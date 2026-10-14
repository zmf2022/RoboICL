"""Translate the policy's canonical dialogue at the provider boundary."""

from __future__ import annotations

from copy import deepcopy
from urllib.parse import urlparse


RESPONSES = "responses"
CHAT_COMPLETIONS = "chat_completions"


def validate_api_mode(value: object) -> str:
    if value not in (RESPONSES, CHAT_COMPLETIONS):
        raise ValueError("api_mode must be responses or chat_completions")
    return value


def validate_endpoint(api_mode: str, endpoint: object) -> str:
    validate_api_mode(api_mode)
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("ASTRA_ENDPOINT must be a non-empty API URL")
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("ASTRA_ENDPOINT must be an absolute HTTP(S) URL")
    suffix = "/responses" if api_mode == RESPONSES else "/chat/completions"
    if not parsed.path.rstrip("/").endswith(suffix):
        raise ValueError(f"{api_mode} endpoint must end in {suffix}")
    return endpoint


def to_wire_payload(payload: dict, api_mode: str) -> dict:
    validate_api_mode(api_mode)
    if api_mode == RESPONSES:
        return deepcopy(payload)
    if payload.get("stream"):
        raise ValueError("Chat Completions currently requires stream=false")
    messages = []
    if payload.get("instructions"):
        messages.append({"role": "system", "content": payload["instructions"]})
    inputs = payload.get("input", [])
    if isinstance(inputs, str):
        inputs = [{"role": "user", "content": inputs}]
    for item in inputs:
        kind = item.get("type", "message")
        if kind == "reasoning":
            # Opaque Responses reasoning is not a portable chat message.
            continue
        if kind == "function_call":
            call = {"id": item["call_id"], "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]}}
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item["call_id"],
                             "content": item["output"]})
        elif kind == "message":
            content = item.get("content", "")
            if isinstance(content, list):
                converted = []
                for part in content:
                    if part["type"] in ("input_text", "output_text"):
                        converted.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "input_image":
                        image = {"url": part["image_url"]}
                        if "detail" in part:
                            # Preserve canonical TRAIN encoding; Chat Completions
                            # uses high for native-resolution image presentation.
                            image["detail"] = "high" if part["detail"] == "original" else part["detail"]
                        converted.append({"type": "image_url", "image_url": image})
                    else:
                        raise ValueError(f"Unsupported chat content: {part['type']}")
                content = converted
            messages.append({"role": item["role"], "content": content})
        else:
            raise ValueError(f"Unsupported chat history item: {kind}")
    wire = {"model": payload["model"], "messages": messages}
    if "tools" in payload:
        wire["tools"] = [{"type": "function", "function": {
            key: deepcopy(tool[key]) for key in ("name", "description", "parameters", "strict")
            if key in tool}} for tool in payload["tools"]]
        if any(tool["type"] != "function" for tool in payload["tools"]):
            raise ValueError("Chat Completions supports only function tools")
    choice = payload.get("tool_choice")
    if choice is not None:
        wire["tool_choice"] = ({"type": "function", "function": {"name": choice["name"]}}
                               if isinstance(choice, dict) else choice)
    if "max_output_tokens" in payload:
        wire["max_tokens"] = payload["max_output_tokens"]
    for key in ("parallel_tool_calls", "temperature", "top_p", "chat_template_kwargs"):
        if key in payload:
            wire[key] = deepcopy(payload[key])
    # Responses reasoning, cache directives, include, and store are not sent.
    return wire


def from_wire_response(response: dict, api_mode: str) -> dict:
    validate_api_mode(api_mode)
    if not isinstance(response, dict):
        raise ValueError("Provider returned a non-object payload")
    if api_mode == RESPONSES:
        return response
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one Chat Completions choice")
    choice = choices[0]
    reason = choice.get("finish_reason")
    if reason not in ("stop", "tool_calls"):
        # Never accept partial action JSON from a length-limited response.
        return {"id": response.get("id"), "status": "incomplete", "output": [],
                "incomplete_details": {"reason": "max_output_tokens" if reason == "length" else reason}}
    message = choice["message"]
    output = []
    if message.get("content"):
        output.append({"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": message["content"]}]})
    for call in message.get("tool_calls") or []:
        function = call["function"]
        if (call.get("type") != "function" or not isinstance(call.get("id"), str)
                or not call["id"].strip() or not isinstance(function.get("name"), str)
                or not isinstance(function.get("arguments"), str)):
            raise ValueError("Malformed Chat Completions tool call")
        output.append({"type": "function_call", "call_id": call["id"],
                       "name": function["name"], "arguments": function["arguments"]})
    usage = response.get("usage") or {}
    normalized_usage = {"input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens")}
    if isinstance(usage.get("prompt_tokens_details"), dict):
        normalized_usage["input_tokens_details"] = deepcopy(usage["prompt_tokens_details"])
    return {"id": response.get("id"), "status": "completed", "output": output,
            "usage": normalized_usage}


def provider_cache_usage(usage: object) -> dict:
    """Normalize optional cache accounting without inventing missing values."""
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = usage.get("input_tokens")
    details = usage.get("input_tokens_details")
    cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else None
    cache_write_tokens = details.get("cache_write_tokens") if isinstance(details, dict) else None
    valid = (
        type(input_tokens) is int and input_tokens >= 0
        and type(cached_tokens) is int and cached_tokens >= 0
    )
    return {
        "input_tokens": input_tokens if type(input_tokens) is int else None,
        "cached_tokens": cached_tokens if type(cached_tokens) is int else None,
        "cache_write_tokens": cache_write_tokens if type(cache_write_tokens) is int else None,
        "cache_reporting_valid": valid,
    }
