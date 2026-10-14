"""Provider contracts: image history, action/result linkage, and fail-closed output."""
from copy import deepcopy
import unittest

from roboicl.policy.provider_compat import (
    from_wire_response, provider_cache_usage, to_wire_payload, validate_endpoint,
)
from roboicl.policy.bounded_policy import _validate_json_response, IncompleteResponseError


class ChatCompletionsTest(unittest.TestCase):
    def payload(self):
        return {
            "model": "local", "instructions": "Control the robot.",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "Current scene"},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,AAAA", "detail": "high"},
            ]}],
            "tools": [{"type": "function", "name": "act", "strict": True,
                       "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "act"},
            "parallel_tool_calls": False, "max_output_tokens": 4096,
            "reasoning": {"effort": "xhigh"}, "include": ["reasoning.encrypted_content"],
            "prompt_cache_options": {"mode": "implicit"}, "prompt_cache_key": "test",
            "store": False, "chat_template_kwargs": {"enable_thinking": False},
        }

    def response(self, reason="tool_calls"):
        return {
            "id": "chat-1", "choices": [{"finish_reason": reason, "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call-1", "type": "function", "function": {
                    "name": "act", "arguments": '{"actions": []}'}}],
            }}],
            "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168,
                      "prompt_tokens_details": {"cached_tokens": 20}},
        }

    def test_image_and_tool_request_without_gpt_only_fields(self):
        payload = self.payload()
        original = deepcopy(payload)
        wire = to_wire_payload(payload, "chat_completions")
        self.assertEqual(payload, original)
        self.assertEqual(wire["messages"][0], {"role": "system", "content": "Control the robot."})
        self.assertEqual(wire["messages"][1]["content"][1], {
            "type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA", "detail": "high"}})
        self.assertEqual(wire["tool_choice"], {"type": "function", "function": {"name": "act"}})
        self.assertEqual(wire["max_tokens"], 4096)
        self.assertEqual(wire["chat_template_kwargs"], {"enable_thinking": False})
        for key in ("reasoning", "include", "store", "prompt_cache_options", "prompt_cache_key"):
            self.assertNotIn(key, wire)

    def test_action_call_survives_next_turn_with_matching_receipt(self):
        response = from_wire_response(self.response(), "chat_completions")
        payload = self.payload()
        payload["input"].extend(response["output"])
        payload["input"].append({"type": "function_call_output", "call_id": "call-1",
                                 "output": '{"executed_steps": 0}'})
        payload["input"].append({"type": "reasoning", "encrypted_content": "opaque"})
        payload["input"].append({"role": "user", "content": "New observation"})
        wire = to_wire_payload(payload, "chat_completions")
        self.assertEqual(wire["messages"][2]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(wire["messages"][3], {"role": "tool", "tool_call_id": "call-1",
                                               "content": '{"executed_steps": 0}'})
        self.assertEqual(wire["messages"][4]["content"], "New observation")
        self.assertEqual(response["usage"]["total_tokens"], 168)
        self.assertEqual(provider_cache_usage(response["usage"])["cached_tokens"], 20)

    def test_vllm_stop_with_tool_call_is_supported(self):
        result = from_wire_response(self.response("stop"), "chat_completions")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["output"][0]["name"], "act")

    def test_truncated_tool_call_is_not_executable(self):
        result = from_wire_response(self.response("length"), "chat_completions")
        self.assertEqual(result["output"], [])
        with self.assertRaises(IncompleteResponseError):
            _validate_json_response(result)

    def test_missing_call_id_is_rejected(self):
        response = self.response()
        del response["choices"][0]["message"]["tool_calls"][0]["id"]
        with self.assertRaisesRegex(ValueError, "Malformed"):
            from_wire_response(response, "chat_completions")

    def test_unknown_cache_usage_is_not_reported_as_a_hit(self):
        response = self.response()
        response["usage"]["prompt_tokens_details"] = None
        result = from_wire_response(response, "chat_completions")
        self.assertFalse(provider_cache_usage(result["usage"])["cache_reporting_valid"])

    def test_responses_payload_remains_unchanged(self):
        payload = self.payload()
        self.assertEqual(to_wire_payload(payload, "responses"), payload)
        response = {"id": "resp-1", "status": "completed", "output": []}
        self.assertIs(from_wire_response(response, "responses"), response)

    def test_endpoint_must_match_selected_protocol(self):
        validate_endpoint("chat_completions", "http://127.0.0.1:8000/v1/chat/completions")
        with self.assertRaises(ValueError):
            validate_endpoint("chat_completions", "http://127.0.0.1:8000/v1/responses")

    def test_chat_streaming_is_explicitly_rejected(self):
        payload = self.payload()
        payload["stream"] = True
        with self.assertRaisesRegex(ValueError, "stream=false"):
            to_wire_payload(payload, "chat_completions")


if __name__ == "__main__":
    unittest.main()
