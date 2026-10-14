"""Run Codex as an observation-only planner at the provider boundary."""
from __future__ import annotations

import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
from uuid import uuid4

_process_lock = threading.Lock()
_active_processes = set()
_closing = False


class CodexCLIError(RuntimeError):
    """No planner response was accepted; never dispatch an action."""


def executable() -> str:
    selected = os.environ.get('ROBOICL_CODEX_CLI', 'codex')
    resolved = shutil.which(str(Path(selected).expanduser()))
    if not resolved:
        raise CodexCLIError('Codex CLI not found; install it or set ROBOICL_CODEX_CLI')
    return resolved


def preflight() -> dict:
    cli = executable()
    try:
        version = subprocess.run([cli, '--version'], capture_output=True, text=True, timeout=15, check=True)
        login = subprocess.run([cli, 'login', 'status'], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        raise CodexCLIError('Cannot inspect Codex CLI installation') from error
    if login.returncode:
        raise CodexCLIError('Codex is not logged in; run codex login before evaluation')
    return {'executable': cli, 'version': version.stdout.strip(), 'logged_in': True}


def prepare(payload: dict, directory: Path) -> tuple[str, list[Path], dict]:
    if payload.get('stream'):
        raise CodexCLIError('Codex CLI requires stream=false')
    if payload.get('max_output_tokens') is not None:
        raise CodexCLIError('Codex CLI cannot enforce max_output_tokens; omit this API option')
    tools = payload.get('tools', [])
    if len(tools) != 1 or tools[0].get('type') != 'function' or tools[0].get('name') != 'act':
        raise CodexCLIError('Codex CLI requires exactly the act function schema')
    transcript = []
    images = []
    for original in payload.get('input', []):
        if original.get('type') == 'reasoning':
            continue
        item = deepcopy(original)
        if isinstance(item.get('content'), list):
            content = []
            for part in item['content']:
                if part.get('type') == 'input_image':
                    url = part.get('image_url', '')
                    if not url.startswith('data:image/jpeg;base64,'):
                        raise CodexCLIError('Codex input images must be embedded JPEGs')
                    path = directory / f'image_{len(images)+1:03d}.jpg'
                    path.write_bytes(base64.b64decode(url.split(',', 1)[1], validate=True))
                    images.append(path)
                    content.append({'type': 'input_text', 'text':
                        f'[Attached image {len(images):03d}: {path.name}; belongs at this transcript position.]'})
                else:
                    content.append(part)
            item['content'] = content
        transcript.append(item)
    prompt = (
        'Act solely as the robotics action policy. Do not use tools, read files, browse, '
        'inspect the simulator, or obtain any other information. Use only the attached images '
        'and the canonical policy transcript below. Prior TRAIN act calls are demonstrations. '
        'Prior LIVE calls and receipts describe what actually executed; corrections may mean '
        'no motion occurred. Return ONLY the JSON arguments for the next LIVE act call, '
        'matching the output schema. Image numbers follow attachment order.\n\n'
        'POLICY INSTRUCTIONS:\n' + payload.get('instructions', '') + '\n\n'
        'CANONICAL TRANSCRIPT:\n' + json.dumps(transcript, ensure_ascii=False) + '\n\n'
        'Generate the next LIVE act arguments for the latest TEST observation now.'
    )
    return prompt, images, deepcopy(tools[0]['parameters'])


def _stop(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def stop_active_requests():
    """Stop planners before asyncio waits for its model worker threads to exit."""
    global _closing
    with _process_lock:
        _closing = True
        processes = list(_active_processes)
    for process in processes:
        _stop(process)


def request(payload: dict, directory: Path, timeout: float | None) -> dict:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    prompt, images, schema = prepare(payload, directory)
    (directory / 'prompt.txt').write_text(prompt, encoding='utf-8')
    schema_path = directory / 'schema.json'
    schema_path.write_text(json.dumps(schema), encoding='utf-8')
    response_path = directory / 'response.json'
    events_path = directory / 'events.jsonl'
    with tempfile.TemporaryDirectory(prefix='workspace-', dir=directory) as workspace:
        command = [executable(), 'exec', '--ignore-user-config', '--ephemeral',
                   '--skip-git-repo-check', '--sandbox', 'read-only', '--cd', workspace,
                   '--model', payload['model'], '-c',
                   'model_reasoning_effort=' + json.dumps(payload.get('reasoning', {}).get('effort', 'xhigh')),
                   '-c', 'web_search="disabled"', '--output-schema', str(schema_path),
                   '--json', '--output-last-message', str(response_path)]
        for image in images:
            command.extend(['--image', str(image)])
        command.append('-')
        (directory / 'command.json').write_text(json.dumps(command, indent=2))
        with events_path.open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
            with _process_lock:
                if _closing:
                    raise CodexCLIError('Policy server is shutting down; no request started')
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout,
                                           stderr=stderr, text=True, start_new_session=True)
                _active_processes.add(process)
            try:
                process.communicate(prompt, timeout=timeout)
            except BaseException as error:
                _stop(process)
                if isinstance(error, subprocess.TimeoutExpired):
                    raise CodexCLIError(f'Codex CLI timed out; see {directory}') from None
                raise
            finally:
                with _process_lock:
                    _active_processes.discard(process)
        if process.returncode:
            raise CodexCLIError(f'Codex CLI exited {process.returncode}; see {directory / "stderr.log"}')
    try:
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        if any(event.get('type') in ('error', 'turn.failed') for event in events):
            raise ValueError('CLI reported a failed turn')
        all_items = [event['item'] for event in events if isinstance(event.get('item'), dict)]
        if any(item.get('type') not in ('agent_message', 'reasoning') for item in all_items):
            raise ValueError('Planner used tools; response cannot be accepted')
        items = [event['item'] for event in events if event.get('type') == 'item.completed']
        completed = [event for event in events if event.get('type') == 'turn.completed']
        if len(completed) != 1:
            raise ValueError('Expected exactly one completed planner turn')
        arguments = json.loads(response_path.read_text())
        if (not isinstance(arguments, dict) or set(arguments) != {'actions', 'execution_note'}
                or not isinstance(arguments['execution_note'], str)
                or len(arguments['execution_note']) > 400):
            raise ValueError('Invalid act arguments')
        messages = [item for item in items if item.get('type') == 'agent_message']
        if not messages or json.loads(messages[-1]['text']) != arguments:
            raise ValueError('Final message differs from output file')
        usage = completed[0]['usage']
        if any(type(usage.get(key)) is not int or usage[key] < 0 for key in ('input_tokens', 'output_tokens')):
            raise ValueError('Missing/invalid CLI usage')
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CodexCLIError(f'Invalid Codex response: {error}; see {directory}') from None
    normalized = {'input_tokens': usage['input_tokens'], 'output_tokens': usage['output_tokens'],
                  'total_tokens': usage['input_tokens'] + usage['output_tokens']}
    details = {target: usage[source] for source, target in (
        ('cached_input_tokens', 'cached_tokens'), ('cache_write_input_tokens', 'cache_write_tokens'))
        if type(usage.get(source)) is int and usage[source] >= 0}
    if details:
        normalized['input_tokens_details'] = details
    return {'id': 'codex-' + uuid4().hex, 'status': 'completed', 'usage': normalized,
            'output': [{'type': 'function_call', 'call_id': 'call-' + uuid4().hex,
                        'name': 'act', 'arguments': json.dumps(arguments, allow_nan=False)}]}
