"""Codex transport preserves evidence, receipts, budgets, and failure boundaries."""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from roboicl.policy import codex_cli
from roboicl.policy.bounded_policy import Model, RequestTransportError, BudgetExceeded
from roboicl.policy.provider_compat import validate_endpoint


class CodexCLITest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        closing = patch.object(codex_cli, '_closing', False)
        closing.start()
        self.addCleanup(closing.stop)

    def payload(self):
        return {'model': 'gpt-6.1-sol', 'instructions': 'Observe and act.',
                'reasoning': {'effort': 'xhigh'},
                'tools': [{'type': 'function', 'name': 'act', 'parameters': {
                    'type': 'object', 'properties': {'actions': {}, 'execution_note': {}},
                    'required': ['actions', 'execution_note'], 'additionalProperties': False}}],
                'input': [{'role': 'user', 'content': [
                    {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' +
                     base64.b64encode(b'original-jpeg-bytes').decode(), 'detail': 'original'}]},
                    {'type': 'function_call', 'name': 'act', 'call_id': 'train-1', 'arguments': '{}'},
                    {'type': 'function_call_output', 'call_id': 'train-1', 'output': '{"executed_steps":0}'},
                    {'type': 'reasoning', 'encrypted_content': 'opaque'}]}

    def cli(self, extra='', exit_code=0):
        path = self.root / 'fake-codex'
        path.write_text(f'''#!{sys.executable}
import json, sys
from pathlib import Path
if '--version' in sys.argv:
    print('codex-cli test'); sys.exit(0)
if sys.argv[1:3] == ['login', 'status']:
    print('Logged in using ChatGPT'); sys.exit({exit_code})
prompt = sys.stdin.read()
{extra}
args = {{'actions': [[0]*14]*5, 'execution_note': 'hold'}}
text = json.dumps(args)
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(text)
print(json.dumps({{'type':'item.completed','item':{{'type':'agent_message','text':text}}}}))
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':100,'output_tokens':20,'cached_input_tokens':30}}}}))
sys.exit({exit_code})
''')
        path.chmod(0o755)
        return str(path)

    def test_prepare_keeps_image_bytes_and_receipt_linkage(self):
        payload = self.payload(); original = deepcopy(payload)
        prompt, images, schema = codex_cli.prepare(payload, self.root)
        self.assertEqual(payload, original)
        self.assertEqual(images[0].read_bytes(), b'original-jpeg-bytes')
        self.assertIn('train-1', prompt)
        self.assertIn('executed_steps', prompt)
        self.assertNotIn('opaque', prompt)
        self.assertNotIn('base64', prompt)
        self.assertEqual(schema, payload['tools'][0]['parameters'])

    def test_request_normalizes_action_and_usage(self):
        with patch.dict(os.environ, {'ROBOICL_CODEX_CLI': self.cli()}):
            self.assertTrue(codex_cli.preflight()['logged_in'])
            response = codex_cli.request(self.payload(), self.root/'request', 5)
        self.assertEqual(response['output'][0]['name'], 'act')
        self.assertTrue(response['output'][0]['call_id'])
        self.assertEqual(response['usage']['total_tokens'], 120)
        self.assertEqual(response['usage']['input_tokens_details']['cached_tokens'], 30)
        command = json.loads((self.root/'request/command.json').read_text())
        self.assertIn('--ignore-user-config', command)
        self.assertIn('--ephemeral', command)
        self.assertEqual(command[command.index('--model')+1], 'gpt-6.1-sol')
        self.assertFalse(list((self.root/'request').glob('workspace-*')))

    def test_tool_use_is_rejected(self):
        extra = "print(json.dumps({'type':'item.completed','item':{'type':'command_execution'}}))"
        with patch.dict(os.environ, {'ROBOICL_CODEX_CLI': self.cli(extra)}):
            with self.assertRaisesRegex(codex_cli.CodexCLIError, 'used tools'):
                codex_cli.request(self.payload(), self.root/'request', 5)

    def test_nonzero_exit_is_rejected(self):
        with patch.dict(os.environ, {'ROBOICL_CODEX_CLI': self.cli(exit_code=1)}):
            with self.assertRaisesRegex(codex_cli.CodexCLIError, 'exited 1'):
                codex_cli.request(self.payload(), self.root/'request', 5)
            with self.assertRaisesRegex(codex_cli.CodexCLIError, 'not logged in'):
                codex_cli.preflight()

    def test_started_tool_without_completion_is_rejected(self):
        extra = "print(json.dumps({'type':'item.started','item':{'type':'command_execution'}}))"
        with patch.dict(os.environ, {'ROBOICL_CODEX_CLI': self.cli(extra)}):
            with self.assertRaisesRegex(codex_cli.CodexCLIError, 'used tools'):
                codex_cli.request(self.payload(), self.root/'request', 5)

    def test_next_turn_keeps_live_call_and_actual_execution_receipt(self):
        payload = self.payload()
        payload['input'].extend([
            {'type': 'function_call', 'call_id': 'live-1', 'name': 'act', 'arguments': '{"actions":[]}'},
            {'type': 'function_call_output', 'call_id': 'live-1', 'output': '{"executed_steps":5}'},
            {'role': 'user', 'content': [{'type': 'input_text', 'text': 'Latest observation step 5'}]},
        ])
        prompt, _, _ = codex_cli.prepare(payload, self.root)
        self.assertEqual(prompt.count('live-1'), 2)
        self.assertIn('Latest observation step 5', prompt)
        self.assertIn('executed_steps', prompt)

    def test_timeout_terminates_planner(self):
        with patch.dict(os.environ, {'ROBOICL_CODEX_CLI': self.cli('import time; time.sleep(10)')}):
            with self.assertRaisesRegex(codex_cli.CodexCLIError, 'timed out'):
                codex_cli.request(self.payload(), self.root/'request', .1)

    def test_server_interrupt_stops_inflight_cli_before_joining_workers(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=sig):
                pid_file = self.root / f'child-{sig}.pid'
                cli = self.cli(f"import os, time; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)")
                payload_file = self.root/'payload.json'
                payload_file.write_text(json.dumps(self.payload()))
                script = self.root/'server.py'
                script.write_text(f'''
import asyncio, json, sys
from types import SimpleNamespace
from pathlib import Path
from roboicl import policy_server
from roboicl.policy.codex_cli import request
class Server:
    def __init__(self, *args): pass
    async def serve_forever(self):
        await asyncio.to_thread(request, json.loads(Path({str(payload_file)!r}).read_text()), Path({str(self.root/f'run-{sig}')!r}), 60)
policy_server.Model = lambda cfg: SimpleNamespace(api_mode='codex_cli')
policy_server.PolicyServer = Server
sys.argv = ['server', '--port', '1', '--model', 'gpt-6.1-sol', '--task', 'deposit_coin', '--seed', '0']
raise SystemExit(policy_server.main())
''')
                environment = dict(os.environ, ROBOICL_CODEX_CLI=cli,
                    PYTHONPATH=str(Path(__file__).resolve().parents[1]))
                process = subprocess.Popen([sys.executable, str(script)], env=environment,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                child_pid = None
                try:
                    deadline = time.monotonic()+10
                    while not pid_file.exists() and process.poll() is None and time.monotonic()<deadline:
                        time.sleep(.02)
                    self.assertTrue(pid_file.exists(), 'CLI never started')
                    child_pid = int(pid_file.read_text())
                    process.send_signal(sig)
                    self.assertEqual(process.wait(timeout=5), 0)
                    with self.assertRaises(ProcessLookupError):
                        os.kill(child_pid, 0)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    if child_pid is not None:
                        try: os.killpg(child_pid, signal.SIGKILL)
                        except ProcessLookupError: pass

    def test_unenforceable_output_limit_is_rejected(self):
        payload = self.payload(); payload['max_output_tokens'] = 1024
        with self.assertRaisesRegex(codex_cli.CodexCLIError, 'cannot enforce'):
            codex_cli.prepare(payload, self.root)

    def test_endpoint_does_not_accept_http(self):
        self.assertEqual(validate_endpoint('codex_cli', 'codex://exec'), 'codex://exec')
        with self.assertRaises(ValueError):
            validate_endpoint('codex_cli', 'https://example.com/responses')

    def model(self):
        model = Model.__new__(Model)
        model.api_mode = 'codex_cli'; model.episode_dir = self.root
        model.started = time.monotonic(); model.request_timeout_s = 5
        model.budget_poisoned = False; model.max_requests = 3
        model.max_seconds = None; model.max_total_tokens = None
        model.request_count = model.reserved_tokens = model.pending_reserved_tokens = model.reported_tokens = 0
        model.payload_metrics = lambda payload: {'reserved_tokens': 200}
        model._log = lambda *args: None
        return model

    def test_cli_settles_the_shared_request_budget(self):
        model = self.model()
        response = {'status': 'completed', 'output': [], 'usage': {'input_tokens': 100, 'output_tokens': 20}}
        with patch.dict(os.environ, {}, clear=True), patch.object(codex_cli, 'request', return_value=response):
            self.assertEqual(model._request(self.payload()), response)
        self.assertEqual(model.request_count, 1)
        self.assertEqual(model.reported_tokens, 120)
        self.assertEqual(model.pending_reserved_tokens, 0)

    def test_failure_retains_budget_and_cannot_retry_or_dispatch(self):
        model = self.model()
        with patch.dict(os.environ, {}, clear=True), patch.object(codex_cli, 'request', side_effect=codex_cli.CodexCLIError('failed')):
            with self.assertRaises(RequestTransportError) as raised:
                model._request(self.payload())
        self.assertFalse(raised.exception.retryable)
        self.assertTrue(model.budget_poisoned)
        self.assertEqual(model.pending_reserved_tokens, 200)
        with self.assertRaises(BudgetExceeded):
            model._request(self.payload())


if __name__ == '__main__':
    unittest.main()
