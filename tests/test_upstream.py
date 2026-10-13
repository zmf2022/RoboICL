import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from roboicl.upstream import verify_curobo, verify_simulator, verify_sources


class BundledSourceTest(unittest.TestCase):
    def test_current_sources_are_complete(self):
        report = verify_sources()
        self.assertEqual(len(report['packages']), 3)
        self.assertTrue(all(p['storage'] == 'vendored' for p in report['packages']))

    def test_missing_entry_point_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'dependency').mkdir()
            lock = root / 'lock.json'
            lock.write_text(json.dumps({'schema': 'test', 'packages': [
                {'name': 'test', 'path': 'dependency', 'commit': 'provenance',
                 'required_files': ['entry.py']}
            ]}))
            with patch('roboicl.upstream.CODE', root):
                with self.assertRaisesRegex(FileNotFoundError, 'entry.py'):
                    verify_sources(lock)

    def test_embedded_repository_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'dependency').mkdir()
            (root / 'dependency/.git').write_text('gitdir: elsewhere')
            lock = root / 'lock.json'
            lock.write_text(json.dumps({'schema': 'test', 'packages': [
                {'name': 'test', 'path': 'dependency', 'commit': 'provenance',
                 'required_files': []}
            ]}))
            with patch('roboicl.upstream.CODE', root):
                with self.assertRaisesRegex(ValueError, 'nested Git metadata'):
                    verify_sources(lock)

    def test_wrong_simulator_is_rejected_before_launch(self):
        probe = subprocess.CompletedProcess([], 0, json.dumps(
            {'python': '3.12', 'isaacsim': '5.1.0.0'}), '')
        with patch('roboicl.upstream.subprocess.run', return_value=probe):
            with self.assertRaisesRegex(ValueError, 'expected 6.0.1.0'):
                verify_simulator(Path('/sim/python'))

    def test_wrong_python_is_rejected_before_launch(self):
        probe = subprocess.CompletedProcess([], 0, json.dumps(
            {'python': '3.11', 'isaacsim': '6.0.1.0'}), '')
        with patch('roboicl.upstream.subprocess.run', return_value=probe):
            with self.assertRaisesRegex(ValueError, 'expected 3.12'):
                verify_simulator(Path('/sim/python'))


    def test_old_curobo_install_is_rejected(self):
        probe = subprocess.CompletedProcess([], 0, json.dumps({
            'installed': True, 'version': '0.7.8',
            'provenance': {'url': 'https://github.com/NVlabs/curobo.git',
                           'vcs_info': {'vcs': 'git', 'commit_id': 'old'}},
            'missing_files': ['batch_motion_planner.py'],
        }), '')
        with patch('roboicl.upstream.subprocess.run', return_value=probe):
            with self.assertRaisesRegex(ValueError, 'newer API'):
                verify_curobo(Path('/sim/python'))

    def test_missing_curobo_install_is_rejected(self):
        probe = subprocess.CompletedProcess([], 0, '{"installed": false}', '')
        with patch('roboicl.upstream.subprocess.run', return_value=probe):
            with self.assertRaisesRegex(ValueError, 'requirements-sim.txt'):
                verify_curobo(Path('/sim/python'))

    def test_pinned_curobo_without_required_api_is_rejected(self):
        expected = json.loads(Path('configs/upstream.lock.json').read_text())['external_packages'][0]
        actual = {'installed': True, 'version': '0.8.0',
                  'provenance': {'url': expected['url'], 'vcs_info': {
                      'vcs': 'git', 'commit_id': expected['commit']}},
                  'missing_files': ['runtime.py']}
        probe = subprocess.CompletedProcess([], 0, json.dumps(actual), '')
        with patch('roboicl.upstream.subprocess.run', return_value=probe):
            with self.assertRaisesRegex(ValueError, 'newer API'):
                verify_curobo(Path('/sim/python'))

    def test_pinned_curobo_install_is_accepted(self):
        expected = json.loads(Path('configs/upstream.lock.json').read_text())['external_packages'][0]
        actual = {'installed': True, 'version': '0.8.0',
                  'provenance': {'url': expected['url'], 'vcs_info': {
                      'vcs': 'git', 'commit_id': expected['commit']}},
                  'missing_files': []}
        probe = subprocess.CompletedProcess([], 0, json.dumps(actual), '')
        with patch('roboicl.upstream.subprocess.run', return_value=probe):
            self.assertEqual(verify_curobo(Path('/sim/python')), actual)


if __name__ == '__main__':
    unittest.main()
