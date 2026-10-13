import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from roboicl.upstream import verify_simulator, verify_sources


class BundledSourceTest(unittest.TestCase):
    def test_current_sources_are_complete(self):
        report = verify_sources()
        self.assertEqual(len(report['packages']), 4)
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


if __name__ == '__main__':
    unittest.main()
