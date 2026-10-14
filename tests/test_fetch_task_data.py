"""Download integrity and resumable-transfer contracts without network calls."""

import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "fetch_task_data", Path(__file__).resolve().parents[1] / "setup/fetch_task_data.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}


class TaskDataDownloadTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.target = Path(self.directory.name) / "episode.hdf5"
        self.content = b"abcdef"
        self.digest = hashlib.sha256(self.content).hexdigest()

    def test_verified_existing_file_uses_no_network(self):
        self.target.write_bytes(self.content)
        with patch.object(fetch.urllib.request, "urlopen") as request:
            fetch.download("https://example.invalid/file", self.target, 6, self.digest)
        request.assert_not_called()

    def test_invalid_existing_file_is_preserved(self):
        self.target.write_bytes(b"user-owned")
        with patch.object(fetch.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(ValueError, "Existing source preserved"):
                fetch.download("https://example.invalid/file", self.target, 6, self.digest)
        self.assertEqual(self.target.read_bytes(), b"user-owned")
        request.assert_not_called()

    def test_resumed_range_is_appended_and_verified(self):
        partial = self.target.with_suffix(".hdf5.partial")
        partial.write_bytes(b"abc")
        response = Response(b"def", 206, {"Content-Range": "bytes 3-5/6"})
        with patch.object(fetch.urllib.request, "urlopen", return_value=response) as request:
            fetch.download("https://example.invalid/file", self.target, 6, self.digest)
        self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=3-")
        self.assertEqual(self.target.read_bytes(), self.content)
        self.assertFalse(partial.exists())

    def test_server_ignoring_range_restarts_partial(self):
        self.target.with_suffix(".hdf5.partial").write_bytes(b"abc")
        with patch.object(fetch.urllib.request, "urlopen", return_value=Response(self.content)):
            fetch.download("https://example.invalid/file", self.target, 6, self.digest)
        self.assertEqual(self.target.read_bytes(), self.content)

    def test_wrong_content_is_not_promoted_to_source(self):
        with patch.object(fetch.urllib.request, "urlopen", return_value=Response(b"xxxxxx")):
            with self.assertRaisesRegex(ValueError, "SHA256 failed"):
                fetch.download("https://example.invalid/file", self.target, 6, self.digest)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.target.with_suffix(".hdf5.partial").read_bytes(), b"xxxxxx")


if __name__ == "__main__":
    unittest.main()
