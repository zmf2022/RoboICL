import json
from pathlib import Path
import tempfile
import unittest

from roboicl.run import validate_camera_probe


class CameraProbeValidationTest(unittest.TestCase):
    def make_probe(self, folder, black=False):
        probe = folder / 'capture/capture_probe.json'
        probe.parent.mkdir()
        rows = [{'route': 'observation', 'frame': frame, 'camera': camera,
                 'shape': [480, 640, 3], 'max': 0 if black else 255}
                for frame in range(10) for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')]
        probe.write_text(json.dumps(rows))

    def test_complete_capture_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.make_probe(folder)
            validate_camera_probe(folder, 0)

    def test_crashed_simulator_cannot_pass_with_probe_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.make_probe(folder)
            with self.assertRaisesRegex(RuntimeError, 'code -11.*client.log'):
                validate_camera_probe(folder, -11)

    def test_missing_probe_points_to_client_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'did not produce.*client.log'):
                validate_camera_probe(Path(tmp), 0)

    def test_black_frames_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.make_probe(folder, black=True)
            with self.assertRaisesRegex(RuntimeError, 'Camera probe invalid'):
                validate_camera_probe(folder, 0)


if __name__ == '__main__':
    unittest.main()
