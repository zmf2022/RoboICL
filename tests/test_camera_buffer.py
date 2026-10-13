"""Test the camera read boundary without Kit, CUDA, or a live scene."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class CameraBufferTest(unittest.TestCase):
    def test_empty_annotator_is_rejected_before_allocating_or_launching(self):
        path = Path(__file__).resolve().parents[1] / 'third_party/RoboDojo/env/camera_manager/capture/camera_view.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CameraView')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_data')
        method = ast.parse(ast.unparse(method)).body[0]
        method.returns = None
        for arg in method.args.args + method.args.kwonlyargs:
            arg.annotation = None
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        empty = SimpleNamespace(size=0, shape=(0,), device='cpu', flatten=lambda: None)
        warp = Mock()
        warp.zeros.return_value = SimpleNamespace(shape=(1, 480, 640, 4), device='cpu')
        namespace = {'ANNOTATOR_SPEC': {'rgba': {'name': 'rgba', 'channels': 4, 'dtype': object()}},
                     'wp': warp, 'reshape_tiled_image': object()}
        exec(compile(module, str(path), 'exec'), namespace)
        annotator = Mock()
        annotator.get_data.return_value = empty
        camera = SimpleNamespace(_annotators={'rgba': annotator}, _render_product_path='/Render/test',
                                 camera_resolution=(640, 480), tiled_resolution=(640, 480), prims=[object()])
        out = SimpleNamespace(shape=(1, 480, 640, 4), device='cuda:0')
        with self.assertRaisesRegex(RuntimeError, 'no delivered frame.*Kit visualizer'):
            namespace['get_data'](camera, 'rgba', out=out)
        warp.zeros.assert_not_called()
        warp.launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
