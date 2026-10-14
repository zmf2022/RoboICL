"""Exercise the actual robot manager boundary without starting Kit."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "isaaclab_pose", ROOT / "third_party/RoboDojo/utils/isaaclab_pose.py")
pose_helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pose_helpers)


class Tensor(np.ndarray):
    """CPU stand-in for the manager's clone/cpu tensor operations."""
    def clone(self):
        return self.copy()

    def cpu(self):
        return self


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def replace(self, **kwargs):
        return Config(**(self.__dict__ | kwargs))


def load_method(name, source=None):
    if source is None:
        source = (ROOT / "third_party/RoboDojo/env/robot_manager/robot_manager.py").read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RobotManager")
    method = deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name))
    method.returns = None
    for arg in method.args.args + method.args.kwonlyargs:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"deepcopy": deepcopy, "np": np, "ENV_REGEX_NAMESPACE": "{ENV_REGEX_NS}",
                 "ArticulationCfg": SimpleNamespace(InitialStateCfg=Config),
                 "wxyz_to_xyzw": pose_helpers.wxyz_to_xyzw,
                 "xyzw_to_wxyz": pose_helpers.xyzw_to_wxyz}
    exec(compile(module, "robot_manager.py", "exec"), namespace)
    return namespace[name]


class IsaacLabPoseTest(unittest.TestCase):
    def test_spawn_keeps_quarter_turn_about_z(self):
        q = float(np.sqrt(0.5))
        robot = SimpleNamespace(entity_origin_pose=[0.3, -0.45, 0.765, q, 0, 0, q])
        scene = Config(init_state=Config(joint_pos={"joint1": 0.0}))
        manager = SimpleNamespace(_get_robot_cfg=lambda **_: scene)
        load_method("_attach_scene_cfg")(manager, "x5", robot, 0, {})
        np.testing.assert_allclose(robot.SceneCfg.init_state.rot, [0, 0, q, q])
        self.assertEqual(robot.SceneCfg.init_state.pos, [0.3, -0.45, 0.765])
        self.assertEqual(robot.entity_origin_pose[-4:], [q, 0, 0, q])

    def test_link_observation_is_wxyz_in_both_coordinate_modes(self):
        native = np.asarray([[[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]]]).view(Tensor)
        robot = SimpleNamespace(robot_name="x5")
        manager = SimpleNamespace(num_envs=1, robot_list=[robot],
                                  robot_key=[SimpleNamespace(body_names=["link6"],
                                      data=SimpleNamespace(body_link_pose_w=native))],
                                  scene=SimpleNamespace(env_origins=np.asarray([[0.2, 0.3, 0.4]])))
        method = load_method("get_link_pose")
        world = method(manager, robot, "link6", is_relative=False)[0]
        relative = method(manager, robot, "link6", is_relative=True)[0]
        np.testing.assert_allclose(world, [1, 2, 3, 0.9, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(relative, [0.8, 1.7, 2.6, 0.9, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(native[0, 0], [1, 2, 3, 0.1, 0.2, 0.3, 0.9])


if __name__ == "__main__":
    unittest.main()
