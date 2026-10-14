"""Run with Isaac Sim extensions loaded to exercise the real Warp prim API."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from roboicl.isaaclab_compat import _restore_warp_world_to_local


class WarpWorldToLocalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import warp as wp
            from pxr import Gf
            import isaacsim.core.utils.warp as backend
            import isaacsim.core.utils.warp.transformations as transformations
        except ImportError as exc:
            raise unittest.SkipTest(f'Isaac Sim runtime required: {exc}')
        cls.wp = wp
        cls.backend = backend
        cls.transformations = transformations
        _restore_warp_world_to_local()

    def convert(self, parents, positions, quaternions, device):
        inputs = [self.wp.from_numpy(np.asarray(x, dtype=np.float32), device=device)
                  for x in (parents, positions, quaternions)]
        outputs = self.backend.get_local_from_world(*inputs, device)
        self.assertTrue(all(str(x.device) == device for x in outputs))
        for actual, expected in zip(inputs, (parents, positions, quaternions)):
            np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)
        return [x.numpy() for x in outputs]

    def test_translation_uses_inverse_parent(self):
        parent = np.eye(4)[None]
        parent[0, 3, :3] = [1, 2, 3]
        positions, quaternions = self.convert(parent, [[1.1, 2.2, 3.3]],
                                              [[1, 0, 0, 0]], 'cpu')
        np.testing.assert_allclose(positions, [[.1, .2, .3]], atol=1e-6)
        np.testing.assert_allclose(quaternions, [[1, 0, 0, 0]], atol=1e-6)

    def test_rotated_batch_and_round_trip(self):
        parent_rotation = Rotation.from_euler('xyz', [[30, 0, 0], [20, -80, -90]], degrees=True)
        local_rotation = Rotation.from_euler('xyz', [[-10, 40, 15], [70, 20, -35]], degrees=True)
        parent_position = np.asarray([[0, -.41, 1.308], [-.3, -.45, .765]])
        local_position = np.asarray([[.1, -.2, .3], [-.2, .1, .001]])
        parents = np.repeat(np.eye(4)[None], 2, axis=0)
        parents[:, :3, :3] = parent_rotation.as_matrix().transpose(0, 2, 1)
        parents[:, 3, :3] = parent_position
        world_position = parent_rotation.apply(local_position) + parent_position
        world_quaternion = (parent_rotation * local_rotation).as_quat()[:, [3, 0, 1, 2]]
        devices = ['cpu'] + (['cuda:0'] if self.wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device):
                p, q = self.convert(parents, world_position, world_quaternion, device)
                np.testing.assert_allclose(p, local_position, atol=1e-6)
                actual_rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]])
                np.testing.assert_allclose(actual_rotation.as_matrix(), local_rotation.as_matrix(), atol=1e-6)
                inputs = [self.wp.from_numpy(np.asarray(x, dtype=np.float32), device=device)
                          for x in (parents, p, q)]
                # The upstream forward helper uses torch matmul. Disable TF32
                # in this numerical check so its approximation cannot mask
                # an inverse-transform regression; production settings stay intact.
                import torch
                previous_tf32 = torch.backends.cuda.matmul.allow_tf32
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    wp_out, wq_out = self.backend.get_world_from_local(*inputs, device)
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = previous_tf32
                np.testing.assert_allclose(wp_out.numpy(), world_position, atol=1e-6)
                np.testing.assert_allclose(Rotation.from_quat(wq_out.numpy()[:, [1, 2, 3, 0]]).as_matrix(),
                                           (parent_rotation * local_rotation).as_matrix(), atol=1e-6)

    def test_empty_batch(self):
        p, q = self.convert(np.empty((0, 4, 4)), np.empty((0, 3)), np.empty((0, 4)), 'cpu')
        self.assertEqual(p.shape, (0, 3))
        self.assertEqual(q.shape, (0, 4))

    def test_install_is_idempotent_and_exports_agree(self):
        function = self.backend.get_local_from_world
        self.assertFalse(_restore_warp_world_to_local())
        self.assertIs(self.backend.get_local_from_world, function)
        self.assertIs(self.transformations.get_local_from_world, function)


if __name__ == '__main__':
    unittest.main()
