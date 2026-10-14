"""Quaternion conversions at the Isaac Lab 3 / RoboDojo boundary.

RoboDojo layouts, observations, transforms3d, and cuRobo use WXYZ.
Isaac Lab 3 asset configuration and state buffers use XYZW.
Isaac Sim's prim and camera APIs still use WXYZ.
"""


def wxyz_to_xyzw(quaternion):
    """Convert a RoboDojo quaternion to Isaac Lab's asset convention."""
    w, x, y, z = quaternion
    return x, y, z, w


def xyzw_to_wxyz(quaternion):
    """Convert an Isaac Lab state quaternion to RoboDojo's convention."""
    x, y, z, w = quaternion
    return w, x, y, z
