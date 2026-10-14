"""Isaac Lab 3.0 / Isaac Sim 6.x compatibility for the pinned RoboDojo integration.

RoboDojo targets Isaac Lab 2.3 on Isaac Sim 5.1 (Python 3.11). The simulator
environment runs Isaac Lab 3.0 on Isaac Sim 6.x (Python 3.12), which changes
several integration contracts:

- ``isaaclab.sim.SimulationCfg.physx`` moved behind ``SimulationCfg.physics``
  and ``PhysxCfg`` moved into the separate ``isaaclab_physx`` package.
- ``isaaclab.utils`` is a lazy-loader package whose ``configclass`` attribute
  is clobbered by the same-named submodule once imported.
- ``isaacsim.core.*`` and other RoboDojo dependencies live inside Isaac Sim
  extension directories that the Isaac Lab 3.0 experience never puts on
  ``sys.path``.
- The legacy compiled ``Semantics`` module was removed; a deprecated helper
  imports it at module scope.
- Kit's extension resolver never sees ``extsDeprecated/``, where Isaac Sim 6
  keeps ``isaacsim.sensors.camera`` and the ``isaacsim.core.*`` chain, so
  RoboDojo's ``--enable`` flags fail before the app starts.
- ``omni.physx`` stopped re-exporting ``acquire_physx_interface`` after its
  extension loads; RoboDojo still calls it through the package.
- Asset configuration and pose buffers changed from WXYZ to XYZW. RoboDojo
  converts these explicitly at its robot manager boundary; prim/camera APIs
  retain their own WXYZ contract and are not patched globally.
- Deprecated Warp prim utilities apply a forward parent transform in
  ``get_local_from_world``. Restore the inverse before prims rewrite poses.

This module restores those contracts within the shared bundled sources.
Only the entry points RoboDojo actually touches are restored; no other Isaac
Lab 2.3 behaviour is emulated, because remaining call sites already import
cleanly from Isaac Lab 3.0.
"""

from __future__ import annotations

import glob
import os
import sys
import types

_COMPAT_FLAG = "_roboicl_isaaclab3_compat"


def _import_physx_cfg():
    """Return Isaac Lab 3.0's PhysxCfg, or None when PhysX is unavailable."""
    try:
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
        return PhysxCfg
    except ImportError:
        return None


def _restore_configclass() -> bool:
    """Restore ``isaaclab.utils.configclass`` after its submodule import.

    Isaac Lab 3.0 exposes ``isaaclab.utils`` through a lazy-loader stub, but
    importing the ``isaaclab.utils.configclass`` submodule rebinds the parent
    package attribute to that module object. RoboDojo calls the decorator, so
    import the submodule first and put the function back explicitly. Once the
    submodule is in ``sys.modules`` Python will not re-bind the attribute, so
    the fix is stable for the rest of the process.
    """
    import isaaclab.utils as utils_pkg

    from isaaclab.utils.configclass import configclass

    if getattr(utils_pkg, "configclass", None) is configclass:
        return False
    utils_pkg.configclass = configclass
    return True


def _expose_isaacsim_extensions() -> int:
    """Expose Isaac Sim 6.x ``isaacsim.*`` and ``omni.*`` extension packages.

    Isaac Sim 6 pip wheels keep ``isaacsim.core.*``, ``isaacsim.sensors.camera``
    and other RoboDojo dependencies inside ``exts*/<ext>/isaacsim/...``, which
    the Isaac Lab 3.0 experience never adds to ``sys.path``. Appending each
    extension's inner ``isaacsim`` directory to ``isaacsim.__path__`` makes the
    namespace importable without editing the site-packages tree. Extension
    ``compat/`` trees re-export legacy ``omni.isaac.*`` schemas the same way.
    """
    import isaacsim

    site_isaac = isaacsim.__path__[0]
    added = 0
    for pattern in ("exts/*", "extsDeprecated/*", "extsUser/*", "extsPhysics/*"):
        for root in glob.glob(os.path.join(site_isaac, pattern)):
            for inner in glob.glob(os.path.join(root, "isaacsim")):
                if inner not in isaacsim.__path__:
                    isaacsim.__path__.append(inner)
                    added += 1
            for compat in glob.glob(os.path.join(root, "compat", "omni")):
                import omni

                if compat not in omni.__path__:
                    omni.__path__.append(compat)
                    added += 1
    return added


def _restore_physics_context_accessor() -> bool:
    """Restore ``SimulationContext.get_physics_context``.

    Isaac Lab 3.0 replaced the ``PhysicsContext`` object with a physics
    manager, dropping this accessor. RoboDojo calls it once at setup and
    keeps the result without ever reading it, so returning the live
    ``PhysicsCfg`` preserves the call contract without emulating 2.3
    behaviour that nothing consumes.
    """
    from isaaclab.sim import SimulationContext

    if hasattr(SimulationContext, "get_physics_context"):
        return False

    def get_physics_context(self):
        return getattr(self, "_physics", None)

    SimulationContext.get_physics_context = get_physics_context
    return True


def deprecated_extension_args() -> str:
    """Register deprecated extensions with Kit without modifying site-packages."""
    import shlex
    import isaacsim

    folder = os.path.join(isaacsim.__path__[0], "extsDeprecated")
    return f"--ext-folder={shlex.quote(folder)}" if os.path.isdir(folder) else ""


def _stub_missing_semantics() -> bool:
    """Stub the removed ``Semantics`` compiled extension.

    ``isaacsim.core.utils.semantics`` (deprecated in Isaac Sim 6.x) imports the
    legacy compiled ``Semantics`` module at module scope, but that module is no
    longer shipped. It is only referenced by the migration helper, which
    RoboDojo never calls; a stub keeps the import graph loadable.
    """
    if "Semantics" in sys.modules:
        return False
    try:
        import Semantics  # noqa: F401
        return False
    except ImportError:
        pass
    stub = types.ModuleType("Semantics")
    stub.SemanticsAPI = None
    sys.modules["Semantics"] = stub
    return True


def install() -> dict:
    """Restore the Isaac Lab 3.0 / Isaac Sim 6.x entry points RoboDojo needs.

    Returns a report describing what was changed. Repeated calls are inert so
    the shim stays idempotent across module reloads in one process.
    """
    import isaaclab.sim as sim

    report = {
        "configclass_restored": _restore_configclass(),
        "isaacsim_extension_paths_added": _expose_isaacsim_extensions(),
        "semantics_stubbed": _stub_missing_semantics(),
        "installed": False,
        "reason": None,
    }
    if getattr(sim, _COMPAT_FLAG, False):
        report["reason"] = "already_installed"
        return report

    base = sim.SimulationCfg
    fields = getattr(base, "__dataclass_fields__", {})
    if "physx" in fields and hasattr(sim, "PhysxCfg"):
        report["reason"] = "native_isaaclab2_api"
        return report

    physx_cfg = _import_physx_cfg()
    if physx_cfg is None and "physics" not in fields:
        raise ImportError(
            "Isaac Lab exposes neither SimulationCfg.physx nor SimulationCfg.physics; "
            "this Isaac Lab revision is unsupported by the RoboDojo integration"
        )

    original_init = base.__init__

    def __init__(self, *args, physx=None, **kwargs):
        # Isaac Lab 3.0 renamed the backend slot; an explicitly supplied
        # default-constructed PhysxCfg keeps its documented meaning.
        if physx is not None and "physics" not in kwargs:
            kwargs["physics"] = physx
        original_init(self, *args, **kwargs)

    base.__init__ = __init__
    if physx_cfg is not None:
        sim.PhysxCfg = physx_cfg
    physx_alias = False
    if "physics" in fields and "physx" not in fields:
        # RoboDojo reads ``sim_cfg.sim.physx`` at reset time; alias the
        # renamed slot so both the constructor and attribute access agree.
        base.physx = property(
            lambda self: self.physics,
            lambda self, value: setattr(self, "physics", value),
        )
        physx_alias = True
    setattr(sim, _COMPAT_FLAG, True)
    report.update({
        "installed": True,
        "physx_cfg_restored": physx_cfg is not None,
        "physx_kwarg_mapped_to": "physics" if "physics" in fields else None,
        "physx_property_aliased": physx_alias,
    })
    return report


def _align_backend_with_warp_view() -> bool:
    """Report the physics view's tensor backend from ``PhysxManager.get_backend``.

    Isaac Lab 3.0 always creates its physics view with the warp backend, but
    the inherited ``get_backend`` derives numpy/torch from the device string.
    Isaac Sim's deprecated prim wrappers pick their tensor utilities from
    ``get_backend()`` and must stay type-consistent with the view they call,
    so report the live view's own backend once it exists. Isaac Lab only
    stores this value (``asset_base._backend``, ``SimulationContext.backend``)
    without branching on it, and RoboDojo reads it for the same consistency
    purpose.
    """
    from isaaclab.physics.physics_manager import PhysicsManager
    from isaaclab_physx.physics.physx_manager import PhysxManager

    if getattr(PhysxManager.get_backend, "_roboicl_warp_view", False):
        return False
    original_get_backend = PhysicsManager.get_backend.__func__

    @classmethod
    def get_backend(cls):
        view = getattr(cls, "_view", None)
        frontend = getattr(view, "_frontend", None) if view is not None else None
        # FrontendWarp / FrontendTorch / FrontendNumpy; omni.physics.tensors
        # keeps the framework name only on the frontend class.
        name = type(frontend).__name__.lower().replace("frontend", "")
        if name in ("warp", "torch", "numpy"):
            return name
        return original_get_backend(cls)

    get_backend._roboicl_warp_view = True
    PhysxManager.get_backend = get_backend
    return True


def _restore_physx_manager_backend_utils() -> bool:
    """Restore ``SimulationManager._get_backend_utils`` on Isaac Lab's manager.

    Isaac Lab 3.0 swaps Isaac Sim's ``isaacsim.core.simulation_manager.
    SimulationManager`` for its own ``PhysxManager``, which drops the
    deprecated backend helper that Isaac Sim's deprecated ``isaacsim.core.
    prims`` package still calls. Re-implement it against ``get_backend()``
    exactly as Isaac Sim does (numpy / torch / warp utility module).
    """
    from isaaclab_physx.physics.physx_manager import PhysxManager

    if hasattr(PhysxManager, "_get_backend_utils"):
        return False

    @classmethod
    def _get_backend_utils(cls):
        backend = cls.get_backend()
        modules = {
            "numpy": "isaacsim.core.utils.numpy",
            "torch": "isaacsim.core.utils.torch",
            "warp": "isaacsim.core.utils.warp",
        }
        if backend not in modules:
            raise ValueError(
                f"Provided backend is not supported: {backend}. Supported: torch, numpy, warp."
            )
        import importlib

        return importlib.import_module(modules[backend])

    PhysxManager._get_backend_utils = _get_backend_utils
    return True


def _restore_warp_world_to_local() -> bool:
    """Correct the deprecated Warp prim utility at its shared call boundary.

    Isaac Sim 6.0.1's get_local_from_world calls its local-to-world kernel.
    XFormPrim uses this to preserve world poses while rewriting xform ops,
    so the forward transform corrupts mounted camera positions and rotations.
    USD matrices use row vectors: local = world * inverse(parent).
    Retain the prim API's WXYZ convention and Warp array/device contract.
    """
    import numpy as np
    import warp as wp
    from pxr import Gf
    import isaacsim.core.utils.warp as backend
    import isaacsim.core.utils.warp.transformations as transformations

    if getattr(transformations.get_local_from_world, "_roboicl_inverse_parent", False):
        return False

    def get_local_from_world(parent_transforms, positions, orientations, device):
        parents = parent_transforms.numpy()
        world_positions = positions.numpy()
        world_orientations = orientations.numpy()
        translations = np.empty((len(world_positions), 3), dtype=np.float32)
        rotations = np.empty((len(world_positions), 4), dtype=np.float32)
        for i, (parent, position, quaternion) in enumerate(
            zip(parents, world_positions, world_orientations)
        ):
            world = Gf.Matrix4d(1)
            world.SetRotate(Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*map(float, quaternion[1:]))))
            world.SetTranslateOnly(Gf.Vec3d(*map(float, position)))
            local = Gf.Transform(world * Gf.Matrix4d(parent.tolist()).GetInverse())
            rotation = local.GetRotation().GetQuat()
            translations[i] = local.GetTranslation()
            rotations[i] = (rotation.GetReal(), *rotation.GetImaginary())
        return (
            wp.from_numpy(translations, device=device),
            wp.from_numpy(rotations, device=device),
        )

    get_local_from_world._roboicl_inverse_parent = True
    # XFormPrim imports the package export; direct callers use the submodule.
    transformations.get_local_from_world = get_local_from_world
    backend.get_local_from_world = get_local_from_world
    return True


def _restore_physics_sim_interface() -> bool:
    """Restore ``SimulationManager._physics_sim_interface`` on PhysxManager.

    Isaac Sim's deprecated prim wrappers call ``flush_changes()`` on this
    interface after writing USD attributes. The interface itself is a plain
    acquire from ``omni.physics.core`` and is independent of which manager
    class owns the timeline, so exposing the same object keeps those writes
    visible to the solver.
    """
    from isaaclab_physx.physics.physx_manager import PhysxManager

    if hasattr(PhysxManager, "_physics_sim_interface"):
        return False
    import omni.physics.core

    PhysxManager._physics_sim_interface = omni.physics.core.get_physics_simulation_interface()
    return True


def _adapt_deprecated_prim_deletion_callback() -> bool:
    """Adapt ``Prim._on_prim_deletion`` to Isaac Lab's event envelope.

    Isaac Lab's event bus dispatches ``IsaacEvents.PRIM_DELETION`` as an
    event object carrying a ``payload`` dict, while Isaac Sim's deprecated
    ``Prim`` class expects the bare ``prim_path`` string. Unwrap both shapes
    so prim views still invalidate when the stage is cleared between resets.
    """
    from isaacsim.core.prims.impl.prim import Prim

    original = Prim._on_prim_deletion
    if getattr(original, "_roboicl_event_payload", False):
        return False

    def _on_prim_deletion(self, payload):
        payload = getattr(payload, "payload", payload)
        if isinstance(payload, dict):
            payload = payload.get("prim_path", "/")
        return original(self, payload)

    _on_prim_deletion._roboicl_event_payload = True
    Prim._on_prim_deletion = _on_prim_deletion
    return True


def _restore_gui_render_predicates() -> dict:
    """Restore ``SimulationContext.has_gui()`` / ``has_rtx_sensors()`` calls.

    Isaac Lab 3.0 turned ``has_gui`` into a read-only property and dropped
    ``has_rtx_sensors`` in favour of ``is_rendering``, while RoboDojo calls
    both as methods. Returning an int-like truthy flag (bool cannot be
    subclassed) for ``has_gui`` keeps Isaac Lab's property reads working and
    lets RoboDojo's call syntax succeed; ``has_rtx_sensors`` reads the same
    setting it did in 2.3.
    """
    from isaaclab.sim import SimulationContext

    report: dict = {"has_gui_callable": False, "has_rtx_sensors_added": False}

    class _CallableFlag(int):
        def __call__(self) -> bool:
            return bool(self)

    original = SimulationContext.__dict__.get("has_gui")
    if isinstance(original, property) and not getattr(original.fget, "_roboicl_callable_bool", False):
        inner = original.fget

        def _fget(self):
            return _CallableFlag(bool(inner(self)))

        _fget._roboicl_callable_bool = True
        SimulationContext.has_gui = property(_fget, original.fset, original.fdel, original.__doc__)
        report["has_gui_callable"] = True

    if "has_rtx_sensors" not in SimulationContext.__dict__:

        def has_rtx_sensors(self) -> bool:
            return bool(self.get_setting("/isaaclab/render/rtx_sensors"))

        has_rtx_sensors._roboicl_added = True
        SimulationContext.has_rtx_sensors = has_rtx_sensors
        report["has_rtx_sensors_added"] = True
    return report


def install_post_launch() -> dict:
    """Restore symbols Isaac Sim 6.x dropped, once the Kit app is running.

    The bindings above must be patched before ``SimulationApp`` starts; this
    covers the contracts that only exist once the extensions are loaded.
    ``omni.physx`` no longer re-exports ``acquire_physx_interface`` (its
    ``scripts.ifaces.__all__`` omits the ``acquire_*`` helpers), but RoboDojo
    calls it to force GPU physics, so re-export it from the surviving binding.
    ``SimulationContext`` is patched here because importing it pulls in
    ``pxr``, which must not be loaded before ``SimulationApp`` starts.
    Idempotent: existing symbols are left untouched.
    """
    import omni.physx

    report: dict = {}
    if not hasattr(omni.physx, "acquire_physx_interface"):
        from omni.physx.bindings._physx import acquire_physx_interface

        omni.physx.acquire_physx_interface = acquire_physx_interface
        report["physx_interface_reexported"] = True
    else:
        report["physx_interface_reexported"] = False
    report["physics_context_restored"] = _restore_physics_context_accessor()
    report["backend_aligned_with_warp_view"] = _align_backend_with_warp_view()
    report["physx_manager_backend_utils_restored"] = _restore_physx_manager_backend_utils()
    report["warp_world_to_local_corrected"] = _restore_warp_world_to_local()
    report["physx_manager_sim_interface_restored"] = _restore_physics_sim_interface()
    report["prim_deletion_payload_adapted"] = _adapt_deprecated_prim_deletion_callback()
    report["gui_render_predicates"] = _restore_gui_render_predicates()
    return report
