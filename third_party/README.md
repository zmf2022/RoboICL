# Bundled dependency sources

These directories are ordinary files in the RoboICL repository. They share the
main repository's history and may be edited directly. No submodule initialization
is required. Original licenses and copyright notices are retained in each tree.

`configs/upstream.lock.json` records the imported upstream revisions and required
entry points. It records provenance rather than forbidding local source changes.
The simulator target is Python 3.12 with Isaac Sim wheel version 6.0.1.0.

- `RoboDojo`: benchmark environments, task definitions, and evaluation client.
- `XPolicyLab`: shared policy transport and data helpers.
- `IsaacLab`: shared simulator framework and PhysX backend.
- `curobo`: shared kinematics and motion planning library.

RoboDojo's nested XPolicyLab, IsaacLab, and cuRobo checkouts are omitted. Import
paths in `roboicl/paths.py` select the four shared directories. The imported
tracked files are retained, including upstream docs, tests, and licenses;
generated files and upstream Git metadata are excluded.

Vendoring changes:

- Git submodule declarations and nested Git metadata are omitted.
- Git LFS filters in dependency `.gitattributes` files are removed. The already
  materialized dependency assets are tracked as ordinary Git blobs, so cloning
  does not depend on uploading them to a separate LFS server. Benchmark assets
  under `data/` still use the separate asset fetch workflow.
- cuRobo's package version is fixed to `0.8.0.post30` for the imported revision,
  so package builds do not depend on an upstream Git checkout or the main
  repository's tags.

Legacy RoboDojo / Isaac Lab 3.0 API adaptation lives in
`roboicl/isaaclab_compat.py`. Deprecated Isaac Sim extensions are registered with
Kit's `--ext-folder` argument instead of writing to the simulator installation.
