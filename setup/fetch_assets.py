"""Fetch the complete RoboDojo Assets tree at the repository's frozen revision.

The full tree is intentional: a task-only file list is unsafe until USD and
material dependencies are resolved transitively. Assets remain outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


CODE = Path(__file__).resolve().parents[1]


def command(cache: Path, env: dict[str, str], *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "http.version=HTTP/1.1", "-C", str(cache), *args],
        env=env,
        text=True,
    ).strip()


def materialize_robot_configs(data_root: Path) -> list[str]:
    """Resolve portable cuRobo templates against this data installation."""
    generated = []
    for template in sorted((data_root / "Assets/Robots").glob("*/curobo_tmp.yml")):
        output = template.with_name("curobo.yml")
        content = template.read_text(encoding="utf-8").replace("${ASSETS_PATH}", str(data_root))
        if not output.exists() or output.read_text(encoding="utf-8") != content:
            output.write_text(content, encoding="utf-8")
        generated.append(str(output.relative_to(data_root)))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=CODE / "data")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    lock = json.loads((CODE / "configs/data.lock.json").read_text(encoding="utf-8"))["assets"]
    target = data_root / "Assets"
    data_root.mkdir(parents=True, exist_ok=True)

    if target.exists() or target.is_symlink():
        raise RuntimeError(f"Existing Assets target preserved; use a clean data root or inspect it first: {target}")

    # Git/LFS staging is temporary; the installed Assets directory contains real files.
    with tempfile.TemporaryDirectory(prefix=".assets-download-", dir=data_root) as staging:
        install_assets(data_root, Path(staging), lock)


def install_assets(data_root: Path, cache: Path, lock: dict) -> None:
    target = data_root / "Assets"
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    command(cache, env, "init")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = os.environ.get("ROBODOJO_ASSET_REPO_URL", f"{endpoint}/datasets/{lock['repo_id']}")
    command(cache, env, "remote", "add", "origin", url)

    command(cache, env, "fetch", "--depth=1", "origin", lock["revision"])
    command(cache, env, "sparse-checkout", "init", "--cone")
    command(cache, env, "sparse-checkout", "set", "Assets")
    command(cache, env, "checkout", "--detach", "FETCH_HEAD")
    actual = command(cache, env, "rev-parse", "HEAD")
    if actual != lock["revision"]:
        raise RuntimeError(f"Asset revision mismatch: {actual}")
    command(cache, env, "lfs", "install", "--local")
    subprocess.run(
        ["git", "-c", "http.version=HTTP/1.1", "-C", str(cache), "lfs", "pull",
         "--include=Assets/**", "--exclude="],
        env=env,
        check=True,
    )
    other_entries = [name for name in command(cache, env, "ls-tree", "--name-only", "HEAD").splitlines()
                     if name != "Assets"]
    excluded = ",".join(other_entries + [f"{name}/**" for name in other_entries])
    command(cache, env, "-c", f"lfs.fetchexclude={excluded}", "lfs", "fsck")

    for name in ("Robots", "Object", "Room", "Material", "Sensor", "Eval_Layout"):
        if not (cache / "Assets" / name).is_dir():
            raise RuntimeError(f"Frozen asset tree is incomplete: Assets/{name}")
    tree = command(cache, env, "ls-tree", "-r", "HEAD", "Assets")
    (cache / "Assets").rename(target)
    generated = materialize_robot_configs(data_root)
    manifest = {
        "schema": "roboicl.assets.install.v1",
        "repo_id": lock["repo_id"],
        "revision": actual,
        "scope": "complete-tree",
        "include": lock["include"],
        "assets_tree_sha256": hashlib.sha256(tree.encode("utf-8")).hexdigest(),
        "assets": str(target),
        "generated": generated,
    }
    (data_root / ".roboicl-assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
