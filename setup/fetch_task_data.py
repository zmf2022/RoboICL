#!/usr/bin/env python3
"""Prepare one verified source episode per published task and all locked references."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from roboicl.config import reference_path, task_names
from roboicl.reference_setting import load_reference_setting


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(url: str) -> list:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def download(url: str, target: Path, size: int, digest: str) -> None:
    if target.exists():
        if target.stat().st_size != size or sha256(target) != digest:
            raise ValueError(f"Existing source preserved but verification failed: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset < size:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=120) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Download server returned a mismatched Content-Range")
                    with partial.open("ab" if append else "wb") as stream:
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
            if partial.stat().st_size != size or sha256(partial) != digest:
                raise ValueError(f"Partial source preserved but size/SHA256 failed: {partial}")
            partial.replace(target)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def write_index(inventory: dict, runtime: Path) -> None:
    labels = {
        "reference_ready": "ICL 已校验：shots=1",
        "source_verified": "原始示范已校验：shots=0",
        "source_verified_reference_unavailable": "公开示范已校验；历史 ICL 不可用：shots=0",
        "not_published": "未发布示范：shots=0",
        "pending": "准备中",
        "failed": "准备失败，见 JSON 清单",
    }
    lines = ["# 任务经验数据索引", "", f"数据根目录：`{inventory['data_root']}`。",
             ""]
    verified = [row for row in inventory["tasks"].values() if row["status"] in (
        "reference_ready", "source_verified", "source_verified_reference_unavailable")]
    refs = sum(row["status"] == "reference_ready" for row in verified)
    gib = sum(row["source_bytes"] for row in verified) / 1024**3
    lines += [f"已校验原始示范：{len(verified)} 个任务，合计 {gib:.2f} GiB；ICL reference 就绪：{refs} 个任务。",
             "", "每个有公开数据的任务准备一条示范。原始 HDF5 和可注入模型的 reference 是不同文件。",
             "", "|任务|准备状态与运行协议|原始经验数据|ICL reference|", "|---|---|---|---|"]
    for task, row in inventory["tasks"].items():
        ready = row["status"] in ("reference_ready", "source_verified", "source_verified_reference_unavailable")
        source = f"[HDF5]({row['source_file']})" if ready else "—"
        reference = (f"[bundle]({row['reference']}/train_reference_bundle.json)"
                     if row["status"] == "reference_ready" else "—")
        lines.append(f"|`{task}`|{labels[row['status']]}|{source}|{reference}|")
    lines += ["", "固定源版本、SHA256、文件字节数、历史源不可用原因见 `task_data_inventory.json`。", ""]
    temporary = runtime / "README.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(runtime / "README.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=CODE / "data")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--hf-cli", default=shutil.which("hf"),
                        help="Official hf executable used when the legacy download returns HTTP 403")
    parser.add_argument("--download-backend", choices=("http", "hub"), default="http",
                        help="Use official Hub/Xet transfers instead of sequential HTTP downloads")
    args = parser.parse_args()
    if args.download_backend == "hub" and not args.hf_cli:
        parser.error("--download-backend hub requires an installed hf client or --hf-cli")
    root = args.data_root.expanduser().resolve()
    runtime = root / "runtime-data"
    runtime.mkdir(parents=True, exist_ok=True)
    setting = load_reference_setting("configs/references/one_shot_j12_b12.json")
    repository = setting["source_repository"]
    repo_id, revision = repository["repo_id"], repository["revision"]
    endpoint = args.endpoint.rstrip("/")
    locked = {row["task"]: row for row in setting["one_shot_references"]}
    directories = read_json(f"{endpoint}/api/datasets/{repo_id}/tree/{revision}/data/RoboDojo?limit=1000")
    published = {Path(row["path"]).name for row in directories if row["type"] == "directory"}
    inventory = {
        "schema": "roboicl.task_data_inventory.v1", "repository": repo_id,
        "revision": revision, "data_root": str(root), "tasks": {},
    }

    def discover(task: str) -> dict:
        if task not in published:
            return {"status": "not_published", "protocol": "zero_shot",
                    "reason": "No task directory in the pinned official dataset."}
        lock = locked.get(task)
        source_revision = lock["source"].get("repository_revision", revision) if lock else revision
        folder = f"data/RoboDojo/{task}/arx_x5/data"
        files = read_json(f"{endpoint}/api/datasets/{repo_id}/tree/{source_revision}/{folder}?limit=1000")
        source_path = lock["source"]["repository_path"] if lock else f"{folder}/episode_0000000.hdf5"
        matches = [row for row in files if row["type"] == "file" and row["path"] == source_path]
        if len(matches) != 1:
            raise ValueError(f"No unique source episode for {task}: {source_path}")
        source = matches[0]
        digest = source.get("lfs", {}).get("oid")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Source metadata lacks SHA256 for {task}")
        if lock and (digest != lock["source"]["sha256"]
                     or lock["source"].get("size", source["size"]) != source["size"]):
            raise ValueError(f"Official source metadata differs from locked reference: {task}")
        install = lock["source"]["install_path"] if lock else f"runtime-data/{task}/train/episode_0000000.hdf5"
        return {"status": "pending", "protocol": "one_shot" if lock else "zero_shot",
                "source_revision": source_revision, "repository_path": source_path,
                "source_file": str(root / install), "source_sha256": digest,
                "source_bytes": source["size"], "locked_reference": bool(lock)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(discover, task): task for task in task_names()}
        for future in as_completed(futures):
            task = futures[future]
            try:
                inventory["tasks"][task] = future.result()
            except Exception as exc:
                inventory["tasks"][task] = {"status": "failed", "error": str(exc)}
    inventory["tasks"] = dict(sorted(inventory["tasks"].items()))
    report = runtime / "task_data_inventory.json"

    def save() -> None:
        temporary = report.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(report)
        write_index(inventory, runtime)

    save()
    jobs = [(task, row) for task, row in inventory["tasks"].items() if row["status"] == "pending"]
    needed = sum(max(0, row["source_bytes"] - (Path(row["source_file"]).stat().st_size
                 if Path(row["source_file"]).exists() else 0)) for _, row in jobs)
    if shutil.disk_usage(root).free < needed + 2 * 1024**3:
        raise RuntimeError(f"Insufficient space for {needed} source bytes plus reference bundles")
    print(f"Published tasks: {len(jobs)}; source bytes to download: {needed}; workers: {args.workers}", flush=True)

    def hub_transfer(row: dict, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        hub_root = root / ".hub-downloads"
        environment = dict(os.environ, HF_ENDPOINT=endpoint,
                           HF_XET_CACHE=str(root / ".hf-xet-cache"))
        with (target.parent / "hub_download.log").open("w") as stream:
            subprocess.run([args.hf_cli, "download", repo_id, row["repository_path"],
                            "--repo-type", "dataset", "--revision", row["source_revision"],
                            "--local-dir", str(hub_root), "--quiet"],
                           env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        fetched = hub_root / row["repository_path"]
        if fetched.stat().st_size != row["source_bytes"] or sha256(fetched) != row["source_sha256"]:
            raise ValueError(f"Hub source failed size/SHA256 verification: {fetched}")
        fetched.replace(target)

    def public_fallback(task: str, row: dict, reason: str) -> dict:
        folder = f"data/RoboDojo/{task}/arx_x5/data"
        files = read_json(f"{endpoint}/api/datasets/{repo_id}/tree/{revision}/{folder}?limit=1000")
        source = next(item for item in files if item["path"] == row["repository_path"])
        digest = source.get("lfs", {}).get("oid")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Public source metadata lacks SHA256 for {task}")
        target = runtime / task / "public" / "train" / Path(source["path"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        cached = root / ".hub-downloads" / source["path"]
        if (not target.exists() and cached.exists() and cached.stat().st_size == source["size"]
                and sha256(cached) == digest):
            cached.replace(target)
        if args.download_backend == "hub" and not target.exists():
            hub_transfer(dict(source_revision=revision, repository_path=source["path"],
                              source_bytes=source["size"], source_sha256=digest), target)
        else:
            download(f"{endpoint}/datasets/{repo_id}/resolve/{revision}/{source['path']}",
                     target, source["size"], digest)
        return dict(row, status="source_verified_reference_unavailable", protocol="zero_shot",
                    source_file=str(target), source_sha256=digest, source_bytes=source["size"],
                    source_revision=revision, reference_error=reason,
                    locked_source={key: row[key] for key in (
                        "source_file", "source_sha256", "source_bytes", "source_revision")})

    def prepare(task: str, row: dict) -> dict:
        target = Path(row["source_file"])
        url = f"{endpoint}/datasets/{repo_id}/resolve/{row['source_revision']}/{row['repository_path']}"
        try:
            if args.download_backend == "hub" and not target.exists():
                try:
                    hub_transfer(row, target)
                except subprocess.CalledProcessError:
                    if row["locked_reference"] and row["source_revision"] != revision:
                        return public_fallback(task, row, "Locked historical source unavailable via official Hub.")
                    raise
            else:
                download(url, target, row["source_bytes"], row["source_sha256"])
        except urllib.error.HTTPError as exc:
            if exc.code != 403:
                raise
            if not args.hf_cli:
                if row["locked_reference"] and row["source_revision"] != revision:
                    return public_fallback(task, row, "Locked historical source returned HTTP 403.")
                raise
            try:
                hub_transfer(row, target)
            except subprocess.CalledProcessError:
                if row["locked_reference"] and row["source_revision"] != revision:
                    return public_fallback(task, row, "Locked historical source returned HTTP 403; official Hub retry failed.")
                raise
        result = dict(row, status="source_verified")
        if row["locked_reference"]:
            log = runtime / task / "prepare_reference.log"
            with log.open("w") as stream:
                subprocess.run([sys.executable, str(CODE / "setup/fetch_reference.py"), task,
                                "--data-root", str(root), "--endpoint", endpoint],
                               cwd=CODE, stdout=stream, stderr=subprocess.STDOUT, check=True)
            bundle_root = reference_path(root, task, locked[task]["reference_horizon"], 12)
            result.update(status="reference_ready", reference=str(bundle_root),
                          train_chunk_count=12, verification=str(bundle_root / "verification.json"))
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare, task, row): task for task, row in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                inventory["tasks"][task] = future.result()
            except Exception as exc:
                inventory["tasks"][task].update(status="failed", error=str(exc))
            save()
            print(f"[{index}/{len(jobs)}] {task}: {inventory['tasks'][task]['status']}", flush=True)
    failed = [task for task, row in inventory["tasks"].items() if row["status"] == "failed"]
    print(f"Inventory: {report}; failures: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
