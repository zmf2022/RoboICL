"""Validate bundled sources and the simulator version without starting Kit."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from roboicl.paths import CODE, configure_imports


def verify_sources(lock_path: Path | None = None) -> dict:
    """Check entry points; upstream commits record provenance, not Git state."""
    lock = json.loads((lock_path or CODE / 'configs/upstream.lock.json').read_text())
    checked = []
    for package in lock['packages']:
        path = CODE / package['path']
        if not path.is_dir():
            raise FileNotFoundError(f'Missing bundled source: {path}; restore it from the main repository')
        if (path / '.git').exists() or (path / '.gitmodules').exists():
            raise ValueError(f'Bundled source must not contain nested Git metadata: {path}')
        for entry in package['required_files']:
            if not (path / entry).is_file():
                raise FileNotFoundError(f'Missing bundled source file: {path / entry}')
        checked.append({'name': package['name'], 'path': package['path'],
                        'upstream_commit': package['commit'], 'storage': 'vendored'})
    return {'schema': lock['schema'], 'packages': checked}


def verify_simulator(interpreter: Path, lock_path: Path | None = None) -> dict:
    """Read wheel metadata without importing Isaac Sim or creating a GPU context."""
    lock = json.loads((lock_path or CODE / 'configs/upstream.lock.json').read_text())
    expected = lock['simulator']
    probe = subprocess.run(
        [str(interpreter), '-c', "import json,sys; from importlib.metadata import version; "
         "print(json.dumps({'python': '.'.join(map(str,sys.version_info[:2])), "
         "'isaacsim': version('isaacsim')}))"],
        capture_output=True, text=True, timeout=30,
    )
    if probe.returncode:
        raise RuntimeError(f'Cannot inspect simulator environment {interpreter}: {probe.stderr.strip()}')
    actual = json.loads(probe.stdout)
    for key in ('python', 'isaacsim'):
        if actual[key] != expected[key]:
            raise ValueError(f'Simulator {key} is {actual[key]}; expected {expected[key]} in {interpreter}')
    return actual


def activate() -> None:
    configure_imports()
