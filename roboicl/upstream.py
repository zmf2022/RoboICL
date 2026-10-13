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


def verify_curobo(interpreter: Path, lock_path: Path | None = None) -> dict:
    """Require the pinned upstream install and RoboDojo's API without loading CUDA."""
    lock = json.loads((lock_path or CODE / 'configs/upstream.lock.json').read_text())
    expected = next(p for p in lock['external_packages'] if p['name'] == 'cuRobo')
    probe_code = """
import json
from importlib.metadata import distribution, PackageNotFoundError
from importlib.util import find_spec
from pathlib import Path
import sys
expected = json.loads(sys.argv[1])
try:
    package = distribution(expected['distribution'])
except PackageNotFoundError:
    print(json.dumps({'installed': False}))
else:
    spec = find_spec(expected['module'])
    root = Path(spec.origin).parent if spec and spec.origin else None
    print(json.dumps({
        'installed': True,
        'version': package.version,
        'provenance': json.loads(package.read_text('direct_url.json') or '{}'),
        'module_path': str(root) if root else None,
        'missing_files': [f for f in expected['required_files'] if root is None or not (root / f).is_file()],
    }))
"""
    probe = subprocess.run(
        [str(interpreter), '-c', probe_code, json.dumps(expected)],
        capture_output=True, text=True, timeout=30,
    )
    if probe.returncode:
        raise RuntimeError(f'Cannot inspect cuRobo in {interpreter}: {probe.stderr.strip()}')
    actual = json.loads(probe.stdout)
    provenance = actual.get('provenance', {})
    vcs = provenance.get('vcs_info', {})
    if (not actual.get('installed') or provenance.get('url') != expected['url']
            or vcs.get('vcs') != 'git' or vcs.get('commit_id') != expected['commit']
            or actual.get('missing_files')):
        raise ValueError(
            f"cuRobo must be installed from {expected['commit']} with RoboDojo's newer API; "
            f"found {actual}. Install setup/requirements-sim.txt in {interpreter}."
        )
    return actual
