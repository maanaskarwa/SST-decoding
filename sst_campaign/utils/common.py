"""Small SST campaign utility helpers for timestamping, parsing run output, and running Python subprocesses with captured metadata."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_DIR_PREFIX = 'RUN_OUTPUT_DIR_JSON '


def timestamp_slug() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def parse_run_output_dir(stdout: str) -> str | None:
    for line in stdout.splitlines():
        if line.startswith(RUN_DIR_PREFIX):
            payload = json.loads(line[len(RUN_DIR_PREFIX) :])
            return str(payload['run_dir'])
    return None


def emit_run_output_dir(run_dir: Path) -> None:
    """Emit structured run-dir metadata for subprocess output parsing."""
    print(RUN_DIR_PREFIX + json.dumps({'run_dir': str(run_dir)}), flush=True)


def run_python_script(
    script: str,
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    merged_env = os.environ.copy()
    merged_env.setdefault('MPLBACKEND', 'Agg')
    merged_env.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    if env:
        merged_env.update(env)

    command = [sys.executable, script, *args]
    started_at = datetime.now(timezone.utc).isoformat()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    ended_at = datetime.now(timezone.utc).isoformat()
    stdout = proc.stdout or ''
    stderr = proc.stderr or ''
    return {
        'command': command,
        'reproducible_command': ' '.join(['uv', 'run', 'python', script, *args]),
        'returncode': int(proc.returncode),
        'stdout': stdout,
        'stderr': stderr,
        'run_dir': parse_run_output_dir(stdout),
        'started_at_utc': started_at,
        'ended_at_utc': ended_at,
    }
