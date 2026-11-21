"""Record which code and environment produced a benchmark result.

Each benchmark script stores the returned dictionary next to its results, so
a number in a results file can be traced to the Pare version, the git commit
and the library versions that produced it.

Usage::

    from provenance import record
    results["_provenance"][tag] = record("lm_eval")
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def record(*extra_packages: str) -> dict:
    """Return the Pare version, git state, timestamp, GPU and library versions.

    Args:
        extra_packages: Names of further installed packages to record, in
                        addition to torch, transformers and datasets.
    """
    import torch

    from pare import __version__

    commit = _git("rev-parse", "HEAD")
    info: dict = {
        "pare_version": __version__,
        "git_commit": commit,
        "git_dirty": bool(_git("status", "--porcelain")) if commit else None,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }
    for package in ("transformers", "datasets", *extra_packages):
        info[package] = _installed_version(package)
    return info


def _git(*args: str) -> str | None:
    """Run a git command in the repository root; None outside a checkout."""
    try:
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _installed_version(package: str) -> str | None:
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None
