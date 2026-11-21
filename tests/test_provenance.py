"""Tests for the provenance record written next to benchmark results."""

import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from provenance import record  # noqa: E402

from pare import __version__


def test_record_names_the_code_and_environment():
    info = record("pytest")
    assert info["pare_version"] == __version__
    assert info["torch"] == torch.__version__
    assert info["pytest"] is not None
    assert "transformers" in info and "datasets" in info
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", info["timestamp_utc"])


def test_record_reports_the_git_commit_in_a_checkout():
    info = record()
    if info["git_commit"] is None:
        # Installed outside a git checkout; nothing more to check.
        assert info["git_dirty"] is None
        return
    assert re.fullmatch(r"[0-9a-f]{40}", info["git_commit"])
    assert isinstance(info["git_dirty"], bool)
