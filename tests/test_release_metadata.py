"""Release metadata must agree, or the PyPI / MCP Registry publishing steps fail halfway through a release."""

import json
import re
import sys
from pathlib import Path

import pytest

import audio_mcp

ROOT = Path(__file__).resolve().parent.parent

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    tomllib = pytest.importorskip("tomli")


def load():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    server = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    return project, server


def test_versions_match_everywhere():
    project, server = load()
    pypi = next(p for p in server["packages"] if p["registryType"] == "pypi")
    assert project["version"] == audio_mcp.__version__ == server["version"] == pypi["version"]


def test_registry_ownership_marker_matches_server_name():
    project, server = load()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(rf"mcp-name:\s*{re.escape(server['name'])}(\s|-->)", readme)
    assert next(p for p in server["packages"] if p["registryType"] == "pypi")["identifier"] == project["name"]


def test_registry_description_fits():
    _, server = load()
    assert len(server["description"]) <= 100
