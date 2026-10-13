# Releasing

1. Update `version` in `pyproject.toml`, `src/audio_mcp/__init__.py` and both places in `server.json` (a test checks
   they match), and move the `[Unreleased]` notes in
   `CHANGELOG.md` under the new version with today's date.
2. Make sure CI is green on `main`.
3. Create a GitHub release with tag `vX.Y.Z` (it must match the version). `.github/workflows/release.yml` builds,
   checks and publishes to PyPI (after you approve the `pypi` deployment), then updates the listing in the official
   MCP Registry (`io.github.mshark1/audio-mcp`) using GitHub OIDC, with no secrets involved.

## One-time PyPI setup (trusted publishing, no tokens)

1. On pypi.org, go to *Your projects → Publishing → Add a new pending publisher*: project `audio-mcp`, owner/repo
   = this GitHub repo, workflow `release.yml`, environment `pypi`.
2. In the GitHub repo, go to *Settings → Environments → New environment* `pypi` (optionally require a reviewer).

To try a release first, publish to TestPyPI by adding `repository-url: https://test.pypi.org/legacy/` to the
publish step and registering the same pending publisher on test.pypi.org.
