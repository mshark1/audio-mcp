## What and why

<!-- What changed, and the problem it solves. Real tool output before/after is very welcome. -->

## How it's verified

<!-- Which test pins this to ground truth (a constructed signal or a reference implementation)? -->

## Checklist

- [ ] `pytest` passes and `ruff check . && ruff format --check .` are clean
- [ ] New behaviour has a ground-truth test; new tools/ops are covered in `tests/test_contract.py`
- [ ] Ran `python scripts/smoke_stdio.py <tmpdir>` if `server.py` changed
- [ ] Changes to tool names, parameters or output keys are listed in `CHANGELOG.md`
- [ ] Written by / with an AI agent? Say which one; it helps reviewers (and is welcome)
