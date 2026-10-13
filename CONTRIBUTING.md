# Contributing

Thanks for helping. Contributions from people and from AI agents are equally welcome and held to the same bar.
If you're an AI coding agent, read [AGENTS.md](AGENTS.md) too; it maps the code and lists the rules the
codebase depends on.

## The bar: small and excellent beats big and approximate

This project deliberately stays modest. A feature belongs here if it:

1. **Does one thing reliably**, verified against **known ground truth**: a synthetic signal whose correct answer
   you know by construction, or an independent reference implementation. "It ran and the numbers look plausible"
   isn't enough.
2. **Is honest about uncertainty.** If it can be wrong (pitch on chords, tempo on complex music), it says so in the
   output, through a confidence value, a flag like `polyphony_likely`, or a comment. A confident wrong answer is
   worse than no answer.
3. **Fails helpfully.** Expected problems raise `AudioError` with a message that tells the caller what to do next.
4. **Keeps output compact.** Agents pay for every token: round numbers, cap lists, prefer one-line summaries.

We removed a key-estimation feature for failing (1) on a nursery rhyme. Removing things is a legitimate
contribution.

## Setup

```bash
git clone https://github.com/mshark1/audio-mcp && cd audio-mcp
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add ,speech for audio_transcribe
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
pytest                                    # all tests; speech tests skip without the extra
python scripts/smoke_stdio.py /tmp/smoke  # starts the real server over stdio and calls the tools
```

CI runs these on Linux, macOS and Windows for Python 3.10 to 3.14, plus a run with the oldest supported dependency
versions.

Include in your PR description:
- **What changed and why**, with real tool output before and after where it helps.
- **How it's verified**: which test pins the behaviour to ground truth.
- **Any change to tool output.** Tool names, parameters and output keys are an API that agents rely on, so call out
  renames or removals explicitly and add them to [CHANGELOG.md](CHANGELOG.md).

## Good first contributions

- New ground-truth tests for existing tools, especially cases you suspect are wrong.
- Better `issues` detections in `analysis.find_issues`, each with a constructed test signal.
- New `audio_process` operations (one function in `process.py` decorated with `@op`).
- Accuracy reports: a file where a tool gives the wrong answer, with what the right answer is and how you know.

## Reporting bugs

If a tool returns `Internal error in <tool>: ...`, that's a bug by definition. Please open an issue with the tool
call (arguments), a description of the input file (format, length, content), and the full message. Agents filing
issues on a user's behalf should include the same.
