"""Command line: `audio-mcp` (no args) runs the MCP server over stdio; subcommands run tools directly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audio_io import AudioError


def _json_arg(value: str):
    p = Path(value)
    return json.loads(p.read_text(encoding="utf-8") if p.is_file() else value)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="audio-mcp", description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("serve", help="run the MCP server over stdio (default)")

    a = sub.add_parser("analyze", help="full measurement report")
    a.add_argument("path")
    a.add_argument("--start", type=float)
    a.add_argument("--end", type=float)
    a.add_argument("--fast", action="store_true", help="skip pitch and rhythm")

    t = sub.add_parser("timeline", help="per-segment levels over time (streams; any length)")
    t.add_argument("path")
    t.add_argument("--segments", type=int, default=24)

    v = sub.add_parser("view", help="render waveform/spectrogram PNG")
    v.add_argument("path")
    v.add_argument("out")
    v.add_argument("--kind", default="both")
    v.add_argument("--start", type=float)
    v.add_argument("--end", type=float)
    v.add_argument("--linear", action="store_true")
    v.add_argument("--pitch", action="store_true")

    m = sub.add_parser("melody", help="transcribe a monophonic line to notes")
    m.add_argument("path")
    m.add_argument("--start", type=float)
    m.add_argument("--end", type=float)

    c = sub.add_parser("compare", help="A/B two files")
    c.add_argument("a")
    c.add_argument("b")

    p = sub.add_parser("process", help="apply an operations list (JSON string or file)")
    p.add_argument("path")
    p.add_argument("operations", type=_json_arg)
    p.add_argument("--out")
    p.add_argument("--overwrite", action="store_true")

    s = sub.add_parser("synth", help="render a synth spec (JSON string or file)")
    s.add_argument("spec", type=_json_arg)
    s.add_argument("out")
    s.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd in (None, "serve"):
        from .server import main as serve

        serve()
        return

    from . import server

    sys.stdout.reconfigure(encoding="utf-8")
    try:
        if args.cmd == "analyze":
            result = server.audio_analyze(args.path, args.start, args.end, pitch=not args.fast, rhythm=not args.fast)
        elif args.cmd == "timeline":
            result = server.audio_timeline(args.path, args.segments)
        elif args.cmd == "view":
            server.audio_view(
                args.path,
                args.kind,
                args.start,
                args.end,
                "linear" if args.linear else "log",
                None,
                args.pitch,
                args.out,
            )
            result = {"image": str(Path(args.out).resolve())}
        elif args.cmd == "melody":
            result = server.audio_melody(args.path, args.start, args.end)
        elif args.cmd == "compare":
            result = server.audio_compare(args.a, args.b)
        elif args.cmd == "process":
            result = server.audio_process(args.path, args.operations, args.out, args.overwrite)
        else:
            result = server.audio_synthesize(args.spec, args.out, args.overwrite)
    except AudioError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
