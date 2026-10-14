"""End-to-end check: start the server over stdio and call the tools exactly as an MCP client would.

Usage:
    python scripts/smoke_stdio.py <scratch_dir>                     # runs `python -m audio_mcp`
    python scripts/smoke_stdio.py <scratch_dir> -- <server command>  # e.g. -- docker run -i --rm -v ... audio-mcp

Exits non-zero if any call that should succeed fails, or any call that should fail does so with an internal error
instead of a clean, explanatory one. The scratch_dir must be visible to the server at the same path.
"""

import asyncio
import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

failures = []


def text_of(result):
    return next((c.text for c in result.content if c.type == "text"), "")


def expect_ok(name, result, kind="text"):
    kinds = [c.type for c in result.content]
    ok = not result.is_error and kind in kinds
    print(f"--- {name}: {'ok' if ok else 'FAILED'} {kinds}")
    print(text_of(result)[:400])
    if not ok:
        failures.append(name)


def expect_clean_error(name, result, must_contain):
    text = text_of(result)
    ok = result.is_error and must_contain in text and "Internal error" not in text
    print(f"--- {name}: {'clean error as expected' if ok else 'FAILED'}")
    print(text[:300])
    if not ok:
        failures.append(name)


async def main(scratch: Path, command: list[str]):
    scratch.mkdir(parents=True, exist_ok=True)
    zap = str(scratch / "zap.wav")
    zap_hp = str(scratch / "zap_hp.wav")
    params = StdioServerParameters(command=command[0], args=command[1:])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = [t.name for t in (await session.list_tools()).tools]
        print("tools:", tools)
        if len(tools) < 12:
            failures.append(f"expected 12 tools, got {len(tools)}")

        spec = {
            "layers": [
                {
                    "wave": "square",
                    "freq": 1800,
                    "freq_end": 200,
                    "duration": 0.3,
                    "envelope": {"attack": 0.002, "release": 0.12},
                    "lowpass": 6000,
                }
            ]
        }
        expect_ok(
            "synthesize", await session.call_tool("audio_synthesize", {"spec": spec, "output": zap, "overwrite": True})
        )
        expect_ok("analyze", await session.call_tool("audio_analyze", {"path": zap}))
        expect_ok("events", await session.call_tool("audio_events", {"path": zap}))
        expect_ok("view", await session.call_tool("audio_view", {"path": zap}), kind="image")
        ops = [{"op": "highpass", "hz": 300}, {"op": "normalize_peak", "db": -3}]
        expect_ok(
            "process",
            await session.call_tool(
                "audio_process", {"path": zap, "operations": ops, "output": zap_hp, "overwrite": True}
            ),
        )
        expect_ok("compare", await session.call_tool("audio_compare", {"path_a": zap, "path_b": zap_hp}))
        expect_ok("check_delivery", await session.call_tool("audio_check_delivery", {"path": zap, "spec": "game_sfx"}))
        expect_clean_error(
            "unknown spec",
            await session.call_tool("audio_check_delivery", {"path": zap, "spec": "netflix"}),
            "Unknown delivery spec",
        )
        expect_clean_error(
            "bad op",
            await session.call_tool("audio_process", {"path": zap, "operations": [{"op": "louder"}]}),
            "Unknown op 'louder'",
        )
        fake_m4a = scratch / "not_really.m4a"
        fake_m4a.write_bytes(bytes(256))
        expect_clean_error(  # FFmpeg missing, or FFmpeg can't decode it: either way a clean, explanatory error
            "undecodable m4a",
            await session.call_tool("audio_analyze", {"path": str(fake_m4a)}),
            "FFmpeg",
        )
        game = scratch / "godot_game"
        (game / "audio" / "sfx").mkdir(parents=True, exist_ok=True)
        (game / "project.godot").write_text("config_version=5\n")
        (game / "audio" / "sfx" / "zap.wav").write_bytes(Path(zap).read_bytes())
        expect_ok("game project", await session.call_tool("audio_check_game_project", {"project_dir": str(game)}))
        expect_clean_error(
            "not a godot project",
            await session.call_tool("audio_check_game_project", {"project_dir": str(scratch)}),
            "No project.godot",
        )
        expect_clean_error(
            "missing file",
            await session.call_tool("audio_analyze", {"path": str(scratch / "nope.wav")}),
            "File not found",
        )


if __name__ == "__main__":
    # Tool output contains sparkline characters; Windows consoles may default to a legacy code page.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    server_cmd = args[args.index("--") + 1 :] if "--" in args else [sys.executable, "-m", "audio_mcp"]
    asyncio.run(main(Path(args[0]), server_cmd))
    if failures:
        sys.exit(f"\nSMOKE TEST FAILED: {failures}")
    print("\nSmoke test passed.")
