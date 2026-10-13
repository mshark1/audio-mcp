"""End-to-end check: start the server as a subprocess over stdio and call every tool like an MCP client would.

Usage: python scripts/smoke_stdio.py <scratch_dir>
"""

import asyncio
import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def show(name, result):
    kinds = [c.type for c in result.content]
    text = next((c.text for c in result.content if c.type == "text"), "")
    status = "ERROR" if result.is_error else "ok"
    print(f"--- {name}: {status} {kinds}")
    print(text[:600] + ("..." if len(text) > 600 else ""))


async def main(scratch: Path):
    scratch.mkdir(parents=True, exist_ok=True)
    zap = str(scratch / "zap.wav")
    params = StdioServerParameters(command=sys.executable, args=["-m", "audio_mcp"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])

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
        show(
            "synthesize", await session.call_tool("audio_synthesize", {"spec": spec, "output": zap, "overwrite": True})
        )
        show("analyze", await session.call_tool("audio_analyze", {"path": zap}))
        view = await session.call_tool("audio_view", {"path": zap})
        show("view", view)
        show(
            "process",
            await session.call_tool(
                "audio_process",
                {
                    "path": zap,
                    "operations": [{"op": "highpass", "hz": 300}, {"op": "normalize_peak", "db": -3}],
                    "output": str(scratch / "zap_hp.wav"),
                    "overwrite": True,
                },
            ),
        )
        show(
            "compare", await session.call_tool("audio_compare", {"path_a": zap, "path_b": str(scratch / "zap_hp.wav")})
        )
        show(
            "bad op (should be a clear error)",
            await session.call_tool("audio_process", {"path": zap, "operations": [{"op": "louder"}]}),
        )
        show(
            "missing file (should be a clear error)",
            await session.call_tool("audio_analyze", {"path": str(scratch / "nope.wav")}),
        )


if __name__ == "__main__":
    # Tool output contains sparkline characters; Windows consoles may default to a legacy code page.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main(Path(sys.argv[1])))
