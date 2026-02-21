#!/usr/bin/env python3
"""
Stocks MCP Client (OpenAI Agents SDK)

Connects to the stocks MCP server via stdio and uses an OpenAI agent
to answer stock questions. The client spawns the server automatically;
you do not need to run the server in a separate terminal.

Usage:
    python client.py "What is the current price of AAPL?"
    python client.py   # uses a default question

Requires: OPENAI_API_KEY in the environment.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Resolve path to server script (same repo: patterns/stocks-mcp/server/server.py)
_CLIENT_DIR = Path(__file__).resolve().parent
_STOCKS_MCP_ROOT = _CLIENT_DIR.parent
_SERVER_SCRIPT = _STOCKS_MCP_ROOT / "server" / "server.py"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Stocks MCP client (OpenAI Agents SDK)")
    parser.add_argument(
        "query",
        nargs="?",
        default="What is the current price of AAPL?",
        help="Question to ask about stocks (default: AAPL price)",
    )
    args = parser.parse_args()

    if not _SERVER_SCRIPT.exists():
        print(f"Server script not found: {_SERVER_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    # openai-agents package provides the "agents" module (use patterns/stocks-mcp/.venv for IDE)
    from agents import Agent, Runner  # pylint: disable=import-error
    from agents.mcp import MCPServerStdio  # pylint: disable=import-error

    async with MCPServerStdio(
        name="Stocks MCP",
        params={
            "command": sys.executable,
            "args": [str(_SERVER_SCRIPT)],
        },
        cache_tools_list=True,
    ) as server:
        agent = Agent(
            name="StocksAssistant",
            instructions="Use the MCP tools to answer stock and market questions. Use get_quote for current price and session data, get_info for sector and company summary.",
            mcp_servers=[server],
        )
        result = await Runner.run(agent, args.query)
        print(result.final_output or "(No response)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
