#!/usr/bin/env python3
"""
Chat API for the stocks MCP client.

Exposes POST /chat so a React (or any) frontend can send messages
and get agent replies. Keeps the MCP server connection open for the app lifetime.

Run from repo root or patterns/stocks-mcp:
  uv run chat_api.py   # or: .venv/bin/python api/chat_api.py
"""

import sys
from pathlib import Path

# Ensure we can import client logic and agents
_API_DIR = Path(__file__).resolve().parent
_STOCKS_MCP_ROOT = _API_DIR.parent
sys.path.insert(0, str(_STOCKS_MCP_ROOT))
_SERVER_SCRIPT = _STOCKS_MCP_ROOT / "server" / "server.py"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents import Agent, Runner  # pylint: disable=import-error
from agents.mcp import MCPServerStdio  # pylint: disable=import-error
import uvicorn  # pylint: disable=import-error

app = FastAPI(title="Stocks MCP Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.on_event("startup")
async def startup():
    if not _SERVER_SCRIPT.exists():
        raise RuntimeError(f"Server script not found: {_SERVER_SCRIPT}")

    stdio_server = MCPServerStdio(
        name="Stocks MCP",
        params={
            "command": sys.executable,
            "args": [str(_SERVER_SCRIPT)],
        },
        cache_tools_list=True,
    )
    await stdio_server.__aenter__()
    app.state.mcp_context = stdio_server
    app.state.agent = Agent(
        name="StocksAssistant",
        instructions="Use the MCP tools to answer stock and market questions. Use get_quote for current price and session data, get_info for sector and company summary.",
        mcp_servers=[stdio_server],
    )
    app.state.runner = Runner


@app.on_event("shutdown")
async def shutdown():
    ctx = getattr(app.state, "mcp_context", None)
    if ctx is not None:
        await ctx.__aexit__(None, None, None)
        app.state.mcp_context = None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Handle a chat message and return the agent's reply using the stocks MCP tools."""
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    agent = getattr(app.state, "agent", None)
    runner = getattr(app.state, "runner", None)
    if not agent or not runner:
        raise HTTPException(status_code=503, detail="Agent not ready")
    try:
        result = await runner.run(agent, req.message.strip())
        return ChatResponse(reply=result.final_output or "(No response)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health():
    return {"status": "ok", "agent_ready": hasattr(app.state, "agent") and app.state.agent is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
