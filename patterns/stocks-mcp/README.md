# Stocks MCP (OpenAI Agents SDK)

A minimal MCP server that returns stock information from publicly available feeds (Yahoo Finance via yfinance), and an OpenAI Agents SDK client that connects to it via stdio.

## Layout

- **server/server.py** — MCP server with tools `get_quote` and `get_info` (yfinance, stdio).
- **client/client.py** — OpenAI Agents SDK client: spawns the server and asks the model to answer stock questions using MCP tools.
- **api/chat_api.py** — FastAPI backend that exposes `POST /chat` for the React chat UI.
- **chat-ui/** — React (Vite) chat client to test the stocks MCP in the browser.

## Prerequisites

- Python 3.10+
- OpenAI API key (for the client only)

## Setup

1. **Install dependencies**

   Recommended (venv; fixes IDE "Unable to import 'mcp'" and keeps deps local):

   ```bash
   cd patterns/stocks-mcp && python -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```

   Then set your IDE/editor Python interpreter to `patterns/stocks-mcp/.venv`.

   Or install into your current environment:

   ```bash
   pip install -r patterns/stocks-mcp/requirements.txt
   ```

2. **Set your OpenAI API key** (required for the client):

   ```bash
   export OPENAI_API_KEY=sk-...
   ```

## Run

**Recommended (single command):** Run only the client. It automatically starts the MCP server as a subprocess; you do not need to run the server in a separate terminal.

From the repo root:

```bash
python patterns/stocks-mcp/client/client.py "What is the current price of AAPL?"
```

Or with a default question:

```bash
python patterns/stocks-mcp/client/client.py
```

The client uses the OpenAI Agents SDK (`Agent`, `Runner`, `MCPServerStdio`) to run the stocks server and get answers from the model using the MCP tools.

## Server tools

- **get_quote(symbol)** — Current price and basic quote data (open, high, low, volume).
- **get_info(symbol)** — Summary info (sector, market cap, short description).

Data is fetched from Yahoo Finance via the yfinance library (no API key required).

## React Chat UI (test in browser)

1. **Start the chat API** (from repo root or `patterns/stocks-mcp`):

   ```bash
   cd patterns/stocks-mcp
   .venv/bin/pip install -r requirements.txt   # if not already
   export OPENAI_API_KEY=sk-...
   .venv/bin/python api/chat_api.py
   ```

   The API runs at `http://localhost:8000`. It starts the MCP server internally and keeps it connected.

2. **Start the React chat UI** (in a second terminal):

   ```bash
   cd patterns/stocks-mcp/chat-ui
   npm install
   npm run dev
   ```

   Open `http://localhost:5173` in your browser. The UI proxies `/chat` and `/health` to the API on port 8000.

3. Type a question (e.g. “What is the current price of AAPL?”) and click Send. The assistant reply will appear in the chat.
