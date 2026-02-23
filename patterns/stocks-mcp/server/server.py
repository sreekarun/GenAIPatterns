#!/usr/bin/env python3
"""
Stocks MCP Server

MCP server that exposes stock data from Yahoo Finance (yfinance).
Tools: get_quote, get_info. Runs over stdio.

Usage:
    python server.py

The client (OpenAI Agents SDK) spawns this process via MCPServerStdio.
"""

from mcp import McpError
from mcp.server.fastmcp import FastMCP
from mcp.types import ErrorData
import json
import logging
import asyncio
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger(__name__)

server = FastMCP("stocks-mcp-server")

try:
    import yfinance as yf
except ImportError:
    yf = None


def _ensure_yfinance() -> None:
    """Raise McpError if yfinance is not installed."""
    if yf is None:
        raise McpError(ErrorData(
            code=-32603,
            message="yfinance is not installed; pip install yfinance",
            data={},
        ))


@server.tool()
async def get_quote(symbol: str) -> str:
    """Get current quote for a stock symbol (price, open, high, low, volume).

    Args:
        symbol: Stock ticker symbol (e.g. AAPL, MSFT, GOOGL).

    Returns:
        JSON string with current price and session OHLCV data.
    """
    if not symbol or not str(symbol).strip():
        raise McpError(ErrorData(
            code=-32602,
            message="Symbol is required",
            data={"parameter": "symbol"},
        ))
    _ensure_yfinance()
    sym = str(symbol).strip().upper()
    try:
        ticker = yf.Ticker(sym)
        info = getattr(ticker, "fast_info", None)
        price = open_ = high = low = vol = None
        if info is not None:
            price = getattr(info, "last_price", None)
            open_ = getattr(info, "open", None)
            high = getattr(info, "day_high", None)
            low = getattr(info, "day_low", None)
            vol = getattr(info, "last_volume", None)
        if price is None:
            full_info = ticker.info or {}
            price = full_info.get("regularMarketPrice") or full_info.get("currentPrice")
            open_ = open_ or full_info.get("regularMarketOpen") or full_info.get("open")
            high = high or full_info.get("dayHigh")
            low = low or full_info.get("dayLow")
            vol = vol or full_info.get("volume")
        result = {
            "symbol": sym,
            "price": price,
            "open": open_,
            "high": high,
            "low": low,
            "volume": vol,
        }
        result = {k: v for k, v in result.items() if v is not None}
        if not result or result.get("price") is None:
            raise McpError(ErrorData(
                code=-32602,
                message=f"No quote data found for symbol: {sym}",
                data={"symbol": sym},
            ))
        logger.info("get_quote %s -> %s", sym, result)
        return json.dumps(result, indent=2)
    except McpError:
        raise
    except Exception as e:
        logger.exception("get_quote failed for %s", sym)
        raise McpError(ErrorData(
            code=-32603,
            message=f"Failed to get quote for {sym}",
            data={"error": str(e)},
        )) from e


@server.tool()
async def get_info(symbol: str) -> str:
    """Get summary info for a stock (sector, market cap, description).

    Args:
        symbol: Stock ticker symbol (e.g. AAPL, MSFT).

    Returns:
        JSON string with sector, marketCap, short description.
    """
    if not symbol or not str(symbol).strip():
        raise McpError(ErrorData(
            code=-32602,
            message="Symbol is required",
            data={"parameter": "symbol"},
        ))
    _ensure_yfinance()
    sym = str(symbol).strip().upper()
    try:
        ticker = yf.Ticker(sym)
        info = ticker.info
        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            raise McpError(ErrorData(
                code=-32602,
                message=f"No info found for symbol: {sym}",
                data={"symbol": sym},
            ))
        # Keep it short: sector, marketCap, short description
        result = {
            "symbol": sym,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "shortName": info.get("shortName"),
            "longName": info.get("longName"),
            "summary": (info.get("longBusinessSummary") or "")[:500],
        }
        result = {k: v for k, v in result.items() if v is not None}
        logger.info("get_info %s -> keys %s", sym, list(result.keys()))
        return json.dumps(result, indent=2)
    except McpError:
        raise
    except Exception as e:
        logger.exception("get_info failed for %s", sym)
        raise McpError(ErrorData(
            code=-32603,
            message=f"Failed to get info for {sym}",
            data={"error": str(e)},
        )) from e


async def main_async() -> None:
    """Run the MCP server asynchronously."""
    logger.info("Starting Stocks MCP Server (stdio)...")
    logger.info("Tools: get_quote, get_info")
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error("Server error: %s", e)
        raise


def main() -> None:
    """Run the MCP server over stdio (used when launched by the client or chat API)."""
    try:
        # Check if there's already a running event loop (cloud environment)
        try:
            loop = asyncio.get_running_loop()
            # Event loop already exists, run as a coroutine
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.submit(asyncio.run, main_async())
        except RuntimeError:
            # No event loop, create one
            asyncio.run(main_async())
    except Exception as e:
        logger.error("Server error: %s", e)
        raise


if __name__ == "__main__":
    main()
