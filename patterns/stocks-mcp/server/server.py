#!/usr/bin/env python3
"""
Stocks MCP Server

MCP server that exposes stock data from Yahoo Finance (yfinance).
Tools: get_quote, get_info.

Usage:
    python server.py
"""

from fastmcp import FastMCP
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger(__name__)

mcp = FastMCP("stocks-mcp-server")

try:
    import yfinance as yf
except ImportError:
    yf = None


@mcp.tool
def get_quote(symbol: str) -> str:
    """Get current quote for a stock symbol (price, open, high, low, volume).

    Args:
        symbol: Stock ticker symbol (e.g. AAPL, MSFT, GOOGL).

    Returns:
        JSON string with current price and session OHLCV data.
    """
    if not symbol or not str(symbol).strip():
        return json.dumps({"error": "Symbol is required"})
    
    if yf is None:
        return json.dumps({"error": "yfinance is not installed; pip install yfinance"})
    
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
            return json.dumps({"error": f"No quote data found for symbol: {sym}"})
        
        logger.info("get_quote %s -> %s", sym, result)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.exception("get_quote failed for %s", sym)
        return json.dumps({"error": f"Failed to get quote for {sym}: {str(e)}"})


@mcp.tool
def get_info(symbol: str) -> str:
    """Get summary info for a stock (sector, market cap, description).

    Args:
        symbol: Stock ticker symbol (e.g. AAPL, MSFT).

    Returns:
        JSON string with sector, marketCap, short description.
    """
    if not symbol or not str(symbol).strip():
        return json.dumps({"error": "Symbol is required"})
    
    if yf is None:
        return json.dumps({"error": "yfinance is not installed; pip install yfinance"})
    
    sym = str(symbol).strip().upper()
    try:
        ticker = yf.Ticker(sym)
        info = ticker.info
        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            return json.dumps({"error": f"No info found for symbol: {sym}"})
        
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
    except Exception as e:
        logger.exception("get_info failed for %s", sym)
        return json.dumps({"error": f"Failed to get info for {sym}: {str(e)}"})


if __name__ == "__main__":
    logger.info("Starting Stocks MCP Server...")
    logger.info("Tools: get_quote, get_info")
    mcp.run()
