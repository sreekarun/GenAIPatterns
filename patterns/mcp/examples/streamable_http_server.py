#!/usr/bin/env python3
"""
Streamable HTTP MCP Server Example

Demonstrates real-time streaming capabilities using HTTP transport
with Server-Sent Events (SSE) for Model Context Protocol.

Features:
- Real-time data streaming
- Progressive result delivery
- HTTP-compatible transport
- Authentication support
- Performance monitoring

Usage:
    python streamable_http_server.py
    
Test streaming endpoints:
    curl http://localhost:8000/health-stream
    curl -H "Accept: text/event-stream" http://localhost:8000/mcp/tools/stream_data_analysis
"""

from mcp import McpServer
from mcp.server.fastapi import create_app
from mcp.types import Tool, Resource, TextContent, McpError
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import json
import time
import logging
import os
from typing import Optional, AsyncGenerator
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
server = McpServer("streamable-http-demo")

# Authentication
security = HTTPBearer()
API_TOKEN = os.getenv("API_TOKEN", "demo-token-12345")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token."""
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Streaming tools
@server.tool()
async def stream_data_analysis(
    dataset_name: str = "sample_data",
    total_items: int = 1000,
    chunk_size: int = 100
) -> str:
    """
    Analyze a dataset progressively, streaming results in real-time.
    
    Args:
        dataset_name: Name of the dataset to analyze
        total_items: Total number of items to process
        chunk_size: Number of items to process per chunk
    """
    logger.info(f"Starting analysis of {dataset_name} with {total_items} items")
    
    processed = 0
    anomalies_found = 0
    
    for i in range(0, total_items, chunk_size):
        # Simulate processing time
        await asyncio.sleep(0.2)
        
        current_chunk = min(chunk_size, total_items - processed)
        processed += current_chunk
        
        # Simulate finding anomalies
        chunk_anomalies = random.randint(0, max(1, current_chunk // 20))
        anomalies_found += chunk_anomalies
        
        percentage = (processed / total_items) * 100
        
        result = {
            "timestamp": time.time(),
            "progress": {
                "processed": processed,
                "total": total_items,
                "percentage": round(percentage, 2)
            },
            "current_chunk": {
                "size": current_chunk,
                "anomalies": chunk_anomalies
            },
            "summary": {
                "total_anomalies": anomalies_found,
                "anomaly_rate": round((anomalies_found / processed) * 100, 2)
            }
        }
        
        yield f"📊 Analysis Progress: {percentage:.1f}% | Anomalies: {anomalies_found}/{processed} ({result['summary']['anomaly_rate']}%)"
        
        if processed >= total_items:
            break
    
    final_result = f"✅ Analysis complete! Processed {processed} items, found {anomalies_found} anomalies ({result['summary']['anomaly_rate']}% rate)"
    logger.info(final_result)
    yield final_result

@server.tool()
async def real_time_monitoring(
    duration_seconds: int = 30,
    interval_seconds: float = 1.0
) -> str:
    """
    Monitor system metrics in real-time and stream the results.
    
    Args:
        duration_seconds: How long to monitor (seconds)
        interval_seconds: Update interval (seconds)
    """
    logger.info(f"Starting real-time monitoring for {duration_seconds} seconds")
    
    start_time = time.time()
    update_count = 0
    
    while time.time() - start_time < duration_seconds:
        current_time = time.strftime("%H:%M:%S")
        elapsed = time.time() - start_time
        
        # Simulate realistic metrics with some variance
        cpu_usage = 45.0 + 20 * (0.5 - random.random()) + 10 * abs(math.sin(elapsed / 10))
        memory_usage = 65.0 + 15 * (0.5 - random.random())
        disk_io = 120.0 + 80 * random.random()
        network_io = 1500 + 800 * random.random()
        
        metrics = {
            "timestamp": time.time(),
            "time": current_time,
            "elapsed": round(elapsed, 1),
            "metrics": {
                "cpu_percent": round(cpu_usage, 1),
                "memory_percent": round(memory_usage, 1),
                "disk_io_mb": round(disk_io, 1),
                "network_io_kb": round(network_io, 1)
            },
            "alerts": []
        }
        
        # Generate alerts for high usage
        if cpu_usage > 80:
            metrics["alerts"].append("⚠️  High CPU usage detected")
        if memory_usage > 85:
            metrics["alerts"].append("⚠️  High memory usage detected")
        
        status_icon = "🟢" if not metrics["alerts"] else "🟡"
        alerts_text = " | ".join(metrics["alerts"]) if metrics["alerts"] else "All systems normal"
        
        yield f"{status_icon} [{current_time}] CPU: {metrics['metrics']['cpu_percent']}% | RAM: {metrics['metrics']['memory_percent']}% | {alerts_text}"
        
        update_count += 1
        await asyncio.sleep(interval_seconds)
    
    final_result = f"🏁 Monitoring complete. Collected {update_count} data points over {duration_seconds} seconds"
    logger.info(final_result)
    yield final_result

@server.tool()
async def stream_file_processing(
    file_pattern: str = "*.log",
    max_files: int = 50
) -> str:
    """
    Process files matching a pattern and stream progress updates.
    
    Args:
        file_pattern: Pattern to match files (e.g., *.log, *.txt)
        max_files: Maximum number of files to process
    """
    logger.info(f"Processing files matching '{file_pattern}' (max: {max_files})")
    
    # Simulate file discovery
    simulated_files = [f"file_{i:03d}.{file_pattern.split('.')[-1]}" for i in range(1, min(max_files + 1, 101))]
    total_files = len(simulated_files)
    processed_files = 0
    total_size = 0
    errors = 0
    
    for i, filename in enumerate(simulated_files):
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Simulate file size and processing
        file_size = random.randint(1024, 10 * 1024 * 1024)  # 1KB to 10MB
        total_size += file_size
        processed_files += 1
        
        # Simulate occasional errors
        if random.random() < 0.05:  # 5% error rate
            errors += 1
            status = "❌ ERROR"
        else:
            status = "✅ OK"
        
        percentage = (processed_files / total_files) * 100
        avg_size = total_size / processed_files / (1024 * 1024)  # MB
        
        yield f"{status} [{processed_files}/{total_files}] {filename} ({file_size // 1024}KB) | Progress: {percentage:.1f}% | Avg: {avg_size:.1f}MB"
        
        if processed_files >= max_files:
            break
    
    final_result = f"🎯 Processing complete! {processed_files} files, {total_size // (1024*1024)}MB total, {errors} errors"
    logger.info(final_result)
    yield final_result

# Resources
@server.resource("streaming://status")
async def get_streaming_status():
    """Get current streaming server status."""
    status = {
        "server_name": "streamable-http-demo",
        "uptime": time.time(),
        "active_streams": random.randint(3, 15),
        "total_requests": random.randint(1000, 5000),
        "version": "1.0.0",
        "features": ["real-time-streaming", "progressive-results", "authentication"]
    }
    
    return TextContent(
        type="text", 
        text=json.dumps(status, indent=2)
    )

@server.list_resources()
async def list_available_resources():
    """List all available resources."""
    return [
        {
            "uri": "streaming://status",
            "name": "Streaming Server Status",
            "description": "Current status and metrics of the streaming server",
            "mimeType": "application/json"
        }
    ]

# Create FastAPI app
app = create_app(server)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom streaming endpoints
@app.get("/health-stream")
async def health_stream():
    """Stream server health data in real-time."""
    async def generate_health_data():
        for i in range(20):  # Stream for 20 updates
            health_data = {
                "timestamp": time.time(),
                "status": "healthy",
                "uptime_seconds": i * 2,
                "active_connections": 5 + (i % 3),
                "memory_usage": 45 + (i % 10),
                "cpu_usage": 30 + random.randint(-5, 15),
                "requests_per_minute": 120 + random.randint(-20, 40)
            }
            
            yield f"data: {json.dumps(health_data)}\n\n"
            await asyncio.sleep(2)
        
        # Send completion event
        yield f"data: {json.dumps({'status': 'stream_complete'})}\n\n"
    
    return StreamingResponse(
        generate_health_data(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.get("/metrics-stream")
async def metrics_stream(token: str = Depends(verify_token)):
    """Stream authenticated metrics data."""
    async def generate_metrics():
        for i in range(15):
            metrics = {
                "timestamp": time.time(),
                "cpu_cores": [
                    {"core": j, "usage": 30 + random.randint(-10, 40)} 
                    for j in range(4)
                ],
                "memory": {
                    "used_gb": 8.5 + random.random() * 2,
                    "total_gb": 16,
                    "percentage": 53.1 + random.random() * 15
                },
                "network": {
                    "rx_mbps": 15.3 + random.random() * 10,
                    "tx_mbps": 8.7 + random.random() * 5
                }
            }
            
            yield f"data: {json.dumps(metrics)}\n\n"
            await asyncio.sleep(3)
    
    return StreamingResponse(
        generate_metrics(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Streamable HTTP MCP Server",
        "version": "1.0.0",
        "endpoints": {
            "mcp": "/mcp",
            "health_stream": "/health-stream",
            "metrics_stream": "/metrics-stream (requires auth)",
            "docs": "/docs"
        },
        "streaming_tools": [
            "stream_data_analysis",
            "real_time_monitoring", 
            "stream_file_processing"
        ],
        "authentication": {
            "required_for": ["/metrics-stream", "/mcp/tools/*"],
            "header": "Authorization: Bearer <token>",
            "demo_token": API_TOKEN
        }
    }

def run_server():
    """Run the streamable HTTP server."""
    import uvicorn
    
    logger.info("🚀 Starting Streamable HTTP MCP Server...")
    logger.info(f"📡 Server will be available at: http://0.0.0.0:8000")
    logger.info(f"📖 API docs available at: http://0.0.0.0:8000/docs")
    logger.info(f"🔐 Demo token: {API_TOKEN}")
    logger.info("🛑 Press Ctrl+C to stop")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    # Import math for monitoring
    import math
    run_server()