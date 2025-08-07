#!/usr/bin/env python3
"""
Streamable HTTP MCP Client Example

Demonstrates how to connect to and consume streaming data from
an MCP server using HTTP transport with Server-Sent Events.

Usage:
    # Start the server first:
    python streamable_http_server.py
    
    # Then run this client:
    python streamable_http_client.py
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import AsyncGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamableHttpClient:
    """Client for consuming MCP streaming HTTP endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000", token: str = None):
        self.base_url = base_url
        self.token = token
        self.headers = {"Accept": "text/event-stream"}
        
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    async def stream_sse_endpoint(self, endpoint: str) -> AsyncGenerator[dict, None]:
        """Stream data from a Server-Sent Events endpoint."""
        url = f"{self.base_url}{endpoint}"
        logger.info(f"🔗 Connecting to SSE stream: {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        logger.error(f"❌ HTTP {response.status}: {await response.text()}")
                        return
                    
                    logger.info(f"✅ Connected to stream, waiting for data...")
                    
                    async for line in response.content:
                        line_str = line.decode().strip()
                        
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]  # Remove "data: " prefix
                            
                            if data_str:
                                try:
                                    data = json.loads(data_str)
                                    yield data
                                except json.JSONDecodeError:
                                    yield {"raw": data_str}
                        elif line_str == "":
                            # Empty line indicates end of event
                            continue
        
        except Exception as e:
            logger.error(f"❌ Stream error: {e}")
            raise
    
    async def call_streaming_tool(self, tool_name: str, **kwargs) -> AsyncGenerator[str, None]:
        """Call a streaming MCP tool and yield results."""
        url = f"{self.base_url}/mcp/tools/{tool_name}/call"
        
        payload = {
            "name": tool_name,
            "arguments": kwargs
        }
        
        logger.info(f"🔧 Calling streaming tool: {tool_name} with args: {kwargs}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self.headers
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"❌ Tool call failed HTTP {response.status}: {error_text}")
                        return
                    
                    logger.info(f"✅ Tool call started, streaming results...")
                    
                    async for line in response.content:
                        line_str = line.decode().strip()
                        if line_str:
                            yield line_str
                            
        except Exception as e:
            logger.error(f"❌ Tool call error: {e}")
            raise
    
    async def get_server_info(self) -> dict:
        """Get basic server information."""
        url = f"{self.base_url}/"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"❌ Failed to get server info: HTTP {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"❌ Server info error: {e}")
            return {}

async def demo_health_stream(client: StreamableHttpClient):
    """Demonstrate health stream consumption."""
    print("\n" + "="*60)
    print("🏥 HEALTH STREAM DEMO")
    print("="*60)
    
    count = 0
    async for health_data in client.stream_sse_endpoint("/health-stream"):
        if health_data.get("status") == "stream_complete":
            print("✅ Health stream completed")
            break
        
        count += 1
        timestamp = health_data.get("timestamp", time.time())
        status = health_data.get("status", "unknown")
        uptime = health_data.get("uptime_seconds", 0)
        connections = health_data.get("active_connections", 0)
        cpu = health_data.get("cpu_usage", 0)
        memory = health_data.get("memory_usage", 0)
        
        print(f"📊 [{count:2d}] Status: {status} | Uptime: {uptime}s | "
              f"Connections: {connections} | CPU: {cpu}% | Memory: {memory}%")
        
        if count >= 10:  # Limit for demo
            print("🛑 Stopping after 10 updates...")
            break

async def demo_metrics_stream(client: StreamableHttpClient):
    """Demonstrate authenticated metrics stream."""
    print("\n" + "="*60)
    print("📈 METRICS STREAM DEMO (Authenticated)")
    print("="*60)
    
    count = 0
    try:
        async for metrics_data in client.stream_sse_endpoint("/metrics-stream"):
            count += 1
            
            cpu_cores = metrics_data.get("cpu_cores", [])
            memory = metrics_data.get("memory", {})
            network = metrics_data.get("network", {})
            
            avg_cpu = sum(core.get("usage", 0) for core in cpu_cores) / len(cpu_cores) if cpu_cores else 0
            memory_pct = memory.get("percentage", 0)
            rx_mbps = network.get("rx_mbps", 0)
            tx_mbps = network.get("tx_mbps", 0)
            
            print(f"📊 [{count:2d}] CPU: {avg_cpu:.1f}% | Memory: {memory_pct:.1f}% | "
                  f"Network: ↓{rx_mbps:.1f} ↑{tx_mbps:.1f} Mbps")
            
            if count >= 5:  # Limit for demo
                print("🛑 Stopping after 5 updates...")
                break
                
    except Exception as e:
        if "401" in str(e):
            print("🔐 Authentication required for metrics stream")
        else:
            print(f"❌ Metrics stream error: {e}")

async def demo_streaming_tools(client: StreamableHttpClient):
    """Demonstrate streaming tool calls."""
    print("\n" + "="*60)
    print("🔧 STREAMING TOOLS DEMO")
    print("="*60)
    
    demos = [
        {
            "name": "stream_data_analysis",
            "args": {"dataset_name": "customer_data", "total_items": 500, "chunk_size": 50},
            "description": "Data Analysis Stream"
        },
        {
            "name": "real_time_monitoring", 
            "args": {"duration_seconds": 10, "interval_seconds": 1},
            "description": "Real-time Monitoring"
        },
        {
            "name": "stream_file_processing",
            "args": {"file_pattern": "*.log", "max_files": 20},
            "description": "File Processing Stream"
        }
    ]
    
    for demo in demos:
        print(f"\n🚀 Starting {demo['description']}...")
        print("-" * 50)
        
        count = 0
        try:
            async for result in client.call_streaming_tool(demo["name"], **demo["args"]):
                count += 1
                print(f"📝 [{count:2d}] {result}")
                
                if count >= 10:  # Limit for demo
                    print("🛑 Stopping after 10 updates...")
                    break
                    
        except Exception as e:
            print(f"❌ Tool error: {e}")
        
        print(f"✅ {demo['description']} demo completed\n")

async def main():
    """Main demo function."""
    print("🌊 Streamable HTTP MCP Client Demo")
    print("=" * 60)
    
    # Create client
    client = StreamableHttpClient()
    
    # Get server info
    print("📋 Getting server information...")
    server_info = await client.get_server_info()
    
    if server_info:
        print(f"✅ Connected to: {server_info.get('message', 'Unknown server')}")
        print(f"📍 Version: {server_info.get('version', 'Unknown')}")
        
        streaming_tools = server_info.get('streaming_tools', [])
        if streaming_tools:
            print(f"🔧 Available streaming tools: {', '.join(streaming_tools)}")
        
        # Show demo token
        auth_info = server_info.get('authentication', {})
        demo_token = auth_info.get('demo_token')
        if demo_token:
            print(f"🔐 Demo token available: {demo_token}")
    else:
        print("⚠️  Could not connect to server. Make sure streamable_http_server.py is running!")
        return
    
    try:
        # Demo 1: Health stream (no auth required)
        await demo_health_stream(client)
        
        # Demo 2: Streaming tools
        await demo_streaming_tools(client)
        
        # Demo 3: Authenticated metrics (if token available)
        if server_info.get('authentication', {}).get('demo_token'):
            print("\n🔐 Testing authenticated endpoint...")
            auth_client = StreamableHttpClient(token=server_info['authentication']['demo_token'])
            await demo_metrics_stream(auth_client)
        
        print("\n" + "="*60)
        print("🎉 All demos completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(main())