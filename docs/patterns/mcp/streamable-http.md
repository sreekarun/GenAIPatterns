# Streamable HTTP Transport for MCP

This guide covers implementing and using Streamable HTTP transport for Model Context Protocol (MCP) servers and clients, providing real-time bidirectional communication capabilities.

## What is Streamable HTTP in MCP?

Streamable HTTP (also known as Server-Sent Events or SSE) is a transport method that enables real-time, streaming communication between MCP clients and servers over HTTP. It provides:

- **Real-time updates**: Stream responses as they're generated
- **Low latency**: Immediate data transmission without polling
- **HTTP compatibility**: Works through standard HTTP infrastructure
- **Bidirectional communication**: Full duplex communication over HTTP
- **Connection persistence**: Long-lived connections for efficient resource usage

## When to Use Streamable HTTP

### Ideal Use Cases

1. **Long-running operations**: Tasks that take significant time to complete
2. **Progressive results**: Operations that can provide intermediate results
3. **Real-time data**: Live data feeds, monitoring, or streaming analytics
4. **Interactive AI assistants**: Conversational flows requiring immediate responses
5. **Web-based integrations**: Browser-based applications needing real-time updates

### Comparison with Other Transports

| Transport | Latency | Complexity | Browser Support | Use Case |
|-----------|---------|------------|-----------------|----------|
| stdio | Very Low | Low | No | Local processes |
| WebSocket | Low | Medium | Yes | Real-time apps |
| **Streamable HTTP** | **Low** | **Medium** | **Yes** | **Web services** |
| Regular HTTP | Medium-High | Low | Yes | Simple requests |

## Implementation Guide

### Prerequisites

```bash
pip install mcp[server]
pip install uvicorn
pip install fastapi
pip install aiohttp
```

### 1. Basic Streamable HTTP Server

```python
from mcp import McpServer
from mcp.server.fastapi import create_app
from mcp.types import Tool, TextContent
import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
import time

# Create MCP server
server = McpServer("streamable-http-server")

@server.tool()
async def stream_data_analysis(dataset: str, chunk_size: int = 100) -> str:
    """Analyze data in chunks, streaming results progressively."""
    total_items = 1000  # Simulated dataset size
    
    for i in range(0, total_items, chunk_size):
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        progress = min(i + chunk_size, total_items)
        percentage = (progress / total_items) * 100
        
        # Yield intermediate result
        yield f"Processed {progress}/{total_items} items ({percentage:.1f}%)"
    
    return "Analysis complete"

@server.tool()
async def real_time_monitoring(duration: int = 10) -> str:
    """Stream real-time monitoring data."""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        current_time = time.strftime("%H:%M:%S")
        cpu_usage = 45.2 + (time.time() % 10)  # Simulated CPU usage
        memory_usage = 62.8 + (time.time() % 5)  # Simulated memory usage
        
        yield f"[{current_time}] CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%"
        await asyncio.sleep(1)
    
    return "Monitoring session ended"

# Create FastAPI app with MCP integration
app = create_app(server)

@app.get("/stream")
async def stream_endpoint():
    """Custom streaming endpoint for direct HTTP access."""
    async def generate_stream():
        for i in range(10):
            data = {
                "timestamp": time.time(),
                "message": f"Stream message {i + 1}",
                "data": {"value": i * 10}
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

def run_server():
    """Run the streamable HTTP server."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
```

### 2. Advanced Streaming Server with Authentication

```python
from mcp import McpServer
from mcp.server.fastapi import create_app
from mcp.types import Tool, McpError
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import json
import os
from typing import Optional

server = McpServer("secure-streaming-server")
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token or API key."""
    expected_token = os.getenv("API_TOKEN", "your-secret-token")
    
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return credentials.credentials

@server.tool()
async def secure_data_stream(
    query: str,
    token: str = None
) -> str:
    """Stream secure data with authentication."""
    if not token:
        raise McpError(
            code=-32602,
            message="Authentication token required",
            data={"parameter": "token"}
        )
    
    # Simulate secure data processing
    for i in range(5):
        await asyncio.sleep(0.5)
        yield f"Secure result {i + 1} for query: {query}"
    
    return "Secure streaming complete"

app = create_app(server)

@app.get("/secure-stream")
async def secure_stream(token: str = Depends(verify_token)):
    """Authenticated streaming endpoint."""
    async def generate_secure_stream():
        for i in range(10):
            data = {
                "timestamp": time.time(),
                "secure_data": f"Authenticated data {i + 1}",
                "user_id": "authenticated_user"
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(
        generate_secure_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
```

### 3. Streamable HTTP Client

```python
import aiohttp
import asyncio
import json
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

class StreamableHttpClient:
    def __init__(self, base_url: str, token: str = None):
        self.base_url = base_url
        self.token = token
        self.headers = {}
        
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    async def connect_to_mcp_server(self):
        """Connect to MCP server via SSE."""
        async with sse_client(f"{self.base_url}/mcp/sse") as client:
            return client
    
    async def stream_tool_call(self, tool_name: str, **kwargs):
        """Call a streaming tool and handle the response."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/tools/{tool_name}/stream"
            
            async with session.post(
                url,
                json=kwargs,
                headers=self.headers
            ) as response:
                
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode().strip())
                            yield data
                        except json.JSONDecodeError:
                            # Handle non-JSON streaming data
                            yield {"raw": line.decode().strip()}
    
    async def listen_to_stream(self, endpoint: str = "/stream"):
        """Listen to a custom streaming endpoint."""
        url = f"{self.base_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                async for line in response.content:
                    if line.startswith(b"data: "):
                        data_str = line[6:].decode().strip()
                        if data_str:
                            try:
                                data = json.loads(data_str)
                                yield data
                            except json.JSONDecodeError:
                                yield {"raw": data_str}

# Usage example
async def main():
    client = StreamableHttpClient("http://localhost:8000")
    
    # Stream tool results
    async for result in client.stream_tool_call(
        "stream_data_analysis",
        dataset="user_data",
        chunk_size=50
    ):
        print(f"Received: {result}")
    
    # Listen to real-time stream
    async for event in client.listen_to_stream():
        print(f"Stream event: {event}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration and Deployment

### 1. Production Configuration

```python
# config/streaming_config.py
import os
from pydantic import BaseSettings

class StreamingConfig(BaseSettings):
    # Server settings
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", 8000))
    workers: int = int(os.getenv("WORKERS", 1))
    
    # Streaming settings
    max_connections: int = 1000
    keepalive_timeout: int = 65
    stream_timeout: int = 300
    
    # Security settings
    api_token: str = os.getenv("API_TOKEN", "")
    cors_origins: list = ["*"]
    
    # Performance settings
    chunk_size: int = 8192
    buffer_size: int = 1024 * 1024  # 1MB
    
    class Config:
        env_file = ".env"

config = StreamingConfig()
```

### 2. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Use uvicorn with proper streaming settings
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--keepalive-timeout", "65"]
```

### 3. Load Balancer Configuration (Nginx)

```nginx
# nginx.conf
upstream mcp_streaming {
    server app1:8000;
    server app2:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name mcp-server.example.com;
    
    location / {
        proxy_pass http://mcp_streaming;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Streaming-specific settings
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
    
    location /stream {
        proxy_pass http://mcp_streaming;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600s;
        
        # SSE headers
        add_header Cache-Control "no-cache";
        add_header Connection "keep-alive";
        add_header Content-Type "text/event-stream";
    }
}
```

## Best Practices

### 1. Connection Management

```python
import asyncio
from typing import Set
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_count = 0
        self.max_connections = 1000
    
    async def connect(self, websocket: WebSocket):
        if self.connection_count >= self.max_connections:
            await websocket.close(code=1013, reason="Too many connections")
            return False
        
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_count += 1
        return True
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_count -= 1
    
    async def broadcast(self, message: str):
        """Broadcast message to all active connections."""
        if self.active_connections:
            tasks = [
                self._safe_send(connection, message)
                for connection in self.active_connections.copy()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _safe_send(self, connection: WebSocket, message: str):
        try:
            await connection.send_text(message)
        except Exception:
            # Connection is broken, remove it
            self.disconnect(connection)
```

### 2. Error Handling and Resilience

```python
import logging
from typing import AsyncGenerator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def resilient_stream():
    """Context manager for resilient streaming operations."""
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            yield
            break
        except ConnectionError as e:
            retry_count += 1
            logger.warning(f"Connection error (attempt {retry_count}): {e}")
            if retry_count >= max_retries:
                logger.error("Max retries exceeded")
                raise
            await asyncio.sleep(2 ** retry_count)  # Exponential backoff
        except Exception as e:
            logger.error(f"Unexpected error in stream: {e}")
            raise

async def robust_streaming_tool(data: str) -> AsyncGenerator[str, None]:
    """Example of a robust streaming tool with error handling."""
    async with resilient_stream():
        chunks = data.split()
        
        for i, chunk in enumerate(chunks):
            try:
                # Simulate processing that might fail
                if chunk.lower() == "error":
                    raise ValueError("Simulated processing error")
                
                result = f"Processed chunk {i + 1}: {chunk.upper()}"
                yield result
                
                # Add small delay to simulate real processing
                await asyncio.sleep(0.1)
                
            except ValueError as e:
                logger.warning(f"Skipping chunk due to error: {e}")
                yield f"Error processing chunk {i + 1}: {str(e)}"
                continue
```

### 3. Performance Optimization

```python
from asyncio import Semaphore
from typing import Optional
import time

class StreamOptimizer:
    def __init__(self, max_concurrent_streams: int = 100):
        self.semaphore = Semaphore(max_concurrent_streams)
        self.active_streams = {}
        self.stream_stats = {
            "total_streams": 0,
            "active_count": 0,
            "avg_duration": 0
        }
    
    async def optimized_stream(
        self, 
        stream_id: str, 
        data_generator,
        chunk_size: int = 1024
    ):
        """Optimized streaming with rate limiting and monitoring."""
        async with self.semaphore:
            start_time = time.time()
            self.active_streams[stream_id] = start_time
            self.stream_stats["active_count"] += 1
            
            try:
                buffer = []
                buffer_size = 0
                
                async for item in data_generator:
                    buffer.append(item)
                    buffer_size += len(str(item))
                    
                    # Yield when buffer reaches chunk_size
                    if buffer_size >= chunk_size:
                        yield "".join(map(str, buffer))
                        buffer = []
                        buffer_size = 0
                
                # Yield remaining buffer
                if buffer:
                    yield "".join(map(str, buffer))
                    
            finally:
                # Update statistics
                duration = time.time() - start_time
                self.stream_stats["total_streams"] += 1
                self.stream_stats["avg_duration"] = (
                    (self.stream_stats["avg_duration"] * (self.stream_stats["total_streams"] - 1) + duration)
                    / self.stream_stats["total_streams"]
                )
                
                del self.active_streams[stream_id]
                self.stream_stats["active_count"] -= 1
```

## Troubleshooting

### Common Issues and Solutions

1. **Connection Timeouts**
   - Increase `keepalive_timeout` and `proxy_read_timeout`
   - Implement heartbeat messages
   - Use connection pooling

2. **Memory Issues with Large Streams**
   - Implement chunking and buffering
   - Use generators instead of loading all data
   - Set appropriate buffer sizes

3. **Browser Compatibility**
   - Ensure proper CORS headers
   - Use EventSource API for SSE
   - Implement fallback mechanisms

4. **Load Balancer Issues**
   - Disable proxy buffering
   - Configure sticky sessions if needed
   - Set appropriate timeout values

### Monitoring and Debugging

```python
import json
import time
from dataclasses import dataclass, asdict

@dataclass
class StreamMetrics:
    stream_id: str
    start_time: float
    bytes_sent: int = 0
    messages_sent: int = 0
    errors: int = 0
    
    def to_dict(self):
        return {
            **asdict(self),
            "duration": time.time() - self.start_time,
            "bytes_per_second": self.bytes_sent / max(time.time() - self.start_time, 0.001),
            "messages_per_second": self.messages_sent / max(time.time() - self.start_time, 0.001)
        }

class StreamMonitor:
    def __init__(self):
        self.metrics = {}
    
    def start_stream(self, stream_id: str):
        self.metrics[stream_id] = StreamMetrics(stream_id, time.time())
    
    def record_message(self, stream_id: str, message_size: int):
        if stream_id in self.metrics:
            self.metrics[stream_id].bytes_sent += message_size
            self.metrics[stream_id].messages_sent += 1
    
    def record_error(self, stream_id: str):
        if stream_id in self.metrics:
            self.metrics[stream_id].errors += 1
    
    def get_stats(self, stream_id: str = None):
        if stream_id:
            return self.metrics.get(stream_id, {}).to_dict()
        return {sid: metrics.to_dict() for sid, metrics in self.metrics.items()}
```

## External Resources

- [Server-Sent Events Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [FastAPI Streaming Response Documentation](https://fastapi.tiangolo.com/advanced/streaming/)
- [MCP Protocol Specification](https://spec.modelcontextprotocol.io/)
- [AIOHTTP Streaming Documentation](https://docs.aiohttp.org/en/stable/client_quickstart.html#streaming-response-content)