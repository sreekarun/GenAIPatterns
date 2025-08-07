# How to Build an MCP Server

This guide covers building Model Context Protocol (MCP) servers using Python, from basic implementations to production-ready services.

## Prerequisites

- Python 3.8 or higher
- Basic understanding of async programming
- Familiarity with JSON-RPC protocol (helpful but not required)

## Installation

```bash
pip install mcp
```

For additional features:
```bash
pip install mcp[server]  # Server-specific dependencies
pip install uvicorn      # For HTTP transport
pip install fastapi      # For web-based servers
```

## Basic Server Implementation

### 1. Simple Tool Server

```python
from mcp import McpServer
from mcp.types import Tool, TextContent
import asyncio

# Create server instance
server = McpServer("my-tool-server")

@server.tool()
async def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers."""
    return a + b

@server.tool()
async def get_weather(city: str) -> str:
    """Get weather information for a city."""
    # In a real implementation, you'd call a weather API
    return f"The weather in {city} is sunny and 72°F"

if __name__ == "__main__":
    # Run with stdio transport
    asyncio.run(server.run_stdio())
```

### 2. Resource Server

```python
from mcp import McpServer
from mcp.types import Resource, TextContent
import json

server = McpServer("resource-server")

@server.resource("config://app")
async def get_app_config():
    """Provide application configuration."""
    config = {
        "app_name": "My Application",
        "version": "1.0.0",
        "features": ["auth", "api", "database"]
    }
    return TextContent(
        type="text",
        text=json.dumps(config, indent=2)
    )

@server.list_resources()
async def list_available_resources():
    """List all available resources."""
    return [
        Resource(
            uri="config://app",
            name="Application Configuration",
            description="Main application configuration",
            mimeType="application/json"
        )
    ]
```

### 3. Prompt Server

```python
from mcp import McpServer
from mcp.types import Prompt, PromptMessage

server = McpServer("prompt-server")

@server.prompt("code-review")
async def code_review_prompt(language: str = "python", style: str = "standard"):
    """Generate a code review prompt."""
    return PromptMessage(
        role="user",
        content=f"""Please review the following {language} code following {style} style guidelines:

Guidelines to check:
- Code quality and readability
- Performance considerations
- Security best practices
- Documentation completeness
- Error handling

Provide constructive feedback with specific suggestions for improvement."""
    )

@server.list_prompts()
async def list_available_prompts():
    """List all available prompts."""
    return [
        Prompt(
            name="code-review",
            description="Generate code review prompts",
            arguments=[
                {"name": "language", "description": "Programming language", "required": False},
                {"name": "style", "description": "Code style guidelines", "required": False}
            ]
        )
    ]
```

## Advanced Server Features

### 4. Server with Authentication

```python
from mcp import McpServer
from mcp.types import Tool
import os

class AuthenticatedServer(McpServer):
    def __init__(self, name: str):
        super().__init__(name)
        self.api_key = os.getenv("API_KEY")
        
    async def validate_request(self, request):
        """Validate incoming requests."""
        # Add your authentication logic here
        auth_header = request.get("headers", {}).get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise Exception("Authentication required")
        
        token = auth_header.split(" ")[1]
        if token != self.api_key:
            raise Exception("Invalid token")

server = AuthenticatedServer("secure-server")

@server.tool()
async def sensitive_operation(data: str) -> str:
    """Perform a sensitive operation that requires authentication."""
    return f"Processed: {data}"
```

### 5. Error Handling and Logging

```python
from mcp import McpServer
from mcp.types import Tool, McpError
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = McpServer("robust-server")

@server.tool()
async def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers with proper error handling."""
    try:
        if b == 0:
            raise McpError(
                code=-32600,
                message="Division by zero is not allowed",
                data={"a": a, "b": b}
            )
        
        result = a / b
        logger.info(f"Division successful: {a} / {b} = {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in division: {str(e)}")
        logger.error(traceback.format_exc())
        raise McpError(
            code=-32603,
            message="Internal server error",
            data={"error": str(e)}
        )

@server.error_handler()
async def handle_errors(error):
    """Global error handler."""
    logger.error(f"Server error: {error}")
    return {
        "error": {
            "code": -32603,
            "message": "Internal server error"
        }
    }
```

## Transport Methods

### Standard I/O Transport

```python
# Run with stdio (default)
if __name__ == "__main__":
    asyncio.run(server.run_stdio())
```

### HTTP Transport with FastAPI

```python
from fastapi import FastAPI
from mcp.server.fastapi import create_mcp_router

app = FastAPI()
mcp_router = create_mcp_router(server)
app.include_router(mcp_router, prefix="/mcp")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### WebSocket Transport

```python
from mcp.server.websocket import run_websocket_server

if __name__ == "__main__":
    asyncio.run(run_websocket_server(server, host="0.0.0.0", port=8080))
```

### Server-Sent Events (SSE) Transport

Server-Sent Events (SSE) provides an ideal transport mechanism for MCP servers that need to stream responses, deliver real-time updates, or handle long-running operations with continuous feedback.

#### Historical Context and Evolution

Server-Sent Events was introduced as part of the HTML5 specification (W3C) around 2009-2011, designed to enable servers to push real-time updates to web browsers over HTTP. Unlike WebSockets which provide bidirectional communication, SSE offers unidirectional server-to-client streaming, making it perfect for scenarios where the client primarily consumes data streams.

The evolution of SSE in web technologies:
- **2009-2011**: Initial HTML5 specification defined SSE
- **2012-2015**: Browser adoption and standardization
- **2016-2020**: Widespread use in real-time applications (dashboards, live feeds)
- **2021-Present**: Adoption in AI/ML systems for streaming model responses

#### Why SSE was Chosen for MCP

SSE was selected as an MCP transport method for several compelling reasons:

1. **Streaming-First Architecture**: Perfect for LLM responses that generate tokens incrementally
2. **HTTP Compatibility**: Works seamlessly with existing web infrastructure (proxies, load balancers)
3. **Automatic Reconnection**: Built-in reconnection handling for robust connections
4. **Simplicity**: Easier to implement and debug compared to WebSockets
5. **Firewall Friendly**: Standard HTTP traffic, no special network configuration required
6. **Event-Driven**: Natural fit for MCP's event-based protocol design

#### Implementation Examples

##### Basic SSE Server Setup

```python
from mcp.server.sse import SseTransport
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()
server = McpServer("sse-example-server")

@server.tool()
async def stream_data(query: str) -> str:
    """Tool that streams data back to client."""
    # Simulate streaming response
    result = f"Processing query: {query}\n"
    for i in range(5):
        await asyncio.sleep(1)
        result += f"Step {i+1} completed\n"
    return result

@app.get("/mcp/sse")
async def mcp_sse_endpoint():
    """SSE endpoint for MCP communication."""
    
    async def event_generator():
        transport = SseTransport(server)
        async for event in transport.stream_events():
            yield f"data: {json.dumps(event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

##### Advanced SSE with Custom Event Types

```python
from mcp.server.sse import SseTransport, SseEvent
from typing import AsyncGenerator
import time

class CustomSseTransport(SseTransport):
    """Custom SSE transport with specialized event handling."""
    
    async def stream_tool_response(self, tool_name: str, args: dict) -> AsyncGenerator[SseEvent, None]:
        """Stream tool execution with progress updates."""
        
        # Send start event
        yield SseEvent(
            event="tool_start",
            data={"tool": tool_name, "timestamp": time.time()}
        )
        
        # Execute tool and stream progress
        try:
            if tool_name == "long_running_analysis":
                for progress in range(0, 101, 10):
                    yield SseEvent(
                        event="tool_progress", 
                        data={"progress": progress, "status": f"Processing... {progress}%"}
                    )
                    await asyncio.sleep(0.5)  # Simulate work
                
                # Send final result
                result = await server.call_tool(tool_name, args)
                yield SseEvent(
                    event="tool_complete",
                    data={"result": result, "timestamp": time.time()}
                )
                
        except Exception as e:
            yield SseEvent(
                event="tool_error",
                data={"error": str(e), "timestamp": time.time()}
            )

@app.get("/mcp/tools/{tool_name}/sse")
async def stream_tool_execution(tool_name: str):
    """SSE endpoint for streaming tool execution."""
    
    async def tool_event_stream():
        transport = CustomSseTransport(server)
        async for event in transport.stream_tool_response(tool_name, {}):
            yield f"event: {event.event}\n"
            yield f"data: {json.dumps(event.data)}\n\n"
    
    return StreamingResponse(tool_event_stream(), media_type="text/event-stream")
```

#### Usage Scenarios and Best Practices

##### When to Use SSE Transport

SSE is ideal for MCP servers when:

- **Streaming LLM Responses**: Real-time token generation for chat applications
- **Long-Running Tools**: Progress updates for data analysis, file processing, or ML model inference
- **Resource Monitoring**: Live updates from databases, APIs, or system metrics
- **Event Broadcasting**: Notifications, alerts, or status changes to multiple clients

##### Implementation Best Practices

```python
# 1. Proper Error Handling and Reconnection
@app.get("/mcp/sse")
async def robust_sse_endpoint():
    async def event_generator():
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                transport = SseTransport(server)
                async for event in transport.stream_events():
                    yield f"data: {json.dumps(event)}\n\n"
                    retry_count = 0  # Reset on successful event
                    
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    yield f"event: retry\n"
                    yield f"data: {json.dumps({'retry_in': 5000})}\n\n"
                    await asyncio.sleep(5)
                else:
                    yield f"event: error\n"
                    yield f"data: {json.dumps({'error': 'Max retries exceeded'})}\n\n"
                    break
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# 2. Client Connection Management
class SseConnectionManager:
    """Manage multiple SSE client connections."""
    
    def __init__(self):
        self.active_connections: dict[str, asyncio.Queue] = {}
    
    async def connect(self, client_id: str) -> asyncio.Queue:
        """Register new SSE client."""
        queue = asyncio.Queue()
        self.active_connections[client_id] = queue
        return queue
    
    async def disconnect(self, client_id: str):
        """Remove SSE client."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def broadcast(self, event: SseEvent):
        """Send event to all connected clients."""
        for client_id, queue in self.active_connections.items():
            try:
                await queue.put(event)
            except Exception:
                await self.disconnect(client_id)

# 3. Performance Optimization
@app.get("/mcp/sse/optimized")
async def optimized_sse_endpoint():
    async def event_generator():
        # Use connection pooling and buffering
        buffer_size = 10
        event_buffer = []
        
        transport = SseTransport(server)
        async for event in transport.stream_events():
            event_buffer.append(event)
            
            # Batch events for better performance
            if len(event_buffer) >= buffer_size:
                for buffered_event in event_buffer:
                    yield f"data: {json.dumps(buffered_event)}\n\n"
                event_buffer.clear()
                
        # Send remaining events
        for remaining_event in event_buffer:
            yield f"data: {json.dumps(remaining_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable proxy buffering
        }
    )
```

#### Comparison with Other Transport Methods

| Feature | SSE | WebSocket | HTTP | stdio |
|---------|-----|-----------|------|-------|
| **Directionality** | Server → Client | Bidirectional | Request/Response | Bidirectional |
| **Reconnection** | Automatic | Manual | N/A | N/A |
| **Infrastructure** | HTTP-friendly | Special handling | Standard | Process-based |
| **Streaming** | Excellent | Excellent | Limited | Good |
| **Complexity** | Low | Medium | Low | Low |
| **Use Case** | Real-time updates | Interactive apps | Simple requests | Local tools |

#### Common Pitfalls and Solutions

```python
# 1. Avoid blocking operations in event streams
# ❌ Bad: Blocking operation
async def bad_event_generator():
    time.sleep(5)  # Blocks entire event loop
    yield "data: delayed\n\n"

# ✅ Good: Non-blocking operation
async def good_event_generator():
    await asyncio.sleep(5)  # Non-blocking
    yield "data: delayed\n\n"

# 2. Handle client disconnections gracefully
@app.get("/mcp/sse/graceful")
async def graceful_sse_endpoint(request: Request):
    async def event_generator():
        try:
            while True:
                # Check if client is still connected
                if await request.is_disconnected():
                    break
                    
                yield f"data: {json.dumps({'ping': time.time()})}\n\n"
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            # Client disconnected
            print("SSE client disconnected")
        finally:
            # Cleanup resources
            pass
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

## Configuration and Environment

### Environment Variables

```python
import os
from dataclasses import dataclass

@dataclass
class ServerConfig:
    name: str = os.getenv("SERVER_NAME", "default-server")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    api_key: str = os.getenv("API_KEY", "")
    database_url: str = os.getenv("DATABASE_URL", "")
    max_connections: int = int(os.getenv("MAX_CONNECTIONS", "100"))

config = ServerConfig()
server = McpServer(config.name)
```

### Configuration File

```python
import json
from pathlib import Path

def load_config(config_path: str = "server_config.json"):
    """Load server configuration from file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}

config = load_config()
server = McpServer(config.get("name", "default-server"))
```

## Testing Your Server

### Unit Tests

```python
import pytest
from mcp.testing import MockMcpClient

@pytest.mark.asyncio
async def test_calculate_sum():
    """Test the calculate_sum tool."""
    client = MockMcpClient(server)
    
    result = await client.call_tool("calculate_sum", {"a": 5, "b": 3})
    assert result == 8

@pytest.mark.asyncio
async def test_division_by_zero():
    """Test error handling for division by zero."""
    client = MockMcpClient(server)
    
    with pytest.raises(McpError) as exc_info:
        await client.call_tool("divide_numbers", {"a": 5, "b": 0})
    
    assert exc_info.value.code == -32600
    assert "Division by zero" in exc_info.value.message
```

### Integration Tests

```python
import asyncio
import subprocess
import json

async def test_server_integration():
    """Test server via stdio transport."""
    # Start server process
    process = subprocess.Popen(
        ["python", "server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send initialization request
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        }
    }
    
    process.stdin.write(json.dumps(init_request) + "\n")
    process.stdin.flush()
    
    # Read response
    response = process.stdout.readline()
    result = json.loads(response)
    
    assert result["result"]["protocolVersion"] == "2024-11-05"
    
    process.terminate()
    process.wait()
```

## Performance Optimization

### Async Best Practices

```python
import asyncio
import aiohttp
from mcp import McpServer

server = McpServer("optimized-server")

# Use connection pooling for external APIs
connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
session = aiohttp.ClientSession(connector=connector)

@server.tool()
async def fetch_data(url: str) -> str:
    """Fetch data from external API efficiently."""
    async with session.get(url) as response:
        return await response.text()

@server.tool()
async def batch_process(items: list) -> list:
    """Process multiple items concurrently."""
    tasks = [process_single_item(item) for item in items]
    return await asyncio.gather(*tasks)

async def process_single_item(item):
    """Process a single item."""
    # Simulate processing
    await asyncio.sleep(0.1)
    return f"processed: {item}"
```

### Caching

```python
from functools import lru_cache
import asyncio
from datetime import datetime, timedelta

class CachedServer(McpServer):
    def __init__(self, name: str):
        super().__init__(name)
        self._cache = {}
        self._cache_ttl = {}
    
    async def cached_call(self, key: str, func, ttl_seconds: int = 300):
        """Cache function results with TTL."""
        now = datetime.now()
        
        if key in self._cache:
            if now < self._cache_ttl[key]:
                return self._cache[key]
        
        result = await func()
        self._cache[key] = result
        self._cache_ttl[key] = now + timedelta(seconds=ttl_seconds)
        
        return result

server = CachedServer("cached-server")

@server.tool()
async def expensive_operation(query: str) -> str:
    """Perform expensive operation with caching."""
    return await server.cached_call(
        f"expensive:{query}",
        lambda: actually_expensive_operation(query),
        ttl_seconds=600
    )
```

## Deployment Considerations

### Production Configuration

```python
from mcp import McpServer
import logging.config

# Production logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'mcp_server.log',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)

# Production server setup
server = McpServer("production-server")

# Add health check endpoint
@server.tool()
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
```

## Next Steps

- [Building an MCP Client](./client.md)
- [Implementation and Hosting](./hosting.md)
- [Best Practices and Known Issues](./best-practices.md)
- [Complete Implementation Examples](../../patterns/mcp/)

## Additional Resources

- [MCP Server Examples](https://github.com/modelcontextprotocol/servers)
- [Python MCP SDK Documentation](https://mcp.readthedocs.io/)
- [MCP Protocol Specification](https://spec.modelcontextprotocol.io/)