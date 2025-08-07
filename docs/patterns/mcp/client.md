# How to Build an MCP Client

This guide covers building Model Context Protocol (MCP) clients that can connect to and interact with MCP servers.

## Prerequisites

- Python 3.8 or higher
- Understanding of async programming
- Basic knowledge of JSON-RPC protocol (see our [JSON-RPC guide](../../json-rpc.md))

## Installation

```bash
pip install mcp
```

For additional client features:
```bash
pip install mcp[client]    # Client-specific dependencies
pip install aiohttp        # For HTTP transport
pip install websockets     # For WebSocket transport
```

## Basic Client Implementation

### 1. Simple MCP Client

```python
from mcp import McpClient
from mcp.client.stdio import stdio_client
import asyncio
import json

async def basic_client_example():
    """Basic example of connecting to an MCP server."""
    
    # Connect to server via stdio
    async with stdio_client("python", "server.py") as client:
        # Initialize the connection
        await client.initialize()
        
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")
        
        # Call a tool
        result = await client.call_tool("calculate_sum", {"a": 5, "b": 3})
        print(f"Result: {result}")
        
        # List available resources
        resources = await client.list_resources()
        print(f"Available resources: {[resource.uri for resource in resources]}")
        
        # Read a resource
        if resources:
            content = await client.read_resource(resources[0].uri)
            print(f"Resource content: {content}")

if __name__ == "__main__":
    asyncio.run(basic_client_example())
```

### 2. HTTP Client

```python
from mcp.client.http import http_client
import asyncio

async def http_client_example():
    """Example of connecting to an MCP server via HTTP."""
    
    async with http_client("http://localhost:8000/mcp") as client:
        await client.initialize()
        
        # Get server capabilities
        capabilities = client.server_capabilities
        print(f"Server capabilities: {capabilities}")
        
        # Use available features
        if capabilities.tools:
            tools = await client.list_tools()
            for tool in tools:
                print(f"Tool: {tool.name} - {tool.description}")
        
        if capabilities.resources:
            resources = await client.list_resources()
            for resource in resources:
                print(f"Resource: {resource.uri} - {resource.name}")

if __name__ == "__main__":
    asyncio.run(http_client_example())
```

### 3. WebSocket Client

```python
from mcp.client.websocket import websocket_client
import asyncio

async def websocket_client_example():
    """Example of connecting to an MCP server via WebSocket."""
    
    async with websocket_client("ws://localhost:8080") as client:
        await client.initialize()
        
        # Real-time interaction with server
        result = await client.call_tool("get_weather", {"city": "New York"})
        print(f"Weather: {result}")

if __name__ == "__main__":
    asyncio.run(websocket_client_example())
```

## Advanced Client Features

### 4. Client with Error Handling

```python
from mcp import McpClient, McpError
from mcp.client.stdio import stdio_client
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustMcpClient:
    def __init__(self, server_command: list):
        self.server_command = server_command
        self.client = None
    
    async def __aenter__(self):
        try:
            self.client = await stdio_client(*self.server_command).__aenter__()
            await self.client.initialize()
            return self
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def safe_call_tool(self, name: str, arguments: dict = None):
        """Safely call a tool with error handling."""
        try:
            result = await self.client.call_tool(name, arguments or {})
            logger.info(f"Tool {name} called successfully")
            return result
        except McpError as e:
            logger.error(f"MCP Error calling {name}: {e.message}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling {name}: {e}")
            return None
    
    async def safe_read_resource(self, uri: str):
        """Safely read a resource with error handling."""
        try:
            content = await self.client.read_resource(uri)
            logger.info(f"Resource {uri} read successfully")
            return content
        except McpError as e:
            logger.error(f"MCP Error reading {uri}: {e.message}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading {uri}: {e}")
            return None

# Usage
async def robust_client_example():
    async with RobustMcpClient(["python", "server.py"]) as client:
        result = await client.safe_call_tool("calculate_sum", {"a": 10, "b": 5})
        if result:
            print(f"Calculation result: {result}")
```

### 5. Async Client Manager

```python
from mcp.client.stdio import stdio_client
from mcp.client.http import http_client
import asyncio
from contextlib import AsyncExitStack

class McpClientManager:
    def __init__(self):
        self.clients = {}
        self.exit_stack = None
    
    async def __aenter__(self):
        self.exit_stack = AsyncExitStack()
        await self.exit_stack.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.exit_stack:
            await self.exit_stack.__aexit__(exc_type, exc_val, exc_tb)
    
    async def add_stdio_client(self, name: str, command: list):
        """Add a stdio-based MCP client."""
        client = await self.exit_stack.enter_async_context(
            stdio_client(*command)
        )
        await client.initialize()
        self.clients[name] = client
        return client
    
    async def add_http_client(self, name: str, url: str):
        """Add an HTTP-based MCP client."""
        client = await self.exit_stack.enter_async_context(
            http_client(url)
        )
        await client.initialize()
        self.clients[name] = client
        return client
    
    async def call_tool_on_best_server(self, tool_name: str, arguments: dict):
        """Call a tool on the first server that supports it."""
        for name, client in self.clients.items():
            try:
                tools = await client.list_tools()
                if any(tool.name == tool_name for tool in tools):
                    result = await client.call_tool(tool_name, arguments)
                    return result, name
            except Exception as e:
                logger.warning(f"Failed to call {tool_name} on {name}: {e}")
                continue
        
        raise Exception(f"No server supports tool: {tool_name}")

# Usage
async def multi_client_example():
    async with McpClientManager() as manager:
        # Add multiple clients
        await manager.add_stdio_client("local", ["python", "local_server.py"])
        await manager.add_http_client("remote", "http://api.example.com/mcp")
        
        # Use the best available server for each tool
        result, server = await manager.call_tool_on_best_server(
            "calculate_sum", {"a": 10, "b": 5}
        )
        print(f"Result from {server}: {result}")
```

## Client Configuration

### 6. Configuration-Driven Client

```python
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ServerConfig:
    name: str
    type: str  # stdio, http, websocket
    config: Dict[str, Any]
    enabled: bool = True

class ConfigurableClient:
    def __init__(self, config_file: str = "client_config.json"):
        self.config = self.load_config(config_file)
        self.clients = {}
    
    def load_config(self, config_file: str) -> Dict[str, ServerConfig]:
        """Load client configuration from file."""
        if not os.path.exists(config_file):
            return {}
        
        with open(config_file) as f:
            data = json.load(f)
        
        configs = {}
        for name, server_data in data.get("servers", {}).items():
            configs[name] = ServerConfig(
                name=name,
                type=server_data["type"],
                config=server_data["config"],
                enabled=server_data.get("enabled", True)
            )
        
        return configs
    
    async def connect_all(self):
        """Connect to all enabled servers."""
        for name, server_config in self.config.items():
            if not server_config.enabled:
                continue
            
            try:
                client = await self.create_client(server_config)
                await client.initialize()
                self.clients[name] = client
                logger.info(f"Connected to {name}")
            except Exception as e:
                logger.error(f"Failed to connect to {name}: {e}")
    
    async def create_client(self, config: ServerConfig):
        """Create a client based on configuration."""
        if config.type == "stdio":
            return stdio_client(*config.config["command"])
        elif config.type == "http":
            return http_client(config.config["url"])
        elif config.type == "websocket":
            return websocket_client(config.config["url"])
        else:
            raise ValueError(f"Unknown client type: {config.type}")

# Example configuration file (client_config.json)
config_example = {
    "servers": {
        "local_tools": {
            "type": "stdio",
            "config": {"command": ["python", "tools_server.py"]},
            "enabled": True
        },
        "data_service": {
            "type": "http",
            "config": {"url": "http://data.example.com/mcp"},
            "enabled": True
        },
        "realtime_service": {
            "type": "websocket",
            "config": {"url": "ws://realtime.example.com"},
            "enabled": False
        }
    }
}
```

## Integration Patterns

### 7. AI Assistant Integration

```python
from mcp.client.stdio import stdio_client
import asyncio
from typing import List, Dict, Any

class AIAssistantWithMCP:
    def __init__(self):
        self.mcp_clients = {}
        self.available_tools = {}
    
    async def setup_mcp_connections(self, server_configs: List[Dict]):
        """Set up connections to MCP servers."""
        for config in server_configs:
            try:
                if config["type"] == "stdio":
                    client = stdio_client(*config["command"])
                else:
                    # Handle other transport types
                    continue
                
                async with client as conn:
                    await conn.initialize()
                    tools = await conn.list_tools()
                    
                    self.mcp_clients[config["name"]] = conn
                    for tool in tools:
                        self.available_tools[tool.name] = {
                            "server": config["name"],
                            "tool": tool
                        }
                        
            except Exception as e:
                print(f"Failed to connect to {config['name']}: {e}")
    
    async def process_user_request(self, user_input: str) -> str:
        """Process user request, potentially using MCP tools."""
        # Simple example: check if user wants to use a tool
        if "calculate" in user_input.lower():
            return await self.handle_calculation_request(user_input)
        elif "weather" in user_input.lower():
            return await self.handle_weather_request(user_input)
        else:
            return self.handle_general_request(user_input)
    
    async def handle_calculation_request(self, request: str) -> str:
        """Handle calculation requests using MCP tools."""
        if "calculate_sum" in self.available_tools:
            tool_info = self.available_tools["calculate_sum"]
            server = self.mcp_clients[tool_info["server"]]
            
            # Extract numbers (simplified)
            import re
            numbers = re.findall(r'\d+', request)
            if len(numbers) >= 2:
                result = await server.call_tool("calculate_sum", {
                    "a": float(numbers[0]),
                    "b": float(numbers[1])
                })
                return f"The sum is: {result}"
        
        return "I cannot perform calculations at the moment."
    
    async def handle_weather_request(self, request: str) -> str:
        """Handle weather requests using MCP tools."""
        if "get_weather" in self.available_tools:
            tool_info = self.available_tools["get_weather"]
            server = self.mcp_clients[tool_info["server"]]
            
            # Extract city (simplified)
            import re
            words = request.split()
            for i, word in enumerate(words):
                if word.lower() in ["in", "for"]:
                    if i + 1 < len(words):
                        city = words[i + 1]
                        result = await server.call_tool("get_weather", {"city": city})
                        return result
        
        return "I cannot get weather information at the moment."
    
    def handle_general_request(self, request: str) -> str:
        """Handle general requests without MCP tools."""
        return f"I understand you said: {request}"

# Usage
async def ai_assistant_example():
    assistant = AIAssistantWithMCP()
    
    # Set up MCP connections
    await assistant.setup_mcp_connections([
        {
            "name": "math_server",
            "type": "stdio",
            "command": ["python", "math_server.py"]
        },
        {
            "name": "weather_server",
            "type": "stdio",
            "command": ["python", "weather_server.py"]
        }
    ])
    
    # Process user requests
    responses = []
    requests = [
        "Calculate 15 + 25",
        "What's the weather in Paris?",
        "Hello, how are you?"
    ]
    
    for request in requests:
        response = await assistant.process_user_request(request)
        responses.append(f"User: {request}\nAssistant: {response}\n")
    
    return responses
```

## Testing MCP Clients

### 8. Client Testing Framework

```python
import pytest
from mcp.testing import MockMcpServer
from mcp import McpClient
import asyncio

class TestMcpClient:
    """Test suite for MCP client functionality."""
    
    @pytest.fixture
    async def mock_server(self):
        """Create a mock MCP server for testing."""
        server = MockMcpServer("test-server")
        
        @server.tool()
        async def test_tool(value: str) -> str:
            return f"processed: {value}"
        
        @server.resource("test://resource")
        async def test_resource():
            return "test resource content"
        
        return server
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, mock_server):
        """Test client initialization."""
        async with mock_server.create_client() as client:
            assert client.server_capabilities is not None
    
    @pytest.mark.asyncio
    async def test_tool_calling(self, mock_server):
        """Test calling tools through client."""
        async with mock_server.create_client() as client:
            result = await client.call_tool("test_tool", {"value": "hello"})
            assert result == "processed: hello"
    
    @pytest.mark.asyncio
    async def test_resource_reading(self, mock_server):
        """Test reading resources through client."""
        async with mock_server.create_client() as client:
            content = await client.read_resource("test://resource")
            assert "test resource content" in str(content)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_server):
        """Test client error handling."""
        async with mock_server.create_client() as client:
            with pytest.raises(Exception):
                await client.call_tool("nonexistent_tool", {})

# Integration tests
@pytest.mark.asyncio
async def test_real_server_integration():
    """Test integration with a real server process."""
    import subprocess
    import tempfile
    import os
    
    # Create a simple test server
    server_code = '''
from mcp import McpServer
import asyncio

server = McpServer("integration-test-server")

@server.tool()
async def echo(message: str) -> str:
    return f"echo: {message}"

if __name__ == "__main__":
    asyncio.run(server.run_stdio())
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(server_code)
        server_file = f.name
    
    try:
        # Test the integration
        from mcp.client.stdio import stdio_client
        
        async with stdio_client("python", server_file) as client:
            await client.initialize()
            
            result = await client.call_tool("echo", {"message": "test"})
            assert result == "echo: test"
            
    finally:
        os.unlink(server_file)
```

## Performance and Optimization

### 9. Connection Pooling

```python
from mcp.client.http import http_client
import asyncio
import aiohttp
from contextlib import asynccontextmanager

class PooledMcpClient:
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connector = None
        self.clients = {}
    
    async def __aenter__(self):
        self.connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.connector:
            await self.connector.close()
    
    @asynccontextmanager
    async def get_client(self, server_url: str):
        """Get a pooled client for the server."""
        if server_url not in self.clients:
            client = http_client(server_url, connector=self.connector)
            await client.initialize()
            self.clients[server_url] = client
        
        yield self.clients[server_url]

# Usage
async def pooled_client_example():
    async with PooledMcpClient(max_connections=20) as pool:
        # Make concurrent requests to different servers
        tasks = []
        
        for i in range(10):
            async with pool.get_client(f"http://server{i}.example.com/mcp") as client:
                task = client.call_tool("process_data", {"id": i})
                tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
```

### 10. Caching Client

```python
from datetime import datetime, timedelta
import hashlib
import json

class CachingMcpClient:
    def __init__(self, base_client, cache_ttl_seconds: int = 300):
        self.base_client = base_client
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.cache = {}
        self.cache_times = {}
    
    def _cache_key(self, method: str, *args, **kwargs):
        """Generate cache key."""
        key_data = f"{method}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cached_valid(self, key: str) -> bool:
        """Check if cached value is still valid."""
        if key not in self.cache_times:
            return False
        return datetime.now() - self.cache_times[key] < self.cache_ttl
    
    async def call_tool(self, name: str, arguments: dict = None):
        """Call tool with caching."""
        cache_key = self._cache_key("call_tool", name, arguments)
        
        if self._is_cached_valid(cache_key):
            return self.cache[cache_key]
        
        result = await self.base_client.call_tool(name, arguments)
        self.cache[cache_key] = result
        self.cache_times[cache_key] = datetime.now()
        
        return result
    
    async def list_tools(self):
        """List tools with caching."""
        cache_key = self._cache_key("list_tools")
        
        if self._is_cached_valid(cache_key):
            return self.cache[cache_key]
        
        result = await self.base_client.list_tools()
        self.cache[cache_key] = result
        self.cache_times[cache_key] = datetime.now()
        
        return result
    
    # Delegate other methods to base client
    def __getattr__(self, name):
        return getattr(self.base_client, name)
```

## Next Steps

- [Implementation and Hosting](./hosting.md)
- [Best Practices and Known Issues](./best-practices.md)
- [Complete Implementation Examples](../../patterns/mcp/)

## Additional Resources

- [MCP Client Examples](https://github.com/modelcontextprotocol/clients)
- [Python MCP SDK Documentation](https://mcp.readthedocs.io/)
- [MCP Protocol Specification](https://spec.modelcontextprotocol.io/)