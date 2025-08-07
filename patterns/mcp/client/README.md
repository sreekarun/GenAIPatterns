# MCP Client Examples

This directory contains MCP client implementation examples, from basic connectivity to advanced integration patterns.

## Examples

### [simple_client.py](./simple_client.py)
A basic MCP client demonstrating:
- Connection to MCP servers
- Tool discovery and calling
- Resource reading
- Basic error handling

**Usage:**
```bash
# Start the simple server first
python ../server/simple_server.py &

# Run the client
python simple_client.py
```

### [advanced_client.py](./advanced_client.py)
An advanced MCP client featuring:
- Multi-server connection management
- Authentication handling
- Connection pooling and retries
- Caching and performance optimization
- Comprehensive error handling

**Usage:**
```bash
# Set environment variables
export SERVER1_URL="http://localhost:8000/mcp"
export SERVER2_COMMAND="python ../server/simple_server.py"
export AUTH_TOKEN="your-auth-token"

# Run the advanced client
python advanced_client.py
```

## Common Patterns

### Basic Client Connection
```python
from mcp.client.stdio import stdio_client

async with stdio_client("python", "server.py") as client:
    await client.initialize()
    
    # Use the client
    tools = await client.list_tools()
    result = await client.call_tool("tool_name", {"param": "value"})
```

### HTTP Client Connection
```python
from mcp.client.http import http_client

async with http_client("http://localhost:8000/mcp") as client:
    await client.initialize()
    
    # Use the client
    result = await client.call_tool("tool_name", {"param": "value"})
```

### Error Handling
```python
from mcp.types import McpError

try:
    result = await client.call_tool("tool_name", arguments)
except McpError as e:
    print(f"MCP Error: {e.message} (code: {e.code})")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Testing Your Client

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest test_clients.py
```

## Integration Examples

See the [examples directory](../examples/) for complete client-server integration scenarios.