# MCP (Model Context Protocol) Pattern

Complete implementation examples and patterns for building MCP servers and clients.

## Overview

This directory contains practical implementations of the Model Context Protocol (MCP), including:
- Simple server and client examples
- Advanced patterns with authentication and error handling
- Integration examples with real-world scenarios
- Deployment and hosting configurations

## Quick Start

1. **Install Dependencies**
   ```bash
   cd patterns/mcp
   pip install -r requirements.txt
   ```

2. **Run Simple Server Example**
   ```bash
   python server/simple_server.py
   ```

3. **Test with Simple Client**
   ```bash
   python client/simple_client.py
   ```

## Directory Structure

```
mcp/
├── README.md                    # This file
├── requirements.txt             # MCP-specific dependencies
├── server/                      # Server implementation examples
│   ├── README.md
│   ├── simple_server.py         # Basic MCP server
│   ├── advanced_server.py       # Production-ready server
│   └── requirements.txt
├── client/                      # Client implementation examples
│   ├── README.md
│   ├── simple_client.py         # Basic MCP client
│   ├── advanced_client.py       # Feature-rich client
│   └── requirements.txt
└── examples/                    # Complete integration examples
    ├── README.md
    ├── basic_integration/       # Simple client-server integration
    └── hosted_solution/         # Cloud deployment example
```

## Key Features Demonstrated

### Server Implementations
- **Tool Registration**: How to create and register MCP tools
- **Resource Management**: Providing read-only data sources
- **Prompt Templates**: Dynamic prompt generation
- **Error Handling**: Robust error management and logging
- **Authentication**: Secure access control
- **Performance**: Async operations and connection pooling

### Client Implementations
- **Connection Management**: Stdio, HTTP, and WebSocket transports
- **Tool Discovery**: Listing and calling available tools
- **Resource Access**: Reading server-provided resources
- **Error Recovery**: Handling connection failures and retries
- **Batch Operations**: Efficient bulk operations

### Integration Examples
- **AI Assistant Integration**: Using MCP in AI applications
- **Multi-Server Clients**: Connecting to multiple MCP servers
- **Load Balancing**: Distributing requests across servers
- **Monitoring**: Health checks and performance metrics

## Getting Started Examples

### 1. Basic Tool Server

```python
from mcp import McpServer
import asyncio

server = McpServer("example-server")

@server.tool()
async def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"

@server.tool()
async def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

if __name__ == "__main__":
    asyncio.run(server.run_stdio())
```

### 2. Basic Client

```python
from mcp.client.stdio import stdio_client
import asyncio

async def main():
    async with stdio_client("python", "server.py") as client:
        await client.initialize()
        
        # List available tools
        tools = await client.list_tools()
        print("Available tools:", [tool.name for tool in tools])
        
        # Call a tool
        result = await client.call_tool("greet", {"name": "World"})
        print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Documentation References

For detailed implementation guidance, see:
- [MCP Concept Overview](../../docs/patterns/mcp/README.md)
- [Building an MCP Server](../../docs/patterns/mcp/server.md)
- [Building an MCP Client](../../docs/patterns/mcp/client.md)
- [Implementation and Hosting](../../docs/patterns/mcp/hosting.md)
- [Best Practices and Known Issues](../../docs/patterns/mcp/best-practices.md)

## Contributing

When adding new examples:
1. Follow the established patterns and conventions
2. Include comprehensive error handling
3. Add appropriate documentation and comments
4. Provide both simple and advanced variations
5. Include tests where applicable

## External Resources

- [Official MCP Documentation](https://modelcontextprotocol.io/)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Community Examples](https://github.com/modelcontextprotocol/servers)