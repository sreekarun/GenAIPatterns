# Basic MCP Integration Example

A complete example demonstrating fundamental client-server integration patterns with MCP.

## Overview

This example shows:
- Setting up a simple MCP server with multiple tools
- Creating a client that discovers and uses server capabilities
- Handling errors gracefully
- Managing resources and configuration
- Basic monitoring and logging

## Components

```
basic_integration/
├── README.md              # This file
├── docker-compose.yml     # Complete development environment
├── run_example.py         # Main example runner
├── server.py              # Example MCP server
├── client.py              # Example MCP client
├── config.json            # Configuration file
└── requirements.txt       # Dependencies
```

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with Docker (recommended)**
   ```bash
   docker-compose up
   ```

3. **Or run manually**
   ```bash
   # Terminal 1: Start server
   python server.py
   
   # Terminal 2: Run client
   python client.py
   ```

4. **Or run the complete example**
   ```bash
   python run_example.py
   ```

## What This Example Demonstrates

### Server Capabilities
- **File Operations**: Read, write, and list files
- **Data Processing**: Transform and analyze text data
- **System Information**: Get system stats and health info
- **Configuration Management**: Provide and update configuration

### Client Features
- **Server Discovery**: Automatically discover available tools and resources
- **Tool Execution**: Call server tools with proper error handling
- **Resource Access**: Read configuration and data resources
- **Health Monitoring**: Check server health and performance

### Integration Patterns
- **Request Batching**: Execute multiple operations efficiently
- **Error Recovery**: Handle failures and retry operations
- **Configuration Management**: Dynamic configuration updates
- **Monitoring**: Track performance and health metrics

## Example Usage Scenarios

### 1. File Management System
```python
# Client requests file operations
files = await client.call_tool("list_files", {"directory": "/data"})
content = await client.call_tool("read_file", {"path": "/data/config.json"})
await client.call_tool("write_file", {"path": "/data/output.txt", "content": "Hello"})
```

### 2. Data Processing Pipeline
```python
# Process data through multiple steps
raw_data = "sample text data"
cleaned = await client.call_tool("clean_text", {"text": raw_data})
analyzed = await client.call_tool("analyze_sentiment", {"text": cleaned})
summary = await client.call_tool("summarize", {"text": cleaned, "max_length": 100})
```

### 3. System Monitoring
```python
# Monitor system health
health = await client.call_tool("get_system_health")
metrics = await client.call_tool("get_performance_metrics")
config = await client.read_resource("config://system")
```

## Configuration

The `config.json` file contains:
```json
{
  "server": {
    "name": "basic-integration-server",
    "log_level": "INFO",
    "max_file_size": 1048576,
    "allowed_directories": ["/tmp", "/data"]
  },
  "client": {
    "timeout": 30,
    "retry_attempts": 3,
    "enable_caching": true
  }
}
```

## Docker Setup

The `docker-compose.yml` provides:
- MCP server container
- Shared volumes for data persistence
- Network configuration for client-server communication
- Environment variable management

## Learning Objectives

After running this example, you'll understand:

1. **Basic MCP Concepts**
   - Tool registration and calling
   - Resource management
   - Error handling

2. **Client-Server Communication**
   - Connection establishment
   - Message exchange patterns
   - Transport layer details

3. **Real-World Integration**
   - Configuration management
   - Error recovery strategies
   - Performance considerations

4. **Development Workflow**
   - Local development setup
   - Testing strategies
   - Debugging techniques

## Extending the Example

You can extend this example by:

1. **Adding New Tools**
   ```python
   @server.tool()
   async def your_custom_tool(param: str) -> str:
       # Your implementation
       return result
   ```

2. **Adding Resources**
   ```python
   @server.resource("custom://resource")
   async def your_resource():
       return TextContent(type="text", text="resource data")
   ```

3. **Modifying Client Logic**
   ```python
   # Add custom client operations
   result = await client.call_tool("your_custom_tool", {"param": "value"})
   ```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure server is running before starting client
   - Check port availability (default: stdio transport)

2. **Tool Not Found**
   - Verify tool is registered on server
   - Check tool name spelling

3. **Permission Errors**
   - Ensure proper file/directory permissions
   - Check `allowed_directories` configuration

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python run_example.py
```

## Next Steps

After mastering this basic example:
1. Explore the [hosted solution](../hosted_solution/) example
2. Review [best practices](../../../docs/patterns/mcp/best-practices.md)
3. Build your own custom integration
4. Deploy to production environment