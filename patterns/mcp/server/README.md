# MCP Server Examples

This directory contains MCP server implementation examples, from basic to production-ready configurations.

## Examples

### [simple_server.py](./simple_server.py)
A basic MCP server demonstrating:
- Tool registration and implementation
- Resource provision
- Prompt templates
- Basic error handling

**Usage:**
```bash
python simple_server.py
```

### [advanced_server.py](./advanced_server.py)
A production-ready MCP server featuring:
- Authentication and authorization
- Database integration
- Connection pooling
- Monitoring and metrics
- Comprehensive error handling
- Async best practices

**Usage:**
```bash
# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost/db"
export API_KEY="your-api-key"
export JWT_SECRET="your-jwt-secret"

# Run the server
python advanced_server.py
```

## Common Patterns

### Tool Implementation
```python
@server.tool()
async def my_tool(param1: str, param2: int = 10) -> str:
    """Tool description for AI assistant."""
    # Validation
    if not param1:
        raise McpError(code=-32602, message="param1 is required")
    
    # Implementation
    result = await process_data(param1, param2)
    return result
```

### Resource Management
```python
@server.resource("config://settings")
async def get_settings():
    """Provide configuration data."""
    return TextContent(
        type="text",
        text=json.dumps(settings_data)
    )
```

### Error Handling
```python
try:
    result = await risky_operation()
    return result
except ValueError as e:
    raise McpError(
        code=-32602,
        message=f"Invalid input: {str(e)}",
        data={"error_type": "validation"}
    )
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise McpError(
        code=-32603,
        message="Internal server error"
    )
```

## Testing Your Server

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest test_servers.py
```

## Deployment

See [hosting documentation](../../../docs/patterns/mcp/hosting.md) for production deployment strategies.