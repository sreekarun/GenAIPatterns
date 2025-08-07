# MCP Integration Examples

This directory contains complete integration examples demonstrating real-world MCP usage patterns.

## Examples

### [basic_integration/](./basic_integration/)
A complete example showing:
- Simple client-server integration
- Tool usage in practice
- Resource management
- Error handling patterns

### [hosted_solution/](./hosted_solution/)
A production-ready example featuring:
- Cloud deployment configuration
- Docker containerization
- Kubernetes manifests
- CI/CD pipeline setup
- Monitoring and logging

## Running Examples

Each example includes its own README with specific setup instructions. Generally:

1. **Navigate to the example directory**
   ```bash
   cd basic_integration  # or hosted_solution
   ```

2. **Follow the example-specific setup**
   See the README.md in each directory for detailed instructions.

3. **Run the example**
   Most examples include run scripts or docker-compose files for easy execution.

## Example Structure

Each example follows this structure:
```
example_name/
├── README.md              # Example overview and setup
├── docker-compose.yml     # Local development setup
├── server/                # Server implementation
├── client/                # Client implementation
├── config/                # Configuration files
├── scripts/               # Utility scripts
└── docs/                  # Example-specific documentation
```

## Learning Path

1. **Start with basic_integration**: Learn fundamental client-server patterns
2. **Progress to hosted_solution**: Understand production deployment
3. **Experiment with configurations**: Modify examples for your use case
4. **Build custom integrations**: Use examples as templates

## Common Integration Patterns

### Simple Request-Response
```python
# Client sends request
result = await client.call_tool("process_data", {"data": input_data})

# Server processes and responds
@server.tool()
async def process_data(data: str) -> str:
    return f"Processed: {data}"
```

### Resource Sharing
```python
# Server provides resources
@server.resource("data://config")
async def get_config():
    return TextContent(type="text", text=json.dumps(config))

# Client reads resources
config = await client.read_resource("data://config")
```

### Batch Processing
```python
# Client sends batch request
operations = [
    {"type": "call_tool", "tool": "process", "arguments": {"data": item}}
    for item in batch_data
]
results = await client.batch_operations(operations)
```

## Best Practices Demonstrated

- **Error handling**: Graceful failure management
- **Resource management**: Proper connection lifecycle
- **Security**: Authentication and authorization patterns
- **Performance**: Caching and connection pooling
- **Monitoring**: Health checks and metrics collection
- **Deployment**: Production-ready configurations

## Contributing Examples

To add a new example:
1. Create a new directory with descriptive name
2. Include comprehensive README
3. Provide working code with comments
4. Add deployment configurations
5. Include tests and validation scripts