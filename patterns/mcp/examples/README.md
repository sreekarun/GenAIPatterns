# MCP Integration Examples

This directory contains complete integration examples demonstrating real-world MCP usage patterns.

## Examples

### [basic_integration/](./basic_integration/)
A complete example showing:
- Simple client-server integration
- Tool usage in practice
- Resource management
- Error handling patterns

### [streamable_http_server.py](./streamable_http_server.py) & [streamable_http_client.py](./streamable_http_client.py)
**New!** Comprehensive Streamable HTTP examples featuring:
- Real-time data streaming with Server-Sent Events (SSE)
- Progressive result delivery for long-running operations
- Authentication and security patterns
- Performance monitoring and metrics
- HTTP-compatible transport for web integration

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

### Streaming Data Processing
```python
# Server provides streaming tool
@server.tool()
async def stream_data_analysis(dataset: str, chunk_size: int = 100) -> str:
    for i in range(0, total_items, chunk_size):
        # Process chunk and yield progress
        await asyncio.sleep(0.1)
        progress = min(i + chunk_size, total_items)
        yield f"Processed {progress}/{total_items} items"
    return "Analysis complete"

# Client consumes stream
async for update in client.call_streaming_tool("stream_data_analysis", 
                                                dataset="user_data"):
    print(f"Progress: {update}")
```

### Real-time Monitoring
```python
# Stream real-time metrics via HTTP
@app.get("/metrics-stream")
async def metrics_stream():
    async def generate_metrics():
        while True:
            metrics = {"cpu": get_cpu_usage(), "memory": get_memory_usage()}
            yield f"data: {json.dumps(metrics)}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(generate_metrics(), media_type="text/event-stream")
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