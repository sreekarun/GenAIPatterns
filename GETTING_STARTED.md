# GenAI Patterns - Getting Started

Welcome to the GenAI Patterns repository! This guide will help you get started with the available patterns, starting with MCP (Model Context Protocol).

## Quick Start

### 1. Set up your environment

```bash
# Clone the repository
git clone https://github.com/sreekarun/GenAIPatterns.git
cd GenAIPatterns

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up MCP pattern specifically
pip install -r patterns/mcp/requirements.txt
```

### 2. Try the MCP Examples

#### Simple Server and Client

```bash
# Terminal 1: Start the simple server
cd patterns/mcp
python server/simple_server.py

# Terminal 2: Run the simple client
cd patterns/mcp
python client/simple_client.py
```

#### Advanced Examples

```bash
# Run the advanced client with demo
python client/advanced_client.py --demo

# Or with custom configuration
python client/advanced_client.py --config client_config.json
```

### 3. Automated Setup

Use the setup script for automated environment configuration:

```bash
# Set up everything for MCP pattern
python scripts/setup.py --pattern mcp --dev

# Validate your setup
python scripts/setup.py --validate --pattern mcp
```

## What's Included

### 📚 Comprehensive Documentation
- [MCP Concept Overview](docs/patterns/mcp/README.md) - Understanding MCP fundamentals
- [Building MCP Servers](docs/patterns/mcp/server.md) - Complete server implementation guide
- [Building MCP Clients](docs/patterns/mcp/client.md) - Client development patterns
- [Hosting & Deployment](docs/patterns/mcp/hosting.md) - Production deployment strategies
- [Best Practices](docs/patterns/mcp/best-practices.md) - Common pitfalls and solutions

### 💻 Working Code Examples
- **Simple Server** (`patterns/mcp/server/simple_server.py`) - Basic MCP server with tools, resources, and prompts
- **Advanced Server** (`patterns/mcp/server/advanced_server.py`) - Production-ready server with auth, metrics, and database integration
- **Simple Client** (`patterns/mcp/client/simple_client.py`) - Basic client with error handling and discovery
- **Advanced Client** (`patterns/mcp/client/advanced_client.py`) - Multi-server client with caching, retries, and performance optimization

### 🚀 Integration Examples
- **Basic Integration** (`patterns/mcp/examples/basic_integration/`) - Complete client-server integration
- **Hosted Solution** (`patterns/mcp/examples/hosted_solution/`) - Cloud deployment configurations

### 🛠 Development Tools
- **Setup Script** (`scripts/setup.py`) - Automated environment setup
- **Configuration Templates** - Ready-to-use configuration files
- **Docker Support** - Containerized deployment options

## Key Features Demonstrated

### Server Capabilities
✅ **Tool Registration** - How to create and register MCP tools  
✅ **Resource Management** - Providing read-only data sources  
✅ **Prompt Templates** - Dynamic prompt generation  
✅ **Error Handling** - Robust error management and logging  
✅ **Authentication** - Secure access control  
✅ **Performance** - Async operations and connection pooling  

### Client Features
✅ **Multi-Transport** - stdio, HTTP, and WebSocket support  
✅ **Tool Discovery** - Automatic server capability detection  
✅ **Resource Access** - Reading server-provided resources  
✅ **Error Recovery** - Connection failures and retry logic  
✅ **Batch Operations** - Efficient bulk operations  
✅ **Caching** - Response caching for performance  

### Production Features
✅ **Cloud Deployment** - AWS, GCP, Azure configurations  
✅ **Containerization** - Docker and Kubernetes manifests  
✅ **Monitoring** - Health checks and metrics collection  
✅ **Security** - Authentication and authorization patterns  
✅ **Scalability** - Load balancing and auto-scaling  

## Architecture Overview

```
GenAI Application
       ↓
   MCP Client ←→ MCP Server ←→ External Services
                     ↓              ↓
                 Tools         Resources
                 Prompts       Data Sources
```

## Example Use Cases

### 1. **AI Assistant with External Tools**
```python
# Client discovers and uses server tools
tools = await client.list_tools()
result = await client.call_tool("analyze_document", {"url": doc_url})
```

### 2. **Multi-Service Integration**
```python
# Connect to multiple specialized servers
database_result = await db_client.call_tool("query", {"sql": query})
api_result = await api_client.call_tool("fetch", {"endpoint": url})
```

### 3. **Resource-Aware Processing**
```python
# Access configuration and data resources
config = await client.read_resource("config://settings")
data = await client.read_resource("data://training_set")
```

## Next Steps

1. **📖 Read the Documentation** - Start with [MCP Concepts](docs/patterns/mcp/README.md)
2. **🔧 Run Examples** - Try the simple server and client
3. **🏗 Build Your Own** - Use examples as templates for your use case
4. **🌟 Contribute** - Add new patterns or improve existing ones

## Coming Soon

- **OpenAI Agentic Framework** patterns
- **LangGraph** implementation examples  
- **Multi-agent systems** patterns
- **RAG (Retrieval-Augmented Generation)** implementations

## Support

- **Documentation**: Comprehensive guides in the `docs/` directory
- **Examples**: Working code in the `patterns/` directory
- **Issues**: Report problems or request features via GitHub issues
- **Resources**: Links to external MCP documentation and community

---

🎯 **Goal**: Provide practical, production-ready patterns for building GenAI applications with modern protocols and frameworks.

🔗 **External Resources**:
- [Official MCP Documentation](https://modelcontextprotocol.io/)
- [MCP GitHub Repository](https://github.com/modelcontextprotocol)
- [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk)