# JSON-RPC 2.0 Documentation

This guide provides comprehensive documentation for JSON-RPC (JavaScript Object Notation Remote Procedure Call), which is the underlying protocol used by many patterns in this repository, particularly the Model Context Protocol (MCP).

## Table of Contents

- [What is JSON-RPC?](#what-is-json-rpc)
- [JSON-RPC 2.0 Specification](#json-rpc-20-specification)
- [Message Types](#message-types)
- [Request Format](#request-format)
- [Response Format](#response-format)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Common Issues](#common-issues)
- [Resources](#resources)

## What is JSON-RPC?

JSON-RPC is a stateless, light-weight remote procedure call (RPC) protocol that uses JSON as the data format. It allows for:

- **Remote procedure calls**: Execute functions on a remote server as if they were local
- **Language agnostic**: Can be used between different programming languages
- **Transport agnostic**: Works over HTTP, WebSockets, TCP, stdin/stdout, and more
- **Simple protocol**: Easy to implement and debug

### Key Benefits

1. **Simplicity**: Minimal overhead and easy to understand
2. **Flexibility**: Works with various transport mechanisms
3. **Standardized**: Well-defined specification (JSON-RPC 2.0)
4. **Debugging friendly**: Human-readable JSON format

## JSON-RPC 2.0 Specification

JSON-RPC 2.0 is the current version and includes several improvements over 1.0:

- **Version identification**: All messages include `"jsonrpc": "2.0"`
- **Named parameters**: Support for both positional and named parameters
- **Batch requests**: Multiple requests can be sent in a single message
- **Notification support**: Fire-and-forget messages without responses

### Core Principles

1. **Request-Response**: Client sends requests, server sends responses
2. **Notifications**: One-way messages that don't expect responses
3. **Batch operations**: Multiple requests/responses in a single message
4. **Error handling**: Standardized error response format

## Message Types

### 1. Request
A request message sent from client to server to invoke a method.

### 2. Response
A response message sent from server to client in reply to a request.

### 3. Notification
A request message without an `id` field, indicating no response is expected.

## Request Format

All requests must include the following fields:

```json
{
  "jsonrpc": "2.0",
  "method": "method_name",
  "params": {...},
  "id": 1
}
```

### Required Fields

- **`jsonrpc`**: Must be exactly `"2.0"`
- **`method`**: String containing the name of the method to be invoked
- **`id`**: Unique identifier for the request (number, string, or null)

### Optional Fields

- **`params`**: Parameters for the method call (object or array)

### Parameter Formats

#### Named Parameters (Object)
```json
{
  "jsonrpc": "2.0",
  "method": "calculate_sum",
  "params": {
    "a": 10,
    "b": 20
  },
  "id": 1
}
```

#### Positional Parameters (Array)
```json
{
  "jsonrpc": "2.0",
  "method": "calculate_sum",
  "params": [10, 20],
  "id": 1
}
```

#### No Parameters
```json
{
  "jsonrpc": "2.0",
  "method": "get_status",
  "id": 1
}
```

## Response Format

### Success Response

```json
{
  "jsonrpc": "2.0",
  "result": {...},
  "id": 1
}
```

### Required Fields

- **`jsonrpc`**: Must be exactly `"2.0"`
- **`result`**: The result of the method call
- **`id`**: Same identifier as the corresponding request

### Error Response

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32601,
    "message": "Method not found",
    "data": {...}
  },
  "id": 1
}
```

## Error Handling

### Standard Error Codes

| Code | Message | Description |
|------|---------|-------------|
| -32700 | Parse error | Invalid JSON was received |
| -32600 | Invalid Request | The JSON sent is not a valid Request object |
| -32601 | Method not found | The method does not exist / is not available |
| -32602 | Invalid params | Invalid method parameter(s) |
| -32603 | Internal error | Internal JSON-RPC error |

### Custom Error Codes

Application-specific errors should use codes outside the reserved range (-32768 to -32000).

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": 1001,
    "message": "Database connection failed",
    "data": {
      "details": "Connection timeout after 30 seconds",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  },
  "id": 1
}
```

## Examples

### Basic Tool Call (MCP Context)

#### Request
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "calculate_sum",
    "arguments": {
      "a": 15,
      "b": 25
    }
  },
  "id": 1
}
```

#### Success Response
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "40"
      }
    ]
  },
  "id": 1
}
```

### Server Initialization (MCP Context)

#### Request
```json
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {}
    },
    "clientInfo": {
      "name": "example-client",
      "version": "1.0.0"
    }
  },
  "id": 1
}
```

#### Response
```json
{
  "jsonrpc": "2.0",
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {
        "listChanged": true
      }
    },
    "serverInfo": {
      "name": "example-server",
      "version": "1.0.0"
    }
  },
  "id": 1
}
```

### Notification (No Response Expected)

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/message",
  "params": {
    "level": "info",
    "logger": "server",
    "data": {
      "message": "Server started successfully"
    }
  }
}
```

### Batch Request

```json
[
  {
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 1
  },
  {
    "jsonrpc": "2.0",
    "method": "resources/list",
    "id": 2
  }
]
```

### Batch Response

```json
[
  {
    "jsonrpc": "2.0",
    "result": {
      "tools": [...]
    },
    "id": 1
  },
  {
    "jsonrpc": "2.0",
    "result": {
      "resources": [...]
    },
    "id": 2
  }
]
```

## Best Practices

### 1. Always Use JSON-RPC 2.0

```python
# ✅ Correct
request = {
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {...},
    "id": 1
}

# ❌ Incorrect - missing or wrong version
request = {
    "jsonrpc": "1.0",  # Wrong version
    "method": "tools/call",
    "params": {...},
    "id": 1
}
```

### 2. Use Meaningful Method Names

```python
# ✅ Good - clear, hierarchical naming
"tools/call"
"tools/list"
"resources/read"
"notifications/message"

# ❌ Poor - unclear or inconsistent
"tc"
"get_tools"
"read-resource"
```

### 3. Validate Input Parameters

```python
from pydantic import BaseModel, ValidationError

class CalculateSumParams(BaseModel):
    a: float
    b: float

def handle_calculate_sum(params):
    try:
        validated_params = CalculateSumParams(**params)
        return validated_params.a + validated_params.b
    except ValidationError as e:
        raise JsonRpcError(-32602, "Invalid params", str(e))
```

### 4. Handle Errors Gracefully

```python
def handle_request(request):
    try:
        # Process request
        result = process_method(request["method"], request.get("params"))
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": request["id"]
        }
    except MethodNotFoundError:
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32601,
                "message": "Method not found"
            },
            "id": request.get("id")
        }
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": str(e)
            },
            "id": request.get("id")
        }
```

### 5. Use Unique Request IDs

```python
import uuid

# Generate unique IDs
request_id = str(uuid.uuid4())

# Or use sequential IDs
request_counter = 0

def get_next_id():
    global request_counter
    request_counter += 1
    return request_counter
```

### 6. Implement Timeouts

```python
import asyncio

async def send_request_with_timeout(request, timeout=30):
    try:
        response = await asyncio.wait_for(
            send_request(request),
            timeout=timeout
        )
        return response
    except asyncio.TimeoutError:
        raise JsonRpcError(-32603, "Request timeout")
```

## Common Issues

### 1. Version Mismatch

**Problem**: Using wrong JSON-RPC version
```json
{
  "jsonrpc": "1.0",  // Wrong!
  "method": "test",
  "id": 1
}
```

**Solution**: Always use "2.0"
```json
{
  "jsonrpc": "2.0",  // Correct!
  "method": "test",
  "id": 1
}
```

### 2. Missing Required Fields

**Problem**: Missing required fields
```json
{
  "method": "test",  // Missing jsonrpc and id
  "params": {}
}
```

**Solution**: Include all required fields
```json
{
  "jsonrpc": "2.0",
  "method": "test",
  "params": {},
  "id": 1
}
```

### 3. Invalid JSON Format

**Problem**: Malformed JSON
```json
{
  "jsonrpc": "2.0",
  "method": "test",
  "id": 1,  // Trailing comma
}
```

**Solution**: Use valid JSON
```json
{
  "jsonrpc": "2.0",
  "method": "test",
  "id": 1
}
```

### 4. ID Mismatch in Response

**Problem**: Response ID doesn't match request ID
```python
# Request
{"jsonrpc": "2.0", "method": "test", "id": 1}

# Response
{"jsonrpc": "2.0", "result": "success", "id": 2}  // Wrong ID!
```

**Solution**: Ensure response ID matches request ID
```python
{"jsonrpc": "2.0", "result": "success", "id": 1}  // Correct!
```

## Transport Implementation Examples

### HTTP Transport

```python
import json
import httpx

async def send_jsonrpc_http(url: str, request: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json=request,
            headers={"Content-Type": "application/json"}
        )
        return response.json()
```

### Stdio Transport

```python
import sys
import json

def send_jsonrpc_stdio(request: dict) -> dict:
    # Send request
    request_str = json.dumps(request)
    sys.stdout.write(request_str + "\n")
    sys.stdout.flush()
    
    # Read response
    response_str = sys.stdin.readline().strip()
    return json.loads(response_str)
```

### WebSocket Transport

```python
import json
import websockets

async def send_jsonrpc_websocket(uri: str, request: dict) -> dict:
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(request))
        response_str = await websocket.recv()
        return json.loads(response_str)
```

## Resources

### Official Specifications
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [JSON Schema for JSON-RPC 2.0](https://json-schema.org/learn/miscellaneous-examples.html#jsonrpc)

### Related Documentation
- [MCP Server Documentation](./patterns/mcp/server.md)
- [MCP Client Documentation](./patterns/mcp/client.md)
- [MCP Best Practices](./patterns/mcp/best-practices.md)

### Libraries and Tools
- [Python: jsonrpclib-pelix](https://pypi.org/project/jsonrpclib-pelix/)
- [Python: aiohttp-jsonrpc](https://pypi.org/project/aiohttp-jsonrpc/)
- [JavaScript: jayson](https://www.npmjs.com/package/jayson)
- [Go: gorilla/rpc](https://github.com/gorilla/rpc)

### Testing Tools
- [JSON-RPC Tester](https://jsonrpc.org/test/)
- [Postman JSON-RPC Testing](https://learning.postman.com/docs/sending-requests/supported-api-frameworks/json-rpc/)