# MCP Best Practices and Known Issues

This guide covers best practices, common pitfalls, and known issues when working with Model Context Protocol (MCP) implementations.

## Best Practices

### 1. Server Design Principles

#### Keep Tools Focused and Single-Purpose

```python
# ❌ Bad: Overly complex tool that does too much
@server.tool()
async def do_everything(action: str, data: str, format: str, output: str) -> str:
    if action == "process":
        # Process data
        pass
    elif action == "format":
        # Format data
        pass
    elif action == "export":
        # Export data
        pass
    # ... many more conditions

# ✅ Good: Focused, single-purpose tools
@server.tool()
async def process_data(data: str) -> str:
    """Process raw data into structured format."""
    return processed_data

@server.tool()
async def format_output(data: str, format_type: str) -> str:
    """Format data according to specified format."""
    return formatted_data

@server.tool()
async def export_data(data: str, destination: str) -> str:
    """Export data to specified destination."""
    return export_result
```

#### Use Proper Type Hints and Validation

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum

class DataFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    XML = "xml"

class ProcessingOptions(BaseModel):
    format: DataFormat
    include_metadata: bool = True
    max_records: int = Field(default=1000, ge=1, le=10000)
    
    @validator('max_records')
    def validate_max_records(cls, v):
        if v > 5000:
            raise ValueError('max_records should not exceed 5000 for performance reasons')
        return v

@server.tool()
async def process_with_options(
    data: str,
    options: ProcessingOptions
) -> str:
    """Process data with validated options."""
    # Implementation with type safety
    return result
```

#### Implement Proper Error Handling

```python
from mcp.types import McpError
import logging

logger = logging.getLogger(__name__)

@server.tool()
async def robust_tool(input_data: str) -> str:
    """Tool with comprehensive error handling."""
    try:
        # Validate input
        if not input_data or len(input_data.strip()) == 0:
            raise McpError(
                code=-32602,
                message="Invalid input: data cannot be empty",
                data={"input_length": len(input_data)}
            )
        
        # Perform operation
        result = await perform_operation(input_data)
        
        # Validate output
        if not result:
            raise McpError(
                code=-32603,
                message="Operation completed but produced no result",
                data={"input": input_data}
            )
        
        logger.info(f"Tool completed successfully for input length: {len(input_data)}")
        return result
        
    except McpError:
        # Re-raise MCP errors as-is
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise McpError(
            code=-32602,
            message=f"Invalid input: {str(e)}",
            data={"error_type": "validation"}
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise McpError(
            code=-32603,
            message="Internal server error",
            data={"error_type": type(e).__name__}
        )

async def perform_operation(data: str) -> str:
    """Perform the actual operation."""
    # Implementation here
    pass
```

### 2. Resource Management

#### Implement Resource Caching

```python
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

class ResourceCache:
    def __init__(self, default_ttl: int = 300):
        self.cache: Dict[str, Any] = {}
        self.cache_times: Dict[str, datetime] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached resource if not expired."""
        if key not in self.cache:
            return None
        
        if datetime.now() - self.cache_times[key] > timedelta(seconds=self.default_ttl):
            # Cache expired
            del self.cache[key]
            del self.cache_times[key]
            return None
        
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set cached resource."""
        self.cache[key] = value
        self.cache_times[key] = datetime.now()

cache = ResourceCache()

@server.resource("config://database")
async def get_database_config():
    """Get database configuration with caching."""
    cached = cache.get("database_config")
    if cached:
        return cached
    
    # Expensive operation to get config
    config = await fetch_database_config()
    cache.set("database_config", config)
    return config
```

#### Use Connection Pooling

```python
import aiohttp
import asyncpg
from contextlib import asynccontextmanager

class ConnectionManager:
    def __init__(self):
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.db_pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Initialize connection pools."""
        # HTTP connection pool
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        self.http_session = aiohttp.ClientSession(connector=connector)
        
        # Database connection pool
        self.db_pool = await asyncpg.create_pool(
            database_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
    
    async def cleanup(self):
        """Clean up connections."""
        if self.http_session:
            await self.http_session.close()
        
        if self.db_pool:
            await self.db_pool.close()
    
    @asynccontextmanager
    async def get_db_connection(self):
        """Get database connection from pool."""
        async with self.db_pool.acquire() as connection:
            yield connection
    
    async def http_request(self, method: str, url: str, **kwargs):
        """Make HTTP request using pooled session."""
        async with self.http_session.request(method, url, **kwargs) as response:
            return await response.json()

# Global connection manager
conn_manager = ConnectionManager()

@server.tool()
async def fetch_external_data(api_endpoint: str) -> dict:
    """Fetch data using connection pool."""
    return await conn_manager.http_request("GET", api_endpoint)
```

### 3. Security Best Practices

#### Input Validation and Sanitization

```python
import re
from html import escape
from urllib.parse import urlparse

class InputValidator:
    @staticmethod
    def validate_url(url: str) -> str:
        """Validate and sanitize URL."""
        if not url:
            raise ValueError("URL cannot be empty")
        
        # Parse URL
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ['http', 'https']:
            raise ValueError("Only HTTP and HTTPS URLs are allowed")
        
        # Check for localhost/private IPs (SSRF protection)
        if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
            raise ValueError("Localhost URLs are not allowed")
        
        return url
    
    @staticmethod
    def validate_sql_input(input_str: str) -> str:
        """Validate input for SQL injection protection."""
        # Remove potential SQL injection patterns
        dangerous_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(UNION|OR|AND)\b.*\b(SELECT|INSERT|UPDATE|DELETE)\b)"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                raise ValueError("Input contains potentially dangerous SQL patterns")
        
        return input_str
    
    @staticmethod
    def sanitize_html(input_str: str) -> str:
        """Sanitize HTML input."""
        return escape(input_str)

validator = InputValidator()

@server.tool()
async def secure_api_call(url: str, query: str) -> str:
    """Make secure API call with input validation."""
    # Validate inputs
    safe_url = validator.validate_url(url)
    safe_query = validator.validate_sql_input(query)
    
    # Proceed with validated inputs
    return await make_api_call(safe_url, safe_query)
```

#### Authentication and Authorization

```python
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class AuthenticationManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.active_sessions: Dict[str, datetime] = {}
    
    def create_token(self, user_id: str, permissions: list) -> str:
        """Create JWT token with user permissions."""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=1),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def check_permission(self, user_payload: dict, required_permission: str) -> bool:
        """Check if user has required permission."""
        user_permissions = user_payload.get('permissions', [])
        return required_permission in user_permissions

auth = AuthenticationManager(os.getenv('JWT_SECRET', 'your-secret-key'))

def require_permission(permission: str):
    """Decorator to require specific permission for tool access."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract token from context (implementation depends on transport)
            token = extract_auth_token()
            
            if not token:
                raise McpError(code=-32401, message="Authentication required")
            
            user_payload = auth.verify_token(token)
            if not user_payload:
                raise McpError(code=-32401, message="Invalid or expired token")
            
            if not auth.check_permission(user_payload, permission):
                raise McpError(code=-32403, message="Insufficient permissions")
            
            # Add user context to kwargs
            kwargs['user_context'] = user_payload
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

@server.tool()
@require_permission('data:read')
async def read_sensitive_data(query: str, user_context: dict = None) -> str:
    """Read sensitive data with permission check."""
    user_id = user_context['user_id']
    # Implement data access with user context
    return f"Data for user {user_id}: {query}"
```

### 4. Performance Optimization

#### Async Best Practices

```python
import asyncio
from asyncio import Semaphore
from typing import List

class PerformantServer:
    def __init__(self, max_concurrent_operations: int = 10):
        self.semaphore = Semaphore(max_concurrent_operations)
        self.operation_cache = {}
    
    async def rate_limited_operation(self, operation_id: str):
        """Perform operation with rate limiting."""
        async with self.semaphore:
            return await self.expensive_operation(operation_id)
    
    async def expensive_operation(self, operation_id: str):
        """Simulate expensive operation."""
        await asyncio.sleep(1)  # Simulate work
        return f"Result for {operation_id}"
    
    async def batch_operations(self, operation_ids: List[str]) -> List[str]:
        """Perform multiple operations concurrently."""
        tasks = [
            self.rate_limited_operation(op_id) 
            for op_id in operation_ids
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

performant_server = PerformantServer()

@server.tool()
async def batch_process(items: List[str]) -> List[str]:
    """Process multiple items efficiently."""
    return await performant_server.batch_operations(items)
```

#### Memory Management

```python
import weakref
import gc
from memory_profiler import profile

class MemoryEfficientServer:
    def __init__(self):
        self.weak_cache = weakref.WeakValueDictionary()
        self.operation_count = 0
    
    async def memory_conscious_operation(self, data: str) -> str:
        """Operation that manages memory usage."""
        self.operation_count += 1
        
        # Force garbage collection every 100 operations
        if self.operation_count % 100 == 0:
            gc.collect()
        
        # Use weak references for caching
        cache_key = hash(data)
        if cache_key in self.weak_cache:
            return self.weak_cache[cache_key]
        
        # Process data
        result = await self.process_large_data(data)
        
        # Store in weak cache
        self.weak_cache[cache_key] = result
        return result
    
    async def process_large_data(self, data: str) -> str:
        """Process data efficiently."""
        # Use generators for large data processing
        def process_chunks():
            chunk_size = 1000
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]
        
        processed_chunks = []
        for chunk in process_chunks():
            # Process chunk
            processed_chunks.append(f"processed:{chunk}")
        
        return "".join(processed_chunks)

memory_efficient = MemoryEfficientServer()
```

### 5. Testing Strategies

#### Comprehensive Unit Testing

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from mcp.testing import MockMcpServer, MockMcpClient

class TestMcpServer:
    @pytest.fixture
    async def mock_server(self):
        """Create mock server for testing."""
        server = MockMcpServer("test-server")
        
        @server.tool()
        async def test_tool(input_data: str) -> str:
            if input_data == "error":
                raise ValueError("Test error")
            return f"processed: {input_data}"
        
        return server
    
    @pytest.mark.asyncio
    async def test_successful_tool_call(self, mock_server):
        """Test successful tool execution."""
        async with mock_server.create_client() as client:
            result = await client.call_tool("test_tool", {"input_data": "hello"})
            assert result == "processed: hello"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_server):
        """Test error handling in tools."""
        async with mock_server.create_client() as client:
            with pytest.raises(Exception) as exc_info:
                await client.call_tool("test_tool", {"input_data": "error"})
            assert "Test error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_tool_with_external_dependency(self):
        """Test tool that depends on external service."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock external API response
            mock_response = AsyncMock()
            mock_response.json.return_value = {"result": "mocked_data"}
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Test tool
            result = await external_api_tool("test_query")
            assert "mocked_data" in result

@pytest.mark.asyncio
async def external_api_tool(query: str) -> str:
    """Tool that calls external API."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/search?q={query}") as response:
            data = await response.json()
            return f"Result: {data['result']}"
```

#### Integration Testing

```python
import asyncio
import subprocess
import tempfile
import os
from pathlib import Path

class TestMcpIntegration:
    @pytest.fixture
    async def server_process(self):
        """Start actual server process for integration testing."""
        # Create temporary server file
        server_code = '''
from mcp import McpServer
import asyncio

server = McpServer("integration-test-server")

@server.tool()
async def integration_tool(data: str) -> str:
    return f"integration: {data}"

if __name__ == "__main__":
    asyncio.run(server.run_stdio())
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(server_code)
            server_file = f.name
        
        # Start server process
        process = subprocess.Popen(
            ['python', server_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        yield process
        
        # Cleanup
        process.terminate()
        process.wait()
        os.unlink(server_file)
    
    @pytest.mark.asyncio
    async def test_stdio_communication(self, server_process):
        """Test communication with server via stdio."""
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
        
        server_process.stdin.write(json.dumps(init_request) + "\n")
        server_process.stdin.flush()
        
        # Read response
        response_line = server_process.stdout.readline()
        response = json.loads(response_line)
        
        assert response["result"]["protocolVersion"] == "2024-11-05"
```

## Known Issues and Solutions

### 1. Common Protocol Issues

#### JSON-RPC Version Mismatch

**Important**: MCP uses JSON-RPC 2.0. For detailed information about JSON-RPC, see our [JSON-RPC guide](../../json-rpc.md).

```python
# ❌ Problem: Using wrong JSON-RPC version
request = {
    "jsonrpc": "1.0",  # Wrong version
    "method": "tools/call",
    "params": {...},
    "id": 1
}

# ✅ Solution: Use correct JSON-RPC 2.0 format
request = {
    "jsonrpc": "2.0",  # Correct version
    "method": "tools/call",
    "params": {...},
    "id": 1
}
```

#### Missing Protocol Version in Initialize

```python
# ❌ Problem: Missing protocol version
@server.initialize()
async def handle_initialize(params):
    return {
        "capabilities": {...}
        # Missing protocolVersion
    }

# ✅ Solution: Include protocol version
@server.initialize()
async def handle_initialize(params):
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {...}
    }
```

### 2. Transport Layer Issues

#### Stdio Buffer Issues

```python
import sys
import json

# ❌ Problem: Not flushing stdout
def send_response(response):
    print(json.dumps(response))
    # Missing flush

# ✅ Solution: Always flush stdout
def send_response(response):
    print(json.dumps(response))
    sys.stdout.flush()
```

#### HTTP Content-Type Issues

```python
from fastapi import Request, HTTPException

# ❌ Problem: Not checking content type
@app.post("/mcp")
async def mcp_endpoint(request: Request):
    body = await request.json()  # May fail with wrong content type

# ✅ Solution: Validate content type
@app.post("/mcp")
async def mcp_endpoint(request: Request):
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(400, "Content-Type must be application/json")
    
    body = await request.json()
```

### 3. Resource Management Issues

#### Memory Leaks in Long-Running Servers

```python
import gc
import weakref
from typing import Dict, Any

class LeakPreventionServer:
    def __init__(self):
        # Use weak references to prevent circular references
        self.clients = weakref.WeakSet()
        self.resources = weakref.WeakValueDictionary()
        self.operation_count = 0
    
    async def add_client(self, client):
        """Add client with weak reference."""
        self.clients.add(client)
    
    async def cleanup_resources(self):
        """Periodic cleanup to prevent memory leaks."""
        self.operation_count += 1
        
        if self.operation_count % 1000 == 0:
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"Memory usage: {memory_mb:.2f} MB")
```

#### Connection Pool Exhaustion

```python
import asyncio
from contextlib import asynccontextmanager

class PoolManager:
    def __init__(self, max_connections: int = 10):
        self.semaphore = asyncio.Semaphore(max_connections)
        self.active_connections = 0
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection with timeout and proper cleanup."""
        try:
            # Wait for available connection with timeout
            await asyncio.wait_for(
                self.semaphore.acquire(), 
                timeout=30.0
            )
            
            self.active_connections += 1
            yield "connection"
            
        except asyncio.TimeoutError:
            raise Exception("Connection pool exhausted")
        finally:
            self.active_connections -= 1
            self.semaphore.release()
    
    def get_stats(self):
        """Get pool statistics."""
        return {
            "active_connections": self.active_connections,
            "available_connections": self.semaphore._value
        }
```

### 4. Error Handling Antipatterns

#### Silent Failures

```python
# ❌ Bad: Silent failures
@server.tool()
async def bad_error_handling(data: str) -> str:
    try:
        result = await risky_operation(data)
        return result
    except:
        return "error"  # Silent failure, no information

# ✅ Good: Proper error reporting
@server.tool()
async def good_error_handling(data: str) -> str:
    try:
        result = await risky_operation(data)
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise McpError(
            code=-32602,
            message=f"Invalid input: {str(e)}",
            data={"input": data, "error_type": "validation"}
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise McpError(
            code=-32603,
            message="Internal server error",
            data={"error_type": type(e).__name__}
        )
```

#### Generic Error Messages

```python
# ❌ Bad: Generic error messages
raise McpError(code=-32603, message="Something went wrong")

# ✅ Good: Specific, actionable error messages
raise McpError(
    code=-32602,
    message="Invalid email format: must contain @ symbol and valid domain",
    data={
        "field": "email",
        "value": email,
        "expected_format": "user@domain.com"
    }
)
```

### 5. Performance Pitfalls

#### Blocking Operations in Async Context

```python
import time
import asyncio

# ❌ Bad: Blocking operation in async function
@server.tool()
async def blocking_tool(data: str) -> str:
    time.sleep(5)  # Blocks entire event loop
    return f"processed: {data}"

# ✅ Good: Proper async operation
@server.tool()
async def async_tool(data: str) -> str:
    await asyncio.sleep(5)  # Non-blocking
    return f"processed: {data}"

# ✅ Good: Run blocking code in thread pool
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

@server.tool()
async def cpu_intensive_tool(data: str) -> str:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, 
        cpu_intensive_operation, 
        data
    )
    return result

def cpu_intensive_operation(data: str) -> str:
    # CPU-intensive work here
    return f"processed: {data}"
```

#### N+1 Query Problem

```python
# ❌ Bad: N+1 queries
@server.tool()
async def get_user_posts_bad(user_ids: List[str]) -> List[dict]:
    results = []
    for user_id in user_ids:  # N queries
        posts = await db.fetch_user_posts(user_id)
        results.append({"user_id": user_id, "posts": posts})
    return results

# ✅ Good: Batch query
@server.tool()
async def get_user_posts_good(user_ids: List[str]) -> List[dict]:
    # Single query to get all posts
    all_posts = await db.fetch_posts_by_users(user_ids)
    
    # Group by user_id
    user_posts = {}
    for post in all_posts:
        user_id = post['user_id']
        if user_id not in user_posts:
            user_posts[user_id] = []
        user_posts[user_id].append(post)
    
    return [
        {"user_id": user_id, "posts": user_posts.get(user_id, [])}
        for user_id in user_ids
    ]
```

## Debugging and Troubleshooting

### 6. Debugging Tools

#### Request/Response Logging

```python
import json
import time
from functools import wraps

def log_mcp_calls(func):
    """Decorator to log MCP calls for debugging."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        call_id = id(func)
        
        logger.info(f"[{call_id}] MCP call started: {func.__name__}")
        logger.debug(f"[{call_id}] Args: {args}, Kwargs: {kwargs}")
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"[{call_id}] MCP call completed in {duration:.3f}s")
            logger.debug(f"[{call_id}] Result: {result}")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{call_id}] MCP call failed after {duration:.3f}s: {e}")
            raise
    
    return wrapper

@server.tool()
@log_mcp_calls
async def debuggable_tool(data: str) -> str:
    """Tool with debug logging."""
    return f"processed: {data}"
```

#### Health Check Implementation

```python
import psutil
import asyncio
from datetime import datetime

@server.tool()
async def health_check() -> dict:
    """Comprehensive health check."""
    try:
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Check async event loop
        loop = asyncio.get_event_loop()
        pending_tasks = len([
            task for task in asyncio.all_tasks(loop) 
            if not task.done()
        ])
        
        # Check external dependencies
        db_healthy = await check_database_health()
        api_healthy = await check_external_api_health()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / 1024 / 1024
            },
            "application": {
                "pending_tasks": pending_tasks,
                "uptime_seconds": time.time() - start_time
            },
            "dependencies": {
                "database": "healthy" if db_healthy else "unhealthy",
                "external_api": "healthy" if api_healthy else "unhealthy"
            }
        }
        
        # Determine overall health
        if not db_healthy or not api_healthy or cpu_percent > 90 or memory.percent > 90:
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

async def check_database_health() -> bool:
    """Check database connectivity."""
    try:
        # Simple query to test database
        await db.execute("SELECT 1")
        return True
    except:
        return False

async def check_external_api_health() -> bool:
    """Check external API health."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.example.com/health", timeout=5) as response:
                return response.status == 200
    except:
        return False
```

## Migration and Upgrade Strategies

### 7. Version Compatibility

```python
from packaging import version

class VersionCompatibilityHandler:
    SUPPORTED_VERSIONS = ["2024-11-05", "2024-10-01"]
    
    def __init__(self, server_version: str = "2024-11-05"):
        self.server_version = server_version
    
    def is_compatible(self, client_version: str) -> bool:
        """Check if client version is compatible."""
        return client_version in self.SUPPORTED_VERSIONS
    
    def get_compatibility_features(self, client_version: str) -> dict:
        """Get available features for client version."""
        if client_version == "2024-11-05":
            return {
                "tools": True,
                "resources": True,
                "prompts": True,
                "streaming": True
            }
        elif client_version == "2024-10-01":
            return {
                "tools": True,
                "resources": True,
                "prompts": False,
                "streaming": False
            }
        else:
            return {}

@server.initialize()
async def handle_initialize(params):
    """Initialize with version compatibility check."""
    client_version = params.get("protocolVersion", "")
    compatibility = VersionCompatibilityHandler()
    
    if not compatibility.is_compatible(client_version):
        raise McpError(
            code=-32600,
            message=f"Unsupported protocol version: {client_version}",
            data={
                "supported_versions": compatibility.SUPPORTED_VERSIONS,
                "client_version": client_version
            }
        )
    
    features = compatibility.get_compatibility_features(client_version)
    
    return {
        "protocolVersion": compatibility.server_version,
        "capabilities": {
            "tools": {"listChanged": True} if features["tools"] else None,
            "resources": {"subscribe": True} if features["resources"] else None,
            "prompts": {"listChanged": True} if features["prompts"] else None
        }
    }
```

## Next Steps

- [Complete Implementation Examples](../../patterns/mcp/)
- [MCP Concept Overview](./README.md)
- [Building an MCP Server](./server.md)
- [Building an MCP Client](./client.md)

## Additional Resources

- [MCP Specification Issues](https://github.com/modelcontextprotocol/specification/issues)
- [Community Discussions](https://github.com/modelcontextprotocol/community)
- [Best Practices Repository](https://github.com/modelcontextprotocol/best-practices)
- [Performance Benchmarks](https://github.com/modelcontextprotocol/benchmarks)