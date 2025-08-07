#!/usr/bin/env python3
"""
Advanced MCP Server Example

A production-ready Model Context Protocol server featuring:
- Authentication and authorization
- Database integration
- Connection pooling
- Monitoring and metrics
- Comprehensive error handling
- Async best practices

Usage:
    # Set environment variables
    export DATABASE_URL="postgresql://user:pass@localhost/db"
    export API_KEY="your-api-key"
    export JWT_SECRET="your-jwt-secret"
    export REDIS_URL="redis://localhost:6379"
    
    # Run the server
    python advanced_server.py

For HTTP transport:
    uvicorn advanced_server:app --host 0.0.0.0 --port 8000
"""

import os
import json
import asyncio
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from functools import wraps

from mcp import McpServer
from mcp.types import Tool, Resource, TextContent, McpError
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Optional dependencies for advanced features
try:
    import aioredis
    import asyncpg
    import jwt
    from prometheus_client import Counter, Histogram, generate_latest
    import structlog
except ImportError as e:
    print(f"Optional dependency missing: {e}")
    print("Install with: pip install aioredis asyncpg PyJWT prometheus-client structlog")

# Configuration
class Config:
    """Server configuration from environment variables."""
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.api_key = os.getenv("API_KEY", "dev-key-123")
        self.jwt_secret = os.getenv("JWT_SECRET", "dev-secret-456")
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.enable_auth = os.getenv("ENABLE_AUTH", "false").lower() == "true"
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"

config = Config()

# Logging setup
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics (if enabled)
if config.enable_metrics:
    try:
        REQUEST_COUNT = Counter('mcp_requests_total', 'Total MCP requests', ['method', 'status'])
        REQUEST_DURATION = Histogram('mcp_request_duration_seconds', 'MCP request duration')
    except:
        logger.warning("Metrics disabled due to missing prometheus-client")
        config.enable_metrics = False

# Connection pools
class ConnectionManager:
    """Manage database and Redis connections."""
    
    def __init__(self):
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_pool: Optional[aioredis.Redis] = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize connection pools."""
        if self.initialized:
            return
        
        try:
            # Database pool
            if config.database_url:
                self.db_pool = await asyncpg.create_pool(
                    config.database_url,
                    min_size=2,
                    max_size=10,
                    command_timeout=30
                )
                logger.info("Database pool initialized")
            
            # Redis pool
            if config.redis_url:
                self.redis_pool = aioredis.from_url(
                    config.redis_url,
                    decode_responses=True
                )
                await self.redis_pool.ping()
                logger.info("Redis connection initialized")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
    
    async def cleanup(self):
        """Clean up connections."""
        if self.db_pool:
            await self.db_pool.close()
        
        if self.redis_pool:
            await self.redis_pool.close()
    
    @asynccontextmanager
    async def get_db_connection(self):
        """Get database connection from pool."""
        if not self.db_pool:
            raise McpError(code=-32603, message="Database not available")
        
        async with self.db_pool.acquire() as connection:
            yield connection
    
    async def cache_get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if not self.redis_pool:
            return None
        
        try:
            return await self.redis_pool.get(key)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    async def cache_set(self, key: str, value: str, ttl: int = 300):
        """Set value in cache with TTL."""
        if not self.redis_pool:
            return
        
        try:
            await self.redis_pool.setex(key, ttl, value)
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

# Global connection manager
conn_manager = ConnectionManager()

# Authentication
class AuthManager:
    """Handle authentication and authorization."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def create_token(self, user_id: str, permissions: List[str]) -> str:
        """Create JWT token."""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=1),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def check_permission(self, user_payload: dict, required_permission: str) -> bool:
        """Check if user has required permission."""
        user_permissions = user_payload.get('permissions', [])
        return required_permission in user_permissions or 'admin' in user_permissions

auth_manager = AuthManager(config.jwt_secret)

# Decorators for monitoring and authentication
def monitor_tool(func):
    """Decorator to monitor tool calls."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        tool_name = func.__name__
        
        try:
            result = await func(*args, **kwargs)
            if config.enable_metrics:
                REQUEST_COUNT.labels(method=tool_name, status='success').inc()
            logger.info(f"Tool {tool_name} completed successfully")
            return result
        except Exception as e:
            if config.enable_metrics:
                REQUEST_COUNT.labels(method=tool_name, status='error').inc()
            logger.error(f"Tool {tool_name} failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            if config.enable_metrics:
                REQUEST_DURATION.observe(duration)
    
    return wrapper

def require_permission(permission: str):
    """Decorator to require specific permission."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not config.enable_auth:
                return await func(*args, **kwargs)
            
            # In a real implementation, you'd extract the token from the request context
            # For this example, we'll assume it's passed in kwargs
            auth_token = kwargs.pop('auth_token', None)
            
            if not auth_token:
                raise McpError(code=-32401, message="Authentication required")
            
            user_payload = auth_manager.verify_token(auth_token)
            if not user_payload:
                raise McpError(code=-32401, message="Invalid or expired token")
            
            if not auth_manager.check_permission(user_payload, permission):
                raise McpError(code=-32403, message="Insufficient permissions")
            
            kwargs['user_context'] = user_payload
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# Create MCP server
server = McpServer("advanced-mcp-server")

# Advanced tool implementations
@server.tool()
@monitor_tool
async def execute_sql_query(query: str, user_context: dict = None) -> dict:
    """Execute a SQL query (with permission checks).
    
    Args:
        query: SQL query to execute
        user_context: User authentication context
        
    Returns:
        Query results
    """
    # Validate query (basic SQL injection prevention)
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
    query_upper = query.upper()
    
    if any(keyword in query_upper for keyword in dangerous_keywords):
        if not user_context or not auth_manager.check_permission(user_context, 'sql:write'):
            raise McpError(
                code=-32403,
                message="Write operations require special permissions",
                data={"required_permission": "sql:write"}
            )
    
    try:
        async with conn_manager.get_db_connection() as conn:
            # Execute query
            if query_upper.startswith('SELECT'):
                rows = await conn.fetch(query)
                return {
                    "rows": [dict(row) for row in rows],
                    "count": len(rows)
                }
            else:
                result = await conn.execute(query)
                return {"result": result}
                
    except Exception as e:
        logger.error(f"SQL query failed: {e}")
        raise McpError(
            code=-32603,
            message="Query execution failed",
            data={"error": str(e)}
        )

@server.tool()
@monitor_tool
async def cached_api_call(url: str, cache_ttl: int = 300) -> dict:
    """Make an API call with caching.
    
    Args:
        url: URL to call
        cache_ttl: Cache time-to-live in seconds
        
    Returns:
        API response data
    """
    # Create cache key
    cache_key = f"api_call:{hashlib.md5(url.encode()).hexdigest()}"
    
    # Try to get from cache first
    cached_result = await conn_manager.cache_get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for {url}")
        return json.loads(cached_result)
    
    # Make API call
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    raise McpError(
                        code=-32603,
                        message=f"API call failed with status {response.status}",
                        data={"url": url, "status": response.status}
                    )
                
                data = await response.json()
                
                # Cache the result
                await conn_manager.cache_set(
                    cache_key, 
                    json.dumps(data), 
                    cache_ttl
                )
                
                logger.info(f"API call successful: {url}")
                return data
                
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise McpError(
            code=-32603,
            message="API call failed",
            data={"error": str(e), "url": url}
        )

@server.tool()
@monitor_tool
async def batch_process_data(items: List[str], batch_size: int = 10) -> List[dict]:
    """Process multiple items in batches.
    
    Args:
        items: List of items to process
        batch_size: Number of items to process in each batch
        
    Returns:
        List of processing results
    """
    if not items:
        return []
    
    if batch_size > 50:
        raise McpError(
            code=-32602,
            message="Batch size too large",
            data={"max_batch_size": 50, "requested": batch_size}
        )
    
    async def process_item(item: str) -> dict:
        """Process a single item."""
        # Simulate processing
        await asyncio.sleep(0.1)
        return {
            "item": item,
            "processed_at": datetime.utcnow().isoformat(),
            "length": len(item),
            "hash": hashlib.md5(item.encode()).hexdigest()
        }
    
    results = []
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch concurrently
        batch_tasks = [process_item(item) for item in batch]
        batch_results = await asyncio.gather(*batch_tasks)
        
        results.extend(batch_results)
        logger.info(f"Processed batch {i//batch_size + 1}: {len(batch)} items")
    
    return results

@server.tool()
@monitor_tool
async def get_server_stats() -> dict:
    """Get server statistics and health information.
    
    Returns:
        Server statistics
    """
    try:
        import psutil
        
        # System stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Application stats
        process = psutil.Process()
        app_memory = process.memory_info()
        
        # Connection stats
        db_available = conn_manager.db_pool is not None
        redis_available = conn_manager.redis_pool is not None
        
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / 1024 / 1024
            },
            "application": {
                "memory_mb": app_memory.rss / 1024 / 1024,
                "connections": {
                    "database": "available" if db_available else "unavailable",
                    "redis": "available" if redis_available else "unavailable"
                }
            },
            "config": {
                "environment": config.environment,
                "auth_enabled": config.enable_auth,
                "metrics_enabled": config.enable_metrics
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get server stats: {e}")
        raise McpError(
            code=-32603,
            message="Failed to get server statistics",
            data={"error": str(e)}
        )

# Resource implementations
@server.resource("config://database")
async def get_database_config():
    """Get database configuration (non-sensitive parts)."""
    if not config.database_url:
        raise McpError(
            code=-32603,
            message="Database not configured"
        )
    
    # Parse URL to extract non-sensitive info
    from urllib.parse import urlparse
    parsed = urlparse(config.database_url)
    
    config_data = {
        "host": parsed.hostname,
        "port": parsed.port,
        "database": parsed.path.lstrip('/') if parsed.path else None,
        "ssl_mode": "prefer",
        "pool_size": "2-10 connections"
    }
    
    return TextContent(
        type="text",
        text=json.dumps(config_data, indent=2)
    )

@server.resource("metrics://prometheus")
async def get_metrics():
    """Get Prometheus metrics."""
    if not config.enable_metrics:
        raise McpError(
            code=-32603,
            message="Metrics not enabled"
        )
    
    try:
        metrics_data = generate_latest().decode('utf-8')
        return TextContent(
            type="text",
            text=metrics_data
        )
    except Exception as e:
        raise McpError(
            code=-32603,
            message="Failed to generate metrics",
            data={"error": str(e)}
        )

@server.list_resources()
async def list_available_resources():
    """List available resources."""
    resources = []
    
    if config.database_url:
        resources.append(Resource(
            uri="config://database",
            name="Database Configuration",
            description="Database connection configuration (non-sensitive)",
            mimeType="application/json"
        ))
    
    if config.enable_metrics:
        resources.append(Resource(
            uri="metrics://prometheus",
            name="Prometheus Metrics",
            description="Server metrics in Prometheus format",
            mimeType="text/plain"
        ))
    
    return resources

# FastAPI integration for HTTP transport
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting Advanced MCP Server...")
    await conn_manager.initialize()
    logger.info("Server initialization complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down server...")
    await conn_manager.cleanup()
    logger.info("Shutdown complete")

app = FastAPI(
    title="Advanced MCP Server",
    description="Production-ready MCP server with advanced features",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Metrics endpoint
@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    if not config.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    
    return generate_latest()

# Authentication endpoint
@app.post("/auth/token")
async def create_auth_token(credentials: dict):
    """Create authentication token."""
    api_key = credentials.get("api_key")
    
    if api_key != config.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # In a real implementation, you'd validate user credentials
    token = auth_manager.create_token(
        user_id="demo_user",
        permissions=["sql:read", "api:call", "data:process"]
    )
    
    return {"access_token": token, "token_type": "bearer"}

# Include MCP endpoints
from mcp.server.fastapi import create_mcp_router
mcp_router = create_mcp_router(server)
app.include_router(mcp_router, prefix="/mcp")

def main():
    """Main entry point for stdio transport."""
    logger.info("Starting Advanced MCP Server (stdio mode)...")
    
    # Initialize connections in sync context
    async def init_and_run():
        await conn_manager.initialize()
        await server.run_stdio()
    
    try:
        asyncio.run(init_and_run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

def run_http_server():
    """Run server with HTTP transport."""
    uvicorn.run(
        "advanced_server:app",
        host="0.0.0.0",
        port=8000,
        log_level=config.log_level.lower(),
        access_log=True
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        run_http_server()
    else:
        main()