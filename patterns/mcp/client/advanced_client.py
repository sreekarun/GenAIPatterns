#!/usr/bin/env python3
"""
Advanced MCP Client Example

An advanced Model Context Protocol client featuring:
- Multi-server connection management
- Authentication handling
- Connection pooling and retries
- Caching and performance optimization
- Comprehensive error handling
- Health monitoring and metrics

Usage:
    # Set environment variables
    export MCP_CONFIG_FILE="client_config.json"
    export AUTH_TOKEN="your-auth-token"
    
    # Run the advanced client
    python advanced_client.py

Configuration file example (client_config.json):
{
    "servers": {
        "local_tools": {
            "type": "stdio",
            "command": ["python", "../server/simple_server.py"],
            "enabled": true,
            "retry_attempts": 3,
            "timeout": 30
        },
        "api_service": {
            "type": "http",
            "url": "http://localhost:8000/mcp",
            "enabled": true,
            "retry_attempts": 5,
            "timeout": 10,
            "headers": {
                "Authorization": "Bearer ${AUTH_TOKEN}"
            }
        }
    },
    "settings": {
        "enable_caching": true,
        "cache_ttl": 300,
        "max_concurrent_requests": 10,
        "enable_metrics": true
    }
}
"""

import os
import json
import asyncio
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
import weakref

from mcp.client.stdio import stdio_client
from mcp.types import McpError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    type: str  # stdio, http, websocket
    enabled: bool = True
    retry_attempts: int = 3
    timeout: int = 30
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClientSettings:
    """Global client settings."""
    enable_caching: bool = True
    cache_ttl: int = 300
    max_concurrent_requests: int = 10
    enable_metrics: bool = True
    connection_timeout: int = 30
    request_timeout: int = 60

class ClientMetrics:
    """Track client performance metrics."""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.server_stats = {}
    
    def record_request(self, server_name: str, duration: float, success: bool):
        """Record a request metric."""
        self.request_count += 1
        self.total_response_time += duration
        
        if not success:
            self.error_count += 1
        
        if server_name not in self.server_stats:
            self.server_stats[server_name] = {
                "requests": 0,
                "errors": 0,
                "total_time": 0.0
            }
        
        stats = self.server_stats[server_name]
        stats["requests"] += 1
        stats["total_time"] += duration
        
        if not success:
            stats["errors"] += 1
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        error_rate = (
            self.error_count / self.request_count 
            if self.request_count > 0 else 0
        )
        
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0
        )
        
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "server_stats": self.server_stats
        }

class ResponseCache:
    """Simple in-memory cache for MCP responses."""
    
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, datetime] = {}
    
    def _generate_key(self, server_name: str, method: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key."""
        key_data = f"{server_name}:{method}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, server_name: str, method: str, args: tuple, kwargs: dict) -> Optional[Any]:
        """Get cached value if not expired."""
        key = self._generate_key(server_name, method, args, kwargs)
        
        if key not in self.cache:
            return None
        
        # Check if expired
        if datetime.now() - self.timestamps[key] > timedelta(seconds=self.ttl):
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        return self.cache[key]
    
    def set(self, server_name: str, method: str, args: tuple, kwargs: dict, value: Any):
        """Set cached value."""
        key = self._generate_key(server_name, method, args, kwargs)
        self.cache[key] = value
        self.timestamps[key] = datetime.now()
    
    def clear(self):
        """Clear all cached values."""
        self.cache.clear()
        self.timestamps.clear()
    
    def cleanup_expired(self):
        """Remove expired entries."""
        now = datetime.now()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if now - timestamp > timedelta(seconds=self.ttl)
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)

class AdvancedMcpClient:
    """Advanced MCP client with multi-server support and advanced features."""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or os.getenv("MCP_CONFIG_FILE", "client_config.json")
        self.servers: Dict[str, ServerConfig] = {}
        self.settings = ClientSettings()
        self.clients: Dict[str, Any] = {}
        self.exit_stack: Optional[AsyncExitStack] = None
        self.cache = ResponseCache()
        self.metrics = ClientMetrics()
        self.semaphore: Optional[asyncio.Semaphore] = None
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file) as f:
                    config_data = json.load(f)
                
                # Parse server configurations
                for name, server_data in config_data.get("servers", {}).items():
                    self.servers[name] = ServerConfig(
                        name=name,
                        type=server_data["type"],
                        enabled=server_data.get("enabled", True),
                        retry_attempts=server_data.get("retry_attempts", 3),
                        timeout=server_data.get("timeout", 30),
                        config=server_data
                    )
                
                # Parse settings
                settings_data = config_data.get("settings", {})
                self.settings = ClientSettings(
                    enable_caching=settings_data.get("enable_caching", True),
                    cache_ttl=settings_data.get("cache_ttl", 300),
                    max_concurrent_requests=settings_data.get("max_concurrent_requests", 10),
                    enable_metrics=settings_data.get("enable_metrics", True),
                    connection_timeout=settings_data.get("connection_timeout", 30),
                    request_timeout=settings_data.get("request_timeout", 60)
                )
                
                # Update cache TTL
                self.cache.ttl = self.settings.cache_ttl
                
                logger.info(f"Loaded configuration from {self.config_file}")
            else:
                logger.warning(f"Config file {self.config_file} not found, using defaults")
                self._create_default_config()
                
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration."""
        self.servers = {
            "local_server": ServerConfig(
                name="local_server",
                type="stdio",
                config={
                    "command": ["python", "../server/simple_server.py"]
                }
            )
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_all()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect_all()
    
    async def connect_all(self):
        """Connect to all configured servers."""
        self.exit_stack = AsyncExitStack()
        await self.exit_stack.__aenter__()
        
        # Create semaphore for concurrent request limiting
        self.semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)
        
        # Connect to each enabled server
        for name, server_config in self.servers.items():
            if not server_config.enabled:
                logger.info(f"Skipping disabled server: {name}")
                continue
            
            try:
                client = await self._create_client(server_config)
                if client:
                    self.clients[name] = client
                    logger.info(f"Connected to server: {name}")
            except Exception as e:
                logger.error(f"Failed to connect to {name}: {e}")
    
    async def disconnect_all(self):
        """Disconnect from all servers."""
        if self.exit_stack:
            await self.exit_stack.__aexit__(None, None, None)
        
        self.clients.clear()
        logger.info("Disconnected from all servers")
    
    async def _create_client(self, server_config: ServerConfig):
        """Create and initialize a client for the given server configuration."""
        try:
            if server_config.type == "stdio":
                command = server_config.config.get("command", [])
                client = await self.exit_stack.enter_async_context(
                    stdio_client(*command)
                )
            elif server_config.type == "http":
                from mcp.client.http import http_client
                url = server_config.config.get("url", "")
                headers = server_config.config.get("headers", {})
                
                # Process environment variables in headers
                processed_headers = {}
                for key, value in headers.items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        env_var = value[2:-1]
                        processed_headers[key] = os.getenv(env_var, "")
                    else:
                        processed_headers[key] = value
                
                client = await self.exit_stack.enter_async_context(
                    http_client(url, headers=processed_headers)
                )
            elif server_config.type == "websocket":
                from mcp.client.websocket import websocket_client
                url = server_config.config.get("url", "")
                client = await self.exit_stack.enter_async_context(
                    websocket_client(url)
                )
            else:
                raise ValueError(f"Unsupported server type: {server_config.type}")
            
            # Initialize the client
            await client.initialize()
            return client
            
        except Exception as e:
            logger.error(f"Failed to create client for {server_config.name}: {e}")
            return None
    
    async def _execute_with_retry(self, server_name: str, operation):
        """Execute operation with retry logic."""
        server_config = self.servers.get(server_name)
        if not server_config:
            raise ValueError(f"Unknown server: {server_name}")
        
        last_exception = None
        
        for attempt in range(server_config.retry_attempts):
            try:
                return await asyncio.wait_for(
                    operation(),
                    timeout=server_config.timeout
                )
            except Exception as e:
                last_exception = e
                if attempt < server_config.retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed for {server_name}, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed for {server_name}: {e}")
        
        raise last_exception
    
    async def _cached_operation(self, server_name: str, method: str, operation, *args, **kwargs):
        """Execute operation with caching if enabled."""
        if not self.settings.enable_caching:
            return await operation()
        
        # Check cache first
        cached_result = self.cache.get(server_name, method, args, kwargs)
        if cached_result is not None:
            if self.settings.enable_metrics:
                self.metrics.record_cache_hit()
            logger.debug(f"Cache hit for {server_name}.{method}")
            return cached_result
        
        # Execute operation
        if self.settings.enable_metrics:
            self.metrics.record_cache_miss()
        
        result = await operation()
        
        # Cache the result
        self.cache.set(server_name, method, args, kwargs, result)
        return result
    
    async def call_tool_on_server(self, server_name: str, tool_name: str, arguments: Dict[str, Any] = None) -> Any:
        """Call a tool on a specific server."""
        if server_name not in self.clients:
            raise ValueError(f"Not connected to server: {server_name}")
        
        client = self.clients[server_name]
        start_time = time.time()
        success = False
        
        async with self.semaphore:
            try:
                async def operation():
                    return await client.call_tool(tool_name, arguments or {})
                
                result = await self._execute_with_retry(
                    server_name,
                    lambda: self._cached_operation(
                        server_name, f"call_tool:{tool_name}", operation, arguments
                    )
                )
                
                success = True
                return result
                
            finally:
                if self.settings.enable_metrics:
                    duration = time.time() - start_time
                    self.metrics.record_request(server_name, duration, success)
    
    async def call_tool_on_any_server(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call a tool on the first server that supports it."""
        errors = {}
        
        for server_name, client in self.clients.items():
            try:
                # Check if server has the tool
                tools = await self.list_tools_on_server(server_name)
                if not any(tool.name == tool_name for tool in tools):
                    continue
                
                result = await self.call_tool_on_server(server_name, tool_name, arguments)
                return {
                    "success": True,
                    "result": result,
                    "server": server_name
                }
                
            except Exception as e:
                errors[server_name] = str(e)
                logger.warning(f"Failed to call {tool_name} on {server_name}: {e}")
        
        raise Exception(f"Tool {tool_name} failed on all servers: {errors}")
    
    async def list_tools_on_server(self, server_name: str) -> List[Any]:
        """List tools available on a specific server."""
        if server_name not in self.clients:
            raise ValueError(f"Not connected to server: {server_name}")
        
        client = self.clients[server_name]
        
        async def operation():
            return await client.list_tools()
        
        return await self._execute_with_retry(
            server_name,
            lambda: self._cached_operation(
                server_name, "list_tools", operation
            )
        )
    
    async def list_all_tools(self) -> Dict[str, List[Any]]:
        """List tools from all connected servers."""
        all_tools = {}
        
        for server_name in self.clients:
            try:
                tools = await self.list_tools_on_server(server_name)
                all_tools[server_name] = tools
            except Exception as e:
                logger.error(f"Failed to list tools from {server_name}: {e}")
                all_tools[server_name] = []
        
        return all_tools
    
    async def read_resource_on_server(self, server_name: str, uri: str) -> Any:
        """Read a resource from a specific server."""
        if server_name not in self.clients:
            raise ValueError(f"Not connected to server: {server_name}")
        
        client = self.clients[server_name]
        
        async def operation():
            return await client.read_resource(uri)
        
        return await self._execute_with_retry(
            server_name,
            lambda: self._cached_operation(
                server_name, f"read_resource:{uri}", operation, uri
            )
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all servers."""
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "servers": {}
        }
        
        unhealthy_count = 0
        
        for server_name, client in self.clients.items():
            try:
                # Try to list tools as a health check
                start_time = time.time()
                await self.list_tools_on_server(server_name)
                response_time = time.time() - start_time
                
                health_status["servers"][server_name] = {
                    "status": "healthy",
                    "response_time": response_time
                }
                
            except Exception as e:
                unhealthy_count += 1
                health_status["servers"][server_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Determine overall status
        total_servers = len(self.clients)
        if unhealthy_count == total_servers:
            health_status["overall_status"] = "unhealthy"
        elif unhealthy_count > 0:
            health_status["overall_status"] = "degraded"
        
        if self.settings.enable_metrics:
            health_status["metrics"] = self.metrics.get_stats()
        
        return health_status
    
    async def batch_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple operations concurrently."""
        async def execute_operation(op):
            try:
                if op["type"] == "call_tool":
                    if "server" in op:
                        result = await self.call_tool_on_server(
                            op["server"], op["tool"], op.get("arguments", {})
                        )
                    else:
                        result = await self.call_tool_on_any_server(
                            op["tool"], op.get("arguments", {})
                        )
                    
                    return {"success": True, "result": result, "operation": op}
                
                elif op["type"] == "read_resource":
                    result = await self.read_resource_on_server(
                        op["server"], op["uri"]
                    )
                    return {"success": True, "result": result, "operation": op}
                
                else:
                    raise ValueError(f"Unknown operation type: {op['type']}")
                    
            except Exception as e:
                return {"success": False, "error": str(e), "operation": op}
        
        # Execute all operations concurrently
        tasks = [execute_operation(op) for op in operations]
        return await asyncio.gather(*tasks)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        if not self.settings.enable_metrics:
            return {"metrics_disabled": True}
        
        return self.metrics.get_stats()
    
    def clear_cache(self):
        """Clear the response cache."""
        self.cache.clear()
        logger.info("Response cache cleared")

async def demo_advanced_client():
    """Demonstrate advanced MCP client features."""
    print("=" * 60)
    print("Advanced MCP Client Demo")
    print("=" * 60)
    
    try:
        async with AdvancedMcpClient() as client:
            # 1. Health check
            print("\n1. Performing health check...")
            health = await client.health_check()
            print(json.dumps(health, indent=2))
            
            # 2. List all tools from all servers
            print("\n2. Listing tools from all servers...")
            all_tools = await client.list_all_tools()
            for server_name, tools in all_tools.items():
                print(f"\nServer '{server_name}' tools:")
                for tool in tools:
                    print(f"  - {tool.name}: {tool.description}")
            
            # 3. Test tool calling with failover
            print("\n3. Testing tool calling with automatic server selection...")
            try:
                result = await client.call_tool_on_any_server("calculate_sum", {"a": 10, "b": 20})
                print(f"Result: {result}")
            except Exception as e:
                print(f"Error: {e}")
            
            # 4. Batch operations
            print("\n4. Testing batch operations...")
            operations = [
                {"type": "call_tool", "tool": "get_current_time"},
                {"type": "call_tool", "tool": "reverse_string", "arguments": {"text": "Batch test"}},
                {"type": "call_tool", "tool": "word_count", "arguments": {"text": "This is a batch operation test"}}
            ]
            
            batch_results = await client.batch_operations(operations)
            for i, result in enumerate(batch_results):
                print(f"Operation {i+1}: {'Success' if result['success'] else 'Failed'}")
                if result['success']:
                    print(f"  Result: {result['result']}")
                else:
                    print(f"  Error: {result['error']}")
            
            # 5. Performance metrics
            print("\n5. Performance metrics...")
            metrics = client.get_metrics()
            print(json.dumps(metrics, indent=2))
            
            print("\n" + "=" * 60)
            print("Advanced demo completed!")
            print("=" * 60)
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nDemo failed: {e}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced MCP Client Example")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    
    args = parser.parse_args()
    
    if args.config:
        os.environ["MCP_CONFIG_FILE"] = args.config
    
    try:
        if args.demo:
            asyncio.run(demo_advanced_client())
        else:
            print("Advanced MCP Client")
            print("Use --demo to run demonstration")
            print("Use --config to specify configuration file")
            
    except KeyboardInterrupt:
        print("\nClient stopped by user")
    except Exception as e:
        logger.error(f"Client error: {e}")
        raise

if __name__ == "__main__":
    main()