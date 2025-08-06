#!/usr/bin/env python3
"""
Simple MCP Client Example

A basic Model Context Protocol client demonstrating:
- Connection to MCP servers via stdio
- Tool discovery and calling
- Resource reading
- Basic error handling

Usage:
    # Start the simple server first (in another terminal)
    python ../server/simple_server.py
    
    # Then run this client
    python simple_client.py

For testing with different transports:
    # HTTP (if server supports it)
    python simple_client.py --transport http --url http://localhost:8000/mcp
    
    # WebSocket (if server supports it)
    python simple_client.py --transport websocket --url ws://localhost:8080
"""

import asyncio
import logging
import json
import argparse
from typing import Dict, Any, List

from mcp.client.stdio import stdio_client
from mcp.types import McpError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMcpClient:
    """A simple MCP client wrapper with basic functionality."""
    
    def __init__(self, transport_type: str = "stdio", **kwargs):
        self.transport_type = transport_type
        self.transport_kwargs = kwargs
        self.client = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Connect to the MCP server."""
        try:
            if self.transport_type == "stdio":
                # Default to simple server command
                command = self.transport_kwargs.get("command", ["python", "../server/simple_server.py"])
                self.client = stdio_client(*command)
            elif self.transport_type == "http":
                from mcp.client.http import http_client
                url = self.transport_kwargs.get("url", "http://localhost:8000/mcp")
                self.client = http_client(url)
            elif self.transport_type == "websocket":
                from mcp.client.websocket import websocket_client
                url = self.transport_kwargs.get("url", "ws://localhost:8080")
                self.client = websocket_client(url)
            else:
                raise ValueError(f"Unsupported transport type: {self.transport_type}")
            
            # Initialize the connection
            await self.client.__aenter__()
            await self.client.initialize()
            
            logger.info(f"Connected to MCP server via {self.transport_type}")
            
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.client:
            try:
                await self.client.__aexit__(None, None, None)
                logger.info("Disconnected from MCP server")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
    
    async def discover_capabilities(self) -> Dict[str, Any]:
        """Discover server capabilities."""
        try:
            capabilities = {
                "tools": [],
                "resources": [],
                "prompts": []
            }
            
            # Get tools
            try:
                tools = await self.client.list_tools()
                capabilities["tools"] = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": getattr(tool, 'inputSchema', None)
                    }
                    for tool in tools
                ]
                logger.info(f"Found {len(tools)} tools")
            except Exception as e:
                logger.warning(f"Failed to list tools: {e}")
            
            # Get resources
            try:
                resources = await self.client.list_resources()
                capabilities["resources"] = [
                    {
                        "uri": resource.uri,
                        "name": resource.name,
                        "description": resource.description,
                        "mimeType": resource.mimeType
                    }
                    for resource in resources
                ]
                logger.info(f"Found {len(resources)} resources")
            except Exception as e:
                logger.warning(f"Failed to list resources: {e}")
            
            # Get prompts
            try:
                prompts = await self.client.list_prompts()
                capabilities["prompts"] = [
                    {
                        "name": prompt.name,
                        "description": prompt.description,
                        "arguments": prompt.arguments
                    }
                    for prompt in prompts
                ]
                logger.info(f"Found {len(prompts)} prompts")
            except Exception as e:
                logger.warning(f"Failed to list prompts: {e}")
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Failed to discover capabilities: {e}")
            return {"error": str(e)}
    
    async def call_tool_safe(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call a tool with error handling."""
        try:
            result = await self.client.call_tool(tool_name, arguments or {})
            logger.info(f"Tool '{tool_name}' executed successfully")
            return {"success": True, "result": result}
        except McpError as e:
            logger.error(f"MCP Error calling '{tool_name}': {e.message} (code: {e.code})")
            return {
                "success": False,
                "error": {
                    "type": "mcp_error",
                    "code": e.code,
                    "message": e.message,
                    "data": e.data
                }
            }
        except Exception as e:
            logger.error(f"Unexpected error calling '{tool_name}': {e}")
            return {
                "success": False,
                "error": {
                    "type": "unexpected_error",
                    "message": str(e)
                }
            }
    
    async def read_resource_safe(self, uri: str) -> Dict[str, Any]:
        """Read a resource with error handling."""
        try:
            content = await self.client.read_resource(uri)
            logger.info(f"Resource '{uri}' read successfully")
            return {
                "success": True,
                "content": content.text if hasattr(content, 'text') else str(content)
            }
        except McpError as e:
            logger.error(f"MCP Error reading '{uri}': {e.message} (code: {e.code})")
            return {
                "success": False,
                "error": {
                    "type": "mcp_error",
                    "code": e.code,
                    "message": e.message
                }
            }
        except Exception as e:
            logger.error(f"Unexpected error reading '{uri}': {e}")
            return {
                "success": False,
                "error": {
                    "type": "unexpected_error",
                    "message": str(e)
                }
            }
    
    async def get_prompt_safe(self, prompt_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get a prompt with error handling."""
        try:
            prompt = await self.client.get_prompt(prompt_name, arguments or {})
            logger.info(f"Prompt '{prompt_name}' retrieved successfully")
            return {
                "success": True,
                "prompt": {
                    "role": prompt.role if hasattr(prompt, 'role') else None,
                    "content": prompt.content if hasattr(prompt, 'content') else str(prompt)
                }
            }
        except McpError as e:
            logger.error(f"MCP Error getting prompt '{prompt_name}': {e.message}")
            return {
                "success": False,
                "error": {
                    "type": "mcp_error",
                    "code": e.code,
                    "message": e.message
                }
            }
        except Exception as e:
            logger.error(f"Unexpected error getting prompt '{prompt_name}': {e}")
            return {
                "success": False,
                "error": {
                    "type": "unexpected_error",
                    "message": str(e)
                }
            }

async def demo_basic_operations():
    """Demonstrate basic MCP client operations."""
    print("=" * 60)
    print("MCP Simple Client Demo")
    print("=" * 60)
    
    try:
        async with SimpleMcpClient() as client:
            # 1. Discover server capabilities
            print("\n1. Discovering server capabilities...")
            capabilities = await client.discover_capabilities()
            print(json.dumps(capabilities, indent=2))
            
            # 2. Test some tools if available
            if capabilities.get("tools"):
                print("\n2. Testing available tools...")
                
                # Test calculate_sum if available
                for tool in capabilities["tools"]:
                    if tool["name"] == "calculate_sum":
                        print(f"\nTesting tool: {tool['name']}")
                        result = await client.call_tool_safe("calculate_sum", {"a": 15, "b": 25})
                        print(f"Result: {result}")
                        break
                
                # Test get_current_time if available
                for tool in capabilities["tools"]:
                    if tool["name"] == "get_current_time":
                        print(f"\nTesting tool: {tool['name']}")
                        result = await client.call_tool_safe("get_current_time")
                        print(f"Result: {result}")
                        break
                
                # Test reverse_string if available
                for tool in capabilities["tools"]:
                    if tool["name"] == "reverse_string":
                        print(f"\nTesting tool: {tool['name']}")
                        result = await client.call_tool_safe("reverse_string", {"text": "Hello MCP!"})
                        print(f"Result: {result}")
                        break
                
                # Test word_count if available
                for tool in capabilities["tools"]:
                    if tool["name"] == "word_count":
                        print(f"\nTesting tool: {tool['name']}")
                        result = await client.call_tool_safe("word_count", {
                            "text": "This is a sample text for word counting.\nIt has multiple lines."
                        })
                        print(f"Result: {result}")
                        break
            
            # 3. Test resources if available
            if capabilities.get("resources"):
                print("\n3. Testing available resources...")
                
                for resource in capabilities["resources"]:
                    print(f"\nReading resource: {resource['uri']}")
                    result = await client.read_resource_safe(resource["uri"])
                    if result["success"]:
                        # Truncate long content for display
                        content = result["content"]
                        if len(content) > 200:
                            content = content[:200] + "..."
                        print(f"Content: {content}")
                    else:
                        print(f"Error: {result['error']}")
            
            # 4. Test prompts if available
            if capabilities.get("prompts"):
                print("\n4. Testing available prompts...")
                
                for prompt in capabilities["prompts"]:
                    print(f"\nTesting prompt: {prompt['name']}")
                    
                    # Use appropriate arguments based on prompt
                    if prompt["name"] == "greeting":
                        result = await client.get_prompt_safe("greeting", {
                            "name": "Alice",
                            "tone": "friendly"
                        })
                    elif prompt["name"] == "analyze-text":
                        result = await client.get_prompt_safe("analyze-text", {
                            "task": "summarize"
                        })
                    else:
                        result = await client.get_prompt_safe(prompt["name"])
                    
                    print(f"Result: {result}")
            
            print("\n" + "=" * 60)
            print("Demo completed successfully!")
            print("=" * 60)
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nDemo failed: {e}")

async def test_error_handling():
    """Test error handling scenarios."""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    try:
        async with SimpleMcpClient() as client:
            # Test calling non-existent tool
            print("\n1. Testing non-existent tool...")
            result = await client.call_tool_safe("non_existent_tool", {"param": "value"})
            print(f"Result: {result}")
            
            # Test calling tool with invalid parameters
            print("\n2. Testing invalid parameters...")
            result = await client.call_tool_safe("reverse_string", {})  # Missing required parameter
            print(f"Result: {result}")
            
            # Test reading non-existent resource
            print("\n3. Testing non-existent resource...")
            result = await client.read_resource_safe("invalid://resource")
            print(f"Result: {result}")
            
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple MCP Client Example")
    parser.add_argument("--transport", choices=["stdio", "http", "websocket"], 
                       default="stdio", help="Transport type")
    parser.add_argument("--url", help="Server URL for HTTP/WebSocket transport")
    parser.add_argument("--command", nargs="+", 
                       default=["python", "../server/simple_server.py"],
                       help="Command to start stdio server")
    parser.add_argument("--test-errors", action="store_true",
                       help="Run error handling tests")
    
    args = parser.parse_args()
    
    # Prepare transport kwargs
    transport_kwargs = {}
    if args.transport == "stdio":
        transport_kwargs["command"] = args.command
    elif args.url:
        transport_kwargs["url"] = args.url
    
    async def run_demo():
        """Run the demo."""
        global SimpleMcpClient
        
        # Update SimpleMcpClient with transport args
        original_init = SimpleMcpClient.__init__
        
        def new_init(self, transport_type=args.transport, **kwargs):
            kwargs.update(transport_kwargs)
            original_init(self, transport_type, **kwargs)
        
        SimpleMcpClient.__init__ = new_init
        
        # Run basic demo
        await demo_basic_operations()
        
        # Run error handling tests if requested
        if args.test_errors:
            await test_error_handling()
    
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\nClient stopped by user")
    except Exception as e:
        logger.error(f"Client error: {e}")
        raise

if __name__ == "__main__":
    main()