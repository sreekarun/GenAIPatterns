#!/usr/bin/env python3
"""
Simple MCP Server Example

A basic Model Context Protocol server demonstrating core concepts:
- Tool registration and implementation
- Resource provision
- Prompt templates
- Basic error handling

Usage:
    python simple_server.py

Test with a client:
    python ../client/simple_client.py
"""

from mcp import McpServer
from mcp.types import Tool, Resource, Prompt, TextContent, PromptMessage, McpError
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
server = McpServer("simple-mcp-server")

# Sample data for resources
SAMPLE_CONFIG = {
    "app_name": "Simple MCP Server",
    "version": "1.0.0",
    "features": ["tools", "resources", "prompts"],
    "created": datetime.now().isoformat()
}

SAMPLE_DOCS = {
    "installation": "pip install mcp",
    "usage": "See examples in the documentation",
    "support": "Visit https://modelcontextprotocol.io/"
}


# Tool implementations
@server.tool()
async def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of a and b
    """
    try:
        result = a + b
        logger.info(f"Calculated sum: {a} + {b} = {result}")
        return result
    except Exception as e:
        logger.error(f"Error calculating sum: {e}")
        raise McpError(
            code=-32603,
            message="Failed to calculate sum",
            data={"error": str(e)}
        )


@server.tool()
async def get_current_time() -> str:
    """Get the current date and time.
    
    Returns:
        Current timestamp in ISO format
    """
    try:
        now = datetime.now().isoformat()
        logger.info(f"Current time requested: {now}")
        return now
    except Exception as e:
        logger.error(f"Error getting current time: {e}")
        raise McpError(
            code=-32603,
            message="Failed to get current time",
            data={"error": str(e)}
        )


@server.tool()
async def reverse_string(text: str) -> str:
    """Reverse a string.
    
    Args:
        text: The string to reverse
        
    Returns:
        The reversed string
    """
    if not text:
        raise McpError(
            code=-32602,
            message="Text parameter is required",
            data={"parameter": "text"}
        )
    
    try:
        reversed_text = text[::-1]
        logger.info(f"Reversed string: '{text}' -> '{reversed_text}'")
        return reversed_text
    except Exception as e:
        logger.error(f"Error reversing string: {e}")
        raise McpError(
            code=-32603,
            message="Failed to reverse string",
            data={"error": str(e)}
        )


@server.tool()
async def word_count(text: str) -> dict:
    """Count words, characters, and lines in text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with word, character, and line counts
    """
    if not text:
        return {"words": 0, "characters": 0, "lines": 0}
    
    try:
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.splitlines())
        
        result = {
            "words": word_count,
            "characters": char_count,
            "lines": line_count
        }
        
        logger.info(f"Text analysis: {result}")
        return result
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise McpError(
            code=-32603,
            message="Failed to analyze text",
            data={"error": str(e)}
        )


# Resource implementations
@server.resource("config://app")
async def get_app_config():
    """Provide application configuration."""
    try:
        logger.info("App configuration requested")
        return TextContent(
            type="text",
            text=json.dumps(SAMPLE_CONFIG, indent=2)
        )
    except Exception as e:
        logger.error(f"Error getting app config: {e}")
        raise McpError(
            code=-32603,
            message="Failed to get app configuration",
            data={"error": str(e)}
        )


@server.resource("docs://help")
async def get_documentation():
    """Provide help documentation."""
    try:
        logger.info("Documentation requested")
        return TextContent(
            type="text",
            text=json.dumps(SAMPLE_DOCS, indent=2)
        )
    except Exception as e:
        logger.error(f"Error getting documentation: {e}")
        raise McpError(
            code=-32603,
            message="Failed to get documentation",
            data={"error": str(e)}
        )


@server.list_resources()
async def list_available_resources():
    """List all available resources."""
    try:
        logger.info("Resource list requested")
        return [
            Resource(
                uri="config://app",
                name="Application Configuration",
                description="Current application configuration and settings",
                mimeType="application/json"
            ),
            Resource(
                uri="docs://help",
                name="Help Documentation",
                description="Help and usage documentation",
                mimeType="application/json"
            )
        ]
    except Exception as e:
        logger.error(f"Error listing resources: {e}")
        raise McpError(
            code=-32603,
            message="Failed to list resources",
            data={"error": str(e)}
        )


# Prompt implementations
@server.prompt("greeting")
async def greeting_prompt(name: str = "User", tone: str = "friendly") -> PromptMessage:
    """Generate a greeting prompt.
    
    Args:
        name: Name of the person to greet
        tone: Tone of the greeting (friendly, formal, casual)
    """
    try:
        tone_styles = {
            "friendly": f"Hello there, {name}! Hope you're having a wonderful day!",
            "formal": f"Good day, {name}. I trust you are well.",
            "casual": f"Hey {name}! What's up?"
        }
        
        greeting = tone_styles.get(tone, tone_styles["friendly"])
        
        logger.info(f"Generated greeting for {name} with {tone} tone")
        
        return PromptMessage(
            role="assistant",
            content=greeting
        )
    except Exception as e:
        logger.error(f"Error generating greeting: {e}")
        raise McpError(
            code=-32603,
            message="Failed to generate greeting",
            data={"error": str(e)}
        )


@server.prompt("analyze-text")
async def text_analysis_prompt(task: str = "summarize") -> PromptMessage:
    """Generate a text analysis prompt.
    
    Args:
        task: Type of analysis (summarize, sentiment, keywords)
    """
    try:
        task_prompts = {
            "summarize": "Please provide a concise summary of the following text, highlighting the main points:",
            "sentiment": "Analyze the sentiment of the following text and explain whether it's positive, negative, or neutral:",
            "keywords": "Extract the key terms and important concepts from the following text:"
        }
        
        prompt_text = task_prompts.get(task, task_prompts["summarize"])
        
        logger.info(f"Generated text analysis prompt for task: {task}")
        
        return PromptMessage(
            role="user",
            content=prompt_text
        )
    except Exception as e:
        logger.error(f"Error generating text analysis prompt: {e}")
        raise McpError(
            code=-32603,
            message="Failed to generate text analysis prompt",
            data={"error": str(e)}
        )


@server.list_prompts()
async def list_available_prompts():
    """List all available prompts."""
    try:
        logger.info("Prompt list requested")
        return [
            Prompt(
                name="greeting",
                description="Generate personalized greetings",
                arguments=[
                    {"name": "name", "description": "Name of person to greet", "required": False},
                    {"name": "tone", "description": "Tone of greeting (friendly, formal, casual)", "required": False}
                ]
            ),
            Prompt(
                name="analyze-text",
                description="Generate text analysis prompts",
                arguments=[
                    {"name": "task", "description": "Analysis task (summarize, sentiment, keywords)", "required": False}
                ]
            )
        ]
    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        raise McpError(
            code=-32603,
            message="Failed to list prompts",
            data={"error": str(e)}
        )


def main():
    """Main entry point."""
    logger.info("Starting Simple MCP Server...")
    logger.info("Available tools: calculate_sum, get_current_time, reverse_string, word_count")
    logger.info("Available resources: config://app, docs://help")
    logger.info("Available prompts: greeting, analyze-text")
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        # Run the server with stdio transport
        asyncio.run(server.run_stdio())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()