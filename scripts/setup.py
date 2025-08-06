#!/usr/bin/env python3
"""
GenAI Patterns Setup Script

This script helps set up the development environment for GenAI Patterns,
particularly for MCP (Model Context Protocol) examples.

Usage:
    python scripts/setup.py [--pattern PATTERN] [--dev]
    
Examples:
    python scripts/setup.py --pattern mcp
    python scripts/setup.py --pattern mcp --dev
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

def run_command(command, cwd=None, check=True):
    """Run a shell command."""
    print(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, cwd=cwd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def setup_python_environment():
    """Set up Python virtual environment."""
    print("Setting up Python environment...")
    
    # Check if we're already in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Already in a virtual environment")
        return
    
    # Create virtual environment if it doesn't exist
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", "venv"])
    
    # Determine activation script path
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate.bat"
        pip_path = venv_path / "Scripts" / "pip"
    else:  # Unix/Linux/macOS
        activate_script = venv_path / "bin" / "activate"
        pip_path = venv_path / "bin" / "pip"
    
    print(f"To activate the virtual environment, run:")
    if os.name == 'nt':
        print(f"    {activate_script}")
    else:
        print(f"    source {activate_script}")
    
    return str(pip_path)

def install_dependencies(pattern=None, dev=False):
    """Install required dependencies."""
    print("Installing dependencies...")
    
    # Determine pip command
    pip_cmd = "pip"
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # We're in a virtual environment
        pip_cmd = "pip"
    else:
        # Check if we have a local venv
        venv_path = Path("venv")
        if venv_path.exists():
            if os.name == 'nt':
                pip_cmd = str(venv_path / "Scripts" / "pip")
            else:
                pip_cmd = str(venv_path / "bin" / "pip")
    
    # Install global requirements
    requirements_files = ["requirements.txt"]
    
    # Add pattern-specific requirements
    if pattern:
        pattern_req = Path(f"patterns/{pattern}/requirements.txt")
        if pattern_req.exists():
            requirements_files.append(str(pattern_req))
    
    # Install development dependencies
    if dev:
        dev_requirements = [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.12.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0"
        ]
        
        for package in dev_requirements:
            run_command([pip_cmd, "install", package])
    
    # Install from requirements files
    for req_file in requirements_files:
        if Path(req_file).exists():
            run_command([pip_cmd, "install", "-r", req_file])

def setup_mcp_pattern():
    """Set up MCP pattern specifically."""
    print("Setting up MCP pattern...")
    
    # Create sample configuration files
    config_dir = Path("patterns/mcp/examples/basic_integration")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample client configuration
    client_config = {
        "servers": {
            "local_server": {
                "type": "stdio",
                "command": ["python", "../server/simple_server.py"],
                "enabled": True,
                "retry_attempts": 3,
                "timeout": 30
            }
        },
        "settings": {
            "enable_caching": True,
            "cache_ttl": 300,
            "max_concurrent_requests": 10,
            "enable_metrics": True
        }
    }
    
    client_config_path = config_dir / "client_config.json"
    if not client_config_path.exists():
        with open(client_config_path, 'w') as f:
            json.dump(client_config, f, indent=2)
        print(f"Created sample client configuration: {client_config_path}")
    
    # Create basic integration example config
    integration_config = {
        "server": {
            "name": "basic-integration-server",
            "log_level": "INFO",
            "max_file_size": 1048576,
            "allowed_directories": ["/tmp", "/data"]
        },
        "client": {
            "timeout": 30,
            "retry_attempts": 3,
            "enable_caching": True
        }
    }
    
    integration_config_path = config_dir / "config.json"
    if not integration_config_path.exists():
        with open(integration_config_path, 'w') as f:
            json.dump(integration_config, f, indent=2)
        print(f"Created integration configuration: {integration_config_path}")

def run_tests(pattern=None):
    """Run tests for the specified pattern."""
    print("Running tests...")
    
    test_paths = []
    if pattern:
        pattern_tests = Path(f"patterns/{pattern}/tests")
        if pattern_tests.exists():
            test_paths.append(str(pattern_tests))
    else:
        # Run all tests
        test_paths = ["tests/"] if Path("tests/").exists() else []
    
    if not test_paths:
        print("No tests found")
        return
    
    # Run pytest
    pytest_cmd = ["python", "-m", "pytest", "-v"] + test_paths
    run_command(pytest_cmd, check=False)

def validate_setup(pattern=None):
    """Validate the setup."""
    print("Validating setup...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Warning: Python 3.8+ is recommended")
    else:
        print(f"✓ Python version: {sys.version}")
    
    # Check if in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Virtual environment active")
    else:
        print("! Not in a virtual environment (recommended to use one)")
    
    # Check required packages
    required_packages = ["mcp"]
    if pattern == "mcp":
        required_packages.extend(["fastapi", "uvicorn", "pydantic"])
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not installed")
    
    # Check pattern-specific setup
    if pattern == "mcp":
        mcp_dir = Path("patterns/mcp")
        if mcp_dir.exists():
            print("✓ MCP pattern directory exists")
            
            # Check for key files
            key_files = [
                "patterns/mcp/server/simple_server.py",
                "patterns/mcp/client/simple_client.py",
                "patterns/mcp/requirements.txt"
            ]
            
            for file_path in key_files:
                if Path(file_path).exists():
                    print(f"✓ {file_path} exists")
                else:
                    print(f"✗ {file_path} missing")
        else:
            print("✗ MCP pattern directory not found")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup GenAI Patterns development environment")
    parser.add_argument("--pattern", choices=["mcp"], help="Specific pattern to set up")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--test", action="store_true", help="Run tests after setup")
    parser.add_argument("--validate", action="store_true", help="Validate setup only")
    
    args = parser.parse_args()
    
    print("GenAI Patterns Setup")
    print("=" * 40)
    
    if args.validate:
        validate_setup(args.pattern)
        return
    
    try:
        # Set up Python environment
        pip_path = setup_python_environment()
        
        # Install dependencies
        install_dependencies(args.pattern, args.dev)
        
        # Pattern-specific setup
        if args.pattern == "mcp":
            setup_mcp_pattern()
        
        # Run tests if requested
        if args.test:
            run_tests(args.pattern)
        
        # Validate setup
        validate_setup(args.pattern)
        
        print("\n" + "=" * 40)
        print("Setup completed successfully!")
        print("\nNext steps:")
        if args.pattern == "mcp":
            print("1. cd patterns/mcp")
            print("2. python server/simple_server.py  # In one terminal")
            print("3. python client/simple_client.py  # In another terminal")
        else:
            print("1. Explore the patterns/ directory")
            print("2. Read the documentation in docs/")
            print("3. Try running examples")
        
    except Exception as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()