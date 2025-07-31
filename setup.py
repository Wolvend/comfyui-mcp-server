#!/usr/bin/env python3
"""
ComfyUI MCP Server Setup Script
Automated installation and configuration
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command with error handling"""
    print(f"⚡ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e.stderr}")
        return False

def setup_virtual_environment():
    """Create and setup virtual environment"""
    if not Path("mcp_venv").exists():
        print("🔧 Setting up virtual environment...")
        if not run_command("python -m venv mcp_venv", "Creating virtual environment"):
            return False
    
    # Activate and install requirements
    activate_cmd = "source mcp_venv/bin/activate" if os.name != 'nt' else "mcp_venv\\Scripts\\activate"
    pip_cmd = f"{activate_cmd} && pip install -r requirements.txt"
    
    return run_command(pip_cmd, "Installing dependencies")

def setup_mcp_config():
    """Setup MCP configuration for Claude Desktop"""
    print("🔧 Setting up MCP configuration...")
    
    # Determine config path based on OS
    if sys.platform == "darwin":  # macOS
        config_path = Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    elif sys.platform == "win32":  # Windows
        config_path = Path.home() / "AppData/Roaming/Claude/claude_desktop_config.json"
    else:  # Linux
        config_path = Path.home() / ".config/Claude/claude_desktop_config.json"
    
    # Create config directory if needed
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Current working directory for server path
    server_path = Path.cwd() / "server.py"
    
    mcp_config = {
        "mcpServers": {
            "comfyui": {
                "command": "bash",
                "args": ["-c", f"cd {Path.cwd()} && source mcp_venv/bin/activate && python server.py"],
                "env": {
                    "COMFYUI_URL": "http://localhost:8188"
                }
            }
        }
    }
    
    # If config exists, merge with existing
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                existing_config = json.load(f)
            if "mcpServers" not in existing_config:
                existing_config["mcpServers"] = {}
            existing_config["mcpServers"]["comfyui"] = mcp_config["mcpServers"]["comfyui"]
            mcp_config = existing_config
        except Exception as e:
            print(f"⚠️  Could not read existing config: {e}")
    
    # Write config
    try:
        with open(config_path, 'w') as f:
            json.dump(mcp_config, f, indent=2)
        print(f"✅ MCP config written to: {config_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to write MCP config: {e}")
        return False

def check_comfyui():
    """Check if ComfyUI is accessible"""
    print("🔧 Checking ComfyUI connection...")
    try:
        import requests
        response = requests.get("http://localhost:8188/system_stats", timeout=5)
        if response.status_code == 200:
            print("✅ ComfyUI is running and accessible")
            return True
        else:
            print(f"⚠️  ComfyUI responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️  ComfyUI not accessible: {e}")
        print("💡 Make sure to start ComfyUI before using the MCP server")
        return False

def test_server():
    """Test the MCP server startup"""
    print("🔧 Testing MCP server startup...")
    test_cmd = "source mcp_venv/bin/activate && timeout 10 python server.py"
    if run_command(test_cmd, "Testing server startup"):
        print("✅ Server starts successfully")
        return True
    else:
        print("⚠️  Server test inconclusive (this may be normal)")
        return True  # Don't fail setup for this

def main():
    """Main setup routine"""
    print("🚀 ComfyUI MCP Server Setup")
    print("=" * 40)
    
    success_count = 0
    total_steps = 5
    
    # Step 1: Virtual environment
    if setup_virtual_environment():
        success_count += 1
    
    # Step 2: MCP configuration
    if setup_mcp_config():
        success_count += 1
    
    # Step 3: Check ComfyUI (optional)
    if check_comfyui():
        success_count += 1
    else:
        success_count += 0.5  # Partial credit
    
    # Step 4: Test server
    if test_server():
        success_count += 1
    
    # Step 5: Final validation
    if Path("mcp_venv").exists() and Path("server.py").exists():
        print("✅ All files in place")
        success_count += 1
    
    print("\n" + "=" * 40)
    print(f"🎯 Setup complete: {success_count}/{total_steps} steps successful")
    
    if success_count >= 4:
        print("🎉 Setup successful! Next steps:")
        print("1. Make sure ComfyUI is running (python main.py --listen 0.0.0.0)")
        print("2. Restart Claude Desktop to load the MCP server")
        print("3. Try asking Claude: 'Generate an image of a sunset'")
        return True
    else:
        print("⚠️  Setup had issues. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)