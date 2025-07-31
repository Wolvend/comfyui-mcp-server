# ComfyUI MCP Server

A powerful Python-based MCP (Model Context Protocol) server that interfaces with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) to enable AI agents like Claude to generate images through natural language commands.

## 🚀 What's New (v1.2.0)

### Major Protocol Fix
- 🔧 **Fixed MCP Connection Issues**: Migrated from WebSocket to proper MCP stdio protocol
- ✅ **Full Claude Desktop Compatibility**: Now works seamlessly with Claude Desktop
- 🚨 **Resolved "Failed to reconnect" errors**

### New Tools Added (11 new, 16 total)
- 📸 **Image Management**: `get_recent_images`, `get_image_metadata`, `cleanup_old_images`
- 🎨 **Advanced Generation**: `batch_generate`, `generate_variations`
- 📊 **System Monitoring**: `get_system_stats`, `get_queue_status`, `clear_comfyui_cache`
- 🔍 **Discovery**: `list_models`, `get_node_info`, `validate_workflow`

### Enhancements
- ✅ Enhanced error handling and validation
- ✅ Extended parameter support for fine-tuned control
- ✅ Proper dependency management with `requirements.txt`
- ✅ Multi-user Git configuration support

## Overview

This MCP server provides a comprehensive interface for AI assistants to:
- 🎨 **Generate images** using ComfyUI workflows with extensive customization
- 📊 **Monitor system** resources, queue status, and server health
- 🗂️ **Manage outputs** including metadata extraction and cleanup
- 🔍 **Discover capabilities** by listing models, nodes, and workflows
- ⚡ **Batch operations** for efficient multi-image generation
- 🎯 **Validate workflows** before execution

## Features

### 🎨 Image Generation
- **Flexible prompts**: Any text description
- **Size control**: Width and height (must be divisible by 8)
- **Workflow selection**: Use different ComfyUI workflows
- **Model selection**: Choose from available checkpoints
- **Advanced parameters**:
  - Seed control for reproducibility
  - Sampling steps (1-150)
  - CFG scale (0-30)
  - Sampler selection (euler, dpm_2, ddim, etc.)
  - Scheduler types (normal, karras, exponential)
  - Denoising strength
  - Negative prompts

### 🛠️ Available MCP Tools (16 Total)

#### Core Generation
1. **`generate_image`** - Generate images with extensive customization options
2. **`batch_generate`** - Generate multiple images with different prompts efficiently
3. **`generate_variations`** - Create variations by modifying prompts

#### System & Status
4. **`get_server_info`** - Get server version, URL, and available models
5. **`health_check`** - Verify ComfyUI connectivity and response time
6. **`get_system_stats`** - Monitor GPU/CPU/memory usage and performance
7. **`get_queue_status`** - Check running and pending generation jobs
8. **`clear_comfyui_cache`** - Free up VRAM by clearing model cache

#### Discovery & Validation
9. **`list_workflows`** - List all available workflow JSON files
10. **`list_models`** - List models by type (checkpoints, LoRAs, VAE, etc.)
11. **`get_node_info`** - Explore available ComfyUI nodes and their parameters
12. **`validate_workflow`** - Validate workflow compatibility before use

#### Output Management
13. **`get_recent_images`** - Retrieve recently generated images with metadata
14. **`get_image_metadata`** - Extract generation parameters from images
15. **`cleanup_old_images`** - Remove old images (with dry-run safety option)
16. **`get_generation_status`** - Check generation progress (placeholder)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Wolvend/comfyui-mcp-server.git
cd comfyui-mcp-server
python -m venv mcp_venv
source mcp_venv/bin/activate
pip install -r requirements.txt

# Configure Claude Desktop (see Installation section)
# Start ComfyUI, then ask Claude to generate images!
```

## Prerequisites

- **Python 3.10+**
- **ComfyUI**: Installed and running (default: `localhost:8188`)
- **Virtual environment** (recommended)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Wolvend/comfyui-mcp-server.git
cd comfyui-mcp-server
```

### 2. Create Virtual Environment
```bash
python -m venv mcp_venv
source mcp_venv/bin/activate  # On Windows: mcp_venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install requests mcp Pillow
```

### 4. Configure MCP for Claude

Add to your Claude Desktop configuration file:

**Location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Configuration:**
```json
{
  "mcpServers": {
    "comfyui": {
      "command": "python",
      "args": ["/path/to/comfyui-mcp-server/server.py"],
      "env": {
        "COMFYUI_URL": "http://localhost:8188"
      }
    }
  }
}
```

### 5. Prepare ComfyUI

1. **Install ComfyUI** ([installation guide](https://github.com/comfyanonymous/ComfyUI))
2. **Download models** and place in `ComfyUI/models/checkpoints/`
3. **Start ComfyUI**:
   ```bash
   cd ComfyUI
   python main.py --listen 0.0.0.0 --port 8188
   ```

### 6. Add Workflows

Place workflow JSON files in the `workflows/` directory:
1. Create workflow in ComfyUI web interface
2. Enable Dev Mode in ComfyUI settings
3. Export with "Save (API Format)"
4. Save to `workflows/` directory

## Usage

### With Claude Desktop

Once configured, you can ask Claude to:

**Basic Generation:**
- "Generate an image of a sunset over mountains"
- "Create a cyberpunk city scene at 1024x768"
- "Make a portrait with seed 12345 for consistency"

**Batch & Variations:**
- "Generate 5 different fantasy landscapes"
- "Create variations of 'a cat' with different styles: realistic, cartoon, oil painting"

**System Management:**
- "Check if ComfyUI is running"
- "Show me the GPU memory usage"
- "List all available Stable Diffusion models"
- "Clean up images older than 7 days (dry run first)"

**Discovery:**
- "What workflows are available?"
- "Show me recent images generated"
- "Get metadata from the last generated image"
- "List all LoRA models"

### Standalone Testing

Run the server directly to test:
```bash
python server.py
```

The server communicates via stdio (standard input/output) using the MCP protocol.

## Configuration

### Environment Variables

- `COMFYUI_URL`: ComfyUI server URL (default: `http://localhost:8188`)
- `DEBUG`: Enable debug logging (set to any value)

### Example with Custom URL
```bash
COMFYUI_URL=http://192.168.1.100:8188 python server.py
```

## Project Structure

```
comfyui-mcp-server/
├── server.py              # MCP server implementation
├── comfyui_client.py      # ComfyUI API interface
├── workflows/             # Workflow JSON files
│   └── basic_api_test.json
├── MCP_COMPATIBILITY.md   # Detailed compatibility guide
├── __init__.py           # Python package file
└── README.md             # This file
```

## Troubleshooting

### Common Issues

1. **"Failed to reconnect to comfyui" (FIXED in v1.2.0)**
   - This was due to WebSocket protocol - now uses proper MCP stdio

2. **"No models available"**
   - Download model files (e.g., SD 1.5) to `ComfyUI/models/checkpoints/`
   - Use `list_models` tool to see what's available

3. **"Failed to connect to ComfyUI"**
   - Ensure ComfyUI is running: `python main.py --listen 0.0.0.0`
   - Check firewall settings
   - Verify COMFYUI_URL environment variable

4. **"Width/height must be divisible by 8"**
   - Use dimensions like 512, 768, 1024, etc.

5. **Memory Issues**
   - Use `clear_comfyui_cache` to free up VRAM
   - Check usage with `get_system_stats`

### Debug Mode

Enable detailed logging:
```bash
DEBUG=1 python server.py
```

## Advanced Usage

### Custom Workflows

1. Create workflow in ComfyUI
2. Note the node IDs for inputs you want to control
3. Modify `DEFAULT_MAPPING` in `comfyui_client.py` if needed

### Batch Scripts

Use the included helper scripts:
- `start_all.sh` - Start both ComfyUI and MCP server
- `start_comfyui.sh` - Start only ComfyUI

## API Reference

### Core Generation Tools

#### generate_image
```python
generate_image(
    prompt: str,                    # Required
    width: int = 512,              # Optional
    height: int = 512,             # Optional
    workflow_id: str = "basic_api_test",  # Optional
    model: str = None,             # Optional
    seed: int = None,              # Optional (-1 for random)
    steps: int = None,             # Optional (1-150)
    cfg_scale: float = None,       # Optional (0-30)
    sampler_name: str = None,      # Optional (euler, dpm_2, ddim, etc.)
    scheduler: str = None,         # Optional (normal, karras, exponential)
    denoise: float = None,         # Optional (0.0-1.0)
    negative_prompt: str = None    # Optional
)
```

#### batch_generate
```python
batch_generate(
    prompts: List[str],            # Required (max 10)
    width: int = 512,              # Optional
    height: int = 512,             # Optional
    seed_increment: bool = True,   # Optional
    base_seed: int = None,         # Optional
    **kwargs                       # Other generate_image parameters
)
```

#### generate_variations
```python
generate_variations(
    base_prompt: str,              # Required
    variations: List[str],         # Required (max 8)
    width: int = 512,              # Optional
    height: int = 512,             # Optional
    base_seed: int = None,         # Optional (fixed for comparison)
    **kwargs                       # Other parameters
)
```

### System Tools

#### get_system_stats
Returns GPU/CPU usage, VRAM info, and system performance metrics.

#### cleanup_old_images
```python
cleanup_old_images(
    days_old: int = 7,             # Images older than N days
    dry_run: bool = True           # Preview without deleting
)
```

## Contributing

Contributions are welcome! 

### Recently Completed ✅
- Fixed MCP connection issues (WebSocket → stdio)
- Added batch generation support
- Implemented 11 new tools for comprehensive functionality
- Added system monitoring and management tools

### Future Improvements
- Real-time progress tracking implementation
- Image-to-image workflows
- ControlNet support
- Animation/video generation
- Custom node integration
- Workflow templates library

## Changelog

### v1.2.0 (Latest)
- 🔧 Fixed critical MCP connection issues by migrating to stdio protocol
- ➕ Added 11 new tools (16 total) for comprehensive ComfyUI control
- 📦 Added proper dependency management with requirements.txt
- 📝 Enhanced documentation and examples
- 🐛 Fixed "Failed to reconnect" errors

### v1.0.0
- Initial release with WebSocket implementation
- Basic image generation support

## License

Apache License 2.0 - See LICENSE file for details

## Acknowledgments

- Original WebSocket implementation by [joenorton](https://github.com/joenorton/comfyui-mcp-server)
- Built for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- Uses [Anthropic's MCP SDK](https://github.com/anthropics/mcp)
- Special thanks to the ComfyUI community

---

For more detailed compatibility information, see [MCP_COMPATIBILITY.md](MCP_COMPATIBILITY.md)