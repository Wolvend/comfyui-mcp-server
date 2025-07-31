# ComfyUI MCP Server

A powerful Python-based MCP (Model Context Protocol) server that interfaces with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) to enable AI agents like Claude to generate images through natural language commands.

## 🚀 What's New

This server has been upgraded to use the **standard MCP stdio protocol** instead of WebSocket, providing:
- ✅ Full compatibility with Claude Desktop and other MCP clients
- ✅ Enhanced error handling and validation
- ✅ Extended parameter support for fine-tuned control
- ✅ Multiple utility tools beyond image generation
- ✅ Environment-based configuration

## Overview

This MCP server allows AI assistants to:
- Generate images using ComfyUI workflows
- Query server status and available models
- List available workflows
- Check ComfyUI health status
- Support extensive customization options

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

### 🛠️ Available MCP Tools

1. **`generate_image`** - Generate images with extensive customization
2. **`get_server_info`** - Get server version and available models
3. **`list_workflows`** - List available workflow files
4. **`health_check`** - Check ComfyUI connectivity
5. **`get_generation_status`** - Check generation progress (placeholder)

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
pip install requests mcp
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
- "Generate an image of a sunset over mountains"
- "Create a cyberpunk city scene at 1024x768"
- "Show me what workflows are available"
- "Check if ComfyUI is running"

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

1. **"No models available"**
   - Download model files (e.g., SD 1.5) to `ComfyUI/models/checkpoints/`

2. **"Failed to connect to ComfyUI"**
   - Ensure ComfyUI is running on the configured port
   - Check firewall settings
   - Verify COMFYUI_URL environment variable

3. **"Width/height must be divisible by 8"**
   - Use dimensions like 512, 768, 1024, etc.

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

### generate_image

```python
generate_image(
    prompt: str,                    # Required
    width: int = 512,              # Optional
    height: int = 512,             # Optional
    workflow_id: str = "basic_api_test",  # Optional
    model: str = None,             # Optional
    seed: int = None,              # Optional
    steps: int = None,             # Optional
    cfg_scale: float = None,       # Optional
    sampler_name: str = None,      # Optional
    scheduler: str = None,         # Optional
    denoise: float = None,         # Optional
    negative_prompt: str = None    # Optional
)
```

## Contributing

Contributions are welcome! Areas for improvement:
- Progress tracking implementation
- Batch generation support
- Image-to-image workflows
- ControlNet support
- Animation/video generation

## License

Apache License 2.0 - See LICENSE file for details

## Acknowledgments

- Original WebSocket implementation by [joenorton](https://github.com/joenorton/comfyui-mcp-server)
- Built for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- Uses [Anthropic's MCP SDK](https://github.com/anthropics/mcp)

---

For more detailed compatibility information, see [MCP_COMPATIBILITY.md](MCP_COMPATIBILITY.md)