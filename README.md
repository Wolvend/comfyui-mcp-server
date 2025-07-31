# ComfyUI MCP Server

A comprehensive Python-based MCP (Model Context Protocol) server that interfaces with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) to enable AI agents like Claude to perform advanced image and video generation, manipulation, and analysis through natural language commands.

## 🚀 What's New (v2.0.0) - MAJOR UPDATE!

### 🎬 Video Generation & Animation
- **Text-to-Video**: Generate videos from text descriptions
- **Image Animation**: Bring static images to life
- **Frame Interpolation**: Smooth transitions between images

### 🎨 Advanced Image Control
- **ControlNet Support**: Guided generation with pose, depth, edges
- **Inpainting/Outpainting**: Smart fill and canvas extension
- **Style Transfer**: Apply artistic styles between images

### 🔧 Professional Enhancement
- **AI Upscaling**: 2x/4x with ESRGAN, Real-ESRGAN
- **Face Restoration**: GFPGAN/CodeFormer integration
- **Background Removal**: Automatic subject isolation
- **Color Grading**: Professional color correction

### 🎯 Creative Tools
- **Multi-Image Blending**: Advanced compositing
- **LoRA Styling**: Apply multiple style models
- **Region Control**: Different prompts for different areas

### 📊 Smart Analysis
- **Prompt Enhancement**: AI-powered prompt optimization
- **Object Detection**: Identify elements in images
- **Quality Comparison**: Technical image analysis
- **Time Estimation**: Predict generation duration

### 🔄 Workflow Automation
- **Animation Sequences**: Multi-prompt animations
- **Batch Styling**: Consistent style across images
- **Progressive Enhancement**: Multi-stage upscaling
- **Template System**: Pre-configured workflows

### ⚡ Real-time Features
- **WebSocket Progress**: Live generation updates with queue status
- **Preview Streaming**: Low-res previews during generation
- **Queue Management**: Real-time queue position and wait time estimates
- **Cancellation**: Stop in-progress tasks immediately

### Previous Updates (v1.2.0)
- ✅ Fixed MCP connection issues (WebSocket → stdio)
- ✅ Full Claude Desktop compatibility
- ✅ Enhanced error handling and validation
- ✅ 16 core tools for generation and management

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

### 🛠️ Available MCP Tools (42 Total)

#### Core Generation (6 tools)
1. **`generate_image`** - Generate images with extensive customization options
2. **`batch_generate`** - Generate multiple images with different prompts efficiently
3. **`generate_variations`** - Create variations by modifying prompts
4. **`generate_video`** - Create videos from text descriptions
5. **`image_to_video`** - Animate static images into videos
6. **`video_interpolation`** - Create smooth transitions between images

#### Advanced Image Control (4 tools)
7. **`controlnet_generate`** - Use ControlNet for pose/depth/edge guidance
8. **`inpaint_image`** - Intelligently fill masked areas
9. **`outpaint_image`** - Extend images beyond original borders
10. **`style_transfer`** - Apply artistic styles between images

#### Image Enhancement (4 tools)
11. **`upscale_image`** - AI-powered 2x/4x upscaling
12. **`face_restore`** - Restore and enhance faces
13. **`remove_background`** - Automatic background removal
14. **`color_correction`** - Professional color grading

#### Creative Tools (4 tools)
15. **`blend_images`** - Blend multiple images with various modes
16. **`apply_lora_styles`** - Apply multiple LoRA style models
17. **`mask_guided_generation`** - Different content in different regions
18. **`create_variations`** - (duplicate of #3)

#### Analysis & Optimization (4 tools)
19. **`analyze_prompt`** - AI-powered prompt improvement
20. **`detect_objects`** - Identify objects in images
21. **`compare_images`** - Technical quality comparison
22. **`estimate_generation_time`** - Predict processing duration

#### Workflow Automation (4 tools)
23. **`create_animation_sequence`** - Multi-prompt animation sequences
24. **`batch_style_apply`** - Consistent style across images
25. **`progressive_upscale`** - Multi-stage quality enhancement
26. **`template_workflows`** - Pre-configured workflow templates

#### Real-time Features (4 tools)
27. **`websocket_progress`** - Live generation progress updates
28. **`preview_stream`** - Low-resolution preview streaming
29. **`queue_priority`** - Manage generation priority
30. **`cancel_generation`** - Stop in-progress tasks

#### System & Status (5 tools)
31. **`get_server_info`** - Get server version, URL, and available models
32. **`health_check`** - Verify ComfyUI connectivity and response time
33. **`get_system_stats`** - Monitor GPU/CPU/memory usage and performance
34. **`get_queue_status`** - Check running and pending generation jobs
35. **`clear_comfyui_cache`** - Free up VRAM by clearing model cache

#### Discovery & Validation (4 tools)
36. **`list_workflows`** - List all available workflow JSON files
37. **`list_models`** - List models by type (checkpoints, LoRAs, VAE, etc.)
38. **`get_node_info`** - Explore available ComfyUI nodes and their parameters
39. **`validate_workflow`** - Validate workflow compatibility before use

#### Output Management (4 tools)
40. **`get_recent_images`** - Retrieve recently generated images with metadata
41. **`get_image_metadata`** - Extract generation parameters from images
42. **`cleanup_old_images`** - Remove old images (with dry-run safety option)
43. **`get_generation_status`** - Check generation progress (placeholder)

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

**Video Generation:**
- "Generate a video of waves crashing on a beach"
- "Animate this image with a zoom-in effect"
- "Create a smooth transition between these two images"

**Advanced Control:**
- "Generate an image using this pose reference with ControlNet"
- "Remove the person from this photo and fill with background"
- "Extend this image 256 pixels to the right"
- "Apply the style of Van Gogh to this photo"

**Enhancement:**
- "Upscale this image 4x with face enhancement"
- "Remove the background from this product photo"
- "Fix the faces in this old photograph"
- "Auto-correct the colors and white balance"

**Creative Tools:**
- "Blend these three images with soft light mode"
- "Apply anime and cyberpunk LoRA styles to this prompt"
- "Generate a cat in the red area and a dog in the blue area"

**Smart Features:**
- "Improve this prompt for better results"
- "What objects are in this image?"
- "Compare the quality of these two images"
- "How long will it take to generate a 4K image with 50 steps?"

**Automation:**
- "Create an animation morphing through: sunrise, noon, sunset, night"
- "Apply this oil painting style to all images in the batch"
- "Progressively upscale this image to 8x resolution"
- "Use the portrait template with my custom settings"

**Real-time Control:**
- "Show me live progress of the current generation"
- "Stream previews every 5 steps"
- "Set this job to high priority"
- "Cancel the current generation"

**System Management:**
- "Check if ComfyUI is running"
- "Show me the GPU memory usage"
- "List all available Stable Diffusion models"
- "Clean up images older than 7 days (dry run first)"

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
- ✅ Fixed MCP connection issues (WebSocket → stdio)
- ✅ Added 26 new advanced tools (42 total)
- ✅ Implemented real ComfyUI node integrations
- ✅ Added comprehensive workflow templates
- ✅ Enhanced model discovery and validation
- ✅ Real-time WebSocket progress tracking
- ✅ Queue management with position tracking
- ✅ ControlNet, upscaling, and inpainting support

### Implementation Status
- **Core Generation**: ✅ Fully implemented
- **Advanced Control**: ✅ ControlNet, inpainting, style transfer working
- **Enhancement**: ✅ Upscaling workflows ready
- **Creative Tools**: ⚠️ Basic implementations (need custom nodes)
- **Analysis**: ⚠️ Placeholder implementations
- **Automation**: ⚠️ Template-based (needs complex workflows)
- **Real-time**: ✅ WebSocket and queue management working

## Changelog

### v2.0.0 (Latest) - MAJOR UPDATE
- 🎬 Added video generation framework (text-to-video, image animation, interpolation)
- 🎨 **IMPLEMENTED** advanced image control (ControlNet, inpainting, outpainting, style transfer)
- 🔧 **IMPLEMENTED** professional enhancement tools (upscaling workflows, face restoration)
- 🎯 Added creative tools framework (blending, LoRA styling, region control)
- 📊 Added smart analysis features (prompt enhancement, object detection, comparison)
- 🔄 Created workflow automation system (animation sequences, batch processing, templates)
- ⚡ **IMPLEMENTED** real-time features (WebSocket progress, queue management, cancellation)
- 📈 Expanded from 16 to 42 total tools
- 🔧 **FIXED** model discovery and parameter mapping
- 🛠️ **ENHANCED** workflow template system with 9 specialized workflows

### v1.2.0
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