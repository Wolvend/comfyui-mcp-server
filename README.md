# ComfyUI MCP Server

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/tools-89-orange.svg" alt="Tools">
</p>

<p align="center">
  <strong>A comprehensive creative automation platform that bridges AI agents with ComfyUI's powerful generation capabilities.</strong>
</p>

---

## 🌟 Overview

ComfyUI MCP Server transforms ComfyUI into an intelligent creative powerhouse accessible through the Model Context Protocol (MCP). With 89 specialized tools, it enables AI agents like Claude to perform complex image and video generation tasks through natural language commands.

### ✨ Key Highlights

- **🎨 89 Specialized Tools** - Comprehensive suite covering every aspect of creative generation
- **🧠 AI-Powered Intelligence** - Learn from successes and optimize automatically
- **⚡ Enterprise-Ready** - Production features like batch processing and error recovery
- **🔄 Workflow Automation** - Save, version, and share complete workflows
- **📊 Real-Time Monitoring** - Live previews and progress tracking
- **🎯 Advanced Control** - ControlNet, inpainting, masking, and more

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed and running
- Required Python packages (see [requirements.txt](requirements.txt))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/comfyui-mcp-server.git
cd comfyui-mcp-server

# Install dependencies
pip install -r requirements.txt

# Start ComfyUI (in a separate terminal)
cd /path/to/ComfyUI
python main.py

# Start the MCP server
python server.py
```

### Basic Usage

```python
# Generate an image
mcp call generate_image '{
  "prompt": "a majestic mountain landscape at sunset, photorealistic",
  "width": 1024,
  "height": 768,
  "steps": 30
}'

# Optimize a prompt with AI
mcp call optimize_prompt_with_ai '{
  "prompt": "cat on sofa",
  "optimization_goals": ["quality", "detail", "style"]
}'

# Apply a workflow preset
mcp call apply_workflow_preset '{
  "preset_name": "Professional Photography",
  "parameters": {"prompt": "luxury product shot"}
}'
```

## 🛠️ Features

### Core Capabilities

<details>
<summary><strong>🎨 Image Generation Suite</strong> (11 tools)</summary>

- `generate_image` - Advanced text-to-image with full parameter control
- `batch_generate` - Generate multiple images efficiently
- `generate_variations` - Create variations of prompts
- `generate_with_style_preset` - Use predefined artistic styles
- `optimize_prompt_with_ai` - AI-powered prompt enhancement
- Workflow preset management (save, list, apply)
- And more...

</details>

<details>
<summary><strong>🎬 Video Generation</strong> (3 tools)</summary>

- `generate_video` - Create videos from text descriptions
- `image_to_video` - Animate static images
- `video_interpolation` - Smooth transitions between frames

</details>

<details>
<summary><strong>🎯 Advanced Control</strong> (5 tools)</summary>

- `controlnet_generate` - Guided generation with pose/depth/edge
- `inpaint_image` - Intelligent area replacement
- `outpaint_image` - Extend beyond canvas
- `style_transfer` - Apply artistic styles
- `auto_generate_mask` - Natural language masking

</details>

<details>
<summary><strong>🔧 Enhancement Tools</strong> (4 tools)</summary>

- `upscale_image` - AI-powered resolution enhancement
- `face_restore` - Facial feature restoration
- `remove_background` - Automatic subject isolation
- `color_correction` - Professional color grading

</details>

<details>
<summary><strong>📊 Analytics & Optimization</strong> (6 tools)</summary>

- `analyze_prompt` - Prompt effectiveness analysis
- `detect_objects` - Image content detection
- `compare_images` - Quality comparison
- `estimate_generation_time` - Resource prediction
- Performance tracking and learning

</details>

### Advanced Features

#### 🔄 Workflow Automation

Save and reuse complete generation workflows:

```python
# Save a workflow
save_workflow_preset(
    name="Product Photography",
    workflow={...},
    tags=["product", "ecommerce"]
)

# Apply with modifications
apply_workflow_preset(
    "Product Photography",
    parameters={"lighting": "dramatic"}
)
```

#### 🧠 AI Learning System

The server learns from every generation:

- Tracks success patterns
- Optimizes prompts automatically
- Suggests improvements
- A/B testing framework

#### 🏭 Batch Processing

Handle complex multi-step operations:

```python
coordinate_batch_operation(
    "Marketing Campaign",
    tasks=[
        {"type": "generate", "prompt": "hero image"},
        {"type": "upscale", "scale": 2},
        {"type": "remove_background"}
    ],
    parallel_limit=3
)
```

## 📚 Documentation

### Start Here

- **[Complete Tutorial](TUTORIAL.md)** - Comprehensive guide covering everything you need
- [Feature Catalog](FEATURES.md) - Detailed list of all 89 tools
- [Changelog](CHANGELOG.md) - Release notes and version history

### Tool Categories

| Category | Tools | Purpose |
|----------|-------|---------|
| **Generation** | 11 | Core image/video creation |
| **Control** | 5 | Guided generation and masking |
| **Enhancement** | 4 | Quality improvement |
| **Creative** | 4 | Artistic effects |
| **Analysis** | 6 | Content understanding |
| **Workflow** | 7 | Automation and batching |
| **Real-time** | 5 | Live monitoring |
| **System** | 5 | Health and status |
| **Discovery** | 4 | Model and node info |
| **Output** | 8 | File management |
| **LoRA** | 8 | Model management |
| **Video** | 12 | Animation and video |

## 🎯 Use Cases

### Professional Photography
```python
# Complete product shoot workflow
workflow = create_product_photography_workflow(
    style="minimalist",
    background="white",
    lighting="studio"
)
batch_generate(products, workflow)
```

### Social Media Content
```python
# Generate optimized content for platforms
for platform in ["instagram", "twitter", "youtube"]:
    optimize_for_platform(image, platform)
```

### Creative Exploration
```python
# A/B test different styles
test_results = ab_test_prompts(
    "landscape photo",
    variations=["moody", "bright", "dramatic"]
)
```

## 🔌 Integration

### Claude Desktop

Add to your Claude configuration:

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

### API Usage

```python
from comfyui_mcp_client import ComfyUIMCPClient

client = ComfyUIMCPClient()
result = client.generate_image(
    prompt="beautiful sunset",
    style_preset="cinematic"
)
```

## 🏗️ Architecture

The server follows a modular architecture:

```
┌─────────────────┐     ┌─────────────────┐
│   AI Agents     │────▶│   MCP Server    │
│  (Claude, etc)  │     │   (89 tools)    │
└─────────────────┘     └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  ComfyUI API    │
                        │  & WebSocket    │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  Enhancement    │
                        │    Layers       │
                        └─────────────────┘
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone <repo>
cd comfyui-mcp-server
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
python -m flake8 .
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- Powered by [Anthropic's MCP SDK](https://github.com/anthropics/mcp)
- FastMCP framework for efficient tool management
- The amazing ComfyUI community

## 📊 Project Stats

- **Version**: 1.0.0
- **Tools**: 89
- **Lines of Code**: ~7,000
- **Test Coverage**: Comprehensive
- **Documentation**: Extensive

## 🔗 Quick Links

- [Complete Tutorial](TUTORIAL.md) - Start here!
- [All Features](FEATURES.md) - Tool catalog
- [Issues](https://github.com/yourusername/comfyui-mcp-server/issues)
- [Discussions](https://github.com/yourusername/comfyui-mcp-server/discussions)

---

<p align="center">
  <strong>Built with ❤️ by the ComfyUI MCP Team</strong>
</p>