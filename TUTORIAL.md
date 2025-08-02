# ComfyUI MCP Server - Complete Tutorial & Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Core Concepts](#core-concepts)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Workflow Automation](#workflow-automation)
7. [AI-Powered Features](#ai-powered-features)
8. [Batch Processing](#batch-processing)
9. [Real-World Examples](#real-world-examples)
10. [API Reference](#api-reference)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)

---

## Introduction

ComfyUI MCP Server is a comprehensive creative automation platform that bridges AI agents with ComfyUI's powerful generation capabilities. With 89 specialized tools, it transforms ComfyUI into an intelligent creative powerhouse.

### What Makes It Special?

- **89 Specialized Tools**: Complete coverage of image/video generation, enhancement, and analysis
- **AI Intelligence**: Learns from every generation to improve results
- **Enterprise Ready**: Production features like batch processing and error recovery
- **Natural Language**: Control complex operations through simple commands

### Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   AI Agents     │────▶│   MCP Server    │────▶│    ComfyUI      │
│ (Claude, etc)   │     │   (89 tools)    │     │   (Workflows)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                         │
         │                       ▼                         │
         │              ┌─────────────────┐               │
         │              │ Learning System │               │
         │              │   (AI Brain)    │               │
         │              └─────────────────┘               │
         │                       │                         │
         └───────────────────────┴─────────────────────────┘
                        Feedback Loop
```

---

## Getting Started

### Prerequisites

1. **Python 3.10+** - Required for the server
2. **ComfyUI** - Must be installed and running
3. **Models** - At least one checkpoint model in ComfyUI
4. **Dependencies** - Install via requirements.txt

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/comfyui-mcp-server.git
cd comfyui-mcp-server

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure ComfyUI URL (if not default)
export COMFYUI_URL="http://localhost:8188"
```

### First Run

```bash
# Terminal 1: Start ComfyUI
cd /path/to/ComfyUI
python main.py

# Terminal 2: Start MCP Server
cd /path/to/comfyui-mcp-server
python server.py
```

You should see:
```
Starting ComfyUI MCP server v1.0.0
ComfyUI URL: http://localhost:8188
Total tools available: 89
ComfyUI connection verified
```

---

## Core Concepts

### Tools

Tools are the building blocks of the MCP server. Each tool performs a specific function:

```python
# Simple tool usage
result = generate_image(
    prompt="sunset over mountains",
    width=1024,
    height=768
)
```

### Workflows

Workflows are ComfyUI node graphs that define generation pipelines:

```python
# Workflows are stored as JSON in the workflows/ directory
workflows/
├── basic_api_test.json      # Simple text-to-image
├── upscale_workflow.json     # Image upscaling
├── inpaint_workflow.json     # Inpainting
└── video_gen_workflow.json   # Video generation
```

### Presets

Presets are saved workflows with metadata:

```python
# Save a workflow for reuse
save_workflow_preset(
    name="Product Photography",
    description="High-quality product shots",
    workflow={...},
    tags=["product", "ecommerce"]
)
```

---

## Basic Usage

### 1. Generate Your First Image

```python
# Basic generation
mcp call generate_image '{
    "prompt": "a beautiful sunset over the ocean",
    "width": 512,
    "height": 512
}'
```

### 2. Improve Your Prompts

```python
# AI-powered prompt optimization
mcp call optimize_prompt_with_ai '{
    "prompt": "cat sitting on chair",
    "optimization_goals": ["clarity", "detail", "style"]
}'
# Returns: "cat sitting on chair, high quality, detailed, professional photography"
```

### 3. Use Style Presets

```python
# Generate with predefined styles
mcp call generate_with_style_preset '{
    "prompt": "mountain landscape",
    "style": "cinematic",
    "quality_preset": "ultra"
}'
```

### 4. Batch Generation

```python
# Generate multiple images
mcp call batch_generate '{
    "prompts": [
        "red apple on white background",
        "green apple on white background",
        "yellow apple on white background"
    ],
    "width": 512,
    "height": 512
}'
```

---

## Advanced Features

### Workflow Management

#### Save Custom Workflows

```python
# Create a reusable workflow
workflow = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {...}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {...}},
    "3": {"class_type": "KSampler", "inputs": {...}}
}

save_workflow_preset(
    name="My Custom Workflow",
    description="Optimized for portraits",
    workflow=workflow,
    tags=["portrait", "custom"]
)
```

#### Share Workflows

```python
# Share with time-limited access
share_result = share_workflow_preset(
    preset_name="My Custom Workflow",
    expiry_hours=72,
    password="secret123"
)
# Returns: {"share_code": "abc123", "expires_at": "..."}
```

### Advanced Control

#### ControlNet Generation

```python
# Use pose guidance
mcp call controlnet_generate '{
    "prompt": "person in business suit",
    "control_image": "pose_reference.png",
    "control_type": "openpose",
    "control_strength": 0.8
}'
```

#### Inpainting

```python
# Replace specific areas
mcp call inpaint_image '{
    "image_path": "original.png",
    "mask_path": "mask.png",
    "prompt": "modern smartphone",
    "strength": 0.9
}'
```

### Enhancement Tools

#### AI Upscaling

```python
# Upscale with AI
mcp call upscale_image '{
    "image_path": "low_res.png",
    "scale": 4,
    "model": "ESRGAN"
}'
```

#### Face Restoration

```python
# Enhance faces in photos
mcp call face_restore '{
    "image_path": "old_photo.jpg",
    "strength": 0.8,
    "model": "GFPGAN"
}'
```

---

## Workflow Automation

### Creating Complex Pipelines

```python
# Define a multi-step operation
tasks = [
    {
        "id": "generate",
        "type": "generate",
        "parameters": {
            "prompt": "product on white background",
            "width": 1024,
            "height": 1024
        }
    },
    {
        "id": "remove_bg",
        "type": "remove_background",
        "parameters": {
            "method": "automatic"
        },
        "depends_on": ["generate"]
    },
    {
        "id": "upscale",
        "type": "upscale",
        "parameters": {
            "scale": 2,
            "model": "ESRGAN"
        },
        "depends_on": ["remove_bg"]
    }
]

# Execute with dependency management
coordinate_batch_operation(
    operation_name="Product Pipeline",
    tasks=tasks,
    parallel_limit=2
)
```

### Conditional Workflows

```python
# Dynamic execution based on conditions
mcp call conditional_workflow '{
    "prompt": "landscape photo",
    "conditions": {
        "if_aspect_ratio": "portrait",
        "then_workflow": "portrait_optimized",
        "else_workflow": "landscape_optimized"
    }
}'
```

---

## AI-Powered Features

### Learning System

The server learns from every generation:

```python
# Track performance metrics
track_generation_metrics(
    prompt_id="gen_12345",
    metrics={
        "quality": 0.92,
        "adherence": 0.88,
        "speed": 0.75
    },
    user_rating=5
)
```

### A/B Testing

```python
# Test different approaches
create_prompt_ab_test(
    test_name="Lighting Study",
    base_prompt="portrait with {lighting} lighting",
    variations=[
        {"lighting": "soft"},
        {"lighting": "dramatic"},
        {"lighting": "natural"}
    ],
    samples_per_variation=10
)
```

### Automatic Optimization

```python
# Create workflow from successful patterns
optimized_workflow = create_workflow_from_metrics(
    metric_threshold={"quality": 0.9},
    time_period_days=30
)
```

---

## Batch Processing

### Simple Batch

```python
# Generate variations
mcp call batch_generate '{
    "prompts": ["cat", "dog", "bird"],
    "width": 512,
    "height": 512,
    "model": "sd_xl_base_1.0.safetensors"
}'
```

### Complex Batch with Dependencies

```python
# Product catalog generation
batch_operation = {
    "operation_name": "Q4 Catalog",
    "tasks": [
        {"id": "hero_1", "type": "generate", "prompt": "hero product shot"},
        {"id": "detail_1", "type": "generate", "prompt": "product details"},
        {"id": "lifestyle_1", "type": "generate", "prompt": "product in use"},
        {"id": "compile", "type": "create_collage", "depends_on": ["hero_1", "detail_1", "lifestyle_1"]}
    ],
    "checkpoint_interval": 10,
    "error_handling": "continue"
}

coordinate_batch_operation(**batch_operation)
```

---

## Real-World Examples

### E-commerce Product Photography

```python
# Complete product photography workflow
def process_product_images(product_name, sku):
    # 1. Generate hero shot
    hero = generate_image(
        prompt=f"{product_name} professional product photography on white background",
        width=2048,
        height=2048,
        steps=40
    )
    
    # 2. Remove background
    transparent = remove_background(
        image_path=hero["image_path"],
        method="automatic"
    )
    
    # 3. Create variations for different platforms
    variations = {}
    for platform in ["amazon", "shopify", "instagram"]:
        optimized = optimize_for_platform(
            image_path=transparent["output_path"],
            platform=platform
        )
        variations[platform] = optimized
    
    # 4. Generate lifestyle shots
    lifestyle_prompts = [
        f"{product_name} being used in modern home",
        f"{product_name} in minimalist setting",
        f"{product_name} with natural lighting"
    ]
    
    lifestyle_shots = batch_generate(
        prompts=lifestyle_prompts,
        style_preset="commercial"
    )
    
    return {
        "sku": sku,
        "hero": hero,
        "transparent": transparent,
        "platform_variations": variations,
        "lifestyle_shots": lifestyle_shots
    }
```

### Social Media Content Creation

```python
# Automated content generation for social media
def create_social_campaign(theme, num_posts=10):
    # 1. Generate base concepts
    concepts = []
    for i in range(num_posts):
        optimized_prompt = optimize_prompt_with_ai(
            prompt=f"{theme} social media post concept {i+1}",
            optimization_goals=["engagement", "trendy", "shareable"]
        )
        concepts.append(optimized_prompt["optimized_prompt"])
    
    # 2. A/B test styles
    style_test = create_prompt_ab_test(
        test_name=f"{theme} Style Test",
        base_prompt=concepts[0],
        variations=[
            {"style": "vibrant"},
            {"style": "minimal"},
            {"style": "retro"}
        ]
    )
    
    # 3. Generate content with winning style
    winning_style = style_test["winner"]["style"]
    
    social_content = batch_generate(
        prompts=concepts,
        style_preset=winning_style,
        quality="high"
    )
    
    # 4. Optimize for each platform
    platform_optimized = {}
    for platform in ["instagram", "twitter", "facebook"]:
        platform_optimized[platform] = []
        for content in social_content["results"]:
            optimized = optimize_for_platform(
                image_path=content["image_path"],
                platform=platform
            )
            platform_optimized[platform].append(optimized)
    
    return {
        "campaign": theme,
        "concepts": concepts,
        "winning_style": winning_style,
        "content": platform_optimized
    }
```

### AI Art Gallery Creation

```python
# Create themed art exhibition
def create_art_exhibition(theme, num_pieces=20):
    # 1. Research and optimize theme
    theme_analysis = analyze_prompt(
        prompt=theme,
        target_style="artistic"
    )
    
    # 2. Generate artwork variations
    artworks = []
    for i in range(num_pieces):
        # Create unique variation
        variation_prompt = f"{theme}, artwork {i+1}, {theme_analysis['style_suggestions'][i % len(theme_analysis['style_suggestions'])]}"
        
        artwork = generate_with_style_preset(
            prompt=variation_prompt,
            style="artistic",
            quality_preset="ultra"
        )
        
        # Analyze composition
        composition = analyze_image_composition(
            image_path=artwork["image_path"]
        )
        
        artworks.append({
            "artwork": artwork,
            "composition": composition,
            "prompt": variation_prompt
        })
    
    # 3. Curate best pieces
    curated = sorted(
        artworks,
        key=lambda x: x["composition"]["aesthetic_score"],
        reverse=True
    )[:10]
    
    # 4. Create gallery catalog
    catalog = create_output_catalog(
        images=[art["artwork"]["image_path"] for art in curated],
        format="html",
        title=f"{theme} Exhibition",
        include_metadata=True
    )
    
    return {
        "theme": theme,
        "total_generated": num_pieces,
        "curated_pieces": len(curated),
        "gallery": catalog,
        "artworks": curated
    }
```

---

## API Reference

### Core Generation Tools

#### generate_image
```python
generate_image(
    prompt: str,                    # Required: Text description
    width: int = 512,              # Image width (divisible by 8)
    height: int = 512,             # Image height (divisible by 8)
    steps: int = 20,               # Sampling steps (1-150)
    cfg_scale: float = 7.0,        # Guidance scale (1.0-30.0)
    seed: Optional[int] = None,    # Random seed (-1 for random)
    model: Optional[str] = None,   # Model checkpoint name
    sampler_name: str = "euler",   # Sampling method
    scheduler: str = "normal",     # Scheduler type
    negative_prompt: str = ""      # What to avoid
) -> Dict[str, Any]
```

#### optimize_prompt_with_ai
```python
optimize_prompt_with_ai(
    prompt: str,                              # Original prompt
    optimization_goals: List[str],            # ["quality", "detail", "style"]
    target_style: Optional[str] = None,       # Target artistic style
    reference_successful: bool = True         # Learn from successes
) -> Dict[str, Any]
```

### Workflow Management

#### save_workflow_preset
```python
save_workflow_preset(
    name: str,                          # Preset name
    description: str,                   # Description
    workflow: Dict[str, Any],          # ComfyUI workflow JSON
    tags: Optional[List[str]] = None,  # Categorization tags
    is_public: bool = False            # Share publicly
) -> Dict[str, Any]
```

### Batch Operations

#### coordinate_batch_operation
```python
coordinate_batch_operation(
    operation_name: str,                      # Operation identifier
    tasks: List[Dict[str, Any]],            # Task list with dependencies
    parallel_limit: int = 3,                 # Max parallel tasks
    error_handling: str = "continue",        # "stop", "retry", "continue"
    progress_webhook: Optional[str] = None   # Progress notification URL
) -> Dict[str, Any]
```

---

## Troubleshooting

### Common Issues

#### 1. Connection Failed
```
Error: Failed to connect to ComfyUI
```
**Solution**: Ensure ComfyUI is running on the correct port:
```bash
# Check if ComfyUI is running
curl http://localhost:8188/system_stats

# If using different port
export COMFYUI_URL="http://localhost:YOUR_PORT"
```

#### 2. Model Not Found
```
Error: Model 'model_name.safetensors' not found
```
**Solution**: Check available models:
```python
mcp call list_models '{"model_type": "checkpoints"}'
```

#### 3. Out of Memory
```
Error: CUDA out of memory
```
**Solution**: Clear cache and reduce batch size:
```python
mcp call clear_comfyui_cache
```

#### 4. Width/Height Error
```
Error: Width/height must be divisible by 8
```
**Solution**: Use multiples of 8:
```python
# Good: 512, 768, 1024, 1280
# Bad: 500, 750, 1000
```

### Performance Optimization

#### Memory Management
```python
# Monitor memory usage
stats = get_system_stats()
if stats["vram_usage_percent"] > 90:
    clear_comfyui_cache()
```

#### Queue Management
```python
# Check queue before submitting
queue = get_queue_status()
if queue["queue_pending"] > 10:
    # Wait or adjust priority
    queue_priority(prompt_id, priority="high")
```

---

## Best Practices

### 1. Prompt Engineering

```python
# ❌ Bad: Vague prompt
"cat"

# ✅ Good: Detailed prompt
"orange tabby cat sitting on vintage leather armchair, warm lighting, cozy interior, professional photography"

# ✅ Better: AI-optimized
optimized = optimize_prompt_with_ai(
    "cat on chair",
    optimization_goals=["clarity", "atmosphere", "quality"]
)
```

### 2. Workflow Organization

```python
# Create logical preset hierarchies
save_workflow_preset(
    name="Photography/Product/Electronics",
    workflow={...},
    tags=["photography", "product", "electronics", "commercial"]
)
```

### 3. Error Handling

```python
# Always handle errors gracefully
try:
    result = generate_image(prompt="test")
except Exception as e:
    # Log error
    logger.error(f"Generation failed: {e}")
    # Try alternative approach
    result = generate_with_style_preset(
        prompt="test",
        style="photorealistic",
        quality_preset="fast"
    )
```

### 4. Resource Management

```python
# Batch operations efficiently
# ❌ Bad: Individual calls
for prompt in prompts:
    generate_image(prompt=prompt)

# ✅ Good: Batch processing
batch_generate(prompts=prompts, parallel_limit=3)
```

### 5. Learning Integration

```python
# Always track metrics for improvement
result = generate_image(prompt="sunset")
track_generation_metrics(
    prompt_id=result["prompt_id"],
    metrics={"quality": 0.9, "speed": 0.8},
    user_rating=5
)
```

---

## Advanced Topics

### Custom Workflows

Create complex multi-node workflows:

```python
workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
    },
    "2": {
        "class_type": "LoraLoader",
        "inputs": {
            "model": ["1", 0],
            "clip": ["1", 1],
            "lora_name": "style_lora.safetensors",
            "strength_model": 0.8
        }
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "your prompt here",
            "clip": ["2", 1]
        }
    }
    # ... more nodes
}
```

### Performance Monitoring

```python
# Get detailed performance metrics
metrics = get_performance_metrics()
print(f"Average generation time: {metrics['avg_generation_time']}s")
print(f"Success rate: {metrics['success_rate']*100}%")
print(f"Queue efficiency: {metrics['queue_efficiency']*100}%")
```

### Integration Examples

#### With Claude Desktop
```json
{
  "mcpServers": {
    "comfyui": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/comfyui-mcp-server"
    }
  }
}
```

#### Python Client
```python
from mcp import Client

client = Client("comfyui")
result = await client.call_tool(
    "generate_image",
    {"prompt": "beautiful landscape"}
)
```

---

## Conclusion

ComfyUI MCP Server provides a comprehensive platform for creative automation. With 89 tools, intelligent features, and enterprise-grade reliability, it transforms ComfyUI into a powerhouse for AI-assisted content creation.

### Key Takeaways

1. **Start Simple**: Begin with basic generation and gradually explore advanced features
2. **Use Intelligence**: Let the AI optimization improve your results
3. **Automate Workflows**: Save time with presets and batch operations
4. **Monitor Performance**: Track metrics to continuously improve
5. **Share Knowledge**: Export and share successful workflows

### Next Steps

- Explore the [complete tool catalog](FEATURES.md)
- Join the community discussions
- Contribute to the project
- Build amazing creative applications!

---

<p align="center">
  <strong>Happy Creating with ComfyUI MCP Server! 🎨</strong>
</p>