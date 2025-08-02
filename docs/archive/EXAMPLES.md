# ComfyUI MCP Server Examples

## 🚀 Quick Start Examples

### Basic Image Generation
```
Ask Claude: "Generate an image of a cyberpunk city at night"
```

### Advanced ControlNet Support
```
Ask Claude: "Use ControlNet to generate an image with this pose reference image, prompt: 'a warrior in medieval armor'"
```

### Image Enhancement & Upscaling
```
Ask Claude: "Upscale this image 4x using ESRGAN"
```

### Batch Generation
```
Ask Claude: "Generate 5 variations of 'a fantasy landscape' with different moods: mystical, dark, bright, ancient, futuristic"
```

## 🎨 Creative Workflows

### Style Transfer
```
Ask Claude: "Apply Van Gogh's starry night style to this photograph"
```

### Inpainting Support
```
Ask Claude: "Remove the person from this image and fill with appropriate background"
```

### LoRA Styling
```
Ask Claude: "Generate a portrait using anime and cyberpunk LoRA styles"
```

## 🔧 System Management

### Check Status
```
Ask Claude: "What's the current ComfyUI queue status?"
```

### Monitor Performance
```
Ask Claude: "Show me system performance metrics"
```

### Detect Capabilities
```
Ask Claude: "What custom nodes are installed in ComfyUI?"
```

## 📊 Real-time Features

### Progress Tracking
```
Ask Claude: "Show me the progress of prompt ID 12345"
```

### Queue Management
```
Ask Claude: "What's my position in the generation queue?"
```

## 🎯 Pro Tips

1. **Use Specific Dimensions**: "Generate at 768x512 for landscape"
2. **Control Seeds**: "Use seed 42 for reproducible results"
3. **Batch with Seeds**: "Generate 3 images with seed increment"
4. **Monitor Resources**: "Check GPU memory before large batch"
5. **Template Workflows**: "Use portrait template with my custom settings"

## 🚨 Troubleshooting Examples

### Check Connection
```
Ask Claude: "Is ComfyUI running and accessible?"
```

### Validate Workflow
```
Ask Claude: "Validate the controlnet_workflow for any issues"
```

### Clear Memory
```
Ask Claude: "Clear ComfyUI cache to free up VRAM"
```

## 📋 Feature Matrix

| Feature | Status | Example Command |
|---------|--------|-----------------|
| Basic Generation | ✅ | "Generate an image of a cat" |
| ControlNet | ✅ | "Use pose control with this image" |
| Upscaling | ✅ | "Upscale this image 2x" |
| Inpainting | ✅ | "Remove object from this area" |
| Batch Processing | ✅ | "Generate 5 different landscapes" |
| Queue Management | ✅ | "Show my queue position" |
| WebSocket Progress | ✅ | "Track progress of my generation" |
| Performance Monitoring | ✅ | "Show system metrics" |
| Custom Node Detection | ✅ | "What extensions are installed?" |

## 🎬 Video Generation Examples  
```
Ask Claude: "Generate a 3-second video of ocean waves"
```
*Note: Requires video generation custom nodes*

## 🔄 Advanced Automation
```
Ask Claude: "Create an animation sequence morphing through: sunrise, noon, sunset, night"
```
*Note: Uses template system with batch processing*