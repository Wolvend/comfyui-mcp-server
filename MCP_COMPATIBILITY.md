# ComfyUI MCP Server - Compatibility Guide

## Overview
This MCP server has been designed for maximum compatibility with different MCP clients and ComfyUI configurations.

## Key Compatibility Features

### 1. **Flexible Parameter Handling**
- All image generation parameters are optional except `prompt`
- Supports both basic and advanced workflows
- Gracefully handles missing or invalid parameters with clear error messages

### 2. **Environment Variable Support**
- `COMFYUI_URL`: Set custom ComfyUI URL (default: http://localhost:8188)
- `COMFYUI_API_KEY`: For future authentication support
- `DEBUG`: Enable debug logging for troubleshooting

### 3. **Robust Error Handling**
- Specific error types for different failure scenarios:
  - Connection errors when ComfyUI is not running
  - Timeout errors for slow operations
  - Validation errors for invalid parameters
- All errors return structured JSON responses

### 4. **Input Validation**
- Image dimensions must be divisible by 8 (SD requirement)
- Maximum dimensions: 2048x2048
- Parameter range validation (steps, cfg_scale, denoise)
- Model name validation against available models

### 5. **Health Monitoring**
- `health_check()`: Verify ComfyUI connectivity
- `get_server_info()`: Get server status and capabilities
- Startup health check with warnings

### 6. **Multiple Tool Endpoints**
- `generate_image`: Main image generation with all parameters
- `list_workflows`: Discover available workflows
- `get_server_info`: Server capabilities and status
- `health_check`: Connection verification

## MCP Protocol Compliance

### Supported MCP Features
- ✅ JSON-RPC over stdio
- ✅ Tool discovery via `list_tools`
- ✅ Typed parameters with validation
- ✅ Structured error responses
- ✅ Server metadata and versioning

### Client Compatibility
- Works with Claude Desktop MCP client
- Compatible with MCP CLI tools
- Supports custom MCP clients following the protocol

## ComfyUI API Compatibility

### Supported ComfyUI Versions
- Tested with ComfyUI 0.3.x
- Should work with any version exposing the standard API

### Workflow Compatibility
- Supports custom workflows via JSON files
- Default workflow provided for basic usage
- Extensible parameter mapping system

## Usage Examples

### Basic Usage
```json
{
  "tool": "generate_image",
  "arguments": {
    "prompt": "a beautiful sunset"
  }
}
```

### Advanced Usage
```json
{
  "tool": "generate_image",
  "arguments": {
    "prompt": "a detailed oil painting of a mountain landscape",
    "width": 1024,
    "height": 768,
    "steps": 30,
    "cfg_scale": 7.5,
    "sampler_name": "dpm_2",
    "seed": 12345,
    "negative_prompt": "blurry, low quality"
  }
}
```

## Troubleshooting

### Debug Mode
Enable debug logging:
```bash
DEBUG=1 python server.py
```

### Common Issues

1. **"Failed to reconnect to comfyui"**
   - Ensure ComfyUI is running on the configured URL
   - Check firewall/network settings
   - Verify COMFYUI_URL environment variable

2. **"Model not found"**
   - Use `get_server_info()` to list available models
   - Ensure model names match exactly (case-sensitive)

3. **Parameter validation errors**
   - Check parameter ranges in tool description
   - Ensure dimensions are divisible by 8

## Future Compatibility

### Planned Features
- [ ] Batch image generation
- [ ] Image-to-image workflows
- [ ] Real-time progress updates
- [ ] Custom node support
- [ ] Authentication/API key support

### API Stability
- Tool names and core parameters will remain stable
- New optional parameters may be added
- Error response format is stable
- Version included in server info for client adaptation