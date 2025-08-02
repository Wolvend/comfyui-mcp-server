#!/usr/bin/env python3
"""
Example usage of ComfyUI MCP Server v2.6.0 new features

These examples demonstrate the new tools added in v2.6.0:
- Workflow Preset Management
- Intelligent Prompt Optimization
- Batch Operations Coordination
- Automatic Mask Generation
"""

# Example 1: Workflow Preset Management
print("=== Example 1: Workflow Preset Management ===")
print("""
# Save a reusable workflow preset
mcp call save_workflow_preset '{
    "name": "Product Photography Studio",
    "description": "Professional product photos with white background",
    "workflow": {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "product photography, studio lighting", "clip": ["1", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"steps": 30, "cfg": 7.5, "sampler_name": "dpmpp_2m", "model": ["1", 0]}}
    },
    "tags": ["product", "ecommerce", "studio"],
    "is_public": false
}'

# List available presets
mcp call list_workflow_presets '{"tags": ["product"]}'

# Apply the preset with custom parameters
mcp call apply_workflow_preset '{
    "preset_name": "Product Photography Studio",
    "parameters": {
        "2": {"text": "luxury watch, studio lighting, white background"}
    }
}'
""")

# Example 2: Intelligent Prompt Optimization
print("\n=== Example 2: Intelligent Prompt Optimization ===")
print("""
# Optimize a basic prompt
mcp call optimize_prompt_with_ai '{
    "prompt": "cat on sofa",
    "optimization_goals": ["clarity", "detail", "style"],
    "target_style": "photorealistic"
}'
# Output: "cat on sofa, photorealistic, high quality, intricate details, studio lighting"

# Generate with style presets
mcp call generate_with_style_preset '{
    "prompt": "mountain landscape at sunset",
    "style": "cinematic",
    "quality_preset": "ultra",
    "auto_optimize": true
}'
""")

# Example 3: Batch Operations
print("\n=== Example 3: Batch Operations Coordination ===")
print("""
# Coordinate multiple operations
mcp call coordinate_batch_operation '{
    "operation_name": "Product Variant Generation",
    "tasks": [
        {
            "id": "red_variant",
            "type": "generate",
            "parameters": {
                "prompt": "red sports car, studio lighting",
                "width": 1024,
                "height": 768
            }
        },
        {
            "id": "blue_variant",
            "type": "generate",
            "parameters": {
                "prompt": "blue sports car, studio lighting",
                "width": 1024,
                "height": 768
            }
        },
        {
            "id": "upscale_red",
            "type": "upscale",
            "parameters": {
                "image_path": "output/red_car.png",
                "scale": 2
            }
        }
    ],
    "parallel_limit": 2,
    "error_handling": "continue"
}'
""")

# Example 4: Automatic Mask Generation
print("\n=== Example 4: Automatic Mask Generation ===")
print("""
# Generate foreground mask
mcp call auto_generate_mask '{
    "image_path": "/path/to/product.jpg",
    "target": "foreground",
    "refinement": "high",
    "preview": true
}'

# Generate background mask for removal
mcp call auto_generate_mask '{
    "image_path": "/path/to/portrait.jpg",
    "target": "background",
    "refinement": "medium",
    "preview": true
}'
""")

# Example 5: Complete E-commerce Workflow
print("\n=== Example 5: Complete E-commerce Product Pipeline ===")
print("""
# Step 1: Create and save the workflow preset
mcp call save_workflow_preset '{
    "name": "Ecommerce Product Pipeline",
    "description": "Complete pipeline: generate, remove bg, upscale, optimize",
    "workflow": {...},
    "tags": ["ecommerce", "automated", "production"]
}'

# Step 2: Batch process multiple products
mcp call coordinate_batch_operation '{
    "operation_name": "Q4 Product Catalog",
    "tasks": [
        {
            "id": "generate_hero_shots",
            "type": "workflow",
            "parameters": {
                "preset_name": "Ecommerce Product Pipeline",
                "parameters": {"prompt": "luxury watch on white background"}
            }
        },
        {
            "id": "create_lifestyle_shots",
            "type": "generate",
            "parameters": {
                "prompt": "luxury watch on wrist, business setting",
                "style": "photorealistic"
            }
        },
        {
            "id": "optimize_for_web",
            "type": "workflow",
            "parameters": {
                "preset_name": "Web Optimization",
                "parameters": {"target_size": "1200x1200"}
            }
        }
    ],
    "parallel_limit": 3,
    "error_handling": "retry",
    "progress_webhook": "https://your-server.com/webhook/progress"
}'
""")

# Example 6: AI-Powered Creative Variations
print("\n=== Example 6: AI-Powered Creative Variations ===")
print("""
# Generate variations with AI optimization
prompts = [
    "sunset over mountains",
    "sunrise over ocean",
    "twilight in forest"
]

for prompt in prompts:
    # First optimize the prompt
    optimized = mcp call optimize_prompt_with_ai '{
        "prompt": "' + prompt + '",
        "optimization_goals": ["artistic", "mood", "atmosphere"],
        "target_style": "cinematic"
    }'
    
    # Then generate with style preset
    mcp call generate_with_style_preset '{
        "prompt": "' + optimized.optimized_prompt + '",
        "style": "cinematic",
        "quality_preset": "ultra"
    }'
""")

print("\n=== Additional Resources ===")
print("""
- Full documentation: /home/wolvend/Desktop/ComfyUI/custom_nodes/comfyui-mcp-server/README.md
- Use case analysis: /home/wolvend/Desktop/ComfyUI/custom_nodes/comfyui-mcp-server/USE_CASES_AND_IMPROVEMENTS.md
- Release notes: /home/wolvend/Desktop/ComfyUI/custom_nodes/comfyui-mcp-server/V2.6.0_RELEASE_NOTES.md

To use these examples:
1. Ensure ComfyUI is running (http://localhost:8188)
2. Start the MCP server: python server.py
3. Use the MCP CLI or integrate with your application
""")