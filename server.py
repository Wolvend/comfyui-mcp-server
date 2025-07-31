#!/usr/bin/env python3
import json
import logging
import os
import sys
from typing import Optional, Dict, Any
from mcp.server.fastmcp import FastMCP
from comfyui_client import ComfyUIClient

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG") else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ComfyUI_MCP_Server")

# Initialize ComfyUI client with environment variable support
comfyui_url = os.getenv("COMFYUI_URL", "http://localhost:8188")
comfyui_api_key = os.getenv("COMFYUI_API_KEY", None)

try:
    comfyui_client = ComfyUIClient(comfyui_url)
    logger.info(f"ComfyUI client initialized (URL: {comfyui_url})")
except Exception as e:
    logger.error(f"Failed to initialize ComfyUI client: {e}")
    sys.exit(1)

# Create FastMCP server with version info
mcp = FastMCP(
    "comfyui-mcp-server",
    version="1.0.0"
)

# Add server capabilities information
@mcp.tool()
def get_server_info() -> Dict[str, Any]:
    """Get information about the ComfyUI MCP server
    
    Returns:
        Server information including version, ComfyUI URL, and available models
    """
    try:
        return {
            "server_version": "1.0.0",
            "comfyui_url": comfyui_url,
            "available_models": comfyui_client.available_models or [],
            "status": "connected"
        }
    except Exception as e:
        logger.error(f"Error getting server info: {e}")
        return {
            "server_version": "1.0.0",
            "comfyui_url": comfyui_url,
            "status": "error",
            "error": str(e)
        }

@mcp.tool()
def list_workflows() -> Dict[str, Any]:
    """List available ComfyUI workflows
    
    Returns:
        Dictionary containing available workflow IDs
    """
    try:
        import glob
        workflow_files = glob.glob("workflows/*.json")
        workflows = [os.path.basename(f).replace('.json', '') for f in workflow_files]
        return {
            "workflows": workflows,
            "count": len(workflows)
        }
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        return {
            "error": str(e),
            "workflows": []
        }

@mcp.tool()
def generate_image(
    prompt: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    workflow_id: Optional[str] = None,
    model: Optional[str] = None,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    sampler_name: Optional[str] = None,
    scheduler: Optional[str] = None,
    denoise: Optional[float] = None,
    negative_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Generate an image using ComfyUI with extensive customization options
    
    Args:
        prompt: The text prompt for image generation (required)
        width: Image width in pixels (default: 512, must be divisible by 8)
        height: Image height in pixels (default: 512, must be divisible by 8)
        workflow_id: Workflow ID to use (default: basic_api_test)
        model: Model checkpoint name to use (optional, must be from available models)
        seed: Random seed for reproducibility (optional, -1 for random)
        steps: Number of sampling steps (optional, typically 20-50)
        cfg_scale: Classifier-free guidance scale (optional, typically 1.0-20.0)
        sampler_name: Sampling method (optional, e.g., 'euler', 'dpm_2', 'ddim')
        scheduler: Scheduler type (optional, e.g., 'normal', 'karras', 'exponential')
        denoise: Denoising strength for img2img (optional, 0.0-1.0)
        negative_prompt: Negative prompt to avoid certain features (optional)
    
    Returns:
        Dictionary containing the image URL or error information
    """
    try:
        # Input validation
        if not prompt or not isinstance(prompt, str):
            return {"error": "Prompt must be a non-empty string"}
        
        # Set defaults with validation
        width = width or 512
        height = height or 512
        workflow_id = workflow_id or "basic_api_test"
        
        # Validate dimensions
        if width <= 0 or height <= 0:
            return {"error": "Width and height must be positive integers"}
        
        if width % 8 != 0 or height % 8 != 0:
            return {"error": "Width and height must be divisible by 8"}
        
        if width > 2048 or height > 2048:
            return {"error": "Width and height must not exceed 2048 pixels"}
        
        # Validate model if specified
        if model and comfyui_client.available_models:
            if model not in comfyui_client.available_models:
                return {
                    "error": f"Model '{model}' not found",
                    "available_models": comfyui_client.available_models
                }
        
        # Validate numeric parameters
        if seed is not None and not isinstance(seed, int):
            return {"error": "Seed must be an integer"}
        
        if steps is not None:
            if not isinstance(steps, int) or steps < 1 or steps > 150:
                return {"error": "Steps must be an integer between 1 and 150"}
        
        if cfg_scale is not None:
            if not isinstance(cfg_scale, (int, float)) or cfg_scale < 0 or cfg_scale > 30:
                return {"error": "CFG scale must be a number between 0 and 30"}
        
        if denoise is not None:
            if not isinstance(denoise, (int, float)) or denoise < 0 or denoise > 1:
                return {"error": "Denoise must be a number between 0 and 1"}
        
        logger.info(f"Generating image - Prompt: '{prompt[:50]}...', Size: {width}x{height}, Workflow: {workflow_id}")
        
        # Build parameters dict for ComfyUI client
        params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "workflow_id": workflow_id
        }
        
        # Add optional parameters if provided
        if model:
            params["model"] = model
        if seed is not None:
            params["seed"] = seed
        if steps is not None:
            params["steps"] = steps
        if cfg_scale is not None:
            params["cfg_scale"] = cfg_scale
        if sampler_name:
            params["sampler_name"] = sampler_name
        if scheduler:
            params["scheduler"] = scheduler
        if denoise is not None:
            params["denoise"] = denoise
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
        
        # Call ComfyUI client
        image_url = comfyui_client.generate_image(**params)
        
        logger.info(f"Image generated successfully: {image_url}")
        
        return {
            "success": True,
            "image_url": image_url,
            "parameters": params
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {"error": f"Invalid parameter: {str(e)}"}
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return {"error": "Failed to connect to ComfyUI. Is it running?"}
    except TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        return {"error": "Request timed out. ComfyUI might be busy or unresponsive."}
    except Exception as e:
        logger.error(f"Unexpected error generating image: {e}", exc_info=True)
        return {"error": f"Unexpected error: {str(e)}"}

@mcp.tool()
def get_generation_status(prompt_id: str) -> Dict[str, Any]:
    """Check the status of an image generation task
    
    Args:
        prompt_id: The ID of the generation task to check
        
    Returns:
        Dictionary containing the status and progress information
    """
    try:
        # This would need to be implemented in comfyui_client
        # For now, return a placeholder
        return {
            "prompt_id": prompt_id,
            "status": "unknown",
            "message": "Status checking not yet implemented"
        }
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        return {"error": str(e)}

# Health check endpoint
@mcp.tool()
def health_check() -> Dict[str, Any]:
    """Check if ComfyUI is accessible and responsive
    
    Returns:
        Health status of the ComfyUI connection
    """
    try:
        import requests
        response = requests.get(f"{comfyui_url}/system_stats", timeout=5)
        if response.status_code == 200:
            return {
                "status": "healthy",
                "comfyui_url": comfyui_url,
                "response_time_ms": int(response.elapsed.total_seconds() * 1000)
            }
        else:
            return {
                "status": "unhealthy",
                "comfyui_url": comfyui_url,
                "http_status": response.status_code
            }
    except requests.exceptions.Timeout:
        return {
            "status": "timeout",
            "comfyui_url": comfyui_url,
            "message": "ComfyUI did not respond within 5 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "comfyui_url": comfyui_url,
            "error": str(e)
        }

if __name__ == "__main__":
    logger.info(f"Starting ComfyUI MCP server v1.0.0")
    logger.info(f"ComfyUI URL: {comfyui_url}")
    logger.info(f"Debug mode: {'ON' if os.getenv('DEBUG') else 'OFF'}")
    
    # Check ComfyUI connection on startup
    health = health_check()
    if health["status"] != "healthy":
        logger.warning(f"ComfyUI health check failed: {health}")
    else:
        logger.info("ComfyUI connection verified")
    
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server crashed: {e}", exc_info=True)
        sys.exit(1)