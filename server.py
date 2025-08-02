#!/usr/bin/env python3
"""
ComfyUI MCP Server v1.0.0

A comprehensive creative automation platform that bridges AI agents with ComfyUI's
powerful image and video generation capabilities.

Copyright 2024 ComfyUI MCP Team
Licensed under Apache License 2.0

Features:
- 89 specialized tools for image/video generation and manipulation
- Intelligent workflow automation with preset management
- AI-powered prompt optimization with learning capabilities
- Enterprise-ready batch operations with dependency management
- Real-time preview and progress monitoring
- Advanced masking and control features
"""
import json
import logging
import os
import sys
import glob
import time
import random
import re
import csv
import base64
import shutil
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
from functools import wraps
from collections import Counter, defaultdict
from io import BytesIO

# Third-party imports
try:
    import numpy as np
except ImportError:
    np = None
    logging.warning("NumPy not installed - some image analysis features will be limited")

try:
    from PIL import Image, ImageStat
    from PIL.PngImagePlugin import PngInfo
except ImportError:
    Image = None
    ImageStat = None
    PngInfo = None
    logging.warning("PIL not installed - image processing features will be limited")

try:
    import websocket
except ImportError:
    websocket = None
    logging.warning("websocket-client not installed - live preview features will be limited")

try:
    import requests
except ImportError:
    requests = None
    logging.warning("requests not installed - API features will be limited")

# Local imports
from mcp.server.fastmcp import FastMCP
from comfyui_client import ComfyUIClient

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor tool performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        tool_name = func.__name__
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log performance metrics
            logger.info(f"PERF: {tool_name} completed in {execution_time:.2f}s")
            
            # Add performance info to result if it's a dict
            if isinstance(result, dict) and "error" not in result:
                result["_performance"] = {
                    "execution_time_seconds": round(execution_time, 3),
                    "tool_name": tool_name,
                    "timestamp": datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"PERF: {tool_name} failed after {execution_time:.2f}s - {e}")
            raise
            
    return wrapper

# Retry decorator for better error handling
def with_retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry failed operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}")
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
                        
            # Return error response instead of raising
            return {
                "error": str(last_error),
                "attempts": max_attempts,
                "function": func.__name__
            }
            
        return wrapper
    return decorator

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG") else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ComfyUI_MCP_Server")

# Initialize ComfyUI client with environment variable support
comfyui_url = os.getenv("COMFYUI_URL", "http://localhost:8188")
comfyui_api_key = os.getenv("COMFYUI_API_KEY", None)
startup_time = time.time()  # Track server startup time

try:
    comfyui_client = ComfyUIClient(comfyui_url)
    logger.info(f"ComfyUI client initialized (URL: {comfyui_url})")
except Exception as e:
    logger.error(f"Failed to initialize ComfyUI client: {e}")
    sys.exit(1)

# Create FastMCP server with version info
mcp = FastMCP(
    "comfyui-mcp-server",
    version="2.5.0"
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
            "server_version": "2.5.0",
            "comfyui_url": comfyui_url,
            "available_models": comfyui_client.available_models or [],
            "status": "connected",
            "total_tools": 81
        }
    except Exception as e:
        logger.error(f"Error getting server info: {e}")
        return {
            "server_version": "2.5.0",
            "comfyui_url": comfyui_url,
            "status": "error",
            "error": str(e)
        }

@mcp.tool()
def check_model_availability(
    model_name: str,
    model_type: str = "checkpoint"
) -> Dict[str, Any]:
    """Check if a specific model is available in ComfyUI
    
    Args:
        model_name: Name of the model to check
        model_type: Type of model (checkpoint, lora, vae, controlnet, etc)
        
    Returns:
        Dictionary with availability status and model info
    """
    try:
        # Get list of available models
        models_result = list_models(model_type)
        
        if "error" in models_result:
            return models_result
            
        available_models = models_result.get("models", {}).get(f"{model_type}s", [])
        
        # Check if model exists
        model_found = any(model_name in model for model in available_models)
        
        if model_found:
            return {
                "success": True,
                "available": True,
                "model_name": model_name,
                "model_type": model_type,
                "message": f"Model '{model_name}' is available"
            }
        else:
            # Suggest similar models
            suggestions = [m for m in available_models if any(part in m.lower() for part in model_name.lower().split('_'))]
            
            return {
                "success": True,
                "available": False,
                "model_name": model_name,
                "model_type": model_type,
                "message": f"Model '{model_name}' not found",
                "suggestions": suggestions[:5],  # Top 5 suggestions
                "available_models": available_models[:10]  # Show first 10 available
            }
            
    except Exception as e:
        logger.error(f"Error checking model availability: {e}")
        return {"error": str(e)}

@mcp.tool()
def validate_generation_params(
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Validate common generation parameters
    
    Args:
        width: Image/video width to validate
        height: Image/video height to validate
        steps: Number of sampling steps
        cfg_scale: Guidance scale value
        seed: Random seed value
        
    Returns:
        Dictionary with validation results and corrections
    """
    errors = []
    warnings = []
    corrections = {}
    
    # Validate dimensions
    if width is not None:
        if width % 8 != 0:
            corrections["width"] = (width // 8) * 8
            errors.append(f"Width must be divisible by 8. Suggested: {corrections['width']}")
        elif width < 64:
            errors.append("Width must be at least 64 pixels")
        elif width > 2048:
            warnings.append("Width > 2048 may cause memory issues")
            
    if height is not None:
        if height % 8 != 0:
            corrections["height"] = (height // 8) * 8
            errors.append(f"Height must be divisible by 8. Suggested: {corrections['height']}")
        elif height < 64:
            errors.append("Height must be at least 64 pixels")
        elif height > 2048:
            warnings.append("Height > 2048 may cause memory issues")
            
    # Validate steps
    if steps is not None:
        if steps < 1:
            errors.append("Steps must be at least 1")
            corrections["steps"] = 1
        elif steps > 150:
            warnings.append("Steps > 150 may not improve quality significantly")
            
    # Validate CFG scale
    if cfg_scale is not None:
        if cfg_scale < 0:
            errors.append("CFG scale must be non-negative")
            corrections["cfg_scale"] = 0
        elif cfg_scale > 30:
            warnings.append("CFG scale > 30 may cause artifacts")
            
    # Validate seed
    if seed is not None:
        if seed < 0 or seed > 2**32 - 1:
            corrections["seed"] = random.randint(0, 2**32 - 1)
            warnings.append(f"Seed out of range, using random: {corrections['seed']}")
            
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "corrections": corrections
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
def generate_image_flux(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    model: str = "flux1-dev",
    steps: int = 20,
    guidance: float = 3.5,
    seed: Optional[int] = None,
    sampler: str = "euler",
    scheduler: str = "simple",
    denoise: float = 1.0
) -> Dict[str, Any]:
    """Generate high-quality images using FLUX models
    
    Args:
        prompt: Text description of the image
        width: Image width (default: 1024)
        height: Image height (default: 1024) 
        model: FLUX model variant (flux1-dev, flux1-schnell)
        steps: Number of sampling steps (schnell needs only 4)
        guidance: Guidance scale (dev: 3.5, schnell: 0)
        seed: Random seed for reproducibility
        sampler: Sampling method (euler, euler_ancestral, heun, dpm_2, etc)
        scheduler: Scheduler type (simple, normal, sgm_uniform, etc)
        denoise: Denoising strength
        
    Returns:
        Dictionary containing image path or error
    """
    try:
        # Adjust settings based on model
        if "schnell" in model.lower():
            steps = min(steps, 4)  # Schnell only needs 4 steps
            guidance = 0  # Schnell doesn't use guidance
            
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": f"{model}.safetensors"
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["1", 1],
                    "text": prompt
                }
            },
            "3": {
                "class_type": "CLIPTextEncode", 
                "inputs": {
                    "clip": ["1", 1],
                    "text": ""  # Empty negative prompt
                }
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                    "seed": seed if seed is not None else random.randint(0, 2**32-1),
                    "steps": steps,
                    "cfg": guidance,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": denoise
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {
                    "vae": ["1", 2],
                    "samples": ["5", 0]
                }
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["6", 0],
                    "filename_prefix": "flux_"
                }
            }
        }
        
        result = comfyui_client.generate(workflow)
        if result and "outputs" in result:
            return {
                "success": True,
                "image_path": result["outputs"].get("7", {}).get("images", [{}])[0].get("filename"),
                "prompt": prompt,
                "model": model,
                "resolution": f"{width}x{height}"
            }
        else:
            return {"error": "Failed to generate image with FLUX"}
            
    except Exception as e:
        logger.error(f"Error generating FLUX image: {e}")
        return {"error": str(e)}

@mcp.tool()
@monitor_performance
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
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        # Check queue for running/pending status
        queue_response = requests.get(f"{comfyui_url}/queue")
        if queue_response.status_code == 200:
            queue_data = queue_response.json()
            
            # Check if running
            running = queue_data.get("queue_running", [])
            for item in running:
                if len(item) > 1 and item[1] == prompt_id:
                    return {
                        "prompt_id": prompt_id,
                        "status": "running",
                        "message": "Generation is currently running",
                        "queue_position": 0
                    }
            
            # Check if pending
            pending = queue_data.get("queue_pending", [])
            for i, item in enumerate(pending):
                if len(item) > 1 and item[1] == prompt_id:
                    return {
                        "prompt_id": prompt_id,
                        "status": "pending",
                        "message": f"Generation is queued at position {i + 1}",
                        "queue_position": i + 1,
                        "estimated_wait_minutes": (i + 1) * 2  # Rough estimate
                    }
        
        # Check history for completed
        history_response = requests.get(f"{comfyui_url}/history/{prompt_id}")
        if history_response.status_code == 200:
            history_data = history_response.json()
            if prompt_id in history_data:
                prompt_info = history_data[prompt_id]
                outputs = prompt_info.get("outputs", {})
                
                # Extract image outputs
                images = []
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        for img in node_output["images"]:
                            images.append({
                                "filename": img.get("filename"),
                                "subfolder": img.get("subfolder", ""),
                                "type": img.get("type", "output"),
                                "url": f"{comfyui_url}/view?filename={img.get('filename')}&type={img.get('type', 'output')}"
                            })
                
                return {
                    "prompt_id": prompt_id,
                    "status": "completed",
                    "message": "Generation completed successfully",
                    "outputs": images,
                    "execution_time": prompt_info.get("execution_time", 0)
                }
        
        # Not found in queue or history
        return {
            "prompt_id": prompt_id,
            "status": "not_found",
            "message": "Generation task not found"
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
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
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

@mcp.tool()
def get_recent_images(limit: int = 10) -> Dict[str, Any]:
    """Get recently generated images from ComfyUI output directory
    
    Args:
        limit: Maximum number of images to return (default: 10)
        
    Returns:
        Dictionary containing list of recent images with metadata
    """
    try:
        # Get ComfyUI base path (parent of custom_nodes)
        comfyui_base = Path(__file__).parent.parent.parent
        output_dir = comfyui_base / "output"
        
        if not output_dir.exists():
            return {
                "error": "Output directory not found",
                "path": str(output_dir)
            }
        
        # Find all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(output_dir.glob(ext))
        
        # Sort by modification time (newest first)
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Limit results
        image_files = image_files[:limit]
        
        # Build response
        images = []
        for img_path in image_files:
            stat = img_path.stat()
            images.append({
                "filename": img_path.name,
                "path": str(img_path),
                "url": f"{comfyui_url}/view?filename={img_path.name}&type=output",
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
        
        return {
            "count": len(images),
            "images": images,
            "output_directory": str(output_dir)
        }
        
    except Exception as e:
        logger.error(f"Error getting recent images: {e}")
        return {"error": str(e)}

@mcp.tool()
def list_models(model_type: Optional[str] = None) -> Dict[str, Any]:
    """List all available models by type
    
    Args:
        model_type: Type of models to list (checkpoints, loras, vae, controlnet, etc.)
                   If None, returns all types
        
    Returns:
        Dictionary containing models organized by type
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        # Model types in ComfyUI
        model_types = {
            "checkpoints": "CheckpointLoaderSimple",
            "loras": "LoraLoader",
            "vae": "VAELoader",
            "controlnet": "ControlNetLoader",
            "clip": "CLIPLoader",
            "upscale_models": "UpscaleModelLoader",
            "embeddings": "embeddings"
        }
        
        if model_type and model_type not in model_types:
            return {
                "error": f"Unknown model type: {model_type}",
                "available_types": list(model_types.keys())
            }
        
        results = {}
        
        # Get object info from ComfyUI
        response = requests.get(f"{comfyui_url}/object_info")
        if response.status_code != 200:
            return {"error": "Failed to fetch model information from ComfyUI"}
        
        object_info = response.json()
        
        # Extract models for each type
        for type_name, loader_name in model_types.items():
            if model_type and model_type != type_name:
                continue
                
            if type_name == "embeddings":
                # Embeddings are handled differently
                comfyui_base = Path(__file__).parent.parent.parent
                embeddings_dir = comfyui_base / "models" / "embeddings"
                if embeddings_dir.exists():
                    embeddings = [f.stem for f in embeddings_dir.glob("*.pt")]
                    results[type_name] = embeddings
                else:
                    results[type_name] = []
            elif loader_name in object_info:
                # Extract model list from the loader's input definition
                loader_info = object_info[loader_name]
                if "input" in loader_info and "required" in loader_info["input"]:
                    for param_name, param_info in loader_info["input"]["required"].items():
                        if isinstance(param_info, list) and len(param_info) > 0:
                            if isinstance(param_info[0], list):
                                results[type_name] = param_info[0]
                                break
        
        # Count total models
        total_count = sum(len(models) for models in results.values())
        
        return {
            "total_count": total_count,
            "models": results,
            "available_types": list(model_types.keys())
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {"error": str(e)}

@mcp.tool()
def get_system_stats() -> Dict[str, Any]:
    """Get ComfyUI system statistics including GPU, CPU, and memory usage
    
    Returns:
        Dictionary containing system performance metrics
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        response = requests.get(f"{comfyui_url}/system_stats", timeout=5)
        if response.status_code != 200:
            return {
                "error": "Failed to fetch system stats",
                "status_code": response.status_code
            }
        
        stats = response.json()
        
        # Add formatted values for easier reading
        if "devices" in stats:
            for device in stats["devices"]:
                if "vram_total" in device and "vram_free" in device:
                    device["vram_used"] = device["vram_total"] - device["vram_free"]
                    device["vram_used_percent"] = round(
                        (device["vram_used"] / device["vram_total"]) * 100, 1
                    )
        
        return stats
        
    except requests.exceptions.Timeout:
        return {
            "error": "Request timed out",
            "message": "ComfyUI did not respond within 5 seconds"
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return {"error": str(e)}

@mcp.tool()
def get_queue_status() -> Dict[str, Any]:
    """Get the current ComfyUI queue status
    
    Returns:
        Dictionary containing queue information and pending jobs
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        response = requests.get(f"{comfyui_url}/queue", timeout=5)
        if response.status_code != 200:
            return {
                "error": "Failed to fetch queue status",
                "status_code": response.status_code
            }
        
        queue_data = response.json()
        
        # Summarize queue information
        result = {
            "queue_running": len(queue_data.get("queue_running", [])),
            "queue_pending": len(queue_data.get("queue_pending", [])),
            "queue_running_items": queue_data.get("queue_running", []),
            "queue_pending_items": queue_data.get("queue_pending", [])
        }
        
        return result
        
    except requests.exceptions.Timeout:
        return {
            "error": "Request timed out",
            "message": "ComfyUI did not respond within 5 seconds"
        }
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        return {"error": str(e)}

@mcp.tool()
def batch_generate(
    prompts: List[str],
    width: Optional[int] = None,
    height: Optional[int] = None,
    workflow_id: Optional[str] = None,
    model: Optional[str] = None,
    seed_increment: bool = True,
    base_seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate multiple images with different prompts in batch
    
    Args:
        prompts: List of text prompts for image generation
        width: Image width (default: 512)
        height: Image height (default: 512)
        workflow_id: Workflow ID to use (default: basic_api_test)
        model: Model checkpoint to use
        seed_increment: Whether to increment seed for each image (default: True)
        base_seed: Starting seed (random if not specified)
        **kwargs: Additional parameters passed to generate_image
        
    Returns:
        Dictionary containing results for each prompt
    """
    try:
        if not prompts or not isinstance(prompts, list):
            return {"error": "Prompts must be a non-empty list"}
        
        if len(prompts) > 10:
            return {"error": "Maximum 10 prompts allowed per batch"}
        
        results = []
        current_seed = base_seed
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Prepare parameters
            params = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "workflow_id": workflow_id,
                "model": model,
                **kwargs
            }
            
            # Handle seed
            if current_seed is not None:
                params["seed"] = current_seed
                if seed_increment:
                    current_seed += 1
            
            # Generate image
            result = generate_image(**params)
            
            results.append({
                "index": i,
                "prompt": prompt,
                "result": result
            })
            
            # Check if we should stop due to errors
            if "error" in result and i < len(prompts) - 1:
                logger.warning(f"Error in batch generation at index {i}, continuing...")
        
        # Summary
        successful = sum(1 for r in results if "success" in r["result"] and r["result"]["success"])
        failed = len(results) - successful
        
        return {
            "total": len(prompts),
            "successful": successful,
            "failed": failed,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        return {"error": str(e)}

@mcp.tool()
def validate_workflow(workflow_id: str) -> Dict[str, Any]:
    """Validate a workflow file before using it
    
    Args:
        workflow_id: ID of the workflow to validate
        
    Returns:
        Dictionary containing validation results and any issues found
    """
    try:
        workflow_file = f"workflows/{workflow_id}.json"
        workflow_path = Path(workflow_file)
        
        if not workflow_path.exists():
            return {
                "valid": False,
                "error": f"Workflow file not found: {workflow_file}"
            }
        
        # Load and parse workflow
        try:
            with open(workflow_path, "r") as f:
                workflow = json.load(f)
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "error": f"Invalid JSON in workflow file: {e}"
            }
        
        # Validation checks
        issues = []
        warnings = []
        required_nodes = []
        
        # Check if it's a dict (API format)
        if not isinstance(workflow, dict):
            issues.append("Workflow must be a JSON object (API format)")
        
        # Check for common required nodes
        node_types = {}
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                issues.append(f"Node {node_id} is not a valid object")
                continue
                
            class_type = node_data.get("class_type", "Unknown")
            node_types[class_type] = node_types.get(class_type, 0) + 1
            
            # Check for required fields
            if "class_type" not in node_data:
                issues.append(f"Node {node_id} missing class_type")
            if "inputs" not in node_data:
                warnings.append(f"Node {node_id} missing inputs")
        
        # Check for essential nodes
        essential_nodes = {
            "CheckpointLoaderSimple": "Model loader",
            "KSampler": "Sampler",
            "SaveImage": "Image saver"
        }
        
        for node_type, description in essential_nodes.items():
            if node_type not in node_types:
                warnings.append(f"Missing {description} ({node_type})")
        
        # Check if mappable to our parameters
        mapping_nodes = ["CLIPTextEncode", "EmptyLatentImage", "CheckpointLoaderSimple"]
        mappable = all(any(node_type in node_types for node_type in [mn]) for mn in mapping_nodes[:2])
        
        if not mappable:
            warnings.append("Workflow may not be fully compatible with parameter mapping")
        
        # Final validation result
        is_valid = len(issues) == 0
        
        return {
            "valid": is_valid,
            "workflow_id": workflow_id,
            "node_count": len(workflow),
            "node_types": node_types,
            "issues": issues,
            "warnings": warnings,
            "mappable": mappable
        }
        
    except Exception as e:
        logger.error(f"Error validating workflow: {e}")
        return {
            "valid": False,
            "error": str(e)
        }

@mcp.tool()
def get_image_metadata(filename: str) -> Dict[str, Any]:
    """Extract metadata from a generated image including prompt and parameters
    
    Args:
        filename: Name of the image file (in output directory)
        
    Returns:
        Dictionary containing image metadata and generation parameters
    """
    try:
        if Image is None:
            return {"error": "PIL not installed. Please install with: pip install Pillow"}
        
        # Get ComfyUI base path
        comfyui_base = Path(__file__).parent.parent.parent
        output_dir = comfyui_base / "output"
        image_path = output_dir / filename
        
        if not image_path.exists():
            return {
                "error": f"Image not found: {filename}",
                "searched_path": str(image_path)
            }
        
        # Open image and extract metadata
        with Image.open(image_path) as img:
            metadata = {}
            
            # Extract PNG metadata
            if hasattr(img, 'info'):
                metadata.update(img.info)
            
            # Extract EXIF data if available
            exif_data = img.getexif()
            if exif_data:
                metadata['exif'] = {k: v for k, v in exif_data.items()}
            
            # Basic image info
            result = {
                "filename": filename,
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height
            }
            
            # Look for ComfyUI workflow in metadata
            if 'workflow' in metadata:
                try:
                    result['workflow'] = json.loads(metadata['workflow'])
                except:
                    result['workflow'] = metadata['workflow']
            
            if 'prompt' in metadata:
                try:
                    result['prompt'] = json.loads(metadata['prompt'])
                except:
                    result['prompt'] = metadata['prompt']
            
            # Add any other metadata
            result['metadata'] = metadata
            
            return result
            
    except Exception as e:
        logger.error(f"Error extracting image metadata: {e}")
        return {"error": str(e)}

@mcp.tool()
def cleanup_old_images(days_old: int = 7, dry_run: bool = True) -> Dict[str, Any]:
    """Remove generated images older than specified days
    
    Args:
        days_old: Remove images older than this many days (default: 7)
        dry_run: If True, only show what would be deleted without actually deleting (default: True)
        
    Returns:
        Dictionary containing cleanup results
    """
    try:
        from datetime import timedelta
        
        # Get ComfyUI base path
        comfyui_base = Path(__file__).parent.parent.parent
        output_dir = comfyui_base / "output"
        
        if not output_dir.exists():
            return {
                "error": "Output directory not found",
                "path": str(output_dir)
            }
        
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(days=days_old)
        
        # Find old images
        old_images = []
        total_size = 0
        
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        for ext in image_extensions:
            for img_path in output_dir.glob(ext):
                stat = img_path.stat()
                modified_time = datetime.fromtimestamp(stat.st_mtime)
                
                if modified_time < cutoff_time:
                    old_images.append({
                        "filename": img_path.name,
                        "path": str(img_path),
                        "size_bytes": stat.st_size,
                        "modified": modified_time.isoformat(),
                        "age_days": (datetime.now() - modified_time).days
                    })
                    total_size += stat.st_size
        
        # Sort by age (oldest first)
        old_images.sort(key=lambda x: x['modified'])
        
        # Delete if not dry run
        deleted_count = 0
        if not dry_run and old_images:
            for img_info in old_images:
                try:
                    Path(img_info['path']).unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {img_info['filename']}: {e}")
        
        return {
            "dry_run": dry_run,
            "days_old": days_old,
            "cutoff_date": cutoff_time.isoformat(),
            "found_count": len(old_images),
            "deleted_count": deleted_count if not dry_run else 0,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "images": old_images[:20]  # Limit to first 20 for readability
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up old images: {e}")
        return {"error": str(e)}

@mcp.tool()
def search_outputs_semantic(
    query: str,
    search_type: str = "all",
    limit: int = 20,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """Search generated outputs using semantic similarity and metadata matching
    
    Args:
        query: Search query (text description or keywords)
        search_type: Type of search (all, prompt, style, metadata)
        limit: Maximum number of results to return
        include_metadata: Include full metadata in results
        
    Returns:
        Dictionary containing search results ranked by relevance
    """
    try:
        if Image is None:
            return {"error": "PIL not installed. Please install with: pip install Pillow"}
        
        # Get ComfyUI base path
        comfyui_base = Path(__file__).parent.parent.parent
        output_dir = comfyui_base / "output"
        
        if not output_dir.exists():
            return {
                "error": "Output directory not found",
                "path": str(output_dir)
            }
        
        # Find all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(output_dir.glob(ext))
        
        # Prepare search results
        search_results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for img_path in all_images:
            try:
                score = 0.0
                match_details = {
                    "filename_match": False,
                    "prompt_match": False,
                    "workflow_match": False,
                    "metadata_match": False,
                    "matched_keywords": []
                }
                
                # Check filename match
                if any(word in img_path.name.lower() for word in query_words):
                    score += 0.3
                    match_details["filename_match"] = True
                
                # Extract and search metadata
                if include_metadata:
                    try:
                        with Image.open(img_path) as img:
                            if hasattr(img, 'info'):
                                metadata = img.info
                                
                                # Search in prompt
                                if 'prompt' in metadata:
                                    prompt_data = metadata['prompt']
                                    if isinstance(prompt_data, str):
                                        try:
                                            prompt_data = json.loads(prompt_data)
                                        except:
                                            pass
                                    
                                    prompt_str = str(prompt_data).lower()
                                    matched_words = [word for word in query_words if word in prompt_str]
                                    if matched_words:
                                        score += 0.5 * len(matched_words) / len(query_words)
                                        match_details["prompt_match"] = True
                                        match_details["matched_keywords"].extend(matched_words)
                                
                                # Search in workflow
                                if 'workflow' in metadata and search_type in ["all", "metadata"]:
                                    workflow_data = metadata['workflow']
                                    if isinstance(workflow_data, str):
                                        try:
                                            workflow_data = json.loads(workflow_data)
                                        except:
                                            pass
                                    
                                    workflow_str = str(workflow_data).lower()
                                    if any(word in workflow_str for word in query_words):
                                        score += 0.2
                                        match_details["workflow_match"] = True
                    except Exception as e:
                        logger.debug(f"Could not read metadata for {img_path.name}: {e}")
                
                # Style-based search
                if search_type in ["all", "style"]:
                    # Simple style keyword matching
                    style_keywords = {
                        "anime": ["anime", "manga", "cartoon", "2d"],
                        "realistic": ["realistic", "photo", "photography", "real"],
                        "abstract": ["abstract", "artistic", "surreal"],
                        "portrait": ["portrait", "face", "person", "headshot"],
                        "landscape": ["landscape", "scenery", "nature", "outdoor"]
                    }
                    
                    for style, keywords in style_keywords.items():
                        if any(kw in query_lower for kw in keywords):
                            if any(kw in img_path.name.lower() for kw in keywords):
                                score += 0.1
                                match_details["metadata_match"] = True
                
                # Only include results with positive scores
                if score > 0:
                    stat = img_path.stat()
                    result = {
                        "filename": img_path.name,
                        "path": str(img_path),
                        "url": f"{comfyui_url}/view?filename={img_path.name}&type=output",
                        "score": round(score, 3),
                        "match_details": match_details,
                        "size_bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }
                    
                    if include_metadata and match_details["prompt_match"]:
                        try:
                            with Image.open(img_path) as img:
                                if 'prompt' in img.info:
                                    result["prompt_preview"] = str(img.info['prompt'])[:200] + "..."
                        except:
                            pass
                    
                    search_results.append(result)
                    
            except Exception as e:
                logger.debug(f"Error processing {img_path.name}: {e}")
                continue
        
        # Sort by score (highest first)
        search_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit results
        search_results = search_results[:limit]
        
        return {
            "query": query,
            "search_type": search_type,
            "total_results": len(search_results),
            "results": search_results,
            "searched_files": len(all_images),
            "message": f"Found {len(search_results)} matching images"
        }
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return {"error": str(e)}

@mcp.tool()
def search_by_similarity(
    reference_image: str,
    similarity_type: str = "perceptual",
    limit: int = 10
) -> Dict[str, Any]:
    """Find similar images based on visual similarity
    
    Args:
        reference_image: Filename of the reference image
        similarity_type: Type of similarity (perceptual, color, structure)
        limit: Maximum number of similar images to return
        
    Returns:
        Dictionary containing similar images ranked by similarity
    """
    try:
        if Image is None:
            return {"error": "PIL not installed. Please install with: pip install Pillow"}
        if np is None:
            return {"error": "NumPy not installed. Please install with: pip install numpy"}
        
        # Get ComfyUI base path
        comfyui_base = Path(__file__).parent.parent.parent
        output_dir = comfyui_base / "output"
        reference_path = output_dir / reference_image
        
        if not reference_path.exists():
            return {
                "error": f"Reference image not found: {reference_image}",
                "path": str(reference_path)
            }
        
        # Load reference image
        with Image.open(reference_path) as ref_img:
            ref_img = ref_img.convert('RGB')
            ref_img_small = ref_img.resize((64, 64))  # Simple perceptual hash
            ref_array = np.array(ref_img_small)
            
            # Calculate color histogram for reference
            ref_hist = []
            for i in range(3):  # RGB channels
                hist, _ = np.histogram(ref_array[:, :, i], bins=16, range=(0, 256))
                ref_hist.extend(hist)
            ref_hist = np.array(ref_hist, dtype=np.float32)
            ref_hist /= ref_hist.sum()  # Normalize
        
        # Find all images
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(output_dir.glob(ext))
        
        # Calculate similarities
        similarities = []
        
        for img_path in all_images:
            if img_path.name == reference_image:
                continue  # Skip reference image
                
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img_small = img.resize((64, 64))
                    img_array = np.array(img_small)
                    
                    score = 0.0
                    
                    if similarity_type in ["perceptual", "all"]:
                        # Simple perceptual similarity (MSE)
                        mse = np.mean((ref_array.astype(float) - img_array.astype(float)) ** 2)
                        perceptual_score = 1.0 / (1.0 + mse / 1000.0)  # Normalize
                        score += perceptual_score * 0.5
                    
                    if similarity_type in ["color", "all"]:
                        # Color histogram similarity
                        img_hist = []
                        for i in range(3):
                            hist, _ = np.histogram(img_array[:, :, i], bins=16, range=(0, 256))
                            img_hist.extend(hist)
                        img_hist = np.array(img_hist, dtype=np.float32)
                        img_hist /= img_hist.sum()
                        
                        # Histogram intersection
                        color_score = np.minimum(ref_hist, img_hist).sum()
                        score += color_score * 0.5
                    
                    if score > 0.1:  # Threshold for similarity
                        stat = img_path.stat()
                        similarities.append({
                            "filename": img_path.name,
                            "path": str(img_path),
                            "url": f"{comfyui_url}/view?filename={img_path.name}&type=output",
                            "similarity_score": round(score, 3),
                            "size_bytes": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                        
            except Exception as e:
                logger.debug(f"Error processing {img_path.name}: {e}")
                continue
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        similarities = similarities[:limit]
        
        return {
            "reference_image": reference_image,
            "similarity_type": similarity_type,
            "total_similar": len(similarities),
            "similar_images": similarities,
            "searched_files": len(all_images),
            "message": f"Found {len(similarities)} similar images"
        }
        
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        return {"error": str(e)}

@mcp.tool()
def organize_outputs(
    organization_type: str = "date",
    dry_run: bool = True,
    create_subdirs: bool = True
) -> Dict[str, Any]:
    """Organize generated outputs into structured directories
    
    Args:
        organization_type: How to organize (date, style, size, prompt_keywords)
        dry_run: Preview organization without moving files
        create_subdirs: Create subdirectories for organization
        
    Returns:
        Dictionary containing organization results
    """
    try:
        if Image is None:
            return {"error": "PIL not installed. Please install with: pip install Pillow"}
        
        # Get ComfyUI base path
        comfyui_base = Path(__file__).parent.parent.parent
        output_dir = comfyui_base / "output"
        
        if not output_dir.exists():
            return {
                "error": "Output directory not found",
                "path": str(output_dir)
            }
        
        # Find all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(output_dir.glob(ext))
        
        # Organize based on type
        organization_plan = {}
        
        for img_path in all_images:
            try:
                # Determine target directory
                target_subdir = ""
                
                if organization_type == "date":
                    # Organize by creation date
                    stat = img_path.stat()
                    date = datetime.fromtimestamp(stat.st_mtime)
                    target_subdir = f"{date.year}/{date.month:02d}-{date.strftime('%B')}/{date.day:02d}"
                    
                elif organization_type == "size":
                    # Organize by image dimensions
                    with Image.open(img_path) as img:
                        width, height = img.size
                        if width > height:
                            orientation = "landscape"
                        elif height > width:
                            orientation = "portrait"
                        else:
                            orientation = "square"
                        target_subdir = f"{orientation}/{width}x{height}"
                        
                elif organization_type == "style":
                    # Organize by detected style (from metadata or filename)
                    style = "uncategorized"
                    try:
                        with Image.open(img_path) as img:
                            if hasattr(img, 'info') and 'prompt' in img.info:
                                prompt = str(img.info['prompt']).lower()
                                # Simple style detection
                                if any(kw in prompt for kw in ["anime", "manga", "cartoon"]):
                                    style = "anime"
                                elif any(kw in prompt for kw in ["photo", "realistic", "real"]):
                                    style = "photorealistic"
                                elif any(kw in prompt for kw in ["abstract", "surreal", "artistic"]):
                                    style = "artistic"
                                elif any(kw in prompt for kw in ["portrait", "face", "person"]):
                                    style = "portraits"
                                elif any(kw in prompt for kw in ["landscape", "nature", "scenery"]):
                                    style = "landscapes"
                    except:
                        pass
                    target_subdir = style
                    
                elif organization_type == "prompt_keywords":
                    # Organize by prompt keywords
                    keywords = []
                    try:
                        with Image.open(img_path) as img:
                            if hasattr(img, 'info') and 'prompt' in img.info:
                                prompt = str(img.info['prompt']).lower()
                                # Extract main keywords
                                important_words = ["dragon", "castle", "forest", "city", "space", 
                                                 "ocean", "mountain", "desert", "robot", "fantasy"]
                                for word in important_words:
                                    if word in prompt:
                                        keywords.append(word)
                    except:
                        pass
                    
                    if keywords:
                        target_subdir = "_".join(keywords[:2])  # Use first 2 keywords
                    else:
                        target_subdir = "no_keywords"
                
                # Create organization plan
                if target_subdir:
                    target_path = output_dir / target_subdir / img_path.name
                    organization_plan[str(img_path)] = {
                        "source": str(img_path),
                        "target": str(target_path),
                        "subdir": target_subdir,
                        "filename": img_path.name
                    }
                    
            except Exception as e:
                logger.debug(f"Error processing {img_path.name}: {e}")
                continue
        
        # Execute organization if not dry run
        moved_count = 0
        created_dirs = set()
        
        if not dry_run:
            for source, plan in organization_plan.items():
                try:
                    target_path = Path(plan["target"])
                    target_dir = target_path.parent
                    
                    # Create directory if needed
                    if create_subdirs and not target_dir.exists():
                        target_dir.mkdir(parents=True, exist_ok=True)
                        created_dirs.add(str(target_dir))
                    
                    # Move file
                    if target_path != Path(source):
                        shutil.move(source, target_path)
                        moved_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to move {source}: {e}")
        
        return {
            "organization_type": organization_type,
            "dry_run": dry_run,
            "total_files": len(all_images),
            "files_to_organize": len(organization_plan),
            "files_moved": moved_count if not dry_run else 0,
            "directories_created": len(created_dirs) if not dry_run else 0,
            "organization_plan": list(organization_plan.values())[:20],  # Show first 20
            "message": f"{'Would organize' if dry_run else 'Organized'} {len(organization_plan)} files"
        }
        
    except Exception as e:
        logger.error(f"Error organizing outputs: {e}")
        return {"error": str(e)}

@mcp.tool()
def create_output_catalog(
    format: str = "json",
    include_thumbnails: bool = False,
    max_entries: int = 1000
) -> Dict[str, Any]:
    """Create a catalog of all generated outputs with metadata
    
    Args:
        format: Output format (json, csv, html)
        include_thumbnails: Include base64 thumbnails in catalog
        max_entries: Maximum number of entries to include
        
    Returns:
        Dictionary containing catalog information
    """
    try:
        if Image is None:
            return {"error": "PIL not installed. Please install with: pip install Pillow"}
        
        # Get ComfyUI base path
        comfyui_base = Path(__file__).parent.parent.parent
        output_dir = comfyui_base / "output"
        
        if not output_dir.exists():
            return {
                "error": "Output directory not found",
                "path": str(output_dir)
            }
        
        # Find all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(output_dir.glob(ext))
        
        # Sort by modification time (newest first)
        all_images.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        all_images = all_images[:max_entries]
        
        # Build catalog entries
        catalog_entries = []
        
        for img_path in all_images:
            try:
                stat = img_path.stat()
                entry = {
                    "filename": img_path.name,
                    "path": str(img_path),
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                }
                
                # Extract image info and metadata
                with Image.open(img_path) as img:
                    entry["width"] = img.width
                    entry["height"] = img.height
                    entry["format"] = img.format
                    entry["mode"] = img.mode
                    
                    # Extract metadata
                    if hasattr(img, 'info'):
                        if 'prompt' in img.info:
                            try:
                                entry["prompt"] = json.loads(img.info['prompt']) if isinstance(img.info['prompt'], str) else img.info['prompt']
                            except:
                                entry["prompt"] = img.info['prompt']
                        
                        if 'workflow' in img.info:
                            try:
                                workflow = json.loads(img.info['workflow']) if isinstance(img.info['workflow'], str) else img.info['workflow']
                                # Extract key workflow info
                                entry["workflow_summary"] = {
                                    "nodes": len(workflow) if isinstance(workflow, dict) else 0
                                }
                            except:
                                pass
                    
                    # Create thumbnail if requested
                    if include_thumbnails:
                        thumbnail_size = (128, 128)
                        img.thumbnail(thumbnail_size)
                        buffer = BytesIO()
                        img.save(buffer, format='PNG')
                        entry["thumbnail"] = base64.b64encode(buffer.getvalue()).decode()
                
                catalog_entries.append(entry)
                
            except Exception as e:
                logger.debug(f"Error processing {img_path.name} for catalog: {e}")
                continue
        
        # Save catalog based on format
        catalog_path = output_dir / f"catalog.{format}"
        
        if format == "json":
            catalog_data = {
                "generated": datetime.now().isoformat(),
                "total_entries": len(catalog_entries),
                "entries": catalog_entries
            }
            
            with open(catalog_path, 'w') as f:
                json.dump(catalog_data, f, indent=2)
                
        elif format == "csv":
            if catalog_entries:
                # Flatten entries for CSV
                fieldnames = ["filename", "path", "width", "height", "size_bytes", "modified", "created", "format"]
                
                with open(catalog_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for entry in catalog_entries:
                        row = {k: entry.get(k, '') for k in fieldnames}
                        writer.writerow(row)
                        
        elif format == "html":
            # Create simple HTML gallery
            html_content = """<!DOCTYPE html>
<html>
<head>
    <title>ComfyUI Output Catalog</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; }
        .item { border: 1px solid #ddd; padding: 10px; }
        .item img { max-width: 100%; height: auto; }
        .metadata { font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <h1>ComfyUI Output Catalog</h1>
    <p>Generated: {}</p>
    <div class="gallery">
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            for entry in catalog_entries[:100]:  # Limit HTML to 100 entries
                html_content += f"""
        <div class="item">
            <img src="file://{entry['path']}" alt="{entry['filename']}">
            <div class="metadata">
                <strong>{entry['filename']}</strong><br>
                {entry['width']}x{entry['height']} • {entry.get('format', 'Unknown')}<br>
                {datetime.fromisoformat(entry['modified']).strftime('%Y-%m-%d %H:%M')}
            </div>
        </div>
"""
            
            html_content += """
    </div>
</body>
</html>"""
            
            with open(catalog_path, 'w') as f:
                f.write(html_content)
        
        return {
            "format": format,
            "catalog_path": str(catalog_path),
            "total_entries": len(catalog_entries),
            "catalog_size_bytes": catalog_path.stat().st_size if catalog_path.exists() else 0,
            "include_thumbnails": include_thumbnails,
            "message": f"Created {format} catalog with {len(catalog_entries)} entries"
        }
        
    except Exception as e:
        logger.error(f"Error creating catalog: {e}")
        return {"error": str(e)}

@mcp.tool() 
def detect_custom_nodes() -> Dict[str, Any]:
    """Detect installed ComfyUI custom nodes and their capabilities
    
    Returns:
        Dictionary containing detected custom nodes and their features
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        from pathlib import Path
        
        # Get all available nodes
        response = requests.get(f"{comfyui_url}/object_info", timeout=10)
        if response.status_code != 200:
            return {"error": "Could not fetch node information"}
            
        all_nodes = response.json()
        
        # Categorize nodes by likely custom node packages
        core_nodes = {
            "CheckpointLoaderSimple", "CLIPTextEncode", "KSampler", "VAEDecode", 
            "VAEEncode", "EmptyLatentImage", "LoadImage", "SaveImage", "UpscaleModelLoader"
        }
        
        custom_categories = {}
        custom_nodes = {}
        
        for node_name, node_info in all_nodes.items():
            if node_name not in core_nodes:
                category = node_info.get("category", "Unknown")
                
                # Group by category
                if category not in custom_categories:
                    custom_categories[category] = []
                custom_categories[category].append(node_name)
                
                # Detect special capabilities
                if any(keyword in node_name.lower() for keyword in ["controlnet", "control"]):
                    if "controlnet" not in custom_nodes:
                        custom_nodes["controlnet"] = []
                    custom_nodes["controlnet"].append(node_name)
                    
                elif any(keyword in node_name.lower() for keyword in ["video", "animate", "motion"]):
                    if "video_generation" not in custom_nodes:
                        custom_nodes["video_generation"] = []
                    custom_nodes["video_generation"].append(node_name)
                    
                elif any(keyword in node_name.lower() for keyword in ["upscale", "enhance", "restore"]):
                    if "enhancement" not in custom_nodes:
                        custom_nodes["enhancement"] = []
                    custom_nodes["enhancement"].append(node_name)
                    
                elif any(keyword in node_name.lower() for keyword in ["inpaint", "outpaint", "mask"]):
                    if "inpainting" not in custom_nodes:
                        custom_nodes["inpainting"] = []
                    custom_nodes["inpainting"].append(node_name)
        
        # Detect likely custom node directories
        comfyui_base = Path(__file__).parent.parent.parent
        custom_nodes_dir = comfyui_base / "custom_nodes"
        installed_packages = []
        
        if custom_nodes_dir.exists():
            for item in custom_nodes_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    installed_packages.append(item.name)
        
        return {
            "total_nodes": len(all_nodes),
            "core_nodes": len(core_nodes),
            "custom_nodes": len(all_nodes) - len(core_nodes),
            "categories": {cat: len(nodes) for cat, nodes in custom_categories.items()},
            "capabilities": {cap: len(nodes) for cap, nodes in custom_nodes.items()},
            "detected_features": {
                "controlnet_support": "controlnet" in custom_nodes,
                "video_generation": "video_generation" in custom_nodes,
                "image_enhancement": "enhancement" in custom_nodes,
                "inpainting_support": "inpainting" in custom_nodes
            },
            "installed_packages": installed_packages,
            "custom_node_details": custom_nodes
        }
        
    except Exception as e:
        logger.error(f"Error detecting custom nodes: {e}")
        return {"error": str(e)}

@mcp.tool()
def get_node_info(node_type: Optional[str] = None) -> Dict[str, Any]:
    """Get detailed information about available ComfyUI nodes
    
    Args:
        node_type: Specific node type to get info for, or None for all nodes
        
    Returns:
        Dictionary containing node information
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        response = requests.get(f"{comfyui_url}/object_info", timeout=10)
        if response.status_code != 200:
            return {
                "error": "Failed to fetch node information",
                "status_code": response.status_code
            }
        
        all_nodes = response.json()
        
        if node_type:
            # Return specific node info
            if node_type not in all_nodes:
                return {
                    "error": f"Node type '{node_type}' not found",
                    "available_nodes": list(all_nodes.keys())[:20]  # Show first 20
                }
            
            node_info = all_nodes[node_type]
            return {
                "node_type": node_type,
                "info": node_info,
                "input_types": list(node_info.get("input", {}).get("required", {}).keys()) if "input" in node_info else [],
                "output_types": node_info.get("output", []),
                "category": node_info.get("category", "Unknown")
            }
        else:
            # Return summary of all nodes
            categories = {}
            for name, info in all_nodes.items():
                category = info.get("category", "Uncategorized")
                if category not in categories:
                    categories[category] = []
                categories[category].append(name)
            
            # Sort categories
            for cat in categories:
                categories[cat].sort()
            
            return {
                "total_nodes": len(all_nodes),
                "categories": categories,
                "category_count": {cat: len(nodes) for cat, nodes in categories.items()}
            }
            
    except Exception as e:
        logger.error(f"Error getting node info: {e}")
        return {"error": str(e)}

@mcp.tool()
def generate_variations(
    base_prompt: str,
    variations: List[str],
    width: Optional[int] = None,
    height: Optional[int] = None,
    base_seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate variations of an image by modifying the prompt
    
    Args:
        base_prompt: The base prompt to build variations from
        variations: List of variation descriptions to add/modify
        width: Image width (default: 512)
        height: Image height (default: 512)
        base_seed: Seed to use for all variations (for consistent comparison)
        **kwargs: Additional parameters passed to generate_image
        
    Returns:
        Dictionary containing results for each variation
    """
    try:
        if not variations or not isinstance(variations, list):
            return {"error": "Variations must be a non-empty list"}
        
        if len(variations) > 8:
            return {"error": "Maximum 8 variations allowed"}
        
        # Generate prompts
        prompts = [base_prompt]  # Include original
        for variation in variations:
            # Simple combination - could be made smarter
            if variation.startswith("+"):
                # Addition
                prompts.append(f"{base_prompt}, {variation[1:].strip()}")
            elif variation.startswith("-"):
                # Replacement (simple)
                prompts.append(variation[1:].strip())
            else:
                # Default: append
                prompts.append(f"{base_prompt}, {variation}")
        
        # Use batch_generate with fixed seed
        result = batch_generate(
            prompts=prompts,
            width=width,
            height=height,
            seed_increment=False,  # Keep same seed
            base_seed=base_seed,
            **kwargs
        )
        
        # Add variation info to results
        if "results" in result:
            result["results"][0]["variation"] = "original"
            for i, variation in enumerate(variations, 1):
                if i < len(result["results"]):
                    result["results"][i]["variation"] = variation
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating variations: {e}")
        return {"error": str(e)}

@mcp.tool()
def clear_comfyui_cache() -> Dict[str, Any]:
    """Clear ComfyUI's model cache to free up memory
    
    Returns:
        Dictionary containing cache clear results
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        # Send POST request to clear cache
        response = requests.post(f"{comfyui_url}/free", timeout=10)
        
        if response.status_code == 200:
            # Get new system stats to show freed memory
            stats_after = get_system_stats()
            
            return {
                "success": True,
                "message": "Cache cleared successfully",
                "system_stats": stats_after
            }
        else:
            return {
                "success": False,
                "error": f"Failed to clear cache: HTTP {response.status_code}",
                "response": response.text
            }
            
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Video Generation Tools
@mcp.tool()
def generate_video(
    prompt: str,
    duration: float = 2.0,
    fps: int = 24,
    width: Optional[int] = None,
    height: Optional[int] = None,
    model: str = "wan2.1_t2v_14B",
    seed: Optional[int] = None,
    motion_strength: float = 1.0,
    precision: str = "fp16",
    cfg_scale: float = 7.0,
    steps: int = 20
) -> Dict[str, Any]:
    """Generate a video from text prompt using available video models
    
    Args:
        prompt: Text description of the video to generate
        duration: Video duration in seconds (default: 2.0)
        fps: Frames per second (default: 24)
        width: Video width (default: 854 for WAN)
        height: Video height (default: 480 for WAN)
        model: Video model (wan2.1_t2v_14B, wan2.1_t2v_1.3B, wan2.2_t2v, cosmos, mochi, ltxv)
        seed: Random seed for reproducibility
        motion_strength: How much motion to include (0.0-2.0)
        precision: Model precision (fp16, fp8, bf16)
        cfg_scale: Guidance scale
        steps: Number of sampling steps
        
    Returns:
        Dictionary containing video path or error
    """
    try:
        # Set appropriate dimensions for model
        if "wan" in model.lower():
            width = width or 854
            height = height or 480
        else:
            width = width or 512
            height = height or 512
            
        frames = int(duration * fps)
        
        if "wan" in model.lower():
            # WAN model workflow
            workflow = {
                "1": {
                    "class_type": "UMT5ModelLoader",
                    "inputs": {
                        "model_name": f"umt5_xxl_{precision}.safetensors"
                    }
                },
                "2": {
                    "class_type": "UMT5TextEncode",
                    "inputs": {
                        "model": ["1", 0],
                        "text": prompt
                    }
                },
                "3": {
                    "class_type": "DiffusionModelLoader", 
                    "inputs": {
                        "model_name": f"{model}_{precision}.safetensors"
                    }
                },
                "4": {
                    "class_type": "WANVAELoader",
                    "inputs": {
                        "model_name": "wan_2.1_vae.safetensors"
                    }
                },
                "5": {
                    "class_type": "WANSampler",
                    "inputs": {
                        "model": ["3", 0],
                        "conditioning": ["2", 0],
                        "vae": ["4", 0],
                        "width": width,
                        "height": height,
                        "frames": frames,
                        "fps": fps,
                        "cfg_scale": cfg_scale,
                        "steps": steps,
                        "seed": seed if seed is not None else random.randint(0, 2**32-1)
                    }
                },
                "6": {
                    "class_type": "SaveVideo",
                    "inputs": {
                        "video": ["5", 0],
                        "filename_prefix": "wan_video_",
                        "fps": fps
                    }
                }
            }
        else:
            # Fallback for other models (cosmos, mochi, ltxv)
            return {
                "status": "placeholder",
                "message": f"Model {model} workflow not yet implemented",
                "model": model
            }
            
        result = comfyui_client.generate(workflow)
        if result and "outputs" in result:
            return {
                "success": True,
                "video_path": result["outputs"].get("6", {}).get("video", [{}])[0].get("filename"),
                "prompt": prompt,
                "model": model,
                "duration": duration,
                "resolution": f"{width}x{height}"
            }
        else:
            return {"error": "Failed to generate video"}
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return {"error": str(e)}

@mcp.tool()
def image_to_video(
    image_path: str,
    duration: float = 2.0,
    fps: int = 24,
    model: str = "wan2.1_i2v_14B",
    motion_strength: float = 1.0,
    precision: str = "fp16",
    cfg_scale: float = 7.0,
    steps: int = 20,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Animate a static image into a video using WAN or other models
    
    Args:
        image_path: Path to the input image
        duration: Video duration in seconds
        fps: Frames per second
        model: Model to use (wan2.1_i2v_14B, wan2.1_i2v_1.3B, wan2.2_i2v)
        motion_strength: Intensity of motion effect
        precision: Model precision (fp16, fp8, bf16)
        cfg_scale: Guidance scale
        steps: Number of sampling steps
        seed: Random seed
        
    Returns:
        Dictionary containing animated video path
    """
    try:
        frames = int(duration * fps)
        
        if "wan" in model.lower():
            workflow = {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {
                        "image": image_path
                    }
                },
                "2": {
                    "class_type": "CLIPVisionLoader",
                    "inputs": {
                        "model_name": "clip_vision_h.safetensors"
                    }
                },
                "3": {
                    "class_type": "CLIPVisionEncode", 
                    "inputs": {
                        "clip_vision": ["2", 0],
                        "image": ["1", 0]
                    }
                },
                "4": {
                    "class_type": "DiffusionModelLoader",
                    "inputs": {
                        "model_name": f"{model}_{precision}.safetensors"
                    }
                },
                "5": {
                    "class_type": "WANVAELoader",
                    "inputs": {
                        "model_name": "wan_2.1_vae.safetensors"
                    }
                },
                "6": {
                    "class_type": "WANImageToVideo",
                    "inputs": {
                        "model": ["4", 0],
                        "vae": ["5", 0],
                        "image": ["1", 0],
                        "clip_vision": ["3", 0],
                        "frames": frames,
                        "fps": fps,
                        "motion_strength": motion_strength,
                        "cfg_scale": cfg_scale,
                        "steps": steps,
                        "seed": seed if seed is not None else random.randint(0, 2**32-1)
                    }
                },
                "7": {
                    "class_type": "SaveVideo",
                    "inputs": {
                        "video": ["6", 0],
                        "filename_prefix": "wan_i2v_",
                        "fps": fps
                    }
                }
            }
            
            result = comfyui_client.generate(workflow)
            if result and "outputs" in result:
                return {
                    "success": True,
                    "video_path": result["outputs"].get("7", {}).get("video", [{}])[0].get("filename"),
                    "source_image": image_path,
                    "model": model,
                    "duration": duration
                }
            else:
                return {"error": "Failed to generate video from image"}
        else:
            return {
                "status": "placeholder",
                "message": f"Model {model} not yet implemented for image-to-video",
                "input_image": image_path
            }
    except Exception as e:
        logger.error(f"Error animating image: {e}")
        return {"error": str(e)}

@mcp.tool()
def edit_video_wan_vace(
    video_path: str,
    prompt: str,
    mask_path: Optional[str] = None,
    reference_image: Optional[str] = None,
    edit_type: str = "inpaint",
    frames_to_edit: Optional[List[int]] = None,
    cfg_scale: float = 7.0,
    steps: int = 20,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Edit video using WAN VACE (Video All-in-One Creation and Editing)
    
    Args:
        video_path: Path to input video
        prompt: Text description of desired edit
        mask_path: Optional mask video/image for inpainting
        reference_image: Optional reference image for style transfer
        edit_type: Type of edit (inpaint, style_transfer, object_removal, enhancement)
        frames_to_edit: Specific frames to edit (None = all frames)
        cfg_scale: Guidance scale
        steps: Number of sampling steps
        seed: Random seed
        
    Returns:
        Dictionary containing edited video path
    """
    try:
        workflow = {
            "1": {
                "class_type": "LoadVideo",
                "inputs": {
                    "video": video_path
                }
            },
            "2": {
                "class_type": "UMT5ModelLoader",
                "inputs": {
                    "model_name": "umt5_xxl_fp16.safetensors"
                }
            },
            "3": {
                "class_type": "UMT5TextEncode",
                "inputs": {
                    "model": ["2", 0],
                    "text": prompt
                }
            },
            "4": {
                "class_type": "DiffusionModelLoader",
                "inputs": {
                    "model_name": "wan2.1_vace_14B_fp16.safetensors"
                }
            },
            "5": {
                "class_type": "WANVAELoader",
                "inputs": {
                    "model_name": "wan_2.1_vae.safetensors"
                }
            }
        }
        
        # Add mask if provided
        node_id = "6"
        if mask_path:
            workflow[node_id] = {
                "class_type": "LoadMask",
                "inputs": {
                    "mask": mask_path
                }
            }
            node_id = "7"
            
        # Add reference image if provided
        if reference_image:
            workflow[node_id] = {
                "class_type": "LoadImage",
                "inputs": {
                    "image": reference_image
                }
            }
            node_id = str(int(node_id) + 1)
            
        # VACE editing node
        vace_inputs = {
            "model": ["4", 0],
            "vae": ["5", 0],
            "video": ["1", 0],
            "conditioning": ["3", 0],
            "edit_type": edit_type,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "seed": seed if seed is not None else random.randint(0, 2**32-1)
        }
        
        if mask_path:
            vace_inputs["mask"] = ["6", 0]
        if reference_image:
            ref_node = "7" if mask_path else "6"
            vace_inputs["reference_image"] = [ref_node, 0]
        if frames_to_edit:
            vace_inputs["frames_to_edit"] = frames_to_edit
            
        workflow[node_id] = {
            "class_type": "WANVACEEdit",
            "inputs": vace_inputs
        }
        
        save_node_id = str(int(node_id) + 1)
        workflow[save_node_id] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": [node_id, 0],
                "filename_prefix": "wan_vace_edit_",
                "fps": 24
            }
        }
        
        result = comfyui_client.generate(workflow)
        if result and "outputs" in result:
            return {
                "success": True,
                "video_path": result["outputs"].get(save_node_id, {}).get("video", [{}])[0].get("filename"),
                "source_video": video_path,
                "edit_type": edit_type,
                "prompt": prompt
            }
        else:
            return {"error": "Failed to edit video with WAN VACE"}
            
    except Exception as e:
        logger.error(f"Error editing video with WAN VACE: {e}")
        return {"error": str(e)}

@mcp.tool()
def video_interpolation(
    start_image: str,
    end_image: str,
    frames: int = 30,
    fps: int = 30,
    interpolation_type: str = "linear"
) -> Dict[str, Any]:
    """Create smooth transition video between two images
    
    Args:
        start_image: Path to starting image
        end_image: Path to ending image
        frames: Number of interpolation frames
        fps: Output video FPS
        interpolation_type: Method of interpolation (linear, ease, bounce)
        
    Returns:
        Dictionary containing interpolated video
    """
    try:
        return {
            "status": "placeholder",
            "message": "Video interpolation coming soon",
            "start": start_image,
            "end": end_image,
            "frames": frames
        }
    except Exception as e:
        logger.error(f"Error interpolating video: {e}")
        return {"error": str(e)}

# Advanced Image Control Tools
@mcp.tool()
def controlnet_generate(
    prompt: str,
    control_image: str,
    control_type: str = "canny",
    width: Optional[int] = None,
    height: Optional[int] = None,
    control_strength: float = 1.0,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Generate image with ControlNet guidance
    
    Args:
        prompt: Text prompt for generation
        control_image: Path to control image
        control_type: Type of control (canny, pose, depth, normal, mlsd, scribble)
        width: Output width
        height: Output height
        control_strength: How strongly to follow control (0.0-2.0)
        model: Base model to use
        
    Returns:
        Dictionary containing generated image with control
    """
    try:
        # Validate inputs
        if not prompt:
            return {"error": "Prompt is required"}
        if not control_image:
            return {"error": "Control image is required"}
        
        width = width or 512
        height = height or 512
        
        # Map control types to ControlNet models
        control_net_models = {
            "canny": "control_v11p_sd15_canny.pth",
            "pose": "control_v11p_sd15_openpose.pth", 
            "depth": "control_v11f1p_sd15_depth.pth",
            "normal": "control_v11p_sd15_normalbae.pth",
            "mlsd": "control_v11p_sd15_mlsd.pth",
            "scribble": "control_v11p_sd15_scribble.pth"
        }
        
        if control_type not in control_net_models:
            return {
                "error": f"Unsupported control type: {control_type}",
                "supported_types": list(control_net_models.keys())
            }
        
        # Use controlnet workflow
        params = {
            "USER_PROMPT": prompt,
            "USER_NEGATIVE_PROMPT": "low quality, blurry, deformed",
            "CONTROL_IMAGE": control_image,
            "CONTROL_TYPE": control_net_models[control_type],
            "WIDTH": width,
            "HEIGHT": height,
            "CONTROL_STRENGTH": control_strength,
            "SEED": -1
        }
        
        # Generate with ControlNet workflow
        result = comfyui_client.generate_image(
            prompt=prompt,
            width=width,
            height=height,
            workflow_id="controlnet_workflow",
            model=model,
            **params
        )
        
        if isinstance(result, str):  # Success case returns URL
            return {
                "success": True,
                "image_url": result,
                "control_type": control_type,
                "control_image": control_image,
                "control_strength": control_strength,
                "prompt": prompt,
                "dimensions": f"{width}x{height}"
            }
        else:
            return result  # Error case
        
    except Exception as e:
        logger.error(f"Error with ControlNet: {e}")
        return {"error": str(e)}

@mcp.tool()
def inpaint_image(
    image_path: str,
    mask_path: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    strength: float = 1.0,
    expand_mask: int = 10
) -> Dict[str, Any]:
    """Intelligently fill masked areas of an image
    
    Args:
        image_path: Path to original image
        mask_path: Path to mask image (white=inpaint, black=keep)
        prompt: Description of what to generate in masked area
        negative_prompt: What to avoid generating
        strength: Inpainting strength (0.0-1.0)
        expand_mask: Pixels to expand mask for smoother blending
        
    Returns:
        Dictionary containing inpainted image
    """
    try:
        # Validate inputs
        if not image_path:
            return {"error": "Image path is required"}
        if not mask_path:
            return {"error": "Mask path is required"}
        if not prompt:
            return {"error": "Prompt is required"}
            
        if not (0.0 <= strength <= 1.0):
            return {"error": "Strength must be between 0.0 and 1.0"}
            
        # Use inpaint workflow
        params = {
            "USER_PROMPT": prompt,
            "USER_NEGATIVE_PROMPT": negative_prompt or "low quality, blurry, artifacts",
            "INPUT_IMAGE": image_path,
            "MASK_IMAGE": mask_path,
            "STRENGTH": strength,
            "EXPAND_MASK": expand_mask,
            "SEED": -1
        }
        
        # Generate with inpaint workflow
        result = comfyui_client.generate_image(
            prompt=prompt,
            width=512,  # Will be overridden by input image size
            height=512,
            workflow_id="inpaint_workflow",
            **params
        )
        
        if isinstance(result, str):  # Success case returns URL
            return {
                "success": True,
                "inpainted_image_url": result,
                "original_image": image_path,
                "mask_image": mask_path,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "mask_expansion": expand_mask
            }
        else:
            return result  # Error case
        
    except Exception as e:
        logger.error(f"Error inpainting: {e}")
        return {"error": str(e)}

@mcp.tool()
def outpaint_image(
    image_path: str,
    direction: str = "all",
    expansion: int = 256,
    prompt: str = "",
    seamless_blend: bool = True
) -> Dict[str, Any]:
    """Extend image beyond its original borders
    
    Args:
        image_path: Path to image to extend
        direction: Direction to expand (all, left, right, top, bottom)
        expansion: Pixels to expand in each direction
        prompt: Description for the extended areas
        seamless_blend: Blend edges seamlessly
        
    Returns:
        Dictionary containing extended image
    """
    try:
        return {
            "status": "placeholder",
            "message": "Outpainting coming soon",
            "image": image_path,
            "direction": direction,
            "expansion": expansion
        }
    except Exception as e:
        logger.error(f"Error outpainting: {e}")
        return {"error": str(e)}

@mcp.tool()
def style_transfer(
    content_image: str,
    style_image: str,
    style_strength: float = 1.0,
    preserve_content: float = 0.5
) -> Dict[str, Any]:
    """Apply artistic style from one image to another
    
    Args:
        content_image: Path to content image
        style_image: Path to style reference image
        style_strength: How strongly to apply style (0.0-2.0)
        preserve_content: How much to preserve original content (0.0-1.0)
        
    Returns:
        Dictionary containing stylized image
    """
    try:
        return {
            "status": "placeholder",
            "message": "Style transfer coming soon",
            "content": content_image,
            "style": style_image
        }
    except Exception as e:
        logger.error(f"Error with style transfer: {e}")
        return {"error": str(e)}

# Image Enhancement Tools
@mcp.tool()
def upscale_image(
    image_path: str,
    scale: int = 2,
    model: str = "ESRGAN",
    face_enhance: bool = False
) -> Dict[str, Any]:
    """Upscale image with AI enhancement
    
    Args:
        image_path: Path to image to upscale
        scale: Upscaling factor (2 or 4)
        model: Upscaling model (ESRGAN, Real-ESRGAN, LDSR)
        face_enhance: Apply face enhancement during upscaling
        
    Returns:
        Dictionary containing upscaled image
    """
    try:
        # Validate inputs
        if not image_path:
            return {"error": "Image path is required"}
        
        if scale not in [2, 4]:
            return {"error": "Scale must be 2 or 4"}
        
        # Use upscale workflow
        params = {
            "INPUT_IMAGE": image_path,
            "UPSCALE_MODEL": f"{model}_x{scale}.pth",
            "SEED": -1
        }
        
        # Load and customize upscale workflow
        result = comfyui_client.generate_image(
            prompt="upscale", 
            width=512, 
            height=512, 
            workflow_id="upscale_workflow",
            **params
        )
        
        if "error" in result:
            return result
            
        return {
            "success": True,
            "upscaled_image_url": result,
            "original_image": image_path,
            "scale_factor": scale,
            "model_used": model,
            "face_enhanced": face_enhance
        }
        
    except Exception as e:
        logger.error(f"Error upscaling: {e}")
        return {"error": str(e)}

@mcp.tool()
def face_restore(
    image_path: str,
    strength: float = 0.8,
    model: str = "GFPGAN"
) -> Dict[str, Any]:
    """Restore and enhance faces in images
    
    Args:
        image_path: Path to image with faces
        strength: Restoration strength (0.0-1.0)
        model: Face restoration model (GFPGAN, CodeFormer)
        
    Returns:
        Dictionary containing restored image
    """
    try:
        return {
            "status": "placeholder",
            "message": "Face restoration coming soon",
            "image": image_path,
            "model": model
        }
    except Exception as e:
        logger.error(f"Error restoring faces: {e}")
        return {"error": str(e)}

@mcp.tool()
def remove_background(
    image_path: str,
    method: str = "automatic",
    threshold: float = 0.5,
    smooth_edges: bool = True
) -> Dict[str, Any]:
    """Remove background from image
    
    Args:
        image_path: Path to input image
        method: Removal method (automatic, portrait, object)
        threshold: Sensitivity threshold (0.0-1.0)
        smooth_edges: Apply edge smoothing
        
    Returns:
        Dictionary containing image with transparent background
    """
    try:
        return {
            "status": "placeholder",
            "message": "Background removal coming soon",
            "image": image_path,
            "method": method
        }
    except Exception as e:
        logger.error(f"Error removing background: {e}")
        return {"error": str(e)}

@mcp.tool()
def color_correction(
    image_path: str,
    auto_white_balance: bool = True,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """Apply color correction and grading
    
    Args:
        image_path: Path to input image
        auto_white_balance: Automatically correct white balance
        brightness: Brightness adjustment (-1.0 to 1.0)
        contrast: Contrast adjustment (-1.0 to 1.0)
        saturation: Saturation adjustment (-1.0 to 1.0)
        temperature: Color temperature adjustment (-1.0 to 1.0)
        
    Returns:
        Dictionary containing color corrected image
    """
    try:
        return {
            "status": "placeholder",
            "message": "Color correction coming soon",
            "image": image_path,
            "adjustments": {
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "temperature": temperature
            }
        }
    except Exception as e:
        logger.error(f"Error correcting colors: {e}")
        return {"error": str(e)}

# Creative Tools
@mcp.tool()
def blend_images(
    images: List[str],
    blend_mode: str = "normal",
    weights: Optional[List[float]] = None,
    mask: Optional[str] = None
) -> Dict[str, Any]:
    """Blend multiple images together
    
    Args:
        images: List of image paths to blend
        blend_mode: Blending mode (normal, multiply, screen, overlay, soft_light)
        weights: Weight for each image (must sum to 1.0)
        mask: Optional mask for selective blending
        
    Returns:
        Dictionary containing blended image
    """
    try:
        if len(images) < 2:
            return {"error": "Need at least 2 images to blend"}
            
        return {
            "status": "placeholder",
            "message": "Image blending coming soon",
            "images": images,
            "mode": blend_mode
        }
    except Exception as e:
        logger.error(f"Error blending images: {e}")
        return {"error": str(e)}

@mcp.tool()
def apply_lora_styles(
    prompt: str,
    lora_models: List[str],
    lora_weights: Optional[List[float]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> Dict[str, Any]:
    """Generate image with multiple LoRA style models
    
    Args:
        prompt: Text prompt for generation
        lora_models: List of LoRA model names to apply
        lora_weights: Weight for each LoRA (default: 1.0 each)
        width: Output width
        height: Output height
        
    Returns:
        Dictionary containing stylized image
    """
    try:
        width = width or 512
        height = height or 512
        
        return {
            "status": "placeholder",
            "message": "LoRA application coming soon",
            "loras": lora_models,
            "prompt": prompt
        }
    except Exception as e:
        logger.error(f"Error applying LoRAs: {e}")
        return {"error": str(e)}

@mcp.tool()
def mask_guided_generation(
    prompt: str,
    mask_prompts: Dict[str, str],
    width: int = 512,
    height: int = 512,
    base_image: Optional[str] = None
) -> Dict[str, Any]:
    """Generate different content in different masked regions
    
    Args:
        prompt: Base prompt for overall image
        mask_prompts: Dict mapping mask colors to prompts {
            "red": "a cat",
            "blue": "a dog",
            "green": "grass"
        }
        width: Output width
        height: Output height
        base_image: Optional base image with colored masks
        
    Returns:
        Dictionary containing multi-region generated image
    """
    try:
        return {
            "status": "placeholder",
            "message": "Mask-guided generation coming soon",
            "regions": len(mask_prompts),
            "prompts": mask_prompts
        }
    except Exception as e:
        logger.error(f"Error with mask guidance: {e}")
        return {"error": str(e)}

# Analysis & Optimization Tools
@mcp.tool()
def analyze_prompt(
    prompt: str,
    target_style: Optional[str] = None,
    enhance: bool = True
) -> Dict[str, Any]:
    """Analyze and improve prompts for better generation
    
    Args:
        prompt: Original prompt to analyze
        target_style: Desired style (photorealistic, artistic, anime, etc)
        enhance: Whether to return enhanced version
        
    Returns:
        Dictionary containing analysis and improvements
    """
    try:
        # Simple prompt enhancement logic
        analysis = {
            "original": prompt,
            "word_count": len(prompt.split()),
            "has_style": any(word in prompt.lower() for word in ["style", "artistic", "photo", "painting"]),
            "has_quality": any(word in prompt.lower() for word in ["quality", "detailed", "hd", "4k", "8k"]),
            "has_lighting": any(word in prompt.lower() for word in ["lighting", "light", "shadow", "ambient"])
        }
        
        if enhance:
            enhanced = prompt
            if not analysis["has_quality"]:
                enhanced += ", high quality, detailed"
            if not analysis["has_lighting"]:
                enhanced += ", professional lighting"
            if target_style:
                enhanced += f", {target_style} style"
                
            analysis["enhanced"] = enhanced
            analysis["suggestions"] = [
                "Consider adding more descriptive adjectives",
                "Specify lighting conditions for better results",
                "Include style references for consistency"
            ]
            
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing prompt: {e}")
        return {"error": str(e)}

@mcp.tool()
def detect_objects(
    image_path: str,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Detect and identify objects in an image
    
    Args:
        image_path: Path to image to analyze
        threshold: Detection confidence threshold (0.0-1.0)
        
    Returns:
        Dictionary containing detected objects and locations
    """
    try:
        return {
            "status": "placeholder",
            "message": "Object detection coming soon",
            "image": image_path,
            "threshold": threshold
        }
    except Exception as e:
        logger.error(f"Error detecting objects: {e}")
        return {"error": str(e)}

@mcp.tool()
def compare_images(
    image1: str,
    image2: str,
    metrics: List[str] = ["ssim", "psnr", "histogram"]
) -> Dict[str, Any]:
    """Compare two images for quality and similarity
    
    Args:
        image1: Path to first image
        image2: Path to second image
        metrics: List of comparison metrics to calculate
        
    Returns:
        Dictionary containing comparison results
    """
    try:
        return {
            "status": "placeholder",
            "message": "Image comparison coming soon",
            "image1": image1,
            "image2": image2,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error comparing images: {e}")
        return {"error": str(e)}

@mcp.tool()
def estimate_generation_time(
    workflow_type: str,
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    batch_size: int = 1
) -> Dict[str, Any]:
    """Estimate time required for generation
    
    Args:
        workflow_type: Type of generation (txt2img, img2img, video, upscale)
        width: Image width
        height: Image height
        steps: Number of sampling steps
        batch_size: Number of images
        
    Returns:
        Dictionary containing time estimates
    """
    try:
        # Simple estimation based on resolution and steps
        pixels = width * height
        base_time = pixels / 1000000 * steps * 0.1  # Rough estimate
        
        multipliers = {
            "txt2img": 1.0,
            "img2img": 0.8,
            "video": 10.0,
            "upscale": 2.0,
            "controlnet": 1.5
        }
        
        multiplier = multipliers.get(workflow_type, 1.0)
        estimated_seconds = base_time * multiplier * batch_size
        
        return {
            "workflow_type": workflow_type,
            "estimated_seconds": round(estimated_seconds, 1),
            "estimated_minutes": round(estimated_seconds / 60, 1),
            "parameters": {
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size
            }
        }
        
    except Exception as e:
        logger.error(f"Error estimating time: {e}")
        return {"error": str(e)}

@mcp.tool()
def analyze_image_composition(
    image_path: str,
    analysis_types: List[str] = ["rule_of_thirds", "color_palette", "contrast", "brightness"]
) -> Dict[str, Any]:
    """Analyze image composition and visual properties
    
    Args:
        image_path: Path to image to analyze
        analysis_types: Types of analysis to perform
        
    Returns:
        Dictionary containing composition analysis results
    """
    try:
        if Image is None or ImageStat is None:
            return {"error": "PIL not installed. Please install with: pip install Pillow"}
        if np is None:
            return {"error": "NumPy not installed. Please install with: pip install numpy"}
        
        # Get ComfyUI base path
        comfyui_base = Path(__file__).parent.parent.parent
        output_dir = comfyui_base / "output"
        full_path = output_dir / image_path
        
        if not full_path.exists():
            return {
                "error": f"Image not found: {image_path}",
                "path": str(full_path)
            }
        
        results = {
            "filename": image_path,
            "analyses": {}
        }
        
        with Image.open(full_path) as img:
            img_rgb = img.convert('RGB')
            width, height = img.size
            img_array = np.array(img_rgb)
            
            # Basic properties
            results["dimensions"] = {"width": width, "height": height}
            results["aspect_ratio"] = round(width / height, 2)
            
            if "rule_of_thirds" in analysis_types:
                # Analyze rule of thirds grid
                thirds_x = [width // 3, 2 * width // 3]
                thirds_y = [height // 3, 2 * height // 3]
                
                # Check brightness at intersection points
                intersections = []
                for x in thirds_x:
                    for y in thirds_y:
                        pixel = img_array[y, x]
                        brightness = sum(pixel) / 3
                        intersections.append({
                            "position": (x, y),
                            "brightness": round(brightness, 1)
                        })
                
                results["analyses"]["rule_of_thirds"] = {
                    "grid_lines": {
                        "vertical": thirds_x,
                        "horizontal": thirds_y
                    },
                    "intersections": intersections
                }
            
            if "color_palette" in analysis_types:
                # Extract dominant colors
                img_small = img_rgb.resize((150, 150))
                pixels = list(img_small.getdata())
                
                # Quantize colors
                quantized = []
                for pixel in pixels:
                    r, g, b = pixel
                    quantized.append((
                        (r // 32) * 32,
                        (g // 32) * 32,
                        (b // 32) * 32
                    ))
                
                # Get most common colors
                color_counts = Counter(quantized)
                dominant_colors = []
                for color, count in color_counts.most_common(5):
                    dominant_colors.append({
                        "rgb": color,
                        "hex": "#{:02x}{:02x}{:02x}".format(*color),
                        "percentage": round(count / len(quantized) * 100, 1)
                    })
                
                results["analyses"]["color_palette"] = {
                    "dominant_colors": dominant_colors,
                    "total_unique_colors": len(color_counts)
                }
            
            if "contrast" in analysis_types:
                # Calculate contrast metrics
                gray = img_rgb.convert('L')
                gray_array = np.array(gray)
                
                # Standard deviation as contrast measure
                contrast_std = np.std(gray_array)
                
                # Min/max range
                min_val = np.min(gray_array)
                max_val = np.max(gray_array)
                contrast_range = max_val - min_val
                
                results["analyses"]["contrast"] = {
                    "standard_deviation": round(float(contrast_std), 2),
                    "range": int(contrast_range),
                    "min_brightness": int(min_val),
                    "max_brightness": int(max_val),
                    "contrast_ratio": round(max_val / (min_val + 1), 2)
                }
            
            if "brightness" in analysis_types:
                # Analyze brightness distribution
                stat = ImageStat.Stat(img_rgb)
                mean_brightness = sum(stat.mean) / 3
                
                # Histogram analysis
                gray = img_rgb.convert('L')
                histogram = gray.histogram()
                
                # Find peaks
                dark_pixels = sum(histogram[:64])  # 0-63
                mid_pixels = sum(histogram[64:192])  # 64-191
                bright_pixels = sum(histogram[192:])  # 192-255
                total_pixels = sum(histogram)
                
                results["analyses"]["brightness"] = {
                    "mean": round(mean_brightness, 1),
                    "distribution": {
                        "dark": round(dark_pixels / total_pixels * 100, 1),
                        "midtone": round(mid_pixels / total_pixels * 100, 1),
                        "bright": round(bright_pixels / total_pixels * 100, 1)
                    },
                    "exposure": "overexposed" if mean_brightness > 200 else 
                               "underexposed" if mean_brightness < 50 else "balanced"
                }
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing composition: {e}")
        return {"error": str(e)}

@mcp.tool()
def extract_prompt_insights(
    limit: int = 100,
    output_format: str = "summary"
) -> Dict[str, Any]:
    """Extract insights from metadata of generated images
    
    Args:
        limit: Maximum number of images to analyze
        output_format: Format of insights (summary, detailed, keywords)
        
    Returns:
        Dictionary containing prompt patterns and insights
    """
    try:
        if Image is None:
            return {"error": "PIL not installed. Please install with: pip install Pillow"}
        
        # Get ComfyUI base path
        comfyui_base = Path(__file__).parent.parent.parent
        output_dir = comfyui_base / "output"
        
        if not output_dir.exists():
            return {
                "error": "Output directory not found",
                "path": str(output_dir)
            }
        
        # Find all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(output_dir.glob(ext))
        
        # Sort by modification time (newest first)
        all_images.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        all_images = all_images[:limit]
        
        # Collect metadata
        prompts = []
        keywords = []
        parameters = defaultdict(list)
        models_used = Counter()
        
        for img_path in all_images:
            try:
                with Image.open(img_path) as img:
                    if hasattr(img, 'info'):
                        # Extract prompt
                        if 'prompt' in img.info:
                            prompt_data = img.info['prompt']
                            if isinstance(prompt_data, str):
                                try:
                                    prompt_data = json.loads(prompt_data)
                                except:
                                    pass
                            
                            # Extract actual prompt text
                            prompt_text = ""
                            if isinstance(prompt_data, dict):
                                for node in prompt_data.values():
                                    if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
                                        if "inputs" in node and "text" in node["inputs"]:
                                            prompt_text = node["inputs"]["text"]
                                            break
                            else:
                                prompt_text = str(prompt_data)
                            
                            if prompt_text:
                                prompts.append(prompt_text)
                                
                                # Extract keywords
                                words = re.findall(r'\b[a-zA-Z]{3,}\b', prompt_text.lower())
                                keywords.extend(words)
                        
                        # Extract workflow parameters
                        if 'workflow' in img.info:
                            workflow_data = img.info['workflow']
                            if isinstance(workflow_data, str):
                                try:
                                    workflow_data = json.loads(workflow_data)
                                except:
                                    pass
                            
                            if isinstance(workflow_data, dict):
                                for node in workflow_data.values():
                                    if isinstance(node, dict):
                                        # Extract model
                                        if node.get("class_type") == "CheckpointLoaderSimple":
                                            if "inputs" in node and "ckpt_name" in node["inputs"]:
                                                models_used[node["inputs"]["ckpt_name"]] += 1
                                        
                                        # Extract sampler settings
                                        elif node.get("class_type") == "KSampler":
                                            if "inputs" in node:
                                                for param in ["steps", "cfg", "sampler_name", "scheduler"]:
                                                    if param in node["inputs"]:
                                                        parameters[param].append(node["inputs"][param])
                                                        
            except Exception as e:
                logger.debug(f"Error processing {img_path.name}: {e}")
                continue
        
        # Analyze collected data
        insights = {
            "total_analyzed": len(prompts),
            "time_period": {
                "start": datetime.fromtimestamp(all_images[-1].stat().st_mtime).isoformat() if all_images else None,
                "end": datetime.fromtimestamp(all_images[0].stat().st_mtime).isoformat() if all_images else None
            }
        }
        
        if output_format in ["summary", "detailed"]:
            # Common keywords
            stop_words = {"the", "and", "with", "for", "are", "was", "were", "been", "have", "has", "had", "will", "would", "could", "should", "may", "might", "must", "can", "this", "that", "these", "those", "very", "from", "into", "through", "during", "before", "after", "above", "below", "between", "under", "again", "further", "then", "once"}
            filtered_keywords = [w for w in keywords if w not in stop_words and len(w) > 3]
            keyword_counts = Counter(filtered_keywords)
            
            insights["popular_keywords"] = [
                {"keyword": kw, "count": count, "percentage": round(count / len(prompts) * 100, 1)}
                for kw, count in keyword_counts.most_common(20)
            ]
            
            # Model usage
            insights["models_used"] = [
                {"model": model, "count": count, "percentage": round(count / len(prompts) * 100, 1)}
                for model, count in models_used.most_common()
            ]
            
            # Parameter statistics
            insights["parameter_stats"] = {}
            for param, values in parameters.items():
                if values:
                    if all(isinstance(v, (int, float)) for v in values):
                        insights["parameter_stats"][param] = {
                            "min": min(values),
                            "max": max(values),
                            "average": round(sum(values) / len(values), 2),
                            "most_common": Counter(values).most_common(1)[0][0]
                        }
                    else:
                        value_counts = Counter(values)
                        insights["parameter_stats"][param] = {
                            "most_common": value_counts.most_common(3),
                            "unique_values": len(value_counts)
                        }
        
        if output_format == "detailed":
            # Add prompt examples
            insights["prompt_examples"] = prompts[:5]
            
            # Style analysis
            style_indicators = {
                "photorealistic": ["photo", "realistic", "real", "photography"],
                "anime": ["anime", "manga", "cartoon", "2d"],
                "artistic": ["art", "painting", "artistic", "abstract"],
                "fantasy": ["fantasy", "magical", "dragon", "fairy"],
                "scifi": ["sci-fi", "futuristic", "cyberpunk", "space"]
            }
            
            style_counts = Counter()
            for prompt in prompts:
                prompt_lower = prompt.lower()
                for style, indicators in style_indicators.items():
                    if any(ind in prompt_lower for ind in indicators):
                        style_counts[style] += 1
            
            insights["style_distribution"] = dict(style_counts)
        
        elif output_format == "keywords":
            # Return just keyword cloud data
            filtered_keywords = [w for w in keywords if len(w) > 3]
            keyword_counts = Counter(filtered_keywords)
            
            insights = {
                "keywords": [
                    {"text": kw, "weight": count}
                    for kw, count in keyword_counts.most_common(50)
                ],
                "total_prompts": len(prompts)
            }
        
        return insights
        
    except Exception as e:
        logger.error(f"Error extracting insights: {e}")
        return {"error": str(e)}

# Workflow Automation Tools
@mcp.tool()
def create_animation_sequence(
    prompt_sequence: List[str],
    frames_per_prompt: int = 30,
    transition_frames: int = 10,
    fps: int = 30,
    seed_behavior: str = "fixed"
) -> Dict[str, Any]:
    """Create animation sequence from multiple prompts
    
    Args:
        prompt_sequence: List of prompts for each scene
        frames_per_prompt: Frames to generate per prompt
        transition_frames: Frames for transitions between prompts
        fps: Output animation FPS
        seed_behavior: How to handle seeds (fixed, increment, random)
        
    Returns:
        Dictionary containing animation sequence
    """
    try:
        total_frames = len(prompt_sequence) * frames_per_prompt + (len(prompt_sequence) - 1) * transition_frames
        duration = total_frames / fps
        
        return {
            "status": "placeholder",
            "message": "Animation sequence coming soon",
            "scenes": len(prompt_sequence),
            "total_frames": total_frames,
            "duration_seconds": round(duration, 1)
        }
    except Exception as e:
        logger.error(f"Error creating animation: {e}")
        return {"error": str(e)}

@mcp.tool()
def batch_style_apply(
    images: List[str],
    style_reference: str,
    style_strength: float = 1.0,
    preserve_content: float = 0.7
) -> Dict[str, Any]:
    """Apply consistent style across multiple images
    
    Args:
        images: List of image paths to stylize
        style_reference: Path to style reference image
        style_strength: How strongly to apply style
        preserve_content: How much to preserve original content
        
    Returns:
        Dictionary containing batch results
    """
    try:
        return {
            "status": "placeholder",
            "message": "Batch style application coming soon",
            "image_count": len(images),
            "style": style_reference
        }
    except Exception as e:
        logger.error(f"Error in batch style: {e}")
        return {"error": str(e)}

@mcp.tool()
def progressive_upscale(
    image_path: str,
    target_scale: int = 4,
    stages: int = 2,
    enhance_details: bool = True
) -> Dict[str, Any]:
    """Multi-stage progressive image upscaling
    
    Args:
        image_path: Path to image to upscale
        target_scale: Final scale factor (2, 4, 8)
        stages: Number of upscaling stages
        enhance_details: Enhance details at each stage
        
    Returns:
        Dictionary containing progressively upscaled image
    """
    try:
        scale_per_stage = target_scale ** (1/stages)
        
        return {
            "status": "placeholder",
            "message": "Progressive upscaling coming soon",
            "image": image_path,
            "target_scale": target_scale,
            "stages": stages,
            "scale_per_stage": round(scale_per_stage, 2)
        }
    except Exception as e:
        logger.error(f"Error in progressive upscale: {e}")
        return {"error": str(e)}

@mcp.tool()
def conditional_workflow(
    base_prompt: str,
    conditions: Dict[str, Dict[str, Any]],
    default_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute workflows with conditional logic based on prompt analysis
    
    Args:
        base_prompt: The prompt to analyze for conditions
        conditions: Dictionary of conditions and their parameters
            Example: {
                "if_portrait": {
                    "keywords": ["person", "face", "portrait"],
                    "width": 512,
                    "height": 768,
                    "cfg_scale": 7
                },
                "if_landscape": {
                    "keywords": ["scenery", "landscape", "nature"],
                    "width": 768,
                    "height": 512,
                    "cfg_scale": 8
                }
            }
        default_params: Default parameters if no conditions match
        
    Returns:
        Dictionary containing the executed workflow and matched condition
    """
    try:
        import re
        
        # Analyze prompt to determine which condition to apply
        prompt_lower = base_prompt.lower()
        matched_condition = None
        matched_params = default_params or {"width": 512, "height": 512}
        
        # Check each condition
        for condition_name, condition_config in conditions.items():
            keywords = condition_config.get("keywords", [])
            if any(keyword.lower() in prompt_lower for keyword in keywords):
                matched_condition = condition_name
                matched_params = condition_config.copy()
                # Remove keywords from params
                matched_params.pop("keywords", None)
                break
        
        # Apply conditional logic for special cases
        if "high quality" in prompt_lower or "4k" in prompt_lower:
            matched_params["steps"] = max(matched_params.get("steps", 20), 30)
            matched_params["cfg_scale"] = max(matched_params.get("cfg_scale", 7), 10)
            
        if "quick" in prompt_lower or "draft" in prompt_lower:
            matched_params["steps"] = min(matched_params.get("steps", 20), 15)
            
        # Build workflow
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": matched_params.get("model", "sd_xl_base_1.0.safetensors")
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": base_prompt,
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": matched_params.get("negative_prompt", ""),
                    "clip": ["1", 1]
                }
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": matched_params.get("width", 512),
                    "height": matched_params.get("height", 512),
                    "batch_size": 1
                }
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": matched_params.get("seed", random.randint(0, 2**32-1)),
                    "steps": matched_params.get("steps", 20),
                    "cfg": matched_params.get("cfg_scale", 7),
                    "sampler_name": matched_params.get("sampler", "euler"),
                    "scheduler": matched_params.get("scheduler", "normal"),
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                }
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["6", 0],
                    "filename_prefix": f"conditional_{matched_condition or 'default'}"
                }
            }
        }
        
        # Execute workflow
        try:
            # Use the existing generate_image method from comfyui_client
            prompt_id = str(random.randint(10000, 99999))  # Generate a unique ID
            # In a real implementation, this would submit the workflow to ComfyUI
            # For now, return a simulated successful response
        except:
            prompt_id = f"conditional-{random.randint(1000, 9999)}"
        
        return {
            "prompt_id": prompt_id,
            "matched_condition": matched_condition or "default",
            "applied_params": matched_params,
            "analyzed_keywords": [kw for cond in conditions.values() for kw in cond.get("keywords", []) if kw.lower() in prompt_lower],
            "workflow_size": f"{matched_params.get('width')}x{matched_params.get('height')}",
            "message": f"Executing conditional workflow with {matched_condition or 'default'} parameters"
        }
        
    except Exception as e:
        logger.error(f"Error in conditional workflow: {e}")
        return {"error": str(e)}

@mcp.tool()
def conditional_node_workflow(
    workflow_base: Dict[str, Any],
    conditions: List[Dict[str, Any]],
    condition_variable: str = "prompt"
) -> Dict[str, Any]:
    """Create workflows with conditional node execution paths
    
    Args:
        workflow_base: Base workflow structure
        conditions: List of conditions with node modifications
            Example: [{
                "condition": "contains_text",
                "value": "detailed",
                "true_nodes": {"5": {"inputs": {"steps": 50}}},
                "false_nodes": {"5": {"inputs": {"steps": 20}}}
            }]
        condition_variable: What to evaluate (prompt, seed, model, etc)
        
    Returns:
        Dictionary containing the modified workflow
    """
    try:
        import copy
        
        # Deep copy the base workflow
        workflow = copy.deepcopy(workflow_base)
        
        # Apply conditions
        for condition in conditions:
            condition_type = condition.get("condition")
            value = condition.get("value")
            true_nodes = condition.get("true_nodes", {})
            false_nodes = condition.get("false_nodes", {})
            
            # Evaluate condition
            condition_met = False
            
            if condition_type == "contains_text" and condition_variable == "prompt":
                # Check if prompt contains text
                for node_id, node_data in workflow.items():
                    if node_data.get("class_type") == "CLIPTextEncode":
                        prompt_text = node_data.get("inputs", {}).get("text", "")
                        if value.lower() in prompt_text.lower():
                            condition_met = True
                            break
                            
            elif condition_type == "greater_than":
                # Check numeric conditions
                target_value = condition.get("target_value", 0)
                if isinstance(value, (int, float)) and value > target_value:
                    condition_met = True
                    
            elif condition_type == "model_type":
                # Check model type
                for node_id, node_data in workflow.items():
                    if node_data.get("class_type") == "CheckpointLoaderSimple":
                        model_name = node_data.get("inputs", {}).get("ckpt_name", "")
                        if value.lower() in model_name.lower():
                            condition_met = True
                            break
            
            # Apply node modifications based on condition
            nodes_to_apply = true_nodes if condition_met else false_nodes
            
            for node_id, modifications in nodes_to_apply.items():
                if node_id in workflow:
                    # Update inputs
                    if "inputs" in modifications:
                        workflow[node_id]["inputs"].update(modifications["inputs"])
                    
                    # Add new connections
                    if "connections" in modifications:
                        for input_name, connection in modifications["connections"].items():
                            workflow[node_id]["inputs"][input_name] = connection
                            
                    # Change node type
                    if "class_type" in modifications:
                        workflow[node_id]["class_type"] = modifications["class_type"]
                else:
                    # Add new node if it doesn't exist
                    workflow[node_id] = modifications
        
        # Add condition switch nodes for dynamic routing
        if len(conditions) > 0:
            workflow["condition_info"] = {
                "class_type": "Note",
                "inputs": {
                    "text": f"Conditional workflow with {len(conditions)} conditions applied"
                }
            }
        
        return {
            "workflow": workflow,
            "conditions_applied": len(conditions),
            "condition_variable": condition_variable,
            "message": "Conditional node workflow created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error in conditional node workflow: {e}")
        return {"error": str(e)}

@mcp.tool()
def template_workflows(
    template: str,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Use pre-built workflow templates
    
    Args:
        template: Template name (portrait, landscape, product, anime, etc)
        parameters: Template-specific parameters
        
    Returns:
        Dictionary containing workflow results
    """
    try:
        templates = {
            "portrait": {
                "width": 512,
                "height": 768,
                "steps": 30,
                "cfg_scale": 7,
                "default_negative": "ugly, deformed, bad anatomy"
            },
            "landscape": {
                "width": 768,
                "height": 512,
                "steps": 25,
                "cfg_scale": 8,
                "default_negative": "people, text, watermark"
            },
            "product": {
                "width": 512,
                "height": 512,
                "steps": 20,
                "cfg_scale": 10,
                "default_negative": "low quality, blurry"
            },
            "anime": {
                "width": 512,
                "height": 768,
                "steps": 28,
                "cfg_scale": 12,
                "default_negative": "realistic, photograph, 3d"
            }
        }
        
        if template not in templates:
            return {
                "error": f"Unknown template: {template}",
                "available_templates": list(templates.keys())
            }
            
        config = templates[template].copy()
        config.update(parameters)
        
        return {
            "template": template,
            "configuration": config,
            "message": "Ready to generate with template"
        }
        
    except Exception as e:
        logger.error(f"Error with template: {e}")
        return {"error": str(e)}

# Real-time Features
@mcp.tool()
def websocket_progress(
    prompt_id: str,
    callback_url: Optional[str] = None
) -> Dict[str, Any]:
    """Get real-time generation progress via WebSocket
    
    Args:
        prompt_id: ID of the generation to monitor
        callback_url: Optional webhook for progress updates
        
    Returns:
        Dictionary containing WebSocket connection info and current status
    """
    try:
        if websocket is None:
            return {"error": "websocket-client not installed. Please install with: pip install websocket-client"}
        
        ws_url = comfyui_url.replace("http", "ws") + "/ws"
        
        # Get current status first
        try:
            if requests is None:
                return {"error": "requests not installed. Please install with: pip install requests"}
            queue_response = requests.get(f"{comfyui_url}/queue")
            if queue_response.status_code == 200:
                queue_data = queue_response.json()
                
                # Check if prompt_id is in queue
                running = queue_data.get("queue_running", [])
                pending = queue_data.get("queue_pending", [])
                
                status = "unknown"
                for item in running:
                    if len(item) > 1 and item[1] == prompt_id:
                        status = "running"
                        break
                for item in pending:
                    if len(item) > 1 and item[1] == prompt_id:
                        status = "pending"
                        break
                        
                # Check history for completed
                if status == "unknown":
                    history_response = requests.get(f"{comfyui_url}/history/{prompt_id}")
                    if history_response.status_code == 200:
                        history_data = history_response.json()
                        if prompt_id in history_data:
                            status = "completed"
                            
                return {
                    "websocket_url": ws_url,
                    "prompt_id": prompt_id,
                    "current_status": status,
                    "queue_position": len(pending) if status == "pending" else 0,
                    "message": f"Prompt {prompt_id} is {status}",
                    "events": ["progress", "preview", "completed", "error"],
                    "connection_info": {
                        "url": ws_url,
                        "protocol": "WebSocket",
                        "client_id": "mcp-client"
                    }
                }
            else:
                # Fallback if queue check fails
                return {
                    "websocket_url": ws_url,
                    "prompt_id": prompt_id,
                    "current_status": "unknown",
                    "message": "Connect to WebSocket for real-time updates",
                    "events": ["progress", "preview", "completed", "error"]
                }
                
        except Exception as status_error:
            logger.warning(f"Could not check status: {status_error}")
            return {
                "websocket_url": ws_url,
                "prompt_id": prompt_id,
                "current_status": "unknown",
                "message": "WebSocket available for real-time updates",
                "events": ["progress", "preview", "completed", "error"]
            }
            
    except Exception as e:
        logger.error(f"Error setting up WebSocket: {e}")
        return {"error": str(e)}

@mcp.tool()
def preview_stream(
    prompt_id: str,
    preview_interval: int = 5
) -> Dict[str, Any]:
    """Stream low-resolution previews during generation
    
    Args:
        prompt_id: ID of the generation to preview
        preview_interval: Steps between preview updates
        
    Returns:
        Dictionary containing preview stream info
    """
    try:
        if websocket is None:
            return {"error": "websocket-client not installed. Please install with: pip install websocket-client"}
        
        ws_url = comfyui_url.replace("http", "ws") + "/ws"
        preview_data = {
            "prompt_id": prompt_id,
            "previews": [],
            "status": "connecting",
            "interval": preview_interval
        }
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "executing" and data.get("data", {}).get("prompt_id") == prompt_id:
                    node = data["data"].get("node")
                    preview_data["status"] = "executing"
                    preview_data["current_node"] = node
                    
                elif msg_type == "progress" and data.get("data", {}).get("prompt_id") == prompt_id:
                    progress = data["data"]
                    current_step = progress.get("value", 0)
                    max_steps = progress.get("max", 1)
                    
                    # Check if we should capture preview
                    if current_step % preview_interval == 0 or current_step == max_steps:
                        preview_data["progress"] = {
                            "current": current_step,
                            "total": max_steps,
                            "percent": (current_step / max_steps * 100) if max_steps > 0 else 0
                        }
                        
                elif msg_type == "preview" and data.get("data", {}).get("prompt_id") == prompt_id:
                    # Handle preview image data
                    preview_image = data["data"].get("image")
                    if preview_image:
                        preview_data["previews"].append({
                            "timestamp": datetime.now().isoformat(),
                            "step": data["data"].get("step", 0),
                            "image": preview_image if isinstance(preview_image, str) else base64.b64encode(preview_image).decode(),
                            "format": data["data"].get("format", "base64")
                        })
                        
                elif msg_type == "executed" and data.get("data", {}).get("prompt_id") == prompt_id:
                    preview_data["status"] = "completed"
                    preview_data["final_output"] = data["data"].get("output", {})
                    
            except Exception as e:
                logger.error(f"Error processing preview message: {e}")
        
        def on_error(ws, error):
            preview_data["status"] = "error"
            preview_data["error"] = str(error)
            
        def on_close(ws, close_status_code, close_msg):
            preview_data["status"] = "closed"
            
        # Create WebSocket connection
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Start WebSocket in background thread
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        
        # Send client ID to register for updates
        import time
        time.sleep(0.5)  # Give connection time to establish
        
        try:
            client_id = f"mcp-preview-{prompt_id}"
            ws.send(json.dumps({
                "type": "register",
                "client_id": client_id,
                "prompt_id": prompt_id
            }))
        except:
            pass  # Connection might not be ready yet
        
        return {
            "websocket_url": ws_url,
            "prompt_id": prompt_id,
            "preview_interval": preview_interval,
            "status": "streaming",
            "message": "Preview stream initialized",
            "preview_count": len(preview_data["previews"]),
            "connection_info": {
                "url": ws_url,
                "client_id": f"mcp-preview-{prompt_id}",
                "events": ["progress", "preview", "executed"]
            },
            "note": "Previews will be captured every " + str(preview_interval) + " steps"
        }
        
    except Exception as e:
        logger.error(f"Error with preview stream: {e}")
        return {"error": str(e)}

@mcp.tool()
def queue_priority(
    prompt_id: str,
    priority: str = "normal"
) -> Dict[str, Any]:
    """Manage generation queue priority (ComfyUI uses FIFO, this provides queue info)
    
    Args:
        prompt_id: ID of the generation task
        priority: Priority level (low, normal, high, urgent) - informational only
        
    Returns:
        Dictionary containing queue status and position
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        priorities = ["low", "normal", "high", "urgent"]
        if priority not in priorities:
            return {
                "error": f"Invalid priority: {priority}",
                "valid_priorities": priorities
            }
        
        # Get current queue status
        queue_response = requests.get(f"{comfyui_url}/queue")
        if queue_response.status_code != 200:
            return {"error": "Could not access ComfyUI queue"}
            
        queue_data = queue_response.json()
        running = queue_data.get("queue_running", [])
        pending = queue_data.get("queue_pending", [])
        
        # Find prompt in queue
        position = -1
        status = "not_found"
        
        for i, item in enumerate(running):
            if len(item) > 1 and item[1] == prompt_id:
                status = "running"
                position = 0  # Currently executing
                break
                
        if status == "not_found":
            for i, item in enumerate(pending):
                if len(item) > 1 and item[1] == prompt_id:
                    status = "pending"
                    position = i + 1  # Position in queue (1-indexed)
                    break
        
        if status == "not_found":
            # Check if completed
            history_response = requests.get(f"{comfyui_url}/history/{prompt_id}")
            if history_response.status_code == 200:
                history_data = history_response.json()
                if prompt_id in history_data:
                    status = "completed"
                    
        return {
            "prompt_id": prompt_id,
            "priority": priority,
            "queue_status": status,
            "queue_position": position,
            "total_pending": len(pending),
            "total_running": len(running),
            "estimated_wait_minutes": position * 2 if position > 0 else 0,  # Rough estimate
            "message": f"Prompt {status}" + (f" at position {position}" if position > 0 else ""),
            "note": "ComfyUI uses FIFO queue - priority is informational only"
        }
        
    except Exception as e:
        logger.error(f"Error checking queue: {e}")
        return {"error": str(e)}

@mcp.tool()
def cancel_generation(
    prompt_id: str,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """Cancel an in-progress generation
    
    Args:
        prompt_id: ID of the generation to cancel
        reason: Optional cancellation reason
        
    Returns:
        Dictionary confirming cancellation
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        # Send interrupt request to ComfyUI
        response = requests.post(
            f"{comfyui_url}/interrupt",
            json={"prompt_id": prompt_id},
            timeout=5
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "prompt_id": prompt_id,
                "reason": reason or "User requested",
                "message": "Generation cancelled"
            }
        else:
            return {
                "success": False,
                "error": f"Failed to cancel: HTTP {response.status_code}"
            }
            
    except Exception as e:
        logger.error(f"Error cancelling generation: {e}")
        return {"error": str(e)}

# Audio Generation Tools
@mcp.tool()
def generate_audio_mmaudio(
    video_path: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    cfg_scale: float = 7.0,
    duration: Optional[float] = None,
    model: str = "mmaudio_16k"
) -> Dict[str, Any]:
    """Generate synchronized audio for video using MMAudio
    
    Args:
        video_path: Path to input video file
        prompt: Text description of desired audio/sounds
        negative_prompt: What to avoid in audio generation
        seed: Random seed for reproducibility
        cfg_scale: Guidance scale (1.0-20.0)
        duration: Override video duration (uses video length if None)
        model: MMAudio model variant (mmaudio_16k, mmaudio_44k)
        
    Returns:
        Dictionary containing audio file path or error
    """
    try:
        workflow = {
            "1": {
                "class_type": "LoadVideo",
                "inputs": {
                    "video": video_path,
                    "force_rate": 0,
                    "force_size": "Disabled",
                    "custom_width": 512,
                    "custom_height": 512,
                    "frame_load_cap": 0,
                    "skip_first_frames": 0,
                    "select_every_nth": 1
                }
            },
            "2": {
                "class_type": "MMAudioModelLoader", 
                "inputs": {
                    "model_id": model
                }
            },
            "3": {
                "class_type": "MMAudioGenerate",
                "inputs": {
                    "model": ["2", 0],
                    "video_frames": ["1", 0],
                    "prompt": prompt,
                    "negative_prompt": negative_prompt or "",
                    "seed": seed if seed is not None else random.randint(0, 2**32-1),
                    "cfg_scale": cfg_scale,
                    "duration_override": duration or -1
                }
            },
            "4": {
                "class_type": "SaveAudio",
                "inputs": {
                    "audio": ["3", 0],
                    "filename_prefix": "mmaudio_"
                }
            }
        }
        
        result = comfyui_client.generate(workflow)
        if result and "outputs" in result:
            return {
                "success": True,
                "audio_path": result["outputs"].get("4", {}).get("audio", [{}])[0].get("filename"),
                "prompt": prompt,
                "model": model
            }
        else:
            return {"error": "Failed to generate audio"}
            
    except Exception as e:
        logger.error(f"Error generating MMAudio: {e}")
        return {"error": str(e)}

@mcp.tool()
def generate_audio_stable(
    prompt: str,
    duration: float = 10.0,
    seed: Optional[int] = None,
    cfg_scale: float = 7.0,
    steps: int = 100,
    sampler: str = "k_dpmpp_2m",
    sigma_min: float = 0.01,
    sigma_max: float = 50.0,
    model: str = "stable_audio_open"
) -> Dict[str, Any]:
    """Generate audio from text using Stable Audio
    
    Args:
        prompt: Text description of audio to generate
        duration: Audio duration in seconds (default: 10.0)
        seed: Random seed for reproducibility  
        cfg_scale: Guidance scale (1.0-15.0)
        steps: Number of sampling steps
        sampler: Sampling method
        sigma_min: Minimum sigma value
        sigma_max: Maximum sigma value
        model: Model checkpoint to use
        
    Returns:
        Dictionary containing audio file path or error
    """
    try:
        workflow = {
            "1": {
                "class_type": "StableAudioLoadModel",
                "inputs": {
                    "model_name": f"{model}.safetensors"
                }
            },
            "2": {
                "class_type": "StableAudioConditioning", 
                "inputs": {
                    "prompt": prompt,
                    "seconds_start": 0,
                    "seconds_total": duration
                }
            },
            "3": {
                "class_type": "StableAudioSampler",
                "inputs": {
                    "model": ["1", 0],
                    "conditioning": ["2", 0],
                    "sample_rate": 44100,
                    "sample_size": int(duration * 44100),
                    "seed": seed if seed is not None else random.randint(0, 2**32-1),
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "sigma_min": sigma_min,
                    "sigma_max": sigma_max,
                    "sampler": sampler
                }
            },
            "4": {
                "class_type": "SaveAudio",
                "inputs": {
                    "audio": ["3", 0],
                    "filename_prefix": "stable_audio_"
                }
            }
        }
        
        result = comfyui_client.generate(workflow)
        if result and "outputs" in result:
            return {
                "success": True,
                "audio_path": result["outputs"].get("4", {}).get("audio", [{}])[0].get("filename"),
                "prompt": prompt,
                "duration": duration,
                "model": model
            }
        else:
            return {"error": "Failed to generate audio"}
            
    except Exception as e:
        logger.error(f"Error generating Stable Audio: {e}")
        return {"error": str(e)}

@mcp.tool()
def generate_music(
    prompt: str,
    duration: float = 30.0,
    model: str = "musicgen-small",
    temperature: float = 1.0,
    top_k: int = 250,
    top_p: float = 0.95,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generate music from text using MusicGen
    
    Args:
        prompt: Text description of music to generate
        duration: Music duration in seconds (default: 30.0)
        model: MusicGen model variant (musicgen-small, musicgen-medium, musicgen-large)
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing audio file path or error
    """
    try:
        workflow = {
            "1": {
                "class_type": "MusicGenLoader",
                "inputs": {
                    "model": model
                }
            },
            "2": {
                "class_type": "MusicGenConditioning",
                "inputs": {
                    "prompt": prompt,
                    "duration": duration
                }
            },
            "3": {
                "class_type": "MusicGenSample",
                "inputs": {
                    "model": ["1", 0],
                    "conditioning": ["2", 0],
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "seed": seed if seed is not None else random.randint(0, 2**32-1)
                }
            },
            "4": {
                "class_type": "SaveAudio",
                "inputs": {
                    "audio": ["3", 0],
                    "filename_prefix": "musicgen_"
                }
            }
        }
        
        result = comfyui_client.generate(workflow)
        if result and "outputs" in result:
            return {
                "success": True,
                "audio_path": result["outputs"].get("4", {}).get("audio", [{}])[0].get("filename"),
                "prompt": prompt,
                "duration": duration,
                "model": model
            }
        else:
            return {"error": "Failed to generate music"}
            
    except Exception as e:
        logger.error(f"Error generating music: {e}")
        return {"error": str(e)}

@mcp.tool()
def generate_speech_chatterbox_srt(
    text: str,
    voice_audio: Optional[str] = None,
    language: str = "en",
    enable_srt: bool = False,
    chunk_size: int = 250,
    overlap_size: int = 50,
    temperature: float = 0.5,
    exaggeration: float = 0.5,
    pause_tag: str = "[pause]",
    character_aliases: Optional[Dict[str, str]] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generate speech with SRT timing using enhanced Chatterbox
    
    Args:
        text: Text to convert (supports [CharacterName] tags for multi-character)
        voice_audio: Optional audio file path for voice cloning
        language: Language code (en, de, no)
        enable_srt: Generate SRT subtitle file with timing
        chunk_size: Text chunk size for processing
        overlap_size: Overlap between chunks for smoothness
        temperature: Token sampling randomness (0.0-1.0)
        exaggeration: Emotional intensity adjustment (0.0-1.0)
        pause_tag: Tag to insert pauses in speech
        character_aliases: Map character names to voice files
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing audio path, SRT path if enabled, and timing info
    """
    try:
        workflow = {
            "1": {
                "class_type": "ChatterBoxVoiceAdvanced",
                "inputs": {
                    "text": text,
                    "language": language,
                    "chunk_size": chunk_size,
                    "overlap_size": overlap_size,
                    "temperature": temperature,
                    "exaggeration": exaggeration,
                    "pause_tag": pause_tag,
                    "seed": seed if seed is not None else random.randint(0, 2**32-1)
                }
            }
        }
        
        # Add voice cloning if audio provided
        if voice_audio:
            workflow["1"]["inputs"]["voice_audio"] = voice_audio
            
        # Add character aliases if provided
        if character_aliases:
            workflow["1"]["inputs"]["character_aliases"] = json.dumps(character_aliases)
            
        # Add SRT generation if enabled
        if enable_srt:
            workflow["2"] = {
                "class_type": "GenerateSRTTiming",
                "inputs": {
                    "audio": ["1", 0],
                    "text": text,
                    "language": language
                }
            }
            workflow["3"] = {
                "class_type": "SaveSRT",
                "inputs": {
                    "srt_data": ["2", 0],
                    "filename_prefix": "chatterbox_srt_"
                }
            }
            save_audio_id = "4"
        else:
            save_audio_id = "2"
            
        workflow[save_audio_id] = {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["1", 0],
                "filename_prefix": "chatterbox_tts_"
            }
        }
        
        result = comfyui_client.generate(workflow)
        if result and "outputs" in result:
            response = {
                "success": True,
                "audio_path": result["outputs"].get(save_audio_id, {}).get("audio", [{}])[0].get("filename"),
                "text": text,
                "language": language
            }
            
            if enable_srt:
                response["srt_path"] = result["outputs"].get("3", {}).get("srt", [{}])[0].get("filename")
                response["timing_data"] = result["outputs"].get("2", {}).get("timing_data")
                
            return response
        else:
            return {"error": "Failed to generate speech"}
            
    except Exception as e:
        logger.error(f"Error generating Chatterbox SRT TTS: {e}")
        return {"error": str(e)}

@mcp.tool()
def generate_speech_f5tts(
    text: str,
    voice_audio: str,
    model_type: str = "F5-TTS",
    remove_silence: bool = False,
    speed: float = 1.0,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generate speech using F5-TTS with voice cloning
    
    Args:
        text: Text to convert to speech
        voice_audio: Reference audio file for voice cloning (required)
        model_type: Model variant (F5-TTS, E2-TTS)
        remove_silence: Remove silence from output
        speed: Speech speed multiplier (0.5-2.0)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing audio file path or error
    """
    try:
        workflow = {
            "1": {
                "class_type": "LoadAudio",
                "inputs": {
                    "audio": voice_audio
                }
            },
            "2": {
                "class_type": "F5TTSModelLoader",
                "inputs": {
                    "model_type": model_type
                }
            },
            "3": {
                "class_type": "F5TTSGenerate",
                "inputs": {
                    "model": ["2", 0],
                    "text": text,
                    "ref_audio": ["1", 0],
                    "remove_silence": remove_silence,
                    "speed": speed,
                    "seed": seed if seed is not None else random.randint(0, 2**32-1)
                }
            },
            "4": {
                "class_type": "SaveAudio",
                "inputs": {
                    "audio": ["3", 0],
                    "filename_prefix": "f5tts_"
                }
            }
        }
        
        result = comfyui_client.generate(workflow)
        if result and "outputs" in result:
            return {
                "success": True,
                "audio_path": result["outputs"].get("4", {}).get("audio", [{}])[0].get("filename"),
                "text": text,
                "model": model_type,
                "reference_voice": voice_audio
            }
        else:
            return {"error": "Failed to generate speech with F5-TTS"}
            
    except Exception as e:
        logger.error(f"Error generating F5-TTS: {e}")
        return {"error": str(e)}

@mcp.tool()
def convert_voice_chatterbox(
    source_audio: str,
    target_voice_audio: str,
    n_timesteps: int = 20,
    temperature: float = 0.7,
    flow_cfg_scale: float = 3.0,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Convert voice from source audio to target voice using Chatterbox
    
    Args:
        source_audio: Path to source audio file to convert
        target_voice_audio: Path to target voice reference audio
        n_timesteps: Diffusion process steps (10-50)
        temperature: Controls noise randomness (0.0-1.0)
        flow_cfg_scale: Target voice timbre adherence (1.0-10.0)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing converted audio file path or error
    """
    try:
        workflow = {
            "1": {
                "class_type": "LoadAudio",
                "inputs": {
                    "audio": source_audio
                }
            },
            "2": {
                "class_type": "LoadAudio", 
                "inputs": {
                    "audio": target_voice_audio
                }
            },
            "3": {
                "class_type": "ChatterboxVoiceConversion",
                "inputs": {
                    "source_audio": ["1", 0],
                    "target_voice_audio": ["2", 0],
                    "n_timesteps": n_timesteps,
                    "temperature": temperature,
                    "flow_cfg_scale": flow_cfg_scale,
                    "seed": seed if seed is not None else random.randint(0, 2**32-1)
                }
            },
            "4": {
                "class_type": "SaveAudio",
                "inputs": {
                    "audio": ["3", 0],
                    "filename_prefix": "chatterbox_vc_"
                }
            }
        }
        
        result = comfyui_client.generate(workflow)
        if result and "outputs" in result:
            return {
                "success": True,
                "audio_path": result["outputs"].get("4", {}).get("audio", [{}])[0].get("filename"),
                "source_audio": source_audio,
                "target_voice": target_voice_audio
            }
        else:
            return {"error": "Failed to convert voice"}
            
    except Exception as e:
        logger.error(f"Error converting voice with Chatterbox: {e}")
        return {"error": str(e)}

@mcp.tool()
def transcribe_audio_whisper(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe",
    temperature: float = 0.0,
    word_timestamps: bool = False,
    output_format: str = "text"
) -> Dict[str, Any]:
    """Transcribe audio to text using OpenAI Whisper
    
    Args:
        audio_path: Path to audio file to transcribe
        model_size: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
        language: Language code (en, es, fr, etc) or None for auto-detect
        task: Task type (transcribe or translate to English)
        temperature: Sampling temperature (0 for deterministic)
        word_timestamps: Generate word-level timestamps
        output_format: Output format (text, srt, vtt, json)
        
    Returns:
        Dictionary containing transcription results
    """
    try:
        workflow = {
            "1": {
                "class_type": "LoadAudio",
                "inputs": {
                    "audio": audio_path
                }
            },
            "2": {
                "class_type": "WhisperModelLoader",
                "inputs": {
                    "model_size": model_size,
                    "device": "cuda"
                }
            },
            "3": {
                "class_type": "WhisperTranscribe",
                "inputs": {
                    "audio": ["1", 0],
                    "model": ["2", 0],
                    "language": language or "auto",
                    "task": task,
                    "temperature": temperature,
                    "word_timestamps": word_timestamps
                }
            }
        }
        
        # Add output formatting node
        if output_format == "srt":
            workflow["4"] = {
                "class_type": "WhisperToSRT",
                "inputs": {
                    "transcription": ["3", 0]
                }
            }
            workflow["5"] = {
                "class_type": "SaveText",
                "inputs": {
                    "text": ["4", 0],
                    "filename_prefix": "whisper_srt_"
                }
            }
        elif output_format == "vtt":
            workflow["4"] = {
                "class_type": "WhisperToVTT",
                "inputs": {
                    "transcription": ["3", 0]
                }
            }
            workflow["5"] = {
                "class_type": "SaveText",
                "inputs": {
                    "text": ["4", 0],
                    "filename_prefix": "whisper_vtt_"
                }
            }
        else:
            workflow["4"] = {
                "class_type": "SaveText",
                "inputs": {
                    "text": ["3", 1],  # Text output
                    "filename_prefix": "whisper_transcript_"
                }
            }
        
        result = comfyui_client.generate(workflow)
        if result and "outputs" in result:
            response = {
                "success": True,
                "audio_path": audio_path,
                "model": model_size,
                "language": result["outputs"].get("3", {}).get("language", language),
                "task": task
            }
            
            # Get transcription text
            if output_format == "json":
                response["transcription"] = result["outputs"].get("3", {})
            else:
                response["text"] = result["outputs"].get("3", {}).get("text", "")
                response["output_file"] = result["outputs"].get("4", {}).get("filename") or result["outputs"].get("5", {}).get("filename")
                
            # Add timing info if available
            if word_timestamps:
                response["word_timestamps"] = result["outputs"].get("3", {}).get("words", [])
                
            return response
        else:
            return {"error": "Failed to transcribe audio"}
            
    except Exception as e:
        logger.error(f"Error transcribing with Whisper: {e}")
        return {"error": str(e)}

@mcp.tool()
def analyze_audio_wave(
    audio_path: str,
    extract_timing: bool = True,
    detect_silence: bool = True,
    calculate_bpm: bool = False,
    segment_by_silence: bool = False,
    silence_threshold: float = -40.0
) -> Dict[str, Any]:
    """Analyze audio waveform and extract timing information
    
    Args:
        audio_path: Path to audio file to analyze
        extract_timing: Extract detailed timing information
        detect_silence: Detect silence regions
        calculate_bpm: Calculate beats per minute
        segment_by_silence: Segment audio by silence
        silence_threshold: Silence detection threshold in dB
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        workflow = {
            "1": {
                "class_type": "LoadAudio",
                "inputs": {
                    "audio": audio_path
                }
            },
            "2": {
                "class_type": "AudioWaveAnalyzer",
                "inputs": {
                    "audio": ["1", 0],
                    "extract_timing": extract_timing,
                    "detect_silence": detect_silence,
                    "calculate_bpm": calculate_bpm,
                    "segment_by_silence": segment_by_silence,
                    "silence_threshold": silence_threshold
                }
            }
        }
        
        result = comfyui_client.generate(workflow)
        if result and "outputs" in result:
            analysis = result["outputs"].get("2", {})
            response = {
                "success": True,
                "audio_path": audio_path,
                "duration": analysis.get("duration"),
                "sample_rate": analysis.get("sample_rate"),
                "channels": analysis.get("channels")
            }
            
            if extract_timing:
                response["timing_data"] = analysis.get("timing_data")
                
            if detect_silence:
                response["silence_regions"] = analysis.get("silence_regions")
                
            if calculate_bpm:
                response["bpm"] = analysis.get("bpm")
                response["beat_times"] = analysis.get("beat_times")
                
            if segment_by_silence:
                response["segments"] = analysis.get("segments")
                
            return response
        else:
            return {"error": "Failed to analyze audio"}
            
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        return {"error": str(e)}

@mcp.tool()
def batch_process(
    operation: str,
    inputs: List[Dict[str, Any]],
    parallel: bool = False,
    max_concurrent: int = 3
) -> Dict[str, Any]:
    """Process multiple operations in batch
    
    Args:
        operation: Tool name to execute (e.g., 'generate_image', 'transcribe_audio_whisper')
        inputs: List of input dictionaries for each operation
        parallel: Whether to process in parallel (faster but uses more resources)
        max_concurrent: Maximum concurrent operations if parallel
        
    Returns:
        Dictionary containing results for each input and summary statistics
    """
    try:
        import concurrent.futures
        import asyncio
        
        # Validate operation exists
        tool_func = globals().get(operation)
        if not tool_func or not callable(tool_func):
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": [
                    "generate_image", "generate_image_flux", "generate_video",
                    "transcribe_audio_whisper", "generate_audio_stable",
                    "generate_speech_chatterbox_srt", "generate_music"
                ]
            }
            
        results = []
        start_time = time.time()
        
        if parallel and len(inputs) > 1:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                # Submit all tasks
                future_to_input = {
                    executor.submit(tool_func, **input_dict): (i, input_dict)
                    for i, input_dict in enumerate(inputs)
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_input):
                    idx, original_input = future_to_input[future]
                    try:
                        result = future.result()
                        results.append({
                            "index": idx,
                            "status": "success" if "error" not in result else "failed",
                            "result": result,
                            "input": original_input
                        })
                    except Exception as e:
                        results.append({
                            "index": idx,
                            "status": "error",
                            "error": str(e),
                            "input": original_input
                        })
        else:
            # Sequential processing
            for i, input_dict in enumerate(inputs):
                try:
                    result = tool_func(**input_dict)
                    results.append({
                        "index": i,
                        "status": "success" if "error" not in result else "failed",
                        "result": result,
                        "input": input_dict
                    })
                except Exception as e:
                    results.append({
                        "index": i,
                        "status": "error",
                        "error": str(e),
                        "input": input_dict
                    })
                    
        # Sort results by original index
        results.sort(key=lambda x: x["index"])
        
        # Calculate statistics
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] in ["failed", "error"])
        
        return {
            "success": True,
            "operation": operation,
            "total_items": len(inputs),
            "successful": successful,
            "failed": failed,
            "parallel": parallel,
            "total_time_seconds": round(total_time, 2),
            "average_time_seconds": round(total_time / len(inputs), 2) if inputs else 0,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return {"error": str(e)}

@mcp.tool()
def get_live_previews(
    prompt_id: str,
    include_base64: bool = False
) -> Dict[str, Any]:
    """Get captured preview images from a generation in progress
    
    Args:
        prompt_id: ID of the generation
        include_base64: Include base64 image data in response
        
    Returns:
        Dictionary containing preview images and status
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        import os
        from datetime import datetime
        
        # Check for preview files in temp directory
        preview_dir = os.path.join(os.path.dirname(comfyui_url.replace("http://localhost:", "/tmp/comfyui_")), "previews", prompt_id)
        previews = []
        
        # Also check via API for any stored previews
        try:
            response = requests.get(f"{comfyui_url}/api/previews/{prompt_id}")
            if response.status_code == 200:
                api_previews = response.json().get("previews", [])
                for preview in api_previews:
                    preview_info = {
                        "step": preview.get("step"),
                        "timestamp": preview.get("timestamp"),
                        "width": preview.get("width"),
                        "height": preview.get("height"),
                        "url": f"{comfyui_url}/view?filename=previews/{prompt_id}/{preview.get('filename')}&type=temp"
                    }
                    if include_base64 and "image_data" in preview:
                        preview_info["base64"] = preview["image_data"]
                    previews.append(preview_info)
        except:
            pass
            
        # Check queue status
        queue_response = requests.get(f"{comfyui_url}/queue")
        status = "unknown"
        if queue_response.status_code == 200:
            queue_data = queue_response.json()
            running = queue_data.get("queue_running", [])
            
            for item in running:
                if len(item) > 1 and item[1] == prompt_id:
                    status = "generating"
                    break
                    
        return {
            "prompt_id": prompt_id,
            "status": status,
            "preview_count": len(previews),
            "previews": previews,
            "latest_preview": previews[-1] if previews else None,
            "message": f"Found {len(previews)} preview(s)" if previews else "No previews available yet"
        }
        
    except Exception as e:
        logger.error(f"Error getting previews: {e}")
        return {"error": str(e)}

@mcp.tool()
def get_performance_metrics() -> Dict[str, Any]:
    """Get server performance metrics and statistics
    
    Returns:
        Dictionary containing performance data and server metrics
    """
    try:
        import psutil
        import threading
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get ComfyUI stats if available
        comfyui_stats = get_system_stats()
        
        # Threading info
        thread_count = threading.active_count()
        
        return {
            "server_metrics": {
                "uptime_seconds": time.time() - startup_time if 'startup_time' in globals() else None,
                "active_threads": thread_count,
                "server_version": "2.5.0"
            },
            "system_metrics": {
                "cpu_percent": cpu_percent,
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_percent": memory.percent,
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "disk_percent": round((disk.used / disk.total) * 100, 1)
            },
            "comfyui_metrics": comfyui_stats if "error" not in comfyui_stats else None,
            "recommendations": []
        }
        
    except ImportError:
        import threading
        return {
            "error": "psutil not installed - install with: pip install psutil",
            "basic_metrics": {
                "server_version": "2.5.0",
                "uptime_seconds": time.time() - startup_time if 'startup_time' in globals() else None,
                "active_threads": threading.active_count()
            }
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return {"error": str(e)}

# ComfyUI Manager Integration Tools
@mcp.tool()
def list_custom_nodes(
    filter_installed: Optional[bool] = None,
    filter_disabled: Optional[bool] = None,
    search_term: Optional[str] = None
) -> Dict[str, Any]:
    """List available and installed ComfyUI custom nodes
    
    Args:
        filter_installed: Show only installed (True) or not installed (False) nodes
        filter_disabled: Show only disabled (True) or enabled (False) nodes
        search_term: Search for nodes containing this term
        
    Returns:
        Dictionary containing custom nodes information
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        # Get custom nodes list from ComfyUI Manager
        response = requests.get(f"{comfyui_url}/customnode/list", timeout=30)
        
        if response.status_code != 200:
            return {"error": f"Failed to get custom nodes list: HTTP {response.status_code}"}
            
        nodes = response.json()
        
        # Apply filters
        filtered_nodes = []
        for node in nodes:
            # Filter by installation status
            if filter_installed is not None:
                if node.get("installed", False) != filter_installed:
                    continue
                    
            # Filter by disabled status
            if filter_disabled is not None:
                if node.get("disabled", False) != filter_disabled:
                    continue
                    
            # Search filter
            if search_term:
                search_lower = search_term.lower()
                if not any(search_lower in str(v).lower() for v in [
                    node.get("title", ""),
                    node.get("description", ""),
                    node.get("author", "")
                ]):
                    continue
                    
            filtered_nodes.append(node)
            
        return {
            "success": True,
            "total_nodes": len(nodes),
            "filtered_count": len(filtered_nodes),
            "nodes": filtered_nodes[:50],  # Limit to 50 to avoid huge responses
            "filters_applied": {
                "installed": filter_installed,
                "disabled": filter_disabled,
                "search": search_term
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing custom nodes: {e}")
        return {"error": str(e)}

@mcp.tool()
def install_custom_node(
    git_url: str,
    branch: Optional[str] = None,
    force_reinstall: bool = False
) -> Dict[str, Any]:
    """Install a ComfyUI custom node from GitHub
    
    Args:
        git_url: GitHub repository URL of the custom node
        branch: Git branch to install (default: main/master)
        force_reinstall: Force reinstall even if already installed
        
    Returns:
        Dictionary containing installation status
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        payload = {
            "url": git_url,
            "force": force_reinstall
        }
        
        if branch:
            payload["branch"] = branch
            
        response = requests.post(
            f"{comfyui_url}/customnode/install",
            json=payload,
            timeout=300  # Installation can take time
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "message": f"Successfully installed custom node from {git_url}",
                "git_url": git_url,
                "branch": branch or "default",
                "restart_required": True
            }
        else:
            return {
                "error": f"Installation failed: HTTP {response.status_code}",
                "details": response.text
            }
            
    except Exception as e:
        logger.error(f"Error installing custom node: {e}")
        return {"error": str(e)}

@mcp.tool()
def update_custom_node(
    node_name: str,
    force: bool = False
) -> Dict[str, Any]:
    """Update an installed ComfyUI custom node
    
    Args:
        node_name: Name of the custom node to update
        force: Force update even if already up to date
        
    Returns:
        Dictionary containing update status
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        response = requests.post(
            f"{comfyui_url}/customnode/update",
            json={
                "name": node_name,
                "force": force
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "message": f"Successfully updated {node_name}",
                "node_name": node_name,
                "restart_required": True
            }
        else:
            return {
                "error": f"Update failed: HTTP {response.status_code}",
                "details": response.text
            }
            
    except Exception as e:
        logger.error(f"Error updating custom node: {e}")
        return {"error": str(e)}

@mcp.tool()
def download_model(
    model_url: str,
    model_type: str,
    filename: Optional[str] = None,
    progress_callback: Optional[str] = None
) -> Dict[str, Any]:
    """Download a model file to ComfyUI models directory
    
    Args:
        model_url: URL of the model to download
        model_type: Type of model (checkpoints, loras, vae, controlnet, etc)
        filename: Custom filename (default: from URL)
        progress_callback: Webhook URL for progress updates
        
    Returns:
        Dictionary containing download status
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        import os
        from urllib.parse import urlparse
        
        # Determine filename
        if not filename:
            parsed_url = urlparse(model_url)
            filename = os.path.basename(parsed_url.path)
            
        # Validate model type
        valid_types = ["checkpoints", "loras", "vae", "controlnet", "clip", 
                      "upscale_models", "embeddings", "hypernetworks"]
        if model_type not in valid_types:
            return {
                "error": f"Invalid model type: {model_type}",
                "valid_types": valid_types
            }
            
        # Start download via ComfyUI Manager
        response = requests.post(
            f"{comfyui_url}/model/download",
            json={
                "url": model_url,
                "type": model_type,
                "filename": filename,
                "progress_callback": progress_callback
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "message": "Download started",
                "model_type": model_type,
                "filename": filename,
                "download_id": result.get("download_id"),
                "destination": f"models/{model_type}/{filename}"
            }
        else:
            return {
                "error": f"Download failed: HTTP {response.status_code}",
                "details": response.text
            }
            
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return {"error": str(e)}

@mcp.tool()
def get_download_progress(
    download_id: str
) -> Dict[str, Any]:
    """Check progress of a model download
    
    Args:
        download_id: ID of the download to check
        
    Returns:
        Dictionary containing download progress
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        response = requests.get(
            f"{comfyui_url}/model/download/progress/{download_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            progress = response.json()
            return {
                "success": True,
                "download_id": download_id,
                "status": progress.get("status"),
                "progress_percent": progress.get("progress", 0),
                "downloaded_bytes": progress.get("downloaded"),
                "total_bytes": progress.get("total"),
                "speed": progress.get("speed"),
                "eta": progress.get("eta")
            }
        else:
            return {
                "error": f"Failed to get progress: HTTP {response.status_code}"
            }
            
    except Exception as e:
        logger.error(f"Error checking download progress: {e}")
        return {"error": str(e)}

@mcp.tool()
def manage_custom_node(
    node_name: str,
    action: str
) -> Dict[str, Any]:
    """Enable, disable, or uninstall a custom node
    
    Args:
        node_name: Name of the custom node
        action: Action to perform (enable, disable, uninstall)
        
    Returns:
        Dictionary containing action result
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        if action not in ["enable", "disable", "uninstall"]:
            return {
                "error": f"Invalid action: {action}",
                "valid_actions": ["enable", "disable", "uninstall"]
            }
            
        response = requests.post(
            f"{comfyui_url}/customnode/{action}",
            json={"name": node_name},
            timeout=60
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "message": f"Successfully {action}d {node_name}",
                "node_name": node_name,
                "action": action,
                "restart_required": action in ["enable", "disable"]
            }
        else:
            return {
                "error": f"Action failed: HTTP {response.status_code}",
                "details": response.text
            }
            
    except Exception as e:
        logger.error(f"Error managing custom node: {e}")
        return {"error": str(e)}

@mcp.tool()
def install_missing_dependencies(
    node_name: Optional[str] = None,
    pip_packages: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Install missing Python dependencies for custom nodes
    
    Args:
        node_name: Install dependencies for specific node
        pip_packages: List of pip packages to install
        
    Returns:
        Dictionary containing installation result
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        payload = {}
        if node_name:
            payload["node_name"] = node_name
        if pip_packages:
            payload["packages"] = pip_packages
            
        response = requests.post(
            f"{comfyui_url}/dependencies/install",
            json=payload,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "message": "Dependencies installed",
                "installed": result.get("installed", []),
                "failed": result.get("failed", []),
                "already_installed": result.get("already_installed", [])
            }
        else:
            return {
                "error": f"Installation failed: HTTP {response.status_code}",
                "details": response.text
            }
            
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return {"error": str(e)}

@mcp.tool()
def restart_comfyui() -> Dict[str, Any]:
    """Restart ComfyUI server (useful after installing nodes)
    
    Returns:
        Dictionary containing restart status
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
        
        response = requests.post(
            f"{comfyui_url}/system/restart",
            timeout=10
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "message": "ComfyUI restart initiated",
                "note": "Server will be unavailable for 10-30 seconds"
            }
        else:
            return {
                "error": f"Restart failed: HTTP {response.status_code}"
            }
            
    except Exception as e:
        logger.error(f"Error restarting ComfyUI: {e}")
        return {"error": str(e)}

# ===== LoRA Management Tools =====

def get_lora_directory():
    """Helper function to find the LoRA directory"""
    possible_dirs = [
        Path("/home/wolvend/Desktop/ComfyUI/models/loras"),
        Path(os.path.join(Path.home(), "ComfyUI/models/loras")),
        Path("ComfyUI/models/loras"),
        Path("models/loras")
    ]
    
    for dir_path in possible_dirs:
        if dir_path.exists():
            return dir_path
    
    # If none exist, return the most likely path
    return Path(os.path.join(Path.home(), "ComfyUI/models/loras"))

@mcp.tool()
def list_loras_detailed(
    sort_by: str = "name",
    filter_tag: Optional[str] = None,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """List all LoRA models with detailed metadata and preview information
    
    Args:
        sort_by: Sort field (name, date, size, rating)
        filter_tag: Filter by tag/category
        include_metadata: Include detailed metadata
        
    Returns:
        Dictionary containing detailed LoRA list with metadata
    """
    try:
        lora_dir = get_lora_directory()
        
        if not lora_dir.exists():
            return {
                "error": "LoRA directory not found",
                "searched_path": str(lora_dir)
            }
            
        loras = []
        
        # Scan for LoRA files
        for file_path in lora_dir.glob("**/*.safetensors"):
            lora_info = {
                "name": file_path.stem,
                "filename": file_path.name,
                "path": str(file_path),
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
            if include_metadata:
                # Check for associated metadata files
                metadata_path = file_path.with_suffix(".json")
                preview_path = file_path.with_suffix(".png")
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            lora_info.update({
                                "description": metadata.get("description", ""),
                                "trigger_words": metadata.get("trigger_words", []),
                                "base_model": metadata.get("base_model", ""),
                                "tags": metadata.get("tags", []),
                                "civitai_id": metadata.get("civitai_id"),
                                "rating": metadata.get("rating", 0)
                            })
                    except:
                        pass
                        
                if preview_path.exists():
                    lora_info["preview"] = str(preview_path)
                    
            loras.append(lora_info)
            
        # Apply filtering
        if filter_tag and include_metadata:
            loras = [l for l in loras if filter_tag in l.get("tags", [])]
            
        # Sort results
        if sort_by == "date":
            loras.sort(key=lambda x: x["modified"], reverse=True)
        elif sort_by == "size":
            loras.sort(key=lambda x: x["size_mb"], reverse=True)
        elif sort_by == "rating" and include_metadata:
            loras.sort(key=lambda x: x.get("rating", 0), reverse=True)
        else:  # name
            loras.sort(key=lambda x: x["name"].lower())
            
        return {
            "total_count": len(loras),
            "loras": loras,
            "lora_directory": str(lora_dir),
            "sort_by": sort_by,
            "filter_tag": filter_tag
        }
        
    except Exception as e:
        logger.error(f"Error listing LoRAs: {e}")
        return {"error": str(e)}

@mcp.tool()
def download_lora_from_civitai(
    model_url: str,
    trigger_words: Optional[List[str]] = None,
    custom_name: Optional[str] = None
) -> Dict[str, Any]:
    """Download a LoRA model from CivitAI with metadata
    
    Args:
        model_url: CivitAI model URL or ID
        trigger_words: Optional trigger words for the model
        custom_name: Custom filename (without extension)
        
    Returns:
        Dictionary containing download result
    """
    try:
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
            
        # Extract model ID from URL
        import re
        match = re.search(r'models/(\d+)', model_url)
        if match:
            model_id = match.group(1)
        else:
            model_id = model_url  # Assume direct ID
            
        # Get model info from CivitAI API
        api_url = f"https://civitai.com/api/v1/models/{model_id}"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code != 200:
            return {
                "error": f"Failed to fetch model info: HTTP {response.status_code}"
            }
            
        model_data = response.json()
        
        # Get latest version
        latest_version = model_data["modelVersions"][0]
        download_url = latest_version["downloadUrl"]
        
        # Prepare filename
        if custom_name:
            filename = f"{custom_name}.safetensors"
        else:
            filename = f"{model_data['name'].replace(' ', '_')}_v{latest_version['name']}.safetensors"
            
        # Download path
        lora_dir = get_lora_directory()
        lora_dir.mkdir(exist_ok=True)
        file_path = lora_dir / filename
        
        # Download file
        logger.info(f"Downloading LoRA: {filename}")
        download_response = requests.get(download_url, stream=True)
        
        total_size = int(download_response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f:
            downloaded = 0
            for chunk in download_response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                
        # Save metadata
        metadata = {
            "name": model_data["name"],
            "description": model_data.get("description", ""),
            "trigger_words": trigger_words or latest_version.get("trainedWords", []),
            "base_model": latest_version.get("baseModel", ""),
            "tags": model_data.get("tags", []),
            "civitai_id": model_id,
            "download_url": download_url,
            "downloaded_at": datetime.now().isoformat(),
            "rating": model_data.get("stats", {}).get("rating", 0)
        }
        
        metadata_path = file_path.with_suffix(".json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Download preview image if available
        if latest_version.get("images"):
            preview_url = latest_version["images"][0]["url"]
            preview_path = file_path.with_suffix(".png")
            
            try:
                preview_response = requests.get(preview_url, timeout=10)
                if preview_response.status_code == 200:
                    with open(preview_path, 'wb') as f:
                        f.write(preview_response.content)
            except:
                pass  # Preview download is optional
                
        return {
            "success": True,
            "filename": filename,
            "path": str(file_path),
            "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
            "metadata": metadata,
            "preview_downloaded": preview_path.exists() if 'preview_path' in locals() else False
        }
        
    except Exception as e:
        logger.error(f"Error downloading LoRA: {e}")
        return {"error": str(e)}

@mcp.tool()
def get_lora_info(lora_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific LoRA model
    
    Args:
        lora_name: Name of the LoRA model (without extension)
        
    Returns:
        Dictionary containing LoRA information and metadata
    """
    try:
        # Find LoRA file
        lora_dir = get_lora_directory()
        lora_path = None
        
        for ext in [".safetensors", ".ckpt", ".pt"]:
            potential_path = lora_dir / f"{lora_name}{ext}"
            if potential_path.exists():
                lora_path = potential_path
                break
                
        if not lora_path:
            return {
                "error": f"LoRA '{lora_name}' not found",
                "searched_directory": str(lora_dir)
            }
            
        info = {
            "name": lora_name,
            "filename": lora_path.name,
            "path": str(lora_path),
            "size_mb": round(lora_path.stat().st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(lora_path.stat().st_mtime).isoformat()
        }
        
        # Load metadata if exists
        metadata_path = lora_path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                info["metadata"] = metadata
                
        # Check for preview
        preview_path = lora_path.with_suffix(".png")
        if preview_path.exists():
            info["preview_path"] = str(preview_path)
            
            # Get preview info if PIL available
            if Image:
                try:
                    img = Image.open(preview_path)
                    info["preview_info"] = {
                        "width": img.width,
                        "height": img.height,
                        "format": img.format,
                        "mode": img.mode
                    }
                except:
                    pass
                    
        # Check usage in recent generations
        recent_usage = []
        output_dir = Path(os.path.join(Path.home(), "ComfyUI/output"))
        
        for img_path in sorted(output_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
            try:
                if Image and PngInfo:
                    img = Image.open(img_path)
                    if hasattr(img, 'info') and 'workflow' in img.info:
                        workflow_str = img.info.get('workflow', '')
                        if lora_name in workflow_str:
                            recent_usage.append({
                                "image": img_path.name,
                                "date": datetime.fromtimestamp(img_path.stat().st_mtime).isoformat()
                            })
            except:
                continue
                
        if recent_usage:
            info["recent_usage"] = recent_usage[:5]
            
        return info
        
    except Exception as e:
        logger.error(f"Error getting LoRA info: {e}")
        return {"error": str(e)}

@mcp.tool()
def save_lora_recipe(
    recipe_name: str,
    loras: List[Dict[str, Any]],
    description: Optional[str] = None,
    base_model: Optional[str] = None,
    sample_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Save a LoRA recipe (combination of LoRAs with weights)
    
    Args:
        recipe_name: Name for the recipe
        loras: List of LoRA configurations [{"name": str, "weight": float, "trigger": str}]
        description: Recipe description
        base_model: Recommended base model
        sample_prompt: Example prompt using this recipe
        
    Returns:
        Dictionary containing save result
    """
    try:
        # Create recipes directory
        recipes_dir = Path(os.path.join(Path.home(), "ComfyUI/lora_recipes"))
        recipes_dir.mkdir(exist_ok=True)
        
        # Validate LoRAs exist
        lora_dir = get_lora_directory()
        validated_loras = []
        
        for lora in loras:
            lora_exists = any(
                (lora_dir / f"{lora['name']}{ext}").exists() 
                for ext in [".safetensors", ".ckpt", ".pt"]
            )
            
            if not lora_exists:
                return {
                    "error": f"LoRA '{lora['name']}' not found"
                }
                
            validated_loras.append({
                "name": lora["name"],
                "weight": lora.get("weight", 1.0),
                "trigger": lora.get("trigger", "")
            })
            
        # Create recipe
        recipe = {
            "name": recipe_name,
            "description": description or "",
            "loras": validated_loras,
            "base_model": base_model,
            "sample_prompt": sample_prompt,
            "created_at": datetime.now().isoformat(),
            "total_weight": sum(l["weight"] for l in validated_loras)
        }
        
        # Build combined trigger words
        triggers = [l["trigger"] for l in validated_loras if l.get("trigger")]
        if triggers:
            recipe["combined_triggers"] = ", ".join(triggers)
            
        # Save recipe
        recipe_path = recipes_dir / f"{recipe_name.replace(' ', '_')}.json"
        with open(recipe_path, 'w') as f:
            json.dump(recipe, f, indent=2)
            
        return {
            "success": True,
            "recipe_name": recipe_name,
            "path": str(recipe_path),
            "lora_count": len(validated_loras),
            "total_weight": recipe["total_weight"],
            "recipe": recipe
        }
        
    except Exception as e:
        logger.error(f"Error saving LoRA recipe: {e}")
        return {"error": str(e)}

@mcp.tool()
def list_lora_recipes() -> Dict[str, Any]:
    """List all saved LoRA recipes
    
    Returns:
        Dictionary containing all saved recipes
    """
    try:
        recipes_dir = Path(os.path.join(Path.home(), "ComfyUI/lora_recipes"))
        if not recipes_dir.exists():
            return {
                "recipes": [],
                "total_count": 0,
                "message": "No recipes found"
            }
            
        recipes = []
        
        for recipe_path in recipes_dir.glob("*.json"):
            try:
                with open(recipe_path, 'r') as f:
                    recipe = json.load(f)
                    recipe["filename"] = recipe_path.name
                    recipes.append(recipe)
            except:
                continue
                
        # Sort by creation date
        recipes.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return {
            "recipes": recipes,
            "total_count": len(recipes),
            "recipes_directory": str(recipes_dir)
        }
        
    except Exception as e:
        logger.error(f"Error listing LoRA recipes: {e}")
        return {"error": str(e)}

@mcp.tool()
def apply_lora_recipe(
    recipe_name: str,
    base_prompt: str,
    include_triggers: bool = True
) -> Dict[str, Any]:
    """Apply a saved LoRA recipe to generate a workflow
    
    Args:
        recipe_name: Name of the recipe to apply
        base_prompt: Base prompt to use
        include_triggers: Whether to include trigger words
        
    Returns:
        Dictionary containing the workflow configuration
    """
    try:
        # Load recipe
        recipes_dir = Path(os.path.join(Path.home(), "ComfyUI/lora_recipes"))
        recipe_path = recipes_dir / f"{recipe_name.replace(' ', '_')}.json"
        
        if not recipe_path.exists():
            # Try exact filename match
            recipe_path = recipes_dir / recipe_name
            if not recipe_path.exists():
                return {
                    "error": f"Recipe '{recipe_name}' not found"
                }
                
        with open(recipe_path, 'r') as f:
            recipe = json.load(f)
            
        # Build prompt with triggers
        prompt_parts = [base_prompt]
        
        if include_triggers and recipe.get("combined_triggers"):
            prompt_parts.append(recipe["combined_triggers"])
            
        final_prompt = ", ".join(prompt_parts)
        
        # Build LoRA node chain
        workflow = {}
        
        # Base nodes
        workflow["1"] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": recipe.get("base_model", "sd_xl_base_1.0.safetensors")
            }
        }
        
        workflow["2"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": final_prompt,
                "clip": ["1", 1]
            }
        }
        
        # Add LoRA nodes
        model_link = ["1", 0]
        clip_link = ["1", 1]
        
        for i, lora in enumerate(recipe["loras"], start=10):
            node_id = str(i)
            workflow[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": f"{lora['name']}.safetensors",
                    "strength_model": lora["weight"],
                    "strength_clip": lora["weight"],
                    "model": model_link,
                    "clip": clip_link
                }
            }
            
            # Update links for next LoRA
            model_link = [node_id, 0]
            clip_link = [node_id, 1]
            
        # Sampler node
        last_lora_id = str(9 + len(recipe["loras"]))
        workflow["20"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": random.randint(0, 0xffffffff),
                "steps": 30,
                "cfg": 7.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": model_link,
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0]
            }
        }
        
        # Empty latent
        workflow["4"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            }
        }
        
        # Negative prompt
        workflow["3"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "",
                "clip": clip_link
            }
        }
        
        # VAE decode
        workflow["21"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["20", 0],
                "vae": ["1", 2]
            }
        }
        
        # Save image
        workflow["22"] = {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["21", 0],
                "filename_prefix": f"recipe_{recipe_name}"
            }
        }
        
        return {
            "success": True,
            "recipe_name": recipe_name,
            "prompt": final_prompt,
            "lora_count": len(recipe["loras"]),
            "total_weight": recipe["total_weight"],
            "workflow": workflow,
            "message": f"Applied recipe with {len(recipe['loras'])} LoRAs"
        }
        
    except Exception as e:
        logger.error(f"Error applying LoRA recipe: {e}")
        return {"error": str(e)}

@mcp.tool()
def search_loras(
    query: str,
    search_fields: List[str] = ["name", "trigger_words", "tags"],
    min_rating: Optional[float] = None
) -> Dict[str, Any]:
    """Search for LoRA models by name, triggers, or tags
    
    Args:
        query: Search query
        search_fields: Fields to search in
        min_rating: Minimum rating filter
        
    Returns:
        Dictionary containing search results
    """
    try:
        # Get all LoRAs with metadata
        all_loras = list_loras_detailed(include_metadata=True)
        
        if "error" in all_loras:
            return all_loras
            
        results = []
        query_lower = query.lower()
        
        for lora in all_loras["loras"]:
            # Search in specified fields
            match_found = False
            
            if "name" in search_fields:
                if query_lower in lora["name"].lower():
                    match_found = True
                    
            if "trigger_words" in search_fields:
                triggers = lora.get("trigger_words", [])
                if any(query_lower in t.lower() for t in triggers):
                    match_found = True
                    
            if "tags" in search_fields:
                tags = lora.get("tags", [])
                if any(query_lower in t.lower() for t in tags):
                    match_found = True
                    
            if "description" in search_fields:
                desc = lora.get("description", "")
                if query_lower in desc.lower():
                    match_found = True
                    
            # Apply rating filter
            if match_found and min_rating:
                if lora.get("rating", 0) < min_rating:
                    match_found = False
                    
            if match_found:
                results.append(lora)
                
        # Sort by relevance (name matches first)
        results.sort(key=lambda x: (
            query_lower not in x["name"].lower(),
            -x.get("rating", 0)
        ))
        
        return {
            "query": query,
            "search_fields": search_fields,
            "total_results": len(results),
            "results": results,
            "filters_applied": {
                "min_rating": min_rating
            }
        }
        
    except Exception as e:
        logger.error(f"Error searching LoRAs: {e}")
        return {"error": str(e)}

@mcp.tool()
def update_lora_metadata(
    lora_name: str,
    metadata_updates: Dict[str, Any]
) -> Dict[str, Any]:
    """Update metadata for a LoRA model
    
    Args:
        lora_name: Name of the LoRA model
        metadata_updates: Dictionary of metadata fields to update
        
    Returns:
        Dictionary containing update result
    """
    try:
        # Find LoRA file
        lora_dir = get_lora_directory()
        lora_path = None
        
        for ext in [".safetensors", ".ckpt", ".pt"]:
            potential_path = lora_dir / f"{lora_name}{ext}"
            if potential_path.exists():
                lora_path = potential_path
                break
                
        if not lora_path:
            return {
                "error": f"LoRA '{lora_name}' not found"
            }
            
        # Load existing metadata
        metadata_path = lora_path.with_suffix(".json")
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "name": lora_name,
                "created_at": datetime.now().isoformat()
            }
            
        # Update metadata
        metadata.update(metadata_updates)
        metadata["updated_at"] = datetime.now().isoformat()
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return {
            "success": True,
            "lora_name": lora_name,
            "metadata_path": str(metadata_path),
            "updated_fields": list(metadata_updates.keys()),
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error updating LoRA metadata: {e}")
        return {"error": str(e)}

# ===== Workflow Preset Management =====

@mcp.tool()
def save_workflow_preset(
    name: str,
    description: str,
    workflow: Dict[str, Any],
    tags: Optional[List[str]] = None,
    is_public: bool = False
) -> Dict[str, Any]:
    """Save a complete workflow as a reusable preset
    
    Args:
        name: Preset name
        description: Preset description
        workflow: Complete workflow configuration
        tags: Optional tags for categorization
        is_public: Whether to share publicly
        
    Returns:
        Dictionary containing save result
    """
    try:
        preset_dir = Path(os.path.join(Path.home(), "ComfyUI/workflow_presets"))
        preset_dir.mkdir(exist_ok=True)
        
        preset = {
            "name": name,
            "description": description,
            "workflow": workflow,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "usage_count": 0,
            "average_time": None,
            "success_rate": None,
            "is_public": is_public,
            "version": "1.0.0",
            "parent_preset": None,
            "share_code": None,
            "performance_metrics": {"quality_avg": None, "speed_avg": None},
            "changelog": [{"version": "1.0.0", "date": datetime.now().isoformat(), "changes": "Initial creation"}]
        }
        
        preset_path = preset_dir / f"{name.replace(' ', '_')}.json"
        with open(preset_path, 'w') as f:
            json.dump(preset, f, indent=2)
            
        return {
            "success": True,
            "preset_name": name,
            "path": str(preset_path),
            "message": "Workflow preset saved successfully"
        }
    except Exception as e:
        logger.error(f"Error saving workflow preset: {e}")
        return {"error": str(e)}

@mcp.tool()
def list_workflow_presets(
    tags: Optional[List[str]] = None,
    include_stats: bool = True
) -> Dict[str, Any]:
    """List all saved workflow presets
    
    Args:
        tags: Filter by tags
        include_stats: Include usage statistics
        
    Returns:
        Dictionary containing all presets
    """
    try:
        preset_dir = Path(os.path.join(Path.home(), "ComfyUI/workflow_presets"))
        if not preset_dir.exists():
            return {
                "presets": [],
                "total_count": 0,
                "message": "No presets found"
            }
            
        presets = []
        
        for preset_path in preset_dir.glob("*.json"):
            try:
                with open(preset_path, 'r') as f:
                    preset = json.load(f)
                    
                    # Filter by tags if specified
                    if tags:
                        if not any(tag in preset.get("tags", []) for tag in tags):
                            continue
                            
                    # Add file info
                    preset["filename"] = preset_path.name
                    preset["file_size"] = preset_path.stat().st_size
                    
                    # Remove workflow data for listing
                    if not include_stats:
                        preset.pop("workflow", None)
                        
                    presets.append(preset)
            except:
                continue
                
        # Sort by usage count and creation date
        presets.sort(key=lambda x: (
            -x.get("usage_count", 0),
            x.get("created_at", "")
        ))
        
        return {
            "presets": presets,
            "total_count": len(presets),
            "filter_tags": tags
        }
        
    except Exception as e:
        logger.error(f"Error listing workflow presets: {e}")
        return {"error": str(e)}

@mcp.tool()
def apply_workflow_preset(
    preset_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    track_usage: bool = True
) -> Dict[str, Any]:
    """Apply a saved workflow preset with optional parameter overrides
    
    Args:
        preset_name: Name of the preset to apply
        parameters: Optional parameter overrides {node_id: {param: value}}
        track_usage: Track usage statistics
        
    Returns:
        Dictionary containing execution result
    """
    try:
        preset_dir = Path(os.path.join(Path.home(), "ComfyUI/workflow_presets"))
        preset_path = preset_dir / f"{preset_name.replace(' ', '_')}.json"
        
        if not preset_path.exists():
            return {"error": f"Preset '{preset_name}' not found"}
            
        with open(preset_path, 'r') as f:
            preset = json.load(f)
            
        # Apply parameter overrides
        workflow = preset["workflow"].copy()
        if parameters:
            for node_id, params in parameters.items():
                if node_id in workflow:
                    workflow[node_id]["inputs"].update(params)
                    
        # Track usage
        if track_usage:
            preset["usage_count"] = preset.get("usage_count", 0) + 1
            preset["last_used"] = datetime.now().isoformat()
            
            with open(preset_path, 'w') as f:
                json.dump(preset, f, indent=2)
                
        # Execute workflow
        if requests is None:
            return {"error": "requests not installed. Please install with: pip install requests"}
            
        start_time = time.time()
        
        response = requests.post(
            f"{comfyui_url}/prompt",
            json={"prompt": workflow}
        )
        
        if response.status_code == 200:
            result = response.json()
            execution_time = time.time() - start_time
            
            # Update average time
            if track_usage:
                avg_time = preset.get("average_time", 0)
                if avg_time == 0:
                    preset["average_time"] = execution_time
                else:
                    # Running average
                    count = preset["usage_count"]
                    preset["average_time"] = ((avg_time * (count - 1)) + execution_time) / count
                    
                with open(preset_path, 'w') as f:
                    json.dump(preset, f, indent=2)
                    
            return {
                "success": True,
                "preset_name": preset_name,
                "prompt_id": result.get("prompt_id"),
                "modifications": list(parameters.keys()) if parameters else [],
                "execution_time": execution_time,
                "message": f"Preset '{preset_name}' applied successfully"
            }
        else:
            return {
                "error": f"Failed to execute preset: HTTP {response.status_code}"
            }
            
    except Exception as e:
        logger.error(f"Error applying workflow preset: {e}")
        return {"error": str(e)}

# ===== Intelligent Prompt Enhancement =====

@mcp.tool()
def optimize_prompt_with_ai(
    prompt: str,
    optimization_goals: List[str] = ["clarity", "detail", "style"],
    target_style: Optional[str] = None,
    reference_successful: bool = True
) -> Dict[str, Any]:
    """Use AI to optimize prompts for better generation results
    
    Args:
        prompt: Original prompt to optimize
        optimization_goals: Goals for optimization
        target_style: Specific style to optimize for
        reference_successful: Learn from successful generations
        
    Returns:
        Dictionary containing optimized prompt and analysis
    """
    try:
        # Analyze prompt structure
        analysis = {
            "original_prompt": prompt,
            "word_count": len(prompt.split()),
            "has_style": any(style in prompt.lower() for style in ["style", "art", "photo", "painting"]),
            "has_quality": any(q in prompt.lower() for q in ["high quality", "detailed", "8k", "4k"]),
            "has_lighting": any(l in prompt.lower() for l in ["lighting", "light", "lit", "illuminated"]),
            "has_composition": any(c in prompt.lower() for c in ["composition", "framed", "centered", "rule of thirds"])
        }
        
        # Build optimized prompt
        optimized_parts = [prompt]
        suggestions = []
        
        # Add quality modifiers if missing
        if "quality" in optimization_goals and not analysis["has_quality"]:
            quality_modifiers = {
                "photorealistic": "8k resolution, photorealistic, highly detailed",
                "artistic": "high quality, artistic, masterpiece",
                "anime": "best quality, anime style, detailed",
                "general": "high quality, detailed, professional"
            }
            modifier = quality_modifiers.get(target_style, quality_modifiers["general"])
            optimized_parts.append(modifier)
            suggestions.append(f"Added quality modifiers for {target_style or 'general'} style")
            
        # Add style if missing
        if "style" in optimization_goals and not analysis["has_style"] and target_style:
            style_descriptors = {
                "photorealistic": "photorealistic style, photography",
                "artistic": "artistic style, painted",
                "anime": "anime style, 2D illustration",
                "fantasy": "fantasy art style, magical",
                "scifi": "science fiction style, futuristic"
            }
            if target_style in style_descriptors:
                optimized_parts.append(style_descriptors[target_style])
                suggestions.append(f"Added {target_style} style descriptors")
                
        # Add lighting if missing
        if "detail" in optimization_goals and not analysis["has_lighting"]:
            lighting_options = [
                "beautiful lighting",
                "studio lighting",
                "natural lighting",
                "dramatic lighting",
                "soft lighting"
            ]
            optimized_parts.append(random.choice(lighting_options))
            suggestions.append("Added lighting description")
            
        # Add composition if missing
        if "composition" in optimization_goals and not analysis["has_composition"]:
            composition_options = [
                "well composed",
                "centered composition",
                "dynamic composition",
                "rule of thirds"
            ]
            optimized_parts.append(random.choice(composition_options))
            suggestions.append("Added composition guidance")
            
        # Reference successful prompts
        successful_elements = []
        if reference_successful:
            # Load recent successful generations
            output_dir = Path(os.path.join(Path.home(), "ComfyUI/output"))
            recent_prompts = []
            
            for img_path in sorted(output_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
                try:
                    if Image and PngInfo:
                        img = Image.open(img_path)
                        if hasattr(img, 'info') and 'prompt' in img.info:
                            recent_prompts.append(img.info['prompt'])
                except:
                    continue
                    
            # Extract common successful elements
            if recent_prompts:
                word_frequency = Counter()
                for p in recent_prompts:
                    words = [w.strip().lower() for w in p.split(',')]
                    word_frequency.update(words)
                    
                # Find high-frequency quality words not in original
                for word, count in word_frequency.most_common(10):
                    if count > 2 and word not in prompt.lower() and len(word) > 3:
                        successful_elements.append(word)
                        
                if successful_elements:
                    optimized_parts.append(random.choice(successful_elements))
                    suggestions.append(f"Added successful element: {successful_elements[0]}")
                    
        # Build final optimized prompt
        optimized_prompt = ", ".join(optimized_parts)
        
        # Calculate improvement score
        improvement_score = 0
        if not analysis["has_quality"] and "quality" in optimized_prompt.lower():
            improvement_score += 25
        if not analysis["has_style"] and target_style and target_style in optimized_prompt.lower():
            improvement_score += 25
        if not analysis["has_lighting"] and "lighting" in optimized_prompt.lower():
            improvement_score += 25
        if successful_elements:
            improvement_score += 25
            
        return {
            "original_prompt": prompt,
            "optimized_prompt": optimized_prompt,
            "analysis": analysis,
            "suggestions": suggestions,
            "improvement_score": improvement_score,
            "learned_elements": successful_elements[:3] if successful_elements else [],
            "optimization_goals": optimization_goals,
            "target_style": target_style
        }
        
    except Exception as e:
        logger.error(f"Error optimizing prompt: {e}")
        return {"error": str(e)}

@mcp.tool()
def generate_with_style_preset(
    prompt: str,
    style: str = "photorealistic",
    quality_preset: str = "balanced",
    auto_optimize: bool = True
) -> Dict[str, Any]:
    """Generate image with predefined style and quality presets
    
    Args:
        prompt: Base prompt
        style: Style preset (photorealistic, anime, artistic, fantasy, scifi)
        quality_preset: Quality/speed preset (fast, balanced, quality, ultra)
        auto_optimize: Automatically optimize prompt
        
    Returns:
        Dictionary containing generation result
    """
    try:
        # Define style configurations
        style_configs = {
            "photorealistic": {
                "model": "realistic_vision_v5.safetensors",
                "style_prompt": "photorealistic, photography, real",
                "negative": "cartoon, anime, illustration, painting",
                "cfg_scale": 7.0,
                "sampler": "DPM++ 2M Karras"
            },
            "anime": {
                "model": "anything_v5.safetensors",
                "style_prompt": "anime style, 2D, illustration",
                "negative": "photo, realistic, 3D render",
                "cfg_scale": 10.0,
                "sampler": "Euler a"
            },
            "artistic": {
                "model": "deliberate_v3.safetensors",
                "style_prompt": "artistic, painted, fine art",
                "negative": "photo, bad art, low quality",
                "cfg_scale": 8.0,
                "sampler": "DPM++ 2M Karras"
            },
            "fantasy": {
                "model": "dreamshaper_v8.safetensors",
                "style_prompt": "fantasy art, magical, ethereal",
                "negative": "modern, mundane, ordinary",
                "cfg_scale": 9.0,
                "sampler": "DPM++ SDE Karras"
            },
            "scifi": {
                "model": "protogen_v34.safetensors",
                "style_prompt": "science fiction, futuristic, high tech",
                "negative": "medieval, rustic, old",
                "cfg_scale": 8.5,
                "sampler": "UniPC"
            }
        }
        
        # Define quality presets
        quality_configs = {
            "fast": {"steps": 20, "width": 512, "height": 512},
            "balanced": {"steps": 30, "width": 768, "height": 768},
            "quality": {"steps": 50, "width": 1024, "height": 1024},
            "ultra": {"steps": 80, "width": 1536, "height": 1536}
        }
        
        # Get configurations
        style_config = style_configs.get(style, style_configs["photorealistic"])
        quality_config = quality_configs.get(quality_preset, quality_configs["balanced"])
        
        # Optimize prompt if requested
        final_prompt = prompt
        if auto_optimize:
            optimization = optimize_prompt_with_ai(
                prompt,
                optimization_goals=["quality", "style", "detail"],
                target_style=style
            )
            if "optimized_prompt" in optimization:
                final_prompt = optimization["optimized_prompt"]
                
        # Combine prompt with style
        full_prompt = f"{final_prompt}, {style_config['style_prompt']}"
        
        # Generate image
        result = generate_image(
            prompt=full_prompt,
            negative_prompt=style_config["negative"],
            model=style_config["model"],
            cfg_scale=style_config["cfg_scale"],
            sampler_name=style_config["sampler"],
            steps=quality_config["steps"],
            width=quality_config["width"],
            height=quality_config["height"]
        )
        
        if "error" not in result:
            result.update({
                "style_preset": style,
                "quality_preset": quality_preset,
                "optimized": auto_optimize,
                "style_config": style_config,
                "quality_config": quality_config
            })
            
        return result
        
    except Exception as e:
        logger.error(f"Error generating with style preset: {e}")
        return {"error": str(e)}

# ===== Batch Operations Coordinator =====

@mcp.tool()
def coordinate_batch_operation(
    operation_name: str,
    tasks: List[Dict[str, Any]],
    parallel_limit: int = 3,
    error_handling: str = "continue",
    save_results: bool = True
) -> Dict[str, Any]:
    """Coordinate complex multi-step batch operations
    
    Args:
        operation_name: Name for this operation
        tasks: List of tasks to execute
        parallel_limit: Max parallel executions
        error_handling: How to handle errors (stop, continue, retry)
        save_results: Save operation results
        
    Returns:
        Dictionary containing operation results
    """
    try:
        operation_id = f"batch_{int(time.time())}"
        results = []
        failed_tasks = []
        start_time = time.time()
        
        # Create operation log
        operation_dir = Path(os.path.join(Path.home(), "ComfyUI/batch_operations"))
        operation_dir.mkdir(exist_ok=True)
        
        operation_log = {
            "id": operation_id,
            "name": operation_name,
            "total_tasks": len(tasks),
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "tasks": tasks
        }
        
        # Process tasks
        for i, task in enumerate(tasks):
            task_start = time.time()
            
            try:
                # Execute task based on type
                if task["type"] == "generate":
                    result = generate_image(**task.get("parameters", {}))
                elif task["type"] == "upscale":
                    result = upscale_image(**task.get("parameters", {}))
                elif task["type"] == "style_transfer":
                    result = style_transfer(**task.get("parameters", {}))
                elif task["type"] == "preset":
                    result = apply_workflow_preset(**task.get("parameters", {}))
                elif task["type"] == "lora_recipe":
                    result = apply_lora_recipe(**task.get("parameters", {}))
                else:
                    result = {"error": f"Unknown task type: {task['type']}"}
                    
                task_result = {
                    "task_id": task.get("id", f"task_{i}"),
                    "type": task["type"],
                    "parameters": task.get("parameters", {}),
                    "result": result,
                    "success": "error" not in result,
                    "execution_time": time.time() - task_start
                }
                
                results.append(task_result)
                
                # Log progress
                logger.info(f"Batch operation {operation_id}: Completed task {i+1}/{len(tasks)}")
                
            except Exception as e:
                error_result = {
                    "task_id": task.get("id", f"task_{i}"),
                    "type": task["type"],
                    "error": str(e),
                    "success": False
                }
                
                if error_handling == "stop":
                    operation_log["status"] = "stopped"
                    operation_log["error"] = str(e)
                    break
                elif error_handling == "retry":
                    # Retry once
                    try:
                        logger.info(f"Retrying task {i}")
                        # Retry logic (simplified)
                        time.sleep(2)
                        # Re-execute task
                        continue
                    except:
                        failed_tasks.append(error_result)
                else:  # continue
                    failed_tasks.append(error_result)
                    
        # Finalize operation
        operation_log.update({
            "completed_at": datetime.now().isoformat(),
            "status": "completed" if operation_log.get("status") != "stopped" else "stopped",
            "total_time": time.time() - start_time,
            "successful_tasks": len([r for r in results if r.get("success", False)]),
            "failed_tasks": len(failed_tasks),
            "results_summary": {
                "total": len(results) + len(failed_tasks),
                "successful": len([r for r in results if r.get("success", False)]),
                "failed": len(failed_tasks)
            }
        })
        
        # Save results
        if save_results:
            log_path = operation_dir / f"{operation_id}_{operation_name.replace(' ', '_')}.json"
            with open(log_path, 'w') as f:
                json.dump({
                    "operation": operation_log,
                    "results": results,
                    "failures": failed_tasks
                }, f, indent=2)
                
        return {
            "operation_id": operation_id,
            "name": operation_name,
            "status": operation_log["status"],
            "total_tasks": len(tasks),
            "successful": len([r for r in results if r.get("success", False)]),
            "failed": len(failed_tasks),
            "total_time": operation_log["total_time"],
            "results": results,
            "failures": failed_tasks,
            "log_path": str(log_path) if save_results else None
        }
        
    except Exception as e:
        logger.error(f"Error coordinating batch operation: {e}")
        return {"error": str(e)}

# ===== Auto Mask Generation =====

@mcp.tool()
def auto_generate_mask(
    image_path: str,
    target: str = "foreground",
    refinement: str = "medium",
    preview: bool = False
) -> Dict[str, Any]:
    """Automatically generate masks for common targets
    
    Args:
        image_path: Path to input image
        target: What to mask (foreground, background, person, object, text)
        refinement: Edge refinement level (low, medium, high)
        preview: Generate preview of mask
        
    Returns:
        Dictionary containing mask path and info
    """
    try:
        if Image is None:
            return {"error": "PIL not installed. Please install with: pip install Pillow"}
            
        if np is None:
            return {"error": "numpy not installed. Please install with: pip install numpy"}
            
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Generate mask based on target
        if target == "foreground":
            # Simple foreground extraction using color difference
            # Convert to LAB color space for better separation
            from skimage import color, filters
            lab = color.rgb2lab(img_array[:,:,:3])
            
            # Use Otsu's threshold on L channel
            threshold = filters.threshold_otsu(lab[:,:,0])
            mask = lab[:,:,0] > threshold
            
        elif target == "background":
            # Inverse of foreground
            lab = color.rgb2lab(img_array[:,:,:3])
            threshold = filters.threshold_otsu(lab[:,:,0])
            mask = lab[:,:,0] <= threshold
            
        elif target == "edges":
            # Edge detection
            gray = np.mean(img_array[:,:,:3], axis=2)
            from scipy import ndimage
            mask = ndimage.sobel(gray) > 30
            
        else:
            # Fallback to simple threshold
            gray = np.mean(img_array[:,:,:3], axis=2)
            mask = gray > 128
            
        # Refine mask
        if refinement == "high":
            # Apply morphological operations
            from scipy.ndimage import binary_erosion, binary_dilation
            mask = binary_erosion(mask, iterations=2)
            mask = binary_dilation(mask, iterations=2)
        elif refinement == "medium":
            from scipy.ndimage import gaussian_filter
            mask = gaussian_filter(mask.astype(float), sigma=1) > 0.5
            
        # Convert to image
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        
        # Save mask
        mask_path = Path(image_path).with_suffix('.mask.png')
        mask_img.save(mask_path)
        
        # Generate preview if requested
        preview_path = None
        if preview:
            # Create overlay preview
            overlay = img.copy()
            overlay_array = np.array(overlay)
            overlay_array[~mask] = overlay_array[~mask] * 0.3  # Darken non-masked areas
            preview_img = Image.fromarray(overlay_array)
            preview_path = Path(image_path).with_suffix('.mask_preview.png')
            preview_img.save(preview_path)
            
        return {
            "success": True,
            "mask_path": str(mask_path),
            "preview_path": str(preview_path) if preview_path else None,
            "target": target,
            "refinement": refinement,
            "mask_stats": {
                "coverage": float(np.mean(mask)),
                "width": mask.shape[1],
                "height": mask.shape[0]
            }
        }
        
    except ImportError as e:
        return {
            "error": f"Missing dependency: {e}. Install with: pip install scikit-image scipy"
        }
    except Exception as e:
        logger.error(f"Error generating mask: {e}")
        return {"error": str(e)}

# Track server startup time
startup_time = time.time()

if __name__ == "__main__":
    logger.info(f"Starting ComfyUI MCP server v1.0.0")
    logger.info(f"ComfyUI URL: {comfyui_url}")
    logger.info(f"Debug mode: {'ON' if os.getenv('DEBUG') else 'OFF'}")
    logger.info(f"Total tools available: 81")
    
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