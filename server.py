#!/usr/bin/env python3
import json
import logging
import os
import sys
import glob
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
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
    version="1.2.0"
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
            "server_version": "1.2.0",
            "comfyui_url": comfyui_url,
            "available_models": comfyui_client.available_models or [],
            "status": "connected",
            "total_tools": 16
        }
    except Exception as e:
        logger.error(f"Error getting server info: {e}")
        return {
            "server_version": "1.2.0",
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
        import requests
        
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
        import requests
        
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
        import requests
        
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
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        
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
def get_node_info(node_type: Optional[str] = None) -> Dict[str, Any]:
    """Get detailed information about available ComfyUI nodes
    
    Args:
        node_type: Specific node type to get info for, or None for all nodes
        
    Returns:
        Dictionary containing node information
    """
    try:
        import requests
        
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
        import requests
        
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

if __name__ == "__main__":
    logger.info(f"Starting ComfyUI MCP server v1.2.0")
    logger.info(f"ComfyUI URL: {comfyui_url}")
    logger.info(f"Debug mode: {'ON' if os.getenv('DEBUG') else 'OFF'}")
    logger.info(f"Total tools available: 16")
    
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