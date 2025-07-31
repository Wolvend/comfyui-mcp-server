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
    version="2.0.0"
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
            "server_version": "2.0.0",
            "comfyui_url": comfyui_url,
            "available_models": comfyui_client.available_models or [],
            "status": "connected",
            "total_tools": 42
        }
    except Exception as e:
        logger.error(f"Error getting server info: {e}")
        return {
            "server_version": "2.0.0",
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

# Video Generation Tools
@mcp.tool()
def generate_video(
    prompt: str,
    duration: float = 2.0,
    fps: int = 24,
    width: Optional[int] = None,
    height: Optional[int] = None,
    model: Optional[str] = None,
    seed: Optional[int] = None,
    motion_strength: float = 1.0
) -> Dict[str, Any]:
    """Generate a video from text prompt using available video models
    
    Args:
        prompt: Text description of the video to generate
        duration: Video duration in seconds (default: 2.0)
        fps: Frames per second (default: 24)
        width: Video width (default: 512)
        height: Video height (default: 512)
        model: Video model to use (cosmos, mochi, ltxv)
        seed: Random seed for reproducibility
        motion_strength: How much motion to include (0.0-2.0)
        
    Returns:
        Dictionary containing video URL or error
    """
    try:
        # Implementation would use CosmosPredict2ImageToVideoLatent or similar
        width = width or 512
        height = height or 512
        
        # For now, using image generation as placeholder
        result = generate_image(
            prompt=f"video frame: {prompt}",
            width=width,
            height=height,
            seed=seed
        )
        
        return {
            "status": "placeholder",
            "message": "Video generation coming soon",
            "frame_preview": result
        }
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return {"error": str(e)}

@mcp.tool()
def image_to_video(
    image_path: str,
    duration: float = 2.0,
    fps: int = 24,
    motion_type: str = "auto",
    motion_strength: float = 1.0
) -> Dict[str, Any]:
    """Animate a static image into a video
    
    Args:
        image_path: Path to the input image
        duration: Video duration in seconds
        fps: Frames per second
        motion_type: Type of motion (auto, zoom, pan, rotate)
        motion_strength: Intensity of motion effect
        
    Returns:
        Dictionary containing animated video URL
    """
    try:
        # Would use image animation nodes
        return {
            "status": "placeholder",
            "message": "Image to video animation coming soon",
            "input_image": image_path
        }
    except Exception as e:
        logger.error(f"Error animating image: {e}")
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
        import websocket
        import threading
        import json as json_lib
        
        ws_url = comfyui_url.replace("http", "ws") + "/ws"
        
        # Get current status first
        try:
            import requests
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
        return {
            "status": "placeholder",
            "message": "Preview streaming coming soon",
            "prompt_id": prompt_id,
            "interval": preview_interval
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
        import requests
        
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
        import requests
        
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

if __name__ == "__main__":
    logger.info(f"Starting ComfyUI MCP server v2.0.0")
    logger.info(f"ComfyUI URL: {comfyui_url}")
    logger.info(f"Debug mode: {'ON' if os.getenv('DEBUG') else 'OFF'}")
    logger.info(f"Total tools available: 42")
    
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