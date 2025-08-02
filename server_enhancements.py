"""
Additional enhancement functions to be added to server.py
These expand the v2.6.0 tools with advanced capabilities
"""

import json
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

# ===== Workflow Preset Enhancements =====

@mcp.tool()
def share_workflow_preset(
    preset_name: str,
    expiry_hours: int = 72,
    password: Optional[str] = None,
    allowed_uses: Optional[int] = None
) -> Dict[str, Any]:
    """Share a workflow preset with time-limited access code
    
    Args:
        preset_name: Name of preset to share
        expiry_hours: Hours until share expires (default 72)
        password: Optional password protection
        allowed_uses: Maximum number of times it can be downloaded
        
    Returns:
        Dictionary containing share code and URL
    """
    try:
        preset_dir = Path(os.path.join(Path.home(), "ComfyUI/workflow_presets"))
        preset_path = preset_dir / f"{preset_name.replace(' ', '_')}.json"
        
        if not preset_path.exists():
            return {"error": f"Preset '{preset_name}' not found"}
            
        # Generate unique share code
        share_code = secrets.token_urlsafe(16)
        
        # Create share record
        shares_dir = preset_dir / "shares"
        shares_dir.mkdir(exist_ok=True)
        
        share_data = {
            "preset_name": preset_name,
            "share_code": share_code,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=expiry_hours)).isoformat(),
            "password_hash": hashlib.sha256(password.encode()).hexdigest() if password else None,
            "allowed_uses": allowed_uses,
            "used_count": 0,
            "access_log": []
        }
        
        share_path = shares_dir / f"{share_code}.json"
        with open(share_path, 'w') as f:
            json.dump(share_data, f, indent=2)
            
        return {
            "success": True,
            "share_code": share_code,
            "expires_at": share_data["expires_at"],
            "share_url": f"comfyui://preset/{share_code}",
            "password_protected": password is not None,
            "usage_limit": allowed_uses
        }
        
    except Exception as e:
        logger.error(f"Error sharing workflow preset: {e}")
        return {"error": str(e)}

@mcp.tool()
def fork_workflow_preset(
    preset_name: str,
    new_name: str,
    modifications: Optional[Dict[str, Any]] = None,
    maintain_lineage: bool = True
) -> Dict[str, Any]:
    """Create a fork of existing workflow preset
    
    Args:
        preset_name: Original preset to fork
        new_name: Name for the forked preset
        modifications: Initial modifications to apply
        maintain_lineage: Track parent-child relationship
        
    Returns:
        Dictionary containing fork result
    """
    try:
        preset_dir = Path(os.path.join(Path.home(), "ComfyUI/workflow_presets"))
        original_path = preset_dir / f"{preset_name.replace(' ', '_')}.json"
        
        if not original_path.exists():
            return {"error": f"Original preset '{preset_name}' not found"}
            
        # Load original preset
        with open(original_path, 'r') as f:
            original = json.load(f)
            
        # Create forked preset
        forked = original.copy()
        forked["name"] = new_name
        forked["created_at"] = datetime.now().isoformat()
        forked["version"] = "1.0.0"
        forked["parent_preset"] = preset_name if maintain_lineage else None
        forked["fork_source"] = {
            "preset": preset_name,
            "version": original.get("version", "1.0.0"),
            "date": datetime.now().isoformat()
        }
        forked["changelog"] = [{
            "version": "1.0.0",
            "date": datetime.now().isoformat(),
            "changes": f"Forked from {preset_name}"
        }]
        
        # Apply modifications if provided
        if modifications:
            workflow = forked["workflow"]
            for node_id, changes in modifications.items():
                if node_id in workflow:
                    workflow[node_id]["inputs"].update(changes)
                    
        # Save forked preset
        fork_path = preset_dir / f"{new_name.replace(' ', '_')}.json"
        with open(fork_path, 'w') as f:
            json.dump(forked, f, indent=2)
            
        return {
            "success": True,
            "preset_name": new_name,
            "parent_preset": preset_name,
            "path": str(fork_path),
            "modifications_applied": len(modifications) if modifications else 0
        }
        
    except Exception as e:
        logger.error(f"Error forking workflow preset: {e}")
        return {"error": str(e)}

# ===== Prompt Optimization Enhancements =====

@mcp.tool()
def track_generation_metrics(
    prompt_id: str,
    metrics: Dict[str, float],
    user_rating: Optional[int] = None,
    tags: Optional[List[str]] = None,
    save_to_history: bool = True
) -> Dict[str, Any]:
    """Track performance metrics for generated images
    
    Args:
        prompt_id: ID of the generation
        metrics: Performance metrics (quality, adherence, speed, etc.)
        user_rating: Optional 1-5 star rating
        tags: Tags for categorization
        save_to_history: Save for learning
        
    Returns:
        Dictionary containing tracking result
    """
    try:
        metrics_dir = Path(os.path.join(Path.home(), "ComfyUI/generation_metrics"))
        metrics_dir.mkdir(exist_ok=True)
        
        # Create metrics record
        metrics_data = {
            "prompt_id": prompt_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "user_rating": user_rating,
            "tags": tags or [],
            "performance_score": sum(metrics.values()) / len(metrics) if metrics else 0
        }
        
        # Save to history if requested
        if save_to_history:
            history_path = metrics_dir / f"{prompt_id}.json"
            with open(history_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            # Update learning database
            learning_db = metrics_dir / "learning_database.json"
            if learning_db.exists():
                with open(learning_db, 'r') as f:
                    db = json.load(f)
            else:
                db = {"total_generations": 0, "patterns": {}, "successful_elements": []}
                
            db["total_generations"] += 1
            if metrics_data["performance_score"] > 0.8:
                # Track successful patterns
                # This is simplified - real implementation would use ML
                db["successful_elements"].append({
                    "score": metrics_data["performance_score"],
                    "tags": tags,
                    "timestamp": metrics_data["timestamp"]
                })
                
            with open(learning_db, 'w') as f:
                json.dump(db, f, indent=2)
                
        return {
            "success": True,
            "prompt_id": prompt_id,
            "performance_score": metrics_data["performance_score"],
            "metrics_saved": save_to_history,
            "recommendation": "High quality generation" if metrics_data["performance_score"] > 0.8 else "Consider optimization"
        }
        
    except Exception as e:
        logger.error(f"Error tracking generation metrics: {e}")
        return {"error": str(e)}

@mcp.tool()
def create_prompt_ab_test(
    test_name: str,
    base_prompt: str,
    variations: List[Dict[str, Any]],
    samples_per_variation: int = 5,
    test_metrics: List[str] = ["quality", "speed", "coherence"]
) -> Dict[str, Any]:
    """Create A/B test for prompt variations
    
    Args:
        test_name: Name for the test
        base_prompt: Base prompt to test variations of
        variations: List of variations to test
        samples_per_variation: Samples to generate per variation
        test_metrics: Metrics to track
        
    Returns:
        Dictionary containing test configuration
    """
    try:
        tests_dir = Path(os.path.join(Path.home(), "ComfyUI/ab_tests"))
        tests_dir.mkdir(exist_ok=True)
        
        # Create test configuration
        test_config = {
            "test_name": test_name,
            "created_at": datetime.now().isoformat(),
            "base_prompt": base_prompt,
            "variations": variations,
            "samples_per_variation": samples_per_variation,
            "test_metrics": test_metrics,
            "status": "configured",
            "results": {},
            "total_samples": len(variations) * samples_per_variation
        }
        
        # Generate variation prompts
        test_prompts = []
        for i, variation in enumerate(variations):
            for sample in range(samples_per_variation):
                modified_prompt = base_prompt
                for key, value in variation.items():
                    if key in base_prompt:
                        modified_prompt = modified_prompt.replace(f"{{{key}}}", str(value))
                    else:
                        modified_prompt += f", {value}"
                        
                test_prompts.append({
                    "variation_id": i,
                    "sample_id": sample,
                    "prompt": modified_prompt,
                    "variation_params": variation
                })
                
        test_config["test_prompts"] = test_prompts
        
        # Save test configuration
        test_path = tests_dir / f"{test_name.replace(' ', '_')}.json"
        with open(test_path, 'w') as f:
            json.dump(test_config, f, indent=2)
            
        return {
            "success": True,
            "test_name": test_name,
            "total_prompts": len(test_prompts),
            "variations": len(variations),
            "test_id": hashlib.md5(test_name.encode()).hexdigest()[:8],
            "next_step": "Run batch_generate with test_prompts to execute test"
        }
        
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}")
        return {"error": str(e)}

# ===== Batch Operation Enhancements =====

@mcp.tool()
def create_batch_dependency_chain(
    operation_name: str,
    tasks: List[Dict[str, Any]],
    dependencies: Dict[str, List[str]],
    optimization_mode: str = "parallel"  # parallel, sequential, smart
) -> Dict[str, Any]:
    """Create batch operation with dependency management
    
    Args:
        operation_name: Name for the operation
        tasks: List of tasks with IDs
        dependencies: Dict of task_id -> [dependency_ids]
        optimization_mode: How to optimize execution
        
    Returns:
        Dictionary containing execution plan
    """
    try:
        # Build dependency graph
        task_map = {task["id"]: task for task in tasks}
        execution_order = []
        completed = set()
        
        # Topological sort for dependency resolution
        while len(completed) < len(tasks):
            for task in tasks:
                task_id = task["id"]
                if task_id in completed:
                    continue
                    
                task_deps = dependencies.get(task_id, [])
                if all(dep in completed for dep in task_deps):
                    execution_order.append(task_id)
                    completed.add(task_id)
                    
        # Group by parallelization opportunity
        execution_groups = []
        current_group = []
        
        for task_id in execution_order:
            task_deps = dependencies.get(task_id, [])
            can_parallel = all(
                dep in [t for group in execution_groups for t in group]
                for dep in task_deps
            )
            
            if can_parallel or not task_deps:
                current_group.append(task_id)
            else:
                if current_group:
                    execution_groups.append(current_group)
                current_group = [task_id]
                
        if current_group:
            execution_groups.append(current_group)
            
        # Create execution plan
        plan = {
            "operation_name": operation_name,
            "total_tasks": len(tasks),
            "execution_groups": execution_groups,
            "parallelization_opportunities": len([g for g in execution_groups if len(g) > 1]),
            "estimated_time_savings": f"{len(tasks) - len(execution_groups)} task intervals",
            "dependency_graph": dependencies,
            "optimization_mode": optimization_mode
        }
        
        return {
            "success": True,
            "execution_plan": plan,
            "can_parallelize": len(execution_groups) < len(tasks),
            "critical_path": max(len(g) for g in execution_groups)
        }
        
    except Exception as e:
        logger.error(f"Error creating dependency chain: {e}")
        return {"error": str(e)}

# ===== Mask Generation Enhancements =====

@mcp.tool()
def generate_semantic_mask(
    image_path: str,
    semantic_prompt: str,
    confidence_threshold: float = 0.7,
    return_all_matches: bool = False
) -> Dict[str, Any]:
    """Generate mask from natural language description
    
    Args:
        image_path: Path to input image
        semantic_prompt: Natural language description (e.g., "the red car on the left")
        confidence_threshold: Minimum confidence for matches
        return_all_matches: Return all matching regions
        
    Returns:
        Dictionary containing semantic mask(s)
    """
    try:
        if Image is None:
            return {"error": "PIL not installed. Please install with: pip install Pillow"}
            
        # This is a simplified implementation
        # Real implementation would use CLIP or similar models
        
        # Parse semantic elements from prompt
        semantic_elements = {
            "colors": ["red", "blue", "green", "yellow", "black", "white"],
            "positions": ["left", "right", "center", "top", "bottom"],
            "objects": ["car", "person", "tree", "building", "sky", "ground"]
        }
        
        found_elements = {
            "colors": [],
            "positions": [],
            "objects": []
        }
        
        prompt_lower = semantic_prompt.lower()
        for category, items in semantic_elements.items():
            for item in items:
                if item in prompt_lower:
                    found_elements[category].append(item)
                    
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Create basic mask (simplified)
        mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=bool)
        
        # Apply position-based masking
        h, w = mask.shape
        if "left" in found_elements["positions"]:
            mask[:, :w//2] = True
        elif "right" in found_elements["positions"]:
            mask[:, w//2:] = True
        elif "center" in found_elements["positions"]:
            mask[h//4:3*h//4, w//4:3*w//4] = True
            
        # Save mask
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_path = Path(image_path).with_suffix('.semantic_mask.png')
        mask_img.save(mask_path)
        
        return {
            "success": True,
            "mask_path": str(mask_path),
            "semantic_elements": found_elements,
            "confidence": 0.5,  # Simplified
            "prompt": semantic_prompt,
            "message": "Note: This is a simplified implementation. Real semantic masking requires advanced models."
        }
        
    except Exception as e:
        logger.error(f"Error generating semantic mask: {e}")
        return {"error": str(e)}

# ===== Integration Tools =====

@mcp.tool()
def create_workflow_from_metrics(
    metric_threshold: Dict[str, float],
    time_period_days: int = 30,
    workflow_type: str = "optimized"
) -> Dict[str, Any]:
    """Create optimized workflow based on historical metrics
    
    Args:
        metric_threshold: Minimum metrics to include
        time_period_days: Days of history to analyze
        workflow_type: Type of workflow to create
        
    Returns:
        Dictionary containing optimized workflow
    """
    try:
        metrics_dir = Path(os.path.join(Path.home(), "ComfyUI/generation_metrics"))
        learning_db = metrics_dir / "learning_database.json"
        
        if not learning_db.exists():
            return {"error": "No metrics history available"}
            
        with open(learning_db, 'r') as f:
            db = json.load(f)
            
        # Analyze successful patterns
        successful_elements = db.get("successful_elements", [])
        
        # Filter by time period
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        recent_successes = [
            elem for elem in successful_elements
            if datetime.fromisoformat(elem["timestamp"]) > cutoff_date
        ]
        
        # Extract common patterns
        common_tags = {}
        for elem in recent_successes:
            for tag in elem.get("tags", []):
                common_tags[tag] = common_tags.get(tag, 0) + 1
                
        # Build optimized workflow
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "high quality, professional",
                    "clip": ["1", 1]
                }
            }
        }
        
        # Add nodes based on successful patterns
        if "upscale" in common_tags and common_tags["upscale"] > 5:
            workflow["10"] = {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {"upscale_model": "ESRGAN", "image": ["prev", 0]}
            }
            
        return {
            "success": True,
            "workflow": workflow,
            "based_on_successes": len(recent_successes),
            "common_patterns": dict(sorted(common_tags.items(), key=lambda x: x[1], reverse=True)[:5]),
            "optimization_score": sum(elem["score"] for elem in recent_successes) / len(recent_successes) if recent_successes else 0
        }
        
    except Exception as e:
        logger.error(f"Error creating workflow from metrics: {e}")
        return {"error": str(e)}

print("Enhancement tools ready to be integrated into server.py!")
print("These add advanced capabilities to the v2.6.0 core tools:")