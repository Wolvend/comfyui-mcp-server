"""
ComfyUI Client Module

Handles communication with ComfyUI API for the MCP Server.
Provides workflow execution, monitoring, and result handling.

Copyright 2024 ComfyUI MCP Team
Part of ComfyUI MCP Server v1.0.0
"""

import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComfyUIClient")

DEFAULT_MAPPING = {
    "prompt": ("6", "text"),
    "width": ("5", "width"),
    "height": ("5", "height"),
    "model": ("4", "ckpt_name")
}

# Workflow-specific parameter mappings
WORKFLOW_MAPPINGS = {
    "basic_api_test": DEFAULT_MAPPING,
    "controlnet_workflow": {
        "USER_PROMPT": ("2", "text"),
        "USER_NEGATIVE_PROMPT": ("3", "text"),
        "CONTROL_IMAGE": ("4", "image"),
        "CONTROL_TYPE": ("5", "control_net_name"),
        "WIDTH": ("7", "width"),
        "HEIGHT": ("7", "height"),
        "CONTROL_STRENGTH": ("6", "strength"),
        "SEED": ("8", "seed")
    },
    "upscale_workflow": {
        "INPUT_IMAGE": ("1", "image"),
        "UPSCALE_MODEL": ("2", "model_name"),
        "SEED": ("4", "seed")
    },
    "inpaint_workflow": {
        "USER_PROMPT": ("2", "text"),
        "USER_NEGATIVE_PROMPT": ("3", "text"),
        "INPUT_IMAGE": ("4", "image"),
        "MASK_IMAGE": ("5", "image"),
        "EXPAND_MASK": ("6", "expand"),
        "STRENGTH": ("8", "denoise"),
        "SEED": ("8", "seed")
    },
    "video_gen_workflow": {
        "USER_PROMPT": ("2", "text"),
        "WIDTH": ("4", "width"),
        "HEIGHT": ("4", "height"),
        "FRAMES": ("4", "frames"),
        "SEED": ("5", "seed"),
        "FPS": ("7", "fps")
    }
}

class ComfyUIClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.available_models = self._get_available_models()

    def _get_available_models(self):
        """Fetch list of available checkpoint models from ComfyUI"""
        try:
            response = requests.get(f"{self.base_url}/object_info/CheckpointLoaderSimple")
            if response.status_code != 200:
                logger.warning("Failed to fetch model list; using default handling")
                return []
            data = response.json()
            models = data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            
            # If no models found via API, try to detect from filesystem
            if not models:
                models = self._scan_model_directory()
            
            logger.info(f"Available models: {models}")
            return models
        except Exception as e:
            logger.warning(f"Error fetching models: {e}")
            return self._scan_model_directory()

    def _scan_model_directory(self):
        """Scan filesystem for model files as fallback"""
        import os
        from pathlib import Path
        
        try:
            # Common ComfyUI model paths
            possible_paths = [
                "/home/wolvend/Desktop/ComfyUI/models/checkpoints",
                "../../../models/checkpoints",
                "./ComfyUI/models/checkpoints"
            ]
            
            for path_str in possible_paths:
                path = Path(path_str)
                if path.exists():
                    models = []
                    for ext in ['.safetensors', '.ckpt', '.pth']:
                        models.extend([f.name for f in path.glob(f'*{ext}')])
                    if models:
                        logger.info(f"Found models in {path}: {models}")
                        return models
            
            logger.warning("No model files found in standard directories")
            return []
        except Exception as e:
            logger.warning(f"Error scanning model directory: {e}")
            return []

    def generate_image(self, prompt, width, height, workflow_id="basic_api_test", model=None, retry_count=3, **kwargs):
        """Generate image with automatic retry on failure"""
        last_error = None
        
        for attempt in range(retry_count):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{retry_count}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                
                return self._generate_image_attempt(prompt, width, height, workflow_id, model, **kwargs)
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt == retry_count - 1:
                    logger.error(f"All {retry_count} attempts failed. Last error: {e}")
                    raise last_error
                    
        raise last_error
    
    def _generate_image_attempt(self, prompt, width, height, workflow_id="basic_api_test", model=None, **kwargs):
        try:
            workflow_file = f"workflows/{workflow_id}.json"
            with open(workflow_file, "r") as f:
                workflow = json.load(f)

            params = {"prompt": prompt, "width": width, "height": height}
            if model:
                # Validate or correct model name
                if model.endswith("'"):  # Strip accidental quote
                    model = model.rstrip("'")
                    logger.info(f"Corrected model name: {model}")
                if self.available_models and model not in self.available_models:
                    raise Exception(f"Model '{model}' not in available models: {self.available_models}")
                params["model"] = model

            # Use workflow-specific mapping if available
            mapping = WORKFLOW_MAPPINGS.get(workflow_id, DEFAULT_MAPPING)
            
            for param_key, value in params.items():
                if param_key in mapping:
                    node_id, input_key = mapping[param_key]
                    if node_id not in workflow:
                        logger.warning(f"Node {node_id} not found in workflow {workflow_id}, skipping parameter {param_key}")
                        continue
                    if "inputs" not in workflow[node_id]:
                        workflow[node_id]["inputs"] = {}
                    workflow[node_id]["inputs"][input_key] = value
                    logger.debug(f"Set {workflow_id}[{node_id}][{input_key}] = {value}")
                else:
                    logger.debug(f"Parameter {param_key} not in mapping for {workflow_id}, passing as kwargs")

            logger.info(f"Submitting workflow {workflow_id} to ComfyUI...")
            response = requests.post(f"{self.base_url}/prompt", json={"prompt": workflow})
            if response.status_code != 200:
                raise Exception(f"Failed to queue workflow: {response.status_code} - {response.text}")

            prompt_id = response.json()["prompt_id"]
            logger.info(f"Queued workflow with prompt_id: {prompt_id}")

            max_attempts = 60  # Increased timeout
            poll_interval = 1
            
            for attempt in range(max_attempts):
                try:
                    # Check queue first for better error messages
                    queue_response = requests.get(f"{self.base_url}/queue", timeout=5)
                    if queue_response.status_code == 200:
                        queue_data = queue_response.json()
                        running = queue_data.get("queue_running", [])
                        pending = queue_data.get("queue_pending", [])
                        
                        # Check if prompt is still in queue
                        in_queue = any(item[1] == prompt_id for item in running + pending if len(item) > 1)
                        if not in_queue and attempt > 5:  # Allow some time for queue processing
                            # Not in queue, check history
                            pass
                        elif in_queue:
                            logger.debug(f"Prompt {prompt_id} still in queue (attempt {attempt + 1})")
                    
                    # Check history
                    history_response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
                    if history_response.status_code != 200:
                        logger.warning(f"History request failed: {history_response.status_code}")
                        time.sleep(poll_interval)
                        continue
                        
                    history = history_response.json()
                    if history.get(prompt_id):
                        history_data = history[prompt_id]
                        
                        # Check for errors in history
                        if "status" in history_data and history_data["status"].get("status_str") == "error":
                            error_messages = history_data["status"].get("messages", [])
                            raise Exception(f"ComfyUI workflow error: {error_messages}")
                        
                        outputs = history_data.get("outputs", {})
                        if outputs:
                            logger.info("Workflow outputs: %s", json.dumps(outputs, indent=2))
                            image_node = next((nid for nid, out in outputs.items() if "images" in out), None)
                            if not image_node:
                                raise Exception(f"No output node with images found: {outputs}")
                            images = outputs[image_node]["images"]
                            if not images:
                                raise Exception("No images in output")
                            image_filename = images[0]["filename"]
                            image_url = f"{self.base_url}/view?filename={image_filename}&subfolder=&type=output"
                            logger.info(f"Generated image URL: {image_url}")
                            return image_url
                        else:
                            logger.debug(f"No outputs yet for {prompt_id} (attempt {attempt + 1})")
                    
                    time.sleep(poll_interval)
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"Request timeout on attempt {attempt + 1}")
                    time.sleep(poll_interval * 2)
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                    time.sleep(poll_interval * 2)
                    
            raise Exception(f"Workflow {prompt_id} didn't complete within {max_attempts} seconds")

        except FileNotFoundError:
            raise Exception(f"Workflow file '{workflow_file}' not found")
        except KeyError as e:
            raise Exception(f"Workflow error - invalid node or input: {e}")
        except requests.RequestException as e:
            raise Exception(f"ComfyUI API error: {e}")