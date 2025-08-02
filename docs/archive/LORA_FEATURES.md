# ComfyUI MCP Server v2.5.0 - LoRA Management Features

## Overview
Version 2.5.0 introduces comprehensive LoRA model management capabilities, inspired by the ComfyUI-Lora-Manager project. These tools streamline organizing, downloading, and applying LoRA models with rich metadata support.

## New LoRA Management Tools (8 tools)

### 1. **list_loras_detailed**
Browse all LoRA models with detailed metadata and preview information.

**Parameters:**
- `sort_by`: Sort field (name, date, size, rating)
- `filter_tag`: Filter by tag/category
- `include_metadata`: Include detailed metadata

**Features:**
- Lists all .safetensors LoRA files
- Shows file size, modification date
- Includes metadata from .json files
- Preview image paths
- Trigger words and tags
- CivitAI ratings

### 2. **download_lora_from_civitai**
Download LoRA models from CivitAI with automatic metadata extraction.

**Parameters:**
- `model_url`: CivitAI model URL or ID
- `trigger_words`: Optional trigger words list
- `custom_name`: Custom filename (without extension)

**Features:**
- Extracts model ID from CivitAI URLs
- Downloads latest model version
- Saves metadata automatically
- Downloads preview images
- Preserves trigger words

### 3. **get_lora_info**
Get detailed information about a specific LoRA model.

**Parameters:**
- `lora_name`: Name of the LoRA model (without extension)

**Returns:**
- File information (size, path, dates)
- Metadata (description, tags, triggers)
- Preview image details
- Recent usage in generations

### 4. **save_lora_recipe**
Save combinations of LoRAs with weights as reusable recipes.

**Parameters:**
- `recipe_name`: Name for the recipe
- `loras`: List of LoRA configurations with weights
- `description`: Recipe description
- `base_model`: Recommended base model
- `sample_prompt`: Example prompt

**Example:**
```python
save_lora_recipe(
    recipe_name="anime_portrait",
    loras=[
        {"name": "anime_style", "weight": 0.8, "trigger": "anime style"},
        {"name": "detailed_eyes", "weight": 0.5, "trigger": "detailed eyes"}
    ],
    description="Perfect for anime portraits with emphasis on eyes",
    base_model="sd_xl_base_1.0.safetensors"
)
```

### 5. **list_lora_recipes**
List all saved LoRA recipes.

**Returns:**
- All saved recipes with metadata
- Creation dates
- LoRA combinations
- Total weights

### 6. **apply_lora_recipe**
Apply a saved LoRA recipe to generate a ComfyUI workflow.

**Parameters:**
- `recipe_name`: Name of the recipe to apply
- `base_prompt`: Base prompt to use
- `include_triggers`: Whether to include trigger words

**Features:**
- Builds complete ComfyUI workflow
- Chains multiple LoRA nodes
- Includes trigger words automatically
- Preserves weight configurations

### 7. **search_loras**
Search for LoRA models by various criteria.

**Parameters:**
- `query`: Search query
- `search_fields`: Fields to search (name, trigger_words, tags, description)
- `min_rating`: Minimum rating filter

**Features:**
- Multi-field search
- Rating filtering
- Relevance-based sorting
- Case-insensitive matching

### 8. **update_lora_metadata**
Update metadata for existing LoRA models.

**Parameters:**
- `lora_name`: Name of the LoRA model
- `metadata_updates`: Dictionary of fields to update

**Example:**
```python
update_lora_metadata(
    "anime_style_v2",
    {
        "description": "High quality anime style LoRA",
        "tags": ["anime", "style", "2D"],
        "trigger_words": ["anime style", "2D illustration"],
        "custom_notes": "Works best with DPM++ 2M Karras"
    }
)
```

## Directory Structure

```
ComfyUI/
├── models/
│   └── loras/                    # LoRA model files
│       ├── model1.safetensors    # LoRA model
│       ├── model1.json           # Metadata
│       └── model1.png            # Preview image
└── lora_recipes/                 # Saved recipes
    └── recipe_name.json          # Recipe configuration
```

## Metadata Format

Each LoRA can have an associated .json file with metadata:

```json
{
    "name": "Anime Style v2",
    "description": "High quality anime style transformation",
    "trigger_words": ["anime style", "2D illustration"],
    "base_model": "SDXL",
    "tags": ["anime", "style", "artistic"],
    "civitai_id": "123456",
    "rating": 4.8,
    "downloaded_at": "2024-01-15T10:30:00",
    "custom_notes": "User notes here"
}
```

## Recipe Format

LoRA recipes are saved as JSON files:

```json
{
    "name": "Perfect Portrait",
    "description": "Ideal combination for portrait photography",
    "loras": [
        {
            "name": "portrait_master",
            "weight": 0.8,
            "trigger": "portrait photography"
        },
        {
            "name": "skin_details",
            "weight": 0.4,
            "trigger": "detailed skin"
        }
    ],
    "base_model": "realistic_vision_v5.safetensors",
    "sample_prompt": "portrait photography of a person, detailed skin",
    "created_at": "2024-01-15T14:20:00",
    "total_weight": 1.2,
    "combined_triggers": "portrait photography, detailed skin"
}
```

## Usage Examples

### 1. Browse and Filter LoRAs
```python
# List all LoRAs sorted by rating
list_loras_detailed(sort_by="rating", include_metadata=True)

# Filter by tag
list_loras_detailed(filter_tag="anime", sort_by="date")
```

### 2. Download from CivitAI
```python
# Download a LoRA from CivitAI
download_lora_from_civitai(
    "https://civitai.com/models/123456",
    trigger_words=["epic fantasy", "detailed armor"],
    custom_name="fantasy_armor_v2"
)
```

### 3. Create and Use Recipes
```python
# Save a recipe
save_lora_recipe(
    "cyberpunk_style",
    [
        {"name": "cyberpunk_2077", "weight": 0.7, "trigger": "cyberpunk style"},
        {"name": "neon_lights", "weight": 0.5, "trigger": "neon lighting"}
    ],
    description="Cyberpunk aesthetic with neon accents"
)

# Apply the recipe
apply_lora_recipe(
    "cyberpunk_style",
    "a character in a futuristic city",
    include_triggers=True
)
```

### 4. Search and Update
```python
# Search for specific LoRAs
search_loras("fantasy", ["name", "tags"], min_rating=4.0)

# Update metadata
update_lora_metadata(
    "old_lora_v1",
    {
        "description": "Updated description",
        "tags": ["updated", "tested"],
        "custom_notes": "Tested with latest ComfyUI"
    }
)
```

## Benefits

1. **Organization**: Keep LoRAs organized with metadata and tags
2. **Discovery**: Search and filter large LoRA collections
3. **Recipes**: Save and reuse successful LoRA combinations
4. **Integration**: Direct CivitAI download with metadata
5. **Workflow Generation**: Automatic ComfyUI workflow creation
6. **Documentation**: Track trigger words and usage notes

## Requirements

- ComfyUI with LoRA support
- Models directory at `ComfyUI/models/loras/`
- Optional: Internet connection for CivitAI downloads

## Future Enhancements

- Batch metadata updates
- LoRA compatibility checking
- Recipe sharing/import/export
- Performance metrics tracking
- Automatic trigger word detection
- Visual recipe builder