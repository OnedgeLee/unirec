# Resources Class Usage Guide

This document explains how to use the new `Resources`, `ResourcesSpec`, and `ResourcesBuilder` classes introduced to provide better type hints and structure for resource management in pipelines.

## Overview

The resource management system consists of three main classes:

1. **`Resources`**: Runtime container for loaded resources with type hints
2. **`ResourcesSpec`**: Configuration specification (typically loaded from YAML)
3. **`ResourcesBuilder`**: Constructs `Resources` from `ResourcesSpec`

The data flow is: **YAML → ResourcesSpec → ResourcesBuilder → Resources**

## Basic Usage

### 1. Creating Resources Directly

```python
import numpy as np
from unirec.core.resources import Resources

# Create resources with typed fields
embeddings = np.random.rand(100, 128).astype(np.float32)
item_ids = [1, 2, 3, 4, 5]
categories = {1: "electronics", 2: "books"}

resources = Resources(
    item_embeddings=embeddings,
    item_ids=item_ids,
    categories=categories
)

# Access resources using dict-like interface
print(resources["item_embeddings"].shape)  # (100, 128)
print(resources.get("item_ids"))  # [1, 2, 3, 4, 5]

# Add custom resources
resources["custom_data"] = "my_value"
```

### 2. Loading from YAML Configuration

```python
from unirec.core.resources import ResourcesSpec, ResourcesBuilder

# Example YAML configuration:
# resources:
#   item_embeddings: path/to/embeddings.npy
#   item_ids: [1, 2, 3, 4, 5]
#   categories:
#     1: "electronics"
#     2: "books"
#   custom_resource: "custom_value"

# Step 1: Load YAML config (using your config loader)
yaml_config = {
    "item_embeddings": "resources/item_emb.npy",
    "item_ids": [1, 2, 3, 4, 5],
    "categories": {1: "electronics", 2: "books"},
    "custom_resource": "custom_value"
}

# Step 2: Create ResourcesSpec from config
spec = ResourcesSpec.from_dict(yaml_config)

# Step 3: Build Resources from spec
resources = ResourcesBuilder.build(spec)

# Now resources are loaded and ready to use
```

### 3. Using Resources in Components

```python
from unirec.core.interfaces import Component

class MyComponent(Component):
    def setup(self, resources):
        super().setup(resources)
        
        # Access resources with type checking
        embeddings = self.require_resource("item_embeddings", np.ndarray)
        item_ids = self.optional_resource("item_ids", list)
        
        # Standard fields have proper type hints
        if "item_embeddings" in resources:
            self.embeddings = resources["item_embeddings"]
```

## Type Safety Benefits

The `Resources` class provides better type hints compared to plain dictionaries:

```python
# Before (plain dict):
resources: dict[str, Any] = {
    "item_embeddings": embeddings,  # No type hint
    "item_ids": item_ids,           # No type hint
}

# After (Resources class):
resources = Resources(
    item_embeddings=embeddings,  # Type: NDArray[np.float32] | None
    item_ids=item_ids,            # Type: list[int] | None
    categories=categories,        # Type: dict[int, str]
)
```

## Standard Resource Fields

The `Resources` class defines common resource types used in recommendation pipelines:

- **`item_embeddings`**: Item embeddings array (N, d) - `NDArray[np.float32] | None`
- **`item_ids`**: List of item IDs aligned to embeddings - `list[int] | None`
- **`item_memmap`**: Memory-mapped item array - `NDArray[np.float32] | None`
- **`categories`**: Mapping of item_id to category - `dict[int, str]`
- **`custom`**: Additional custom resources - `dict[str, Any]`

## Backward Compatibility

The implementation maintains full backward compatibility with existing code using dictionaries:

```python
# Old code still works
resources_dict = {"model": "path/to/model", "index": "path/to/index"}
component.setup(resources_dict)  # Still works!

# New code with Resources
resources_obj = Resources()
resources_obj.custom["model"] = "path/to/model"
component.setup(resources_obj)  # Also works!
```

## Complete Example

Here's a complete example showing the recommended workflow:

```python
import yaml
from unirec.core.resources import ResourcesSpec, ResourcesBuilder
from unirec.core.interfaces import Component

# 1. Load YAML configuration
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# 2. Create ResourcesSpec from YAML
resources_config = config.get("resources", {})
spec = ResourcesSpec.from_dict(resources_config)

# 3. Build Resources
resources = ResourcesBuilder.build(spec)

# 4. Use in pipeline components
class MyRetriever(Component):
    def setup(self, resources):
        super().setup(resources)
        # Access with type safety
        self.embeddings = self.require_resource("item_embeddings", np.ndarray)
        self.item_ids = self.optional_resource("item_ids", list)

retriever = MyRetriever()
retriever.setup(resources)
```

## Migration Guide

To migrate existing code to use the new `Resources` class:

1. **No changes required for basic usage** - existing dict-based code continues to work
2. **Optional upgrade path**:
   - Create `ResourcesSpec` from your YAML configs
   - Use `ResourcesBuilder.build()` to create `Resources` objects
   - Pass `Resources` objects to `component.setup()` instead of dicts

3. **Benefit**: Better IDE autocomplete and type checking with the new classes

## Testing

The `Resources` classes are fully tested with 18 comprehensive tests covering:
- Resource initialization and access
- Dict-like interface
- ResourcesSpec creation from config
- ResourcesBuilder construction
- Component integration
- Backward compatibility

Run tests with: `pytest tests/test_resources.py -v`
