"""Resource management for pipeline components.

This module provides type-safe resource management through:
- Resources: Runtime container for loaded resources
- ResourcesSpec: Configuration specification (loaded from YAML)
- ResourcesBuilder: Constructs Resources from ResourcesSpec
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray


@dataclass
class Resources:
    """Type-safe container for runtime resources.
    
    This class provides type hints and validation for common resource types
    used in recommendation pipelines.
    
    Attributes:
        item_embeddings: Item embeddings array (N, d)
        item_ids: Optional list of item IDs aligned to embeddings
        item_memmap: Optional memory-mapped item array
        categories: Optional mapping of item_id to category
        custom: Additional custom resources as key-value pairs
    """
    
    item_embeddings: NDArray[np.float32] | None = None
    item_ids: list[int] | None = None
    item_memmap: NDArray[np.float32] | None = None
    categories: dict[int, str] = field(default_factory=dict)
    custom: dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a resource by key, checking both standard and custom resources.
        
        Args:
            key: Resource key to retrieve
            default: Default value if key not found
            
        Returns:
            Resource value or default
        """
        # Check standard attributes first
        if hasattr(self, key):
            val = getattr(self, key)
            if val is not None:
                return val
        
        # Check custom resources
        return self.custom.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get a resource by key using dict-like access.
        
        Args:
            key: Resource key to retrieve
            
        Returns:
            Resource value
            
        Raises:
            KeyError: If key not found
        """
        # Check standard attributes first
        if hasattr(self, key):
            val = getattr(self, key)
            if val is not None:
                return val
        
        # Check custom resources
        if key in self.custom:
            return self.custom[key]
        
        raise KeyError(f"Resource '{key}' not found")
    
    def __setitem__(self, key: str, value: Any):
        """Set a resource by key using dict-like access.
        
        Args:
            key: Resource key to set
            value: Resource value
        """
        # Check if it's a standard attribute
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.custom[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if a resource exists.
        
        Args:
            key: Resource key to check
            
        Returns:
            True if resource exists, False otherwise
        """
        if hasattr(self, key) and getattr(self, key) is not None:
            return True
        return key in self.custom


@dataclass
class ResourcesSpec:
    """Configuration specification for resources (typically loaded from YAML).
    
    This class represents the resource configuration before resources are loaded.
    It stores paths and configuration that will be used to construct Resources.
    
    Attributes:
        item_embeddings: Path or specification for item embeddings
        item_ids: Path or specification for item IDs
        item_memmap: Path or specification for memory-mapped item array
        categories: Path or specification for categories
        custom: Additional custom resource specifications
    """
    
    item_embeddings: str | dict[str, Any] | None = None
    item_ids: str | list[int] | None = None
    item_memmap: str | dict[str, Any] | None = None
    categories: str | dict[int, str] | None = None
    custom: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ResourcesSpec":
        """Create ResourcesSpec from a configuration dictionary.
        
        Args:
            config: Configuration dictionary (typically from YAML)
            
        Returns:
            ResourcesSpec instance
        """
        # Extract known keys
        spec = cls(
            item_embeddings=config.get("item_embeddings"),
            item_ids=config.get("item_ids"),
            item_memmap=config.get("item_memmap"),
            categories=config.get("categories"),
        )
        
        # Store remaining keys in custom
        known_keys = {"item_embeddings", "item_ids", "item_memmap", "categories"}
        for key, value in config.items():
            if key not in known_keys:
                spec.custom[key] = value
        
        return spec


class ResourcesBuilder:
    """Builds Resources from ResourcesSpec.
    
    This class handles loading resources from paths and specifications,
    converting ResourcesSpec into a runtime Resources object.
    """
    
    @staticmethod
    def _load_array(source: str | dict[str, Any]) -> NDArray[np.float32]:
        """Load numpy array from path or specification.
        
        Args:
            source: File path string or dict with 'path' key
            
        Returns:
            Loaded numpy array
        """
        if isinstance(source, dict):
            path = source.get("path", source.get("file"))
        else:
            path = source
        
        if path is None:
            raise ValueError("Array source must have a path")
        
        return np.load(path).astype(np.float32, copy=False)
    
    @staticmethod
    def _load_categories(source: str | dict[int, str]) -> dict[int, str]:
        """Load categories from path or dict.
        
        Args:
            source: File path string or dict mapping
            
        Returns:
            Category dictionary
        """
        if isinstance(source, dict):
            return source
        
        # If string, assume it's a path to load
        # For now, return empty dict if string path (can be extended later)
        return {}
    
    @classmethod
    def build(cls, spec: ResourcesSpec) -> Resources:
        """Build Resources from ResourcesSpec.
        
        Args:
            spec: ResourcesSpec with resource specifications
            
        Returns:
            Resources with loaded resources
        """
        resources = Resources()
        
        # Load item embeddings
        if spec.item_embeddings is not None:
            resources.item_embeddings = cls._load_array(spec.item_embeddings)
        
        # Load item IDs
        if spec.item_ids is not None:
            if isinstance(spec.item_ids, list):
                resources.item_ids = spec.item_ids
            else:
                # Could be a path to a file with IDs
                try:
                    resources.item_ids = list(np.load(spec.item_ids))
                except Exception:
                    # If loading fails, set to None
                    resources.item_ids = None
        
        # Load item memmap
        if spec.item_memmap is not None:
            resources.item_memmap = cls._load_array(spec.item_memmap)
        
        # Load categories
        if spec.categories is not None:
            resources.categories = cls._load_categories(spec.categories)
        
        # Copy custom resources
        resources.custom = spec.custom.copy()
        
        return resources
