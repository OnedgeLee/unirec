"""Concrete implementation of resource management for item-based recommendation.

This module provides a concrete implementation of the generic resource interfaces
for managing item embeddings, IDs, and related resources.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray

from .resources import Resources, ResourcesSpec, ResourcesBuilder


@dataclass
class ItemResources(Resources):
    """Concrete runtime container for item-related resources.
    
    This provides type-safe access to item resources with proper type hints.
    IDE autocomplete will work on these typed attributes.
    
    Attributes:
        item_embeddings: Item embeddings array (N, d)
        item_ids: Optional list of item IDs aligned to embeddings
        item_memmap: Optional memory-mapped item array
        categories: Mapping of item_id to category
        extra: Additional custom resources as key-value pairs
    """
    
    item_embeddings: NDArray[np.float32] | None = None
    item_ids: list[int] | None = None
    item_memmap: NDArray[np.float32] | None = None
    categories: dict[int, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ItemResourcesSpec(ResourcesSpec):
    """Concrete specification for item resources (typically from YAML).
    
    This represents the resource configuration before resources are loaded.
    Fields can be paths (strings) or direct values.
    
    Attributes:
        item_embeddings: Path to embeddings file or dict with 'path' key
        item_ids: Path to IDs file or direct list of IDs
        item_memmap: Path to memory-mapped array or dict with 'path' key
        categories: Path to categories file or direct dict
        extra: Additional custom resource specifications
    """
    
    item_embeddings: str | dict[str, Any] | None = None
    item_ids: str | list[int] | None = None
    item_memmap: str | dict[str, Any] | None = None
    categories: str | dict[int, str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ItemResourcesSpec":
        """Create ItemResourcesSpec from a configuration dictionary.
        
        This is typically used to load specs from YAML configuration.
        
        Args:
            config: Configuration dictionary (from YAML)
            
        Returns:
            ItemResourcesSpec instance
        """
        # Extract known keys
        spec = cls(
            item_embeddings=config.get("item_embeddings"),
            item_ids=config.get("item_ids"),
            item_memmap=config.get("item_memmap"),
            categories=config.get("categories"),
        )
        
        # Store remaining keys in extra
        known_keys = {"item_embeddings", "item_ids", "item_memmap", "categories"}
        for key, value in config.items():
            if key not in known_keys:
                spec.extra[key] = value
        
        return spec


class ItemResourcesBuilder(ResourcesBuilder[ItemResources, ItemResourcesSpec]):
    """Builder for constructing ItemResources from ItemResourcesSpec.
    
    This handles loading resources from file paths and specifications,
    converting ItemResourcesSpec into runtime ItemResources.
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
    def build(cls, spec: ItemResourcesSpec) -> ItemResources:
        """Build ItemResources from ItemResourcesSpec.
        
        Args:
            spec: ItemResourcesSpec with resource specifications
            
        Returns:
            ItemResources with loaded resources
        """
        resources = ItemResources()
        
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
        
        # Copy extra resources
        resources.extra = spec.extra.copy()
        
        return resources
