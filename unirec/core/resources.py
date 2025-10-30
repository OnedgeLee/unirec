"""Generic resource management interfaces for pipeline components.

This module provides generic interfaces for type-safe resource management:
- Resources[T]: Generic interface for runtime resource containers
- ResourcesSpec[T]: Generic interface for resource specifications (from YAML)
- ResourcesBuilder[T, S]: Generic interface for constructing Resources from ResourcesSpec

The type parameter T represents the concrete resource type.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


# Type variable for the concrete Resources type
TResources = TypeVar("TResources", bound="Resources")
# Type variable for the concrete ResourcesSpec type
TResourcesSpec = TypeVar("TResourcesSpec", bound="ResourcesSpec")


class Resources(ABC):
    """Abstract base interface for runtime resource containers.
    
    Concrete subclasses should define typed attributes for specific resources.
    This provides better type hints than dict[str, Any] for IDE autocomplete.
    
    Example:
        class ItemResources(Resources):
            item_embeddings: NDArray[np.float32]
            item_ids: list[int]
            categories: dict[int, str]
    """
    pass


class ResourcesSpec(ABC):
    """Abstract base interface for resource specifications.
    
    This represents resource configuration before loading (typically from YAML).
    Concrete subclasses should define fields matching their Resources counterpart.
    
    Example:
        @dataclass
        class ItemResourcesSpec(ResourcesSpec):
            item_embeddings: str  # path to embeddings file
            item_ids: str | list[int]  # path or direct list
            categories: str | dict[int, str]  # path or direct dict
    """
    pass


class ResourcesBuilder(ABC, Generic[TResources, TResourcesSpec]):
    """Abstract base interface for building Resources from ResourcesSpec.
    
    Concrete implementations should handle loading resources from paths
    and specifications, converting ResourcesSpec into runtime Resources.
    
    Example:
        class ItemResourcesBuilder(ResourcesBuilder[ItemResources, ItemResourcesSpec]):
            @classmethod
            def build(cls, spec: ItemResourcesSpec) -> ItemResources:
                # Load embeddings from path
                embeddings = np.load(spec.item_embeddings)
                # Create runtime resources
                return ItemResources(
                    item_embeddings=embeddings,
                    item_ids=spec.item_ids if isinstance(spec.item_ids, list)
                              else load_ids(spec.item_ids),
                    categories=spec.categories if isinstance(spec.categories, dict)
                                else load_categories(spec.categories)
                )
    """
    
    @classmethod
    @abstractmethod
    def build(cls, spec: TResourcesSpec) -> TResources:
        """Build Resources from ResourcesSpec.
        
        Args:
            spec: ResourcesSpec with resource specifications
            
        Returns:
            Resources with loaded resources
        """
        ...
