"""Tests for unirec.core.resources and item_resources modules."""

import numpy as np

from unirec.core.resources import Resources, ResourcesSpec, ResourcesBuilder
from unirec.core.item_resources import ItemResources, ItemResourcesSpec, ItemResourcesBuilder
from unirec.core.interfaces import Component
from unirec.core.state import PipelineState
import unirec.data.context.user_context as user_ctx

UserProfileContext = user_ctx.UserProfileContext
UserSessionContext = user_ctx.UserSessionContext


# Test component using generic Resources
class TestComponent(Component[ItemResources]):
    """Concrete component for testing with ItemResources."""

    component_kind = "testable"

    def run(self, state: PipelineState) -> PipelineState:
        return state


def test_resources_interface():
    """Test Resources interface is abstract."""
    assert issubclass(ItemResources, Resources)
    
    # Resources is abstract, so we can't instantiate it directly
    # but we can instantiate concrete subclasses
    item_resources = ItemResources()
    assert isinstance(item_resources, Resources)


def test_item_resources_initialization():
    """Test ItemResources initialization."""
    resources = ItemResources()
    assert resources.item_embeddings is None
    assert resources.item_ids is None
    assert resources.item_memmap is None
    assert resources.categories == {}
    assert resources.extra == {}


def test_item_resources_with_values():
    """Test ItemResources with initial values."""
    embeddings = np.random.rand(10, 128).astype(np.float32)
    item_ids = [1, 2, 3, 4, 5]
    categories = {1: "cat1", 2: "cat2"}
    
    resources = ItemResources(
        item_embeddings=embeddings,
        item_ids=item_ids,
        categories=categories,
    )
    
    assert resources.item_embeddings is not None
    assert np.array_equal(resources.item_embeddings, embeddings)
    assert resources.item_ids == item_ids
    assert resources.categories == categories


def test_item_resources_attribute_access():
    """Test ItemResources attribute access (not dict-like)."""
    embeddings = np.random.rand(5, 64).astype(np.float32)
    resources = ItemResources(item_embeddings=embeddings)
    
    # Access via attributes
    assert resources.item_embeddings is not None
    assert np.array_equal(resources.item_embeddings, embeddings)
    
    # Should not have dict-like methods
    assert not hasattr(resources, '__getitem__')
    assert not hasattr(resources, '__setitem__')


def test_resources_spec_interface():
    """Test ResourcesSpec interface is abstract."""
    assert issubclass(ItemResourcesSpec, ResourcesSpec)
    
    item_spec = ItemResourcesSpec()
    assert isinstance(item_spec, ResourcesSpec)


def test_item_resources_spec_initialization():
    """Test ItemResourcesSpec initialization."""
    spec = ItemResourcesSpec()
    assert spec.item_embeddings is None
    assert spec.item_ids is None
    assert spec.item_memmap is None
    assert spec.categories is None
    assert spec.extra == {}


def test_item_resources_spec_with_values():
    """Test ItemResourcesSpec with values."""
    spec = ItemResourcesSpec(
        item_embeddings="path/to/embeddings.npy",
        item_ids=[1, 2, 3],
        categories={"cat1": "category1"},
    )
    
    assert spec.item_embeddings == "path/to/embeddings.npy"
    assert spec.item_ids == [1, 2, 3]
    assert spec.categories == {"cat1": "category1"}


def test_item_resources_spec_from_dict():
    """Test ItemResourcesSpec.from_dict()."""
    config = {
        "item_embeddings": "path/to/emb.npy",
        "item_ids": [10, 20, 30],
        "categories": {"a": "catA"},
        "custom_resource": "custom_value",
    }
    
    spec = ItemResourcesSpec.from_dict(config)
    
    assert spec.item_embeddings == "path/to/emb.npy"
    assert spec.item_ids == [10, 20, 30]
    assert spec.categories == {"a": "catA"}
    assert spec.extra["custom_resource"] == "custom_value"


def test_item_resources_spec_from_dict_empty():
    """Test ItemResourcesSpec.from_dict() with empty dict."""
    spec = ItemResourcesSpec.from_dict({})
    
    assert spec.item_embeddings is None
    assert spec.item_ids is None
    assert spec.item_memmap is None
    assert spec.categories is None
    assert spec.extra == {}


def test_resources_builder_interface():
    """Test ResourcesBuilder interface."""
    assert issubclass(ItemResourcesBuilder, ResourcesBuilder)
    
    # Builder should have build method
    assert hasattr(ItemResourcesBuilder, 'build')


def test_item_resources_builder_build_empty():
    """Test ItemResourcesBuilder.build() with empty spec."""
    spec = ItemResourcesSpec()
    resources = ItemResourcesBuilder.build(spec)
    
    assert resources.item_embeddings is None
    assert resources.item_ids is None
    assert resources.item_memmap is None
    assert resources.categories == {}
    assert resources.extra == {}


def test_item_resources_builder_with_item_ids_list():
    """Test ItemResourcesBuilder with item IDs as list."""
    spec = ItemResourcesSpec(item_ids=[1, 2, 3, 4, 5])
    resources = ItemResourcesBuilder.build(spec)
    
    assert resources.item_ids == [1, 2, 3, 4, 5]


def test_item_resources_builder_with_categories_dict():
    """Test ItemResourcesBuilder with categories as dict."""
    categories = {1: "cat1", 2: "cat2", 3: "cat3"}
    spec = ItemResourcesSpec(categories=categories)
    resources = ItemResourcesBuilder.build(spec)
    
    assert resources.categories == categories


def test_item_resources_builder_with_extra():
    """Test ItemResourcesBuilder with extra resources."""
    spec = ItemResourcesSpec()
    spec.extra = {"model": "path/to/model", "config": {"param": "value"}}
    
    resources = ItemResourcesBuilder.build(spec)
    
    assert resources.extra["model"] == "path/to/model"
    assert resources.extra["config"] == {"param": "value"}


def test_component_setup_with_item_resources():
    """Test Component.setup() with ItemResources object."""
    comp = TestComponent()
    
    embeddings = np.random.rand(10, 64).astype(np.float32)
    resources = ItemResources(item_embeddings=embeddings, item_ids=[1, 2, 3])
    
    comp.setup(resources)
    
    assert comp.resources is resources
    # Access via attributes
    assert comp.resources.item_embeddings is not None
    assert comp.resources.item_ids == [1, 2, 3]


def test_component_setup_with_dict_backward_compat():
    """Test Component.setup() still works with dict (backward compatibility)."""
    comp = TestComponent()
    
    resources_dict = {"model": "path/to/model", "index": "path/to/index"}
    comp.setup(resources_dict)
    
    assert comp.resources == resources_dict
    assert comp.resources["model"] == "path/to/model"


def test_component_require_resource_with_item_resources():
    """Test Component.require_resource() with ItemResources object."""
    comp = TestComponent()
    
    embeddings = np.random.rand(5, 32).astype(np.float32)
    resources = ItemResources(item_embeddings=embeddings)
    resources.extra["my_resource"] = 42
    
    comp.setup(resources)
    
    # Test require_resource with standard field
    result = comp.require_resource("item_embeddings", np.ndarray)
    assert np.array_equal(result, embeddings)
    
    # Test require_resource with extra field
    result = comp.require_resource("my_resource", int)
    assert result == 42


def test_component_optional_resource_with_item_resources():
    """Test Component.optional_resource() with ItemResources object."""
    comp = TestComponent()
    
    resources = ItemResources()
    resources.extra["optional_res"] = "value"
    
    comp.setup(resources)
    
    # Test optional_resource that exists
    result = comp.optional_resource("optional_res", str)
    assert result == "value"
    
    # Test optional_resource that doesn't exist
    result = comp.optional_resource("missing", str)
    assert result is None


def test_integration_yaml_to_item_resources():
    """Test full integration: YAML config -> ItemResourcesSpec -> ItemResources."""
    # Simulate YAML config
    yaml_config = {
        "item_embeddings": None,  # Would be path in real scenario
        "item_ids": [100, 200, 300],
        "categories": {100: "electronics", 200: "books"},
        "custom_data": "custom_value",
    }
    
    # Step 1: Create ItemResourcesSpec from config
    spec = ItemResourcesSpec.from_dict(yaml_config)
    
    assert spec.item_ids == [100, 200, 300]
    assert spec.categories == {100: "electronics", 200: "books"}
    assert spec.extra["custom_data"] == "custom_value"
    
    # Step 2: Build ItemResources from spec
    resources = ItemResourcesBuilder.build(spec)
    
    assert resources.item_ids == [100, 200, 300]
    assert resources.categories == {100: "electronics", 200: "books"}
    assert resources.extra["custom_data"] == "custom_value"
    
    # Step 3: Use ItemResources in Component
    comp = TestComponent()
    comp.setup(resources)
    
    assert comp.require_resource("item_ids", list) == [100, 200, 300]
    assert comp.require_resource("categories", dict) == {100: "electronics", 200: "books"}


def test_type_safety_benefit():
    """Test that ItemResources provides better type hints than dict."""
    # With ItemResources, IDE can autocomplete attributes
    resources = ItemResources(
        item_embeddings=np.zeros((10, 64), dtype=np.float32),
        item_ids=[1, 2, 3],
    )
    
    # These have proper type hints
    assert isinstance(resources.item_embeddings, np.ndarray)
    assert isinstance(resources.item_ids, list)
    
    # Contrast with dict[str, Any] which has no type info for comparison
    # (dict_resources["item_embeddings"] would have type Any with no IDE help)
