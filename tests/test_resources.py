"""Tests for unirec.core.resources module."""

import pytest
import numpy as np
from unirec.core.resources import Resources, ResourcesSpec, ResourcesBuilder
from unirec.core.interfaces import Component
from unirec.core.state import PipelineState
import unirec.data.context.user_context as user_ctx

UserProfileContext = user_ctx.UserProfileContext
UserSessionContext = user_ctx.UserSessionContext


class TestableComponent(Component):
    """Concrete component for testing with Resources."""

    component_kind = "testable"

    def run(self, state: PipelineState) -> PipelineState:
        return state


def test_resources_initialization():
    """Test Resources initialization."""
    resources = Resources()
    assert resources.item_embeddings is None
    assert resources.item_ids is None
    assert resources.item_memmap is None
    assert resources.categories == {}
    assert resources.custom == {}


def test_resources_with_values():
    """Test Resources with initial values."""
    embeddings = np.random.rand(10, 128).astype(np.float32)
    item_ids = [1, 2, 3, 4, 5]
    categories = {1: "cat1", 2: "cat2"}
    
    resources = Resources(
        item_embeddings=embeddings,
        item_ids=item_ids,
        categories=categories,
    )
    
    assert resources.item_embeddings is not None
    assert np.array_equal(resources.item_embeddings, embeddings)
    assert resources.item_ids == item_ids
    assert resources.categories == categories


def test_resources_get_method():
    """Test Resources.get() method."""
    resources = Resources()
    resources.custom["my_resource"] = "value"
    
    # Test getting custom resource
    assert resources.get("my_resource") == "value"
    
    # Test getting with default
    assert resources.get("missing", "default") == "default"
    
    # Test getting None value
    assert resources.get("item_embeddings") is None


def test_resources_dict_access():
    """Test Resources with dict-like access."""
    embeddings = np.random.rand(5, 64).astype(np.float32)
    resources = Resources(item_embeddings=embeddings)
    
    # Test __getitem__
    assert np.array_equal(resources["item_embeddings"], embeddings)
    
    # Test __setitem__ for custom
    resources["custom_key"] = "custom_value"
    assert resources["custom_key"] == "custom_value"
    
    # Test __contains__
    assert "item_embeddings" in resources
    assert "custom_key" in resources
    assert "missing_key" not in resources


def test_resources_dict_access_missing_key():
    """Test Resources raises KeyError for missing keys."""
    resources = Resources()
    
    with pytest.raises(KeyError):
        _ = resources["missing_key"]


def test_resources_spec_initialization():
    """Test ResourcesSpec initialization."""
    spec = ResourcesSpec()
    assert spec.item_embeddings is None
    assert spec.item_ids is None
    assert spec.item_memmap is None
    assert spec.categories is None
    assert spec.custom == {}


def test_resources_spec_with_values():
    """Test ResourcesSpec with values."""
    spec = ResourcesSpec(
        item_embeddings="path/to/embeddings.npy",
        item_ids=[1, 2, 3],
        categories={"cat1": "category1"},
    )
    
    assert spec.item_embeddings == "path/to/embeddings.npy"
    assert spec.item_ids == [1, 2, 3]
    assert spec.categories == {"cat1": "category1"}


def test_resources_spec_from_dict():
    """Test ResourcesSpec.from_dict()."""
    config = {
        "item_embeddings": "path/to/emb.npy",
        "item_ids": [10, 20, 30],
        "categories": {"a": "catA"},
        "custom_resource": "custom_value",
    }
    
    spec = ResourcesSpec.from_dict(config)
    
    assert spec.item_embeddings == "path/to/emb.npy"
    assert spec.item_ids == [10, 20, 30]
    assert spec.categories == {"a": "catA"}
    assert spec.custom["custom_resource"] == "custom_value"


def test_resources_spec_from_dict_empty():
    """Test ResourcesSpec.from_dict() with empty dict."""
    spec = ResourcesSpec.from_dict({})
    
    assert spec.item_embeddings is None
    assert spec.item_ids is None
    assert spec.item_memmap is None
    assert spec.categories is None
    assert spec.custom == {}


def test_resources_builder_build_empty():
    """Test ResourcesBuilder.build() with empty spec."""
    spec = ResourcesSpec()
    resources = ResourcesBuilder.build(spec)
    
    assert resources.item_embeddings is None
    assert resources.item_ids is None
    assert resources.item_memmap is None
    assert resources.categories == {}
    assert resources.custom == {}


def test_resources_builder_with_item_ids_list():
    """Test ResourcesBuilder with item IDs as list."""
    spec = ResourcesSpec(item_ids=[1, 2, 3, 4, 5])
    resources = ResourcesBuilder.build(spec)
    
    assert resources.item_ids == [1, 2, 3, 4, 5]


def test_resources_builder_with_categories_dict():
    """Test ResourcesBuilder with categories as dict."""
    categories = {1: "cat1", 2: "cat2", 3: "cat3"}
    spec = ResourcesSpec(categories=categories)
    resources = ResourcesBuilder.build(spec)
    
    assert resources.categories == categories


def test_resources_builder_with_custom():
    """Test ResourcesBuilder with custom resources."""
    spec = ResourcesSpec()
    spec.custom = {"model": "path/to/model", "config": {"param": "value"}}
    
    resources = ResourcesBuilder.build(spec)
    
    assert resources.custom["model"] == "path/to/model"
    assert resources.custom["config"] == {"param": "value"}


def test_component_setup_with_resources_object():
    """Test Component.setup() with Resources object."""
    comp = TestableComponent()
    
    embeddings = np.random.rand(10, 64).astype(np.float32)
    resources = Resources(item_embeddings=embeddings, item_ids=[1, 2, 3])
    
    comp.setup(resources)
    
    assert comp.resources is resources
    assert "item_embeddings" in comp.resources
    assert "item_ids" in comp.resources


def test_component_setup_with_dict():
    """Test Component.setup() still works with dict (backward compatibility)."""
    comp = TestableComponent()
    
    resources_dict = {"model": "path/to/model", "index": "path/to/index"}
    comp.setup(resources_dict)
    
    assert comp.resources == resources_dict
    assert comp.resources["model"] == "path/to/model"


def test_component_require_resource_with_resources_object():
    """Test Component.require_resource() with Resources object."""
    comp = TestableComponent()
    
    embeddings = np.random.rand(5, 32).astype(np.float32)
    resources = Resources(item_embeddings=embeddings)
    resources.custom["my_resource"] = 42
    
    comp.setup(resources)
    
    # Test require_resource with standard field
    result = comp.require_resource("item_embeddings", np.ndarray)
    assert np.array_equal(result, embeddings)
    
    # Test require_resource with custom field
    result = comp.require_resource("my_resource", int)
    assert result == 42


def test_component_optional_resource_with_resources_object():
    """Test Component.optional_resource() with Resources object."""
    comp = TestableComponent()
    
    resources = Resources()
    resources.custom["optional_res"] = "value"
    
    comp.setup(resources)
    
    # Test optional_resource that exists
    result = comp.optional_resource("optional_res", str)
    assert result == "value"
    
    # Test optional_resource that doesn't exist
    result = comp.optional_resource("missing", str)
    assert result is None


def test_integration_yaml_to_resources():
    """Test full integration: YAML config -> ResourcesSpec -> Resources."""
    # Simulate YAML config
    yaml_config = {
        "item_embeddings": None,  # Would be path in real scenario
        "item_ids": [100, 200, 300],
        "categories": {100: "electronics", 200: "books"},
        "custom_data": "custom_value",
    }
    
    # Step 1: Create ResourcesSpec from config
    spec = ResourcesSpec.from_dict(yaml_config)
    
    assert spec.item_ids == [100, 200, 300]
    assert spec.categories == {100: "electronics", 200: "books"}
    assert spec.custom["custom_data"] == "custom_value"
    
    # Step 2: Build Resources from spec
    resources = ResourcesBuilder.build(spec)
    
    assert resources.item_ids == [100, 200, 300]
    assert resources.categories == {100: "electronics", 200: "books"}
    assert resources.custom["custom_data"] == "custom_value"
    
    # Step 3: Use Resources in Component
    comp = TestableComponent()
    comp.setup(resources)
    
    assert comp.require_resource("item_ids", list) == [100, 200, 300]
    assert comp.require_resource("categories", dict) == {100: "electronics", 200: "books"}
