"""Tests for unirec.core.interfaces Loadable functionality."""

import pytest
from unirec.core.interfaces import Loadable


class TestableLoadable(Loadable):
    """Concrete loadable for testing."""

    pass


def test_loadable_initialization():
    """Test Loadable initialization with params."""
    loadable = TestableLoadable(param1="value1", param2=42)
    assert loadable.params["param1"] == "value1"
    assert loadable.params["param2"] == 42


def test_loadable_empty_initialization():
    """Test Loadable initialization without params."""
    loadable = TestableLoadable()
    assert loadable.params == {}
    assert loadable.resources == {}


def test_loadable_require_param():
    """Test require_param with valid parameter."""
    loadable = TestableLoadable(my_param=123)
    value = loadable.require_param("my_param", int)
    assert value == 123


def test_loadable_require_param_missing():
    """Test require_param raises KeyError when missing."""
    loadable = TestableLoadable()
    with pytest.raises(KeyError, match="missing required param"):
        loadable.require_param("missing_param", str)


def test_loadable_require_param_wrong_type():
    """Test require_param raises TypeError for wrong type."""
    loadable = TestableLoadable(my_param="string")
    with pytest.raises(TypeError, match="must be"):
        loadable.require_param("my_param", int)


def test_loadable_require_param_with_default():
    """Test require_param uses default when missing."""
    loadable = TestableLoadable()
    value = loadable.require_param("missing_param", str, default_value="default")
    assert value == "default"
    # Default should be stored in params
    assert loadable.params["missing_param"] == "default"


def test_loadable_optional_param():
    """Test optional_param with valid parameter."""
    loadable = TestableLoadable(my_param="test")
    value = loadable.optional_param("my_param", str)
    assert value == "test"


def test_loadable_optional_param_missing():
    """Test optional_param returns None when missing."""
    loadable = TestableLoadable()
    value = loadable.optional_param("missing_param", str)
    assert value is None


def test_loadable_optional_param_wrong_type():
    """Test optional_param raises TypeError for wrong type."""
    loadable = TestableLoadable(my_param=123)
    with pytest.raises(TypeError, match="must be"):
        loadable.optional_param("my_param", str)


def test_loadable_setup():
    """Test loadable setup with resources."""
    loadable = TestableLoadable()
    resources = {"model": "path/to/model", "index": "path/to/index"}
    loadable.setup(resources)
    assert loadable.resources == resources


def test_loadable_require_resource():
    """Test require_resource with valid resource."""
    loadable = TestableLoadable()
    loadable.setup({"my_resource": 42})
    value = loadable.require_resource("my_resource", int)
    assert value == 42


def test_loadable_require_resource_missing():
    """Test require_resource raises KeyError when missing."""
    loadable = TestableLoadable()
    loadable.setup({})
    with pytest.raises(KeyError, match="missing required resource"):
        loadable.require_resource("missing_resource", str)


def test_loadable_require_resource_wrong_type():
    """Test require_resource raises TypeError for wrong type."""
    loadable = TestableLoadable()
    loadable.setup({"my_resource": "string"})
    with pytest.raises(TypeError, match="must be"):
        loadable.require_resource("my_resource", int)


def test_loadable_require_resource_with_default():
    """Test require_resource uses default when missing."""
    loadable = TestableLoadable()
    loadable.setup({})
    value = loadable.require_resource("missing_resource", str, default_value="default")
    assert value == "default"
    # Default should be stored in resources
    assert loadable.resources["missing_resource"] == "default"


def test_loadable_optional_resource():
    """Test optional_resource with valid resource."""
    loadable = TestableLoadable()
    loadable.setup({"my_resource": "test"})
    value = loadable.optional_resource("my_resource", str)
    assert value == "test"


def test_loadable_optional_resource_missing():
    """Test optional_resource returns None when missing."""
    loadable = TestableLoadable()
    loadable.setup({})
    value = loadable.optional_resource("missing_resource", str)
    assert value is None


def test_loadable_optional_resource_wrong_type():
    """Test optional_resource raises TypeError for wrong type."""
    loadable = TestableLoadable()
    loadable.setup({"my_resource": 123})
    with pytest.raises(TypeError, match="must be"):
        loadable.optional_resource("my_resource", str)


def test_loadable_close():
    """Test loadable close method (no-op by default)."""
    loadable = TestableLoadable()
    # Should not raise any error
    loadable.close()


def test_loadable_params_and_resources_independent():
    """Test that params and resources are independent."""
    loadable = TestableLoadable(param1="value1")
    loadable.setup({"resource1": "res_value1"})
    
    assert "param1" in loadable.params
    assert "param1" not in loadable.resources
    assert "resource1" in loadable.resources
    assert "resource1" not in loadable.params
