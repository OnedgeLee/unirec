"""Tests for unirec.core.interfaces Component functionality."""

import pytest
from unirec.core.interfaces import Component
from unirec.core.state import PipelineState
from unirec.data.encodable import UserEncodable
import unirec.data.context.user_context as user_ctx

UserProfileContext = user_ctx.UserProfileContext
UserSessionContext = user_ctx.UserSessionContext


class TestableComponent(Component):
    """Concrete component for testing."""

    component_kind = "testable"

    def run(self, state: PipelineState) -> PipelineState:
        return state


def test_component_initialization():
    """Test Component initialization with params."""
    comp = TestableComponent(id="test_comp", param1="value1")
    assert comp.id == "test_comp"
    assert comp.params["param1"] == "value1"


def test_component_default_id():
    """Test that component gets class name as default id."""
    comp = TestableComponent()
    assert comp.id == "TestableComponent"


def test_component_require_param():
    """Test require_param with valid parameter."""
    comp = TestableComponent(my_param=123)
    value = comp.require_param("my_param", int)
    assert value == 123


def test_component_require_param_missing():
    """Test require_param raises KeyError when missing."""
    comp = TestableComponent()
    with pytest.raises(KeyError):
        comp.require_param("missing_param", str)


def test_component_require_param_wrong_type():
    """Test require_param raises TypeError for wrong type."""
    comp = TestableComponent(my_param="string")
    with pytest.raises(TypeError):
        comp.require_param("my_param", int)


def test_component_require_param_with_default():
    """Test require_param uses default when missing."""
    comp = TestableComponent()
    value = comp.require_param("missing_param", str, default_value="default")
    assert value == "default"
    # Default should be stored in params
    assert comp.params["missing_param"] == "default"


def test_component_optional_param():
    """Test optional_param with valid parameter."""
    comp = TestableComponent(my_param="test")
    value = comp.optional_param("my_param", str)
    assert value == "test"


def test_component_optional_param_missing():
    """Test optional_param returns None when missing."""
    comp = TestableComponent()
    value = comp.optional_param("missing_param", str)
    assert value is None


def test_component_optional_param_wrong_type():
    """Test optional_param raises TypeError for wrong type."""
    comp = TestableComponent(my_param=123)
    with pytest.raises(TypeError):
        comp.optional_param("my_param", str)


def test_component_setup():
    """Test component setup with resources."""
    comp = TestableComponent()
    resources = {"model": "path/to/model", "index": "path/to/index"}
    comp.setup(resources)
    assert comp.resources == resources


def test_component_require_resource():
    """Test require_resource with valid resource."""
    comp = TestableComponent()
    comp.setup({"my_resource": 42})
    value = comp.require_resource("my_resource", int)
    assert value == 42


def test_component_require_resource_missing():
    """Test require_resource raises KeyError when missing."""
    comp = TestableComponent()
    comp.setup({})
    with pytest.raises(KeyError):
        comp.require_resource("missing_resource", str)


def test_component_require_resource_wrong_type():
    """Test require_resource raises TypeError for wrong type."""
    comp = TestableComponent()
    comp.setup({"my_resource": "string"})
    with pytest.raises(TypeError):
        comp.require_resource("my_resource", int)


def test_component_require_resource_with_default():
    """Test require_resource uses default when missing."""
    comp = TestableComponent()
    comp.setup({})
    value = comp.require_resource("missing_resource", str, default_value="default")
    assert value == "default"
    # Default should be stored in resources
    assert comp.resources["missing_resource"] == "default"


def test_component_optional_resource():
    """Test optional_resource with valid resource."""
    comp = TestableComponent()
    comp.setup({"my_resource": "test"})
    value = comp.optional_resource("my_resource", str)
    assert value == "test"


def test_component_optional_resource_missing():
    """Test optional_resource returns None when missing."""
    comp = TestableComponent()
    comp.setup({})
    value = comp.optional_resource("missing_resource", str)
    assert value is None


def test_component_optional_resource_wrong_type():
    """Test optional_resource raises TypeError for wrong type."""
    comp = TestableComponent()
    comp.setup({"my_resource": 123})
    with pytest.raises(TypeError):
        comp.optional_resource("my_resource", str)


def test_component_run():
    """Test component run method."""
    comp = TestableComponent()
    profile = UserProfileContext(_id=1, _meta={})
    session = UserSessionContext(_meta={})
    user = UserEncodable(profile=profile, session=session)
    state = PipelineState(user=user, context={})

    result = comp.run(state)
    assert result == state


def test_component_close():
    """Test component close method (no-op by default)."""
    comp = TestableComponent()
    # Should not raise any error
    comp.close()
