"""Tests for unirec.core.registry module."""

import pytest
from unirec.core.registry import register, create, REGISTRY
from unirec.core.interfaces import Component
from unirec.core.state import PipelineState
from unirec.data.encodable import UserEncodable
import unirec.data.context.user_context as user_ctx

UserProfileContext = user_ctx.UserProfileContext
UserSessionContext = user_ctx.UserSessionContext


class TestComponent(Component):
    """Test component for registry tests."""

    component_kind = "test_component"

    def run(self, state: PipelineState) -> PipelineState:
        return state


def test_register_decorator():
    """Test that register decorator adds component to registry."""
    initial_count = len(REGISTRY.get("test_kind", {}))

    @register("test_kind")
    class DemoComponent(Component):
        component_kind = "test_kind"

        def run(self, state: PipelineState) -> PipelineState:
            return state

    assert "test_kind" in REGISTRY
    assert len(REGISTRY["test_kind"]) > initial_count
    fqn = f"{DemoComponent.__module__}.{DemoComponent.__name__}"
    assert fqn in REGISTRY["test_kind"]


def test_create_from_registry():
    """Test creating component from registry."""

    @register("test_creation")
    class CreatableComponent(Component):
        component_kind = "test_creation"

        def __init__(self, **params):
            super().__init__(**params)

        def run(self, state: PipelineState) -> PipelineState:
            return state

    fqn = f"{CreatableComponent.__module__}.{CreatableComponent.__name__}"
    comp = create("test_creation", fqn, id="test_comp")

    assert isinstance(comp, CreatableComponent)
    assert comp.id == "test_comp"


def test_create_with_params():
    """Test creating component with parameters."""

    @register("param_test")
    class ParamComponent(Component):
        component_kind = "param_test"

        def run(self, state: PipelineState) -> PipelineState:
            return state

    fqn = f"{ParamComponent.__module__}.{ParamComponent.__name__}"
    comp = create("param_test", fqn, id="param_comp", custom_param="value")

    assert comp.params["id"] == "param_comp"
    assert comp.params["custom_param"] == "value"


def test_create_nonexistent_kind():
    """Test that creating with wrong kind still works if impl is valid."""
    # If component is not in registry, it will try to import from module
    with pytest.raises((KeyError, AttributeError, ModuleNotFoundError)):
        create("nonexistent_kind", "nonexistent.module.Class")
