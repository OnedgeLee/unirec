"""Tests for unirec.core.version module."""

import pytest
from packaging.version import Version
from unirec.core.version import Versioned


class ConcreteVersioned(Versioned):
    """Concrete implementation for testing."""

    VERSION = Version("1.2.3")


def test_versioned_version():
    """Test that VERSION is properly defined and accessible."""
    assert ConcreteVersioned.VERSION == Version("1.2.3")
    assert ConcreteVersioned.version() == Version("1.2.3")


def test_versioned_instance_version():
    """Test that version() works on instances."""
    obj = ConcreteVersioned()
    assert obj.version() == Version("1.2.3")


def test_versioned_requires_version():
    """Test that concrete classes must define VERSION."""
    with pytest.raises(TypeError):

        class MissingVersion(Versioned):
            pass


def test_versioned_version_must_be_version_type():
    """Test that VERSION must be a Version instance."""
    with pytest.raises(TypeError):

        class WrongVersionType(Versioned):
            VERSION = "1.2.3"
