"""Tests for unirec.core.fingerprint module."""

import pytest
from packaging.version import Version
from unirec.core.fingerprint import (
    Fingerprint,
    Fingerprinter,
    Fingerprintable,
    fingerprint_field,
)
from dataclasses import dataclass


def test_fingerprinter_creation():
    """Test Fingerprinter initialization."""
    fp = Fingerprinter()
    assert fp.hash_algo == "blake2s"
    assert fp.hash_len == 32


def test_fingerprinter_custom_params():
    """Test Fingerprinter with custom parameters."""
    fp = Fingerprinter(hash_algo="sha256", hash_len=16)
    assert fp.hash_algo == "sha256"
    assert fp.hash_len == 16


def test_fingerprinter_min_hash_len():
    """Test that hash_len must be at least MIN_HASH_LEN."""
    with pytest.raises(ValueError):
        Fingerprinter(hash_len=8)  # Below minimum


def test_fingerprinter_has_version():
    """Test that Fingerprinter has VERSION."""
    assert hasattr(Fingerprinter, "VERSION")
    assert isinstance(Fingerprinter.VERSION, Version)


def test_fingerprint_from_simple_object():
    """Test creating fingerprint from simple object."""

    @dataclass
    class SimpleObj:
        value: int = fingerprint_field()

    fp = Fingerprinter()
    obj = SimpleObj(value=42)
    fingerprint = fp.make(obj)

    assert isinstance(fingerprint, Fingerprint)
    assert fingerprint.fqn  # Should have fully qualified name
    assert fingerprint.json_str  # Should have json_str
    # Test key without accessing digest cached_property
    key_parts = fingerprint.key.split(":")
    assert len(key_parts) == 3


def test_fingerprint_key_format():
    """Test fingerprint key format."""

    @dataclass
    class SimpleObj:
        value: int = fingerprint_field()

    fp = Fingerprinter()
    obj = SimpleObj(value=42)
    fingerprint = fp.make(obj)

    # Key format: algo:hash_len:digest
    parts = fingerprint.key.split(":")
    assert len(parts) == 3
    assert parts[0] == fp.hash_algo
    assert parts[1] == str(fp.hash_len)
    assert len(parts[2]) > 0  # Has digest


def test_fingerprinter_key_method():
    """Test Fingerprinter key() method."""

    @dataclass
    class SimpleObj:
        value: int = fingerprint_field()

    fp = Fingerprinter()
    obj = SimpleObj(value=42)
    key = fp.key(obj)

    assert isinstance(key, str)
    assert ":" in key


def test_fingerprint_consistency():
    """Test that same object produces same fingerprint."""

    @dataclass
    class SimpleObj:
        value: int = fingerprint_field()

    fp = Fingerprinter()
    obj1 = SimpleObj(value=42)
    obj2 = SimpleObj(value=42)

    key1 = fp.key(obj1)
    key2 = fp.key(obj2)

    assert key1 == key2


def test_fingerprint_different_for_different_values():
    """Test that different objects produce different fingerprints."""

    @dataclass
    class SimpleObj:
        value: int = fingerprint_field()

    fp = Fingerprinter()
    obj1 = SimpleObj(value=42)
    obj2 = SimpleObj(value=43)

    key1 = fp.key(obj1)
    key2 = fp.key(obj2)

    assert key1 != key2


def test_fingerprintable_mixin():
    """Test Fingerprintable mixin."""

    @dataclass
    class MyObj(Fingerprintable):
        value: int = fingerprint_field()

    obj = MyObj(value=42)
    assert hasattr(obj, "key")
    key = obj.key
    assert isinstance(key, str)
    assert ":" in key


def test_fingerprintable_cached_property():
    """Test that fingerprint is cached."""

    @dataclass
    class MyObj(Fingerprintable):
        value: int = fingerprint_field()

    obj = MyObj(value=42)
    fp1 = obj.fingerprint
    fp2 = obj.fingerprint

    # Should be the same object (cached)
    assert fp1 is fp2


def test_fingerprint_cannot_be_subclassed():
    """Test that Fingerprint is final and cannot be subclassed."""
    with pytest.raises(TypeError):

        class MyFingerprint(Fingerprint):
            pass


def test_fingerprinter_cannot_be_subclassed():
    """Test that Fingerprinter is final and cannot be subclassed."""
    with pytest.raises(TypeError):

        class MyFingerprinter(Fingerprinter):
            pass


def test_fingerprint_field_basic():
    """Test fingerprint_field decorator."""

    @dataclass
    class MyObj:
        tracked: int = fingerprint_field()
        not_tracked: int = 0

    fp = Fingerprinter()
    obj = MyObj(tracked=42, not_tracked=100)

    # tracked field should be in fingerprint
    fingerprint = fp.make(obj)
    assert fingerprint is not None


def test_fingerprint_with_empty_fqn():
    """Test that Fingerprint requires non-empty fqn."""
    with pytest.raises(ValueError):
        Fingerprint(json_str='[["test", "value"]]', algo="sha256", hash_len=16, fqn="")
