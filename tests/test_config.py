"""Tests for unirec.core.config module."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unirec.core.config import Config, StageCfg, load_config


def test_stage_cfg_is_dict():
    """Test that StageCfg behaves like a dict."""
    cfg = StageCfg({"key": "value", "num": 42})
    assert cfg["key"] == "value"
    assert cfg["num"] == 42


def test_config_is_dict():
    """Test that Config behaves like a dict."""
    cfg = Config({"stages": [], "resources": {}})
    assert "stages" in cfg
    assert cfg["stages"] == []


def test_load_config_from_file():
    """Test loading config from YAML file."""
    test_data = {"stages": [{"kind": "test"}], "resources": {"model": "path"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.dump(test_data, f)
        temp_path = f.name

    try:
        cfg = load_config(temp_path)
        assert isinstance(cfg, Config)
        assert cfg["stages"] == [{"kind": "test"}]
        assert cfg["resources"]["model"] == "path"
    finally:
        Path(temp_path).unlink()


def test_load_config_with_empty_file():
    """Test loading config from nearly empty YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write("{}")  # Write empty dict instead of completely empty file
        temp_path = f.name

    try:
        cfg = load_config(temp_path)
        assert isinstance(cfg, Config)
        assert len(cfg) == 0
    finally:
        Path(temp_path).unlink()


def test_load_config_nonexistent_file():
    """Test that loading nonexistent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_config("/tmp/nonexistent_config_file_test.yml")
