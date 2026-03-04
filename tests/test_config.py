"""
Tests for ConfigManager.
"""

import json
import os
import tempfile
import pytest

from config.config_manager import ConfigManager, DEFAULT_CONFIG


class TestConfigManager:
    def test_defaults(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg = ConfigManager(path)
            assert cfg.get("capture.roi_width") == 640
            assert cfg.get("aim.enabled") is False
            assert cfg.get("nonexistent.key", "default") == "default"
        finally:
            os.unlink(path)

    def test_set_and_get(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg = ConfigManager(path)
            cfg.set("aim.sensitivity", 2.5)
            assert cfg.get("aim.sensitivity") == 2.5
        finally:
            os.unlink(path)

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg = ConfigManager(path)
            cfg.set("capture.roi_width", 800)
            cfg.save_now()

            # Load into new instance
            cfg2 = ConfigManager.__new__(ConfigManager)
            cfg2.__init__(path)
            assert cfg2.get("capture.roi_width") == 800
        finally:
            os.unlink(path)

    def test_reset_defaults(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg = ConfigManager(path)
            cfg.set("capture.roi_width", 999)
            cfg.reset_defaults()
            assert cfg.get("capture.roi_width") == 640
        finally:
            os.unlink(path)

    def test_section(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg = ConfigManager(path)
            capture = cfg.section("capture")
            assert isinstance(capture, dict)
            assert "roi_width" in capture
            # Ensure deep copy (mutating returned dict doesn't affect config)
            capture["roi_width"] = 9999
            assert cfg.get("capture.roi_width") == 640
        finally:
            os.unlink(path)

    def test_listener(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg = ConfigManager(path)
            changes = []
            cfg.add_listener(lambda k, v: changes.append((k, v)))
            cfg.set("aim.sensitivity", 3.0)
            assert len(changes) == 1
            assert changes[0] == ("aim.sensitivity", 3.0)
        finally:
            os.unlink(path)

    def test_migration(self):
        """Test config migration from v1 to v2."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"_version": 1, "capture": {"roi_width": 512}}, f)
            path = f.name
        try:
            cfg = ConfigManager(path)
            # Should have migrated and added filters section
            assert cfg.get("filters.type") == "one_euro"
            assert cfg.get("capture.roi_width") == 512  # preserved
        finally:
            os.unlink(path)
