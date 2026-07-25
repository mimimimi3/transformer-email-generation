"""Helpers for loading project YAML configuration files."""

import yaml


def load_config(config_path):
    """Loads project settings from a YAML config file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
