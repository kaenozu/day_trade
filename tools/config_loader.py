#!/usr/bin/env python3
"""設定ローダー"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"

    def load_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        if not environment:
            environment = os.getenv("ENVIRONMENT", "development")

        config_file = self.config_dir / f"config_{environment}.json"

        if not config_file.exists():
            config_file = self.config_dir / "config.json"

        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_value(self, key_path: str, default: Any = None) -> Any:
        config = self.load_config()
        keys = key_path.split('.')
        value = config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
