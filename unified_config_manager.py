#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合設定管理システム - Issue #960対応
散在する193の設定ファイルを統合管理
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from functools import lru_cache
from dataclasses import dataclass, field
import threading
import hashlib
from datetime import datetime
import jsonschema

@dataclass
class ConfigValidationError(Exception):
    """設定検証エラー"""
    message: str
    config_type: str
    errors: List[str] = field(default_factory=list)

@dataclass
class ConfigCache:
    """設定キャッシュ"""
    content: Dict[str, Any]
    timestamp: datetime
    hash_value: str

class UnifiedConfigManager:
    """統合設定管理システム"""

    def __init__(self, config_root: str = "config_unified", environment: str = "development"):
        self.config_root = Path(config_root)
        self.environment = environment
        self.cache: Dict[str, ConfigCache] = {}
        self.cache_lock = threading.RLock()
        self.logger = self._setup_logging()
        self.schemas = self._load_schemas()

        # 設定ディレクトリ構造の初期化
        self._ensure_directory_structure()

    def _setup_logging(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger('unified_config')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _ensure_directory_structure(self) -> None:
        """統合設定ディレクトリ構造の確保"""
        directories = [
            self.config_root / "core",
            self.config_root / "environments",
            self.config_root / "features",
            self.config_root / "schemas"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Config directory structure ensured: {self.config_root}")

    def _load_schemas(self) -> Dict[str, Dict]:
        """設定検証スキーマの読み込み"""
        schemas = {}
        schema_dir = self.config_root / "schemas"

        if schema_dir.exists():
            for schema_file in schema_dir.glob("*.json"):
                try:
                    with open(schema_file, 'r', encoding='utf-8') as f:
                        schema_name = schema_file.stem
                        schemas[schema_name] = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load schema {schema_file}: {e}")

        return schemas

    @lru_cache(maxsize=128)
    def get_config(self, config_type: str, force_reload: bool = False) -> Dict[str, Any]:
        """統合設定取得（キャッシュ付き）"""
        if force_reload and config_type in self.cache:
            del self.cache[config_type]
            self.get_config.cache_clear()

        with self.cache_lock:
            # キャッシュチェック
            if config_type in self.cache:
                cached = self.cache[config_type]
                if self._is_cache_valid(cached, config_type):
                    self.logger.debug(f"Config loaded from cache: {config_type}")
                    return cached.content.copy()

            # 設定読み込み
            config = self._load_config(config_type)

            # キャッシュ更新
            self.cache[config_type] = ConfigCache(
                content=config.copy(),
                timestamp=datetime.now(),
                hash_value=self._calculate_hash(config)
            )

            self.logger.info(f"Config loaded: {config_type}")
            return config

    def _load_config(self, config_type: str) -> Dict[str, Any]:
        """設定の読み込み"""
        # 基本設定の読み込み
        base_config = self._load_base_config(config_type)

        # 環境別設定の読み込み
        env_config = self._load_environment_config(config_type)

        # 設定のマージ
        merged_config = self._merge_configs(base_config, env_config)

        # 設定の検証
        self._validate_config(config_type, merged_config)

        return merged_config

    def _load_base_config(self, config_type: str) -> Dict[str, Any]:
        """基本設定の読み込み"""
        config_file = self.config_root / "core" / f"{config_type}.yaml"

        if not config_file.exists():
            # フォールバック: features ディレクトリ
            config_file = self.config_root / "features" / f"{config_type}.yaml"

        if not config_file.exists():
            self.logger.warning(f"Base config not found: {config_type}")
            return {}

        return self._load_yaml_file(config_file)

    def _load_environment_config(self, config_type: str) -> Dict[str, Any]:
        """環境別設定の読み込み"""
        env_file = self.config_root / "environments" / f"{self.environment}.yaml"

        if not env_file.exists():
            self.logger.debug(f"Environment config not found: {self.environment}")
            return {}

        env_data = self._load_yaml_file(env_file)

        # 特定設定タイプの環境設定を抽出
        return env_data.get(config_type, {})

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """YAMLファイルの読み込み"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f) or {}

            if not isinstance(content, dict):
                self.logger.warning(f"Invalid YAML structure in {file_path}")
                return {}

            # 環境変数置換
            return self._substitute_environment_variables(content)

        except Exception as e:
            self.logger.error(f"Failed to load YAML file {file_path}: {e}")
            return {}

    def _substitute_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """環境変数の置換"""
        def substitute_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                default_value = ""

                if ":" in env_var:
                    env_var, default_value = env_var.split(":", 1)

                return os.getenv(env_var, default_value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value

        return substitute_value(config)

    def _merge_configs(self, base_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """設定のマージ"""
        merged = base_config.copy()

        def deep_merge(target: Dict, source: Dict):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value

        deep_merge(merged, env_config)
        return merged

    def _validate_config(self, config_type: str, config: Dict[str, Any]) -> None:
        """設定の妥当性検証"""
        if config_type not in self.schemas:
            self.logger.debug(f"No schema found for config type: {config_type}")
            return

        schema = self.schemas[config_type]

        try:
            jsonschema.validate(config, schema)
            self.logger.debug(f"Config validation passed: {config_type}")
        except jsonschema.ValidationError as e:
            raise ConfigValidationError(
                message=f"Config validation failed for {config_type}",
                config_type=config_type,
                errors=[str(e)]
            )

    def _is_cache_valid(self, cached: ConfigCache, config_type: str) -> bool:
        """キャッシュの有効性チェック"""
        # 簡易版: タイムスタンプチェック
        cache_duration = 300  # 5分
        age = (datetime.now() - cached.timestamp).total_seconds()
        return age < cache_duration

    def _calculate_hash(self, config: Dict[str, Any]) -> str:
        """設定のハッシュ値計算"""
        config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(config_str.encode()).hexdigest()

    def set_config(self, config_type: str, config: Dict[str, Any], environment: Optional[str] = None) -> None:
        """設定の保存"""
        target_env = environment or self.environment

        # 設定検証
        self._validate_config(config_type, config)

        if target_env == "base":
            # 基本設定として保存
            config_file = self.config_root / "core" / f"{config_type}.yaml"
        else:
            # 環境別設定として保存
            env_file = self.config_root / "environments" / f"{target_env}.yaml"

            # 既存環境設定の読み込み
            env_data = {}
            if env_file.exists():
                env_data = self._load_yaml_file(env_file)

            # 設定の更新
            env_data[config_type] = config
            config = env_data
            config_file = env_file

        # ファイル保存
        self._save_yaml_file(config_file, config)

        # キャッシュクリア
        with self.cache_lock:
            if config_type in self.cache:
                del self.cache[config_type]
            self.get_config.cache_clear()

        self.logger.info(f"Config saved: {config_type} -> {config_file}")

    def _save_yaml_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """YAMLファイルの保存"""
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)

    def reload_config(self, config_type: str) -> None:
        """設定の再読み込み"""
        self.get_config(config_type, force_reload=True)

    def list_config_types(self) -> List[str]:
        """利用可能な設定タイプの一覧"""
        config_types = set()

        # core ディレクトリ
        core_dir = self.config_root / "core"
        if core_dir.exists():
            for file_path in core_dir.glob("*.yaml"):
                config_types.add(file_path.stem)

        # features ディレクトリ
        features_dir = self.config_root / "features"
        if features_dir.exists():
            for file_path in features_dir.glob("*.yaml"):
                config_types.add(file_path.stem)

        return sorted(list(config_types))

    def get_environment_info(self) -> Dict[str, Any]:
        """環境情報の取得"""
        return {
            "current_environment": self.environment,
            "config_root": str(self.config_root),
            "available_environments": self._get_available_environments(),
            "cache_status": {
                "cached_configs": list(self.cache.keys()),
                "cache_size": len(self.cache)
            }
        }

    def _get_available_environments(self) -> List[str]:
        """利用可能な環境の一覧"""
        env_dir = self.config_root / "environments"
        if not env_dir.exists():
            return []

        return [f.stem for f in env_dir.glob("*.yaml")]

    def create_default_configs(self) -> None:
        """デフォルト設定の作成"""
        # アプリケーション基本設定
        app_config = {
            "name": "day-trade-personal",
            "version": "2.1.0",
            "debug": False,
            "log_level": "INFO",
            "timezone": "Asia/Tokyo"
        }

        # データベース設定
        database_config = {
            "url": "sqlite:///data/trading.db",
            "pool_size": 10,
            "echo": False,
            "connect_args": {
                "check_same_thread": False
            }
        }

        # セキュリティ設定
        security_config = {
            "secret_key": "${SECRET_KEY:default-secret-key}",
            "encryption_key": "${ENCRYPTION_KEY:}",
            "session_timeout": 3600,
            "max_login_attempts": 5,
            "password_min_length": 8
        }

        # ML/AI設定
        ml_config = {
            "enabled": True,
            "model_path": "models/",
            "batch_size": 32,
            "gpu_enabled": False,
            "cache_predictions": True
        }

        # 設定保存
        configs = {
            "application": app_config,
            "database": database_config,
            "security": security_config,
            "ml_models": ml_config
        }

        for config_type, config_data in configs.items():
            self.set_config(config_type, config_data, "base")

        # 環境別設定
        env_configs = {
            "development": {
                "application": {"debug": True, "log_level": "DEBUG"},
                "database": {"echo": True}
            },
            "staging": {
                "application": {"debug": False, "log_level": "INFO"},
                "database": {"pool_size": 5}
            },
            "production": {
                "application": {"debug": False, "log_level": "WARNING"},
                "security": {"strict_mode": True}
            }
        }

        for env_name, env_data in env_configs.items():
            env_file = self.config_root / "environments" / f"{env_name}.yaml"
            self._save_yaml_file(env_file, env_data)

        self.logger.info("Default configurations created")

def create_config_schemas() -> None:
    """設定検証スキーマの作成"""
    schemas_dir = Path("config_unified/schemas")
    schemas_dir.mkdir(parents=True, exist_ok=True)

    # アプリケーション設定スキーマ
    app_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"},
            "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
            "debug": {"type": "boolean"},
            "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]}
        },
        "required": ["name", "version"]
    }

    # データベース設定スキーマ
    db_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "format": "uri"},
            "pool_size": {"type": "integer", "minimum": 1, "maximum": 100},
            "echo": {"type": "boolean"}
        },
        "required": ["url"]
    }

    # スキーマ保存
    schemas = {
        "application": app_schema,
        "database": db_schema
    }

    for schema_name, schema_data in schemas.items():
        schema_file = schemas_dir / f"{schema_name}.json"
        with open(schema_file, 'w', encoding='utf-8') as f:
            json.dump(schema_data, f, indent=2, ensure_ascii=False)

    print(f"Configuration schemas created in {schemas_dir}")

def main():
    """メイン実行関数"""
    print("Unified Config Manager - Issue #960")
    print("=" * 50)

    # スキーマ作成
    create_config_schemas()

    # 統合設定管理システムの初期化
    config_manager = UnifiedConfigManager()

    # デフォルト設定の作成
    config_manager.create_default_configs()

    # 設定テスト
    try:
        app_config = config_manager.get_config("application")
        print(f"Application config loaded: {app_config['name']} v{app_config['version']}")

        db_config = config_manager.get_config("database")
        print(f"Database config loaded: {db_config['url']}")

        # 環境情報表示
        env_info = config_manager.get_environment_info()
        print(f"Environment: {env_info['current_environment']}")
        print(f"Available configs: {config_manager.list_config_types()}")

        print("\\nUnified Config Manager successfully initialized!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()