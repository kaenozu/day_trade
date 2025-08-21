"""
統一設定管理システム

システム全体の設定を一元管理
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
import yaml
import os
import threading
import hashlib
from contextlib import contextmanager

T = TypeVar('T')


class ConfigSource(Enum):
    """設定ソース"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"
    MEMORY = "memory"


class ConfigFormat(Enum):
    """設定フォーマット"""
    JSON = "json"
    YAML = "yaml"
    INI = "ini"
    ENV = "env"


@dataclass
class ConfigEntry:
    """設定エントリ"""
    key: str
    value: Any
    source: ConfigSource
    format: ConfigFormat
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigSchema:
    """設定スキーマ"""
    name: str
    required_keys: List[str]
    optional_keys: List[str] = field(default_factory=list)
    default_values: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Callable] = field(default_factory=dict)


class ConfigValidator:
    """設定バリデーター"""

    def __init__(self):
        self._schemas: Dict[str, ConfigSchema] = {}

    def register_schema(self, schema: ConfigSchema) -> None:
        """スキーマ登録"""
        self._schemas[schema.name] = schema

    def validate(self, config_name: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """設定検証"""
        if config_name not in self._schemas:
            return {"valid": True, "errors": []}

        schema = self._schemas[config_name]
        errors = []

        # 必須キーチェック
        for key in schema.required_keys:
            if key not in config_data:
                errors.append(f"Required key '{key}' is missing")

        # バリデーションルール適用
        for key, rule in schema.validation_rules.items():
            if key in config_data:
                try:
                    if not rule(config_data[key]):
                        errors.append(f"Validation failed for key '{key}'")
                except Exception as e:
                    errors.append(f"Validation error for key '{key}': {str(e)}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "schema": config_name
        }


class ConfigLoader(ABC):
    """設定ローダー基底クラス"""

    @abstractmethod
    def load(self, source: str) -> Dict[str, Any]:
        """設定読み込み"""
        pass

    @abstractmethod
    def save(self, config: Dict[str, Any], destination: str) -> bool:
        """設定保存"""
        pass


class FileConfigLoader(ConfigLoader):
    """ファイル設定ローダー"""

    def load(self, file_path: str) -> Dict[str, Any]:
        """ファイルから設定読み込み"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        suffix = path.suffix.lower()

        with open(path, 'r', encoding='utf-8') as f:
            if suffix == '.json':
                return json.load(f)
            elif suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

    def save(self, config: Dict[str, Any], file_path: str) -> bool:
        """ファイルに設定保存"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            suffix = path.suffix.lower()

            with open(path, 'w', encoding='utf-8') as f:
                if suffix == '.json':
                    json.dump(config, f, indent=2, ensure_ascii=False)
                elif suffix in ['.yaml', '.yml']:
                    yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
                else:
                    raise ValueError(f"Unsupported file format: {suffix}")

            return True
        except Exception as e:
            return False


class EnvironmentConfigLoader(ConfigLoader):
    """環境変数設定ローダー"""

    def __init__(self, prefix: str = "DAY_TRADE_"):
        self.prefix = prefix

    def load(self, source: str = "") -> Dict[str, Any]:
        """環境変数から設定読み込み"""
        config = {}

        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                config_key = key[len(self.prefix):].lower()

                # 型推論
                if value.lower() in ['true', 'false']:
                    config[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    config[config_key] = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    config[config_key] = float(value)
                else:
                    config[config_key] = value

        return config

    def save(self, config: Dict[str, Any], destination: str = "") -> bool:
        """環境変数に設定保存（実際には設定しない）"""
        return False


class ConfigWatcher:
    """設定変更監視"""

    def __init__(self):
        self._watchers: List[Callable[[str, Any, Any], None]] = []
        self._file_watches: Dict[str, float] = {}

    def add_watcher(self, callback: Callable[[str, Any, Any], None]) -> None:
        """変更監視コールバック追加"""
        self._watchers.append(callback)

    def watch_file(self, file_path: str) -> None:
        """ファイル監視追加"""
        path = Path(file_path)
        if path.exists():
            self._file_watches[file_path] = path.stat().st_mtime

    def check_file_changes(self) -> List[str]:
        """ファイル変更チェック"""
        changed_files = []

        for file_path, last_mtime in self._file_watches.items():
            path = Path(file_path)
            if path.exists():
                current_mtime = path.stat().st_mtime
                if current_mtime > last_mtime:
                    self._file_watches[file_path] = current_mtime
                    changed_files.append(file_path)

        return changed_files

    def notify_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """変更通知"""
        for watcher in self._watchers:
            try:
                watcher(key, old_value, new_value)
            except Exception as e:
                print(f"Config watcher error: {e}")


class ConfigCache:
    """設定キャッシュ"""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, ConfigEntry] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """キャッシュから取得"""
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                # TTLチェック
                elapsed = (datetime.now() - entry.last_updated).total_seconds()
                if elapsed < self.ttl_seconds:
                    return entry.value
                else:
                    del self._cache[key]
            return None

    def set(self, key: str, value: Any, source: ConfigSource, format: ConfigFormat) -> None:
        """キャッシュに設定"""
        with self._lock:
            self._cache[key] = ConfigEntry(
                key=key,
                value=value,
                source=source,
                format=format
            )

    def invalidate(self, key: str = None) -> None:
        """キャッシュ無効化"""
        with self._lock:
            if key:
                self._cache.pop(key, None)
            else:
                self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計"""
        with self._lock:
            return {
                "size": len(self._cache),
                "entries": list(self._cache.keys()),
                "oldest_entry": min(
                    (entry.last_updated for entry in self._cache.values()),
                    default=None
                )
            }


class UnifiedConfigManager:
    """統一設定管理マネージャー"""

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._loaders: Dict[ConfigSource, ConfigLoader] = {
            ConfigSource.FILE: FileConfigLoader(),
            ConfigSource.ENVIRONMENT: EnvironmentConfigLoader()
        }
        self._validator = ConfigValidator()
        self._watcher = ConfigWatcher()
        self._cache = ConfigCache()
        self._lock = threading.RLock()
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._active_profile = "default"

        # デフォルトスキーマ登録
        self._register_default_schemas()

    def _register_default_schemas(self) -> None:
        """デフォルトスキーマ登録"""
        # 取引設定スキーマ
        trading_schema = ConfigSchema(
            name="trading",
            required_keys=["max_positions", "risk_tolerance"],
            optional_keys=["stop_loss_percent", "take_profit_percent"],
            default_values={
                "stop_loss_percent": 5.0,
                "take_profit_percent": 10.0
            },
            validation_rules={
                "max_positions": lambda x: isinstance(x, int) and x > 0,
                "risk_tolerance": lambda x: isinstance(x, (int, float)) and 0 < x <= 1
            }
        )
        self._validator.register_schema(trading_schema)

        # データベース設定スキーマ
        database_schema = ConfigSchema(
            name="database",
            required_keys=["url"],
            optional_keys=["pool_size", "timeout"],
            default_values={
                "pool_size": 10,
                "timeout": 30
            }
        )
        self._validator.register_schema(database_schema)

    def load_from_file(self, file_path: str, profile: str = None) -> bool:
        """ファイルから設定読み込み"""
        try:
            loader = self._loaders[ConfigSource.FILE]
            config_data = loader.load(file_path)

            profile_name = profile or self._active_profile

            with self._lock:
                if profile_name not in self._profiles:
                    self._profiles[profile_name] = {}

                self._profiles[profile_name].update(config_data)

                # アクティブプロファイルの場合、メイン設定も更新
                if profile_name == self._active_profile:
                    self._config.update(config_data)

            # ファイル監視追加
            self._watcher.watch_file(file_path)

            return True
        except Exception as e:
            print(f"Failed to load config from {file_path}: {e}")
            return False

    def load_from_environment(self) -> bool:
        """環境変数から設定読み込み"""
        try:
            loader = self._loaders[ConfigSource.ENVIRONMENT]
            env_config = loader.load()

            with self._lock:
                self._config.update(env_config)

            return True
        except Exception as e:
            print(f"Failed to load config from environment: {e}")
            return False

    def set_profile(self, profile_name: str) -> bool:
        """プロファイル設定"""
        with self._lock:
            if profile_name in self._profiles:
                self._active_profile = profile_name
                self._config.clear()
                self._config.update(self._profiles[profile_name])
                return True
            return False

    def get_active_profile(self) -> str:
        """アクティブプロファイル取得"""
        return self._active_profile

    def get_profiles(self) -> List[str]:
        """プロファイル一覧取得"""
        return list(self._profiles.keys())

    def get(self, key: str, default: Any = None) -> Any:
        """設定値取得"""
        # キャッシュから取得試行
        cached_value = self._cache.get(key)
        if cached_value is not None:
            return cached_value

        # ドット記法対応
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

            # キャッシュに保存
            self._cache.set(key, value, ConfigSource.MEMORY, ConfigFormat.JSON)

            return value
        except Exception:
            return default

    def set(self, key: str, value: Any, persist: bool = False) -> bool:
        """設定値設定"""
        try:
            old_value = self.get(key)

            # ドット記法対応
            keys = key.split('.')
            config_ref = self._config

            with self._lock:
                for k in keys[:-1]:
                    if k not in config_ref:
                        config_ref[k] = {}
                    config_ref = config_ref[k]

                config_ref[keys[-1]] = value

            # キャッシュ無効化
            self._cache.invalidate(key)

            # 変更通知
            self._watcher.notify_change(key, old_value, value)

            # 永続化
            if persist:
                self.save_to_file(f"config_{self._active_profile}.json")

            return True
        except Exception as e:
            print(f"Failed to set config {key}: {e}")
            return False

    def save_to_file(self, file_path: str, profile: str = None) -> bool:
        """ファイルに設定保存"""
        try:
            loader = self._loaders[ConfigSource.FILE]
            profile_name = profile or self._active_profile

            config_to_save = self._profiles.get(profile_name, self._config)

            return loader.save(config_to_save, file_path)
        except Exception as e:
            print(f"Failed to save config to {file_path}: {e}")
            return False

    def validate_config(self, config_name: str = None) -> Dict[str, Any]:
        """設定検証"""
        if config_name:
            # 特定設定の検証
            config_section = self.get(config_name, {})
            return self._validator.validate(config_name, config_section)
        else:
            # 全設定の検証
            results = {}
            for schema_name in ["trading", "database"]:
                config_section = self.get(schema_name, {})
                results[schema_name] = self._validator.validate(schema_name, config_section)
            return results

    def watch_changes(self, callback: Callable[[str, Any, Any], None]) -> None:
        """設定変更監視"""
        self._watcher.add_watcher(callback)

    def check_file_changes(self) -> bool:
        """ファイル変更チェックと再読み込み"""
        changed_files = self._watcher.check_file_changes()

        for file_path in changed_files:
            print(f"Config file changed: {file_path}")
            self.load_from_file(file_path)

        return len(changed_files) > 0

    def get_config_info(self) -> Dict[str, Any]:
        """設定情報取得"""
        return {
            "active_profile": self._active_profile,
            "profiles": self.get_profiles(),
            "config_size": len(self._config),
            "cache_stats": self._cache.get_stats(),
            "validation_results": self.validate_config()
        }

    def reset_to_defaults(self, config_name: str = None) -> bool:
        """デフォルト値にリセット"""
        try:
            if config_name:
                # 特定設定のリセット
                if config_name in self._validator._schemas:
                    schema = self._validator._schemas[config_name]
                    with self._lock:
                        self._config[config_name] = schema.default_values.copy()
            else:
                # 全設定のリセット
                with self._lock:
                    self._config.clear()
                    for schema in self._validator._schemas.values():
                        self._config[schema.name] = schema.default_values.copy()

            # キャッシュクリア
            self._cache.invalidate()

            return True
        except Exception as e:
            print(f"Failed to reset config: {e}")
            return False

    @contextmanager
    def temp_config(self, temp_config: Dict[str, Any]):
        """一時設定コンテキスト"""
        original_values = {}

        try:
            # 元の値を保存して一時値を設定
            for key, value in temp_config.items():
                original_values[key] = self.get(key)
                self.set(key, value)

            yield

        finally:
            # 元の値に復元
            for key, original_value in original_values.items():
                if original_value is not None:
                    self.set(key, original_value)


# グローバル設定マネージャー
global_config_manager = UnifiedConfigManager()