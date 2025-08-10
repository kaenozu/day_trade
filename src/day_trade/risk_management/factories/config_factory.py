#!/usr/bin/env python3
"""
Configuration Provider Factory
設定プロバイダーファクトリー

設定管理システムの動的生成とプラグイン管理
"""

from typing import Dict, Type, Any, Optional, List, Union
from enum import Enum
import importlib
import os
from pathlib import Path

from ..exceptions.risk_exceptions import ConfigurationError, ValidationError

class ConfigProviderType(Enum):
    """設定プロバイダータイプ"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"
    VAULT = "vault"
    KUBERNETES = "kubernetes"
    CONSUL = "consul"
    ETCD = "etcd"
    PLUGIN = "plugin"

class ConfigFormat(Enum):
    """設定ファイル形式"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    XML = "xml"
    PROPERTIES = "properties"

class IConfigProvider:
    """設定プロバイダーインターフェース"""

    def get(self, key: str, default: Any = None) -> Any:
        """設定値取得"""
        raise NotImplementedError

    def set(self, key: str, value: Any) -> bool:
        """設定値設定"""
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """設定値削除"""
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        """設定キー存在確認"""
        raise NotImplementedError

    def get_all(self) -> Dict[str, Any]:
        """全設定取得"""
        raise NotImplementedError

    def reload(self) -> bool:
        """設定リロード"""
        raise NotImplementedError

    def validate(self) -> List[str]:
        """設定検証"""
        return []

    def backup(self) -> bool:
        """設定バックアップ"""
        return True

    def restore(self, backup_id: str) -> bool:
        """設定復元"""
        return True

class ConfigProviderFactory:
    """設定プロバイダーファクトリー"""

    def __init__(self):
        self._provider_registry: Dict[ConfigProviderType, Type[IConfigProvider]] = {}
        self._plugin_registry: Dict[str, Type[IConfigProvider]] = {}
        self._config_schemas: Dict[ConfigProviderType, Dict[str, Any]] = {}
        self._instance_cache: Dict[str, IConfigProvider] = {}
        self._watchers: Dict[str, List[callable]] = {}

        # 組み込みプロバイダーを登録
        self._register_builtin_providers()

    def _register_builtin_providers(self):
        """組み込みプロバイダー登録"""
        try:
            # ファイルベース設定
            self.register_provider(
                ConfigProviderType.FILE,
                "src.day_trade.risk_management.config.file_provider",
                "FileConfigProvider",
                {
                    "file_path": {"type": str, "required": True},
                    "format": {"type": str, "default": "json"},
                    "auto_reload": {"type": bool, "default": False},
                    "watch_changes": {"type": bool, "default": False},
                    "backup_count": {"type": int, "default": 5},
                    "encoding": {"type": str, "default": "utf-8"}
                }
            )

            # 環境変数設定
            self.register_provider(
                ConfigProviderType.ENVIRONMENT,
                "src.day_trade.risk_management.config.env_provider",
                "EnvironmentConfigProvider",
                {
                    "prefix": {"type": str, "default": ""},
                    "separator": {"type": str, "default": "_"},
                    "case_sensitive": {"type": bool, "default": False},
                    "type_conversion": {"type": bool, "default": True},
                    "default_values": {"type": dict, "default": {}}
                }
            )

            # データベース設定
            self.register_provider(
                ConfigProviderType.DATABASE,
                "src.day_trade.risk_management.config.db_provider",
                "DatabaseConfigProvider",
                {
                    "connection_string": {"type": str, "required": True},
                    "table_name": {"type": str, "default": "configurations"},
                    "key_column": {"type": str, "default": "key"},
                    "value_column": {"type": str, "default": "value"},
                    "cache_ttl_seconds": {"type": int, "default": 300},
                    "auto_sync": {"type": bool, "default": True}
                }
            )

            # リモート設定
            self.register_provider(
                ConfigProviderType.REMOTE,
                "src.day_trade.risk_management.config.remote_provider",
                "RemoteConfigProvider",
                {
                    "url": {"type": str, "required": True},
                    "auth_type": {"type": str, "default": "none"},
                    "auth_credentials": {"type": dict, "default": {}},
                    "polling_interval_seconds": {"type": int, "default": 60},
                    "timeout_seconds": {"type": int, "default": 30},
                    "retry_attempts": {"type": int, "default": 3}
                }
            )

            # HashiCorp Vault設定
            self.register_provider(
                ConfigProviderType.VAULT,
                "src.day_trade.risk_management.config.vault_provider",
                "VaultConfigProvider",
                {
                    "url": {"type": str, "required": True},
                    "token": {"type": str, "required": False},
                    "role_id": {"type": str, "required": False},
                    "secret_id": {"type": str, "required": False},
                    "mount_point": {"type": str, "default": "secret"},
                    "path_prefix": {"type": str, "default": ""},
                    "verify_ssl": {"type": bool, "default": True}
                }
            )

            # Kubernetes ConfigMap設定
            self.register_provider(
                ConfigProviderType.KUBERNETES,
                "src.day_trade.risk_management.config.k8s_provider",
                "KubernetesConfigProvider",
                {
                    "namespace": {"type": str, "default": "default"},
                    "config_map_name": {"type": str, "required": True},
                    "kubeconfig_path": {"type": str, "required": False},
                    "context": {"type": str, "required": False},
                    "watch_changes": {"type": bool, "default": True}
                }
            )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to register some builtin config providers: {e}")

    def register_provider(
        self,
        provider_type: ConfigProviderType,
        module_path: str,
        class_name: str,
        config_schema: Optional[Dict[str, Any]] = None
    ) -> bool:
        """プロバイダー登録"""
        try:
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)

            if not issubclass(provider_class, IConfigProvider):
                raise ConfigurationError(
                    f"Config provider class {class_name} must implement IConfigProvider interface",
                    config_key=f"config.provider.{provider_type.value}"
                )

            self._provider_registry[provider_type] = provider_class

            if config_schema:
                self._config_schemas[provider_type] = config_schema

            return True

        except Exception as e:
            raise ConfigurationError(
                f"Failed to register config provider type {provider_type.value}",
                config_key=f"config.provider.{provider_type.value}",
                cause=e
            ) from e

    def register_plugin_provider(
        self,
        plugin_name: str,
        provider_class: Type[IConfigProvider],
        config_schema: Optional[Dict[str, Any]] = None
    ) -> bool:
        """プラグインプロバイダー登録"""
        try:
            if not issubclass(provider_class, IConfigProvider):
                raise ConfigurationError(
                    f"Plugin config provider {plugin_name} must implement IConfigProvider interface",
                    config_key=f"config.plugin.{plugin_name}"
                )

            self._plugin_registry[plugin_name] = provider_class

            if config_schema:
                self._config_schemas[ConfigProviderType.PLUGIN] = self._config_schemas.get(
                    ConfigProviderType.PLUGIN, {}
                )
                self._config_schemas[ConfigProviderType.PLUGIN][plugin_name] = config_schema

            return True

        except Exception as e:
            raise ConfigurationError(
                f"Failed to register plugin config provider {plugin_name}",
                config_key=f"config.plugin.{plugin_name}",
                cause=e
            ) from e

    def create_config_provider(
        self,
        provider_type: ConfigProviderType,
        config: Optional[Dict[str, Any]] = None,
        plugin_name: Optional[str] = None,
        use_cache: bool = True
    ) -> IConfigProvider:
        """設定プロバイダーインスタンス作成"""

        # キャッシュキー生成
        cache_key = self._generate_cache_key(provider_type, plugin_name, config)

        # キャッシュからインスタンス取得
        if use_cache and cache_key in self._instance_cache:
            return self._instance_cache[cache_key]

        # 設定検証
        validated_config = self._validate_config(provider_type, config, plugin_name)

        # プロバイダークラス取得
        provider_class = self._get_provider_class(provider_type, plugin_name)

        try:
            # インスタンス作成
            instance = self._create_provider_instance(provider_class, validated_config)

            # キャッシュに保存
            if use_cache:
                self._instance_cache[cache_key] = instance

            return instance

        except Exception as e:
            raise ConfigurationError(
                f"Failed to create config provider instance of type {provider_type.value}",
                config_key=f"config.provider.{provider_type.value}",
                cause=e
            ) from e

    def create_hierarchical_provider(
        self,
        provider_configs: List[Dict[str, Any]],
        merge_strategy: str = "override"  # "override", "merge", "append"
    ) -> 'HierarchicalConfigProvider':
        """階層設定プロバイダー作成"""

        providers = []

        for config in provider_configs:
            provider_type = ConfigProviderType(config["type"])
            plugin_name = config.get("plugin_name")
            provider_config = config.get("config", {})
            priority = config.get("priority", 0)

            provider = self.create_config_provider(
                provider_type,
                provider_config,
                plugin_name,
                use_cache=False
            )

            providers.append({
                "provider": provider,
                "priority": priority,
                "name": config.get("name", provider_type.value)
            })

        # 優先度順にソート
        providers.sort(key=lambda x: x["priority"], reverse=True)

        # 階層プロバイダー作成
        from ..config.hierarchical_provider import HierarchicalConfigProvider

        return HierarchicalConfigProvider(
            providers=[p["provider"] for p in providers],
            merge_strategy=merge_strategy
        )

    def create_encrypted_provider(
        self,
        base_provider: IConfigProvider,
        encryption_key: str,
        encrypted_keys: List[str] = None
    ) -> 'EncryptedConfigProvider':
        """暗号化設定プロバイダー作成"""

        # 暗号化プロバイダー作成
        from ..config.encrypted_provider import EncryptedConfigProvider

        return EncryptedConfigProvider(
            base_provider=base_provider,
            encryption_key=encryption_key,
            encrypted_keys=encrypted_keys or []
        )

    def create_cached_provider(
        self,
        base_provider: IConfigProvider,
        cache_ttl_seconds: int = 300,
        max_cache_size: int = 1000
    ) -> 'CachedConfigProvider':
        """キャッシュ設定プロバイダー作成"""

        # キャッシュプロバイダー作成
        from ..config.cached_provider import CachedConfigProvider

        return CachedConfigProvider(
            base_provider=base_provider,
            cache_ttl_seconds=cache_ttl_seconds,
            max_cache_size=max_cache_size
        )

    def create_config_watcher(
        self,
        provider: IConfigProvider,
        watched_keys: List[str],
        callback: callable,
        polling_interval_seconds: int = 30
    ) -> 'ConfigWatcher':
        """設定監視システム作成"""

        # 設定監視システム作成
        from ..config.config_watcher import ConfigWatcher

        watcher = ConfigWatcher(
            provider=provider,
            watched_keys=watched_keys,
            callback=callback,
            polling_interval_seconds=polling_interval_seconds
        )

        # 監視システムを登録
        for key in watched_keys:
            if key not in self._watchers:
                self._watchers[key] = []
            self._watchers[key].append(watcher)

        return watcher

    def validate_configuration(
        self,
        provider: IConfigProvider,
        schema: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """設定検証"""

        errors = {}
        warnings = {}

        config_data = provider.get_all()

        # 必須フィールドチェック
        for field_name, field_schema in schema.items():
            if field_schema.get("required", False) and field_name not in config_data:
                errors.setdefault("missing_fields", []).append(field_name)

            if field_name in config_data:
                value = config_data[field_name]

                # 型チェック
                expected_type = field_schema.get("type")
                if expected_type and not isinstance(value, expected_type):
                    errors.setdefault("type_errors", []).append({
                        "field": field_name,
                        "expected": expected_type.__name__,
                        "actual": type(value).__name__
                    })

                # 値範囲チェック
                min_value = field_schema.get("min")
                max_value = field_schema.get("max")
                if min_value is not None and value < min_value:
                    errors.setdefault("range_errors", []).append({
                        "field": field_name,
                        "value": value,
                        "min": min_value
                    })
                if max_value is not None and value > max_value:
                    errors.setdefault("range_errors", []).append({
                        "field": field_name,
                        "value": value,
                        "max": max_value
                    })

                # 列挙値チェック
                allowed_values = field_schema.get("enum")
                if allowed_values and value not in allowed_values:
                    errors.setdefault("enum_errors", []).append({
                        "field": field_name,
                        "value": value,
                        "allowed": allowed_values
                    })

        # 非推奨フィールドチェック
        for field_name, value in config_data.items():
            if field_name in schema:
                field_schema = schema[field_name]
                if field_schema.get("deprecated", False):
                    warnings.setdefault("deprecated_fields", []).append({
                        "field": field_name,
                        "replacement": field_schema.get("replacement")
                    })

        return {"errors": errors, "warnings": warnings}

    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """利用可能プロバイダー一覧取得"""

        available = {}

        # 組み込みプロバイダー
        for provider_type in self._provider_registry:
            config_schema = self._config_schemas.get(provider_type, {})
            available[provider_type.value] = {
                "type": "builtin",
                "name": provider_type.value,
                "config_schema": config_schema,
                "supported_features": self._get_provider_features(provider_type)
            }

        # プラグインプロバイダー
        for plugin_name, provider_class in self._plugin_registry.items():
            try:
                temp_instance = provider_class({})
                if hasattr(temp_instance, 'get_metadata'):
                    metadata = temp_instance.get_metadata()
                    available[plugin_name] = {
                        "type": "plugin",
                        "name": metadata.get("name", plugin_name),
                        "version": metadata.get("version", "unknown"),
                        "description": metadata.get("description", ""),
                        "config_schema": self._config_schemas.get(
                            ConfigProviderType.PLUGIN, {}
                        ).get(plugin_name, {})
                    }
                else:
                    available[plugin_name] = {
                        "type": "plugin",
                        "name": plugin_name
                    }
            except Exception:
                available[plugin_name] = {
                    "type": "plugin",
                    "name": plugin_name,
                    "status": "unavailable"
                }

        return available

    def clear_cache(self):
        """インスタンスキャッシュクリア"""
        # アクティブな接続を適切に閉じる
        for instance in self._instance_cache.values():
            if hasattr(instance, 'close'):
                try:
                    instance.close()
                except Exception:
                    pass

        self._instance_cache.clear()

    def stop_all_watchers(self):
        """全監視システム停止"""
        all_watchers = set()
        for watchers in self._watchers.values():
            all_watchers.update(watchers)

        for watcher in all_watchers:
            if hasattr(watcher, 'stop'):
                try:
                    watcher.stop()
                except Exception:
                    pass

        self._watchers.clear()

    def _validate_config(
        self,
        provider_type: ConfigProviderType,
        config: Optional[Dict[str, Any]],
        plugin_name: Optional[str]
    ) -> Dict[str, Any]:
        """設定検証"""

        if config is None:
            config = {}

        # スキーマ取得
        if provider_type == ConfigProviderType.PLUGIN and plugin_name:
            schema = self._config_schemas.get(ConfigProviderType.PLUGIN, {}).get(plugin_name, {})
        else:
            schema = self._config_schemas.get(provider_type, {})

        validated_config = {}

        # 必須フィールドチェックとデフォルト値適用
        for field_name, field_schema in schema.items():
            if field_schema.get("required", False) and field_name not in config:
                raise ValidationError(
                    f"Required configuration field '{field_name}' is missing for {provider_type.value}",
                    field_name=field_name,
                    validation_rules=["required"]
                )

            if field_name not in config and "default" in field_schema:
                validated_config[field_name] = field_schema["default"]
            elif field_name in config:
                # 型チェック
                expected_type = field_schema.get("type")
                if expected_type and not isinstance(config[field_name], expected_type):
                    raise ValidationError(
                        f"Configuration field '{field_name}' must be of type {expected_type.__name__}",
                        field_name=field_name,
                        invalid_value=config[field_name],
                        validation_rules=[f"type:{expected_type.__name__}"]
                    )
                validated_config[field_name] = config[field_name]

        # 追加設定フィールドも含める
        for field_name, field_value in config.items():
            if field_name not in validated_config:
                validated_config[field_name] = field_value

        return validated_config

    def _get_provider_class(
        self,
        provider_type: ConfigProviderType,
        plugin_name: Optional[str]
    ) -> Type[IConfigProvider]:
        """プロバイダークラス取得"""

        if provider_type == ConfigProviderType.PLUGIN:
            if not plugin_name:
                raise ConfigurationError(
                    "Plugin name is required for plugin config provider type",
                    config_key="plugin_name"
                )

            if plugin_name not in self._plugin_registry:
                raise ConfigurationError(
                    f"Plugin config provider '{plugin_name}' is not registered",
                    config_key=f"config.plugin.{plugin_name}"
                )

            return self._plugin_registry[plugin_name]

        else:
            if provider_type not in self._provider_registry:
                raise ConfigurationError(
                    f"Config provider type '{provider_type.value}' is not registered",
                    config_key=f"config.provider.{provider_type.value}"
                )

            return self._provider_registry[provider_type]

    def _create_provider_instance(
        self,
        provider_class: Type[IConfigProvider],
        config: Dict[str, Any]
    ) -> IConfigProvider:
        """プロバイダーインスタンス作成"""
        return provider_class(config)

    def _get_provider_features(self, provider_type: ConfigProviderType) -> List[str]:
        """プロバイダー機能取得"""
        features = ["get", "set", "delete", "exists", "get_all"]

        if provider_type in [ConfigProviderType.DATABASE, ConfigProviderType.REMOTE]:
            features.extend(["persistence", "remote_access", "concurrent_access"])

        if provider_type == ConfigProviderType.FILE:
            features.extend(["persistence", "backup", "version_control"])

        if provider_type == ConfigProviderType.VAULT:
            features.extend(["encryption", "access_control", "audit_log"])

        if provider_type == ConfigProviderType.KUBERNETES:
            features.extend(["watch_changes", "automatic_reload"])

        return features

    def _generate_cache_key(
        self,
        provider_type: ConfigProviderType,
        plugin_name: Optional[str],
        config: Optional[Dict[str, Any]]
    ) -> str:
        """キャッシュキー生成"""
        import hashlib
        import json

        key_data = {
            "type": provider_type.value,
            "plugin": plugin_name,
            "config": config or {}
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

# グローバルファクトリーインスタンス
_global_config_factory: Optional[ConfigProviderFactory] = None

def get_config_factory() -> ConfigProviderFactory:
    """グローバル設定ファクトリー取得"""
    global _global_config_factory
    if _global_config_factory is None:
        _global_config_factory = ConfigProviderFactory()
    return _global_config_factory

def create_config_provider(
    provider_type: ConfigProviderType,
    config: Optional[Dict[str, Any]] = None,
    plugin_name: Optional[str] = None
) -> IConfigProvider:
    """設定プロバイダー作成（便利関数）"""
    factory = get_config_factory()
    return factory.create_config_provider(provider_type, config, plugin_name)

# 設定ヘルパー関数

def load_config_from_file(file_path: str, format_type: Optional[str] = None) -> Dict[str, Any]:
    """ファイルから設定読み込み"""

    if format_type is None:
        # ファイル拡張子から形式を推定
        ext = Path(file_path).suffix.lower()
        format_map = {
            '.json': 'json',
            '.yaml': 'yaml', '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini', '.cfg': 'ini',
            '.xml': 'xml',
            '.properties': 'properties'
        }
        format_type = format_map.get(ext, 'json')

    provider = create_config_provider(
        ConfigProviderType.FILE,
        {"file_path": file_path, "format": format_type}
    )

    return provider.get_all()

def load_config_from_env(prefix: str = "", separator: str = "_") -> Dict[str, Any]:
    """環境変数から設定読み込み"""

    provider = create_config_provider(
        ConfigProviderType.ENVIRONMENT,
        {"prefix": prefix, "separator": separator}
    )

    return provider.get_all()
