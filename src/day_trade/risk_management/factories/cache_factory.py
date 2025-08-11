#!/usr/bin/env python3
"""
Cache Provider Factory
キャッシュプロバイダーファクトリー

様々なキャッシュプロバイダーの動的生成とプラグイン管理
"""

import importlib
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from ..exceptions.risk_exceptions import ConfigurationError, ValidationError
from ..interfaces.cache_interfaces import (
    ICacheEvictionPolicy,
    ICacheProvider,
    ICacheSerializer,
)


class CacheProviderType(Enum):
    """キャッシュプロバイダータイプ"""

    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"
    PLUGIN = "plugin"


class SerializerType(Enum):
    """シリアライザータイプ"""

    PICKLE = "pickle"
    JSON = "json"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"
    CUSTOM = "custom"


class EvictionPolicyType(Enum):
    """立ち退きポリシータイプ"""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    RANDOM = "random"
    NONE = "none"


class CacheProviderFactory:
    """キャッシュプロバイダーファクトリー"""

    def __init__(self):
        self._provider_registry: Dict[CacheProviderType, Type[ICacheProvider]] = {}
        self._serializer_registry: Dict[SerializerType, Type[ICacheSerializer]] = {}
        self._eviction_policy_registry: Dict[
            EvictionPolicyType, Type[ICacheEvictionPolicy]
        ] = {}
        self._plugin_registry: Dict[str, Type[ICacheProvider]] = {}
        self._config_schemas: Dict[CacheProviderType, Dict[str, Any]] = {}
        self._instance_cache: Dict[str, ICacheProvider] = {}

        # 組み込みプロバイダーを登録
        self._register_builtin_providers()
        self._register_builtin_serializers()
        self._register_builtin_eviction_policies()

    def _register_builtin_providers(self):
        """組み込みプロバイダー登録"""
        try:
            # メモリキャッシュ
            self.register_provider(
                CacheProviderType.MEMORY,
                "src.day_trade.risk_management.cache.memory_cache",
                "MemoryCacheProvider",
                {
                    "max_size": {"type": int, "default": 1000},
                    "ttl_seconds": {"type": int, "default": 3600},
                    "eviction_policy": {"type": str, "default": "lru"},
                    "thread_safe": {"type": bool, "default": True},
                },
            )

            # Redisキャッシュ
            self.register_provider(
                CacheProviderType.REDIS,
                "src.day_trade.risk_management.cache.redis_cache",
                "RedisCacheProvider",
                {
                    "host": {"type": str, "default": "localhost"},
                    "port": {"type": int, "default": 6379},
                    "db": {"type": int, "default": 0},
                    "password": {"type": str, "required": False},
                    "connection_pool_size": {"type": int, "default": 10},
                    "socket_timeout": {"type": float, "default": 2.0},
                    "socket_connect_timeout": {"type": float, "default": 2.0},
                },
            )

            # ファイルキャッシュ
            self.register_provider(
                CacheProviderType.FILE,
                "src.day_trade.risk_management.cache.file_cache",
                "FileCacheProvider",
                {
                    "cache_dir": {"type": str, "default": "./cache"},
                    "max_file_size_mb": {"type": int, "default": 100},
                    "max_cache_size_gb": {"type": int, "default": 1},
                    "cleanup_interval_hours": {"type": int, "default": 24},
                    "compression": {"type": bool, "default": True},
                },
            )

            # 分散キャッシュ
            self.register_provider(
                CacheProviderType.DISTRIBUTED,
                "src.day_trade.risk_management.cache.distributed_cache",
                "DistributedCacheProvider",
                {
                    "nodes": {"type": list, "required": True},
                    "replication_factor": {"type": int, "default": 2},
                    "consistency_level": {"type": str, "default": "quorum"},
                    "partition_strategy": {"type": str, "default": "hash_ring"},
                },
            )

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to register some builtin cache providers: {e}")

    def _register_builtin_serializers(self):
        """組み込みシリアライザー登録"""
        try:
            from ..cache.serializers import (
                JsonSerializer,
                MsgPackSerializer,
                PickleSerializer,
            )

            self._serializer_registry.update(
                {
                    SerializerType.PICKLE: PickleSerializer,
                    SerializerType.JSON: JsonSerializer,
                    SerializerType.MSGPACK: MsgPackSerializer,
                }
            )

        except ImportError:
            pass  # シリアライザーが利用できない場合は無視

    def _register_builtin_eviction_policies(self):
        """組み込み立ち退きポリシー登録"""
        try:
            from ..cache.eviction_policies import (
                FIFOEvictionPolicy,
                LFUEvictionPolicy,
                LRUEvictionPolicy,
                RandomEvictionPolicy,
                TTLEvictionPolicy,
            )

            self._eviction_policy_registry.update(
                {
                    EvictionPolicyType.LRU: LRUEvictionPolicy,
                    EvictionPolicyType.LFU: LFUEvictionPolicy,
                    EvictionPolicyType.FIFO: FIFOEvictionPolicy,
                    EvictionPolicyType.TTL: TTLEvictionPolicy,
                    EvictionPolicyType.RANDOM: RandomEvictionPolicy,
                }
            )

        except ImportError:
            pass  # ポリシーが利用できない場合は無視

    def register_provider(
        self,
        provider_type: CacheProviderType,
        module_path: str,
        class_name: str,
        config_schema: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """プロバイダー登録"""
        try:
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)

            if not issubclass(provider_class, ICacheProvider):
                raise ConfigurationError(
                    f"Cache provider class {class_name} must implement ICacheProvider interface",
                    config_key=f"cache.provider.{provider_type.value}",
                )

            self._provider_registry[provider_type] = provider_class

            if config_schema:
                self._config_schemas[provider_type] = config_schema

            return True

        except Exception as e:
            raise ConfigurationError(
                f"Failed to register cache provider type {provider_type.value}",
                config_key=f"cache.provider.{provider_type.value}",
                cause=e,
            ) from e

    def register_plugin_provider(
        self,
        plugin_name: str,
        provider_class: Type[ICacheProvider],
        config_schema: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """プラグインプロバイダー登録"""
        try:
            if not issubclass(provider_class, ICacheProvider):
                raise ConfigurationError(
                    f"Plugin cache provider {plugin_name} must implement ICacheProvider interface",
                    config_key=f"cache.plugin.{plugin_name}",
                )

            self._plugin_registry[plugin_name] = provider_class

            if config_schema:
                self._config_schemas[
                    CacheProviderType.PLUGIN
                ] = self._config_schemas.get(CacheProviderType.PLUGIN, {})
                self._config_schemas[CacheProviderType.PLUGIN][
                    plugin_name
                ] = config_schema

            return True

        except Exception as e:
            raise ConfigurationError(
                f"Failed to register plugin cache provider {plugin_name}",
                config_key=f"cache.plugin.{plugin_name}",
                cause=e,
            ) from e

    def create_provider(
        self,
        provider_type: CacheProviderType,
        config: Optional[Dict[str, Any]] = None,
        plugin_name: Optional[str] = None,
        use_cache: bool = True,
    ) -> ICacheProvider:
        """プロバイダーインスタンス作成"""

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
            # シリアライザー作成
            serializer = self._create_serializer(
                validated_config.get("serializer", "pickle")
            )

            # 立ち退きポリシー作成
            eviction_policy = self._create_eviction_policy(
                validated_config.get("eviction_policy", "lru"), validated_config
            )

            # インスタンス作成
            instance = self._create_provider_instance(
                provider_class, validated_config, serializer, eviction_policy
            )

            # キャッシュに保存
            if use_cache:
                self._instance_cache[cache_key] = instance

            return instance

        except Exception as e:
            raise ConfigurationError(
                f"Failed to create cache provider instance of type {provider_type.value}",
                config_key=f"cache.provider.{provider_type.value}",
                cause=e,
            ) from e

    def create_hybrid_provider(
        self,
        l1_config: Dict[str, Any],  # 高速キャッシュ（メモリなど）
        l2_config: Dict[str, Any],  # 大容量キャッシュ（Redisなど）
        promotion_policy: str = "frequency",
        demotion_policy: str = "lru",
    ) -> ICacheProvider:
        """ハイブリッドキャッシュプロバイダー作成"""

        l1_provider = self.create_provider(
            CacheProviderType(l1_config["type"]),
            l1_config.get("config", {}),
            use_cache=False,
        )

        l2_provider = self.create_provider(
            CacheProviderType(l2_config["type"]),
            l2_config.get("config", {}),
            use_cache=False,
        )

        # ハイブリッドプロバイダー作成
        from ..cache.hybrid_cache import HybridCacheProvider

        return HybridCacheProvider(
            l1_provider=l1_provider,
            l2_provider=l2_provider,
            promotion_policy=promotion_policy,
            demotion_policy=demotion_policy,
        )

    def create_distributed_cluster(
        self,
        node_configs: List[Dict[str, Any]],
        replication_factor: int = 2,
        consistency_level: str = "quorum",
    ) -> ICacheProvider:
        """分散クラスターキャッシュ作成"""

        nodes = []
        for node_config in node_configs:
            provider = self.create_provider(
                CacheProviderType(node_config["type"]),
                node_config.get("config", {}),
                use_cache=False,
            )
            nodes.append(provider)

        # 分散プロバイダー作成
        from ..cache.distributed_cache import DistributedCacheProvider

        return DistributedCacheProvider(
            nodes=nodes,
            replication_factor=replication_factor,
            consistency_level=consistency_level,
        )

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
                "supported_features": self._get_provider_features(provider_type),
            }

        # プラグインプロバイダー
        for plugin_name, provider_class in self._plugin_registry.items():
            try:
                temp_instance = provider_class({})
                if hasattr(temp_instance, "get_metadata"):
                    metadata = temp_instance.get_metadata()
                    available[plugin_name] = {
                        "type": "plugin",
                        "name": metadata.get("name", plugin_name),
                        "version": metadata.get("version", "unknown"),
                        "description": metadata.get("description", ""),
                        "config_schema": self._config_schemas.get(
                            CacheProviderType.PLUGIN, {}
                        ).get(plugin_name, {}),
                    }
                else:
                    available[plugin_name] = {"type": "plugin", "name": plugin_name}
            except Exception:
                available[plugin_name] = {
                    "type": "plugin",
                    "name": plugin_name,
                    "status": "unavailable",
                }

        return available

    def clear_cache(self):
        """インスタンスキャッシュクリア"""
        # アクティブな接続を適切に閉じる
        for instance in self._instance_cache.values():
            if hasattr(instance, "close"):
                try:
                    instance.close()
                except Exception:
                    pass

        self._instance_cache.clear()

    def _validate_config(
        self,
        provider_type: CacheProviderType,
        config: Optional[Dict[str, Any]],
        plugin_name: Optional[str],
    ) -> Dict[str, Any]:
        """設定検証"""

        if config is None:
            config = {}

        # スキーマ取得
        if provider_type == CacheProviderType.PLUGIN and plugin_name:
            schema = self._config_schemas.get(CacheProviderType.PLUGIN, {}).get(
                plugin_name, {}
            )
        else:
            schema = self._config_schemas.get(provider_type, {})

        validated_config = {}

        # 必須フィールドチェックとデフォルト値適用
        for field_name, field_schema in schema.items():
            if field_schema.get("required", False) and field_name not in config:
                raise ValidationError(
                    f"Required configuration field '{field_name}' is missing",
                    field_name=field_name,
                    validation_rules=["required"],
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
                        validation_rules=[f"type:{expected_type.__name__}"],
                    )
                validated_config[field_name] = config[field_name]

        # 追加設定フィールドも含める
        for field_name, field_value in config.items():
            if field_name not in validated_config:
                validated_config[field_name] = field_value

        return validated_config

    def _get_provider_class(
        self, provider_type: CacheProviderType, plugin_name: Optional[str]
    ) -> Type[ICacheProvider]:
        """プロバイダークラス取得"""

        if provider_type == CacheProviderType.PLUGIN:
            if not plugin_name:
                raise ConfigurationError(
                    "Plugin name is required for plugin cache provider type",
                    config_key="plugin_name",
                )

            if plugin_name not in self._plugin_registry:
                raise ConfigurationError(
                    f"Plugin cache provider '{plugin_name}' is not registered",
                    config_key=f"cache.plugin.{plugin_name}",
                )

            return self._plugin_registry[plugin_name]

        else:
            if provider_type not in self._provider_registry:
                raise ConfigurationError(
                    f"Cache provider type '{provider_type.value}' is not registered",
                    config_key=f"cache.provider.{provider_type.value}",
                )

            return self._provider_registry[provider_type]

    def _create_serializer(self, serializer_type: str) -> ICacheSerializer:
        """シリアライザー作成"""
        try:
            serializer_enum = SerializerType(serializer_type.lower())
            if serializer_enum in self._serializer_registry:
                serializer_class = self._serializer_registry[serializer_enum]
                return serializer_class()
            else:
                # デフォルトのPickleシリアライザー
                from ..cache.serializers import PickleSerializer

                return PickleSerializer()

        except (ValueError, KeyError):
            # 不明なシリアライザーの場合はPickleを使用
            from ..cache.serializers import PickleSerializer

            return PickleSerializer()

    def _create_eviction_policy(
        self, policy_type: str, config: Dict[str, Any]
    ) -> ICacheEvictionPolicy:
        """立ち退きポリシー作成"""
        try:
            policy_enum = EvictionPolicyType(policy_type.lower())
            if policy_enum in self._eviction_policy_registry:
                policy_class = self._eviction_policy_registry[policy_enum]
                return policy_class(config)
            else:
                # デフォルトのLRUポリシー
                from ..cache.eviction_policies import LRUEvictionPolicy

                return LRUEvictionPolicy(config)

        except (ValueError, KeyError):
            # 不明なポリシーの場合はLRUを使用
            from ..cache.eviction_policies import LRUEvictionPolicy

            return LRUEvictionPolicy(config)

    def _create_provider_instance(
        self,
        provider_class: Type[ICacheProvider],
        config: Dict[str, Any],
        serializer: ICacheSerializer,
        eviction_policy: ICacheEvictionPolicy,
    ) -> ICacheProvider:
        """プロバイダーインスタンス作成"""

        # 設定にシリアライザーとポリシーを追加
        enhanced_config = config.copy()
        enhanced_config["serializer"] = serializer
        enhanced_config["eviction_policy"] = eviction_policy

        return provider_class(enhanced_config)

    def _get_provider_features(self, provider_type: CacheProviderType) -> List[str]:
        """プロバイダー機能取得"""
        features = ["get", "set", "delete", "exists"]

        if provider_type in [CacheProviderType.REDIS, CacheProviderType.DISTRIBUTED]:
            features.extend(["persistence", "clustering", "replication"])

        if provider_type == CacheProviderType.MEMORY:
            features.extend(["high_performance", "thread_safe"])

        if provider_type == CacheProviderType.FILE:
            features.extend(["persistence", "large_capacity"])

        return features

    def _generate_cache_key(
        self,
        provider_type: CacheProviderType,
        plugin_name: Optional[str],
        config: Optional[Dict[str, Any]],
    ) -> str:
        """キャッシュキー生成"""
        import hashlib
        import json

        key_data = {
            "type": provider_type.value,
            "plugin": plugin_name,
            "config": config or {},
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


# グローバルファクトリーインスタンス
_global_cache_factory: Optional[CacheProviderFactory] = None


def get_cache_factory() -> CacheProviderFactory:
    """グローバルキャッシュファクトリー取得"""
    global _global_cache_factory
    if _global_cache_factory is None:
        _global_cache_factory = CacheProviderFactory()
    return _global_cache_factory


def create_cache_provider(
    provider_type: CacheProviderType,
    config: Optional[Dict[str, Any]] = None,
    plugin_name: Optional[str] = None,
) -> ICacheProvider:
    """キャッシュプロバイダー作成（便利関数）"""
    factory = get_cache_factory()
    return factory.create_provider(provider_type, config, plugin_name)
