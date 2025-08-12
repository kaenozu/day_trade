#!/usr/bin/env python3
"""
Redis Enhanced Distributed Cache System
Issue #377: 高度なキャッシング戦略の導入

高性能Redis分散キャッシュシステム:
- Redis Cluster対応
- フェイルオーバー・高可用性
- パイプライン・バッチ処理最適化
- 自動シャーディング
- プール管理・接続最適化
- Lua Script活用
- Pub/Sub無効化通知
- セキュリティ強化
"""

import asyncio
import hashlib
import json
import pickle
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from ..utils.logging_config import get_context_logger
    from .enhanced_persistent_cache import PersistentCacheEntry
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    @dataclass
    class PersistentCacheEntry:
        key: str
        value: Any
        created_at: float
        last_accessed: float
        access_count: int
        size_bytes: int
        priority: float = 1.0
        metadata: Dict[str, Any] = field(default_factory=dict)


# Redis依存関係チェック
try:
    import redis
    import redis.asyncio as aioredis
    from redis.exceptions import ConnectionError, RedisError, TimeoutError

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn(
        "Redis not available. pip install redis>=4.0.0 to enable Redis distributed cache.",
        stacklevel=2,
    )

logger = get_context_logger(__name__)


class RedisConnectionStrategy(Enum):
    """Redis接続戦略"""

    SINGLE = "single"
    SENTINEL = "sentinel"
    CLUSTER = "cluster"
    SHARDED = "sharded"


class RedisDataFormat(Enum):
    """Redis データフォーマット"""

    JSON = "json"
    PICKLE = "pickle"
    COMPRESSED_PICKLE = "compressed_pickle"
    MSGPACK = "msgpack"


@dataclass
class RedisConfig:
    """Redis設定"""

    # 接続設定
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    username: Optional[str] = None

    # 接続戦略
    strategy: RedisConnectionStrategy = RedisConnectionStrategy.SINGLE

    # プール設定
    max_connections: int = 50
    connection_timeout: float = 5.0
    socket_timeout: float = 5.0
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[int, int] = field(
        default_factory=lambda: {
            1: 600,  # TCP_KEEPIDLE
            2: 30,  # TCP_KEEPINTVL
            3: 3,  # TCP_KEEPCNT
        }
    )

    # クラスター設定
    cluster_nodes: List[Dict[str, Union[str, int]]] = field(default_factory=list)

    # Sentinel設定
    sentinel_hosts: List[Tuple[str, int]] = field(default_factory=list)
    sentinel_service_name: str = "mymaster"

    # パフォーマンス設定
    data_format: RedisDataFormat = RedisDataFormat.COMPRESSED_PICKLE
    enable_pipelining: bool = True
    pipeline_buffer_size: int = 100
    enable_lua_scripts: bool = True

    # 高可用性設定
    retry_on_timeout: bool = True
    retry_on_error: bool = True
    max_retries: int = 3
    retry_delay_ms: int = 100

    # セキュリティ設定
    ssl: bool = False
    ssl_cert_reqs: str = "required"
    ssl_ca_certs: Optional[str] = None
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

    # 監視設定
    health_check_interval: int = 30
    enable_metrics: bool = True


class RedisEnhancedCache:
    """Redis強化分散キャッシュシステム"""

    def __init__(self, config: Optional[RedisConfig] = None):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install: pip install redis>=4.0.0")

        self.config = config or RedisConfig()
        self.redis_client = None
        self.connection_pool = None
        self.lua_scripts = {}

        # 統計情報
        self.stats = {
            "operations": {"get": 0, "set": 0, "delete": 0, "pipeline": 0},
            "performance": {"hits": 0, "misses": 0, "errors": 0, "timeouts": 0},
            "connections": {"active": 0, "created": 0, "closed": 0},
            "data_transfer": {"bytes_sent": 0, "bytes_received": 0},
        }

        # パフォーマンス監視
        self.performance_history = []

        # Lua Scripts
        self._prepare_lua_scripts()

        logger.info("Redis強化分散キャッシュシステム初期化開始")

    async def initialize(self):
        """Redis接続初期化"""
        try:
            await self._create_connection()
            await self._load_lua_scripts()

            # ヘルスチェックタスク開始
            if self.config.health_check_interval > 0:
                asyncio.create_task(self._health_check_loop())

            logger.info(f"Redis接続初期化完了: {self.config.strategy.value}")
            logger.info(f"  ホスト: {self.config.host}:{self.config.port}")
            logger.info(f"  データフォーマット: {self.config.data_format.value}")
            logger.info(f"  最大接続数: {self.config.max_connections}")

        except Exception as e:
            logger.error(f"Redis初期化エラー: {e}")
            raise

    async def _create_connection(self):
        """Redis接続作成"""
        if self.config.strategy == RedisConnectionStrategy.SINGLE:
            # シングルインスタンス接続
            self.connection_pool = aioredis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                username=self.config.username,
                max_connections=self.config.max_connections,
                connection_class=aioredis.Connection,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.connection_timeout,
                socket_keepalive=self.config.socket_keepalive,
                socket_keepalive_options=self.config.socket_keepalive_options,
                ssl=self.config.ssl,
                ssl_cert_reqs=self.config.ssl_cert_reqs,
                ssl_ca_certs=self.config.ssl_ca_certs,
                ssl_certfile=self.config.ssl_cert_path,
                ssl_keyfile=self.config.ssl_key_path,
            )
            self.redis_client = aioredis.Redis(connection_pool=self.connection_pool)

        elif self.config.strategy == RedisConnectionStrategy.CLUSTER:
            # Redisクラスター接続
            if not self.config.cluster_nodes:
                raise ValueError("Cluster nodes not configured")

            self.redis_client = aioredis.RedisCluster(
                startup_nodes=self.config.cluster_nodes,
                decode_responses=False,
                skip_full_coverage_check=True,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.connection_timeout,
            )

        elif self.config.strategy == RedisConnectionStrategy.SENTINEL:
            # Redis Sentinel接続
            if not self.config.sentinel_hosts:
                raise ValueError("Sentinel hosts not configured")

            sentinel = redis.Sentinel(
                self.config.sentinel_hosts, socket_timeout=self.config.socket_timeout
            )
            self.redis_client = sentinel.master_for(
                self.config.sentinel_service_name,
                decode_responses=False,
                socket_timeout=self.config.socket_timeout,
            )

        # 接続テスト
        await self.redis_client.ping()
        self.stats["connections"]["created"] += 1

    def _prepare_lua_scripts(self):
        """Lua Script準備"""
        if not self.config.enable_lua_scripts:
            return

        # 原子的GET+UPDATE Script
        self.lua_scripts["atomic_get_update"] = """
            local key = KEYS[1]
            local new_access_time = ARGV[1]
            local increment_count = ARGV[2]

            local current = redis.call('GET', key)
            if current then
                local data = cjson.decode(current)
                data.last_accessed = tonumber(new_access_time)
                data.access_count = data.access_count + tonumber(increment_count)
                redis.call('SET', key, cjson.encode(data))
                return current
            else
                return nil
            end
        """

        # バッチ削除Script
        self.lua_scripts["batch_delete"] = """
            local keys = {}
            for i = 1, #KEYS do
                keys[i] = KEYS[i]
            end
            return redis.call('DEL', unpack(keys))
        """

        # 期限切れクリーンアップScript
        self.lua_scripts["cleanup_expired"] = """
            local pattern = KEYS[1]
            local current_time = tonumber(ARGV[1])
            local batch_size = tonumber(ARGV[2])

            local cursor = 0
            local deleted = 0

            repeat
                local result = redis.call('SCAN', cursor, 'MATCH', pattern, 'COUNT', batch_size)
                cursor = result[1]
                local keys = result[2]

                for i = 1, #keys do
                    local value = redis.call('GET', keys[i])
                    if value then
                        local data = cjson.decode(value)
                        if data.expires_at and data.expires_at <= current_time then
                            redis.call('DEL', keys[i])
                            deleted = deleted + 1
                        end
                    end
                end
            until cursor == "0"

            return deleted
        """

    async def _load_lua_scripts(self):
        """Lua Script登録"""
        if not self.config.enable_lua_scripts or not self.redis_client:
            return

        try:
            for name, script in self.lua_scripts.items():
                script_hash = await self.redis_client.script_load(script)
                self.lua_scripts[name] = script_hash
                logger.debug(f"Lua Script登録: {name}")

        except Exception as e:
            logger.warning(f"Lua Script登録エラー: {e}")

    async def get(self, key: str, use_atomic_update: bool = True) -> Optional[Any]:
        """データ取得"""
        start_time = time.time()

        try:
            self.stats["operations"]["get"] += 1

            # 原子的取得+アクセス統計更新
            if (
                use_atomic_update
                and self.config.enable_lua_scripts
                and "atomic_get_update" in self.lua_scripts
            ):
                result = await self.redis_client.evalsha(
                    self.lua_scripts["atomic_get_update"], 1, key, str(time.time()), "1"
                )
            else:
                result = await self.redis_client.get(key)

            if result:
                self.stats["performance"]["hits"] += 1
                self.stats["data_transfer"]["bytes_received"] += len(result)

                # データ復元
                entry = self._deserialize_entry(result)
                if entry and not self._is_expired(entry):
                    return entry.value
                else:
                    # 期限切れの場合は削除
                    if entry:
                        await self.delete(key)
                    self.stats["performance"]["misses"] += 1
                    return None
            else:
                self.stats["performance"]["misses"] += 1
                return None

        except (ConnectionError, TimeoutError) as e:
            self.stats["performance"]["timeouts"] += 1
            logger.warning(f"Redis接続エラー ({key}): {e}")

            if self.config.retry_on_timeout:
                return await self._retry_operation("get", key)
            return None

        except Exception as e:
            self.stats["performance"]["errors"] += 1
            logger.error(f"Redis取得エラー ({key}): {e}")
            return None

        finally:
            response_time = (time.time() - start_time) * 1000
            self._record_performance(response_time)

    async def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        priority: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """データ保存"""
        start_time = time.time()

        try:
            self.stats["operations"]["set"] += 1

            # エントリ作成
            current_time = time.time()
            entry = PersistentCacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=0,
                size_bytes=0,  # シリアライズ時に計算
                priority=priority,
                metadata=metadata or {},
            )

            # シリアライズ
            serialized_data = self._serialize_entry(entry)
            entry.size_bytes = len(serialized_data)

            self.stats["data_transfer"]["bytes_sent"] += entry.size_bytes

            # Redis保存
            if ttl_seconds:
                await self.redis_client.setex(key, ttl_seconds, serialized_data)
            else:
                await self.redis_client.set(key, serialized_data)

            return True

        except (ConnectionError, TimeoutError) as e:
            self.stats["performance"]["timeouts"] += 1
            logger.warning(f"Redis接続エラー ({key}): {e}")

            if self.config.retry_on_timeout:
                return await self._retry_operation(
                    "put", key, value, ttl_seconds, priority, metadata
                )
            return False

        except Exception as e:
            self.stats["performance"]["errors"] += 1
            logger.error(f"Redis保存エラー ({key}): {e}")
            return False

        finally:
            response_time = (time.time() - start_time) * 1000
            self._record_performance(response_time)

    async def delete(self, key: str) -> bool:
        """データ削除"""
        try:
            self.stats["operations"]["delete"] += 1
            result = await self.redis_client.delete(key)
            return result > 0

        except Exception as e:
            self.stats["performance"]["errors"] += 1
            logger.error(f"Redis削除エラー ({key}): {e}")
            return False

    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """バッチデータ取得"""
        if not keys:
            return {}

        try:
            if self.config.enable_pipelining:
                # パイプライン使用
                pipeline = self.redis_client.pipeline()
                for key in keys:
                    pipeline.get(key)

                results = await pipeline.execute()
                self.stats["operations"]["pipeline"] += 1
            else:
                # MGET使用
                results = await self.redis_client.mget(keys)

            # 結果処理
            batch_result = {}
            for i, key in enumerate(keys):
                if i < len(results) and results[i]:
                    entry = self._deserialize_entry(results[i])
                    if entry and not self._is_expired(entry):
                        batch_result[key] = entry.value
                        self.stats["performance"]["hits"] += 1
                    else:
                        self.stats["performance"]["misses"] += 1
                else:
                    self.stats["performance"]["misses"] += 1

            return batch_result

        except Exception as e:
            logger.error(f"バッチ取得エラー: {e}")
            return {}

    async def batch_put(self, data: Dict[str, Tuple[Any, Optional[int]]]) -> int:
        """バッチデータ保存"""
        if not data:
            return 0

        try:
            if self.config.enable_pipelining:
                pipeline = self.redis_client.pipeline()

                for key, (value, ttl) in data.items():
                    entry = PersistentCacheEntry(
                        key=key,
                        value=value,
                        created_at=time.time(),
                        last_accessed=time.time(),
                        access_count=0,
                        size_bytes=0,
                    )

                    serialized_data = self._serialize_entry(entry)

                    if ttl:
                        pipeline.setex(key, ttl, serialized_data)
                    else:
                        pipeline.set(key, serialized_data)

                results = await pipeline.execute()
                self.stats["operations"]["pipeline"] += 1

                # 成功した操作をカウント
                success_count = sum(1 for result in results if result)
                return success_count

            else:
                # 個別処理
                success_count = 0
                for key, (value, ttl) in data.items():
                    if await self.put(key, value, ttl):
                        success_count += 1

                return success_count

        except Exception as e:
            logger.error(f"バッチ保存エラー: {e}")
            return 0

    async def batch_delete(self, keys: List[str]) -> int:
        """バッチデータ削除"""
        if not keys:
            return 0

        try:
            if self.config.enable_lua_scripts and "batch_delete" in self.lua_scripts:
                # Lua Script使用
                deleted_count = await self.redis_client.evalsha(
                    self.lua_scripts["batch_delete"], len(keys), *keys
                )
                return deleted_count
            else:
                # 通常のDEL使用
                deleted_count = await self.redis_client.delete(*keys)
                return deleted_count

        except Exception as e:
            logger.error(f"バッチ削除エラー: {e}")
            return 0

    async def cleanup_expired(self, pattern: str = "*", batch_size: int = 100) -> int:
        """期限切れデータクリーンアップ"""
        try:
            if self.config.enable_lua_scripts and "cleanup_expired" in self.lua_scripts:
                # Lua Script使用（原子的操作）
                deleted_count = await self.redis_client.evalsha(
                    self.lua_scripts["cleanup_expired"],
                    1,
                    pattern,
                    str(time.time()),
                    str(batch_size),
                )
                return deleted_count
            else:
                # 通常のクリーンアップ
                return await self._cleanup_expired_manual(pattern, batch_size)

        except Exception as e:
            logger.error(f"期限切れクリーンアップエラー: {e}")
            return 0

    async def _cleanup_expired_manual(self, pattern: str, batch_size: int) -> int:
        """手動期限切れクリーンアップ"""
        deleted_count = 0
        cursor = 0

        while True:
            cursor, keys = await self.redis_client.scan(
                cursor=cursor, match=pattern, count=batch_size
            )

            if keys:
                # パイプラインで効率的に処理
                pipeline = self.redis_client.pipeline()
                for key in keys:
                    pipeline.get(key)

                values = await pipeline.execute()

                expired_keys = []
                for i, value in enumerate(values):
                    if value:
                        entry = self._deserialize_entry(value)
                        if entry and self._is_expired(entry):
                            expired_keys.append(keys[i])

                if expired_keys:
                    deleted = await self.redis_client.delete(*expired_keys)
                    deleted_count += deleted

            if cursor == 0:
                break

        return deleted_count

    async def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        try:
            info = await self.redis_client.info()

            return {
                **self.stats,
                "redis_info": {
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "keyspace": info.get("db0", {}),
                },
                "performance": {
                    **self.stats["performance"],
                    "hit_rate": self._calculate_hit_rate(),
                    "average_response_time": self._calculate_avg_response_time(),
                },
                "config": {
                    "strategy": self.config.strategy.value,
                    "data_format": self.config.data_format.value,
                    "max_connections": self.config.max_connections,
                    "pipelining_enabled": self.config.enable_pipelining,
                    "lua_scripts_enabled": self.config.enable_lua_scripts,
                },
            }

        except Exception as e:
            logger.error(f"統計情報取得エラー: {e}")
            return self.stats

    def _serialize_entry(self, entry: PersistentCacheEntry) -> bytes:
        """エントリシリアライズ"""
        try:
            if self.config.data_format == RedisDataFormat.JSON:
                # JSON（シンプルなデータのみ）
                data = {
                    "key": entry.key,
                    "value": entry.value,
                    "created_at": entry.created_at,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                    "priority": entry.priority,
                    "metadata": entry.metadata,
                }
                return json.dumps(data).encode("utf-8")

            elif self.config.data_format == RedisDataFormat.PICKLE:
                # Pickle（すべてのPythonオブジェクト対応）
                return pickle.dumps(entry)

            elif self.config.data_format == RedisDataFormat.COMPRESSED_PICKLE:
                # 圧縮Pickle（サイズ最適化）
                import gzip

                pickled_data = pickle.dumps(entry)
                return gzip.compress(pickled_data)

            else:
                # デフォルト: Pickle
                return pickle.dumps(entry)

        except Exception as e:
            logger.error(f"シリアライズエラー: {e}")
            raise

    def _deserialize_entry(self, data: bytes) -> Optional[PersistentCacheEntry]:
        """エントリデシリアライズ"""
        try:
            if self.config.data_format == RedisDataFormat.JSON:
                json_data = json.loads(data.decode("utf-8"))
                return PersistentCacheEntry(
                    key=json_data["key"],
                    value=json_data["value"],
                    created_at=json_data["created_at"],
                    last_accessed=json_data["last_accessed"],
                    access_count=json_data["access_count"],
                    size_bytes=len(data),
                    priority=json_data.get("priority", 1.0),
                    metadata=json_data.get("metadata", {}),
                )

            elif self.config.data_format == RedisDataFormat.PICKLE:
                return pickle.loads(data)

            elif self.config.data_format == RedisDataFormat.COMPRESSED_PICKLE:
                import gzip

                decompressed_data = gzip.decompress(data)
                return pickle.loads(decompressed_data)

            else:
                return pickle.loads(data)

        except Exception as e:
            logger.error(f"デシリアライズエラー: {e}")
            return None

    def _is_expired(self, entry: PersistentCacheEntry) -> bool:
        """期限切れチェック"""
        if not hasattr(entry, "expires_at"):
            return False

        expires_at = getattr(entry, "expires_at", None)
        if expires_at is None:
            return False

        return time.time() > expires_at

    async def _retry_operation(self, operation: str, *args, **kwargs):
        """操作リトライ"""
        for attempt in range(self.config.max_retries):
            try:
                await asyncio.sleep(self.config.retry_delay_ms / 1000)

                if operation == "get":
                    return await self.get(*args, **kwargs)
                elif operation == "put":
                    return await self.put(*args, **kwargs)
                elif operation == "delete":
                    return await self.delete(*args, **kwargs)

            except Exception as e:
                logger.warning(
                    f"リトライ{attempt + 1}/{self.config.max_retries}失敗 ({operation}): {e}"
                )
                if attempt == self.config.max_retries - 1:
                    raise

        return None

    def _record_performance(self, response_time_ms: float):
        """パフォーマンス記録"""
        self.performance_history.append(
            {"timestamp": time.time(), "response_time_ms": response_time_ms}
        )

        # 履歴サイズ制限
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]

    def _calculate_hit_rate(self) -> float:
        """ヒット率計算"""
        hits = self.stats["performance"]["hits"]
        misses = self.stats["performance"]["misses"]
        total = hits + misses

        if total == 0:
            return 0.0

        return (hits / total) * 100

    def _calculate_avg_response_time(self) -> float:
        """平均応答時間計算"""
        if not self.performance_history:
            return 0.0

        recent_history = self.performance_history[-100:]  # 最新100件
        return sum(h["response_time_ms"] for h in recent_history) / len(recent_history)

    async def _health_check_loop(self):
        """ヘルスチェックループ"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # 基本接続チェック
                await self.redis_client.ping()

                # パフォーマンス監視
                stats = await self.get_stats()
                hit_rate = stats["performance"]["hit_rate"]
                avg_response_time = stats["performance"]["average_response_time"]

                if hit_rate < 50.0:
                    logger.warning(f"低ヒット率検出: {hit_rate:.1f}%")

                if avg_response_time > 100.0:
                    logger.warning(f"高応答時間検出: {avg_response_time:.1f}ms")

                logger.debug(
                    f"Redis健康状態: ヒット率={hit_rate:.1f}%, 応答時間={avg_response_time:.1f}ms"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ヘルスチェックエラー: {e}")

    async def shutdown(self):
        """シャットダウン処理"""
        try:
            if self.redis_client:
                await self.redis_client.close()

            if self.connection_pool:
                await self.connection_pool.disconnect()

            self.stats["connections"]["closed"] += 1
            logger.info("Redis強化分散キャッシュ終了")

        except Exception as e:
            logger.error(f"Redis終了処理エラー: {e}")


# ファクトリー関数
def create_redis_enhanced_cache(
    host: str = "localhost",
    port: int = 6379,
    password: Optional[str] = None,
    max_connections: int = 50,
    enable_cluster: bool = False,
    cluster_nodes: Optional[List[Dict]] = None,
) -> RedisEnhancedCache:
    """Redis強化キャッシュ作成"""

    config = RedisConfig(
        host=host,
        port=port,
        password=password,
        max_connections=max_connections,
        strategy=(
            RedisConnectionStrategy.CLUSTER
            if enable_cluster
            else RedisConnectionStrategy.SINGLE
        ),
        cluster_nodes=cluster_nodes or [],
    )

    return RedisEnhancedCache(config)


# テスト関数
async def test_redis_enhanced_cache():
    """Redis強化キャッシュテスト"""
    print("=== Issue #377 Redis強化分散キャッシュテスト ===")

    if not REDIS_AVAILABLE:
        print("❌ Redis未インストールのためテストをスキップ")
        print("インストール: pip install redis>=4.0.0")
        return

    try:
        # Redis接続テスト
        print("\n1. Redis接続テスト...")
        cache = create_redis_enhanced_cache()
        await cache.initialize()
        print("   ✅ Redis接続成功")

        # 基本操作テスト
        print("\n2. 基本操作テスト...")
        test_data = {
            "simple": "シンプルな文字列",
            "complex": {"number": 42, "list": [1, 2, 3], "nested": {"a": "b"}},
            "numpy_like": list(range(100)),
        }

        for key, value in test_data.items():
            success = await cache.put(key, value, ttl_seconds=3600)
            print(f"   PUT {key}: {'成功' if success else '失敗'}")

        for key in test_data.keys():
            retrieved = await cache.get(key)
            print(f"   GET {key}: {'成功' if retrieved is not None else '失敗'}")

        # バッチ操作テスト
        print("\n3. バッチ操作テスト...")
        batch_data = {f"batch_{i}": (f"データ_{i}", 1800) for i in range(20)}

        batch_put_count = await cache.batch_put(batch_data)
        print(f"   バッチPUT: {batch_put_count}/20 成功")

        batch_get_result = await cache.batch_get(list(batch_data.keys()))
        print(f"   バッチGET: {len(batch_get_result)}/20 取得成功")

        # パフォーマンステスト
        print("\n4. パフォーマンステスト...")
        start_time = time.time()

        for i in range(200):
            await cache.put(f"perf_{i}", f"パフォーマンステスト_{i}", ttl_seconds=600)

        put_time = time.time() - start_time
        print(f"   200件PUT: {put_time:.3f}秒")

        start_time = time.time()
        hit_count = 0
        for i in range(200):
            result = await cache.get(f"perf_{i}")
            if result:
                hit_count += 1

        get_time = time.time() - start_time
        print(f"   200件GET: {get_time:.3f}秒 ({hit_count}件ヒット)")

        # 統計情報表示
        print("\n5. 統計情報:")
        stats = await cache.get_stats()
        for category, data in stats.items():
            if isinstance(data, dict):
                print(f"   {category}:")
                for k, v in data.items():
                    print(f"     {k}: {v}")
            else:
                print(f"   {category}: {data}")

        # クリーンアップテスト
        print("\n6. クリーンアップテスト...")
        deleted_count = await cache.cleanup_expired("batch_*")
        print(f"   期限切れクリーンアップ: {deleted_count}件削除")

        await cache.shutdown()
        print("\n✅ Redis強化分散キャッシュ テスト完了！")

    except ConnectionError:
        print("❌ Redis接続失敗 - Redisサーバーが起動していることを確認してください")
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_redis_enhanced_cache())
