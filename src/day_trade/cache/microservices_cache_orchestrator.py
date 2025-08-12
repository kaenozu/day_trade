#!/usr/bin/env python3
"""
マイクロサービス対応キャッシュオーケストレーター
Issue #377 + #418: 高度キャッシング戦略とマイクロサービス連携

分散環境でのキャッシュ一貫性、レプリケーション、自動フェイルオーバーを実現
"""

import asyncio
import hashlib
import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis as AsyncRedis

    ASYNC_REDIS_AVAILABLE = True
except ImportError:
    ASYNC_REDIS_AVAILABLE = False

try:
    from ..utils.logging_config import get_context_logger
    from .distributed_cache_system import DistributedCacheManager
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)

    class DistributedCacheManager:
        pass


logger = get_context_logger(__name__)


class CacheConsistencyLevel(Enum):
    """キャッシュ一貫性レベル"""

    EVENTUAL = "eventual"  # 結果整合性
    STRONG = "strong"  # 強一貫性
    WEAK = "weak"  # 弱一貫性


class CacheRegion(Enum):
    """キャッシュ領域分類"""

    MARKET_DATA = "market_data"  # 市場データ（高速更新）
    ANALYSIS_RESULTS = "analysis"  # 分析結果（中期間保持）
    USER_SESSIONS = "user_sessions"  # ユーザーセッション（短期間）
    CONFIGURATIONS = "config"  # 設定データ（長期間）
    TRADING_POSITIONS = "positions"  # 取引ポジション（強一貫性必須）


@dataclass
class CacheEntry:
    """統合キャッシュエントリ"""

    key: str
    value: Any
    region: CacheRegion
    consistency_level: CacheConsistencyLevel
    ttl_seconds: int
    created_at: float
    version: int = 1
    metadata: Dict[str, Any] = None
    node_id: str = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.node_id is None:
            self.node_id = str(uuid.uuid4())[:8]

    @property
    def is_expired(self) -> bool:
        return time.time() > (self.created_at + self.ttl_seconds)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CacheReplicationStrategy(ABC):
    """キャッシュレプリケーション戦略インターフェース"""

    @abstractmethod
    async def replicate(self, entry: CacheEntry, nodes: List[str]) -> bool:
        """レプリケーション実行"""
        pass

    @abstractmethod
    async def verify_consistency(self, key: str, nodes: List[str]) -> bool:
        """一貫性検証"""
        pass


class MasterSlaveReplication(CacheReplicationStrategy):
    """マスター・スレーブ レプリケーション"""

    def __init__(self, master_node: str, slave_nodes: List[str]):
        self.master_node = master_node
        self.slave_nodes = slave_nodes
        self.replication_lag_threshold = 1.0  # seconds

    async def replicate(self, entry: CacheEntry, nodes: List[str]) -> bool:
        """マスターへの書き込み後、非同期でスレーブにレプリケート"""
        success_count = 0

        try:
            # 1. マスターに書き込み
            if await self._write_to_node(entry, self.master_node):
                success_count += 1

            # 2. スレーブに非同期レプリケート
            replication_tasks = []
            for slave_node in self.slave_nodes:
                if slave_node in nodes:
                    task = asyncio.create_task(
                        self._replicate_to_slave(entry, slave_node)
                    )
                    replication_tasks.append(task)

            if replication_tasks:
                results = await asyncio.gather(
                    *replication_tasks, return_exceptions=True
                )
                success_count += sum(1 for r in results if r is True)

            # マスター書き込み成功で成功とみなす（非同期レプリケーション）
            return success_count >= 1

        except Exception as e:
            logger.error(f"レプリケーション失敗 {entry.key}: {e}")
            return False

    async def verify_consistency(self, key: str, nodes: List[str]) -> bool:
        """マスター・スレーブ間の一貫性検証"""
        try:
            master_value = await self._read_from_node(key, self.master_node)
            if master_value is None:
                return True  # キーが存在しない場合は一貫

            # スレーブでの値を確認
            inconsistent_slaves = []
            for slave_node in self.slave_nodes:
                if slave_node in nodes:
                    slave_value = await self._read_from_node(key, slave_node)
                    if slave_value != master_value:
                        inconsistent_slaves.append(slave_node)

            if inconsistent_slaves:
                logger.warning(f"一貫性不整合検出 {key}: slaves={inconsistent_slaves}")
                # 不整合スレーブの修復
                await self._repair_inconsistent_slaves(
                    key, master_value, inconsistent_slaves
                )

            return len(inconsistent_slaves) == 0

        except Exception as e:
            logger.error(f"一貫性検証失敗 {key}: {e}")
            return False

    async def _write_to_node(self, entry: CacheEntry, node: str) -> bool:
        """指定ノードへの書き込み"""
        # 実装は具体的なバックエンドに依存
        return True

    async def _read_from_node(self, key: str, node: str) -> Any:
        """指定ノードからの読み込み"""
        # 実装は具体的なバックエンドに依存
        return None

    async def _replicate_to_slave(self, entry: CacheEntry, slave_node: str) -> bool:
        """スレーブへのレプリケーション"""
        try:
            await asyncio.sleep(0.01)  # レプリケーション遅延シミュレート
            return await self._write_to_node(entry, slave_node)
        except Exception as e:
            logger.error(f"スレーブレプリケーション失敗 {slave_node}: {e}")
            return False

    async def _repair_inconsistent_slaves(
        self, key: str, correct_value: Any, slave_nodes: List[str]
    ):
        """不整合スレーブの修復"""
        repair_tasks = []
        for slave_node in slave_nodes:
            # 正しい値でスレーブを修復
            entry = CacheEntry(
                key=key,
                value=correct_value,
                region=CacheRegion.MARKET_DATA,  # デフォルト
                consistency_level=CacheConsistencyLevel.STRONG,
                ttl_seconds=3600,
                created_at=time.time(),
            )
            task = asyncio.create_task(self._write_to_node(entry, slave_node))
            repair_tasks.append(task)

        if repair_tasks:
            await asyncio.gather(*repair_tasks, return_exceptions=True)
            logger.info(f"不整合スレーブ修復完了 {key}: {len(repair_tasks)}台")


class EventualConsistencyReplication(CacheReplicationStrategy):
    """結果整合性レプリケーション"""

    def __init__(self, nodes: List[str], min_success_ratio: float = 0.5):
        self.nodes = nodes
        self.min_success_ratio = min_success_ratio
        self.propagation_delay = 0.1  # seconds

    async def replicate(self, entry: CacheEntry, nodes: List[str]) -> bool:
        """結果整合性での並列レプリケーション"""
        target_nodes = [n for n in nodes if n in self.nodes]
        if not target_nodes:
            return False

        # 全ノードに並列書き込み
        replication_tasks = []
        for node in target_nodes:
            task = asyncio.create_task(self._write_with_retry(entry, node))
            replication_tasks.append(task)

        try:
            results = await asyncio.gather(*replication_tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            success_ratio = success_count / len(target_nodes)

            logger.debug(
                f"結果整合性レプリケーション {entry.key}: {success_count}/{len(target_nodes)} ノード成功"
            )

            return success_ratio >= self.min_success_ratio

        except Exception as e:
            logger.error(f"結果整合性レプリケーション失敗 {entry.key}: {e}")
            return False

    async def verify_consistency(self, key: str, nodes: List[str]) -> bool:
        """結果整合性検証（緩和条件）"""
        target_nodes = [n for n in nodes if n in self.nodes]
        if not target_nodes:
            return True

        # 複数ノードから値を読み取り
        read_tasks = []
        for node in target_nodes:
            task = asyncio.create_task(self._read_from_node(key, node))
            read_tasks.append(task)

        try:
            results = await asyncio.gather(*read_tasks, return_exceptions=True)
            values = [
                r for r in results if not isinstance(r, Exception) and r is not None
            ]

            if not values:
                return True  # 値が存在しない場合は一貫

            # 最も一般的な値が過半数を占めれば一貫とみなす
            value_counts = {}
            for value in values:
                value_hash = hashlib.sha256(str(value).encode()).hexdigest()
                value_counts[value_hash] = value_counts.get(value_hash, 0) + 1

            max_count = max(value_counts.values())
            consistency_ratio = max_count / len(values)

            is_consistent = consistency_ratio >= 0.6  # 60%以上で一貫とみなす

            if not is_consistent:
                logger.warning(
                    f"結果整合性違反 {key}: consistency_ratio={consistency_ratio}"
                )

            return is_consistent

        except Exception as e:
            logger.error(f"結果整合性検証失敗 {key}: {e}")
            return False

    async def _write_with_retry(
        self, entry: CacheEntry, node: str, max_retries: int = 2
    ) -> bool:
        """リトライ付き書き込み"""
        for attempt in range(max_retries + 1):
            try:
                await asyncio.sleep(self.propagation_delay * attempt)
                return await self._write_to_node(entry, node)
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"書き込み最終失敗 {node}: {e}")
                    return False
                await asyncio.sleep(0.1 * (2**attempt))  # Exponential backoff

        return False

    async def _write_to_node(self, entry: CacheEntry, node: str) -> bool:
        """ノードへの書き込み"""
        # 実装は具体的なバックエンドに依存
        return True

    async def _read_from_node(self, key: str, node: str) -> Any:
        """ノードからの読み込み"""
        # 実装は具体的なバックエンドに依存
        return None


class MicroservicesCacheOrchestrator:
    """マイクロサービス対応キャッシュオーケストレーター"""

    def __init__(
        self,
        cache_nodes: List[str] = None,
        replication_strategy: CacheReplicationStrategy = None,
        consistency_check_interval: int = 60,  # seconds
    ):
        self.cache_nodes = cache_nodes or ["redis://localhost:6379"]
        self.replication_strategy = (
            replication_strategy or EventualConsistencyReplication(self.cache_nodes)
        )
        self.consistency_check_interval = consistency_check_interval

        # 各サービス用のキャッシュ管理
        self.service_caches = {}
        self.region_configs = self._setup_region_configs()

        # パフォーマンス統計
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "replication_successes": 0,
            "replication_failures": 0,
            "consistency_violations": 0,
            "consistency_repairs": 0,
        }
        self._stats_lock = threading.RLock()

        # バックグラウンドタスク
        self._consistency_task = None
        self._running = False

        logger.info(
            f"マイクロサービスキャッシュオーケストレーター初期化: {len(self.cache_nodes)}ノード"
        )

    def _setup_region_configs(self) -> Dict[CacheRegion, Dict[str, Any]]:
        """リージョン別設定"""
        return {
            CacheRegion.MARKET_DATA: {
                "consistency_level": CacheConsistencyLevel.EVENTUAL,
                "default_ttl": 30,  # 30秒（高速更新データ）
                "replication_factor": 3,
                "priority": "high",
            },
            CacheRegion.ANALYSIS_RESULTS: {
                "consistency_level": CacheConsistencyLevel.EVENTUAL,
                "default_ttl": 1800,  # 30分（分析結果）
                "replication_factor": 2,
                "priority": "medium",
            },
            CacheRegion.TRADING_POSITIONS: {
                "consistency_level": CacheConsistencyLevel.STRONG,
                "default_ttl": 300,  # 5分（重要データ）
                "replication_factor": 5,
                "priority": "critical",
            },
            CacheRegion.USER_SESSIONS: {
                "consistency_level": CacheConsistencyLevel.WEAK,
                "default_ttl": 3600,  # 1時間（セッション）
                "replication_factor": 1,
                "priority": "low",
            },
            CacheRegion.CONFIGURATIONS: {
                "consistency_level": CacheConsistencyLevel.STRONG,
                "default_ttl": 86400,  # 24時間（設定）
                "replication_factor": 3,
                "priority": "high",
            },
        }

    def get_service_cache(self, service_name: str) -> DistributedCacheManager:
        """サービス専用キャッシュ取得"""
        if service_name not in self.service_caches:
            # サービス専用設定
            cache_config = {
                "host": "localhost",
                "port": 6379,
                "db": hash(service_name) % 16,  # サービス毎にDB分離
            }

            self.service_caches[service_name] = DistributedCacheManager(
                backend_type="redis" if ASYNC_REDIS_AVAILABLE else "memory",
                backend_config=cache_config,
                enable_fallback=True,
            )

            logger.info(
                f"サービス専用キャッシュ作成: {service_name} (DB={cache_config['db']})"
            )

        return self.service_caches[service_name]

    async def set(
        self,
        key: str,
        value: Any,
        region: CacheRegion,
        service_name: str = "default",
        ttl_seconds: Optional[int] = None,
        consistency_level: Optional[CacheConsistencyLevel] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """統合キャッシュ設定"""
        region_config = self.region_configs[region]

        # デフォルト値設定
        if ttl_seconds is None:
            ttl_seconds = region_config["default_ttl"]
        if consistency_level is None:
            consistency_level = region_config["consistency_level"]

        # キャッシュエントリ作成
        entry = CacheEntry(
            key=self._generate_cache_key(key, region, service_name),
            value=value,
            region=region,
            consistency_level=consistency_level,
            ttl_seconds=ttl_seconds,
            created_at=time.time(),
            metadata=metadata or {},
        )

        try:
            # 1. ローカルキャッシュに設定
            service_cache = self.get_service_cache(service_name)
            local_success = service_cache.set(entry.key, entry.to_dict(), ttl_seconds)

            # 2. レプリケーション戦略に基づく分散設定
            if consistency_level != CacheConsistencyLevel.WEAK:
                replication_success = await self.replication_strategy.replicate(
                    entry, self.cache_nodes
                )

                with self._stats_lock:
                    if replication_success:
                        self.stats["replication_successes"] += 1
                    else:
                        self.stats["replication_failures"] += 1

                # 強一貫性の場合はレプリケーション成功必須
                if consistency_level == CacheConsistencyLevel.STRONG:
                    success = local_success and replication_success
                else:
                    success = local_success or replication_success
            else:
                success = local_success

            if success:
                logger.debug(
                    f"キャッシュ設定成功 [{region.value}] {service_name}:{key}"
                )
            else:
                logger.warning(
                    f"キャッシュ設定失敗 [{region.value}] {service_name}:{key}"
                )

            return success

        except Exception as e:
            logger.error(f"キャッシュ設定エラー {service_name}:{key}: {e}")
            return False

    async def get(
        self,
        key: str,
        region: CacheRegion,
        service_name: str = "default",
        default: Any = None,
    ) -> Any:
        """統合キャッシュ取得"""
        cache_key = self._generate_cache_key(key, region, service_name)

        try:
            # 1. ローカルキャッシュから取得試行
            service_cache = self.get_service_cache(service_name)
            cached_entry = service_cache.get(cache_key)

            if cached_entry is not None:
                with self._stats_lock:
                    self.stats["cache_hits"] += 1

                # エントリ構造の場合は値を抽出
                if isinstance(cached_entry, dict) and "value" in cached_entry:
                    logger.debug(
                        f"キャッシュヒット [{region.value}] {service_name}:{key}"
                    )
                    return cached_entry["value"]
                else:
                    return cached_entry

            # 2. 分散キャッシュから取得試行（結果整合性対応）
            if region in [CacheRegion.MARKET_DATA, CacheRegion.TRADING_POSITIONS]:
                distributed_value = await self._get_from_distributed_nodes(cache_key)
                if distributed_value is not None:
                    # ローカルキャッシュに同期
                    service_cache.set(cache_key, distributed_value, 300)

                    with self._stats_lock:
                        self.stats["cache_hits"] += 1

                    logger.debug(
                        f"分散キャッシュヒット [{region.value}] {service_name}:{key}"
                    )
                    return distributed_value

            with self._stats_lock:
                self.stats["cache_misses"] += 1

            logger.debug(f"キャッシュミス [{region.value}] {service_name}:{key}")
            return default

        except Exception as e:
            logger.error(f"キャッシュ取得エラー {service_name}:{key}: {e}")
            return default

    async def delete(
        self,
        key: str,
        region: CacheRegion,
        service_name: str = "default",
    ) -> bool:
        """統合キャッシュ削除"""
        cache_key = self._generate_cache_key(key, region, service_name)

        try:
            # 1. ローカルキャッシュから削除
            service_cache = self.get_service_cache(service_name)
            local_success = service_cache.delete(cache_key)

            # 2. 分散ノードからも削除
            distributed_success = await self._delete_from_distributed_nodes(cache_key)

            success = local_success or distributed_success

            if success:
                logger.debug(
                    f"キャッシュ削除成功 [{region.value}] {service_name}:{key}"
                )

            return success

        except Exception as e:
            logger.error(f"キャッシュ削除エラー {service_name}:{key}: {e}")
            return False

    async def invalidate_region(
        self,
        region: CacheRegion,
        service_name: str = "default",
        pattern: str = "*",
    ) -> int:
        """リージョン全体のキャッシュ無効化"""
        try:
            service_cache = self.get_service_cache(service_name)
            region_pattern = f"{region.value}:{service_name}:{pattern}"

            cleared_count = service_cache.clear(region_pattern)

            logger.info(
                f"リージョンキャッシュ無効化 [{region.value}] {service_name}: {cleared_count}件"
            )

            return cleared_count

        except Exception as e:
            logger.error(f"リージョン無効化エラー [{region.value}] {service_name}: {e}")
            return 0

    async def start_consistency_monitoring(self):
        """一貫性監視開始"""
        if self._consistency_task is not None:
            return

        self._running = True
        self._consistency_task = asyncio.create_task(self._consistency_monitor_loop())
        logger.info("キャッシュ一貫性監視開始")

    async def stop_consistency_monitoring(self):
        """一貫性監視停止"""
        if self._consistency_task is None:
            return

        self._running = False
        self._consistency_task.cancel()

        try:
            await self._consistency_task
        except asyncio.CancelledError:
            pass

        self._consistency_task = None
        logger.info("キャッシュ一貫性監視停止")

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        with self._stats_lock:
            base_stats = self.stats.copy()

        # サービス別統計
        service_stats = {}
        for service_name, cache in self.service_caches.items():
            service_stats[service_name] = cache.get_stats()

        return {
            **base_stats,
            "service_caches": service_stats,
            "cache_nodes": len(self.cache_nodes),
            "active_services": len(self.service_caches),
            "hit_rate": base_stats["cache_hits"]
            / max(base_stats["cache_hits"] + base_stats["cache_misses"], 1),
            "replication_success_rate": base_stats["replication_successes"]
            / max(
                base_stats["replication_successes"]
                + base_stats["replication_failures"],
                1,
            ),
        }

    def _generate_cache_key(
        self, key: str, region: CacheRegion, service_name: str
    ) -> str:
        """統合キャッシュキー生成"""
        return f"{region.value}:{service_name}:{key}"

    async def _get_from_distributed_nodes(self, key: str) -> Any:
        """分散ノードからの取得"""
        # 実際の実装では各ノードにアクセス
        return None

    async def _delete_from_distributed_nodes(self, key: str) -> bool:
        """分散ノードからの削除"""
        # 実際の実装では各ノードにアクセス
        return True

    async def _consistency_monitor_loop(self):
        """一貫性監視ループ"""
        while self._running:
            try:
                await asyncio.sleep(self.consistency_check_interval)

                if not self._running:
                    break

                await self._check_consistency()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"一貫性監視エラー: {e}")
                await asyncio.sleep(10)  # エラー時は短い間隔で再試行

    async def _check_consistency(self):
        """一貫性チェック"""
        try:
            # クリティカルなリージョンの一貫性をチェック
            critical_regions = [
                CacheRegion.TRADING_POSITIONS,
                CacheRegion.CONFIGURATIONS,
            ]

            for region in critical_regions:
                # サンプルキーでの一貫性チェック（実際の実装では全キーをチェック）
                sample_key = f"{region.value}:sample"

                is_consistent = await self.replication_strategy.verify_consistency(
                    sample_key, self.cache_nodes
                )

                if not is_consistent:
                    with self._stats_lock:
                        self.stats["consistency_violations"] += 1

                    logger.warning(f"一貫性違反検出 [{region.value}]: {sample_key}")

                    # 修復処理
                    await self._repair_inconsistency(sample_key, region)

                    with self._stats_lock:
                        self.stats["consistency_repairs"] += 1

            logger.debug("一貫性チェック完了")

        except Exception as e:
            logger.error(f"一貫性チェックエラー: {e}")

    async def _repair_inconsistency(self, key: str, region: CacheRegion):
        """一貫性修復"""
        try:
            # 実際の実装では、マスターノードから正しい値を取得し、
            # 不整合ノードを修復する
            logger.info(f"一貫性修復実行 [{region.value}]: {key}")

        except Exception as e:
            logger.error(f"一貫性修復エラー {key}: {e}")


# グローバルインスタンス
_global_orchestrator: Optional[MicroservicesCacheOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_cache_orchestrator(
    cache_nodes: List[str] = None,
    replication_strategy: CacheReplicationStrategy = None,
) -> MicroservicesCacheOrchestrator:
    """グローバルキャッシュオーケストレーター取得"""
    global _global_orchestrator

    if _global_orchestrator is None:
        with _orchestrator_lock:
            if _global_orchestrator is None:
                _global_orchestrator = MicroservicesCacheOrchestrator(
                    cache_nodes=cache_nodes,
                    replication_strategy=replication_strategy,
                )

    return _global_orchestrator


# 便利なデコレータ
def microservice_cache(
    region: CacheRegion,
    service_name: str,
    ttl_seconds: int = None,
    consistency_level: CacheConsistencyLevel = None,
):
    """マイクロサービス対応キャッシュデコレータ"""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            orchestrator = get_cache_orchestrator()

            # キャッシュキー生成
            cache_key = (
                f"{func.__name__}:"
                + hashlib.sha256(str(args).encode() + str(kwargs).encode()).hexdigest()[
                    :16
                ]
            )

            # キャッシュ取得試行
            cached_result = await orchestrator.get(cache_key, region, service_name)
            if cached_result is not None:
                return cached_result

            # 関数実行
            result = await func(*args, **kwargs)

            # 結果をキャッシュ
            await orchestrator.set(
                cache_key,
                result,
                region,
                service_name,
                ttl_seconds=ttl_seconds,
                consistency_level=consistency_level,
            )

            return result

        def sync_wrapper(*args, **kwargs):
            # 同期関数の場合は非同期ラッパーを実行
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            finally:
                loop.close()

        # 関数が非同期かどうかを判定
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


if __name__ == "__main__":
    # テスト実行
    async def main():
        print(
            "=== Issue #377 + #418 マイクロサービス対応キャッシュオーケストレーター ==="
        )

        # 1. オーケストレーター初期化
        cache_nodes = ["redis://localhost:6379", "redis://localhost:6380"]
        replication_strategy = EventualConsistencyReplication(
            cache_nodes, min_success_ratio=0.5
        )

        orchestrator = MicroservicesCacheOrchestrator(
            cache_nodes=cache_nodes,
            replication_strategy=replication_strategy,
        )

        print("\n1. サービス別キャッシュテスト")

        # 市場データサービスのキャッシュ
        await orchestrator.set(
            "AAPL_price",
            150.25,
            CacheRegion.MARKET_DATA,
            service_name="market-data-service",
            ttl_seconds=30,
        )

        # 分析サービスのキャッシュ
        await orchestrator.set(
            "AAPL_analysis",
            {"trend": "bullish", "score": 0.85},
            CacheRegion.ANALYSIS_RESULTS,
            service_name="analysis-service",
            ttl_seconds=1800,
        )

        # 取引サービスのキャッシュ（強一貫性）
        await orchestrator.set(
            "position_123",
            {"symbol": "AAPL", "quantity": 100, "avg_price": 149.50},
            CacheRegion.TRADING_POSITIONS,
            service_name="trading-engine-service",
            consistency_level=CacheConsistencyLevel.STRONG,
        )

        print("✅ 各サービスにデータをキャッシュしました")

        # データ取得テスト
        market_price = await orchestrator.get(
            "AAPL_price", CacheRegion.MARKET_DATA, "market-data-service"
        )
        analysis_result = await orchestrator.get(
            "AAPL_analysis", CacheRegion.ANALYSIS_RESULTS, "analysis-service"
        )
        position_data = await orchestrator.get(
            "position_123", CacheRegion.TRADING_POSITIONS, "trading-engine-service"
        )

        print(f"市場データ: {market_price}")
        print(f"分析結果: {analysis_result}")
        print(f"ポジション: {position_data}")

        print("\n2. 統計情報")
        stats = orchestrator.get_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")

        print("\n3. デコレータテスト")

        @microservice_cache(
            CacheRegion.ANALYSIS_RESULTS, "analysis-service", ttl_seconds=300
        )
        async def complex_analysis(symbol: str, timeframe: str):
            await asyncio.sleep(0.1)  # 複雑な分析をシミュレート
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "result": "complex_analysis_result",
                "timestamp": time.time(),
            }

        start_time = time.perf_counter()
        result1 = await complex_analysis("TSLA", "1H")
        first_time = (time.perf_counter() - start_time) * 1000

        start_time = time.perf_counter()
        result2 = await complex_analysis("TSLA", "1H")  # キャッシュから取得
        cached_time = (time.perf_counter() - start_time) * 1000

        print(f"初回分析: {first_time:.1f}ms")
        print(f"キャッシュ取得: {cached_time:.1f}ms")
        print(f"高速化率: {first_time / cached_time:.1f}x")
        print(f"結果一致: {result1['result'] == result2['result']}")

        # 一貫性監視開始（デモ）
        print("\n4. 一貫性監視デモ")
        await orchestrator.start_consistency_monitoring()
        await asyncio.sleep(2)  # 短時間の監視
        await orchestrator.stop_consistency_monitoring()

        print("\n=== マイクロサービス対応キャッシュオーケストレーターテスト完了 ===")

    # 実行
    asyncio.run(main())
