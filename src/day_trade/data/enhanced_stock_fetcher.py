#!/usr/bin/env python3
"""
Enhanced Stock Fetcher - 高度キャッシング統合版
Issue #377: 高度なキャッシング戦略の導入

永続キャッシュ、分散キャッシュ、スマート無効化戦略を統合したstock_fetcher
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# 基本モジュール
try:
    from ..utils.exceptions import APIError, DataError, NetworkError, ValidationError
    from ..utils.logging_config import get_context_logger, log_performance_metric
    from .stock_fetcher import StockFetcher as BaseStockFetcher
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)

    def log_performance_metric(*args, **kwargs):
        pass

    # フォールバック例外クラス
    class APIError(Exception):
        pass

    class NetworkError(Exception):
        pass

    class DataError(Exception):
        pass

    class ValidationError(Exception):
        pass

    # フォールバック基底クラス
    class BaseStockFetcher:
        def __init__(self, **kwargs):
            self.logger = logging.getLogger(__name__)


# 高度キャッシュシステム
try:
    from ..cache.distributed_cache_system import (
        distributed_cache,
        get_distributed_cache,
    )
    from ..cache.persistent_cache_system import get_persistent_cache, persistent_cache
    from ..cache.smart_invalidation_strategies import (
        CacheEntry,
        DependencyBasedStrategy,
        EventDrivenStrategy,
        InvalidationTrigger,
        LFUStrategy,
        LRUStrategy,
        SmartInvalidationManager,
        TTLStrategy,
        get_invalidation_manager,
    )

    ADVANCED_CACHE_AVAILABLE = True
except ImportError:
    # フォールバックキャッシュ実装

    def get_persistent_cache(**kwargs):
        return None

    def get_distributed_cache(**kwargs):
        return None

    def get_invalidation_manager(**kwargs):
        return None

    def persistent_cache(**kwargs):
        def decorator(func):
            return func

        return decorator

    def distributed_cache(**kwargs):
        def decorator(func):
            return func

        return decorator

    ADVANCED_CACHE_AVAILABLE = False

logger = get_context_logger(__name__)


class CacheConfig:
    """キャッシュ設定管理"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # デフォルト設定
        self.defaults = {
            # 永続キャッシュ設定
            "persistent_cache_enabled": True,
            "persistent_storage_type": "sqlite",
            "persistent_storage_path": "data/stock_cache",
            "persistent_compression": True,
            "persistent_auto_cleanup": True,
            # 分散キャッシュ設定
            "distributed_cache_enabled": False,  # デフォルト無効
            "distributed_backend_type": "memory",
            "distributed_backend_config": {},
            "distributed_serialization": "pickle",
            # スマート無効化設定
            "smart_invalidation_enabled": True,
            "max_cache_size": 10000,
            "memory_pressure_threshold": 0.8,
            # TTL設定
            "price_ttl_seconds": 30,
            "historical_ttl_seconds": 300,
            "company_info_ttl_seconds": 3600,
            "bulk_operations_ttl_seconds": 60,
            # キャッシュレイヤー設定
            "enable_multi_layer_cache": True,
            "l1_memory_size": 1000,
            "l2_persistent_enabled": True,
            "l3_distributed_enabled": False,
        }

    def get(self, key: str, default=None):
        """設定値取得"""
        return self.config.get(key, self.defaults.get(key, default))

    def set(self, key: str, value: Any):
        """設定値更新"""
        self.config[key] = value


class MultiLayerCacheManager:
    """マルチレイヤーキャッシュ管理"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.stats = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "l3_hits": 0,
            "l3_misses": 0,
            "total_requests": 0,
        }
        import threading

        self._lock = threading.RLock()

        # L1: メモリキャッシュ（インスタンス内）
        self.l1_cache = {}
        self.l1_max_size = config.get("l1_memory_size")

        # L2: 永続キャッシュ
        self.l2_cache = None
        if config.get("l2_persistent_enabled") and ADVANCED_CACHE_AVAILABLE:
            try:
                self.l2_cache = get_persistent_cache(
                    storage_type=config.get("persistent_storage_type"),
                    storage_path=config.get("persistent_storage_path"),
                )
            except Exception as e:
                logger.warning(f"L2永続キャッシュ初期化失敗: {e}")

        # L3: 分散キャッシュ
        self.l3_cache = None
        if config.get("l3_distributed_enabled") and ADVANCED_CACHE_AVAILABLE:
            try:
                self.l3_cache = get_distributed_cache(
                    backend_type=config.get("distributed_backend_type"),
                    backend_config=config.get("distributed_backend_config"),
                )
            except Exception as e:
                logger.warning(f"L3分散キャッシュ初期化失敗: {e}")

        # スマート無効化マネージャー
        self.invalidation_manager = None
        if config.get("smart_invalidation_enabled") and ADVANCED_CACHE_AVAILABLE:
            try:
                strategies = [
                    TTLStrategy(),
                    LRUStrategy(max_idle_seconds=3600),
                    LFUStrategy(min_frequency_threshold=0.01),
                ]
                self.invalidation_manager = get_invalidation_manager(
                    strategies=strategies, max_cache_size=config.get("max_cache_size")
                )
            except Exception as e:
                logger.warning(f"スマート無効化マネージャー初期化失敗: {e}")

        logger.info(
            "マルチレイヤーキャッシュマネージャー初期化完了",
            extra={
                "l1_enabled": True,
                "l2_enabled": self.l2_cache is not None,
                "l3_enabled": self.l3_cache is not None,
                "smart_invalidation": self.invalidation_manager is not None,
            },
        )

    def get(self, key: str, default=None):
        """マルチレイヤーキャッシュ取得"""
        with self._lock:
            self.stats["total_requests"] += 1

            # L1: メモリキャッシュ
            if key in self.l1_cache:
                value, timestamp = self.l1_cache[key]
                self.stats["l1_hits"] += 1
                return value
            self.stats["l1_misses"] += 1

            # L2: 永続キャッシュ
            if self.l2_cache:
                try:
                    value = self.l2_cache.get(key)
                    if value is not None:
                        self.stats["l2_hits"] += 1
                        # L1にも保存
                        self._set_l1(key, value)
                        return value
                except Exception as e:
                    logger.debug(f"L2キャッシュ取得エラー: {e}")
                self.stats["l2_misses"] += 1

            # L3: 分散キャッシュ
            if self.l3_cache:
                try:
                    value = self.l3_cache.get(key)
                    if value is not None:
                        self.stats["l3_hits"] += 1
                        # 上位層にも保存
                        self._set_l1(key, value)
                        if self.l2_cache:
                            self.l2_cache.set(key, value)
                        return value
                except Exception as e:
                    logger.debug(f"L3キャッシュ取得エラー: {e}")
                self.stats["l3_misses"] += 1

            return default

    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """マルチレイヤーキャッシュ保存"""
        with self._lock:
            # L1: メモリキャッシュ
            self._set_l1(key, value)

            # L2: 永続キャッシュ
            if self.l2_cache:
                try:
                    self.l2_cache.set(key, value, ttl_seconds=ttl_seconds)
                except Exception as e:
                    logger.debug(f"L2キャッシュ設定エラー: {e}")

            # L3: 分散キャッシュ
            if self.l3_cache:
                try:
                    self.l3_cache.set(key, value, ttl_seconds=ttl_seconds)
                except Exception as e:
                    logger.debug(f"L3キャッシュ設定エラー: {e}")

            # スマート無効化マネージャーに追加
            if self.invalidation_manager:
                try:
                    entry = CacheEntry(
                        key=key,
                        value=value,
                        created_at=time.time(),
                        last_accessed=time.time(),
                        access_count=1,
                        size_bytes=len(str(value).encode("utf-8")),
                    )
                    self.invalidation_manager.add_entry(key, entry)
                except Exception as e:
                    logger.debug(f"無効化マネージャー追加エラー: {e}")

    def _set_l1(self, key: str, value: Any):
        """L1キャッシュ設定（LRU削除付き）"""
        if len(self.l1_cache) >= self.l1_max_size:
            # 最古のエントリを削除
            oldest_key = min(self.l1_cache.keys(), key=lambda k: self.l1_cache[k][1])
            del self.l1_cache[oldest_key]

        self.l1_cache[key] = (value, time.time())

    def invalidate(self, key: str):
        """特定キーの無効化"""
        with self._lock:
            # L1から削除
            if key in self.l1_cache:
                del self.l1_cache[key]

            # L2から削除
            if self.l2_cache:
                try:
                    self.l2_cache.delete(key)
                except Exception as e:
                    logger.debug(f"L2キャッシュ削除エラー: {e}")

            # L3から削除
            if self.l3_cache:
                try:
                    self.l3_cache.delete(key)
                except Exception as e:
                    logger.debug(f"L3キャッシュ削除エラー: {e}")

            # スマート無効化マネージャーから削除
            if self.invalidation_manager:
                try:
                    self.invalidation_manager.remove_entry(key)
                except Exception as e:
                    logger.debug(f"無効化マネージャー削除エラー: {e}")

    def clear_all(self):
        """全キャッシュクリア"""
        with self._lock:
            # L1クリア
            self.l1_cache.clear()

            # L2クリア
            if self.l2_cache:
                try:
                    self.l2_cache.clear()
                except Exception:
                    pass

            # L3クリア
            if self.l3_cache:
                try:
                    self.l3_cache.clear()
                except Exception:
                    pass

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計取得"""
        with self._lock:
            total_hits = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]
            total_misses = (
                self.stats["l1_misses"] + self.stats["l2_misses"] + self.stats["l3_misses"]
            )

            stats = {
                **self.stats,
                "l1_cache_size": len(self.l1_cache),
                "l1_hit_rate": self.stats["l1_hits"] / max(self.stats["total_requests"], 1),
                "overall_hit_rate": total_hits / max(total_hits + total_misses, 1),
                "cache_layers_active": sum(
                    [
                        1,  # L1 always active
                        1 if self.l2_cache else 0,
                        1 if self.l3_cache else 0,
                    ]
                ),
            }

            # 各レイヤーの統計も追加
            if self.l2_cache:
                try:
                    l2_stats = self.l2_cache.get_stats()
                    stats["l2_stats"] = l2_stats
                except Exception:
                    pass

            if self.l3_cache:
                try:
                    l3_stats = self.l3_cache.get_stats()
                    stats["l3_stats"] = l3_stats
                except Exception:
                    pass

            return stats


class EnhancedStockFetcher(BaseStockFetcher):
    """高度キャッシング戦略統合版Stock Fetcher"""

    def __init__(self, cache_config: Dict[str, Any] = None, **kwargs):
        """
        初期化

        Args:
            cache_config: キャッシュ設定辞書
            **kwargs: BaseStockFetcher用パラメータ
        """
        super().__init__(**kwargs)

        # キャッシュ設定初期化
        self.cache_config = CacheConfig(cache_config)

        # マルチレイヤーキャッシュマネージャー初期化
        self.cache_manager = MultiLayerCacheManager(self.cache_config)

        # パフォーマンス統計
        self.performance_stats = {
            "api_calls": 0,
            "cache_hits": 0,
            "total_requests": 0,
            "avg_response_time_ms": 0.0,
            "cache_hit_rate": 0.0,
        }

        logger.info(
            "Enhanced Stock Fetcher 初期化完了",
            extra={
                "advanced_cache_available": ADVANCED_CACHE_AVAILABLE,
                "multi_layer_enabled": self.cache_config.get("enable_multi_layer_cache"),
            },
        )

    def _generate_cache_key(self, method: str, *args, **kwargs) -> str:
        """キャッシュキー生成"""
        import hashlib

        key_parts = [method] + [str(arg) for arg in args]
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_parts.extend([f"{k}={v}" for k, v in sorted_kwargs])

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _cached_operation(
        self, operation_name: str, operation_func, ttl_seconds: int, *args, **kwargs
    ):
        """キャッシュ付き操作実行"""
        start_time = time.time()
        self.performance_stats["total_requests"] += 1

        # キャッシュキー生成
        cache_key = f"{operation_name}:{self._generate_cache_key(operation_name, *args, **kwargs)}"

        # キャッシュから取得試行
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            self.performance_stats["cache_hits"] += 1
            elapsed_ms = (time.time() - start_time) * 1000

            # 統計更新
            self._update_performance_stats(elapsed_ms)

            logger.debug(
                f"キャッシュヒット: {operation_name}",
                extra={"cache_key": cache_key, "elapsed_ms": elapsed_ms},
            )
            return cached_result

        # API実行
        try:
            self.performance_stats["api_calls"] += 1
            result = operation_func(*args, **kwargs)

            # 結果をキャッシュに保存
            if result is not None:
                self.cache_manager.set(cache_key, result, ttl_seconds)

            elapsed_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(elapsed_ms)

            log_performance_metric(f"stock_fetcher_{operation_name}", elapsed_ms, "ms")

            logger.debug(
                f"API実行完了: {operation_name}",
                extra={"elapsed_ms": elapsed_ms, "cache_key": cache_key},
            )

            return result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(
                f"API実行エラー: {operation_name}",
                extra={"error": str(e), "elapsed_ms": elapsed_ms},
            )
            raise

    def _update_performance_stats(self, elapsed_ms: float):
        """パフォーマンス統計更新"""
        # 移動平均でレスポンス時間更新
        alpha = 0.1
        self.performance_stats["avg_response_time_ms"] = (
            self.performance_stats["avg_response_time_ms"] * (1 - alpha) + elapsed_ms * alpha
        )

        # キャッシュヒット率更新
        total_requests = self.performance_stats["total_requests"]
        if total_requests > 0:
            self.performance_stats["cache_hit_rate"] = (
                self.performance_stats["cache_hits"] / total_requests
            )

    def get_current_price(self, code: str) -> Optional[Dict[str, float]]:
        """現在価格取得（キャッシュ強化版）"""
        ttl = self.cache_config.get("price_ttl_seconds")
        return self._cached_operation("get_current_price", super().get_current_price, ttl, code)

    def get_historical_data(
        self, code: str, period: str = "1mo", interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """ヒストリカルデータ取得（キャッシュ強化版）"""
        ttl = self.cache_config.get("historical_ttl_seconds")
        return self._cached_operation(
            "get_historical_data",
            super().get_historical_data,
            ttl,
            code,
            period,
            interval,
        )

    def get_historical_data_range(
        self,
        code: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """期間指定ヒストリカルデータ取得（キャッシュ強化版）"""
        ttl = self.cache_config.get("historical_ttl_seconds")
        return self._cached_operation(
            "get_historical_data_range",
            super().get_historical_data_range,
            ttl,
            code,
            start_date,
            end_date,
            interval,
        )

    def get_company_info(self, code: str) -> Optional[Dict[str, Any]]:
        """企業情報取得（キャッシュ強化版）"""
        ttl = self.cache_config.get("company_info_ttl_seconds")
        return self._cached_operation("get_company_info", super().get_company_info, ttl, code)

    def bulk_get_current_prices(
        self, codes: List[str], batch_size: int = 50, delay: float = 0.1
    ) -> Dict[str, Optional[Dict]]:
        """一括現在価格取得（キャッシュ強化版）"""
        if not codes:
            return {}

        results = {}
        uncached_codes = []

        # まずキャッシュから取得
        for code in codes:
            cache_key = f"get_current_price:{self._generate_cache_key('get_current_price', code)}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                results[code] = cached_result
                self.performance_stats["cache_hits"] += 1
            else:
                uncached_codes.append(code)
            self.performance_stats["total_requests"] += 1

        # キャッシュにないものをバルク取得
        if uncached_codes:
            ttl = self.cache_config.get("price_ttl_seconds")
            bulk_results = self._cached_operation(
                "bulk_get_current_prices",
                super().bulk_get_current_prices,
                ttl,
                uncached_codes,
                batch_size,
                delay,
            )

            # 個別にキャッシュ保存
            for code, result in bulk_results.items():
                if result is not None:
                    cache_key = (
                        f"get_current_price:{self._generate_cache_key('get_current_price', code)}"
                    )
                    self.cache_manager.set(cache_key, result, ttl)
                results[code] = result

        logger.info(
            f"一括価格取得完了: {len(results)}件, キャッシュヒット: {len(codes) - len(uncached_codes)}件"
        )
        return results

    def invalidate_cache(self, pattern: str = None):
        """キャッシュ無効化"""
        if pattern:
            # パターンマッチング無効化（簡易実装）
            if self.cache_manager.invalidation_manager:
                try:
                    self.cache_manager.invalidation_manager.trigger_event(
                        "pattern_invalidation", pattern=pattern
                    )
                except Exception as e:
                    logger.debug(f"パターン無効化エラー: {e}")
        else:
            # 全キャッシュクリア
            self.cache_manager.clear_all()

        logger.info(f"キャッシュ無効化完了: pattern={pattern}")

    def invalidate_symbol_cache(self, symbol: str):
        """特定銘柄のキャッシュ無効化"""
        patterns = [
            f"get_current_price:{self._generate_cache_key('get_current_price', symbol)}",
            f"get_historical_data:{symbol}",
            f"get_company_info:{self._generate_cache_key('get_company_info', symbol)}",
        ]

        for pattern in patterns:
            self.cache_manager.invalidate(pattern)

        logger.info(f"銘柄キャッシュ無効化完了: {symbol}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計取得"""
        cache_stats = self.cache_manager.get_stats()

        return {
            "performance_stats": self.performance_stats,
            "cache_stats": cache_stats,
            "cache_config": {
                "multi_layer_enabled": self.cache_config.get("enable_multi_layer_cache"),
                "persistent_enabled": self.cache_config.get("persistent_cache_enabled"),
                "distributed_enabled": self.cache_config.get("distributed_cache_enabled"),
                "smart_invalidation_enabled": self.cache_config.get("smart_invalidation_enabled"),
            },
            "advanced_cache_available": ADVANCED_CACHE_AVAILABLE,
        }

    def optimize_cache_settings(self) -> Dict[str, Any]:
        """キャッシュ設定最適化"""
        stats = self.get_cache_stats()
        recommendations = {}

        # ヒット率に基づく推奨
        hit_rate = stats["performance_stats"]["cache_hit_rate"]
        if hit_rate < 0.3:
            recommendations["increase_ttl"] = "TTLを延長してヒット率を改善"
        elif hit_rate > 0.9:
            recommendations["decrease_ttl"] = "TTLを短縮してメモリ効率を改善"

        # API呼び出し頻度に基づく推奨
        api_ratio = stats["performance_stats"]["api_calls"] / max(
            stats["performance_stats"]["total_requests"], 1
        )
        if api_ratio > 0.8:
            recommendations["enable_persistent"] = "永続キャッシュを有効化"
            if not self.cache_config.get("l3_distributed_enabled"):
                recommendations["consider_distributed"] = "分散キャッシュの検討"

        return {
            "current_stats": stats,
            "recommendations": recommendations,
            "optimization_score": hit_rate * 100,  # 0-100のスコア
        }

    def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        health = {
            "status": "healthy",
            "cache_manager_active": self.cache_manager is not None,
            "l1_cache_size": len(self.cache_manager.l1_cache),
            "advanced_cache_available": ADVANCED_CACHE_AVAILABLE,
            "issues": [],
        }

        # L2キャッシュ確認
        if self.cache_manager.l2_cache:
            try:
                l2_stats = self.cache_manager.l2_cache.get_stats()
                health["l2_persistent_healthy"] = True
                health["l2_entries"] = l2_stats.get("total_operations", 0)
            except Exception as e:
                health["l2_persistent_healthy"] = False
                health["issues"].append(f"L2永続キャッシュエラー: {e}")

        # L3キャッシュ確認
        if self.cache_manager.l3_cache:
            try:
                l3_stats = self.cache_manager.l3_cache.get_stats()
                health["l3_distributed_healthy"] = True
                health["l3_connected"] = l3_stats.get("primary_connected", False)
            except Exception as e:
                health["l3_distributed_healthy"] = False
                health["issues"].append(f"L3分散キャッシュエラー: {e}")

        if health["issues"]:
            health["status"] = "warning"

        return health


# 工場関数
def create_enhanced_stock_fetcher(
    cache_config: Dict[str, Any] = None, **fetcher_kwargs
) -> EnhancedStockFetcher:
    """
    Enhanced Stock Fetcher インスタンス作成

    Args:
        cache_config: キャッシュ設定
        **fetcher_kwargs: StockFetcher用パラメータ

    Returns:
        設定済みEnhancedStockFetcher
    """
    default_cache_config = {
        "persistent_cache_enabled": True,
        "persistent_storage_type": "sqlite",
        "persistent_compression": True,
        "smart_invalidation_enabled": True,
        "enable_multi_layer_cache": True,
        "price_ttl_seconds": 30,
        "historical_ttl_seconds": 300,
        "company_info_ttl_seconds": 3600,
    }

    if cache_config:
        default_cache_config.update(cache_config)

    return EnhancedStockFetcher(cache_config=default_cache_config, **fetcher_kwargs)


if __name__ == "__main__":
    # テスト実行
    print("=== Enhanced Stock Fetcher テスト ===")

    # デフォルト設定でインスタンス作成
    fetcher = create_enhanced_stock_fetcher()

    print("\n1. ヘルスチェック")
    health = fetcher.health_check()
    print(f"ステータス: {health['status']}")
    print(f"高度キャッシュ利用可能: {health['advanced_cache_available']}")

    print("\n2. 現在価格取得テスト")
    try:
        # 初回実行（API呼び出し）
        start_time = time.time()
        price1 = fetcher.get_current_price("7203")
        first_time = time.time() - start_time

        # 2回目実行（キャッシュヒット）
        start_time = time.time()
        price2 = fetcher.get_current_price("7203")
        cached_time = time.time() - start_time

        print(f"初回実行時間: {first_time:.3f}秒")
        print(f"キャッシュ実行時間: {cached_time:.3f}秒")
        print(f"高速化率: {first_time/max(cached_time, 0.001):.1f}x")

    except Exception as e:
        print(f"価格取得テストエラー: {e}")

    print("\n3. キャッシュ統計")
    stats = fetcher.get_cache_stats()
    perf_stats = stats["performance_stats"]
    print(f"総リクエスト: {perf_stats['total_requests']}")
    print(f"キャッシュヒット: {perf_stats['cache_hits']}")
    print(f"ヒット率: {perf_stats['cache_hit_rate']:.2%}")
    print(f"平均レスポンス時間: {perf_stats['avg_response_time_ms']:.1f}ms")

    print("\n4. 最適化推奨")
    optimization = fetcher.optimize_cache_settings()
    print(f"最適化スコア: {optimization['optimization_score']:.1f}/100")
    if optimization["recommendations"]:
        for key, recommendation in optimization["recommendations"].items():
            print(f"- {key}: {recommendation}")
    else:
        print("- 現在の設定は最適です")

    print("\n=== Enhanced Stock Fetcher テスト完了 ===")
