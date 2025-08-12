#!/usr/bin/env python3
"""
HFT超低レイテンシ最適化エンジン
Issue #434: 本番環境パフォーマンス最終最適化 - HFT <50μs目標達成

CPUキャッシュ最適化、メモリ事前割り当て、SIMD並列処理によるμ秒レベル最適化
"""

import ctypes
import mmap
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import numba
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)

# システム最適化定数
CACHE_LINE_SIZE = 64  # CPUキャッシュライン
HUGE_PAGE_SIZE = 2 * 1024 * 1024  # 2MB Huge Pages
L1_CACHE_SIZE = 32 * 1024  # L1キャッシュサイズ
L2_CACHE_SIZE = 256 * 1024  # L2キャッシュサイズ


@dataclass
class HFTConfig:
    """HFT最適化設定"""

    # レイテンシ目標
    target_latency_us: float = 50.0
    strict_latency_us: float = 10.0  # 厳格モード

    # メモリ最適化
    preallocated_memory_mb: int = 100
    use_huge_pages: bool = True
    numa_aware: bool = True

    # CPU最適化
    cpu_affinity: List[int] = None
    enable_simd: bool = True
    vectorization_width: int = 8

    # 並列処理
    max_threads: int = 2
    thread_pinning: bool = True

    # プロファイリング
    enable_profiling: bool = True
    profile_every_n_ops: int = 1000


class MemoryPool:
    """超高速メモリプール - 事前割り当て済みメモリ管理"""

    def __init__(self, pool_size_mb: int = 100, use_huge_pages: bool = True):
        self.pool_size = pool_size_mb * 1024 * 1024
        self.use_huge_pages = use_huge_pages

        # メモリプール初期化
        self._init_memory_pool()

        # メモリ管理構造
        self.free_blocks = []
        self.used_blocks = {}
        self.allocation_lock = threading.RLock()

        # 統計情報
        self.stats = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "current_usage_bytes": 0,
            "peak_usage_bytes": 0,
            "allocation_time_ns": [],
        }

        logger.info(
            f"メモリプール初期化完了: {pool_size_mb}MB (huge_pages={use_huge_pages})"
        )

    def _init_memory_pool(self):
        """メモリプール初期化"""
        try:
            # Windows対応のmmap実装
            if os.name == "nt":  # Windows
                try:
                    self.memory_pool = mmap.mmap(-1, self.pool_size)
                    logger.info("Windows メモリマップ初期化完了")
                except Exception as e:
                    logger.warning(f"Windows mmap失敗: {e}")
                    self.memory_pool = bytearray(self.pool_size)
            else:
                # Linux/Unix
                if self.use_huge_pages and os.path.exists("/proc/sys/vm/nr_hugepages"):
                    # Huge Pages使用（Linux専用）
                    self.memory_pool = mmap.mmap(
                        -1,
                        self.pool_size,
                        mmap.MAP_PRIVATE
                        | mmap.MAP_ANONYMOUS
                        | getattr(mmap, "MAP_HUGETLB", 0),
                        mmap.PROT_READ | mmap.PROT_WRITE,
                    )
                    logger.info("Huge Pages メモリプール初期化完了")
                else:
                    # 通常メモリ
                    self.memory_pool = mmap.mmap(
                        -1,
                        self.pool_size,
                        mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                        mmap.PROT_READ | mmap.PROT_WRITE,
                    )
                    logger.info("通常メモリプール初期化完了")

                # メモリをゼロクリア（初期化時のみ）
                if hasattr(self.memory_pool, "seek"):
                    self.memory_pool.seek(0)
                    self.memory_pool.write(b"\x00" * self.pool_size)

        except Exception as e:
            logger.error(f"メモリプール初期化失敗: {e}")
            # フォールバック - 通常のbytearray
            self.memory_pool = bytearray(self.pool_size)
            logger.info("bytearray フォールバック初期化完了")

    def allocate(self, size: int, alignment: int = CACHE_LINE_SIZE) -> Optional[int]:
        """超高速メモリ割り当て"""
        start_time = time.perf_counter_ns()

        # アライメント調整
        aligned_size = (size + alignment - 1) & ~(alignment - 1)

        with self.allocation_lock:
            # フリーブロック検索
            for i, (offset, block_size) in enumerate(self.free_blocks):
                if block_size >= aligned_size:
                    # ブロック使用
                    if block_size > aligned_size:
                        # 残りを新しいフリーブロックとして追加
                        self.free_blocks[i] = (
                            offset + aligned_size,
                            block_size - aligned_size,
                        )
                    else:
                        # ブロック完全使用
                        del self.free_blocks[i]

                    # 使用ブロック記録
                    self.used_blocks[offset] = aligned_size

                    # 統計更新
                    self.stats["total_allocations"] += 1
                    self.stats["current_usage_bytes"] += aligned_size
                    self.stats["peak_usage_bytes"] = max(
                        self.stats["peak_usage_bytes"],
                        self.stats["current_usage_bytes"],
                    )

                    allocation_time = time.perf_counter_ns() - start_time
                    self.stats["allocation_time_ns"].append(allocation_time)

                    return offset

            # 新規メモリ領域を割り当て
            if hasattr(self, "_next_offset"):
                if self._next_offset + aligned_size <= self.pool_size:
                    offset = self._next_offset
                    self._next_offset += aligned_size
                    self.used_blocks[offset] = aligned_size

                    # 統計更新
                    self.stats["total_allocations"] += 1
                    self.stats["current_usage_bytes"] += aligned_size
                    self.stats["peak_usage_bytes"] = max(
                        self.stats["peak_usage_bytes"],
                        self.stats["current_usage_bytes"],
                    )

                    return offset
            else:
                self._next_offset = aligned_size
                self.used_blocks[0] = aligned_size

                self.stats["total_allocations"] += 1
                self.stats["current_usage_bytes"] += aligned_size
                self.stats["peak_usage_bytes"] = aligned_size

                return 0

        logger.error(f"メモリ割り当て失敗: {size}bytes")
        return None

    def deallocate(self, offset: int):
        """超高速メモリ解放"""
        with self.allocation_lock:
            if offset in self.used_blocks:
                size = self.used_blocks[offset]
                del self.used_blocks[offset]

                # フリーブロックに追加
                self.free_blocks.append((offset, size))
                self.free_blocks.sort(key=lambda x: x[0])  # オフセット順ソート

                # 統計更新
                self.stats["total_deallocations"] += 1
                self.stats["current_usage_bytes"] -= size

                # 連続フリーブロックをマージ
                self._merge_free_blocks()

    def _merge_free_blocks(self):
        """連続フリーブロックのマージ"""
        if len(self.free_blocks) < 2:
            return

        merged = []
        current_offset, current_size = self.free_blocks[0]

        for offset, size in self.free_blocks[1:]:
            if current_offset + current_size == offset:
                # 連続ブロック - マージ
                current_size += size
            else:
                # 非連続 - 現在ブロック追加
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size

        # 最後のブロック追加
        merged.append((current_offset, current_size))
        self.free_blocks = merged

    def get_memory_view(self, offset: int, size: int) -> memoryview:
        """メモリビュー取得（ゼロコピー）"""
        if isinstance(self.memory_pool, mmap.mmap):
            return memoryview(self.memory_pool)[offset : offset + size]
        else:
            return memoryview(self.memory_pool)[offset : offset + size]

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        with self.allocation_lock:
            avg_allocation_time = (
                sum(self.stats["allocation_time_ns"])
                / len(self.stats["allocation_time_ns"])
                if self.stats["allocation_time_ns"]
                else 0
            )

            return {
                **self.stats,
                "avg_allocation_time_ns": avg_allocation_time,
                "pool_utilization": self.stats["current_usage_bytes"] / self.pool_size,
                "free_blocks_count": len(self.free_blocks),
                "used_blocks_count": len(self.used_blocks),
            }


if NUMBA_AVAILABLE:

    @jit(nopython=True, nogil=True, cache=True)
    def vectorized_feature_calculation(
        prices: np.ndarray, volumes: np.ndarray
    ) -> np.ndarray:
        """SIMD最適化特徴量計算"""
        n = len(prices)
        features = np.empty((n, 8), dtype=np.float64)

        # 並列ベクトル演算
        for i in prange(n):
            if i >= 20:  # 十分な履歴データがある場合
                # 1. 移動平均（高速化版）
                ma_5 = 0.0
                ma_20 = 0.0
                for j in range(5):
                    ma_5 += prices[i - j]
                for j in range(20):
                    ma_20 += prices[i - j]
                ma_5 /= 5.0
                ma_20 /= 20.0

                # 2. RSI（簡略化版）
                gain = 0.0
                loss = 0.0
                for j in range(1, 15):
                    diff = prices[i - j + 1] - prices[i - j]
                    if diff > 0:
                        gain += diff
                    else:
                        loss -= diff

                rsi = 100.0 - (100.0 / (1.0 + (gain / max(loss, 1e-8))))

                # 3. ボリューム率
                vol_ma = 0.0
                for j in range(5):
                    vol_ma += volumes[i - j]
                vol_ma /= 5.0
                vol_ratio = volumes[i] / max(vol_ma, 1e-8)

                # 4. 価格変化率
                price_change = (prices[i] - prices[i - 1]) / max(prices[i - 1], 1e-8)

                # 5. ボラティリティ（簡略版）
                volatility = 0.0
                for j in range(1, 6):
                    diff = prices[i - j + 1] - prices[i - j]
                    volatility += diff * diff
                volatility = (volatility / 5.0) ** 0.5

                # 特徴量設定
                features[i, 0] = ma_5
                features[i, 1] = ma_20
                features[i, 2] = rsi
                features[i, 3] = vol_ratio
                features[i, 4] = price_change
                features[i, 5] = volatility
                features[i, 6] = prices[i]
                features[i, 7] = volumes[i]
            else:
                # 不十分なデータの場合はゼロ埋め
                for j in range(8):
                    features[i, j] = 0.0

        return features

    @jit(nopython=True, nogil=True, cache=True)
    def ultra_fast_price_prediction(features: np.ndarray, weights: np.ndarray) -> float:
        """超高速価格予測（線形モデル）"""
        prediction = 0.0
        for i in range(min(len(features), len(weights))):
            prediction += features[i] * weights[i]
        return prediction

else:

    def vectorized_feature_calculation(
        prices: np.ndarray, volumes: np.ndarray
    ) -> np.ndarray:
        """フォールバック特徴量計算"""
        logger.warning("Numba未利用のためフォールバック実行")
        n = len(prices)
        features = np.zeros((n, 8))

        for i in range(20, n):
            features[i, 0] = np.mean(prices[i - 5 : i])  # MA5
            features[i, 1] = np.mean(prices[i - 20 : i])  # MA20
            features[i, 6] = prices[i]
            features[i, 7] = volumes[i]

        return features

    def ultra_fast_price_prediction(features: np.ndarray, weights: np.ndarray) -> float:
        """フォールバック価格予測"""
        return np.dot(features[: len(weights)], weights)


class HFTOptimizer:
    """HFT超低レイテンシ最適化エンジン"""

    def __init__(self, config: HFTConfig = None):
        self.config = config or HFTConfig()

        # システム最適化
        self._setup_system_optimization()

        # メモリプール初期化
        self.memory_pool = MemoryPool(
            self.config.preallocated_memory_mb, self.config.use_huge_pages
        )

        # 予測モデル重み（事前学習済み想定）
        self.model_weights = np.random.normal(0, 0.1, 8).astype(np.float64)

        # パフォーマンス統計
        self.performance_stats = {
            "total_predictions": 0,
            "avg_latency_us": 0.0,
            "latency_histogram": np.zeros(100),  # 0-99μs
            "under_target_percentage": 0.0,
            "cache_hit_rate": 0.0,
        }
        self._stats_lock = threading.RLock()

        logger.info(
            f"HFT最適化エンジン初期化完了 (目標: {self.config.target_latency_us}μs)"
        )

    def _setup_system_optimization(self):
        """システムレベル最適化"""
        try:
            # CPU親和性設定（Linux専用）
            if self.config.cpu_affinity and hasattr(os, "sched_setaffinity"):
                try:
                    os.sched_setaffinity(0, set(self.config.cpu_affinity))
                    logger.info(f"CPU親和性設定: {self.config.cpu_affinity}")
                except OSError as e:
                    logger.warning(f"CPU親和性設定失敗: {e}")

            # プロセス優先度設定（可能な場合）
            if hasattr(os, "setpriority"):
                try:
                    os.setpriority(os.PRIO_PROCESS, 0, -10)  # 高優先度
                    logger.info("プロセス優先度設定完了")
                except (PermissionError, OSError) as e:
                    logger.warning(f"プロセス優先度設定失敗: {e}")

            # Windows用プロセス優先度設定
            if os.name == "nt":
                try:
                    import psutil

                    process = psutil.Process()
                    process.nice(psutil.HIGH_PRIORITY_CLASS)
                    logger.info("Windows プロセス優先度設定完了")
                except Exception as e:
                    logger.warning(f"Windows プロセス優先度設定失敗: {e}")

            # スレッドピニング準備
            if self.config.thread_pinning:
                self._setup_thread_pinning()

        except Exception as e:
            logger.error(f"システム最適化エラー: {e}")

    def _setup_thread_pinning(self):
        """スレッドピニング設定"""
        # 実装は環境依存のため、基本的な準備のみ
        self.thread_affinity_map = {}
        if (
            self.config.cpu_affinity
            and len(self.config.cpu_affinity) >= self.config.max_threads
        ):
            for i in range(self.config.max_threads):
                self.thread_affinity_map[i] = self.config.cpu_affinity[i]

    def predict_ultra_fast(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> Dict[str, Any]:
        """超高速予測実行（メイン処理）"""
        start_time = time.perf_counter_ns()

        try:
            # 1. 特徴量計算（SIMD最適化）
            feature_start = time.perf_counter_ns()
            features = vectorized_feature_calculation(prices, volumes)
            feature_time = time.perf_counter_ns() - feature_start

            # 2. 最新の特徴量で予測
            prediction_start = time.perf_counter_ns()
            latest_features = features[-1] if len(features) > 0 else np.zeros(8)
            prediction = ultra_fast_price_prediction(
                latest_features, self.model_weights
            )
            prediction_time = time.perf_counter_ns() - prediction_start

            # 3. 総実行時間計算
            total_time_ns = time.perf_counter_ns() - start_time
            total_time_us = total_time_ns / 1000.0

            # 4. 統計更新
            self._update_performance_stats(total_time_us)

            # 5. 結果返却
            result = {
                "prediction": float(prediction),
                "latency_us": total_time_us,
                "feature_time_ns": feature_time,
                "prediction_time_ns": prediction_time,
                "under_target": total_time_us < self.config.target_latency_us,
                "features_shape": features.shape,
                "timestamp_ns": time.perf_counter_ns(),
            }

            # レイテンシ警告
            if total_time_us > self.config.target_latency_us:
                logger.warning(
                    f"レイテンシ目標超過: {total_time_us:.2f}μs > {self.config.target_latency_us}μs"
                )

            return result

        except Exception as e:
            logger.error(f"超高速予測エラー: {e}")
            return {
                "prediction": 0.0,
                "latency_us": 999999.0,
                "error": str(e),
                "under_target": False,
            }

    def batch_predict_optimized(
        self, symbols_data: Dict[str, Dict[str, np.ndarray]], batch_size: int = 10
    ) -> Dict[str, Any]:
        """最適化バッチ予測"""
        batch_start_time = time.perf_counter_ns()

        results = {}
        latencies = []

        # バッチ処理
        symbols = list(symbols_data.keys())
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i : i + batch_size]

            batch_results = {}
            batch_latency_start = time.perf_counter_ns()

            for symbol in batch_symbols:
                data = symbols_data[symbol]
                prices = data.get("prices", np.array([]))
                volumes = data.get("volumes", np.array([]))

                if len(prices) > 0 and len(volumes) > 0:
                    prediction_result = self.predict_ultra_fast(prices, volumes)
                    batch_results[symbol] = prediction_result
                    latencies.append(prediction_result.get("latency_us", 0))

            batch_latency = (time.perf_counter_ns() - batch_latency_start) / 1000.0

            results.update(batch_results)

            logger.debug(
                f"バッチ処理完了: {len(batch_results)}銘柄, {batch_latency:.2f}μs"
            )

        total_batch_time = (time.perf_counter_ns() - batch_start_time) / 1000.0

        return {
            "results": results,
            "batch_stats": {
                "total_symbols": len(results),
                "total_batch_time_us": total_batch_time,
                "avg_latency_per_symbol_us": np.mean(latencies) if latencies else 0,
                "max_latency_us": np.max(latencies) if latencies else 0,
                "under_target_rate": (
                    np.mean(
                        [
                            1 if lat < self.config.target_latency_us else 0
                            for lat in latencies
                        ]
                    )
                    if latencies
                    else 0
                ),
            },
        }

    def _update_performance_stats(self, latency_us: float):
        """パフォーマンス統計更新"""
        with self._stats_lock:
            self.performance_stats["total_predictions"] += 1

            # 移動平均レイテンシ更新
            total_preds = self.performance_stats["total_predictions"]
            current_avg = self.performance_stats["avg_latency_us"]
            self.performance_stats["avg_latency_us"] = (
                current_avg * (total_preds - 1) + latency_us
            ) / total_preds

            # ヒストグラム更新
            histogram_index = min(int(latency_us), 99)
            self.performance_stats["latency_histogram"][histogram_index] += 1

            # 目標達成率更新
            under_target_count = np.sum(
                self.performance_stats["latency_histogram"][
                    : int(self.config.target_latency_us)
                ]
            )
            self.performance_stats["under_target_percentage"] = (
                under_target_count / total_preds * 100.0
            )

            # プロファイリング
            if (
                self.config.enable_profiling
                and total_preds % self.config.profile_every_n_ops == 0
            ):
                self._log_performance_profile()

    def _log_performance_profile(self):
        """パフォーマンスプロファイル出力"""
        with self._stats_lock:
            stats = self.performance_stats.copy()
            memory_stats = self.memory_pool.get_stats()

            logger.info("=== HFT パフォーマンスプロファイル ===")
            logger.info(f"総予測数: {stats['total_predictions']}")
            logger.info(f"平均レイテンシ: {stats['avg_latency_us']:.2f}μs")
            logger.info(f"目標達成率: {stats['under_target_percentage']:.1f}%")
            logger.info(f"メモリ使用率: {memory_stats['pool_utilization']:.1f}%")
            logger.info(
                f"メモリ平均割り当て時間: {memory_stats['avg_allocation_time_ns']:.0f}ns"
            )

    def get_optimization_report(self) -> Dict[str, Any]:
        """最適化レポート生成"""
        with self._stats_lock:
            perf_stats = self.performance_stats.copy()

        memory_stats = self.memory_pool.get_stats()

        # レイテンシ分析
        histogram = perf_stats["latency_histogram"]
        percentiles = {
            "p50": self._calculate_percentile(histogram, 0.5),
            "p90": self._calculate_percentile(histogram, 0.9),
            "p95": self._calculate_percentile(histogram, 0.95),
            "p99": self._calculate_percentile(histogram, 0.99),
            "p99.9": self._calculate_percentile(histogram, 0.999),
        }

        return {
            "config": {
                "target_latency_us": self.config.target_latency_us,
                "strict_latency_us": self.config.strict_latency_us,
                "memory_pool_mb": self.config.preallocated_memory_mb,
                "cpu_affinity": self.config.cpu_affinity,
                "simd_enabled": self.config.enable_simd,
            },
            "performance": {
                **perf_stats,
                "percentiles_us": percentiles,
            },
            "memory": memory_stats,
            "optimization_score": min(
                100.0,
                perf_stats["under_target_percentage"]
                + (100 - memory_stats["pool_utilization"] * 100) * 0.1,
            ),
        }

    def _calculate_percentile(self, histogram: np.ndarray, percentile: float) -> float:
        """ヒストグラムからパーセンタイル計算"""
        total_count = np.sum(histogram)
        if total_count == 0:
            return 0.0

        target_count = total_count * percentile
        cumulative_count = 0

        for i, count in enumerate(histogram):
            cumulative_count += count
            if cumulative_count >= target_count:
                return float(i)

        return float(len(histogram) - 1)

    def cleanup(self):
        """リソースクリーンアップ"""
        try:
            # メモリプール解放
            if hasattr(self.memory_pool, "memory_pool"):
                if hasattr(self.memory_pool.memory_pool, "close"):
                    self.memory_pool.memory_pool.close()

            logger.info("HFT最適化エンジンクリーンアップ完了")
        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")


# グローバルインスタンス
_global_hft_optimizer: Optional[HFTOptimizer] = None
_optimizer_lock = threading.Lock()


def get_hft_optimizer(config: HFTConfig = None) -> HFTOptimizer:
    """グローバルHFT最適化エンジン取得"""
    global _global_hft_optimizer

    if _global_hft_optimizer is None:
        with _optimizer_lock:
            if _global_hft_optimizer is None:
                _global_hft_optimizer = HFTOptimizer(config)

    return _global_hft_optimizer


def hft_optimized(target_latency_us: float = 50.0):
    """HFT最適化デコレータ"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            optimizer = get_hft_optimizer()

            start_time = time.perf_counter_ns()
            result = func(*args, **kwargs)
            execution_time_us = (time.perf_counter_ns() - start_time) / 1000.0

            if execution_time_us > target_latency_us:
                logger.warning(
                    f"関数 {func.__name__} レイテンシ超過: {execution_time_us:.2f}μs"
                )

            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    # HFT最適化デモ
    print("=== HFT超低レイテンシ最適化エンジン ===")

    # 設定
    config = HFTConfig(
        target_latency_us=25.0,
        preallocated_memory_mb=50,
        use_huge_pages=False,  # デモ用
        cpu_affinity=[0, 1],
        enable_simd=True,
    )

    optimizer = HFTOptimizer(config)

    # テストデータ生成
    np.random.seed(42)
    test_prices = np.random.normal(100, 5, 1000).astype(np.float64)
    test_volumes = np.random.normal(10000, 1000, 1000).astype(np.float64)

    print("\n1. 単一予測テスト")
    for i in range(5):
        result = optimizer.predict_ultra_fast(test_prices, test_volumes)
        print(
            f"予測 {i+1}: {result['prediction']:.4f}, "
            f"レイテンシ: {result['latency_us']:.2f}μs, "
            f"目標達成: {result['under_target']}"
        )

    print("\n2. バッチ予測テスト")
    symbols_data = {}
    for i in range(10):
        symbol = f"STOCK_{i:03d}"
        symbols_data[symbol] = {
            "prices": test_prices + np.random.normal(0, 1, len(test_prices)),
            "volumes": test_volumes + np.random.normal(0, 100, len(test_volumes)),
        }

    batch_result = optimizer.batch_predict_optimized(symbols_data, batch_size=5)
    batch_stats = batch_result["batch_stats"]

    print(f"バッチ処理: {batch_stats['total_symbols']}銘柄")
    print(f"総処理時間: {batch_stats['total_batch_time_us']:.2f}μs")
    print(f"平均レイテンシ: {batch_stats['avg_latency_per_symbol_us']:.2f}μs")
    print(f"目標達成率: {batch_stats['under_target_rate']:.1%}")

    print("\n3. 最適化レポート")
    report = optimizer.get_optimization_report()
    print(f"最適化スコア: {report['optimization_score']:.1f}/100")
    print(f"平均レイテンシ: {report['performance']['avg_latency_us']:.2f}μs")
    print(f"目標達成率: {report['performance']['under_target_percentage']:.1f}%")
    print(f"99%レイテンシ: {report['performance']['percentiles_us']['p99']:.1f}μs")
    print(f"メモリ使用率: {report['memory']['pool_utilization']:.1%}")

    optimizer.cleanup()
    print("\n=== HFT最適化デモ完了 ===")
