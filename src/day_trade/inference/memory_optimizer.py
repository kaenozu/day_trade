#!/usr/bin/env python3
"""
メモリ最適化システム
Memory Optimization System for ML Inference

Issue #761: MLモデル推論パイプラインの高速化と最適化 Phase 2
"""

import gc
import psutil
import time
import logging
import threading
import weakref
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import mmap
import pickle

# GPU メモリ管理
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# ログ設定
logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """メモリ最適化設定"""
    # モデルプール設定
    max_models_in_memory: int = 5
    model_cache_ttl_seconds: int = 3600  # 1時間
    enable_model_sharing: bool = True

    # 特徴量キャッシュ設定
    feature_cache_size_mb: int = 512
    feature_cache_ttl_seconds: int = 300  # 5分
    enable_feature_compression: bool = True

    # メモリ監視設定
    memory_warning_threshold: float = 0.8  # 80%
    memory_critical_threshold: float = 0.9  # 90%
    gc_trigger_threshold: float = 0.85  # 85%

    # ゼロコピー設定
    enable_zero_copy: bool = True
    mmap_threshold_mb: int = 100

    # GPU設定
    gpu_memory_fraction: float = 0.8
    enable_gpu_memory_growth: bool = True


@dataclass
class MemoryStats:
    """メモリ統計"""
    total_memory_gb: float = 0.0
    used_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    memory_usage_percent: float = 0.0
    models_in_memory: int = 0
    feature_cache_size_mb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gc_collections: int = 0
    memory_leaks_detected: int = 0


class LRUCache:
    """メモリ効率的なLRUキャッシュ"""

    def __init__(self, max_size: int, ttl_seconds: Optional[int] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.access_times = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """値取得"""
        with self.lock:
            if key not in self.cache:
                return None

            # TTL チェック
            if self.ttl_seconds:
                access_time = self.access_times.get(key, 0)
                if time.time() - access_time > self.ttl_seconds:
                    del self.cache[key]
                    del self.access_times[key]
                    return None

            # LRU更新
            value = self.cache.pop(key)
            self.cache[key] = value
            self.access_times[key] = time.time()

            return value

    def put(self, key: str, value: Any) -> None:
        """値設定"""
        with self.lock:
            # 既存キーの場合は更新
            if key in self.cache:
                self.cache.pop(key)

            # サイズ制限チェック
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                if oldest_key in self.access_times:
                    del self.access_times[oldest_key]

            # 新しい値を追加
            self.cache[key] = value
            self.access_times[key] = time.time()

    def clear(self) -> None:
        """キャッシュクリア"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

    def size(self) -> int:
        """キャッシュサイズ"""
        return len(self.cache)

    def memory_usage(self) -> float:
        """メモリ使用量推定（MB）"""
        try:
            import sys
            total_size = 0
            for value in self.cache.values():
                total_size += sys.getsizeof(value)
            return total_size / (1024 * 1024)  # MB
        except:
            return 0.0


class ModelPool:
    """モデルプール管理"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.reference_counts: Dict[str, int] = defaultdict(int)

    def load_model(self, model_id: str, model_loader: Callable[[], Any]) -> Any:
        """モデル読み込み"""
        with self.lock:
            # キャッシュ確認
            if model_id in self.models:
                self.access_times[model_id] = time.time()
                self.reference_counts[model_id] += 1
                logger.debug(f"Model cache hit: {model_id}")
                return self.models[model_id]

            # 容量チェックと古いモデル削除
            self._evict_old_models()

            # 新しいモデル読み込み
            logger.info(f"Loading model: {model_id}")
            start_time = time.time()

            try:
                model = model_loader()
                load_time = time.time() - start_time

                # メタデータ保存
                metadata = {
                    "load_time": load_time,
                    "created_at": time.time(),
                    "access_count": 1,
                    "memory_size_mb": self._estimate_model_size(model)
                }

                self.models[model_id] = model
                self.model_metadata[model_id] = metadata
                self.access_times[model_id] = time.time()
                self.reference_counts[model_id] = 1

                logger.info(f"Model loaded: {model_id} ({load_time:.2f}s, {metadata['memory_size_mb']:.1f}MB)")
                return model

            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                raise

    def release_model(self, model_id: str) -> None:
        """モデル参照カウント減少"""
        with self.lock:
            if model_id in self.reference_counts:
                self.reference_counts[model_id] = max(0, self.reference_counts[model_id] - 1)

    def _evict_old_models(self) -> None:
        """古いモデルの削除"""
        while len(self.models) >= self.config.max_models_in_memory:
            # 参照カウントが0で最も古いモデルを削除
            evict_candidates = [
                (model_id, access_time)
                for model_id, access_time in self.access_times.items()
                if self.reference_counts[model_id] == 0
            ]

            if not evict_candidates:
                # 全てのモデルが使用中の場合は強制削除
                evict_candidates = [(model_id, access_time) for model_id, access_time in self.access_times.items()]

            # 最も古いモデルを削除
            oldest_model = min(evict_candidates, key=lambda x: x[1])[0]
            self._remove_model(oldest_model)

    def _remove_model(self, model_id: str) -> None:
        """モデル削除"""
        if model_id in self.models:
            logger.info(f"Evicting model: {model_id}")
            del self.models[model_id]
            del self.model_metadata[model_id]
            del self.access_times[model_id]
            if model_id in self.reference_counts:
                del self.reference_counts[model_id]

    def _estimate_model_size(self, model: Any) -> float:
        """モデルサイズ推定（MB）"""
        try:
            import sys
            if hasattr(model, '__sizeof__'):
                return model.__sizeof__() / (1024 * 1024)
            else:
                return sys.getsizeof(model) / (1024 * 1024)
        except:
            return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """プール統計"""
        with self.lock:
            total_memory = sum(meta.get('memory_size_mb', 0) for meta in self.model_metadata.values())

            return {
                "models_count": len(self.models),
                "total_memory_mb": total_memory,
                "model_details": {
                    model_id: {
                        "memory_mb": meta.get('memory_size_mb', 0),
                        "access_count": meta.get('access_count', 0),
                        "reference_count": self.reference_counts.get(model_id, 0),
                        "age_seconds": time.time() - meta.get('created_at', 0)
                    }
                    for model_id, meta in self.model_metadata.items()
                }
            }

    def clear(self) -> None:
        """プールクリア"""
        with self.lock:
            logger.info("Clearing model pool")
            self.models.clear()
            self.model_metadata.clear()
            self.access_times.clear()
            self.reference_counts.clear()


class FeatureCache:
    """特徴量キャッシュ"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.cache = LRUCache(
            max_size=1000,  # アイテム数制限
            ttl_seconds=config.feature_cache_ttl_seconds
        )
        self.compression_enabled = config.enable_feature_compression
        self.max_size_mb = config.feature_cache_size_mb

    def get_features(self, cache_key: str) -> Optional[np.ndarray]:
        """特徴量取得"""
        cached_data = self.cache.get(cache_key)
        if cached_data is None:
            return None

        try:
            if self.compression_enabled:
                # 圧縮データを展開
                import lz4.frame
                decompressed = lz4.frame.decompress(cached_data)
                return pickle.loads(decompressed)
            else:
                return cached_data
        except Exception as e:
            logger.error(f"Feature decompression failed: {e}")
            return None

    def store_features(self, cache_key: str, features: np.ndarray) -> bool:
        """特徴量保存"""
        try:
            # メモリサイズチェック
            if self.cache.memory_usage() > self.max_size_mb:
                self._cleanup_cache()

            if self.compression_enabled:
                # 特徴量を圧縮
                import lz4.frame
                pickled_data = pickle.dumps(features)
                compressed_data = lz4.frame.compress(pickled_data)
                self.cache.put(cache_key, compressed_data)
            else:
                self.cache.put(cache_key, features)

            return True

        except Exception as e:
            logger.error(f"Feature caching failed: {e}")
            return False

    def _cleanup_cache(self) -> None:
        """キャッシュクリーンアップ"""
        # メモリ使用量が制限を超えている場合、半分クリア
        current_size = self.cache.size()
        target_size = current_size // 2

        while self.cache.size() > target_size:
            # LRUキャッシュなので自動的に古いものが削除される
            oldest_key = next(iter(self.cache.cache))
            self.cache.cache.pop(oldest_key, None)

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計"""
        return {
            "cache_size": self.cache.size(),
            "memory_usage_mb": self.cache.memory_usage(),
            "compression_enabled": self.compression_enabled
        }


class MemoryMonitor:
    """メモリ監視"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.stats = MemoryStats()
        self.monitoring = False
        self.callbacks: List[Callable[[MemoryStats], None]] = []

    def start_monitoring(self, interval_seconds: float = 5.0) -> None:
        """監視開始"""
        self.monitoring = True

        def monitor_loop():
            while self.monitoring:
                try:
                    self.update_stats()
                    self._check_thresholds()

                    # コールバック実行
                    for callback in self.callbacks:
                        try:
                            callback(self.stats)
                        except Exception as e:
                            logger.error(f"Memory monitor callback failed: {e}")

                    time.sleep(interval_seconds)

                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
                    time.sleep(interval_seconds)

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self) -> None:
        """監視停止"""
        self.monitoring = False
        logger.info("Memory monitoring stopped")

    def update_stats(self) -> None:
        """統計更新"""
        # システムメモリ
        memory = psutil.virtual_memory()
        self.stats.total_memory_gb = memory.total / (1024**3)
        self.stats.used_memory_gb = memory.used / (1024**3)
        self.stats.available_memory_gb = memory.available / (1024**3)
        self.stats.memory_usage_percent = memory.percent / 100.0

        # GPU メモリ（利用可能な場合）
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            self.stats.gpu_memory_used_gb = gpu_memory

        # GC統計
        self.stats.gc_collections = sum(gc.get_stats()[i]['collections'] for i in range(len(gc.get_stats())))

    def _check_thresholds(self) -> None:
        """閾値チェック"""
        usage = self.stats.memory_usage_percent

        if usage > self.config.memory_critical_threshold:
            logger.critical(f"Critical memory usage: {usage:.1%}")
            self._emergency_cleanup()
        elif usage > self.config.memory_warning_threshold:
            logger.warning(f"High memory usage: {usage:.1%}")

        if usage > self.config.gc_trigger_threshold:
            logger.info("Triggering garbage collection")
            self._trigger_gc()

    def _emergency_cleanup(self) -> None:
        """緊急メモリクリーンアップ"""
        logger.info("Performing emergency memory cleanup")
        gc.collect()

        # GPU メモリクリア（可能な場合）
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _trigger_gc(self) -> None:
        """ガベージコレクション実行"""
        before = self.stats.used_memory_gb
        gc.collect()
        self.update_stats()
        after = self.stats.used_memory_gb

        freed_mb = (before - after) * 1024
        if freed_mb > 0:
            logger.info(f"GC freed {freed_mb:.1f}MB")

    def add_callback(self, callback: Callable[[MemoryStats], None]) -> None:
        """監視コールバック追加"""
        self.callbacks.append(callback)

    def get_stats(self) -> MemoryStats:
        """現在の統計取得"""
        self.update_stats()
        return self.stats


class ZeroCopyManager:
    """ゼロコピー管理"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.mapped_files: Dict[str, mmap.mmap] = {}

    @contextmanager
    def zero_copy_array(self, data: np.ndarray):
        """ゼロコピー配列コンテキスト"""
        if not self.config.enable_zero_copy:
            yield data
            return

        try:
            # データサイズチェック
            size_mb = data.nbytes / (1024 * 1024)

            if size_mb > self.config.mmap_threshold_mb:
                # 大きなデータはmemory-mapped arrayを使用
                temp_file = f"/tmp/zero_copy_{id(data)}.npy"
                np.save(temp_file, data)

                with open(temp_file, 'r+b') as f:
                    # NumPyヘッダーをスキップ
                    np.lib.format.read_magic(f)
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)

                    # メモリマップ作成
                    mm = mmap.mmap(f.fileno(), 0)
                    mapped_array = np.frombuffer(mm, dtype=dtype).reshape(shape)

                    yield mapped_array

                    mm.close()

                # 一時ファイル削除
                os.unlink(temp_file)
            else:
                # 小さなデータは通常の配列を使用
                yield data

        except Exception as e:
            logger.error(f"Zero-copy operation failed: {e}")
            yield data

    def create_shared_array(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """共有メモリ配列作成"""
        try:
            import multiprocessing as mp
            shared_array = mp.Array('f', int(np.prod(shape)))
            return np.frombuffer(shared_array.get_obj(), dtype=dtype).reshape(shape)
        except Exception as e:
            logger.error(f"Shared array creation failed: {e}")
            return np.zeros(shape, dtype=dtype)


class MemoryOptimizer:
    """メモリ最適化システム"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.model_pool = ModelPool(config)
        self.feature_cache = FeatureCache(config)
        self.memory_monitor = MemoryMonitor(config)
        self.zero_copy_manager = ZeroCopyManager(config)

        # GPU初期化
        self._initialize_gpu_memory()

        # メモリ監視開始
        self.memory_monitor.add_callback(self._memory_callback)
        self.memory_monitor.start_monitoring()

    def _initialize_gpu_memory(self) -> None:
        """GPU メモリ初期化"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # メモリ成長設定
            if self.config.enable_gpu_memory_growth:
                torch.cuda.empty_cache()

            # メモリ使用率制限
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(self.config.gpu_memory_fraction)

            logger.info(f"GPU memory initialized: {torch.cuda.get_device_name()}")

    def _memory_callback(self, stats: MemoryStats) -> None:
        """メモリ監視コールバック"""
        if stats.memory_usage_percent > self.config.memory_warning_threshold:
            # 高メモリ使用時の自動クリーンアップ
            self.cleanup_unused_resources()

    def get_model(self, model_id: str, model_loader: Callable[[], Any]) -> Any:
        """最適化されたモデル取得"""
        return self.model_pool.load_model(model_id, model_loader)

    def release_model(self, model_id: str) -> None:
        """モデル解放"""
        self.model_pool.release_model(model_id)

    def cache_features(self, cache_key: str, features: np.ndarray) -> bool:
        """特徴量キャッシング"""
        return self.feature_cache.store_features(cache_key, features)

    def get_cached_features(self, cache_key: str) -> Optional[np.ndarray]:
        """キャッシュ済み特徴量取得"""
        return self.feature_cache.get_features(cache_key)

    @contextmanager
    def optimized_inference(self, model_id: str, model_loader: Callable[[], Any]):
        """最適化された推論コンテキスト"""
        model = None
        try:
            # モデル取得
            model = self.get_model(model_id, model_loader)
            yield model
        finally:
            # モデル解放
            if model:
                self.release_model(model_id)

    def cleanup_unused_resources(self) -> None:
        """未使用リソースクリーンアップ"""
        logger.info("Cleaning up unused resources")

        # 古いキャッシュエントリ削除
        self.feature_cache._cleanup_cache()

        # ガベージコレクション
        gc.collect()

        # GPU メモリクリア
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """包括的統計取得"""
        return {
            "memory_stats": self.memory_monitor.get_stats().__dict__,
            "model_pool_stats": self.model_pool.get_stats(),
            "feature_cache_stats": self.feature_cache.get_stats(),
            "gpu_available": TORCH_AVAILABLE and torch.cuda.is_available(),
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3) if TORCH_AVAILABLE and torch.cuda.is_available() else 0
        }

    def shutdown(self) -> None:
        """システム終了"""
        logger.info("Shutting down memory optimizer")
        self.memory_monitor.stop_monitoring()
        self.model_pool.clear()
        self.feature_cache.cache.clear()


# 使用例とテスト
async def test_memory_optimizer():
    """メモリ最適化テスト"""

    # 設定
    config = MemoryConfig(
        max_models_in_memory=3,
        feature_cache_size_mb=100,
        memory_warning_threshold=0.8
    )

    # 最適化システム初期化
    optimizer = MemoryOptimizer(config)

    try:
        print("=== Memory Optimization Test ===")

        # ダミーモデルローダー
        def create_dummy_model(model_id: str):
            return f"DummyModel_{model_id}_{time.time()}"

        # モデルプールテスト
        print("\n1. Model Pool Test")
        for i in range(5):
            model_id = f"model_{i}"
            with optimizer.optimized_inference(model_id, lambda: create_dummy_model(model_id)) as model:
                print(f"  Using model: {model}")

        # 特徴量キャッシュテスト
        print("\n2. Feature Cache Test")
        for i in range(3):
            features = np.random.randn(100, 10).astype(np.float32)
            cache_key = f"features_{i}"

            # キャッシュ保存
            success = optimizer.cache_features(cache_key, features)
            print(f"  Cached features_{i}: {success}")

            # キャッシュ取得
            cached = optimizer.get_cached_features(cache_key)
            print(f"  Retrieved features_{i}: {cached is not None}")

        # 統計表示
        print("\n3. Memory Statistics")
        stats = optimizer.get_comprehensive_stats()

        memory_stats = stats["memory_stats"]
        print(f"  System Memory: {memory_stats['used_memory_gb']:.1f}GB / {memory_stats['total_memory_gb']:.1f}GB ({memory_stats['memory_usage_percent']:.1%})")

        model_stats = stats["model_pool_stats"]
        print(f"  Models in Memory: {model_stats['models_count']} ({model_stats['total_memory_mb']:.1f}MB)")

        cache_stats = stats["feature_cache_stats"]
        print(f"  Feature Cache: {cache_stats['cache_size']} items ({cache_stats['memory_usage_mb']:.1f}MB)")

        # ゼロコピーテスト
        print("\n4. Zero-Copy Test")
        large_array = np.random.randn(1000, 1000).astype(np.float32)
        with optimizer.zero_copy_manager.zero_copy_array(large_array) as zero_copy_array:
            result = np.sum(zero_copy_array)
            print(f"  Zero-copy computation result: {result:.2f}")

        # クリーンアップテスト
        print("\n5. Cleanup Test")
        optimizer.cleanup_unused_resources()
        print("  Cleanup completed")

        final_stats = optimizer.get_comprehensive_stats()
        print(f"  Memory after cleanup: {final_stats['memory_stats']['memory_usage_percent']:.1%}")

    except Exception as e:
        print(f"Memory optimization test failed: {e}")

    finally:
        optimizer.shutdown()


if __name__ == "__main__":
    asyncio.run(test_memory_optimizer())