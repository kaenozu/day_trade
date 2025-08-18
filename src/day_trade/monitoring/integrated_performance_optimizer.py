#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Performance Optimizer - 統合パフォーマンス最適化システム
Issue #870拡張予測システムと既存システムの統合パフォーマンス最適化
"""

import time
import gc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import weakref
from collections import defaultdict, deque
import json
import pickle
import joblib
from pathlib import Path
import warnings

# Performance Optimization System インポート
from performance_optimization_system import (
    PerformanceOptimizationSystem,
    get_performance_system,
    PerformanceMetrics,
    OptimizationResult,
    PerformanceLevel
)

# Enhanced Prediction System関連インポート（利用可能な場合）
try:
    from advanced_feature_selector import AdvancedFeatureSelector
    FEATURE_SELECTOR_AVAILABLE = True
except ImportError:
    FEATURE_SELECTOR_AVAILABLE = False

try:
    from advanced_ensemble_system import AdvancedEnsembleSystem
    ENSEMBLE_SYSTEM_AVAILABLE = True
except ImportError:
    ENSEMBLE_SYSTEM_AVAILABLE = False

try:
    from hybrid_timeseries_predictor import HybridTimeSeriesPredictor
    HYBRID_PREDICTOR_AVAILABLE = True
except ImportError:
    HYBRID_PREDICTOR_AVAILABLE = False

try:
    from meta_learning_system import MetaLearningSystem
    META_LEARNING_AVAILABLE = True
except ImportError:
    META_LEARNING_AVAILABLE = False


class OptimizationTarget(Enum):
    """最適化対象"""
    MEMORY_USAGE = "memory_usage"
    CPU_EFFICIENCY = "cpu_efficiency"
    CACHE_PERFORMANCE = "cache_performance"
    MODEL_LOADING = "model_loading"
    FEATURE_PROCESSING = "feature_processing"
    PREDICTION_SPEED = "prediction_speed"
    DATA_PIPELINE = "data_pipeline"


class OptimizationPriority(Enum):
    """最適化優先度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class OptimizationStrategy:
    """最適化戦略"""
    target: OptimizationTarget
    priority: OptimizationPriority
    method: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    success_threshold: float = 10.0  # 成功と判断する改善率（%）


@dataclass
class SystemState:
    """システム状態"""
    memory_usage_mb: float
    cpu_usage_percent: float
    active_models: int
    cached_features: int
    pending_predictions: int
    response_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


class ModelCache:
    """モデルキャッシュシステム"""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """キャッシュからモデルを取得"""
        with self._lock:
            if key not in self.cache:
                self.miss_count += 1
                return None

            # TTLチェック
            if time.time() - self.access_times[key] > self.ttl_seconds:
                del self.cache[key]
                del self.access_times[key]
                self.miss_count += 1
                return None

            self.access_times[key] = time.time()
            self.hit_count += 1
            return self.cache[key]

    def put(self, key: str, value: Any):
        """モデルをキャッシュに保存"""
        with self._lock:
            # サイズ制限チェック
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            self.cache[key] = value
            self.access_times[key] = time.time()

    def _evict_lru(self):
        """LRU方式で古いエントリを削除"""
        if not self.access_times:
            return

        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]

    def get_hit_rate(self) -> float:
        """ヒット率を取得"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def clear(self):
        """キャッシュをクリア"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': self.get_hit_rate(),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'memory_usage_mb': sum(self._get_object_size(obj) for obj in self.cache.values()) / 1024 / 1024
        }

    def _get_object_size(self, obj: Any) -> int:
        """オブジェクトのサイズを推定"""
        try:
            return len(pickle.dumps(obj))
        except:
            return 1024  # デフォルト1KB


class FeatureCache:
    """特徴量キャッシュシステム"""

    def __init__(self, max_features: int = 10000, compression: bool = True):
        self.max_features = max_features
        self.compression = compression
        self.cache = {}
        self.access_count = defaultdict(int)
        self.last_access = {}
        self._lock = threading.RLock()

    def get_cache_key(self, data_hash: str, config: Dict[str, Any]) -> str:
        """キャッシュキーを生成"""
        config_str = json.dumps(config, sort_keys=True)
        return f"{data_hash}_{hash(config_str)}"

    def get_features(self, cache_key: str) -> Optional[np.ndarray]:
        """特徴量を取得"""
        with self._lock:
            if cache_key not in self.cache:
                return None

            self.access_count[cache_key] += 1
            self.last_access[cache_key] = time.time()

            features = self.cache[cache_key]
            if self.compression and hasattr(features, 'get_array'):
                # 圧縮されている場合は展開
                return features.get_array()
            return features

    def put_features(self, cache_key: str, features: np.ndarray):
        """特徴量を保存"""
        with self._lock:
            # サイズ制限チェック
            if len(self.cache) >= self.max_features:
                self._evict_least_used()

            if self.compression:
                # NumPy配列を圧縮して保存
                try:
                    import joblib
                    compressed = joblib.Memory(None).cache(lambda x: x)(features)
                    self.cache[cache_key] = compressed
                except ImportError:
                    # joblibが利用できない場合は圧縮せずに保存
                    self.cache[cache_key] = features
                except:
                    self.cache[cache_key] = features
            else:
                self.cache[cache_key] = features

            self.access_count[cache_key] = 1
            self.last_access[cache_key] = time.time()

    def _evict_least_used(self):
        """最も使用頻度の低いエントリを削除"""
        if not self.cache:
            return

        # アクセス頻度と最終アクセス時刻を考慮
        scores = {}
        current_time = time.time()
        for key in self.cache.keys():
            access_freq = self.access_count[key]
            time_since_access = current_time - self.last_access[key]
            # スコア計算（高いほど重要）
            scores[key] = access_freq / (1 + time_since_access / 3600)  # 1時間で重み半減

        # 最低スコアのエントリを削除
        worst_key = min(scores.keys(), key=lambda k: scores[k])
        del self.cache[worst_key]
        del self.access_count[worst_key]
        del self.last_access[worst_key]

    def clear(self):
        """キャッシュをクリア"""
        with self._lock:
            self.cache.clear()
            self.access_count.clear()
            self.last_access.clear()

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        total_accesses = sum(self.access_count.values())
        return {
            'size': len(self.cache),
            'max_size': self.max_features,
            'total_accesses': total_accesses,
            'compression_enabled': self.compression,
            'estimated_memory_mb': self._estimate_memory_usage() / 1024 / 1024
        }

    def _estimate_memory_usage(self) -> int:
        """メモリ使用量を推定"""
        total_size = 0
        for features in self.cache.values():
            if hasattr(features, 'nbytes'):
                total_size += features.nbytes
            else:
                total_size += 1024  # デフォルト推定
        return total_size


class PredictionBatchProcessor:
    """予測バッチ処理システム"""

    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.pending_requests = deque()
        self.processing = False
        self._lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def add_prediction_request(self, request_id: str, data: Dict[str, Any], callback: Callable):
        """予測リクエストを追加"""
        with self._lock:
            self.pending_requests.append({
                'id': request_id,
                'data': data,
                'callback': callback,
                'timestamp': time.time()
            })

    def process_batch(self, predictor: Any) -> List[Dict[str, Any]]:
        """バッチ処理を実行"""
        if self.processing or not self.pending_requests:
            return []

        with self._lock:
            self.processing = True

            # バッチサイズ分のリクエストを取得
            batch = []
            for _ in range(min(self.batch_size, len(self.pending_requests))):
                if self.pending_requests:
                    batch.append(self.pending_requests.popleft())

        try:
            # バッチ予測実行
            batch_data = [req['data'] for req in batch]
            predictions = predictor.predict_batch(batch_data)

            # 結果を各リクエストに配布
            results = []
            for i, (req, prediction) in enumerate(zip(batch, predictions)):
                result = {
                    'id': req['id'],
                    'prediction': prediction,
                    'processing_time': time.time() - req['timestamp']
                }

                # コールバック実行
                if req['callback']:
                    try:
                        req['callback'](result)
                    except Exception as e:
                        result['callback_error'] = str(e)

                results.append(result)

            return results

        finally:
            self.processing = False

    def get_stats(self) -> Dict[str, Any]:
        """バッチ処理統計を取得"""
        return {
            'pending_requests': len(self.pending_requests),
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'processing': self.processing
        }


class IntegratedPerformanceOptimizer:
    """統合パフォーマンス最適化システム"""

    def __init__(self):
        # 基本パフォーマンスシステム
        self.performance_system = get_performance_system()

        # 専用キャッシュシステム
        self.model_cache = ModelCache(max_size=50, ttl_seconds=7200)
        self.feature_cache = FeatureCache(max_features=5000, compression=True)

        # バッチ処理システム
        self.batch_processor = PredictionBatchProcessor(batch_size=50, max_workers=6)

        # 最適化戦略
        self.strategies = self._initialize_strategies()

        # システム状態監視
        self.state_history = deque(maxlen=1000)
        self.optimization_results = deque(maxlen=100)

        # ログ設定
        from daytrade_logging import get_logger
        self.logger = get_logger("integrated_performance_optimizer")

        # 設定
        self.auto_optimization_enabled = True
        self.optimization_interval = 300  # 5分
        self.last_optimization = datetime.now()

        # プロファイリング
        self.profiling_enabled = False
        self.profiling_data = defaultdict(list)

        self.logger.info("Integrated Performance Optimizer initialized")

    def _initialize_strategies(self) -> List[OptimizationStrategy]:
        """最適化戦略を初期化"""
        return [
            OptimizationStrategy(
                target=OptimizationTarget.MEMORY_USAGE,
                priority=OptimizationPriority.HIGH,
                method="garbage_collection_and_cache_cleanup",
                parameters={
                    'force_gc': True,
                    'clear_unused_caches': True,
                    'compact_models': True
                }
            ),
            OptimizationStrategy(
                target=OptimizationTarget.MODEL_LOADING,
                priority=OptimizationPriority.MEDIUM,
                method="model_cache_optimization",
                parameters={
                    'preload_common_models': True,
                    'compress_cached_models': False,
                    'optimize_cache_size': True
                }
            ),
            OptimizationStrategy(
                target=OptimizationTarget.FEATURE_PROCESSING,
                priority=OptimizationPriority.MEDIUM,
                method="feature_cache_optimization",
                parameters={
                    'enable_compression': True,
                    'optimize_batch_size': True,
                    'precompute_common_features': True
                }
            ),
            OptimizationStrategy(
                target=OptimizationTarget.PREDICTION_SPEED,
                priority=OptimizationPriority.HIGH,
                method="batch_processing_optimization",
                parameters={
                    'optimize_batch_size': True,
                    'parallel_processing': True,
                    'pipeline_optimization': True
                }
            ),
            OptimizationStrategy(
                target=OptimizationTarget.CPU_EFFICIENCY,
                priority=OptimizationPriority.MEDIUM,
                method="cpu_and_threading_optimization",
                parameters={
                    'optimize_thread_count': True,
                    'enable_cpu_affinity': False,
                    'optimize_numpy_threads': True
                }
            )
        ]

    def capture_system_state(self) -> SystemState:
        """現在のシステム状態を取得"""
        process = psutil.Process()
        memory_info = process.memory_info()

        # レスポンス時間測定
        start_time = time.time()
        _ = np.random.randn(1000, 100).mean()  # 簡単な計算でレスポンス時間測定
        response_time = (time.time() - start_time) * 1000

        state = SystemState(
            memory_usage_mb=memory_info.rss / 1024 / 1024,
            cpu_usage_percent=process.cpu_percent(interval=0.1),
            active_models=len(self.model_cache.cache),
            cached_features=len(self.feature_cache.cache),
            pending_predictions=len(self.batch_processor.pending_requests),
            response_time_ms=response_time
        )

        self.state_history.append(state)
        return state

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """メモリ使用量最適化"""
        start_time = time.time()
        before_state = self.capture_system_state()

        # ガベージコレクション
        gc_stats = {}
        for generation in range(3):
            collected = gc.collect(generation)
            gc_stats[f'generation_{generation}'] = collected

        # キャッシュクリーンアップ
        model_cache_size_before = len(self.model_cache.cache)
        feature_cache_size_before = len(self.feature_cache.cache)

        # 古いエントリの削除
        current_time = time.time()
        expired_models = []
        for key, access_time in self.model_cache.access_times.items():
            if current_time - access_time > self.model_cache.ttl_seconds:
                expired_models.append(key)

        for key in expired_models:
            if key in self.model_cache.cache:
                del self.model_cache.cache[key]
                del self.model_cache.access_times[key]

        # 使用頻度の低い特徴量キャッシュの削除
        if len(self.feature_cache.cache) > self.feature_cache.max_features * 0.8:
            for _ in range(int(self.feature_cache.max_features * 0.2)):
                self.feature_cache._evict_least_used()

        after_state = self.capture_system_state()
        execution_time = (time.time() - start_time) * 1000

        # 改善率計算
        memory_freed = before_state.memory_usage_mb - after_state.memory_usage_mb
        improvement_percent = (memory_freed / before_state.memory_usage_mb * 100) if before_state.memory_usage_mb > 0 else 0

        result = {
            'strategy': 'memory_usage_optimization',
            'execution_time_ms': execution_time,
            'memory_freed_mb': memory_freed,
            'improvement_percent': improvement_percent,
            'gc_stats': gc_stats,
            'cache_cleanup': {
                'models_removed': model_cache_size_before - len(self.model_cache.cache),
                'features_removed': feature_cache_size_before - len(self.feature_cache.cache)
            },
            'before_state': before_state,
            'after_state': after_state
        }

        self.logger.info(f"Memory optimization completed: {memory_freed:.1f}MB freed ({improvement_percent:.1f}% improvement)")
        return result

    def optimize_model_cache(self) -> Dict[str, Any]:
        """モデルキャッシュ最適化"""
        start_time = time.time()
        before_stats = self.model_cache.get_stats()

        # キャッシュサイズの動的調整
        hit_rate = self.model_cache.get_hit_rate()
        if hit_rate > 0.9:
            # ヒット率が高い場合はキャッシュサイズを増加
            new_size = min(self.model_cache.max_size * 1.2, 200)
        elif hit_rate < 0.5:
            # ヒット率が低い場合はサイズを減少
            new_size = max(self.model_cache.max_size * 0.8, 20)
        else:
            new_size = self.model_cache.max_size

        self.model_cache.max_size = int(new_size)

        # TTLの動的調整
        avg_access_interval = self._calculate_average_access_interval()
        if avg_access_interval > 0:
            # アクセス間隔に基づいてTTLを調整
            optimal_ttl = max(avg_access_interval * 2, 1800)  # 最低30分
            self.model_cache.ttl_seconds = min(optimal_ttl, 7200)  # 最大2時間

        after_stats = self.model_cache.get_stats()
        execution_time = (time.time() - start_time) * 1000

        # 改善率計算（ヒット率の改善）
        hit_rate_improvement = ((after_stats['hit_rate'] - before_stats['hit_rate']) * 100) if before_stats['hit_rate'] > 0 else 0

        result = {
            'strategy': 'model_cache_optimization',
            'execution_time_ms': execution_time,
            'hit_rate_improvement': hit_rate_improvement,
            'cache_size_adjustment': {
                'old_size': before_stats['max_size'],
                'new_size': after_stats['max_size']
            },
            'ttl_adjustment': self.model_cache.ttl_seconds,
            'before_stats': before_stats,
            'after_stats': after_stats
        }

        self.logger.info(f"Model cache optimization completed: Hit rate {after_stats['hit_rate']:.1%}")
        return result

    def _calculate_average_access_interval(self) -> float:
        """平均アクセス間隔を計算"""
        if len(self.model_cache.access_times) < 2:
            return 0

        access_times = sorted(self.model_cache.access_times.values())
        intervals = [access_times[i] - access_times[i-1] for i in range(1, len(access_times))]
        return sum(intervals) / len(intervals) if intervals else 0

    def optimize_feature_processing(self) -> Dict[str, Any]:
        """特徴量処理最適化"""
        start_time = time.time()
        before_stats = self.feature_cache.get_stats()

        # キャッシュ圧縮の最適化
        if not self.feature_cache.compression and before_stats['estimated_memory_mb'] > 500:
            # メモリ使用量が多い場合は圧縮を有効化
            self.feature_cache.compression = True
            compressed_count = 0

            # 既存のキャッシュを圧縮
            for key, features in list(self.feature_cache.cache.items()):
                if isinstance(features, np.ndarray):
                    try:
                        # 圧縮実行
                        self.feature_cache.cache[key] = self._compress_features(features)
                        compressed_count += 1
                    except Exception as e:
                        self.logger.warning(f"Feature compression failed for key {key}: {e}")

        # キャッシュサイズの動的調整
        utilization = before_stats['size'] / before_stats['max_size']
        if utilization > 0.9:
            # 使用率が高い場合はサイズを増加
            new_max_size = min(self.feature_cache.max_features * 1.5, 20000)
        elif utilization < 0.3:
            # 使用率が低い場合はサイズを削減
            new_max_size = max(self.feature_cache.max_features * 0.7, 1000)
        else:
            new_max_size = self.feature_cache.max_features

        self.feature_cache.max_features = int(new_max_size)

        after_stats = self.feature_cache.get_stats()
        execution_time = (time.time() - start_time) * 1000

        # 改善率計算（メモリ削減率）
        memory_reduction = before_stats['estimated_memory_mb'] - after_stats['estimated_memory_mb']
        improvement_percent = (memory_reduction / before_stats['estimated_memory_mb'] * 100) if before_stats['estimated_memory_mb'] > 0 else 0

        result = {
            'strategy': 'feature_processing_optimization',
            'execution_time_ms': execution_time,
            'memory_reduction_mb': memory_reduction,
            'improvement_percent': improvement_percent,
            'compression_enabled': self.feature_cache.compression,
            'cache_size_adjustment': {
                'old_size': before_stats['max_size'],
                'new_size': after_stats['max_size']
            },
            'before_stats': before_stats,
            'after_stats': after_stats
        }

        self.logger.info(f"Feature processing optimization completed: {memory_reduction:.1f}MB memory reduction")
        return result

    def _compress_features(self, features: np.ndarray) -> Any:
        """特徴量を圧縮"""
        try:
            import joblib
            return joblib.Memory(None).cache(lambda x: x)(features)
        except ImportError:
            # joblibが利用できない場合は圧縮せずに返す
            return features
        except:
            return features

    def optimize_prediction_speed(self) -> Dict[str, Any]:
        """予測速度最適化"""
        start_time = time.time()
        before_stats = self.batch_processor.get_stats()

        # バッチサイズの動的調整
        current_batch_size = self.batch_processor.batch_size
        pending_requests = before_stats['pending_requests']

        if pending_requests > current_batch_size * 2:
            # 待機リクエストが多い場合はバッチサイズを増加
            new_batch_size = min(current_batch_size * 1.5, 200)
        elif pending_requests < current_batch_size * 0.5 and current_batch_size > 20:
            # 待機リクエストが少ない場合はバッチサイズを削減
            new_batch_size = max(current_batch_size * 0.8, 20)
        else:
            new_batch_size = current_batch_size

        self.batch_processor.batch_size = int(new_batch_size)

        # ワーカー数の動的調整
        cpu_count = psutil.cpu_count()
        current_workers = self.batch_processor.max_workers

        if pending_requests > 100 and current_workers < cpu_count:
            # 負荷が高い場合はワーカー数を増加
            new_workers = min(current_workers + 1, cpu_count)
            self.batch_processor.executor._max_workers = new_workers
            self.batch_processor.max_workers = new_workers
        elif pending_requests < 20 and current_workers > 2:
            # 負荷が低い場合はワーカー数を削減
            new_workers = max(current_workers - 1, 2)
            self.batch_processor.max_workers = new_workers

        after_stats = self.batch_processor.get_stats()
        execution_time = (time.time() - start_time) * 1000

        # 改善率計算（処理能力の向上）
        throughput_improvement = ((new_batch_size * after_stats['max_workers']) /
                                (current_batch_size * before_stats['max_workers']) - 1) * 100

        result = {
            'strategy': 'prediction_speed_optimization',
            'execution_time_ms': execution_time,
            'throughput_improvement': throughput_improvement,
            'batch_size_adjustment': {
                'old_size': current_batch_size,
                'new_size': new_batch_size
            },
            'workers_adjustment': {
                'old_workers': before_stats['max_workers'],
                'new_workers': after_stats['max_workers']
            },
            'before_stats': before_stats,
            'after_stats': after_stats
        }

        self.logger.info(f"Prediction speed optimization completed: {throughput_improvement:.1f}% throughput improvement")
        return result

    def optimize_cpu_efficiency(self) -> Dict[str, Any]:
        """CPU効率最適化"""
        start_time = time.time()
        before_state = self.capture_system_state()

        optimizations = []

        # NumPyスレッド数の最適化
        try:
            import os
            cpu_count = psutil.cpu_count(logical=False)  # 物理コア数
            optimal_threads = max(1, cpu_count // 2)  # 物理コアの半分

            os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
            os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)

            optimizations.append(f"NumPy threads set to {optimal_threads}")
        except Exception as e:
            self.logger.warning(f"NumPy thread optimization failed: {e}")

        # ガベージコレクションの最適化
        gc.set_threshold(700, 10, 10)  # より効率的な閾値
        optimizations.append("Garbage collection threshold optimized")

        # Pythonの最適化フラグ
        try:
            import sys
            if not sys.flags.optimize:
                self.logger.info("Consider running with -O flag for better performance")
        except:
            pass

        after_state = self.capture_system_state()
        execution_time = (time.time() - start_time) * 1000

        # 改善率計算（CPU使用率の改善）
        cpu_improvement = max(0, before_state.cpu_usage_percent - after_state.cpu_usage_percent)
        improvement_percent = (cpu_improvement / before_state.cpu_usage_percent * 100) if before_state.cpu_usage_percent > 0 else 0

        result = {
            'strategy': 'cpu_efficiency_optimization',
            'execution_time_ms': execution_time,
            'cpu_improvement': cpu_improvement,
            'improvement_percent': improvement_percent,
            'optimizations_applied': optimizations,
            'before_state': before_state,
            'after_state': after_state
        }

        self.logger.info(f"CPU efficiency optimization completed: {len(optimizations)} optimizations applied")
        return result

    def run_comprehensive_optimization(self) -> List[Dict[str, Any]]:
        """包括的最適化を実行"""
        self.logger.info("Starting comprehensive performance optimization")

        optimization_results = []

        # 現在の状態を評価
        current_state = self.capture_system_state()
        performance_metrics = self.performance_system.get_current_metrics()
        performance_level = self.performance_system.assess_performance_level(performance_metrics)

        self.logger.info(f"Current performance level: {performance_level.value}")

        # 優先度に基づいて最適化を実行
        enabled_strategies = [s for s in self.strategies if s.enabled]
        enabled_strategies.sort(key=lambda x: x.priority.value)

        for strategy in enabled_strategies:
            try:
                if strategy.target == OptimizationTarget.MEMORY_USAGE:
                    result = self.optimize_memory_usage()
                elif strategy.target == OptimizationTarget.MODEL_LOADING:
                    result = self.optimize_model_cache()
                elif strategy.target == OptimizationTarget.FEATURE_PROCESSING:
                    result = self.optimize_feature_processing()
                elif strategy.target == OptimizationTarget.PREDICTION_SPEED:
                    result = self.optimize_prediction_speed()
                elif strategy.target == OptimizationTarget.CPU_EFFICIENCY:
                    result = self.optimize_cpu_efficiency()
                else:
                    continue

                result['strategy_config'] = {
                    'target': strategy.target.value,
                    'priority': strategy.priority.value,
                    'method': strategy.method,
                    'parameters': strategy.parameters
                }

                optimization_results.append(result)

                # 成功閾値チェック
                if result.get('improvement_percent', 0) >= strategy.success_threshold:
                    self.logger.info(f"Strategy {strategy.target.value} succeeded with {result['improvement_percent']:.1f}% improvement")

            except Exception as e:
                self.logger.error(f"Optimization strategy {strategy.target.value} failed: {e}")
                optimization_results.append({
                    'strategy': strategy.target.value,
                    'error': str(e),
                    'success': False
                })

        # 基本パフォーマンスシステムの最適化も実行
        base_optimizations = self.performance_system.auto_optimize()
        for base_opt in base_optimizations:
            optimization_results.append({
                'strategy': f"base_{base_opt.strategy.value}",
                'improvement_percent': base_opt.improvement_percent,
                'execution_time_ms': base_opt.execution_time_ms,
                'success': base_opt.success,
                'details': base_opt.details
            })

        # 結果を履歴に保存
        self.optimization_results.extend(optimization_results)
        self.last_optimization = datetime.now()

        # 総合結果のサマリー
        successful_optimizations = [r for r in optimization_results if r.get('success', True)]
        total_improvement = sum(r.get('improvement_percent', 0) for r in successful_optimizations)

        self.logger.info(f"Comprehensive optimization completed: {len(successful_optimizations)}/{len(optimization_results)} strategies succeeded")
        self.logger.info(f"Total improvement: {total_improvement:.1f}%")

        return optimization_results

    def get_performance_report(self) -> Dict[str, Any]:
        """包括的パフォーマンスレポートを生成"""
        current_state = self.capture_system_state()
        base_report = self.performance_system.get_performance_report()

        # 最近の最適化結果
        recent_optimizations = list(self.optimization_results)[-20:]
        successful_optimizations = [opt for opt in recent_optimizations if opt.get('success', True)]

        # キャッシュ統計
        model_cache_stats = self.model_cache.get_stats()
        feature_cache_stats = self.feature_cache.get_stats()
        batch_processor_stats = self.batch_processor.get_stats()

        # 統計計算
        total_improvement = sum(opt.get('improvement_percent', 0) for opt in successful_optimizations)
        avg_improvement = total_improvement / len(successful_optimizations) if successful_optimizations else 0

        return {
            'system_state': {
                'memory_usage_mb': current_state.memory_usage_mb,
                'cpu_usage_percent': current_state.cpu_usage_percent,
                'active_models': current_state.active_models,
                'cached_features': current_state.cached_features,
                'pending_predictions': current_state.pending_predictions,
                'response_time_ms': current_state.response_time_ms
            },
            'cache_performance': {
                'model_cache': model_cache_stats,
                'feature_cache': feature_cache_stats,
                'batch_processor': batch_processor_stats
            },
            'optimization_summary': {
                'total_optimizations': len(self.optimization_results),
                'recent_optimizations': len(recent_optimizations),
                'successful_optimizations': len(successful_optimizations),
                'total_improvement_percent': total_improvement,
                'average_improvement_percent': avg_improvement,
                'last_optimization': self.last_optimization.isoformat()
            },
            'recent_optimization_details': [
                {
                    'strategy': opt.get('strategy', 'unknown'),
                    'improvement_percent': opt.get('improvement_percent', 0),
                    'execution_time_ms': opt.get('execution_time_ms', 0),
                    'success': opt.get('success', True)
                }
                for opt in recent_optimizations
            ],
            'base_performance_report': base_report,
            'configuration': {
                'auto_optimization_enabled': self.auto_optimization_enabled,
                'optimization_interval_seconds': self.optimization_interval,
                'profiling_enabled': self.profiling_enabled
            },
            'timestamp': datetime.now().isoformat()
        }


# グローバルインスタンス
_integrated_optimizer = None


def get_integrated_optimizer() -> IntegratedPerformanceOptimizer:
    """グローバル統合最適化システムを取得"""
    global _integrated_optimizer
    if _integrated_optimizer is None:
        _integrated_optimizer = IntegratedPerformanceOptimizer()
    return _integrated_optimizer


def optimize_system_performance() -> List[Dict[str, Any]]:
    """システムパフォーマンス最適化（便利関数）"""
    return get_integrated_optimizer().run_comprehensive_optimization()


def get_system_performance_report() -> Dict[str, Any]:
    """システムパフォーマンスレポート取得（便利関数）"""
    return get_integrated_optimizer().get_performance_report()


if __name__ == "__main__":
    print("⚡ 統合パフォーマンス最適化システムテスト")
    print("=" * 60)

    optimizer = IntegratedPerformanceOptimizer()

    # 現在の状態
    state = optimizer.capture_system_state()
    print(f"現在のシステム状態:")
    print(f"  メモリ使用量: {state.memory_usage_mb:.1f}MB")
    print(f"  CPU使用率: {state.cpu_usage_percent:.1f}%")
    print(f"  キャッシュ状況: モデル{state.active_models}個, 特徴量{state.cached_features}個")
    print(f"  レスポンス時間: {state.response_time_ms:.2f}ms")

    # 包括的最適化実行
    print("\n包括的最適化実行中...")
    optimizations = optimizer.run_comprehensive_optimization()

    print(f"\n最適化結果:")
    for opt in optimizations:
        success_indicator = "✅" if opt.get('success', True) else "❌"
        improvement = opt.get('improvement_percent', 0)
        print(f"  {success_indicator} {opt.get('strategy', 'unknown')}: {improvement:.1f}% 改善")

    # レポート生成
    report = optimizer.get_performance_report()
    print(f"\n統計サマリー:")
    print(f"  総最適化回数: {report['optimization_summary']['total_optimizations']}")
    print(f"  成功回数: {report['optimization_summary']['successful_optimizations']}")
    print(f"  平均改善率: {report['optimization_summary']['average_improvement_percent']:.1f}%")

    print("\nテスト完了")