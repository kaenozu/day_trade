#!/usr/bin/env python3
"""
高度な推論最適化システム
Advanced Inference Optimization System

Issue #761: MLモデル推論パイプラインの高速化と最適化 Phase 4
"""

import asyncio
import logging
import time
import json
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import threading
import psutil

# 機械学習フレームワーク
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import sklearn.metrics
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ログ設定
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """高度最適化設定"""
    # 動的モデル選択
    enable_dynamic_model_selection: bool = True
    model_selection_strategy: str = "performance_based"  # "performance_based", "latency_based", "accuracy_based"
    model_switch_threshold: float = 0.1  # 10%の性能改善で切り替え

    # 推論結果キャッシング
    enable_inference_caching: bool = True
    cache_similarity_threshold: float = 0.95  # 95%類似で キャッシュヒット
    cache_ttl_seconds: int = 300  # 5分
    max_cache_entries: int = 10000

    # プロファイリング
    enable_profiling: bool = True
    profiling_sample_rate: float = 0.1  # 10%をプロファイル
    profiling_window_size: int = 1000

    # A/Bテスト
    enable_ab_testing: bool = True
    ab_test_split_ratio: float = 0.1  # 10%がテストグループ
    ab_test_duration_hours: int = 24

    # 自動最適化
    enable_auto_optimization: bool = True
    optimization_interval_hours: int = 6
    auto_tuning_enabled: bool = True


@dataclass
class ModelPerformanceMetrics:
    """モデル性能メトリクス"""
    model_id: str
    total_predictions: int = 0
    avg_latency_ms: float = 0.0
    accuracy: float = 0.0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


@dataclass
class InferenceCacheEntry:
    """推論キャッシュエントリ"""
    input_hash: str
    output: np.ndarray
    model_id: str
    confidence: float
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)


class DynamicModelSelector:
    """動的モデル選択器"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.model_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self.selection_history: deque = deque(maxlen=1000)
        self.lock = threading.RLock()

    def register_model(self, model_id: str) -> None:
        """モデル登録"""
        with self.lock:
            if model_id not in self.model_metrics:
                self.model_metrics[model_id] = ModelPerformanceMetrics(model_id=model_id)
                logger.info(f"Registered model for dynamic selection: {model_id}")

    def update_model_metrics(self, model_id: str, latency_ms: float, accuracy: float, throughput: float) -> None:
        """モデルメトリクス更新"""
        with self.lock:
            if model_id not in self.model_metrics:
                self.register_model(model_id)

            metrics = self.model_metrics[model_id]
            metrics.total_predictions += 1

            # 移動平均でメトリクス更新
            alpha = 0.1  # 学習率
            metrics.avg_latency_ms = (1 - alpha) * metrics.avg_latency_ms + alpha * latency_ms
            metrics.accuracy = (1 - alpha) * metrics.accuracy + alpha * accuracy
            metrics.throughput_per_second = (1 - alpha) * metrics.throughput_per_second + alpha * throughput
            metrics.last_updated = datetime.now()

    def select_optimal_model(self, available_models: List[str], strategy: Optional[str] = None) -> str:
        """最適モデル選択"""
        with self.lock:
            if not available_models:
                raise ValueError("No models available for selection")

            strategy = strategy or self.config.model_selection_strategy

            # 利用可能なモデルのメトリクス取得
            model_scores = {}

            for model_id in available_models:
                if model_id not in self.model_metrics:
                    self.register_model(model_id)

                metrics = self.model_metrics[model_id]

                if strategy == "latency_based":
                    # 低レイテンシ優先（逆数で計算）
                    score = 1.0 / (metrics.avg_latency_ms + 1.0)
                elif strategy == "accuracy_based":
                    # 高精度優先
                    score = metrics.accuracy
                elif strategy == "performance_based":
                    # バランス型：精度とレイテンシの調和平均
                    if metrics.accuracy > 0 and metrics.avg_latency_ms > 0:
                        score = 2 * metrics.accuracy * (1000.0 / metrics.avg_latency_ms) / (
                            metrics.accuracy + (1000.0 / metrics.avg_latency_ms)
                        )
                    else:
                        score = 0.0
                else:
                    # デフォルト：スループット優先
                    score = metrics.throughput_per_second

                model_scores[model_id] = score

            # 最高スコアのモデル選択
            best_model = max(model_scores.keys(), key=lambda x: model_scores[x])

            # 選択履歴記録
            self.selection_history.append({
                "timestamp": datetime.now(),
                "selected_model": best_model,
                "strategy": strategy,
                "scores": model_scores.copy()
            })

            logger.debug(f"Selected optimal model: {best_model} (strategy: {strategy}, score: {model_scores[best_model]:.3f})")
            return best_model

    def should_switch_model(self, current_model: str, candidate_model: str) -> bool:
        """モデル切り替え判定"""
        with self.lock:
            if current_model not in self.model_metrics or candidate_model not in self.model_metrics:
                return False

            current_metrics = self.model_metrics[current_model]
            candidate_metrics = self.model_metrics[candidate_model]

            # 性能改善率計算
            if current_metrics.avg_latency_ms > 0:
                latency_improvement = (current_metrics.avg_latency_ms - candidate_metrics.avg_latency_ms) / current_metrics.avg_latency_ms
            else:
                latency_improvement = 0.0

            accuracy_improvement = candidate_metrics.accuracy - current_metrics.accuracy

            # 総合改善スコア
            total_improvement = 0.5 * latency_improvement + 0.5 * accuracy_improvement

            return total_improvement > self.config.model_switch_threshold

    def get_model_rankings(self) -> List[Tuple[str, float]]:
        """モデルランキング取得"""
        with self.lock:
            rankings = []

            for model_id, metrics in self.model_metrics.items():
                # 複合スコア計算
                if metrics.avg_latency_ms > 0:
                    score = metrics.accuracy * (1000.0 / metrics.avg_latency_ms) * metrics.throughput_per_second
                else:
                    score = 0.0

                rankings.append((model_id, score))

            return sorted(rankings, key=lambda x: x[1], reverse=True)


class InferenceCache:
    """推論結果キャッシュ"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache: Dict[str, InferenceCacheEntry] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()

        # クリーンアップタスク開始
        self._start_cleanup_task()

    def _compute_input_hash(self, input_data: np.ndarray, model_id: str) -> str:
        """入力データハッシュ計算"""
        # 入力データとモデルIDを組み合わせてハッシュ化
        data_bytes = input_data.tobytes()
        combined = data_bytes + model_id.encode('utf-8')
        return hashlib.sha256(combined).hexdigest()

    def _compute_similarity(self, array1: np.ndarray, array2: np.ndarray) -> float:
        """配列類似度計算"""
        try:
            # コサイン類似度計算
            dot_product = np.dot(array1.flatten(), array2.flatten())
            norm1 = np.linalg.norm(array1.flatten())
            norm2 = np.linalg.norm(array2.flatten())

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # 0以上の値のみ

        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0

    def get_cached_result(self, input_data: np.ndarray, model_id: str) -> Optional[Tuple[np.ndarray, float]]:
        """キャッシュ結果取得"""
        if not self.config.enable_inference_caching:
            return None

        with self.lock:
            input_hash = self._compute_input_hash(input_data, model_id)

            # 完全一致チェック
            if input_hash in self.cache:
                entry = self.cache[input_hash]
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                self.access_times[input_hash] = time.time()
                logger.debug(f"Cache hit (exact): {input_hash[:8]}")
                return entry.output, entry.confidence

            # 類似度ベースの検索
            best_match = None
            best_similarity = 0.0

            for cached_hash, entry in self.cache.items():
                if entry.model_id != model_id:
                    continue

                # 元の入力データを復元（メタデータから）
                # 実際の実装では、入力データも保存する必要があります
                # ここでは簡略化のため、ハッシュベースの完全一致のみ実装
                pass

            # 類似度が閾値を超える場合はキャッシュヒット
            if best_similarity > self.config.cache_similarity_threshold and best_match:
                entry = self.cache[best_match]
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                self.access_times[best_match] = time.time()
                logger.debug(f"Cache hit (similar): {best_match[:8]} (similarity: {best_similarity:.3f})")
                return entry.output, entry.confidence * best_similarity

            return None

    def store_result(self, input_data: np.ndarray, output: np.ndarray, model_id: str, confidence: float) -> None:
        """結果キャッシュ保存"""
        if not self.config.enable_inference_caching:
            return

        with self.lock:
            input_hash = self._compute_input_hash(input_data, model_id)

            # キャッシュサイズ制限チェック
            if len(self.cache) >= self.config.max_cache_entries:
                self._evict_oldest_entries()

            # 新しいエントリ作成
            entry = InferenceCacheEntry(
                input_hash=input_hash,
                output=output.copy(),
                model_id=model_id,
                confidence=confidence,
                created_at=datetime.now()
            )

            self.cache[input_hash] = entry
            self.access_times[input_hash] = time.time()

            logger.debug(f"Cache stored: {input_hash[:8]}")

    def _evict_oldest_entries(self) -> None:
        """古いエントリを削除"""
        # アクセス時間順でソートして古いものから削除
        sorted_entries = sorted(self.access_times.items(), key=lambda x: x[1])

        # 10%のエントリを削除
        num_to_evict = max(1, len(sorted_entries) // 10)

        for hash_key, _ in sorted_entries[:num_to_evict]:
            if hash_key in self.cache:
                del self.cache[hash_key]
            if hash_key in self.access_times:
                del self.access_times[hash_key]

        logger.debug(f"Evicted {num_to_evict} cache entries")

    def _start_cleanup_task(self) -> None:
        """定期クリーンアップタスク開始"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(60)  # 1分ごと
                    self._cleanup_expired_entries()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")

        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()

    def _cleanup_expired_entries(self) -> None:
        """期限切れエントリクリーンアップ"""
        with self.lock:
            current_time = datetime.now()
            expired_keys = []

            for hash_key, entry in self.cache.items():
                age = (current_time - entry.created_at).total_seconds()
                if age > self.config.cache_ttl_seconds:
                    expired_keys.append(hash_key)

            for key in expired_keys:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計取得"""
        with self.lock:
            total_accesses = sum(entry.access_count for entry in self.cache.values())

            return {
                "cache_size": len(self.cache),
                "total_accesses": total_accesses,
                "hit_rate": total_accesses / max(1, len(self.cache)),
                "memory_usage_mb": len(self.cache) * 0.1  # 推定値
            }


class PerformanceProfiler:
    """パフォーマンスプロファイラー"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.profiling_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.system_metrics: deque = deque(maxlen=config.profiling_window_size)
        self.lock = threading.RLock()

        # システムメトリクス収集開始
        self._start_system_monitoring()

    def profile_inference(self, model_id: str, input_data: np.ndarray, inference_func: Callable) -> Tuple[Any, Dict[str, float]]:
        """推論プロファイリング"""
        if not self.config.enable_profiling:
            result = inference_func()
            return result, {}

        # サンプリング判定
        if np.random.random() > self.config.profiling_sample_rate:
            result = inference_func()
            return result, {}

        # プロファイリング開始
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

        try:
            # 推論実行
            result = inference_func()

            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

            # メトリクス計算
            metrics = {
                "latency_ms": (end_time - start_time) * 1000,
                "memory_delta_mb": end_memory - start_memory,
                "input_shape": input_data.shape,
                "input_size_mb": input_data.nbytes / (1024 * 1024),
                "timestamp": time.time()
            }

            # プロファイリングデータ保存
            with self.lock:
                self.profiling_data[model_id].append(metrics)

                # データサイズ制限
                if len(self.profiling_data[model_id]) > self.config.profiling_window_size:
                    self.profiling_data[model_id] = self.profiling_data[model_id][-self.config.profiling_window_size:]

            return result, metrics

        except Exception as e:
            # エラーの場合もプロファイリングデータ記録
            error_metrics = {
                "latency_ms": (time.perf_counter() - start_time) * 1000,
                "error": str(e),
                "timestamp": time.time()
            }

            with self.lock:
                self.profiling_data[model_id].append(error_metrics)

            raise

    def _start_system_monitoring(self) -> None:
        """システム監視開始"""
        def monitor_loop():
            while True:
                try:
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()

                    metrics = {
                        "timestamp": time.time(),
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_available_gb": memory.available / (1024**3)
                    }

                    with self.lock:
                        self.system_metrics.append(metrics)

                    time.sleep(5)  # 5秒間隔

                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                    time.sleep(5)

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def analyze_performance_trends(self, model_id: str) -> Dict[str, Any]:
        """パフォーマンス傾向分析"""
        with self.lock:
            if model_id not in self.profiling_data or not self.profiling_data[model_id]:
                return {}

            data = self.profiling_data[model_id]
            latencies = [d.get("latency_ms", 0) for d in data if "latency_ms" in d]
            memory_deltas = [d.get("memory_delta_mb", 0) for d in data if "memory_delta_mb" in d]

            if not latencies:
                return {}

            analysis = {
                "sample_count": len(latencies),
                "avg_latency_ms": np.mean(latencies),
                "median_latency_ms": np.median(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "latency_std": np.std(latencies),
                "avg_memory_delta_mb": np.mean(memory_deltas) if memory_deltas else 0,
                "error_rate": len([d for d in data if "error" in d]) / len(data)
            }

            # 傾向分析（線形回帰）
            if len(latencies) > 10:
                try:
                    timestamps = np.array([d.get("timestamp", 0) for d in data if "latency_ms" in d])
                    latency_array = np.array(latencies)

                    # 線形回帰でトレンド計算
                    coeffs = np.polyfit(timestamps, latency_array, 1)
                    analysis["latency_trend_slope"] = coeffs[0] * 3600  # per hour

                except Exception as e:
                    logger.error(f"Trend analysis failed: {e}")
                    analysis["latency_trend_slope"] = 0.0

            return analysis

    def get_system_health(self) -> Dict[str, Any]:
        """システム健全性取得"""
        with self.lock:
            if not self.system_metrics:
                return {}

            recent_metrics = list(self.system_metrics)[-60:]  # 直近5分

            cpu_values = [m["cpu_percent"] for m in recent_metrics]
            memory_values = [m["memory_percent"] for m in recent_metrics]

            return {
                "avg_cpu_percent": np.mean(cpu_values),
                "max_cpu_percent": np.max(cpu_values),
                "avg_memory_percent": np.mean(memory_values),
                "max_memory_percent": np.max(memory_values),
                "sample_count": len(recent_metrics)
            }


class ABTester:
    """A/Bテストシステム"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.test_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.lock = threading.RLock()

    def start_ab_test(self, test_name: str, model_a: str, model_b: str, metrics_to_compare: List[str]) -> bool:
        """A/Bテスト開始"""
        if not self.config.enable_ab_testing:
            return False

        with self.lock:
            if test_name in self.active_tests:
                logger.warning(f"A/B test {test_name} already active")
                return False

            self.active_tests[test_name] = {
                "model_a": model_a,
                "model_b": model_b,
                "metrics_to_compare": metrics_to_compare,
                "start_time": datetime.now(),
                "duration_hours": self.config.ab_test_duration_hours,
                "split_ratio": self.config.ab_test_split_ratio,
                "group_a_count": 0,
                "group_b_count": 0
            }

            logger.info(f"Started A/B test: {test_name} ({model_a} vs {model_b})")
            return True

    def assign_test_group(self, test_name: str, user_id: str) -> Optional[str]:
        """テストグループ割り当て"""
        with self.lock:
            if test_name not in self.active_tests:
                return None

            test_config = self.active_tests[test_name]

            # テスト期間チェック
            elapsed = datetime.now() - test_config["start_time"]
            if elapsed.total_seconds() > test_config["duration_hours"] * 3600:
                return None

            # ハッシュベースの安定した割り当て
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            is_group_b = (hash_value % 100) < (test_config["split_ratio"] * 100)

            if is_group_b:
                test_config["group_b_count"] += 1
                return test_config["model_b"]
            else:
                test_config["group_a_count"] += 1
                return test_config["model_a"]

    def record_test_result(self, test_name: str, model_id: str, metrics: Dict[str, float]) -> None:
        """テスト結果記録"""
        with self.lock:
            if test_name not in self.active_tests:
                return

            result = {
                "timestamp": datetime.now(),
                "model_id": model_id,
                "metrics": metrics.copy()
            }

            self.test_results[test_name].append(result)

    def analyze_test_results(self, test_name: str) -> Dict[str, Any]:
        """テスト結果分析"""
        with self.lock:
            if test_name not in self.active_tests or test_name not in self.test_results:
                return {}

            test_config = self.active_tests[test_name]
            results = self.test_results[test_name]

            model_a = test_config["model_a"]
            model_b = test_config["model_b"]

            # グループ別データ分離
            group_a_results = [r for r in results if r["model_id"] == model_a]
            group_b_results = [r for r in results if r["model_id"] == model_b]

            if not group_a_results or not group_b_results:
                return {"status": "insufficient_data"}

            analysis = {
                "test_name": test_name,
                "group_a_size": len(group_a_results),
                "group_b_size": len(group_b_results),
                "metrics_comparison": {}
            }

            # メトリクス比較
            for metric in test_config["metrics_to_compare"]:
                group_a_values = [r["metrics"].get(metric, 0) for r in group_a_results if metric in r["metrics"]]
                group_b_values = [r["metrics"].get(metric, 0) for r in group_b_results if metric in r["metrics"]]

                if group_a_values and group_b_values:
                    a_mean = np.mean(group_a_values)
                    b_mean = np.mean(group_b_values)

                    # 統計的有意性テスト（簡易版）
                    if SKLEARN_AVAILABLE:
                        try:
                            from scipy import stats
                            _, p_value = stats.ttest_ind(group_a_values, group_b_values)
                            is_significant = p_value < 0.05
                        except ImportError:
                            p_value = 0.5
                            is_significant = False
                    else:
                        p_value = 0.5
                        is_significant = abs(a_mean - b_mean) > 0.1 * max(a_mean, b_mean)

                    analysis["metrics_comparison"][metric] = {
                        "group_a_mean": a_mean,
                        "group_b_mean": b_mean,
                        "improvement": (b_mean - a_mean) / a_mean if a_mean > 0 else 0,
                        "p_value": p_value,
                        "is_significant": is_significant,
                        "winner": "B" if b_mean > a_mean and is_significant else "A" if a_mean > b_mean and is_significant else "tie"
                    }

            return analysis

    def stop_test(self, test_name: str) -> Dict[str, Any]:
        """テスト停止と最終分析"""
        with self.lock:
            if test_name not in self.active_tests:
                return {}

            final_analysis = self.analyze_test_results(test_name)

            # テスト停止
            del self.active_tests[test_name]

            logger.info(f"Stopped A/B test: {test_name}")
            return final_analysis


class AdvancedOptimizer:
    """高度な最適化システム"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.model_selector = DynamicModelSelector(config)
        self.inference_cache = InferenceCache(config)
        self.profiler = PerformanceProfiler(config)
        self.ab_tester = ABTester(config)

        # 自動最適化
        self.auto_optimization_enabled = config.enable_auto_optimization
        self.last_optimization = datetime.now()

        if self.auto_optimization_enabled:
            self._start_auto_optimization()

    async def optimized_inference(self,
                                 input_data: np.ndarray,
                                 available_models: List[str],
                                 inference_func: Callable[[str], Any],
                                 user_id: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """最適化された推論実行"""

        # A/Bテスト対象モデル選択
        selected_model = None
        if user_id:
            for test_name in self.ab_tester.active_tests:
                test_model = self.ab_tester.assign_test_group(test_name, user_id)
                if test_model and test_model in available_models:
                    selected_model = test_model
                    break

        # 動的モデル選択（A/Bテストで選択されなかった場合）
        if not selected_model:
            selected_model = self.model_selector.select_optimal_model(available_models)

        # キャッシュチェック
        cached_result = self.inference_cache.get_cached_result(input_data, selected_model)
        if cached_result:
            output, confidence = cached_result
            return output, {
                "model_used": selected_model,
                "cache_hit": True,
                "confidence": confidence,
                "latency_ms": 0.1  # キャッシュヒットは超高速
            }

        # プロファイリング付き推論実行
        def inference_wrapper():
            return inference_func(selected_model)

        start_time = time.perf_counter()
        result, profiling_metrics = self.profiler.profile_inference(
            selected_model, input_data, inference_wrapper
        )
        total_latency = (time.perf_counter() - start_time) * 1000

        # 結果をキャッシュ
        if hasattr(result, 'confidence'):
            confidence = result.confidence
        else:
            confidence = 0.95  # デフォルト信頼度

        if isinstance(result, np.ndarray):
            self.inference_cache.store_result(input_data, result, selected_model, confidence)

        # モデル性能メトリクス更新
        accuracy = confidence  # 簡略化
        throughput = 1000.0 / total_latency if total_latency > 0 else 0
        self.model_selector.update_model_metrics(selected_model, total_latency, accuracy, throughput)

        # A/Bテスト結果記録
        if user_id:
            for test_name in self.ab_tester.active_tests:
                test_config = self.ab_tester.active_tests[test_name]
                if selected_model in [test_config["model_a"], test_config["model_b"]]:
                    metrics = {
                        "latency_ms": total_latency,
                        "accuracy": accuracy,
                        "throughput": throughput
                    }
                    self.ab_tester.record_test_result(test_name, selected_model, metrics)

        return result, {
            "model_used": selected_model,
            "cache_hit": False,
            "latency_ms": total_latency,
            "confidence": confidence,
            "profiling_metrics": profiling_metrics
        }

    def _start_auto_optimization(self) -> None:
        """自動最適化開始"""
        def optimization_loop():
            while self.auto_optimization_enabled:
                try:
                    time.sleep(3600)  # 1時間間隔

                    # 最適化間隔チェック
                    if (datetime.now() - self.last_optimization).total_seconds() > self.config.optimization_interval_hours * 3600:
                        self._perform_auto_optimization()
                        self.last_optimization = datetime.now()

                except Exception as e:
                    logger.error(f"Auto optimization error: {e}")

        optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        optimization_thread.start()
        logger.info("Started auto optimization")

    def _perform_auto_optimization(self) -> None:
        """自動最適化実行"""
        logger.info("Performing automatic optimization")

        # パフォーマンス分析
        model_rankings = self.model_selector.get_model_rankings()

        if len(model_rankings) > 1:
            best_model, best_score = model_rankings[0]
            second_model, second_score = model_rankings[1]

            # 大きな性能差がある場合はA/Bテスト開始
            if best_score > second_score * 1.2:  # 20%以上の差
                test_name = f"auto_test_{int(time.time())}"
                self.ab_tester.start_ab_test(
                    test_name,
                    second_model,
                    best_model,
                    ["latency_ms", "accuracy", "throughput"]
                )

        # システム健全性チェック
        system_health = self.profiler.get_system_health()
        if system_health.get("avg_cpu_percent", 0) > 80:
            logger.warning("High CPU usage detected during auto optimization")

        # キャッシュクリーンアップ
        cache_stats = self.inference_cache.get_cache_stats()
        if cache_stats.get("cache_size", 0) > self.config.max_cache_entries * 0.8:
            logger.info("Cache cleanup triggered by auto optimization")

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """包括的レポート生成"""
        return {
            "model_rankings": self.model_selector.get_model_rankings(),
            "cache_stats": self.inference_cache.get_cache_stats(),
            "system_health": self.profiler.get_system_health(),
            "active_ab_tests": list(self.ab_tester.active_tests.keys()),
            "optimization_config": asdict(self.config),
            "last_optimization": self.last_optimization.isoformat()
        }

    def shutdown(self) -> None:
        """システム終了"""
        self.auto_optimization_enabled = False
        logger.info("Advanced optimizer shutdown")


# 使用例とテスト
async def test_advanced_optimizer():
    """高度最適化テスト"""

    # 設定
    config = OptimizationConfig(
        enable_dynamic_model_selection=True,
        enable_inference_caching=True,
        enable_profiling=True,
        enable_ab_testing=True
    )

    # 最適化システム初期化
    optimizer = AdvancedOptimizer(config)

    try:
        print("=== Advanced Optimization Test ===")

        # テスト用推論関数
        def mock_inference(model_id: str) -> np.ndarray:
            # モデルごとに異なる特性をシミュレート
            if "fast" in model_id:
                time.sleep(0.001)  # 高速
                return np.random.randn(1, 5).astype(np.float32)
            elif "accurate" in model_id:
                time.sleep(0.01)   # 低速だが高精度
                return np.random.randn(1, 10).astype(np.float32)
            else:
                time.sleep(0.005)  # 中間
                return np.random.randn(1, 8).astype(np.float32)

        # テストモデル
        available_models = ["fast_model", "accurate_model", "balanced_model"]

        # 複数回推論実行
        print(f"\n1. Multiple Inference Test")
        for i in range(20):
            input_data = np.random.randn(1, 20).astype(np.float32)
            user_id = f"user_{i % 5}"  # 5人のユーザーを循環

            result, metadata = await optimizer.optimized_inference(
                input_data, available_models, mock_inference, user_id
            )

            print(f"  Inference {i+1}: Model {metadata['model_used']}, "
                  f"Latency {metadata['latency_ms']:.2f}ms, "
                  f"Cache hit: {metadata['cache_hit']}")

        # A/Bテスト開始
        print(f"\n2. A/B Test")
        optimizer.ab_tester.start_ab_test(
            "speed_vs_accuracy",
            "fast_model",
            "accurate_model",
            ["latency_ms", "accuracy"]
        )

        # A/Bテスト用推論
        for i in range(10):
            input_data = np.random.randn(1, 20).astype(np.float32)
            user_id = f"test_user_{i}"

            result, metadata = await optimizer.optimized_inference(
                input_data, available_models, mock_inference, user_id
            )

        # A/Bテスト結果分析
        ab_results = optimizer.ab_tester.analyze_test_results("speed_vs_accuracy")
        print(f"  A/B Test Results: {ab_results.get('metrics_comparison', {})}")

        # パフォーマンス分析
        print(f"\n3. Performance Analysis")
        for model_id in available_models:
            analysis = optimizer.profiler.analyze_performance_trends(model_id)
            if analysis:
                print(f"  {model_id}: Avg latency {analysis.get('avg_latency_ms', 0):.2f}ms, "
                      f"P95 {analysis.get('p95_latency_ms', 0):.2f}ms")

        # 包括的レポート
        print(f"\n4. Comprehensive Report")
        report = optimizer.get_comprehensive_report()

        print(f"  Model Rankings:")
        for i, (model, score) in enumerate(report["model_rankings"]):
            print(f"    {i+1}. {model}: {score:.3f}")

        print(f"  Cache Stats: {report['cache_stats']}")
        print(f"  System Health: {report['system_health']}")

    except Exception as e:
        print(f"Advanced optimization test failed: {e}")

    finally:
        optimizer.shutdown()


if __name__ == "__main__":
    asyncio.run(test_advanced_optimizer())