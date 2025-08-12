#!/usr/bin/env python3
"""
適応的キャッシュ戦略システム
Issue #377: 高度キャッシング戦略の追加最適化

機械学習ベースの動的キャッシュ最適化とパフォーマンス予測を実装
"""

import asyncio
import json
import math
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class CachePattern(Enum):
    """キャッシュアクセスパターン"""

    HOT = "hot"  # 頻繁アクセス
    COLD = "cold"  # 稀なアクセス
    BURST = "burst"  # バーストアクセス
    PERIODIC = "periodic"  # 周期的アクセス
    DECLINING = "declining"  # アクセス減少傾向
    GROWING = "growing"  # アクセス増加傾向


class OptimizationStrategy(Enum):
    """最適化戦略"""

    LRU_ENHANCED = "lru_enhanced"  # 拡張LRU
    FREQUENCY_BASED = "frequency_based"  # 頻度ベース
    PREDICTIVE = "predictive"  # 予測ベース
    ADAPTIVE_TTL = "adaptive_ttl"  # 動的TTL
    WORKLOAD_AWARE = "workload_aware"  # ワークロード対応


@dataclass
class CacheAccessMetrics:
    """キャッシュアクセス メトリクス"""

    key: str
    access_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    last_access_time: float = 0
    creation_time: float = 0
    access_times: List[float] = None
    value_size_bytes: int = 0
    access_pattern: CachePattern = CachePattern.COLD

    def __post_init__(self):
        if self.access_times is None:
            self.access_times = []
        if self.creation_time == 0:
            self.creation_time = time.time()

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / max(total, 1)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.creation_time

    @property
    def last_access_age_seconds(self) -> float:
        if self.last_access_time == 0:
            return self.age_seconds
        return time.time() - self.last_access_time

    @property
    def access_frequency_per_hour(self) -> float:
        if self.age_seconds < 1:
            return self.access_count
        return self.access_count / (self.age_seconds / 3600)

    def record_access(self, is_hit: bool = True):
        """アクセス記録"""
        current_time = time.time()

        self.access_count += 1
        self.last_access_time = current_time

        if is_hit:
            self.hit_count += 1
        else:
            self.miss_count += 1

        # 最新のアクセス時刻を記録（最大100件）
        self.access_times.append(current_time)
        if len(self.access_times) > 100:
            self.access_times = self.access_times[-100:]

        # アクセスパターンを更新
        self._update_access_pattern()

    def _update_access_pattern(self):
        """アクセスパターンの更新"""
        if len(self.access_times) < 5:
            self.access_pattern = CachePattern.COLD
            return

        recent_times = self.access_times[-20:]  # 最新20アクセス
        intervals = []

        for i in range(1, len(recent_times)):
            intervals.append(recent_times[i] - recent_times[i - 1])

        if not intervals:
            return

        avg_interval = statistics.mean(intervals)
        interval_variance = statistics.variance(intervals) if len(intervals) > 1 else 0

        # パターン判定
        if avg_interval < 5:  # 5秒以内
            if interval_variance < 1:
                self.access_pattern = CachePattern.HOT
            else:
                self.access_pattern = CachePattern.BURST
        elif avg_interval < 60:  # 1分以内
            if interval_variance < 10:
                self.access_pattern = CachePattern.PERIODIC
            else:
                self.access_pattern = CachePattern.HOT
        else:
            # 長期トレンド分析
            if len(self.access_times) >= 10:
                recent_rate = len(self.access_times[-5:]) / max(
                    self.access_times[-1] - self.access_times[-5], 1
                )
                old_rate = len(self.access_times[-10:-5]) / max(
                    self.access_times[-5] - self.access_times[-10], 1
                )

                if recent_rate > old_rate * 1.5:
                    self.access_pattern = CachePattern.GROWING
                elif recent_rate < old_rate * 0.5:
                    self.access_pattern = CachePattern.DECLINING
                else:
                    self.access_pattern = CachePattern.COLD


class MLCachePredictor:
    """機械学習ベース キャッシュ予測器"""

    def __init__(self, enable_ml: bool = None):
        self.enable_ml = enable_ml if enable_ml is not None else SKLEARN_AVAILABLE

        if self.enable_ml and not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. ML predictions disabled.")
            self.enable_ml = False

        # 特徴量とターゲットデータ
        self.feature_history = deque(maxlen=1000)
        self.target_history = deque(maxlen=1000)

        # 訓練済みモデル
        self.access_predictor = None
        self.ttl_predictor = None

        # モデル更新タイミング
        self.last_training_time = 0
        self.training_interval = 300  # 5分
        self.min_samples_for_training = 50

        logger.info(f"ML キャッシュ予測器初期化: enable_ml={self.enable_ml}")

    def record_cache_event(
        self, metrics: CacheAccessMetrics, was_accessed: bool, optimal_ttl: float = None
    ):
        """キャッシュイベント記録（ML学習データ）"""
        if not self.enable_ml:
            return

        try:
            # 特徴量生成
            features = self._extract_features(metrics)

            # アクセス予測用データ
            self.feature_history.append(features)
            self.target_history.append(1.0 if was_accessed else 0.0)

            # 定期的モデル更新
            if (
                time.time() - self.last_training_time > self.training_interval
                and len(self.feature_history) >= self.min_samples_for_training
            ):
                asyncio.create_task(self._train_models_async())

        except Exception as e:
            logger.error(f"ML データ記録エラー: {e}")

    def predict_access_probability(self, metrics: CacheAccessMetrics) -> float:
        """アクセス確率予測"""
        if not self.enable_ml or self.access_predictor is None:
            return self._fallback_access_prediction(metrics)

        try:
            features = self._extract_features(metrics)
            features_array = np.array(features).reshape(1, -1)

            if hasattr(self.access_predictor, "predict_proba"):
                probabilities = self.access_predictor.predict_proba(features_array)
                return probabilities[0][1]  # クラス1（アクセスあり）の確率
            else:
                prediction = self.access_predictor.predict(features_array)
                return max(0.0, min(1.0, prediction[0]))

        except Exception as e:
            logger.error(f"アクセス確率予測エラー: {e}")
            return self._fallback_access_prediction(metrics)

    def predict_optimal_ttl(self, metrics: CacheAccessMetrics) -> int:
        """最適TTL予測"""
        if not self.enable_ml or self.ttl_predictor is None:
            return self._fallback_ttl_prediction(metrics)

        try:
            features = self._extract_features(metrics)
            features_array = np.array(features).reshape(1, -1)

            predicted_ttl = self.ttl_predictor.predict(features_array)[0]

            # TTL範囲制限
            return max(30, min(86400, int(predicted_ttl)))

        except Exception as e:
            logger.error(f"TTL予測エラー: {e}")
            return self._fallback_ttl_prediction(metrics)

    def _extract_features(self, metrics: CacheAccessMetrics) -> List[float]:
        """特徴量抽出"""
        features = [
            metrics.access_count,
            metrics.hit_rate,
            metrics.age_seconds,
            metrics.last_access_age_seconds,
            metrics.access_frequency_per_hour,
            metrics.value_size_bytes / 1024,  # KB単位
            len(metrics.access_times),
        ]

        # アクセスパターンをワンホットエンコーディング
        pattern_features = [0.0] * len(CachePattern)
        pattern_features[list(CachePattern).index(metrics.access_pattern)] = 1.0
        features.extend(pattern_features)

        # 時間的特徴量
        current_hour = time.localtime().tm_hour
        features.extend(
            [
                math.sin(2 * math.pi * current_hour / 24),  # 時間の周期性
                math.cos(2 * math.pi * current_hour / 24),
            ]
        )

        return features

    async def _train_models_async(self):
        """非同期モデル訓練"""
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._train_models)
        except Exception as e:
            logger.error(f"非同期モデル訓練エラー: {e}")

    def _train_models(self):
        """モデル訓練"""
        if not self.enable_ml or len(self.feature_history) < self.min_samples_for_training:
            return

        try:
            # データ準備
            X = np.array(list(self.feature_history))
            y_access = np.array(list(self.target_history))

            # アクセス予測モデル訓練
            if SKLEARN_AVAILABLE:
                self.access_predictor = RandomForestRegressor(
                    n_estimators=20, max_depth=10, random_state=42, n_jobs=1
                )
                self.access_predictor.fit(X, y_access)

                logger.debug(f"アクセス予測モデル訓練完了: {len(X)}サンプル")

            self.last_training_time = time.time()

        except Exception as e:
            logger.error(f"モデル訓練エラー: {e}")

    def _fallback_access_prediction(self, metrics: CacheAccessMetrics) -> float:
        """フォールバック アクセス予測"""
        # ヒューリスティックベースの予測
        base_probability = min(metrics.hit_rate, 0.9)

        # アクセスパターンによる調整
        pattern_multiplier = {
            CachePattern.HOT: 1.5,
            CachePattern.BURST: 1.2,
            CachePattern.PERIODIC: 1.0,
            CachePattern.GROWING: 1.3,
            CachePattern.DECLINING: 0.7,
            CachePattern.COLD: 0.5,
        }

        # 最終アクセスからの時間による減衰
        time_factor = math.exp(-metrics.last_access_age_seconds / 3600)

        probability = base_probability * pattern_multiplier[metrics.access_pattern] * time_factor

        return max(0.0, min(1.0, probability))

    def _fallback_ttl_prediction(self, metrics: CacheAccessMetrics) -> int:
        """フォールバック TTL予測"""
        # アクセスパターンベースのTTL
        base_ttl = {
            CachePattern.HOT: 300,  # 5分
            CachePattern.BURST: 600,  # 10分
            CachePattern.PERIODIC: 1800,  # 30分
            CachePattern.GROWING: 900,  # 15分
            CachePattern.DECLINING: 1800,  # 30分
            CachePattern.COLD: 3600,  # 1時間
        }

        ttl = base_ttl[metrics.access_pattern]

        # アクセス頻度による調整
        if metrics.access_frequency_per_hour > 10:
            ttl = int(ttl * 0.5)  # 高頻度なら短く
        elif metrics.access_frequency_per_hour < 1:
            ttl = int(ttl * 2.0)  # 低頻度なら長く

        return max(60, min(86400, ttl))


class AdaptiveCacheStrategy:
    """適応的キャッシュ戦略"""

    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE_TTL,
        enable_ml_predictions: bool = None,
        metrics_window_size: int = 1000,
    ):
        self.strategy = strategy
        self.metrics_window_size = metrics_window_size

        # メトリクス管理
        self.key_metrics: Dict[str, CacheAccessMetrics] = {}
        self.global_metrics = {
            "total_accesses": 0,
            "total_hits": 0,
            "total_misses": 0,
            "average_access_interval": 60.0,
            "cache_memory_usage": 0,
            "eviction_count": 0,
        }

        # ML予測器
        self.ml_predictor = MLCachePredictor(enable_ml_predictions)

        # 最適化パラメータ
        self.optimization_params = self._initialize_optimization_params()

        # 戦略固有の状態
        self.strategy_state = {}

        # スレッドセーフティ
        self._lock = threading.RLock()

        logger.info(f"適応的キャッシュ戦略初期化: {strategy.value}")

    def _initialize_optimization_params(self) -> Dict[str, Any]:
        """最適化パラメータ初期化"""
        return {
            # LRU拡張パラメータ
            "lru_frequency_weight": 0.3,
            "lru_recency_weight": 0.7,
            # 頻度ベースパラメータ
            "frequency_decay_rate": 0.1,
            "frequency_threshold": 5,
            # 予測ベースパラメータ
            "prediction_confidence_threshold": 0.7,
            "prediction_window_hours": 2,
            # 動的TTLパラメータ
            "ttl_adjustment_factor": 0.2,
            "min_ttl": 30,
            "max_ttl": 86400,
            # ワークロード対応パラメータ
            "workload_adaptation_rate": 0.1,
            "memory_pressure_threshold": 0.8,
        }

    def should_cache_item(
        self,
        key: str,
        value: Any,
        default_ttl: int,
        value_size: int = 0,
        context: Dict[str, Any] = None,
    ) -> Tuple[bool, int]:
        """アイテムをキャッシュすべきか判定"""
        with self._lock:
            try:
                metrics = self._get_or_create_metrics(key)

                if value_size > 0:
                    metrics.value_size_bytes = value_size

                # 戦略別判定
                if self.strategy == OptimizationStrategy.PREDICTIVE:
                    return self._predictive_caching_decision(metrics, default_ttl, context)

                elif self.strategy == OptimizationStrategy.FREQUENCY_BASED:
                    return self._frequency_based_decision(metrics, default_ttl, context)

                elif self.strategy == OptimizationStrategy.ADAPTIVE_TTL:
                    return self._adaptive_ttl_decision(metrics, default_ttl, context)

                elif self.strategy == OptimizationStrategy.WORKLOAD_AWARE:
                    return self._workload_aware_decision(metrics, default_ttl, context)

                else:  # LRU_ENHANCED
                    return self._lru_enhanced_decision(metrics, default_ttl, context)

            except Exception as e:
                logger.error(f"キャッシュ判定エラー {key}: {e}")
                return True, default_ttl  # エラー時はデフォルト動作

    def should_evict_item(
        self,
        key: str,
        current_cache_size: int,
        memory_pressure: float = 0.0,
    ) -> bool:
        """アイテムを追い出すべきか判定"""
        with self._lock:
            try:
                metrics = self._get_or_create_metrics(key)

                # 基本的な追い出し判定
                if metrics.access_pattern == CachePattern.COLD:
                    return True

                if metrics.last_access_age_seconds > 3600:  # 1時間未アクセス
                    return True

                # メモリ圧迫時の高度な判定
                if memory_pressure > self.optimization_params["memory_pressure_threshold"]:
                    # アクセス確率予測
                    access_probability = self.ml_predictor.predict_access_probability(metrics)

                    # 価値スコア計算
                    value_score = self._calculate_cache_value_score(metrics, access_probability)

                    # 閾値による判定
                    eviction_threshold = 0.3 - (memory_pressure - 0.8) * 2  # メモリ圧迫で閾値低下

                    return value_score < eviction_threshold

                return False

            except Exception as e:
                logger.error(f"追い出し判定エラー {key}: {e}")
                return False

    def record_cache_access(self, key: str, is_hit: bool, access_context: Dict[str, Any] = None):
        """キャッシュアクセス記録"""
        with self._lock:
            try:
                metrics = self._get_or_create_metrics(key)
                metrics.record_access(is_hit)

                # グローバル統計更新
                self.global_metrics["total_accesses"] += 1
                if is_hit:
                    self.global_metrics["total_hits"] += 1
                else:
                    self.global_metrics["total_misses"] += 1

                # ML学習データとして記録
                self.ml_predictor.record_cache_event(metrics, was_accessed=True)

                # 最適化パラメータの動的調整
                if self.global_metrics["total_accesses"] % 100 == 0:
                    self._adjust_optimization_parameters()

            except Exception as e:
                logger.error(f"アクセス記録エラー {key}: {e}")

    def get_optimization_suggestions(self) -> Dict[str, Any]:
        """最適化提案生成"""
        with self._lock:
            try:
                suggestions = {
                    "strategy": self.strategy.value,
                    "recommendations": [],
                    "metrics_summary": self._generate_metrics_summary(),
                    "performance_insights": [],
                }

                # 全体的なヒット率チェック
                total_requests = (
                    self.global_metrics["total_hits"] + self.global_metrics["total_misses"]
                )
                hit_rate = self.global_metrics["total_hits"] / max(total_requests, 1)

                if hit_rate < 0.5:
                    suggestions["recommendations"].append(
                        {
                            "type": "hit_rate_improvement",
                            "message": f"ヒット率が低い ({hit_rate:.1%}) - TTL延長またはキャッシュサイズ拡大を検討",
                            "priority": "high",
                        }
                    )

                # アクセスパターン分析
                pattern_distribution = self._analyze_access_patterns()

                if pattern_distribution.get(CachePattern.COLD, 0) > 0.4:
                    suggestions["recommendations"].append(
                        {
                            "type": "cold_data_cleanup",
                            "message": "40%以上のデータが稀にしかアクセスされていません - 定期クリーンアップを推奨",
                            "priority": "medium",
                        }
                    )

                # ML予測精度
                if self.ml_predictor.enable_ml:
                    suggestions["performance_insights"].append(
                        {
                            "type": "ml_prediction_status",
                            "message": f"ML予測が有効です。学習データ: {len(self.ml_predictor.feature_history)}件",
                            "status": "active",
                        }
                    )

                # メモリ使用量最適化
                if self.global_metrics["cache_memory_usage"] > 0:
                    memory_efficiency = (
                        hit_rate * 100 / (self.global_metrics["cache_memory_usage"] / (1024**2))
                    )

                    if memory_efficiency < 10:  # ヒット率%/MB
                        suggestions["recommendations"].append(
                            {
                                "type": "memory_efficiency",
                                "message": f"メモリ効率が低い ({memory_efficiency:.1f} hits%/MB) - 大きなオブジェクトの見直しを推奨",
                                "priority": "medium",
                            }
                        )

                return suggestions

            except Exception as e:
                logger.error(f"最適化提案生成エラー: {e}")
                return {"error": str(e)}

    def _get_or_create_metrics(self, key: str) -> CacheAccessMetrics:
        """メトリクス取得または作成"""
        if key not in self.key_metrics:
            self.key_metrics[key] = CacheAccessMetrics(key=key)

            # メトリクス数制限
            if len(self.key_metrics) > self.metrics_window_size:
                # 最も古いメトリクスを削除
                oldest_key = min(
                    self.key_metrics.keys(),
                    key=lambda k: self.key_metrics[k].last_access_time,
                )
                del self.key_metrics[oldest_key]

        return self.key_metrics[key]

    def _predictive_caching_decision(
        self, metrics: CacheAccessMetrics, default_ttl: int, context: Dict[str, Any]
    ) -> Tuple[bool, int]:
        """予測ベースキャッシング判定"""
        access_probability = self.ml_predictor.predict_access_probability(metrics)

        # 高確率でアクセスされると予測される場合のみキャッシュ
        should_cache = (
            access_probability >= self.optimization_params["prediction_confidence_threshold"]
        )

        if should_cache:
            # 予測TTL使用
            predicted_ttl = self.ml_predictor.predict_optimal_ttl(metrics)
            return True, predicted_ttl

        return False, default_ttl

    def _frequency_based_decision(
        self, metrics: CacheAccessMetrics, default_ttl: int, context: Dict[str, Any]
    ) -> Tuple[bool, int]:
        """頻度ベース判定"""
        frequency_threshold = self.optimization_params["frequency_threshold"]

        # アクセス頻度が閾値以上ならキャッシュ
        should_cache = metrics.access_frequency_per_hour >= frequency_threshold

        if should_cache:
            # 頻度に応じたTTL調整
            frequency_multiplier = min(2.0, metrics.access_frequency_per_hour / frequency_threshold)
            adjusted_ttl = int(default_ttl * frequency_multiplier)

            return True, max(
                self.optimization_params["min_ttl"],
                min(self.optimization_params["max_ttl"], adjusted_ttl),
            )

        return False, default_ttl

    def _adaptive_ttl_decision(
        self, metrics: CacheAccessMetrics, default_ttl: int, context: Dict[str, Any]
    ) -> Tuple[bool, int]:
        """動的TTL判定"""
        # 常にキャッシュするが、TTLを適応的に調整

        # アクセスパターンベースTTL
        pattern_ttl = self.ml_predictor.predict_optimal_ttl(metrics)

        # ヒット率によるTTL調整
        hit_rate_factor = (
            1.0 + (metrics.hit_rate - 0.5) * self.optimization_params["ttl_adjustment_factor"]
        )

        adjusted_ttl = int(pattern_ttl * hit_rate_factor)

        final_ttl = max(
            self.optimization_params["min_ttl"],
            min(self.optimization_params["max_ttl"], adjusted_ttl),
        )

        return True, final_ttl

    def _workload_aware_decision(
        self, metrics: CacheAccessMetrics, default_ttl: int, context: Dict[str, Any]
    ) -> Tuple[bool, int]:
        """ワークロード対応判定"""
        # 現在のワークロード状況を考慮
        memory_pressure = context.get("memory_pressure", 0.0) if context else 0.0

        # 高負荷時は選択的キャッシュ
        if memory_pressure > 0.7:
            access_probability = self.ml_predictor.predict_access_probability(metrics)
            should_cache = access_probability > 0.8  # 高確率のみ
        else:
            should_cache = True

        if should_cache:
            # 負荷に応じたTTL調整
            pressure_factor = 1.0 - memory_pressure * 0.5
            adjusted_ttl = int(default_ttl * pressure_factor)

            return True, max(60, adjusted_ttl)

        return False, default_ttl

    def _lru_enhanced_decision(
        self, metrics: CacheAccessMetrics, default_ttl: int, context: Dict[str, Any]
    ) -> Tuple[bool, int]:
        """拡張LRU判定"""
        # LRU + 頻度を組み合わせた判定
        recency_score = 1.0 / max(1.0, metrics.last_access_age_seconds / 3600)
        frequency_score = min(1.0, metrics.access_frequency_per_hour / 10.0)

        combined_score = (
            recency_score * self.optimization_params["lru_recency_weight"]
            + frequency_score * self.optimization_params["lru_frequency_weight"]
        )

        should_cache = combined_score > 0.3

        return should_cache, default_ttl

    def _calculate_cache_value_score(
        self, metrics: CacheAccessMetrics, access_probability: float
    ) -> float:
        """キャッシュ価値スコア計算"""
        # 複数の要因を組み合わせた価値スコア

        # アクセス確率重み
        probability_score = access_probability

        # ヒット率重み
        hit_rate_score = metrics.hit_rate

        # 頻度重み（対数スケール）
        frequency_score = min(1.0, math.log10(max(1, metrics.access_frequency_per_hour)) / 2.0)

        # サイズコスト（大きなオブジェクトは価値減少）
        size_cost = max(0.1, 1.0 - metrics.value_size_bytes / (1024 * 1024))  # 1MB基準

        # 総合スコア
        value_score = (
            probability_score * 0.4 + hit_rate_score * 0.3 + frequency_score * 0.2 + size_cost * 0.1
        )

        return value_score

    def _adjust_optimization_parameters(self):
        """最適化パラメータの動的調整"""
        try:
            hit_rate = self._calculate_global_hit_rate()

            # ヒット率が低い場合はより積極的にキャッシュ
            if hit_rate < 0.5:
                self.optimization_params["prediction_confidence_threshold"] *= 0.95
                self.optimization_params["frequency_threshold"] *= 0.9

            # ヒット率が高い場合はより選択的に
            elif hit_rate > 0.8:
                self.optimization_params["prediction_confidence_threshold"] *= 1.05
                self.optimization_params["frequency_threshold"] *= 1.1

            # 境界値制限
            self.optimization_params["prediction_confidence_threshold"] = max(
                0.3,
                min(0.9, self.optimization_params["prediction_confidence_threshold"]),
            )
            self.optimization_params["frequency_threshold"] = max(
                1.0, min(20.0, self.optimization_params["frequency_threshold"])
            )

            logger.debug("最適化パラメータ動的調整完了")

        except Exception as e:
            logger.error(f"パラメータ調整エラー: {e}")

    def _calculate_global_hit_rate(self) -> float:
        """グローバルヒット率計算"""
        total_requests = self.global_metrics["total_hits"] + self.global_metrics["total_misses"]
        return self.global_metrics["total_hits"] / max(total_requests, 1)

    def _generate_metrics_summary(self) -> Dict[str, Any]:
        """メトリクス要約生成"""
        total_keys = len(self.key_metrics)
        total_requests = self.global_metrics["total_hits"] + self.global_metrics["total_misses"]

        return {
            "total_cached_keys": total_keys,
            "total_requests": total_requests,
            "global_hit_rate": self._calculate_global_hit_rate(),
            "average_key_access_count": sum(m.access_count for m in self.key_metrics.values())
            / max(total_keys, 1),
            "memory_usage_mb": self.global_metrics["cache_memory_usage"] / (1024**2),
            "ml_predictions_enabled": self.ml_predictor.enable_ml,
        }

    def _analyze_access_patterns(self) -> Dict[CachePattern, float]:
        """アクセスパターン分析"""
        if not self.key_metrics:
            return {}

        pattern_counts = defaultdict(int)
        total_keys = len(self.key_metrics)

        for metrics in self.key_metrics.values():
            pattern_counts[metrics.access_pattern] += 1

        return {pattern: count / total_keys for pattern, count in pattern_counts.items()}


if __name__ == "__main__":
    # テスト実行
    async def main():
        print("=== Issue #377 適応的キャッシュ戦略システムテスト ===")

        # 1. 各戦略のテスト
        strategies = [
            OptimizationStrategy.ADAPTIVE_TTL,
            OptimizationStrategy.PREDICTIVE,
            OptimizationStrategy.FREQUENCY_BASED,
        ]

        for strategy in strategies:
            print(f"\n--- {strategy.value} 戦略テスト ---")

            cache_strategy = AdaptiveCacheStrategy(
                strategy=strategy,
                enable_ml_predictions=True,
            )

            # シミュレートされたアクセスパターン
            test_keys = ["hot_data", "cold_data", "burst_data"]
            access_patterns = [20, 2, 10]  # アクセス回数

            for key, access_count in zip(test_keys, access_patterns):
                for i in range(access_count):
                    # キャッシング判定
                    should_cache, ttl = cache_strategy.should_cache_item(
                        key=key,
                        value=f"test_data_{i}",
                        default_ttl=3600,
                        value_size=1024 * (i + 1),
                    )

                    print(f"  {key}: キャッシュ={should_cache}, TTL={ttl}秒")

                    # アクセス記録
                    cache_strategy.record_cache_access(key, is_hit=True)

                    # 短い間隔でシミュレート
                    await asyncio.sleep(0.001)

            # 最適化提案
            suggestions = cache_strategy.get_optimization_suggestions()
            print(f"  最適化提案: {len(suggestions['recommendations'])}件")
            for rec in suggestions["recommendations"]:
                print(f"    - {rec['message']} ({rec['priority']})")

        print("\n2. ML予測テスト")

        if SKLEARN_AVAILABLE:
            predictor = MLCachePredictor(enable_ml=True)

            # テスト用メトリクス
            test_metrics = CacheAccessMetrics(
                key="ml_test",
                access_count=15,
                hit_count=12,
                miss_count=3,
                access_pattern=CachePattern.HOT,
            )

            # いくつかのイベントを記録
            for i in range(20):
                predictor.record_cache_event(test_metrics, was_accessed=i % 3 != 0)

            # 予測実行
            access_prob = predictor.predict_access_probability(test_metrics)
            optimal_ttl = predictor.predict_optimal_ttl(test_metrics)

            print(f"  アクセス確率予測: {access_prob:.2f}")
            print(f"  最適TTL予測: {optimal_ttl}秒")
        else:
            print("  scikit-learn not available - フォールバック予測使用")

        print("\n3. パフォーマンス比較テスト")

        # 異なる戦略でのパフォーマンス比較
        performance_results = {}

        for strategy in [
            OptimizationStrategy.LRU_ENHANCED,
            OptimizationStrategy.ADAPTIVE_TTL,
        ]:
            cache_strategy = AdaptiveCacheStrategy(strategy=strategy)

            start_time = time.perf_counter()

            # 大量のキャッシュ判定処理
            for i in range(1000):
                key = f"perf_test_{i % 100}"
                should_cache, ttl = cache_strategy.should_cache_item(
                    key, f"data_{i}", 3600, value_size=1024
                )
                cache_strategy.record_cache_access(key, is_hit=i % 3 != 0)

            elapsed_time = (time.perf_counter() - start_time) * 1000
            performance_results[strategy.value] = elapsed_time

            print(f"  {strategy.value}: {elapsed_time:.1f}ms")

        print("\n=== 適応的キャッシュ戦略システムテスト完了 ===")

    # 実行
    asyncio.run(main())
