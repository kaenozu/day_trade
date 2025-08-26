#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Monitor - Performance Evaluator
パフォーマンス評価クラス
"""

import asyncio
import logging
import time
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
from collections import defaultdict, deque

from .enums_and_models import (
    PerformanceMetrics,
    PerformanceStatus,
    PerformanceAlert,
    AlertLevel
)
from .config_manager import EnhancedPerformanceConfigManager

logger = logging.getLogger(__name__)

# 機械学習関連
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 精度検証システム
try:
    from prediction_accuracy_validator import PredictionAccuracyValidator
    ACCURACY_VALIDATOR_AVAILABLE = True
except ImportError:
    ACCURACY_VALIDATOR_AVAILABLE = False


class PerformanceEvaluator:
    """パフォーマンス評価クラス"""

    def __init__(self, config_manager: EnhancedPerformanceConfigManager):
        self.config_manager = config_manager
        self.performance_history = defaultdict(deque)
        self.performance_cache = {}
        self.accuracy_validator = None

        # 精度検証システムの初期化
        if ACCURACY_VALIDATOR_AVAILABLE:
            try:
                self.accuracy_validator = PredictionAccuracyValidator()
                logger.info("PredictionAccuracyValidator統合完了")
            except Exception as e:
                logger.warning(f"PredictionAccuracyValidator統合失敗: {e}")
                self.accuracy_validator = None

    def is_cache_valid(self) -> bool:
        """キャッシュの有効性チェック"""
        if not self.performance_cache:
            return False

        # 最新エントリの時刻をチェック
        latest_time = max(p.timestamp for p in self.performance_cache.values())
        elapsed = datetime.now() - latest_time

        # 1時間以内なら有効
        return elapsed.total_seconds() < 3600

    async def evaluate_model_performance(
            self, symbol: str, model_type: str
    ) -> Optional[PerformanceMetrics]:
        """個別モデル性能評価"""
        try:
            if not self.accuracy_validator:
                # 模擬データで評価
                return self._generate_mock_performance(symbol, model_type)

            # 実際の精度検証
            monitoring_config = self.config_manager.get_monitoring_config()
            validation_hours = monitoring_config.get('validation_hours', 168)

            if hasattr(self.accuracy_validator, 'validate_current_system_accuracy'):
                result = await self.accuracy_validator.validate_current_system_accuracy(
                    [symbol], validation_hours
                )
                accuracy = (
                    result.overall_accuracy 
                    if hasattr(result, 'overall_accuracy') else 0.0
                )
            else:
                test_symbols = [symbol]
                results = await self.accuracy_validator.validate_prediction_accuracy(
                    test_symbols
                )
                if symbol not in results:
                    return None
                result = results[symbol]
                accuracy = result.get('accuracy', 0.0)

            # 予測時間測定（模擬）
            start_time = time.time()
            prediction_time = (time.time() - start_time) * 1000

            # PerformanceMetricsオブジェクト作成
            metrics = PerformanceMetrics(
                symbol=symbol,
                model_type=model_type,
                timestamp=datetime.now(),
                accuracy=accuracy,
                precision=(
                    result.get('precision', 0.0) 
                    if isinstance(result, dict) 
                    else getattr(result, 'precision', 0.0)
                ),
                recall=(
                    result.get('recall', 0.0) 
                    if isinstance(result, dict) 
                    else getattr(result, 'recall', 0.0)
                ),
                f1_score=(
                    result.get('f1_score', 0.0) 
                    if isinstance(result, dict) 
                    else getattr(result, 'f1_score', 0.0)
                ),
                prediction_time_ms=prediction_time,
                confidence_avg=(
                    result.get('confidence_avg', 0.0) 
                    if isinstance(result, dict) 
                    else getattr(result, 'confidence_avg', 0.0)
                ),
                sample_size=(
                    result.get('sample_size', 0) 
                    if isinstance(result, dict) 
                    else getattr(result, 'sample_count', 0)
                ),
                prediction_accuracy=getattr(result, 'prediction_accuracy', 0.0),
                return_prediction=getattr(result, 'return_prediction', 0.0),
                volatility_prediction=getattr(result, 'volatility_prediction', 0.0),
                source='enhanced_validator'
            )

            # ステータス判定
            metrics.status = self.determine_performance_status(metrics)

            return metrics

        except Exception as e:
            logger.error(f"モデル性能評価エラー {symbol}_{model_type}: {e}")
            return None

    def _generate_mock_performance(
            self, symbol: str, model_type: str
    ) -> PerformanceMetrics:
        """模擬性能データ生成"""
        # 模擬的な性能値生成
        base_accuracy = 0.78 + np.random.normal(0, 0.05)
        base_accuracy = max(0.6, min(0.95, base_accuracy))

        metrics = PerformanceMetrics(
            symbol=symbol,
            model_type=model_type,
            timestamp=datetime.now(),
            accuracy=base_accuracy,
            precision=base_accuracy + np.random.normal(0, 0.02),
            recall=base_accuracy + np.random.normal(0, 0.02),
            f1_score=base_accuracy + np.random.normal(0, 0.02),
            prediction_time_ms=np.random.uniform(50, 200),
            confidence_avg=base_accuracy + np.random.normal(0, 0.03),
            sample_size=np.random.randint(100, 1000)
        )

        # 値の範囲調整
        for attr in ['precision', 'recall', 'f1_score', 'confidence_avg']:
            value = getattr(metrics, attr)
            if value is not None:
                setattr(metrics, attr, max(0.0, min(1.0, value)))

        metrics.status = self.determine_performance_status(metrics)
        return metrics

    def determine_performance_status(
            self, metrics: PerformanceMetrics
    ) -> PerformanceStatus:
        """性能ステータス判定"""
        thresholds = self.config_manager.config.get(
            'performance_thresholds', {}
        ).get('accuracy', {})
        accuracy = metrics.accuracy or 0.0

        if accuracy >= thresholds.get('target_threshold', 0.90):
            return PerformanceStatus.EXCELLENT
        elif accuracy >= thresholds.get('warning_threshold', 0.80):
            return PerformanceStatus.GOOD
        elif accuracy >= thresholds.get('minimum_threshold', 0.75):
            return PerformanceStatus.WARNING
        elif accuracy >= thresholds.get('critical_threshold', 0.70):
            return PerformanceStatus.CRITICAL
        else:
            return PerformanceStatus.CRITICAL

    async def check_and_generate_alerts(self, performance: PerformanceMetrics) -> List[PerformanceAlert]:
        """性能アラートチェックと生成"""
        alerts = []
        metrics_to_check = {
            'accuracy': performance.accuracy,
            'prediction_accuracy': performance.prediction_accuracy,
            'return_prediction': performance.return_prediction,
            'volatility_prediction': performance.volatility_prediction,
            'confidence': performance.confidence_avg
        }

        for metric_name, value in metrics_to_check.items():
            if value is None:
                continue

            threshold = self.config_manager.get_threshold(metric_name, performance.symbol)

            if value < threshold:
                # アラートレベル決定
                if value < threshold * 0.85:  # 15%以上の劣化
                    level = AlertLevel.CRITICAL
                    action = "即座に再学習を検討"
                elif value < threshold * 0.95:  # 5%以上の劣化
                    level = AlertLevel.WARNING
                    action = "監視を強化し、再学習を準備"
                else:
                    level = AlertLevel.INFO
                    action = "継続監視"

                alert = PerformanceAlert(
                    timestamp=datetime.now(),
                    alert_level=level,
                    symbol=performance.symbol,
                    model_type=performance.model_type,
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=threshold,
                    message=(
                        f"{metric_name}が閾値を下回りました: "
                        f"{value:.2f} < {threshold:.2f}"
                    ),
                    recommended_action=action
                )

                alerts.append(alert)

                logger.warning(
                    f"アラート生成: {performance.symbol} {metric_name} "
                    f"{value:.2f} < {threshold:.2f}"
                )

        return alerts

    def check_thresholds(self, metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """閾値チェック"""
        alerts = []
        thresholds = self.config_manager.config.get('performance_thresholds', {})

        # 精度閾値チェック
        accuracy_thresholds = thresholds.get('accuracy', {})
        if metrics.accuracy is not None:
            if metrics.accuracy < accuracy_thresholds.get('critical_threshold', 0.70):
                alert = PerformanceAlert(
                    id=f"accuracy_critical_{metrics.symbol}_{metrics.model_type}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=metrics.symbol,
                    model_type=metrics.model_type,
                    alert_level=AlertLevel.CRITICAL,
                    metric_name="accuracy",
                    current_value=metrics.accuracy,
                    threshold_value=accuracy_thresholds.get('critical_threshold', 0.70),
                    message=f"精度が緊急閾値を下回りました: {metrics.accuracy:.3f}"
                )
                alerts.append(alert)
            elif metrics.accuracy < accuracy_thresholds.get('minimum_threshold', 0.75):
                alert = PerformanceAlert(
                    id=f"accuracy_warning_{metrics.symbol}_{metrics.model_type}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=metrics.symbol,
                    model_type=metrics.model_type,
                    alert_level=AlertLevel.WARNING,
                    metric_name="accuracy",
                    current_value=metrics.accuracy,
                    threshold_value=accuracy_thresholds.get('minimum_threshold', 0.75),
                    message=f"精度が最低閾値を下回りました: {metrics.accuracy:.3f}"
                )
                alerts.append(alert)

        return alerts

    def check_performance_history(
            self, key: str, current_metrics: PerformanceMetrics
    ) -> List[PerformanceAlert]:
        """履歴比較チェック"""
        alerts = []

        if key not in self.performance_history or len(self.performance_history[key]) < 3:
            return alerts

        history = list(self.performance_history[key])
        recent_accuracy = [m.accuracy for m in history[-3:] if m.accuracy is not None]

        if len(recent_accuracy) >= 3 and current_metrics.accuracy is not None:
            avg_recent = np.mean(recent_accuracy[:-1])  # 最新を除く平均
            degradation = avg_recent - current_metrics.accuracy

            threshold_drop = (
                self.config_manager.config.get('retraining', {})
                .get('triggers', {})
                .get('accuracy_degradation', {})
                .get('threshold_drop', 0.05)
            )

            if degradation > threshold_drop:
                alert = PerformanceAlert(
                    id=f"degradation_{current_metrics.symbol}_{current_metrics.model_type}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=current_metrics.symbol,
                    model_type=current_metrics.model_type,
                    alert_level=AlertLevel.ERROR,
                    metric_name="accuracy_degradation",
                    current_value=current_metrics.accuracy,
                    threshold_value=avg_recent - threshold_drop,
                    message=f"精度が大幅に低下しました: {degradation:.3f}ポイント低下"
                )
                alerts.append(alert)

        return alerts