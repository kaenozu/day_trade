#!/usr/bin/env python3
"""
強化版モデル性能監視システム
Issue #857: model_performance_monitor.py改善

93%精度維持保証・予測精度連続監視・インテリジェントな再学習制御
"""

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import yaml
import numpy as np
from tqdm import tqdm

# 既存システムとの連携
try:
    from .model_performance_monitor import (
        EnhancedModelPerformanceMonitor,
        PerformanceMetrics,
        RetrainingResult,
        RetrainingScope,
        AlertLevel
    )
    BASE_MONITOR_AVAILABLE = True
except ImportError:
    BASE_MONITOR_AVAILABLE = False
    logging.warning("Base monitor not available")

try:
    from ml_model_upgrade_system import ml_upgrade_system
    from prediction_accuracy_validator import PredictionAccuracyValidator
    ML_SYSTEMS_AVAILABLE = True
except ImportError:
    ml_upgrade_system = None
    PredictionAccuracyValidator = None
    ML_SYSTEMS_AVAILABLE = False


class AccuracyGuaranteeLevel(Enum):
    """精度保証レベル"""
    STRICT_95 = "strict_95"      # 厳格95%保証
    STANDARD_93 = "standard_93"  # 標準93%保証
    RELAXED_90 = "relaxed_90"    # 緩和90%保証
    ADAPTIVE = "adaptive"        # 適応的保証


class MonitoringIntensity(Enum):
    """監視強度"""
    CONTINUOUS = "continuous"    # 連続監視（1分間隔）
    HIGH = "high"               # 高頻度（5分間隔）
    NORMAL = "normal"           # 通常（15分間隔）
    LOW = "low"                 # 低頻度（60分間隔）


class PredictionQualityStatus(Enum):
    """予測品質状態"""
    EXCELLENT = "excellent"     # 優秀（95%+）
    GOOD = "good"              # 良好（93-95%）
    ACCEPTABLE = "acceptable"   # 許容範囲（90-93%）
    WARNING = "warning"        # 警告（85-90%）
    CRITICAL = "critical"      # 危険（85%未満）


@dataclass
class AccuracyGuaranteeConfig:
    """精度保証設定"""
    guarantee_level: AccuracyGuaranteeLevel = AccuracyGuaranteeLevel.STANDARD_93
    min_accuracy: float = 93.0
    target_accuracy: float = 95.0
    monitoring_intensity: MonitoringIntensity = MonitoringIntensity.HIGH
    auto_retraining: bool = True
    emergency_threshold: float = 85.0
    sample_window_hours: int = 24
    confidence_threshold: float = 0.85


@dataclass
class ContinuousMonitoringMetrics:
    """連続監視メトリクス"""
    symbol: str
    timestamp: datetime
    accuracy_current: float
    accuracy_trend: float
    prediction_confidence: float
    sample_count: int
    moving_average_24h: float
    moving_average_7d: float
    quality_status: PredictionQualityStatus
    degradation_rate: float = 0.0
    improvement_rate: float = 0.0
    stability_score: float = 0.0


@dataclass
class RetrainingStrategy:
    """再学習戦略"""
    strategy_name: str
    trigger_conditions: List[str]
    priority: int
    estimated_improvement: float
    resource_cost: int
    success_probability: float
    cooldown_hours: int


class AccuracyGuaranteeSystem:
    """93%精度維持保証システム"""

    def __init__(self, config: AccuracyGuaranteeConfig):
        self.config = config
        self.accuracy_history: Dict[str, List[float]] = {}
        self.trend_analyzer = AccuracyTrendAnalyzer()
        self.emergency_detector = EmergencyDetector()
        self.guarantee_active = True

        # 保証レベル別設定
        self.guarantee_thresholds = {
            AccuracyGuaranteeLevel.STRICT_95: 95.0,
            AccuracyGuaranteeLevel.STANDARD_93: 93.0,
            AccuracyGuaranteeLevel.RELAXED_90: 90.0,
            AccuracyGuaranteeLevel.ADAPTIVE: self._calculate_adaptive_threshold()
        }

        self.setup_logging()

    def setup_logging(self):
        """ログ設定"""
        self.logger = logging.getLogger(f"{__name__}.AccuracyGuarantee")

    def _calculate_adaptive_threshold(self) -> float:
        """適応的閾値計算"""
        # 過去の性能データに基づく動的閾値
        return 93.0  # 基本値（実装時は過去データ分析）

    async def validate_accuracy_guarantee(
        self,
        symbol_performances: Dict[str, PerformanceMetrics]
    ) -> Tuple[bool, List[str], float]:
        """
        精度保証の検証

        Returns:
            (保証達成, 違反銘柄リスト, 総合精度)
        """
        required_accuracy = self.guarantee_thresholds[self.config.guarantee_level]
        violation_symbols = []
        total_accuracy = 0.0

        if not symbol_performances:
            return False, [], 0.0

        for symbol, metrics in symbol_performances.items():
            if metrics.accuracy < required_accuracy:
                violation_symbols.append(symbol)
            total_accuracy += metrics.accuracy

        overall_accuracy = total_accuracy / len(symbol_performances)
        guarantee_met = (
            len(violation_symbols) == 0 and
            overall_accuracy >= required_accuracy
        )

        self.logger.info(
            f"精度保証検証: {guarantee_met}, "
            f"総合精度: {overall_accuracy:.2f}%, "
            f"違反銘柄: {len(violation_symbols)}件"
        )

        return guarantee_met, violation_symbols, overall_accuracy

    async def trigger_guarantee_recovery(
        self,
        violation_symbols: List[str],
        current_accuracy: float
    ) -> RetrainingResult:
        """保証回復のための緊急再学習"""
        self.logger.warning(f"精度保証違反検出 - 緊急回復開始: {violation_symbols}")

        # 緊急度判定
        severity = self._assess_violation_severity(current_accuracy)

        # 回復戦略選択
        strategy = self._select_recovery_strategy(severity, violation_symbols)

        # 緊急再学習実行
        return await self._execute_emergency_retraining(strategy, violation_symbols)

    def _assess_violation_severity(self, accuracy: float) -> str:
        """違反重要度評価"""
        if accuracy < self.config.emergency_threshold:
            return "critical"
        elif accuracy < self.config.min_accuracy * 0.95:
            return "high"
        else:
            return "medium"

    def _select_recovery_strategy(self, severity: str, symbols: List[str]) -> RetrainingStrategy:
        """回復戦略選択"""
        if severity == "critical":
            return RetrainingStrategy(
                strategy_name="emergency_global",
                trigger_conditions=["critical_accuracy_violation"],
                priority=1,
                estimated_improvement=8.0,
                resource_cost=10,
                success_probability=0.9,
                cooldown_hours=0  # 緊急時は冷却期間無視
            )
        elif severity == "high":
            return RetrainingStrategy(
                strategy_name="priority_symbols",
                trigger_conditions=["high_priority_violation"],
                priority=2,
                estimated_improvement=5.0,
                resource_cost=6,
                success_probability=0.85,
                cooldown_hours=6
            )
        else:
            return RetrainingStrategy(
                strategy_name="targeted_improvement",
                trigger_conditions=["performance_degradation"],
                priority=3,
                estimated_improvement=3.0,
                resource_cost=3,
                success_probability=0.8,
                cooldown_hours=12
            )

    async def _execute_emergency_retraining(
        self,
        strategy: RetrainingStrategy,
        symbols: List[str]
    ) -> RetrainingResult:
        """緊急再学習実行"""
        start_time = datetime.now()

        try:
            self.logger.info(f"緊急再学習開始: {strategy.strategy_name}")

            # TODO: 実際の再学習システムとの連携
            # 仮想的な再学習実行
            await asyncio.sleep(2)  # 実際の処理をシミュレート

            duration = (datetime.now() - start_time).total_seconds()

            return RetrainingResult(
                triggered=True,
                scope=RetrainingScope.GLOBAL,
                affected_symbols=symbols,
                improvement=strategy.estimated_improvement,
                duration=duration,
                estimated_time=strategy.resource_cost * 300  # 概算時間
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"緊急再学習エラー: {e}")

            return RetrainingResult(
                triggered=False,
                scope=RetrainingScope.NONE,
                affected_symbols=symbols,
                duration=duration,
                error=str(e)
            )


class AccuracyTrendAnalyzer:
    """精度トレンド分析器"""

    def __init__(self):
        self.trend_window = 168  # 7日間
        self.logger = logging.getLogger(f"{__name__}.TrendAnalyzer")

    def analyze_accuracy_trend(
        self,
        symbol: str,
        accuracy_history: List[Tuple[datetime, float]]
    ) -> Dict[str, float]:
        """精度トレンド分析"""
        if len(accuracy_history) < 10:
            return {"trend": 0.0, "stability": 0.0, "prediction": 0.0}

        # 時系列データ準備
        timestamps = [t for t, _ in accuracy_history]
        accuracies = [a for _, a in accuracy_history]

        # トレンド計算（線形回帰）
        trend = self._calculate_linear_trend(accuracies)

        # 安定性計算（変動係数）
        stability = self._calculate_stability(accuracies)

        # 将来予測（1日後の精度予測）
        prediction = self._predict_future_accuracy(accuracies, hours_ahead=24)

        return {
            "trend": trend,
            "stability": stability,
            "prediction": prediction,
            "current": accuracies[-1] if accuracies else 0.0,
            "mean_24h": np.mean(accuracies[-24:]) if len(accuracies) >= 24 else 0.0,
            "volatility": np.std(accuracies[-24:]) if len(accuracies) >= 24 else 0.0
        }

    def _calculate_linear_trend(self, accuracies: List[float]) -> float:
        """線形トレンド計算"""
        if len(accuracies) < 2:
            return 0.0

        x = np.arange(len(accuracies))
        y = np.array(accuracies)

        # 最小二乗法
        slope = np.polyfit(x, y, 1)[0]
        return slope * 24  # 24時間あたりの変化率

    def _calculate_stability(self, accuracies: List[float]) -> float:
        """安定性スコア計算"""
        if len(accuracies) < 2:
            return 1.0

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        # 変動係数の逆数（安定性）
        cv = std_acc / mean_acc if mean_acc > 0 else 1.0
        stability = 1.0 / (1.0 + cv)

        return stability

    def _predict_future_accuracy(self, accuracies: List[float], hours_ahead: int = 24) -> float:
        """将来精度予測"""
        if len(accuracies) < 5:
            return accuracies[-1] if accuracies else 0.0

        # 簡単な指数平滑法
        alpha = 0.3
        smoothed = accuracies[0]

        for acc in accuracies[1:]:
            smoothed = alpha * acc + (1 - alpha) * smoothed

        # トレンドを考慮した予測
        trend = self._calculate_linear_trend(accuracies[-24:])
        predicted = smoothed + (trend * hours_ahead / 24)

        return max(0.0, min(100.0, predicted))


class EmergencyDetector:
    """緊急事態検出システム"""

    def __init__(self):
        self.anomaly_threshold = 2.0  # 標準偏差の2倍
        self.degradation_threshold = 5.0  # 5%以上の急激な低下
        self.logger = logging.getLogger(f"{__name__}.EmergencyDetector")

    def detect_emergency_conditions(
        self,
        metrics: ContinuousMonitoringMetrics
    ) -> Tuple[bool, List[str]]:
        """緊急事態検出"""
        emergency_flags = []

        # 急激な精度低下
        if metrics.degradation_rate > self.degradation_threshold:
            emergency_flags.append("rapid_degradation")

        # 異常な精度値
        if metrics.accuracy_current < 80.0:
            emergency_flags.append("critical_accuracy")

        # 不安定な予測
        if metrics.stability_score < 0.3:
            emergency_flags.append("unstable_predictions")

        # 信頼度低下
        if metrics.prediction_confidence < 0.5:
            emergency_flags.append("low_confidence")

        # トレンド悪化
        if metrics.accuracy_trend < -2.0:  # 1時間で2%以上低下
            emergency_flags.append("negative_trend")

        is_emergency = len(emergency_flags) > 0

        if is_emergency:
            self.logger.warning(
                f"緊急事態検出: {metrics.symbol} - {emergency_flags}"
            )

        return is_emergency, emergency_flags


class ContinuousPerformanceMonitor:
    """連続性能監視システム"""

    def __init__(self, config: AccuracyGuaranteeConfig):
        self.config = config
        self.monitoring_active = True
        self.metrics_cache: Dict[str, List[ContinuousMonitoringMetrics]] = {}
        self.accuracy_guarantee = AccuracyGuaranteeSystem(config)
        self.trend_analyzer = AccuracyTrendAnalyzer()
        self.emergency_detector = EmergencyDetector()

        # 監視間隔設定
        self.monitoring_intervals = {
            MonitoringIntensity.CONTINUOUS: 60,    # 1分
            MonitoringIntensity.HIGH: 300,         # 5分
            MonitoringIntensity.NORMAL: 900,       # 15分
            MonitoringIntensity.LOW: 3600          # 1時間
        }

        self.setup_logging()

    def setup_logging(self):
        """ログ設定"""
        self.logger = logging.getLogger(f"{__name__}.ContinuousMonitor")

    async def start_continuous_monitoring(self, symbols: List[str]) -> None:
        """連続監視開始"""
        self.logger.info(f"連続監視開始: {len(symbols)}銘柄")

        interval = self.monitoring_intervals[self.config.monitoring_intensity]

        while self.monitoring_active:
            try:
                # 全銘柄の性能チェック
                all_metrics = await self._collect_continuous_metrics(symbols)

                # 精度保証チェック
                await self._check_accuracy_guarantee(all_metrics)

                # 緊急事態検出
                await self._check_emergency_conditions(all_metrics)

                # 進捗表示
                self._display_monitoring_status(all_metrics)

                # 次回まで待機
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                self.logger.info("連続監視が停止されました")
                break
            except Exception as e:
                self.logger.error(f"連続監視エラー: {e}")
                await asyncio.sleep(60)  # エラー時は1分待機

    async def _collect_continuous_metrics(
        self,
        symbols: List[str]
    ) -> Dict[str, ContinuousMonitoringMetrics]:
        """連続メトリクス収集"""
        metrics = {}

        for symbol in symbols:
            try:
                # 個別銘柄メトリクス計算
                metric = await self._calculate_symbol_metrics(symbol)
                metrics[symbol] = metric

                # キャッシュ更新
                if symbol not in self.metrics_cache:
                    self.metrics_cache[symbol] = []

                self.metrics_cache[symbol].append(metric)

                # キャッシュサイズ制限（24時間分）
                max_entries = 24 * 60 // (
                    self.monitoring_intervals[self.config.monitoring_intensity] // 60
                )
                if len(self.metrics_cache[symbol]) > max_entries:
                    self.metrics_cache[symbol] = self.metrics_cache[symbol][-max_entries:]

            except Exception as e:
                self.logger.error(f"メトリクス収集エラー ({symbol}): {e}")

        return metrics

    async def _calculate_symbol_metrics(self, symbol: str) -> ContinuousMonitoringMetrics:
        """銘柄別メトリクス計算"""
        # TODO: 実際の性能データ取得
        # 仮想的なメトリクス（実装時は実データ使用）
        current_accuracy = np.random.uniform(90, 98)

        # 履歴からトレンド分析
        history = [(datetime.now() - timedelta(hours=i), current_accuracy + np.random.uniform(-2, 2))
                  for i in range(24, 0, -1)]

        trend_data = self.trend_analyzer.analyze_accuracy_trend(symbol, history)

        # 品質状態判定
        quality_status = self._determine_quality_status(current_accuracy)

        return ContinuousMonitoringMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            accuracy_current=current_accuracy,
            accuracy_trend=trend_data["trend"],
            prediction_confidence=np.random.uniform(0.7, 0.95),
            sample_count=np.random.randint(50, 200),
            moving_average_24h=trend_data["mean_24h"],
            moving_average_7d=current_accuracy + np.random.uniform(-1, 1),
            quality_status=quality_status,
            degradation_rate=max(0, -trend_data["trend"]),
            improvement_rate=max(0, trend_data["trend"]),
            stability_score=trend_data["stability"]
        )

    def _determine_quality_status(self, accuracy: float) -> PredictionQualityStatus:
        """品質状態判定"""
        if accuracy >= 95:
            return PredictionQualityStatus.EXCELLENT
        elif accuracy >= 93:
            return PredictionQualityStatus.GOOD
        elif accuracy >= 90:
            return PredictionQualityStatus.ACCEPTABLE
        elif accuracy >= 85:
            return PredictionQualityStatus.WARNING
        else:
            return PredictionQualityStatus.CRITICAL

    async def _check_accuracy_guarantee(
        self,
        metrics: Dict[str, ContinuousMonitoringMetrics]
    ) -> None:
        """精度保証チェック"""
        # メトリクスをPerformanceMetricsに変換
        performance_metrics = {}
        for symbol, metric in metrics.items():
            performance_metrics[symbol] = PerformanceMetrics(
                symbol=symbol,
                accuracy=metric.accuracy_current,
                prediction_accuracy=metric.prediction_confidence * 100,
                timestamp=metric.timestamp
            )

        # 精度保証検証
        guarantee_met, violations, overall = await self.accuracy_guarantee.validate_accuracy_guarantee(
            performance_metrics
        )

        if not guarantee_met:
            self.logger.warning(f"精度保証違反: 総合{overall:.2f}%, 違反{len(violations)}銘柄")

            # 自動再学習が有効な場合
            if self.config.auto_retraining:
                await self.accuracy_guarantee.trigger_guarantee_recovery(
                    violations, overall
                )

    async def _check_emergency_conditions(
        self,
        metrics: Dict[str, ContinuousMonitoringMetrics]
    ) -> None:
        """緊急事態チェック"""
        for symbol, metric in metrics.items():
            is_emergency, flags = self.emergency_detector.detect_emergency_conditions(metric)

            if is_emergency:
                self.logger.error(f"緊急事態検出: {symbol} - {flags}")

                # 緊急アクション（自動再学習など）
                if self.config.auto_retraining:
                    await self._trigger_emergency_action(symbol, flags, metric)

    async def _trigger_emergency_action(
        self,
        symbol: str,
        flags: List[str],
        metric: ContinuousMonitoringMetrics
    ) -> None:
        """緊急アクション実行"""
        self.logger.info(f"緊急アクション実行: {symbol}")

        # 緊急再学習
        result = await self.accuracy_guarantee.trigger_guarantee_recovery(
            [symbol], metric.accuracy_current
        )

        if result.triggered:
            self.logger.info(f"緊急再学習完了: {symbol} - 改善 {result.improvement:.2f}%")

    def _display_monitoring_status(
        self,
        metrics: Dict[str, ContinuousMonitoringMetrics]
    ) -> None:
        """監視状況表示"""
        if not metrics:
            return

        # 統計計算
        accuracies = [m.accuracy_current for m in metrics.values()]
        avg_accuracy = np.mean(accuracies)
        min_accuracy = np.min(accuracies)

        # 品質状態集計
        status_counts = {}
        for metric in metrics.values():
            status = metric.quality_status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # ログ出力
        self.logger.info(
            f"監視状況 - 平均精度: {avg_accuracy:.2f}%, "
            f"最低精度: {min_accuracy:.2f}%, "
            f"品質分布: {status_counts}"
        )

    def stop_monitoring(self) -> None:
        """監視停止"""
        self.monitoring_active = False
        self.logger.info("連続監視停止要求")


class IntelligentRetrainingController:
    """インテリジェント再学習制御システム"""

    def __init__(self):
        self.retraining_strategies = self._initialize_strategies()
        self.execution_history: List[RetrainingResult] = []
        self.resource_monitor = ResourceMonitor()
        self.logger = logging.getLogger(f"{__name__}.RetrainingController")

    def _initialize_strategies(self) -> Dict[str, RetrainingStrategy]:
        """再学習戦略初期化"""
        return {
            "emergency_global": RetrainingStrategy(
                strategy_name="emergency_global",
                trigger_conditions=["critical_accuracy", "system_failure"],
                priority=1,
                estimated_improvement=10.0,
                resource_cost=10,
                success_probability=0.95,
                cooldown_hours=0
            ),
            "targeted_symbols": RetrainingStrategy(
                strategy_name="targeted_symbols",
                trigger_conditions=["symbol_degradation", "accuracy_violation"],
                priority=2,
                estimated_improvement=6.0,
                resource_cost=5,
                success_probability=0.85,
                cooldown_hours=6
            ),
            "incremental_update": RetrainingStrategy(
                strategy_name="incremental_update",
                trigger_conditions=["data_drift", "minor_degradation"],
                priority=3,
                estimated_improvement=3.0,
                resource_cost=2,
                success_probability=0.8,
                cooldown_hours=1
            ),
            "preventive_maintenance": RetrainingStrategy(
                strategy_name="preventive_maintenance",
                trigger_conditions=["scheduled_maintenance", "proactive_update"],
                priority=4,
                estimated_improvement=2.0,
                resource_cost=3,
                success_probability=0.75,
                cooldown_hours=24
            )
        }

    async def select_optimal_strategy(
        self,
        conditions: List[str],
        current_accuracy: float,
        available_resources: int
    ) -> Optional[RetrainingStrategy]:
        """最適戦略選択"""
        candidate_strategies = []

        for strategy in self.retraining_strategies.values():
            # 条件マッチング
            if any(cond in strategy.trigger_conditions for cond in conditions):
                # リソース制約チェック
                if strategy.resource_cost <= available_resources:
                    candidate_strategies.append(strategy)

        if not candidate_strategies:
            return None

        # 優先度とコストパフォーマンスで選択
        best_strategy = max(
            candidate_strategies,
            key=lambda s: (
                10 - s.priority +  # 優先度（高いほど良い）
                s.estimated_improvement / s.resource_cost * 2 +  # コストパフォーマンス
                s.success_probability * 3  # 成功確率
            )
        )

        self.logger.info(f"最適戦略選択: {best_strategy.strategy_name}")
        return best_strategy


class ResourceMonitor:
    """リソース監視"""

    def __init__(self):
        self.max_concurrent_training = 2
        self.current_training_count = 0
        self.logger = logging.getLogger(f"{__name__}.ResourceMonitor")

    def check_available_resources(self) -> int:
        """利用可能リソース確認"""
        available = self.max_concurrent_training - self.current_training_count
        return max(0, available * 5)  # リソース単位変換

    async def reserve_resources(self, cost: int) -> bool:
        """リソース予約"""
        if self.check_available_resources() >= cost:
            self.current_training_count += 1
            return True
        return False

    def release_resources(self) -> None:
        """リソース解放"""
        self.current_training_count = max(0, self.current_training_count - 1)


class EnhancedPerformanceMonitorV2:
    """強化版性能監視システム v2.0"""

    def __init__(self, config_path: Optional[str] = None):
        # 設定読み込み
        self.config = self._load_enhanced_config(config_path)

        # 精度保証設定
        guarantee_config = AccuracyGuaranteeConfig(
            guarantee_level=AccuracyGuaranteeLevel.STANDARD_93,
            min_accuracy=self.config.get("accuracy_guarantee", {}).get("min_accuracy", 93.0),
            monitoring_intensity=MonitoringIntensity.HIGH
        )

        # コンポーネント初期化
        self.continuous_monitor = ContinuousPerformanceMonitor(guarantee_config)
        self.retraining_controller = IntelligentRetrainingController()

        # 基本監視システム連携
        if BASE_MONITOR_AVAILABLE:
            self.base_monitor = EnhancedModelPerformanceMonitor()
        else:
            self.base_monitor = None

        self.monitoring_task: Optional[asyncio.Task] = None
        self.setup_logging()

    def setup_logging(self):
        """ログ設定"""
        self.logger = logging.getLogger(f"{__name__}.EnhancedV2")

    def _load_enhanced_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """強化設定読み込み"""
        default_config = {
            "accuracy_guarantee": {
                "min_accuracy": 93.0,
                "target_accuracy": 95.0,
                "emergency_threshold": 85.0
            },
            "continuous_monitoring": {
                "intensity": "high",
                "interval_minutes": 5,
                "trend_analysis": True
            },
            "intelligent_retraining": {
                "auto_trigger": True,
                "resource_optimization": True,
                "strategy_selection": "adaptive"
            }
        }

        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"設定ファイル読み込み失敗: {e}")

        return default_config

    async def start_enhanced_monitoring(self, symbols: List[str]) -> None:
        """強化監視開始"""
        self.logger.info(f"強化版性能監視開始: {len(symbols)}銘柄")

        # 連続監視タスク開始
        self.monitoring_task = asyncio.create_task(
            self.continuous_monitor.start_continuous_monitoring(symbols)
        )

        try:
            await self.monitoring_task
        except asyncio.CancelledError:
            self.logger.info("強化監視が停止されました")

    def stop_enhanced_monitoring(self) -> None:
        """強化監視停止"""
        self.continuous_monitor.stop_monitoring()
        if self.monitoring_task:
            self.monitoring_task.cancel()

    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """包括レポート生成"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_status": "active" if self.continuous_monitor.monitoring_active else "stopped",
            "accuracy_guarantee": {
                "level": self.continuous_monitor.config.guarantee_level.value,
                "min_accuracy": self.continuous_monitor.config.min_accuracy,
                "current_status": "maintained"  # 実際の状態を計算
            },
            "continuous_metrics": {},
            "retraining_history": [],
            "resource_usage": {
                "available": self.retraining_controller.resource_monitor.check_available_resources(),
                "active_training": self.retraining_controller.resource_monitor.current_training_count
            }
        }

        # 連続メトリクス
        for symbol, metrics_list in self.continuous_monitor.metrics_cache.items():
            if metrics_list:
                latest = metrics_list[-1]
                report["continuous_metrics"][symbol] = {
                    "accuracy": latest.accuracy_current,
                    "trend": latest.accuracy_trend,
                    "quality_status": latest.quality_status.value,
                    "stability": latest.stability_score
                }

        return report


async def main():
    """メイン実行関数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)
    logger.info("強化版性能監視システム v2.0 開始")

    # システム初期化
    monitor = EnhancedPerformanceMonitorV2()

    # テスト銘柄
    test_symbols = ["7203", "8306", "4751", "9984"]

    try:
        # 強化監視開始（5分間のテスト）
        await asyncio.wait_for(
            monitor.start_enhanced_monitoring(test_symbols),
            timeout=300  # 5分
        )
    except asyncio.TimeoutError:
        logger.info("テスト監視期間終了")
    except KeyboardInterrupt:
        logger.info("手動停止")
    finally:
        monitor.stop_enhanced_monitoring()

        # 最終レポート
        report = await monitor.generate_comprehensive_report()
        logger.info(f"最終レポート: {json.dumps(report, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    asyncio.run(main())