#!/usr/bin/env python3
"""
Prometheus メトリクス収集システム
Day Trade システム専用メトリクス定義・収集機能
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

@dataclass
class MetricConfig:
    """メトリクス設定"""
    name: str
    help: str
    labels: List[str] = None
    buckets: List[float] = None

class PrometheusMetricsCollector:
    """メインメトリクス収集器"""

    def __init__(self, registry: CollectorRegistry = None):
        self.registry = registry or CollectorRegistry()
        self._initialize_metrics()
        self._last_collection_time = time.time()

    def _initialize_metrics(self):
        """基本メトリクス初期化"""

        # システム情報
        self.system_info = Info(
            'day_trade_system_info',
            'Day Trade システム情報',
            registry=self.registry
        )

        # 収集統計
        self.metrics_collection_total = Counter(
            'day_trade_metrics_collection_total',
            'メトリクス収集回数',
            ['collector_type'],
            registry=self.registry
        )

        self.metrics_collection_duration = Histogram(
            'day_trade_metrics_collection_duration_seconds',
            'メトリクス収集時間',
            ['collector_type'],
            buckets=[0.001, 0.01, 0.1, 1.0, 5.0],
            registry=self.registry
        )

        # システム初期化
        self.system_info.info({
            'version': '2.0.0',
            'environment': 'production',
            'start_time': datetime.now().isoformat()
        })

    def collect_all_metrics(self) -> Dict[str, Any]:
        """全メトリクス収集"""

        start_time = time.time()

        try:
            # 基本統計更新
            self.metrics_collection_total.labels(
                collector_type='all'
            ).inc()

            collection_time = time.time() - start_time
            self.metrics_collection_duration.labels(
                collector_type='all'
            ).observe(collection_time)

            return {
                'status': 'success',
                'collection_time': collection_time,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"メトリクス収集エラー: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class RiskManagementMetrics:
    """リスク管理システムメトリクス"""

    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        self._initialize_risk_metrics()

    def _initialize_risk_metrics(self):
        """リスク管理メトリクス初期化"""

        # リスク分析実行回数
        self.risk_analysis_total = Counter(
            'day_trade_risk_analysis_total',
            'リスク分析実行回数',
            ['analysis_type', 'risk_level'],
            registry=self.registry
        )

        # リスク分析処理時間
        self.risk_analysis_duration = Histogram(
            'day_trade_risk_analysis_duration_seconds',
            'リスク分析処理時間',
            ['analysis_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )

        # 現在のリスクスコア
        self.current_risk_score = Gauge(
            'day_trade_current_risk_score',
            '現在のリスクスコア',
            ['component', 'symbol'],
            registry=self.registry
        )

        # リスクアラート発生回数
        self.risk_alerts_total = Counter(
            'day_trade_risk_alerts_total',
            'リスクアラート発生回数',
            ['alert_level', 'component'],
            registry=self.registry
        )

        # 不正検知結果
        self.fraud_detection_total = Counter(
            'day_trade_fraud_detection_total',
            '不正検知実行回数',
            ['result', 'confidence_level'],
            registry=self.registry
        )

        # 不正検知精度
        self.fraud_detection_accuracy = Gauge(
            'day_trade_fraud_detection_accuracy',
            '不正検知精度',
            ['model_type'],
            registry=self.registry
        )

        # リスク分析コンポーネント状態
        self.risk_component_status = Gauge(
            'day_trade_risk_component_status',
            'リスクコンポーネント状態 (1=正常, 0=異常)',
            ['component'],
            registry=self.registry
        )

    def record_risk_analysis(self, analysis_type: str, risk_level: str,
                           duration: float):
        """リスク分析記録"""

        self.risk_analysis_total.labels(
            analysis_type=analysis_type,
            risk_level=risk_level
        ).inc()

        self.risk_analysis_duration.labels(
            analysis_type=analysis_type
        ).observe(duration)

    def update_risk_score(self, component: str, symbol: str, score: float):
        """リスクスコア更新"""

        self.current_risk_score.labels(
            component=component,
            symbol=symbol
        ).set(score)

    def record_fraud_detection(self, result: str, confidence: str,
                             accuracy: float = None, model_type: str = 'ensemble'):
        """不正検知結果記録"""

        self.fraud_detection_total.labels(
            result=result,
            confidence_level=confidence
        ).inc()

        if accuracy is not None:
            self.fraud_detection_accuracy.labels(
                model_type=model_type
            ).set(accuracy)

class TradingPerformanceMetrics:
    """取引パフォーマンスメトリクス"""

    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        self._initialize_trading_metrics()

    def _initialize_trading_metrics(self):
        """取引メトリクス初期化"""

        # 取引実行回数
        self.trades_total = Counter(
            'day_trade_trades_total',
            '取引実行回数',
            ['trade_type', 'symbol', 'result'],
            registry=self.registry
        )

        # 取引執行時間
        self.trade_execution_duration = Histogram(
            'day_trade_execution_duration_seconds',
            '取引執行時間',
            ['trade_type'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0],
            registry=self.registry
        )

        # ポートフォリオ価値
        self.portfolio_value = Gauge(
            'day_trade_portfolio_value',
            'ポートフォリオ価値',
            ['currency'],
            registry=self.registry
        )

        # 未実現損益
        self.unrealized_pnl = Gauge(
            'day_trade_unrealized_pnl',
            '未実現損益',
            ['symbol', 'currency'],
            registry=self.registry
        )

        # 実現損益
        self.realized_pnl = Counter(
            'day_trade_realized_pnl_total',
            '実現損益累計',
            ['symbol', 'currency'],
            registry=self.registry
        )

        # 勝率
        self.win_rate = Gauge(
            'day_trade_win_rate',
            '勝率',
            ['strategy', 'timeframe'],
            registry=self.registry
        )

        # 最大ドローダウン
        self.max_drawdown = Gauge(
            'day_trade_max_drawdown',
            '最大ドローダウン',
            ['strategy'],
            registry=self.registry
        )

class AIEngineMetrics:
    """AI エンジンメトリクス"""

    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        self._initialize_ai_metrics()

    def _initialize_ai_metrics(self):
        """AI エンジンメトリクス初期化"""

        # AI予測実行回数
        self.ai_predictions_total = Counter(
            'day_trade_ai_predictions_total',
            'AI予測実行回数',
            ['model_type', 'symbol'],
            registry=self.registry
        )

        # AI予測処理時間
        self.ai_prediction_duration = Histogram(
            'day_trade_ai_prediction_duration_seconds',
            'AI予測処理時間',
            ['model_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )

        # 予測精度
        self.prediction_accuracy = Gauge(
            'day_trade_prediction_accuracy',
            '予測精度',
            ['model_type', 'timeframe'],
            registry=self.registry
        )

        # 生成AI API呼び出し
        self.generative_ai_calls_total = Counter(
            'day_trade_generative_ai_calls_total',
            '生成AI API呼び出し回数',
            ['provider', 'model', 'result'],
            registry=self.registry
        )

        # GPU使用率
        self.gpu_utilization = Gauge(
            'day_trade_gpu_utilization_percent',
            'GPU使用率',
            ['gpu_id'],
            registry=self.registry
        )

        # モデル訓練状況
        self.model_training_status = Gauge(
            'day_trade_model_training_status',
            'モデル訓練状況 (1=訓練中, 0=停止)',
            ['model_type'],
            registry=self.registry
        )

class SystemHealthMetrics:
    """システムヘルスメトリクス"""

    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        self._initialize_health_metrics()
        self._start_system_monitoring()

    def _initialize_health_metrics(self):
        """ヘルスメトリクス初期化"""

        # CPU使用率
        self.cpu_usage = Gauge(
            'day_trade_cpu_usage_percent',
            'CPU使用率',
            registry=self.registry
        )

        # メモリ使用量
        self.memory_usage = Gauge(
            'day_trade_memory_usage_bytes',
            'メモリ使用量',
            ['type'],  # used, available, total
            registry=self.registry
        )

        # ディスク使用量
        self.disk_usage = Gauge(
            'day_trade_disk_usage_bytes',
            'ディスク使用量',
            ['mount_point', 'type'],
            registry=self.registry
        )

        # アクティブ接続数
        self.active_connections = Gauge(
            'day_trade_active_connections',
            'アクティブ接続数',
            ['connection_type'],
            registry=self.registry
        )

        # データベース接続プール
        self.db_connections = Gauge(
            'day_trade_db_connections',
            'データベース接続数',
            ['status'],  # active, idle, total
            registry=self.registry
        )

        # アプリケーションエラー
        self.application_errors = Counter(
            'day_trade_application_errors_total',
            'アプリケーションエラー回数',
            ['error_type', 'component'],
            registry=self.registry
        )

    def _start_system_monitoring(self):
        """システム監視開始"""

        def update_system_metrics():
            while True:
                try:
                    # CPU使用率
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_usage.set(cpu_percent)

                    # メモリ使用量
                    memory = psutil.virtual_memory()
                    self.memory_usage.labels(type='used').set(memory.used)
                    self.memory_usage.labels(type='available').set(memory.available)
                    self.memory_usage.labels(type='total').set(memory.total)

                    # ディスク使用量
                    disk = psutil.disk_usage('/')
                    self.disk_usage.labels(mount_point='/', type='used').set(disk.used)
                    self.disk_usage.labels(mount_point='/', type='total').set(disk.total)

                except Exception as e:
                    logger.error(f"システムメトリクス更新エラー: {e}")

                time.sleep(10)  # 10秒間隔で更新

        # バックグラウンドで実行
        monitoring_thread = threading.Thread(
            target=update_system_metrics,
            daemon=True
        )
        monitoring_thread.start()

    def record_error(self, error_type: str, component: str):
        """エラー記録"""

        self.application_errors.labels(
            error_type=error_type,
            component=component
        ).inc()

# グローバル インスタンス
_default_registry = CollectorRegistry()
_metrics_collector = PrometheusMetricsCollector(_default_registry)
_risk_metrics = RiskManagementMetrics(_default_registry)
_trading_metrics = TradingPerformanceMetrics(_default_registry)
_ai_metrics = AIEngineMetrics(_default_registry)
_health_metrics = SystemHealthMetrics(_default_registry)

def get_metrics_collector() -> PrometheusMetricsCollector:
    """デフォルトメトリクス収集器取得"""
    return _metrics_collector

def get_risk_metrics() -> RiskManagementMetrics:
    """リスク管理メトリクス取得"""
    return _risk_metrics

def get_trading_metrics() -> TradingPerformanceMetrics:
    """取引メトリクス取得"""
    return _trading_metrics

def get_ai_metrics() -> AIEngineMetrics:
    """AI メトリクス取得"""
    return _ai_metrics

def get_health_metrics() -> SystemHealthMetrics:
    """ヘルスメトリクス取得"""
    return _health_metrics

def get_registry() -> CollectorRegistry:
    """デフォルトレジストリ取得"""
    return _default_registry
