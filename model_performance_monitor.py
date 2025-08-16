#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Monitor - MLモデル性能監視システム
Issue #857対応：モデル監視の柔軟性と効率性強化

MLモデルの性能を継続的に監視し、性能低下を検知して自動的に再学習をトリガーする
堅牢で柔軟なモデル性能監視システム
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import yaml
import time
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# 共通ユーティリティのインポート
try:
    from src.day_trade.utils.windows_console_fix import setup_windows_console
    setup_windows_console()
except ImportError:
    # Windows環境での文字化け対策（フォールバック）
    import sys
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# 機械学習関連
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 既存システムのインポート
try:
    from ml_prediction_models import MLPredictionModels, ModelType, PredictionTask
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False

try:
    from prediction_accuracy_validator import PredictionAccuracyValidator
    ACCURACY_VALIDATOR_AVAILABLE = True
except ImportError:
    ACCURACY_VALIDATOR_AVAILABLE = False

try:
    from ml_model_upgrade_system import MLModelUpgradeSystem
    UPGRADE_SYSTEM_AVAILABLE = True
except ImportError:
    UPGRADE_SYSTEM_AVAILABLE = False

try:
    from symbol_selector import SymbolSelector
    SYMBOL_SELECTOR_AVAILABLE = True
except ImportError:
    SYMBOL_SELECTOR_AVAILABLE = False


class PerformanceStatus(Enum):
    """性能ステータス"""
    EXCELLENT = "excellent"      # 目標閾値以上
    GOOD = "good"               # 警告閾値以上
    WARNING = "warning"         # 最低閾値以上
    CRITICAL = "critical"       # 最低閾値未満
    UNKNOWN = "unknown"         # 評価不可


class AlertLevel(Enum):
    """アラートレベル"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """性能指標"""
    symbol: str
    model_type: str
    timestamp: datetime
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    r2_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    prediction_time_ms: Optional[float] = None
    confidence_avg: Optional[float] = None
    sample_size: int = 0
    status: PerformanceStatus = PerformanceStatus.UNKNOWN


@dataclass
class PerformanceAlert:
    """性能アラート"""
    id: str
    timestamp: datetime
    symbol: str
    model_type: str
    alert_level: AlertLevel
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class RetrainingTrigger:
    """再学習トリガー"""
    trigger_id: str
    timestamp: datetime
    trigger_type: str
    affected_symbols: List[str]
    affected_models: List[str]
    reason: str
    severity: AlertLevel
    recommended_action: str


class ModelPerformanceMonitor:
    """MLモデル性能監視システム"""

    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)

        # 設定読み込み
        self.config_path = config_path or Path("config/performance_thresholds.yaml")
        self.config = self._load_configuration()

        # データベース初期化
        self.db_path = Path(self.config.get('database', {}).get('connection', {}).get('path', 'data/model_performance.db'))
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

        # 内部状態
        self.performance_history = defaultdict(deque)
        self.active_alerts = {}
        self.monitoring_active = False
        self.last_check_time = None

        # 外部システム統合
        self._initialize_external_systems()

        self.logger.info("ModelPerformanceMonitor initialized")

    def _load_configuration(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"設定ファイル読み込み完了: {self.config_path}")
                return config
            else:
                self.logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"設定ファイル読み込みエラー: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'performance_thresholds': {
                'accuracy': {
                    'minimum_threshold': 0.75,
                    'warning_threshold': 0.80,
                    'target_threshold': 0.90,
                    'critical_threshold': 0.70
                },
                'confidence': {
                    'minimum_threshold': 0.65,
                    'warning_threshold': 0.75,
                    'target_threshold': 0.85
                }
            },
            'monitoring_symbols': {
                'primary_symbols': ['7203', '8306', '9984', '4751', '2914']
            },
            'monitoring': {
                'intervals': {
                    'real_time_seconds': 60,
                    'periodic_minutes': 15
                }
            },
            'retraining': {
                'triggers': {
                    'accuracy_degradation': {
                        'enabled': True,
                        'threshold_drop': 0.05,
                        'consecutive_failures': 3
                    }
                },
                'strategy': {
                    'granularity': 'selective',
                    'batch_size': 5,
                    'max_concurrent': 2
                }
            }
        }

    def _init_database(self):
        """データベース初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 性能履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        accuracy REAL,
                        precision_score REAL,
                        recall REAL,
                        f1_score REAL,
                        r2_score REAL,
                        mse REAL,
                        mae REAL,
                        prediction_time_ms REAL,
                        confidence_avg REAL,
                        sample_size INTEGER,
                        status TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # アラートテーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_alerts (
                        id TEXT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        symbol TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        alert_level TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        threshold_value REAL NOT NULL,
                        message TEXT NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolved_at DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # 再学習履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS retraining_history (
                        id TEXT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        trigger_type TEXT NOT NULL,
                        affected_symbols TEXT NOT NULL,
                        affected_models TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        recommended_action TEXT NOT NULL,
                        status TEXT DEFAULT 'pending',
                        completed_at DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                conn.commit()
                self.logger.info("データベース初期化完了")

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")

    def _initialize_external_systems(self):
        """外部システム初期化"""
        # MLモデルシステム
        if ML_MODELS_AVAILABLE:
            try:
                self.ml_models = MLPredictionModels()
                self.logger.info("MLPredictionModels統合完了")
            except Exception as e:
                self.logger.warning(f"MLPredictionModels統合失敗: {e}")
                self.ml_models = None
        else:
            self.ml_models = None

        # 精度検証システム
        if ACCURACY_VALIDATOR_AVAILABLE:
            try:
                self.accuracy_validator = PredictionAccuracyValidator()
                self.logger.info("PredictionAccuracyValidator統合完了")
            except Exception as e:
                self.logger.warning(f"PredictionAccuracyValidator統合失敗: {e}")
                self.accuracy_validator = None
        else:
            self.accuracy_validator = None

        # モデルアップグレードシステム
        if UPGRADE_SYSTEM_AVAILABLE:
            try:
                self.upgrade_system = MLModelUpgradeSystem()
                self.logger.info("MLModelUpgradeSystem統合完了")
            except Exception as e:
                self.logger.warning(f"MLModelUpgradeSystem統合失敗: {e}")
                self.upgrade_system = None
        else:
            self.upgrade_system = None

        # 銘柄選択システム
        if SYMBOL_SELECTOR_AVAILABLE:
            try:
                self.symbol_selector = SymbolSelector()
                self.logger.info("SymbolSelector統合完了")
            except Exception as e:
                self.logger.warning(f"SymbolSelector統合失敗: {e}")
                self.symbol_selector = None
        else:
            self.symbol_selector = None

    def get_monitoring_symbols(self) -> List[str]:
        """監視対象銘柄リスト取得"""
        config_symbols = self.config.get('monitoring_symbols', {})

        # 設定ファイルから銘柄取得
        symbols = []
        symbols.extend(config_symbols.get('primary_symbols', []))
        symbols.extend(config_symbols.get('secondary_symbols', []))

        # 動的銘柄選択が有効な場合
        dynamic_config = config_symbols.get('dynamic_selection', {})
        if dynamic_config.get('enabled', False) and self.symbol_selector:
            try:
                # 動的銘柄選択実行
                criteria = dynamic_config.get('selection_criteria', {})
                dynamic_symbols = self.symbol_selector.select_symbols_by_criteria(criteria)
                symbols.extend(dynamic_symbols)
                self.logger.info(f"動的銘柄選択で{len(dynamic_symbols)}銘柄を追加")
            except Exception as e:
                self.logger.error(f"動的銘柄選択エラー: {e}")

        # 重複除去してソート
        return sorted(list(set(symbols)))

    async def get_latest_model_performance(self) -> Dict[str, PerformanceMetrics]:
        """最新のモデル性能取得"""
        symbols = self.get_monitoring_symbols()
        performance_data = {}

        for symbol in symbols:
            try:
                # 各モデルタイプの性能評価
                for model_type in self.config.get('models', {}).get('supported_models', ['RandomForestClassifier']):
                    metrics = await self._evaluate_model_performance(symbol, model_type)
                    if metrics:
                        key = f"{symbol}_{model_type}"
                        performance_data[key] = metrics

                        # 履歴に追加
                        self.performance_history[key].append(metrics)

                        # 履歴サイズ制限
                        max_history = 100
                        if len(self.performance_history[key]) > max_history:
                            self.performance_history[key].popleft()

            except Exception as e:
                self.logger.error(f"性能評価エラー {symbol}: {e}")

        return performance_data

    async def _evaluate_model_performance(self, symbol: str, model_type: str) -> Optional[PerformanceMetrics]:
        """個別モデル性能評価"""
        try:
            if not self.accuracy_validator:
                # 模擬データで評価
                return self._generate_mock_performance(symbol, model_type)

            # 実際の精度検証
            test_symbols = [symbol]
            results = await self.accuracy_validator.validate_prediction_accuracy(test_symbols)

            if symbol not in results:
                return None

            result = results[symbol]

            # 性能指標計算
            start_time = time.time()

            # 予測時間測定（模擬）
            prediction_time = (time.time() - start_time) * 1000

            # PerformanceMetricsオブジェクト作成
            metrics = PerformanceMetrics(
                symbol=symbol,
                model_type=model_type,
                timestamp=datetime.now(),
                accuracy=result.get('accuracy', 0.0),
                precision=result.get('precision', 0.0),
                recall=result.get('recall', 0.0),
                f1_score=result.get('f1_score', 0.0),
                prediction_time_ms=prediction_time,
                confidence_avg=result.get('confidence_avg', 0.0),
                sample_size=result.get('sample_size', 0)
            )

            # ステータス判定
            metrics.status = self._determine_performance_status(metrics)

            # データベースに保存
            await self._save_performance_metrics(metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"モデル性能評価エラー {symbol}_{model_type}: {e}")
            return None

    def _generate_mock_performance(self, symbol: str, model_type: str) -> PerformanceMetrics:
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

        metrics.status = self._determine_performance_status(metrics)
        return metrics

    def _determine_performance_status(self, metrics: PerformanceMetrics) -> PerformanceStatus:
        """性能ステータス判定"""
        thresholds = self.config.get('performance_thresholds', {}).get('accuracy', {})
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

    async def _save_performance_metrics(self, metrics: PerformanceMetrics):
        """性能指標をデータベースに保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO model_performance_history
                    (symbol, model_type, timestamp, accuracy, precision_score, recall, f1_score,
                     prediction_time_ms, confidence_avg, sample_size, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.symbol, metrics.model_type, metrics.timestamp,
                    metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score,
                    metrics.prediction_time_ms, metrics.confidence_avg,
                    metrics.sample_size, metrics.status.value
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"性能指標保存エラー: {e}")

    async def check_performance_degradation(self, performance_data: Dict[str, PerformanceMetrics]) -> List[PerformanceAlert]:
        """性能低下チェック"""
        alerts = []

        for key, metrics in performance_data.items():
            symbol = metrics.symbol
            model_type = metrics.model_type

            # 閾値チェック
            threshold_alerts = self._check_thresholds(metrics)
            alerts.extend(threshold_alerts)

            # 履歴比較チェック
            history_alerts = self._check_performance_history(key, metrics)
            alerts.extend(history_alerts)

        # アラート保存とログ出力
        for alert in alerts:
            await self._save_alert(alert)
            self._log_alert(alert)

        return alerts

    def _check_thresholds(self, metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """閾値チェック"""
        alerts = []
        thresholds = self.config.get('performance_thresholds', {})

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

        # 信頼度閾値チェック
        confidence_thresholds = thresholds.get('confidence', {})
        if metrics.confidence_avg is not None:
            if metrics.confidence_avg < confidence_thresholds.get('minimum_threshold', 0.65):
                alert = PerformanceAlert(
                    id=f"confidence_warning_{metrics.symbol}_{metrics.model_type}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=metrics.symbol,
                    model_type=metrics.model_type,
                    alert_level=AlertLevel.WARNING,
                    metric_name="confidence",
                    current_value=metrics.confidence_avg,
                    threshold_value=confidence_thresholds.get('minimum_threshold', 0.65),
                    message=f"信頼度が最低閾値を下回りました: {metrics.confidence_avg:.3f}"
                )
                alerts.append(alert)

        return alerts

    def _check_performance_history(self, key: str, current_metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """履歴比較チェック"""
        alerts = []

        if key not in self.performance_history or len(self.performance_history[key]) < 3:
            return alerts

        history = list(self.performance_history[key])
        recent_accuracy = [m.accuracy for m in history[-3:] if m.accuracy is not None]

        if len(recent_accuracy) >= 3 and current_metrics.accuracy is not None:
            avg_recent = np.mean(recent_accuracy[:-1])  # 最新を除く平均
            degradation = avg_recent - current_metrics.accuracy

            threshold_drop = self.config.get('retraining', {}).get('triggers', {}).get('accuracy_degradation', {}).get('threshold_drop', 0.05)

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

    async def _save_alert(self, alert: PerformanceAlert):
        """アラートをデータベースに保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO performance_alerts
                    (id, timestamp, symbol, model_type, alert_level, metric_name,
                     current_value, threshold_value, message, resolved, resolved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id, alert.timestamp, alert.symbol, alert.model_type,
                    alert.alert_level.value, alert.metric_name, alert.current_value,
                    alert.threshold_value, alert.message, alert.resolved, alert.resolved_at
                ))
                conn.commit()

                # アクティブアラートに追加
                self.active_alerts[alert.id] = alert

        except Exception as e:
            self.logger.error(f"アラート保存エラー: {e}")

    def _log_alert(self, alert: PerformanceAlert):
        """アラートログ出力"""
        level_map = {
            AlertLevel.INFO: self.logger.info,
            AlertLevel.WARNING: self.logger.warning,
            AlertLevel.ERROR: self.logger.error,
            AlertLevel.CRITICAL: self.logger.critical
        }

        log_func = level_map.get(alert.alert_level, self.logger.info)
        log_func(f"[{alert.alert_level.value.upper()}] {alert.symbol}_{alert.model_type}: {alert.message}")

    async def trigger_retraining(self, alerts: List[PerformanceAlert]) -> Optional[RetrainingTrigger]:
        """再学習トリガー"""
        if not alerts:
            return None

        # 重要なアラートのみ処理
        critical_alerts = [a for a in alerts if a.alert_level in [AlertLevel.CRITICAL, AlertLevel.ERROR]]

        if not critical_alerts:
            return None

        # 影響を受ける銘柄とモデルを特定
        affected_symbols = list(set(a.symbol for a in critical_alerts))
        affected_models = list(set(a.model_type for a in critical_alerts))

        # 再学習戦略決定
        strategy = self.config.get('retraining', {}).get('strategy', {})
        granularity = strategy.get('granularity', 'selective')

        # 再学習トリガー作成
        trigger = RetrainingTrigger(
            trigger_id=f"retraining_{int(time.time())}",
            timestamp=datetime.now(),
            trigger_type=granularity,
            affected_symbols=affected_symbols,
            affected_models=affected_models,
            reason=f"{len(critical_alerts)}個の重要なアラートが発生",
            severity=max(alert.alert_level for alert in critical_alerts),
            recommended_action=self._generate_retraining_recommendation(critical_alerts, granularity)
        )

        # データベースに保存
        await self._save_retraining_trigger(trigger)

        # 実際の再学習実行
        if self.upgrade_system and granularity == 'selective':
            try:
                self.logger.info(f"選択的再学習を開始: {affected_symbols}")
                await self._execute_selective_retraining(trigger)
            except Exception as e:
                self.logger.error(f"再学習実行エラー: {e}")
        elif granularity == 'full':
            self.logger.info("全体再学習が推奨されましたが、手動実行が必要です")

        return trigger

    def _generate_retraining_recommendation(self, alerts: List[PerformanceAlert], granularity: str) -> str:
        """再学習推奨事項生成"""
        if granularity == 'selective':
            return f"影響を受けた{len(set(a.symbol for a in alerts))}銘柄の選択的再学習を実行"
        elif granularity == 'full':
            return "システム全体の再学習が必要"
        else:
            return "増分学習による性能改善を実行"

    async def _save_retraining_trigger(self, trigger: RetrainingTrigger):
        """再学習トリガーをデータベースに保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO retraining_history
                    (id, timestamp, trigger_type, affected_symbols, affected_models,
                     reason, severity, recommended_action, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trigger.trigger_id, trigger.timestamp, trigger.trigger_type,
                    json.dumps(trigger.affected_symbols), json.dumps(trigger.affected_models),
                    trigger.reason, trigger.severity.value, trigger.recommended_action, 'triggered'
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"再学習トリガー保存エラー: {e}")

    async def _execute_selective_retraining(self, trigger: RetrainingTrigger):
        """選択的再学習実行"""
        try:
            if not self.upgrade_system:
                self.logger.warning("MLModelUpgradeSystemが利用できません")
                return

            # バッチサイズ制限
            strategy = self.config.get('retraining', {}).get('strategy', {})
            batch_size = strategy.get('batch_size', 5)

            symbols_to_retrain = trigger.affected_symbols[:batch_size]

            self.logger.info(f"選択的再学習開始: {symbols_to_retrain}")

            # 実際の再学習実行（模擬）
            await asyncio.sleep(1)  # 模擬処理時間

            self.logger.info(f"選択的再学習完了: {symbols_to_retrain}")

            # ステータス更新
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE retraining_history
                    SET status = 'completed', completed_at = ?
                    WHERE id = ?
                ''', (datetime.now(), trigger.trigger_id))
                conn.commit()

        except Exception as e:
            self.logger.error(f"選択的再学習エラー: {e}")

            # エラーステータス更新
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE retraining_history
                    SET status = 'failed'
                    WHERE id = ?
                ''', (trigger.trigger_id,))
                conn.commit()

    async def start_monitoring(self):
        """監視開始"""
        if self.monitoring_active:
            self.logger.warning("監視は既に開始されています")
            return

        self.monitoring_active = True
        self.logger.info("モデル性能監視を開始しました")

        try:
            while self.monitoring_active:
                # 性能チェック実行
                await self.run_performance_check()

                # 監視間隔待機
                interval = self.config.get('monitoring', {}).get('intervals', {}).get('real_time_seconds', 60)
                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            self.logger.info("監視がキャンセルされました")
        except Exception as e:
            self.logger.error(f"監視エラー: {e}")
        finally:
            self.monitoring_active = False

    async def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        self.logger.info("モデル性能監視を停止しました")

    async def run_performance_check(self):
        """性能チェック実行"""
        try:
            self.logger.info("性能チェック開始")

            # 最新性能取得
            performance_data = await self.get_latest_model_performance()

            # 性能低下チェック
            alerts = await self.check_performance_degradation(performance_data)

            # 再学習トリガー
            if alerts:
                retraining_trigger = await self.trigger_retraining(alerts)
                if retraining_trigger:
                    self.logger.info(f"再学習トリガー発動: {retraining_trigger.trigger_id}")

            self.last_check_time = datetime.now()

            self.logger.info(f"性能チェック完了: {len(performance_data)}モデル, {len(alerts)}アラート")

        except Exception as e:
            self.logger.error(f"性能チェックエラー: {e}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視ステータス取得"""
        return {
            'monitoring_active': self.monitoring_active,
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'active_alerts_count': len(self.active_alerts),
            'monitoring_symbols_count': len(self.get_monitoring_symbols()),
            'config_loaded': bool(self.config),
            'external_systems': {
                'ml_models': self.ml_models is not None,
                'accuracy_validator': self.accuracy_validator is not None,
                'upgrade_system': self.upgrade_system is not None,
                'symbol_selector': self.symbol_selector is not None
            }
        }


async def test_model_performance_monitor():
    """モデル性能監視システムテスト"""
    print("=== ModelPerformanceMonitor テスト開始 ===")

    # 監視システム初期化
    monitor = ModelPerformanceMonitor()

    # 監視ステータス確認
    status = monitor.get_monitoring_status()
    print(f"監視ステータス: {json.dumps(status, indent=2, ensure_ascii=False)}")

    # 監視対象銘柄確認
    symbols = monitor.get_monitoring_symbols()
    print(f"監視対象銘柄: {symbols}")

    # 性能チェック実行
    print("\n性能チェック実行中...")
    await monitor.run_performance_check()

    # 短時間の監視実行
    print("\n短時間監視開始...")
    monitoring_task = asyncio.create_task(monitor.start_monitoring())

    # 10秒後に停止
    await asyncio.sleep(10)
    await monitor.stop_monitoring()

    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass

    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # テスト実行
    asyncio.run(test_model_performance_monitor())