#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Monitor - モデル性能監視システム（改善版）

MLモデルの性能を監視し、閾値を下回った場合に段階的再学習をトリガーする
Issue #857対応：外部設定対応、動的監視、詳細な再学習制御

改善項目：
1. 性能閾値の完全外部化
2. 動的監視対象銘柄システム
3. 段階的再学習トリガー（銘柄別、部分、全体）
4. symbol_selector連携
5. Windows環境対策統合
"""

import asyncio
import sqlite3
import yaml
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Dict, List, Any, Set
from dataclasses import dataclass
from enum import Enum

# 共通ユーティリティのインポート
from src.day_trade.utils.encoding_fix import apply_windows_encoding_fix

# 既存システムとの連携
try:
    from ml_model_upgrade_system import ml_upgrade_system
    from prediction_accuracy_validator import PredictionAccuracyValidator
    ML_SYSTEMS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Import warning: {e}. Some functionality may be limited.")
    ml_upgrade_system = None
    PredictionAccuracyValidator = None
    ML_SYSTEMS_AVAILABLE = False

try:
    from symbol_selector import SymbolSelector
    SYMBOL_SELECTOR_AVAILABLE = True
except ImportError:
    SYMBOL_SELECTOR_AVAILABLE = False

# Windows環境対応
apply_windows_encoding_fix()

# ロギング設定
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """アラートレベル"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RetrainingScope(Enum):
    """再学習スコープ"""
    NONE = "none"
    INCREMENTAL = "incremental"
    SYMBOL = "symbol"
    PARTIAL = "partial"
    GLOBAL = "global"


@dataclass
class PerformanceMetrics:
    """性能メトリクス情報"""
    symbol: str
    accuracy: float
    prediction_accuracy: float = 0.0
    return_prediction: float = 0.0
    volatility_prediction: float = 0.0
    sample_count: int = 0
    timestamp: datetime = None
    source: str = "validator"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class RetrainingResult:
    """再学習結果情報"""
    triggered: bool
    scope: RetrainingScope
    affected_symbols: List[str]
    improvement: float = 0.0
    duration: float = 0.0
    error: Optional[str] = None
    estimated_time: float = 0.0
    actual_benefit: float = 0.0


@dataclass
class PerformanceAlert:
    """性能アラート"""
    timestamp: datetime
    level: AlertLevel
    symbol: str
    metric: str
    current_value: float
    threshold: float
    message: str
    recommended_action: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EnhancedPerformanceConfigManager:
    """改善版性能設定管理クラス"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = (
            config_path or Path("config/performance_monitoring.yaml")
        )
        self.config = {}
        self.last_loaded = None
        self._load_configuration()

    def _load_configuration(self):
        """設定ファイルを読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                self.last_loaded = datetime.now()
                logger.info(f"設定ファイルを読み込みました: {self.config_path}")
            else:
                logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
                self._create_default_config()

        except Exception as e:
            logger.error(f"設定読み込みエラー: {e}")
            self._load_default_config()

    def _create_default_config(self):
        """デフォルト設定ファイルの作成"""
        default_config = {
            'performance_thresholds': {
                'accuracy': 90.0,
                'prediction_accuracy': 85.0,
                'return_prediction': 75.0,
                'volatility_prediction': 80.0
            },
            'monitoring': {
                'default_symbols': ["7203", "8306", "4751"],
                'dynamic_monitoring': {'enabled': True},
                'validation_hours': 168,
                'min_samples': 10
            },
            'retraining': {
                'granular_mode': True,
                'cooldown_hours': 24,
                'global_threshold': 80.0,
                'symbol_specific_threshold': 85.0,
                'partial_threshold': 88.0
            },
            'system': {
                'log_level': 'INFO'
            }
        }

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    default_config, f, default_flow_style=False, allow_unicode=True
                )
            logger.info(f"デフォルト設定ファイルを作成: {self.config_path}")
            self.config = default_config
        except Exception as e:
            logger.error(f"デフォルト設定ファイル作成エラー: {e}")
            self._load_default_config()

    def _load_default_config(self):
        """デフォルト設定を読み込み"""
        self.config = {
            'performance_thresholds': {
                'accuracy': 90.0,
                'prediction_accuracy': 85.0,
                'return_prediction': 75.0,
                'volatility_prediction': 80.0
            },
            'monitoring': {
                'default_symbols': ["7203", "8306", "4751"],
                'dynamic_monitoring': {'enabled': False},
                'validation_hours': 168,
                'min_samples': 10
            },
            'retraining': {
                'granular_mode': True,
                'cooldown_hours': 24,
                'global_threshold': 80.0,
                'symbol_specific_threshold': 85.0,
                'partial_threshold': 88.0
            },
            'system': {
                'log_level': 'INFO'
            }
        }
        logger.info("デフォルト設定を使用します")

    def get_threshold(self, metric: str, symbol: str = None) -> float:
        """閾値取得（銘柄別対応）"""
        thresholds = self.config.get('performance_thresholds', {})

        # 銘柄別閾値チェック
        if symbol and 'symbol_specific' in thresholds:
            symbol_category = self._categorize_symbol(symbol)
            symbol_thresholds = thresholds['symbol_specific'].get(
                symbol_category, {}
            )
            if metric in symbol_thresholds:
                return symbol_thresholds[metric]

        # デフォルト閾値
        return thresholds.get(metric, 90.0)

    def _categorize_symbol(self, symbol: str) -> str:
        """銘柄カテゴリ分類"""
        high_volume = ["7203", "8306", "9984", "6758"]
        if symbol in high_volume:
            return "high_volume_stocks"
        return "mid_volume_stocks"

    def get_monitoring_config(self) -> Dict[str, Any]:
        """監視設定取得"""
        return self.config.get('monitoring', {})

    def get_retraining_config(self) -> Dict[str, Any]:
        """再学習設定取得"""
        return self.config.get('retraining', {})

    def reload_if_modified(self):
        """設定ファイルが更新されていれば再読み込み"""
        try:
            if self.config_path.exists():
                file_mtime = datetime.fromtimestamp(
                    self.config_path.stat().st_mtime
                )
                if self.last_loaded is None or file_mtime > self.last_loaded:
                    self._load_configuration()
        except Exception as e:
            logger.error(f"設定ファイル更新チェックエラー: {e}")


class DynamicSymbolManager:
    """動的監視対象銘柄管理クラス"""

    def __init__(self, config_manager: EnhancedPerformanceConfigManager):
        self.config_manager = config_manager
        self.symbol_selector = None
        self.last_update = None
        self.current_symbols = set()

        # symbol_selector連携
        if SYMBOL_SELECTOR_AVAILABLE:
            try:
                self.symbol_selector = SymbolSelector()
                logger.info("SymbolSelectorとの連携を開始しました")
            except Exception as e:
                logger.warning(f"SymbolSelector初期化失敗: {e}")

    def get_monitoring_symbols(self) -> List[str]:
        """現在の監視対象銘柄を取得"""
        monitoring_config = self.config_manager.get_monitoring_config()

        # 基本監視銘柄
        default_symbols = ["7203", "8306", "4751"]
        base_symbols = set(monitoring_config.get('default_symbols', default_symbols))

        # 動的監視が有効な場合
        dynamic_config = monitoring_config.get('dynamic_monitoring', {})
        if dynamic_config.get('enabled', False):
            dynamic_symbols = self._get_dynamic_symbols()
            base_symbols.update(dynamic_symbols)

            # 最大銘柄数制限
            max_symbols = dynamic_config.get('max_symbols', 10)
            if len(base_symbols) > max_symbols:
                # 優先度順にソート（将来的にはより高度な選択ロジック）
                base_symbols = set(list(base_symbols)[:max_symbols])

        return list(base_symbols)

    def _get_dynamic_symbols(self) -> Set[str]:
        """動的監視銘柄を取得"""
        dynamic_symbols = set()

        try:
            if self.symbol_selector:
                # symbol_selectorから活発な銘柄を取得
                active_symbols = self.symbol_selector.get_recommended_symbols(
                    limit=5
                )
                dynamic_symbols.update(active_symbols)

                # 高ボラティリティ銘柄を追加
                volatile_symbols = self.symbol_selector.get_high_volatility_symbols(
                    limit=3
                )
                dynamic_symbols.update(volatile_symbols)

        except Exception as e:
            logger.error(f"動的監視銘柄取得エラー: {e}")

        return dynamic_symbols

    def should_update_symbols(self) -> bool:
        """監視銘柄を更新すべきかチェック"""
        if self.last_update is None:
            return True

        monitoring_config = self.config_manager.get_monitoring_config()
        dynamic_config = monitoring_config.get('dynamic_monitoring', {})
        update_interval = dynamic_config.get('update_interval_hours', 24)

        elapsed = datetime.now() - self.last_update
        return elapsed.total_seconds() > (update_interval * 3600)


class GranularRetrainingManager:
    """段階的再学習管理クラス"""

    def __init__(self, config_manager: EnhancedPerformanceConfigManager):
        self.config_manager = config_manager
        self.last_retraining = {}  # スコープ別の最終実行時刻
        self.retraining_history = []

    def determine_retraining_scope(
            self, symbol_performances: Dict[str, PerformanceMetrics]
    ) -> RetrainingScope:
        """再学習スコープを決定"""
        retraining_config = self.config_manager.get_retraining_config()

        if not symbol_performances:
            return RetrainingScope.NONE

        # 全体性能計算
        total_accuracy = sum(p.accuracy for p in symbol_performances.values())
        overall_accuracy = total_accuracy / len(symbol_performances)

        # グローバル再学習判定
        global_threshold = retraining_config.get('global_threshold', 80.0)
        if overall_accuracy < global_threshold:
            return RetrainingScope.GLOBAL

        # 段階的再学習が有効な場合
        if retraining_config.get('granular_mode', True):
            # 部分再学習判定
            partial_threshold = retraining_config.get('partial_threshold', 88.0)
            low_performance_count = sum(1 for p in symbol_performances.values()
                                        if p.accuracy < partial_threshold)

            if low_performance_count >= len(symbol_performances) * 0.3:  # 30%以上が低性能
                return RetrainingScope.PARTIAL

            # 銘柄別再学習判定
            symbol_threshold = retraining_config.get('symbol_specific_threshold', 85.0)
            critical_symbols = [s for s, p in symbol_performances.items()
                                if p.accuracy < symbol_threshold]

            if critical_symbols:
                return RetrainingScope.SYMBOL

        # 増分学習判定（データ変化検出など）
        if self._should_incremental_update(symbol_performances):
            return RetrainingScope.INCREMENTAL

        return RetrainingScope.NONE

    def _should_incremental_update(
            self, symbol_performances: Dict[str, PerformanceMetrics]
    ) -> bool:
        """増分学習の必要性判定"""
        # 実装例：性能の微小な劣化や新しいデータパターンの検出
        # ここでは簡略化して、5%以内の劣化で増分学習を提案
        for perf in symbol_performances.values():
            if 85.0 <= perf.accuracy < 90.0:  # 軽微な劣化
                return True
        return False

    def check_cooldown_period(self, scope: RetrainingScope) -> bool:
        """冷却期間チェック"""
        if scope == RetrainingScope.NONE:
            return True

        retraining_config = self.config_manager.get_retraining_config()
        cooldown_hours = retraining_config.get('cooldown_hours', 24)

        # スコープ別の冷却期間調整
        scope_cooldown = {
            RetrainingScope.INCREMENTAL: cooldown_hours * 0.25,  # 6時間
            RetrainingScope.SYMBOL: cooldown_hours * 0.5,        # 12時間
            RetrainingScope.PARTIAL: cooldown_hours * 0.75,      # 18時間
            RetrainingScope.GLOBAL: cooldown_hours               # 24時間
        }

        required_cooldown = scope_cooldown.get(scope, cooldown_hours)
        last_time = self.last_retraining.get(scope)

        if last_time is None:
            return True

        elapsed = datetime.now() - last_time
        return elapsed.total_seconds() > (required_cooldown * 3600)

    def update_retraining_time(self, scope: RetrainingScope):
        """再学習実行時刻を更新"""
        self.last_retraining[scope] = datetime.now()


class EnhancedModelPerformanceMonitor:
    """
    改善版MLモデル性能監視システム

    主な機能:
    - 完全外部設定による性能閾値管理
    - 動的監視対象銘柄システム
    - 段階的再学習トリガー（増分・銘柄別・部分・全体）
    - symbol_selector連携
    - 詳細なパフォーマンス追跡と分析
    """

    def __init__(self, config_path: Optional[Path] = None,
                 upgrade_db_path: Optional[Path] = None,
                 advanced_ml_db_path: Optional[Path] = None):

        self.config_manager = EnhancedPerformanceConfigManager(config_path)
        self.symbol_manager = DynamicSymbolManager(self.config_manager)
        self.retraining_manager = GranularRetrainingManager(self.config_manager)

        # データベースパス
        self.upgrade_db_path = (
            upgrade_db_path or Path("ml_models_data/upgrade_system.db")
        )
        self.advanced_ml_db_path = (
            advanced_ml_db_path or Path("ml_models_data/advanced_ml_predictions.db")
        )

        # 性能追跡
        self.performance_cache = {}
        self.alerts = []

        # 外部システム初期化
        if PredictionAccuracyValidator:
            self.accuracy_validator = PredictionAccuracyValidator()
        else:
            self.accuracy_validator = None
            logger.warning("PredictionAccuracyValidatorが利用できません")

        # データベース初期化
        self._init_database()

        # ログレベル設定
        log_level = self.config_manager.config.get(
            'system', {}
        ).get('log_level', 'INFO')
        logger.setLevel(getattr(logging, log_level))

        logger.info("Enhanced Model Performance Monitor initialized")

    def _init_database(self):
        """データベースの初期化"""
        try:
            with sqlite3.connect(self.upgrade_db_path) as conn:
                cursor = conn.cursor()

                # 改善版性能履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS enhanced_performance_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        value REAL NOT NULL,
                        threshold REAL,
                        sample_count INTEGER,
                        timestamp TEXT NOT NULL,
                        source TEXT DEFAULT 'validator',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # 改善版再学習履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS enhanced_retraining_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trigger_time TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        affected_symbols TEXT,
                        improvement REAL,
                        duration REAL,
                        estimated_time REAL,
                        actual_benefit REAL,
                        status TEXT DEFAULT 'pending',
                        error_message TEXT,
                        config_snapshot TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # アラート履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        metric TEXT NOT NULL,
                        current_value REAL,
                        threshold REAL,
                        message TEXT,
                        recommended_action TEXT,
                        resolved BOOLEAN DEFAULT FALSE,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                conn.commit()
                logger.info("データベーステーブルを初期化しました")

        except Exception as e:
            logger.error(f"データベース初期化に失敗しました: {e}")
            raise

    async def get_latest_model_performance(self,
                                           symbols: Optional[List[str]] = None,
                                           force_refresh: bool = False
                                           ) -> Dict[str, PerformanceMetrics]:
        """
        最新のモデル性能を取得

        Args:
            symbols: 評価対象銘柄（Noneの場合は動的取得）
            force_refresh: キャッシュを無視して強制更新

        Returns:
            Dict[str, PerformanceMetrics]: 銘柄別性能メトリクス
        """
        # 設定ファイル更新チェック
        self.config_manager.reload_if_modified()

        # 監視対象銘柄決定
        if symbols is None:
            symbols = self.symbol_manager.get_monitoring_symbols()

        logger.info(f"モデル性能を取得します - 対象銘柄: {symbols}")

        # キャッシュチェック
        if not force_refresh and self._is_cache_valid():
            cached_results = {s: self.performance_cache[s] for s in symbols
                              if s in self.performance_cache}
            if len(cached_results) == len(symbols):
                logger.info("キャッシュから性能データを取得しました")
                return cached_results

        if not self.accuracy_validator:
            logger.warning("PredictionAccuracyValidatorが利用できません")
            return {}

        performance_results = {}
        monitoring_config = self.config_manager.get_monitoring_config()
        validation_hours = monitoring_config.get('validation_hours', 168)

        try:
            # 銘柄別性能評価
            for symbol in symbols:
                try:
                    # 個別銘柄の性能評価
                    metrics = await self.accuracy_validator.validate_current_system_accuracy(  # noqa: E501
                        [symbol], validation_hours
                    )

                    performance = PerformanceMetrics(
                        symbol=symbol,
                        accuracy=metrics.overall_accuracy,
                        prediction_accuracy=getattr(
                            metrics, 'prediction_accuracy', 0.0
                        ),
                        return_prediction=getattr(metrics, 'return_prediction', 0.0),
                        volatility_prediction=getattr(
                            metrics, 'volatility_prediction', 0.0
                        ),
                        sample_count=getattr(metrics, 'sample_count', 0),
                        source='enhanced_validator'
                    )

                    performance_results[symbol] = performance
                    self.performance_cache[symbol] = performance

                    # 性能データをデータベースに記録
                    await self._record_enhanced_performance_history(performance)

                    # 閾値チェックとアラート生成
                    await self._check_and_generate_alerts(performance)

                except Exception as e:
                    logger.error(f"銘柄 {symbol} の性能評価に失敗: {e}")
                    continue

            logger.info(f"モデル性能を取得しました - {len(performance_results)}銘柄")
            return performance_results

        except Exception as e:
            logger.error(f"モデル性能の取得に失敗しました: {e}")
            return {}

    def _is_cache_valid(self) -> bool:
        """キャッシュの有効性チェック"""
        if not self.performance_cache:
            return False

        # 最新エントリの時刻をチェック
        latest_time = max(p.timestamp for p in self.performance_cache.values())
        elapsed = datetime.now() - latest_time

        # 1時間以内なら有効
        return elapsed.total_seconds() < 3600

    async def _record_enhanced_performance_history(
        self, performance: PerformanceMetrics
    ):
        """改善版性能履歴記録"""
        try:
            with sqlite3.connect(self.upgrade_db_path) as conn:
                cursor = conn.cursor()
                timestamp = performance.timestamp.isoformat()

                # メトリクス別に記録
                metrics = {
                    'accuracy': performance.accuracy,
                    'prediction_accuracy': performance.prediction_accuracy,
                    'return_prediction': performance.return_prediction,
                    'volatility_prediction': performance.volatility_prediction
                }

                for metric_name, value in metrics.items():
                    threshold = self.config_manager.get_threshold(
                        metric_name, performance.symbol
                    )

                    cursor.execute('''
                        INSERT INTO enhanced_performance_history
                        (
                            symbol, metric_name, value, threshold,
                            sample_count, timestamp, source
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (performance.symbol, metric_name, value, threshold,
                          performance.sample_count, timestamp, performance.source))

                conn.commit()

        except Exception as e:
            logger.error(f"改善版性能履歴の記録に失敗: {e}")

    async def _check_and_generate_alerts(self, performance: PerformanceMetrics):
        """性能アラートチェックと生成"""
        metrics_to_check = {
            'accuracy': performance.accuracy,
            'prediction_accuracy': performance.prediction_accuracy,
            'return_prediction': performance.return_prediction,
            'volatility_prediction': performance.volatility_prediction
        }

        for metric_name, value in metrics_to_check.items():
            threshold = self.config_manager.get_threshold(
                metric_name, performance.symbol
            )

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
                    level=level,
                    symbol=performance.symbol,
                    metric=metric_name,
                    current_value=value,
                    threshold=threshold,
                    message=(
                        f"{metric_name}が閾値を下回りました: "
                        f"{value:.2f}% < {threshold:.2f}%"
                    ),
                    recommended_action=action
                )

                self.alerts.append(alert)
                await self._record_alert(alert)

                logger.warning(
                    f"アラート生成: {performance.symbol} {metric_name} "
                    f"{value:.2f}% < {threshold:.2f}%"
                )

    async def _record_alert(self, alert: PerformanceAlert):
        """アラートをデータベースに記録"""
        try:
            with sqlite3.connect(self.upgrade_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_alerts
                    (timestamp, level, symbol, metric, current_value, threshold,
                     message, recommended_action)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.timestamp.isoformat(),
                    alert.level.value,
                    alert.symbol,
                    alert.metric,
                    alert.current_value,
                    alert.threshold,
                    alert.message,
                    alert.recommended_action
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"アラート記録に失敗: {e}")

    async def check_and_trigger_enhanced_retraining(self) -> RetrainingResult:
        """
        改善版性能チェックと段階的再学習のトリガー

        Returns:
            RetrainingResult: 詳細な再学習実行結果
        """
        logger.info("改善版モデル性能監視を開始します")

        # 最新性能の取得
        symbol_performances = await self.get_latest_model_performance()

        if not symbol_performances:
            logger.warning("性能データがないため、再学習チェックをスキップします")
            return RetrainingResult(
                triggered=False,
                scope=RetrainingScope.NONE,
                affected_symbols=[]
            )

        # 再学習スコープ決定
        retraining_scope = self.retraining_manager.determine_retraining_scope(
            symbol_performances
        )

        if retraining_scope == RetrainingScope.NONE:
            logger.info("性能は閾値以上です。再学習は不要です")
            return RetrainingResult(
                triggered=False,
                scope=RetrainingScope.NONE,
                affected_symbols=list(symbol_performances.keys())
            )

        # 冷却期間チェック
        if not self.retraining_manager.check_cooldown_period(retraining_scope):
            logger.info(f"{retraining_scope.value}再学習は冷却期間中のためスキップします")
            return RetrainingResult(
                triggered=False,
                scope=retraining_scope,
                affected_symbols=list(symbol_performances.keys())
            )

        # 再学習の実行
        result = await self._execute_enhanced_retraining(
            retraining_scope, symbol_performances
        )

        # 結果をデータベースに記録
        await self._record_enhanced_retraining_result(
            result
        )

        # 再学習時刻を更新
        if result.triggered:
            self.retraining_manager.update_retraining_time(retraining_scope)

        return result

    async def _execute_enhanced_retraining(
            self, scope: RetrainingScope,
            symbol_performances: Dict[str, PerformanceMetrics]
    ) -> RetrainingResult:
        """改善版再学習の実行"""
        if not ml_upgrade_system:
            return RetrainingResult(
                triggered=False,
                scope=scope,
                affected_symbols=list(symbol_performances.keys()),
                error="ml_upgrade_system not available"
            )

        start_time = datetime.now()
        affected_symbols = list(symbol_performances.keys())

        # 再学習設定から推定時間を取得
        retraining_config = self.config_manager.get_retraining_config()
        scopes_config = retraining_config.get('scopes', {})
        scope_config = scopes_config.get(scope.value, {})
        estimated_time = scope_config.get('estimated_time', 1800)  # デフォルト30分

        try:
            logger.info(f"{scope.value}再学習を開始します - 対象: {affected_symbols}")
            logger.info(f"推定実行時間: {estimated_time}秒")

            # スコープに応じた再学習実行
            if scope == RetrainingScope.GLOBAL:
                report = await ml_upgrade_system.run_complete_system_upgrade()
                await ml_upgrade_system.integrate_best_models(report)
                improvement = report.overall_improvement

            elif scope == RetrainingScope.PARTIAL:
                # 低性能銘柄群の部分再学習
                low_perf_symbols = [
                    s for s, p in symbol_performances.items()
                    if p.accuracy < self.config_manager.get_threshold('accuracy', s)
                ]
                affected_symbols = low_perf_symbols

                # TODO: ml_upgrade_systemに部分再学習機能を追加
                report = await ml_upgrade_system.run_complete_system_upgrade()
                await ml_upgrade_system.integrate_best_models(report)
                # 部分再学習の効果調整
                improvement = report.overall_improvement * 0.7

            elif scope == RetrainingScope.SYMBOL:
                # 特定銘柄の再学習
                critical_symbols = [
                    s for s, p in symbol_performances.items()
                    if p.accuracy < self.config_manager.get_threshold('accuracy', s)
                ]
                affected_symbols = critical_symbols

                # TODO: ml_upgrade_systemに銘柄指定再学習機能を追加
                report = await ml_upgrade_system.run_complete_system_upgrade()
                await ml_upgrade_system.integrate_best_models(report)
                improvement = report.overall_improvement * 0.5  # 銘柄別再学習の効果調整

            elif scope == RetrainingScope.INCREMENTAL:
                # 増分学習
                # TODO: 増分学習機能の実装
                logger.info("増分学習は現在開発中です。通常の再学習を実行します")
                report = await ml_upgrade_system.run_complete_system_upgrade()
                await ml_upgrade_system.integrate_best_models(report)
                improvement = report.overall_improvement * 0.3  # 増分学習の効果調整

            else:
                return RetrainingResult(
                    triggered=False,
                    scope=scope,
                    affected_symbols=affected_symbols,
                    error=f"Unknown retraining scope: {scope.value}"
                )

            # 実行時間の計算
            duration = (datetime.now() - start_time).total_seconds()

            # 実際の効果測定（簡略化）
            actual_benefit = improvement if improvement > 0 else 0.0

            logger.info(f"再学習が完了しました - 改善: {improvement:.2f}%, 所要時間: {duration:.1f}秒")

            return RetrainingResult(
                triggered=True,
                scope=scope,
                affected_symbols=affected_symbols,
                improvement=improvement,
                duration=duration,
                estimated_time=estimated_time,
                actual_benefit=actual_benefit
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"再学習実行エラー: {e}"
            logger.error(error_msg)

            return RetrainingResult(
                triggered=True,
                scope=scope,
                affected_symbols=affected_symbols,
                duration=duration,
                estimated_time=estimated_time,
                error=error_msg
            )

    async def _record_enhanced_retraining_result(self, result: RetrainingResult):
        """改善版再学習結果をデータベースに記録"""
        try:
            with sqlite3.connect(self.upgrade_db_path) as conn:
                cursor = conn.cursor()

                # 設定スナップショットを保存
                config_snapshot = yaml.dump(self.config_manager.config)

                cursor.execute('''
                    INSERT INTO enhanced_retraining_history
                    (trigger_time, scope, affected_symbols, improvement,
                     duration, estimated_time, actual_benefit, status,
                     error_message, config_snapshot)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    result.scope.value,
                    ','.join(result.affected_symbols),
                    result.improvement,
                    result.duration,
                    result.estimated_time,
                    result.actual_benefit,
                    'success' if result.error is None else 'error',
                    result.error,
                    config_snapshot
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"改善版再学習結果の記録に失敗: {e}")

    def get_enhanced_performance_summary(self) -> Dict[str, Any]:
        """改善版性能監視の要約情報を取得"""
        try:
            with sqlite3.connect(self.upgrade_db_path) as conn:
                cursor = conn.cursor()

                # 最新の性能情報（銘柄別）
                cursor.execute('''
                    SELECT symbol, metric_name, value, threshold, timestamp
                    FROM enhanced_performance_history
                    WHERE timestamp >= datetime('now', '-24 hours')
                    ORDER BY timestamp DESC
                    LIMIT 50
                ''')
                recent_performance = cursor.fetchall()

                # 再学習履歴
                cursor.execute('''
                    SELECT scope, improvement, status, trigger_time, duration,
                           estimated_time, actual_benefit
                    FROM enhanced_retraining_history
                    ORDER BY trigger_time DESC
                    LIMIT 10
                ''')
                recent_retraining = cursor.fetchall()

                # アクティブなアラート
                cursor.execute('''
                    SELECT level, symbol, metric, current_value, threshold, message
                    FROM performance_alerts
                    WHERE resolved = FALSE
                    ORDER BY timestamp DESC
                    LIMIT 20
                ''')
                active_alerts = cursor.fetchall()

                # 現在の監視銘柄
                current_symbols = self.symbol_manager.get_monitoring_symbols()

                return {
                    'timestamp': datetime.now().isoformat(),
                    'config_path': str(self.config_manager.config_path),
                    'monitoring_symbols': current_symbols,
                    'symbol_count': len(current_symbols),
                    'dynamic_monitoring': (
                        self.config_manager.get_monitoring_config()
                        .get('dynamic_monitoring', {})
                        .get('enabled', False)
                    ),
                    'recent_performance': recent_performance,
                    'recent_retraining': recent_retraining,
                    'active_alerts': active_alerts,
                    'alert_count': len(active_alerts),
                    'cache_valid': self._is_cache_valid(),
                    'integrations': {
                        'symbol_selector': SYMBOL_SELECTOR_AVAILABLE,
                        'ml_systems': ML_SYSTEMS_AVAILABLE
                    }
                }

        except Exception as e:
            logger.error(f"改善版サマリー情報の取得に失敗: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }


# ユーティリティ関数
def create_enhanced_performance_monitor(
        config_path: Optional[str] = None
) -> EnhancedModelPerformanceMonitor:
    """EnhancedModelPerformanceMonitorインスタンスの作成"""
    path = Path(config_path) if config_path else None
    return EnhancedModelPerformanceMonitor(config_path=path)


if __name__ == "__main__":
    # 基本的な動作確認
    async def main():
        logger.info("Enhanced ModelPerformanceMonitor テスト開始")

        monitor = create_enhanced_performance_monitor()

        # 性能取得テスト
        performance_data = await monitor.get_latest_model_performance()
        logger.info(f"性能データ取得: {len(performance_data)}銘柄")

        # 再学習チェックテスト
        result = await monitor.check_and_trigger_enhanced_retraining()
        logger.info(f"再学習チェック結果: {result}")

        # サマリー取得テスト
        summary = monitor.get_enhanced_performance_summary()
        symbol_count = summary.get('symbol_count', 0)
        alert_count = summary.get('alert_count', 0)
        logger.info(f"サマリー: 監視銘柄{symbol_count}件, アラート{alert_count}件")

    # テスト実行
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
