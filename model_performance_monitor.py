#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Monitor - モデル性能監視システム

MLモデルの性能を監視し、閾値を下回った場合に再学習をトリガーする
改善版: 外部設定対応、柔軟な監視対象、段階的再学習対応
"""

import asyncio
import sqlite3
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

# 共通ユーティリティのインポート
from src.day_trade.utils.encoding_utils import setup_windows_encoding

# 既存システムとの連携
try:
    from ml_model_upgrade_system import ml_upgrade_system
    from prediction_accuracy_validator import PredictionAccuracyValidator
except ImportError as e:
    logging.warning(f"Import warning: {e}. Some functionality may be limited.")
    ml_upgrade_system = None
    PredictionAccuracyValidator = None

# Windows環境対応
setup_windows_encoding()

# ロギング設定
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能メトリクス情報"""
    accuracy: float
    prediction_accuracy: float = 0.0
    return_prediction: float = 0.0
    volatility_prediction: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class RetrainingResult:
    """再学習結果情報"""
    triggered: bool
    scope: str  # 'symbol', 'partial', 'global'
    affected_symbols: List[str]
    improvement: float = 0.0
    duration: float = 0.0
    error: Optional[str] = None

class ModelPerformanceMonitor:
    """
    MLモデル性能監視システム

    主な機能:
    - 外部設定ファイルからの閾値管理
    - 動的な監視対象銘柄
    - 段階的再学習トリガー
    - 詳細なパフォーマンス追跡
    """

    def __init__(self, config_path: Optional[Path] = None,
                 upgrade_db_path: Optional[Path] = None,
                 advanced_ml_db_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/performance_thresholds.yaml")
        self.upgrade_db_path = upgrade_db_path or Path("ml_models_data/upgrade_system.db")
        self.advanced_ml_db_path = advanced_ml_db_path or Path("ml_models_data/advanced_ml_predictions.db")

        # 設定とコンポーネントの初期化
        self.config = {}
        self.thresholds = {}
        self.monitoring_symbols = []
        self.last_retraining = {}  # 銘柄別の最終再学習時刻

        # 外部システムの初期化
        if PredictionAccuracyValidator:
            self.accuracy_validator = PredictionAccuracyValidator()
        else:
            self.accuracy_validator = None

        self._init_database()
        self.load_configuration()

    def _init_database(self):
        """データベースの初期化"""
        try:
            with sqlite3.connect(self.upgrade_db_path) as conn:
                cursor = conn.cursor()

                # 性能閾値テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_thresholds (
                        metric_name TEXT PRIMARY KEY,
                        threshold_value REAL,
                        last_updated TEXT,
                        source TEXT DEFAULT 'config'
                    )
                ''')

                # 性能履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        metric_name TEXT,
                        value REAL,
                        timestamp TEXT,
                        source TEXT
                    )
                ''')

                # 再学習履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS retraining_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trigger_time TEXT,
                        scope TEXT,
                        affected_symbols TEXT,
                        improvement REAL,
                        duration REAL,
                        status TEXT,
                        error_message TEXT
                    )
                ''')

                conn.commit()
                logger.info("データベーステーブルを初期化しました")

        except Exception as e:
            logger.error(f"データベース初期化に失敗しました: {e}")
            raise

    def load_configuration(self):
        """外部設定ファイルからの設定読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"設定ファイルを読み込みました: {self.config_path}")
            else:
                logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
                self._create_default_config()

            # 閾値の設定
            self.thresholds = self.config.get('performance_thresholds', {
                'accuracy': 90.0,
                'prediction_accuracy': 85.0,
                'return_prediction': 75.0,
                'volatility_prediction': 80.0
            })

            # 監視対象銘柄の設定
            monitoring_config = self.config.get('monitoring', {})
            self.monitoring_symbols = monitoring_config.get('default_symbols',
                                                           ["7203", "8306", "4751"])
            self.validation_hours = monitoring_config.get('validation_hours', 168)
            self.min_samples = monitoring_config.get('min_samples', 10)

            # ログレベルの設定
            log_level = self.config.get('system', {}).get('log_level', 'INFO')
            logger.setLevel(getattr(logging, log_level))

            logger.info(f"設定を読み込みました - 閾値: {self.thresholds}, 監視銘柄: {self.monitoring_symbols}")

        except Exception as e:
            logger.error(f"設定読み込みに失敗しました: {e}")
            self._load_default_thresholds()

    def _create_default_config(self):
        """デフォルト設定ファイルの作成"""
        default_config = {
            'performance_thresholds': {
                'accuracy': 90.0,
                'prediction_accuracy': 85.0
            },
            'monitoring': {
                'default_symbols': ["7203", "8306", "4751"],
                'validation_hours': 168
            },
            'retraining': {
                'granular_mode': True,
                'cooldown_hours': 24
            }
        }

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"デフォルト設定ファイルを作成しました: {self.config_path}")
            self.config = default_config
        except Exception as e:
            logger.error(f"デフォルト設定ファイルの作成に失敗しました: {e}")

    def _load_default_thresholds(self):
        """デフォルト閾値の読み込み"""
        self.thresholds = {
            'accuracy': 90.0,
            'prediction_accuracy': 85.0,
            'return_prediction': 75.0,
            'volatility_prediction': 80.0
        }
        self.monitoring_symbols = ["7203", "8306", "4751"]
        self.validation_hours = 168
        logger.info("デフォルト閾値を設定しました")

    def get_monitoring_symbols(self) -> List[str]:
        """
        現在の監視対象銘柄を取得

        将来的にはsymbol_selectorとの連携や動的生成も可能
        """
        # 設定ファイルから基本銘柄を取得
        symbols = self.monitoring_symbols.copy()

        # TODO: symbol_selector.pyとの連携
        # symbols.extend(symbol_selector.get_active_symbols())

        return list(set(symbols))  # 重複除去

    async def get_latest_model_performance(self,
                                         symbols: Optional[List[str]] = None) -> PerformanceMetrics:
        """
        最新のモデル性能を取得

        Args:
            symbols: 評価対象銘柄（Noneの場合はデフォルト）

        Returns:
            PerformanceMetrics: 性能メトリクス
        """
        if symbols is None:
            symbols = self.get_monitoring_symbols()

        logger.info(f"モデル性能を取得します - 対象銘柄: {symbols}")

        if not self.accuracy_validator:
            logger.warning("PredictionAccuracyValidatorが利用できません")
            return PerformanceMetrics(accuracy=0.0)

        try:
            # PredictionAccuracyValidatorを使用した性能評価
            metrics = await self.accuracy_validator.validate_current_system_accuracy(
                symbols, self.validation_hours)

            performance = PerformanceMetrics(
                accuracy=metrics.overall_accuracy,
                prediction_accuracy=getattr(metrics, 'prediction_accuracy', 0.0),
                return_prediction=getattr(metrics, 'return_prediction', 0.0),
                volatility_prediction=getattr(metrics, 'volatility_prediction', 0.0)
            )

            # 性能履歴をデータベースに記録
            await self._record_performance_history(symbols, performance)

            logger.info(f"モデル性能を取得しました - 精度: {performance.accuracy:.2f}%")
            return performance

        except Exception as e:
            logger.error(f"モデル性能の取得に失敗しました: {e}")
            return PerformanceMetrics(accuracy=0.0)

    async def _record_performance_history(self, symbols: List[str],
                                        performance: PerformanceMetrics):
        """性能履歴をデータベースに記録"""
        try:
            with sqlite3.connect(self.upgrade_db_path) as conn:
                cursor = conn.cursor()
                timestamp = performance.timestamp.isoformat()

                # 各銘柄と全体の性能を記録
                for symbol in symbols:
                    cursor.execute('''
                        INSERT INTO performance_history
                        (symbol, metric_name, value, timestamp, source)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (symbol, 'accuracy', performance.accuracy, timestamp, 'validator'))

                # 全体の性能も記録
                cursor.execute('''
                    INSERT INTO performance_history
                    (symbol, metric_name, value, timestamp, source)
                    VALUES (?, ?, ?, ?, ?)
                ''', ('OVERALL', 'accuracy', performance.accuracy, timestamp, 'validator'))

                conn.commit()

        except Exception as e:
            logger.error(f"性能履歴の記録に失敗しました: {e}")

    async def check_and_trigger_retraining(self) -> RetrainingResult:
        """
        性能チェックと段階的再学習のトリガー

        Returns:
            RetrainingResult: 再学習実行結果
        """
        logger.info("モデル性能監視を開始します")

        # 最新性能の取得
        symbols = self.get_monitoring_symbols()
        performance = await self.get_latest_model_performance(symbols)

        if performance.accuracy == 0.0:
            logger.warning("性能データがないため、再学習チェックをスキップします")
            return RetrainingResult(triggered=False, scope="none", affected_symbols=[])

        # 再学習の必要性を判定
        retraining_scope = self._determine_retraining_scope(performance)

        if retraining_scope == "none":
            logger.info("性能は閾値以上です。再学習は不要です")
            return RetrainingResult(triggered=False, scope="none", affected_symbols=[])

        # 冷却期間チェック
        if not self._check_cooldown_period(retraining_scope):
            logger.info(f"{retraining_scope}再学習は冷却期間中のためスキップします")
            return RetrainingResult(triggered=False, scope=retraining_scope,
                                  affected_symbols=symbols)

        # 再学習の実行
        result = await self._execute_retraining(retraining_scope, symbols, performance)

        # 結果をデータベースに記録
        await self._record_retraining_result(result)

        return result

    def _determine_retraining_scope(self, performance: PerformanceMetrics) -> str:
        """再学習の範囲を決定"""
        accuracy = performance.accuracy

        # グローバル閾値チェック
        global_threshold = self.config.get('retraining', {}).get('global_threshold', 85.0)
        if accuracy < global_threshold:
            return "global"

        # 段階的再学習が有効な場合
        if self.config.get('retraining', {}).get('granular_mode', True):
            symbol_threshold = self.config.get('retraining', {}).get('symbol_specific_threshold', 80.0)
            if accuracy < symbol_threshold:
                return "partial"

        # 基本的な精度閾値
        if accuracy < self.thresholds.get('accuracy', 90.0):
            return "standard"

        return "none"

    def _check_cooldown_period(self, scope: str) -> bool:
        """冷却期間のチェック"""
        cooldown_hours = self.config.get('retraining', {}).get('cooldown_hours', 24)

        last_time = self.last_retraining.get(scope)
        if last_time is None:
            return True

        elapsed = datetime.now() - last_time
        return elapsed.total_seconds() > (cooldown_hours * 3600)

    async def _execute_retraining(self, scope: str, symbols: List[str],
                                performance: PerformanceMetrics) -> RetrainingResult:
        """再学習の実行"""
        if not ml_upgrade_system:
            return RetrainingResult(
                triggered=False, scope=scope, affected_symbols=symbols,
                error="ml_upgrade_system not available"
            )

        start_time = datetime.now()

        try:
            logger.info(f"{scope}再学習を開始します - 対象: {symbols}")

            # スコープに応じた再学習実行
            if scope == "global":
                report = await ml_upgrade_system.run_complete_system_upgrade()
                integration_results = await ml_upgrade_system.integrate_best_models(report)
                improvement = report.overall_improvement

            elif scope in ["partial", "standard"]:
                # 部分的再学習（将来実装予定）
                # TODO: ml_upgrade_systemに銘柄指定再学習機能を追加
                report = await ml_upgrade_system.run_complete_system_upgrade()
                integration_results = await ml_upgrade_system.integrate_best_models(report)
                improvement = report.overall_improvement

            else:
                return RetrainingResult(
                    triggered=False, scope=scope, affected_symbols=symbols,
                    error=f"Unknown retraining scope: {scope}"
                )

            # 実行時間の計算
            duration = (datetime.now() - start_time).total_seconds()

            # 最終実行時刻を更新
            self.last_retraining[scope] = datetime.now()

            logger.info(f"再学習が完了しました - 改善: {improvement:.2f}%, 所要時間: {duration:.1f}秒")

            return RetrainingResult(
                triggered=True, scope=scope, affected_symbols=symbols,
                improvement=improvement, duration=duration
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"再学習実行エラー: {e}"
            logger.error(error_msg)

            return RetrainingResult(
                triggered=True, scope=scope, affected_symbols=symbols,
                duration=duration, error=error_msg
            )

    async def _record_retraining_result(self, result: RetrainingResult):
        """再学習結果をデータベースに記録"""
        try:
            with sqlite3.connect(self.upgrade_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO retraining_history
                    (trigger_time, scope, affected_symbols, improvement,
                     duration, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    result.scope,
                    ','.join(result.affected_symbols),
                    result.improvement,
                    result.duration,
                    'success' if result.error is None else 'error',
                    result.error
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"再学習結果の記録に失敗しました: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """性能監視の要約情報を取得"""
        try:
            with sqlite3.connect(self.upgrade_db_path) as conn:
                cursor = conn.cursor()

                # 最新の性能情報
                cursor.execute('''
                    SELECT metric_name, value, timestamp
                    FROM performance_history
                    WHERE symbol = 'OVERALL'
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''')
                recent_performance = cursor.fetchall()

                # 再学習履歴
                cursor.execute('''
                    SELECT scope, improvement, status, trigger_time
                    FROM retraining_history
                    ORDER BY trigger_time DESC
                    LIMIT 5
                ''')
                recent_retraining = cursor.fetchall()

                return {
                    'current_thresholds': self.thresholds,
                    'monitoring_symbols': self.monitoring_symbols,
                    'recent_performance': recent_performance,
                    'recent_retraining': recent_retraining,
                    'config_path': str(self.config_path)
                }

        except Exception as e:
            logger.error(f"サマリー情報の取得に失敗しました: {e}")
            return {}

# ユーティリティ関数
def create_performance_monitor(config_path: Optional[str] = None) -> ModelPerformanceMonitor:
    """ModelPerformanceMonitorインスタンスの作成"""
    path = Path(config_path) if config_path else None
    return ModelPerformanceMonitor(config_path=path)

if __name__ == "__main__":
    # 基本的な動作確認
    async def main():
        logger.info("ModelPerformanceMonitor テスト開始")

        monitor = create_performance_monitor()
        result = await monitor.check_and_trigger_retraining()

        logger.info(f"監視結果: {result}")
        summary = monitor.get_performance_summary()
        logger.info(f"サマリー: {summary}")

    # テスト実行
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main())
