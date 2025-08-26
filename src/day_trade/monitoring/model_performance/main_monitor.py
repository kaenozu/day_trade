#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Monitor - Main Monitor Class
メインモニタークラス
"""

import asyncio
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque

warnings.filterwarnings('ignore')

# 共通ユーティリティのインポート
try:
    from src.day_trade.utils.encoding_fix import apply_windows_encoding_fix
    apply_windows_encoding_fix()
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

# 既存システムのインポート
try:
    from ml_prediction_models import MLPredictionModels, ModelType, PredictionTask
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False

# ローカルインポート
from .enums_and_models import (
    PerformanceMetrics, PerformanceAlert, RetrainingResult, RetrainingScope
)
from .config_manager import EnhancedPerformanceConfigManager
from .symbol_manager import DynamicSymbolManager
from .retraining_manager import GranularRetrainingManager
from .database_manager import DatabaseManager
from .performance_evaluator import PerformanceEvaluator
from .alert_manager import AlertManager

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """
    統合版MLモデル性能監視システム

    主な機能:
    - 完全外部設定による性能閾値管理
    - 動的監視対象銘柄システム
    - 段階的再学習トリガー（増分・銘柄別・部分・全体）
    - symbol_selector連携
    - 詳細なパフォーマンス追跡と分析
    - 包括的なアラート管理
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)

        # 設定管理システム初期化
        self.config_manager = EnhancedPerformanceConfigManager(config_path)
        self.symbol_manager = DynamicSymbolManager(self.config_manager)
        self.retraining_manager = GranularRetrainingManager(self.config_manager)

        # データベース初期化
        db_path = Path(
            self.config_manager.config.get('database', {})
            .get('connection', {})
            .get('path', 'data/model_performance.db')
        )
        self.database_manager = DatabaseManager(db_path)

        # コンポーネント初期化
        self.performance_evaluator = PerformanceEvaluator(self.config_manager)
        self.alert_manager = AlertManager(self.config_manager)

        # 内部状態
        self.monitoring_active = False
        self.last_check_time = None

        # 外部システム統合
        self._initialize_external_systems()

        # ログレベル設定
        log_level = self.config_manager.config.get(
            'system', {}
        ).get('log_level', 'INFO')
        self.logger.setLevel(getattr(logging, log_level))

        self.logger.info("統合版ModelPerformanceMonitor初期化完了")

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

    def get_monitoring_symbols(self) -> List[str]:
        """監視対象銘柄リスト取得"""
        return self.symbol_manager.get_monitoring_symbols()

    async def get_latest_model_performance(
            self, symbols: Optional[List[str]] = None, force_refresh: bool = False
    ) -> Dict[str, PerformanceMetrics]:
        """最新のモデル性能取得"""
        # 設定ファイル更新チェック
        self.config_manager.reload_if_modified()

        # 監視対象銘柄決定
        if symbols is None:
            symbols = self.get_monitoring_symbols()

        self.logger.info(f"モデル性能を取得します - 対象銘柄: {symbols}")

        # キャッシュチェック
        if not force_refresh and self.performance_evaluator.is_cache_valid():
            cached_results = {
                s: self.performance_evaluator.performance_cache[s] 
                for s in symbols
                if s in self.performance_evaluator.performance_cache
            }
            if len(cached_results) == len(symbols):
                self.logger.info("キャッシュから性能データを取得しました")
                return cached_results

        performance_data = {}

        for symbol in symbols:
            try:
                # 各モデルタイプの性能評価
                supported_models = (
                    self.config_manager.config.get('models', {})
                    .get('supported_models', ['RandomForestClassifier'])
                )
                for model_type in supported_models:
                    metrics = await self.performance_evaluator.evaluate_model_performance(
                        symbol, model_type
                    )
                    if metrics:
                        key = f"{symbol}_{model_type}"
                        performance_data[key] = metrics

                        # 履歴に追加
                        self.performance_evaluator.performance_history[key].append(metrics)

                        # 履歴サイズ制限
                        max_history = 100
                        if len(self.performance_evaluator.performance_history[key]) > max_history:
                            self.performance_evaluator.performance_history[key].popleft()

                        # キャッシュ更新
                        self.performance_evaluator.performance_cache[symbol] = metrics

                        # データベースに保存
                        await self.database_manager.save_performance_metrics(metrics)

                        # 閾値チェックとアラート生成
                        alerts = await self.performance_evaluator.check_and_generate_alerts(metrics)
                        for alert in alerts:
                            await self.database_manager.save_alert(alert)
                            self.alert_manager.add_alert_to_active(alert)

            except Exception as e:
                self.logger.error(f"性能評価エラー {symbol}: {e}")

        self.logger.info(f"モデル性能を取得しました - {len(performance_data)}項目")
        return performance_data

    async def check_and_trigger_enhanced_retraining(self) -> RetrainingResult:
        """改善版性能チェックと段階的再学習のトリガー"""
        self.logger.info("改善版モデル性能監視を開始します")

        # 最新性能の取得
        symbol_performances = await self.get_latest_model_performance()

        if not symbol_performances:
            self.logger.warning("性能データがないため、再学習チェックをスキップします")
            return RetrainingResult(
                triggered=False,
                scope=RetrainingScope.NONE,
                affected_symbols=[]
            )

        # 単一値の性能メトリクスに変換
        aggregated_performances = {}
        for key, metrics in symbol_performances.items():
            symbol = metrics.symbol
            if symbol not in aggregated_performances:
                aggregated_performances[symbol] = metrics

        # 再学習スコープ決定
        retraining_scope = self.retraining_manager.determine_retraining_scope(
            aggregated_performances
        )

        if retraining_scope == RetrainingScope.NONE:
            self.logger.info("性能は閾値以上です。再学習は不要です")
            return RetrainingResult(
                triggered=False,
                scope=RetrainingScope.NONE,
                affected_symbols=list(aggregated_performances.keys())
            )

        # 冷却期間チェック
        if not self.retraining_manager.check_cooldown_period(retraining_scope):
            self.logger.info(f"{retraining_scope.value}再学習は冷却期間中のためスキップします")
            return RetrainingResult(
                triggered=False,
                scope=retraining_scope,
                affected_symbols=list(aggregated_performances.keys())
            )

        # 再学習の実行
        result = await self.retraining_manager.execute_enhanced_retraining(
            retraining_scope, aggregated_performances
        )

        # 結果をデータベースに記録
        await self.database_manager.record_enhanced_retraining_result(
            result, self.config_manager.config
        )

        # 再学習時刻を更新
        if result.triggered:
            self.retraining_manager.update_retraining_time(retraining_scope)

        return result

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
                interval = (
                    self.config_manager.config.get('monitoring', {})
                    .get('intervals', {})
                    .get('real_time_seconds', 60)
                )
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
            alerts = await self.alert_manager.check_performance_degradation(
                performance_data
            )

            # 再学習トリガー
            if alerts:
                retraining_trigger = await self.alert_manager.trigger_retraining(alerts)
                if retraining_trigger:
                    await self.database_manager.save_retraining_trigger(retraining_trigger)
                    self.logger.info(f"再学習トリガー発動: {retraining_trigger.trigger_id}")

            self.last_check_time = datetime.now()

            self.logger.info(
                f"性能チェック完了: {len(performance_data)}モデル, {len(alerts)}アラート"
            )

        except Exception as e:
            self.logger.error(f"性能チェックエラー: {e}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視ステータス取得"""
        return {
            'monitoring_active': self.monitoring_active,
            'last_check_time': (
                self.last_check_time.isoformat() if self.last_check_time else None
            ),
            'active_alerts_count': self.alert_manager.get_active_alert_count(),
            'monitoring_symbols_count': len(self.get_monitoring_symbols()),
            'config_loaded': bool(self.config_manager.config),
            'cache_valid': self.performance_evaluator.is_cache_valid(),
            'external_systems': {
                'ml_models': self.ml_models is not None,
                'symbol_selector': self.symbol_manager.symbol_selector is not None
            }
        }

    def get_enhanced_performance_summary(self) -> Dict[str, Any]:
        """改善版性能監視の要約情報を取得"""
        try:
            db_summary = self.database_manager.get_enhanced_performance_summary()
            current_symbols = self.get_monitoring_symbols()

            summary = {
                'timestamp': datetime.now().isoformat(),
                'config_path': str(self.config_manager.config_path),
                'monitoring_symbols': current_symbols,
                'symbol_count': len(current_symbols),
                'dynamic_monitoring': (
                    self.config_manager.get_monitoring_config()
                    .get('dynamic_monitoring', {})
                    .get('enabled', False)
                ),
                'cache_valid': self.performance_evaluator.is_cache_valid(),
                **db_summary
            }

            return summary

        except Exception as e:
            self.logger.error(f"改善版サマリー情報の取得に失敗: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }


# ユーティリティ関数
def create_enhanced_performance_monitor(
        config_path: Optional[str] = None
) -> ModelPerformanceMonitor:
    """ModelPerformanceMonitorインスタンスの作成"""
    path = Path(config_path) if config_path else None
    return ModelPerformanceMonitor(config_path=path)


# レガシー関数（後方互換性）
async def test_model_performance_monitor():
    """モデル性能監視システムテスト"""
    print("=== ModelPerformanceMonitor テスト開始 ===")

    # 監視システム初期化
    monitor = ModelPerformanceMonitor()

    # 監視ステータス確認
    status = monitor.get_monitoring_status()
    print(f"監視ステータス: {status}")

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
    # 基本的な動作確認
    async def main():
        logger.info("統合版ModelPerformanceMonitor テスト開始")

        monitor = create_enhanced_performance_monitor()

        # 性能取得テスト
        performance_data = await monitor.get_latest_model_performance()
        logger.info(f"性能データ取得: {len(performance_data)}項目")

        # 再学習チェックテスト
        result = await monitor.check_and_trigger_enhanced_retraining()
        logger.info(f"再学習チェック結果: {result}")

        # サマリー取得テスト
        summary = monitor.get_enhanced_performance_summary()
        symbol_count = summary.get('symbol_count', 0)
        alert_count = summary.get('alert_count', 0)
        logger.info(f"サマリー: 監視銘柄{symbol_count}件, アラート{alert_count}件")

    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # テスト実行
    asyncio.run(main())