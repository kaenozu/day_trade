#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Monitor - メイン性能監視システム

MLモデル性能監視システムのメインコーディネーター
各モジュールを統合して包括的な性能監視とアラート管理を提供
"""

import asyncio
import logging
import warnings
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Windows環境対策
warnings.filterwarnings('ignore')

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

# 内部モジュールのインポート
from .config import EnhancedPerformanceConfigManager
from .symbol_manager import DynamicSymbolManager
from .retraining_manager import GranularRetrainingManager
from .metrics_evaluator import MetricsEvaluator
from .alert_manager import AlertManager
from .database_manager import DatabaseManager
from .types import PerformanceMetrics, PerformanceAlert, RetrainingResult, RetrainingScope

# 外部システム連携チェック
try:
    from ml_prediction_models import MLPredictionModels
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False

try:
    from ml_model_upgrade_system import MLModelUpgradeSystem, ml_upgrade_system
    UPGRADE_SYSTEM_AVAILABLE = True
except ImportError:
    ml_upgrade_system = None
    UPGRADE_SYSTEM_AVAILABLE = False

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

    Attributes:
        config_manager: 設定管理システム
        symbol_manager: 動的銘柄管理システム
        retraining_manager: 再学習管理システム
        metrics_evaluator: メトリクス評価システム
        alert_manager: アラート管理システム
        database_manager: データベース管理システム
        monitoring_active: 監視アクティブフラグ
        performance_cache: 性能データキャッシュ
        last_check_time: 最終チェック時刻
    """

    def __init__(self, config_path: Optional[Path] = None):
        """初期化
        
        Args:
            config_path: 設定ファイルのパス（未指定時はデフォルト）
        """
        self.logger = logging.getLogger(__name__)

        # 設定管理システム初期化
        self.config_manager = EnhancedPerformanceConfigManager(config_path)
        
        # 各管理システムの初期化
        self.symbol_manager = DynamicSymbolManager(self.config_manager)
        self.retraining_manager = GranularRetrainingManager(self.config_manager)
        self.metrics_evaluator = MetricsEvaluator(self.config_manager)
        self.database_manager = DatabaseManager(self.config_manager)
        
        # アラート管理（データベース保存コールバック付き）
        self.alert_manager = AlertManager(
            self.config_manager, 
            save_callback=self.database_manager.save_alert
        )

        # 内部状態
        self.performance_cache = {}
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

        # モデルアップグレードシステム
        if UPGRADE_SYSTEM_AVAILABLE:
            try:
                self.upgrade_system = (MLModelUpgradeSystem() 
                                     if ml_upgrade_system is None else ml_upgrade_system)
                self.logger.info("MLModelUpgradeSystem統合完了")
            except Exception as e:
                self.logger.warning(f"MLModelUpgradeSystem統合失敗: {e}")
                self.upgrade_system = None
        else:
            self.upgrade_system = None

    def get_monitoring_symbols(self) -> List[str]:
        """監視対象銘柄リスト取得
        
        Returns:
            監視対象銘柄のリスト
        """
        return self.symbol_manager.get_monitoring_symbols()

    async def get_latest_model_performance(
            self, symbols: Optional[List[str]] = None, force_refresh: bool = False
    ) -> Dict[str, PerformanceMetrics]:
        """最新のモデル性能取得
        
        Args:
            symbols: 対象銘柄リスト（未指定時は全監視銘柄）
            force_refresh: 強制的にキャッシュを更新するかどうか
            
        Returns:
            銘柄・モデル別性能メトリクス辞書
        """
        # 設定ファイル更新チェック
        self.config_manager.reload_if_modified()

        # 監視対象銘柄決定
        if symbols is None:
            symbols = self.get_monitoring_symbols()

        self.logger.info(f"モデル性能を取得します - 対象銘柄: {symbols}")

        # キャッシュチェック
        if not force_refresh and self._is_cache_valid():
            cached_results = {s: self.performance_cache[s] for s in symbols
                              if s in self.performance_cache}
            if len(cached_results) == len(symbols):
                self.logger.info("キャッシュから性能データを取得しました")
                return cached_results

        performance_data = {}

        for symbol in symbols:
            try:
                # 各モデルタイプの性能評価
                supported_models = (self.config_manager.config.get('models', {})
                                  .get('supported_models', ['RandomForestClassifier']))
                
                for model_type in supported_models:
                    metrics = await self.metrics_evaluator.evaluate_model_performance(
                        symbol, model_type
                    )
                    
                    if metrics:
                        key = f"{symbol}_{model_type}"
                        performance_data[key] = metrics

                        # データベースに保存
                        await self.database_manager.save_performance_metrics(metrics)

                        # キャッシュ更新
                        self.performance_cache[symbol] = metrics

                        # アラートチェック
                        alerts = await self.alert_manager.check_and_generate_alerts(metrics)
                        
                        if alerts:
                            self.logger.info(f"アラート生成: {len(alerts)}件 ({symbol})")

            except Exception as e:
                self.logger.error(f"性能評価エラー {symbol}: {e}")

        self.logger.info(f"モデル性能を取得しました - {len(performance_data)}項目")
        return performance_data

    def _is_cache_valid(self) -> bool:
        """キャッシュの有効性チェック
        
        Returns:
            キャッシュが有効な場合はTrue
        """
        if not self.performance_cache:
            return False

        # 最新エントリの時刻をチェック
        latest_time = max(p.timestamp for p in self.performance_cache.values())
        elapsed = datetime.now() - latest_time

        # 1時間以内なら有効
        return elapsed.total_seconds() < 3600

    async def check_and_trigger_enhanced_retraining(self) -> RetrainingResult:
        """改善版性能チェックと段階的再学習のトリガー
        
        Returns:
            再学習実行結果
        """
        # 再学習実行器の初期化（遅延初期化）
        from .retraining_executor import RetrainingExecutor
        
        if not hasattr(self, '_retraining_executor'):
            self._retraining_executor = RetrainingExecutor(
                self.config_manager,
                self.retraining_manager,
                self.database_manager
            )
        
        # 最新性能の取得
        symbol_performances = await self.get_latest_model_performance()
        
        # 再学習チェックと実行
        return await self._retraining_executor.check_and_trigger_enhanced_retraining(
            symbol_performances
        )

    async def check_performance_degradation(
            self, performance_data: Dict[str, PerformanceMetrics]
    ) -> List[PerformanceAlert]:
        """性能低下チェック
        
        Args:
            performance_data: 銘柄・モデル別性能データ
            
        Returns:
            検出されたアラートのリスト
        """
        return await self.alert_manager.check_performance_degradation(performance_data)

    async def start_monitoring(self):
        """監視開始"""
        # 監視制御器の初期化（遅延初期化）
        from .retraining_executor import MonitoringController
        
        if not hasattr(self, '_monitoring_controller'):
            # 再学習実行器の初期化
            from .retraining_executor import RetrainingExecutor
            retraining_executor = RetrainingExecutor(
                self.config_manager,
                self.retraining_manager,
                self.database_manager
            )
            
            # 監視制御器の初期化
            self._monitoring_controller = MonitoringController(
                self.config_manager,
                self.get_latest_model_performance,
                self.check_performance_degradation,
                retraining_executor
            )
        
        await self._monitoring_controller.start_monitoring()

    async def stop_monitoring(self):
        """監視停止"""
        if hasattr(self, '_monitoring_controller'):
            await self._monitoring_controller.stop_monitoring()

    async def run_performance_check(self):
        """性能チェック実行"""
        if hasattr(self, '_monitoring_controller'):
            await self._monitoring_controller.run_performance_check()
        else:
            self.logger.warning("監視制御器が初期化されていません")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視ステータス取得
        
        Returns:
            監視状態を表す辞書
        """
        base_status = {
            'config_loaded': bool(self.config_manager.config),
            'cache_valid': self._is_cache_valid(),
            'monitoring_symbols_count': len(self.get_monitoring_symbols()),
            'external_systems': {
                'ml_models': self.ml_models is not None,
                'upgrade_system': self.upgrade_system is not None,
                'symbol_selector': self.symbol_manager.symbol_selector is not None
            }
        }
        
        if hasattr(self, '_monitoring_controller'):
            controller_status = self._monitoring_controller.get_monitoring_status()
            base_status.update(controller_status)
        else:
            base_status.update({
                'monitoring_active': False,
                'last_check_time': None
            })
            
        return base_status

    def get_enhanced_performance_summary(self) -> Dict[str, Any]:
        """改善版性能監視の要約情報を取得
        
        Returns:
            要約情報辞書
        """
        try:
            # データベースからサマリーを取得
            summary = self.database_manager.get_performance_summary()
            
            # 現在の監視銘柄情報を追加
            current_symbols = self.get_monitoring_symbols()
            summary.update({
                'config_path': str(self.config_manager.config_path),
                'monitoring_symbols': current_symbols,
                'symbol_count': len(current_symbols),
                'dynamic_monitoring': (
                    self.config_manager.get_monitoring_config()
                    .get('dynamic_monitoring', {})
                    .get('enabled', False)
                ),
                'cache_valid': self._is_cache_valid(),
            })
            
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
    """ModelPerformanceMonitorインスタンスの作成
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        初期化されたModelPerformanceMonitorインスタンス
    """
    path = Path(config_path) if config_path else None
    return ModelPerformanceMonitor(config_path=path)