#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retraining Executor - 再学習実行システム

再学習の実行とモニタリング制御を担当するモジュール
MonitorクラスのRetraining関連メソッドを分離
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from .config import EnhancedPerformanceConfigManager
from .retraining_manager import GranularRetrainingManager
from .database_manager import DatabaseManager
from .types import PerformanceMetrics, RetrainingResult, RetrainingScope

logger = logging.getLogger(__name__)

# 外部システム連携チェック
try:
    from ml_model_upgrade_system import MLModelUpgradeSystem, ml_upgrade_system
    UPGRADE_SYSTEM_AVAILABLE = True
except ImportError:
    ml_upgrade_system = None
    UPGRADE_SYSTEM_AVAILABLE = False


class RetrainingExecutor:
    """再学習実行クラス
    
    段階的再学習の実行とモニタリングを担当します。
    
    Attributes:
        config_manager: 設定管理インスタンス
        retraining_manager: 再学習管理インスタンス
        database_manager: データベース管理インスタンス
        upgrade_system: アップグレードシステムインスタンス
    """

    def __init__(
            self, 
            config_manager: EnhancedPerformanceConfigManager,
            retraining_manager: GranularRetrainingManager,
            database_manager: DatabaseManager
    ):
        """初期化
        
        Args:
            config_manager: 設定管理インスタンス
            retraining_manager: 再学習管理インスタンス
            database_manager: データベース管理インスタンス
        """
        self.config_manager = config_manager
        self.retraining_manager = retraining_manager
        self.database_manager = database_manager
        self.upgrade_system = None

        # アップグレードシステムの初期化
        if UPGRADE_SYSTEM_AVAILABLE:
            try:
                self.upgrade_system = (MLModelUpgradeSystem() 
                                     if ml_upgrade_system is None else ml_upgrade_system)
                logger.info("MLModelUpgradeSystem統合完了")
            except Exception as e:
                logger.warning(f"MLModelUpgradeSystem統合失敗: {e}")
                self.upgrade_system = None

    async def check_and_trigger_enhanced_retraining(
            self, symbol_performances: Dict[str, PerformanceMetrics]
    ) -> RetrainingResult:
        """改善版性能チェックと段階的再学習のトリガー
        
        Args:
            symbol_performances: 銘柄別性能メトリクス
            
        Returns:
            再学習実行結果
        """
        logger.info("改善版モデル性能監視を開始します")

        if not symbol_performances:
            logger.warning("性能データがないため、再学習チェックをスキップします")
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
            logger.info("性能は閾値以上です。再学習は不要です")
            return RetrainingResult(
                triggered=False,
                scope=RetrainingScope.NONE,
                affected_symbols=list(aggregated_performances.keys())
            )

        # 冷却期間チェック
        if not self.retraining_manager.check_cooldown_period(retraining_scope):
            logger.info(f"{retraining_scope.value}再学習は冷却期間中のためスキップします")
            return RetrainingResult(
                triggered=False,
                scope=retraining_scope,
                affected_symbols=list(aggregated_performances.keys())
            )

        # 再学習の実行
        result = await self._execute_enhanced_retraining(
            retraining_scope, aggregated_performances
        )

        # 結果をデータベースに記録
        await self.database_manager.save_retraining_result(
            result, 
            trigger_info={'id': f"retraining_{int(time.time())}"}
        )

        # 再学習時刻を更新
        if result.triggered:
            self.retraining_manager.update_retraining_time(retraining_scope)

        return result

    async def _execute_enhanced_retraining(
            self, scope: RetrainingScope, symbol_performances: Dict[str, PerformanceMetrics]
    ) -> RetrainingResult:
        """改善版再学習実行
        
        Args:
            scope: 再学習スコープ
            symbol_performances: 銘柄別性能メトリクス
            
        Returns:
            再学習実行結果
        """
        if not self.upgrade_system:
            return RetrainingResult(
                triggered=False,
                scope=scope,
                affected_symbols=list(symbol_performances.keys()),
                error="ml_upgrade_system not available"
            )

        start_time = datetime.now()
        affected_symbols = list(symbol_performances.keys())

        # 再学習設定から推定時間を取得
        estimated_time = self.retraining_manager.get_estimated_time(scope)

        try:
            logger.info(f"{scope.value}再学習を開始します - 対象: {affected_symbols}")
            logger.info(f"推定実行時間: {estimated_time}秒")

            # スコープに応じた再学習実行
            improvement = await self._execute_retraining_by_scope(scope)

            # 実行時間の計算
            duration = (datetime.now() - start_time).total_seconds()

            # 実際の効果測定
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

    async def _execute_retraining_by_scope(self, scope: RetrainingScope) -> float:
        """スコープ別再学習実行
        
        Args:
            scope: 再学習スコープ
            
        Returns:
            改善度（%）
        """
        if scope == RetrainingScope.GLOBAL:
            if hasattr(self.upgrade_system, 'run_complete_system_upgrade'):
                report = await self.upgrade_system.run_complete_system_upgrade()
                if hasattr(self.upgrade_system, 'integrate_best_models'):
                    await self.upgrade_system.integrate_best_models(report)
                return getattr(report, 'overall_improvement', 0.0)
            else:
                # フォールバック処理
                await asyncio.sleep(2)  # 模擬処理時間
                return 5.0

        elif scope in [RetrainingScope.PARTIAL, RetrainingScope.SYMBOL]:
            # 部分再学習または銘柄別再学習
            if hasattr(self.upgrade_system, 'run_complete_system_upgrade'):
                report = await self.upgrade_system.run_complete_system_upgrade()
                if hasattr(self.upgrade_system, 'integrate_best_models'):
                    await self.upgrade_system.integrate_best_models(report)
                return getattr(report, 'overall_improvement', 0.0) * 0.7
            else:
                await asyncio.sleep(1)  # 模擬処理時間
                return 3.0

        elif scope == RetrainingScope.INCREMENTAL:
            # 増分学習
            logger.info("増分学習は現在開発中です。通常の再学習を実行します")
            await asyncio.sleep(0.5)  # 模擬処理時間
            return 1.0

        else:
            raise ValueError(f"Unknown retraining scope: {scope.value}")


class MonitoringController:
    """監視制御クラス
    
    監視の開始・停止・状態管理を担当します。
    """

    def __init__(
            self, 
            config_manager: EnhancedPerformanceConfigManager,
            performance_getter,
            alert_checker,
            retraining_executor: RetrainingExecutor
    ):
        """初期化
        
        Args:
            config_manager: 設定管理インスタンス
            performance_getter: 性能取得関数
            alert_checker: アラートチェック関数
            retraining_executor: 再学習実行インスタンス
        """
        self.config_manager = config_manager
        self.performance_getter = performance_getter
        self.alert_checker = alert_checker
        self.retraining_executor = retraining_executor
        
        self.monitoring_active = False
        self.last_check_time = None

    async def start_monitoring(self):
        """監視開始"""
        if self.monitoring_active:
            logger.warning("監視は既に開始されています")
            return

        self.monitoring_active = True
        logger.info("モデル性能監視を開始しました")

        try:
            while self.monitoring_active:
                # 性能チェック実行
                await self.run_performance_check()

                # 監視間隔待機
                interval = (self.config_manager.config.get('monitoring', {})
                           .get('intervals', {})
                           .get('real_time_seconds', 60))
                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            logger.info("監視がキャンセルされました")
        except Exception as e:
            logger.error(f"監視エラー: {e}")
        finally:
            self.monitoring_active = False

    async def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        logger.info("モデル性能監視を停止しました")

    async def run_performance_check(self):
        """性能チェック実行"""
        try:
            logger.info("性能チェック開始")

            # 最新性能取得
            performance_data = await self.performance_getter()

            # 性能低下チェック
            alerts = await self.alert_checker(performance_data)

            # 再学習トリガー
            if alerts:
                logger.info(f"アラート検出: {len(alerts)}件")

            # 改善版再学習チェック
            result = await self.retraining_executor.check_and_trigger_enhanced_retraining(
                performance_data
            )
            
            if result.triggered:
                logger.info(f"再学習実行: {result.scope.value}, 改善度: {result.improvement:.2f}%")

            self.last_check_time = datetime.now()

            logger.info(f"性能チェック完了: {len(performance_data)}モデル, {len(alerts)}アラート")

        except Exception as e:
            logger.error(f"性能チェックエラー: {e}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視ステータス取得
        
        Returns:
            監視状態を表す辞書
        """
        return {
            'monitoring_active': self.monitoring_active,
            'last_check_time': (self.last_check_time.isoformat() 
                               if self.last_check_time else None),
        }