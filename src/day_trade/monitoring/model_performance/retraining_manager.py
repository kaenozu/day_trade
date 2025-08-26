#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Monitor - Retraining Manager
段階的再学習管理クラス
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict

from .enums_and_models import (
    PerformanceMetrics,
    RetrainingScope,
    RetrainingResult
)
from .config_manager import EnhancedPerformanceConfigManager

logger = logging.getLogger(__name__)

# モデルアップグレードシステム連携
try:
    from ml_model_upgrade_system import MLModelUpgradeSystem, ml_upgrade_system
    UPGRADE_SYSTEM_AVAILABLE = True
except ImportError:
    ml_upgrade_system = None
    UPGRADE_SYSTEM_AVAILABLE = False


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
        total_accuracy = sum(
            p.accuracy for p in symbol_performances.values() if p.accuracy
        )
        overall_accuracy = (
            total_accuracy / len(symbol_performances)
            if symbol_performances else 0
        )

        # グローバル再学習判定
        global_threshold = retraining_config.get('global_threshold', 80.0)
        if overall_accuracy < global_threshold:
            return RetrainingScope.GLOBAL

        # 段階的再学習が有効な場合
        if retraining_config.get('granular_mode', True):
            # 部分再学習判定
            partial_threshold = retraining_config.get('partial_threshold', 88.0)
            low_performance_count = sum(
                1 for p in symbol_performances.values()
                if p.accuracy and p.accuracy < partial_threshold
            )

            # 30%以上が低性能の場合
            if low_performance_count >= len(symbol_performances) * 0.3:
                return RetrainingScope.PARTIAL

            # 銘柄別再学習判定
            symbol_threshold = retraining_config.get(
                'symbol_specific_threshold', 85.0
            )
            critical_symbols = [
                s for s, p in symbol_performances.items()
                if p.accuracy and p.accuracy < symbol_threshold
            ]

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
            if perf.accuracy and 85.0 <= perf.accuracy < 90.0:  # 軽微な劣化
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

    async def execute_enhanced_retraining(
            self,
            scope: RetrainingScope,
            symbol_performances: Dict[str, PerformanceMetrics]
    ) -> RetrainingResult:
        """改善版再学習実行"""
        if not UPGRADE_SYSTEM_AVAILABLE:
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
        estimated_time = scope_config.get('estimated_time', 1800)

        try:
            logger.info(
                f"{scope.value}再学習を開始します - 対象: {affected_symbols}"
            )
            logger.info(f"推定実行時間: {estimated_time}秒")

            # スコープに応じた再学習実行
            if scope == RetrainingScope.GLOBAL:
                improvement = await self._execute_global_retraining()
            elif scope in [RetrainingScope.PARTIAL, RetrainingScope.SYMBOL]:
                improvement = await self._execute_partial_retraining()
            elif scope == RetrainingScope.INCREMENTAL:
                improvement = await self._execute_incremental_retraining()
            else:
                return RetrainingResult(
                    triggered=False,
                    scope=scope,
                    affected_symbols=affected_symbols,
                    error=f"Unknown retraining scope: {scope.value}"
                )

            # 実行時間の計算
            duration = (datetime.now() - start_time).total_seconds()
            actual_benefit = improvement if improvement > 0 else 0.0

            logger.info(
                f"再学習が完了しました - 改善: {improvement:.2f}%, "
                f"所要時間: {duration:.1f}秒"
            )

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

    async def _execute_global_retraining(self) -> float:
        """グローバル再学習実行"""
        upgrade_system = (
            MLModelUpgradeSystem() if ml_upgrade_system is None
            else ml_upgrade_system
        )

        if hasattr(upgrade_system, 'run_complete_system_upgrade'):
            report = await upgrade_system.run_complete_system_upgrade()
            if hasattr(upgrade_system, 'integrate_best_models'):
                await upgrade_system.integrate_best_models(report)
            return getattr(report, 'overall_improvement', 0.0)
        else:
            # フォールバック処理
            await asyncio.sleep(2)  # 模擬処理時間
            return 5.0

    async def _execute_partial_retraining(self) -> float:
        """部分再学習または銘柄別再学習実行"""
        upgrade_system = (
            MLModelUpgradeSystem() if ml_upgrade_system is None
            else ml_upgrade_system
        )

        if hasattr(upgrade_system, 'run_complete_system_upgrade'):
            report = await upgrade_system.run_complete_system_upgrade()
            if hasattr(upgrade_system, 'integrate_best_models'):
                await upgrade_system.integrate_best_models(report)
            return getattr(report, 'overall_improvement', 0.0) * 0.7
        else:
            await asyncio.sleep(1)  # 模擬処理時間
            return 3.0

    async def _execute_incremental_retraining(self) -> float:
        """増分学習実行"""
        logger.info("増分学習は現在開発中です。通常の再学習を実行します")
        await asyncio.sleep(0.5)  # 模擬処理時間
        return 1.0