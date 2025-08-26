#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Granular Retraining Manager - 段階的再学習管理

MLモデルの段階的再学習を管理するモジュール
性能低下の度合いに応じて、増分・銘柄別・部分・全体の再学習スコープを決定
"""

import logging
from datetime import datetime
from typing import Dict
from .config import EnhancedPerformanceConfigManager
from .types import PerformanceMetrics, RetrainingScope

logger = logging.getLogger(__name__)


class GranularRetrainingManager:
    """段階的再学習管理クラス
    
    モデルの性能状況に応じて最適な再学習スコープを決定し、
    冷却期間を管理して効率的な再学習を実現します。
    
    Attributes:
        config_manager: 設定管理インスタンス
        last_retraining: スコープ別の最終実行時刻辞書
        retraining_history: 再学習履歴リスト
    """

    def __init__(self, config_manager: EnhancedPerformanceConfigManager):
        """初期化
        
        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager
        self.last_retraining = {}  # スコープ別の最終実行時刻
        self.retraining_history = []

    def determine_retraining_scope(
            self, symbol_performances: Dict[str, PerformanceMetrics]
    ) -> RetrainingScope:
        """再学習スコープを決定
        
        銘柄ごとの性能メトリクスを分析して、最適な再学習スコープを決定します。
        
        Args:
            symbol_performances: 銘柄別性能メトリクス辞書
            
        Returns:
            決定された再学習スコープ
        """
        retraining_config = self.config_manager.get_retraining_config()

        if not symbol_performances:
            logger.debug("性能データなし、再学習不要")
            return RetrainingScope.NONE

        # 全体性能計算
        valid_performances = [p for p in symbol_performances.values() if p.accuracy is not None]
        if not valid_performances:
            logger.debug("有効な精度データなし、再学習不要")
            return RetrainingScope.NONE

        total_accuracy = sum(p.accuracy for p in valid_performances)
        overall_accuracy = total_accuracy / len(valid_performances)
        
        logger.debug(f"全体平均精度: {overall_accuracy:.3f}")

        # グローバル再学習判定
        global_threshold = retraining_config.get('global_threshold', 80.0) / 100.0
        if overall_accuracy < global_threshold:
            logger.info(f"全体精度が閾値を下回る: {overall_accuracy:.3f} < {global_threshold:.3f}")
            return RetrainingScope.GLOBAL

        # 段階的再学習が有効な場合
        if retraining_config.get('granular_mode', True):
            # 部分再学習判定
            partial_threshold = retraining_config.get('partial_threshold', 88.0) / 100.0
            low_performance_count = sum(1 for p in valid_performances
                                        if p.accuracy < partial_threshold)

            low_performance_ratio = low_performance_count / len(valid_performances)
            
            if low_performance_ratio >= 0.3:  # 30%以上が低性能
                logger.info(f"低性能銘柄比率が高い: {low_performance_ratio:.1%}")
                return RetrainingScope.PARTIAL

            # 銘柄別再学習判定
            symbol_threshold = retraining_config.get('symbol_specific_threshold', 85.0) / 100.0
            critical_symbols = [s for s, p in symbol_performances.items()
                                if p.accuracy is not None and p.accuracy < symbol_threshold]

            if critical_symbols:
                logger.info(f"低性能銘柄: {critical_symbols}")
                return RetrainingScope.SYMBOL

        # 増分学習判定
        if self._should_incremental_update(valid_performances):
            logger.info("増分学習が適用可能")
            return RetrainingScope.INCREMENTAL

        logger.debug("再学習は不要")
        return RetrainingScope.NONE

    def _should_incremental_update(
            self, performances: list
    ) -> bool:
        """増分学習の必要性判定
        
        Args:
            performances: 性能メトリクスのリスト
            
        Returns:
            増分学習が必要な場合はTrue
        """
        # 軽微な劣化で増分学習を提案（85-90%の範囲）
        for perf in performances:
            if perf.accuracy and 0.85 <= perf.accuracy < 0.90:
                logger.debug(f"増分学習対象: 精度 {perf.accuracy:.3f}")
                return True
        return False

    def check_cooldown_period(self, scope: RetrainingScope) -> bool:
        """冷却期間チェック
        
        再学習スコープに応じた冷却期間をチェックして、実行可能か判定します。
        
        Args:
            scope: 再学習スコープ
            
        Returns:
            実行可能な場合はTrue
        """
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
            logger.debug(f"{scope.value}は初回実行のため実行可能")
            return True

        elapsed = datetime.now() - last_time
        elapsed_hours = elapsed.total_seconds() / 3600
        
        is_ready = elapsed_hours >= required_cooldown
        
        if is_ready:
            logger.debug(f"{scope.value}冷却期間完了: {elapsed_hours:.1f}h >= {required_cooldown:.1f}h")
        else:
            logger.debug(f"{scope.value}冷却期間中: {elapsed_hours:.1f}h < {required_cooldown:.1f}h")
            
        return is_ready

    def update_retraining_time(self, scope: RetrainingScope):
        """再学習実行時刻を更新
        
        Args:
            scope: 実行された再学習スコープ
        """
        current_time = datetime.now()
        self.last_retraining[scope] = current_time
        logger.info(f"{scope.value}再学習実行時刻を更新: {current_time}")

    def get_estimated_time(self, scope: RetrainingScope) -> float:
        """推定実行時間を取得
        
        Args:
            scope: 再学習スコープ
            
        Returns:
            推定実行時間（秒）
        """
        retraining_config = self.config_manager.get_retraining_config()
        scopes_config = retraining_config.get('scopes', {})
        scope_config = scopes_config.get(scope.value, {})
        return scope_config.get('estimated_time', 1800)  # デフォルト30分

    def get_retraining_status(self) -> Dict:
        """再学習ステータスを取得
        
        Returns:
            再学習ステータス辞書
        """
        status = {
            'last_retraining_times': {},
            'cooldown_status': {},
            'next_available_times': {}
        }

        current_time = datetime.now()
        retraining_config = self.config_manager.get_retraining_config()
        cooldown_hours = retraining_config.get('cooldown_hours', 24)

        scope_cooldown = {
            RetrainingScope.INCREMENTAL: cooldown_hours * 0.25,
            RetrainingScope.SYMBOL: cooldown_hours * 0.5,
            RetrainingScope.PARTIAL: cooldown_hours * 0.75,
            RetrainingScope.GLOBAL: cooldown_hours
        }

        for scope in RetrainingScope:
            if scope == RetrainingScope.NONE:
                continue
                
            last_time = self.last_retraining.get(scope)
            status['last_retraining_times'][scope.value] = (
                last_time.isoformat() if last_time else None
            )
            
            if last_time:
                elapsed_hours = (current_time - last_time).total_seconds() / 3600
                required_hours = scope_cooldown.get(scope, cooldown_hours)
                is_ready = elapsed_hours >= required_hours
                
                status['cooldown_status'][scope.value] = {
                    'ready': is_ready,
                    'elapsed_hours': round(elapsed_hours, 2),
                    'required_hours': required_hours,
                    'remaining_hours': max(0, required_hours - elapsed_hours)
                }
                
                if not is_ready:
                    next_available = last_time.timestamp() + (required_hours * 3600)
                    next_available_dt = datetime.fromtimestamp(next_available)
                    status['next_available_times'][scope.value] = next_available_dt.isoformat()
            else:
                status['cooldown_status'][scope.value] = {
                    'ready': True,
                    'elapsed_hours': None,
                    'required_hours': scope_cooldown.get(scope, cooldown_hours),
                    'remaining_hours': 0
                }

        return status

    def reset_cooldown(self, scope: RetrainingScope):
        """冷却期間をリセット
        
        Args:
            scope: リセットする再学習スコープ
        """
        if scope in self.last_retraining:
            del self.last_retraining[scope]
            logger.info(f"{scope.value}の冷却期間をリセットしました")

    def get_recommended_action(self, scope: RetrainingScope, symbol_count: int = 0) -> str:
        """推奨アクションを取得
        
        Args:
            scope: 再学習スコープ
            symbol_count: 影響を受ける銘柄数
            
        Returns:
            推奨アクションの説明
        """
        action_map = {
            RetrainingScope.NONE: "現在の性能は良好です。継続監視を行います。",
            RetrainingScope.INCREMENTAL: "軽微な性能低下が検出されました。増分学習を実行します。",
            RetrainingScope.SYMBOL: f"特定銘柄（{symbol_count}銘柄）の性能低下。銘柄別再学習を実行します。",
            RetrainingScope.PARTIAL: "複数銘柄で性能低下が検出されました。部分再学習を実行します。",
            RetrainingScope.GLOBAL: "システム全体の性能が大幅に低下しています。全体再学習を実行します。"
        }
        
        return action_map.get(scope, "不明なスコープです。")