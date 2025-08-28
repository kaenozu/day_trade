#!/usr/bin/env python3
"""
回復・復旧管理機能
データソースの問題に対する自動回復アクションを管理します
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List

from .database_operations import DatabaseOperations
from .enums import RecoveryAction
from .models import DataSourceConfig, DataSourceState, FreshnessCheck, IntegrityCheck, MonitoringStats


class RecoveryManager:
    """回復・復旧管理システム
    
    データソースの問題を検出した際に、設定に基づいて
    自動回復アクションを実行する機能を提供します。
    """
    
    def __init__(self, db_operations: DatabaseOperations):
        """回復管理システムを初期化
        
        Args:
            db_operations: データベース操作インスタンス
        """
        self.logger = logging.getLogger(__name__)
        self.db_operations = db_operations
        
        # 回復アクションコールバック
        self.recovery_callbacks: Dict[RecoveryAction, List[Callable]] = defaultdict(list)
        
        # 統計情報
        self.stats = MonitoringStats()
        
        # 回復実行履歴
        self.recovery_history: Dict[str, List[Dict]] = defaultdict(list)
    
    async def evaluate_recovery_actions(
        self,
        config: DataSourceConfig,
        freshness_check: FreshnessCheck,
        integrity_checks: List[IntegrityCheck]
    ):
        """回復アクション評価・実行
        
        データソースの状態を評価し、必要に応じて回復アクションを実行します。
        
        Args:
            config: データソース設定
            freshness_check: 鮮度チェック結果
            integrity_checks: 整合性チェック結果リスト
        """
        if not config.enable_recovery:
            self.logger.debug(f"回復機能無効: {config.source_id}")
            return
        
        try:
            # データソース状態取得
            source_state = self.db_operations.get_source_state(config.source_id)
            if not source_state:
                # 初回の場合は状態を作成
                source_state = DataSourceState(
                    source_id=config.source_id,
                    current_status="unknown",
                    consecutive_failures=0,
                    recovery_attempts=0
                )
            
            # 問題判定
            has_issues = await self._assess_data_health(
                freshness_check, integrity_checks
            )
            
            if not has_issues:
                # 問題なし - 状態をリセット
                await self._reset_recovery_state(config, source_state)
                return
            
            # 回復が必要
            self.logger.info(f"回復アクション評価開始: {config.source_id}")
            
            # 回復条件チェック
            if await self._should_attempt_recovery(config, source_state):
                await self._execute_recovery_strategy(config, source_state)
                self.stats.recovery_actions_taken += 1
            else:
                self.logger.info(
                    f"回復条件未満: {config.source_id} "
                    f"(試行回数: {source_state.recovery_attempts}/{config.max_retry_attempts})"
                )
            
        except Exception as e:
            self.logger.error(f"回復アクション評価エラー ({config.source_id}): {e}")
    
    async def _assess_data_health(
        self,
        freshness_check: FreshnessCheck,
        integrity_checks: List[IntegrityCheck]
    ) -> bool:
        """データ健全性評価
        
        Args:
            freshness_check: 鮮度チェック結果
            integrity_checks: 整合性チェック結果リスト
            
        Returns:
            問題があるかどうか（True: 問題あり, False: 健全）
        """
        from .enums import FreshnessStatus
        
        # 鮮度問題チェック
        if freshness_check.status != FreshnessStatus.FRESH:
            return True
        
        # 整合性問題チェック
        for check in integrity_checks:
            if not check.passed:
                return True
        
        return False
    
    async def _should_attempt_recovery(
        self,
        config: DataSourceConfig,
        source_state: DataSourceState
    ) -> bool:
        """回復実行判定
        
        Args:
            config: データソース設定
            source_state: データソース状態
            
        Returns:
            回復を実行すべきかどうか
        """
        # 最大回復試行回数チェック
        if source_state.recovery_attempts >= config.max_retry_attempts:
            self.logger.warning(
                f"最大回復試行回数に到達: {config.source_id} "
                f"({source_state.recovery_attempts}/{config.max_retry_attempts})"
            )
            return False
        
        # 連続失敗回数による判定
        if source_state.consecutive_failures < 2:
            self.logger.debug(
                f"連続失敗回数不足: {config.source_id} "
                f"({source_state.consecutive_failures}回)"
            )
            return False
        
        # クールダウン期間チェック（前回の回復から一定時間経過）
        if await self._is_in_cooldown_period(config.source_id):
            self.logger.debug(f"クールダウン期間中: {config.source_id}")
            return False
        
        return True
    
    async def _is_in_cooldown_period(self, source_id: str) -> bool:
        """クールダウン期間判定
        
        Args:
            source_id: データソースID
            
        Returns:
            クールダウン期間中かどうか
        """
        history = self.recovery_history.get(source_id, [])
        if not history:
            return False
        
        # 最後の回復実行から30分以内はクールダウン
        last_recovery = history[-1]
        last_time = datetime.fromisoformat(last_recovery["timestamp"])
        current_time = datetime.now(timezone.utc)
        
        cooldown_minutes = 30
        time_diff = (current_time - last_time).total_seconds() / 60
        
        return time_diff < cooldown_minutes
    
    async def _execute_recovery_strategy(
        self,
        config: DataSourceConfig,
        source_state: DataSourceState
    ):
        """回復戦略実行
        
        Args:
            config: データソース設定
            source_state: データソース状態
        """
        strategy = config.recovery_strategy
        
        self.logger.info(
            f"回復戦略実行: {config.source_id} - {strategy.value}"
        )
        
        recovery_result = None
        
        try:
            if strategy == RecoveryAction.RETRY:
                recovery_result = await self._execute_retry_action(config)
            
            elif strategy == RecoveryAction.FALLBACK:
                recovery_result = await self._execute_fallback_action(config)
            
            elif strategy == RecoveryAction.MANUAL:
                recovery_result = await self._execute_manual_action(config)
            
            elif strategy == RecoveryAction.DISABLE:
                recovery_result = await self._execute_disable_action(config)
            
            else:
                self.logger.warning(f"未知の回復戦略: {strategy}")
                return
            
            # 回復実行記録
            await self._record_recovery_attempt(config, strategy, recovery_result)
            
            # データソース状態更新
            source_state.recovery_attempts += 1
            await self.db_operations.update_source_state(
                config.source_id, source_state
            )
            
        except Exception as e:
            self.logger.error(f"回復戦略実行エラー ({config.source_id}): {e}")
            await self._record_recovery_attempt(
                config, strategy, {"success": False, "error": str(e)}
            )
    
    async def _execute_retry_action(
        self, 
        config: DataSourceConfig
    ) -> Dict:
        """リトライアクション実行
        
        Args:
            config: データソース設定
            
        Returns:
            実行結果
        """
        self.logger.info(f"データソースリトライ実行: {config.source_id}")
        
        # リトライ前の待機
        await asyncio.sleep(5)
        
        # 実際の実装では:
        # - API接続の再試行
        # - データベース接続の再確立
        # - ファイル読み込みの再実行
        # - キャッシュクリア
        # など、データソース種別に応じたリトライ処理を実行
        
        # コールバック実行
        await self._execute_recovery_callbacks(RecoveryAction.RETRY, config)
        
        return {
            "success": True,
            "action": "retry",
            "details": "データソース接続リトライ実行",
            "wait_time_seconds": 5,
        }
    
    async def _execute_fallback_action(
        self, 
        config: DataSourceConfig
    ) -> Dict:
        """フォールバックアクション実行
        
        Args:
            config: データソース設定
            
        Returns:
            実行結果
        """
        self.logger.info(f"フォールバック実行: {config.source_id}")
        
        # 実際の実装では:
        # - バックアップデータソースへの切り替え
        # - キャッシュデータの利用
        # - 代替エンドポイントの使用
        # - データの補間処理
        # など、フォールバック処理を実行
        
        # コールバック実行
        await self._execute_recovery_callbacks(RecoveryAction.FALLBACK, config)
        
        return {
            "success": True,
            "action": "fallback",
            "details": "バックアップデータソースに切り替え",
            "fallback_source": f"backup_{config.source_id}",
        }
    
    async def _execute_manual_action(
        self, 
        config: DataSourceConfig
    ) -> Dict:
        """手動介入要求アクション実行
        
        Args:
            config: データソース設定
            
        Returns:
            実行結果
        """
        self.logger.warning(f"手動介入要求: {config.source_id}")
        
        # 実際の実装では:
        # - 運用チームへの通知送信
        # - チケットシステムでのインシデント作成
        # - 緊急連絡の実施
        # - 監視システムでのアラート強化
        # など、手動介入を促す処理を実行
        
        # コールバック実行
        await self._execute_recovery_callbacks(RecoveryAction.MANUAL, config)
        
        return {
            "success": True,
            "action": "manual_intervention",
            "details": "運用チームに手動介入を要求",
            "notification_sent": True,
        }
    
    async def _execute_disable_action(
        self, 
        config: DataSourceConfig
    ) -> Dict:
        """無効化アクション実行
        
        Args:
            config: データソース設定
            
        Returns:
            実行結果
        """
        self.logger.warning(f"データソース無効化: {config.source_id}")
        
        # 実際の実装では:
        # - データソースの監視停止
        # - 依存サービスへの通知
        # - 代替処理への切り替え
        # - 状態の永続化
        # など、データソース無効化処理を実行
        
        # コールバック実行
        await self._execute_recovery_callbacks(RecoveryAction.DISABLE, config)
        
        return {
            "success": True,
            "action": "disable",
            "details": f"データソース {config.source_id} を無効化",
            "disabled_until_manual_reactivation": True,
        }
    
    async def _execute_recovery_callbacks(
        self,
        action: RecoveryAction,
        config: DataSourceConfig
    ):
        """回復アクションコールバック実行
        
        Args:
            action: 回復アクション種別
            config: データソース設定
        """
        callbacks = self.recovery_callbacks.get(action, [])
        
        for callback in callbacks:
            try:
                await callback(config, action)
            except Exception as e:
                self.logger.error(
                    f"回復コールバックエラー ({action.value}): {e}"
                )
    
    async def _record_recovery_attempt(
        self,
        config: DataSourceConfig,
        strategy: RecoveryAction,
        result: Dict
    ):
        """回復試行記録
        
        Args:
            config: データソース設定
            strategy: 実行した回復戦略
            result: 実行結果
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy.value,
            "result": result,
            "source_config": {
                "source_type": config.source_type.value,
                "monitoring_level": config.monitoring_level.value,
            }
        }
        
        self.recovery_history[config.source_id].append(record)
        
        # 履歴サイズ制限（直近50件まで）
        if len(self.recovery_history[config.source_id]) > 50:
            self.recovery_history[config.source_id] = self.recovery_history[config.source_id][-50:]
        
        self.logger.info(
            f"回復試行記録: {config.source_id} - "
            f"{strategy.value} ({'成功' if result.get('success') else '失敗'})"
        )
    
    async def _reset_recovery_state(
        self,
        config: DataSourceConfig,
        source_state: DataSourceState
    ):
        """回復状態リセット
        
        Args:
            config: データソース設定
            source_state: データソース状態
        """
        # 連続失敗回数と回復試行回数をリセット
        source_state.consecutive_failures = 0
        source_state.recovery_attempts = 0
        source_state.current_status = "healthy"
        source_state.last_success = datetime.now(timezone.utc)
        
        await self.db_operations.update_source_state(
            config.source_id, source_state
        )
    
    def add_recovery_callback(
        self, 
        action: RecoveryAction, 
        callback: Callable
    ):
        """回復アクションコールバック追加
        
        Args:
            action: 回復アクション種別
            callback: 実行するコールバック関数
        """
        self.recovery_callbacks[action].append(callback)
        self.logger.info(f"回復コールバック追加: {action.value}")
    
    def remove_recovery_callback(
        self, 
        action: RecoveryAction, 
        callback: Callable
    ):
        """回復アクションコールバック削除
        
        Args:
            action: 回復アクション種別
            callback: 削除するコールバック関数
        """
        if callback in self.recovery_callbacks[action]:
            self.recovery_callbacks[action].remove(callback)
            self.logger.info(f"回復コールバック削除: {action.value}")
    
    def get_recovery_statistics(self) -> Dict:
        """回復統計情報取得
        
        Returns:
            回復統計情報
        """
        # 回復履歴の統計計算
        total_attempts = sum(
            len(history) for history in self.recovery_history.values()
        )
        
        successful_attempts = 0
        strategy_counts = defaultdict(int)
        
        for history in self.recovery_history.values():
            for record in history:
                if record.get("result", {}).get("success", False):
                    successful_attempts += 1
                strategy_counts[record["strategy"]] += 1
        
        success_rate = (
            (successful_attempts / total_attempts * 100) 
            if total_attempts > 0 else 0.0
        )
        
        return {
            "total_recovery_attempts": total_attempts,
            "successful_recovery_attempts": successful_attempts,
            "recovery_success_rate": round(success_rate, 2),
            "recovery_by_strategy": dict(strategy_counts),
            "sources_with_recovery_history": len(self.recovery_history),
            "actions_taken_total": self.stats.recovery_actions_taken,
        }
    
    def get_recovery_history(
        self, 
        source_id: str, 
        limit: int = 10
    ) -> List[Dict]:
        """回復履歴取得
        
        Args:
            source_id: データソースID
            limit: 取得件数制限
            
        Returns:
            回復履歴のリスト
        """
        history = self.recovery_history.get(source_id, [])
        return sorted(
            history, 
            key=lambda x: x["timestamp"], 
            reverse=True
        )[:limit]