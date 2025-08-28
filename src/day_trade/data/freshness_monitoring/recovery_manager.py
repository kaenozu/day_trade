#!/usr/bin/env python3
"""
回復管理モジュール
データソースの回復アクションの評価と実行を管理
"""

import asyncio
import logging
from collections import defaultdict
from typing import Callable, Dict, List

from .database import DatabaseManager
from .models import (
    DataSourceConfig,
    FreshnessCheck,
    FreshnessStatus,
    IntegrityCheck,
    RecoveryAction,
)


class RecoveryManager:
    """回復管理クラス"""

    def __init__(self, database_manager: DatabaseManager):
        self.logger = logging.getLogger(__name__)
        self.database_manager = database_manager
        self.recovery_callbacks: Dict[RecoveryAction, List[Callable]] = defaultdict(list)

    def add_recovery_callback(self, action: RecoveryAction, callback: Callable):
        """回復アクションコールバック追加"""
        self.recovery_callbacks[action].append(callback)

    async def evaluate_recovery_actions(
        self,
        config: DataSourceConfig,
        freshness: FreshnessCheck,
        integrity: List[IntegrityCheck],
    ):
        """回復アクション評価・実行"""
        if not config.enable_recovery:
            return

        try:
            # 回復が必要かどうかを判定
            needs_recovery = await self._needs_recovery(freshness, integrity)
            if not needs_recovery:
                return

            # 連続失敗回数と回復試行回数を確認
            source_state = self.database_manager.get_source_state(config.source_id)
            if not source_state:
                self.logger.warning(f"データソース状態が見つかりません: {config.source_id}")
                return

            consecutive_failures = source_state[0] if source_state else 0
            recovery_attempts = source_state[1] if source_state else 0

            # 最大試行回数チェック
            if recovery_attempts >= config.max_retry_attempts:
                self.logger.warning(
                    f"最大回復試行回数に到達: {config.source_id} ({recovery_attempts}/{config.max_retry_attempts})"
                )
                return

            # 回復戦略に応じてアクション実行
            await self._execute_recovery_action(config, consecutive_failures)

        except Exception as e:
            self.logger.error(f"回復アクション評価エラー ({config.source_id}): {e}")

    async def _needs_recovery(
        self, freshness: FreshnessCheck, integrity: List[IntegrityCheck]
    ) -> bool:
        """回復が必要かどうかを判定"""
        # 鮮度問題
        if freshness.status != FreshnessStatus.FRESH:
            return True

        # 整合性問題
        if any(not check.passed for check in integrity):
            return True

        return False

    async def _execute_recovery_action(
        self, config: DataSourceConfig, consecutive_failures: int
    ):
        """回復アクション実行"""
        try:
            if config.recovery_strategy == RecoveryAction.RETRY:
                await self._execute_retry_action(config, consecutive_failures)
            elif config.recovery_strategy == RecoveryAction.FALLBACK:
                await self._execute_fallback_action(config, consecutive_failures)
            elif config.recovery_strategy == RecoveryAction.MANUAL:
                await self._execute_manual_action(config, consecutive_failures)
            elif config.recovery_strategy == RecoveryAction.DISABLE:
                await self._execute_disable_action(config, consecutive_failures)

            # 回復試行回数更新
            self.database_manager.update_recovery_attempts(config.source_id)

        except Exception as e:
            self.logger.error(f"回復アクション実行エラー ({config.source_id}): {e}")

    async def _execute_retry_action(
        self, config: DataSourceConfig, consecutive_failures: int
    ):
        """リトライアクション実行"""
        self.logger.info(f"データソースリトライ実行: {config.source_id}")

        # 段階的な待機時間（指数バックオフ）
        wait_time = min(5 * (2 ** consecutive_failures), 300)  # 最大5分
        await asyncio.sleep(wait_time)

        # 実際の実装では、データソースへの再接続・再取得を実行
        # ここではログ出力のみ
        self.logger.info(f"リトライ実行完了: {config.source_id} (待機時間: {wait_time}秒)")

        # コールバック実行
        await self._execute_callbacks(RecoveryAction.RETRY, config)

    async def _execute_fallback_action(
        self, config: DataSourceConfig, consecutive_failures: int
    ):
        """フェイルオーバーアクション実行"""
        self.logger.info(f"フェイルオーバー実行: {config.source_id}")

        # 実際の実装では、バックアップソースに切り替え
        # 設定ファイルから代替エンドポイントを取得して切り替える
        fallback_url = config.connection_params.get("fallback_url")
        if fallback_url:
            self.logger.info(f"代替エンドポイントに切り替え: {fallback_url}")
        else:
            self.logger.warning(f"代替エンドポイントが設定されていません: {config.source_id}")

        # コールバック実行
        await self._execute_callbacks(RecoveryAction.FALLBACK, config)

    async def _execute_manual_action(
        self, config: DataSourceConfig, consecutive_failures: int
    ):
        """手動介入アクション実行"""
        self.logger.warning(
            f"手動介入が必要です: {config.source_id} (連続失敗: {consecutive_failures}回)"
        )

        # 実際の実装では、管理者への通知を送信
        # メール、Slack、PagerDutyなどの通知システムを呼び出す
        
        # コールバック実行
        await self._execute_callbacks(RecoveryAction.MANUAL, config)

    async def _execute_disable_action(
        self, config: DataSourceConfig, consecutive_failures: int
    ):
        """無効化アクション実行"""
        self.logger.warning(
            f"データソースを一時無効化: {config.source_id} (連続失敗: {consecutive_failures}回)"
        )

        # 実際の実装では、監視対象から一時的に除外
        # またはポーリング頻度を大幅に下げる
        
        # コールバック実行
        await self._execute_callbacks(RecoveryAction.DISABLE, config)

    async def _execute_callbacks(self, action: RecoveryAction, config: DataSourceConfig):
        """回復アクションコールバック実行"""
        callbacks = self.recovery_callbacks.get(action, [])
        for callback in callbacks:
            try:
                await callback(config)
            except Exception as e:
                self.logger.error(f"回復コールバックエラー ({action.value}): {e}")

    def get_recovery_strategy_description(self, strategy: RecoveryAction) -> str:
        """回復戦略の説明取得"""
        descriptions = {
            RecoveryAction.RETRY: "データソースへの再接続・再取得を試行",
            RecoveryAction.FALLBACK: "代替データソースへのフェイルオーバー",
            RecoveryAction.MANUAL: "管理者への手動介入要請",
            RecoveryAction.DISABLE: "データソースの一時的な無効化",
        }
        return descriptions.get(strategy, "不明な回復戦略")

    async def test_recovery_action(self, config: DataSourceConfig) -> bool:
        """回復アクションのテスト実行"""
        try:
            self.logger.info(f"回復アクションテスト開始: {config.source_id}")
            
            if config.recovery_strategy == RecoveryAction.RETRY:
                # リトライテスト（短時間）
                await asyncio.sleep(1)
                success = True
            elif config.recovery_strategy == RecoveryAction.FALLBACK:
                # フェイルオーバーテスト
                fallback_url = config.connection_params.get("fallback_url")
                success = fallback_url is not None
            else:
                # その他のアクションは常に成功とみなす
                success = True
                
            self.logger.info(f"回復アクションテスト結果: {config.source_id} = {'成功' if success else '失敗'}")
            return success
            
        except Exception as e:
            self.logger.error(f"回復アクションテストエラー ({config.source_id}): {e}")
            return False

    def get_recovery_statistics(self, source_id: str) -> dict:
        """回復統計情報取得"""
        try:
            source_state = self.database_manager.get_source_state(source_id)
            if not source_state:
                return {"error": "データソース状態が見つかりません"}

            return {
                "source_id": source_id,
                "consecutive_failures": source_state[0] if source_state else 0,
                "recovery_attempts": source_state[1] if source_state else 0,
                "current_status": source_state[2] if source_state else "unknown",
            }

        except Exception as e:
            self.logger.error(f"回復統計取得エラー ({source_id}): {e}")
            return {"error": str(e)}