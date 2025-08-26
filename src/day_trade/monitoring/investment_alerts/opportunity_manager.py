#!/usr/bin/env python3
"""
投資機会アラートシステム - 機会管理
"""

import asyncio
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from ...utils.logging_config import get_context_logger
from .enums import OpportunitySeverity, OpportunityType
from .models import AlertConfig, InvestmentOpportunity

logger = get_context_logger(__name__)


class OpportunityManager:
    """投資機会管理クラス"""

    def __init__(self, config: AlertConfig):
        """機会管理の初期化"""
        self.config = config
        self.opportunities: List[InvestmentOpportunity] = []
        self.alert_handlers: Dict[OpportunitySeverity, List[Callable]] = {
            severity: [] for severity in OpportunitySeverity
        }
        self.recent_opportunities: Dict[str, datetime] = {}

    def register_alert_handler(
        self,
        severity: OpportunitySeverity,
        handler: Callable,
    ) -> None:
        """アラートハンドラー登録"""
        self.alert_handlers[severity].append(handler)
        logger.info(f"機会アラートハンドラー登録: {severity.value}")

    async def process_opportunity(self, opportunity: InvestmentOpportunity) -> None:
        """機会処理"""
        # 市場状況フィルター
        if self.config.enable_market_condition_filter:
            if not await self._passes_market_filter(opportunity):
                return

        # 品質フィルター
        if not await self._passes_quality_filter(opportunity):
            return

        # アラート生成
        await self._generate_opportunity_alert(opportunity)

    async def _passes_market_filter(self, opportunity: InvestmentOpportunity) -> bool:
        """市場状況フィルター"""
        # TODO: 市場状況の参照方法を修正する必要がある
        # market_condition = self.detector.market_condition
        # if not market_condition:
        #     return True
        
        # # 高ボラティリティ期間を避ける
        # if (
        #     self.config.avoid_high_volatility_periods
        #     and market_condition.volatility_level == "high"
        # ):
        #     return False

        # return not (
        #     market_condition.market_sentiment < self.config.min_market_sentiment
        # )
        
        # 一時的に常にTrueを返す
        return True

    async def _passes_quality_filter(self, opportunity: InvestmentOpportunity) -> bool:
        """品質フィルター"""
        # 最低信頼度チェック
        if opportunity.confidence_score < self.config.min_confidence_score:
            return False

        # 最低利益可能性チェック
        if opportunity.profit_potential < self.config.min_profit_potential:
            return False

        return not (opportunity.risk_reward_ratio < self.config.min_risk_reward_ratio)

    async def _generate_opportunity_alert(
        self,
        opportunity: InvestmentOpportunity,
    ) -> None:
        """機会アラート生成"""
        alert_key = f"{opportunity.symbol}_{opportunity.opportunity_type.value}"

        # アラート抑制チェック
        if not await self._should_generate_opportunity_alert(alert_key):
            return

        # 機会追加
        self.opportunities.append(opportunity)

        # アラートハンドラー実行
        await self._execute_opportunity_handlers(opportunity)

        # アラート抑制記録
        self.recent_opportunities[alert_key] = datetime.now()

        logger.info(f"投資機会アラート: {opportunity.message}")

    async def _should_generate_opportunity_alert(self, alert_key: str) -> bool:
        """機会アラート生成判定"""
        if alert_key in self.recent_opportunities:
            last_alert = self.recent_opportunities[alert_key]
            elapsed_minutes = (datetime.now() - last_alert).total_seconds() / 60
            if elapsed_minutes < self.config.opportunity_cooldown_minutes:
                return False

        # 時間あたりアラート数制限
        current_hour = datetime.now().hour
        hourly_count = len(
            [
                opp
                for opp in self.opportunities
                if opp.timestamp.hour == current_hour
                and opp.timestamp.date() == datetime.now().date()
            ]
        )

        return not (hourly_count >= self.config.max_opportunities_per_hour)

    async def _execute_opportunity_handlers(
        self,
        opportunity: InvestmentOpportunity,
    ) -> None:
        """機会ハンドラー実行"""
        handlers = self.alert_handlers.get(opportunity.severity, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(opportunity)
                else:
                    handler(opportunity)
            except Exception as e:
                logger.error(f"機会アラートハンドラー実行エラー: {e}")

    async def cleanup_opportunity_history(self) -> None:
        """機会履歴クリーンアップ"""
        cutoff_time = datetime.now() - timedelta(
            days=self.config.alert_history_retention_days
        )

        # 機会履歴クリーンアップ
        self.opportunities = [
            opp for opp in self.opportunities if opp.timestamp > cutoff_time
        ]

        # 最近の機会記録クリーンアップ
        recent_cutoff = datetime.now() - timedelta(
            minutes=self.config.opportunity_cooldown_minutes * 2
        )
        self.recent_opportunities = {
            key: timestamp
            for key, timestamp in self.recent_opportunities.items()
            if timestamp > recent_cutoff
        }

    async def get_active_opportunities(
        self,
        symbol: Optional[str] = None,
        opportunity_type: Optional[OpportunityType] = None,
        severity: Optional[OpportunitySeverity] = None,
    ) -> List[InvestmentOpportunity]:
        """アクティブ機会取得"""
        active_opportunities = [opp for opp in self.opportunities if not opp.executed]

        if symbol:
            active_opportunities = [
                opp for opp in active_opportunities if opp.symbol == symbol
            ]

        if opportunity_type:
            active_opportunities = [
                opp
                for opp in active_opportunities
                if opp.opportunity_type == opportunity_type
            ]

        if severity:
            active_opportunities = [
                opp for opp in active_opportunities if opp.severity == severity
            ]

        return sorted(active_opportunities, key=lambda o: o.timestamp, reverse=True)

    async def execute_opportunity(
        self,
        opportunity_id: str,
        execution_note: Optional[str] = None,
    ) -> bool:
        """機会実行"""
        for opportunity in self.opportunities:
            if (
                opportunity.opportunity_id == opportunity_id
                and not opportunity.executed
            ):
                opportunity.executed = True
                opportunity.executed_at = datetime.now()
                if execution_note:
                    opportunity.details["execution_note"] = execution_note

                logger.info(f"投資機会実行: {opportunity_id}")
                return True

        return False