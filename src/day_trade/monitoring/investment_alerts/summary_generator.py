#!/usr/bin/env python3
"""
投資機会アラートシステム - 概要生成器
"""

from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Optional

from .enums import OpportunityType, OpportunitySeverity
from .models import InvestmentOpportunity


class SummaryGenerator:
    """概要生成器"""

    @staticmethod
    async def generate_opportunity_summary(
        opportunities: List[InvestmentOpportunity],
        active_configs_count: int,
        total_configs_count: int,
        is_running: bool,
        market_condition: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """機会概要生成"""
        active_opportunities = [opp for opp in opportunities if not opp.executed]

        # 重要度別集計
        severity_counts = {}
        for severity in OpportunitySeverity:
            severity_counts[severity.value] = len(
                [opp for opp in active_opportunities if opp.severity == severity]
            )

        # 機会タイプ別集計
        type_counts = {}
        for opp_type in OpportunityType:
            type_counts[opp_type.value] = len(
                [
                    opp
                    for opp in active_opportunities
                    if opp.opportunity_type == opp_type
                ]
            )

        # 銘柄別集計
        symbol_counts = {}
        for opp in active_opportunities:
            symbol_counts[opp.symbol] = symbol_counts.get(opp.symbol, 0) + 1

        # 統計情報
        if active_opportunities:
            avg_confidence = mean(
                [opp.confidence_score for opp in active_opportunities]
            )
            avg_profit_potential = mean(
                [opp.profit_potential for opp in active_opportunities]
            )
        else:
            avg_confidence = 0
            avg_profit_potential = 0

        return {
            "timestamp": datetime.now().isoformat(),
            "total_active_opportunities": len(active_opportunities),
            "severity_breakdown": severity_counts,
            "type_breakdown": type_counts,
            "symbol_breakdown": dict(
                sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),  # トップ10
            "statistics": {
                "average_confidence_score": avg_confidence,
                "average_profit_potential": avg_profit_potential,
                "total_opportunities_generated": len(opportunities),
                "executed_opportunities": len(
                    [opp for opp in opportunities if opp.executed]
                ),
            },
            "market_condition": {
                "market_trend": (
                    market_condition.market_trend if market_condition else None
                ),
                "volatility_level": (
                    market_condition.volatility_level if market_condition else None
                ),
                "market_sentiment": (
                    market_condition.market_sentiment if market_condition else None
                ),
            },
            "monitoring_status": {
                "is_running": is_running,
                "active_configs": active_configs_count,
                "total_configs": total_configs_count,
            },
        }