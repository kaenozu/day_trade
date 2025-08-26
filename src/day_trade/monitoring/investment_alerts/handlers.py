#!/usr/bin/env python3
"""
投資機会アラートシステム - アラートハンドラー
"""

import json
from pathlib import Path

from ...utils.logging_config import get_context_logger
from .models import InvestmentOpportunity

logger = get_context_logger(__name__)


async def log_opportunity_handler(opportunity: InvestmentOpportunity) -> None:
    """ログ機会ハンドラー"""
    logger.info(
        f"投資機会: {opportunity.message} "
        f"(信頼度: {opportunity.confidence_score:.2f}, "
        f"利益可能性: {opportunity.profit_potential:.1f}%)"
    )


def console_opportunity_handler(opportunity: InvestmentOpportunity) -> None:
    """コンソール機会ハンドラー"""
    print(
        f"[{opportunity.timestamp.strftime('%H:%M:%S')}] "
        f"{opportunity.severity.value.upper()}: {opportunity.message}"
    )
    print(
        f"  アクション: {opportunity.recommended_action.value}, "
        f"現在価格: {opportunity.current_price:.2f}"
    )
    print(
        f"  利益可能性: {opportunity.profit_potential:.1f}%, "
        f"信頼度: {opportunity.confidence_score:.2f}"
    )


async def file_opportunity_handler(opportunity: InvestmentOpportunity) -> None:
    """ファイル機会ハンドラー"""
    alert_file = Path("alerts") / "investment_opportunities.log"
    alert_file.parent.mkdir(exist_ok=True)

    opportunity_data = {
        "timestamp": opportunity.timestamp.isoformat(),
        "opportunity_id": opportunity.opportunity_id,
        "symbol": opportunity.symbol,
        "opportunity_type": opportunity.opportunity_type.value,
        "severity": opportunity.severity.value,
        "recommended_action": opportunity.recommended_action.value,
        "current_price": opportunity.current_price,
        "target_price": opportunity.target_price,
        "profit_potential": opportunity.profit_potential,
        "confidence_score": opportunity.confidence_score,
        "time_horizon": opportunity.time_horizon.value,
        "risk_level": opportunity.risk_level,
        "message": opportunity.message,
        "technical_indicators": opportunity.technical_indicators,
    }

    with open(alert_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(opportunity_data, ensure_ascii=False) + "\n")