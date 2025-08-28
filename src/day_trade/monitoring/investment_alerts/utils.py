#!/usr/bin/env python3
"""
投資機会アラートシステム - ユーティリティ関数
"""

import asyncio

from .handlers import console_opportunity_handler, file_opportunity_handler, log_opportunity_handler
from .models import AlertConfig
from .system import InvestmentOpportunityAlertSystem
from .enums import OpportunitySeverity


async def setup_investment_opportunity_monitoring() -> InvestmentOpportunityAlertSystem:
    """投資機会監視セットアップ"""
    config = AlertConfig(
        min_confidence_score=0.7,
        min_profit_potential=5.0,
        min_risk_reward_ratio=2.0,
        max_opportunities_per_hour=5,
    )

    alert_system = InvestmentOpportunityAlertSystem(config)

    # アラートハンドラー登録
    alert_system.register_alert_handler(
        OpportunitySeverity.MEDIUM, console_opportunity_handler
    )
    alert_system.register_alert_handler(
        OpportunitySeverity.HIGH, log_opportunity_handler
    )
    alert_system.register_alert_handler(
        OpportunitySeverity.CRITICAL, file_opportunity_handler
    )

    return alert_system


async def main():
    """メイン関数"""
    alert_system = await setup_investment_opportunity_monitoring()

    # 監視開始
    await alert_system.start_monitoring()

    # 15秒間監視実行
    await asyncio.sleep(15)

    # 機会概要取得
    summary = await alert_system.get_opportunity_summary()
    print(f"アクティブ投資機会数: {summary['total_active_opportunities']}")
    print(f"平均信頼度: {summary['statistics']['average_confidence_score']:.2f}")
    print(
        f"平均利益可能性: {summary['statistics']['average_profit_potential']:.1f}%"
    )

    # アクティブ機会表示
    active_opportunities = await alert_system.get_active_opportunities()
    for opp in active_opportunities[:5]:  # 上位5件
        print(
            f"機会: {opp.message} ({opp.symbol}, 信頼度: {opp.confidence_score:.2f})"
        )

    # 監視停止
    await alert_system.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())