import asyncio
import numpy as np
import logging
import uuid
from datetime import datetime, timedelta

# Import necessary classes and functions from the main module
from performance_tracker import PerformanceTracker, Trade, TradeType, TradeResult, RiskLevel, Portfolio, PerformanceMetrics

# Configure logging for the test file
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_performance_tracker():
    """パフォーマンス追跡システムのテスト"""
    logger.info("=== 包括的パフォーマンス追跡システム テスト ===")

    tracker = PerformanceTracker()
    await tracker.ainit() # Call the async initializer

    logger.info(f"デフォルトポートフォリオID: {tracker.default_portfolio_id}")
    logger.info(f"ベンチマーク期待リターン: {tracker.benchmark_return * 100}%")

    # サンプル取引データ作成・記録
    logger.info(f"\n[ サンプル取引データ作成 ]")
    sample_trades = []

    for i in range(30):
        # エントリー
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            symbol="7203",
            name="トヨタ自動車",
            trade_type=TradeType.BUY,
            entry_date=datetime.now() - timedelta(days=i*2),
            entry_price=3000 + np.random.randint(-50, 50),
            quantity=100,
            entry_amount=300000 + np.random.randint(-5000, 5000),
            risk_level=RiskLevel.LOW if i % 3 == 0 else RiskLevel.MEDIUM,
            confidence_score=np.random.uniform(70, 95),
            sector="自動車",
            theme="EV・蓄電池",
            trading_session=f"SESSION_{i//10}",
            strategy_used="デイトレード戦略",
            notes=f"テスト取引{i}"
        )

        # 決済（70%の確率で決済済み）
        if np.random.random() < 0.7:
            trade.exit_date = trade.entry_date + timedelta(hours=np.random.randint(1, 8))
            trade.exit_price = trade.entry_price * (1 + np.random.uniform(-0.08, 0.12))
            trade.exit_amount = trade.exit_price * trade.quantity
            trade.profit_loss = trade.exit_amount - trade.entry_amount
            trade.profit_loss_pct = trade.profit_loss / trade.entry_amount * 100
            trade.trade_result = TradeResult.PROFIT if trade.profit_loss > 0 else TradeResult.LOSS

        await tracker.record_trade(trade)
        sample_trades.append(trade)

    logger.info(f"作成した取引データ: {len(sample_trades)}件")

    # ポートフォリオ状況確認
    logger.info(f"\n[ ポートフォリオ状況 ]")
    portfolio = await tracker.get_portfolio()
    if portfolio:
        logger.info(f"初期資本: {portfolio.initial_capital:,.0f}円")
        logger.info(f"現在資本: {portfolio.current_capital:,.0f}円")
        logger.info(f"総リターン: {portfolio.total_return:.2f}%")
        logger.info(f"総取引数: {portfolio.total_trades}")
        logger.info(f"勝率: {portfolio.win_rate:.1f}%")

    # パフォーマンス指標計算
    logger.info(f"\n[ パフォーマンス指標 (30日) ]")
    metrics = await tracker.calculate_performance_metrics(30)

    logger.info(f"総リターン: {metrics.total_return_pct:.2f}%")
    logger.info(f"年率換算: {metrics.annualized_return:.2f}%")
    logger.info(f"ボラティリティ: {metrics.volatility:.2f}%")
    logger.info(f"シャープレシオ: {metrics.sharpe_ratio:.2f}")
    logger.info(f"最大ドローダウン: {metrics.max_drawdown:.2f}%")
    logger.info(f"プロフィットファクター: {metrics.profit_factor:.2f}")
    logger.info(f"アルファ: {metrics.alpha:.2f}%")

    # 包括的レポート生成
    logger.info(f"\n[ 包括的パフォーマンスレポート ]")
    report = await tracker.generate_comprehensive_report()

    if "error" not in report:
        portfolio_summary = report["portfolio_summary"]
        logger.info(f"ポートフォリオ名: {portfolio_summary['portfolio_name']}")
        logger.info(f"総リターン: {portfolio_summary['total_return']:.2f}%")
        logger.info(f"勝率: {portfolio_summary['win_rate']:.1f}%")

        perf_30d = report["performance_metrics"]["30_days"]
        logger.info(f"\n30日パフォーマンス:")
        logger.info(f"  年率リターン: {perf_30d['annualized_return']:.2f}%")
        logger.info(f"  シャープレシオ: {perf_30d['sharpe_ratio']:.2f}")

        risk_analysis = report["risk_analysis"]
        logger.info(f"\nリスク分析:")
        logger.info(f"  リスクレベル: {risk_analysis.get('risk_level', 'N/A')}")
        logger.info(f"  分散化スコア: {risk_analysis.get('diversification_score', 0):.1f}")

        risk_recs = risk_analysis.get('risk_recommendations', [])
        if risk_recs:
            logger.info(f"\nリスク管理提言:")
            for rec in risk_recs[:2]:
                logger.info(f"  • {rec}")
    else:
        logger.error(f"レポート生成エラー: {report['error']}")

    logger.info(f"\n=== 包括的パフォーマンス追跡システム テスト完了 ===")

if __name__ == "__main__":
    asyncio.run(test_performance_tracker())