import unittest
import asyncio
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

from performance_tracker import PerformanceTracker, DBTrade, TradeType, TradeResult

logging.basicConfig(level=logging.INFO)

class TestPerformanceTracker(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.db_path = Path("test_tracker.db")
        self.tracker = PerformanceTracker(db_path=str(self.db_path))

    async def asyncSetUp(self):
        await self.tracker.initialize_db()

    async def asyncTearDown(self):
        await self.tracker.close()
        if self.db_path.exists():
            os.remove(self.db_path)

    async def test_full_scenario(self):
        # 1. Record a new trade
        trade1_id = "test_trade_1"
        trade1_data = {
            "trade_id": trade1_id,
            "symbol": "AAPL",
            "trade_type": TradeType.BUY.value,
            "entry_price": 150.0,
            "quantity": 10,
            "entry_date": datetime.now() - timedelta(days=10)
        }
        await self.tracker.record_trade(trade1_data)

        # 2. Verify trade was recorded
        retrieved_trade = await self.tracker.get_trade(trade1_id)
        self.assertIsNotNone(retrieved_trade)
        self.assertEqual(retrieved_trade.symbol, "AAPL")
        self.assertEqual(retrieved_trade.trade_result, TradeResult.OPEN.value)

        # 3. Close the trade with a profit
        await self.tracker.close_trade(trade1_id, exit_price=160.0)
        closed_trade = await self.tracker.get_trade(trade1_id)
        self.assertEqual(closed_trade.trade_result, TradeResult.PROFIT.value)
        self.assertAlmostEqual(closed_trade.profit_loss, 100.0)

        # 4. Check portfolio update
        portfolio = await self.tracker.get_portfolio()
        self.assertEqual(portfolio.total_trades, 1)
        self.assertEqual(portfolio.winning_trades, 1)
        self.assertAlmostEqual(portfolio.current_capital, 1000100.0)

        # 5. Calculate performance metrics
        metrics = await self.tracker.calculate_performance_metrics(days_back=15)
        self.assertEqual(metrics["total_trades"], 1)
        self.assertEqual(metrics["win_rate"], 100.0)
        self.assertAlmostEqual(metrics["total_pnl"], 100.0)

if __name__ == '__main__':
    unittest.main()