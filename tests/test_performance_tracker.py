
import unittest
import asyncio
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
import uuid

# performance_tracker.pyから必要なクラスをインポート
from performance_tracker import PerformanceTracker, Trade, Portfolio, PerformanceMetrics

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestPerformanceTracker(unittest.IsolatedAsyncioTestCase):
    """PerformanceTrackerの統合テスト"""

    def setUp(self):
        """テストのセットアップ"""
        self.db_path = Path(f"test_performance_{uuid.uuid4()}.db")
        self.tracker = PerformanceTracker(db_path=str(self.db_path))

    async def asyncTearDown(self):
        """テストの後片付け"""
        await self.tracker.close()
        if self.db_path.exists():
            os.remove(self.db_path)

    async def test_full_performance_tracking_scenario(self):
        """
        トレードの追加からパフォーマンス計算、レポート生成までの一連のシナリオをテストする。
        """
        # 1. データベースの初期化をテスト
        await self.tracker.initialize_db()
        self.assertTrue(self.db_path.exists())

        # 2. トレードを追加
        trades = [
            Trade(symbol="AAPL", entry_price=150.0, exit_price=155.0, quantity=10, trade_type="BUY", entry_time=datetime.now() - timedelta(days=5), exit_time=datetime.now() - timedelta(days=4)),
            Trade(symbol="GOOG", entry_price=2800.0, exit_price=2750.0, quantity=2, trade_type="SELL", entry_time=datetime.now() - timedelta(days=3), exit_time=datetime.now() - timedelta(days=2)),
            Trade(symbol="MSFT", entry_price=300.0, exit_price=310.0, quantity=5, trade_type="BUY", entry_time=datetime.now() - timedelta(days=1), exit_time=datetime.now()),
        ]
        for trade in trades:
            await self.tracker.add_trade(trade)

        # 3. データベースからトレードを読み込み、数が正しいことを確認
        all_trades = await self.tracker.get_trades()
        self.assertEqual(len(all_trades), 3)
        self.assertEqual(all_trades[0].symbol, "AAPL")

        # 4. パフォーマンスメトリクスを計算
        metrics = await self.tracker.calculate_performance_metrics(days_back=10)
        self.assertIsInstance(metrics, PerformanceMetrics)
        
        # 簡単なメトリクスの検証
        self.assertEqual(metrics.total_trades, 3)
        self.assertEqual(metrics.winning_trades, 2)
        self.assertEqual(metrics.losing_trades, 1)
        self.assertAlmostEqual(metrics.win_rate, 2/3 * 100)
        # PnLの計算: (155-150)*10 + (2800-2750)*2 + (310-300)*5 = 50 + 100 + 50 = 200
        # 注: SELLトレードのPnLは (entry - exit) * quantity
        self.assertAlmostEqual(metrics.total_pnl, 200.0)

        # 5. ポートフォリオを取得 (このテストでは明示的に設定していないので、空かデフォルト値のはず)
        portfolio = await self.tracker.get_portfolio()
        self.assertIsInstance(portfolio, Portfolio) # デフォルトの空ポートフォリオが返されるはず

        # 6. レポートを生成
        report = await self.tracker.generate_report(days_back=10)
        self.assertIn("Performance Report", report)
        self.assertIn("Total PnL: 200.00", report)
        self.assertIn("Win Rate: 66.67%", report)

        # 7. 最後のトレードを取得
        last_trade = await self.tracker.get_last_trade()
        self.assertIsNotNone(last_trade)
        self.assertEqual(last_trade.symbol, "MSFT")

if __name__ == '__main__':
    unittest.main()
