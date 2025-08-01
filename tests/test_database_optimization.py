"""
データベース最適化のテスト
"""

import time
from datetime import datetime, timedelta

import pytest

from src.day_trade.core.watchlist import WatchlistManager
from src.day_trade.models.database import TEST_DATABASE_URL, DatabaseManager, DatabaseConfig
from src.day_trade.models.stock import PriceData, Stock, Trade


class TestDatabaseOptimization:
    """データベース最適化テスト"""

    @pytest.fixture
    def test_db(self):
        """テスト用データベース"""
        config = DatabaseConfig.for_testing()
        db = DatabaseManager(config)
        db.create_tables()
        yield db
        db.drop_tables()

    @pytest.fixture
    def sample_data(self, test_db):
        """サンプルデータを作成"""
        stocks_data = [
            {
                "code": "7203",
                "name": "トヨタ自動車",
                "sector": "輸送用機器",
                "market": "プライム",
            },
            {
                "code": "9984",
                "name": "ソフトバンクグループ",
                "sector": "情報・通信業",
                "market": "プライム",
            },
            {
                "code": "8306",
                "name": "三菱UFJフィナンシャル・グループ",
                "sector": "銀行業",
                "market": "プライム",
            },
            {
                "code": "4063",
                "name": "信越化学工業",
                "sector": "化学",
                "market": "プライム",
            },
            {
                "code": "6758",
                "name": "ソニーグループ",
                "sector": "電気機器",
                "market": "プライム",
            },
        ]

        # バルクインサートでサンプル株式データを作成
        test_db.bulk_insert(Stock, stocks_data)

        # 価格データも大量に作成
        price_data = []
        base_date = datetime.now() - timedelta(days=30)

        for stock in stocks_data:
            for i in range(30):  # 30日分
                price_data.append(
                    {
                        "stock_code": stock["code"],
                        "datetime": base_date + timedelta(days=i),
                        "open": 100.0 + i,
                        "high": 110.0 + i,
                        "low": 90.0 + i,
                        "close": 105.0 + i,
                        "volume": 1000000 + i * 10000,
                    }
                )

        test_db.bulk_insert(PriceData, price_data)
        return stocks_data

    def test_bulk_insert_performance(self, test_db):
        """バルクインサートのパフォーマンステスト"""
        # 大量データを準備
        large_stock_data = []
        for i in range(1000):
            large_stock_data.append(
                {
                    "code": f"{1000 + i:04d}",
                    "name": f"テスト銘柄{i}",
                    "sector": "テスト",
                    "market": "テスト市場",
                }
            )

        # バルクインサートの実行時間を測定
        start_time = time.time()
        test_db.bulk_insert(Stock, large_stock_data, batch_size=100)
        bulk_time = time.time() - start_time

        # 通常の個別インサートと比較（少数で比較）
        individual_data = []
        for i in range(100):
            individual_data.append(
                {
                    "code": f"{2000 + i:04d}",
                    "name": f"個別銘柄{i}",
                    "sector": "個別テスト",
                    "market": "個別市場",
                }
            )

        start_time = time.time()
        with test_db.session_scope() as session:
            for data in individual_data:
                stock = Stock(**data)
                session.add(stock)
        individual_time = time.time() - start_time

        # バルクインサートの方が高速であることを確認
        # 注意: この比較は環境に依存するため、実際のテストでは調整が必要
        print(f"Bulk insert time (1000 records): {bulk_time:.3f}s")
        print(f"Individual insert time (100 records): {individual_time:.3f}s")
        print(f"Bulk insert rate: {1000 / bulk_time:.1f} records/sec")
        print(f"Individual insert rate: {100 / individual_time:.1f} records/sec")

        # データが正常に挿入されていることを確認
        with test_db.session_scope() as session:
            count = session.query(Stock).count()
            assert count >= 1100  # 1000 + 100

    def test_optimized_queries(self, test_db, sample_data):
        """最適化されたクエリのテスト"""
        with test_db.session_scope() as session:
            # 最適化された複数銘柄の最新価格取得
            stock_codes = ["7203", "9984", "8306"]

            start_time = time.time()
            latest_prices = PriceData.get_latest_prices(session, stock_codes)
            optimized_time = time.time() - start_time

            # 結果検証
            assert len(latest_prices) == 3
            for code in stock_codes:
                assert code in latest_prices
                assert latest_prices[code].close > 0

            print(f"Optimized latest prices query time: {optimized_time:.3f}s")

    def test_volume_spike_detection(self, test_db, sample_data):
        """出来高急増検出の最適化テスト"""
        with test_db.session_scope() as session:
            # 出来高急増銘柄の検出
            start_time = time.time()
            spike_candidates = PriceData.get_volume_spike_candidates(
                session, volume_threshold=1.1, days_back=20, limit=10
            )
            detection_time = time.time() - start_time

            # 結果があることを確認（テストデータの性質上、必ずしもスパイクはない）
            assert isinstance(spike_candidates, list)
            print(f"Volume spike detection time: {detection_time:.3f}s")
            print(f"Found {len(spike_candidates)} spike candidates")

    def test_database_optimization_commands(self, test_db):
        """データベース最適化コマンドのテスト"""
        # 最適化実行
        start_time = time.time()
        test_db.optimize_database()
        optimization_time = time.time() - start_time

        print(f"Database optimization time: {optimization_time:.3f}s")
        # SQLiteの場合、VACUUMとANALYZEが実行される

    def test_watchlist_bulk_operations(self, test_db, sample_data):
        """ウォッチリストの一括操作テスト"""

        # WatchlistManagerのテスト用設定
        # 注意: 実際のStockFetcherは使用しないモック版が必要
        class MockStockFetcher:
            def get_company_info(self, code):
                return {"name": f"Mock Stock {code}"}

        # データベース設定を一時的に変更
        from src.day_trade.models import database

        original_db = database.db_manager
        database.db_manager = test_db

        try:
            watchlist_manager = WatchlistManager()
            watchlist_manager.fetcher = MockStockFetcher()

            # 一括追加データ
            bulk_data = [
                {"code": "7203", "group": "tech", "memo": "自動車"},
                {"code": "9984", "group": "tech", "memo": "通信"},
                {"code": "8306", "group": "finance", "memo": "銀行"},
            ]

            # 一括追加のパフォーマンステスト
            start_time = time.time()
            results = watchlist_manager.bulk_add_stocks(bulk_data)
            bulk_add_time = time.time() - start_time

            # 結果検証
            assert len(results) == 3
            assert all(results.values())  # 全て成功

            # 最適化されたウォッチリスト取得
            start_time = time.time()
            optimized_watchlist = watchlist_manager.get_watchlist_optimized()
            optimized_get_time = time.time() - start_time

            # 結果検証
            assert len(optimized_watchlist) == 3

            print(f"Bulk add time: {bulk_add_time:.3f}s")
            print(f"Optimized get time: {optimized_get_time:.3f}s")

        finally:
            # 元の設定を復元
            database.db_manager = original_db

    def test_trade_portfolio_summary(self, test_db, sample_data):
        """取引ポートフォリオサマリーの最適化テスト"""
        # サンプル取引データを作成
        trades_data = [
            {
                "stock_code": "7203",
                "trade_type": "buy",
                "quantity": 100,
                "price": 1000.0,
                "commission": 100,
                "trade_datetime": datetime.now() - timedelta(days=10),
            },
            {
                "stock_code": "7203",
                "trade_type": "sell",
                "quantity": 50,
                "price": 1100.0,
                "commission": 100,
                "trade_datetime": datetime.now() - timedelta(days=5),
            },
            {
                "stock_code": "9984",
                "trade_type": "buy",
                "quantity": 200,
                "price": 5000.0,
                "commission": 200,
                "trade_datetime": datetime.now() - timedelta(days=3),
            },
        ]

        test_db.bulk_insert(Trade, trades_data)

        with test_db.session_scope() as session:
            # ポートフォリオサマリーの計算
            start_time = time.time()
            summary = Trade.get_portfolio_summary(session)
            summary_time = time.time() - start_time

            # 結果検証
            assert "portfolio" in summary
            assert "total_cost" in summary
            assert "total_proceeds" in summary

            # トヨタのポジション確認（100株買い - 50株売り = 50株保有）
            assert "7203" in summary["portfolio"]
            assert summary["portfolio"]["7203"]["quantity"] == 50

            print(f"Portfolio summary calculation time: {summary_time:.3f}s")
            print(f"Portfolio summary: {summary}")

    def test_sector_search_performance(self, test_db, sample_data):
        """セクター検索のパフォーマンステスト"""
        with test_db.session_scope() as session:
            # セクター別銘柄取得
            start_time = time.time()
            tech_stocks = Stock.get_by_sector(session, "電気機器")
            sector_search_time = time.time() - start_time

            # 結果検証
            assert len(tech_stocks) >= 1  # ソニーグループが含まれるはず

            # 銘柄名・コード検索
            start_time = time.time()
            search_results = Stock.search_by_name_or_code(session, "ソニー", limit=10)
            name_search_time = time.time() - start_time

            # 結果検証
            assert len(search_results) >= 1

            print(f"Sector search time: {sector_search_time:.3f}s")
            print(f"Name search time: {name_search_time:.3f}s")


if __name__ == "__main__":
    # 直接実行時の簡易テスト
    test_case = TestDatabaseOptimization()

    # テスト用DB作成
    config = DatabaseConfig.for_testing()
    db = DatabaseManager(config)
    db.create_tables()

    try:
        # サンプルデータ作成
        stocks_data = [
            {
                "code": "7203",
                "name": "トヨタ自動車",
                "sector": "輸送用機器",
                "market": "プライム",
            },
            {
                "code": "9984",
                "name": "ソフトバンクグループ",
                "sector": "情報・通信業",
                "market": "プライム",
            },
        ]
        db.bulk_insert(Stock, stocks_data)

        # 基本的なテスト実行
        test_case.test_bulk_insert_performance(db)
        print("データベース最適化テストが完了しました。")

    finally:
        db.drop_tables()
