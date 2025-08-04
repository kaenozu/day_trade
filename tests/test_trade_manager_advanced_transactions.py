"""
TradeManagerの高度なトランザクション機能のテスト
"""

from decimal import Decimal
from unittest.mock import patch

import pytest

from src.day_trade.core.trade_manager import TradeManager, TradeType
from src.day_trade.models.database import db_manager
from src.day_trade.utils.exceptions import DatabaseError


@pytest.fixture
def trade_manager():
    """テスト用TradeManagerインスタンス"""
    # データベースのテーブルを確実に作成
    db_manager.create_tables()
    return TradeManager(load_from_db=False)


@pytest.fixture
def sample_trades_data():
    """サンプル取引データ"""
    return [
        {
            "symbol": "7203",
            "trade_type": TradeType.BUY,
            "quantity": 100,
            "price": Decimal("2500"),
            "notes": "バッチ取引1",
        },
        {
            "symbol": "8306",
            "trade_type": TradeType.BUY,
            "quantity": 50,
            "price": Decimal("800"),
            "notes": "バッチ取引2",
        },
        {
            "symbol": "9984",
            "trade_type": TradeType.BUY,
            "quantity": 10,
            "price": Decimal("15000"),
            "notes": "バッチ取引3",
        },
    ]


class TestBatchTradeOperations:
    """一括取引操作のテスト"""

    def test_add_trades_batch_memory_only(self, trade_manager, sample_trades_data):
        """メモリ内のみの一括取引追加テスト"""
        # 実行
        trade_ids = trade_manager.add_trades_batch(
            sample_trades_data, persist_to_db=False
        )

        # 検証
        assert len(trade_ids) == 3
        assert len(trade_manager.trades) == 3
        assert len(trade_manager.positions) == 3

        # 各ポジションの確認
        assert "7203" in trade_manager.positions
        assert "8306" in trade_manager.positions
        assert "9984" in trade_manager.positions

        # ポジション詳細確認
        position_7203 = trade_manager.get_position("7203")
        assert position_7203.quantity == 100
        assert position_7203.average_price > Decimal("2500")  # 手数料込み

    def test_add_trades_batch_with_db_persistence(
        self, trade_manager, sample_trades_data
    ):
        """データベース永続化を含む一括取引追加テスト"""
        # 実行
        trade_ids = trade_manager.add_trades_batch(
            sample_trades_data, persist_to_db=True
        )

        # 検証
        assert len(trade_ids) == 3
        assert len(trade_manager.trades) == 3

        # データベースに保存されていることを確認
        with db_manager.session_scope() as session:
            from src.day_trade.models.stock import Trade as DBTrade

            db_trades = session.query(DBTrade).all()
            assert len(db_trades) >= 3  # 他のテストからの残りがある可能性

    def test_add_trades_batch_empty_data(self, trade_manager):
        """空データでの一括取引追加テスト"""
        # 実行
        trade_ids = trade_manager.add_trades_batch([], persist_to_db=False)

        # 検証
        assert trade_ids == []
        assert len(trade_manager.trades) == 0

    def test_add_trades_batch_rollback_on_error(self, trade_manager):
        """エラー時のロールバックテスト"""
        # 正常な取引データと無効なデータを混在（より確実にエラーを発生させる）
        invalid_trades_data = [
            {
                "symbol": "7203",
                "trade_type": TradeType.BUY,
                "quantity": 100,
                "price": Decimal("2500"),
            },
            {
                "symbol": "INVALID",
                "trade_type": TradeType.BUY,
                "quantity": 50,
                "price": "INVALID_PRICE",  # 無効な価格（文字列）
            },
        ]

        # 実行前の状態確認
        initial_trades_count = len(trade_manager.trades)
        initial_positions_count = len(trade_manager.positions)

        # エラーが発生することを確認
        with pytest.raises((ValueError, TypeError, AttributeError)):
            trade_manager.add_trades_batch(invalid_trades_data, persist_to_db=False)

        # ロールバック確認
        assert len(trade_manager.trades) == initial_trades_count
        assert len(trade_manager.positions) == initial_positions_count

    def test_add_trades_batch_partial_processing(
        self, trade_manager, sample_trades_data
    ):
        """部分的な処理でのトランザクション保護テスト"""
        # DB永続化でエラーを発生させるためのモック
        with patch("src.day_trade.models.stock.Trade.create_buy_trade") as mock_create:
            # 2回目の呼び出しでエラーを発生
            mock_create.side_effect = [None, DatabaseError("DB Error"), None]

            # エラーが発生することを確認
            with pytest.raises(DatabaseError):
                trade_manager.add_trades_batch(sample_trades_data, persist_to_db=True)

            # すべての処理がロールバックされていることを確認
            assert len(trade_manager.trades) == 0
            assert len(trade_manager.positions) == 0


class TestDataClearOperations:
    """データクリア操作のテスト"""

    def test_clear_all_data_memory_only(self, trade_manager, sample_trades_data):
        """メモリ内のみの全データクリアテスト"""
        # 事前にデータを追加
        trade_manager.add_trades_batch(sample_trades_data, persist_to_db=False)
        assert len(trade_manager.trades) > 0
        assert len(trade_manager.positions) > 0

        # 実行
        trade_manager.clear_all_data(persist_to_db=False)

        # 検証
        assert len(trade_manager.trades) == 0
        assert len(trade_manager.positions) == 0
        assert len(trade_manager.realized_pnl) == 0
        assert trade_manager._trade_counter == 0

    def test_clear_all_data_with_db(self, trade_manager, sample_trades_data):
        """データベースを含む全データクリアテスト"""
        # 事前にデータを追加
        trade_manager.add_trades_batch(sample_trades_data, persist_to_db=True)

        # データベースにデータが存在することを確認
        with db_manager.session_scope() as session:
            from src.day_trade.models.stock import Trade as DBTrade

            initial_db_count = session.query(DBTrade).count()
            assert initial_db_count > 0

        # 実行
        trade_manager.clear_all_data(persist_to_db=True)

        # メモリ内データのクリア確認
        assert len(trade_manager.trades) == 0
        assert len(trade_manager.positions) == 0
        assert len(trade_manager.realized_pnl) == 0

        # データベースのクリア確認
        with db_manager.session_scope() as session:
            from src.day_trade.models.stock import Trade as DBTrade

            final_db_count = session.query(DBTrade).count()
            assert final_db_count == 0

    def test_clear_all_data_rollback_on_error(self, trade_manager, sample_trades_data):
        """データクリア時のエラーロールバックテスト"""
        # 事前にデータを追加
        trade_manager.add_trades_batch(sample_trades_data, persist_to_db=False)
        initial_trades = trade_manager.trades.copy()
        initial_positions = trade_manager.positions.copy()

        # DBエラーをシミュレート
        with patch.object(db_manager, "transaction_scope") as mock_transaction:
            mock_transaction.side_effect = DatabaseError("DB Connection Error")

            # エラーが発生することを確認
            with pytest.raises(DatabaseError):
                trade_manager.clear_all_data(persist_to_db=True)

            # データが復元されていることを確認
            assert len(trade_manager.trades) == len(initial_trades)
            assert len(trade_manager.positions) == len(initial_positions)


class TestImprovedLoadOperations:
    """改良された読み込み操作のテスト"""

    def test_load_trades_from_db_atomic_operation(self, trade_manager):
        """データベース読み込みの原子性テスト"""
        # 事前にメモリ内データを作成
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )
        trade_manager.trades.copy()

        # データベースに別のデータを追加
        with db_manager.transaction_scope() as session:
            from src.day_trade.models.stock import Stock
            from src.day_trade.models.stock import Trade as DBTrade

            # 銘柄マスタ確認・作成
            stock = session.query(Stock).filter(Stock.code == "8306").first()
            if not stock:
                stock = Stock(
                    code="8306",
                    name="三菱UFJ",
                    market="東証プライム",
                    sector="金融",
                    industry="銀行",
                )
                session.add(stock)
                session.flush()

            # DB取引追加
            DBTrade.create_buy_trade(
                session=session,
                stock_code="8306",
                quantity=50,
                price=800.0,
                commission=100.0,
                memo="DB読み込みテスト",
            )

        # データベースから再読み込み
        trade_manager._load_trades_from_db()

        # メモリ内データがDBデータに置き換わっていることを確認
        assert len(trade_manager.trades) >= 1  # DBから読み込んだデータ

        # 特定の取引が存在することを確認
        db_trades = [t for t in trade_manager.trades if t.symbol == "8306"]
        assert len(db_trades) >= 1

    def test_sync_with_db_rollback_on_error(self, trade_manager, sample_trades_data):
        """sync_with_dbのエラー時ロールバックテスト"""
        # 事前にメモリ内データを作成
        trade_manager.add_trades_batch(sample_trades_data, persist_to_db=False)
        initial_trades = trade_manager.trades.copy()
        initial_positions = trade_manager.positions.copy()

        # _load_trades_from_dbでエラーをシミュレート
        with patch.object(trade_manager, "_load_trades_from_db") as mock_load:
            mock_load.side_effect = DatabaseError("DB Load Error")

            # エラーが発生することを確認
            with pytest.raises(DatabaseError):
                trade_manager.sync_with_db()

            # メモリ内データが復元されていることを確認
            assert len(trade_manager.trades) == len(initial_trades)
            assert len(trade_manager.positions) == len(initial_positions)


class TestTransactionIntegrity:
    """トランザクション整合性のテスト"""

    def test_complex_transaction_scenario(self, trade_manager):
        """複雑なトランザクションシナリオのテスト"""
        # 複数銘柄での複雑な取引シナリオ
        complex_trades = [
            {
                "symbol": "7203",
                "trade_type": TradeType.BUY,
                "quantity": 100,
                "price": Decimal("2500"),
            },
            {
                "symbol": "7203",
                "trade_type": TradeType.BUY,
                "quantity": 200,
                "price": Decimal("2450"),
            },
            {
                "symbol": "8306",
                "trade_type": TradeType.BUY,
                "quantity": 1000,
                "price": Decimal("800"),
            },
            {
                "symbol": "7203",
                "trade_type": TradeType.SELL,
                "quantity": 150,
                "price": Decimal("2600"),
            },
        ]

        # 一括処理で実行
        trade_ids = trade_manager.add_trades_batch(complex_trades, persist_to_db=True)

        # 結果検証
        assert len(trade_ids) == 4
        assert len(trade_manager.trades) == 4

        # ポジション確認（7203は一部売却済み、8306は保有継続）
        assert "7203" in trade_manager.positions
        assert "8306" in trade_manager.positions

        position_7203 = trade_manager.get_position("7203")
        assert position_7203.quantity == 150  # 300 - 150

        position_8306 = trade_manager.get_position("8306")
        assert position_8306.quantity == 1000

        # 実現損益確認
        assert len(trade_manager.realized_pnl) == 1  # 7203の売却分

    @pytest.mark.skip(reason="Concurrent access test issues - temporary skip for CI")
    def test_concurrent_access_simulation(self, trade_manager):
        """同時アクセスシミュレーションテスト"""
        # 複数の操作を順次実行してデータの整合性を確認
        trades_batch_1 = [
            {
                "symbol": "7203",
                "trade_type": TradeType.BUY,
                "quantity": 100,
                "price": Decimal("2500"),
            }
        ]
        trades_batch_2 = [
            {
                "symbol": "8306",
                "trade_type": TradeType.BUY,
                "quantity": 50,
                "price": Decimal("800"),
            }
        ]

        # バッチ1実行
        trade_ids_1 = trade_manager.add_trades_batch(trades_batch_1, persist_to_db=True)

        # バッチ2実行
        trade_ids_2 = trade_manager.add_trades_batch(trades_batch_2, persist_to_db=True)

        # データベース同期
        trade_manager.sync_with_db()

        # 整合性確認
        assert len(trade_manager.trades) >= 2
        assert len(trade_manager.positions) >= 2

        # すべての取引IDが異なることを確認
        all_trade_ids = trade_ids_1 + trade_ids_2
        assert len(all_trade_ids) == len(set(all_trade_ids))
