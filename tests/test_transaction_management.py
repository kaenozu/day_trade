"""
DB操作のトランザクション管理テスト
"""

from unittest.mock import Mock, patch

import pytest

from src.day_trade.core.trade_operations import TradeOperationError, TradeOperations
from src.day_trade.models.database import DatabaseConfig, DatabaseManager
from src.day_trade.models.stock import Stock, Trade


class TestTransactionManagement:
    """トランザクション管理のテスト"""

    @pytest.fixture
    def test_db_manager(self):
        """テスト用データベースマネージャー"""
        config = DatabaseConfig.for_testing()
        db_manager = DatabaseManager(config)
        db_manager.create_tables()
        return db_manager

    @pytest.fixture
    def trade_operations(self):
        """取引操作インスタンス"""
        with patch("src.day_trade.core.trade_operations.db_manager") as mock_db:
            mock_db.transaction_scope.return_value.__enter__ = Mock()
            mock_db.transaction_scope.return_value.__exit__ = Mock(return_value=None)
            return TradeOperations()

    def test_transaction_scope_basic(self, test_db_manager):
        """基本的なトランザクションスコープのテスト"""
        with test_db_manager.transaction_scope() as session:
            # テストデータ作成
            stock = Stock(code="TEST", name="Test Stock")
            session.add(stock)
            session.flush()

            # セッション内で確認
            found_stock = session.query(Stock).filter(Stock.code == "TEST").first()
            assert found_stock is not None
            assert found_stock.name == "Test Stock"

        # トランザクション完了後も存在することを確認
        with test_db_manager.session_scope() as session:
            found_stock = session.query(Stock).filter(Stock.code == "TEST").first()
            assert found_stock is not None

    def test_transaction_rollback_on_error(self, test_db_manager):
        """エラー時のロールバック機能テスト"""
        from src.day_trade.utils.exceptions import DatabaseError

        try:
            with test_db_manager.transaction_scope() as session:
                # テストデータ作成
                stock = Stock(code="ROLLBACK", name="Rollback Test")
                session.add(stock)
                session.flush()

                # 意図的にエラーを発生
                raise ValueError("Test error")
        except DatabaseError:
            # DatabaseErrorに変換されることを確認
            pass

        # ロールバックされて存在しないことを確認
        with test_db_manager.session_scope() as session:
            found_stock = session.query(Stock).filter(Stock.code == "ROLLBACK").first()
            assert found_stock is None

    def test_atomic_operation(self, test_db_manager):
        """アトミック操作のテスト"""

        def operation1(session):
            stock1 = Stock(code="ATOMIC1", name="Atomic Test 1")
            session.add(stock1)

        def operation2(session):
            stock2 = Stock(code="ATOMIC2", name="Atomic Test 2")
            session.add(stock2)

        # アトミック操作実行
        test_db_manager.atomic_operation([operation1, operation2])

        # 両方のレコードが存在することを確認
        with test_db_manager.session_scope() as session:
            stock1 = session.query(Stock).filter(Stock.code == "ATOMIC1").first()
            stock2 = session.query(Stock).filter(Stock.code == "ATOMIC2").first()
            assert stock1 is not None
            assert stock2 is not None

    def test_bulk_operations_with_transaction(self, test_db_manager):
        """一括操作でのトランザクション管理テスト"""
        # テスト用データ
        stock_data = [
            {"code": "BULK1", "name": "Bulk Test 1"},
            {"code": "BULK2", "name": "Bulk Test 2"},
            {"code": "BULK3", "name": "Bulk Test 3"},
        ]

        # 一括挿入
        test_db_manager.bulk_insert(Stock, stock_data)

        # データが正しく挿入されていることを確認
        with test_db_manager.session_scope() as session:
            count = session.query(Stock).filter(Stock.code.like("BULK%")).count()
            assert count == 3

    @patch("src.day_trade.core.trade_operations.StockFetcher")
    def test_buy_stock_transaction(self, mock_fetcher_class, test_db_manager):
        """買い注文のトランザクション管理テスト"""
        # モックの設定
        mock_fetcher = Mock()
        mock_fetcher.get_company_info.return_value = {
            "name": "Test Company",
            "sector": "Technology",
            "industry": "Software",
        }
        mock_fetcher.get_current_price.return_value = {"price": 1000.0}
        mock_fetcher_class.return_value = mock_fetcher

        # データベース設定をパッチ
        with patch("src.day_trade.core.trade_operations.db_manager", test_db_manager):
            trade_ops = TradeOperations(mock_fetcher)

            # 買い注文実行
            result = trade_ops.buy_stock("TEST001", 100, price=1000.0, commission=100.0)

            # 結果確認
            assert result["success"] is True
            assert result["stock_code"] == "TEST001"
            assert result["quantity"] == 100
            assert result["price"] == 1000.0
            assert result["total_amount"] == 100100.0  # price * quantity + commission

            # データベース内容確認
            with test_db_manager.session_scope() as session:
                stock = session.query(Stock).filter(Stock.code == "TEST001").first()
                assert stock is not None
                assert stock.name == "Test Company"

                trade = (
                    session.query(Trade).filter(Trade.stock_code == "TEST001").first()
                )
                assert trade is not None
                assert trade.trade_type == "buy"
                assert trade.quantity == 100

    @patch("src.day_trade.core.trade_operations.StockFetcher")
    def test_sell_stock_insufficient_holdings(
        self, mock_fetcher_class, test_db_manager
    ):
        """売り注文での保有不足エラーテスト"""
        # モックの設定
        mock_fetcher = Mock()
        mock_fetcher.get_company_info.return_value = {
            "name": "Test Company",
            "sector": "Technology",
        }
        mock_fetcher_class.return_value = mock_fetcher

        # テスト用銘柄を事前に追加
        with test_db_manager.session_scope() as session:
            stock = Stock(code="SELL001", name="Sell Test")
            session.add(stock)

        with patch("src.day_trade.core.trade_operations.db_manager", test_db_manager):
            trade_ops = TradeOperations(mock_fetcher)

            # 保有していない銘柄の売り注文
            with pytest.raises(TradeOperationError) as exc_info:
                trade_ops.sell_stock("SELL001", 100, price=1000.0)

            assert "売却数量が保有数量を上回ります" in str(exc_info.value)

    def test_retry_mechanism(self, test_db_manager):
        """再試行メカニズムのテスト - エラー検出ロジックのみテスト"""
        from sqlalchemy.exc import OperationalError

        # デッドロックエラーが再試行可能と判定されることを確認
        deadlock_error = OperationalError("deadlock detected", None, None)
        assert test_db_manager._is_retriable_error(deadlock_error) is True

        # ロックタイムアウトエラーが再試行可能と判定されることを確認
        lock_error = OperationalError("database is locked", None, None)
        assert test_db_manager._is_retriable_error(lock_error) is True

        # 通常のエラーは再試行不可と判定されることを確認
        normal_error = ValueError("normal error")
        assert test_db_manager._is_retriable_error(normal_error) is False

    def test_deadlock_detection(self, test_db_manager):
        """デッドロック検出のテスト"""
        from sqlalchemy.exc import OperationalError

        # デッドロックエラーをシミュレート
        error = OperationalError("deadlock detected", None, None)
        assert test_db_manager._is_retriable_error(error) is True

        # 非再試行エラーをテスト
        error = ValueError("invalid value")
        assert test_db_manager._is_retriable_error(error) is False


class TestTradeOperations:
    """取引操作クラスのテスト"""

    @pytest.fixture
    def mock_stock_fetcher(self):
        """モック株価取得インスタンス"""
        fetcher = Mock()
        fetcher.get_company_info.return_value = {
            "name": "Mock Company",
            "sector": "Technology",
            "industry": "Software",
        }
        fetcher.get_current_price.return_value = {"price": 1500.0}
        return fetcher

    def test_trade_operation_error_handling(self, mock_stock_fetcher):
        """取引操作のエラーハンドリングテスト"""
        with patch("src.day_trade.core.trade_operations.db_manager") as mock_db:
            # セッションでエラーが発生するようにモック
            mock_session = Mock()
            mock_session.query.side_effect = Exception("Database error")
            mock_db.transaction_scope.return_value.__enter__.return_value = mock_session

            trade_ops = TradeOperations(mock_stock_fetcher)

            with pytest.raises(TradeOperationError) as exc_info:
                trade_ops.buy_stock("ERROR", 100)

            assert "買い注文処理エラー" in str(exc_info.value)

    def test_batch_operations(self, mock_stock_fetcher):
        """バッチ操作のテスト"""
        with patch("src.day_trade.core.trade_operations.db_manager") as mock_db:
            mock_session = Mock()
            mock_db.transaction_scope.return_value.__enter__.return_value = mock_session

            trade_ops = TradeOperations(mock_stock_fetcher)

            operations = [
                {"action": "buy", "stock_code": "BATCH1", "quantity": 100},
                {"action": "buy", "stock_code": "BATCH2", "quantity": 200},
            ]

            # バッチ操作実行（実装が完了していないためモックで動作確認）
            result = trade_ops.batch_trade_operations(operations)

            # 基本的な構造の確認
            assert "success" in result
            assert "total_operations" in result
            assert result["total_operations"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
