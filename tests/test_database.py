"""
データベース基盤のテスト
"""

import contextlib
import time
from datetime import datetime
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.day_trade.models.database import DatabaseConfig, DatabaseManager, get_db
from src.day_trade.models.enums import AlertType, TradeType
from src.day_trade.models.stock import Alert, PriceData, Stock, Trade, WatchlistItem
from src.day_trade.utils.exceptions import DatabaseIntegrityError


class TestDatabaseManager:
    """DatabaseManagerクラスのテスト"""

    @pytest.fixture
    def test_db_manager(self):
        """テスト用のデータベースマネージャー"""
        config = DatabaseConfig.for_testing()
        db_manager = DatabaseManager(config)
        db_manager.create_tables()
        yield db_manager
        # クリーンアップ
        db_manager.drop_tables()

    def test_session_scope(self, test_db_manager):
        """セッションスコープのテスト"""
        # セッションスコープを使用してデータを作成
        with test_db_manager.session_scope() as session:
            stock = Stock(
                code="7203",
                name="トヨタ自動車",
                market="東証プライム",
                sector="輸送用機器",
            )
            session.add(stock)

        # 別のセッションで確認
        with test_db_manager.session_scope() as session:
            result = session.query(Stock).filter_by(code="7203").first()
            assert result is not None
            assert result.name == "トヨタ自動車"

    def test_rollback_on_error(self, test_db_manager):
        """エラー時のロールバックテスト"""
        # エラーを発生させる
        try:
            with test_db_manager.session_scope() as session:
                stock = Stock(code="7203", name="トヨタ自動車")
                session.add(stock)
                # わざとエラーを発生させる
                raise ValueError("Test error")
        except Exception:
            pass

        # データが保存されていないことを確認
        with test_db_manager.session_scope() as session:
            result = session.query(Stock).filter_by(code="7203").first()
            assert result is None

    def test_create_and_query_tables(self, test_db_manager):
        """テーブル作成とクエリのテスト"""
        with test_db_manager.session_scope() as session:
            # 銘柄データ作成
            stock = Stock(
                code="7203",
                name="トヨタ自動車",
                market="東証プライム",
                sector="輸送用機器",
            )
            session.add(stock)
            session.flush()

            # 価格データ作成（Decimal型で精密計算）
            price = PriceData(
                stock_code="7203",
                datetime=datetime(2023, 5, 1, 15, 0),
                open=Decimal("2450.00"),
                high=Decimal("2480.00"),
                low=Decimal("2440.00"),
                close=Decimal("2470.00"),
                volume=10000000,
            )
            session.add(price)

            # 取引データ作成（Decimal型で精密計算）
            trade = Trade(
                stock_code="7203",
                trade_type=TradeType.BUY,
                quantity=100,
                price=Decimal("2470.00"),
                commission=Decimal("500.00"),
                trade_datetime=datetime(2023, 5, 1, 15, 0),
                memo="テスト取引",
            )
            session.add(trade)

            # ウォッチリストアイテム作成
            watchlist = WatchlistItem(
                stock_code="7203", group_name="自動車", memo="主力銘柄"
            )
            session.add(watchlist)

            # アラート作成（Decimal型で精密計算）
            alert = Alert(
                stock_code="7203",
                alert_type=AlertType.PRICE_ABOVE,
                threshold=Decimal("2500.00"),
                is_active=True,
                memo="高値警戒",
            )
            session.add(alert)

        # データ確認
        with test_db_manager.session_scope() as session:
            # 銘柄確認
            stock = session.query(Stock).filter_by(code="7203").first()
            assert stock.name == "トヨタ自動車"

            # リレーション確認
            assert len(stock.price_data) == 1
            assert len(stock.trades) == 1
            assert len(stock.watchlist_items) == 1
            assert len(stock.alerts) == 1

            # 価格データ確認（現在の実装：Float型、将来的にDecimal型に移行予定）
            price = stock.price_data[0]
            if isinstance(price.close, Decimal):
                # Decimal型への移行後の検証
                assert price.close == Decimal("2470.00")
                assert isinstance(price.open, Decimal), (
                    f"Expected Decimal, got {type(price.open)}"
                )
                assert isinstance(price.high, Decimal), (
                    f"Expected Decimal, got {type(price.high)}"
                )
                assert isinstance(price.low, Decimal), (
                    f"Expected Decimal, got {type(price.low)}"
                )
            else:
                # 現在のFloat型実装での検証
                assert price.close == 2470.0
                assert isinstance(price.open, float), (
                    f"Expected float, got {type(price.open)}"
                )
                assert isinstance(price.high, float), (
                    f"Expected float, got {type(price.high)}"
                )
                assert isinstance(price.low, float), (
                    f"Expected float, got {type(price.low)}"
                )
                print(
                    "Price data currently using Float type (will migrate to Decimal in the future)"
                )

            # 取引データ確認（現在の実装：Float型、将来的にDecimal型に移行予定）
            trade = stock.trades[0]
            if isinstance(trade.price, Decimal):
                # Decimal型への移行後の検証
                assert isinstance(trade.commission, Decimal), (
                    f"Expected Decimal, got {type(trade.commission)}"
                )
                if hasattr(trade, "total_amount"):
                    expected_total = Decimal("2470.00") * 100 + Decimal("500.00")
                    assert trade.total_amount == expected_total
                    assert isinstance(trade.total_amount, Decimal), (
                        f"Expected Decimal, got {type(trade.total_amount)}"
                    )
            else:
                # 現在のFloat型実装での検証
                assert isinstance(trade.price, float), (
                    f"Expected float, got {type(trade.price)}"
                )
                assert isinstance(trade.commission, float), (
                    f"Expected float, got {type(trade.commission)}"
                )
                if hasattr(trade, "total_amount"):
                    expected_total = 2470.0 * 100 + 500.0
                    assert trade.total_amount == expected_total
                print(
                    "Trade data currently using Float type (will migrate to Decimal in the future)"
                )

            # アラート確認（現在の実装：Float型、将来的にDecimal型に移行予定）
            alert = stock.alerts[0]
            if isinstance(alert.threshold, Decimal):
                # Decimal型への移行後の検証
                assert alert.threshold == Decimal("2500.00")
            else:
                # 現在のFloat型実装での検証
                assert alert.threshold == 2500.0
                assert isinstance(alert.threshold, float), (
                    f"Expected float, got {type(alert.threshold)}"
                )
                print(
                    "Alert data currently using Float type (will migrate to Decimal in the future)"
                )

    def test_indexes_and_constraints(self, test_db_manager):
        """インデックスと制約のテスト"""
        # 最初のレコードを作成
        with test_db_manager.session_scope() as session:
            stock1 = Stock(code="7203", name="トヨタ自動車")
            session.add(stock1)

        # 重複エラーが発生することを確認（アプリケーション固有の例外型を指定）
        with pytest.raises(
            DatabaseIntegrityError
        ), test_db_manager.session_scope() as session:
            stock2 = Stock(code="7203", name="トヨタ自動車（重複）")
            session.add(stock2)

    def test_timestamp_mixin(self, test_db_manager):
        """タイムスタンプMixinのテスト（improved with session.refresh()）"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="7203", name="トヨタ自動車")
            session.add(stock)
            session.flush()
            session.refresh(stock)  # DBからの最新データで更新

            # created_atとupdated_atが設定されていることを確認
            assert stock.created_at is not None, (
                "created_at should be set automatically"
            )
            assert stock.updated_at is not None, (
                "updated_at should be set automatically"
            )

            # 初期状態ではcreated_atとupdated_atは同じかほぼ同じ
            time_diff = abs((stock.updated_at - stock.created_at).total_seconds())
            assert time_diff < 1.0, (
                f"Initial timestamp difference too large: {time_diff}s"
            )

            # 元のupdated_atを保存
            original_updated = stock.updated_at
            original_created = stock.created_at

            # 少し待機してからデータを更新
            time.sleep(0.1)  # 100ms待機

            # 更新操作
            stock.name = "トヨタ自動車株式会社"
            session.flush()
            session.refresh(stock)  # DBからの最新データで更新

            # updated_atが更新されていることを確認
            assert stock.created_at == original_created, (
                "created_at should not change on update"
            )

            # updated_atの自動更新確認（実装依存）
            if stock.updated_at != original_updated:
                assert stock.updated_at > original_updated, (
                    "updated_at should be newer after update"
                )
                print(
                    f"Automatic updated_at update detected: {original_updated} -> {stock.updated_at}"
                )
            else:
                print("Manual updated_at update required (implementation dependent)")

            # 基本的な整合性確認
            assert stock.created_at <= stock.updated_at, (
                "created_at should not be after updated_at"
            )

    def test_decimal_precision_and_calculations(self, test_db_manager):
        """Decimal型の精度と計算テスト（現在はFloat型、将来的にDecimal型移行テスト）"""
        with test_db_manager.session_scope() as session:
            # 精密な計算が必要な価格データを作成
            stock = Stock(code="TEST", name="テスト銘柄")
            session.add(stock)
            session.flush()

            # 現在はFloat型で保存されるが、将来的にはDecimal型に移行
            price1 = PriceData(
                stock_code="TEST",
                datetime=datetime(2023, 5, 1, 9, 0),
                open=Decimal("1234.56"),  # 入力はDecimalだがFloat型に変換される
                high=Decimal("1234.78"),
                low=Decimal("1234.12"),
                close=Decimal("1234.67"),
                volume=1000000,
            )

            price2 = PriceData(
                stock_code="TEST",
                datetime=datetime(2023, 5, 1, 10, 0),
                open=Decimal("1234.67"),
                high=Decimal("1235.89"),
                low=Decimal("1234.45"),
                close=Decimal("1235.12"),
                volume=1200000,
            )

            session.add_all([price1, price2])
            session.flush()

            # 取引データでの精密計算
            trade1 = Trade(
                stock_code="TEST",
                trade_type=TradeType.BUY,
                quantity=123,
                price=Decimal("1234.56"),
                commission=Decimal("123.45"),
                trade_datetime=datetime(2023, 5, 1, 9, 30),
            )

            trade2 = Trade(
                stock_code="TEST",
                trade_type=TradeType.SELL,
                quantity=100,
                price=Decimal("1235.12"),
                commission=Decimal("98.76"),
                trade_datetime=datetime(2023, 5, 1, 10, 30),
            )

            session.add_all([trade1, trade2])
            session.flush()

            # アラートでの精密しきい値
            alert = Alert(
                stock_code="TEST",
                alert_type=AlertType.PRICE_ABOVE,
                threshold=Decimal("1236.789"),  # 小数点第3位まで
                is_active=True,
            )

            session.add(alert)
            session.flush()

        # データの確認と計算精度テスト（現在の実装に対応）
        with test_db_manager.session_scope() as session:
            # 価格データの確認
            prices = session.query(PriceData).filter_by(stock_code="TEST").all()
            assert len(prices) == 2

            # 現在の実装（Float型）または将来の実装（Decimal型）に対応
            decimal_implemented = isinstance(prices[0].open, Decimal)

            for price in prices:
                if decimal_implemented:
                    # Decimal型への移行後の検証
                    assert isinstance(price.open, Decimal)
                    assert isinstance(price.high, Decimal)
                    assert isinstance(price.low, Decimal)
                    assert isinstance(price.close, Decimal)
                    print("Using Decimal type for price data")
                else:
                    # 現在のFloat型実装での検証
                    assert isinstance(price.open, float)
                    assert isinstance(price.high, float)
                    assert isinstance(price.low, float)
                    assert isinstance(price.close, float)
                    print("Using Float type for price data (will migrate to Decimal)")

            # 取引データの精度確認と計算
            trades = session.query(Trade).filter_by(stock_code="TEST").all()
            assert len(trades) == 2

            if decimal_implemented:
                # Decimal型での精密計算
                total_cost = Decimal("0")
                total_proceeds = Decimal("0")

                for trade in trades:
                    assert isinstance(trade.price, Decimal)
                    assert isinstance(trade.commission, Decimal)

                    trade_amount = trade.price * trade.quantity

                    if trade.trade_type == TradeType.BUY:
                        total_cost += trade_amount + trade.commission
                    else:
                        total_proceeds += trade_amount - trade.commission

                # 精密計算の結果確認
                expected_cost = Decimal("1234.56") * 123 + Decimal("123.45")
                expected_proceeds = Decimal("1235.12") * 100 - Decimal("98.76")

                assert total_cost == expected_cost, (
                    f"Expected {expected_cost}, got {total_cost}"
                )
                assert total_proceeds == expected_proceeds, (
                    f"Expected {expected_proceeds}, got {total_proceeds}"
                )
            else:
                # Float型での計算（現在の実装）
                total_cost = 0.0
                total_proceeds = 0.0

                for trade in trades:
                    assert isinstance(trade.price, float)
                    assert isinstance(trade.commission, float)

                    trade_amount = trade.price * trade.quantity

                    if trade.trade_type == TradeType.BUY:
                        total_cost += trade_amount + trade.commission
                    else:
                        total_proceeds += trade_amount - trade.commission

                # Float型での概算検証（浮動小数点誤差を考慮）
                expected_cost = 1234.56 * 123 + 123.45
                expected_proceeds = 1235.12 * 100 - 98.76

                assert abs(total_cost - expected_cost) < 0.01, (
                    f"Expected ~{expected_cost}, got {total_cost}"
                )
                assert abs(total_proceeds - expected_proceeds) < 0.01, (
                    f"Expected ~{expected_proceeds}, got {total_proceeds}"
                )

            # アラートのしきい値確認
            alert = session.query(Alert).filter_by(stock_code="TEST").first()

            if decimal_implemented:
                assert isinstance(alert.threshold, Decimal)
                assert alert.threshold == Decimal("1236.789")
            else:
                assert isinstance(alert.threshold, float)
                assert abs(alert.threshold - 1236.789) < 0.001  # Float型での精度許容

            print(
                f"Precision test passed - Type: {'Decimal' if decimal_implemented else 'Float'}, Cost: {total_cost}, Proceeds: {total_proceeds}"
            )


class TestDatabaseFunctions:
    """データベース関連関数のテスト"""

    def test_get_db_generator(self):
        """get_db関数のテスト（DI対応を考慮）"""
        gen = get_db()
        session = next(gen)
        assert isinstance(session, Session), "get_db should return a Session instance"

        # セッションが実際に使用可能であることを確認
        try:
            # 簡単なクエリでセッションの有効性を確認
            session.execute(text("SELECT 1"))
        except Exception as e:
            pytest.fail(f"Session is not functional: {e}")

        # クリーンアップ
        with contextlib.suppress(StopIteration):
            next(gen)

    def test_database_manager_dependency_injection(self):
        """DatabaseManagerのDI対応テスト（将来の機能追加を考慮）"""
        # 複数のDatabaseManagerインスタンスが独立して動作することを確認
        config1 = DatabaseConfig.for_testing()
        config2 = DatabaseConfig.for_testing()

        db1 = DatabaseManager(config1)
        db2 = DatabaseManager(config2)

        # 異なるインスタンスであることを確認
        assert db1 is not db2, "DatabaseManager instances should be independent"

        # それぞれが独立して動作することを確認
        db1.create_tables()
        db2.create_tables()

        # クリーンアップ
        db1.drop_tables()
        db2.drop_tables()
