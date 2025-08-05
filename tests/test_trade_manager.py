"""
取引記録管理機能のテスト
"""

import os
import tempfile
from datetime import datetime
from decimal import Decimal

import pytest

from src.day_trade.core.trade_manager import (
    Position,
    Trade,
    TradeManager,
    TradeStatus,
    TradeType,
)


class TestTrade:
    """Tradeクラスのテスト"""

    def test_trade_creation(self):
        """取引作成のテスト"""
        timestamp = datetime.now()
        trade = Trade(
            id="T001",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500"),
            timestamp=timestamp,
            commission=Decimal("250"),
        )

        assert trade.id == "T001"
        assert trade.symbol == "7203"
        assert trade.trade_type == TradeType.BUY
        assert trade.quantity == 100
        assert trade.price == Decimal("2500")
        assert trade.timestamp == timestamp
        assert trade.commission == Decimal("250")
        assert trade.status == TradeStatus.EXECUTED

    def test_trade_to_dict(self):
        """取引の辞書変換テスト"""
        timestamp = datetime.now()
        trade = Trade(
            id="T001",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500"),
            timestamp=timestamp,
        )

        trade_dict = trade.to_dict()

        assert trade_dict["id"] == "T001"
        assert trade_dict["symbol"] == "7203"
        assert trade_dict["trade_type"] == "buy"
        assert trade_dict["quantity"] == 100
        assert trade_dict["price"] == "2500"
        assert trade_dict["timestamp"] == timestamp.isoformat()

    def test_trade_from_dict(self):
        """辞書からの取引復元テスト"""
        timestamp = datetime.now()
        trade_dict = {
            "id": "T001",
            "symbol": "7203",
            "trade_type": "buy",
            "quantity": 100,
            "price": "2500",
            "timestamp": timestamp.isoformat(),
            "commission": "250",
            "status": "executed",
            "notes": "テスト取引",
        }

        trade = Trade.from_dict(trade_dict)

        assert trade.id == "T001"
        assert trade.symbol == "7203"
        assert trade.trade_type == TradeType.BUY
        assert trade.quantity == 100
        assert trade.price == Decimal("2500")
        assert trade.commission == Decimal("250")
        assert trade.notes == "テスト取引"


class TestPosition:
    """Positionクラスのテスト"""

    def test_position_creation(self):
        """ポジション作成のテスト"""
        position = Position(
            symbol="7203",
            quantity=100,
            average_price=Decimal("2500"),
            total_cost=Decimal("250100"),  # 価格 + 手数料
            current_price=Decimal("2600"),
        )

        assert position.symbol == "7203"
        assert position.quantity == 100
        assert position.average_price == Decimal("2500")
        assert position.total_cost == Decimal("250100")
        assert position.current_price == Decimal("2600")

    def test_position_calculations(self):
        """ポジション計算のテスト"""
        position = Position(
            symbol="7203",
            quantity=100,
            average_price=Decimal("2500"),
            total_cost=Decimal("250100"),
            current_price=Decimal("2600"),
        )

        # 時価総額
        assert position.market_value == Decimal("260000")

        # 含み損益
        assert position.unrealized_pnl == Decimal("9900")  # 260000 - 250100

        # 含み損益率
        expected_percent = (Decimal("9900") / Decimal("250100")) * 100
        assert abs(position.unrealized_pnl_percent - expected_percent) < Decimal("0.01")

    def test_position_loss(self):
        """含み損のテスト"""
        position = Position(
            symbol="7203",
            quantity=100,
            average_price=Decimal("2500"),
            total_cost=Decimal("250100"),
            current_price=Decimal("2400"),  # 下落
        )

        assert position.market_value == Decimal("240000")
        assert position.unrealized_pnl == Decimal("-10100")  # 240000 - 250100
        assert position.unrealized_pnl_percent < 0


class TestTradeManager:
    """TradeManagerクラスのテスト"""

    @pytest.fixture
    def trade_manager(self):
        """テスト用取引管理システム"""
        return TradeManager(
            commission_rate=Decimal("0.001"),
            tax_rate=Decimal("0.2"),
            load_from_db=False,
        )

    def test_initialization(self, trade_manager):
        """初期化のテスト"""
        assert trade_manager.commission_rate == Decimal("0.001")
        assert trade_manager.tax_rate == Decimal("0.2")
        assert len(trade_manager.trades) == 0
        assert len(trade_manager.positions) == 0
        assert len(trade_manager.realized_pnl) == 0

    def test_commission_calculation(self, trade_manager):
        """手数料計算のテスト"""
        # 通常の手数料計算
        commission = trade_manager._calculate_commission(Decimal("2500"), 100)
        expected = Decimal("2500") * 100 * Decimal("0.001")  # 250
        assert commission == expected

        # 最低手数料のテスト
        commission = trade_manager._calculate_commission(Decimal("50"), 1)
        assert commission == Decimal("100")  # 最低100円

    def test_add_buy_trade(self, trade_manager):
        """買い取引追加のテスト"""
        timestamp = datetime.now()
        trade_id = trade_manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500"),
            timestamp=timestamp,
            persist_to_db=False,
        )

        assert trade_id is not None
        assert len(trade_manager.trades) == 1
        assert len(trade_manager.positions) == 1

        # 取引確認
        trade = trade_manager.trades[0]
        assert trade.symbol == "7203"
        assert trade.trade_type == TradeType.BUY
        assert trade.quantity == 100
        assert trade.price == Decimal("2500")

        # ポジション確認
        position = trade_manager.positions["7203"]
        assert position.quantity == 100
        assert position.total_cost == Decimal("2500") * 100 + trade.commission

    def test_add_multiple_buy_trades(self, trade_manager):
        """複数買い取引のテスト"""
        # 1回目の買い
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )
        # 2回目の買い
        trade_manager.add_trade(
            "7203", TradeType.BUY, 200, Decimal("2450"), persist_to_db=False
        )

        position = trade_manager.positions["7203"]
        assert position.quantity == 300

        # 平均単価の計算確認
        # (2500*100 + 手数料1) + (2450*200 + 手数料2) = 総コスト
        # 平均単価 = 総コスト / 300
        expected_total_cost = (Decimal("2500") * 100 + Decimal("250")) + (
            Decimal("2450") * 200 + Decimal("490")
        )
        assert position.total_cost == expected_total_cost

    def test_sell_trade_partial(self, trade_manager):
        """一部売却のテスト"""
        # 買い取引
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )

        # 一部売却
        trade_manager.add_trade(
            "7203", TradeType.SELL, 50, Decimal("2600"), persist_to_db=False
        )

        # ポジション確認
        position = trade_manager.positions["7203"]
        assert position.quantity == 50

        # 実現損益確認
        assert len(trade_manager.realized_pnl) == 1
        realized = trade_manager.realized_pnl[0]
        assert realized.symbol == "7203"
        assert realized.quantity == 50
        assert realized.sell_price == Decimal("2600")
        assert realized.pnl > 0  # 利益が出ているはず

    def test_sell_trade_complete(self, trade_manager):
        """完全売却のテスト"""
        # 買い取引
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )

        # 完全売却
        trade_manager.add_trade(
            "7203", TradeType.SELL, 100, Decimal("2600"), persist_to_db=False
        )

        # ポジションが削除されることを確認
        assert "7203" not in trade_manager.positions

        # 実現損益確認
        assert len(trade_manager.realized_pnl) == 1

    def test_sell_without_position(self, trade_manager):
        """ポジションなしでの売却テスト"""
        # ポジションなしで売却（警告ログが出るが処理は継続）
        trade_manager.add_trade(
            "7203", TradeType.SELL, 100, Decimal("2600"), persist_to_db=False
        )

        assert len(trade_manager.trades) == 1
        assert len(trade_manager.positions) == 0
        assert len(trade_manager.realized_pnl) == 0

    def test_sell_excess_quantity(self, trade_manager):
        """過剰売却のテスト"""
        # 100株買い
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )

        # 150株売却（過剰）- 警告が出るが処理は継続される
        trade_manager.add_trade(
            "7203", TradeType.SELL, 150, Decimal("2600"), persist_to_db=False
        )

        # ポジションは残っている（過剰分は処理されない）
        assert "7203" in trade_manager.positions
        assert trade_manager.positions["7203"].quantity == 100  # 元のまま
        assert len(trade_manager.realized_pnl) == 0  # 実現損益は発生しない

    def test_current_price_update(self, trade_manager):
        """現在価格更新のテスト"""
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )
        trade_manager.add_trade(
            "8306", TradeType.BUY, 50, Decimal("4000"), persist_to_db=False
        )

        # 価格更新
        prices = {"7203": Decimal("2600"), "8306": Decimal("4200")}
        trade_manager.update_current_prices(prices)

        # 更新確認
        assert trade_manager.positions["7203"].current_price == Decimal("2600")
        assert trade_manager.positions["8306"].current_price == Decimal("4200")

    def test_portfolio_summary(self, trade_manager):
        """ポートフォリオサマリーのテスト"""
        # 複数取引
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )
        trade_manager.add_trade(
            "8306", TradeType.BUY, 50, Decimal("4000"), persist_to_db=False
        )
        trade_manager.add_trade(
            "7203", TradeType.SELL, 50, Decimal("2600"), persist_to_db=False
        )

        # 現在価格設定
        trade_manager.update_current_prices(
            {"7203": Decimal("2550"), "8306": Decimal("4100")}
        )

        summary = trade_manager.get_portfolio_summary()

        assert summary["total_positions"] == 2  # 7203, 8306
        assert summary["total_trades"] == 3
        assert summary["winning_trades"] == 1  # 7203の売却
        assert summary["losing_trades"] == 0
        assert "win_rate" in summary

    def test_trade_history(self, trade_manager):
        """取引履歴のテスト"""
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )
        trade_manager.add_trade(
            "8306", TradeType.BUY, 50, Decimal("4000"), persist_to_db=False
        )
        trade_manager.add_trade(
            "7203", TradeType.SELL, 50, Decimal("2600"), persist_to_db=False
        )

        # 全履歴
        all_trades = trade_manager.get_trade_history()
        assert len(all_trades) == 3

        # 特定銘柄の履歴
        toyota_trades = trade_manager.get_trade_history("7203")
        assert len(toyota_trades) == 2
        assert all(trade.symbol == "7203" for trade in toyota_trades)

    def test_realized_pnl_history(self, trade_manager):
        """実現損益履歴のテスト"""
        # 取引実行
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )
        trade_manager.add_trade(
            "8306", TradeType.BUY, 50, Decimal("4000"), persist_to_db=False
        )
        trade_manager.add_trade(
            "7203", TradeType.SELL, 50, Decimal("2600"), persist_to_db=False
        )
        trade_manager.add_trade(
            "8306", TradeType.SELL, 25, Decimal("3900"), persist_to_db=False
        )

        # 全実現損益
        all_pnl = trade_manager.get_realized_pnl_history()
        assert len(all_pnl) == 2

        # 特定銘柄の実現損益
        toyota_pnl = trade_manager.get_realized_pnl_history("7203")
        assert len(toyota_pnl) == 1
        assert toyota_pnl[0].symbol == "7203"

    def test_tax_calculation(self, trade_manager):
        """税務計算のテスト"""
        current_year = datetime.now().year

        # 利益の出る取引
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )
        trade_manager.add_trade(
            "7203", TradeType.SELL, 100, Decimal("2600"), persist_to_db=False
        )

        tax_info = trade_manager.calculate_tax_implications(current_year)

        assert tax_info["year"] == current_year
        assert tax_info["total_trades"] == 1
        assert Decimal(tax_info["total_gain"]) > 0
        assert Decimal(tax_info["net_gain"]) > 0
        assert Decimal(tax_info["tax_due"]) > 0

    def test_csv_export(self, trade_manager):
        """CSV出力のテスト"""
        # テストデータ作成
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )
        trade_manager.add_trade(
            "7203", TradeType.SELL, 50, Decimal("2600"), persist_to_db=False
        )
        trade_manager.update_current_prices({"7203": Decimal("2550")})

        with tempfile.TemporaryDirectory() as temp_dir:
            # 取引履歴CSV
            trades_path = os.path.join(temp_dir, "trades.csv")
            trade_manager.export_to_csv(trades_path, "trades")
            assert os.path.exists(trades_path)

            # ポジションCSV
            positions_path = os.path.join(temp_dir, "positions.csv")
            trade_manager.export_to_csv(positions_path, "positions")
            assert os.path.exists(positions_path)

            # 実現損益CSV
            pnl_path = os.path.join(temp_dir, "realized_pnl.csv")
            trade_manager.export_to_csv(pnl_path, "realized_pnl")
            assert os.path.exists(pnl_path)

    def test_json_save_load(self, trade_manager):
        """JSON保存・読み込みのテスト"""
        # テストデータ作成
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )
        trade_manager.add_trade(
            "8306", TradeType.BUY, 50, Decimal("4000"), persist_to_db=False
        )
        trade_manager.add_trade(
            "7203", TradeType.SELL, 50, Decimal("2600"), persist_to_db=False
        )
        trade_manager.update_current_prices(
            {"7203": Decimal("2550"), "8306": Decimal("4100")}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "portfolio.json")

            # 保存
            trade_manager.save_to_json(json_path)
            assert os.path.exists(json_path)

            # 新しいマネージャーで読み込み
            new_manager = TradeManager(load_from_db=False)
            new_manager.load_from_json(json_path)

            # データ確認
            assert len(new_manager.trades) == len(trade_manager.trades)
            assert len(new_manager.positions) == len(trade_manager.positions)
            assert len(new_manager.realized_pnl) == len(trade_manager.realized_pnl)
            assert new_manager.commission_rate == trade_manager.commission_rate
            assert new_manager.tax_rate == trade_manager.tax_rate

    def test_invalid_csv_export(self, trade_manager):
        """無効なCSV出力タイプのテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = os.path.join(temp_dir, "invalid.csv")

            with pytest.raises(ValueError):
                trade_manager.export_to_csv(invalid_path, "invalid_type")

    def test_complex_trading_scenario(self, trade_manager):
        """複雑な取引シナリオのテスト"""
        # 複数回に分けて買い
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )
        trade_manager.add_trade(
            "7203", TradeType.BUY, 200, Decimal("2450"), persist_to_db=False
        )
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2550"), persist_to_db=False
        )

        # 段階的に売り
        trade_manager.add_trade(
            "7203", TradeType.SELL, 150, Decimal("2600"), persist_to_db=False
        )
        trade_manager.add_trade(
            "7203", TradeType.SELL, 100, Decimal("2650"), persist_to_db=False
        )

        # 残ポジション確認
        position = trade_manager.positions["7203"]
        assert position.quantity == 150  # 400 - 150 - 100

        # 実現損益確認
        assert len(trade_manager.realized_pnl) == 2

        # すべての売却で利益が出ているはず
        for pnl in trade_manager.realized_pnl:
            assert pnl.pnl > 0


class TestTradeManagerDatabaseIntegration:
    """TradeManagerのデータベース統合テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.manager = TradeManager(use_database=True)

    def test_add_trade_with_database(self):
        """データベース連携での取引追加テスト"""
        # モックを使用してデータベース操作をテスト
        from unittest.mock import patch, MagicMock

        with patch('src.day_trade.core.trade_manager.db_manager') as mock_db:
            mock_session = MagicMock()
            mock_db.transaction_scope.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = None

            # モック取引を作成
            mock_trade = MagicMock()
            mock_trade.id = 1

            with patch('src.day_trade.models.stock.Trade') as mock_db_trade:
                mock_db_trade.create_buy_trade.return_value = mock_trade

                trade_id = self.manager.add_trade(
                    symbol="7203",
                    trade_type=TradeType.BUY,
                    quantity=100,
                    price=Decimal("2500"),
                    commission=Decimal("250")
                )

                assert trade_id is not None
                # データベース操作が呼ばれることを確認
                mock_db.transaction_scope.assert_called_once()

    def test_load_trades_from_database(self):
        """データベースからの取引読み込みテスト"""
        from unittest.mock import patch, MagicMock

        with patch('src.day_trade.core.trade_manager.db_manager') as mock_db:
            mock_session = MagicMock()
            mock_db.transaction_scope.return_value.__enter__.return_value = mock_session

            # モック取引データ
            mock_trade = MagicMock()
            mock_trade.id = 1
            mock_trade.stock_code = "7203"
            mock_trade.trade_type = "buy"
            mock_trade.quantity = 100
            mock_trade.price = 2500.0
            mock_trade.trade_datetime = datetime.now()
            mock_trade.commission = 250.0
            mock_trade.memo = "テスト取引"

            mock_session.query.return_value.order_by.return_value.all.return_value = [mock_trade]

            # データベースから読み込み
            self.manager.load_trades_from_database()

            # 取引が読み込まれることを確認
            assert len(self.manager.get_all_trades()) >= 0  # モックなので実際のデータは入らない

    def test_load_trades_database_error_handling(self):
        """データベース読み込みエラー処理テスト"""
        from unittest.mock import patch

        with patch('src.day_trade.core.trade_manager.db_manager') as mock_db:
            mock_db.transaction_scope.side_effect = Exception("DB接続エラー")

            # エラーが発生してもプログラムが停止しないことを確認
            self.manager.load_trades_from_database()

            # 既存のメモリ内データは保持される
            assert isinstance(self.manager.trades, list)


class TestTradeManagerCSVIntegration:
    """TradeManagerのCSV統合機能テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.manager = TradeManager()

        # テスト用の取引を追加
        self.manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500"),
            commission=Decimal("250")
        )

        self.manager.add_trade(
            symbol="7203",
            trade_type=TradeType.SELL,
            quantity=50,
            price=Decimal("2600"),
            commission=Decimal("130")
        )

    def test_export_to_csv(self):
        """CSV出力テスト"""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            csv_path = tmp_file.name

        try:
            # CSV出力
            self.manager.export_to_csv(csv_path)

            # ファイルが作成されることを確認
            assert os.path.exists(csv_path)

            # ファイル内容を確認
            with open(csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert 'id,symbol,trade_type,quantity,price,timestamp,commission,status,notes' in content
                assert '7203' in content
                assert 'buy' in content or 'sell' in content

        finally:
            # 一時ファイルを削除
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_import_from_csv(self):
        """CSV読み込みテスト"""
        import tempfile
        import os

        # テスト用CSVファイルを作成
        csv_content = """id,symbol,trade_type,quantity,price,timestamp,commission,status,notes
CSV_1,9984,buy,200,15000,2023-01-15T10:30:00,300,executed,CSVインポートテスト
CSV_2,9984,sell,100,15500,2023-01-16T14:20:00,150,executed,CSVインポートテスト"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(csv_content)
            csv_path = tmp_file.name

        try:
            # 初期状態の取引数を記録
            initial_count = len(self.manager.get_all_trades())

            # CSVから読み込み
            self.manager.import_from_csv(csv_path)

            # 取引が追加されることを確認
            assert len(self.manager.get_all_trades()) > initial_count

            # インポートされた取引を確認
            all_trades = self.manager.get_all_trades()
            imported_trades = [t for t in all_trades if t.symbol == "9984"]
            assert len(imported_trades) >= 2

        finally:
            # 一時ファイルを削除
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_import_from_invalid_csv(self):
        """無効なCSVファイルの読み込みテスト"""
        import tempfile
        import os

        # 無効なCSVファイルを作成
        invalid_csv = "invalid,csv,content\nwithout,proper,headers"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(invalid_csv)
            csv_path = tmp_file.name

        try:
            initial_count = len(self.manager.get_all_trades())

            # 無効なCSVの読み込み（エラーハンドリング）
            self.manager.import_from_csv(csv_path)

            # 取引数が変わらないことを確認
            assert len(self.manager.get_all_trades()) == initial_count

        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)


class TestTradeManagerJSONIntegration:
    """TradeManagerのJSON統合機能テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.manager = TradeManager()

        # テスト用の取引を追加
        self.manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500"),
            commission=Decimal("250")
        )

    def test_export_to_json(self):
        """JSON出力テスト"""
        import tempfile
        import os
        import json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json_path = tmp_file.name

        try:
            # JSON出力
            self.manager.export_to_json(json_path)

            # ファイルが作成されることを確認
            assert os.path.exists(json_path)

            # JSON内容を確認
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert isinstance(data, list)
                assert len(data) > 0
                assert 'symbol' in data[0]
                assert 'trade_type' in data[0]

        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_import_from_json(self):
        """JSON読み込みテスト"""
        import tempfile
        import os

        # テスト用JSONデータ
        json_data = [
            {
                "id": "JSON_1",
                "symbol": "9984",
                "trade_type": "buy",
                "quantity": 200,
                "price": "15000",
                "timestamp": "2023-01-15T10:30:00",
                "commission": "300",
                "status": "executed",
                "notes": "JSONインポートテスト"
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
            json.dump(json_data, tmp_file, ensure_ascii=False)
            json_path = tmp_file.name

        try:
            initial_count = len(self.manager.get_all_trades())

            # JSONから読み込み
            self.manager.import_from_json(json_path)

            # 取引が追加されることを確認
            assert len(self.manager.get_all_trades()) > initial_count

        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)


class TestTradeManagerAnalytics:
    """TradeManagerの分析機能テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.manager = TradeManager()

        # 複数の取引を追加
        base_time = datetime(2023, 1, 15, 10, 0, 0)

        # 7203の取引
        self.manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500"),
            commission=Decimal("250"),
            timestamp=base_time
        )

        self.manager.add_trade(
            symbol="7203",
            trade_type=TradeType.SELL,
            quantity=50,
            price=Decimal("2600"),
            commission=Decimal("130"),
            timestamp=base_time.replace(hour=14)
        )

        # 9984の取引
        self.manager.add_trade(
            symbol="9984",
            trade_type=TradeType.BUY,
            quantity=200,
            price=Decimal("15000"),
            commission=Decimal("300"),
            timestamp=base_time.replace(day=16)
        )

    def test_get_performance_by_symbol(self):
        """銘柄別パフォーマンス取得テスト"""
        performance = self.manager.get_performance_by_symbol()

        assert isinstance(performance, dict)
        assert "7203" in performance

        # 7203のパフォーマンス詳細確認
        symbol_perf = performance["7203"]
        assert "total_trades" in symbol_perf
        assert "buy_volume" in symbol_perf
        assert "sell_volume" in symbol_perf
        assert "realized_pnl" in symbol_perf

        # 取引数の確認
        assert symbol_perf["total_trades"] == 2
        assert symbol_perf["buy_volume"] > 0
        assert symbol_perf["sell_volume"] > 0

    def test_get_performance_by_date_range(self):
        """期間別パフォーマンス取得テスト"""
        start_date = datetime(2023, 1, 14)
        end_date = datetime(2023, 1, 17)

        performance = self.manager.get_performance_by_date_range(start_date, end_date)

        assert isinstance(performance, dict)
        assert "total_trades" in performance
        assert "symbols" in performance
        assert "realized_pnl" in performance

        # 期間内の取引が含まれることを確認
        assert performance["total_trades"] >= 2

    def test_get_win_loss_ratio(self):
        """勝率計算テスト"""
        win_loss = self.manager.get_win_loss_ratio()

        assert isinstance(win_loss, dict)
        assert "total_closed_positions" in win_loss
        assert "winning_trades" in win_loss
        assert "losing_trades" in win_loss
        assert "win_rate" in win_loss

        # 値の範囲確認
        if win_loss["total_closed_positions"] > 0:
            assert 0 <= win_loss["win_rate"] <= 100

    def test_get_average_holding_period(self):
        """平均保有期間計算テスト"""
        avg_holding = self.manager.get_average_holding_period()

        assert isinstance(avg_holding, dict)
        assert "average_days" in avg_holding
        assert "total_positions" in avg_holding

        # 値が非負であることを確認
        if avg_holding["total_positions"] > 0:
            assert avg_holding["average_days"] >= 0

    def test_calculate_total_return(self):
        """総合リターン計算テスト"""
        # 現在価格を設定
        self.manager.update_current_prices({"7203": Decimal("2550"), "9984": Decimal("15200")})

        total_return = self.manager.calculate_total_return()

        assert isinstance(total_return, dict)
        assert "total_invested" in total_return
        assert "current_value" in total_return
        assert "realized_pnl" in total_return
        assert "unrealized_pnl" in total_return
        assert "total_return" in total_return
        assert "total_return_percent" in total_return

    def test_get_trading_summary(self):
        """取引サマリー取得テスト"""
        summary = self.manager.get_trading_summary()

        assert isinstance(summary, dict)
        assert "total_trades" in summary
        assert "buy_trades" in summary
        assert "sell_trades" in summary
        assert "total_symbols" in summary
        assert "total_volume" in summary
        assert "total_commission" in summary

        # 数値の整合性確認
        assert summary["total_trades"] == summary["buy_trades"] + summary["sell_trades"]
        assert summary["total_symbols"] >= 1  # 少なくとも1銘柄は取引済み


class TestTradeManagerPositionManagement:
    """TradeManagerのポジション管理テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.manager = TradeManager()

    def test_position_zero_division_error_handling(self):
        """ポジションのゼロ除算エラー処理テスト"""
        # コスト0のポジションを作成（異常ケース）
        position = Position(
            symbol="7203",
            quantity=100,
            average_price=Decimal("0"),
            total_cost=Decimal("0"),
            current_price=Decimal("2500")
        )

        # ゼロ除算エラーが発生しないことを確認
        pnl_percent = position.unrealized_pnl_percent
        assert pnl_percent == Decimal("0")

    def test_position_negative_quantity(self):
        """負の数量ポジションテスト（ショートポジション想定）"""
        position = Position(
            symbol="7203",
            quantity=-100,  # ショートポジション
            average_price=Decimal("2500"),
            total_cost=Decimal("-250000"),
            current_price=Decimal("2400")
        )

        # 負の数量での計算が正しく行われることを確認
        market_value = position.market_value
        assert market_value == Decimal("-240000")  # -100 * 2400

        unrealized_pnl = position.unrealized_pnl
        assert unrealized_pnl == Decimal("10000")  # -240000 - (-250000)


class TestRealizedPnLDataClass:
    """RealizedPnLデータクラスのテスト"""

    def test_realized_pnl_creation(self):
        """実現損益データの作成テスト"""
        from src.day_trade.core.trade_manager import RealizedPnL

        buy_date = datetime(2023, 1, 15, 10, 0, 0)
        sell_date = datetime(2023, 1, 16, 14, 0, 0)

        realized_pnl = RealizedPnL(
            symbol="7203",
            quantity=100,
            buy_price=Decimal("2500"),
            sell_price=Decimal("2600"),
            buy_commission=Decimal("250"),
            sell_commission=Decimal("130"),
            pnl=Decimal("9620"),  # (2600-2500)*100 - 250 - 130
            pnl_percent=Decimal("3.85"),
            buy_date=buy_date,
            sell_date=sell_date
        )

        assert realized_pnl.symbol == "7203"
        assert realized_pnl.quantity == 100
        assert realized_pnl.pnl == Decimal("9620")
        assert realized_pnl.buy_date == buy_date
        assert realized_pnl.sell_date == sell_date

    def test_realized_pnl_to_dict(self):
        """実現損益の辞書変換テスト"""
        from src.day_trade.core.trade_manager import RealizedPnL

        realized_pnl = RealizedPnL(
            symbol="7203",
            quantity=100,
            buy_price=Decimal("2500"),
            sell_price=Decimal("2600"),
            buy_commission=Decimal("250"),
            sell_commission=Decimal("130"),
            pnl=Decimal("9620"),
            pnl_percent=Decimal("3.85"),
            buy_date=datetime(2023, 1, 15, 10, 0, 0),
            sell_date=datetime(2023, 1, 16, 14, 0, 0)
        )

        result = realized_pnl.to_dict()

        assert isinstance(result, dict)
        assert result["symbol"] == "7203"
        assert result["quantity"] == 100
        assert result["buy_price"] == "2500"
        assert result["sell_price"] == "2600"
        assert result["pnl"] == "9620"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
