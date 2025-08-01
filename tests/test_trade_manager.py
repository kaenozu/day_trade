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
        return TradeManager(commission_rate=Decimal("0.001"), tax_rate=Decimal("0.2"))

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
        trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        # 2回目の買い
        trade_manager.add_trade("7203", TradeType.BUY, 200, Decimal("2450"))

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
        trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        # 一部売却
        trade_manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))

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
        trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        # 完全売却
        trade_manager.add_trade("7203", TradeType.SELL, 100, Decimal("2600"))

        # ポジションが削除されることを確認
        assert "7203" not in trade_manager.positions

        # 実現損益確認
        assert len(trade_manager.realized_pnl) == 1

    def test_sell_without_position(self, trade_manager):
        """ポジションなしでの売却テスト"""
        # ポジションなしで売却（警告ログが出るが処理は継続）
        trade_manager.add_trade("7203", TradeType.SELL, 100, Decimal("2600"))

        assert len(trade_manager.trades) == 1
        assert len(trade_manager.positions) == 0
        assert len(trade_manager.realized_pnl) == 0

    def test_sell_excess_quantity(self, trade_manager):
        """過剰売却のテスト"""
        # 100株買い
        trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        # 150株売却（過剰）- 警告が出るが処理は継続される
        trade_manager.add_trade("7203", TradeType.SELL, 150, Decimal("2600"))

        # ポジションは残っている（過剰分は処理されない）
        assert "7203" in trade_manager.positions
        assert trade_manager.positions["7203"].quantity == 100  # 元のまま
        assert len(trade_manager.realized_pnl) == 0  # 実現損益は発生しない

    def test_current_price_update(self, trade_manager):
        """現在価格更新のテスト"""
        trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        trade_manager.add_trade("8306", TradeType.BUY, 50, Decimal("4000"))

        # 価格更新
        prices = {"7203": Decimal("2600"), "8306": Decimal("4200")}
        trade_manager.update_current_prices(prices)

        # 更新確認
        assert trade_manager.positions["7203"].current_price == Decimal("2600")
        assert trade_manager.positions["8306"].current_price == Decimal("4200")

    def test_portfolio_summary(self, trade_manager):
        """ポートフォリオサマリーのテスト"""
        # 複数取引
        trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        trade_manager.add_trade("8306", TradeType.BUY, 50, Decimal("4000"))
        trade_manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))

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
        trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        trade_manager.add_trade("8306", TradeType.BUY, 50, Decimal("4000"))
        trade_manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))

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
        trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        trade_manager.add_trade("8306", TradeType.BUY, 50, Decimal("4000"))
        trade_manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))
        trade_manager.add_trade("8306", TradeType.SELL, 25, Decimal("3900"))

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
        trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        trade_manager.add_trade("7203", TradeType.SELL, 100, Decimal("2600"))

        tax_info = trade_manager.calculate_tax_implications(current_year)

        assert tax_info["year"] == current_year
        assert tax_info["total_trades"] == 1
        assert Decimal(tax_info["total_gain"]) > 0
        assert Decimal(tax_info["net_gain"]) > 0
        assert Decimal(tax_info["tax_due"]) > 0

    def test_csv_export(self, trade_manager):
        """CSV出力のテスト"""
        # テストデータ作成
        trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        trade_manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))
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
        trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        trade_manager.add_trade("8306", TradeType.BUY, 50, Decimal("4000"))
        trade_manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))
        trade_manager.update_current_prices(
            {"7203": Decimal("2550"), "8306": Decimal("4100")}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "portfolio.json")

            # 保存
            trade_manager.save_to_json(json_path)
            assert os.path.exists(json_path)

            # 新しいマネージャーで読み込み
            new_manager = TradeManager()
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
        trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        trade_manager.add_trade("7203", TradeType.BUY, 200, Decimal("2450"))
        trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("2550"))

        # 段階的に売り
        trade_manager.add_trade("7203", TradeType.SELL, 150, Decimal("2600"))
        trade_manager.add_trade("7203", TradeType.SELL, 100, Decimal("2650"))

        # 残ポジション確認
        position = trade_manager.positions["7203"]
        assert position.quantity == 150  # 400 - 150 - 100

        # 実現損益確認
        assert len(trade_manager.realized_pnl) == 2

        # すべての売却で利益が出ているはず
        for pnl in trade_manager.realized_pnl:
            assert pnl.pnl > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
