"""
TradeManagerのカバレッジ向上を目的とした追加テスト
"""

import csv
import json
import os
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.day_trade.core.trade_manager import (
    TradeManager,
    TradeType,
    quantize_decimal,
)


class TestQuantizeDecimal:
    """quantize_decimal関数のテスト"""

    def test_quantize_decimal_default(self):
        """デフォルト精度での量子化テスト"""
        result = quantize_decimal(Decimal("123.456789"))
        assert result == Decimal("123.46")

    def test_quantize_decimal_custom_precision(self):
        """カスタム精度での量子化テスト"""
        result = quantize_decimal(Decimal("123.456789"), 4)
        assert result == Decimal("123.4568")

    def test_quantize_decimal_zero_precision(self):
        """精度0での量子化テスト"""
        result = quantize_decimal(Decimal("123.456789"), 0)
        assert result == Decimal("123")


class TestTradeManagerFileOperations:
    """TradeManagerのファイル操作テスト"""

    def test_export_to_csv_success(self):
        """CSV出力成功テスト"""
        manager = TradeManager()
        
        # テスト用の取引を追加
        manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500")
        )
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            try:
                manager.export_to_csv(tmp_file.name)
                
                # ファイルが作成されたことを確認
                assert os.path.exists(tmp_file.name)
                
                # CSVの内容を確認
                with open(tmp_file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                    assert '7203' in content
                    assert 'BUY' in content
                    
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)

    def test_export_to_csv_file_error(self):
        """CSV出力ファイルエラーテスト"""
        manager = TradeManager()
        
        # 存在しないディレクトリへの書き込み
        with pytest.raises(Exception):
            manager.export_to_csv("/nonexistent/path/test.csv")

    def test_export_to_json_success(self):
        """JSON出力成功テスト"""
        manager = TradeManager()
        
        # テスト用の取引を追加
        manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500")
        )
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            try:
                manager.export_to_json(tmp_file.name)
                
                # ファイルが作成されたことを確認
                assert os.path.exists(tmp_file.name)
                
                # JSONの内容を確認
                with open(tmp_file.name, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    assert len(data['trades']) > 0
                    assert data['trades'][0]['symbol'] == '7203'
                    
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)

    def test_import_from_json_success(self):
        """JSON読み込み成功テスト"""
        manager = TradeManager()
        
        # テストデータを作成
        test_data = {
            "trades": [
                {
                    "id": "T001",
                    "symbol": "7203",
                    "trade_type": "BUY",
                    "quantity": 100,
                    "price": "2500",
                    "timestamp": datetime.now().isoformat(),
                    "commission": "250",
                    "status": "executed",
                    "notes": ""
                }
            ],
            "positions": {},
            "realized_pnl_history": []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            try:
                json.dump(test_data, tmp_file, ensure_ascii=False, indent=2)
                tmp_file.flush()
                
                # JSONから読み込み
                manager.import_from_json(tmp_file.name)
                
                # データが正しく読み込まれたことを確認
                assert len(manager.trades) == 1
                assert manager.trades[0].symbol == "7203"
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)

    def test_import_from_json_file_not_found(self):
        """JSON読み込みファイル不存在テスト"""
        manager = TradeManager()
        
        with pytest.raises(FileNotFoundError):
            manager.import_from_json("nonexistent_file.json")

    def test_import_from_json_invalid_format(self):
        """JSON読み込み無効フォーマットテスト"""
        manager = TradeManager()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            try:
                tmp_file.write("invalid json content")
                tmp_file.flush()
                
                with pytest.raises(json.JSONDecodeError):
                    manager.import_from_json(tmp_file.name)
                    
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)


class TestTradeManagerAnalytics:
    """TradeManagerの分析機能テスト"""

    def test_get_portfolio_summary(self):
        """ポートフォリオサマリー取得テスト"""
        manager = TradeManager()
        
        # テスト用の取引を追加
        manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500")
        )
        
        summary = manager.get_portfolio_summary()
        
        assert 'total_value' in summary
        assert 'cash_balance' in summary
        assert 'positions' in summary
        assert isinstance(summary['total_value'], Decimal)

    def test_get_trade_history(self):
        """取引履歴取得テスト"""
        manager = TradeManager()
        
        # テスト用の取引を追加
        manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500")
        )
        
        history = manager.get_trade_history()
        
        assert len(history) == 1
        assert history[0]['symbol'] == '7203'
        assert history[0]['trade_type'] == 'BUY'

    def test_get_realized_pnl_history(self):
        """実現損益履歴取得テスト"""
        manager = TradeManager()
        
        # 買い→売りの取引を実行
        manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500")
        )
        manager.add_trade(
            symbol="7203",
            trade_type=TradeType.SELL,
            quantity=50,
            price=Decimal("2600")
        )
        
        pnl_history = manager.get_realized_pnl_history()
        
        assert len(pnl_history) > 0
        assert 'symbol' in pnl_history[0]
        assert 'pnl' in pnl_history[0]

    def test_calculate_tax(self):
        """税金計算テスト"""
        manager = TradeManager()
        
        # 利益のある実現損益を作成
        profit = Decimal("10000")
        tax = manager.calculate_tax(profit)
        
        expected_tax = profit * manager.tax_rate
        assert tax == expected_tax

    def test_calculate_tax_loss(self):
        """損失時の税金計算テスト"""
        manager = TradeManager()
        
        # 損失の場合
        loss = Decimal("-5000")
        tax = manager.calculate_tax(loss)
        
        # 損失の場合は税金なし
        assert tax == Decimal("0")


class TestTradeManagerComplexScenarios:
    """TradeManagerの複合シナリオテスト"""

    def test_multiple_symbols_trading(self):
        """複数銘柄取引テスト"""
        manager = TradeManager()
        
        # 複数銘柄での取引
        symbols = ["7203", "6758", "4063"]
        
        for symbol in symbols:
            manager.add_trade(
                symbol=symbol,
                trade_type=TradeType.BUY,
                quantity=100,
                price=Decimal("1000")
            )
        
        # 各銘柄のポジションが作成されていることを確認
        assert len(manager.positions) == 3
        for symbol in symbols:
            assert symbol in manager.positions
            assert manager.positions[symbol].quantity == 100

    def test_partial_sales_scenario(self):
        """部分売却シナリオテスト"""
        manager = TradeManager()
        
        # 1000株購入
        manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=1000,
            price=Decimal("2500")
        )
        
        # 300株売却
        manager.add_trade(
            symbol="7203",
            trade_type=TradeType.SELL,
            quantity=300,
            price=Decimal("2600")
        )
        
        # 残りポジション確認
        position = manager.positions["7203"]
        assert position.quantity == 700
        
        # 実現損益が記録されていることを確認
        assert len(manager.realized_pnl_history) == 1

    def test_price_update_scenario(self):
        """価格更新シナリオテスト"""
        manager = TradeManager()
        
        # ポジション作成
        manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500")
        )
        
        # 価格を複数回更新
        prices = [Decimal("2600"), Decimal("2400"), Decimal("2700")]
        
        for price in prices:
            manager.update_current_price("7203", price)
            position = manager.positions["7203"]
            assert position.current_price == price


class TestTradeManagerErrorHandling:
    """TradeManagerのエラーハンドリングテスト"""

    def test_invalid_trade_parameters(self):
        """無効な取引パラメータテスト"""
        manager = TradeManager()
        
        # 無効な銘柄コード
        with pytest.raises(ValueError):
            manager.add_trade(
                symbol="",
                trade_type=TradeType.BUY,
                quantity=100,
                price=Decimal("2500")
            )
        
        # 負の数量
        with pytest.raises(ValueError):
            manager.add_trade(
                symbol="7203",
                trade_type=TradeType.BUY,
                quantity=-100,
                price=Decimal("2500")
            )
        
        # 負の価格
        with pytest.raises(ValueError):
            manager.add_trade(
                symbol="7203",
                trade_type=TradeType.BUY,
                quantity=100,
                price=Decimal("-2500")
            )

    def test_position_edge_cases(self):
        """ポジションのエッジケーステスト"""
        manager = TradeManager()
        
        # ポジションがない状態での価格更新
        manager.update_current_price("7203", Decimal("2500"))
        # エラーが発生しないことを確認
        
        # 存在しない銘柄での売却試行
        with pytest.raises(ValueError):
            manager.add_trade(
                symbol="7203",
                trade_type=TradeType.SELL,
                quantity=100,
                price=Decimal("2500")
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])