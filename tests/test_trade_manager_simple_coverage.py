"""
TradeManagerの簡潔で実用的なカバレッジ向上テスト
実際のAPIに基づいて作成
"""

import os
import tempfile
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import pytest

from src.day_trade.core.trade_manager import (
    TradeManager,
    TradeType,
    validate_positive_decimal,
    validate_file_path,
    quantize_decimal,
    safe_decimal_conversion,
    mask_sensitive_info,
)


class TestUtilityFunctions:
    """ユーティリティ関数テスト"""
    
    def test_quantize_decimal(self):
        """Decimal量子化テスト"""
        result = quantize_decimal(Decimal("123.456"), 2)
        assert result == Decimal("123.46")
        
        result = quantize_decimal(Decimal("123.456"), 0)
        assert result == Decimal("123")

    def test_safe_decimal_conversion(self):
        """安全なDecimal変換テスト"""
        # 文字列から
        result = safe_decimal_conversion("123.45")
        assert result == Decimal("123.45")
        
        # カンマ区切り
        result = safe_decimal_conversion("1,234.56")
        assert result == Decimal("1234.56")
        
        # デフォルト値テスト
        result = safe_decimal_conversion("invalid", Decimal("0"))
        assert result == Decimal("0")

    def test_mask_sensitive_info(self):
        """機密情報マスキングテスト"""
        result = mask_sensitive_info("1234567890")
        assert result == "1*******0"
        
        result = mask_sensitive_info("ABC")
        assert result == "A*C"
        
        result = mask_sensitive_info("")
        assert result == ""

    def test_validate_positive_decimal(self):
        """正数検証テスト"""
        # 成功
        validate_positive_decimal(Decimal("100"), "テスト")
        
        # 失敗
        with pytest.raises(ValueError):
            validate_positive_decimal(Decimal("-100"), "テスト")

    def test_validate_file_path(self):
        """ファイルパス検証テスト"""
        with tempfile.NamedTemporaryFile() as tmp:
            validate_file_path(tmp.name)
        
        with pytest.raises(ValueError):
            validate_file_path("")


class TestTradeManagerBasic:
    """TradeManagerの基本機能テスト"""
    
    def test_initialization(self):
        """初期化テスト"""
        manager = TradeManager()
        assert manager.initial_capital == Decimal("1000000")
        assert manager.cash_balance == Decimal("1000000")
        assert len(manager.trades) == 0
        assert len(manager.positions) == 0

    def test_basic_trade_operations(self):
        """基本取引操作テスト"""
        manager = TradeManager()
        
        # 買い注文
        trade_id = manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        assert trade_id is not None
        assert len(manager.trades) == 1
        assert "7203" in manager.positions
        
        # 売り注文
        manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))
        assert len(manager.trades) == 2
        assert manager.positions["7203"].quantity == 50

    def test_price_updates(self):
        """価格更新テスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        
        manager.update_current_price("7203", Decimal("2600"))
        position = manager.positions["7203"]
        assert position.current_price == Decimal("2600")

    def test_portfolio_analytics(self):
        """ポートフォリオ分析テスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        
        # サマリー取得
        summary = manager.get_portfolio_summary()
        assert "total_value" in summary
        assert "positions" in summary
        
        # 取引履歴取得
        history = manager.get_trade_history()
        assert len(history) == 1
        
        # 実現損益履歴（売却後）
        manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))
        pnl_history = manager.get_realized_pnl_history()
        assert len(pnl_history) > 0

    def test_tax_calculation(self):
        """税金計算テスト"""
        manager = TradeManager()
        
        # 利益の税金
        tax = manager.calculate_tax(Decimal("10000"))
        assert tax > Decimal("0")
        
        # 損失の税金（ゼロ）
        tax = manager.calculate_tax(Decimal("-5000"))
        assert tax == Decimal("0")


class TestFileOperations:
    """ファイル操作テスト"""
    
    def test_csv_export(self):
        """CSV出力テスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            try:
                manager.export_to_csv(tmp.name)
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_json_export(self):
        """JSON出力テスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            try:
                manager.export_to_json(tmp.name)
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)


class TestErrorHandling:
    """エラーハンドリングテスト"""
    
    def test_invalid_trade_parameters(self):
        """無効な取引パラメータテスト"""
        manager = TradeManager()
        
        # 空の銘柄コード
        with pytest.raises(ValueError):
            manager.add_trade("", TradeType.BUY, 100, Decimal("2500"))
        
        # 負の数量
        with pytest.raises(ValueError):
            manager.add_trade("7203", TradeType.BUY, -100, Decimal("2500"))
        
        # 負の価格
        with pytest.raises(ValueError):
            manager.add_trade("7203", TradeType.BUY, 100, Decimal("-2500"))

    def test_insufficient_position_for_sale(self):
        """売却時のポジション不足テスト"""
        manager = TradeManager()
        
        # ポジションなしでの売却
        with pytest.raises(ValueError):
            manager.add_trade("7203", TradeType.SELL, 100, Decimal("2500"))

    @patch("builtins.open", side_effect=PermissionError("Access denied"))
    def test_file_operation_errors(self, mock_open):
        """ファイル操作エラーテスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        
        with pytest.raises(PermissionError):
            manager.export_to_csv("test.csv")


class TestComplexScenarios:
    """複合シナリオテスト"""
    
    def test_multiple_trades_same_symbol(self):
        """同一銘柄での複数取引テスト"""
        manager = TradeManager()
        
        # 複数回に分けて購入
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2400"))
        manager.add_trade("7203", TradeType.BUY, 200, Decimal("2500"))
        manager.add_trade("7203", TradeType.BUY, 150, Decimal("2600"))
        
        position = manager.positions["7203"]
        assert position.quantity == 450
        
        # 部分売却
        manager.add_trade("7203", TradeType.SELL, 250, Decimal("2700"))
        assert position.quantity == 200
        
        # 完全売却
        manager.add_trade("7203", TradeType.SELL, 200, Decimal("2800"))
        assert "7203" not in manager.positions

    def test_multiple_symbols(self):
        """複数銘柄取引テスト"""
        manager = TradeManager()
        
        symbols = ["7203", "6758", "4063"]
        for symbol in symbols:
            manager.add_trade(symbol, TradeType.BUY, 100, Decimal("1000"))
            manager.update_current_price(symbol, Decimal("1100"))
        
        assert len(manager.positions) == 3
        summary = manager.get_portfolio_summary()
        assert len(summary["positions"]) == 3

    def test_commission_calculation(self):
        """手数料計算テスト"""
        manager = TradeManager(commission_rate=Decimal("0.001"))
        
        trade_id = manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        trade = manager.trades[0]
        
        # 手数料が正しく計算されていることを確認
        expected_commission = Decimal("2500") * 100 * Decimal("0.001")
        assert trade.commission == expected_commission


if __name__ == "__main__":
    pytest.main([__file__, "-v"])