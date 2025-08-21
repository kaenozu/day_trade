"""
TradeManagerの100%カバレッジを目指す戦略的テスト
未カバー行を効率的にテストする
"""

import math
import os
import tempfile
from datetime import datetime
from decimal import Decimal, InvalidOperation
from unittest.mock import patch, MagicMock

import pytest

from src.day_trade.core.trade_manager import (
    TradeManager,
    TradeType,
    TradeStatus,
    safe_decimal_conversion,
    validate_positive_decimal,
    validate_file_path,
)


class TestSafeDecimalConversionComprehensive:
    """safe_decimal_conversion関数の完全カバレッジテスト"""
    
    def test_none_value(self):
        """None値のテスト"""
        with pytest.raises(ValueError, match="がNoneです"):
            safe_decimal_conversion(None)
    
    def test_int_conversion(self):
        """int変換テスト"""
        result = safe_decimal_conversion(123)
        assert result == Decimal("123")
    
    def test_string_empty(self):
        """空文字列テスト"""
        with pytest.raises(ValueError, match="空の文字列"):
            safe_decimal_conversion("")
        
        with pytest.raises(ValueError, match="空の文字列"):
            safe_decimal_conversion("   ")
    
    def test_dangerous_string_patterns(self):
        """危険な文字列パターンテスト"""
        dangerous_values = ["inf", "-inf", "nan", "null", "undefined"]
        for value in dangerous_values:
            with pytest.raises(ValueError, match="無効な数値文字列"):
                safe_decimal_conversion(value)
            # 大文字小文字混在もテスト
            with pytest.raises(ValueError, match="無効な数値文字列"):
                safe_decimal_conversion(value.upper())
    
    def test_invalid_string_format(self):
        """無効な文字列形式テスト"""
        invalid_values = ["abc", "12.34.56", "12a34", "++123"]
        for value in invalid_values:
            with pytest.raises(ValueError, match="数値形式ではありません"):
                safe_decimal_conversion(value)
    
    def test_comma_separated_string(self):
        """カンマ区切り文字列テスト"""
        result = safe_decimal_conversion("1,234.56")
        assert result == Decimal("1234.56")
    
    def test_float_special_values(self):
        """float特殊値テスト"""
        # 無限大
        with pytest.raises(ValueError, match="無限大またはNaN"):
            safe_decimal_conversion(float('inf'))
        
        with pytest.raises(ValueError, match="無限大またはNaN"):
            safe_decimal_conversion(float('-inf'))
        
        # NaN
        with pytest.raises(ValueError, match="無限大またはNaN"):
            safe_decimal_conversion(float('nan'))
    
    def test_float_extreme_values(self):
        """float極端値テスト"""
        # 極大値
        with pytest.raises(ValueError, match="値が大きすぎます"):
            safe_decimal_conversion(1e16)
        
        # 極小値
        with pytest.raises(ValueError, match="値が小さすぎます"):
            safe_decimal_conversion(1e-15)
    
    def test_float_valid_conversion(self):
        """float有効変換テスト"""
        result = safe_decimal_conversion(123.456)
        assert isinstance(result, Decimal)
    
    def test_invalid_type(self):
        """無効な型テスト"""
        with pytest.raises(TypeError, match="サポートされていない型"):
            safe_decimal_conversion([1, 2, 3])
        
        with pytest.raises(TypeError, match="サポートされていない型"):
            safe_decimal_conversion({"key": "value"})
    
    def test_default_value_usage(self):
        """デフォルト値使用テスト"""
        default = Decimal("999")
        result = safe_decimal_conversion("invalid", "テスト", default)
        assert result == default
    
    def test_decimal_infinity_check(self):
        """Decimal無限大チェックテスト"""
        with pytest.raises(ValueError, match="Decimal値が無限大またはNaN"):
            safe_decimal_conversion(Decimal('inf'))


class TestValidationFunctionsComprehensive:
    """バリデーション関数の完全テスト"""
    
    def test_validate_positive_decimal_type_error(self):
        """型エラーテスト"""
        with pytest.raises(TypeError, match="Decimal型である必要があります"):
            validate_positive_decimal("123", "テスト値")
    
    def test_validate_positive_decimal_zero_allowed(self):
        """ゼロ許可テスト"""
        # ゼロ許可の場合
        result = validate_positive_decimal(Decimal("0"), "テスト値", allow_zero=True)
        assert result == Decimal("0")
        
        # 負数はエラー
        with pytest.raises(ValueError, match="0以上である必要があります"):
            validate_positive_decimal(Decimal("-1"), "テスト値", allow_zero=True)
    
    def test_validate_file_path_nonexistent(self):
        """存在しないファイルパステスト"""
        with pytest.raises(ValueError, match="ファイル操作"):
            validate_file_path("/nonexistent/path/file.txt")


class TestTradeManagerEdgeCases:
    """TradeManagerのエッジケーステスト"""
    
    def test_initialization_with_custom_params(self):
        """カスタムパラメータでの初期化テスト"""
        manager = TradeManager(
            initial_capital=Decimal("500000"),
            commission_rate=Decimal("0.002"),
            tax_rate=Decimal("0.25"),
            persist_to_db=False
        )
        assert manager.initial_capital == Decimal("500000")
        assert manager.commission_rate == Decimal("0.002")
        assert manager.tax_rate == Decimal("0.25")
    
    def test_add_trade_with_notes(self):
        """ノート付き取引追加テスト"""
        manager = TradeManager()
        trade_id = manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500"),
            notes="テスト取引"
        )
        assert trade_id is not None
        trade = manager.trades[0]
        assert trade.notes == "テスト取引"
    
    def test_add_trade_with_status(self):
        """ステータス付き取引追加テスト"""
        manager = TradeManager()
        trade_id = manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500"),
            status=TradeStatus.PENDING
        )
        trade = manager.trades[0]
        assert trade.status == TradeStatus.PENDING
    
    def test_update_current_price_new_position(self):
        """新しいポジションの価格更新テスト"""
        manager = TradeManager()
        # ポジションがない状態で価格更新（エラーなし）
        manager.update_current_price("7203", Decimal("2500"))
        # この場合は何も起こらない
    
    def test_minimum_commission(self):
        """最小手数料テスト"""
        manager = TradeManager(
            commission_rate=Decimal("0.001"),
            minimum_commission=Decimal("100")
        )
        trade_id = manager.add_trade("7203", TradeType.BUY, 10, Decimal("50"))  # 手数料0.5円
        trade = manager.trades[0]
        assert trade.commission == Decimal("100")  # 最小手数料が適用される
    
    def test_calculate_tax_edge_cases(self):
        """税金計算エッジケーステスト"""
        manager = TradeManager()
        
        # ゼロ利益
        tax = manager.calculate_tax(Decimal("0"))
        assert tax == Decimal("0")
        
        # 非常に小さな利益
        tax = manager.calculate_tax(Decimal("0.01"))
        assert tax > Decimal("0")
    
    def test_portfolio_summary_with_current_prices(self):
        """現在価格付きポートフォリオサマリーテスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        manager.add_trade("6758", TradeType.BUY, 200, Decimal("1000"))
        
        # 現在価格を設定
        manager.update_current_price("7203", Decimal("2600"))
        manager.update_current_price("6758", Decimal("1100"))
        
        summary = manager.get_portfolio_summary()
        assert "total_value" in summary
        assert "unrealized_pnl" in summary
    
    def test_export_specific_data_types(self):
        """特定データタイプの出力テスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            try:
                # ポジションデータの出力
                manager.export_to_csv(tmp.name, "positions")
                assert os.path.exists(tmp.name)
                
                # 実現損益データの出力
                manager.export_to_csv(tmp.name, "realized_pnl")
                assert os.path.exists(tmp.name)
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)


class TestErrorHandlingPaths:
    """エラーハンドリングパスのテスト"""
    
    def test_invalid_symbol_validation(self):
        """無効な銘柄コード検証テスト"""
        manager = TradeManager()
        
        # 空文字列
        with pytest.raises(ValueError):
            manager.add_trade("", TradeType.BUY, 100, Decimal("2500"))
        
        # 空白のみ
        with pytest.raises(ValueError):
            manager.add_trade("   ", TradeType.BUY, 100, Decimal("2500"))
    
    def test_invalid_quantity_validation(self):
        """無効な数量検証テスト"""
        manager = TradeManager()
        
        # ゼロ数量
        with pytest.raises(ValueError):
            manager.add_trade("7203", TradeType.BUY, 0, Decimal("2500"))
        
        # 負の数量
        with pytest.raises(ValueError):
            manager.add_trade("7203", TradeType.BUY, -100, Decimal("2500"))
    
    def test_invalid_price_validation(self):
        """無効な価格検証テスト"""
        manager = TradeManager()
        
        # ゼロ価格
        with pytest.raises(ValueError):
            manager.add_trade("7203", TradeType.BUY, 100, Decimal("0"))
        
        # 負の価格
        with pytest.raises(ValueError):
            manager.add_trade("7203", TradeType.BUY, 100, Decimal("-100"))
    
    def test_insufficient_position_for_sale(self):
        """売却時のポジション不足テスト"""
        manager = TradeManager()
        
        # ポジションなしでの売却
        with pytest.raises(ValueError, match="ポジションが存在しません"):
            manager.add_trade("7203", TradeType.SELL, 100, Decimal("2500"))
        
        # 不十分なポジションでの売却
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        with pytest.raises(ValueError, match="売却数量が保有数量を超えています"):
            manager.add_trade("7203", TradeType.SELL, 150, Decimal("2600"))


class TestFileOperationPaths:
    """ファイル操作パスのテスト"""
    
    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_csv_export_permission_error(self, mock_open):
        """CSV出力権限エラーテスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        
        with pytest.raises(PermissionError):
            manager.export_to_csv("test.csv")
    
    @patch("builtins.open", side_effect=OSError("Disk full"))
    def test_json_export_disk_error(self, mock_open):
        """JSON出力ディスクエラーテスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        
        with pytest.raises(OSError):
            manager.export_to_json("test.json")
    
    def test_invalid_export_data_type(self):
        """無効な出力データタイプテスト"""
        manager = TradeManager()
        
        with pytest.raises(ValueError, match="サポートされていないデータタイプ"):
            manager.export_to_csv("test.csv", "invalid_type")


class TestDatabaseIntegrationPaths:
    """データベース統合パスのテスト"""
    
    @patch('src.day_trade.core.trade_manager.db_manager')
    def test_database_enabled_initialization(self, mock_db):
        """データベース有効時の初期化テスト"""
        mock_db.get_session.return_value.__enter__.return_value = MagicMock()
        
        manager = TradeManager(persist_to_db=True)
        assert manager.persist_to_db == True
    
    @patch('src.day_trade.core.trade_manager.db_manager')
    def test_database_save_trade(self, mock_db):
        """データベース取引保存テスト"""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db_trade = MagicMock()
        mock_db_trade.id = 123
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None
        
        manager = TradeManager(persist_to_db=True)
        trade_id = manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        
        assert trade_id is not None
        mock_session.add.assert_called()
        mock_session.commit.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])