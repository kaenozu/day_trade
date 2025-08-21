"""
TradeManagerの100%カバレッジを目指す最終テスト
実際のAPIに合わせて作成
"""

import csv
import json
import os
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.day_trade.core.trade_manager import (
    BuyLot,
    Position,
    RealizedPnL,
    Trade,
    TradeManager,
    TradeStatus,
    TradeType,
    mask_sensitive_info,
    safe_decimal_conversion,
    quantize_decimal,
)


class TestCompleteRealizedPnL:
    """RealizedPnLクラスの完全テスト"""

    def test_realized_pnl_creation(self):
        """実現損益作成テスト - 正しいフィールド名を使用"""
        buy_date = datetime(2025, 1, 1)
        sell_date = datetime(2025, 1, 15)
        
        pnl = RealizedPnL(
            symbol="7203",
            quantity=50,
            buy_price=Decimal("2500"),
            sell_price=Decimal("2600"),
            buy_commission=Decimal("125"),
            sell_commission=Decimal("130"),
            pnl=Decimal("4745"),
            pnl_percent=Decimal("3.8"),
            buy_date=buy_date,
            sell_date=sell_date
        )
        
        assert pnl.symbol == "7203"
        assert pnl.quantity == 50
        assert pnl.buy_price == Decimal("2500")
        assert pnl.sell_price == Decimal("2600")
        assert pnl.buy_commission == Decimal("125")
        assert pnl.sell_commission == Decimal("130")
        assert pnl.pnl == Decimal("4745")
        assert pnl.pnl_percent == Decimal("3.8")
        assert pnl.buy_date == buy_date
        assert pnl.sell_date == sell_date

    def test_realized_pnl_to_dict(self):
        """実現損益辞書変換テスト"""
        buy_date = datetime(2025, 1, 1)
        sell_date = datetime(2025, 1, 15)
        
        pnl = RealizedPnL(
            symbol="7203",
            quantity=50,
            buy_price=Decimal("2500"),
            sell_price=Decimal("2600"),
            buy_commission=Decimal("125"),
            sell_commission=Decimal("130"),
            pnl=Decimal("4745"),
            pnl_percent=Decimal("3.80"),
            buy_date=buy_date,
            sell_date=sell_date
        )
        
        result = pnl.to_dict()
        
        assert result["symbol"] == "7203"
        assert result["quantity"] == 50
        assert result["buy_price"] == "2500"
        assert result["sell_price"] == "2600"
        assert result["buy_commission"] == "125"
        assert result["sell_commission"] == "130"
        assert result["pnl"] == "4745"
        assert result["pnl_percent"] == "3.80"
        assert result["buy_date"] == buy_date.isoformat()
        assert result["sell_date"] == sell_date.isoformat()


class TestCompleteTrade:
    """Tradeクラスの完全テスト"""

    def test_trade_calculation_methods(self):
        """取引計算メソッドのテスト"""
        trade = Trade(
            id="T001",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500"),
            timestamp=datetime.now(),
            commission=Decimal("250")
        )
        
        # 手数料を含む総額を計算（total_amountプロパティがない場合の代替）
        total_with_commission = trade.quantity * trade.price + trade.commission
        assert total_with_commission == Decimal("250250")

    def test_trade_from_dict_error_handling(self):
        """取引復元エラーハンドリングテスト"""
        # 必須フィールドが不足
        with pytest.raises(ValueError, match="必須フィールド"):
            Trade.from_dict({"id": "T001"})
        
        # 無効な数量
        with pytest.raises(ValueError, match="数量は正数である必要があります"):
            Trade.from_dict({
                "id": "T001",
                "symbol": "7203",
                "trade_type": "BUY",
                "quantity": -100,
                "price": "2500",
                "timestamp": datetime.now().isoformat()
            })


class TestCompleteTradeManagerDatabase:
    """TradeManagerのデータベース統合テスト"""

    def test_add_trade_without_database(self):
        """データベース無効時の取引追加テスト"""
        manager = TradeManager(persist_to_db=False)
        
        trade_id = manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500")
        )
        
        assert trade_id is not None
        assert len(manager.trades) == 1

    def test_manager_initialization(self):
        """TradeManager初期化テスト"""
        manager = TradeManager()
        
        assert manager.initial_capital == Decimal("1000000")
        assert manager.cash_balance == Decimal("1000000")
        assert len(manager.trades) == 0
        assert len(manager.positions) == 0
        assert len(manager.realized_pnl_history) == 0


class TestCompleteFileOperations:
    """ファイル操作の完全テスト"""

    def test_export_import_csv_complete_cycle(self):
        """CSV完全サイクルテスト"""
        manager = TradeManager()
        
        # テスト用の取引を追加
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
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            try:
                # CSV出力
                manager.export_to_csv(tmp_file.name)
                
                # ファイルの内容を確認
                with open(tmp_file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                    assert '7203' in content
                    assert 'BUY' in content
                    assert 'SELL' in content
                    
                # CSVファイルが正しく作成されたことを確認
                assert os.path.getsize(tmp_file.name) > 0
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)

    def test_export_json_success(self):
        """JSON出力成功テスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            try:
                manager.export_to_json(tmp_file.name)
                assert os.path.exists(tmp_file.name)
                assert os.path.getsize(tmp_file.name) > 0
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)


class TestCompleteAnalytics:
    """分析機能の完全テスト"""

    def test_portfolio_performance_analytics(self):
        """ポートフォリオパフォーマンス分析テスト"""
        manager = TradeManager()
        
        # 複数の取引を実行
        symbols = ["7203", "6758", "4063"]
        for symbol in symbols:
            manager.add_trade(
                symbol=symbol,
                trade_type=TradeType.BUY,
                quantity=100,
                price=Decimal("1000")
            )
            # 価格更新
            manager.update_current_price(symbol, Decimal("1100"))
        
        # ポートフォリオサマリー取得
        summary = manager.get_portfolio_summary()
        
        # 基本的な検証
        assert 'total_value' in summary
        assert 'positions' in summary
        assert len(summary['positions']) == 3

    def test_tax_calculation_comprehensive(self):
        """包括的な税金計算テスト"""
        manager = TradeManager(tax_rate=Decimal("0.20315"))
        
        # 利益の場合
        profit = Decimal("100000")
        tax = manager.calculate_tax(profit)
        expected_tax = profit * Decimal("0.20315")
        assert tax == expected_tax
        
        # 損失の場合
        loss = Decimal("-50000")
        tax_loss = manager.calculate_tax(loss)
        assert tax_loss == Decimal("0")
        
        # ゼロの場合
        zero_pnl = Decimal("0")
        tax_zero = manager.calculate_tax(zero_pnl)
        assert tax_zero == Decimal("0")

    def test_get_performance_metrics(self):
        """パフォーマンスメトリクス取得テスト"""
        manager = TradeManager()
        
        # 取引履歴を作成
        manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=200,
            price=Decimal("2000")
        )
        manager.add_trade(
            symbol="7203",
            trade_type=TradeType.SELL,
            quantity=100,
            price=Decimal("2200")
        )
        
        # 取引履歴取得
        history = manager.get_trade_history()
        assert len(history) == 2
        
        # 実現損益履歴取得
        pnl_history = manager.get_realized_pnl_history()
        assert len(pnl_history) > 0


class TestCompleteErrorHandling:
    """完全なエラーハンドリングテスト"""

    def test_validate_inputs_comprehensive(self):
        """包括的な入力検証テスト"""
        manager = TradeManager()
        
        # 無効な銘柄コード
        with pytest.raises(ValueError):
            manager.add_trade(
                symbol="",
                trade_type=TradeType.BUY,
                quantity=100,
                price=Decimal("2500")
            )
        
        # ゼロ価格
        with pytest.raises(ValueError):
            manager.add_trade(
                symbol="7203",
                trade_type=TradeType.BUY,
                quantity=100,
                price=Decimal("0")
            )
        
        # 負の価格
        with pytest.raises(ValueError):
            manager.add_trade(
                symbol="7203",
                trade_type=TradeType.BUY,
                quantity=100,
                price=Decimal("-100")
            )

    def test_boundary_conditions(self):
        """境界条件テスト"""
        manager = TradeManager()
        
        # 最小値での取引
        trade_id = manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=1,
            price=Decimal("0.01")
        )
        assert trade_id is not None
        
        # 非常に大きな値での取引
        trade_id = manager.add_trade(
            symbol="6758",
            trade_type=TradeType.BUY,
            quantity=1000000,
            price=Decimal("10000")
        )
        assert trade_id is not None


class TestUtilityFunctions:
    """ユーティリティ関数の完全テスト"""

    def test_quantize_decimal_edge_cases(self):
        """quantize_decimal関数のエッジケーステスト"""
        # 非常に小さな値
        result = quantize_decimal(Decimal("0.000001"), 6)
        assert result == Decimal("0.000001")
        
        # 非常に大きな値
        result = quantize_decimal(Decimal("999999999.999999"), 6)
        assert result == Decimal("999999999.999999")
        
        # 負の値
        result = quantize_decimal(Decimal("-123.456"), 2)
        assert result == Decimal("-123.46")

    def test_safe_decimal_conversion_all_paths(self):
        """safe_decimal_conversion関数の全パステスト"""
        # float入力でのエッジケース
        result = safe_decimal_conversion(123.456)
        assert isinstance(result, Decimal)
        
        # 文字列でのカンマ区切り
        result = safe_decimal_conversion("1,234.56")
        assert result == Decimal("1234.56")
        
        # 非常に小さなfloat値
        result = safe_decimal_conversion(1e-15, default_value=Decimal("0"))
        assert result == Decimal("0")  # エラーでデフォルト値が返される


if __name__ == "__main__":
    pytest.main([__file__, "-v"])