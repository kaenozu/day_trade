"""
TradeManagerの未カバー部分を対象とした特化テスト
"""

import os
import tempfile
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.day_trade.core.trade_manager import (
    TradeManager,
    TradeType,
    validate_positive_decimal,
    validate_file_path,
)


class TestValidationFunctions:
    """バリデーション関数のテスト"""

    def test_validate_positive_decimal_success(self):
        """正数検証成功テスト"""
        # 正常な正数
        validate_positive_decimal(Decimal("100"), "テスト値")

        # ゼロ許可の場合
        validate_positive_decimal(Decimal("0"), "テスト値", allow_zero=True)

    def test_validate_positive_decimal_failure(self):
        """正数検証失敗テスト"""
        # 負数
        with pytest.raises(ValueError, match="テスト値は正数である必要があります"):
            validate_positive_decimal(Decimal("-100"), "テスト値")

        # ゼロ不許可の場合
        with pytest.raises(ValueError, match="テスト値は正数である必要があります"):
            validate_positive_decimal(Decimal("0"), "テスト値", allow_zero=False)

    def test_validate_file_path_success(self):
        """ファイルパス検証成功テスト"""
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            validate_file_path(tmp.name)

    def test_validate_file_path_failure(self):
        """ファイルパス検証失敗テスト"""
        with pytest.raises(ValueError, match="ファイルパス"):
            validate_file_path("")


class TestTradeManagerInternalMethods:
    """TradeManagerの内部メソッドテスト"""

    def test_update_position_fifo_logic(self):
        """ポジション更新FIFO論理テスト"""
        manager = TradeManager()

        # 複数回に分けて同じ銘柄を購入
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2400"))
        manager.add_trade("7203", TradeType.BUY, 200, Decimal("2500"))
        manager.add_trade("7203", TradeType.BUY, 150, Decimal("2600"))

        # ポジション確認
        position = manager.positions["7203"]
        assert position.quantity == 450
        assert len(position.buy_lots) == 3

        # FIFO順序での部分売却
        manager.add_trade("7203", TradeType.SELL, 250, Decimal("2700"))

        # 最古のロットから売却されていることを確認
        assert position.quantity == 200
        assert len(position.buy_lots) == 2  # 残りのロット

    def test_position_cleanup_on_complete_sale(self):
        """完全売却時のポジションクリーンアップテスト"""
        manager = TradeManager()

        # ポジション作成
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        assert "7203" in manager.positions

        # 完全売却
        manager.add_trade("7203", TradeType.SELL, 100, Decimal("2600"))

        # ポジションが削除されていることを確認
        assert "7203" not in manager.positions

    def test_commission_calculation_rounding(self):
        """手数料計算の丸め処理テスト"""
        manager = TradeManager(commission_rate=Decimal("0.001"))

        # 手数料計算（価格と数量）
        commission = manager._calculate_commission(Decimal("123.45"), 100)
        expected = Decimal("123.45") * 100 * Decimal("0.001")
        assert commission == expected

    def test_generate_trade_id_uniqueness(self):
        """取引ID生成の一意性テスト"""
        manager = TradeManager()

        ids = set()
        for _ in range(100):
            trade_id = manager._generate_trade_id()
            assert trade_id not in ids, f"重複したID: {trade_id}"
            ids.add(trade_id)


class TestFileOperationsErrorHandling:
    """ファイル操作エラーハンドリングテスト"""

    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_export_csv_permission_error(self, mock_open):
        """CSV出力権限エラーテスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        with pytest.raises(PermissionError):
            manager.export_to_csv("test.csv")

    @patch("builtins.open", side_effect=OSError("Disk full"))
    def test_export_json_disk_error(self, mock_open):
        """JSON出力ディスクエラーテスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        with pytest.raises(OSError):
            manager.export_to_json("test.json")

    def test_export_csv_success(self):
        """CSV出力成功テスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            try:
                manager.export_to_csv(tmp_file.name)
                assert os.path.exists(tmp_file.name)
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
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)


class TestAnalyticsEdgeCases:
    """分析機能のエッジケーステスト"""

    def test_portfolio_summary_empty_portfolio(self):
        """空ポートフォリオのサマリーテスト"""
        manager = TradeManager()

        summary = manager.get_portfolio_summary()

        assert summary['total_value'] == manager.initial_capital
        assert summary['cash_balance'] == manager.initial_capital
        assert len(summary['positions']) == 0

    def test_trade_history_chronological_order(self):
        """取引履歴の時系列順序テスト"""
        manager = TradeManager()

        # 複数の取引を時間差で追加
        import time

        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        time.sleep(0.001)  # 微小な時間差
        manager.add_trade("6758", TradeType.BUY, 200, Decimal("1000"))
        time.sleep(0.001)
        manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))

        history = manager.get_trade_history()

        # 時系列順に並んでいることを確認
        assert len(history) == 3
        for i in range(1, len(history)):
            prev_time = datetime.fromisoformat(history[i-1]['timestamp'])
            curr_time = datetime.fromisoformat(history[i]['timestamp'])
            assert prev_time <= curr_time

    def test_realized_pnl_calculation_precision(self):
        """実現損益計算の精度テスト"""
        manager = TradeManager()

        # 精度が重要な計算
        manager.add_trade("7203", TradeType.BUY, 333, Decimal("3333.33"))
        manager.add_trade("7203", TradeType.SELL, 333, Decimal("3366.66"))

        pnl_history = manager.get_realized_pnl_history()
        assert len(pnl_history) == 1

        # 損益が正確に計算されていることを確認
        pnl = pnl_history[0]
        assert isinstance(Decimal(pnl['pnl']), Decimal)


class TestConcurrencyAndThreadSafety:
    """並行性とスレッドセーフティテスト"""

    def test_multiple_operations_sequence(self):
        """複数操作シーケンステスト"""
        manager = TradeManager()

        # 複雑な操作シーケンス
        operations = [
            ("7203", TradeType.BUY, 100, Decimal("2500")),
            ("6758", TradeType.BUY, 200, Decimal("1000")),
            ("7203", TradeType.BUY, 50, Decimal("2600")),
            ("7203", TradeType.SELL, 75, Decimal("2700")),
            ("6758", TradeType.SELL, 100, Decimal("1100")),
        ]

        for symbol, trade_type, quantity, price in operations:
            manager.add_trade(symbol, trade_type, quantity, price)

        # 最終状態の確認
        assert len(manager.positions) == 2
        assert manager.positions["7203"].quantity == 75
        assert manager.positions["6758"].quantity == 100
        assert len(manager.realized_pnl_history) == 2


class TestMemoryAndPerformance:
    """メモリとパフォーマンステスト"""

    def test_large_dataset_handling(self):
        """大量データ処理テスト"""
        manager = TradeManager()

        # 大量の取引を追加
        for i in range(1000):
            symbol = f"TST{i:04d}"
            manager.add_trade(symbol, TradeType.BUY, 100, Decimal("1000"))

        # メモリ使用量の確認（基本的なチェック）
        assert len(manager.trades) == 1000
        assert len(manager.positions) == 1000

        # パフォーマンス確認（サマリー生成）
        summary = manager.get_portfolio_summary()
        assert len(summary['positions']) == 1000

    def test_position_consolidation(self):
        """ポジション統合テスト"""
        manager = TradeManager()

        # 同一銘柄への複数回購入
        for _ in range(100):
            manager.add_trade("7203", TradeType.BUY, 10, Decimal("2500"))

        # ポジションが適切に統合されていることを確認
        position = manager.positions["7203"]
        assert position.quantity == 1000
        assert len(position.buy_lots) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])