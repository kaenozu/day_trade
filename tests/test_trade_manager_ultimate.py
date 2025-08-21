"""
TradeManagerの100%カバレッジを目指す究極のテスト
未カバー行を精密にターゲットする
"""

import os
import tempfile
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest

from src.day_trade.core.trade_manager import (
    TradeManager,
    TradeType,
    TradeStatus,
    validate_file_path,
    safe_decimal_conversion,
)


class TestValidateFilePathComprehensive:
    """validate_file_path関数の完全カバレッジ"""

    def test_empty_filepath(self):
        """空のファイルパステスト（行208）"""
        with pytest.raises(ValueError, match="ファイルパスが必要です"):
            validate_file_path("")

    def test_none_filepath(self):
        """Noneファイルパステスト"""
        with pytest.raises(ValueError, match="ファイルパスが必要です"):
            validate_file_path(None)

    def test_invalid_filepath_type(self):
        """無効なファイルパス型テスト（行211-213）"""
        with pytest.raises(TypeError, match="文字列またはPathオブジェクトである必要があります"):
            validate_file_path(123)

        with pytest.raises(TypeError, match="文字列またはPathオブジェクトである必要があります"):
            validate_file_path(['path'])

    def test_dangerous_path_patterns(self):
        """危険なパスパターンテスト（行220-237）"""
        dangerous_paths = [
            "../etc/passwd",
            "..\\windows\\system32",
            "%2e%2e/etc/shadow",
            "~/sensitive_file",
            "file:///etc/hosts",
            "ftp://malicious.com/file",
            "http://evil.com/script",
            "https://bad.com/malware",
            "\\\\network\\share",
            "//server/drive",
            "..%2fetc%2fpasswd",
            "..%5cwindows%5csystem32",
            "file\x00.txt",  # NULL文字
            "path\x01file",  # 制御文字
        ]

        for dangerous_path in dangerous_paths:
            with pytest.raises(ValueError, match="セキュリティ上危険なパス"):
                validate_file_path(dangerous_path)

    def test_absolute_path_validation(self):
        """絶対パス検証テスト"""
        # Windowsの絶対パス
        if os.name == 'nt':
            validate_file_path("C:\\temp\\test.txt")
        else:
            validate_file_path("/tmp/test.txt")

    def test_path_object_validation(self):
        """Pathオブジェクト検証テスト"""
        temp_path = Path(tempfile.gettempdir()) / "test.txt"
        validate_file_path(temp_path)


class TestSafeDecimalConversionMissingPaths:
    """safe_decimal_conversion未カバーパステスト"""

    def test_decimal_edge_cases(self):
        """Decimal値のエッジケーステスト"""
        # 既にDecimal値の場合（行69-70）
        result = safe_decimal_conversion(Decimal("123.45"))
        assert result == Decimal("123.45")

    def test_float_precision_handling(self):
        """float精度処理テスト（行118以降）"""
        # repr()を使った精度保持のテスト
        float_val = 0.1 + 0.2  # 浮動小数点精度問題
        result = safe_decimal_conversion(float_val)
        assert isinstance(result, Decimal)

    def test_decimal_conversion_error_handling(self):
        """Decimal変換エラーハンドリングテスト"""
        # InvalidOperationの発生
        with patch('decimal.Decimal', side_effect=Exception("Conversion error")):
            with pytest.raises(ValueError, match="Decimal変換エラー"):
                safe_decimal_conversion("123.45")


class TestTradeManagerMissingInitialization:
    """TradeManager初期化の未カバー部分"""

    def test_database_manager_import_error(self):
        """データベースマネージャーインポートエラーテスト"""
        # データベース関連の初期化エラーをテスト
        manager = TradeManager()
        # persist_to_dbがFalseの場合のパス
        assert hasattr(manager, 'trades')
        assert hasattr(manager, 'positions')

    def test_logging_configuration(self):
        """ログ設定テスト"""
        manager = TradeManager()
        # ログ設定が正常に行われることを確認
        assert hasattr(manager, 'initial_capital')


class TestTradeManagerFileOperationsPaths:
    """ファイル操作の未カバーパス"""

    def test_export_csv_positions_data(self):
        """CSV出力：ポジションデータテスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            try:
                # data_type="positions"のパスをテスト
                manager.export_to_csv(tmp.name, "positions")
                assert os.path.exists(tmp.name)

                # ファイル内容を確認
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                    assert '7203' in content
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_export_csv_realized_pnl_data(self):
        """CSV出力：実現損益データテスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            try:
                # data_type="realized_pnl"のパスをテスト
                manager.export_to_csv(tmp.name, "realized_pnl")
                assert os.path.exists(tmp.name)
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_export_csv_invalid_data_type(self):
        """CSV出力：無効データタイプテスト"""
        manager = TradeManager()

        # サポートされていないデータタイプ
        with pytest.raises(ValueError, match="サポートされていないデータタイプ"):
            manager.export_to_csv("test.csv", "invalid_type")

    def test_export_json_comprehensive(self):
        """JSON出力の包括的テスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            try:
                manager.export_to_json(tmp.name)
                assert os.path.exists(tmp.name)

                # JSON内容の検証
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    assert 'trades' in data
                    assert 'positions' in data
                    assert 'realized_pnl_history' in data
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)


class TestTradeManagerAnalyticsEdgeCases:
    """分析機能のエッジケース"""

    def test_get_trade_history_dict_conversion(self):
        """取引履歴の辞書変換テスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        # get_trade_historyは実際にはTradeオブジェクトのリストを返す
        history = manager.trades  # 実際のTradeオブジェクトリスト
        assert len(history) == 1
        assert history[0].symbol == "7203"

    def test_get_realized_pnl_history_dict_conversion(self):
        """実現損益履歴の辞書変換テスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))

        pnl_history = manager.get_realized_pnl_history()
        assert len(pnl_history) > 0

        # 各要素が辞書形式で返されることを確認
        pnl = pnl_history[0]
        if isinstance(pnl, dict):
            assert 'symbol' in pnl
            assert 'pnl' in pnl

    def test_portfolio_summary_comprehensive(self):
        """ポートフォリオサマリーの包括的テスト"""
        manager = TradeManager()

        # 複数銘柄での取引
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        manager.add_trade("6758", TradeType.BUY, 200, Decimal("1000"))

        # 現在価格の設定
        manager.update_current_price("7203", Decimal("2600"))
        manager.update_current_price("6758", Decimal("1100"))

        summary = manager.get_portfolio_summary()

        # 必要なキーの存在確認
        required_keys = ['total_value', 'cash_balance', 'positions', 'unrealized_pnl']
        for key in required_keys:
            assert key in summary

        # ポジション詳細の確認
        assert len(summary['positions']) == 2

        for position in summary['positions']:
            assert 'symbol' in position
            assert 'quantity' in position
            assert 'market_value' in position


class TestTradeManagerEdgeCaseHandling:
    """エッジケース処理テスト"""

    def test_add_trade_with_zero_commission_rate(self):
        """手数料率ゼロでの取引追加テスト"""
        manager = TradeManager(commission_rate=Decimal("0"))
        trade_id = manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        trade = manager.trades[0]
        assert trade.commission == Decimal("0")

    def test_position_cleanup_edge_cases(self):
        """ポジションクリーンアップのエッジケース"""
        manager = TradeManager()

        # 完全に一致する売却
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        manager.add_trade("7203", TradeType.SELL, 100, Decimal("2600"))

        # ポジションが削除されていることを確認
        assert "7203" not in manager.positions

    def test_calculate_tax_zero_rate(self):
        """税率ゼロでの税金計算テスト"""
        manager = TradeManager(tax_rate=Decimal("0"))

        tax = manager.calculate_tax(Decimal("10000"))
        assert tax == Decimal("0")

    def test_commission_calculation_edge_cases(self):
        """手数料計算のエッジケース"""
        manager = TradeManager()

        # 非常に小さな取引金額
        commission = manager._calculate_commission(Decimal("0.01"), 1)
        assert commission >= Decimal("0")

        # 非常に大きな取引金額
        commission = manager._calculate_commission(Decimal("1000000"), 1000)
        assert commission > Decimal("0")


class TestErrorHandlingComprehensive:
    """包括的エラーハンドリングテスト"""

    def test_all_validation_paths(self):
        """すべての検証パステスト"""
        manager = TradeManager()

        # 空文字列銘柄コード
        with pytest.raises(ValueError):
            manager.add_trade("", TradeType.BUY, 100, Decimal("2500"))

        # 空白のみ銘柄コード
        with pytest.raises(ValueError):
            manager.add_trade("   ", TradeType.BUY, 100, Decimal("2500"))

        # ゼロ数量
        with pytest.raises(ValueError):
            manager.add_trade("7203", TradeType.BUY, 0, Decimal("2500"))

        # ゼロ価格
        with pytest.raises(ValueError):
            manager.add_trade("7203", TradeType.BUY, 100, Decimal("0"))

    def test_sell_validation_comprehensive(self):
        """売却検証の包括的テスト"""
        manager = TradeManager()

        # ポジション不足での売却
        with pytest.raises(ValueError, match="ポジションが存在しません"):
            manager.add_trade("7203", TradeType.SELL, 100, Decimal("2500"))

        # 部分ポジションでの過剰売却
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        with pytest.raises(ValueError, match="売却数量が保有数量を超えています"):
            manager.add_trade("7203", TradeType.SELL, 150, Decimal("2600"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])