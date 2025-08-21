"""
TradeManagerのファイルI/O機能完全制覇テスト
未カバー行1414-1597を集中攻撃
"""

import csv
import json
import os
import tempfile
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest

from src.day_trade.core.trade_manager import (
    TradeManager,
    TradeType,
    TradeStatus,
)


class TestBatchTradeAddition:
    """一括取引追加機能テスト（行1414-1597）"""
    
    def test_add_trades_batch_empty_data(self):
        """空の取引データ一括追加テスト（行1421-1423）"""
        manager = TradeManager()
        
        # 空のリストを渡す
        result = manager.add_trades_batch([])
        
        # 空のリストが返される
        assert result == []
    
    def test_add_trades_batch_memory_only(self):
        """メモリのみ一括取引追加テスト"""
        manager = TradeManager()
        
        trades_data = [
            {
                "symbol": "7203",
                "trade_type": TradeType.BUY,
                "quantity": 100,
                "price": Decimal("2500"),
                "notes": "バッチ取引1"
            },
            {
                "symbol": "6758", 
                "trade_type": TradeType.BUY,
                "quantity": 200,
                "price": Decimal("1000"),
                "notes": "バッチ取引2"
            }
        ]
        
        # persist_to_db=Falseでバッチ追加
        trade_ids = manager.add_trades_batch(trades_data, persist_to_db=False)
        
        # 2つの取引IDが返される
        assert len(trade_ids) == 2
        assert len(manager.trades) == 2
        assert len(manager.positions) == 2
    
    def test_add_trades_batch_with_backup_restoration(self):
        """バックアップ復元テスト（行1427-1431）"""
        manager = TradeManager()
        
        # 初期取引を追加
        manager.add_trade("EXISTING", TradeType.BUY, 50, Decimal("1500"))
        initial_trade_count = len(manager.trades)
        
        # 不正なデータを含むバッチ（エラーを引き起こす）
        trades_data = [
            {
                "symbol": "7203",
                "trade_type": TradeType.BUY,
                "quantity": 100,
                "price": Decimal("2500")
            },
            {
                "symbol": "",  # 無効な銘柄コード
                "trade_type": TradeType.BUY,
                "quantity": 200,
                "price": Decimal("1000")
            }
        ]
        
        # バッチ追加でエラーが発生する
        with pytest.raises(ValueError):
            manager.add_trades_batch(trades_data, persist_to_db=False)
        
        # 初期状態に復元されている
        assert len(manager.trades) == initial_trade_count
    
    def test_add_trades_batch_with_commission_calculation(self):
        """手数料自動計算テスト（行1448-1449）"""
        manager = TradeManager(commission_rate=Decimal("0.001"))
        
        trades_data = [
            {
                "symbol": "7203",
                "trade_type": TradeType.BUY,
                "quantity": 100,
                "price": Decimal("2500")
                # commissionを省略 -> 自動計算される
            }
        ]
        
        trade_ids = manager.add_trades_batch(trades_data, persist_to_db=False)
        
        # 手数料が自動計算されている
        trade = manager.trades[0]
        expected_commission = Decimal("2500") * 100 * Decimal("0.001")
        assert trade.commission == expected_commission


class TestCsvExportPaths:
    """CSV出力機能の完全パステスト"""
    
    def test_export_to_csv_trades_complete(self):
        """取引データCSV出力の完全テスト"""
        manager = TradeManager()
        
        # 複数の取引を追加
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"), notes="テスト取引1")
        manager.add_trade("6758", TradeType.BUY, 200, Decimal("1000"), notes="テスト取引2")
        manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"), notes="テスト取引3")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            try:
                # 取引データのCSV出力
                manager.export_to_csv(tmp.name, "trades")
                
                # ファイルの存在確認
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
                
                # CSV内容の詳細確認
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                    # 3つの取引が出力されている
                    assert len(rows) == 3
                    
                    # ヘッダーの確認
                    expected_headers = ['id', 'symbol', 'trade_type', 'quantity', 'price', 'timestamp', 'commission', 'status', 'notes']
                    assert all(header in reader.fieldnames for header in expected_headers)
                    
                    # 最初の行の詳細確認
                    first_row = rows[0]
                    assert first_row['symbol'] == '7203'
                    assert first_row['trade_type'] == 'BUY'
                    assert first_row['quantity'] == '100'
                    assert first_row['notes'] == 'テスト取引1'
                    
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
    
    def test_export_to_csv_positions_complete(self):
        """ポジションデータCSV出力の完全テスト"""
        manager = TradeManager()
        
        # ポジションを作成
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        manager.add_trade("6758", TradeType.BUY, 200, Decimal("1000"))
        
        # 現在価格を設定（update_current_pricesを使用）
        manager.update_current_prices({"7203": Decimal("2600"), "6758": Decimal("1100")})
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            try:
                # ポジションデータのCSV出力
                manager.export_to_csv(tmp.name, "positions")
                
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
                
                # CSV内容の確認
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                    # 2つのポジションが出力されている
                    assert len(rows) == 2
                    
                    # 必要なフィールドの確認
                    expected_headers = ['symbol', 'quantity', 'average_price', 'total_cost', 'current_price', 'market_value', 'unrealized_pnl']
                    assert all(header in reader.fieldnames for header in expected_headers)
                    
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
    
    def test_export_to_csv_realized_pnl_complete(self):
        """実現損益データCSV出力の完全テスト"""
        manager = TradeManager()
        
        # 実現損益を生成する取引
        manager.add_trade("7203", TradeType.BUY, 200, Decimal("2500"))
        manager.add_trade("7203", TradeType.SELL, 100, Decimal("2600"))
        manager.add_trade("7203", TradeType.SELL, 50, Decimal("2700"))
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            try:
                # 実現損益データのCSV出力
                manager.export_to_csv(tmp.name, "realized_pnl")
                
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
                
                # CSV内容の確認
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                    # 実現損益レコードが出力されている
                    assert len(rows) >= 1
                    
                    # 必要なフィールドの確認
                    expected_headers = ['symbol', 'quantity', 'buy_price', 'sell_price', 'pnl']
                    for header in expected_headers:
                        assert header in reader.fieldnames, f"Header '{header}' not found"
                    
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)


class TestJsonExportPaths:
    """JSON出力機能の完全パステスト"""
    
    def test_export_to_json_complete_structure(self):
        """JSON出力の完全構造テスト"""
        manager = TradeManager()
        
        # 複雑なデータ構造を作成
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"), notes="JSON取引1")
        manager.add_trade("6758", TradeType.BUY, 200, Decimal("1000"), notes="JSON取引2")
        manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"), notes="JSON取引3")
        
        # 現在価格を設定
        manager.update_current_prices({"7203": Decimal("2650"), "6758": Decimal("1050")})
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            try:
                # JSON出力
                manager.export_to_json(tmp.name)
                
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
                
                # JSON構造の詳細確認
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # トップレベル構造の確認
                    assert 'trades' in data
                    assert 'positions' in data
                    assert 'realized_pnl_history' in data
                    assert 'metadata' in data
                    
                    # 取引データの確認
                    trades = data['trades']
                    assert len(trades) == 3
                    assert all('id' in trade for trade in trades)
                    assert all('symbol' in trade for trade in trades)
                    assert all('notes' in trade for trade in trades)
                    
                    # ポジションデータの確認
                    positions = data['positions']
                    assert len(positions) == 2
                    
                    # 実現損益データの確認
                    pnl_history = data['realized_pnl_history']
                    assert isinstance(pnl_history, list)
                    
                    # メタデータの確認
                    metadata = data['metadata']
                    assert 'export_timestamp' in metadata
                    assert 'total_trades' in metadata
                    assert 'total_positions' in metadata
                    
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)


class TestFileOperationErrorHandling:
    """ファイル操作エラーハンドリング完全テスト"""
    
    @patch("builtins.open")
    def test_csv_export_write_error(self, mock_open):
        """CSV出力書き込みエラーテスト"""
        # ファイル書き込みエラーをシミュレート
        mock_open.side_effect = IOError("Write failed")
        
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        
        with pytest.raises(IOError):
            manager.export_to_csv("test.csv", "trades")
    
    @patch("builtins.open")
    def test_json_export_write_error(self, mock_open):
        """JSON出力書き込みエラーテスト"""
        # ファイル書き込みエラーをシミュレート
        mock_open.side_effect = IOError("Write failed")
        
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        
        with pytest.raises(IOError):
            manager.export_to_json("test.json")
    
    def test_csv_export_invalid_directory(self):
        """CSV出力無効ディレクトリテスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        
        # 存在しないディレクトリへの出力
        with pytest.raises(Exception):
            manager.export_to_csv("/nonexistent/directory/test.csv", "trades")
    
    def test_json_export_invalid_directory(self):
        """JSON出力無効ディレクトリテスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        
        # 存在しないディレクトリへの出力
        with pytest.raises(Exception):
            manager.export_to_json("/nonexistent/directory/test.json")


class TestUpdateCurrentPricesMethod:
    """update_current_pricesメソッドテスト"""
    
    def test_update_current_prices_multiple_symbols(self):
        """複数銘柄の現在価格一括更新テスト"""
        manager = TradeManager()
        
        # 複数銘柄のポジション作成
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        manager.add_trade("6758", TradeType.BUY, 200, Decimal("1000"))
        manager.add_trade("4063", TradeType.BUY, 150, Decimal("3000"))
        
        # 価格辞書で一括更新
        price_dict = {
            "7203": Decimal("2600"),
            "6758": Decimal("1100"),
            "4063": Decimal("3200")
        }
        
        manager.update_current_prices(price_dict)
        
        # 各ポジションの価格が更新されていることを確認
        assert manager.positions["7203"].current_price == Decimal("2600")
        assert manager.positions["6758"].current_price == Decimal("1100")
        assert manager.positions["4063"].current_price == Decimal("3200")
    
    def test_update_current_prices_partial_symbols(self):
        """部分的な銘柄価格更新テスト"""
        manager = TradeManager()
        
        # 複数銘柄のポジション作成
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        manager.add_trade("6758", TradeType.BUY, 200, Decimal("1000"))
        
        # 一部の銘柄のみ価格更新
        price_dict = {
            "7203": Decimal("2600"),
            # 6758は更新しない
            "9999": Decimal("5000")  # 存在しない銘柄
        }
        
        manager.update_current_prices(price_dict)
        
        # 存在する銘柄のみ更新される
        assert manager.positions["7203"].current_price == Decimal("2600")
        assert manager.positions["6758"].current_price == Decimal("0")  # デフォルト値のまま


if __name__ == "__main__":
    pytest.main([__file__, "-v"])