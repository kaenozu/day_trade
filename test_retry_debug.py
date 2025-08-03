"""
リトライ機能のデバッグテスト
"""

import requests
from unittest.mock import Mock, patch
from src.day_trade.data.stock_fetcher import StockFetcher

def test_retry_debug():
    """リトライ機能の動作を確認"""
    fetcher = StockFetcher(retry_count=3, retry_delay=0.1)

    print("=== リトライ機能デバッグ ===")

    # モックティッカーを作成
    mock_ticker = Mock()
    mock_ticker.history.side_effect = requests.exceptions.ConnectionError("Connection failed")

    try:
        with patch.object(fetcher, '_create_ticker', return_value=mock_ticker):
            result = fetcher.get_current_price("7203")
            print(f"結果: {result}")
    except Exception as e:
        print(f"例外: {type(e).__name__}: {e}")

    # 統計を確認
    stats = fetcher.get_retry_stats()
    print(f"統計: {stats}")

    # リトライ可能エラーのテスト
    print(f"ConnectionErrorがリトライ可能: {fetcher._is_retryable_error(requests.exceptions.ConnectionError('test'))}")

if __name__ == "__main__":
    test_retry_debug()
