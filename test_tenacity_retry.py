"""
tenacityリトライ機能のテスト
Issue #185: API耐障害性強化のテスト
"""

import time
import unittest
from unittest.mock import Mock, patch

import requests

from src.day_trade.data.stock_fetcher import StockFetcher
from src.day_trade.utils.exceptions import NetworkError


class TestTenacityRetry(unittest.TestCase):
    """tenacityベースのリトライ機能テスト"""

    def setUp(self):
        """テストセットアップ"""
        self.fetcher = StockFetcher(
            retry_count=3,
            retry_delay=0.1,  # テスト用に短縮
        )

    def test_successful_request_no_retry(self):
        """成功時はリトライしないことをテスト"""
        # モックデータを作成
        mock_ticker = Mock()
        mock_ticker.history.return_value = Mock(empty=False)

        with patch.object(self.fetcher, "_create_ticker", return_value=mock_ticker):
            # リクエスト成功
            self.fetcher.get_current_price("7203")

        # 統計を確認
        stats = self.fetcher.get_retry_stats()
        self.assertEqual(stats["total_requests"], 1)
        self.assertEqual(stats["successful_requests"], 1)
        self.assertEqual(stats["failed_requests"], 0)
        self.assertEqual(stats["total_retries"], 0)

    def test_retry_on_network_error(self):
        """ネットワークエラー時のリトライテスト"""
        mock_ticker = Mock()

        # 最初の2回は例外、3回目は成功
        mock_ticker.history.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            requests.exceptions.Timeout("Request timeout"),
            Mock(empty=False),  # 3回目は成功
        ]

        with patch.object(self.fetcher, "_create_ticker", return_value=mock_ticker):
            # リトライが発生して最終的に成功
            self.fetcher.get_current_price("7203")

        # 統計を確認
        stats = self.fetcher.get_retry_stats()
        self.assertEqual(stats["total_requests"], 1)
        self.assertEqual(stats["successful_requests"], 1)
        self.assertEqual(stats["total_retries"], 2)  # 2回リトライ

    def test_max_retries_exceeded(self):
        """最大リトライ回数を超えた場合のテスト"""
        mock_ticker = Mock()

        # すべて失敗
        mock_ticker.history.side_effect = requests.exceptions.ConnectionError(
            "Connection failed"
        )

        with patch.object(
            self.fetcher, "_create_ticker", return_value=mock_ticker
        ), self.assertRaises(NetworkError):
            self.fetcher.get_current_price("7203")

        # 統計を確認
        stats = self.fetcher.get_retry_stats()
        self.assertEqual(stats["total_requests"], 1)
        self.assertEqual(stats["successful_requests"], 0)
        self.assertEqual(stats["failed_requests"], 1)
        self.assertEqual(stats["total_retries"], 3)  # 設定した最大リトライ回数

    def test_non_retryable_error(self):
        """リトライ不可能なエラーのテスト"""
        mock_ticker = Mock()

        # 401エラー（認証エラー）はリトライしない
        http_error = requests.exceptions.HTTPError("401 Unauthorized")
        http_error.response = Mock()
        http_error.response.status_code = 401
        mock_ticker.history.side_effect = http_error

        with patch.object(
            self.fetcher, "_create_ticker", return_value=mock_ticker
        ), self.assertRaises(NetworkError):
            self.fetcher.get_current_price("7203")

        # 統計を確認（リトライされていない）
        stats = self.fetcher.get_retry_stats()
        self.assertEqual(stats["total_retries"], 0)

    def test_exponential_backoff(self):
        """指数バックオフのテスト"""
        mock_ticker = Mock()
        mock_ticker.history.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            requests.exceptions.ConnectionError("Connection failed"),
            Mock(empty=False),  # 3回目は成功
        ]

        start_time = time.time()

        with patch.object(self.fetcher, "_create_ticker", return_value=mock_ticker):
            self.fetcher.get_current_price("7203")

        elapsed_time = time.time() - start_time

        # 指数バックオフにより、少なくとも0.1 + 0.2 = 0.3秒以上かかるはず
        self.assertGreater(elapsed_time, 0.3)

    def test_retry_statistics_accuracy(self):
        """リトライ統計の正確性テスト"""
        # 複数回のリクエストで統計の正確性を確認
        mock_ticker = Mock()

        # 1回目: 成功
        # 2回目: 1回リトライして成功
        # 3回目: 失敗
        call_count = 0

        def side_effect_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Mock(empty=False)  # 成功
            elif call_count == 2:
                raise requests.exceptions.ConnectionError("Connection failed")
            elif call_count == 3:
                return Mock(empty=False)  # リトライ後成功
            else:
                raise requests.exceptions.ConnectionError("Connection failed")

        mock_ticker.history.side_effect = side_effect_func

        with patch.object(self.fetcher, "_create_ticker", return_value=mock_ticker):
            # 1回目: 成功
            self.fetcher.get_current_price("7203")

            # 2回目: リトライして成功
            self.fetcher.get_current_price("9984")

            # 3回目: 失敗
            with self.assertRaises(NetworkError):
                mock_ticker.history.side_effect = requests.exceptions.ConnectionError(
                    "Connection failed"
                )
                self.fetcher.get_current_price("8306")

        # 統計を確認
        stats = self.fetcher.get_retry_stats()
        self.assertEqual(stats["total_requests"], 3)
        self.assertEqual(stats["successful_requests"], 2)
        self.assertEqual(stats["failed_requests"], 1)
        self.assertGreater(stats["total_retries"], 0)

        # 成功率の確認
        self.assertAlmostEqual(stats["success_rate"], 2 / 3, places=2)


if __name__ == "__main__":
    # テスト実行
    unittest.main()
