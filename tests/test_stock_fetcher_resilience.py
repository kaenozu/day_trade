"""
StockFetcherの耐障害性強化機能のテスト
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
import requests.exceptions as req_exc
from tenacity import RetryError

from src.day_trade.data.stock_fetcher import StockFetcher, DataNotFoundError
from src.day_trade.utils.exceptions import NetworkError


@pytest.fixture
def stock_fetcher():
    """テスト用StockFetcherインスタンス"""
    return StockFetcher(retry_count=3, retry_delay=0.1)


class TestTenacityRetryMechanism:
    """tenacityリトライ機構のテスト"""

    def test_resilient_api_call_decorator(self, stock_fetcher):
        """resilient_api_callデコレータの基本動作テスト"""
        call_count = 0

        @stock_fetcher.resilient_api_call(max_attempts=3, min_wait=0.1, max_wait=1.0)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise req_exc.ConnectionError("Connection failed")
            return "success"

        # 2回目で成功する
        result = test_function()
        assert result == "success"
        assert call_count == 2

    def test_resilient_api_call_max_attempts(self, stock_fetcher):
        """最大試行回数のテスト"""
        call_count = 0

        @stock_fetcher.resilient_api_call(max_attempts=2, min_wait=0.1, max_wait=1.0)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise req_exc.ConnectionError("Always fails")

        # 最大試行回数まで失敗
        with pytest.raises(req_exc.ConnectionError):
            test_function()
        assert call_count == 2

    def test_non_retryable_error_immediate_failure(self, stock_fetcher):
        """リトライ不可能なエラーでの即座の失敗テスト"""
        call_count = 0

        @stock_fetcher.resilient_api_call(max_attempts=3, min_wait=0.1, max_wait=1.0)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")

        # ValueError はリトライ対象に含まれないため、即座に失敗
        with pytest.raises(ValueError):
            test_function()
        assert call_count == 1


class TestGetCurrentPriceResilience:
    """get_current_priceの耐障害性テスト"""

    @patch('src.day_trade.data.stock_fetcher.yf.Ticker')
    def test_get_current_price_connection_error_retry(self, mock_ticker_class, stock_fetcher):
        """接続エラー時のリトライテスト"""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # 最初の2回は接続エラー、3回目で成功
        mock_ticker.info = Mock(side_effect=[
            req_exc.ConnectionError("Connection failed"),
            req_exc.ConnectionError("Connection failed"),
            {
                "currentPrice": 2500.0,
                "previousClose": 2450.0,
                "volume": 1000000,
                "marketCap": 50000000000
            }
        ])

        result = stock_fetcher.get_current_price("7203")

        assert result is not None
        assert result["current_price"] == 2500.0
        assert mock_ticker.info.call_count == 3

    @patch('src.day_trade.data.stock_fetcher.yf.Ticker')
    def test_get_current_price_timeout_retry(self, mock_ticker_class, stock_fetcher):
        """タイムアウトエラー時のリトライテスト"""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # 最初の1回はタイムアウト、2回目で成功
        mock_ticker.info = Mock(side_effect=[
            req_exc.Timeout("Request timeout"),
            {
                "currentPrice": 1800.0,
                "previousClose": 1750.0,
                "volume": 500000
            }
        ])

        result = stock_fetcher.get_current_price("8306")

        assert result is not None
        assert result["current_price"] == 1800.0
        assert mock_ticker.info.call_count == 2

    @patch('src.day_trade.data.stock_fetcher.yf.Ticker')
    def test_get_current_price_http_error_retry(self, mock_ticker_class, stock_fetcher):
        """HTTPエラー時のリトライテスト"""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # HTTPエラー（500）をシミュレート
        http_error = req_exc.HTTPError("500 Server Error")
        mock_response = Mock()
        mock_response.status_code = 500
        http_error.response = mock_response

        mock_ticker.info = Mock(side_effect=[
            http_error,
            {
                "currentPrice": 15000.0,
                "previousClose": 14800.0,
                "volume": 200000
            }
        ])

        result = stock_fetcher.get_current_price("9984")

        assert result is not None
        assert result["current_price"] == 15000.0
        assert mock_ticker.info.call_count == 2

    @patch('src.day_trade.data.stock_fetcher.yf.Ticker')
    def test_get_current_price_max_retries_exceeded(self, mock_ticker_class, stock_fetcher):
        """最大リトライ回数超過時のテスト"""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # 常に接続エラーで失敗
        mock_ticker.info = Mock(side_effect=req_exc.ConnectionError("Always fails"))

        with pytest.raises(req_exc.ConnectionError):
            stock_fetcher.get_current_price("7203")

        # 最大リトライ回数（5回）まで試行されることを確認
        assert mock_ticker.info.call_count == 5


class TestGetHistoricalDataResilience:
    """get_historical_dataの耐障害性テスト"""

    @patch('src.day_trade.data.stock_fetcher.yf.Ticker')
    def test_get_historical_data_network_error_retry(self, mock_ticker_class, stock_fetcher):
        """ネットワークエラー時のリトライテスト"""
        import pandas as pd

        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # サンプルデータフレーム
        sample_df = pd.DataFrame({
            'Open': [2500.0, 2520.0],
            'High': [2550.0, 2570.0],
            'Low': [2480.0, 2500.0],
            'Close': [2520.0, 2540.0],
            'Volume': [1000000, 1100000]
        })

        # 最初の1回はネットワークエラー、2回目で成功
        mock_ticker.history = Mock(side_effect=[
            NetworkError("Network error"),
            sample_df
        ])

        result = stock_fetcher.get_historical_data("7203", period="5d")

        assert result is not None
        assert len(result) == 2
        assert mock_ticker.history.call_count == 2

    @patch('src.day_trade.data.stock_fetcher.yf.Ticker')
    def test_get_historical_data_empty_response_handling(self, mock_ticker_class, stock_fetcher):
        """空のレスポンス処理テスト"""
        import pandas as pd

        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # 空のDataFrameを返す
        empty_df = pd.DataFrame()
        mock_ticker.history = Mock(return_value=empty_df)

        with pytest.raises(DataNotFoundError):
            stock_fetcher.get_historical_data("INVALID")


class TestGetCompanyInfoResilience:
    """get_company_infoの耐障害性テスト"""

    @patch('src.day_trade.data.stock_fetcher.yf.Ticker')
    def test_get_company_info_retry_success(self, mock_ticker_class, stock_fetcher):
        """企業情報取得のリトライ成功テスト"""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # 最初は接続エラー、2回目で成功
        mock_ticker.info = Mock(side_effect=[
            req_exc.ConnectionError("Connection failed"),
            {
                "longName": "Toyota Motor Corporation",
                "sector": "Consumer Cyclical",
                "industry": "Auto Manufacturers",
                "marketCap": 200000000000,
                "website": "https://www.toyota.com",
                "country": "Japan"
            }
        ])

        result = stock_fetcher.get_company_info("7203")

        assert result is not None
        assert result["name"] == "Toyota Motor Corporation"
        assert result["sector"] == "Consumer Cyclical"
        assert mock_ticker.info.call_count == 2


class TestRetryConfiguration:
    """リトライ設定のテスト"""

    def test_custom_retry_parameters(self):
        """カスタムリトライパラメータのテスト"""
        # カスタム設定でStockFetcherを作成
        fetcher = StockFetcher(retry_count=5, retry_delay=0.5)

        assert fetcher.retry_count == 5
        assert fetcher.retry_delay == 0.5

    def test_retry_decorator_parameter_override(self, stock_fetcher):
        """リトライデコレータのパラメータオーバーライドテスト"""
        call_count = 0

        @stock_fetcher.resilient_api_call(max_attempts=2, min_wait=0.1, max_wait=1.0)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise req_exc.ConnectionError("Always fails")

        with pytest.raises(req_exc.ConnectionError):
            test_function()

        # max_attemptsで指定した2回まで試行
        assert call_count == 2


class TestErrorClassification:
    """エラー分類のテスト"""

    def test_retryable_error_detection(self, stock_fetcher):
        """リトライ可能エラーの検出テスト"""
        # リトライ可能なエラー
        assert stock_fetcher._is_retryable_error(req_exc.ConnectionError("Connection failed"))
        assert stock_fetcher._is_retryable_error(req_exc.Timeout("Timeout"))

        # HTTPエラー（500番台）
        http_error = req_exc.HTTPError("500 Server Error")
        mock_response = Mock()
        mock_response.status_code = 500
        http_error.response = mock_response
        assert stock_fetcher._is_retryable_error(http_error)

        # リトライ不可能なエラー
        assert not stock_fetcher._is_retryable_error(ValueError("Invalid value"))
        assert not stock_fetcher._is_retryable_error(KeyError("Missing key"))

    def test_non_retryable_http_errors(self, stock_fetcher):
        """リトライ不可能なHTTPエラーのテスト"""
        # 404エラー（リトライ不可能）
        http_error_404 = req_exc.HTTPError("404 Not Found")
        mock_response = Mock()
        mock_response.status_code = 404
        http_error_404.response = mock_response
        assert not stock_fetcher._is_retryable_error(http_error_404)

        # 401エラー（リトライ不可能）
        http_error_401 = req_exc.HTTPError("401 Unauthorized")
        mock_response = Mock()
        mock_response.status_code = 401
        http_error_401.response = mock_response
        assert not stock_fetcher._is_retryable_error(http_error_401)


class TestConcurrentRetryBehavior:
    """並行処理でのリトライ動作テスト"""

    @patch('src.day_trade.data.stock_fetcher.yf.Ticker')
    def test_get_realtime_data_partial_failures(self, mock_ticker_class, stock_fetcher):
        """リアルタイムデータ取得での部分的失敗テスト"""
        def create_mock_ticker(symbol):
            mock_ticker = Mock()
            if symbol == "7203.T":
                # 成功ケース
                mock_ticker.info = {
                    "currentPrice": 2500.0,
                    "previousClose": 2450.0,
                    "volume": 1000000
                }
            elif symbol == "8306.T":
                # リトライ後成功ケース
                mock_ticker.info = Mock(side_effect=[
                    req_exc.ConnectionError("Connection failed"),
                    {
                        "currentPrice": 800.0,
                        "previousClose": 790.0,
                        "volume": 2000000
                    }
                ])
            else:
                # 完全失敗ケース
                mock_ticker.info = Mock(side_effect=req_exc.ConnectionError("Always fails"))

            return mock_ticker

        mock_ticker_class.side_effect = create_mock_ticker

        codes = ["7203", "8306", "9999"]  # 9999は存在しない銘柄
        result = stock_fetcher.get_realtime_data(codes)

        # 成功した銘柄のみ結果に含まれる
        assert len(result) >= 1  # 少なくとも7203は成功するはず
        if "7203" in result:
            assert result["7203"]["current_price"] == 2500.0
