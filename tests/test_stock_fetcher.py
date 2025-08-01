"""
株価データ取得モジュールのテスト
"""

import sys
import os
import pytest
import time
from datetime import datetime
import pandas as pd
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from day_trade.data.stock_fetcher import (  # noqa: E402
    StockFetcher,
    DataCache,
    cache_with_ttl,
    StockFetcherError,
    InvalidSymbolError,
    DataNotFoundError,
)


class TestDataCache:
    """DataCacheのテストクラス"""

    def test_cache_set_and_get(self):
        """キャッシュの設定と取得のテスト"""
        cache = DataCache(ttl_seconds=60)

        # データを設定
        cache.set("test_key", "test_value")

        # データを取得
        result = cache.get("test_key")
        assert result == "test_value"

    def test_cache_expiry(self):
        """キャッシュの有効期限のテスト"""
        cache = DataCache(ttl_seconds=1)

        # データを設定
        cache.set("test_key", "test_value")

        # すぐに取得（有効）
        result = cache.get("test_key")
        assert result == "test_value"

        # 時間を進めてから取得（無効）
        import time

        time.sleep(1.1)
        result = cache.get("test_key")
        assert result is None

    def test_cache_clear(self):
        """キャッシュクリアのテスト"""
        cache = DataCache(ttl_seconds=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.size() == 2

        cache.clear()
        assert cache.size() == 0
        assert cache.get("key1") is None


class TestStockFetcher:
    """StockFetcherクラスのテスト"""

    @pytest.fixture
    def fetcher(self):
        """StockFetcherインスタンスを作成"""
        return StockFetcher(
            cache_size=2,
            price_cache_ttl=5,
            historical_cache_ttl=10,
            retry_count=2,
            retry_delay=0.1,
        )

    @pytest.fixture
    def mock_ticker_info(self):
        """モックのティッカー情報"""
        return {
            "currentPrice": 2500.0,
            "previousClose": 2480.0,
            "volume": 1000000,
            "longName": "Toyota Motor Corporation",
            "sector": "Consumer Cyclical",
            "industry": "Auto Manufacturers",
            "marketCap": 35000000000000,
        }

    def test_format_symbol(self, fetcher):
        """証券コードのフォーマットテスト"""
        # 市場コードなしの場合
        assert fetcher._format_symbol("7203") == "7203.T"

        # 市場コード指定の場合
        assert fetcher._format_symbol("7203", "O") == "7203.O"

        # すでに市場コードがある場合
        assert fetcher._format_symbol("7203.T") == "7203.T"

    @patch("yfinance.Ticker")
    def test_get_current_price_success(
        self, mock_ticker_class, fetcher, mock_ticker_info
    ):
        """現在価格取得の正常系テスト"""
        # モックの設定
        mock_ticker = Mock()
        mock_ticker.info = mock_ticker_info
        mock_ticker_class.return_value = mock_ticker

        # テスト実行
        result = fetcher.get_current_price("7203")

        # 検証
        assert result is not None
        assert result["symbol"] == "7203.T"
        assert result["current_price"] == 2500.0
        assert result["previous_close"] == 2480.0
        assert result["change"] == 20.0
        assert result["change_percent"] == pytest.approx(0.806, rel=0.01)
        assert result["volume"] == 1000000
        assert isinstance(result["timestamp"], datetime)

    @patch("yfinance.Ticker")
    def test_get_current_price_no_data(self, mock_ticker_class, fetcher):
        """現在価格が取得できない場合のテスト"""
        # モックの設定（価格情報なし）
        mock_ticker = Mock()
        mock_ticker.info = {}
        mock_ticker_class.return_value = mock_ticker

        # テスト実行
        with pytest.raises(DataNotFoundError):
            fetcher.get_current_price("9999")

    @patch("day_trade.data.stock_fetcher.yf.Ticker")
    def test_get_current_price_exception(self, mock_ticker_class, fetcher):
        """例外発生時のテスト"""
        # キャッシュをクリアしてテスト開始
        fetcher.clear_all_caches()

        # モックの設定（例外を発生させる）
        mock_ticker_class.side_effect = Exception("API Error")

        # テスト実行
        with pytest.raises(StockFetcherError):
            fetcher.get_current_price("EXCEPTION_TEST")

    @patch("yfinance.Ticker")
    def test_get_historical_data_success(self, mock_ticker_class, fetcher):
        """ヒストリカルデータ取得の正常系テスト"""
        # モックデータの作成
        dates = pd.date_range(end=datetime.now(), periods=5, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [2480, 2490, 2485, 2495, 2500],
                "High": [2495, 2500, 2495, 2505, 2510],
                "Low": [2475, 2485, 2480, 2490, 2495],
                "Close": [2490, 2485, 2495, 2500, 2505],
                "Volume": [1000000, 1100000, 900000, 1200000, 1050000],
            },
            index=dates,
        )

        # モックの設定
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker

        # テスト実行
        result = fetcher.get_historical_data("7203", period="5d", interval="1d")

        # 検証
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        mock_ticker.history.assert_called_once_with(period="5d", interval="1d")

    @patch("yfinance.Ticker")
    def test_get_historical_data_empty(self, mock_ticker_class, fetcher):
        """空のヒストリカルデータの場合のテスト"""
        # モックの設定（空のDataFrame）
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        # テスト実行
        with pytest.raises(DataNotFoundError):
            fetcher.get_historical_data("9999")

    @patch("yfinance.Ticker")
    def test_get_historical_data_range(self, mock_ticker_class, fetcher):
        """期間指定でのヒストリカルデータ取得テスト"""
        # モックデータの作成
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 5)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [2400, 2410, 2420, 2415, 2425],
                "Close": [2410, 2420, 2415, 2425, 2430],
                "Volume": [1000000] * 5,
            },
            index=dates,
        )

        # モックの設定
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker

        # テスト実行（文字列での日付指定）
        result = fetcher.get_historical_data_range(
            "7203", "2023-01-01", "2023-01-05", interval="1d"
        )

        # 検証
        assert result is not None
        assert len(result) == 5
        mock_ticker.history.assert_called_once()

        # datetimeオブジェクトでの日付指定もテスト
        result2 = fetcher.get_historical_data_range(
            "7203", start_date, end_date, interval="1d"
        )
        assert result2 is not None

    @patch("day_trade.data.stock_fetcher.StockFetcher.get_current_price")
    def test_get_realtime_data(self, mock_get_current_price, fetcher):
        """複数銘柄のリアルタイムデータ取得テスト（並行処理検証含む）"""

        # get_current_price のモック設定
        # 各コードに対して異なるデータを返すように設定
        def mock_side_effect(code):
            time.sleep(0.1)  # 処理時間をシミュレート
            if code == "7203":
                return {
                    "symbol": "7203.T",
                    "current_price": 2500.0,
                    "previous_close": 2480.0,
                    "change": 20.0,
                    "change_percent": 0.806,
                    "volume": 1000000,
                    "timestamp": datetime.now(),
                }
            elif code == "6758":
                return {
                    "symbol": "6758.T",
                    "current_price": 15000.0,
                    "previous_close": 14900.0,
                    "change": 100.0,
                    "change_percent": 0.67,
                    "volume": 500000,
                    "timestamp": datetime.now(),
                }
            elif code == "9984":
                return {
                    "symbol": "9984.T",
                    "current_price": 70000.0,
                    "previous_close": 69500.0,
                    "change": 500.0,
                    "change_percent": 0.72,
                    "volume": 200000,
                    "timestamp": datetime.now(),
                }
            else:
                return None

        mock_get_current_price.side_effect = mock_side_effect

        codes = ["7203", "6758", "9984"]

        # 実行時間の計測
        start_time = time.time()
        result = fetcher.get_realtime_data(codes)
        end_time = time.time()

        # 検証
        assert len(result) == 3
        assert "7203" in result
        assert result["7203"]["current_price"] == 2500.0
        assert "6758" in result
        assert result["6758"]["current_price"] == 15000.0
        assert "9984" in result
        assert result["9984"]["current_price"] == 70000.0

        # get_current_price が各コードに対して1回ずつ呼び出されたことを確認
        assert mock_get_current_price.call_count == len(codes)

        # 並行処理による時間短縮の検証（厳密な時間ではないが、直列処理より短いことを期待）
        # 各呼び出しに0.1秒かかるとすると、直列では 0.3秒以上かかるが、並行なら 0.1秒強で終わるはず
        # expected_min_time = 0.1  # 最も長い処理時間 (各スレッドが同時に実行されるため) - 未使用
        expected_max_time = 0.3  # 直列で実行された場合の合計時間 - 余裕を持つ
        assert (end_time - start_time) < expected_max_time

    @patch("yfinance.Ticker")
    def test_get_company_info(self, mock_ticker_class, fetcher, mock_ticker_info):
        """企業情報取得のテスト"""
        # モックの設定
        mock_ticker = Mock()
        mock_ticker.info = mock_ticker_info
        mock_ticker_class.return_value = mock_ticker

        # テスト実行
        result = fetcher.get_company_info("7203")

        # 検証
        assert result is not None
        assert result["symbol"] == "7203.T"
        assert result["name"] == "Toyota Motor Corporation"
        assert result["sector"] == "Consumer Cyclical"
        assert result["industry"] == "Auto Manufacturers"
        assert result["market_cap"] == 35000000000000

    @patch("day_trade.data.stock_fetcher.yf.Ticker")
    def test_lru_cache(self, mock_ticker_class, fetcher):
        """キャッシュの動作テスト"""
        # キャッシュをクリアしてテスト開始
        fetcher.clear_all_caches()

        # モックの設定
        mock_ticker = Mock()
        mock_ticker.info = {
            "currentPrice": 2500.0,
            "previousClose": 2480.0,
            "volume": 100,
            "longName": "Test Corp",
            "marketCap": 10000,
        }
        mock_ticker_class.return_value = mock_ticker

        # 同じ銘柄を複数回取得
        result1 = fetcher.get_current_price("7203")
        result2 = fetcher.get_current_price("7203")
        result3 = fetcher.get_current_price("7203")

        # 結果が同じであることを確認
        assert result1 == result2 == result3

        # TTLキャッシュが動作していることを確認（複数回呼び出しても値は同じ）
        assert result1["current_price"] == 2500.0

    def test_validate_symbol(self, fetcher):
        """シンボル妥当性チェックのテスト"""
        # 正常なシンボル
        fetcher._validate_symbol("7203")
        fetcher._validate_symbol("AAPL")

        # 無効なシンボル
        with pytest.raises(InvalidSymbolError):
            fetcher._validate_symbol("")

        with pytest.raises(InvalidSymbolError):
            fetcher._validate_symbol(None)

        with pytest.raises(InvalidSymbolError):
            fetcher._validate_symbol("1")

    def test_validate_period_interval(self, fetcher):
        """期間・間隔妥当性チェックのテスト"""
        # 正常な組み合わせ
        fetcher._validate_period_interval("1mo", "1d")
        fetcher._validate_period_interval("1d", "5m")

        # 無効な期間
        with pytest.raises(InvalidSymbolError):
            fetcher._validate_period_interval("invalid", "1d")

        # 無効な間隔
        with pytest.raises(InvalidSymbolError):
            fetcher._validate_period_interval("1mo", "invalid")

        # 分足データの期間制限
        with pytest.raises(InvalidSymbolError):
            fetcher._validate_period_interval("1mo", "5m")

    def test_is_retryable_error(self, fetcher):
        """リトライ可能エラー判定のテスト"""
        # リトライ可能なエラー
        assert fetcher._is_retryable_error(Exception("Connection timeout"))
        assert fetcher._is_retryable_error(Exception("Network error"))
        assert fetcher._is_retryable_error(Exception("500 Internal Server Error"))

        # リトライ不可能なエラー
        assert not fetcher._is_retryable_error(Exception("Invalid symbol"))
        assert not fetcher._is_retryable_error(Exception("Permission denied"))

    @patch("day_trade.data.stock_fetcher.yf.Ticker")
    def test_get_current_price_with_errors(self, mock_ticker_class, fetcher):
        """株価取得でエラーハンドリングのテスト"""
        # 空のinfoでDataNotFoundError
        mock_ticker = Mock()
        mock_ticker.info = {}
        mock_ticker_class.return_value = mock_ticker

        with pytest.raises(DataNotFoundError):
            fetcher.get_current_price("INVALID")

    def test_clear_all_caches(self, fetcher):
        """キャッシュクリアのテスト"""
        # キャッシュクリアを実行（エラーが出ないことを確認）
        fetcher.clear_all_caches()

    def test_cache_with_ttl_decorator(self):
        """TTLキャッシュデコレータのテスト"""
        call_count = 0

        @cache_with_ttl(1)
        def test_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # 最初の呼び出し
        result1 = test_function(5)
        assert result1 == 10
        assert call_count == 1

        # 2回目の呼び出し（キャッシュヒット）
        result2 = test_function(5)
        assert result2 == 10
        assert call_count == 1  # 呼び出し回数は変わらない

        # キャッシュクリア
        test_function.clear_cache()

        # 3回目の呼び出し（キャッシュクリア後）
        result3 = test_function(5)
        assert result3 == 10
        assert call_count == 2  # 呼び出し回数が増える
