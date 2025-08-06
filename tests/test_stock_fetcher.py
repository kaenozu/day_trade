"""
株価データ取得モジュールのテスト
"""

import time
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.day_trade.data.stock_fetcher import (
    DataCache,
    DataNotFoundError,
    InvalidSymbolError,
    StockFetcher,
    StockFetcherError,
    cache_with_ttl,
)
from src.day_trade.utils.exceptions import NetworkError


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

        assert len(cache._cache) == 2

        cache.clear()
        assert len(cache._cache) == 0
        assert cache.get("key1") is None

    def test_cache_stale_while_revalidate(self):
        """stale-while-revalidate機能のテスト"""
        cache = DataCache(ttl_seconds=1, stale_while_revalidate=2)

        # データを設定
        cache.set("test_key", "test_value")

        # すぐに取得（フレッシュ）
        result = cache.get("test_key")
        assert result == "test_value"

        # TTL期限切れまで待機
        time.sleep(1.1)

        # 通常の取得（期限切れなのでNone）
        result = cache.get("test_key", allow_stale=False)
        assert result is None

        # キャッシュにデータを再設定（上記でNoneを返した際にキーが削除されるため）
        cache.set("test_key", "test_value")
        time.sleep(1.1)  # TTL期限切れまで待機

        # stale許可で取得（まだstale期間内なので取得可能）
        result = cache.get("test_key", allow_stale=True)
        assert result == "test_value"

        # stale統計の確認
        stats = cache.get_cache_stats()
        assert stats["stale_hit_count"] >= 1

    def test_cache_lru_eviction(self):
        """LRU evictionのテスト"""
        cache = DataCache(ttl_seconds=60, max_size=2)

        # 最大サイズまで設定
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache._cache) == 2

        # key1にアクセスして最近使用にする
        cache.get("key1")

        # 新しいキーを追加（key2が追い出される）
        cache.set("key3", "value3")

        assert len(cache._cache) == 2
        assert cache.get("key1") == "value1"  # まだ存在
        assert cache.get("key2") is None  # 追い出された
        assert cache.get("key3") == "value3"  # 新しく追加

    def test_cache_environment_variables(self):
        """環境変数による設定のテスト"""
        with patch.dict(
            "os.environ",
            {
                "STOCK_CACHE_TTL_SECONDS": "120",
                "STOCK_CACHE_MAX_SIZE": "500",
                "STOCK_CACHE_STALE_SECONDS": "300",
            },
        ):
            cache = DataCache()
            assert cache.ttl_seconds == 120
            assert cache.max_size == 500
            assert cache.stale_while_revalidate == 300

    def test_cache_stats(self):
        """キャッシュ統計のテスト"""
        cache = DataCache(ttl_seconds=60)

        # 初期状態
        stats = cache.get_cache_stats()
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0
        assert stats["stale_hit_count"] == 0

        # データ設定と取得
        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key2")  # miss

        stats = cache.get_cache_stats()
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1

    def test_cache_key_validation(self):
        """キャッシュキーの検証テスト"""
        cache = DataCache(ttl_seconds=60)

        # 正常なキー
        cache.set("valid_key", "value")
        assert cache.get("valid_key") == "value"

        # 空のキー
        cache.set("", "empty_key_value")
        result = cache.get("")
        # 空のキーでも動作することを確認
        assert result == "empty_key_value"


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

    @patch("src.day_trade.data.stock_fetcher.yf.Ticker")
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

    @patch("src.day_trade.data.stock_fetcher.StockFetcher.get_current_price")
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

    @patch("src.day_trade.data.stock_fetcher.yf.Ticker")
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
        # timestamp は比較から除外する
        # result.pop('timestamp') を使用してtimestampを削除し、残りの辞書を比較
        result1.pop("timestamp", None)
        result2.pop("timestamp", None)
        result3.pop("timestamp", None)

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
        import requests.exceptions as req_exc

        # リトライ可能なエラー (ConnectionError, Timeout)
        assert fetcher._is_retryable_error(req_exc.ConnectionError("Connection failed"))
        assert fetcher._is_retryable_error(req_exc.Timeout("Request timed out"))

        # リトライ可能なHTTPエラー (5xx, 408, 429)
        mock_response_500 = Mock()
        mock_response_500.status_code = 500
        http_error_500 = req_exc.HTTPError(
            "500 Internal Server Error", response=mock_response_500
        )
        assert fetcher._is_retryable_error(http_error_500)

        mock_response_408 = Mock()
        mock_response_408.status_code = 408
        http_error_408 = req_exc.HTTPError(
            "408 Request Timeout", response=mock_response_408
        )
        assert fetcher._is_retryable_error(http_error_408)

        # リトライ不可能なエラー (その他のExceptionや、リトライ対象外のHTTPエラー)
        assert not fetcher._is_retryable_error(Exception("Some other error"))

        mock_response_401 = Mock()
        mock_response_401.status_code = 401
        http_error_401 = req_exc.HTTPError(
            "401 Unauthorized", response=mock_response_401
        )
        assert not fetcher._is_retryable_error(http_error_401)

    @patch("src.day_trade.data.stock_fetcher.yf.Ticker")
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

    def test_fetcher_with_custom_cache_settings(self):
        """カスタムキャッシュ設定でのStockFetcher初期化テスト"""
        fetcher = StockFetcher(
            cache_size=10,
            price_cache_ttl=30,
            historical_cache_ttl=120,
            retry_count=5,
            retry_delay=0.5,
        )

        assert fetcher.retry_count == 5
        assert fetcher.retry_delay == 0.5

    def test_get_current_price_with_retry(self, fetcher):
        """リトライ機能のテスト（ロジックの確認）"""
        # リトライ可能エラーの判定テスト
        import requests.exceptions as req_exc

        # ConnectionErrorはリトライ可能
        conn_error = req_exc.ConnectionError("Network error")
        assert fetcher._is_retryable_error(conn_error)

        # TimeoutErrorもリトライ可能
        timeout_error = req_exc.Timeout("Request timeout")
        assert fetcher._is_retryable_error(timeout_error)

        # HTTPError（500番台）もリトライ可能
        http_error = req_exc.HTTPError("Internal server error")
        mock_response = Mock()
        mock_response.status_code = 500
        http_error.response = mock_response
        assert fetcher._is_retryable_error(http_error)

        # HTTPError（400番台）はリトライ不可
        http_error_400 = req_exc.HTTPError("Bad request")
        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        http_error_400.response = mock_response_400
        assert not fetcher._is_retryable_error(http_error_400)

        # 一般的なExceptionはリトライ不可
        general_error = Exception("General error")
        assert not fetcher._is_retryable_error(general_error)

    def test_get_current_price_retry_exhausted(self, fetcher):
        """リトライ統計のテスト"""
        # リトライ統計の初期状態を確認
        stats = fetcher.get_retry_stats()
        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "failed_requests" in stats
        assert "total_retries" in stats

        # 統計の初期値確認
        assert stats["total_requests"] >= 0
        assert stats["successful_requests"] >= 0
        assert stats["failed_requests"] >= 0

    def test_format_symbol_edge_cases(self, fetcher):
        """シンボルフォーマットのエッジケースのテスト"""
        # 小文字での入力（東証デフォルト）
        assert fetcher._format_symbol("aapl") == "aapl.T"

        # 数字のみ（日本株）
        assert fetcher._format_symbol("1234") == "1234.T"

        # 既存の市場コード（別市場指定）
        assert (
            fetcher._format_symbol("7203.T", "O") == "7203.T"
        )  # 既存の市場コードはそのまま

    @patch("src.day_trade.data.stock_fetcher.yf.Ticker")
    def test_get_historical_data_with_different_intervals(
        self, mock_ticker_class, fetcher
    ):
        """様々な間隔でのヒストリカルデータ取得テスト"""
        # モックデータの作成
        dates = pd.date_range(end=datetime.now(), periods=24, freq="h")
        mock_df = pd.DataFrame(
            {
                "Open": range(2400, 2424),
                "High": range(2410, 2434),
                "Low": range(2390, 2414),
                "Close": range(2405, 2429),
                "Volume": [1000000] * 24,
            },
            index=dates,
        )

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker

        # 時間足データの取得
        result = fetcher.get_historical_data("7203", period="1d", interval="1h")

        assert result is not None
        assert len(result) == 24
        mock_ticker.history.assert_called_with(period="1d", interval="1h")

    @patch("src.day_trade.data.stock_fetcher.yf.Ticker")
    def test_get_company_info_missing_fields(self, mock_ticker_class, fetcher):
        """企業情報の一部フィールドが欠損している場合のテスト"""
        # 十分な情報がある場合（5つ以上のフィールド）
        mock_ticker = Mock()
        mock_ticker.info = {
            "longName": "Test Corporation",
            "sector": "Technology",
            "industry": "Software",
            "marketCap": 1000000000,
            "currency": "USD",
            "exchange": "NASDAQ",
            # website, city, country は欠損
        }
        mock_ticker_class.return_value = mock_ticker

        result = fetcher.get_company_info("TEST")

        assert result is not None
        assert result["name"] == "Test Corporation"
        assert result["sector"] == "Technology"
        assert result["industry"] == "Software"
        assert result["market_cap"] == 1000000000
        assert "website" in result  # キーは存在するがNoneまたはデフォルト値
        assert result["website"] is None

    def test_validate_period_interval_edge_cases(self, fetcher):
        """期間・間隔妥当性チェックのエッジケース"""
        # 正常なケース
        fetcher._validate_period_interval("1y", "1d")
        fetcher._validate_period_interval("max", "1wk")

        # 分足の期間制限（1d, 5dのみ許可）
        fetcher._validate_period_interval("1d", "5m")
        fetcher._validate_period_interval("5d", "5m")

        # 分足で無効な期間（1mo など）
        with pytest.raises(InvalidSymbolError):
            fetcher._validate_period_interval("1mo", "5m")

        # 時間足のテスト
        fetcher._validate_period_interval("5d", "1h")

    def test_concurrent_access_thread_safety(self, fetcher):
        """並行アクセスでのスレッドセーフティテスト（基本機能確認）"""
        # LRUキャッシュのスレッドセーフ性確認（基本テスト）
        assert hasattr(fetcher, "_get_ticker")
        assert callable(fetcher._get_ticker)

        # キャッシュクリア機能の確認
        fetcher.clear_all_caches()

        # リトライ統計機能の確認（スレッドセーフ）
        stats1 = fetcher.get_retry_stats()
        stats2 = fetcher.get_retry_stats()

        # 統計データの一貫性確認
        assert isinstance(stats1, dict)
        assert isinstance(stats2, dict)
        assert "total_requests" in stats1
        assert "total_requests" in stats2

    def test_clear_all_caches_with_data(self, fetcher):
        """データがあるキャッシュのクリアテスト"""
        # キャッシュに何かデータを入れる（モック）
        with patch.object(fetcher, "get_current_price") as mock_get_price:
            mock_get_price.return_value = {"current_price": 1000.0}
            fetcher.get_current_price("TEST")

        # キャッシュクリア実行
        fetcher.clear_all_caches()

        # エラーが発生しないことを確認
        assert True

    @patch("src.day_trade.data.stock_fetcher.yf.Ticker")
    def test_network_error_handling(self, mock_ticker_class, fetcher):
        """ネットワークエラーのハンドリングテスト"""
        import requests.exceptions as req_exc

        # ConnectionErrorを発生させる
        mock_ticker_class.side_effect = req_exc.ConnectionError("Network unreachable")

        with pytest.raises(StockFetcherError):
            fetcher.get_current_price("NETWORK_ERROR_TEST")

    def test_environment_based_configuration(self):
        """環境変数ベースの設定テスト"""
        with patch.dict(
            "os.environ",
            {
                "STOCK_FETCHER_RETRY_COUNT": "10",
                "STOCK_FETCHER_RETRY_DELAY": "2.0",
                "STOCK_FETCHER_CACHE_SIZE": "1000",
            },
        ):
            # 環境変数から設定を読み込む場合のテスト
            # （実際のStockFetcherが環境変数を読み込む場合）
            fetcher = StockFetcher()
            assert fetcher is not None

    def test_cache_with_ttl_decorator_edge_cases(self):
        """TTLキャッシュデコレータのエッジケースのテスト"""
        call_count = 0

        @cache_with_ttl(1)
        def test_function_with_error(x):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call error")
            return x * 2

        # デコレータがエラーを適切にハンドリングすることを確認
        # （実装によってはエラーがキャッチされて別の例外に変換される可能性があります）
        from contextlib import suppress

        with suppress(ValueError, StockFetcherError):
            test_function_with_error(5)

        # 複数回呼び出してキャッシュが正しく動作することを確認
        call_count = 0  # リセット

        @cache_with_ttl(1)
        def test_function_success(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # 成功ケースでのキャッシュ動作確認
        result1 = test_function_success(5)
        result2 = test_function_success(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # キャッシュヒットで2回目は呼ばれない

    def test_date_range_validation_edge_cases(self, fetcher):
        """日付範囲検証のエッジケースのテスト"""
        from datetime import datetime

        # 正常なケース
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 10)
        fetcher._validate_date_range(start, end)

        # 文字列での日付指定
        fetcher._validate_date_range("2023-01-01", "2023-01-10")

        # 不正な開始日形式
        with pytest.raises(InvalidSymbolError):
            fetcher._validate_date_range("invalid-date", "2023-01-10")

        # 不正な終了日形式
        with pytest.raises(InvalidSymbolError):
            fetcher._validate_date_range("2023-01-01", "invalid-date")

        # 開始日が終了日以降
        with pytest.raises(InvalidSymbolError):
            fetcher._validate_date_range("2023-01-10", "2023-01-01")

    @patch("src.day_trade.data.stock_fetcher.yf.Ticker")
    def test_historical_data_empty_response(self, mock_ticker_class, fetcher):
        """空のヒストリカルデータ応答のテスト"""
        # 空のDataFrameを返すモック
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        # DataNotFoundErrorが発生することを確認
        with pytest.raises(DataNotFoundError):
            fetcher.get_historical_data("EMPTY_TEST")

    @patch("src.day_trade.data.stock_fetcher.yf.Ticker")
    def test_company_info_insufficient_data(self, mock_ticker_class, fetcher):
        """企業情報が不十分な場合のテスト"""
        # 情報が少ない（5未満）モックデータ
        mock_ticker = Mock()
        mock_ticker.info = {"longName": "Test", "sector": "Tech"}  # 2つのフィールドのみ
        mock_ticker_class.return_value = mock_ticker

        # DataNotFoundErrorが発生することを確認
        with pytest.raises(DataNotFoundError):
            fetcher.get_company_info("INSUFFICIENT_TEST")

    def test_error_handling_edge_cases(self, fetcher):
        """エラーハンドリングのエッジケースのテスト"""
        # 既存のカスタム例外はそのまま再発生
        existing_error = StockFetcherError("existing error")
        with pytest.raises(StockFetcherError):
            fetcher._handle_error(existing_error)

        existing_invalid = InvalidSymbolError("invalid symbol")
        with pytest.raises(InvalidSymbolError):
            fetcher._handle_error(existing_invalid)

        existing_not_found = DataNotFoundError("data not found")
        with pytest.raises(DataNotFoundError):
            fetcher._handle_error(existing_not_found)

    def test_cache_performance_report(self, fetcher):
        """キャッシュパフォーマンスレポートのテスト"""
        # キャッシュパフォーマンスレポートを取得
        report = fetcher.get_cache_performance_report()

        assert isinstance(report, dict)
        assert "cache_enabled" in report
        assert "auto_tuning_enabled" in report
        assert "last_adjustment" in report
        assert "adjustment_interval" in report

    @patch("src.day_trade.data.stock_fetcher.yf.Ticker")
    def test_get_historical_data_range_success(self, mock_ticker_class, fetcher):
        """期間指定ヒストリカルデータ取得の成功テスト"""
        from datetime import datetime

        # モックデータの作成
        dates = pd.date_range("2023-01-01", "2023-01-05", freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [2400, 2410, 2420, 2415, 2425],
                "High": [2420, 2430, 2440, 2435, 2445],
                "Low": [2380, 2390, 2400, 2395, 2405],
                "Close": [2410, 2420, 2415, 2425, 2430],
                "Volume": [1000000] * 5,
            },
            index=dates,
        )

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker

        # 文字列の日付で実行
        result = fetcher.get_historical_data_range(
            "TEST_RANGE", "2023-01-01", "2023-01-05", "1d"
        )

        assert result is not None
        assert len(result) == 5

        # datetimeオブジェクトの日付で実行
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 5)
        result2 = fetcher.get_historical_data_range(
            "TEST_RANGE2", start_date, end_date, "1d"
        )

        assert result2 is not None
        assert len(result2) == 5

    @patch("src.day_trade.data.stock_fetcher.yf.Ticker")
    def test_get_current_price_debug_logging(self, mock_ticker_class):
        """デバッグロギング有効時のテスト"""
        # デバッグロギングを有効にしたStockFetcherを作成
        with patch.dict("os.environ", {"STOCK_FETCHER_DEBUG": "true"}):
            debug_fetcher = StockFetcher()

            # モックの設定
            mock_ticker = Mock()
            mock_ticker.info = {
                "currentPrice": 2500.0,
                "previousClose": 2480.0,
                "volume": 1000000,
                "longName": "Debug Test Corp",
                "marketCap": 10000,
            }
            mock_ticker_class.return_value = mock_ticker

            # デバッグモードでの実行
            result = debug_fetcher.get_current_price("DEBUG_TEST")

            assert result is not None
            assert result["current_price"] == 2500.0
            assert debug_fetcher.enable_debug_logging

    def test_auto_cache_tuning_disabled(self):
        """自動キャッシュチューニング無効時のテスト"""
        with patch.dict("os.environ", {"AUTO_CACHE_TUNING": "false"}):
            tuning_disabled_fetcher = StockFetcher()

            assert not tuning_disabled_fetcher.auto_cache_tuning_enabled

            # _maybe_adjust_cache_settings が何もしないことを確認
            tuning_disabled_fetcher._maybe_adjust_cache_settings()  # エラーが発生しないことを確認

    def test_validate_symbol_edge_cases(self, fetcher):
        """シンボル検証のエッジケースのテスト"""
        # 正常なシンボル
        fetcher._validate_symbol("AAPL")
        fetcher._validate_symbol("7203")

        # 無効なシンボル（None）
        with pytest.raises(InvalidSymbolError):
            fetcher._validate_symbol(None)

        # 無効なシンボル（空文字列）
        with pytest.raises(InvalidSymbolError):
            fetcher._validate_symbol("")

        # 無効なシンボル（短すぎる）
        with pytest.raises(InvalidSymbolError):
            fetcher._validate_symbol("A")

        # 無効なシンボル（数値型）
        with pytest.raises(InvalidSymbolError):
            fetcher._validate_symbol(123)

    def test_string_based_error_classification(self, fetcher):
        """文字列ベースのエラー分類のテスト"""
        # ネットワーク関連エラー
        network_errors = [
            Exception("connection failed"),
            Exception("network timeout"),
            Exception("timeout occurred"),
        ]

        for error in network_errors:
            with pytest.raises(NetworkError):
                fetcher._handle_error(error)

        # 無効なシンボル関連エラー
        symbol_errors = [
            Exception("symbol not found"),
            Exception("invalid ticker symbol"),
        ]

        for error in symbol_errors:
            with pytest.raises(InvalidSymbolError):
                fetcher._handle_error(error)

        # データ未検出エラー
        data_errors = [
            Exception("no data available"),
            Exception("empty dataset returned"),
        ]

        for error in data_errors:
            with pytest.raises(DataNotFoundError):
                fetcher._handle_error(error)

        # その他のエラー
        with pytest.raises(StockFetcherError):
            fetcher._handle_error(Exception("unknown error type"))

    def test_record_retry_statistics(self, fetcher):
        """リトライ統計記録のテスト"""
        # 初期統計の確認
        initial_stats = fetcher.get_retry_stats()
        initial_requests = initial_stats["total_requests"]

        # リクエスト開始を記録
        fetcher._record_request_start()

        # リクエスト成功を記録
        fetcher._record_request_success()

        # リトライ試行を記録
        fetcher._record_retry_attempt()

        # リトライ成功を記録
        fetcher._record_retry_success()

        # 統計の確認
        stats = fetcher.get_retry_stats()
        assert stats["total_requests"] == initial_requests + 1
        assert stats["successful_requests"] >= 1
        assert stats["total_retries"] >= 1
        assert stats["retry_success"] >= 1

    def test_date_range_validation_future_date_warning(self, fetcher):
        """未来の日付での警告テスト"""
        from datetime import datetime, timedelta

        # 現在時刻から1年後の日付（未来）
        future_date = datetime.now() + timedelta(days=365)
        past_date = datetime.now() - timedelta(days=30)

        # 未来の日付を終了日に指定（警告が出るが例外は発生しない）
        # fetcherインスタンスのloggerを使用
        with patch.object(fetcher, "logger"):
            fetcher._validate_date_range(past_date, future_date)
            # エラーが発生しないことを確認（警告のみ）

    @patch("src.day_trade.data.stock_fetcher.yf.Ticker")
    def test_current_price_performance_logging(self, mock_ticker_class, fetcher):
        """パフォーマンスロギングのテスト"""
        # モックの設定
        mock_ticker = Mock()
        mock_ticker.info = {
            "currentPrice": 2500.0,
            "previousClose": 2480.0,
            "volume": 1000000,
            "longName": "Performance Test Corp",
            "marketCap": 10000,
        }
        mock_ticker_class.return_value = mock_ticker

        # パフォーマンスロギング付きで実行
        result = fetcher.get_current_price("PERF_TEST")

        assert result is not None
        assert result["current_price"] == 2500.0

    def test_error_handling_network_exceptions(self, fetcher):
        """ネットワーク例外の高度なハンドリングテスト"""
        import requests.exceptions as req_exc

        # HTTPError（404）のテスト
        http_error_404 = req_exc.HTTPError("Not Found")
        mock_response = Mock()
        mock_response.status_code = 404
        http_error_404.response = mock_response

        # handle_network_exceptionが呼ばれることを確認
        with patch(
            "src.day_trade.data.stock_fetcher.handle_network_exception"
        ) as mock_handle:
            mock_handle.side_effect = NetworkError("Converted network error")

            with pytest.raises(NetworkError):
                fetcher._handle_error(http_error_404)

            mock_handle.assert_called_once_with(http_error_404)

    def test_record_request_failure_with_different_errors(self, fetcher):
        """異なるエラータイプでのリクエスト失敗記録テスト"""
        # 初期統計
        initial_stats = fetcher.get_retry_stats()
        initial_failures = initial_stats["failed_requests"]
        # initial_error_types = len(initial_stats["errors_by_type"])  # 未使用のため削除

        # 異なるタイプのエラーを記録
        fetcher._record_request_failure(ValueError("test error"))
        fetcher._record_request_failure(ConnectionError("connection error"))
        fetcher._record_request_failure(ValueError("another test error"))

        # 統計の確認
        stats = fetcher.get_retry_stats()
        assert stats["failed_requests"] == initial_failures + 3
        assert "ValueError" in stats["errors_by_type"]
        assert "ConnectionError" in stats["errors_by_type"]
        assert stats["errors_by_type"]["ValueError"] >= 2  # 2回記録
        assert stats["errors_by_type"]["ConnectionError"] >= 1  # 1回記録

    def test_success_rate_calculation(self, fetcher):
        """成功率計算のテスト"""
        # リクエストとリトライの統計を記録
        fetcher._record_request_start()
        fetcher._record_request_success()

        fetcher._record_request_start()
        fetcher._record_request_failure(Exception("test"))

        fetcher._record_retry_attempt()
        fetcher._record_retry_success()

        # 統計の確認
        stats = fetcher.get_retry_stats()

        # 成功率の計算が含まれていることを確認
        if stats["total_requests"] > 0:
            assert "success_rate" in stats
            assert "failure_rate" in stats
            assert isinstance(stats["success_rate"], float)
            assert isinstance(stats["failure_rate"], float)

        # リトライ成功率の計算が含まれていることを確認
        if stats["total_retries"] > 0:
            assert "retry_success_rate" in stats
            assert isinstance(stats["retry_success_rate"], float)

    @patch("src.day_trade.data.stock_fetcher.yf.Ticker")
    def test_get_historical_data_timezone_handling(self, mock_ticker_class, fetcher):
        """タイムゾーン処理のテスト"""
        import pytz

        # タイムゾーン付きのデータを作成
        dates = pd.date_range("2023-01-01", "2023-01-03", freq="D", tz=pytz.UTC)
        mock_df = pd.DataFrame(
            {
                "Open": [2400, 2410, 2420],
                "High": [2420, 2430, 2440],
                "Low": [2380, 2390, 2400],
                "Close": [2410, 2420, 2430],
                "Volume": [1000000] * 3,
            },
            index=dates,
        )

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker

        # タイムゾーン処理が正しく動作することを確認
        result = fetcher.get_historical_data("TZ_TEST")

        assert result is not None
        assert len(result) == 3
        # インデックスのタイムゾーンが除去されていることを確認
        assert result.index.tz is None
