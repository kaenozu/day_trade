"""
株価データ取得モジュール
yfinanceを使用してリアルタイムおよびヒストリカルな株価データを取得
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd
import yfinance as yf
from functools import lru_cache, wraps

logger = logging.getLogger(__name__)


class DataCache:
    """データキャッシュクラス"""

    def __init__(self, ttl_seconds: int = 60):
        """
        Args:
            ttl_seconds: キャッシュの有効期限（秒）
        """
        self.ttl_seconds = ttl_seconds
        self._cache = {}

    def get(self, key: str) -> Optional[any]:
        """キャッシュから値を取得"""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return value
            else:
                # 期限切れのキャッシュを削除
                del self._cache[key]
        return None

    def set(self, key: str, value: any) -> None:
        """キャッシュに値を設定"""
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()

    def size(self) -> int:
        """キャッシュサイズを取得"""
        return len(self._cache)


def cache_with_ttl(ttl_seconds: int):
    """TTL付きキャッシュデコレータ"""
    cache = DataCache(ttl_seconds)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # キャッシュキーを生成
            cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"

            # キャッシュから取得を試行
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"キャッシュヒット: {cache_key}")
                return cached_result

            # キャッシュにない場合は実行
            result = func(*args, **kwargs)
            if result is not None:
                cache.set(cache_key, result)
                logger.debug(f"キャッシュに保存: {cache_key}")

            return result

        # キャッシュ管理メソッドを追加
        wrapper.clear_cache = cache.clear
        wrapper.cache_size = cache.size

        return wrapper

    return decorator


class StockFetcherError(Exception):
    """株価データ取得エラー"""

    pass


class NetworkError(StockFetcherError):
    """ネットワークエラー"""

    pass


class InvalidSymbolError(StockFetcherError):
    """無効なシンボルエラー"""

    pass


class DataNotFoundError(StockFetcherError):
    """データが見つからないエラー"""

    pass


class StockFetcher:
    """株価データ取得クラス"""

    def __init__(
        self,
        cache_size: int = 128,
        price_cache_ttl: int = 30,
        historical_cache_ttl: int = 300,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            cache_size: LRUキャッシュのサイズ
            price_cache_ttl: 価格データキャッシュのTTL（秒）
            historical_cache_ttl: ヒストリカルデータキャッシュのTTL（秒）
            retry_count: リトライ回数
            retry_delay: リトライ間隔（秒）
        """
        self.cache_size = cache_size
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        # LRUキャッシュを動的に設定
        self._get_ticker = lru_cache(maxsize=cache_size)(self._create_ticker)

        # キャッシュTTLを設定
        self.price_cache_ttl = price_cache_ttl
        self.historical_cache_ttl = historical_cache_ttl

    def _create_ticker(self, symbol: str) -> yf.Ticker:
        """Tickerオブジェクトを作成（内部メソッド）"""
        return yf.Ticker(symbol)

    def _retry_on_error(self, func, *args, **kwargs):
        """エラー時のリトライ機能"""
        last_exception = None

        for attempt in range(self.retry_count):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # ネットワークエラーまたは一時的なエラーの場合のみリトライ
                if self._is_retryable_error(e):
                    if attempt < self.retry_count - 1:
                        logger.warning(
                            f"リトライ {attempt + 1}/{self.retry_count}: {e}"
                        )
                        time.sleep(self.retry_delay * (attempt + 1))  # 指数バックオフ
                        continue
                else:
                    # リトライ不可能なエラーは即座に再発生
                    break

        # 最終的にエラーを再発生
        self._handle_error(last_exception)

    def _is_retryable_error(self, error: Exception) -> bool:
        """リトライ可能なエラーかどうかを判定"""
        error_message = str(error).lower()

        # ネットワーク関連のエラー
        retryable_patterns = [
            "connection",
            "timeout",
            "network",
            "temporary",
            "500",
            "502",
            "503",
            "504",  # HTTPサーバーエラー
            "read timeout",
            "connection timeout",
        ]

        return any(pattern in error_message for pattern in retryable_patterns)

    def _handle_error(self, error: Exception) -> None:
        """エラーを適切な例外クラスに変換して再発生"""
        error_message = str(error).lower()

        if (
            "connection" in error_message
            or "network" in error_message
            or "timeout" in error_message
        ):
            raise NetworkError(f"ネットワークエラー: {error}") from error
        elif "not found" in error_message or "invalid" in error_message:
            raise InvalidSymbolError(f"無効なシンボル: {error}") from error
        elif "no data" in error_message or "empty" in error_message:
            raise DataNotFoundError(f"データが見つかりません: {error}") from error
        else:
            raise StockFetcherError(f"株価データ取得エラー: {error}") from error

    def _validate_symbol(self, symbol: str) -> None:
        """シンボルの妥当性をチェック"""
        if not symbol or not isinstance(symbol, str):
            raise InvalidSymbolError(f"無効なシンボル: {symbol}")

        if len(symbol) < 2:
            raise InvalidSymbolError(f"シンボルが短すぎます: {symbol}")

    def _validate_date_range(
        self, start_date: Union[str, datetime], end_date: Union[str, datetime]
    ) -> None:
        """日付範囲の妥当性をチェック"""
        if isinstance(start_date, str):
            try:
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError as e:
                raise InvalidSymbolError(f"無効な開始日形式: {start_date}") from e

        if isinstance(end_date, str):
            try:
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError as e:
                raise InvalidSymbolError(f"無効な終了日形式: {end_date}") from e

        if start_date >= end_date:
            raise InvalidSymbolError(
                f"開始日が終了日以降です: {start_date} >= {end_date}"
            )

        if end_date > datetime.now():
            logger.warning(f"終了日が未来の日付です: {end_date}")

    def clear_all_caches(self) -> None:
        """すべてのキャッシュをクリア"""
        self._get_ticker.cache_clear()

        # メソッドレベルのキャッシュもクリア
        if hasattr(self.get_current_price, "clear_cache"):
            self.get_current_price.clear_cache()
        if hasattr(self.get_historical_data, "clear_cache"):
            self.get_historical_data.clear_cache()
        if hasattr(self.get_historical_data_range, "clear_cache"):
            self.get_historical_data_range.clear_cache()

        logger.info("すべてのキャッシュをクリアしました")

    def _format_symbol(self, code: str, market: str = "T") -> str:
        """
        証券コードをyfinance形式にフォーマット

        Args:
            code: 証券コード（例：7203）
            market: 市場コード（T:東証、デフォルト）

        Returns:
            フォーマット済みシンボル（例：7203.T）
        """
        # すでに市場コードが付いている場合はそのまま返す
        if "." in code:
            return code
        return f"{code}.{market}"

    @cache_with_ttl(30)  # 30秒キャッシュ
    def get_current_price(self, code: str) -> Optional[Dict[str, float]]:
        """
        現在の株価情報を取得

        Args:
            code: 証券コード

        Returns:
            価格情報の辞書（現在値、前日終値、変化額、変化率など）

        Raises:
            InvalidSymbolError: 無効なシンボル
            NetworkError: ネットワークエラー
            DataNotFoundError: データが見つからない
            StockFetcherError: その他のエラー
        """

        def _get_price():
            self._validate_symbol(code)
            symbol = self._format_symbol(code)
            ticker = self._get_ticker(symbol)
            info = ticker.info

            # infoが空または無効な場合
            if not info or len(info) < 5:
                raise DataNotFoundError(f"企業情報を取得できません: {symbol}")

            # 基本情報を取得
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            previous_close = info.get("previousClose")

            if current_price is None:
                raise DataNotFoundError(f"現在価格を取得できません: {symbol}")

            # 変化額・変化率を計算
            change = current_price - previous_close if previous_close else 0
            change_percent = (change / previous_close * 100) if previous_close else 0

            return {
                "symbol": symbol,
                "current_price": current_price,
                "previous_close": previous_close,
                "change": change,
                "change_percent": change_percent,
                "volume": info.get("volume", 0),
                "market_cap": info.get("marketCap"),
                "timestamp": datetime.now(),
            }

        return self._retry_on_error(_get_price)

    @cache_with_ttl(300)  # 5分キャッシュ
    def get_historical_data(
        self, code: str, period: str = "1mo", interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        ヒストリカルデータを取得

        Args:
            code: 証券コード
            period: 期間（1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max）
            interval: 間隔（1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo）

        Returns:
            価格データのDataFrame（Open, High, Low, Close, Volume）

        Raises:
            InvalidSymbolError: 無効なシンボル
            NetworkError: ネットワークエラー
            DataNotFoundError: データが見つからない
            StockFetcherError: その他のエラー
        """

        def _get_historical():
            self._validate_symbol(code)
            self._validate_period_interval(period, interval)

            symbol = self._format_symbol(code)
            ticker = self._get_ticker(symbol)

            # データ取得
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                raise DataNotFoundError(f"ヒストリカルデータが空です: {symbol}")

            # インデックスをタイムゾーン除去
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # データの妥当性チェック
            if len(df) == 0:
                raise DataNotFoundError(f"有効なデータがありません: {symbol}")

            return df

        return self._retry_on_error(_get_historical)

    def _validate_period_interval(self, period: str, interval: str) -> None:
        """期間と間隔の妥当性をチェック"""
        valid_periods = [
            "1d",
            "5d",
            "1mo",
            "3mo",
            "6mo",
            "1y",
            "2y",
            "5y",
            "10y",
            "ytd",
            "max",
        ]
        valid_intervals = [
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ]

        if period not in valid_periods:
            raise InvalidSymbolError(f"無効な期間: {period}. 有効な値: {valid_periods}")

        if interval not in valid_intervals:
            raise InvalidSymbolError(
                f"無効な間隔: {interval}. 有効な値: {valid_intervals}"
            )

        # 分足データは短期間のみ対応
        if interval.endswith("m") and period not in ["1d", "5d"]:
            raise InvalidSymbolError(
                "分足データは1d または 5d 期間のみサポートされています"
            )

    @cache_with_ttl(300)  # 5分キャッシュ
    def get_historical_data_range(
        self,
        code: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        指定期間のヒストリカルデータを取得

        Args:
            code: 証券コード
            start_date: 開始日
            end_date: 終了日
            interval: 間隔

        Returns:
            価格データのDataFrame

        Raises:
            InvalidSymbolError: 無効なシンボル
            NetworkError: ネットワークエラー
            DataNotFoundError: データが見つからない
            StockFetcherError: その他のエラー
        """

        def _get_historical_range():
            self._validate_symbol(code)
            self._validate_date_range(start_date, end_date)

            symbol = self._format_symbol(code)
            ticker = self._get_ticker(symbol)

            # 文字列の場合はdatetimeに変換
            start_dt = start_date
            end_dt = end_date
            if isinstance(start_date, str):
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            # データ取得
            df = ticker.history(start=start_dt, end=end_dt, interval=interval)

            if df.empty:
                raise DataNotFoundError(
                    f"指定期間のヒストリカルデータが空です: {symbol}"
                )

            # インデックスをタイムゾーン除去
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            return df

        return self._retry_on_error(_get_historical_range)

    def get_realtime_data(self, codes: List[str]) -> Dict[str, Dict[str, float]]:
        """
        複数銘柄のリアルタイムデータを一括取得

        Args:
            codes: 証券コードのリスト

        Returns:
            銘柄コードをキーとした価格情報の辞書
        """
        if not codes or not isinstance(codes, list):
            raise InvalidSymbolError(f"無効な銘柄コードリスト: {codes}")

        results = {}
        failed_codes = []

        for code in codes:
            try:
                data = self.get_current_price(code)
                if data:
                    results[code] = data
                else:
                    failed_codes.append(code)
            except Exception as e:
                logger.warning(f"銘柄 {code} の取得に失敗: {e}")
                failed_codes.append(code)

        if failed_codes:
            logger.info(f"取得に失敗した銘柄: {failed_codes}")

        return results

    @cache_with_ttl(3600)  # 1時間キャッシュ
    def get_company_info(self, code: str) -> Optional[Dict[str, any]]:
        """
        企業情報を取得

        Args:
            code: 証券コード

        Returns:
            企業情報の辞書

        Raises:
            InvalidSymbolError: 無効なシンボル
            NetworkError: ネットワークエラー
            DataNotFoundError: データが見つからない
            StockFetcherError: その他のエラー
        """

        def _get_company_info():
            self._validate_symbol(code)
            symbol = self._format_symbol(code)
            ticker = self._get_ticker(symbol)
            info = ticker.info

            if not info or len(info) < 5:
                raise DataNotFoundError(f"企業情報を取得できません: {symbol}")

            return {
                "symbol": symbol,
                "name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "employees": info.get("fullTimeEmployees"),
                "description": info.get("longBusinessSummary"),
                "website": info.get("website"),
                "headquarters": info.get("city"),
                "country": info.get("country"),
                "currency": info.get("currency"),
                "exchange": info.get("exchange"),
            }

        return self._retry_on_error(_get_company_info)


# 使用例
if __name__ == "__main__":
    # ロギング設定
    logging.basicConfig(level=logging.INFO)

    # インスタンス作成
    fetcher = StockFetcher()

    # トヨタ自動車の現在価格を取得
    print("=== 現在価格 ===")
    current = fetcher.get_current_price("7203")
    if current:
        print(f"銘柄: {current['symbol']}")
        print(f"現在値: {current['current_price']:,.0f}円")
        print(
            f"前日比: {current['change']:+,.0f}円 ({current['change_percent']:+.2f}%)"
        )

    # ヒストリカルデータを取得
    print("\n=== ヒストリカルデータ（過去5日） ===")
    hist = fetcher.get_historical_data("7203", period="5d", interval="1d")
    if hist is not None:
        print(hist[["Open", "High", "Low", "Close", "Volume"]])

    # 企業情報を取得
    print("\n=== 企業情報 ===")
    info = fetcher.get_company_info("7203")
    if info:
        print(f"企業名: {info['name']}")
        print(f"セクター: {info['sector']}")
        print(f"業種: {info['industry']}")
