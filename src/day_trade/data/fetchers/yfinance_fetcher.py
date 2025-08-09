"""
Yahoo Finance データフェッチャー

yfinanceライブラリを使用した株価データ取得
"""

import os
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import yfinance as yf

from ...utils.logging_config import get_context_logger, log_api_call, log_performance_metric
from ...utils.exceptions import InvalidSymbolError, DataNotFoundError, ValidationError
from ..cache.cache_decorators import cache_with_ttl
from .base_fetcher import BaseFetcher

logger = get_context_logger(__name__)


class YFinanceFetcher(BaseFetcher):
    """Yahoo Finance専用データフェッチャー"""

    def __init__(
        self,
        cache_size: int = 128,
        price_cache_ttl: int = 30,
        historical_cache_ttl: int = 300,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        初期化

        Args:
            cache_size: LRUキャッシュのサイズ
            price_cache_ttl: 価格データキャッシュのTTL（秒）
            historical_cache_ttl: ヒストリカルデータキャッシュのTTL（秒）
            retry_count: リトライ回数
            retry_delay: リトライ間隔（秒）
        """
        super().__init__(cache_size, retry_count, retry_delay)
        
        self.price_cache_ttl = price_cache_ttl
        self.historical_cache_ttl = historical_cache_ttl

        # LRUキャッシュを動的に設定
        self._get_ticker = lru_cache(maxsize=cache_size)(self._create_ticker)

        logger.info(
            f"YFinanceFetcher初期化完了: "
            f"price_ttl={price_cache_ttl}s, historical_ttl={historical_cache_ttl}s"
        )

    def _create_ticker(self, symbol: str) -> yf.Ticker:
        """Tickerオブジェクトを作成（内部メソッド）"""
        return yf.Ticker(symbol)

    def _fetch_data(self, symbol: str, **kwargs) -> Any:
        """基底クラスの抽象メソッド実装"""
        data_type = kwargs.get('data_type', 'current_price')
        
        if data_type == 'current_price':
            return self._get_current_price_raw(symbol)
        elif data_type == 'historical':
            return self._get_historical_data_raw(symbol, **kwargs)
        elif data_type == 'company_info':
            return self._get_company_info_raw(symbol)
        else:
            raise ValueError(f"未サポートのデータタイプ: {data_type}")

    def _validate_symbol(self, symbol: str) -> None:
        """銘柄コードの検証"""
        if not symbol or not isinstance(symbol, str):
            raise InvalidSymbolError(
                message="銘柄コードが無効です",
                symbol=symbol,
                error_code="INVALID_SYMBOL_FORMAT",
            )

        if len(symbol.strip()) == 0:
            raise InvalidSymbolError(
                message="銘柄コードが空です",
                symbol=symbol,
                error_code="EMPTY_SYMBOL",
            )

    def _validate_date_range(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
    ) -> None:
        """日付範囲の検証"""
        if period and (start_date or end_date):
            raise ValidationError(
                message="periodとstart_date/end_dateは同時に指定できません",
                error_code="CONFLICTING_DATE_PARAMS",
            )

        if start_date and end_date:
            try:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                if start >= end:
                    raise ValidationError(
                        message="開始日が終了日より後になっています",
                        error_code="INVALID_DATE_RANGE",
                    )
            except Exception as e:
                raise ValidationError(
                    message=f"日付形式が無効です: {e}",
                    error_code="INVALID_DATE_FORMAT",
                )

    def _format_symbol(self, code: str, market: str = "T") -> str:
        """日本株式用の銘柄コード形式変換"""
        try:
            # 4桁の数字かどうかチェック
            if code.isdigit() and len(code) == 4:
                # 東証の場合は .T を付与
                if market.upper() == "T":
                    return f"{code}.T"
                else:
                    return f"{code}.{market.upper()}"
            
            # 既に適切な形式の場合はそのまま返す
            if "." in code:
                return code
            
            # その他の場合はそのまま返す（米国株等）
            return code
            
        except Exception as e:
            logger.error(f"銘柄コード形式変換エラー: {code} - {e}")
            return code

    @cache_with_ttl(ttl_seconds=30)  # 30秒キャッシュ
    def get_current_price(self, code: str) -> Optional[Dict[str, float]]:
        """
        現在価格取得（キャッシュ付き）

        Args:
            code: 銘柄コード

        Returns:
            価格情報辞書
        """
        try:
            self._validate_symbol(code)
            symbol = self._format_symbol(code)
            
            return self._retry_on_error(self._get_current_price_raw, symbol)
            
        except Exception as e:
            logger.error(f"現在価格取得エラー: {code} - {e}")
            return None

    def _get_current_price_raw(self, symbol: str) -> Dict[str, float]:
        """現在価格の実際の取得処理"""
        with log_api_call("yfinance_current_price", symbol):
            ticker = self._get_ticker(symbol)
            
            # 高速な情報取得
            info = ticker.fast_info
            
            if not info:
                raise DataNotFoundError(
                    message=f"価格データが見つかりません: {symbol}",
                    symbol=symbol,
                    error_code="NO_PRICE_DATA",
                )

            result = {
                "current_price": float(info.get("lastPrice", 0)),
                "previous_close": float(info.get("previousClose", 0)),
                "open": float(info.get("open", 0)),
                "high": float(info.get("dayHigh", 0)),
                "low": float(info.get("dayLow", 0)),
                "volume": int(info.get("volume", 0)),
                "market_cap": float(info.get("marketCap", 0)),
                "timestamp": datetime.now().isoformat(),
            }

            # 価格変化の計算
            if result["previous_close"] > 0:
                price_change = result["current_price"] - result["previous_close"]
                result["price_change"] = price_change
                result["price_change_percent"] = (price_change / result["previous_close"]) * 100

            log_performance_metric(
                "current_price_success", 
                {"symbol": symbol, "price": result["current_price"]}
            )

            return result

    @cache_with_ttl(ttl_seconds=300)  # 5分キャッシュ
    def get_historical_data(
        self,
        code: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        履歴データ取得（キャッシュ付き）

        Args:
            code: 銘柄コード
            period: 期間（1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max）
            interval: 間隔（1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo）

        Returns:
            履歴データのDataFrame
        """
        try:
            self._validate_symbol(code)
            self._validate_period_interval(period, interval)
            symbol = self._format_symbol(code)
            
            return self._retry_on_error(
                self._get_historical_data_raw, 
                symbol, 
                period=period, 
                interval=interval
            )
            
        except Exception as e:
            logger.error(f"履歴データ取得エラー: {code} - {e}")
            return None

    def _validate_period_interval(self, period: str, interval: str) -> None:
        """期間と間隔の組み合わせを検証"""
        # 有効な期間
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        if period not in valid_periods:
            raise ValidationError(
                message=f"無効な期間: {period}",
                error_code="INVALID_PERIOD",
            )

        # 有効な間隔
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        if interval not in valid_intervals:
            raise ValidationError(
                message=f"無効な間隔: {interval}",
                error_code="INVALID_INTERVAL",
            )

        # 期間と間隔の組み合わせ制限
        minute_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
        if interval in minute_intervals:
            # 分足データは60日以内に制限
            long_periods = ["2y", "5y", "10y", "max"]
            if period in long_periods:
                raise ValidationError(
                    message=f"分足データは長期間({period})では取得できません",
                    error_code="INVALID_PERIOD_INTERVAL_COMBINATION",
                )

    def _get_historical_data_raw(
        self, 
        symbol: str, 
        period: str = "1mo", 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """履歴データの実際の取得処理"""
        with log_api_call("yfinance_historical", symbol):
            ticker = self._get_ticker(symbol)
            
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise DataNotFoundError(
                    message=f"履歴データが見つかりません: {symbol}",
                    symbol=symbol,
                    error_code="NO_HISTORICAL_DATA",
                )

            # データフレームのクリーンアップ
            df = df.dropna()
            
            # カラム名を日本語に変換
            column_mapping = {
                "Open": "始値",
                "High": "高値", 
                "Low": "安値",
                "Close": "終値",
                "Volume": "出来高",
                "Dividends": "配当",
                "Stock Splits": "株式分割",
            }
            
            df = df.rename(columns=column_mapping)
            
            # メタデータを追加
            df.attrs = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "fetched_at": datetime.now().isoformat(),
                "rows": len(df),
            }

            log_performance_metric(
                "historical_data_success",
                {"symbol": symbol, "rows": len(df), "period": period}
            )

            return df

    @cache_with_ttl(ttl_seconds=300)  # 5分キャッシュ
    def get_historical_data_range(
        self,
        code: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        日付範囲指定での履歴データ取得

        Args:
            code: 銘柄コード
            start_date: 開始日（YYYY-MM-DD）
            end_date: 終了日（YYYY-MM-DD）
            interval: 間隔

        Returns:
            履歴データのDataFrame
        """
        try:
            self._validate_symbol(code)
            self._validate_date_range(start_date, end_date)
            symbol = self._format_symbol(code)
            
            return self._retry_on_error(
                self._get_historical_data_range_raw,
                symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            
        except Exception as e:
            logger.error(f"日付範囲履歴データ取得エラー: {code} - {e}")
            return None

    def _get_historical_data_range_raw(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """日付範囲指定履歴データの実際の取得処理"""
        with log_api_call("yfinance_historical_range", symbol):
            ticker = self._get_ticker(symbol)
            
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise DataNotFoundError(
                    message=f"指定期間の履歴データが見つかりません: {symbol}",
                    symbol=symbol,
                    error_code="NO_RANGE_DATA",
                )

            # データ処理
            df = df.dropna()
            
            column_mapping = {
                "Open": "始値",
                "High": "高値",
                "Low": "安値", 
                "Close": "終値",
                "Volume": "出来高",
                "Dividends": "配当",
                "Stock Splits": "株式分割",
            }
            
            df = df.rename(columns=column_mapping)
            
            df.attrs = {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "interval": interval,
                "fetched_at": datetime.now().isoformat(),
                "rows": len(df),
            }

            return df

    @cache_with_ttl(ttl_seconds=600)  # 10分キャッシュ
    def get_company_info(self, code: str) -> Optional[Dict[str, Any]]:
        """
        企業情報取得（キャッシュ付き）

        Args:
            code: 銘柄コード

        Returns:
            企業情報辞書
        """
        try:
            self._validate_symbol(code)
            symbol = self._format_symbol(code)
            
            return self._retry_on_error(self._get_company_info_raw, symbol)
            
        except Exception as e:
            logger.error(f"企業情報取得エラー: {code} - {e}")
            return None

    def _get_company_info_raw(self, symbol: str) -> Dict[str, Any]:
        """企業情報の実際の取得処理"""
        with log_api_call("yfinance_company_info", symbol):
            ticker = self._get_ticker(symbol)
            
            info = ticker.info
            
            if not info:
                raise DataNotFoundError(
                    message=f"企業情報が見つかりません: {symbol}",
                    symbol=symbol,
                    error_code="NO_COMPANY_INFO",
                )

            # 重要な情報を抽出・整形
            result = {
                "symbol": symbol,
                "company_name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "pe_ratio": info.get("forwardPE", info.get("trailingPE", 0)),
                "pb_ratio": info.get("priceToBook", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 0),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0),
                "average_volume": info.get("averageVolume", 0),
                "shares_outstanding": info.get("sharesOutstanding", 0),
                "website": info.get("website", ""),
                "business_summary": info.get("businessSummary", ""),
                "full_info": info,  # 完全な情報も含める
                "fetched_at": datetime.now().isoformat(),
            }

            log_performance_metric(
                "company_info_success",
                {"symbol": symbol, "company_name": result["company_name"]}
            )

            return result

    def get_realtime_data(self, codes: List[str]) -> Dict[str, Dict[str, float]]:
        """
        複数銘柄のリアルタイムデータを一括取得

        Args:
            codes: 銘柄コードリスト

        Returns:
            銘柄別価格データ辞書
        """
        results = {}
        
        for code in codes:
            try:
                price_data = self.get_current_price(code)
                if price_data:
                    results[code] = price_data
                else:
                    logger.warning(f"リアルタイムデータ取得失敗: {code}")
                    results[code] = {}
                    
            except Exception as e:
                logger.error(f"リアルタイムデータ取得エラー: {code} - {e}")
                results[code] = {}

        logger.info(f"リアルタイムデータ一括取得完了: {len(codes)}銘柄中{len([r for r in results.values() if r])}銘柄成功")
        return results

    def get_cache_performance_report(self) -> Dict[str, Any]:
        """キャッシュパフォーマンスレポートを取得"""
        try:
            # 各キャッシュデコレータの統計を取得
            current_price_stats = getattr(self.get_current_price, 'get_stats', lambda: {})()
            historical_stats = getattr(self.get_historical_data, 'get_stats', lambda: {})()
            company_info_stats = getattr(self.get_company_info, 'get_stats', lambda: {})()

            return {
                "current_price_cache": current_price_stats,
                "historical_data_cache": historical_stats, 
                "company_info_cache": company_info_stats,
                "lru_cache_info": self.get_cache_info(),
                "overall_performance": self.get_performance_summary(),
            }
            
        except Exception as e:
            logger.error(f"キャッシュパフォーマンスレポート取得エラー: {e}")
            return {"error": str(e)}

    def clear_all_caches(self) -> None:
        """全てのキャッシュをクリア"""
        try:
            # デコレータキャッシュをクリア
            if hasattr(self.get_current_price, 'clear_cache'):
                self.get_current_price.clear_cache()
            
            if hasattr(self.get_historical_data, 'clear_cache'):
                self.get_historical_data.clear_cache()
            
            if hasattr(self.get_company_info, 'clear_cache'):
                self.get_company_info.clear_cache()
            
            # LRUキャッシュをクリア
            self.clear_cache()
            
            logger.info("全キャッシュクリア完了")
            
        except Exception as e:
            logger.error(f"キャッシュクリアエラー: {e}")

    def optimize_cache_settings(self) -> Dict[str, Any]:
        """キャッシュ設定の最適化提案"""
        performance_report = self.get_cache_performance_report()
        recommendations = []

        try:
            # 現在価格キャッシュの分析
            current_stats = performance_report.get("current_price_cache", {})
            if current_stats.get("hit_rate", 0) < 0.5:
                recommendations.append({
                    "cache": "current_price",
                    "issue": "低いヒット率",
                    "current_hit_rate": current_stats.get("hit_rate", 0),
                    "suggestion": "TTLを延長することを検討（現在30秒）"
                })

            # 履歴データキャッシュの分析
            historical_stats = performance_report.get("historical_data_cache", {})
            if historical_stats.get("hit_rate", 0) < 0.7:
                recommendations.append({
                    "cache": "historical_data",
                    "issue": "低いヒット率",
                    "current_hit_rate": historical_stats.get("hit_rate", 0),
                    "suggestion": "TTLを延長することを検討（現在5分）"
                })

            # LRUキャッシュの分析
            lru_info = performance_report.get("lru_cache_info", {})
            if lru_info.get("hit_rate", 0) < 0.6:
                recommendations.append({
                    "cache": "lru_ticker",
                    "issue": "低いヒット率",
                    "current_hit_rate": lru_info.get("hit_rate", 0),
                    "suggestion": f"キャッシュサイズを増加することを検討（現在{self.cache_size}）"
                })

            return {
                "recommendations": recommendations,
                "performance_report": performance_report,
                "analysis_timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"キャッシュ最適化分析エラー: {e}")
            return {"error": str(e), "recommendations": []}