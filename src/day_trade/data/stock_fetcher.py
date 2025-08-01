"""
株価データ取得モジュール
yfinanceを使用してリアルタイムおよびヒストリカルな株価データを取得
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
import yfinance as yf
from functools import lru_cache

logger = logging.getLogger(__name__)


class StockFetcher:
    """株価データ取得クラス"""
    
    def __init__(self, cache_size: int = 128):
        """
        Args:
            cache_size: LRUキャッシュのサイズ
        """
        self.cache_size = cache_size
        # LRUキャッシュを動的に設定
        self._get_ticker = lru_cache(maxsize=cache_size)(self._create_ticker)
    
    def _create_ticker(self, symbol: str) -> yf.Ticker:
        """Tickerオブジェクトを作成（内部メソッド）"""
        return yf.Ticker(symbol)
    
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
    
    def get_current_price(self, code: str) -> Optional[Dict[str, float]]:
        """
        現在の株価情報を取得
        
        Args:
            code: 証券コード
            
        Returns:
            価格情報の辞書（現在値、前日終値、変化額、変化率など）
        """
        try:
            symbol = self._format_symbol(code)
            ticker = self._get_ticker(symbol)
            info = ticker.info
            
            # 基本情報を取得
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            previous_close = info.get("previousClose")
            
            if current_price is None:
                logger.warning(f"現在価格を取得できません: {symbol}")
                return None
            
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
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"株価取得エラー ({code}): {e}")
            return None
    
    def get_historical_data(
        self,
        code: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        ヒストリカルデータを取得
        
        Args:
            code: 証券コード
            period: 期間（1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max）
            interval: 間隔（1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo）
            
        Returns:
            価格データのDataFrame（Open, High, Low, Close, Volume）
        """
        try:
            symbol = self._format_symbol(code)
            ticker = self._get_ticker(symbol)
            
            # データ取得
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"ヒストリカルデータが空です: {symbol}")
                return None
            
            # インデックスをタイムゾーン除去
            df.index = df.index.tz_localize(None)
            
            return df
            
        except Exception as e:
            logger.error(f"ヒストリカルデータ取得エラー ({code}): {e}")
            return None
    
    def get_historical_data_range(
        self,
        code: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
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
        """
        try:
            symbol = self._format_symbol(code)
            ticker = self._get_ticker(symbol)
            
            # 文字列の場合はdatetimeに変換
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
            # データ取得
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.warning(f"ヒストリカルデータが空です: {symbol}")
                return None
            
            # インデックスをタイムゾーン除去
            df.index = df.index.tz_localize(None)
            
            return df
            
        except Exception as e:
            logger.error(f"期間指定データ取得エラー ({code}): {e}")
            return None
    
    def get_realtime_data(self, codes: List[str]) -> Dict[str, Dict[str, float]]:
        """
        複数銘柄のリアルタイムデータを一括取得
        
        Args:
            codes: 証券コードのリスト
            
        Returns:
            銘柄コードをキーとした価格情報の辞書
        """
        results = {}
        
        for code in codes:
            data = self.get_current_price(code)
            if data:
                results[code] = data
        
        return results
    
    def get_company_info(self, code: str) -> Optional[Dict[str, any]]:
        """
        企業情報を取得
        
        Args:
            code: 証券コード
            
        Returns:
            企業情報の辞書
        """
        try:
            symbol = self._format_symbol(code)
            ticker = self._get_ticker(symbol)
            info = ticker.info
            
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
            }
            
        except Exception as e:
            logger.error(f"企業情報取得エラー ({code}): {e}")
            return None


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
        print(f"前日比: {current['change']:+,.0f}円 ({current['change_percent']:+.2f}%)")
    
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