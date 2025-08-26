#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Technical Data Provider - 価格データ取得・生成モジュール
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Optional
import sys
import os

# Windows環境での文字化け対策
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from enhanced_symbol_manager import EnhancedSymbolManager
    ENHANCED_SYMBOLS_AVAILABLE = True
except ImportError:
    ENHANCED_SYMBOLS_AVAILABLE = False

try:
    from real_data_provider_v2 import real_data_provider, MultiSourceDataProvider
    REAL_DATA_PROVIDER_AVAILABLE = True
except ImportError:
    REAL_DATA_PROVIDER_AVAILABLE = False


class AdvancedDataProvider:
    """高度データプロバイダー"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.price_data_cache = {}
        
        if ENHANCED_SYMBOLS_AVAILABLE:
            self.symbol_manager = EnhancedSymbolManager()
    
    async def get_price_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        価格データ取得（実データプロバイダー対応）
        
        Args:
            symbol: 銘柄コード
            period: 取得期間
            
        Returns:
            価格データ DataFrame
        """
        cache_key = f"{symbol}_{period}"
        
        if cache_key in self.price_data_cache:
            cached_data, cache_time = self.price_data_cache[cache_key]
            if datetime.now() - cache_time < timedelta(minutes=15):
                return cached_data

        try:
            # 実データプロバイダーV2使用
            if REAL_DATA_PROVIDER_AVAILABLE:
                df = await real_data_provider.get_stock_data(symbol, period)
                
                if df is not None and not df.empty:
                    # キャッシュに保存
                    self.price_data_cache[cache_key] = (df, datetime.now())
                    self.logger.info(f"Real data fetched for {symbol}: {len(df)} days")
                    return df

            # フォールバック: 従来のYahoo Finance
            if YFINANCE_AVAILABLE:
                ticker_symbol = f"{symbol}.T" if not symbol.endswith('.T') else symbol
                ticker = yf.Ticker(ticker_symbol)
                df = ticker.history(period=period)
                
                if not df.empty:
                    self.price_data_cache[cache_key] = (df, datetime.now())
                    self.logger.info(f"Yahoo Finance data fetched for {symbol}: {len(df)} days")
                    return df

            # 最後のフォールバック: サンプルデータ
            self.logger.warning(f"Using sample data for {symbol}")
            return self._generate_sample_data(symbol, period)

        except Exception as e:
            self.logger.warning(f"Failed to fetch real data for {symbol}: {e}")
            return self._generate_sample_data(symbol, period)

    def _generate_sample_data(self, symbol: str, period: str) -> pd.DataFrame:
        """
        サンプル価格データ生成
        
        Args:
            symbol: 銘柄コード
            period: 期間
            
        Returns:
            サンプルデータ DataFrame
        """
        days = {
            "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
            "6mo": 180, "1y": 365
        }.get(period, 30)

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # ランダムウォーク + トレンド
        np.random.seed(hash(symbol) % 2**32)
        base_price = 1000 + (hash(symbol) % 5000)

        returns = np.random.normal(0.001, 0.02, days)  # 日次リターン
        trend = np.linspace(0, np.random.normal(0, 0.1), days)
        cumulative_returns = np.cumsum(returns + trend)
        prices = base_price * np.exp(cumulative_returns)

        # OHLC計算
        opens = prices * (1 + np.random.normal(0, 0.005, days))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.01, days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.01, days)))

        volumes = np.random.lognormal(14, 0.5, days).astype(int) * 1000

        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=dates)

        return df