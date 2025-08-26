#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Provider for Technical Analysis
技術分析用データプロバイダー
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict

# Windows環境での文字化け対策
import sys
import os
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
    from real_data_provider_v2 import real_data_provider, MultiSourceDataProvider
    REAL_DATA_PROVIDER_AVAILABLE = True
except ImportError:
    REAL_DATA_PROVIDER_AVAILABLE = False


class TechnicalDataProvider:
    """技術分析用データプロバイダークラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.price_data_cache = {}

    async def get_price_data(self, symbol: str, period: str) -> pd.DataFrame:
        """価格データ取得（実データプロバイダー対応）"""

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
        """サンプル価格データ生成"""

        days = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}.get(period, 30)

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

    def calculate_composite_score(self, trend: Dict, momentum: Dict, volatility: Dict) -> float:
        """複合スコア計算"""
        try:
            score = 50.0  # 中立からスタート

            # トレンド評価
            if 'SMA_5' in trend and 'SMA_20' in trend:
                if trend['SMA_5'] > trend['SMA_20']:
                    score += 10
                else:
                    score -= 10

            # MACD評価
            if 'MACD' in trend and 'MACD_Signal' in trend:
                if trend['MACD'] > trend['MACD_Signal']:
                    score += 8
                else:
                    score -= 8

            # RSI評価
            if 'RSI_14' in momentum:
                rsi = momentum['RSI_14']
                if 30 <= rsi <= 70:
                    score += 5  # 適正レンジ
                elif rsi > 80:
                    score -= 10  # 買われすぎ
                elif rsi < 20:
                    score += 10  # 売られすぎ（反転期待）

            # ボラティリティ評価
            if 'BB_Position' in volatility:
                bb_pos = volatility['BB_Position']
                if 20 <= bb_pos <= 80:
                    score += 3  # 適正範囲
                else:
                    score -= 3  # 極端な位置

            return max(0, min(100, score))

        except Exception as e:
            self.logger.error(f"Composite score calculation error: {e}")
            return 50.0