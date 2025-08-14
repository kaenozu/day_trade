#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider - 実戦投入用リアルデータ取得システム

Yahoo Finance APIを使用して実際の株価データを取得
ダミーデータから実データへの完全移行
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import time as time_module

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Windows Console API対応
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

@dataclass
class RealStockData:
    """実際の株価データ"""
    symbol: str
    current_price: float
    open_price: float
    high_price: float
    low_price: float
    volume: int
    previous_close: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None

    @property
    def change_percent(self) -> float:
        """前日比パーセント"""
        if self.previous_close and self.previous_close > 0:
            return ((self.current_price - self.previous_close) / self.previous_close) * 100
        return 0.0

    @property
    def day_range_percent(self) -> float:
        """当日値幅パーセント"""
        if self.low_price > 0:
            return ((self.high_price - self.low_price) / self.low_price) * 100
        return 0.0

class RealDataProvider:
    """
    実戦投入用リアルデータプロバイダー
    Yahoo Finance APIを使用して実際の株価データを取得
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 対象銘柄（東証コード）
        self.target_symbols = {
            "7203.T": "トヨタ自動車",
            "9984.T": "ソフトバンクG",
            "6758.T": "ソニーG",
            "8306.T": "三菱UFJ",
            "4689.T": "LINEヤフー",
            "8058.T": "三菱商事",
            "6501.T": "日立製作所",
            "4063.T": "信越化学",
            "6098.T": "リクルートHD",
            "9432.T": "NTT"
        }

        # キャッシュ機能（API制限回避）
        self.data_cache = {}
        self.cache_duration = 60  # 1分間キャッシュ

    def get_real_stock_data(self, symbol: str) -> Optional[RealStockData]:
        """
        指定銘柄の実際の株価データを取得

        Args:
            symbol: 銘柄コード（例：7203.T）

        Returns:
            RealStockData: 実際の株価データ
        """
        try:
            # キャッシュチェック
            cache_key = f"{symbol}_{int(time_module.time() / self.cache_duration)}"
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]

            # Yahoo Finance APIでデータ取得
            ticker = yf.Ticker(symbol)

            # 基本情報取得
            info = ticker.info
            hist = ticker.history(period="2d")  # 2日分で前日比計算

            if hist.empty:
                self.logger.warning(f"No data available for {symbol}")
                return None

            # 最新データ取得
            latest = hist.iloc[-1]
            previous = hist.iloc[-2] if len(hist) > 1 else latest

            # RealStockDataオブジェクト作成
            stock_data = RealStockData(
                symbol=symbol,
                current_price=float(latest['Close']),
                open_price=float(latest['Open']),
                high_price=float(latest['High']),
                low_price=float(latest['Low']),
                volume=int(latest['Volume']) if not pd.isna(latest['Volume']) else 0,
                previous_close=float(previous['Close']),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                dividend_yield=info.get('dividendYield')
            )

            # キャッシュ保存
            self.data_cache[cache_key] = stock_data

            return stock_data

        except Exception as e:
            self.logger.error(f"Failed to get real data for {symbol}: {e}")
            return None

    async def get_multiple_stocks_data(self, symbols: List[str] = None) -> Dict[str, RealStockData]:
        """
        複数銘柄の実際の株価データを並行取得

        Args:
            symbols: 銘柄コードリスト（未指定時は全対象銘柄）

        Returns:
            Dict[str, RealStockData]: 銘柄コード -> 株価データ
        """
        if symbols is None:
            symbols = list(self.target_symbols.keys())

        # 並行処理で高速取得
        loop = asyncio.get_event_loop()
        tasks = []

        for symbol in symbols:
            task = loop.run_in_executor(None, self.get_real_stock_data, symbol)
            tasks.append((symbol, task))

        results = {}
        for symbol, task in tasks:
            try:
                data = await task
                if data:
                    results[symbol] = data
            except Exception as e:
                self.logger.error(f"Failed to get data for {symbol}: {e}")

        return results

    def get_market_status(self) -> Dict[str, any]:
        """
        現在の市場状況を取得

        Returns:
            Dict: 市場状況情報
        """
        try:
            # 東証ETFで市場状況判定（先物の代替）
            nikkei_etf = yf.Ticker("1321.T")  # NEXT FUNDS 日経225連動型上場投信
            data = nikkei_etf.history(period="1d")

            if not data.empty:
                latest = data.iloc[-1]
                # 現在時刻による簡易市場判定
                now = datetime.now().time()
                is_open = (time(9, 0) <= now <= time(15, 0) and
                          datetime.now().weekday() < 5)  # 月-金

                return {
                    "market_open": is_open,
                    "nikkei_etf": float(latest['Close']),
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            self.logger.error(f"Failed to get market status: {e}")

        return {
            "market_open": False,
            "nikkei_etf": None,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def calculate_technical_indicators(self, symbol: str, period: int = 20) -> Dict[str, float]:
        """
        テクニカル指標を実データから計算

        Args:
            symbol: 銘柄コード
            period: 計算期間

        Returns:
            Dict: テクニカル指標
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{period + 5}d")  # 余裕をもって取得

            if len(hist) < period:
                return {}

            # 移動平均
            ma5 = hist['Close'].rolling(window=5).mean().iloc[-1]
            ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]

            # ボラティリティ
            volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100  # 年率

            # RSI（簡易版）
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]

            return {
                "ma5": float(ma5) if not pd.isna(ma5) else 0,
                "ma20": float(ma20) if not pd.isna(ma20) else 0,
                "volatility": float(volatility) if not pd.isna(volatility) else 0,
                "rsi": float(rsi) if not pd.isna(rsi) else 50
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate indicators for {symbol}: {e}")
            return {}

class RealDataAnalysisEngine:
    """
    実データに基づく分析エンジン
    従来のダミーデータベース分析から実データベース分析へ移行
    """

    def __init__(self):
        self.data_provider = RealDataProvider()
        self.logger = logging.getLogger(__name__)

    async def analyze_daytrading_opportunities(self, limit: int = 5) -> List[Dict[str, any]]:
        """
        実データに基づくデイトレード機会分析

        Args:
            limit: 推奨銘柄数

        Returns:
            List[Dict]: デイトレード推奨リスト
        """
        try:
            # 実データ取得
            real_data = await self.data_provider.get_multiple_stocks_data()

            if not real_data:
                self.logger.warning("No real data available")
                return []

            recommendations = []

            for symbol, data in real_data.items():
                # テクニカル指標取得
                indicators = self.data_provider.calculate_technical_indicators(symbol)

                # 実データに基づくスコア計算
                score = self._calculate_real_trading_score(data, indicators)

                # 推奨情報作成
                recommendation = {
                    "symbol": symbol.replace('.T', ''),
                    "name": self.data_provider.target_symbols.get(symbol, ""),
                    "current_price": data.current_price,
                    "change_percent": data.change_percent,
                    "volume": data.volume,
                    "volatility": indicators.get("volatility", 0),
                    "rsi": indicators.get("rsi", 50),
                    "ma5": indicators.get("ma5", 0),
                    "ma20": indicators.get("ma20", 0),
                    "trading_score": score,
                    "signal": self._generate_trading_signal(data, indicators, score),
                    "confidence": min(95, max(60, score)),  # 実データベースの信頼度
                    "target_profit": self._calculate_target_profit(data, indicators),
                    "stop_loss": self._calculate_stop_loss(data, indicators)
                }

                recommendations.append(recommendation)

            # スコア順にソート
            recommendations.sort(key=lambda x: x["trading_score"], reverse=True)

            return recommendations[:limit]

        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            return []

    def _calculate_real_trading_score(self, data: RealStockData, indicators: Dict[str, float]) -> float:
        """
        実データに基づく取引スコア計算
        """
        score = 0.0

        # 出来高評価（流動性）
        if data.volume > 1000000:  # 100万株以上
            score += 25
        elif data.volume > 500000:
            score += 15
        elif data.volume > 100000:
            score += 10

        # ボラティリティ評価（デイトレード適性）
        volatility = indicators.get("volatility", 0)
        if 2.0 <= volatility <= 8.0:  # 適度なボラティリティ
            score += 30
        elif volatility > 8.0:  # 高ボラティリティ
            score += 20

        # RSI評価（買われ過ぎ・売られ過ぎ）
        rsi = indicators.get("rsi", 50)
        if 30 <= rsi <= 70:  # 適正範囲
            score += 20
        elif rsi < 30 or rsi > 70:  # 反転期待
            score += 15

        # 移動平均評価（トレンド）
        ma5 = indicators.get("ma5", 0)
        ma20 = indicators.get("ma20", 0)
        if ma5 > ma20 and data.current_price > ma5:  # 上昇トレンド
            score += 15

        # 当日値幅評価
        day_range = data.day_range_percent
        if 1.0 <= day_range <= 5.0:  # 適度な値動き
            score += 10

        return min(100.0, score)

    def _generate_trading_signal(self, data: RealStockData, indicators: Dict[str, float], score: float) -> str:
        """実データに基づく売買シグナル生成"""
        rsi = indicators.get("rsi", 50)
        ma5 = indicators.get("ma5", 0)

        if score >= 80:
            if rsi < 30:
                return "★強い買い★"
            elif data.current_price > ma5:
                return "●買い●"
            else:
                return "△やや買い△"
        elif score >= 60:
            return "…待機…"
        else:
            return "▽売り▽"

    def _calculate_target_profit(self, data: RealStockData, indicators: Dict[str, float]) -> float:
        """利確目標計算"""
        volatility = indicators.get("volatility", 3.0)
        return min(5.0, max(1.5, volatility * 0.8))  # ボラティリティベース

    def _calculate_stop_loss(self, data: RealStockData, indicators: Dict[str, float]) -> float:
        """損切りライン計算"""
        volatility = indicators.get("volatility", 3.0)
        return min(3.0, max(1.0, volatility * 0.5))  # 利確の半分程度


# テスト実行用
async def main():
    """実データプロバイダーのテスト"""
    print("=== 実戦投入テスト: リアルデータ取得 ===")

    provider = RealDataProvider()
    analysis_engine = RealDataAnalysisEngine()

    # 市場状況確認
    market_status = provider.get_market_status()
    print(f"市場状況: {market_status}")

    # 実データ分析実行
    recommendations = await analysis_engine.analyze_daytrading_opportunities(5)

    print("\n=== 実データベース推奨 ===")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['symbol']} ({rec['name']})")
        print(f"   現在価格: {rec['current_price']:.2f}円 ({rec['change_percent']:+.2f}%)")
        print(f"   シグナル: {rec['signal']}")
        print(f"   スコア: {rec['trading_score']:.0f}/100")
        print(f"   出来高: {rec['volume']:,}株")
        print()

if __name__ == "__main__":
    asyncio.run(main())