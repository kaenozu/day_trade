#!/usr/bin/env python3
"""
投資機会アラートシステム - 機会検出器
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger
from .models import InvestmentOpportunity, OpportunityConfig, MarketCondition

logger = get_context_logger(__name__)


class OpportunityDetector:
    """投資機会検出器"""

    def __init__(self):
        """機会検出器の初期化"""
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.technical_indicators: Dict[str, Dict[str, float]] = {}
        self.market_condition: Optional[MarketCondition] = None

    async def update_market_condition(self) -> None:
        """市場状況更新"""
        # 模擬的な市場状況データ
        market_trends = ["bull", "bear", "sideways"]
        volatility_levels = ["low", "medium", "high"]
        volume_trends = ["increasing", "decreasing", "stable"]

        # 市場センチメント（-1から1の間）
        market_sentiment = random.uniform(-0.5, 0.8)

        # セクターパフォーマンス
        sectors = ["technology", "finance", "healthcare", "energy", "consumer"]
        sector_performance = {sector: random.uniform(-3.0, 5.0) for sector in sectors}

        self.market_condition = MarketCondition(
            timestamp=datetime.now(),
            market_trend=random.choice(market_trends),
            volatility_level=random.choice(volatility_levels),
            volume_trend=random.choice(volume_trends),
            sector_performance=sector_performance,
            market_sentiment=market_sentiment,
            fear_greed_index=random.uniform(20, 80),
        )

    async def update_price_data(self) -> None:
        """価格データ更新"""
        # 模擬的な価格データ生成
        symbols = [
            "TOPIX",
            "N225",
            "7203",
            "8306",
            "9984",
            "4502",
            "7182",
            "6501",
            "8058",
        ]

        for symbol in symbols:
            # 過去30日分の価格データ生成
            dates = pd.date_range(end=datetime.now(), periods=30, freq="D")

            # ランダムウォーク価格生成
            base_price = 1000 + hash(symbol) % 5000
            returns = np.random.normal(0.001, 0.02, 30)  # 平均0.1%、標準偏差2%
            prices = [base_price]

            for return_rate in returns[1:]:
                prices.append(prices[-1] * (1 + return_rate))

            # OHLCV データ生成
            data = []
            for i, (date, close) in enumerate(zip(dates, prices)):
                open_price = close * random.uniform(0.99, 1.01)
                high = max(open_price, close) * random.uniform(1.0, 1.02)
                low = min(open_price, close) * random.uniform(0.98, 1.0)
                volume = random.randint(100000, 1000000)

                data.append(
                    {
                        "Date": date,
                        "Open": open_price,
                        "High": high,
                        "Low": low,
                        "Close": close,
                        "Volume": volume,
                    }
                )

            self.price_data[symbol] = pd.DataFrame(data).set_index("Date")

    async def calculate_technical_indicators(self) -> None:
        """テクニカル指標計算"""
        for symbol, df in self.price_data.items():
            if len(df) < 20:  # 最低限必要なデータポイント
                continue

            try:
                indicators = {}

                # 移動平均
                indicators["SMA_5"] = df["Close"].rolling(5).mean().iloc[-1]
                indicators["SMA_20"] = df["Close"].rolling(20).mean().iloc[-1]
                indicators["EMA_12"] = df["Close"].ewm(span=12).mean().iloc[-1]
                indicators["EMA_26"] = df["Close"].ewm(span=26).mean().iloc[-1]

                # RSI
                delta = df["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                indicators["RSI"] = (100 - (100 / (1 + rs))).iloc[-1]

                # MACD
                indicators["MACD"] = indicators["EMA_12"] - indicators["EMA_26"]
                macd_signal = pd.Series([indicators["MACD"]]).ewm(span=9).mean()
                indicators["MACD_Signal"] = macd_signal.iloc[-1]
                indicators["MACD_Histogram"] = (
                    indicators["MACD"] - indicators["MACD_Signal"]
                )

                # ボリンジャーバンド
                bb_period = 20
                sma = df["Close"].rolling(bb_period).mean()
                std = df["Close"].rolling(bb_period).std()
                indicators["BB_Upper"] = (sma + 2 * std).iloc[-1]
                indicators["BB_Lower"] = (sma - 2 * std).iloc[-1]
                indicators["BB_Middle"] = sma.iloc[-1]

                # 現在価格のボリンジャーバンド位置
                current_price = df["Close"].iloc[-1]
                bb_width = indicators["BB_Upper"] - indicators["BB_Lower"]
                if bb_width > 0:
                    indicators["BB_Position"] = (
                        current_price - indicators["BB_Lower"]
                    ) / bb_width
                else:
                    indicators["BB_Position"] = 0.5

                # 出来高指標
                indicators["Volume_SMA"] = df["Volume"].rolling(20).mean().iloc[-1]
                indicators["Volume_Ratio"] = (
                    df["Volume"].iloc[-1] / indicators["Volume_SMA"]
                )

                # 価格変化率
                indicators["Price_Change_1D"] = (
                    df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1
                ) * 100
                indicators["Price_Change_5D"] = (
                    df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1
                ) * 100

                self.technical_indicators[symbol] = indicators

            except Exception as e:
                logger.error(f"テクニカル指標計算エラー {symbol}: {e}")

    async def detect_opportunities(self, 
                                 opportunity_configs: Dict[str, OpportunityConfig]) -> List[InvestmentOpportunity]:
        """投資機会検出"""
        opportunities = []

        for rule_id, config in opportunity_configs.items():
            if not config.enabled:
                continue

            try:
                # 機会分析実行
                config_opportunities = await self._analyze_opportunity(config)
                opportunities.extend(config_opportunities)

            except Exception as e:
                logger.error(f"機会検出エラー {rule_id}: {e}")

        return opportunities

    async def _analyze_opportunity(self, config: OpportunityConfig) -> List[InvestmentOpportunity]:
        """機会分析"""
        opportunities = []

        # 対象銘柄取得
        target_symbols = config.symbols
        if "*" in target_symbols:
            target_symbols = list(self.price_data.keys())

        for symbol in target_symbols:
            if symbol not in self.technical_indicators:
                continue

            try:
                opportunity = await self._analyze_symbol_opportunity(symbol, config)
                if opportunity:
                    opportunities.append(opportunity)
            except Exception as e:
                logger.error(f"銘柄機会分析エラー {symbol}: {e}")

        return opportunities

    async def _analyze_symbol_opportunity(
        self,
        symbol: str,
        config: OpportunityConfig,
    ) -> Optional[InvestmentOpportunity]:
        """銘柄機会分析"""
        indicators = self.technical_indicators.get(symbol, {})
        price_data = self.price_data.get(symbol)

        if not indicators or price_data is None or len(price_data) == 0:
            return None

        current_price = price_data["Close"].iloc[-1]

        # 機会タイプ別の分析を実行
        from .analyzers import OpportunityAnalyzer
        analyzer = OpportunityAnalyzer()
        
        return await analyzer.analyze_opportunity(
            symbol, config, indicators, current_price
        )