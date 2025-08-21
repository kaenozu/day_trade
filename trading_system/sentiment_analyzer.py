import logging
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

from .datastructures import MarketSentiment


class AdvancedSentimentAnalyzer:
    """高度センチメント分析"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def analyze_market_sentiment(self, symbol: str, market_data: pd.DataFrame) -> MarketSentiment:
        """総合市場センチメント分析"""
        try:
            # テクニカルセンチメント
            technical_sentiment = self._analyze_technical_sentiment(market_data)

            # ファンダメンタルセンチメント（模擬）
            fundamental_sentiment = self._analyze_fundamental_sentiment(symbol)

            # ニュースセンチメント（模擬）
            news_sentiment = await self._analyze_news_sentiment(symbol)

            # 総合センチメント計算
            weights = {'technical': 0.4, 'fundamental': 0.3, 'news': 0.3}
            sentiment_score = (
                technical_sentiment * weights['technical'] +
                fundamental_sentiment * weights['fundamental'] +
                news_sentiment * weights['news']
            )

            # 信頼度計算
            confidence = self._calculate_sentiment_confidence(
                technical_sentiment, fundamental_sentiment, news_sentiment
            )

            # 主要要因特定
            key_factors = self._identify_key_factors(
                technical_sentiment, fundamental_sentiment, news_sentiment
            )

            return MarketSentiment(
                sentiment_score=sentiment_score,
                confidence=confidence,
                key_factors=key_factors,
                news_sentiment=news_sentiment,
                technical_sentiment=technical_sentiment,
                fundamental_sentiment=fundamental_sentiment,
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return MarketSentiment(
                sentiment_score=0.0,
                confidence=0.5,
                key_factors=['分析エラー'],
                news_sentiment=0.0,
                technical_sentiment=0.0,
                fundamental_sentiment=0.0,
                timestamp=datetime.now()
            )

    def _analyze_technical_sentiment(self, data: pd.DataFrame) -> float:
        """テクニカルセンチメント分析"""
        if data.empty or len(data) < 20:
            return 0.0

        sentiment_signals = []

        try:
            # RSI分析
            rsi = self._calculate_rsi(data['Close'], 14)
            if len(rsi) > 0:
                rsi_latest = rsi.iloc[-1]
                if rsi_latest > 70:
                    sentiment_signals.append(-0.3)  # 売られすぎ
                elif rsi_latest < 30:
                    sentiment_signals.append(0.3)   # 買われすぎ
                else:
                    sentiment_signals.append((rsi_latest - 50) / 100)

            # 移動平均分析
            if 'Close' in data.columns and len(data) >= 20:
                sma_20 = data['Close'].rolling(20).mean().iloc[-1]
                current_price = data['Close'].iloc[-1]
                ma_signal = (current_price - sma_20) / sma_20
                sentiment_signals.append(np.clip(ma_signal, -1, 1))

            # ボリューム分析
            if 'Volume' in data.columns and len(data) >= 10:
                avg_volume = data['Volume'].rolling(10).mean().iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume
                if volume_ratio > 1.5:
                    sentiment_signals.append(0.2)
                elif volume_ratio < 0.5:
                    sentiment_signals.append(-0.2)

            # MACD分析
            macd, macd_signal = self._calculate_macd(data['Close'])
            if len(macd) > 0 and len(macd_signal) > 0:
                macd_diff = macd.iloc[-1] - macd_signal.iloc[-1]
                sentiment_signals.append(np.clip(macd_diff / 10, -1, 1))

            # 総合センチメント
            if sentiment_signals:
                return np.clip(np.mean(sentiment_signals), -1, 1)
            else:
                return 0.0

        except Exception as e:
            self.logger.warning(f"Technical sentiment analysis error: {e}")
            return 0.0

    def _analyze_fundamental_sentiment(self, symbol: str) -> float:
        """ファンダメンタルセンチメント分析（模擬）"""
        # 実装では、企業の財務指標、業績予想、格付け変更などを分析
        # ここでは模擬的な分析を実装
        np.random.seed(hash(symbol) % 2**32)

        # 業種別セクター強度（模擬）
        sector_strength = np.random.uniform(-0.3, 0.3)

        # PER/PBR分析（模擬）
        valuation_score = np.random.uniform(-0.2, 0.2)

        # 業績トレンド（模擬）
        earnings_trend = np.random.uniform(-0.2, 0.2)

        fundamental_sentiment = sector_strength + valuation_score + earnings_trend
        return np.clip(fundamental_sentiment, -1, 1)

    async def _analyze_news_sentiment(self, symbol: str) -> float:
        """ニュースセンチメント分析（模擬）"""
        # 実装では、ニュースAPIから関連ニュースを取得し、
        # 自然言語処理でセンチメントスコアを算出
        # ここでは模擬的な分析を実装

        # シンボルベースの模擬ニュースセンチメント
        np.random.seed((hash(symbol) + int(datetime.now().timestamp())) % 2**32)

        # 最近のニュースセンチメント（模擬）
        recent_news_sentiment = np.random.uniform(-0.5, 0.5)

        # ニュース量による信頼度調整
        news_volume_factor = np.random.uniform(0.7, 1.0)

        return recent_news_sentiment * news_volume_factor

    def _calculate_sentiment_confidence(self, technical: float, fundamental: float, news: float) -> float:
        """センチメント信頼度計算"""
        # センチメント間の一致度を基に信頼度を算出
        sentiments = [technical, fundamental, news]

        # 標準偏差が小さいほど一致度が高い
        std_dev = np.std(sentiments)
        confidence = max(0.3, 1.0 - std_dev)

        # 絶対値の平均が大きいほど信頼度が高い
        abs_mean = np.mean([abs(s) for s in sentiments])
        confidence *= (0.5 + abs_mean * 0.5)

        return np.clip(confidence, 0.0, 1.0)

    def _identify_key_factors(self, technical: float, fundamental: float, news: float) -> List[str]:
        """主要要因特定"""
        factors = []

        # 最も影響の大きい要因を特定
        abs_values = {
            'テクニカル要因': abs(technical),
            'ファンダメンタル要因': abs(fundamental),
            'ニュース要因': abs(news)
        }

        # 影響度順にソート
        sorted_factors = sorted(abs_values.items(), key=lambda x: x[1], reverse=True)

        for factor, impact in sorted_factors:
            if impact > 0.1:  # 十分な影響度がある場合のみ
                factors.append(factor)

        if not factors:
            factors.append('明確な主導要因なし')

        return factors[:3]  # 上位3要因まで

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD計算"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
