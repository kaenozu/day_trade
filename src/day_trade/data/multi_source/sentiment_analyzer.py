#!/usr/bin/env python3
"""
多角的データ収集システム - センチメント分析

Issue #322: ML Data Shortage Problem Resolution
センチメント分析器の実装
"""

from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from textblob import TextBlob

from .base import DataCollector
from .models import CollectedData

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class SentimentAnalyzer(DataCollector):
    """センチメント分析器"""

    def __init__(self):
        """初期化"""
        self.analyzer = TextBlob
        self.sentiment_cache = {}

    async def collect_data(self, symbol: str, **kwargs) -> CollectedData:
        """
        センチメントデータ収集

        Args:
            symbol: 銘柄シンボル
            **kwargs: 追加パラメータ（news_dataなど）

        Returns:
            CollectedData: 収集されたセンチメントデータ
        """
        try:
            # ニュースデータからの感情分析
            news_data = kwargs.get("news_data", [])
            sentiment_scores = []

            for article in news_data:
                sentiment = self._analyze_sentiment(
                    article.get("title", "") + " " + article.get("summary", "")
                )
                sentiment_scores.append(sentiment)

            # 総合センチメント算出
            if sentiment_scores:
                avg_sentiment = np.mean([s["compound"] for s in sentiment_scores])
                positive_ratio = np.mean(
                    [1 if s["compound"] > 0.1 else 0 for s in sentiment_scores]
                )
                negative_ratio = np.mean(
                    [1 if s["compound"] < -0.1 else 0 for s in sentiment_scores]
                )
            else:
                avg_sentiment = 0.0
                positive_ratio = 0.0
                negative_ratio = 0.0

            sentiment_data = {
                "overall_sentiment": avg_sentiment,
                "positive_ratio": positive_ratio,
                "negative_ratio": negative_ratio,
                "neutral_ratio": 1.0 - positive_ratio - negative_ratio,
                "article_sentiments": sentiment_scores,
                "confidence": min(
                    len(sentiment_scores) / 10, 1.0
                ),  # 記事数ベース信頼度
            }

            quality_score = self._evaluate_sentiment_quality(sentiment_data)

            return CollectedData(
                symbol=symbol,
                data_type="sentiment",
                data=sentiment_data,
                source="textblob",
                timestamp=datetime.now(),
                quality_score=quality_score,
                confidence=sentiment_data["confidence"],
            )

        except Exception as e:
            logger.error(f"センチメント分析エラー {symbol}: {e}")
            return CollectedData(
                symbol=symbol,
                data_type="sentiment",
                data={"overall_sentiment": 0.0},
                source="textblob",
                timestamp=datetime.now(),
                quality_score=0.0,
            )

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        テキストのセンチメント分析

        Args:
            text: 分析対象テキスト

        Returns:
            Dict[str, float]: センチメントスコア
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
            subjectivity = (
                blob.sentiment.subjectivity
            )  # 0 (objective) to 1 (subjective)

            # VADER風のスコア変換
            return {
                "compound": polarity,
                "positive": max(polarity, 0),
                "negative": abs(min(polarity, 0)),
                "neutral": 1 - abs(polarity),
                "subjectivity": subjectivity,
            }
        except Exception as e:
            logger.error(f"テキスト分析エラー: {e}")
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "subjectivity": 0.5,
            }

    def _evaluate_sentiment_quality(self, sentiment_data: Dict) -> float:
        """
        センチメント品質評価

        Args:
            sentiment_data: センチメントデータ

        Returns:
            float: 品質スコア（0-1）
        """
        confidence = sentiment_data.get("confidence", 0.0)
        article_count = len(sentiment_data.get("article_sentiments", []))

        # 記事数と信頼度ベースの品質スコア
        article_score = min(article_count / 10, 1.0)
        quality_score = confidence * 0.6 + article_score * 0.4

        return quality_score

    def get_health_status(self) -> Dict[str, Any]:
        """
        ヘルス状態取得

        Returns:
            Dict[str, Any]: ヘルス状態情報
        """
        return {
            "collector": "sentiment",
            "status": "active",
            "analyzer": "TextBlob",
            "cache_size": len(self.sentiment_cache),
        }