#!/usr/bin/env python3
"""
Next-Gen AI Trading - Sentiment Analysis Module
センチメント分析モジュール

FinBERT・ニュース分析・市場心理指標・ソーシャルメディア感情解析
"""

from .news_analyzer import (
    NewsAnalyzer,
    NewsArticle,
    NewsConfig,
    NewsSentimentResult,
    NewsSource,
)
from .sentiment_engine import (
    MarketSentimentIndicator,
    SentimentConfig,
    SentimentEngine,
    SentimentResult,
    create_sentiment_engine,
)
from .social_analyzer import (
    SocialConfig,
    SocialMediaAnalyzer,
    SocialPost,
    SocialSentimentResult,
)

__all__ = [
    # センチメントエンジン
    "SentimentEngine",
    "SentimentConfig",
    "SentimentResult",
    "MarketSentimentIndicator",
    "create_sentiment_engine",
    # ニュース分析
    "NewsAnalyzer",
    "NewsConfig",
    "NewsSource",
    "NewsArticle",
    "NewsSentimentResult",
    # ソーシャル分析
    "SocialMediaAnalyzer",
    "SocialConfig",
    "SocialPost",
    "SocialSentimentResult",
]
