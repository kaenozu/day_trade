#!/usr/bin/env python3
"""
Next-Gen AI News Analyzer
高度ニュース感情分析システム

多言語ニュース収集・感情解析・重要度判定・リアルタイム処理
"""

import time
import asyncio
import warnings
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse

# ウェブスクレイピング・API
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False

try:
    import feedparser
    RSS_AVAILABLE = True
except ImportError:
    RSS_AVAILABLE = False

# NewsAPI対応
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

from .sentiment_engine import SentimentEngine, SentimentResult, create_sentiment_engine
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class NewsConfig:
    """ニュース分析設定"""
    # API設定
    newsapi_key: Optional[str] = None
    max_articles_per_source: int = 50

    # ソース設定
    enabled_sources: List[str] = field(default_factory=lambda: [
        "reuters", "bloomberg", "financial-times", "wall-street-journal",
        "cnbc", "marketwatch", "yahoo-finance", "seeking-alpha"
    ])

    # RSS設定
    rss_feeds: List[str] = field(default_factory=lambda: [
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://rss.cnn.com/rss/money_latest.rss",
        "https://feeds.finance.yahoo.com/rss/2.0/headlines"
    ])

    # 検索設定
    search_keywords: List[str] = field(default_factory=lambda: [
        "stock market", "financial market", "investment", "trading",
        "earnings", "economic", "finance", "market analysis"
    ])

    # フィルタリング設定
    min_relevance_score: float = 0.3
    language: str = "en"
    max_age_hours: int = 24

    # 処理設定
    batch_size: int = 20
    request_delay: float = 1.0  # レート制限対応
    timeout: int = 10

@dataclass
class NewsSource:
    """ニュースソース定義"""
    name: str
    url: str
    source_type: str  # "rss", "api", "scrape"
    reliability_score: float = 1.0
    language: str = "en"
    update_frequency: int = 60  # 分

@dataclass
class NewsArticle:
    """ニュース記事"""
    title: str
    content: str
    url: str
    source: str
    author: Optional[str] = None
    published_at: Optional[datetime] = None

    # 分析結果
    sentiment_result: Optional[SentimentResult] = None
    relevance_score: float = 0.0
    importance_score: float = 0.0
    keywords: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)

    # メタデータ
    fetched_at: datetime = field(default_factory=datetime.now)
    language: str = "en"

@dataclass
class NewsSentimentResult:
    """ニュースセンチメント分析結果"""
    articles: List[NewsArticle]
    overall_sentiment: float
    sentiment_distribution: Dict[str, int]
    top_keywords: List[Tuple[str, int]]
    source_breakdown: Dict[str, Dict[str, Any]]
    time_analysis: Dict[str, float]
    confidence_score: float
    analysis_timestamp: datetime = field(default_factory=datetime.now)

class NewsAnalyzer:
    """高度ニュース分析システム"""

    def __init__(self, config: Optional[NewsConfig] = None):
        self.config = config or NewsConfig()

        # センチメントエンジン
        self.sentiment_engine = create_sentiment_engine()

        # NewsAPI初期化
        self.newsapi = None
        if NEWSAPI_AVAILABLE and self.config.newsapi_key:
            try:
                self.newsapi = NewsApiClient(api_key=self.config.newsapi_key)
            except Exception as e:
                logger.warning(f"NewsAPI初期化失敗: {e}")

        # 記事キャッシュ
        self.article_cache = {}
        self.processed_urls = set()

        # パフォーマンス統計
        self.fetch_stats = {
            "total_fetched": 0,
            "successful_fetches": 0,
            "failed_fetches": 0,
            "cache_hits": 0
        }

        logger.info("News Analyzer 初期化完了")

    async def fetch_news(self,
                        keywords: List[str] = None,
                        sources: List[str] = None,
                        hours_back: int = None) -> List[NewsArticle]:
        """ニュース記事取得"""

        keywords = keywords or self.config.search_keywords
        sources = sources or self.config.enabled_sources
        hours_back = hours_back or self.config.max_age_hours

        logger.info(f"ニュース取得開始: {len(keywords)} キーワード, {len(sources)} ソース")

        all_articles = []

        # 並行取得
        tasks = []

        # NewsAPI経由
        if self.newsapi:
            tasks.append(self._fetch_from_newsapi(keywords, sources, hours_back))

        # RSS Feed経由
        if RSS_AVAILABLE:
            for feed_url in self.config.rss_feeds:
                tasks.append(self._fetch_from_rss(feed_url))

        # 実行
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"記事取得エラー: {result}")

        # 重複除去
        unique_articles = self._deduplicate_articles(all_articles)

        # フィルタリング
        filtered_articles = self._filter_articles(unique_articles, keywords)

        logger.info(f"ニュース取得完了: {len(filtered_articles)} 記事")

        return filtered_articles

    async def _fetch_from_newsapi(self, keywords: List[str], sources: List[str], hours_back: int) -> List[NewsArticle]:
        """NewsAPI経由での記事取得"""

        if not self.newsapi:
            return []

        articles = []

        try:
            # 日時範囲設定
            from_date = datetime.now() - timedelta(hours=hours_back)

            for keyword in keywords[:3]:  # レート制限対応で3キーワードまで
                try:
                    # 記事検索
                    response = self.newsapi.get_everything(
                        q=keyword,
                        sources=','.join(sources[:10]),  # 最大10ソース
                        from_param=from_date.strftime('%Y-%m-%d'),
                        language=self.config.language,
                        sort_by='publishedAt',
                        page_size=self.config.max_articles_per_source
                    )

                    if response['status'] == 'ok':
                        for article_data in response['articles']:
                            article = self._parse_newsapi_article(article_data, keyword)
                            if article:
                                articles.append(article)

                    # レート制限対応
                    await asyncio.sleep(self.config.request_delay)

                except Exception as e:
                    logger.error(f"NewsAPI検索エラー ({keyword}): {e}")

        except Exception as e:
            logger.error(f"NewsAPI全体エラー: {e}")

        return articles

    async def _fetch_from_rss(self, feed_url: str) -> List[NewsArticle]:
        """RSS Feed経由での記事取得"""

        if not RSS_AVAILABLE:
            return []

        articles = []

        try:
            logger.debug(f"RSS取得: {feed_url}")

            # フィード解析
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                logger.warning(f"RSS解析警告: {feed_url}")

            for entry in feed.entries[:self.config.max_articles_per_source]:
                article = self._parse_rss_entry(entry, feed_url)
                if article:
                    articles.append(article)

        except Exception as e:
            logger.error(f"RSS取得エラー ({feed_url}): {e}")

        return articles

    def _parse_newsapi_article(self, article_data: Dict, keyword: str) -> Optional[NewsArticle]:
        """NewsAPI記事データ解析"""

        try:
            # 必須フィールドチェック
            if not article_data.get('title') or not article_data.get('url'):
                return None

            # 重複チェック
            url = article_data['url']
            if url in self.processed_urls:
                return None

            # 日時解析
            published_at = None
            if article_data.get('publishedAt'):
                try:
                    published_at = datetime.fromisoformat(
                        article_data['publishedAt'].replace('Z', '+00:00')
                    )
                except:
                    pass

            # 記事作成
            article = NewsArticle(
                title=article_data.get('title', ''),
                content=article_data.get('description', '') + ' ' + article_data.get('content', ''),
                url=url,
                source=article_data.get('source', {}).get('name', 'unknown'),
                author=article_data.get('author'),
                published_at=published_at,
                keywords=[keyword]
            )

            self.processed_urls.add(url)
            return article

        except Exception as e:
            logger.error(f"NewsAPI記事解析エラー: {e}")
            return None

    def _parse_rss_entry(self, entry, feed_url: str) -> Optional[NewsArticle]:
        """RSS記事解析"""

        try:
            # 必須フィールドチェック
            if not hasattr(entry, 'title') or not hasattr(entry, 'link'):
                return None

            # 重複チェック
            url = entry.link
            if url in self.processed_urls:
                return None

            # 日時解析
            published_at = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                try:
                    published_at = datetime(*entry.published_parsed[:6])
                except:
                    pass

            # コンテンツ抽出
            content = ""
            if hasattr(entry, 'summary'):
                content = entry.summary
            elif hasattr(entry, 'description'):
                content = entry.description

            # HTMLタグ除去
            if content:
                content = re.sub(r'<[^>]+>', '', content)

            # ソース名抽出
            source_name = urlparse(feed_url).netloc

            article = NewsArticle(
                title=entry.title,
                content=content,
                url=url,
                source=source_name,
                published_at=published_at
            )

            self.processed_urls.add(url)
            return article

        except Exception as e:
            logger.error(f"RSS記事解析エラー: {e}")
            return None

    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """記事重複除去"""

        seen_urls = set()
        seen_titles = set()
        unique_articles = []

        for article in articles:
            # URL重複チェック
            if article.url in seen_urls:
                continue

            # タイトル類似度チェック（簡易）
            title_key = re.sub(r'[^\w\s]', '', article.title.lower())
            title_words = set(title_key.split())

            is_similar = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                if title_words and seen_words:
                    jaccard = len(title_words & seen_words) / len(title_words | seen_words)
                    if jaccard > 0.8:  # 80%以上類似
                        is_similar = True
                        break

            if not is_similar:
                unique_articles.append(article)
                seen_urls.add(article.url)
                seen_titles.add(title_key)

        return unique_articles

    def _filter_articles(self, articles: List[NewsArticle], keywords: List[str]) -> List[NewsArticle]:
        """記事フィルタリング"""

        filtered = []

        for article in articles:
            # 年齢フィルター
            if article.published_at:
                age = datetime.now() - article.published_at.replace(tzinfo=None)
                if age.total_seconds() > self.config.max_age_hours * 3600:
                    continue

            # 関連性スコア計算
            relevance = self._calculate_relevance(article, keywords)
            article.relevance_score = relevance

            if relevance >= self.config.min_relevance_score:
                filtered.append(article)

        # 関連性順でソート
        filtered.sort(key=lambda a: a.relevance_score, reverse=True)

        return filtered

    def _calculate_relevance(self, article: NewsArticle, keywords: List[str]) -> float:
        """関連性スコア計算"""

        text = (article.title + " " + article.content).lower()

        # キーワードマッチング
        keyword_matches = 0
        for keyword in keywords:
            if keyword.lower() in text:
                keyword_matches += 1

        # 金融関連キーワード
        financial_keywords = [
            'stock', 'market', 'trading', 'investment', 'finance', 'economic',
            'earnings', 'profit', 'revenue', 'growth', 'decline', 'bull', 'bear'
        ]

        financial_matches = 0
        for word in financial_keywords:
            if word in text:
                financial_matches += 1

        # スコア計算
        keyword_score = min(keyword_matches / max(len(keywords), 1), 1.0)
        financial_score = min(financial_matches / len(financial_keywords), 1.0)

        # 重み付き合計
        relevance = keyword_score * 0.7 + financial_score * 0.3

        return relevance

    def analyze_articles(self, articles: List[NewsArticle]) -> NewsSentimentResult:
        """記事センチメント分析"""

        if not articles:
            return self._create_empty_result()

        logger.info(f"記事分析開始: {len(articles)} 記事")

        # センチメント分析実行
        analyzed_articles = []
        sentiments = []

        for article in articles:
            # タイトルとコンテンツの統合
            full_text = f"{article.title}. {article.content}"

            # センチメント分析
            sentiment_result = self.sentiment_engine.analyze_text(full_text)
            article.sentiment_result = sentiment_result

            # 重要度スコア計算
            article.importance_score = self._calculate_importance(article)

            # キーワード抽出
            article.keywords = self._extract_keywords(full_text)

            analyzed_articles.append(article)
            sentiments.append(sentiment_result.sentiment_score)

        # 全体分析
        overall_sentiment = np.mean(sentiments) if sentiments else 0.0

        # センチメント分布
        sentiment_distribution = {
            'positive': len([s for s in analyzed_articles if s.sentiment_result.sentiment_label == 'positive']),
            'negative': len([s for s in analyzed_articles if s.sentiment_result.sentiment_label == 'negative']),
            'neutral': len([s for s in analyzed_articles if s.sentiment_result.sentiment_label == 'neutral'])
        }

        # キーワード集計
        all_keywords = []
        for article in analyzed_articles:
            all_keywords.extend(article.keywords)

        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        # ソース別分析
        source_breakdown = self._analyze_by_source(analyzed_articles)

        # 時系列分析
        time_analysis = self._analyze_by_time(analyzed_articles)

        # 信頼度スコア
        confidences = [a.sentiment_result.confidence for a in analyzed_articles]
        confidence_score = np.mean(confidences) if confidences else 0.0

        result = NewsSentimentResult(
            articles=analyzed_articles,
            overall_sentiment=overall_sentiment,
            sentiment_distribution=sentiment_distribution,
            top_keywords=top_keywords,
            source_breakdown=source_breakdown,
            time_analysis=time_analysis,
            confidence_score=confidence_score
        )

        logger.info(f"記事分析完了: 全体センチメント={overall_sentiment:.3f}")

        return result

    def _calculate_importance(self, article: NewsArticle) -> float:
        """記事重要度スコア計算"""

        importance = 0.0

        # センチメント強度
        if article.sentiment_result:
            importance += abs(article.sentiment_result.sentiment_score) * 0.3

        # 関連性スコア
        importance += article.relevance_score * 0.4

        # ソース信頼性（簡易）
        source_weights = {
            'reuters': 1.0,
            'bloomberg': 1.0,
            'financial-times': 0.9,
            'wall-street-journal': 0.9,
            'cnbc': 0.8,
            'marketwatch': 0.7
        }

        source_weight = source_weights.get(article.source.lower(), 0.5)
        importance += source_weight * 0.2

        # 新しさ
        if article.published_at:
            hours_old = (datetime.now() - article.published_at.replace(tzinfo=None)).total_seconds() / 3600
            freshness = max(0, 1 - hours_old / 24)  # 24時間で線形減衰
            importance += freshness * 0.1

        return min(importance, 1.0)

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """キーワード抽出（簡易）"""

        # 金融関連キーワード
        financial_terms = [
            'stock', 'market', 'trading', 'investment', 'earnings', 'profit',
            'revenue', 'growth', 'decline', 'bull', 'bear', 'volatility',
            'portfolio', 'dividend', 'bond', 'currency', 'commodity'
        ]

        text_lower = text.lower()
        found_keywords = []

        for term in financial_terms:
            if term in text_lower:
                found_keywords.append(term)

        # 企業名抽出（大文字で始まる単語）
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        for word in words[:5]:  # 最初の5つまで
            if len(word) > 3 and word.lower() not in ['the', 'and', 'but', 'for']:
                found_keywords.append(word.lower())

        return found_keywords[:max_keywords]

    def _analyze_by_source(self, articles: List[NewsArticle]) -> Dict[str, Dict[str, Any]]:
        """ソース別分析"""

        source_data = {}

        for article in articles:
            source = article.source
            if source not in source_data:
                source_data[source] = {
                    'count': 0,
                    'sentiments': [],
                    'importance_scores': []
                }

            source_data[source]['count'] += 1
            if article.sentiment_result:
                source_data[source]['sentiments'].append(article.sentiment_result.sentiment_score)
            source_data[source]['importance_scores'].append(article.importance_score)

        # 統計計算
        for source, data in source_data.items():
            data['avg_sentiment'] = np.mean(data['sentiments']) if data['sentiments'] else 0.0
            data['avg_importance'] = np.mean(data['importance_scores'])
            data['sentiment_std'] = np.std(data['sentiments']) if len(data['sentiments']) > 1 else 0.0

        return source_data

    def _analyze_by_time(self, articles: List[NewsArticle]) -> Dict[str, float]:
        """時系列分析"""

        # 時間帯別分析
        hourly_sentiments = {}

        for article in articles:
            if article.published_at and article.sentiment_result:
                hour = article.published_at.hour
                if hour not in hourly_sentiments:
                    hourly_sentiments[hour] = []
                hourly_sentiments[hour].append(article.sentiment_result.sentiment_score)

        # 平均計算
        hourly_averages = {}
        for hour, sentiments in hourly_sentiments.items():
            hourly_averages[f'hour_{hour:02d}'] = np.mean(sentiments)

        return hourly_averages

    def _create_empty_result(self) -> NewsSentimentResult:
        """空の結果作成"""
        return NewsSentimentResult(
            articles=[],
            overall_sentiment=0.0,
            sentiment_distribution={'positive': 0, 'negative': 0, 'neutral': 0},
            top_keywords=[],
            source_breakdown={},
            time_analysis={},
            confidence_score=0.0
        )

    def get_trending_topics(self, articles: List[NewsArticle], limit: int = 10) -> List[Dict[str, Any]]:
        """トレンディングトピック抽出"""

        if not articles:
            return []

        # キーワード重み付き集計
        keyword_data = {}

        for article in articles:
            for keyword in article.keywords:
                if keyword not in keyword_data:
                    keyword_data[keyword] = {
                        'count': 0,
                        'total_importance': 0.0,
                        'sentiments': [],
                        'articles': []
                    }

                keyword_data[keyword]['count'] += 1
                keyword_data[keyword]['total_importance'] += article.importance_score
                if article.sentiment_result:
                    keyword_data[keyword]['sentiments'].append(article.sentiment_result.sentiment_score)
                keyword_data[keyword]['articles'].append(article.title)

        # トレンドスコア計算
        trending_topics = []
        for keyword, data in keyword_data.items():
            if data['count'] >= 2:  # 最低2記事
                trend_score = (data['count'] * data['total_importance'] / data['count'])
                avg_sentiment = np.mean(data['sentiments']) if data['sentiments'] else 0.0

                trending_topics.append({
                    'keyword': keyword,
                    'trend_score': trend_score,
                    'article_count': data['count'],
                    'avg_sentiment': avg_sentiment,
                    'avg_importance': data['total_importance'] / data['count'],
                    'sample_articles': data['articles'][:3]
                })

        # スコア順ソート
        trending_topics.sort(key=lambda x: x['trend_score'], reverse=True)

        return trending_topics[:limit]

    def export_analysis(self, result: NewsSentimentResult, format: str = "json") -> str:
        """分析結果エクスポート"""

        if format == "json":
            # JSON形式
            export_data = {
                "analysis_timestamp": result.analysis_timestamp.isoformat(),
                "overall_sentiment": result.overall_sentiment,
                "confidence_score": result.confidence_score,
                "total_articles": len(result.articles),
                "sentiment_distribution": result.sentiment_distribution,
                "top_keywords": result.top_keywords,
                "source_breakdown": result.source_breakdown,
                "articles": [
                    {
                        "title": article.title,
                        "source": article.source,
                        "sentiment_label": article.sentiment_result.sentiment_label if article.sentiment_result else None,
                        "sentiment_score": article.sentiment_result.sentiment_score if article.sentiment_result else 0,
                        "importance_score": article.importance_score,
                        "url": article.url
                    }
                    for article in result.articles
                ]
            }

            return json.dumps(export_data, indent=2, ensure_ascii=False)

        elif format == "csv":
            # CSV形式（記事一覧）
            df_data = []
            for article in result.articles:
                df_data.append({
                    "title": article.title,
                    "source": article.source,
                    "published_at": article.published_at.isoformat() if article.published_at else "",
                    "sentiment_label": article.sentiment_result.sentiment_label if article.sentiment_result else "",
                    "sentiment_score": article.sentiment_result.sentiment_score if article.sentiment_result else 0,
                    "confidence": article.sentiment_result.confidence if article.sentiment_result else 0,
                    "importance_score": article.importance_score,
                    "relevance_score": article.relevance_score,
                    "url": article.url
                })

            df = pd.DataFrame(df_data)
            return df.to_csv(index=False)

        else:
            raise ValueError(f"未対応のエクスポート形式: {format}")

# 便利関数
def analyze_financial_news(keywords: List[str] = None, hours_back: int = 24) -> NewsSentimentResult:
    """金融ニュース分析（簡易インターフェース）"""

    async def _analyze():
        analyzer = NewsAnalyzer()
        articles = await analyzer.fetch_news(keywords=keywords, hours_back=hours_back)
        return analyzer.analyze_articles(articles)

    return asyncio.run(_analyze())

if __name__ == "__main__":
    # ニュース分析テスト
    print("=== Next-Gen AI News Analyzer テスト ===")

    # 分析実行（非同期）
    async def test_news_analysis():
        analyzer = NewsAnalyzer()

        # テスト用キーワード
        test_keywords = ["stock market", "technology stocks", "AI investment"]

        print(f"ニュース取得テスト: {test_keywords}")

        # 記事取得
        articles = await analyzer.fetch_news(keywords=test_keywords, hours_back=24)
        print(f"取得記事数: {len(articles)}")

        if articles:
            # 分析実行
            result = analyzer.analyze_articles(articles[:10])  # 最初の10記事

            print(f"\n分析結果:")
            print(f"全体センチメント: {result.overall_sentiment:.3f}")
            print(f"信頼度: {result.confidence_score:.3f}")
            print(f"センチメント分布: {result.sentiment_distribution}")
            print(f"上位キーワード: {result.top_keywords[:5]}")

            # トレンディングトピック
            trending = analyzer.get_trending_topics(articles[:10])
            print(f"\nトレンディングトピック:")
            for topic in trending[:3]:
                print(f"  {topic['keyword']}: スコア={topic['trend_score']:.3f}, 記事数={topic['article_count']}")

            # エクスポートテスト
            json_export = analyzer.export_analysis(result, "json")
            print(f"\nJSON エクスポート長: {len(json_export)} 文字")

        else:
            print("記事が取得できませんでした（テストモード）")

    # テスト実行
    try:
        asyncio.run(test_news_analysis())
    except Exception as e:
        print(f"テストエラー: {e}")

    print("\n=== テスト完了 ===")
