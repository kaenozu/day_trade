#!/usr/bin/env python3
"""
Next-Gen AI Social Media Analyzer
ソーシャルメディア感情分析システム

Twitter・Reddit・Discord・多言語対応・リアルタイム感情解析
"""

import asyncio
import json
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ソーシャルメディアAPI
try:
    import tweepy

    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False

try:
    import praw  # Reddit

    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

try:
    import discord

    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

# データ処理
try:
    import requests

    WEB_REQUESTS_AVAILABLE = True
except ImportError:
    WEB_REQUESTS_AVAILABLE = False

from ..utils.logging_config import get_context_logger
from .sentiment_engine import SentimentResult, create_sentiment_engine

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class SocialConfig:
    """ソーシャル分析設定"""

    # Twitter API設定
    twitter_bearer_token: Optional[str] = None
    twitter_consumer_key: Optional[str] = None
    twitter_consumer_secret: Optional[str] = None
    twitter_access_token: Optional[str] = None
    twitter_access_token_secret: Optional[str] = None

    # Reddit API設定
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "NextGenAITrader/1.0"

    # Discord設定
    discord_token: Optional[str] = None
    monitored_channels: List[str] = field(default_factory=list)

    # 検索設定
    search_keywords: List[str] = field(
        default_factory=lambda: [
            "$SPY",
            "$QQQ",
            "$AAPL",
            "$MSFT",
            "$TSLA",
            "stock market",
            "trading",
            "investment",
            "crypto",
            "bitcoin",
        ]
    )

    # 収集設定
    max_posts_per_platform: int = 100
    max_age_hours: int = 24
    min_engagement_score: float = 0.1

    # フィルタリング設定
    filter_retweets: bool = True
    filter_spam: bool = True
    min_text_length: int = 20

    # 言語設定
    languages: List[str] = field(default_factory=lambda: ["en", "ja"])

    # レート制限
    request_delay: float = 1.0
    batch_size: int = 50


@dataclass
class SocialPost:
    """ソーシャルメディア投稿"""

    text: str
    platform: str  # "twitter", "reddit", "discord"
    post_id: str
    author: str
    created_at: datetime

    # エンゲージメント指標
    likes: int = 0
    retweets: int = 0
    comments: int = 0
    upvotes: int = 0  # Reddit
    downvotes: int = 0  # Reddit

    # 分析結果
    sentiment_result: Optional[SentimentResult] = None
    engagement_score: float = 0.0
    influence_score: float = 0.0
    keywords: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)

    # メタデータ
    language: str = "en"
    is_verified: bool = False
    follower_count: int = 0
    url: Optional[str] = None


@dataclass
class SocialSentimentResult:
    """ソーシャルセンチメント分析結果"""

    posts: List[SocialPost]
    overall_sentiment: float
    platform_breakdown: Dict[str, Dict[str, Any]]
    trending_hashtags: List[Tuple[str, int]]
    influential_users: List[Dict[str, Any]]
    engagement_analysis: Dict[str, float]
    temporal_analysis: Dict[str, float]
    confidence_score: float
    analysis_timestamp: datetime = field(default_factory=datetime.now)


class SocialMediaAnalyzer:
    """ソーシャルメディア分析システム"""

    def __init__(self, config: Optional[SocialConfig] = None):
        self.config = config or SocialConfig()

        # センチメントエンジン
        self.sentiment_engine = create_sentiment_engine()

        # API クライアント初期化
        self.twitter_api = None
        self.reddit_api = None
        self.discord_client = None

        self._initialize_apis()

        # データキャッシュ
        self.post_cache = {}
        self.processed_post_ids = set()
        self.user_influence_cache = {}

        # パフォーマンス統計
        self.fetch_stats = {
            "twitter_fetched": 0,
            "reddit_fetched": 0,
            "discord_fetched": 0,
            "total_processed": 0,
            "cache_hits": 0,
        }

        logger.info("Social Media Analyzer 初期化完了")

    def _initialize_apis(self):
        """API初期化"""

        # Twitter API v2初期化
        if TWITTER_AVAILABLE and self.config.twitter_bearer_token:
            try:
                self.twitter_api = tweepy.Client(
                    bearer_token=self.config.twitter_bearer_token,
                    consumer_key=self.config.twitter_consumer_key,
                    consumer_secret=self.config.twitter_consumer_secret,
                    access_token=self.config.twitter_access_token,
                    access_token_secret=self.config.twitter_access_token_secret,
                    wait_on_rate_limit=True,
                )
                logger.info("Twitter API 初期化完了")
            except Exception as e:
                logger.error(f"Twitter API 初期化エラー: {e}")

        # Reddit API初期化
        if REDDIT_AVAILABLE and self.config.reddit_client_id:
            try:
                self.reddit_api = praw.Reddit(
                    client_id=self.config.reddit_client_id,
                    client_secret=self.config.reddit_client_secret,
                    user_agent=self.config.reddit_user_agent,
                )
                logger.info("Reddit API 初期化完了")
            except Exception as e:
                logger.error(f"Reddit API 初期化エラー: {e}")

    async def collect_social_data(
        self,
        keywords: List[str] = None,
        platforms: List[str] = None,
        hours_back: int = None,
    ) -> List[SocialPost]:
        """ソーシャルメディアデータ収集"""

        keywords = keywords or self.config.search_keywords
        platforms = platforms or ["twitter", "reddit"]
        hours_back = hours_back or self.config.max_age_hours

        logger.info(f"ソーシャルデータ収集開始: {len(keywords)} キーワード, {platforms}")

        all_posts = []

        # プラットフォーム別収集
        tasks = []

        if "twitter" in platforms and self.twitter_api:
            tasks.append(self._collect_twitter_data(keywords, hours_back))

        if "reddit" in platforms and self.reddit_api:
            tasks.append(self._collect_reddit_data(keywords, hours_back))

        # 並行実行
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, list):
                    all_posts.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"ソーシャルデータ収集エラー: {result}")

        # フィルタリング・重複除去
        filtered_posts = self._filter_and_deduplicate_posts(all_posts)

        logger.info(f"ソーシャルデータ収集完了: {len(filtered_posts)} 投稿")

        return filtered_posts

    async def _collect_twitter_data(self, keywords: List[str], hours_back: int) -> List[SocialPost]:
        """Twitter データ収集"""

        if not self.twitter_api:
            return []

        posts = []

        try:
            # 日時範囲設定
            start_time = datetime.now() - timedelta(hours=hours_back)

            for keyword in keywords[:5]:  # レート制限対応
                try:
                    logger.debug(f"Twitter検索: {keyword}")

                    # ツイート検索
                    tweets = tweepy.Paginator(
                        self.twitter_api.search_recent_tweets,
                        query=f"{keyword} -is:retweet" if self.config.filter_retweets else keyword,
                        tweet_fields=[
                            "created_at",
                            "author_id",
                            "public_metrics",
                            "lang",
                            "context_annotations",
                        ],
                        user_fields=["verified", "public_metrics"],
                        expansions=["author_id"],
                        start_time=start_time,
                        max_results=100,
                    ).flatten(limit=self.config.max_posts_per_platform // len(keywords))

                    for tweet in tweets:
                        post = self._parse_twitter_post(tweet, keyword)
                        if post:
                            posts.append(post)

                    # レート制限対応
                    await asyncio.sleep(self.config.request_delay)

                except Exception as e:
                    logger.error(f"Twitter検索エラー ({keyword}): {e}")

        except Exception as e:
            logger.error(f"Twitter全体エラー: {e}")

        self.fetch_stats["twitter_fetched"] += len(posts)
        return posts

    async def _collect_reddit_data(self, keywords: List[str], hours_back: int) -> List[SocialPost]:
        """Reddit データ収集"""

        if not self.reddit_api:
            return []

        posts = []

        try:
            # 対象サブレディット
            target_subreddits = [
                "investing",
                "stocks",
                "SecurityAnalysis",
                "ValueInvesting",
                "financialindependence",
                "StockMarket",
                "wallstreetbets",
                "cryptocurrency",
                "Bitcoin",
                "ethereum",
            ]

            cutoff_time = time.time() - (hours_back * 3600)

            for subreddit_name in target_subreddits[:5]:  # 制限
                try:
                    subreddit = self.reddit_api.subreddit(subreddit_name)

                    # 新着投稿取得
                    for submission in subreddit.new(limit=20):
                        if submission.created_utc < cutoff_time:
                            continue

                        # キーワードフィルタ
                        if self._matches_keywords(
                            submission.title + " " + submission.selftext, keywords
                        ):
                            post = self._parse_reddit_post(submission)
                            if post:
                                posts.append(post)

                        # コメントも収集
                        try:
                            submission.comments.replace_more(limit=0)
                            for comment in submission.comments.list()[:5]:  # 上位5コメント
                                if (
                                    hasattr(comment, "created_utc")
                                    and comment.created_utc >= cutoff_time
                                ):
                                    if self._matches_keywords(comment.body, keywords):
                                        comment_post = self._parse_reddit_comment(
                                            comment, submission.title
                                        )
                                        if comment_post:
                                            posts.append(comment_post)
                        except:
                            pass

                    # レート制限対応
                    await asyncio.sleep(self.config.request_delay)

                except Exception as e:
                    logger.error(f"Reddit subreddit エラー ({subreddit_name}): {e}")

        except Exception as e:
            logger.error(f"Reddit全体エラー: {e}")

        self.fetch_stats["reddit_fetched"] += len(posts)
        return posts

    def _parse_twitter_post(self, tweet, keyword: str) -> Optional[SocialPost]:
        """Twitter投稿解析"""

        try:
            # 基本情報抽出
            text = tweet.text
            if len(text) < self.config.min_text_length:
                return None

            # 重複チェック
            if tweet.id in self.processed_post_ids:
                return None

            # メトリクス取得
            metrics = tweet.public_metrics or {}

            # ハッシュタグ・メンション抽出
            hashtags = re.findall(r"#(\w+)", text)
            mentions = re.findall(r"@(\w+)", text)

            # 投稿作成
            post = SocialPost(
                text=text,
                platform="twitter",
                post_id=str(tweet.id),
                author=str(tweet.author_id) if hasattr(tweet, "author_id") else "unknown",
                created_at=tweet.created_at,
                likes=metrics.get("like_count", 0),
                retweets=metrics.get("retweet_count", 0),
                comments=metrics.get("reply_count", 0),
                keywords=[keyword],
                hashtags=hashtags,
                mentions=mentions,
                language=getattr(tweet, "lang", "en"),
                url=f"https://twitter.com/user/status/{tweet.id}",
            )

            # エンゲージメントスコア計算
            post.engagement_score = self._calculate_engagement_score(post)

            self.processed_post_ids.add(tweet.id)
            return post

        except Exception as e:
            logger.error(f"Twitter投稿解析エラー: {e}")
            return None

    def _parse_reddit_post(self, submission) -> Optional[SocialPost]:
        """Reddit投稿解析"""

        try:
            text = submission.title + " " + (submission.selftext or "")
            if len(text.strip()) < self.config.min_text_length:
                return None

            # 重複チェック
            if submission.id in self.processed_post_ids:
                return None

            post = SocialPost(
                text=text,
                platform="reddit",
                post_id=submission.id,
                author=str(submission.author) if submission.author else "unknown",
                created_at=datetime.fromtimestamp(submission.created_utc),
                upvotes=submission.score,
                comments=submission.num_comments,
                url=f"https://reddit.com{submission.permalink}",
            )

            # エンゲージメントスコア計算
            post.engagement_score = self._calculate_engagement_score(post)

            self.processed_post_ids.add(submission.id)
            return post

        except Exception as e:
            logger.error(f"Reddit投稿解析エラー: {e}")
            return None

    def _parse_reddit_comment(self, comment, parent_title: str) -> Optional[SocialPost]:
        """Redditコメント解析"""

        try:
            if len(comment.body) < self.config.min_text_length:
                return None

            # 重複チェック
            if comment.id in self.processed_post_ids:
                return None

            post = SocialPost(
                text=f"Re: {parent_title[:50]}... - {comment.body}",
                platform="reddit",
                post_id=comment.id,
                author=str(comment.author) if comment.author else "unknown",
                created_at=datetime.fromtimestamp(comment.created_utc),
                upvotes=comment.score,
                url=f"https://reddit.com{comment.permalink}",
            )

            post.engagement_score = self._calculate_engagement_score(post)

            self.processed_post_ids.add(comment.id)
            return post

        except Exception as e:
            logger.error(f"Redditコメント解析エラー: {e}")
            return None

    def _matches_keywords(self, text: str, keywords: List[str]) -> bool:
        """キーワードマッチング"""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)

    def _filter_and_deduplicate_posts(self, posts: List[SocialPost]) -> List[SocialPost]:
        """投稿フィルタリング・重複除去"""

        filtered = []
        seen_texts = set()

        for post in posts:
            # スパムフィルタ
            if self.config.filter_spam and self._is_spam(post):
                continue

            # エンゲージメント閾値
            if post.engagement_score < self.config.min_engagement_score:
                continue

            # テキスト重複チェック（簡易）
            text_hash = hash(post.text.lower().strip())
            if text_hash in seen_texts:
                continue

            # 言語フィルタ
            if post.language not in self.config.languages:
                continue

            filtered.append(post)
            seen_texts.add(text_hash)

        # エンゲージメント順ソート
        filtered.sort(key=lambda p: p.engagement_score, reverse=True)

        return filtered

    def _calculate_engagement_score(self, post: SocialPost) -> float:
        """エンゲージメントスコア計算"""

        score = 0.0

        if post.platform == "twitter":
            # Twitter エンゲージメント
            score = (post.likes * 1.0 + post.retweets * 3.0 + post.comments * 2.0) / 10.0

        elif post.platform == "reddit":
            # Reddit エンゲージメント
            score = (post.upvotes * 1.0 + post.comments * 2.0) / 5.0

        # 正規化
        return min(score / 100.0, 1.0)  # 最大1.0に正規化

    def _is_spam(self, post: SocialPost) -> bool:
        """スパム判定（簡易）"""

        text = post.text.lower()

        # スパムキーワード
        spam_indicators = [
            "buy now",
            "click here",
            "free money",
            "guaranteed profit",
            "pump and dump",
            "get rich quick",
            "easy money",
        ]

        spam_count = sum(1 for indicator in spam_indicators if indicator in text)

        # 重複文字・記号チェック
        repeat_chars = len(re.findall(r"(.)\1{3,}", text))  # 4回以上連続
        excessive_caps = len(re.findall(r"[A-Z]{5,}", text))  # 5文字以上大文字

        # スパム判定
        return spam_count >= 2 or repeat_chars >= 3 or excessive_caps >= 3

    def analyze_social_sentiment(self, posts: List[SocialPost]) -> SocialSentimentResult:
        """ソーシャルセンチメント分析"""

        if not posts:
            return self._create_empty_social_result()

        logger.info(f"ソーシャルセンチメント分析開始: {len(posts)} 投稿")

        # センチメント分析実行
        analyzed_posts = []
        sentiments = []

        for post in posts:
            # センチメント分析
            sentiment_result = self.sentiment_engine.analyze_text(post.text)
            post.sentiment_result = sentiment_result

            # 影響力スコア計算
            post.influence_score = self._calculate_influence_score(post)

            analyzed_posts.append(post)
            sentiments.append(sentiment_result.sentiment_score)

        # 重み付き全体センチメント（エンゲージメント重み）
        weights = [post.engagement_score for post in analyzed_posts]
        if sum(weights) > 0:
            overall_sentiment = np.average(sentiments, weights=weights)
        else:
            overall_sentiment = np.mean(sentiments) if sentiments else 0.0

        # プラットフォーム別分析
        platform_breakdown = self._analyze_by_platform(analyzed_posts)

        # トレンディングハッシュタグ
        trending_hashtags = self._get_trending_hashtags(analyzed_posts)

        # 影響力のあるユーザー
        influential_users = self._get_influential_users(analyzed_posts)

        # エンゲージメント分析
        engagement_analysis = self._analyze_engagement_patterns(analyzed_posts)

        # 時系列分析
        temporal_analysis = self._analyze_temporal_patterns(analyzed_posts)

        # 信頼度スコア
        confidences = [post.sentiment_result.confidence for post in analyzed_posts]
        confidence_score = np.mean(confidences) if confidences else 0.0

        result = SocialSentimentResult(
            posts=analyzed_posts,
            overall_sentiment=overall_sentiment,
            platform_breakdown=platform_breakdown,
            trending_hashtags=trending_hashtags,
            influential_users=influential_users,
            engagement_analysis=engagement_analysis,
            temporal_analysis=temporal_analysis,
            confidence_score=confidence_score,
        )

        logger.info(f"ソーシャルセンチメント分析完了: 全体={overall_sentiment:.3f}")

        return result

    def _calculate_influence_score(self, post: SocialPost) -> float:
        """影響力スコア計算"""

        influence = 0.0

        # エンゲージメント重み
        influence += post.engagement_score * 0.4

        # センチメント強度重み
        if post.sentiment_result:
            influence += abs(post.sentiment_result.sentiment_score) * 0.3

        # プラットフォーム重み
        platform_weights = {"twitter": 1.0, "reddit": 0.8, "discord": 0.6}
        influence += platform_weights.get(post.platform, 0.5) * 0.2

        # 認証アカウント重み
        if post.is_verified:
            influence += 0.1

        return min(influence, 1.0)

    def _analyze_by_platform(self, posts: List[SocialPost]) -> Dict[str, Dict[str, Any]]:
        """プラットフォーム別分析"""

        platform_data = {}

        for post in posts:
            platform = post.platform
            if platform not in platform_data:
                platform_data[platform] = {
                    "count": 0,
                    "sentiments": [],
                    "engagement_scores": [],
                    "influence_scores": [],
                }

            platform_data[platform]["count"] += 1
            if post.sentiment_result:
                platform_data[platform]["sentiments"].append(post.sentiment_result.sentiment_score)
            platform_data[platform]["engagement_scores"].append(post.engagement_score)
            platform_data[platform]["influence_scores"].append(post.influence_score)

        # 統計計算
        for platform, data in platform_data.items():
            data["avg_sentiment"] = np.mean(data["sentiments"]) if data["sentiments"] else 0.0
            data["avg_engagement"] = np.mean(data["engagement_scores"])
            data["avg_influence"] = np.mean(data["influence_scores"])
            data["sentiment_std"] = (
                np.std(data["sentiments"]) if len(data["sentiments"]) > 1 else 0.0
            )

        return platform_data

    def _get_trending_hashtags(
        self, posts: List[SocialPost], limit: int = 20
    ) -> List[Tuple[str, int]]:
        """トレンディングハッシュタグ取得"""

        hashtag_counts = {}
        hashtag_engagement = {}

        for post in posts:
            for hashtag in post.hashtags:
                hashtag_lower = hashtag.lower()
                hashtag_counts[hashtag_lower] = hashtag_counts.get(hashtag_lower, 0) + 1

                if hashtag_lower not in hashtag_engagement:
                    hashtag_engagement[hashtag_lower] = []
                hashtag_engagement[hashtag_lower].append(post.engagement_score)

        # エンゲージメント重み付きスコア
        weighted_hashtags = []
        for hashtag, count in hashtag_counts.items():
            if count >= 2:  # 最低2回出現
                avg_engagement = np.mean(hashtag_engagement[hashtag])
                weighted_score = count * (1 + avg_engagement)
                weighted_hashtags.append((hashtag, weighted_score))

        # スコア順ソート
        weighted_hashtags.sort(key=lambda x: x[1], reverse=True)

        return [(tag, int(score)) for tag, score in weighted_hashtags[:limit]]

    def _get_influential_users(
        self, posts: List[SocialPost], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """影響力のあるユーザー取得"""

        user_data = {}

        for post in posts:
            author = post.author
            if author not in user_data:
                user_data[author] = {
                    "post_count": 0,
                    "total_engagement": 0.0,
                    "total_influence": 0.0,
                    "sentiments": [],
                    "platforms": set(),
                }

            user_data[author]["post_count"] += 1
            user_data[author]["total_engagement"] += post.engagement_score
            user_data[author]["total_influence"] += post.influence_score
            user_data[author]["platforms"].add(post.platform)

            if post.sentiment_result:
                user_data[author]["sentiments"].append(post.sentiment_result.sentiment_score)

        # 影響力スコア計算
        influential_users = []
        for author, data in user_data.items():
            if data["post_count"] >= 2:  # 最低2投稿
                avg_engagement = data["total_engagement"] / data["post_count"]
                avg_influence = data["total_influence"] / data["post_count"]
                avg_sentiment = np.mean(data["sentiments"]) if data["sentiments"] else 0.0

                influence_score = avg_influence * data["post_count"] * (1 + avg_engagement)

                influential_users.append(
                    {
                        "author": author,
                        "influence_score": influence_score,
                        "post_count": data["post_count"],
                        "avg_engagement": avg_engagement,
                        "avg_sentiment": avg_sentiment,
                        "platforms": list(data["platforms"]),
                    }
                )

        # 影響力順ソート
        influential_users.sort(key=lambda x: x["influence_score"], reverse=True)

        return influential_users[:limit]

    def _analyze_engagement_patterns(self, posts: List[SocialPost]) -> Dict[str, float]:
        """エンゲージメントパターン分析"""

        analysis = {}

        # プラットフォーム別エンゲージメント
        platform_engagement = {}
        for post in posts:
            if post.platform not in platform_engagement:
                platform_engagement[post.platform] = []
            platform_engagement[post.platform].append(post.engagement_score)

        for platform, scores in platform_engagement.items():
            analysis[f"{platform}_avg_engagement"] = np.mean(scores)

        # センチメント別エンゲージメント
        positive_engagement = [
            p.engagement_score
            for p in posts
            if p.sentiment_result and p.sentiment_result.sentiment_label == "positive"
        ]
        negative_engagement = [
            p.engagement_score
            for p in posts
            if p.sentiment_result and p.sentiment_result.sentiment_label == "negative"
        ]

        if positive_engagement:
            analysis["positive_avg_engagement"] = np.mean(positive_engagement)
        if negative_engagement:
            analysis["negative_avg_engagement"] = np.mean(negative_engagement)

        # 全体統計
        all_engagement = [p.engagement_score for p in posts]
        if all_engagement:
            analysis["overall_avg_engagement"] = np.mean(all_engagement)
            analysis["engagement_std"] = np.std(all_engagement)
            analysis["high_engagement_ratio"] = len([e for e in all_engagement if e > 0.5]) / len(
                all_engagement
            )

        return analysis

    def _analyze_temporal_patterns(self, posts: List[SocialPost]) -> Dict[str, float]:
        """時系列パターン分析"""

        analysis = {}

        # 時間帯別分析
        hourly_sentiments = {}
        hourly_engagement = {}

        for post in posts:
            hour = post.created_at.hour

            if hour not in hourly_sentiments:
                hourly_sentiments[hour] = []
                hourly_engagement[hour] = []

            if post.sentiment_result:
                hourly_sentiments[hour].append(post.sentiment_result.sentiment_score)
            hourly_engagement[hour].append(post.engagement_score)

        # 平均計算
        for hour in range(24):
            if hour in hourly_sentiments and hourly_sentiments[hour]:
                analysis[f"hour_{hour:02d}_sentiment"] = np.mean(hourly_sentiments[hour])
            if hour in hourly_engagement and hourly_engagement[hour]:
                analysis[f"hour_{hour:02d}_engagement"] = np.mean(hourly_engagement[hour])

        # ピーク時間帯検出
        if hourly_engagement:
            peak_hour = max(
                hourly_engagement.keys(),
                key=lambda h: np.mean(hourly_engagement[h]) if hourly_engagement[h] else 0,
            )
            analysis["peak_engagement_hour"] = peak_hour

        return analysis

    def _create_empty_social_result(self) -> SocialSentimentResult:
        """空のソーシャル結果作成"""
        return SocialSentimentResult(
            posts=[],
            overall_sentiment=0.0,
            platform_breakdown={},
            trending_hashtags=[],
            influential_users=[],
            engagement_analysis={},
            temporal_analysis={},
            confidence_score=0.0,
        )

    def export_social_analysis(self, result: SocialSentimentResult, format: str = "json") -> str:
        """ソーシャル分析結果エクスポート"""

        if format == "json":
            export_data = {
                "analysis_timestamp": result.analysis_timestamp.isoformat(),
                "overall_sentiment": result.overall_sentiment,
                "confidence_score": result.confidence_score,
                "total_posts": len(result.posts),
                "platform_breakdown": result.platform_breakdown,
                "trending_hashtags": result.trending_hashtags,
                "influential_users": result.influential_users,
                "engagement_analysis": result.engagement_analysis,
                "posts": [
                    {
                        "platform": post.platform,
                        "text": post.text[:200] + "..." if len(post.text) > 200 else post.text,
                        "author": post.author,
                        "sentiment_label": (
                            post.sentiment_result.sentiment_label if post.sentiment_result else None
                        ),
                        "sentiment_score": (
                            post.sentiment_result.sentiment_score if post.sentiment_result else 0
                        ),
                        "engagement_score": post.engagement_score,
                        "influence_score": post.influence_score,
                        "hashtags": post.hashtags,
                        "created_at": post.created_at.isoformat(),
                    }
                    for post in result.posts
                ],
            }

            return json.dumps(export_data, indent=2, ensure_ascii=False)

        else:
            raise ValueError(f"未対応のエクスポート形式: {format}")

    def get_analysis_summary(self) -> Dict[str, Any]:
        """分析サマリー取得"""
        return {
            "fetch_stats": self.fetch_stats,
            "cache_size": len(self.post_cache),
            "processed_posts": len(self.processed_post_ids),
            "api_status": {
                "twitter": self.twitter_api is not None,
                "reddit": self.reddit_api is not None,
                "discord": self.discord_client is not None,
            },
        }


# 便利関数
def analyze_social_sentiment(
    keywords: List[str] = None, platforms: List[str] = None, hours_back: int = 24
) -> SocialSentimentResult:
    """ソーシャルセンチメント分析（簡易インターフェース）"""

    async def _analyze():
        analyzer = SocialMediaAnalyzer()
        posts = await analyzer.collect_social_data(
            keywords=keywords, platforms=platforms, hours_back=hours_back
        )
        return analyzer.analyze_social_sentiment(posts)

    return asyncio.run(_analyze())


if __name__ == "__main__":
    # ソーシャル分析テスト
    print("=== Next-Gen AI Social Media Analyzer テスト ===")

    async def test_social_analysis():
        analyzer = SocialMediaAnalyzer()

        # テスト用キーワード
        test_keywords = ["$AAPL", "stock market", "AI investment"]

        print(f"ソーシャルデータ収集テスト: {test_keywords}")

        # データ収集（モック）
        mock_posts = [
            SocialPost(
                text="Very bullish on $AAPL today! Great earnings report #StockMarket #Investing",
                platform="twitter",
                post_id="12345",
                author="trader_pro",
                created_at=datetime.now(),
                likes=100,
                retweets=50,
                hashtags=["StockMarket", "Investing"],
                keywords=["$AAPL"],
            ),
            SocialPost(
                text="Market looking bearish. Time to be cautious with tech stocks.",
                platform="reddit",
                post_id="67890",
                author="market_analyst",
                created_at=datetime.now() - timedelta(hours=2),
                upvotes=25,
                comments=15,
                keywords=["market", "tech stocks"],
            ),
        ]

        print(f"テスト投稿数: {len(mock_posts)}")

        # センチメント分析
        result = analyzer.analyze_social_sentiment(mock_posts)

        print("\n分析結果:")
        print(f"全体センチメント: {result.overall_sentiment:.3f}")
        print(f"信頼度: {result.confidence_score:.3f}")
        print(f"プラットフォーム別: {list(result.platform_breakdown.keys())}")
        print(f"トレンディングハッシュタグ: {result.trending_hashtags[:3]}")

        # エクスポートテスト
        json_export = analyzer.export_social_analysis(result, "json")
        print(f"\nJSON エクスポート長: {len(json_export)} 文字")

        # サマリー
        summary = analyzer.get_analysis_summary()
        print(f"分析サマリー: {summary}")

    # テスト実行
    try:
        asyncio.run(test_social_analysis())
    except Exception as e:
        print(f"テストエラー: {e}")

    print("\n=== テスト完了 ===")
