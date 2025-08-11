#!/usr/bin/env python3
"""
Next-Gen AI Sentiment Engine
高度センチメント分析エンジン

FinBERT・GPT-4・多言語対応・リアルタイム感情解析システム
"""

import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# 自然言語処理ライブラリ
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# 多言語対応
try:
    from googletrans import Translator

    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class SentimentConfig:
    """センチメント分析設定"""

    # モデル設定
    finbert_model: str = "ProsusAI/finbert"
    use_gpu: bool = True
    batch_size: int = 16
    max_length: int = 512

    # 多言語設定
    enable_translation: bool = True
    target_language: str = "en"

    # 分析設定
    confidence_threshold: float = 0.7
    sentiment_smoothing: float = 0.1
    temporal_weight_decay: float = 0.95

    # キャッシュ設定
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1時間

    # 集約設定
    aggregation_window: int = 24  # 24時間
    min_samples: int = 3


@dataclass
class SentimentResult:
    """センチメント分析結果"""

    text: str
    sentiment_label: str  # positive, negative, neutral
    sentiment_score: float  # -1.0 ~ 1.0
    confidence: float  # 0.0 ~ 1.0
    emotions: Optional[Dict[str, float]] = None
    model_used: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class MarketSentimentIndicator:
    """市場センチメント指標"""

    overall_sentiment: float  # -1.0 ~ 1.0
    sentiment_strength: float  # 0.0 ~ 1.0
    sentiment_volatility: float  # 変動性
    sentiment_trend: float  # トレンド (-1.0 ~ 1.0)
    fear_greed_index: float  # 恐怖・貪欲指標 (0 ~ 100)
    market_mood: str  # "bullish", "bearish", "neutral"
    confidence_level: float  # 信頼度
    sample_count: int
    timestamp: datetime


class SentimentEngine:
    """高度センチメント分析エンジン"""

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()

        # デバイス設定
        if TRANSFORMERS_AVAILABLE and self.config.use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            logger.info("GPU使用でセンチメント分析")
        else:
            self.device = "cpu"
            logger.info("CPU使用でセンチメント分析")

        # モデル初期化
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}

        # 翻訳機能
        self.translator = None
        if TRANSLATION_AVAILABLE and self.config.enable_translation:
            try:
                self.translator = Translator()
            except Exception as e:
                logger.warning(f"翻訳機能初期化失敗: {e}")

        # センチメントキャッシュ
        self.sentiment_cache = {}
        self.analysis_history = []

        # 初期化
        self._initialize_models()

        logger.info("Sentiment Engine 初期化完了")

    def _initialize_models(self):
        """モデル初期化"""

        # FinBERT (金融特化)
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"FinBERTモデル読み込み: {self.config.finbert_model}")

                self.tokenizers["finbert"] = AutoTokenizer.from_pretrained(
                    self.config.finbert_model
                )
                self.models[
                    "finbert"
                ] = AutoModelForSequenceClassification.from_pretrained(
                    self.config.finbert_model
                ).to(self.device)

                self.pipelines["finbert"] = pipeline(
                    "sentiment-analysis",
                    model=self.models["finbert"],
                    tokenizer=self.tokenizers["finbert"],
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True,
                )

                logger.info("FinBERT初期化完了")

            except Exception as e:
                logger.error(f"FinBERT初期化失敗: {e}")

        # VADER (NLTK)
        if NLTK_AVAILABLE:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                logger.info("VADER分析器初期化完了")
            except Exception as e:
                logger.warning(f"VADER初期化失敗: {e}")
                self.vader_analyzer = None

    def analyze_text(self, text: str, model: str = "finbert") -> SentimentResult:
        """単一テキストのセンチメント分析"""

        if not text or not text.strip():
            return SentimentResult(
                text=text,
                sentiment_label="neutral",
                sentiment_score=0.0,
                confidence=0.0,
                model_used="none",
                timestamp=time.time(),
            )

        # キャッシュチェック
        if self.config.cache_enabled:
            cache_key = f"{model}:{hash(text)}"
            if cache_key in self.sentiment_cache:
                cached_result = self.sentiment_cache[cache_key]
                if time.time() - cached_result.timestamp < self.config.cache_ttl:
                    return cached_result

        # 前処理
        processed_text = self._preprocess_text(text)

        # 翻訳（必要に応じて）
        if self._needs_translation(processed_text):
            processed_text = self._translate_text(processed_text)

        # モデル別分析
        result = None
        if model == "finbert" and "finbert" in self.pipelines:
            result = self._analyze_with_finbert(processed_text)
        elif model == "vader" and self.vader_analyzer:
            result = self._analyze_with_vader(processed_text)
        elif model == "textblob" and TEXTBLOB_AVAILABLE:
            result = self._analyze_with_textblob(processed_text)
        else:
            # フォールバック分析
            result = self._fallback_analysis(processed_text)

        # 結果を履歴に追加
        self.analysis_history.append(result)

        # キャッシュに保存
        if self.config.cache_enabled:
            self.sentiment_cache[cache_key] = result

        return result

    def _analyze_with_finbert(self, text: str) -> SentimentResult:
        """FinBERTによる分析"""

        try:
            # トークン長チェック
            if len(text) > self.config.max_length:
                text = text[: self.config.max_length]

            # 推論実行
            outputs = self.pipelines["finbert"](text)

            # 結果解析
            scores = {item["label"].lower(): item["score"] for item in outputs[0]}

            # 最高スコアのラベル
            best_label = max(scores.keys(), key=lambda k: scores[k])
            best_score = scores[best_label]

            # センチメントスコア計算 (-1.0 ~ 1.0)
            if best_label == "positive":
                sentiment_score = scores.get("positive", 0) - scores.get("negative", 0)
            elif best_label == "negative":
                sentiment_score = scores.get("negative", 0) - scores.get("positive", 0)
                sentiment_score = -sentiment_score
            else:
                sentiment_score = 0.0

            return SentimentResult(
                text=text,
                sentiment_label=best_label,
                sentiment_score=sentiment_score,
                confidence=best_score,
                emotions=scores,
                model_used="finbert",
                timestamp=time.time(),
                metadata={"full_scores": scores},
            )

        except Exception as e:
            logger.error(f"FinBERT分析エラー: {e}")
            return self._fallback_analysis(text)

    def _analyze_with_vader(self, text: str) -> SentimentResult:
        """VADERによる分析"""

        try:
            scores = self.vader_analyzer.polarity_scores(text)

            # ラベル決定
            compound = scores["compound"]
            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"

            return SentimentResult(
                text=text,
                sentiment_label=label,
                sentiment_score=compound,
                confidence=abs(compound),
                emotions={
                    "positive": scores["pos"],
                    "negative": scores["neg"],
                    "neutral": scores["neu"],
                },
                model_used="vader",
                timestamp=time.time(),
                metadata={"vader_scores": scores},
            )

        except Exception as e:
            logger.error(f"VADER分析エラー: {e}")
            return self._fallback_analysis(text)

    def _analyze_with_textblob(self, text: str) -> SentimentResult:
        """TextBlobによる分析"""

        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1.0 ~ 1.0
            subjectivity = blob.sentiment.subjectivity  # 0.0 ~ 1.0

            # ラベル決定
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"

            return SentimentResult(
                text=text,
                sentiment_label=label,
                sentiment_score=polarity,
                confidence=subjectivity,
                model_used="textblob",
                timestamp=time.time(),
                metadata={"polarity": polarity, "subjectivity": subjectivity},
            )

        except Exception as e:
            logger.error(f"TextBlob分析エラー: {e}")
            return self._fallback_analysis(text)

    def _fallback_analysis(self, text: str) -> SentimentResult:
        """フォールバック分析（ルールベース）"""

        # 簡単なキーワードベース分析
        positive_words = [
            "good",
            "great",
            "excellent",
            "positive",
            "up",
            "rise",
            "gain",
            "profit",
            "bull",
            "bullish",
            "strong",
            "high",
            "increase",
            "growth",
            "buy",
        ]

        negative_words = [
            "bad",
            "terrible",
            "negative",
            "down",
            "fall",
            "loss",
            "bear",
            "bearish",
            "weak",
            "low",
            "decrease",
            "decline",
            "sell",
            "crash",
            "drop",
        ]

        text_lower = text.lower()

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        total_words = len(text.split())

        if pos_count > neg_count:
            label = "positive"
            score = min(pos_count / max(total_words, 1), 1.0)
        elif neg_count > pos_count:
            label = "negative"
            score = -min(neg_count / max(total_words, 1), 1.0)
        else:
            label = "neutral"
            score = 0.0

        confidence = abs(score)

        return SentimentResult(
            text=text,
            sentiment_label=label,
            sentiment_score=score,
            confidence=confidence,
            model_used="fallback",
            timestamp=time.time(),
            metadata={
                "positive_words": pos_count,
                "negative_words": neg_count,
                "total_words": total_words,
            },
        )

    def analyze_batch(
        self, texts: List[str], model: str = "finbert"
    ) -> List[SentimentResult]:
        """バッチテキスト分析"""

        if not texts:
            return []

        logger.info(f"バッチ分析開始: {len(texts)} テキスト")

        results = []
        batch_size = self.config.batch_size

        # バッチ処理
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = [self.analyze_text(text, model) for text in batch]
            results.extend(batch_results)

        logger.info(f"バッチ分析完了: {len(results)} 結果")
        return results

    def calculate_market_sentiment(
        self,
        texts: List[str] = None,
        sentiment_results: List[SentimentResult] = None,
        symbol: str = None,
    ) -> MarketSentimentIndicator:
        """市場センチメント指標計算"""

        # 入力データ準備
        if sentiment_results is None:
            if texts is None:
                # 履歴からデータ取得
                recent_results = self._get_recent_analysis(
                    hours=self.config.aggregation_window
                )
            else:
                recent_results = self.analyze_batch(texts)
        else:
            recent_results = sentiment_results

        if len(recent_results) < self.config.min_samples:
            logger.warning(
                f"サンプル数不足: {len(recent_results)} < {self.config.min_samples}"
            )
            return self._create_neutral_indicator(len(recent_results))

        # 基本統計計算
        sentiment_scores = [r.sentiment_score for r in recent_results]
        confidences = [r.confidence for r in recent_results]

        # 重み付き平均（信頼度による重み付け）
        if sum(confidences) > 0:
            weighted_sentiment = np.average(sentiment_scores, weights=confidences)
        else:
            weighted_sentiment = np.mean(sentiment_scores)

        # センチメント強度（絶対値の平均）
        sentiment_strength = np.mean([abs(score) for score in sentiment_scores])

        # センチメント変動性
        sentiment_volatility = (
            np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.0
        )

        # トレンド計算（時系列重み付け）
        sentiment_trend = self._calculate_sentiment_trend(recent_results)

        # Fear & Greed Index (0-100)
        fear_greed_index = self._calculate_fear_greed_index(
            weighted_sentiment, sentiment_volatility, sentiment_strength
        )

        # 市場ムード判定
        market_mood = self._determine_market_mood(
            weighted_sentiment, sentiment_strength
        )

        # 信頼度レベル
        confidence_level = np.mean(confidences) if confidences else 0.0

        return MarketSentimentIndicator(
            overall_sentiment=weighted_sentiment,
            sentiment_strength=sentiment_strength,
            sentiment_volatility=sentiment_volatility,
            sentiment_trend=sentiment_trend,
            fear_greed_index=fear_greed_index,
            market_mood=market_mood,
            confidence_level=confidence_level,
            sample_count=len(recent_results),
            timestamp=datetime.now(),
        )

    def _preprocess_text(self, text: str) -> str:
        """テキスト前処理"""

        # HTML/XMLタグ除去
        text = re.sub(r"<[^>]+>", "", text)

        # URL除去
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # メンション・ハッシュタグの整理
        text = re.sub(r"@\w+", "@USER", text)
        text = re.sub(r"#(\w+)", r"\1", text)

        # 余分な空白除去
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _needs_translation(self, text: str) -> bool:
        """翻訳が必要かチェック"""
        if not self.config.enable_translation:
            return False

        # 英語検出（簡易）
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in text if c.isalpha())

        if total_chars == 0:
            return False

        english_ratio = english_chars / total_chars
        return english_ratio < 0.7  # 70%未満が英語でない場合

    def _translate_text(self, text: str) -> str:
        """テキスト翻訳"""
        if not self.translator:
            return text

        try:
            result = self.translator.translate(text, dest=self.config.target_language)
            return result.text
        except Exception as e:
            logger.warning(f"翻訳エラー: {e}")
            return text

    def _get_recent_analysis(self, hours: int = 24) -> List[SentimentResult]:
        """最近の分析結果取得"""
        cutoff_time = time.time() - (hours * 3600)
        return [r for r in self.analysis_history if r.timestamp >= cutoff_time]

    def _calculate_sentiment_trend(self, results: List[SentimentResult]) -> float:
        """センチメントトレンド計算"""
        if len(results) < 2:
            return 0.0

        # 時系列ソート
        sorted_results = sorted(results, key=lambda x: x.timestamp)

        # 重み付き移動平均によるトレンド
        weights = np.array(
            [
                self.config.temporal_weight_decay**i
                for i in range(len(sorted_results) - 1, -1, -1)
            ]
        )
        scores = np.array([r.sentiment_score for r in sorted_results])

        if len(scores) > 1:
            # 線形回帰によるトレンド
            x = np.arange(len(scores))
            trend = np.polyfit(x, scores, 1)[0]  # 傾き
            return np.clip(trend * len(scores), -1.0, 1.0)

        return 0.0

    def _calculate_fear_greed_index(
        self, sentiment: float, volatility: float, strength: float
    ) -> float:
        """Fear & Greed Index計算"""

        # センチメントスコア正規化 (50 + sentiment * 50)
        sentiment_component = 50 + sentiment * 50

        # ボラティリティ逆転 (低い方が貪欲)
        volatility_component = max(0, 100 - volatility * 100)

        # 強度成分 (強いセンチメントは極端さを表す)
        strength_component = strength * 100

        # 重み付き平均
        fear_greed = (
            sentiment_component * 0.5
            + volatility_component * 0.3
            + strength_component * 0.2
        )

        return np.clip(fear_greed, 0, 100)

    def _determine_market_mood(self, sentiment: float, strength: float) -> str:
        """市場ムード判定"""

        if strength < 0.1:
            return "neutral"
        elif sentiment > 0.2:
            return "bullish"
        elif sentiment < -0.2:
            return "bearish"
        else:
            return "neutral"

    def _create_neutral_indicator(self, sample_count: int) -> MarketSentimentIndicator:
        """中立指標作成"""
        return MarketSentimentIndicator(
            overall_sentiment=0.0,
            sentiment_strength=0.0,
            sentiment_volatility=0.0,
            sentiment_trend=0.0,
            fear_greed_index=50.0,
            market_mood="neutral",
            confidence_level=0.0,
            sample_count=sample_count,
            timestamp=datetime.now(),
        )

    def get_analysis_summary(self, hours: int = 24) -> Dict[str, Any]:
        """分析サマリー取得"""

        recent_results = self._get_recent_analysis(hours)

        if not recent_results:
            return {"status": "no_data"}

        # 基本統計
        sentiment_scores = [r.sentiment_score for r in recent_results]
        confidences = [r.confidence for r in recent_results]

        # モデル使用統計
        model_counts = {}
        for result in recent_results:
            model = result.model_used
            model_counts[model] = model_counts.get(model, 0) + 1

        # ラベル分布
        label_counts = {}
        for result in recent_results:
            label = result.sentiment_label
            label_counts[label] = label_counts.get(label, 0) + 1

        return {
            "analysis_period_hours": hours,
            "total_analyses": len(recent_results),
            "average_sentiment": np.mean(sentiment_scores),
            "sentiment_std": np.std(sentiment_scores),
            "average_confidence": np.mean(confidences),
            "model_distribution": model_counts,
            "sentiment_distribution": label_counts,
            "latest_analysis": recent_results[-1].timestamp if recent_results else None,
        }

    def clear_cache(self):
        """キャッシュクリア"""
        self.sentiment_cache.clear()
        logger.info("センチメントキャッシュクリア完了")

    def clear_history(self):
        """履歴クリア"""
        self.analysis_history.clear()
        logger.info("分析履歴クリア完了")


# 便利関数
def create_sentiment_engine(config_dict: Optional[Dict] = None) -> SentimentEngine:
    """センチメントエンジン作成"""
    if config_dict:
        config = SentimentConfig(**config_dict)
    else:
        config = SentimentConfig()

    return SentimentEngine(config)


def analyze_financial_text(text: str, model: str = "finbert") -> SentimentResult:
    """金融テキスト分析（簡易インターフェース）"""
    engine = create_sentiment_engine()
    return engine.analyze_text(text, model)


if __name__ == "__main__":
    # センチメントエンジンテスト
    print("=== Next-Gen AI Sentiment Engine テスト ===")

    # エンジン作成
    engine = create_sentiment_engine()

    # テストテキスト
    test_texts = [
        "The stock market is showing strong bullish momentum today with significant gains.",
        "Market volatility increases as investors fear economic downturn.",
        "Corporate earnings exceed expectations, driving positive sentiment.",
        "債券市場は安定しており、投資家の信頼が高まっています。",  # 日本語
        "The company's quarterly results were disappointing, causing stock prices to fall.",
    ]

    print(f"\nテキスト分析開始: {len(test_texts)} サンプル")

    # 個別分析
    results = []
    for i, text in enumerate(test_texts):
        result = engine.analyze_text(text, model="finbert")
        results.append(result)

        print(f"\nテキスト {i+1}: {text[:60]}...")
        print(f"センチメント: {result.sentiment_label}")
        print(f"スコア: {result.sentiment_score:.3f}")
        print(f"信頼度: {result.confidence:.3f}")
        print(f"モデル: {result.model_used}")

    # 市場指標計算
    print("\n市場センチメント指標計算...")
    market_indicator = engine.calculate_market_sentiment(sentiment_results=results)

    print(f"全体センチメント: {market_indicator.overall_sentiment:.3f}")
    print(f"センチメント強度: {market_indicator.sentiment_strength:.3f}")
    print(f"変動性: {market_indicator.sentiment_volatility:.3f}")
    print(f"トレンド: {market_indicator.sentiment_trend:.3f}")
    print(f"Fear & Greed Index: {market_indicator.fear_greed_index:.1f}")
    print(f"市場ムード: {market_indicator.market_mood}")
    print(f"信頼度レベル: {market_indicator.confidence_level:.3f}")

    # 分析サマリー
    summary = engine.get_analysis_summary()
    print(f"\n分析サマリー: {summary}")

    print("\n=== テスト完了 ===")
