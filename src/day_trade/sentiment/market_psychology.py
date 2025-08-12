#!/usr/bin/env python3
"""
Next-Gen AI Market Psychology System
市場心理指標システム

Fear & Greed Index・VIX・Put/Call Ratio・センチメント統合分析
"""

import json
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# 統計・数値計算
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..utils.logging_config import get_context_logger
from .news_analyzer import NewsAnalyzer

# プロジェクト内インポート
from .sentiment_engine import (
    create_sentiment_engine,
)
from .social_analyzer import SocialMediaAnalyzer

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class MarketPsychologyConfig:
    """市場心理分析設定"""

    # データソース重み
    news_weight: float = 0.35
    social_weight: float = 0.25
    technical_weight: float = 0.25
    macro_weight: float = 0.15

    # 指標計算設定
    fear_greed_smoothing: float = 0.1
    sentiment_lookback_days: int = 7
    volatility_window: int = 20

    # 閾値設定
    extreme_fear_threshold: float = 20.0
    fear_threshold: float = 40.0
    greed_threshold: float = 60.0
    extreme_greed_threshold: float = 80.0

    # 更新頻度
    update_interval_minutes: int = 15
    cache_ttl_minutes: int = 30


@dataclass
class TechnicalIndicators:
    """テクニカル指標データ"""

    vix: float = 0.0
    put_call_ratio: float = 0.0
    advance_decline_ratio: float = 0.0
    new_high_low_ratio: float = 0.0
    margin_debt_ratio: float = 0.0
    insider_trading_ratio: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MacroIndicators:
    """マクロ経済指標データ"""

    interest_rate: float = 0.0
    inflation_rate: float = 0.0
    unemployment_rate: float = 0.0
    gdp_growth: float = 0.0
    currency_strength: float = 0.0
    commodity_index: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketPsychologyIndex:
    """市場心理総合指標"""

    # 主要指標
    fear_greed_index: float  # 0-100
    sentiment_score: float  # -1.0 ~ 1.0
    confidence_level: float  # 0.0 ~ 1.0
    market_mood: str  # "extreme_fear", "fear", "neutral", "greed", "extreme_greed"

    # 構成要素
    news_sentiment: float
    social_sentiment: float
    technical_sentiment: float
    macro_sentiment: float

    # 統計情報
    historical_percentile: float  # 過去データでの位置
    volatility_adjustment: float
    trend_momentum: float

    # メタデータ
    data_quality_score: float
    sample_sizes: Dict[str, int]
    calculation_timestamp: datetime = field(default_factory=datetime.now)

    def get_interpretation(self) -> Dict[str, str]:
        """指標解釈"""
        interpretations = {}

        # Fear & Greed Index解釈
        if self.fear_greed_index <= 20:
            interpretations["fear_greed"] = "極度の恐怖 - 買い機会の可能性"
        elif self.fear_greed_index <= 40:
            interpretations["fear_greed"] = "恐怖 - 慎重な楽観視"
        elif self.fear_greed_index <= 60:
            interpretations["fear_greed"] = "中立 - バランスの取れた市場"
        elif self.fear_greed_index <= 80:
            interpretations["fear_greed"] = "貪欲 - 注意が必要"
        else:
            interpretations["fear_greed"] = "極度の貪欲 - 売り機会の可能性"

        # センチメント解釈
        if self.sentiment_score >= 0.3:
            interpretations["sentiment"] = "強気センチメント"
        elif self.sentiment_score >= 0.1:
            interpretations["sentiment"] = "やや強気"
        elif self.sentiment_score <= -0.3:
            interpretations["sentiment"] = "弱気センチメント"
        elif self.sentiment_score <= -0.1:
            interpretations["sentiment"] = "やや弱気"
        else:
            interpretations["sentiment"] = "中立的センチメント"

        return interpretations


class MarketPsychologyAnalyzer:
    """市場心理分析システム"""

    def __init__(self, config: Optional[MarketPsychologyConfig] = None):
        self.config = config or MarketPsychologyConfig()

        # 分析エンジン初期化
        self.sentiment_engine = create_sentiment_engine()
        self.news_analyzer = NewsAnalyzer()
        self.social_analyzer = SocialMediaAnalyzer()

        # データキャッシュ
        self.indicator_cache = {}
        self.historical_data = []

        # 統計処理
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)

        logger.info("Market Psychology Analyzer 初期化完了")

    async def calculate_market_psychology(
        self,
        symbols: List[str] = None,
        include_news: bool = True,
        include_social: bool = True,
        include_technical: bool = True,
        include_macro: bool = True,
    ) -> MarketPsychologyIndex:
        """市場心理指標計算"""

        logger.info("市場心理指標計算開始")
        start_time = time.time()

        # 各データソースの分析
        analysis_results = {}
        data_quality_scores = {}
        sample_sizes = {}

        # ニュース分析
        if include_news:
            try:
                news_articles = await self.news_analyzer.fetch_news(
                    keywords=symbols or ["stock market", "financial market"],
                    hours_back=24,
                )
                news_result = self.news_analyzer.analyze_articles(news_articles)
                analysis_results["news"] = news_result.overall_sentiment
                data_quality_scores["news"] = news_result.confidence_score
                sample_sizes["news"] = len(news_articles)

                logger.info(f"ニュース分析完了: {len(news_articles)} 記事")
            except Exception as e:
                logger.error(f"ニュース分析エラー: {e}")
                analysis_results["news"] = 0.0
                data_quality_scores["news"] = 0.0
                sample_sizes["news"] = 0

        # ソーシャル分析
        if include_social:
            try:
                social_posts = await self.social_analyzer.collect_social_data(
                    keywords=symbols or ["$SPY", "$QQQ", "stock market"], hours_back=24
                )
                social_result = self.social_analyzer.analyze_social_sentiment(social_posts)
                analysis_results["social"] = social_result.overall_sentiment
                data_quality_scores["social"] = social_result.confidence_score
                sample_sizes["social"] = len(social_posts)

                logger.info(f"ソーシャル分析完了: {len(social_posts)} 投稿")
            except Exception as e:
                logger.error(f"ソーシャル分析エラー: {e}")
                analysis_results["social"] = 0.0
                data_quality_scores["social"] = 0.0
                sample_sizes["social"] = 0

        # テクニカル分析
        if include_technical:
            technical_indicators = self._get_technical_indicators()
            analysis_results["technical"] = self._calculate_technical_sentiment(
                technical_indicators
            )
            data_quality_scores["technical"] = 0.8  # 固定値（データの信頼性は高い）
            sample_sizes["technical"] = 6  # 6つの指標

        # マクロ経済分析
        if include_macro:
            macro_indicators = self._get_macro_indicators()
            analysis_results["macro"] = self._calculate_macro_sentiment(macro_indicators)
            data_quality_scores["macro"] = 0.7  # 固定値
            sample_sizes["macro"] = 6  # 6つの指標

        # 重み付き総合センチメント計算
        weighted_sentiment = self._calculate_weighted_sentiment(analysis_results)

        # Fear & Greed Index計算
        fear_greed_index = self._calculate_fear_greed_index(analysis_results)

        # 信頼度計算
        confidence_level = self._calculate_confidence_level(data_quality_scores, sample_sizes)

        # 市場ムード判定
        market_mood = self._determine_market_mood(fear_greed_index)

        # 履歴パーセンタイル
        historical_percentile = self._calculate_historical_percentile(fear_greed_index)

        # ボラティリティ調整
        volatility_adjustment = self._calculate_volatility_adjustment()

        # トレンドモメンタム
        trend_momentum = self._calculate_trend_momentum()

        # 総合データ品質スコア
        data_quality_score = (
            np.mean(list(data_quality_scores.values())) if data_quality_scores else 0.0
        )

        # 結果作成
        psychology_index = MarketPsychologyIndex(
            fear_greed_index=fear_greed_index,
            sentiment_score=weighted_sentiment,
            confidence_level=confidence_level,
            market_mood=market_mood,
            news_sentiment=analysis_results.get("news", 0.0),
            social_sentiment=analysis_results.get("social", 0.0),
            technical_sentiment=analysis_results.get("technical", 0.0),
            macro_sentiment=analysis_results.get("macro", 0.0),
            historical_percentile=historical_percentile,
            volatility_adjustment=volatility_adjustment,
            trend_momentum=trend_momentum,
            data_quality_score=data_quality_score,
            sample_sizes=sample_sizes,
        )

        # 履歴に追加
        self.historical_data.append(psychology_index)

        # 古いデータを削除（最新100件のみ保持）
        if len(self.historical_data) > 100:
            self.historical_data = self.historical_data[-100:]

        calculation_time = time.time() - start_time
        logger.info(f"市場心理指標計算完了: {calculation_time:.2f}秒")
        logger.info(
            f"Fear & Greed Index: {fear_greed_index:.1f}, センチメント: {weighted_sentiment:.3f}"
        )

        return psychology_index

    def _get_technical_indicators(self) -> TechnicalIndicators:
        """テクニカル指標取得（模擬データ）"""

        # 実際の実装では外部データソースから取得
        # ここでは模擬データを生成

        return TechnicalIndicators(
            vix=np.random.uniform(15, 35),  # VIX: 15-35
            put_call_ratio=np.random.uniform(0.7, 1.3),  # Put/Call: 0.7-1.3
            advance_decline_ratio=np.random.uniform(0.8, 1.2),  # A/D: 0.8-1.2
            new_high_low_ratio=np.random.uniform(0.5, 2.0),  # H/L: 0.5-2.0
            margin_debt_ratio=np.random.uniform(0.8, 1.5),  # Margin: 0.8-1.5
            insider_trading_ratio=np.random.uniform(0.6, 1.4),  # Insider: 0.6-1.4
        )

    def _get_macro_indicators(self) -> MacroIndicators:
        """マクロ経済指標取得（模擬データ）"""

        # 実際の実装では経済データAPIから取得

        return MacroIndicators(
            interest_rate=np.random.uniform(0.01, 0.06),  # 1-6%
            inflation_rate=np.random.uniform(0.01, 0.05),  # 1-5%
            unemployment_rate=np.random.uniform(0.03, 0.08),  # 3-8%
            gdp_growth=np.random.uniform(0.01, 0.04),  # 1-4%
            currency_strength=np.random.uniform(0.9, 1.1),  # 90-110
            commodity_index=np.random.uniform(90, 110),  # 90-110
        )

    def _calculate_technical_sentiment(self, indicators: TechnicalIndicators) -> float:
        """テクニカル指標からセンチメント計算"""

        # VIX (恐怖指数): 低い方が楽観的
        vix_sentiment = np.clip((35 - indicators.vix) / 20, -1, 1)  # 15-35 -> 1 to -1

        # Put/Call比率: 低い方が楽観的
        pc_sentiment = np.clip((1.0 - indicators.put_call_ratio) / 0.3, -1, 1)

        # 騰落レシオ: 高い方が楽観的
        ad_sentiment = np.clip((indicators.advance_decline_ratio - 1.0) / 0.2, -1, 1)

        # 新高値/新安値比率: 高い方が楽観的
        hl_sentiment = np.clip((indicators.new_high_low_ratio - 1.0) / 0.5, -1, 1)

        # マージンデビット: 高い方が楽観的（但し過度は注意）
        margin_sentiment = np.clip((indicators.margin_debt_ratio - 1.0) / 0.3, -1, 1)

        # インサイダー取引: 低い方が楽観的
        insider_sentiment = np.clip((1.0 - indicators.insider_trading_ratio) / 0.4, -1, 1)

        # 加重平均
        weights = [0.3, 0.2, 0.2, 0.15, 0.1, 0.05]
        sentiments = [
            vix_sentiment,
            pc_sentiment,
            ad_sentiment,
            hl_sentiment,
            margin_sentiment,
            insider_sentiment,
        ]

        technical_sentiment = np.average(sentiments, weights=weights)

        return np.clip(technical_sentiment, -1.0, 1.0)

    def _calculate_macro_sentiment(self, indicators: MacroIndicators) -> float:
        """マクロ指標からセンチメント計算"""

        # 金利: 適度な水準が良い（2-4%）
        rate_sentiment = 1.0 - abs(indicators.interest_rate - 0.03) / 0.03

        # インフレ: 低い方が良い（2%以下）
        inflation_sentiment = 1.0 - max(0, indicators.inflation_rate - 0.02) / 0.03

        # 失業率: 低い方が良い
        unemployment_sentiment = 1.0 - indicators.unemployment_rate / 0.08

        # GDP成長: 高い方が良い
        gdp_sentiment = indicators.gdp_growth / 0.04

        # 通貨強度: 安定が良い（1.0に近い）
        currency_sentiment = 1.0 - abs(indicators.currency_strength - 1.0) / 0.1

        # 商品指数: 安定が良い（100に近い）
        commodity_sentiment = 1.0 - abs(indicators.commodity_index - 100) / 20

        # 加重平均
        weights = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]
        sentiments = [
            rate_sentiment,
            inflation_sentiment,
            unemployment_sentiment,
            gdp_sentiment,
            currency_sentiment,
            commodity_sentiment,
        ]

        macro_sentiment = np.average(sentiments, weights=weights)

        # -1 ~ 1に正規化
        return np.clip(macro_sentiment * 2 - 1, -1.0, 1.0)

    def _calculate_weighted_sentiment(self, analysis_results: Dict[str, float]) -> float:
        """重み付き総合センチメント計算"""

        weights = {
            "news": self.config.news_weight,
            "social": self.config.social_weight,
            "technical": self.config.technical_weight,
            "macro": self.config.macro_weight,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for source, sentiment in analysis_results.items():
            if source in weights:
                weight = weights[source]
                weighted_sum += sentiment * weight
                total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0

    def _calculate_fear_greed_index(self, analysis_results: Dict[str, float]) -> float:
        """Fear & Greed Index計算"""

        # センチメントスコアを0-100に変換
        base_score = 50  # 中立点

        # 各要素の寄与度
        contributions = {}

        if "news" in analysis_results:
            contributions["news"] = analysis_results["news"] * 25 * self.config.news_weight

        if "social" in analysis_results:
            contributions["social"] = analysis_results["social"] * 25 * self.config.social_weight

        if "technical" in analysis_results:
            contributions["technical"] = (
                analysis_results["technical"] * 25 * self.config.technical_weight
            )

        if "macro" in analysis_results:
            contributions["macro"] = analysis_results["macro"] * 25 * self.config.macro_weight

        # 総合スコア計算
        total_contribution = sum(contributions.values())
        fear_greed_score = base_score + total_contribution

        # 平滑化（前回値との加重平均）
        if self.historical_data:
            prev_score = self.historical_data[-1].fear_greed_index
            fear_greed_score = prev_score * self.config.fear_greed_smoothing + fear_greed_score * (
                1 - self.config.fear_greed_smoothing
            )

        return np.clip(fear_greed_score, 0.0, 100.0)

    def _calculate_confidence_level(
        self, data_quality_scores: Dict[str, float], sample_sizes: Dict[str, int]
    ) -> float:
        """信頼度レベル計算"""

        if not data_quality_scores:
            return 0.0

        # データ品質の重み付き平均
        quality_scores = list(data_quality_scores.values())
        avg_quality = np.mean(quality_scores)

        # サンプルサイズによる信頼度調整
        total_samples = sum(sample_sizes.values())
        sample_confidence = min(total_samples / 100.0, 1.0)  # 100サンプルで満点

        # 総合信頼度
        confidence = avg_quality * 0.7 + sample_confidence * 0.3

        return np.clip(confidence, 0.0, 1.0)

    def _determine_market_mood(self, fear_greed_index: float) -> str:
        """市場ムード判定"""

        if fear_greed_index <= self.config.extreme_fear_threshold:
            return "extreme_fear"
        elif fear_greed_index <= self.config.fear_threshold:
            return "fear"
        elif fear_greed_index <= self.config.greed_threshold:
            return "neutral"
        elif fear_greed_index <= self.config.extreme_greed_threshold:
            return "greed"
        else:
            return "extreme_greed"

    def _calculate_historical_percentile(self, current_value: float) -> float:
        """過去データでのパーセンタイル計算"""

        if len(self.historical_data) < 10:
            return 50.0  # データ不足時は中央値

        historical_values = [
            data.fear_greed_index for data in self.historical_data[-30:]
        ]  # 過去30点
        percentile = stats.percentileofscore(historical_values, current_value)

        return percentile

    def _calculate_volatility_adjustment(self) -> float:
        """ボラティリティ調整計算"""

        if len(self.historical_data) < self.config.volatility_window:
            return 0.0

        # 過去データの標準偏差
        recent_values = [
            data.fear_greed_index for data in self.historical_data[-self.config.volatility_window :]
        ]
        volatility = np.std(recent_values)

        # 正規化（0-1）
        normalized_volatility = min(volatility / 20.0, 1.0)  # 20が最大ボラティリティと仮定

        return normalized_volatility

    def _calculate_trend_momentum(self) -> float:
        """トレンドモメンタム計算"""

        if len(self.historical_data) < 5:
            return 0.0

        # 過去5点での線形回帰
        recent_values = [data.fear_greed_index for data in self.historical_data[-5:]]
        x = np.arange(len(recent_values))

        if len(recent_values) > 1:
            slope, _, r_value, _, _ = stats.linregress(x, recent_values)
            # 傾きを正規化
            momentum = np.clip(slope / 5.0, -1.0, 1.0)  # 1ステップあたり最大5の変化と仮定
            return momentum * r_value**2  # R²で重み付け

        return 0.0

    def get_market_psychology_summary(self, index: MarketPsychologyIndex) -> Dict[str, Any]:
        """市場心理サマリー作成"""

        interpretations = index.get_interpretation()

        # レーダーチャート用データ
        radar_data = {
            "ニュースセンチメント": (index.news_sentiment + 1) * 50,  # 0-100に変換
            "ソーシャルセンチメント": (index.social_sentiment + 1) * 50,
            "テクニカル指標": (index.technical_sentiment + 1) * 50,
            "マクロ経済": (index.macro_sentiment + 1) * 50,
        }

        # アラート生成
        alerts = []
        if index.fear_greed_index <= 20:
            alerts.append("極度の恐怖状態 - 買い機会の可能性")
        elif index.fear_greed_index >= 80:
            alerts.append("極度の貪欲状態 - 利益確定を検討")

        if index.volatility_adjustment > 0.8:
            alerts.append("市場ボラティリティが高水準")

        if abs(index.trend_momentum) > 0.7:
            direction = "上昇" if index.trend_momentum > 0 else "下降"
            alerts.append(f"強い{direction}トレンド継続中")

        return {
            "fear_greed_index": index.fear_greed_index,
            "market_mood": index.market_mood,
            "sentiment_score": index.sentiment_score,
            "confidence_level": index.confidence_level,
            "interpretations": interpretations,
            "radar_data": radar_data,
            "alerts": alerts,
            "historical_percentile": index.historical_percentile,
            "data_quality_score": index.data_quality_score,
            "sample_sizes": index.sample_sizes,
            "timestamp": index.calculation_timestamp.isoformat(),
        }

    def export_psychology_analysis(self, index: MarketPsychologyIndex, format: str = "json") -> str:
        """市場心理分析エクスポート"""

        if format == "json":
            summary = self.get_market_psychology_summary(index)

            export_data = {
                **summary,
                "detailed_components": {
                    "news_sentiment": index.news_sentiment,
                    "social_sentiment": index.social_sentiment,
                    "technical_sentiment": index.technical_sentiment,
                    "macro_sentiment": index.macro_sentiment,
                    "volatility_adjustment": index.volatility_adjustment,
                    "trend_momentum": index.trend_momentum,
                },
            }

            return json.dumps(export_data, indent=2, ensure_ascii=False)

        else:
            raise ValueError(f"未対応のエクスポート形式: {format}")

    def get_historical_trend(self, days: int = 30) -> Dict[str, List[float]]:
        """過去トレンド取得"""

        if not self.historical_data:
            return {}

        # 過去N日のデータ
        recent_data = self.historical_data[-days:]

        return {
            "timestamps": [data.calculation_timestamp.isoformat() for data in recent_data],
            "fear_greed_index": [data.fear_greed_index for data in recent_data],
            "sentiment_score": [data.sentiment_score for data in recent_data],
            "confidence_level": [data.confidence_level for data in recent_data],
            "news_sentiment": [data.news_sentiment for data in recent_data],
            "social_sentiment": [data.social_sentiment for data in recent_data],
            "technical_sentiment": [data.technical_sentiment for data in recent_data],
            "macro_sentiment": [data.macro_sentiment for data in recent_data],
        }


# 便利関数
def analyze_market_psychology(symbols: List[str] = None) -> MarketPsychologyIndex:
    """市場心理分析（簡易インターフェース）"""

    async def _analyze():
        analyzer = MarketPsychologyAnalyzer()
        return await analyzer.calculate_market_psychology(symbols=symbols)

    import asyncio

    return asyncio.run(_analyze())


if __name__ == "__main__":
    # 市場心理分析テスト
    print("=== Next-Gen AI Market Psychology System テスト ===")

    async def test_market_psychology():
        analyzer = MarketPsychologyAnalyzer()

        print("市場心理指標計算テスト実行中...")

        # 分析実行
        psychology_index = await analyzer.calculate_market_psychology(
            symbols=["SPY", "QQQ", "AAPL"],
            include_news=True,
            include_social=True,
            include_technical=True,
            include_macro=True,
        )

        print("\n=== 市場心理分析結果 ===")
        print(f"Fear & Greed Index: {psychology_index.fear_greed_index:.1f}")
        print(f"市場ムード: {psychology_index.market_mood}")
        print(f"センチメントスコア: {psychology_index.sentiment_score:.3f}")
        print(f"信頼度レベル: {psychology_index.confidence_level:.3f}")

        print("\n=== 構成要素 ===")
        print(f"ニュースセンチメント: {psychology_index.news_sentiment:.3f}")
        print(f"ソーシャルセンチメント: {psychology_index.social_sentiment:.3f}")
        print(f"テクニカル指標: {psychology_index.technical_sentiment:.3f}")
        print(f"マクロ経済指標: {psychology_index.macro_sentiment:.3f}")

        print("\n=== 統計情報 ===")
        print(f"過去パーセンタイル: {psychology_index.historical_percentile:.1f}%")
        print(f"ボラティリティ調整: {psychology_index.volatility_adjustment:.3f}")
        print(f"トレンドモメンタム: {psychology_index.trend_momentum:.3f}")
        print(f"データ品質スコア: {psychology_index.data_quality_score:.3f}")

        # 解釈表示
        interpretations = psychology_index.get_interpretation()
        print("\n=== 解釈 ===")
        for key, interpretation in interpretations.items():
            print(f"{key}: {interpretation}")

        # サマリー作成
        summary = analyzer.get_market_psychology_summary(psychology_index)
        print("\n=== アラート ===")
        for alert in summary["alerts"]:
            print(f"⚠️  {alert}")

        # エクスポートテスト
        json_export = analyzer.export_psychology_analysis(psychology_index)
        print(f"\nJSON エクスポート長: {len(json_export)} 文字")

        # 複数回実行して履歴作成
        print("\n履歴データ作成中...")
        for i in range(3):
            await analyzer.calculate_market_psychology()
            await asyncio.sleep(0.1)  # 短い間隔

        # トレンド取得
        trend_data = analyzer.get_historical_trend(days=10)
        print(f"履歴データ点数: {len(trend_data.get('fear_greed_index', []))}")

    # テスト実行
    import asyncio

    try:
        asyncio.run(test_market_psychology())
    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback

        traceback.print_exc()

    print("\n=== テスト完了 ===")
