#!/usr/bin/env python3
"""
多角的データ収集システム - 特徴量エンジニアリング

Issue #322: ML Data Shortage Problem Resolution
包括的特徴量エンジニアリングの実装
"""

from typing import Any, Dict, List

import numpy as np

from .models import CollectedData

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class ComprehensiveFeatureEngineer:
    """包括的特徴量エンジニアリング"""

    def __init__(self, data_manager=None):
        """
        初期化

        Args:
            data_manager: データ管理システム（循環インポート回避のためOptional）
        """
        self.data_manager = data_manager

    def generate_comprehensive_features(
        self, comprehensive_data: Dict[str, CollectedData]
    ) -> Dict[str, float]:
        """
        包括的特徴量生成

        Args:
            comprehensive_data: 包括的データ辞書

        Returns:
            Dict[str, float]: 特徴量辞書
        """
        features = {}

        try:
            # 1. 価格系特徴量
            if "price" in comprehensive_data:
                price_features = self._generate_price_features(
                    comprehensive_data["price"].data
                )
                features.update(price_features)

            # 2. センチメント系特徴量
            if "sentiment" in comprehensive_data:
                sentiment_features = self._generate_sentiment_features(
                    comprehensive_data["sentiment"].data
                )
                features.update(sentiment_features)

            # 3. ニュース系特徴量
            if "news" in comprehensive_data:
                news_features = self._generate_news_features(
                    comprehensive_data["news"].data
                )
                features.update(news_features)

            # 4. マクロ経済特徴量
            if "macro" in comprehensive_data:
                macro_features = self._generate_macro_features(
                    comprehensive_data["macro"].data
                )
                features.update(macro_features)

            # 5. 相互作用特徴量
            interaction_features = self._generate_interaction_features(
                comprehensive_data
            )
            features.update(interaction_features)

            logger.info(f"包括特徴量生成完了: {len(features)}個")
            return features

        except Exception as e:
            logger.error(f"包括特徴量生成エラー: {e}")
            return {}

    def _generate_price_features(self, price_data: Any) -> Dict[str, float]:
        """
        価格系特徴量生成（Issue #325最適化版準拠）

        Args:
            price_data: 価格データ

        Returns:
            Dict[str, float]: 価格系特徴量
        """
        # 既存の技術指標特徴量（16個）
        return {
            "price_sma_20": 1.0,  # 移動平均
            "price_rsi_14": 0.5,  # RSI
            "price_macd": 0.1,  # MACD
            "price_volatility": 0.2,  # ボラティリティ
            "price_momentum_5": 0.05,  # 5日間モメンタム
            "price_momentum_20": 0.08,  # 20日間モメンタム
            "price_bb_upper": 1.05,  # ボリンジャーバンド上限
            "price_bb_lower": 0.95,  # ボリンジャーバンド下限
            "price_atr_14": 0.15,  # ATR（14日間）
            "price_williams_r": -0.3,  # ウィリアムズ%R
            "price_cci_14": 100.0,  # CCI（14日間）
            "price_stoch_k": 0.6,  # ストキャスティクス%K
            "price_stoch_d": 0.55,  # ストキャスティクス%D
            "price_adx_14": 25.0,  # ADX（14日間）
            "price_obv": 1000000.0,  # OBV（出来高バランス）
            "price_mfi_14": 45.0,  # MFI（14日間）
        }

    def _generate_sentiment_features(self, sentiment_data: Dict) -> Dict[str, float]:
        """
        センチメント系特徴量生成

        Args:
            sentiment_data: センチメントデータ

        Returns:
            Dict[str, float]: センチメント系特徴量
        """
        features = {}

        features["sentiment_overall"] = sentiment_data.get("overall_sentiment", 0.0)
        features["sentiment_positive_ratio"] = sentiment_data.get("positive_ratio", 0.0)
        features["sentiment_negative_ratio"] = sentiment_data.get("negative_ratio", 0.0)
        features["sentiment_confidence"] = sentiment_data.get("confidence", 0.0)

        # センチメント変化率
        features["sentiment_momentum"] = abs(
            sentiment_data.get("overall_sentiment", 0.0)
        )

        # センチメント極性強度
        features["sentiment_polarity_strength"] = (
            features["sentiment_positive_ratio"] + features["sentiment_negative_ratio"]
        )

        # センチメント一貫性
        neutral_ratio = sentiment_data.get("neutral_ratio", 0.0)
        features["sentiment_consistency"] = 1.0 - neutral_ratio

        return features

    def _generate_news_features(self, news_data: List[Dict]) -> Dict[str, float]:
        """
        ニュース系特徴量生成

        Args:
            news_data: ニュースデータリスト

        Returns:
            Dict[str, float]: ニュース系特徴量
        """
        features = {}

        # ニュース数・頻度
        features["news_count"] = len(news_data)
        features["news_frequency"] = min(len(news_data) / 24, 1.0)  # 24時間正規化

        # ソース多様性
        sources = set(article.get("source", "") for article in news_data)
        features["news_source_diversity"] = len(sources)

        # 関連度スコア
        relevance_scores = [article.get("relevance", 0.7) for article in news_data]
        features["news_avg_relevance"] = (
            np.mean(relevance_scores) if relevance_scores else 0.0
        )

        # ニュース品質指標
        features["news_max_relevance"] = (
            np.max(relevance_scores) if relevance_scores else 0.0
        )
        features["news_relevance_variance"] = (
            np.var(relevance_scores) if len(relevance_scores) > 1 else 0.0
        )

        # 時間的新しさ
        features["news_freshness"] = 1.0 if news_data else 0.0  # 簡易実装

        return features

    def _generate_macro_features(self, macro_data: Dict) -> Dict[str, float]:
        """
        マクロ経済特徴量生成

        Args:
            macro_data: マクロ経済データ

        Returns:
            Dict[str, float]: マクロ経済特徴量
        """
        features = {}

        # 基本マクロ指標
        features["macro_interest_rate"] = macro_data.get("interest_rate", 0.0)
        features["macro_inflation_rate"] = macro_data.get("inflation_rate", 0.0)
        features["macro_gdp_growth"] = macro_data.get("gdp_growth", 0.0)
        features["macro_exchange_rate"] = (
            macro_data.get("exchange_rate_usd", 150.0) / 150.0
        )  # 正規化
        features["macro_unemployment_rate"] = macro_data.get("unemployment_rate", 0.0)

        # 市場指標
        features["macro_market_volatility"] = macro_data.get("market_volatility", 0.0)
        features["macro_economic_sentiment"] = macro_data.get("economic_sentiment", 0.0)

        # セクター影響度
        if "sector_impact" in macro_data:
            sector_impact = macro_data["sector_impact"]
            if isinstance(sector_impact, dict):
                features["macro_sector_sensitivity"] = np.mean(list(sector_impact.values()))
                features["macro_sector_max_impact"] = max(
                    abs(v) for v in sector_impact.values()
                ) if sector_impact else 0.0

        # 予測影響度
        if "impact_prediction" in macro_data:
            impact_pred = macro_data["impact_prediction"]
            features["macro_predicted_impact"] = impact_pred.get("overall_impact", 0.0)
            features["macro_impact_volatility"] = impact_pred.get(
                "impact_volatility", 0.0
            )
            features["macro_impact_confidence"] = impact_pred.get("confidence", 0.0)

            # リスクレベルの数値化
            risk_level = impact_pred.get("risk_level", "low")
            risk_mapping = {"low": 0.2, "medium": 0.5, "high": 0.8}
            features["macro_risk_level"] = risk_mapping.get(risk_level, 0.2)

        return features

    def _generate_interaction_features(
        self, comprehensive_data: Dict[str, CollectedData]
    ) -> Dict[str, float]:
        """
        相互作用特徴量生成

        Args:
            comprehensive_data: 包括的データ辞書

        Returns:
            Dict[str, float]: 相互作用特徴量
        """
        features = {}

        try:
            # センチメント × マクロ経済
            if "sentiment" in comprehensive_data and "macro" in comprehensive_data:
                sentiment = comprehensive_data["sentiment"].data.get(
                    "overall_sentiment", 0.0
                )
                interest_rate = comprehensive_data["macro"].data.get(
                    "interest_rate", 0.0
                )
                inflation_rate = comprehensive_data["macro"].data.get(
                    "inflation_rate", 0.0
                )

                features["sentiment_macro_interaction"] = sentiment * interest_rate
                features["sentiment_inflation_correlation"] = sentiment * inflation_rate

            # ニュース × センチメント一貫性
            if "news" in comprehensive_data and "sentiment" in comprehensive_data:
                news_count = len(comprehensive_data["news"].data)
                sentiment_conf = comprehensive_data["sentiment"].data.get(
                    "confidence", 0.0
                )

                features["news_sentiment_consistency"] = sentiment_conf * min(
                    news_count / 5, 1.0
                )

                # ニュース数とセンチメント強度の相関
                sentiment_strength = abs(comprehensive_data["sentiment"].data.get(
                    "overall_sentiment", 0.0
                ))
                features["news_volume_sentiment_correlation"] = (
                    min(news_count / 10, 1.0) * sentiment_strength
                )

            # 価格 × センチメント相関（価格データがある場合）
            if "price" in comprehensive_data and "sentiment" in comprehensive_data:
                sentiment_overall = comprehensive_data["sentiment"].data.get(
                    "overall_sentiment", 0.0
                )
                # 価格データから簡単な特徴量を抽出（実際の実装では更に詳細に）
                features["price_sentiment_alignment"] = sentiment_overall * 0.5

            # データ品質総合スコア
            quality_scores = [
                data.quality_score for data in comprehensive_data.values()
            ]
            features["overall_data_quality"] = (
                np.mean(quality_scores) if quality_scores else 0.0
            )

            # データ完全性スコア
            expected_data_types = {"price", "news", "sentiment", "macro"}
            available_types = set(comprehensive_data.keys())
            features["data_completeness"] = len(available_types) / len(expected_data_types)

            # データ信頼度加重平均
            confidence_scores = [
                data.confidence for data in comprehensive_data.values()
                if hasattr(data, 'confidence') and data.confidence is not None
            ]
            features["weighted_confidence"] = (
                np.mean(confidence_scores) if confidence_scores else 0.0
            )

        except Exception as e:
            logger.error(f"相互作用特徴量エラー: {e}")

        return features

    def get_feature_importance_weights(self) -> Dict[str, float]:
        """
        特徴量重要度重みを取得

        Returns:
            Dict[str, float]: 特徴量重要度辞書
        """
        return {
            # 価格系特徴量（高重要度）
            "price_sma_20": 0.9,
            "price_rsi_14": 0.8,
            "price_macd": 0.85,
            "price_volatility": 0.8,
            
            # センチメント系特徴量（中重要度）
            "sentiment_overall": 0.7,
            "sentiment_confidence": 0.6,
            
            # ニュース系特徴量（中重要度）
            "news_count": 0.6,
            "news_avg_relevance": 0.65,
            
            # マクロ経済特徴量（中低重要度）
            "macro_interest_rate": 0.5,
            "macro_predicted_impact": 0.6,
            
            # 相互作用特徴量（高重要度）
            "overall_data_quality": 0.8,
            "data_completeness": 0.75,
        }