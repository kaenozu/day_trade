"""
メタ特徴量計算モジュール
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="ensemble_meta_features")


class MetaFeatureCalculator:
    """メタ特徴量計算クラス"""

    def __init__(self):
        """初期化"""
        pass

    def calculate_meta_features(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> Dict[str, Any]:
        """
        メタ特徴量を計算
        
        Args:
            df: 価格データのDataFrame
            indicators: テクニカル指標のDataFrame
            patterns: チャートパターン認識結果
            
        Returns:
            メタ特徴量の辞書
        """
        try:
            meta_features = {}

            # 市場状況の特徴量
            if len(df) >= 20:
                # ボラティリティ特徴量
                volatility_features = self._calculate_volatility_features(df)
                meta_features.update(volatility_features)

                # トレンド特徴量
                trend_features = self._calculate_trend_features(df)
                meta_features.update(trend_features)

                # 価格位置特徴量
                price_features = self._calculate_price_features(df)
                meta_features.update(price_features)

            # テクニカル指標の特徴量
            if indicators is not None and not indicators.empty:
                indicator_features = self._calculate_indicator_features(indicators)
                meta_features.update(indicator_features)

            # 出来高の特徴量
            if "Volume" in df.columns and len(df) >= 10:
                volume_features = self._calculate_volume_features(df)
                meta_features.update(volume_features)

            # パターン特徴量
            if patterns:
                pattern_features = self._calculate_pattern_features(patterns)
                meta_features.update(pattern_features)

            return meta_features

        except Exception as e:
            logger.error(f"メタ特徴量計算エラー: {e}")
            return {}

    def _calculate_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """ボラティリティ関連特徴量を計算"""
        features = {}
        try:
            returns = df["Close"].pct_change().dropna()
            if len(returns) > 0:
                # 年率ボラティリティ
                features["volatility"] = returns.std() * np.sqrt(252)
                features["mean_return"] = returns.mean()

                # 最近のボラティリティ vs 長期ボラティリティ
                if len(returns) >= 40:
                    recent_vol = returns.tail(20).std() * np.sqrt(252)
                    long_term_vol = returns.std() * np.sqrt(252)
                    features["volatility_ratio"] = (
                        recent_vol / long_term_vol if long_term_vol > 0 else 1.0
                    )

                # リターンの偏度と尖度
                if len(returns) >= 30:
                    features["return_skewness"] = returns.skew()
                    features["return_kurtosis"] = returns.kurtosis()

        except Exception as e:
            logger.warning(f"ボラティリティ特徴量計算エラー: {e}")

        return features

    def _calculate_trend_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """トレンド関連特徴量を計算"""
        features = {}
        try:
            # 移動平均による トレンド強度
            if len(df) >= 50:
                sma_20 = df["Close"].rolling(20).mean()
                sma_50 = df["Close"].rolling(50).mean()
                if not sma_20.empty and not sma_50.empty:
                    features["trend_strength"] = (
                        sma_20.iloc[-1] / sma_50.iloc[-1] - 1
                    ) * 100

            # 価格の線形トレンド
            if len(df) >= 30:
                close_prices = df["Close"].iloc[-30:].values
                x = np.arange(len(close_prices))
                slope, _ = np.polyfit(x, close_prices, 1)
                features["price_trend_slope"] = slope

            # 移動平均の傾き
            if len(df) >= 20:
                sma_20 = df["Close"].rolling(20).mean()
                if len(sma_20.dropna()) >= 5:
                    recent_sma = sma_20.dropna().tail(5).values
                    x = np.arange(len(recent_sma))
                    sma_slope, _ = np.polyfit(x, recent_sma, 1)
                    features["sma_trend_slope"] = sma_slope

        except Exception as e:
            logger.warning(f"トレンド特徴量計算エラー: {e}")

        return features

    def _calculate_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """価格関連特徴量を計算"""
        features = {}
        try:
            current_price = df["Close"].iloc[-1]

            # 過去のレンジ内での位置
            if len(df) >= 20:
                high_20 = df["High"].rolling(20).max().iloc[-1]
                low_20 = df["Low"].rolling(20).min().iloc[-1]
                if high_20 != low_20:
                    features["price_position"] = (current_price - low_20) / (
                        high_20 - low_20
                    )

            # 異なる期間での価格位置
            for period in [10, 50]:
                if len(df) >= period:
                    high_period = df["High"].rolling(period).max().iloc[-1]
                    low_period = df["Low"].rolling(period).min().iloc[-1]
                    if high_period != low_period:
                        features[f"price_position_{period}d"] = (
                            current_price - low_period
                        ) / (high_period - low_period)

            # 価格変化率
            for period in [1, 5, 10]:
                if len(df) > period:
                    past_price = df["Close"].iloc[-period - 1]
                    features[f"price_change_{period}d"] = (
                        (current_price - past_price) / past_price * 100
                        if past_price > 0
                        else 0.0
                    )

        except Exception as e:
            logger.warning(f"価格特徴量計算エラー: {e}")

        return features

    def _calculate_indicator_features(self, indicators: pd.DataFrame) -> Dict[str, float]:
        """テクニカル指標特徴量を計算"""
        features = {}
        try:
            # RSI レベル
            if "RSI" in indicators.columns:
                rsi = indicators["RSI"].iloc[-1]
                features["rsi_level"] = rsi
                # RSI の極値判定
                features["rsi_overbought"] = 1.0 if rsi > 70 else 0.0
                features["rsi_oversold"] = 1.0 if rsi < 30 else 0.0

            # MACD 特徴量
            if "MACD" in indicators.columns and "MACD_Signal" in indicators.columns:
                macd = indicators["MACD"].iloc[-1]
                macd_signal = indicators["MACD_Signal"].iloc[-1]
                features["macd_divergence"] = macd - macd_signal

                # MACD ヒストグラム
                if "MACD_Histogram" in indicators.columns:
                    features["macd_histogram"] = indicators["MACD_Histogram"].iloc[-1]

            # ボリンジャーバンド
            if all(
                col in indicators.columns
                for col in ["BB_Upper", "BB_Middle", "BB_Lower"]
            ):
                current_price = indicators.index[-1]  # 仮定
                bb_upper = indicators["BB_Upper"].iloc[-1]
                bb_lower = indicators["BB_Lower"].iloc[-1]
                bb_middle = indicators["BB_Middle"].iloc[-1]

                if bb_upper != bb_lower:
                    # ボリンジャーバンド内位置 (仮の実装)
                    features["bb_position"] = 0.5  # 実際の価格データが必要

            # ストキャスティクス
            if "%K" in indicators.columns and "%D" in indicators.columns:
                stoch_k = indicators["%K"].iloc[-1]
                stoch_d = indicators["%D"].iloc[-1]
                features["stochastic_k"] = stoch_k
                features["stochastic_d"] = stoch_d
                features["stochastic_divergence"] = stoch_k - stoch_d

        except Exception as e:
            logger.warning(f"指標特徴量計算エラー: {e}")

        return features

    def _calculate_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """出来高特徴量を計算"""
        features = {}
        try:
            current_volume = df["Volume"].iloc[-1]

            # 出来高比率
            for period in [10, 20]:
                if len(df) >= period:
                    avg_volume = df["Volume"].rolling(period).mean().iloc[-1]
                    features[f"volume_ratio_{period}d"] = (
                        current_volume / avg_volume if avg_volume > 0 else 1.0
                    )

            # 出来高トレンド
            if len(df) >= 10:
                recent_volumes = df["Volume"].tail(10).values
                x = np.arange(len(recent_volumes))
                volume_slope, _ = np.polyfit(x, recent_volumes, 1)
                features["volume_trend"] = volume_slope

            # 出来高と価格の関係
            if len(df) >= 20:
                volume_change = df["Volume"].pct_change().tail(20)
                price_change = df["Close"].pct_change().tail(20)
                if len(volume_change.dropna()) > 1 and len(price_change.dropna()) > 1:
                    correlation = np.corrcoef(
                        volume_change.dropna(), price_change.dropna()
                    )[0, 1]
                    features["volume_price_correlation"] = (
                        correlation if not np.isnan(correlation) else 0.0
                    )

        except Exception as e:
            logger.warning(f"出来高特徴量計算エラー: {e}")

        return features

    def _calculate_pattern_features(self, patterns: Dict) -> Dict[str, float]:
        """パターン特徴量を計算"""
        features = {}
        try:
            # チャートパターンの存在
            pattern_count = 0
            bullish_patterns = 0
            bearish_patterns = 0

            for pattern_name, pattern_info in patterns.items():
                if isinstance(pattern_info, dict) and pattern_info.get("detected", False):
                    pattern_count += 1
                    features[f"pattern_{pattern_name}"] = 1.0

                    # パターンの強気/弱気判定
                    if "bullish" in pattern_name.lower() or "support" in pattern_name.lower():
                        bullish_patterns += 1
                    elif "bearish" in pattern_name.lower() or "resistance" in pattern_name.lower():
                        bearish_patterns += 1

            features["total_patterns"] = pattern_count
            features["bullish_patterns"] = bullish_patterns
            features["bearish_patterns"] = bearish_patterns

            # パターンバランス
            if pattern_count > 0:
                features["pattern_bias"] = (bullish_patterns - bearish_patterns) / pattern_count

        except Exception as e:
            logger.warning(f"パターン特徴量計算エラー: {e}")

        return features

    def get_feature_summary(self, meta_features: Dict[str, Any]) -> Dict[str, Any]:
        """メタ特徴量のサマリーを取得"""
        if not meta_features:
            return {"feature_count": 0, "categories": []}

        categories = []
        if any("volatility" in key for key in meta_features.keys()):
            categories.append("volatility")
        if any("trend" in key for key in meta_features.keys()):
            categories.append("trend")
        if any("price" in key for key in meta_features.keys()):
            categories.append("price")
        if any("rsi" in key or "macd" in key for key in meta_features.keys()):
            categories.append("indicators")
        if any("volume" in key for key in meta_features.keys()):
            categories.append("volume")
        if any("pattern" in key for key in meta_features.keys()):
            categories.append("patterns")

        return {
            "feature_count": len(meta_features),
            "categories": categories,
            "numeric_features": sum(
                1 for v in meta_features.values() if isinstance(v, (int, float))
            ),
        }