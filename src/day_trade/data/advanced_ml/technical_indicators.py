#!/usr/bin/env python3
"""
Advanced ML Engine Technical Indicators Module

ML拡張テクニカル指標計算機能
"""

from typing import Dict

import pandas as pd

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class TechnicalIndicatorCalculator:
    """ML拡張テクニカル指標計算クラス"""

    def calculate_advanced_technical_indicators(
        self, data: pd.DataFrame, symbol: str = "UNKNOWN"
    ) -> Dict[str, float]:
        """
        高度テクニカル指標計算（ML拡張版）

        Args:
            data: 価格データ
            symbol: 銘柄コード

        Returns:
            テクニカル指標スコア辞書
        """
        try:
            if data is None or data.empty:
                logger.warning(f"データが空です: {symbol}")
                return self._get_default_ml_scores()

            # 基本的なテクニカル指標計算
            close_prices = (
                data["終値"] if "終値" in data.columns else data.get("Close", pd.Series())
            )

            if close_prices.empty or len(close_prices) < 20:
                logger.warning(f"価格データが不足: {symbol} ({len(close_prices)} 件)")
                return self._get_default_ml_scores()

            # ML強化テクニカル指標
            ml_scores = {}

            try:
                # トレンド強度スコア (ML拡張)
                sma_20 = close_prices.rolling(20).mean()
                sma_50 = (
                    close_prices.rolling(50).mean()
                    if len(close_prices) >= 50
                    else sma_20
                )
                trend_strength = self._calculate_ml_trend_strength(
                    close_prices, sma_20, sma_50
                )
                ml_scores["trend_strength"] = min(100, max(0, trend_strength))

                # 価格変動予測スコア
                volatility = (
                    close_prices.pct_change().rolling(20).std().iloc[-1]
                    if len(close_prices) > 20
                    else 0.02
                )
                volume_data = data.get("出来高", data.get("Volume", pd.Series()))
                volatility_score = self._calculate_volatility_score(
                    close_prices, volatility, volume_data
                )
                ml_scores["volatility_prediction"] = min(100, max(0, volatility_score))

                # パターン認識スコア (簡易版)
                pattern_score = self._calculate_pattern_recognition_score(close_prices)
                ml_scores["pattern_recognition"] = min(100, max(0, pattern_score))

                logger.debug(f"ML指標計算完了: {symbol}")
                return ml_scores

            except Exception as e:
                logger.warning(f"ML指標計算でエラー {symbol}: {e}")
                return self._get_default_ml_scores()

        except Exception as e:
            logger.error(f"高度テクニカル指標計算エラー {symbol}: {e}")
            return self._get_default_ml_scores()

    def _calculate_ml_trend_strength(
        self, prices: pd.Series, sma_20: pd.Series, sma_50: pd.Series
    ) -> float:
        """ML拡張トレンド強度計算"""
        try:
            # 現在価格と移動平均の関係
            current_price = prices.iloc[-1]
            current_sma20 = (
                sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else current_price
            )
            current_sma50 = (
                sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else current_price
            )

            # 価格位置スコア
            price_position = ((current_price - current_sma20) / current_sma20) * 100

            # トレンド方向性
            ma_trend = (
                ((current_sma20 - current_sma50) / current_sma50) * 100
                if current_sma50 != 0
                else 0
            )

            # 勢い計算
            momentum = (
                prices.pct_change(5).iloc[-1] * 100 if len(prices) > 5 else 0
            )

            # 統合スコア
            trend_score = (
                (price_position * 0.4) + (ma_trend * 0.3) + (momentum * 0.3) + 50
            )

            return trend_score

        except Exception:
            return 50.0  # 中立値

    def _calculate_volatility_score(
        self, prices: pd.Series, volatility: float, volume: pd.Series
    ) -> float:
        """ボラティリティ予測スコア"""
        try:
            # ボラティリティ正規化 (0-100スケール)
            volatility_normalized = min(100, volatility * 1000)  # 0.1 = 100

            # 出来高影響
            volume_factor = 1.0
            if not volume.empty and len(volume) > 20:
                avg_volume = volume.rolling(20).mean().iloc[-1]
                current_volume = volume.iloc[-1]
                if avg_volume > 0:
                    volume_factor = min(2.0, current_volume / avg_volume)

            # 予測スコア
            prediction_score = volatility_normalized * volume_factor

            return min(100, prediction_score)

        except Exception:
            return 50.0

    def _calculate_pattern_recognition_score(self, prices: pd.Series) -> float:
        """パターン認識スコア (簡易版)"""
        try:
            if len(prices) < 10:
                return 50.0

            # 最近の価格パターン分析
            recent_prices = prices.tail(10)

            # 連続上昇/下降の検出
            changes = recent_prices.pct_change().dropna()

            # 上昇連続度
            up_streak = 0
            down_streak = 0
            for change in changes:
                if change > 0:
                    up_streak += 1
                    down_streak = 0
                elif change < 0:
                    down_streak += 1
                    up_streak = 0

            # パターン強度
            if up_streak >= 3:
                pattern_score = 60 + min(20, up_streak * 5)
            elif down_streak >= 3:
                pattern_score = 40 - min(20, down_streak * 5)
            else:
                # 価格変動の安定性
                stability = 1 / (1 + changes.std()) if changes.std() > 0 else 0.5
                pattern_score = 50 + (stability - 0.5) * 40

            return min(100, max(0, pattern_score))

        except Exception:
            return 50.0

    def _get_default_ml_scores(self) -> Dict[str, float]:
        """デフォルトMLスコア"""
        return {
            "trend_strength": 50.0,
            "volatility_prediction": 50.0,
            "pattern_recognition": 50.0,
        }