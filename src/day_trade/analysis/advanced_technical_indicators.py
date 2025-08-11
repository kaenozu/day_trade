#!/usr/bin/env python3
"""
高度テクニカル指標分析システム
Issue #315: 高度テクニカル指標・ML機能拡張

Elliott Wave・Ichimoku・Fibonacci・高度ボリンジャーバンド分析
"""

import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 依存パッケージチェック
try:
    import talib

    TALIB_AVAILABLE = True
    logger.info("TA-Lib利用可能")
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib未インストール - 基本実装を使用")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class AdvancedTechnicalIndicators:
    """
    高度テクニカル指標計算クラス

    従来の基本指標に加え、複雑なパターン認識・波動分析を実装
    """

    def __init__(self):
        """初期化"""
        self.fibonacci_levels = [
            0.0,
            0.236,
            0.382,
            0.5,
            0.618,
            0.786,
            1.0,
            1.618,
            2.618,
        ]
        self.elliott_wave_patterns = {}

        logger.info("高度テクニカル指標システム初期化完了")

    def calculate_ichimoku_cloud(
        self,
        data: pd.DataFrame,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_span_b_period: int = 52,
        displacement: int = 26,
    ) -> pd.DataFrame:
        """
        一目均衡表（Ichimoku Cloud）計算

        Args:
            data: OHLC価格データ
            tenkan_period: 転換線期間
            kijun_period: 基準線期間
            senkou_span_b_period: 先行スパンB期間
            displacement: 先行スパンの未来描画期間

        Returns:
            一目均衡表指標を含むDataFrame
        """
        try:
            df = data.copy()

            # 転換線 = (9期間の最高値 + 最安値) ÷ 2
            tenkan_high = df["High"].rolling(window=tenkan_period).max()
            tenkan_low = df["Low"].rolling(window=tenkan_period).min()
            df["tenkan_sen"] = (tenkan_high + tenkan_low) / 2

            # 基準線 = (26期間の最高値 + 最安値) ÷ 2
            kijun_high = df["High"].rolling(window=kijun_period).max()
            kijun_low = df["Low"].rolling(window=kijun_period).min()
            df["kijun_sen"] = (kijun_high + kijun_low) / 2

            # 先行スパンA = (転換線 + 基準線) ÷ 2を26期間未来にプロット
            df["senkou_span_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(
                displacement
            )

            # 先行スパンB = (52期間の最高値 + 最安値) ÷ 2を26期間未来にプロット
            senkou_b_high = df["High"].rolling(window=senkou_span_b_period).max()
            senkou_b_low = df["Low"].rolling(window=senkou_span_b_period).min()
            df["senkou_span_b"] = ((senkou_b_high + senkou_b_low) / 2).shift(
                displacement
            )

            # 遅行スパン = 終値を26期間過去にプロット
            df["chikou_span"] = df["Close"].shift(-displacement)

            # 雲の厚さ
            df["cloud_thickness"] = abs(df["senkou_span_a"] - df["senkou_span_b"])

            # 雲の色（上昇雲/下降雲）
            df["cloud_color"] = np.where(
                df["senkou_span_a"] > df["senkou_span_b"], "bullish", "bearish"
            )

            # 価格と雲の位置関係
            cloud_top = np.maximum(df["senkou_span_a"], df["senkou_span_b"])
            cloud_bottom = np.minimum(df["senkou_span_a"], df["senkou_span_b"])

            df["price_cloud_position"] = np.where(
                df["Close"] > cloud_top,
                "above_cloud",
                np.where(df["Close"] < cloud_bottom, "below_cloud", "in_cloud"),
            )

            # 一目均衡表の総合シグナル
            df["ichimoku_signal"] = self._calculate_ichimoku_signal(df)

            logger.info("一目均衡表計算完了")
            return df

        except Exception as e:
            logger.error(f"一目均衡表計算エラー: {e}")
            return data.copy()

    def _calculate_ichimoku_signal(self, df: pd.DataFrame) -> pd.Series:
        """一目均衡表の総合シグナル計算"""
        try:
            signals = pd.Series(index=df.index, dtype="object")

            for i in range(len(df)):
                score = 0
                signal_reasons = []

                # 転換線と基準線のクロス
                if i > 0:
                    if (
                        df["tenkan_sen"].iloc[i] > df["kijun_sen"].iloc[i]
                        and df["tenkan_sen"].iloc[i - 1] <= df["kijun_sen"].iloc[i - 1]
                    ):
                        score += 2
                        signal_reasons.append("転換線上抜け")
                    elif (
                        df["tenkan_sen"].iloc[i] < df["kijun_sen"].iloc[i]
                        and df["tenkan_sen"].iloc[i - 1] >= df["kijun_sen"].iloc[i - 1]
                    ):
                        score -= 2
                        signal_reasons.append("転換線下抜け")

                # 価格と雲の位置
                if df["price_cloud_position"].iloc[i] == "above_cloud":
                    score += 1
                elif df["price_cloud_position"].iloc[i] == "below_cloud":
                    score -= 1

                # 雲の色
                if df["cloud_color"].iloc[i] == "bullish":
                    score += 0.5
                elif df["cloud_color"].iloc[i] == "bearish":
                    score -= 0.5

                # 遅行スパンの位置
                if i >= 26:
                    chikou_vs_price = df["Close"].iloc[i - 26]
                    current_price = df["Close"].iloc[i]
                    if df["chikou_span"].iloc[i] > chikou_vs_price:
                        score += 1
                        signal_reasons.append("遅行スパン上抜け")
                    elif df["chikou_span"].iloc[i] < chikou_vs_price:
                        score -= 1
                        signal_reasons.append("遅行スパン下抜け")

                # シグナル分類
                if score >= 3:
                    signals.iloc[i] = "strong_buy"
                elif score >= 1:
                    signals.iloc[i] = "buy"
                elif score <= -3:
                    signals.iloc[i] = "strong_sell"
                elif score <= -1:
                    signals.iloc[i] = "sell"
                else:
                    signals.iloc[i] = "hold"

            return signals

        except Exception as e:
            logger.error(f"一目均衡表シグナル計算エラー: {e}")
            return pd.Series(["hold"] * len(df), index=df.index)

    def detect_fibonacci_retracements(
        self, data: pd.DataFrame, swing_period: int = 20, min_swing_percent: float = 5.0
    ) -> Dict[str, any]:
        """
        フィボナッチリトレースメント自動検出

        Args:
            data: OHLC価格データ
            swing_period: スイング検出期間
            min_swing_percent: 最小スイング率（%）

        Returns:
            フィボナッチレベル情報辞書
        """
        try:
            # 最近のスイングハイ・ローを検出
            highs = self._find_swing_highs(data, swing_period)
            lows = self._find_swing_lows(data, swing_period)

            fibonacci_levels = {}
            current_price = data["Close"].iloc[-1]

            # 最新の主要スイング検出
            recent_high = None
            recent_low = None

            # 最近30日以内のスイングを検索
            lookback_days = 30
            recent_data = data.tail(lookback_days)

            for date, price in highs.items():
                if date in recent_data.index:
                    swing_size = abs(price - current_price) / current_price * 100
                    if swing_size >= min_swing_percent:
                        if recent_high is None or price > recent_high[1]:
                            recent_high = (date, price)

            for date, price in lows.items():
                if date in recent_data.index:
                    swing_size = abs(price - current_price) / current_price * 100
                    if swing_size >= min_swing_percent:
                        if recent_low is None or price < recent_low[1]:
                            recent_low = (date, price)

            # フィボナッチリトレースメント計算
            if recent_high and recent_low:
                high_price = recent_high[1]
                low_price = recent_low[1]
                price_range = high_price - low_price

                # 上昇トレンド内のリトレースメント
                if recent_high[0] > recent_low[0]:  # ハイが後に発生
                    fibonacci_levels = {
                        "trend_direction": "uptrend",
                        "swing_high": high_price,
                        "swing_low": low_price,
                        "levels": {},
                    }

                    for level in self.fibonacci_levels:
                        fib_price = high_price - (price_range * level)
                        distance_to_current = abs(fib_price - current_price)
                        fibonacci_levels["levels"][f"fib_{level:.3f}"] = {
                            "price": fib_price,
                            "distance": distance_to_current,
                            "distance_pct": (distance_to_current / current_price) * 100,
                        }

                # 下降トレンド内のリトレースメント
                else:
                    fibonacci_levels = {
                        "trend_direction": "downtrend",
                        "swing_high": high_price,
                        "swing_low": low_price,
                        "levels": {},
                    }

                    for level in self.fibonacci_levels:
                        fib_price = low_price + (price_range * level)
                        distance_to_current = abs(fib_price - current_price)
                        fibonacci_levels["levels"][f"fib_{level:.3f}"] = {
                            "price": fib_price,
                            "distance": distance_to_current,
                            "distance_pct": (distance_to_current / current_price) * 100,
                        }

                # 現在価格に最も近いフィボナッチレベルを特定
                closest_level = min(
                    fibonacci_levels["levels"].items(), key=lambda x: x[1]["distance"]
                )
                fibonacci_levels["closest_level"] = closest_level

                logger.info(
                    f"フィボナッチリトレースメント検出完了: {fibonacci_levels['trend_direction']}"
                )

            else:
                fibonacci_levels = {
                    "trend_direction": "neutral",
                    "message": "明確なスイングが検出されませんでした",
                }

            return fibonacci_levels

        except Exception as e:
            logger.error(f"フィボナッチリトレースメント検出エラー: {e}")
            return {"trend_direction": "error", "message": str(e)}

    def _find_swing_highs(self, data: pd.DataFrame, period: int) -> Dict:
        """スイングハイ検出"""
        try:
            highs = {}
            high_prices = data["High"]

            for i in range(period, len(data) - period):
                current_high = high_prices.iloc[i]

                # 前後期間の最高値と比較
                left_max = high_prices.iloc[i - period : i].max()
                right_max = high_prices.iloc[i + 1 : i + period + 1].max()

                if current_high > left_max and current_high > right_max:
                    highs[data.index[i]] = current_high

            return highs

        except Exception as e:
            logger.error(f"スイングハイ検出エラー: {e}")
            return {}

    def _find_swing_lows(self, data: pd.DataFrame, period: int) -> Dict:
        """スイングロー検出"""
        try:
            lows = {}
            low_prices = data["Low"]

            for i in range(period, len(data) - period):
                current_low = low_prices.iloc[i]

                # 前後期間の最安値と比較
                left_min = low_prices.iloc[i - period : i].min()
                right_min = low_prices.iloc[i + 1 : i + period + 1].min()

                if current_low < left_min and current_low < right_min:
                    lows[data.index[i]] = current_low

            return lows

        except Exception as e:
            logger.error(f"スイングロー検出エラー: {e}")
            return {}

    def calculate_advanced_bollinger_bands(
        self,
        data: pd.DataFrame,
        periods: Optional[List[int]] = None,
        std_devs: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        高度ボリンジャーバンド分析

        Args:
            data: OHLC価格データ
            periods: 計算期間リスト
            std_devs: 標準偏差倍数リスト

        Returns:
            複数のボリンジャーバンドを含むDataFrame
        """
        try:
            df = data.copy()

            if periods is None:
                periods = [20, 50]
            if std_devs is None:
                std_devs = [1.0, 2.0, 3.0]

            for period in periods:
                sma = df["Close"].rolling(window=period).mean()
                std = df["Close"].rolling(window=period).std()

                for std_dev in std_devs:
                    upper_col = f"bb_upper_{period}_{std_dev}"
                    lower_col = f"bb_lower_{period}_{std_dev}"

                    df[upper_col] = sma + (std * std_dev)
                    df[lower_col] = sma - (std * std_dev)

                    # バンド位置
                    position_col = f"bb_position_{period}_{std_dev}"
                    df[position_col] = (df["Close"] - df[lower_col]) / (
                        df[upper_col] - df[lower_col]
                    )

                    # バンド幅（ボラティリティ指標）
                    width_col = f"bb_width_{period}_{std_dev}"
                    df[width_col] = (df[upper_col] - df[lower_col]) / sma

                # スクイーズ検出（20期間, 2σをベース）
                if period == 20 and 2.0 in std_devs:
                    current_width = df[f"bb_width_{period}_2.0"]
                    avg_width = current_width.rolling(20).mean()
                    df["bb_squeeze"] = current_width < avg_width * 0.8

                # ボリンジャーバンド戦略シグナル
                df[f"bb_signal_{period}"] = self._calculate_bb_signals(df, period)

            logger.info(
                f"高度ボリンジャーバンド計算完了: {len(periods)}期間 x {len(std_devs)}標準偏差"
            )
            return df

        except Exception as e:
            logger.error(f"高度ボリンジャーバンド計算エラー: {e}")
            return data.copy()

    def _calculate_bb_signals(self, df: pd.DataFrame, period: int) -> pd.Series:
        """ボリンジャーバンド戦略シグナル計算"""
        try:
            signals = pd.Series(index=df.index, dtype="object")

            upper_2 = f"bb_upper_{period}_2.0"
            lower_2 = f"bb_lower_{period}_2.0"
            position_2 = f"bb_position_{period}_2.0"
            width = f"bb_width_{period}_2.0"

            for i in range(1, len(df)):
                signal = "hold"

                # バンドタッチとリバーサル
                if df[position_2].iloc[i] <= 0.05:  # 下部バンド近く
                    if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
                        signal = "buy"
                elif df[position_2].iloc[i] >= 0.95:  # 上部バンド近く
                    if df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
                        signal = "sell"

                # スクイーズブレイクアウト
                if "bb_squeeze" in df.columns:
                    if df["bb_squeeze"].iloc[i - 1] and not df["bb_squeeze"].iloc[i]:
                        # ブレイクアウト方向を判定
                        if df["Close"].iloc[i] > df["Close"].iloc[i - 5 : i].mean():
                            signal = "breakout_buy"
                        else:
                            signal = "breakout_sell"

                signals.iloc[i] = signal

            return signals

        except Exception as e:
            logger.error(f"BBシグナル計算エラー: {e}")
            return pd.Series(["hold"] * len(df), index=df.index)

    def detect_elliott_wave_patterns(
        self, data: pd.DataFrame, min_wave_size: float = 2.0, max_lookback: int = 100
    ) -> Dict[str, any]:
        """
        エリオット波動パターン検出

        Args:
            data: OHLC価格データ
            min_wave_size: 最小波動サイズ（%）
            max_lookback: 最大遡り期間

        Returns:
            エリオット波動パターン情報
        """
        try:
            # 最近のデータのみを使用
            recent_data = data.tail(max_lookback)

            # ジグザグライン生成（ノイズ除去）
            zigzag = self._generate_zigzag(recent_data, min_wave_size)

            if len(zigzag) < 5:
                return {
                    "pattern": "insufficient_data",
                    "wave_count": len(zigzag),
                    "message": "波動検出に十分なデータがありません",
                }

            # 5波動パターン検出（推進波）
            impulse_pattern = self._detect_impulse_waves(zigzag)

            # 3波動パターン検出（修正波）
            corrective_pattern = self._detect_corrective_waves(zigzag)

            # 最も可能性の高いパターンを選択
            elliott_analysis = {
                "zigzag_points": len(zigzag),
                "impulse_pattern": impulse_pattern,
                "corrective_pattern": corrective_pattern,
                "current_price": float(data["Close"].iloc[-1]),
                "analysis_period": len(recent_data),
            }

            # パターン信頼度計算
            if impulse_pattern["confidence"] > corrective_pattern["confidence"]:
                elliott_analysis["primary_pattern"] = "impulse"
                elliott_analysis["pattern_confidence"] = impulse_pattern["confidence"]
                elliott_analysis["wave_count"] = 5
                elliott_analysis["next_expected"] = impulse_pattern["next_expected"]
            else:
                elliott_analysis["primary_pattern"] = "corrective"
                elliott_analysis["pattern_confidence"] = corrective_pattern[
                    "confidence"
                ]
                elliott_analysis["wave_count"] = 3
                elliott_analysis["next_expected"] = corrective_pattern["next_expected"]

            logger.info(
                f"エリオット波動分析完了: {elliott_analysis['primary_pattern']}パターン"
            )
            return elliott_analysis

        except Exception as e:
            logger.error(f"エリオット波動検出エラー: {e}")
            return {"pattern": "error", "message": str(e), "wave_count": 0}

    def _generate_zigzag(
        self, data: pd.DataFrame, threshold_pct: float
    ) -> List[Tuple[datetime, float, str]]:
        """
        ジグザグライン生成（高値・安値の識別）

        Returns:
            List[(日付, 価格, 高安タイプ)]
        """
        try:
            zigzag = []
            prices = data["Close"].values
            dates = data.index

            if len(prices) < 3:
                return zigzag

            # 初期トレンド方向を判定
            trend = 1 if prices[1] > prices[0] else -1
            extreme_idx = 0
            extreme_price = prices[0]

            for i in range(1, len(prices)):
                current_price = prices[i]
                change_pct = abs(current_price - extreme_price) / extreme_price * 100

                if trend == 1:  # 上昇トレンド
                    if current_price > extreme_price:
                        # 新高値更新
                        extreme_price = current_price
                        extreme_idx = i
                    elif change_pct >= threshold_pct:
                        # 十分な下落でトレンド転換
                        zigzag.append((dates[extreme_idx], extreme_price, "high"))
                        trend = -1
                        extreme_price = current_price
                        extreme_idx = i
                else:  # 下降トレンド
                    if current_price < extreme_price:
                        # 新安値更新
                        extreme_price = current_price
                        extreme_idx = i
                    elif change_pct >= threshold_pct:
                        # 十分な上昇でトレンド転換
                        zigzag.append((dates[extreme_idx], extreme_price, "low"))
                        trend = 1
                        extreme_price = current_price
                        extreme_idx = i

            # 最後のポイントを追加
            zigzag.append(
                (dates[extreme_idx], extreme_price, "high" if trend == 1 else "low")
            )

            return zigzag

        except Exception as e:
            logger.error(f"ジグザグ生成エラー: {e}")
            return []

    def _detect_impulse_waves(
        self, zigzag: List[Tuple[datetime, float, str]]
    ) -> Dict[str, any]:
        """5波推進波パターン検出"""
        try:
            if len(zigzag) < 5:
                return {"confidence": 0, "waves": [], "next_expected": "unknown"}

            # 最新5波を取得
            recent_waves = zigzag[-5:]
            confidence = 0
            wave_analysis = []

            # エリオット波動の基本ルールをチェック
            # Wave 1: 初期推進波
            # Wave 2: 修正波（Wave 1の50-80%戻し）
            # Wave 3: 最強推進波（Wave 1より大きい）
            # Wave 4: 修正波（Wave 1と重複しない）
            # Wave 5: 最終推進波

            prices = [wave[1] for wave in recent_waves]
            wave_types = [wave[2] for wave in recent_waves]

            # パターン判定
            if (
                len(set([i % 2 for i, wtype in enumerate(wave_types)])) == 2
            ):  # 高安が交互
                confidence += 30

            # Wave 3が最大であることを確認
            if len(prices) >= 4:
                wave_sizes = []
                for i in range(0, len(prices) - 1, 2):
                    if i + 1 < len(prices):
                        size = abs(prices[i + 1] - prices[i])
                        wave_sizes.append(size)

                if (
                    len(wave_sizes) >= 2 and max(wave_sizes) == wave_sizes[1]
                ):  # Wave 3が最大
                    confidence += 40

            # Wave 2の戻し比率チェック
            if len(prices) >= 3:
                wave1_size = abs(prices[1] - prices[0])
                wave2_size = abs(prices[2] - prices[1])
                retracement_ratio = wave2_size / wave1_size if wave1_size > 0 else 0

                if 0.3 <= retracement_ratio <= 0.8:  # 適切な戻し比率
                    confidence += 20

            # 次の予想
            next_expected = (
                "corrective_wave"
                if len(recent_waves) >= 5
                else f"wave_{len(recent_waves)+1}"
            )

            return {
                "confidence": confidence,
                "waves": recent_waves,
                "next_expected": next_expected,
                "wave_count": len(recent_waves),
            }

        except Exception as e:
            logger.error(f"推進波検出エラー: {e}")
            return {"confidence": 0, "waves": [], "next_expected": "unknown"}

    def _detect_corrective_waves(
        self, zigzag: List[Tuple[datetime, float, str]]
    ) -> Dict[str, any]:
        """3波修正波パターン検出"""
        try:
            if len(zigzag) < 3:
                return {"confidence": 0, "waves": [], "next_expected": "unknown"}

            # 最新3波を取得
            recent_waves = zigzag[-3:]
            confidence = 0

            prices = [wave[1] for wave in recent_waves]
            wave_types = [wave[2] for wave in recent_waves]

            # ABC修正波の基本パターン
            # A波: 初期修正
            # B波: 一時的反発（A波の38-78%戻し）
            # C波: 最終修正（A波と同程度かそれ以上）

            if len(prices) == 3:
                # A波とC波のサイズ比較
                a_wave_size = abs(prices[1] - prices[0])
                c_wave_size = abs(prices[2] - prices[1])

                if a_wave_size > 0:
                    size_ratio = c_wave_size / a_wave_size
                    if 0.8 <= size_ratio <= 1.5:  # C波がA波と類似サイズ
                        confidence += 50

                # B波の戻し比率
                if len(zigzag) >= 4:
                    prev_trend_size = abs(zigzag[-4][1] - prices[0])
                    b_wave_size = abs(prices[1] - prices[0])

                    if prev_trend_size > 0:
                        b_retracement = b_wave_size / prev_trend_size
                        if 0.3 <= b_retracement <= 0.8:
                            confidence += 30

            next_expected = (
                "impulse_wave"
                if len(recent_waves) >= 3
                else f"corrective_wave_{len(recent_waves)+1}"
            )

            return {
                "confidence": confidence,
                "waves": recent_waves,
                "next_expected": next_expected,
                "wave_count": len(recent_waves),
            }

        except Exception as e:
            logger.error(f"修正波検出エラー: {e}")
            return {"confidence": 0, "waves": [], "next_expected": "unknown"}

    def calculate_volatility_indicators(
        self, data: pd.DataFrame, periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        高度ボラティリティ指標計算

        Args:
            data: OHLC価格データ
            periods: 計算期間リスト

        Returns:
            各種ボラティリティ指標を含むDataFrame
        """
        try:
            df = data.copy()
            if periods is None:
                periods = [10, 20, 50]
            returns = df["Close"].pct_change()

            for period in periods:
                # 実現ボラティリティ（年率化）
                df[f"realized_vol_{period}"] = returns.rolling(period).std() * np.sqrt(
                    252
                )

                # Parkinson推定量（高値・安値ベース）
                hl_returns = np.log(df["High"] / df["Low"])
                df[f"parkinson_vol_{period}"] = hl_returns.rolling(period).apply(
                    lambda x: np.sqrt(np.sum(x**2) / (4 * np.log(2) * len(x)) * 252)
                )

                # Garman-Klass推定量
                gk_returns = 0.5 * (
                    np.log(df["High"] / df["Close"]) * np.log(df["High"] / df["Open"])
                ) + 0.5 * (
                    np.log(df["Low"] / df["Close"]) * np.log(df["Low"] / df["Open"])
                )
                df[f"garman_klass_vol_{period}"] = (
                    gk_returns.rolling(period).sum().apply(lambda x: np.sqrt(x * 252))
                )

                # ATR（Average True Range）
                tr1 = df["High"] - df["Low"]
                tr2 = abs(df["High"] - df["Close"].shift(1))
                tr3 = abs(df["Low"] - df["Close"].shift(1))
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df[f"atr_{period}"] = true_range.rolling(period).mean()

                # ボラティリティ傾向
                current_vol = df[f"realized_vol_{period}"]
                vol_trend = current_vol / current_vol.rolling(period).mean()
                df[f"vol_trend_{period}"] = vol_trend

                # ボラティリティレジーム
                vol_percentile = current_vol.rolling(period * 5).rank(pct=True)
                df[f"vol_regime_{period}"] = pd.cut(
                    vol_percentile,
                    bins=[0, 0.25, 0.75, 1.0],
                    labels=["low", "normal", "high"],
                )

            # VIX風指標（オプション価格がない場合の代替）
            df["vix_like"] = self._calculate_vix_like_indicator(df, returns)

            logger.info(f"高度ボラティリティ指標計算完了: {len(periods)}期間")
            return df

        except Exception as e:
            logger.error(f"ボラティリティ指標計算エラー: {e}")
            return data.copy()

    def _calculate_vix_like_indicator(
        self, df: pd.DataFrame, returns: pd.Series, window: int = 30
    ) -> pd.Series:
        """VIX様指標計算（GARCH風モデル）"""
        try:
            # 簡単なGARCH(1,1)風モデル
            alpha = 0.1
            beta = 0.85
            omega = 0.05

            vol_forecast = pd.Series(index=df.index, dtype=float)
            vol_forecast.iloc[0] = returns.std()

            for i in range(1, len(returns)):
                if pd.notna(returns.iloc[i - 1]):
                    vol_forecast.iloc[i] = np.sqrt(
                        omega
                        + alpha * (returns.iloc[i - 1] ** 2)
                        + beta * (vol_forecast.iloc[i - 1] ** 2)
                    )
                else:
                    vol_forecast.iloc[i] = vol_forecast.iloc[i - 1]

            # 年率化してVIX風スケール（0-100）に変換
            vix_like = vol_forecast * np.sqrt(252) * 100

            return vix_like

        except Exception as e:
            logger.error(f"VIX様指標計算エラー: {e}")
            return pd.Series([20] * len(df), index=df.index)  # デフォルト値

    def generate_comprehensive_signal(
        self, data: pd.DataFrame, weights: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        総合的なテクニカル分析シグナル生成

        Args:
            data: 全指標を含むDataFrame
            weights: 各指標の重み

        Returns:
            総合シグナルを含むDataFrame
        """
        try:
            df = data.copy()

            # デフォルト重み
            if weights is None:
                weights = {
                    "ichimoku": 0.3,
                    "bollinger": 0.25,
                    "fibonacci": 0.2,
                    "elliott": 0.15,
                    "volatility": 0.1,
                }

            # 各シグナルをスコア化
            signal_scores = pd.DataFrame(index=df.index)

            # 一目均衡表シグナル
            if "ichimoku_signal" in df.columns:
                ichimoku_map = {
                    "strong_buy": 2,
                    "buy": 1,
                    "hold": 0,
                    "sell": -1,
                    "strong_sell": -2,
                }
                signal_scores["ichimoku"] = (
                    df["ichimoku_signal"].map(ichimoku_map).fillna(0)
                )

            # ボリンジャーバンドシグナル（20期間をメイン使用）
            bb_columns = [col for col in df.columns if col.startswith("bb_signal_20")]
            if bb_columns:
                bb_map = {
                    "breakout_buy": 2,
                    "buy": 1,
                    "hold": 0,
                    "sell": -1,
                    "breakout_sell": -2,
                }
                signal_scores["bollinger"] = df[bb_columns[0]].map(bb_map).fillna(0)

            # フィボナッチ近接度（別途計算が必要）
            signal_scores["fibonacci"] = 0  # 簡略化

            # エリオット波動（別途計算が必要）
            signal_scores["elliott"] = 0  # 簡略化

            # ボラティリティレジーム
            vol_regime_cols = [col for col in df.columns if "vol_regime_" in col]
            if vol_regime_cols:
                vol_map = {"low": 1, "normal": 0, "high": -0.5}
                signal_scores["volatility"] = (
                    df[vol_regime_cols[0]].map(vol_map).fillna(0)
                )

            # 加重平均で総合スコア計算
            df["technical_score"] = 0
            for indicator, weight in weights.items():
                if indicator in signal_scores.columns:
                    df["technical_score"] += signal_scores[indicator] * weight

            # 総合シグナル判定
            df["comprehensive_signal"] = pd.cut(
                df["technical_score"],
                bins=[-float("inf"), -1.0, -0.3, 0.3, 1.0, float("inf")],
                labels=["strong_sell", "sell", "hold", "buy", "strong_buy"],
            )

            # 信頼度計算（指標の一致度）
            available_signals = signal_scores.dropna(axis=1)
            if len(available_signals.columns) > 1:
                # 標準偏差が小さいほど一致度が高い
                signal_std = available_signals.std(axis=1)
                max_std = np.sqrt(
                    len(available_signals.columns)
                )  # 理論上の最大標準偏差
                df["signal_confidence"] = (1 - signal_std / max_std) * 100
            else:
                df["signal_confidence"] = 50

            logger.info("総合テクニカル分析シグナル生成完了")
            return df

        except Exception as e:
            logger.error(f"総合シグナル生成エラー: {e}")
            data["comprehensive_signal"] = "hold"
            data["signal_confidence"] = 50
            return data


if __name__ == "__main__":
    # テスト実行
    print("=== 高度テクニカル指標分析システム テスト ===")

    # サンプルデータ生成
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    np.random.seed(42)

    # より現実的な株価データ生成
    base_price = 2500
    trend = 0.0002  # 微妙な上昇トレンド
    volatility = 0.02

    prices = [base_price]
    for i in range(1, len(dates)):
        # ランダムウォーク + トレンド + 季節性
        seasonal = 0.001 * np.sin(2 * np.pi * i / 252)  # 年次季節性
        random_change = np.random.normal(trend + seasonal, volatility)
        new_price = prices[-1] * (1 + random_change)
        prices.append(max(new_price, 100))  # 価格が100を下回らないよう制限

    # OHLCV生成
    sample_data = pd.DataFrame(index=dates)
    sample_data["Close"] = prices
    sample_data["Open"] = [p * np.random.uniform(0.99, 1.01) for p in prices]
    sample_data["High"] = [
        max(o, c) * np.random.uniform(1.00, 1.03)
        for o, c in zip(sample_data["Open"], sample_data["Close"])
    ]
    sample_data["Low"] = [
        min(o, c) * np.random.uniform(0.97, 1.00)
        for o, c in zip(sample_data["Open"], sample_data["Close"])
    ]
    sample_data["Volume"] = np.random.randint(1000000, 10000000, len(dates))

    try:
        indicator = AdvancedTechnicalIndicators()

        print(f"サンプルデータ: {len(sample_data)}日分")
        print(
            f"価格範囲: {sample_data['Close'].min():.2f} - {sample_data['Close'].max():.2f}"
        )

        # 1. 一目均衡表テスト
        print("\n1. 一目均衡表計算テスト")
        ichimoku_data = indicator.calculate_ichimoku_cloud(sample_data)
        print(
            f"✅ 一目均衡表計算完了: {len([c for c in ichimoku_data.columns if 'tenkan' in c or 'kijun' in c or 'senkou' in c])}指標"
        )
        print(f"   最新シグナル: {ichimoku_data['ichimoku_signal'].iloc[-1]}")
        print(f"   雲の位置: {ichimoku_data['price_cloud_position'].iloc[-1]}")

        # 2. フィボナッチリトレースメントテスト
        print("\n2. フィボナッチリトレースメント検出テスト")
        fibonacci_result = indicator.detect_fibonacci_retracements(sample_data)
        print(f"✅ フィボナッチ検出完了: {fibonacci_result['trend_direction']}")
        if "levels" in fibonacci_result:
            closest = fibonacci_result.get("closest_level")
            if closest:
                print(f"   最近接レベル: {closest[0]} = {closest[1]['price']:.2f}")

        # 3. 高度ボリンジャーバンドテスト
        print("\n3. 高度ボリンジャーバンド分析テスト")
        bb_data = indicator.calculate_advanced_bollinger_bands(sample_data)
        bb_columns = [c for c in bb_data.columns if c.startswith("bb_")]
        print(f"✅ ボリンジャーバンド計算完了: {len(bb_columns)}指標")
        if "bb_squeeze" in bb_data.columns:
            squeeze_count = bb_data["bb_squeeze"].sum()
            print(f"   スクイーズ検出: {squeeze_count}回")

        # 4. エリオット波動テスト
        print("\n4. エリオット波動パターン検出テスト")
        elliott_result = indicator.detect_elliott_wave_patterns(sample_data)
        print(
            f"✅ エリオット波動分析完了: {elliott_result.get('primary_pattern', 'unknown')}"
        )
        print(f"   パターン信頼度: {elliott_result.get('pattern_confidence', 0):.1f}%")
        print(f"   検出波動数: {elliott_result.get('zigzag_points', 0)}")

        # 5. ボラティリティ指標テスト
        print("\n5. 高度ボラティリティ指標テスト")
        vol_data = indicator.calculate_volatility_indicators(sample_data)
        vol_columns = [c for c in vol_data.columns if "vol" in c.lower() or "atr" in c]
        print(f"✅ ボラティリティ指標計算完了: {len(vol_columns)}指標")
        if "vix_like" in vol_data.columns:
            current_vix = vol_data["vix_like"].iloc[-1]
            print(f"   VIX風指標: {current_vix:.1f}")

        # 6. 総合シグナルテスト
        print("\n6. 総合テクニカル分析シグナルテスト")

        # 全指標を組み合わせ
        combined_data = sample_data.copy()
        for col in ichimoku_data.columns:
            if col not in combined_data.columns:
                combined_data[col] = ichimoku_data[col]
        for col in bb_data.columns:
            if col not in combined_data.columns:
                combined_data[col] = bb_data[col]
        for col in vol_data.columns:
            if col not in combined_data.columns:
                combined_data[col] = vol_data[col]

        comprehensive_data = indicator.generate_comprehensive_signal(combined_data)
        print("✅ 総合シグナル生成完了")
        print(f"   最新シグナル: {comprehensive_data['comprehensive_signal'].iloc[-1]}")
        print(
            f"   シグナル信頼度: {comprehensive_data['signal_confidence'].iloc[-1]:.1f}%"
        )
        print(f"   総合スコア: {comprehensive_data['technical_score'].iloc[-1]:.2f}")

        # 統計サマリー
        print("\n📊 分析統計:")
        print(f"   計算指標総数: {len(comprehensive_data.columns)}個")
        print(
            f"   一目均衡表関連: {len([c for c in comprehensive_data.columns if any(x in c for x in ['tenkan', 'kijun', 'senkou', 'chikou', 'ichimoku'])])}個"
        )
        print(
            f"   ボリンジャーバンド関連: {len([c for c in comprehensive_data.columns if c.startswith('bb_')])}個"
        )
        print(
            f"   ボラティリティ関連: {len([c for c in comprehensive_data.columns if 'vol' in c.lower() or 'atr' in c])}個"
        )

        # 最近のシグナル分布
        recent_signals = (
            comprehensive_data["comprehensive_signal"].tail(30).value_counts()
        )
        print(f"   最近30日のシグナル分布: {dict(recent_signals)}")

        print("\n✅ 高度テクニカル指標分析システム テスト完了！")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
