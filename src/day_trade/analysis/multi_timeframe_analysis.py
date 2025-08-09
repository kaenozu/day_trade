#!/usr/bin/env python3
"""
マルチタイムフレーム分析システム
Issue #315: 高度テクニカル指標・ML機能拡張

複数時間軸（日足・週足・月足）を統合した包括的トレンド分析
"""

import warnings
from datetime import datetime
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger
from .advanced_technical_indicators import AdvancedTechnicalIndicators

logger = get_context_logger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class MultiTimeframeAnalyzer:
    """
    マルチタイムフレーム分析クラス

    日足・週足・月足の複数時間軸でトレンド分析を実行し、
    時間軸間の整合性チェックと統合判定を提供
    """

    def __init__(self):
        """初期化"""
        self.timeframes = {
            "daily": {"period": "D", "name": "日足", "weight": 0.4},
            "weekly": {"period": "W", "name": "週足", "weight": 0.35},
            "monthly": {"period": "M", "name": "月足", "weight": 0.25},
        }

        self.advanced_indicators = AdvancedTechnicalIndicators()
        self.analysis_cache = {}

        logger.info("マルチタイムフレーム分析システム初期化完了")

    def resample_to_timeframe(
        self, data: pd.DataFrame, timeframe: str, method: str = "last"
    ) -> pd.DataFrame:
        """
        データを指定時間軸にリサンプリング

        Args:
            data: 元の価格データ（日足想定）
            timeframe: 'daily', 'weekly', 'monthly'
            method: リサンプリング方法

        Returns:
            リサンプリングされたDataFrame
        """
        try:
            if timeframe not in self.timeframes:
                logger.error(f"サポートされていない時間軸: {timeframe}")
                return data.copy()

            if timeframe == "daily":
                return data.copy()  # 日足はそのまま

            period = self.timeframes[timeframe]["period"]

            # OHLCV形式でリサンプリング
            resampled = pd.DataFrame()

            # 各列の適切な集約方法を定義
            agg_methods = {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }

            for col, agg_method in agg_methods.items():
                if col in data.columns:
                    if agg_method == "first":
                        resampled[col] = data[col].resample(period).first()
                    elif agg_method == "max":
                        resampled[col] = data[col].resample(period).max()
                    elif agg_method == "min":
                        resampled[col] = data[col].resample(period).min()
                    elif agg_method == "last":
                        resampled[col] = data[col].resample(period).last()
                    elif agg_method == "sum":
                        resampled[col] = data[col].resample(period).sum()

            # NaN値を削除
            resampled = resampled.dropna()

            logger.info(
                f"{timeframe}リサンプリング完了: {len(data)} → {len(resampled)}期間"
            )
            return resampled

        except Exception as e:
            logger.error(f"リサンプリングエラー ({timeframe}): {e}")
            return data.copy()

    def calculate_timeframe_indicators(
        self, data: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        """
        指定時間軸でテクニカル指標を計算

        Args:
            data: 価格データ
            timeframe: 時間軸

        Returns:
            テクニカル指標を含むDataFrame
        """
        try:
            # リサンプリング
            tf_data = self.resample_to_timeframe(data, timeframe)

            if tf_data.empty:
                logger.warning(f"リサンプリング後データが空: {timeframe}")
                return pd.DataFrame()

            # 基本テクニカル指標
            df = tf_data.copy()

            # 移動平均（期間を時間軸に応じて調整）
            periods = self._get_periods_for_timeframe(timeframe)

            for period in periods["sma"]:
                if len(df) > period:
                    df[f"sma_{period}"] = df["Close"].rolling(period).mean()

            for period in periods["ema"]:
                if len(df) > period:
                    df[f"ema_{period}"] = df["Close"].ewm(span=period).mean()

            # RSI
            if len(df) > periods["rsi"]:
                delta = df["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(periods["rsi"]).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(periods["rsi"]).mean()
                rs = gain / loss
                df["rsi"] = 100 - (100 / (1 + rs))

            # MACD
            if len(df) > max(periods["macd"]["fast"], periods["macd"]["slow"]):
                ema_fast = df["Close"].ewm(span=periods["macd"]["fast"]).mean()
                ema_slow = df["Close"].ewm(span=periods["macd"]["slow"]).mean()
                df["macd"] = ema_fast - ema_slow
                df["macd_signal"] = (
                    df["macd"].ewm(span=periods["macd"]["signal"]).mean()
                )
                df["macd_histogram"] = df["macd"] - df["macd_signal"]

            # ボリンジャーバンド
            if len(df) > periods["bb"]:
                sma = df["Close"].rolling(periods["bb"]).mean()
                std = df["Close"].rolling(periods["bb"]).std()
                df["bb_upper"] = sma + (std * 2)
                df["bb_lower"] = sma - (std * 2)
                df["bb_position"] = (df["Close"] - df["bb_lower"]) / (
                    df["bb_upper"] - df["bb_lower"]
                )

            # 一目均衡表（期間調整）
            ichimoku_periods = periods["ichimoku"]
            if len(df) > max(ichimoku_periods.values()):
                df = self.advanced_indicators.calculate_ichimoku_cloud(
                    df,
                    tenkan_period=ichimoku_periods["tenkan"],
                    kijun_period=ichimoku_periods["kijun"],
                    senkou_span_b_period=ichimoku_periods["senkou_b"],
                )

            # トレンド判定
            df["trend_direction"] = self._calculate_trend_direction(df, timeframe)
            df["trend_strength"] = self._calculate_trend_strength(df, timeframe)

            # サポート・レジスタンスレベル
            df = self._identify_support_resistance_levels(df, timeframe)

            logger.info(f"{timeframe}指標計算完了: {len(df.columns)}指標")
            return df

        except Exception as e:
            logger.error(f"{timeframe}指標計算エラー: {e}")
            return pd.DataFrame()

    def _get_periods_for_timeframe(self, timeframe: str) -> Dict[str, Union[int, Dict]]:
        """時間軸に応じた指標期間を取得"""
        base_periods = {
            "daily": {
                "sma": [5, 20, 50, 200],
                "ema": [12, 26],
                "rsi": 14,
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bb": 20,
                "ichimoku": {"tenkan": 9, "kijun": 26, "senkou_b": 52},
            },
            "weekly": {
                "sma": [4, 13, 26, 52],  # 約1, 3, 6ヶ月, 1年
                "ema": [8, 17],
                "rsi": 9,
                "macd": {"fast": 8, "slow": 17, "signal": 6},
                "bb": 13,
                "ichimoku": {"tenkan": 6, "kijun": 17, "senkou_b": 34},
            },
            "monthly": {
                "sma": [3, 6, 12, 24],  # 3ヶ月, 6ヶ月, 1年, 2年
                "ema": [5, 10],
                "rsi": 6,
                "macd": {"fast": 5, "slow": 10, "signal": 4},
                "bb": 6,
                "ichimoku": {"tenkan": 3, "kijun": 8, "senkou_b": 16},
            },
        }

        return base_periods.get(timeframe, base_periods["daily"])

    def _calculate_trend_direction(self, df: pd.DataFrame, timeframe: str) -> pd.Series:
        """トレンド方向計算"""
        try:
            trend_signals = pd.Series(index=df.index, dtype="object")

            for i in range(len(df)):
                score = 0

                # 移動平均による判定
                if "sma_20" in df.columns or "sma_13" in df.columns:
                    sma_col = "sma_20" if "sma_20" in df.columns else "sma_13"
                    if pd.notna(df[sma_col].iloc[i]):
                        if df["Close"].iloc[i] > df[sma_col].iloc[i]:
                            score += 1
                        else:
                            score -= 1

                # MACD判定
                if "macd" in df.columns and "macd_signal" in df.columns:
                    if pd.notna(df["macd"].iloc[i]) and pd.notna(
                        df["macd_signal"].iloc[i]
                    ):
                        if df["macd"].iloc[i] > df["macd_signal"].iloc[i]:
                            score += 1
                        else:
                            score -= 1

                # 一目均衡表判定
                if "ichimoku_signal" in df.columns:
                    signal = df["ichimoku_signal"].iloc[i]
                    if signal in ["buy", "strong_buy"]:
                        score += 1
                    elif signal in ["sell", "strong_sell"]:
                        score -= 1

                # トレンド分類
                if score >= 2:
                    trend_signals.iloc[i] = "strong_uptrend"
                elif score == 1:
                    trend_signals.iloc[i] = "uptrend"
                elif score == -1:
                    trend_signals.iloc[i] = "downtrend"
                elif score <= -2:
                    trend_signals.iloc[i] = "strong_downtrend"
                else:
                    trend_signals.iloc[i] = "sideways"

            return trend_signals

        except Exception as e:
            logger.error(f"トレンド方向計算エラー: {e}")
            return pd.Series(["sideways"] * len(df), index=df.index)

    def _calculate_trend_strength(self, df: pd.DataFrame, timeframe: str) -> pd.Series:
        """トレンド強度計算（0-100）"""
        try:
            strength_scores = pd.Series(index=df.index, dtype=float)

            for i in range(20, len(df)):  # 最低20期間のデータが必要
                strength = 50  # ベースライン

                # 価格モメンタム
                if i >= 10:
                    price_change = (
                        df["Close"].iloc[i] - df["Close"].iloc[i - 10]
                    ) / df["Close"].iloc[i - 10]
                    strength += price_change * 500  # スケール調整

                # RSI強度
                if "rsi" in df.columns and pd.notna(df["rsi"].iloc[i]):
                    rsi = df["rsi"].iloc[i]
                    if rsi > 70 or rsi < 30:
                        strength += 20  # 極端なRSIは強いトレンド

                # MACD histogram
                if "macd_histogram" in df.columns and pd.notna(
                    df["macd_histogram"].iloc[i]
                ):
                    macd_hist = df["macd_histogram"].iloc[i]
                    strength += abs(macd_hist) * 1000  # MACD histogramの絶対値

                # ボラティリティ考慮
                if i >= 20:
                    volatility = df["Close"].iloc[i - 20 : i].pct_change().std()
                    if volatility > 0:
                        strength += min(20, volatility * 500)  # 高ボラは強いトレンド

                # 一目均衡表雲の厚さ
                if "cloud_thickness" in df.columns and pd.notna(
                    df["cloud_thickness"].iloc[i]
                ):
                    cloud_thickness = df["cloud_thickness"].iloc[i]
                    current_price = df["Close"].iloc[i]
                    if current_price > 0:
                        thickness_ratio = cloud_thickness / current_price
                        strength += thickness_ratio * 200

                # 0-100に正規化
                strength_scores.iloc[i] = max(0, min(100, strength))

            # 初期値を50で埋める
            strength_scores.fillna(50, inplace=True)

            return strength_scores

        except Exception as e:
            logger.error(f"トレンド強度計算エラー: {e}")
            return pd.Series([50] * len(df), index=df.index)

    def _identify_support_resistance_levels(
        self, df: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        """サポート・レジスタンスレベル特定"""
        try:
            # 時間軸に応じた検出期間
            lookback_periods = {"daily": 50, "weekly": 26, "monthly": 12}
            lookback = lookback_periods.get(timeframe, 50)

            if len(df) < lookback:
                df["support_level"] = np.nan
                df["resistance_level"] = np.nan
                return df

            support_levels = []
            resistance_levels = []

            for i in range(lookback, len(df)):
                # 指定期間内の価格データ
                window_data = df.iloc[i - lookback : i]

                # サポートレベル（最安値付近の価格帯）
                low_prices = window_data["Low"]
                min_price = low_prices.min()

                # 最安値の±2%以内の価格を候補とする
                support_candidates = low_prices[low_prices <= min_price * 1.02]
                support_level = support_candidates.median()
                support_levels.append(support_level)

                # レジスタンスレベル（最高値付近の価格帯）
                high_prices = window_data["High"]
                max_price = high_prices.max()

                # 最高値の±2%以内の価格を候補とする
                resistance_candidates = high_prices[high_prices >= max_price * 0.98]
                resistance_level = resistance_candidates.median()
                resistance_levels.append(resistance_level)

            # データフレームに追加
            df["support_level"] = np.nan
            df["resistance_level"] = np.nan

            df.iloc[lookback:, df.columns.get_loc("support_level")] = support_levels
            df.iloc[
                lookback:, df.columns.get_loc("resistance_level")
            ] = resistance_levels

            # サポート・レジスタンス突破の検出
            df["support_break"] = (df["Close"] < df["support_level"]) & (
                df["Close"].shift(1) >= df["support_level"].shift(1)
            )
            df["resistance_break"] = (df["Close"] > df["resistance_level"]) & (
                df["Close"].shift(1) <= df["resistance_level"].shift(1)
            )

            return df

        except Exception as e:
            logger.error(f"サポート・レジスタンス計算エラー: {e}")
            df["support_level"] = np.nan
            df["resistance_level"] = np.nan
            return df

    def analyze_multiple_timeframes(
        self, data: pd.DataFrame, symbol: str = "UNKNOWN"
    ) -> Dict[str, any]:
        """
        複数時間軸統合分析

        Args:
            data: 日足価格データ
            symbol: 銘柄コード

        Returns:
            統合分析結果辞書
        """
        try:
            analysis_results = {
                "symbol": symbol,
                "analysis_timestamp": datetime.now().isoformat(),
                "timeframes": {},
                "integrated_analysis": {},
            }

            # 各時間軸で分析実行
            timeframe_data = {}

            for tf_key, tf_info in self.timeframes.items():
                logger.info(f"{tf_info['name']}分析開始: {symbol}")

                # 指標計算
                tf_indicators = self.calculate_timeframe_indicators(data, tf_key)

                if tf_indicators.empty:
                    logger.warning(f"{tf_key}分析スキップ: データ不足")
                    continue

                timeframe_data[tf_key] = tf_indicators

                # 最新の分析結果を抽出
                latest_data = tf_indicators.iloc[-1] if len(tf_indicators) > 0 else None

                if latest_data is not None:
                    tf_analysis = {
                        "timeframe": tf_info["name"],
                        "data_points": len(tf_indicators),
                        "current_price": float(latest_data["Close"]),
                        "trend_direction": latest_data.get(
                            "trend_direction", "unknown"
                        ),
                        "trend_strength": float(latest_data.get("trend_strength", 50)),
                        "technical_indicators": {},
                    }

                    # 主要テクニカル指標
                    if "rsi" in latest_data and pd.notna(latest_data["rsi"]):
                        tf_analysis["technical_indicators"]["rsi"] = float(
                            latest_data["rsi"]
                        )

                    if "macd" in latest_data and pd.notna(latest_data["macd"]):
                        tf_analysis["technical_indicators"]["macd"] = float(
                            latest_data["macd"]
                        )

                    if "bb_position" in latest_data and pd.notna(
                        latest_data["bb_position"]
                    ):
                        tf_analysis["technical_indicators"]["bb_position"] = float(
                            latest_data["bb_position"]
                        )

                    # サポート・レジスタンス
                    if "support_level" in latest_data and pd.notna(
                        latest_data["support_level"]
                    ):
                        tf_analysis["support_level"] = float(
                            latest_data["support_level"]
                        )

                    if "resistance_level" in latest_data and pd.notna(
                        latest_data["resistance_level"]
                    ):
                        tf_analysis["resistance_level"] = float(
                            latest_data["resistance_level"]
                        )

                    # 一目均衡表シグナル
                    if "ichimoku_signal" in latest_data:
                        tf_analysis["ichimoku_signal"] = str(
                            latest_data["ichimoku_signal"]
                        )

                    analysis_results["timeframes"][tf_key] = tf_analysis

            # 統合分析実行
            if len(analysis_results["timeframes"]) >= 2:
                integrated = self._perform_integrated_analysis(
                    analysis_results["timeframes"]
                )
                analysis_results["integrated_analysis"] = integrated

                logger.info(f"マルチタイムフレーム分析完了: {symbol}")
            else:
                logger.warning(
                    f"統合分析スキップ: 分析可能な時間軸が不足 ({len(analysis_results['timeframes'])})"
                )
                analysis_results["integrated_analysis"] = {
                    "overall_trend": "insufficient_data",
                    "confidence": 0,
                    "message": "統合分析に十分な時間軸データがありません",
                }

            return analysis_results

        except Exception as e:
            logger.error(f"マルチタイムフレーム分析エラー ({symbol}): {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
            }

    def _perform_integrated_analysis(
        self, timeframe_results: Dict[str, Dict]
    ) -> Dict[str, any]:
        """統合分析実行"""
        try:
            # トレンド方向の統合判定
            trend_votes = {}
            trend_weights = {}

            for tf_key, tf_data in timeframe_results.items():
                trend = tf_data.get("trend_direction", "sideways")
                strength = tf_data.get("trend_strength", 50)
                weight = self.timeframes[tf_key]["weight"]

                # トレンド投票
                if trend not in trend_votes:
                    trend_votes[trend] = 0
                    trend_weights[trend] = 0

                trend_votes[trend] += weight
                trend_weights[trend] += weight * (strength / 100)

            # 最有力トレンドの特定
            if trend_votes:
                dominant_trend = max(trend_votes.keys(), key=lambda x: trend_votes[x])
                trend_confidence = trend_votes[dominant_trend] * 100

                # 強度重み付け
                weighted_confidence = trend_weights.get(dominant_trend, 0) * 100
            else:
                dominant_trend = "sideways"
                trend_confidence = 0
                weighted_confidence = 0

            # 時間軸間の整合性チェック
            consistency_score = self._calculate_timeframe_consistency(timeframe_results)

            # 統合シグナル生成
            integrated_signal = self._generate_integrated_signal(
                dominant_trend,
                weighted_confidence,
                consistency_score,
                timeframe_results,
            )

            # リスク評価
            risk_assessment = self._assess_multi_timeframe_risk(timeframe_results)

            # 投資推奨
            investment_recommendation = self._generate_investment_recommendation(
                integrated_signal, risk_assessment, timeframe_results
            )

            return {
                "overall_trend": dominant_trend,
                "trend_confidence": float(weighted_confidence),
                "consistency_score": float(consistency_score),
                "integrated_signal": integrated_signal,
                "risk_assessment": risk_assessment,
                "investment_recommendation": investment_recommendation,
                "timeframe_agreement": self._analyze_timeframe_agreement(
                    timeframe_results
                ),
            }

        except Exception as e:
            logger.error(f"統合分析エラー: {e}")
            return {"overall_trend": "error", "error": str(e)}

    def _calculate_timeframe_consistency(
        self, timeframe_results: Dict[str, Dict]
    ) -> float:
        """時間軸間の整合性スコア計算（0-100）"""
        try:
            if len(timeframe_results) < 2:
                return 0

            # トレンド方向の一致度
            trends = [tf["trend_direction"] for tf in timeframe_results.values()]
            unique_trends = set(trends)

            if len(unique_trends) == 1:
                trend_consistency = 100
            elif len(unique_trends) == 2:
                trend_consistency = 50
            else:
                trend_consistency = 0

            # テクニカル指標の一致度
            indicator_consistency = 0
            indicator_count = 0

            # RSIの一致度（全て過買われ、過売られ、中立で一致するか）
            rsi_values = []
            for tf_data in timeframe_results.values():
                rsi = tf_data.get("technical_indicators", {}).get("rsi")
                if rsi is not None:
                    if rsi > 70:
                        rsi_values.append("overbought")
                    elif rsi < 30:
                        rsi_values.append("oversold")
                    else:
                        rsi_values.append("neutral")

            if len(rsi_values) >= 2:
                if len(set(rsi_values)) == 1:
                    indicator_consistency += 30
                elif len(set(rsi_values)) == 2:
                    indicator_consistency += 10
                indicator_count += 1

            # MACDシグナルの一致度
            macd_signals = []
            for tf_data in timeframe_results.values():
                macd = tf_data.get("technical_indicators", {}).get("macd")
                if macd is not None:
                    macd_signals.append("positive" if macd > 0 else "negative")

            if len(macd_signals) >= 2:
                if len(set(macd_signals)) == 1:
                    indicator_consistency += 20
                indicator_count += 1

            # 一目均衡表の一致度
            ichimoku_signals = []
            for tf_data in timeframe_results.values():
                ichimoku = tf_data.get("ichimoku_signal")
                if ichimoku:
                    if ichimoku in ["buy", "strong_buy"]:
                        ichimoku_signals.append("bullish")
                    elif ichimoku in ["sell", "strong_sell"]:
                        ichimoku_signals.append("bearish")
                    else:
                        ichimoku_signals.append("neutral")

            if len(ichimoku_signals) >= 2:
                if len(set(ichimoku_signals)) == 1:
                    indicator_consistency += 30
                elif len(set(ichimoku_signals)) == 2:
                    indicator_consistency += 10
                indicator_count += 1

            # 平均化
            if indicator_count > 0:
                indicator_consistency = indicator_consistency / indicator_count

            # 全体の整合性スコア
            overall_consistency = trend_consistency * 0.6 + indicator_consistency * 0.4

            return max(0, min(100, overall_consistency))

        except Exception as e:
            logger.error(f"整合性計算エラー: {e}")
            return 0

    def _generate_integrated_signal(
        self,
        dominant_trend: str,
        confidence: float,
        consistency: float,
        timeframe_results: Dict[str, Dict],
    ) -> Dict[str, any]:
        """統合シグナル生成"""
        try:
            # 基本シグナル判定
            if (
                dominant_trend in ["strong_uptrend", "uptrend"]
                and confidence >= 60
                and consistency >= 70
            ):
                signal_action = "BUY"
                signal_strength = "STRONG"
            elif dominant_trend in ["strong_uptrend", "uptrend"] and confidence >= 40:
                signal_action = "BUY"
                signal_strength = "MODERATE"
            elif (
                dominant_trend in ["strong_downtrend", "downtrend"]
                and confidence >= 60
                and consistency >= 70
            ):
                signal_action = "SELL"
                signal_strength = "STRONG"
            elif (
                dominant_trend in ["strong_downtrend", "downtrend"] and confidence >= 40
            ):
                signal_action = "SELL"
                signal_strength = "MODERATE"
            else:
                signal_action = "HOLD"
                signal_strength = "WEAK"

            # 調整要因チェック
            adjustment_factors = []

            # 短期と長期の不一致チェック
            if "daily" in timeframe_results and "monthly" in timeframe_results:
                daily_trend = timeframe_results["daily"]["trend_direction"]
                monthly_trend = timeframe_results["monthly"]["trend_direction"]

                if daily_trend != monthly_trend:
                    if daily_trend in [
                        "strong_downtrend",
                        "downtrend",
                    ] and monthly_trend in ["uptrend", "strong_uptrend"]:
                        adjustment_factors.append("短期下落・長期上昇の調整局面")
                        if signal_action == "SELL":
                            signal_strength = "WEAK"
                    elif daily_trend in [
                        "uptrend",
                        "strong_uptrend",
                    ] and monthly_trend in ["downtrend", "strong_downtrend"]:
                        adjustment_factors.append("短期上昇・長期下落の調整局面")
                        if signal_action == "BUY":
                            signal_strength = "WEAK"

            # オーバーボート・オーバーソールドチェック
            extreme_rsi_count = 0
            for tf_data in timeframe_results.values():
                rsi = tf_data.get("technical_indicators", {}).get("rsi")
                if rsi is not None:
                    if rsi > 80:
                        extreme_rsi_count += 1
                        adjustment_factors.append("RSI過買われ水準")
                    elif rsi < 20:
                        extreme_rsi_count += 1
                        adjustment_factors.append("RSI過売られ水準")

            if extreme_rsi_count >= 2:  # 複数時間軸で極端
                if signal_action in ["BUY", "SELL"]:
                    signal_strength = (
                        "MODERATE" if signal_strength == "STRONG" else "WEAK"
                    )

            return {
                "action": signal_action,
                "strength": signal_strength,
                "confidence": float(confidence),
                "consistency": float(consistency),
                "dominant_trend": dominant_trend,
                "adjustment_factors": adjustment_factors,
                "signal_score": self._calculate_signal_score(
                    signal_action, signal_strength, confidence, consistency
                ),
            }

        except Exception as e:
            logger.error(f"統合シグナル生成エラー: {e}")
            return {
                "action": "HOLD",
                "strength": "WEAK",
                "confidence": 0,
                "consistency": 0,
                "error": str(e),
            }

    def _calculate_signal_score(
        self, action: str, strength: str, confidence: float, consistency: float
    ) -> float:
        """シグナルスコア計算（-100 to +100）"""
        try:
            base_score = 0

            # アクションベーススコア
            if action == "BUY":
                base_score = 50
            elif action == "SELL":
                base_score = -50
            else:  # HOLD
                base_score = 0

            # 強度による調整
            strength_multiplier = {"STRONG": 1.0, "MODERATE": 0.7, "WEAK": 0.4}.get(
                strength, 0.4
            )

            base_score *= strength_multiplier

            # 信頼度による調整
            confidence_adjustment = (confidence / 100) * 0.3
            consistency_adjustment = (consistency / 100) * 0.2

            final_score = base_score * (
                1 + confidence_adjustment + consistency_adjustment
            )

            return max(-100, min(100, final_score))

        except Exception as e:
            logger.error(f"シグナルスコア計算エラー: {e}")
            return 0

    def _assess_multi_timeframe_risk(
        self, timeframe_results: Dict[str, Dict]
    ) -> Dict[str, any]:
        """マルチタイムフレームリスク評価"""
        try:
            risk_factors = []
            risk_score = 0  # 0-100, 高いほど危険

            # ボラティリティリスク
            high_vol_count = 0
            for tf_key, tf_data in timeframe_results.items():
                strength = tf_data.get("trend_strength", 50)
                if strength > 80:  # 高強度トレンド = 高ボラティリティ
                    high_vol_count += 1
                    risk_factors.append(f"{tf_data['timeframe']}高ボラティリティ")

            risk_score += high_vol_count * 15

            # トレンド不整合リスク
            trends = [tf["trend_direction"] for tf in timeframe_results.values()]
            unique_trends = len(set(trends))
            if unique_trends >= 3:
                risk_score += 30
                risk_factors.append("時間軸間トレンド不整合")
            elif unique_trends == 2:
                risk_score += 15
                risk_factors.append("一部時間軸トレンド相違")

            # 極端なテクニカル指標リスク
            extreme_indicators = 0
            for tf_data in timeframe_results.values():
                indicators = tf_data.get("technical_indicators", {})

                # 極端なRSI
                rsi = indicators.get("rsi")
                if rsi is not None and (rsi > 85 or rsi < 15):
                    extreme_indicators += 1
                    risk_factors.append(f"極端なRSI({rsi:.1f})")

                # ボリンジャーバンド極端位置
                bb_pos = indicators.get("bb_position")
                if bb_pos is not None and (bb_pos > 0.95 or bb_pos < 0.05):
                    extreme_indicators += 1
                    risk_factors.append("ボリンジャーバンド極端位置")

            risk_score += extreme_indicators * 10

            # サポート・レジスタンス近接リスク
            sr_risk = 0
            for tf_data in timeframe_results.values():
                current_price = tf_data.get("current_price", 0)
                support = tf_data.get("support_level")
                resistance = tf_data.get("resistance_level")

                if support and current_price > 0:
                    support_distance = abs(current_price - support) / current_price
                    if support_distance < 0.02:  # 2%以内
                        sr_risk += 1
                        risk_factors.append(f"{tf_data['timeframe']}サポート近接")

                if resistance and current_price > 0:
                    resistance_distance = (
                        abs(current_price - resistance) / current_price
                    )
                    if resistance_distance < 0.02:  # 2%以内
                        sr_risk += 1
                        risk_factors.append(f"{tf_data['timeframe']}レジスタンス近接")

            risk_score += sr_risk * 8

            # リスクレベル分類
            if risk_score >= 70:
                risk_level = "HIGH"
            elif risk_score >= 40:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            return {
                "risk_level": risk_level,
                "risk_score": max(0, min(100, risk_score)),
                "risk_factors": risk_factors,
                "total_risk_factors": len(risk_factors),
            }

        except Exception as e:
            logger.error(f"リスク評価エラー: {e}")
            return {"risk_level": "UNKNOWN", "risk_score": 50, "error": str(e)}

    def _generate_investment_recommendation(
        self,
        integrated_signal: Dict,
        risk_assessment: Dict,
        timeframe_results: Dict[str, Dict],
    ) -> Dict[str, any]:
        """投資推奨生成"""
        try:
            action = integrated_signal.get("action", "HOLD")
            strength = integrated_signal.get("strength", "WEAK")
            risk_level = risk_assessment.get("risk_level", "MEDIUM")

            # 基本推奨
            if action == "BUY" and strength == "STRONG" and risk_level == "LOW":
                recommendation = "STRONG_BUY"
                position_size = "FULL"
            elif action == "BUY" and (strength == "STRONG" or risk_level == "LOW"):
                recommendation = "BUY"
                position_size = "LARGE"
            elif action == "BUY" and strength == "MODERATE":
                recommendation = "WEAK_BUY"
                position_size = "SMALL"
            elif action == "SELL" and strength == "STRONG" and risk_level == "LOW":
                recommendation = "STRONG_SELL"
                position_size = "FULL"
            elif action == "SELL" and (strength == "STRONG" or risk_level == "LOW"):
                recommendation = "SELL"
                position_size = "LARGE"
            elif action == "SELL" and strength == "MODERATE":
                recommendation = "WEAK_SELL"
                position_size = "SMALL"
            else:
                recommendation = "HOLD"
                position_size = "NEUTRAL"

            # リスク調整
            if risk_level == "HIGH":
                if position_size in ["FULL", "LARGE"]:
                    position_size = "SMALL"
                elif position_size == "SMALL":
                    position_size = "MINIMAL"
                    recommendation = (
                        f"CAUTIOUS_{action}" if action != "HOLD" else "HOLD"
                    )

            # 推奨理由生成
            reasons = []

            # トレンド理由
            dominant_trend = integrated_signal.get("dominant_trend", "sideways")
            if dominant_trend != "sideways":
                reasons.append(f"複数時間軸で{dominant_trend}を確認")

            # 整合性理由
            consistency = integrated_signal.get("consistency", 0)
            if consistency >= 70:
                reasons.append("時間軸間の高い整合性")
            elif consistency < 40:
                reasons.append("時間軸間の整合性に懸念")

            # リスク理由
            risk_factors = risk_assessment.get("risk_factors", [])
            if len(risk_factors) == 0:
                reasons.append("明確なリスク要因なし")
            elif len(risk_factors) >= 3:
                reasons.append("複数のリスク要因を確認")

            # 価格位置理由
            support_breaks = 0
            resistance_breaks = 0
            for tf_data in timeframe_results.values():
                current_price = tf_data.get("current_price", 0)
                support = tf_data.get("support_level")
                resistance = tf_data.get("resistance_level")

                if support and current_price < support:
                    support_breaks += 1
                if resistance and current_price > resistance:
                    resistance_breaks += 1

            if resistance_breaks >= 2:
                reasons.append("複数時間軸でレジスタンス突破")
            elif support_breaks >= 2:
                reasons.append("複数時間軸でサポート割れ")

            return {
                "recommendation": recommendation,
                "position_size": position_size,
                "confidence": integrated_signal.get("confidence", 0),
                "reasons": reasons,
                "holding_period": self._suggest_holding_period(
                    timeframe_results, dominant_trend
                ),
                "stop_loss_suggestion": self._calculate_stop_loss(
                    timeframe_results, action
                ),
                "take_profit_suggestion": self._calculate_take_profit(
                    timeframe_results, action
                ),
            }

        except Exception as e:
            logger.error(f"投資推奨生成エラー: {e}")
            return {
                "recommendation": "HOLD",
                "position_size": "NEUTRAL",
                "error": str(e),
            }

    def _suggest_holding_period(
        self, timeframe_results: Dict[str, Dict], dominant_trend: str
    ) -> str:
        """保有期間推奨"""
        try:
            if dominant_trend in ["strong_uptrend", "strong_downtrend"]:
                if "monthly" in timeframe_results:
                    return "LONG_TERM"  # 3-6ヶ月
                elif "weekly" in timeframe_results:
                    return "MEDIUM_TERM"  # 1-3ヶ月
                else:
                    return "SHORT_TERM"  # 1-4週間
            elif dominant_trend in ["uptrend", "downtrend"]:
                return "MEDIUM_TERM"
            else:
                return "SHORT_TERM"

        except Exception:
            return "SHORT_TERM"

    def _calculate_stop_loss(
        self, timeframe_results: Dict[str, Dict], action: str
    ) -> Optional[float]:
        """ストップロス計算"""
        try:
            if action not in ["BUY", "SELL"]:
                return None

            current_price = None
            support_level = None
            resistance_level = None

            # 日足データを優先使用
            if "daily" in timeframe_results:
                tf_data = timeframe_results["daily"]
                current_price = tf_data.get("current_price")
                support_level = tf_data.get("support_level")
                resistance_level = tf_data.get("resistance_level")

            if not current_price:
                return None

            if action == "BUY" and support_level:
                # 買いポジション: サポートレベルの少し下
                stop_loss = support_level * 0.98
            elif action == "SELL" and resistance_level:
                # 売りポジション: レジスタンスレベルの少し上
                stop_loss = resistance_level * 1.02
            else:
                # デフォルト: 現在価格の±5%
                multiplier = 0.95 if action == "BUY" else 1.05
                stop_loss = current_price * multiplier

            return float(stop_loss)

        except Exception as e:
            logger.error(f"ストップロス計算エラー: {e}")
            return None

    def _calculate_take_profit(
        self, timeframe_results: Dict[str, Dict], action: str
    ) -> Optional[float]:
        """利益確定価格計算"""
        try:
            if action not in ["BUY", "SELL"]:
                return None

            current_price = None
            resistance_level = None
            support_level = None

            # 日足データを優先使用
            if "daily" in timeframe_results:
                tf_data = timeframe_results["daily"]
                current_price = tf_data.get("current_price")
                resistance_level = tf_data.get("resistance_level")
                support_level = tf_data.get("support_level")

            if not current_price:
                return None

            if action == "BUY" and resistance_level:
                # 買いポジション: レジスタンスレベルの少し下
                take_profit = resistance_level * 0.98
            elif action == "SELL" and support_level:
                # 売りポジション: サポートレベルの少し上
                take_profit = support_level * 1.02
            else:
                # デフォルト: 現在価格の±10%
                multiplier = 1.10 if action == "BUY" else 0.90
                take_profit = current_price * multiplier

            return float(take_profit)

        except Exception as e:
            logger.error(f"利益確定計算エラー: {e}")
            return None

    def _analyze_timeframe_agreement(
        self, timeframe_results: Dict[str, Dict]
    ) -> Dict[str, any]:
        """時間軸合意分析"""
        try:
            agreements = {
                "trend_agreement": [],
                "technical_agreement": [],
                "signal_agreement": [],
            }

            # トレンド合意
            trends = [
                (tf, data["trend_direction"]) for tf, data in timeframe_results.items()
            ]
            for i, (tf1, trend1) in enumerate(trends):
                for tf2, trend2 in trends[i + 1 :]:
                    if trend1 == trend2:
                        agreements["trend_agreement"].append(f"{tf1}-{tf2}: {trend1}")

            # テクニカル指標合意
            for tf, data in timeframe_results.items():
                rsi = data.get("technical_indicators", {}).get("rsi")
                if rsi:
                    if rsi > 70:
                        agreements["technical_agreement"].append(f"{tf}: RSI過買われ")
                    elif rsi < 30:
                        agreements["technical_agreement"].append(f"{tf}: RSI過売られ")

            return agreements

        except Exception as e:
            logger.error(f"時間軸合意分析エラー: {e}")
            return {}


if __name__ == "__main__":
    # テスト実行
    print("=== マルチタイムフレーム分析システム テスト ===")

    # サンプルデータ生成（2年間の日足データ）
    dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")
    np.random.seed(42)

    # より複雑な価格パターンを生成
    base_price = 2500
    trend_periods = [
        (0, 100, 0.001),  # 100日間上昇トレンド
        (100, 200, -0.0005),  # 100日間下降トレンド
        (200, 300, 0.0003),  # 100日間横ばい
        (300, 500, 0.0012),  # 200日間強い上昇
        (500, 600, -0.0008),  # 100日間調整
        (600, len(dates), 0.0005),  # 残り期間緩やかな上昇
    ]

    prices = [base_price]
    volatility = 0.02

    for i in range(1, len(dates)):
        # 現在のトレンド期間を特定
        current_trend = 0
        for start, end, trend in trend_periods:
            if start <= i < end:
                current_trend = trend
                break

        # ランダムウォーク + トレンド + 週末効果
        weekday_effect = -0.0002 if dates[i].weekday() == 4 else 0  # 金曜日効果
        seasonal_effect = 0.0005 * np.sin(2 * np.pi * i / 252)  # 年次季節性

        random_change = np.random.normal(
            current_trend + weekday_effect + seasonal_effect, volatility
        )
        new_price = prices[-1] * (1 + random_change)
        prices.append(max(new_price, 500))  # 価格下限設定

    # OHLCV生成
    sample_data = pd.DataFrame(index=dates)
    sample_data["Close"] = prices
    sample_data["Open"] = [p * np.random.uniform(0.995, 1.005) for p in prices]
    sample_data["High"] = [
        max(o, c) * np.random.uniform(1.000, 1.025)
        for o, c in zip(sample_data["Open"], sample_data["Close"])
    ]
    sample_data["Low"] = [
        min(o, c) * np.random.uniform(0.975, 1.000)
        for o, c in zip(sample_data["Open"], sample_data["Close"])
    ]
    sample_data["Volume"] = np.random.randint(1000000, 20000000, len(dates))

    try:
        analyzer = MultiTimeframeAnalyzer()

        print(f"サンプルデータ: {len(sample_data)}日分")
        print(
            f"価格範囲: {sample_data['Close'].min():.2f} - {sample_data['Close'].max():.2f}"
        )

        # 各時間軸でのリサンプリングテスト
        print("\n1. 時間軸リサンプリングテスト")
        for tf in ["daily", "weekly", "monthly"]:
            resampled = analyzer.resample_to_timeframe(sample_data, tf)
            print(f"✅ {tf}リサンプリング: {len(sample_data)} → {len(resampled)}期間")

        # 単一時間軸指標計算テスト
        print("\n2. 単一時間軸指標計算テスト")
        for tf in ["daily", "weekly", "monthly"]:
            tf_indicators = analyzer.calculate_timeframe_indicators(sample_data, tf)
            if not tf_indicators.empty:
                print(
                    f"✅ {tf}指標計算完了: {len(tf_indicators.columns)}指標, {len(tf_indicators)}期間"
                )

                # 最新データの表示
                latest = tf_indicators.iloc[-1]
                trend = latest.get("trend_direction", "unknown")
                strength = latest.get("trend_strength", 0)
                print(f"   最新トレンド: {trend} (強度: {strength:.1f})")
            else:
                print(f"❌ {tf}指標計算失敗")

        # マルチタイムフレーム統合分析テスト
        print("\n3. マルチタイムフレーム統合分析テスト")
        integrated_analysis = analyzer.analyze_multiple_timeframes(
            sample_data, "TEST_STOCK"
        )

        if "error" not in integrated_analysis:
            print("✅ 統合分析完了")

            # 時間軸別結果
            print("\n📊 時間軸別分析結果:")
            for tf, result in integrated_analysis["timeframes"].items():
                print(f"   {result['timeframe']}:")
                print(f"     トレンド: {result['trend_direction']}")
                print(f"     強度: {result['trend_strength']:.1f}")
                print(f"     現在価格: {result['current_price']:.2f}")

                if "technical_indicators" in result:
                    indicators = result["technical_indicators"]
                    if "rsi" in indicators:
                        print(f"     RSI: {indicators['rsi']:.1f}")
                    if "bb_position" in indicators:
                        print(f"     BB位置: {indicators['bb_position']:.2f}")

            # 統合結果
            integrated = integrated_analysis["integrated_analysis"]
            print("\n🔍 統合分析結果:")
            print(f"   総合トレンド: {integrated['overall_trend']}")
            print(f"   トレンド信頼度: {integrated['trend_confidence']:.1f}%")
            print(f"   整合性スコア: {integrated['consistency_score']:.1f}%")

            # 統合シグナル
            signal = integrated["integrated_signal"]
            print("\n📈 統合シグナル:")
            print(f"   アクション: {signal['action']}")
            print(f"   強度: {signal['strength']}")
            print(f"   シグナルスコア: {signal['signal_score']:.1f}")

            # リスク評価
            risk = integrated["risk_assessment"]
            print("\n⚠️  リスク評価:")
            print(f"   リスクレベル: {risk['risk_level']}")
            print(f"   リスクスコア: {risk['risk_score']:.1f}")
            print(f"   リスク要因数: {risk['total_risk_factors']}")
            if risk["risk_factors"]:
                for factor in risk["risk_factors"][:3]:  # 上位3個表示
                    print(f"     - {factor}")

            # 投資推奨
            recommendation = integrated["investment_recommendation"]
            print("\n💡 投資推奨:")
            print(f"   推奨: {recommendation['recommendation']}")
            print(f"   ポジションサイズ: {recommendation['position_size']}")
            print(f"   保有期間: {recommendation['holding_period']}")

            if recommendation.get("stop_loss_suggestion"):
                print(f"   ストップロス: {recommendation['stop_loss_suggestion']:.2f}")
            if recommendation.get("take_profit_suggestion"):
                print(f"   利益確定: {recommendation['take_profit_suggestion']:.2f}")

            print("\n📋 推奨理由:")
            for reason in recommendation.get("reasons", []):
                print(f"     - {reason}")

        else:
            print(f"❌ 統合分析エラー: {integrated_analysis['error']}")

        print("\n✅ マルチタイムフレーム分析システム テスト完了！")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
