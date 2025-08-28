#!/usr/bin/env python3
"""
マルチタイムフレーム分析システム - トレンド分析モジュール
Issue #315: 高度テクニカル指標・ML機能拡張

トレンド方向・強度・サポートレジスタンス分析機能を提供
"""

import warnings
from typing import Dict

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TrendAnalyzer:
    """
    トレンド分析クラス
    
    トレンド方向・強度分析とサポート・レジスタンスレベル特定機能を提供
    """
    
    def __init__(self):
        """初期化"""
        logger.info("トレンド分析システム初期化完了")
    
    def calculate_trend_direction(
        self, df: pd.DataFrame, timeframe: str
    ) -> pd.Series:
        """
        トレンド方向計算
        
        Args:
            df: テクニカル指標計算済みDataFrame
            timeframe: 時間軸
            
        Returns:
            トレンド方向Series
        """
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
    
    def calculate_trend_strength(
        self, df: pd.DataFrame, timeframe: str
    ) -> pd.Series:
        """
        トレンド強度計算（0-100）
        
        Args:
            df: テクニカル指標計算済みDataFrame
            timeframe: 時間軸
            
        Returns:
            トレンド強度Series
        """
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
    
    def identify_support_resistance_levels(
        self, df: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        """
        サポート・レジスタンスレベル特定
        
        Args:
            df: 価格データ
            timeframe: 時間軸
            
        Returns:
            サポート・レジスタンス情報を含むDataFrame
        """
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
            df.iloc[lookback:, df.columns.get_loc("resistance_level")] = (
                resistance_levels
            )

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
    
    def analyze_trend_data(
        self, df: pd.DataFrame, timeframe: str
    ) -> Dict[str, any]:
        """
        トレンドデータ統合分析
        
        Args:
            df: 価格データ
            timeframe: 時間軸
            
        Returns:
            トレンド分析結果辞書
        """
        try:
            # トレンド方向・強度計算
            df["trend_direction"] = self.calculate_trend_direction(df, timeframe)
            df["trend_strength"] = self.calculate_trend_strength(df, timeframe)
            
            # サポート・レジスタンスレベル特定
            df = self.identify_support_resistance_levels(df, timeframe)
            
            # 最新データ取得
            latest_data = df.iloc[-1] if len(df) > 0 else None
            
            if latest_data is None:
                return {}
            
            analysis = {
                "trend_direction": latest_data.get("trend_direction", "unknown"),
                "trend_strength": float(latest_data.get("trend_strength", 50)),
                "support_level": None,
                "resistance_level": None,
            }
            
            # サポート・レジスタンス
            if "support_level" in latest_data and pd.notna(latest_data["support_level"]):
                analysis["support_level"] = float(latest_data["support_level"])
            
            if "resistance_level" in latest_data and pd.notna(latest_data["resistance_level"]):
                analysis["resistance_level"] = float(latest_data["resistance_level"])
            
            return analysis
            
        except Exception as e:
            logger.error(f"トレンドデータ分析エラー: {e}")
            return {}