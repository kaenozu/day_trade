#!/usr/bin/env python3
"""
マルチタイムフレーム分析システム - テクニカル指標計算モジュール
Issue #315: 高度テクニカル指標・ML機能拡張

各時間軸でのテクニカル指標計算機能を提供
"""

import warnings
from typing import Dict, Union

import pandas as pd

from ...utils.logging_config import get_context_logger
from ..advanced_technical_indicators import AdvancedTechnicalIndicators
from .timeframe_resampler import TimeframeResampler

logger = get_context_logger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TechnicalCalculator:
    """
    テクニカル指標計算クラス
    
    各時間軸でのテクニカル指標を計算する機能を提供
    """
    
    def __init__(self):
        """初期化"""
        self.resampler = TimeframeResampler()
        self.advanced_indicators = AdvancedTechnicalIndicators()
        
        logger.info("テクニカル指標計算システム初期化完了")
    
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
            tf_data = self.resampler.resample_to_timeframe(data, timeframe)

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

            logger.info(f"{timeframe}指標計算完了: {len(df.columns)}指標")
            return df

        except Exception as e:
            logger.error(f"{timeframe}指標計算エラー: {e}")
            return pd.DataFrame()
    
    def _get_periods_for_timeframe(self, timeframe: str) -> Dict[str, Union[int, Dict]]:
        """
        時間軸に応じた指標期間を取得
        
        Args:
            timeframe: 時間軸キー
            
        Returns:
            指標期間設定辞書
        """
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
    
    def get_indicator_summary(self, df: pd.DataFrame, timeframe: str) -> Dict[str, any]:
        """
        テクニカル指標サマリーを取得
        
        Args:
            df: 指標計算済みDataFrame
            timeframe: 時間軸
            
        Returns:
            指標サマリー辞書
        """
        try:
            if df.empty:
                return {}
            
            latest_data = df.iloc[-1]
            summary = {
                "timeframe": timeframe,
                "data_points": len(df),
                "current_price": float(latest_data["Close"]),
                "technical_indicators": {},
            }
            
            # 主要テクニカル指標
            if "rsi" in latest_data and pd.notna(latest_data["rsi"]):
                summary["technical_indicators"]["rsi"] = float(latest_data["rsi"])
            
            if "macd" in latest_data and pd.notna(latest_data["macd"]):
                summary["technical_indicators"]["macd"] = float(latest_data["macd"])
            
            if "bb_position" in latest_data and pd.notna(latest_data["bb_position"]):
                summary["technical_indicators"]["bb_position"] = float(
                    latest_data["bb_position"]
                )
            
            # 一目均衡表シグナル
            if "ichimoku_signal" in latest_data:
                summary["ichimoku_signal"] = str(latest_data["ichimoku_signal"])
            
            return summary
            
        except Exception as e:
            logger.error(f"指標サマリー取得エラー: {e}")
            return {}