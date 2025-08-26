#!/usr/bin/env python3
"""
マルチタイムフレーム分析システム - データリサンプリングモジュール
Issue #315: 高度テクニカル指標・ML機能拡張

時系列データを異なる時間軸にリサンプリングする機能を提供
"""

import warnings
from typing import Dict

import pandas as pd

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TimeframeResampler:
    """
    時間軸リサンプリングクラス
    
    日足データを週足・月足にリサンプリングする機能を提供
    """
    
    def __init__(self):
        """初期化"""
        self.timeframes = {
            "daily": {"period": "D", "name": "日足", "weight": 0.4},
            "weekly": {"period": "W", "name": "週足", "weight": 0.35},
            "monthly": {"period": "M", "name": "月足", "weight": 0.25},
        }
        
        logger.info("タイムフレームリサンプラー初期化完了")
    
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
    
    def get_timeframe_info(self, timeframe: str) -> Dict[str, any]:
        """
        時間軸情報を取得
        
        Args:
            timeframe: 時間軸キー
            
        Returns:
            時間軸情報辞書
        """
        return self.timeframes.get(timeframe, {})
    
    def get_all_timeframes(self) -> Dict[str, Dict]:
        """
        すべての時間軸情報を取得
        
        Returns:
            すべての時間軸情報
        """
        return self.timeframes.copy()