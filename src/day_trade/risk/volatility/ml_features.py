#!/usr/bin/env python3
"""
機械学習特徴量生成モジュール

ボラティリティ予測のための包括的な特徴量エンジニアリング:
- 価格・リターン特徴量
- ボラティリティ関連特徴量
- テクニカル指標特徴量
- 時系列特徴量（ラグ、時間効果）
- マーケット構造特徴量
"""

import numpy as np
import pandas as pd
from typing import Optional

from .base import VolatilityEngineBase
from .realized_volatility import RealizedVolatilityCalculator
from .vix_indicator import VIXIndicatorCalculator
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class MLFeatureGenerator(VolatilityEngineBase):
    """
    機械学習特徴量生成クラス

    ボラティリティ予測に有効な多様な特徴量を生成します。
    """

    def __init__(self, model_cache_dir: str = "data/volatility_models"):
        """
        初期化

        Args:
            model_cache_dir: モデルキャッシュディレクトリ
        """
        super().__init__(model_cache_dir)
        
        self.rv_calculator = RealizedVolatilityCalculator(model_cache_dir)
        self.vix_calculator = VIXIndicatorCalculator(model_cache_dir)
        
        logger.info("ML特徴量生成器初期化完了")

    def prepare_ml_features_for_volatility(
        self, data: pd.DataFrame, lookback_days: int = 60
    ) -> pd.DataFrame:
        """
        ボラティリティ予測用機械学習特徴量準備

        Args:
            data: OHLCV価格データ
            lookback_days: 特徴量作成に使用する過去データ日数

        Returns:
            特徴量DataFrame
        """
        try:
            features = pd.DataFrame(index=data.index)

            # 基本価格特徴量
            features = self._add_price_features(features, data)

            # ボラティリティ特徴量
            features = self._add_volatility_features(features, data)

            # テクニカル指標特徴量
            features = self._add_technical_features(features, data)

            # 出来高特徴量
            features = self._add_volume_features(features, data)

            # 時系列・ラグ特徴量
            features = self._add_time_series_features(features, data)

            # VIX風指標特徴量
            features = self._add_vix_features(features, data)

            # マーケット構造特徴量
            features = self._add_market_structure_features(features, data)

            # NaN値処理
            features = features.fillna(method="ffill").fillna(method="bfill").fillna(0)

            logger.info(f"ML特徴量準備完了: {len(features.columns)}特徴量")
            return features

        except Exception as e:
            logger.error(f"ML特徴量準備エラー: {e}")
            return pd.DataFrame(index=data.index)

    def _add_price_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        基本価格特徴量を追加

        Args:
            features: 特徴量DataFrame
            data: 価格データ

        Returns:
            特徴量が追加されたDataFrame
        """
        # 基本リターン特徴量
        returns = np.log(data["Close"] / data["Close"].shift(1))
        features["returns"] = returns
        features["abs_returns"] = np.abs(returns)
        features["squared_returns"] = returns**2

        # 価格レンジ指標
        features["high_low_ratio"] = (data["High"] - data["Low"]) / data["Close"]
        features["open_close_ratio"] = (
            np.abs(data["Open"] - data["Close"]) / data["Close"]
        )

        # 価格位置指標
        features["close_position_in_range"] = (
            (data["Close"] - data["Low"]) / (data["High"] - data["Low"])
        )

        return features

    def _add_volatility_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        ボラティリティ関連特徴量を追加

        Args:
            features: 特徴量DataFrame
            data: 価格データ

        Returns:
            特徴量が追加されたDataFrame
        """
        # 複数期間の実現ボラティリティ
        for period in [5, 10, 20, 30]:
            vol_feature = self.rv_calculator.calculate_realized_volatility(
                data, window=period, annualize=True
            )
            features[f"realized_vol_{period}"] = vol_feature

        # ボラティリティクラスタリング指標
        for lag in [1, 2, 3, 5]:
            features[f"vol_lag_{lag}"] = features["realized_vol_20"].shift(lag)
            features[f"return_lag_{lag}"] = features["returns"].shift(lag)

        # ボラティリティ比率
        features["vol_ratio_5_20"] = features["realized_vol_5"] / features["realized_vol_20"]
        features["vol_ratio_10_30"] = features["realized_vol_10"] / features["realized_vol_30"]

        return features

    def _add_technical_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        テクニカル指標特徴量を追加

        Args:
            features: 特徴量DataFrame
            data: 価格データ

        Returns:
            特徴量が追加されたDataFrame
        """
        # RSI
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features["rsi"] = 100 - (100 / (1 + rs))

        # ボリンジャーバンド
        sma_20 = data["Close"].rolling(20).mean()
        std_20 = data["Close"].rolling(20).std()
        features["bb_position"] = (data["Close"] - (sma_20 - 2 * std_20)) / (
            4 * std_20
        )
        features["bb_width"] = (4 * std_20) / sma_20

        # MACD
        ema_12 = data["Close"].ewm(span=12).mean()
        ema_26 = data["Close"].ewm(span=26).mean()
        features["macd"] = ema_12 - ema_26
        features["macd_signal"] = features["macd"].ewm(span=9).mean()
        features["macd_histogram"] = features["macd"] - features["macd_signal"]

        # 移動平均乖離率
        for period in [5, 20, 50]:
            ma = data["Close"].rolling(period).mean()
            features[f"ma_deviation_{period}"] = (data["Close"] - ma) / ma

        return features

    def _add_volume_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        出来高関連特徴量を追加

        Args:
            features: 特徴量DataFrame
            data: 価格データ

        Returns:
            特徴量が追加されたDataFrame
        """
        # 基本出来高指標
        features["volume"] = data["Volume"]
        features["volume_ma"] = data["Volume"].rolling(20).mean()
        features["volume_ratio"] = data["Volume"] / features["volume_ma"]
        features["price_volume"] = data["Close"] * data["Volume"]

        # 出来高加重価格
        features["vwap"] = (
            (data["Close"] * data["Volume"]).rolling(20).sum()
            / data["Volume"].rolling(20).sum()
        )
        features["price_vwap_ratio"] = data["Close"] / features["vwap"]

        # 出来高トレンド
        features["volume_trend"] = data["Volume"].rolling(5).mean().pct_change()

        return features

    def _add_time_series_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        時系列・時間効果特徴量を追加

        Args:
            features: 特徴量DataFrame
            data: 価格データ

        Returns:
            特徴量が追加されたDataFrame
        """
        # リターンの移動統計量
        for window in [10, 20, 50]:
            features[f"return_mean_{window}"] = features["returns"].rolling(window).mean()
            features[f"return_std_{window}"] = features["returns"].rolling(window).std()
            features[f"return_skew_{window}"] = features["returns"].rolling(window).skew()
            features[f"return_kurt_{window}"] = features["returns"].rolling(window).kurt()

        # 時間系特徴量（曜日効果・月効果）
        features["day_of_week"] = data.index.dayofweek
        features["day_of_month"] = data.index.day
        features["month"] = data.index.month

        # 季節調整項
        features["seasonal_factor"] = np.sin(
            2 * np.pi * data.index.dayofyear / 365.25
        )

        # エクストリームリターン指標
        features["extreme_return_5d"] = (
            np.abs(features["returns"].rolling(5).sum()) 
            > features["returns"].rolling(60).std() * 2
        ).astype(int)

        return features

    def _add_vix_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        VIX風指標特徴量を追加

        Args:
            features: 特徴量DataFrame
            data: 価格データ

        Returns:
            特徴量が追加されたDataFrame
        """
        # VIX風指標とその派生指標
        features["vix_like"] = self.vix_calculator.calculate_vix_like_indicator(data)
        features["vix_change"] = features["vix_like"].diff()
        features["vix_relative"] = (
            features["vix_like"] / features["vix_like"].rolling(30).mean()
        )

        # VIXレジーム
        features["vix_regime"] = features["vix_like"].apply(
            self.vix_calculator.classify_vix_regime
        )
        
        # レジーム数値化
        regime_mapping = {
            "complacency": 1, "normal": 2, "concern": 3, "fear": 4, "panic": 5
        }
        features["vix_regime_numeric"] = features["vix_regime"].map(regime_mapping)

        return features

    def _add_market_structure_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        マーケット構造特徴量を追加

        Args:
            features: 特徴量DataFrame
            data: 価格データ

        Returns:
            特徴量が追加されたDataFrame
        """
        # ギャップ指標
        features["overnight_gap"] = (
            data["Open"] - data["Close"].shift(1)
        ) / data["Close"].shift(1)
        
        # イントラデイ動き
        features["intraday_range"] = (data["High"] - data["Low"]) / data["Open"]
        features["intraday_direction"] = (data["Close"] - data["Open"]) / data["Open"]

        # 価格効率性（自己相関）
        for lag in [1, 5, 10]:
            features[f"return_autocorr_{lag}"] = (
                features["returns"].rolling(50).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
                )
            )

        return features

    def add_lag_features(
        self, features: pd.DataFrame, target_columns: list, max_lags: int = 5
    ) -> pd.DataFrame:
        """
        指定した列にラグ特徴量を追加

        Args:
            features: 特徴量DataFrame
            target_columns: ラグを追加する列名のリスト
            max_lags: 最大ラグ期間

        Returns:
            ラグ特徴量が追加されたDataFrame
        """
        result = features.copy()

        for col in target_columns:
            if col in features.columns:
                for lag in range(1, max_lags + 1):
                    result[f"{col}_lag_{lag}"] = features[col].shift(lag)

        logger.info(f"ラグ特徴量追加完了: {len(target_columns)}列 x {max_lags}ラグ")
        return result

    def select_important_features(
        self, features: pd.DataFrame, target: pd.Series, top_k: int = 50
    ) -> pd.DataFrame:
        """
        重要特徴量選択（相関ベース簡易選択）

        Args:
            features: 特徴量DataFrame
            target: ターゲット変数
            top_k: 選択する特徴量数

        Returns:
            選択された特徴量DataFrame
        """
        try:
            # 特徴量とターゲットの相関計算
            correlations = {}
            for col in features.columns:
                if col in features.columns and not features[col].isnull().all():
                    corr = abs(features[col].corr(target))
                    if not np.isnan(corr):
                        correlations[col] = corr

            # 上位k個選択
            selected_features = sorted(
                correlations.keys(), key=lambda x: correlations[x], reverse=True
            )[:top_k]

            selected_df = features[selected_features]
            logger.info(f"重要特徴量選択完了: {len(selected_features)}特徴量")

            return selected_df

        except Exception as e:
            logger.error(f"特徴量選択エラー: {e}")
            return features

    def get_feature_summary(self, features: pd.DataFrame) -> dict:
        """
        特徴量の概要統計を取得

        Args:
            features: 特徴量DataFrame

        Returns:
            概要統計辞書
        """
        try:
            summary = {
                "feature_count": len(features.columns),
                "sample_count": len(features),
                "null_counts": features.isnull().sum().to_dict(),
                "feature_types": {},
                "basic_stats": {},
            }

            # 特徴量タイプ分類
            for col in features.columns:
                if "vol" in col:
                    feature_type = "volatility"
                elif "return" in col:
                    feature_type = "return"
                elif "vix" in col:
                    feature_type = "vix"
                elif "volume" in col:
                    feature_type = "volume"
                elif "day" in col or "month" in col:
                    feature_type = "time"
                else:
                    feature_type = "other"

                if feature_type not in summary["feature_types"]:
                    summary["feature_types"][feature_type] = 0
                summary["feature_types"][feature_type] += 1

            # 基本統計
            numeric_features = features.select_dtypes(include=[np.number])
            for col in numeric_features.columns:
                summary["basic_stats"][col] = {
                    "mean": float(numeric_features[col].mean()),
                    "std": float(numeric_features[col].std()),
                    "min": float(numeric_features[col].min()),
                    "max": float(numeric_features[col].max()),
                }

            return summary

        except Exception as e:
            logger.error(f"特徴量概要統計エラー: {e}")
            return {"error": str(e)}