#!/usr/bin/env python3
"""
Data Processing Module
データ処理モジュール

このモジュールは機械学習のための
データ前処理機能を提供します。

Functions:
    prepare_data: 高度データ前処理パイプライン
    engineer_features: 特徴量エンジニアリング
    create_sequences: 時系列シーケンスデータ作成
"""

import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger
# MacroEconomicFeaturesのインポートをオプション化
try:
    from ..macro_economic_features import MacroEconomicFeatures
except ImportError:
    # フォールバック用のダミークラス
    class MacroEconomicFeatures:
        def add_macro_features(self, data, symbol):
            return data
from .config import ModelConfig
from .technical_indicators import (
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
    extract_fft_features,
)

logger = get_context_logger(__name__)


class DataProcessor:
    """
    データ処理クラス
    
    機械学習用のデータ前処理を担当します。
    
    Attributes:
        config: モデル設定
        scaler: データスケーラー
        feature_selector: 特徴量選択器
        macro_features: マクロ経済特徴量エンジン
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.scaler = None
        self.feature_selector = None
        self.macro_features = MacroEconomicFeatures()
    
    def prepare_data(
        self,
        market_data: pd.DataFrame,
        target_column: str = "終値",
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        高度データ前処理パイプライン
        
        Args:
            market_data: 市場データ
            target_column: ターゲット列名
            feature_columns: 特徴量列名リスト
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (特徴量配列, ターゲット配列)
        """
        logger.info(f"データ前処理開始: {len(market_data)} レコード")

        if feature_columns is None:
            # 数値カラムを自動選択
            numeric_columns = market_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            feature_columns = [
                col for col in numeric_columns if col != target_column
            ]

        # 基本特徴量エンジニアリング
        processed_data = self._engineer_features(market_data, feature_columns)

        # マクロ経済特徴量の追加
        try:
            symbol = getattr(market_data, 'symbol', 'UNKNOWN')
            processed_data = self.macro_features.add_macro_features(
                processed_data, symbol
            )
            logger.info(f"マクロ経済特徴量統合完了: {symbol}")
        except Exception as e:
            logger.warning(f"マクロ経済特徴量統合スキップ: {e}")

        # 系列データ作成
        sequences, targets = self._create_sequences(
            processed_data,
            target_column,
            self.config.sequence_length,
            self.config.prediction_horizon,
        )

        logger.info(f"前処理完了: {sequences.shape} -> {targets.shape}")
        return sequences, targets

    def _engineer_features(
        self, data: pd.DataFrame, feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        高度特徴量エンジニアリング - Issue #709対応最適化版
        
        ベクトル化演算で大幅高速化、並列処理対応
        
        Args:
            data: 入力データ
            feature_columns: 特徴量列名リスト
            
        Returns:
            pd.DataFrame: 処理済みデータ
        """
        start_time = time.time()

        result = data[feature_columns + ["終値"]].copy()

        # 最適化されたテクニカル指標計算
        target_columns = ["終値", "高値", "安値"]
        available_columns = [col for col in target_columns if col in result.columns]

        # 並列化可能な期間パラメータ
        ma_periods = [5, 10, 20, 50, 100, 200]
        vol_periods = [10, 20]
        momentum_periods = [5, 10, 20]

        # バッチ処理で移動平均計算（メモリ効率化）
        for col in available_columns:
            col_series = result[col]
            col_pct = col_series.pct_change()  # 1回計算して再利用

            # 移動平均とEMAをバッチ計算
            for period in ma_periods:
                result[f"{col}_MA_{period}"] = (
                    col_series.rolling(period, min_periods=1).mean()
                )
                result[f"{col}_EMA_{period}"] = (
                    col_series.ewm(span=period, min_periods=1).mean()
                )

            # ボラティリティ指標をバッチ計算
            for period in vol_periods:
                result[f"{col}_volatility_{period}"] = (
                    col_pct.rolling(period, min_periods=1).std()
                )

            # モメンタム指標をバッチ計算
            for period in momentum_periods:
                result[f"{col}_momentum_{period}"] = col_pct * period
                result[f"{col}_roc_{period}"] = (
                    (col_series / col_series.shift(period) - 1) * 100
                )

        # RSI (複数期間)
        if "終値" in result.columns:
            for period in [14, 21, 30]:
                result[f"RSI_{period}"] = calculate_rsi(result["終値"], period)

        # MACD
        if "終値" in result.columns:
            macd_line, macd_signal, macd_histogram = calculate_macd(
                result["終値"]
            )
            result["MACD"] = macd_line
            result["MACD_Signal"] = macd_signal
            result["MACD_Histogram"] = macd_histogram

        # ボリンジャーバンド
        if "終値" in result.columns:
            for period in [20, 50]:
                bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
                    result["終値"], period
                )
                result[f"BB_Upper_{period}"] = bb_upper
                result[f"BB_Middle_{period}"] = bb_middle
                result[f"BB_Lower_{period}"] = bb_lower
                result[f"BB_Width_{period}"] = (bb_upper - bb_lower) / bb_middle
                result[f"BB_Position_{period}"] = (result["終値"] - bb_lower) / (
                    bb_upper - bb_lower
                )

        # 高度テクニカル指標の追加
        self._add_advanced_indicators(result)

        # フーリエ変換特徴量（周期性検出）
        if "終値" in result.columns and len(result) >= 100:
            fft_features = extract_fft_features(result["終値"], n_features=10)
            for i, feature in enumerate(fft_features):
                result[f"FFT_feature_{i}"] = feature

        # 欠損値処理
        result = result.fillna(method="ffill").fillna(method="bfill")

        # 正規化
        result = self._normalize_features(result)

        # 特徴量選択
        if len(result.columns) > self.config.num_features:
            result = self._select_features(result)

        processing_time = time.time() - start_time
        logger.info(
            f"特徴量エンジニアリング完了: {result.shape[1]} 特徴量 "
            f"({processing_time:.3f}秒)"
        )
        return result

    def _add_advanced_indicators(self, result: pd.DataFrame):
        """高度テクニカル指標の追加"""
        if "終値" not in result.columns:
            return

        # Williams %R
        for period in [14, 21]:
            high_col = "高値" if "高値" in result.columns else "終値"
            low_col = "安値" if "安値" in result.columns else "終値"
            if high_col in result.columns and low_col in result.columns:
                highest_high = result[high_col].rolling(period).max()
                lowest_low = result[low_col].rolling(period).min()
                result[f"Williams_R_{period}"] = (
                    -100 * (highest_high - result["終値"]) /
                    (highest_high - lowest_low)
                )

        # Stochastic Oscillator
        for period in [14, 21]:
            high_col = "高値" if "高値" in result.columns else "終値"
            low_col = "安値" if "安値" in result.columns else "終値"
            if high_col in result.columns and low_col in result.columns:
                lowest_low = result[low_col].rolling(period).min()
                highest_high = result[high_col].rolling(period).max()
                k_percent = (
                    100 * (result["終値"] - lowest_low) /
                    (highest_high - lowest_low)
                )
                result[f"Stoch_K_{period}"] = k_percent
                result[f"Stoch_D_{period}"] = k_percent.rolling(3).mean()

        # VWAP (Volume Weighted Average Price)
        volume_col = "出来高" if "出来高" in result.columns else None
        if volume_col and volume_col in result.columns:
            result["VWAP"] = (
                (result["終値"] * result[volume_col]).cumsum() /
                result[volume_col].cumsum()
            )
            result["VWAP_ratio"] = result["終値"] / result["VWAP"]

    def _normalize_features(self, result: pd.DataFrame) -> pd.DataFrame:
        """特徴量の正規化"""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        result[numeric_columns] = scaler.fit_transform(result[numeric_columns])
        self.scaler = scaler
        
        return result

    def _select_features(self, result: pd.DataFrame) -> pd.DataFrame:
        """特徴量選択（相関による）"""
        correlation_with_target = (
            result.corr()["終値"].abs().sort_values(ascending=False)
        )
        selected_features = correlation_with_target.head(
            self.config.num_features
        ).index.tolist()
        return result[selected_features]

    def _create_sequences(
        self, data: pd.DataFrame, target_col: str, seq_len: int, pred_horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        時系列シーケンスデータ作成 - Issue #709対応最適化版
        
        NumPy vectorized operations使用で大幅高速化
        
        Args:
            data: データフレーム
            target_col: ターゲット列名
            seq_len: シーケンス長
            pred_horizon: 予測期間
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (シーケンス, ターゲット)
        """
        start_time = time.time()

        data_values = data.values.astype(np.float32)  # メモリ効率化
        target_values = data[target_col].values.astype(np.float32)

        # 事前にサイズ計算
        n_sequences = len(data_values) - seq_len - pred_horizon + 1
        n_features = data_values.shape[1]

        if n_sequences <= 0:
            logger.warning(
                f"シーケンス数不足: data_len={len(data_values)}, "
                f"seq_len={seq_len}, pred_horizon={pred_horizon}"
            )
            return np.array([]), np.array([])

        # 効率的な配列事前確保
        sequences = np.empty((n_sequences, seq_len, n_features), dtype=np.float32)
        targets = np.empty((n_sequences, pred_horizon), dtype=np.float32)

        # ベクタ化されたスライシング使用
        for i in range(n_sequences):
            sequences[i] = data_values[i:i + seq_len]
            targets[i] = target_values[i + seq_len:i + seq_len + pred_horizon]

        # さらなる最適化: stride_tricksを使用（メモリ効率的）
        try:
            from numpy.lib.stride_tricks import sliding_window_view

            # NumPy 1.20+でsliding_window_viewが利用可能
            sequences_optimized = sliding_window_view(
                data_values[:-pred_horizon],
                window_shape=(seq_len, n_features)
            ).squeeze()

            targets_optimized = sliding_window_view(
                target_values[seq_len:],
                window_shape=pred_horizon
            ).squeeze()

            # 形状調整
            if sequences_optimized.ndim == 2:
                sequences_optimized = sequences_optimized.reshape(
                    -1, seq_len, n_features
                )
            if targets_optimized.ndim == 1:
                targets_optimized = targets_optimized.reshape(-1, pred_horizon)

            sequences = sequences_optimized.astype(np.float32)
            targets = targets_optimized.astype(np.float32)

        except ImportError:
            # NumPy < 1.20の場合はストライド最適化をスキップ
            pass
        except Exception as e:
            logger.warning(f"高速スライディングウィンドウ最適化失敗: {e}")

        processing_time = time.time() - start_time
        logger.info(
            f"シーケンスデータ作成完了: {sequences.shape} -> {targets.shape} "
            f"({processing_time:.3f}秒)"
        )

        return sequences, targets


def prepare_data(
    market_data: pd.DataFrame,
    config: ModelConfig,
    target_column: str = "終値",
    feature_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    データ前処理のファクトリ関数
    
    Args:
        market_data: 市場データ
        config: モデル設定
        target_column: ターゲット列名
        feature_columns: 特徴量列名リスト
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (特徴量配列, ターゲット配列)
    """
    processor = DataProcessor(config)
    return processor.prepare_data(market_data, target_column, feature_columns)