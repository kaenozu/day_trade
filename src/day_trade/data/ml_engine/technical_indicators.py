#!/usr/bin/env python3
"""
Technical Indicators Module
テクニカル指標モジュール

このモジュールは様々なテクニカル指標の
計算機能を提供します。

Functions:
    calculate_rsi: RSI計算
    calculate_macd: MACD計算
    calculate_bollinger_bands: ボリンジャーバンド計算
    extract_fft_features: FFT特徴量抽出
    calculate_ml_trend_strength: ML拡張トレンド強度計算
    calculate_volatility_score: ボラティリティ予測スコア
    calculate_pattern_recognition_score: パターン認識スコア
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI（Relative Strength Index）計算
    
    Args:
        prices: 価格系列
        period: 計算期間
        
    Returns:
        pd.Series: RSI値
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD（Moving Average Convergence Divergence）計算
    
    Args:
        prices: 価格系列
        fast: 短期EMA期間
        slow: 長期EMA期間
        signal: シグナル線EMA期間
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (MACD線, シグナル線, ヒストグラム)
    """
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram


def calculate_bollinger_bands(
    prices: pd.Series, period: int = 20, std_dev: float = 2
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    ボリンジャーバンド計算
    
    Args:
        prices: 価格系列
        period: 移動平均期間
        std_dev: 標準偏差倍率
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (上限バンド, 中央線, 下限バンド)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def extract_fft_features(
    prices: pd.Series, n_features: int = 10
) -> List[pd.Series]:
    """
    FFT特徴量抽出 - Issue #707対応最適化版
    
    ベクトル化演算でCPU効率を向上
    
    Args:
        prices: 価格系列
        n_features: 抽出する特徴量数
        
    Returns:
        List[pd.Series]: FFT特徴量リスト
    """
    try:
        prices_clean = prices.dropna()
        if len(prices_clean) < n_features:
            logger.warning(
                f"データ数不足でFFT特徴量作成不可: "
                f"{len(prices_clean)} < {n_features}"
            )
            return [
                pd.Series([0.0] * len(prices), index=prices.index)
                for _ in range(n_features)
            ]

        # FFT計算（ベクトル化）
        fft = np.fft.fft(prices_clean.values)

        # 一括振幅計算（ループ除去）
        amplitudes = np.abs(fft[1:n_features + 1])

        # 効率的な特徴量系列作成
        fft_features = [
            pd.Series([amplitude] * len(prices), index=prices.index, dtype=np.float32)
            for amplitude in amplitudes
        ]

        return fft_features

    except Exception as e:
        logger.warning(f"FFT特徴量抽出エラー: {e}")
        # フォールバック: ゼロ特徴量
        return [
            pd.Series([0.0] * len(prices), index=prices.index)
            for _ in range(n_features)
        ]


def calculate_ml_trend_strength(
    prices: pd.Series, sma_20: pd.Series, sma_50: pd.Series
) -> float:
    """
    ML拡張トレンド強度計算
    
    Args:
        prices: 価格系列
        sma_20: 20日移動平均
        sma_50: 50日移動平均
        
    Returns:
        float: トレンド強度スコア
    """
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
            if current_sma50 != 0 else 0
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


def calculate_volatility_score(
    prices: pd.Series, volatility: float, volume: pd.Series
) -> float:
    """
    ボラティリティ予測スコア
    
    Args:
        prices: 価格系列
        volatility: ボラティリティ
        volume: 出来高系列
        
    Returns:
        float: ボラティリティスコア
    """
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


def calculate_pattern_recognition_score(prices: pd.Series) -> float:
    """
    パターン認識スコア (簡易版)
    
    Args:
        prices: 価格系列
        
    Returns:
        float: パターン認識スコア
    """
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


def calculate_advanced_technical_indicators(
    data: pd.DataFrame, symbol: str = "UNKNOWN"
) -> Dict[str, float]:
    """
    高度テクニカル指標計算（ML拡張版）
    
    Args:
        data: 価格データ
        symbol: 銘柄コード
        
    Returns:
        Dict[str, float]: テクニカル指標スコア辞書
    """
    try:
        if data is None or data.empty:
            logger.warning(f"データが空です: {symbol}")
            return get_default_ml_scores()

        # 基本的なテクニカル指標計算
        close_prices = (
            data["終値"] if "終値" in data.columns
            else data.get("Close", pd.Series())
        )

        if close_prices.empty or len(close_prices) < 20:
            logger.warning(
                f"価格データが不足: {symbol} ({len(close_prices)} 件)"
            )
            return get_default_ml_scores()

        # ML強化テクニカル指標
        ml_scores = {}

        try:
            # トレンド強度スコア (ML拡張)
            sma_20 = close_prices.rolling(20).mean()
            sma_50 = (
                close_prices.rolling(50).mean() if len(close_prices) >= 50
                else sma_20
            )
            trend_strength = calculate_ml_trend_strength(
                close_prices, sma_20, sma_50
            )
            ml_scores["trend_strength"] = min(100, max(0, trend_strength))

            # 価格変動予測スコア
            volatility = (
                close_prices.pct_change().rolling(20).std().iloc[-1]
                if len(close_prices) > 20 else 0.02
            )
            volume_data = data.get("出来高", data.get("Volume", pd.Series()))
            volatility_score = calculate_volatility_score(
                close_prices, volatility, volume_data
            )
            ml_scores["volatility_prediction"] = min(100, max(0, volatility_score))

            # パターン認識スコア (簡易版)
            pattern_score = calculate_pattern_recognition_score(close_prices)
            ml_scores["pattern_recognition"] = min(100, max(0, pattern_score))

            logger.debug(f"ML指標計算完了: {symbol}")
            return ml_scores

        except Exception as e:
            logger.warning(f"ML指標計算でエラー {symbol}: {e}")
            return get_default_ml_scores()

    except Exception as e:
        logger.error(f"高度テクニカル指標計算エラー {symbol}: {e}")
        return get_default_ml_scores()


def get_default_ml_scores() -> Dict[str, float]:
    """
    デフォルトMLスコア
    
    Returns:
        Dict[str, float]: デフォルトスコア辞書
    """
    return {
        "trend_strength": 50.0,
        "volatility_prediction": 50.0,
        "pattern_recognition": 50.0,
    }


# Williams %R計算
def calculate_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Williams %R計算
    
    Args:
        high: 高値系列
        low: 安値系列
        close: 終値系列
        period: 計算期間
        
    Returns:
        pd.Series: Williams %R値
    """
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r


# Stochastic Oscillator計算
def calculate_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, 
    k_period: int = 14, d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator計算
    
    Args:
        high: 高値系列
        low: 安値系列
        close: 終値系列
        k_period: %K期間
        d_period: %D期間
        
    Returns:
        Tuple[pd.Series, pd.Series]: (%K, %D)
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent.rolling(d_period).mean()
    return k_percent, d_percent


# ATR (Average True Range)計算
def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    ATR（Average True Range）計算
    
    Args:
        high: 高値系列
        low: 安値系列
        close: 終値系列
        period: 計算期間
        
    Returns:
        pd.Series: ATR値
    """
    high_low = high - low
    high_close_prev = np.abs(high - close.shift(1))
    low_close_prev = np.abs(low - close.shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr = true_range.rolling(period).mean()
    return atr