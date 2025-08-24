#!/usr/bin/env python3
"""
ML Engine Utilities Module
MLエンジンユーティリティモジュール

このモジュールは機械学習エンジンで使用される
ユーティリティ関数を提供します。

Functions:
    extract_fft_features_async: 非同期FFT特徴量抽出
    measure_inference_time_optimized: 最適化された推論時間測定
    measure_inference_time_async: 非同期推論時間測定
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


async def extract_fft_features_async(
    prices: pd.Series, n_features: int = 10
) -> List[pd.Series]:
    """
    非同期FFT特徴量抽出 - Issue #707対応

    CPU集約的な処理を非同期処理で最適化

    Args:
        prices: 価格系列
        n_features: 抽出する特徴量数

    Returns:
        List[pd.Series]: FFT特徴量リスト
    """

    def _compute_fft_chunk(chunk_data):
        """FFTチャンク計算"""
        try:
            fft_chunk = np.fft.fft(chunk_data)
            return np.abs(fft_chunk[1:n_features + 1])
        except Exception as e:
            logger.warning(f"FFTチャンク計算エラー: {e}")
            return np.zeros(n_features)

    try:
        prices_clean = prices.dropna()
        if len(prices_clean) < n_features:
            return [
                pd.Series([0.0] * len(prices), index=prices.index)
                for _ in range(n_features)
            ]

        # チャンクサイズ計算
        chunk_size = max(256, len(prices_clean) // 4)  # 最小256、最大4分割
        chunks = [
            prices_clean.values[i:i + chunk_size]
            for i in range(0, len(prices_clean), chunk_size)
        ]

        # 非同期でFFT処理
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = [
                loop.run_in_executor(executor, _compute_fft_chunk, chunk)
                for chunk in chunks if len(chunk) >= n_features
            ]

            if tasks:
                results = await asyncio.gather(*tasks)
                # 結果をマージ（平均）
                amplitudes = np.mean(results, axis=0)
            else:
                # フォールバック: 同期処理
                fft = np.fft.fft(prices_clean.values)
                amplitudes = np.abs(fft[1:n_features + 1])

        # 効率的な特徴量系列作成
        fft_features = [
            pd.Series([amplitude] * len(prices), index=prices.index, dtype=np.float32)
            for amplitude in amplitudes
        ]

        return fft_features

    except Exception as e:
        logger.warning(f"非同期FFT特徴量抽出エラー: {e}")
        # フォールバック: 同期処理
        from .technical_indicators import extract_fft_features
        return extract_fft_features(prices, n_features)


def measure_inference_time_optimized(
    model, test_data: pd.DataFrame, n_iterations: int = 10
) -> Optional[float]:
    """
    最適化された推論時間測定 - Issue #707対応

    Args:
        model: 推論対象モデル
        test_data: テストデータ
        n_iterations: 測定回数

    Returns:
        Optional[float]: 平均推論時間(ms)、エラー時はNone
    """
    try:
        import concurrent.futures

        test_sample = test_data.tail(10)

        def single_inference():
            """単一推論実行"""
            start = time.time()
            try:
                _ = model.predict(test_sample)
                return (time.time() - start) * 1000  # ms変換
            except Exception as e:
                logger.warning(f"推論時間測定エラー: {e}")
                return None

        # 並列推論時間測定（I/O待機を活用）
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(4, n_iterations)
        ) as executor:
            futures = [executor.submit(single_inference) for _ in range(n_iterations)]

            inference_times = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    inference_times.append(result)

        if inference_times:
            avg_time = np.mean(inference_times)
            logger.info(
                f"並列推論時間測定完了: {len(inference_times)}回測定、"
                f"平均{avg_time:.2f}ms"
            )
            return avg_time
        else:
            return None

    except Exception as e:
        logger.warning(f"最適化推論時間測定失敗: {e}")
        return None


async def measure_inference_time_async(
    model, test_data: pd.DataFrame, n_iterations: int = 10
) -> Optional[float]:
    """
    非同期推論時間測定 - Issue #707対応

    Args:
        model: 推論対象モデル
        test_data: テストデータ
        n_iterations: 測定回数

    Returns:
        Optional[float]: 平均推論時間(ms)、エラー時はNone
    """
    try:
        test_sample = test_data.tail(10)

        async def async_inference():
            """非同期推論実行"""
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                start = time.time()
                try:
                    result = await loop.run_in_executor(
                        executor, model.predict, test_sample
                    )
                    return (time.time() - start) * 1000  # ms変換
                except Exception as e:
                    logger.warning(f"非同期推論時間測定エラー: {e}")
                    return None

        # 非同期推論時間測定
        tasks = [async_inference() for _ in range(n_iterations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 有効な結果のみ抽出
        inference_times = [
            result for result in results
            if isinstance(result, (int, float)) and result is not None
        ]

        if inference_times:
            avg_time = np.mean(inference_times)
            logger.info(
                f"非同期推論時間測定完了: {len(inference_times)}回測定、"
                f"平均{avg_time:.2f}ms"
            )
            return avg_time
        else:
            return None

    except Exception as e:
        logger.warning(f"非同期推論時間測定失敗: {e}")
        return None


def validate_data_shape(data: np.ndarray, expected_shape: tuple) -> bool:
    """
    データ形状の検証

    Args:
        data: 検証対象データ
        expected_shape: 期待する形状

    Returns:
        bool: 形状が正しいかどうか
    """
    if data is None:
        return False

    if len(data.shape) != len(expected_shape):
        return False

    for i, (actual, expected) in enumerate(zip(data.shape, expected_shape)):
        if expected is not None and actual != expected:
            return False

    return True


def normalize_features(features: np.ndarray, scaler=None) -> tuple:
    """
    特徴量の正規化

    Args:
        features: 正規化対象の特徴量
        scaler: 使用するスケーラー（Noneの場合は新規作成）

    Returns:
        tuple: (正規化済み特徴量, スケーラー)
    """
    try:
        from sklearn.preprocessing import StandardScaler

        if scaler is None:
            scaler = StandardScaler()
            normalized = scaler.fit_transform(features)
        else:
            normalized = scaler.transform(features)

        return normalized, scaler

    except ImportError:
        logger.warning("scikit-learnが利用できません。正規化をスキップします。")
        return features, None
    except Exception as e:
        logger.error(f"特徴量正規化エラー: {e}")
        return features, scaler


def calculate_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    相関行列計算

    Args:
        data: 相関を計算するデータフレーム

    Returns:
        pd.DataFrame: 相関行列
    """
    try:
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        return correlation_matrix
    except Exception as e:
        logger.error(f"相関行列計算エラー: {e}")
        return pd.DataFrame()


def detect_outliers(data: pd.Series, method: str = "iqr", threshold: float = 1.5) -> pd.Series:
    """
    外れ値検出

    Args:
        data: 検出対象のデータ系列
        method: 検出手法（"iqr" または "zscore"）
        threshold: 閾値

    Returns:
        pd.Series: 外れ値フラグ（True: 外れ値）
    """
    try:
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)

        elif method == "zscore":
            from scipy import stats
            z_scores = np.abs(stats.zscore(data))
            outliers = z_scores > threshold

        else:
            logger.warning(f"未知の外れ値検出手法: {method}")
            return pd.Series([False] * len(data), index=data.index)

        return outliers

    except ImportError:
        logger.warning("scipyが利用できません。外れ値検出をスキップします。")
        return pd.Series([False] * len(data), index=data.index)
    except Exception as e:
        logger.error(f"外れ値検出エラー: {e}")
        return pd.Series([False] * len(data), index=data.index)


def smooth_time_series(data: pd.Series, method: str = "ema", window: int = 10) -> pd.Series:
    """
    時系列データの平滑化

    Args:
        data: 平滑化対象のデータ系列
        method: 平滑化手法（"sma", "ema", "median"）
        window: ウィンドウサイズ

    Returns:
        pd.Series: 平滑化済みデータ系列
    """
    try:
        if method == "sma":
            smoothed = data.rolling(window=window).mean()
        elif method == "ema":
            smoothed = data.ewm(span=window).mean()
        elif method == "median":
            smoothed = data.rolling(window=window).median()
        else:
            logger.warning(f"未知の平滑化手法: {method}")
            return data

        # 欠損値を元データで補完
        smoothed = smoothed.fillna(data)
        return smoothed

    except Exception as e:
        logger.error(f"時系列平滑化エラー: {e}")
        return data


def calculate_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """
    特徴量重要度計算

    Args:
        model: 訓練済みモデル
        feature_names: 特徴量名リスト

    Returns:
        Dict[str, float]: 特徴量重要度辞書
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            logger.warning("特徴量重要度を取得できません")
            return {}

        # 正規化
        if len(importances) > 0:
            importances = importances / np.sum(importances)

        importance_dict = dict(zip(feature_names[:len(importances)], importances))
        return importance_dict

    except Exception as e:
        logger.error(f"特徴量重要度計算エラー: {e}")
        return {}


def memory_usage_info() -> Dict[str, float]:
    """
    メモリ使用量情報取得

    Returns:
        Dict[str, float]: メモリ使用量情報
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / (1024 * 1024),  # MB
            "vms_mb": memory_info.vms / (1024 * 1024),  # MB
            "percent": process.memory_percent(),
        }

    except ImportError:
        logger.warning("psutilが利用できません。メモリ情報を取得できません。")
        return {}
    except Exception as e:
        logger.error(f"メモリ使用量取得エラー: {e}")
        return {}