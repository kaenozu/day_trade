"""
pandas最適化ユーティリティ

pandas DataFrameの処理速度を大幅に向上させる最適化機能。
NumPy vectorization、メモリ効率的なdtype、並列処理を活用。
Phase 2: パフォーマンス最適化プロジェクト対応
"""

import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import wraps
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .logging_config import get_context_logger
from .performance_analyzer import profile_performance
from .performance_config import get_performance_config

logger = get_context_logger(__name__)

# NumPy警告を抑制
warnings.filterwarnings("ignore", category=np.RankWarning)


def optimize_dataframe_dtypes(
    df: pd.DataFrame, aggressive: bool = False
) -> pd.DataFrame:
    """
    DataFrameのデータ型を最適化してメモリ使用量を削減

    Args:
        df: 最適化するDataFrame
        aggressive: 積極的な最適化（精度低下のリスク）

    Returns:
        pd.DataFrame: 最適化されたDataFrame
    """
    optimized = df.copy()
    original_memory = df.memory_usage(deep=True).sum()

    for col in optimized.columns:
        col_type = optimized[col].dtype

        if col_type == "object":
            # 文字列列をcategoryに変換（カーディナリティが低い場合）
            unique_ratio = optimized[col].nunique() / len(optimized[col])
            if unique_ratio < 0.5:  # ユニーク値が50%以下の場合
                optimized[col] = optimized[col].astype("category")

        elif np.issubdtype(col_type, np.integer):
            # 整数型の最適化
            c_min = optimized[col].min()
            c_max = optimized[col].max()

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                optimized[col] = optimized[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                optimized[col] = optimized[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                optimized[col] = optimized[col].astype(np.int32)

        elif np.issubdtype(col_type, np.floating):
            # 浮動小数点型の最適化
            if aggressive:
                # 積極的最適化: float64 → float32（精度低下のリスク）
                if col_type == np.float64:
                    optimized[col] = pd.to_numeric(optimized[col], downcast="float")
            else:
                # 保守的最適化: 値域に基づく最適化
                c_min = optimized[col].min()
                c_max = optimized[col].max()

                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    optimized[col] = optimized[col].astype(np.float32)

    optimized_memory = optimized.memory_usage(deep=True).sum()
    memory_reduction = (original_memory - optimized_memory) / original_memory * 100

    logger.info(
        "DataFrame dtypes optimized",
        original_memory_mb=original_memory / 1024**2,
        optimized_memory_mb=optimized_memory / 1024**2,
        memory_reduction_percent=memory_reduction,
        aggressive=aggressive,
    )

    return optimized


def chunked_processing(chunk_size: Optional[int] = None):
    """
    大量データを効率的に処理するためのチャンク処理デコレータ

    Args:
        chunk_size: チャンクサイズ（Noneの場合は設定から取得）
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            config = get_performance_config()
            actual_chunk_size = chunk_size or config.compute.chunk_size

            if len(df) <= actual_chunk_size:
                # 小さなデータはそのまま処理
                return func(df, *args, **kwargs)

            # チャンクに分割して処理
            results = []
            total_chunks = len(df) // actual_chunk_size + 1

            logger.info(
                "Starting chunked processing",
                total_rows=len(df),
                chunk_size=actual_chunk_size,
                total_chunks=total_chunks,
            )

            for i in range(0, len(df), actual_chunk_size):
                chunk = df.iloc[i : i + actual_chunk_size]
                chunk_result = func(chunk, *args, **kwargs)
                results.append(chunk_result)

                if i % (actual_chunk_size * 10) == 0:  # 進捗ログ
                    logger.debug(
                        "Chunk processing progress",
                        processed_chunks=len(results),
                        total_chunks=total_chunks,
                    )

            # 結果をマージ
            if isinstance(results[0], (pd.DataFrame, pd.Series)):
                final_result = pd.concat(results, ignore_index=True)
            else:
                final_result = np.concatenate(results)

            logger.info("Chunked processing completed")
            return final_result

        return wrapper

    return decorator


@profile_performance
def vectorized_technical_indicators(
    df: pd.DataFrame, price_col: str = "close", volume_col: Optional[str] = None
) -> pd.DataFrame:
    """
    ベクトル化されたテクニカル指標の高速計算

    Args:
        df: 価格データのDataFrame
        price_col: 価格列名
        volume_col: 出来高列名（オプション）

    Returns:
        pd.DataFrame: テクニカル指標が追加されたDataFrame
    """
    result = df.copy()
    prices = result[price_col].values

    # Simple Moving Average (SMA) - ベクトル化版
    for period in [5, 10, 20, 50]:
        col_name = f"sma_{period}"
        result[col_name] = pd.Series(prices).rolling(window=period).mean()

    # Exponential Moving Average (EMA) - ベクトル化版
    for period in [12, 26]:
        col_name = f"ema_{period}"
        result[col_name] = pd.Series(prices).ewm(span=period).mean()

    # RSI (Relative Strength Index) - 高速化版
    delta = np.diff(prices, prepend=prices[0])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    # ベクトル化された指数移動平均
    alpha = 1.0 / 14
    avg_gains = pd.Series(gains).ewm(alpha=alpha).mean()
    avg_losses = pd.Series(losses).ewm(alpha=alpha).mean()

    rs = avg_gains / avg_losses
    result["rsi"] = 100 - (100 / (1 + rs))

    # MACD - ベクトル化版
    ema_12 = pd.Series(prices).ewm(span=12).mean()
    ema_26 = pd.Series(prices).ewm(span=26).mean()
    result["macd"] = ema_12 - ema_26
    result["macd_signal"] = result["macd"].ewm(span=9).mean()
    result["macd_histogram"] = result["macd"] - result["macd_signal"]

    # Bollinger Bands - ベクトル化版
    sma_20 = pd.Series(prices).rolling(window=20).mean()
    std_20 = pd.Series(prices).rolling(window=20).std()
    result["bb_upper"] = sma_20 + (std_20 * 2)
    result["bb_lower"] = sma_20 - (std_20 * 2)
    result["bb_middle"] = sma_20

    # Volume指標（出来高データがある場合）
    if volume_col and volume_col in result.columns:
        volumes = result[volume_col].values

        # Volume Moving Average
        result["volume_sma_20"] = pd.Series(volumes).rolling(window=20).mean()

        # On-Balance Volume (OBV)
        price_changes = np.diff(prices, prepend=prices[0])
        obv_changes = np.where(
            price_changes > 0, volumes, np.where(price_changes < 0, -volumes, 0)
        )
        result["obv"] = np.cumsum(obv_changes)

    logger.info(
        "Vectorized technical indicators calculated",
        input_rows=len(df),
        indicators_added=len([col for col in result.columns if col not in df.columns]),
    )

    return result


def parallel_group_operations(
    df: pd.DataFrame,
    group_col: str,
    operation_func: Callable,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    グループ操作の並列処理実行

    Args:
        df: 処理対象のDataFrame
        group_col: グループ化する列名
        operation_func: 各グループに適用する関数
        max_workers: 最大ワーカー数

    Returns:
        pd.DataFrame: 処理結果
    """
    config = get_performance_config()
    actual_max_workers = max_workers or config.compute.max_workers

    # グループを分割
    groups = [group for _, group in df.groupby(group_col)]

    logger.info(
        "Starting parallel group operations",
        total_groups=len(groups),
        max_workers=actual_max_workers,
    )

    if len(groups) <= 1 or actual_max_workers <= 1:
        # 並列化の意味がない場合は通常処理
        return df.groupby(group_col).apply(operation_func).reset_index(drop=True)

    # 並列処理実行
    if config.compute.use_multiprocessing and len(groups) > 10:
        # 大量のグループがある場合はプロセス並列
        with ProcessPoolExecutor(max_workers=actual_max_workers) as executor:
            results = list(executor.map(operation_func, groups))
    else:
        # スレッド並列で軽量処理
        with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
            results = list(executor.map(operation_func, groups))

    # 結果をマージ
    final_result = pd.concat(results, ignore_index=True)

    logger.info("Parallel group operations completed", processed_groups=len(results))

    return final_result


@chunked_processing()
def memory_efficient_calculation(
    df: pd.DataFrame,
    calculation_func: Callable,
    result_dtype: Optional[np.dtype] = None,
) -> pd.Series:
    """
    メモリ効率的な大量データ計算

    Args:
        df: 計算対象のDataFrame
        calculation_func: 計算関数
        result_dtype: 結果のデータ型

    Returns:
        pd.Series: 計算結果
    """
    result = calculation_func(df)

    if result_dtype and hasattr(result, "astype"):
        result = result.astype(result_dtype)

    return result


class OptimizedDataFrameProcessor:
    """最適化されたDataFrame処理クラス"""

    def __init__(self):
        self.config = get_performance_config()
        self.logger = logger

    @profile_performance
    def optimize_for_computation(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算処理に最適化されたDataFrameを作成"""

        # 1. データ型最適化
        optimized_df = optimize_dataframe_dtypes(df, aggressive=False)

        # 2. インデックス最適化
        if not optimized_df.index.is_monotonic_increasing:
            optimized_df = optimized_df.sort_index()

        # 3. 欠損値の効率的な処理
        if optimized_df.isnull().any().any():
            # 数値列: 0で埋める
            numeric_cols = optimized_df.select_dtypes(include=[np.number]).columns
            optimized_df[numeric_cols] = optimized_df[numeric_cols].fillna(0)

            # カテゴリ列: モードで埋める
            categorical_cols = optimized_df.select_dtypes(
                include=["object", "category"]
            ).columns
            for col in categorical_cols:
                mode_value = optimized_df[col].mode()
                if len(mode_value) > 0:
                    optimized_df[col] = optimized_df[col].fillna(mode_value[0])

        self.logger.info(
            "DataFrame optimized for computation",
            original_shape=df.shape,
            optimized_shape=optimized_df.shape,
        )

        return optimized_df

    def batch_statistical_analysis(
        self, df: pd.DataFrame, group_cols: List[str], value_cols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        バッチ統計分析の高速実行

        Args:
            df: 分析対象のDataFrame
            group_cols: グループ化する列のリスト
            value_cols: 分析対象の値列のリスト

        Returns:
            Dict[str, pd.DataFrame]: 統計分析結果
        """
        results = {}

        # 基本統計量（ベクトル化）
        numeric_df = df[value_cols].select_dtypes(include=[np.number])
        results["basic_stats"] = numeric_df.describe()

        # グループ別統計量（並列処理）
        for group_col in group_cols:
            if group_col in df.columns:
                group_stats = df.groupby(group_col)[value_cols].agg(
                    ["mean", "median", "std", "min", "max", "count"]
                )
                results[f"group_stats_{group_col}"] = group_stats

        # 相関行列（効率的な計算）
        if len(value_cols) > 1:
            correlation_matrix = numeric_df.corr(method="pearson")
            results["correlation"] = correlation_matrix

        self.logger.info(
            "Batch statistical analysis completed",
            group_cols=group_cols,
            value_cols=value_cols,
            result_categories=list(results.keys()),
        )

        return results


# グローバルインスタンス
_processor = None


def get_optimized_processor() -> OptimizedDataFrameProcessor:
    """最適化されたDataFrame処理器のインスタンスを取得"""
    global _processor
    if _processor is None:
        _processor = OptimizedDataFrameProcessor()
    return _processor
