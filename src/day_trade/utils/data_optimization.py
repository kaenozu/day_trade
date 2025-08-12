#!/usr/bin/env python3
"""
データI/O最適化ユーティリティ
Issue #378: データI/Oとデータ処理の最適化 - データ構造と操作の効率化

パフォーマンス最適化機能:
1. データ型の最適化でメモリ効率向上
2. DataFrame操作のベクトル化で高速化
3. 不必要なデータコピーの回避
4. インデックスの最適化
"""

import gc
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil

from .logging_config import get_context_logger

logger = get_context_logger(__name__)

# メモリ使用量監視の閾値（MB）
MEMORY_WARNING_THRESHOLD = 1000  # 1GB
MEMORY_CRITICAL_THRESHOLD = 2000  # 2GB


def memory_monitor(func):
    """メモリ使用量監視デコレータ"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 実行前のメモリ使用量
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)

            # 実行後のメモリ使用量
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = memory_after - memory_before
            elapsed_time = time.perf_counter() - start_time

            # ログ出力
            logger.debug(
                f"{func.__name__} 実行完了",
                extra={
                    "execution_time_ms": round(elapsed_time * 1000, 2),
                    "memory_before_mb": round(memory_before, 2),
                    "memory_after_mb": round(memory_after, 2),
                    "memory_diff_mb": round(memory_diff, 2),
                },
            )

            # メモリ警告
            if memory_after > MEMORY_CRITICAL_THRESHOLD:
                logger.warning(f"メモリ使用量が臨界値を超過: {memory_after:.2f}MB")
            elif memory_after > MEMORY_WARNING_THRESHOLD:
                logger.info(f"メモリ使用量警告: {memory_after:.2f}MB")

            return result

        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            logger.error(
                f"{func.__name__} 実行失敗: {e}",
                extra={
                    "execution_time_ms": round(elapsed_time * 1000, 2),
                    "memory_before_mb": round(memory_before, 2),
                },
            )
            raise

    return wrapper


class DataFrameOptimizer:
    """DataFrame最適化クラス"""

    def __init__(self):
        """初期化"""
        self.optimization_stats = {
            "memory_saved_mb": 0.0,
            "dtype_optimizations": 0,
            "copy_eliminations": 0,
            "vectorizations": 0,
        }

    @memory_monitor
    def optimize_dtypes(self, df: pd.DataFrame, copy: bool = False) -> pd.DataFrame:
        """
        データ型の最適化でメモリ使用量を削減

        Args:
            df: 最適化対象のDataFrame
            copy: データをコピーするかどうか

        Returns:
            pd.DataFrame: 最適化されたDataFrame
        """
        if copy:
            df = df.copy()

        memory_before = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        optimizations_applied = 0

        for col in df.columns:
            original_dtype = df[col].dtype

            if original_dtype == "object":
                # 文字列型の最適化
                if self._is_categorical_candidate(df[col]):
                    df[col] = df[col].astype("category")
                    optimizations_applied += 1
                    logger.debug(f"列 '{col}' を category 型に変換")

            elif pd.api.types.is_numeric_dtype(df[col]):
                # 数値型の最適化
                optimized_dtype = self._get_optimal_numeric_dtype(df[col])
                if optimized_dtype != original_dtype:
                    try:
                        df[col] = df[col].astype(optimized_dtype)
                        optimizations_applied += 1
                        logger.debug(
                            f"列 '{col}' を {original_dtype} から {optimized_dtype} に変換"
                        )
                    except (ValueError, OverflowError):
                        logger.warning(
                            f"列 '{col}' の型変換に失敗: {original_dtype} -> {optimized_dtype}"
                        )

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # 日時型の最適化
                if df[col].dt.tz is None and not df[col].isna().any():
                    # タイムゾーンなしで欠損値がない場合はより効率的な型を使用
                    try:
                        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S")
                        optimizations_applied += 1
                        logger.debug(f"列 '{col}' の日時型を最適化")
                    except:
                        pass

        memory_after = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        memory_saved = memory_before - memory_after

        self.optimization_stats["memory_saved_mb"] += memory_saved
        self.optimization_stats["dtype_optimizations"] += optimizations_applied

        logger.info(
            "データ型最適化完了",
            extra={
                "memory_before_mb": round(memory_before, 2),
                "memory_after_mb": round(memory_after, 2),
                "memory_saved_mb": round(memory_saved, 2),
                "memory_reduction_percent": round((memory_saved / memory_before) * 100, 1),
                "optimizations_applied": optimizations_applied,
            },
        )

        return df

    def _is_categorical_candidate(self, series: pd.Series) -> bool:
        """カテゴリ型候補かどうかを判定"""
        if series.dtype != "object":
            return False

        unique_count = series.nunique()
        total_count = len(series)

        # ユニーク値の比率が30%未満の場合はカテゴリ型候補
        return unique_count / total_count < 0.3 and unique_count < 1000

    def _get_optimal_numeric_dtype(self, series: pd.Series) -> str:
        """最適な数値データ型を取得"""
        if pd.api.types.is_integer_dtype(series):
            # 整数型の最適化
            min_val = series.min()
            max_val = series.max()

            if min_val >= 0:  # 非負整数
                if max_val < 256:
                    return "uint8"
                elif max_val < 65536:
                    return "uint16"
                elif max_val < 4294967296:
                    return "uint32"
                else:
                    return "uint64"
            else:  # 符号付き整数
                if min_val >= -128 and max_val < 128:
                    return "int8"
                elif min_val >= -32768 and max_val < 32768:
                    return "int16"
                elif min_val >= -2147483648 and max_val < 2147483648:
                    return "int32"
                else:
                    return "int64"

        elif pd.api.types.is_float_dtype(series):
            # 浮動小数点型の最適化
            # NaN値の有無をチェック
            if series.isna().any():
                # NaN値がある場合は精度を保持
                if series.dtype == "float64":
                    # 値の範囲をチェックしてfloat32で表現可能かどうか
                    try:
                        # float32の範囲に収まるかチェック
                        series_copy = series.dropna()
                        if (series_copy.abs() <= np.finfo(np.float32).max).all():
                            return "float32"
                    except:
                        pass
                return "float64"
            else:
                # NaN値がない場合はfloat32を試す
                return "float32"

        return str(series.dtype)

    @memory_monitor
    def vectorize_operations(
        self, df: pd.DataFrame, operations: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        DataFrame操作のベクトル化

        Args:
            df: 対象のDataFrame
            operations: 適用する操作のリスト

        Returns:
            pd.DataFrame: 処理後のDataFrame
        """
        optimized_df = df.copy()
        vectorizations_applied = 0

        for operation in operations:
            op_type = operation.get("type")

            if op_type == "technical_indicator":
                # テクニカル指標の計算
                result = self._calculate_technical_indicator_vectorized(optimized_df, operation)
                if result is not None:
                    optimized_df = result
                    vectorizations_applied += 1

            elif op_type == "rolling_calculation":
                # ローリング計算
                result = self._apply_rolling_calculation_vectorized(optimized_df, operation)
                if result is not None:
                    optimized_df = result
                    vectorizations_applied += 1

            elif op_type == "mathematical_operation":
                # 数学的操作
                result = self._apply_mathematical_operation_vectorized(optimized_df, operation)
                if result is not None:
                    optimized_df = result
                    vectorizations_applied += 1

        self.optimization_stats["vectorizations"] += vectorizations_applied

        logger.info(
            "ベクトル化操作完了",
            extra={
                "operations_applied": vectorizations_applied,
                "total_operations": len(operations),
            },
        )

        return optimized_df

    def _calculate_technical_indicator_vectorized(
        self, df: pd.DataFrame, operation: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """ベクトル化されたテクニカル指標計算"""
        indicator_type = operation.get("indicator")
        column = operation.get("column", "Close")
        period = operation.get("period", 20)

        if column not in df.columns:
            logger.warning(f"列 '{column}' が存在しません")
            return None

        try:
            if indicator_type == "sma":
                # 単純移動平均
                df[f"SMA_{period}"] = df[column].rolling(window=period, min_periods=1).mean()

            elif indicator_type == "ema":
                # 指数移動平均
                df[f"EMA_{period}"] = df[column].ewm(span=period, adjust=False).mean()

            elif indicator_type == "bollinger_bands":
                # ボリンジャーバンド
                sma = df[column].rolling(window=period).mean()
                std = df[column].rolling(window=period).std()
                df[f"BB_Upper_{period}"] = sma + (std * 2)
                df[f"BB_Lower_{period}"] = sma - (std * 2)
                df[f"BB_Middle_{period}"] = sma

            elif indicator_type == "rsi":
                # RSI
                delta = df[column].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = (-delta).where(delta < 0, 0).rolling(window=period).mean()
                rs = gain / loss
                df[f"RSI_{period}"] = 100 - (100 / (1 + rs))

            elif indicator_type == "macd":
                # MACD
                fast_period = operation.get("fast_period", 12)
                slow_period = operation.get("slow_period", 26)
                signal_period = operation.get("signal_period", 9)

                ema_fast = df[column].ewm(span=fast_period).mean()
                ema_slow = df[column].ewm(span=slow_period).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=signal_period).mean()

                df["MACD"] = macd_line
                df["MACD_Signal"] = signal_line
                df["MACD_Histogram"] = macd_line - signal_line

            return df

        except Exception as e:
            logger.error(f"テクニカル指標計算エラー: {e}")
            return None

    def _apply_rolling_calculation_vectorized(
        self, df: pd.DataFrame, operation: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """ベクトル化されたローリング計算"""
        column = operation.get("column")
        window = operation.get("window", 20)
        calc_type = operation.get("calculation")

        if column not in df.columns:
            logger.warning(f"列 '{column}' が存在しません")
            return None

        try:
            rolling = df[column].rolling(window=window, min_periods=1)

            if calc_type == "mean":
                df[f"{column}_rolling_mean_{window}"] = rolling.mean()
            elif calc_type == "std":
                df[f"{column}_rolling_std_{window}"] = rolling.std()
            elif calc_type == "min":
                df[f"{column}_rolling_min_{window}"] = rolling.min()
            elif calc_type == "max":
                df[f"{column}_rolling_max_{window}"] = rolling.max()
            elif calc_type == "sum":
                df[f"{column}_rolling_sum_{window}"] = rolling.sum()
            elif calc_type == "volatility":
                # 価格変化率の標準偏差（ボラティリティ）
                returns = df[column].pct_change()
                df[f"{column}_volatility_{window}"] = returns.rolling(window=window).std()

            return df

        except Exception as e:
            logger.error(f"ローリング計算エラー: {e}")
            return None

    def _apply_mathematical_operation_vectorized(
        self, df: pd.DataFrame, operation: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """ベクトル化された数学的操作"""
        operation_type = operation.get("operation")
        columns = operation.get("columns", [])
        result_column = operation.get("result_column")

        if not all(col in df.columns for col in columns):
            missing_cols = [col for col in columns if col not in df.columns]
            logger.warning(f"列が存在しません: {missing_cols}")
            return None

        try:
            if operation_type == "ratio":
                # 比率計算
                if len(columns) == 2:
                    df[result_column] = df[columns[0]] / df[columns[1]].replace(0, np.nan)

            elif operation_type == "difference":
                # 差分計算
                if len(columns) == 2:
                    df[result_column] = df[columns[0]] - df[columns[1]]

            elif operation_type == "percentage_change":
                # 変化率計算
                if len(columns) == 1:
                    df[result_column] = df[columns[0]].pct_change()

            elif operation_type == "log_return":
                # 対数収益率
                if len(columns) == 1:
                    df[result_column] = np.log(df[columns[0]] / df[columns[0]].shift(1))

            elif operation_type == "z_score":
                # Zスコア正規化
                if len(columns) == 1:
                    col = columns[0]
                    window = operation.get("window", 20)
                    rolling_mean = df[col].rolling(window=window).mean()
                    rolling_std = df[col].rolling(window=window).std()
                    df[result_column] = (df[col] - rolling_mean) / rolling_std

            return df

        except Exception as e:
            logger.error(f"数学的操作エラー: {e}")
            return None

    @memory_monitor
    def eliminate_unnecessary_copies(self, df: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        """
        不必要なデータコピーの回避

        Args:
            df: 対象のDataFrame
            operations: 実行する操作のリスト

        Returns:
            pd.DataFrame: 最適化されたDataFrame
        """
        # ビューを使用した効率的な操作
        optimized_df = df  # コピーを避けてビューを使用
        copy_eliminations = 0

        for operation in operations:
            if operation == "drop_duplicates":
                # 重複行削除（インプレース操作）
                if optimized_df.duplicated().any():
                    optimized_df = optimized_df.drop_duplicates(keep="first")
                    copy_eliminations += 1

            elif operation == "fillna":
                # 欠損値処理（効率的な方法）
                if optimized_df.isna().any().any():
                    # 数値列は前方穴埋め、その他は特定値で穴埋め
                    numeric_cols = optimized_df.select_dtypes(include=[np.number]).columns
                    categorical_cols = optimized_df.select_dtypes(
                        include=["category", "object"]
                    ).columns

                    if len(numeric_cols) > 0:
                        optimized_df[numeric_cols] = optimized_df[numeric_cols].fillna(
                            method="ffill"
                        )
                    if len(categorical_cols) > 0:
                        optimized_df[categorical_cols] = optimized_df[categorical_cols].fillna(
                            "Unknown"
                        )
                    copy_eliminations += 1

            elif operation == "sort_values":
                # 効率的なソート
                if "timestamp" in optimized_df.columns:
                    optimized_df = optimized_df.sort_values("timestamp")
                    copy_eliminations += 1

        self.optimization_stats["copy_eliminations"] += copy_eliminations

        logger.info("コピー最適化完了", extra={"copy_eliminations": copy_eliminations})

        return optimized_df

    @memory_monitor
    def optimize_index(
        self, df: pd.DataFrame, index_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        インデックスの最適化

        Args:
            df: 対象のDataFrame
            index_columns: インデックスに設定する列のリスト

        Returns:
            pd.DataFrame: インデックス最適化されたDataFrame
        """
        if index_columns is None:
            # デフォルトのインデックス最適化
            if "timestamp" in df.columns:
                index_columns = ["timestamp"]
            elif "date" in df.columns:
                index_columns = ["date"]
            else:
                logger.info("インデックス最適化用の列が見つかりません")
                return df

        try:
            # インデックス設定
            optimized_df = df.set_index(index_columns)

            # インデックスのソート（検索性能向上）
            if not optimized_df.index.is_monotonic_increasing:
                optimized_df = optimized_df.sort_index()

            logger.info(
                "インデックス最適化完了",
                extra={
                    "index_columns": index_columns,
                    "is_sorted": optimized_df.index.is_monotonic_increasing,
                },
            )

            return optimized_df

        except Exception as e:
            logger.error(f"インデックス最適化エラー: {e}")
            return df

    def get_optimization_stats(self) -> Dict[str, Any]:
        """最適化統計情報を取得"""
        return self.optimization_stats.copy()

    def reset_stats(self):
        """統計情報をリセット"""
        self.optimization_stats = {
            "memory_saved_mb": 0.0,
            "dtype_optimizations": 0,
            "copy_eliminations": 0,
            "vectorizations": 0,
        }


class ChunkedDataProcessor:
    """チャンク処理による大規模データ最適化"""

    def __init__(self, chunk_size: int = 10000):
        """
        初期化

        Args:
            chunk_size: チャンクサイズ
        """
        self.chunk_size = chunk_size
        self.optimizer = DataFrameOptimizer()

    @memory_monitor
    def process_large_dataframe(self, df: pd.DataFrame, processing_func, **kwargs) -> pd.DataFrame:
        """
        大規模DataFrameのチャンク処理

        Args:
            df: 処理対象のDataFrame
            processing_func: 各チャンクに適用する処理関数
            **kwargs: 処理関数に渡す追加引数

        Returns:
            pd.DataFrame: 処理後のDataFrame
        """
        if len(df) <= self.chunk_size:
            # チャンク処理が不要な場合
            return processing_func(df, **kwargs)

        logger.info(
            "チャンク処理開始",
            extra={
                "total_rows": len(df),
                "chunk_size": self.chunk_size,
                "num_chunks": len(df) // self.chunk_size + 1,
            },
        )

        processed_chunks = []

        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i : i + self.chunk_size]

            try:
                # ガベージコレクション（メモリ効率化）
                if i > 0 and i % (self.chunk_size * 10) == 0:
                    gc.collect()

                processed_chunk = processing_func(chunk, **kwargs)
                processed_chunks.append(processed_chunk)

                logger.debug(
                    f"チャンク処理完了: {i//self.chunk_size + 1}/{len(df)//self.chunk_size + 1}"
                )

            except Exception as e:
                logger.error(f"チャンク処理エラー (行 {i}-{i+len(chunk)}): {e}")
                # エラーが発生したチャンクは元のデータを使用
                processed_chunks.append(chunk)

        # 結果の結合
        result = pd.concat(processed_chunks, ignore_index=True)

        # メモリクリーンアップ
        del processed_chunks
        gc.collect()

        logger.info("チャンク処理完了", extra={"result_rows": len(result)})

        return result


def create_optimized_dataframe(data: Dict[str, Any], optimize_dtypes: bool = True) -> pd.DataFrame:
    """
    最適化されたDataFrameを作成

    Args:
        data: DataFrameに変換するデータ
        optimize_dtypes: データ型の最適化を行うかどうか

    Returns:
        pd.DataFrame: 最適化されたDataFrame
    """
    df = pd.DataFrame(data)

    if optimize_dtypes:
        optimizer = DataFrameOptimizer()
        df = optimizer.optimize_dtypes(df)

    return df


def benchmark_dataframe_operations(
    df: pd.DataFrame, operations: List[Dict[str, Any]], iterations: int = 3
) -> Dict[str, Any]:
    """
    DataFrame操作のベンチマーク

    Args:
        df: ベンチマーク対象のDataFrame
        operations: 実行する操作のリスト
        iterations: 実行回数

    Returns:
        Dict[str, Any]: ベンチマーク結果
    """
    optimizer = DataFrameOptimizer()
    results = {
        "original_memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "operation_results": [],
    }

    for operation in operations:
        operation_name = operation.get("name", "unknown")
        operation_times = []

        for i in range(iterations):
            start_time = time.perf_counter()

            try:
                if operation.get("type") == "dtype_optimization":
                    _ = optimizer.optimize_dtypes(df.copy())
                elif operation.get("type") == "vectorization":
                    _ = optimizer.vectorize_operations(df.copy(), operation.get("operations", []))

                elapsed_time = time.perf_counter() - start_time
                operation_times.append(elapsed_time * 1000)  # ms

            except Exception as e:
                logger.error(f"ベンチマーク操作エラー ({operation_name}): {e}")
                operation_times.append(float("inf"))

        results["operation_results"].append(
            {
                "name": operation_name,
                "avg_time_ms": np.mean(operation_times),
                "min_time_ms": np.min(operation_times),
                "max_time_ms": np.max(operation_times),
                "std_time_ms": np.std(operation_times),
            }
        )

    return results
