#!/usr/bin/env python3
"""
専門的テストアサーション
Specialized Test Assertions

Issue #760: 包括的テスト自動化と検証フレームワークの構築
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Callable
import pytest
from unittest.mock import Mock
import psutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

# ログ設定
logger = logging.getLogger(__name__)


@dataclass
class PerformanceThresholds:
    """パフォーマンス閾値設定"""
    max_execution_time_ms: float = 1000.0
    max_memory_usage_mb: float = 512.0
    min_throughput_per_sec: float = 100.0
    max_cpu_usage_percent: float = 80.0


class PerformanceAssertions:
    """パフォーマンステスト アサーション"""

    @staticmethod
    def assert_execution_time(
        func: Callable,
        max_time_ms: float,
        *args,
        **kwargs
    ) -> float:
        """実行時間アサーション"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = (time.perf_counter() - start_time) * 1000

        assert execution_time <= max_time_ms, (
            f"Execution time {execution_time:.2f}ms exceeds limit {max_time_ms}ms"
        )

        logger.info(f"Execution time: {execution_time:.2f}ms (limit: {max_time_ms}ms)")
        return execution_time

    @staticmethod
    async def assert_async_execution_time(
        async_func: Callable,
        max_time_ms: float,
        *args,
        **kwargs
    ) -> float:
        """非同期実行時間アサーション"""
        start_time = time.perf_counter()
        result = await async_func(*args, **kwargs)
        execution_time = (time.perf_counter() - start_time) * 1000

        assert execution_time <= max_time_ms, (
            f"Async execution time {execution_time:.2f}ms exceeds limit {max_time_ms}ms"
        )

        logger.info(f"Async execution time: {execution_time:.2f}ms (limit: {max_time_ms}ms)")
        return execution_time

    @staticmethod
    def assert_memory_usage(
        func: Callable,
        max_memory_mb: float,
        *args,
        **kwargs
    ) -> float:
        """メモリ使用量アサーション"""
        import tracemalloc

        tracemalloc.start()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        result = func(*args, **kwargs)

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024

        assert peak_mb <= max_memory_mb, (
            f"Peak memory usage {peak_mb:.2f}MB exceeds limit {max_memory_mb}MB"
        )

        logger.info(f"Memory usage: {peak_mb:.2f}MB (limit: {max_memory_mb}MB)")
        return peak_mb

    @staticmethod
    def assert_throughput(
        func: Callable,
        data_items: List[Any],
        min_throughput: float
    ) -> float:
        """スループットアサーション"""
        start_time = time.perf_counter()

        for item in data_items:
            func(item)

        total_time = time.perf_counter() - start_time
        throughput = len(data_items) / total_time

        assert throughput >= min_throughput, (
            f"Throughput {throughput:.2f} items/sec below minimum {min_throughput}"
        )

        logger.info(f"Throughput: {throughput:.2f} items/sec (minimum: {min_throughput})")
        return throughput

    @staticmethod
    def assert_cpu_usage(
        func: Callable,
        max_cpu_percent: float,
        duration_seconds: float = 1.0,
        *args,
        **kwargs
    ) -> float:
        """CPU使用率アサーション"""
        import threading
        import time

        cpu_measurements = []
        stop_monitoring = threading.Event()

        def monitor_cpu():
            while not stop_monitoring.is_set():
                cpu_measurements.append(psutil.cpu_percent(interval=0.1))

        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()

        try:
            result = func(*args, **kwargs)
            time.sleep(duration_seconds)
        finally:
            stop_monitoring.set()
            monitor_thread.join()

        if cpu_measurements:
            avg_cpu = np.mean(cpu_measurements)
            max_cpu = np.max(cpu_measurements)

            assert avg_cpu <= max_cpu_percent, (
                f"Average CPU usage {avg_cpu:.1f}% exceeds limit {max_cpu_percent}%"
            )

            logger.info(f"CPU usage: avg={avg_cpu:.1f}%, max={max_cpu:.1f}% (limit: {max_cpu_percent}%)")
            return avg_cpu
        else:
            return 0.0


class MLModelAssertions:
    """機械学習モデル アサーション"""

    @staticmethod
    def assert_model_accuracy(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        min_accuracy: float,
        task_type: str = "regression"
    ) -> float:
        """モデル精度アサーション"""
        predictions = model.predict(X_test)

        if task_type == "regression":
            from sklearn.metrics import r2_score, mean_squared_error
            accuracy = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)

            assert accuracy >= min_accuracy, (
                f"Model R² score {accuracy:.3f} below minimum {min_accuracy}"
            )

            logger.info(f"Model performance: R²={accuracy:.3f}, MSE={mse:.3f}")

        elif task_type == "classification":
            from sklearn.metrics import accuracy_score, classification_report
            accuracy = accuracy_score(y_test, predictions)

            assert accuracy >= min_accuracy, (
                f"Model accuracy {accuracy:.3f} below minimum {min_accuracy}"
            )

            logger.info(f"Classification accuracy: {accuracy:.3f}")

        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        return accuracy

    @staticmethod
    def assert_prediction_consistency(
        model: Any,
        X_test: np.ndarray,
        tolerance: float = 1e-6,
        runs: int = 5
    ) -> float:
        """予測一貫性アサーション"""
        predictions_list = []

        for _ in range(runs):
            predictions = model.predict(X_test)
            predictions_list.append(predictions)

        # 予測値の分散計算
        predictions_array = np.array(predictions_list)
        variance = np.var(predictions_array, axis=0)
        max_variance = np.max(variance)

        assert max_variance <= tolerance, (
            f"Prediction variance {max_variance:.8f} exceeds tolerance {tolerance}"
        )

        logger.info(f"Prediction consistency: max_variance={max_variance:.8f}")
        return max_variance

    @staticmethod
    def assert_model_convergence(
        training_history: Dict[str, List[float]],
        metric_name: str = "loss",
        min_improvement: float = 0.01,
        patience: int = 10
    ) -> bool:
        """モデル収束アサーション"""
        if metric_name not in training_history:
            raise ValueError(f"Metric '{metric_name}' not found in training history")

        metrics = training_history[metric_name]

        if len(metrics) < patience + 1:
            pytest.skip(f"Insufficient training epochs for convergence check")

        # 最後のpatience期間での改善チェック
        recent_metrics = metrics[-patience:]
        best_recent = min(recent_metrics)
        best_overall = min(metrics[:-patience]) if len(metrics) > patience else float('inf')

        improvement = (best_overall - best_recent) / best_overall if best_overall != 0 else 0

        assert improvement >= min_improvement, (
            f"Model convergence insufficient: {improvement:.3f} < {min_improvement}"
        )

        logger.info(f"Model convergence: {improvement:.3f} improvement")
        return True

    @staticmethod
    def assert_no_overfitting(
        train_scores: List[float],
        val_scores: List[float],
        max_gap: float = 0.1
    ) -> float:
        """過学習チェックアサーション"""
        if len(train_scores) != len(val_scores):
            raise ValueError("Train and validation scores must have same length")

        # 最後のスコアでギャップ計算
        train_score = train_scores[-1]
        val_score = val_scores[-1]

        gap = abs(train_score - val_score)

        assert gap <= max_gap, (
            f"Overfitting detected: gap {gap:.3f} exceeds limit {max_gap}"
        )

        logger.info(f"Overfitting check: gap={gap:.3f} (limit: {max_gap})")
        return gap


class DataQualityAssertions:
    """データ品質 アサーション"""

    @staticmethod
    def assert_no_missing_values(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """欠損値なしアサーション"""
        check_columns = columns or df.columns

        for col in check_columns:
            missing_count = df[col].isnull().sum()
            assert missing_count == 0, (
                f"Column '{col}' has {missing_count} missing values"
            )

        logger.info(f"No missing values found in {len(check_columns)} columns")

    @staticmethod
    def assert_data_range(
        df: pd.DataFrame,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> None:
        """データ範囲アサーション"""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        col_data = df[column]

        if min_value is not None:
            actual_min = col_data.min()
            assert actual_min >= min_value, (
                f"Column '{column}' minimum value {actual_min} below limit {min_value}"
            )

        if max_value is not None:
            actual_max = col_data.max()
            assert actual_max <= max_value, (
                f"Column '{column}' maximum value {actual_max} above limit {max_value}"
            )

        logger.info(f"Data range check passed for column '{column}'")

    @staticmethod
    def assert_data_uniqueness(
        df: pd.DataFrame,
        columns: List[str],
        min_unique_ratio: float = 0.95
    ) -> float:
        """データ一意性アサーション"""
        total_rows = len(df)

        for col in columns:
            unique_count = df[col].nunique()
            unique_ratio = unique_count / total_rows

            assert unique_ratio >= min_unique_ratio, (
                f"Column '{col}' uniqueness ratio {unique_ratio:.3f} below minimum {min_unique_ratio}"
            )

            logger.info(f"Column '{col}' uniqueness: {unique_ratio:.3f}")

        return unique_ratio

    @staticmethod
    def assert_data_distribution(
        df: pd.DataFrame,
        column: str,
        expected_mean: Optional[float] = None,
        expected_std: Optional[float] = None,
        tolerance: float = 0.1
    ) -> Dict[str, float]:
        """データ分布アサーション"""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        col_data = df[column].dropna()
        actual_mean = col_data.mean()
        actual_std = col_data.std()

        stats = {"mean": actual_mean, "std": actual_std}

        if expected_mean is not None:
            mean_diff = abs(actual_mean - expected_mean) / expected_mean
            assert mean_diff <= tolerance, (
                f"Column '{column}' mean {actual_mean:.3f} deviates from expected {expected_mean:.3f} by {mean_diff:.3f}"
            )

        if expected_std is not None:
            std_diff = abs(actual_std - expected_std) / expected_std
            assert std_diff <= tolerance, (
                f"Column '{column}' std {actual_std:.3f} deviates from expected {expected_std:.3f} by {std_diff:.3f}"
            )

        logger.info(f"Data distribution check passed for column '{column}': mean={actual_mean:.3f}, std={actual_std:.3f}")
        return stats


class IntegrationAssertions:
    """統合テスト アサーション"""

    @staticmethod
    def assert_api_response_time(
        api_client: Any,
        endpoint: str,
        max_response_time_ms: float,
        **request_kwargs
    ) -> float:
        """API応答時間アサーション"""
        start_time = time.perf_counter()

        response = api_client.get(endpoint, **request_kwargs)

        response_time = (time.perf_counter() - start_time) * 1000

        assert response_time <= max_response_time_ms, (
            f"API response time {response_time:.2f}ms exceeds limit {max_response_time_ms}ms"
        )

        logger.info(f"API response time: {response_time:.2f}ms (limit: {max_response_time_ms}ms)")
        return response_time

    @staticmethod
    def assert_database_connection(db_client: Any, timeout_seconds: float = 5.0) -> None:
        """データベース接続アサーション"""
        start_time = time.perf_counter()

        try:
            # 簡単なクエリでテスト
            if hasattr(db_client, 'execute'):
                db_client.execute("SELECT 1")
            elif hasattr(db_client, 'ping'):
                db_client.ping()
            else:
                # フォールバック: __str__ メソッド呼び出し
                str(db_client)

            connection_time = time.perf_counter() - start_time

            assert connection_time <= timeout_seconds, (
                f"Database connection time {connection_time:.2f}s exceeds timeout {timeout_seconds}s"
            )

            logger.info(f"Database connection successful in {connection_time:.2f}s")

        except Exception as e:
            pytest.fail(f"Database connection failed: {e}")

    @staticmethod
    def assert_cache_performance(
        cache_client: Any,
        key: str,
        value: Any,
        max_set_time_ms: float = 10.0,
        max_get_time_ms: float = 5.0
    ) -> Dict[str, float]:
        """キャッシュ性能アサーション"""
        # SET 性能テスト
        start_time = time.perf_counter()
        cache_client.set(key, value)
        set_time = (time.perf_counter() - start_time) * 1000

        assert set_time <= max_set_time_ms, (
            f"Cache SET time {set_time:.2f}ms exceeds limit {max_set_time_ms}ms"
        )

        # GET 性能テスト
        start_time = time.perf_counter()
        retrieved_value = cache_client.get(key)
        get_time = (time.perf_counter() - start_time) * 1000

        assert get_time <= max_get_time_ms, (
            f"Cache GET time {get_time:.2f}ms exceeds limit {max_get_time_ms}ms"
        )

        # 値の一致確認
        assert retrieved_value == value, "Cache value mismatch"

        logger.info(f"Cache performance: SET={set_time:.2f}ms, GET={get_time:.2f}ms")
        return {"set_time_ms": set_time, "get_time_ms": get_time}

    @staticmethod
    def assert_system_resources(
        max_cpu_percent: float = 80.0,
        max_memory_percent: float = 80.0,
        min_disk_space_gb: float = 1.0
    ) -> Dict[str, float]:
        """システムリソースアサーション"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        assert cpu_percent <= max_cpu_percent, (
            f"CPU usage {cpu_percent:.1f}% exceeds limit {max_cpu_percent}%"
        )

        # メモリ使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        assert memory_percent <= max_memory_percent, (
            f"Memory usage {memory_percent:.1f}% exceeds limit {max_memory_percent}%"
        )

        # ディスク空き容量
        disk = psutil.disk_usage('/')
        free_space_gb = disk.free / (1024**3)
        assert free_space_gb >= min_disk_space_gb, (
            f"Free disk space {free_space_gb:.1f}GB below minimum {min_disk_space_gb}GB"
        )

        metrics = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "free_disk_gb": free_space_gb
        }

        logger.info(f"System resources: CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}%, Disk={free_space_gb:.1f}GB")
        return metrics


# カスタムアサーション関数群
def assert_inference_speed(
    inference_func: Callable,
    input_data: np.ndarray,
    max_time_ms: float = 100.0
) -> float:
    """推論速度アサーション（Issue #761対応）"""
    return PerformanceAssertions.assert_execution_time(
        inference_func, max_time_ms, input_data
    )


def assert_model_memory_efficiency(
    model_loader: Callable,
    max_memory_mb: float = 256.0
) -> float:
    """モデルメモリ効率アサーション（Issue #761対応）"""
    return PerformanceAssertions.assert_memory_usage(
        model_loader, max_memory_mb
    )


def assert_parallel_processing_speedup(
    serial_func: Callable,
    parallel_func: Callable,
    data: Any,
    min_speedup_ratio: float = 2.0
) -> float:
    """並列処理速度向上アサーション（Issue #761対応）"""
    # シリアル実行時間測定
    serial_time = PerformanceAssertions.assert_execution_time(
        serial_func, float('inf'), data
    )

    # 並列実行時間測定
    parallel_time = PerformanceAssertions.assert_execution_time(
        parallel_func, float('inf'), data
    )

    speedup_ratio = serial_time / parallel_time

    assert speedup_ratio >= min_speedup_ratio, (
        f"Parallel speedup {speedup_ratio:.2f}x below minimum {min_speedup_ratio}x"
    )

    logger.info(f"Parallel processing speedup: {speedup_ratio:.2f}x")
    return speedup_ratio


# 使用例とテスト
if __name__ == "__main__":
    # サンプルデータでテスト
    import asyncio

    def sample_function(x):
        time.sleep(0.01)  # 10ms sleep
        return x * 2

    # パフォーマンステスト例
    execution_time = PerformanceAssertions.assert_execution_time(
        sample_function, 50.0, 5
    )
    print(f"Execution time: {execution_time:.2f}ms")

    # データ品質テスト例
    df = pd.DataFrame({
        'price': np.random.uniform(100, 200, 1000),
        'volume': np.random.randint(1000, 10000, 1000)
    })

    DataQualityAssertions.assert_no_missing_values(df)
    DataQualityAssertions.assert_data_range(df, 'price', min_value=50, max_value=250)

    print("All assertions passed!")