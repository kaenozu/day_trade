#!/usr/bin/env python3
"""
パフォーマンス最適化ユーティリティ

Issue #165: アプリケーション全体の処理速度向上に向けた最適化
このモジュールは、アプリケーションのパフォーマンスボトルネックを特定し、
最適化を実行するためのユーティリティを提供します。
"""

import asyncio
import cProfile
import io
import pstats
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil


@dataclass
class PerformanceMetrics:
    """パフォーマンス測定結果"""

    function_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: float
    data_size: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None


class PerformanceProfiler:
    """パフォーマンスプロファイラー"""

    def __init__(self, enable_detailed_profiling: bool = False):
        self.enable_detailed = enable_detailed_profiling
        self.metrics: List[PerformanceMetrics] = []
        self.profiler: Optional[cProfile.Profile] = None

    def profile_function(self, func: Callable) -> Callable:
        """関数のパフォーマンスをプロファイルするデコレータ"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._measure_performance(func, *args, **kwargs)

        return wrapper

    def _measure_performance(self, func: Callable, *args, **kwargs) -> Any:
        """パフォーマンス測定の実行"""
        # システムリソース測定開始
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = psutil.cpu_percent()
        start_time = time.perf_counter()

        # 詳細プロファイリング開始
        if self.enable_detailed and self.profiler is None:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

        try:
            # 関数実行
            result = func(*args, **kwargs)

            # 測定終了
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            peak_memory = (
                process.memory_info().peak_wss / 1024 / 1024
                if hasattr(process.memory_info(), "peak_wss")
                else end_memory
            )
            end_cpu = psutil.cpu_percent()

            # メトリクス記録
            metrics = PerformanceMetrics(
                function_name=func.__name__,
                execution_time=end_time - start_time,
                memory_usage_mb=end_memory - start_memory,
                cpu_usage_percent=(start_cpu + end_cpu) / 2,
                peak_memory_mb=peak_memory,
                data_size=self._estimate_data_size(result),
            )

            self.metrics.append(metrics)
            return result

        finally:
            if self.enable_detailed and self.profiler:
                self.profiler.disable()

    def _estimate_data_size(self, data: Any) -> Optional[int]:
        """データサイズの推定"""
        if isinstance(data, pd.DataFrame):
            return data.memory_usage(deep=True).sum()
        elif isinstance(data, (list, tuple, dict)):
            return len(data)
        return None

    def get_profile_stats(self) -> str:
        """詳細プロファイル統計を取得"""
        if not self.profiler:
            return "詳細プロファイリングが有効化されていません"

        stats_buffer = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stats_buffer)
        stats.sort_stats("cumulative")
        stats.print_stats(20)  # 上位20関数
        return stats_buffer.getvalue()

    def get_summary_report(self) -> Dict[str, Any]:
        """サマリーレポートを取得"""
        if not self.metrics:
            return {"message": "測定データがありません"}

        total_time = sum(m.execution_time for m in self.metrics)
        avg_memory = sum(m.memory_usage_mb for m in self.metrics) / len(self.metrics)
        max_memory = max(m.peak_memory_mb for m in self.metrics)

        return {
            "total_functions_profiled": len(self.metrics),
            "total_execution_time": total_time,
            "average_memory_usage_mb": avg_memory,
            "peak_memory_usage_mb": max_memory,
            "slowest_functions": sorted(
                self.metrics, key=lambda x: x.execution_time, reverse=True
            )[:5],
            "memory_intensive_functions": sorted(
                self.metrics, key=lambda x: x.memory_usage_mb, reverse=True
            )[:5],
        }


class DataFetchOptimizer:
    """データ取得の最適化クラス"""

    def __init__(self, max_workers: int = 4, chunk_size: int = 50):
        self.max_workers = max_workers
        self.chunk_size = chunk_size

    async def fetch_multiple_async(
        self, symbols: List[str], fetch_func: Callable, **kwargs
    ) -> Dict[str, Any]:
        """複数銘柄の非同期並列取得"""

        async def fetch_symbol(symbol: str) -> Tuple[str, Any]:
            try:
                # 同期関数を非同期で実行
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: fetch_func(symbol, **kwargs)
                )
                return symbol, result
            except Exception as e:
                return symbol, e

        # 並列実行
        tasks = [fetch_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 結果整理
        success_results = {}
        errors = {}

        for symbol, result in results:
            if isinstance(result, Exception):
                errors[symbol] = result
            else:
                success_results[symbol] = result

        return {
            "success": success_results,
            "errors": errors,
            "success_count": len(success_results),
            "error_count": len(errors),
        }

    def fetch_multiple_threaded(
        self, symbols: List[str], fetch_func: Callable, **kwargs
    ) -> Dict[str, Any]:
        """複数銘柄のスレッド並列取得"""
        success_results = {}
        errors = {}

        # チャンク分割
        chunks = [
            symbols[i : i + self.chunk_size]
            for i in range(0, len(symbols), self.chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # チャンク単位で並列実行
            for chunk in chunks:
                future_to_symbol = {
                    executor.submit(fetch_func, symbol, **kwargs): symbol
                    for symbol in chunk
                }

                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        success_results[symbol] = result
                    except Exception as e:
                        errors[symbol] = e

        return {
            "success": success_results,
            "errors": errors,
            "success_count": len(success_results),
            "error_count": len(errors),
        }

    def optimize_bulk_request(
        self,
        symbols: List[str],
        bulk_fetch_func: Callable,
        single_fetch_func: Callable,
        bulk_threshold: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """一括取得と個別取得の最適な組み合わせ"""
        if len(symbols) >= bulk_threshold:
            try:
                # 一括取得を試行
                return bulk_fetch_func(symbols, **kwargs)
            except Exception as e:
                warnings.warn(
                    f"一括取得に失敗、個別取得にフォールバック: {e}", stacklevel=2
                )
                # 個別取得にフォールバック
                return self.fetch_multiple_threaded(
                    symbols, single_fetch_func, **kwargs
                )
        else:
            # 少数の場合は個別取得
            return self.fetch_multiple_threaded(symbols, single_fetch_func, **kwargs)


class DatabaseOptimizer:
    """データベース操作の最適化クラス"""

    @staticmethod
    def optimize_bulk_insert(data: List[Dict], session, model_class) -> Dict[str, Any]:
        """一括挿入の最適化"""
        start_time = time.perf_counter()

        try:
            # SQLAlchemyの一括挿入を使用
            session.bulk_insert_mappings(model_class, data)
            session.commit()

            execution_time = time.perf_counter() - start_time
            return {
                "success": True,
                "inserted_count": len(data),
                "execution_time": execution_time,
                "records_per_second": len(data) / execution_time,
            }
        except Exception as e:
            session.rollback()
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.perf_counter() - start_time,
            }

    @staticmethod
    def optimize_bulk_update(data: List[Dict], session, model_class) -> Dict[str, Any]:
        """一括更新の最適化"""
        start_time = time.perf_counter()

        try:
            # SQLAlchemyの一括更新を使用
            session.bulk_update_mappings(model_class, data)
            session.commit()

            execution_time = time.perf_counter() - start_time
            return {
                "success": True,
                "updated_count": len(data),
                "execution_time": execution_time,
                "records_per_second": len(data) / execution_time,
            }
        except Exception as e:
            session.rollback()
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.perf_counter() - start_time,
            }


class CalculationOptimizer:
    """計算処理の最適化クラス"""

    @staticmethod
    def vectorize_technical_indicators(
        data: pd.DataFrame, indicators: List[str], **params
    ) -> pd.DataFrame:
        """テクニカル指標計算のベクトル化"""
        result = data.copy()

        # 並列でベクトル化計算
        for indicator in indicators:
            if indicator == "sma":
                period = params.get("sma_period", 20)
                result[f"sma_{period}"] = data["close"].rolling(window=period).mean()

            elif indicator == "ema":
                period = params.get("ema_period", 12)
                result[f"ema_{period}"] = data["close"].ewm(span=period).mean()

            elif indicator == "rsi":
                period = params.get("rsi_period", 14)
                delta = data["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                result[f"rsi_{period}"] = 100 - (100 / (1 + rs))

            elif indicator == "bollinger":
                period = params.get("bb_period", 20)
                std_dev = params.get("bb_std", 2)
                sma = data["close"].rolling(window=period).mean()
                std = data["close"].rolling(window=period).std()
                result[f"bb_upper_{period}"] = sma + (std * std_dev)
                result[f"bb_lower_{period}"] = sma - (std * std_dev)
                result[f"bb_middle_{period}"] = sma

        return result

    @staticmethod
    def optimize_backtest_calculation(
        data: pd.DataFrame,
        strategy_func: Callable,
        initial_capital: float = 100000,
        chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """バックテスト計算の最適化"""
        start_time = time.perf_counter()

        # データをチャンク分割して処理
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

        portfolio_value = initial_capital
        positions = {}
        trades = []

        for chunk in chunks:
            # チャンク単位でベクトル化処理
            signals = strategy_func(chunk)

            # 取引処理（ベクトル化）
            for idx, signal in signals.iterrows():
                if signal.get("action") == "buy":
                    # 買い処理
                    shares = portfolio_value * 0.1 / signal["price"]  # 10%投資
                    positions[signal["symbol"]] = (
                        positions.get(signal["symbol"], 0) + shares
                    )
                    portfolio_value -= shares * signal["price"]
                    trades.append(
                        {
                            "date": idx,
                            "symbol": signal["symbol"],
                            "action": "buy",
                            "shares": shares,
                            "price": signal["price"],
                        }
                    )
                elif signal.get("action") == "sell" and signal["symbol"] in positions:
                    # 売り処理
                    shares = positions[signal["symbol"]]
                    portfolio_value += shares * signal["price"]
                    del positions[signal["symbol"]]
                    trades.append(
                        {
                            "date": idx,
                            "symbol": signal["symbol"],
                            "action": "sell",
                            "shares": shares,
                            "price": signal["price"],
                        }
                    )

        execution_time = time.perf_counter() - start_time

        return {
            "final_portfolio_value": portfolio_value,
            "total_trades": len(trades),
            "execution_time": execution_time,
            "trades_per_second": len(trades) / execution_time
            if execution_time > 0
            else 0,
            "return_percentage": (portfolio_value - initial_capital)
            / initial_capital
            * 100,
        }


@contextmanager
def performance_monitor(operation_name: str):
    """パフォーマンス監視コンテキストマネージャー"""
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024

    print(f">> {operation_name} 開始")

    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory

        print(f"<< {operation_name} 完了")
        print(f"   実行時間: {execution_time:.3f}秒")
        print(f"   メモリ使用量変化: {memory_delta:+.2f}MB")


# パフォーマンステスト用のサンプル関数
def create_sample_data(rows: int = 10000) -> pd.DataFrame:
    """テスト用のサンプルデータ作成"""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=rows, freq="D")

    return pd.DataFrame(
        {
            "date": dates,
            "open": np.random.randn(rows).cumsum() + 100,
            "high": np.random.randn(rows).cumsum() + 105,
            "low": np.random.randn(rows).cumsum() + 95,
            "close": np.random.randn(rows).cumsum() + 100,
            "volume": np.random.randint(1000000, 10000000, rows),
        }
    )


if __name__ == "__main__":
    # パフォーマンステストの実行例
    print(">> パフォーマンス最適化ツール - テスト実行")

    # プロファイラーの初期化
    profiler = PerformanceProfiler(enable_detailed_profiling=True)

    # テストデータ作成
    with performance_monitor("サンプルデータ作成"):
        test_data = create_sample_data(10000)

    # 計算最適化テスト
    optimizer = CalculationOptimizer()

    @profiler.profile_function
    def test_technical_indicators():
        return optimizer.vectorize_technical_indicators(
            test_data, ["sma", "ema", "rsi", "bollinger"]
        )

    with performance_monitor("テクニカル指標計算"):
        result = test_technical_indicators()

    # 結果表示
    print("\n>> プロファイリング結果:")
    summary = profiler.get_summary_report()
    print(f"総実行時間: {summary['total_execution_time']:.3f}秒")
    print(f"平均メモリ使用量: {summary['average_memory_usage_mb']:.2f}MB")
    print(f"最大メモリ使用量: {summary['peak_memory_usage_mb']:.2f}MB")

    print("\n詳細プロファイル:")
    print(profiler.get_profile_stats())
