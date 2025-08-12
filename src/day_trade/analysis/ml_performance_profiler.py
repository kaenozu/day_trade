#!/usr/bin/env python3
"""
ML処理パフォーマンス・プロファイラー
Issue #325: ML処理のボトルネック詳細プロファイリング

MLコンポーネントの詳細パフォーマンス分析・ボトルネック特定ツール
"""

import cProfile
import io
import logging
import pstats
import threading
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import psutil

try:
    from ..analysis.advanced_technical_indicators import AdvancedTechnicalIndicators
    from ..data.advanced_ml_engine import AdvancedMLEngine
    from ..data.lstm_time_series_model import LSTMTimeSeriesModel
    from ..utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


@dataclass
class PerformanceMetrics:
    """パフォーマンス測定結果"""

    component_name: str
    method_name: str
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    function_calls: int
    data_size: int = 0
    error_occurred: bool = False
    error_message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ComponentBenchmark:
    """コンポーネントベンチマーク結果"""

    component_name: str
    total_execution_time: float
    total_memory_usage: float
    method_metrics: List[PerformanceMetrics] = field(default_factory=list)
    bottleneck_methods: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)


class ResourceMonitor:
    """リソース使用量監視"""

    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None

    def start_monitoring(self, interval: float = 0.1):
        """監視開始"""
        self.monitoring = True
        self.metrics = []

        def monitor_loop():
            while self.monitoring:
                try:
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                    cpu_percent = self.process.cpu_percent()

                    self.metrics.append(
                        {
                            "timestamp": time.time(),
                            "memory_mb": memory_mb,
                            "cpu_percent": cpu_percent,
                        }
                    )

                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"リソース監視エラー: {e}")
                    break

        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """監視停止と結果取得"""
        self.monitoring = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        if not self.metrics:
            return {
                "avg_memory_mb": 0,
                "peak_memory_mb": 0,
                "avg_cpu_percent": 0,
                "peak_cpu_percent": 0,
            }

        memory_values = [m["memory_mb"] for m in self.metrics]
        cpu_values = [m["cpu_percent"] for m in self.metrics]

        return {
            "avg_memory_mb": np.mean(memory_values),
            "peak_memory_mb": np.max(memory_values),
            "avg_cpu_percent": np.mean(cpu_values),
            "peak_cpu_percent": np.max(cpu_values),
        }


class MLPerformanceProfiler:
    """ML処理パフォーマンス・プロファイラー"""

    def __init__(self):
        """初期化"""
        self.resource_monitor = ResourceMonitor()
        self.profiling_results = {}
        self.benchmark_results = []

        # メモリ追跡開始
        tracemalloc.start()

        logger.info("ML性能プロファイラー初期化完了")

    @contextmanager
    def profile_method(self, component_name: str, method_name: str, data_size: int = 0):
        """メソッド実行プロファイリング コンテキストマネージャー"""
        # プロファイリング開始
        profiler = cProfile.Profile()

        # リソース監視開始
        self.resource_monitor.start_monitoring()

        # メモリ使用量記録
        initial_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        start_time = time.time()

        try:
            profiler.enable()
            yield
            profiler.disable()

            # 実行時間計算
            execution_time = time.time() - start_time

            # メモリ使用量取得
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            current_memory_mb = current_memory / 1024 / 1024
            peak_memory_mb = peak_memory / 1024 / 1024
            memory_usage = current_memory_mb - initial_memory

            # リソース監視結果取得
            resource_stats = self.resource_monitor.stop_monitoring()

            # プロファイリング統計取得
            stats_io = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_io)
            stats.sort_stats("cumulative")
            function_calls = stats.total_calls

            # 結果記録
            metrics = PerformanceMetrics(
                component_name=component_name,
                method_name=method_name,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                peak_memory_mb=peak_memory_mb,
                cpu_usage_percent=resource_stats["avg_cpu_percent"],
                function_calls=function_calls,
                data_size=data_size,
                error_occurred=False,
            )

            self._record_metrics(metrics)

            logger.info(
                f"プロファイリング完了: {component_name}.{method_name} "
                f"({execution_time:.2f}秒, {memory_usage:.1f}MB)"
            )

        except Exception as e:
            # エラー時の記録
            execution_time = time.time() - start_time
            self.resource_monitor.stop_monitoring()

            metrics = PerformanceMetrics(
                component_name=component_name,
                method_name=method_name,
                execution_time=execution_time,
                memory_usage_mb=0,
                peak_memory_mb=0,
                cpu_usage_percent=0,
                function_calls=0,
                data_size=data_size,
                error_occurred=True,
                error_message=str(e),
            )

            self._record_metrics(metrics)
            logger.error(f"プロファイリングエラー: {component_name}.{method_name}: {e}")
            raise

    def _record_metrics(self, metrics: PerformanceMetrics):
        """メトリクス記録"""
        component_name = metrics.component_name

        if component_name not in self.profiling_results:
            self.profiling_results[component_name] = []

        self.profiling_results[component_name].append(metrics)

    def benchmark_advanced_ml_engine(
        self, test_data: pd.DataFrame
    ) -> ComponentBenchmark:
        """AdvancedMLEngineのベンチマーク"""
        logger.info("AdvancedMLEngine ベンチマーク開始")

        component_name = "AdvancedMLEngine"
        benchmark_start = time.time()
        total_memory_start = tracemalloc.get_traced_memory()[0] / 1024 / 1024

        try:
            engine = AdvancedMLEngine(fast_mode=False)

            # 1. 特徴量準備のプロファイリング
            with self.profile_method(
                component_name, "prepare_features", len(test_data)
            ):
                features = engine.prepare_features(test_data)

            # 2. モデル訓練のプロファイリング
            if not features.empty:
                with self.profile_method(component_name, "train_models", len(features)):
                    results = engine.train_models(features, test_symbol="PROFILE_TEST")

            # 3. アンサンブル予測のプロファイリング
            if not features.empty:
                with self.profile_method(
                    component_name, "ensemble_predict", len(features)
                ):
                    predictions = engine.ensemble_predict(
                        features, test_symbol="PROFILE_TEST"
                    )

        except Exception as e:
            logger.error(f"AdvancedMLEngine ベンチマークエラー: {e}")

        # ベンチマーク結果作成
        total_time = time.time() - benchmark_start
        total_memory_end = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        total_memory_used = total_memory_end - total_memory_start

        benchmark = ComponentBenchmark(
            component_name=component_name,
            total_execution_time=total_time,
            total_memory_usage=total_memory_used,
            method_metrics=self.profiling_results.get(component_name, []),
        )

        # ボトルネック分析
        benchmark.bottleneck_methods = self._identify_bottlenecks(
            benchmark.method_metrics
        )
        benchmark.optimization_suggestions = self._generate_optimization_suggestions(
            benchmark
        )

        self.benchmark_results.append(benchmark)
        logger.info(f"AdvancedMLEngine ベンチマーク完了: {total_time:.2f}秒")

        return benchmark

    def benchmark_lstm_model(self, test_data: pd.DataFrame) -> ComponentBenchmark:
        """LSTMモデルのベンチマーク"""
        logger.info("LSTMTimeSeriesModel ベンチマーク開始")

        component_name = "LSTMTimeSeriesModel"
        benchmark_start = time.time()
        total_memory_start = tracemalloc.get_traced_memory()[0] / 1024 / 1024

        try:
            lstm_model = LSTMTimeSeriesModel(
                sequence_length=20,  # 高速化のため短縮
                prediction_horizon=3,
                lstm_units=[32, 16],  # 軽量化
            )

            # 1. LSTM特徴量準備のプロファイリング
            with self.profile_method(
                component_name, "prepare_lstm_features", len(test_data)
            ):
                features = lstm_model.prepare_lstm_features(test_data)

            # 2. 系列データ作成のプロファイリング
            if not features.empty:
                with self.profile_method(
                    component_name, "create_sequences", len(features)
                ):
                    X, y = lstm_model.create_sequences(features)

            # 3. モデル構築のプロファイリング（軽量版）
            if len(X) > 0:
                with self.profile_method(
                    component_name, "build_lstm_model", X.shape[1]
                ):
                    model = lstm_model.build_lstm_model((X.shape[1], X.shape[2]))

        except Exception as e:
            logger.error(f"LSTMTimeSeriesModel ベンチマークエラー: {e}")

        # ベンチマーク結果作成
        total_time = time.time() - benchmark_start
        total_memory_end = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        total_memory_used = total_memory_end - total_memory_start

        benchmark = ComponentBenchmark(
            component_name=component_name,
            total_execution_time=total_time,
            total_memory_usage=total_memory_used,
            method_metrics=self.profiling_results.get(component_name, []),
        )

        # ボトルネック分析
        benchmark.bottleneck_methods = self._identify_bottlenecks(
            benchmark.method_metrics
        )
        benchmark.optimization_suggestions = self._generate_optimization_suggestions(
            benchmark
        )

        self.benchmark_results.append(benchmark)
        logger.info(f"LSTMTimeSeriesModel ベンチマーク完了: {total_time:.2f}秒")

        return benchmark

    def benchmark_technical_indicators(
        self, test_data: pd.DataFrame
    ) -> ComponentBenchmark:
        """高度テクニカル指標のベンチマーク"""
        logger.info("AdvancedTechnicalIndicators ベンチマーク開始")

        component_name = "AdvancedTechnicalIndicators"
        benchmark_start = time.time()
        total_memory_start = tracemalloc.get_traced_memory()[0] / 1024 / 1024

        try:
            technical_analyzer = AdvancedTechnicalIndicators()

            # 1. 一目均衡表計算のプロファイリング
            with self.profile_method(
                component_name, "calculate_ichimoku_cloud", len(test_data)
            ):
                ichimoku_result = technical_analyzer.calculate_ichimoku_cloud(test_data)

            # 2. フィボナッチ分析のプロファイリング
            with self.profile_method(
                component_name, "analyze_fibonacci_levels", len(test_data)
            ):
                fibonacci_result = technical_analyzer.analyze_fibonacci_levels(
                    test_data
                )

            # 3. エリオット波動分析のプロファイリング
            with self.profile_method(
                component_name, "detect_elliott_waves", len(test_data)
            ):
                elliott_result = technical_analyzer.detect_elliott_waves(test_data)

        except Exception as e:
            logger.error(f"AdvancedTechnicalIndicators ベンチマークエラー: {e}")

        # ベンチマーク結果作成
        total_time = time.time() - benchmark_start
        total_memory_end = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        total_memory_used = total_memory_end - total_memory_start

        benchmark = ComponentBenchmark(
            component_name=component_name,
            total_execution_time=total_time,
            total_memory_usage=total_memory_used,
            method_metrics=self.profiling_results.get(component_name, []),
        )

        # ボトルネック分析
        benchmark.bottleneck_methods = self._identify_bottlenecks(
            benchmark.method_metrics
        )
        benchmark.optimization_suggestions = self._generate_optimization_suggestions(
            benchmark
        )

        self.benchmark_results.append(benchmark)
        logger.info(f"AdvancedTechnicalIndicators ベンチマーク完了: {total_time:.2f}秒")

        return benchmark

    def _identify_bottlenecks(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """ボトルネック特定"""
        if not metrics:
            return []

        # 実行時間でソート
        sorted_metrics = sorted(metrics, key=lambda x: x.execution_time, reverse=True)

        bottlenecks = []
        total_time = sum(m.execution_time for m in metrics)

        for metric in sorted_metrics:
            # 全体の10%以上を占める場合はボトルネック
            if metric.execution_time / total_time > 0.1:
                bottlenecks.append(
                    f"{metric.method_name} ({metric.execution_time:.2f}秒)"
                )

        return bottlenecks

    def _generate_optimization_suggestions(
        self, benchmark: ComponentBenchmark
    ) -> List[str]:
        """最適化提案生成"""
        suggestions = []

        # 実行時間ベースの提案
        if benchmark.total_execution_time > 10:
            suggestions.append("実行時間が長い - 並列処理の導入を検討")

        # メモリ使用量ベースの提案
        if benchmark.total_memory_usage > 500:  # 500MB以上
            suggestions.append(
                "メモリ使用量が多い - データのチャンク処理やキャッシュ最適化を検討"
            )

        # ボトルネック別の提案
        for bottleneck in benchmark.bottleneck_methods:
            if "prepare_features" in bottleneck:
                suggestions.append(
                    "特徴量準備の最適化: pandas-taの効率的な使用、不要な計算削除"
                )
            elif "train_models" in bottleneck:
                suggestions.append(
                    "モデル訓練の最適化: ハイパーパラメータ調整、早期停止の導入"
                )
            elif "lstm" in bottleneck.lower():
                suggestions.append("LSTM処理の最適化: バッチサイズ調整、GPU使用の検討")

        return suggestions

    def generate_comprehensive_report(self) -> str:
        """包括的レポート生成"""
        report = f"""
ML処理パフォーマンス・プロファイリング レポート
{"=" * 60}
生成日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

【概要】
- 分析対象コンポーネント: {len(self.benchmark_results)}個
- 総プロファイリング時間: {sum(b.total_execution_time for b in self.benchmark_results):.2f}秒
- 総メモリ使用量: {sum(b.total_memory_usage for b in self.benchmark_results):.1f}MB

【コンポーネント別詳細】
"""

        for benchmark in self.benchmark_results:
            report += f"""
{benchmark.component_name}:
  実行時間: {benchmark.total_execution_time:.2f}秒
  メモリ使用: {benchmark.total_memory_usage:.1f}MB
  ボトルネック: {", ".join(benchmark.bottleneck_methods[:3]) or "なし"}

  最適化提案:
"""
            for suggestion in benchmark.optimization_suggestions:
                report += f"    - {suggestion}\n"

            report += "\n  メソッド別詳細:\n"
            for metric in benchmark.method_metrics:
                status = "❌エラー" if metric.error_occurred else "✅正常"
                report += f"    {metric.method_name}: {metric.execution_time:.2f}秒, {metric.memory_usage_mb:.1f}MB {status}\n"

        report += """
【総合評価】
"""

        # パフォーマンス評価
        total_time = sum(b.total_execution_time for b in self.benchmark_results)
        total_memory = sum(b.total_memory_usage for b in self.benchmark_results)

        if total_time < 5:
            performance_grade = "A (優秀)"
        elif total_time < 15:
            performance_grade = "B (良好)"
        elif total_time < 30:
            performance_grade = "C (普通)"
        else:
            performance_grade = "D (要改善)"

        if total_memory < 200:
            memory_grade = "A (優秀)"
        elif total_memory < 500:
            memory_grade = "B (良好)"
        elif total_memory < 1000:
            memory_grade = "C (普通)"
        else:
            memory_grade = "D (要改善)"

        report += f"パフォーマンス評価: {performance_grade}\n"
        report += f"メモリ効率評価: {memory_grade}\n"

        # 優先改善項目
        all_bottlenecks = []
        for benchmark in self.benchmark_results:
            all_bottlenecks.extend(benchmark.bottleneck_methods)

        if all_bottlenecks:
            report += "\n【優先改善項目】\n"
            for bottleneck in all_bottlenecks[:5]:  # 上位5つ
                report += f"- {bottleneck}\n"

        report += f"\n{'=' * 60}\n"

        return report

    def save_report(self, filename: str = None):
        """レポートファイル保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ml_performance_report_{timestamp}.txt"

        report = self.generate_comprehensive_report()

        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"パフォーマンスレポート保存: {filename}")
        return filename


def generate_test_data(symbol: str = "TEST", days: int = 200) -> pd.DataFrame:
    """テスト用データ生成"""
    dates = pd.date_range(start="2023-01-01", periods=days)
    np.random.seed(42)

    base_price = 2500
    returns = np.random.normal(0.0005, 0.02, days)
    prices = [base_price]

    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, base_price * 0.5))

    prices = prices[1:]  # 最初の要素を削除

    return pd.DataFrame(
        {
            "Open": [p * np.random.uniform(0.995, 1.005) for p in prices],
            "High": [
                max(o, c) * np.random.uniform(1.000, 1.02)
                for o, c in zip(prices, prices)
            ],
            "Low": [
                min(o, c) * np.random.uniform(0.98, 1.000)
                for o, c in zip(prices, prices)
            ],
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, days),
        },
        index=dates,
    )


if __name__ == "__main__":
    print("=== ML処理パフォーマンス・プロファイラー テスト ===")

    try:
        # プロファイラー初期化
        profiler = MLPerformanceProfiler()

        # テストデータ生成
        print("1. テストデータ生成...")
        test_data = generate_test_data(days=150)  # 高速化のため短縮
        print(f"   テストデータ: {len(test_data)}日分")

        # 各コンポーネントのベンチマーク実行
        print("2. AdvancedMLEngine ベンチマーク...")
        ml_benchmark = profiler.benchmark_advanced_ml_engine(test_data)
        print(f"   実行時間: {ml_benchmark.total_execution_time:.2f}秒")
        print(f"   メモリ使用: {ml_benchmark.total_memory_usage:.1f}MB")

        print("3. LSTMTimeSeriesModel ベンチマーク...")
        lstm_benchmark = profiler.benchmark_lstm_model(test_data)
        print(f"   実行時間: {lstm_benchmark.total_execution_time:.2f}秒")
        print(f"   メモリ使用: {lstm_benchmark.total_memory_usage:.1f}MB")

        print("4. AdvancedTechnicalIndicators ベンチマーク...")
        tech_benchmark = profiler.benchmark_technical_indicators(test_data)
        print(f"   実行時間: {tech_benchmark.total_execution_time:.2f}秒")
        print(f"   メモリ使用: {tech_benchmark.total_memory_usage:.1f}MB")

        # 包括的レポート生成
        print("5. 包括的レポート生成...")
        report_filename = profiler.save_report()

        # サマリー表示
        print("\n=== ベンチマーク結果サマリー ===")
        total_time = sum(b.total_execution_time for b in profiler.benchmark_results)
        total_memory = sum(b.total_memory_usage for b in profiler.benchmark_results)

        print(f"総実行時間: {total_time:.2f}秒")
        print(f"総メモリ使用量: {total_memory:.1f}MB")
        print(f"レポート保存: {report_filename}")

        # トップボトルネック表示
        all_bottlenecks = []
        for benchmark in profiler.benchmark_results:
            all_bottlenecks.extend(benchmark.bottleneck_methods)

        if all_bottlenecks:
            print("\n主要ボトルネック:")
            for bottleneck in all_bottlenecks[:3]:
                print(f"  - {bottleneck}")

        print("\n✅ ML処理パフォーマンス・プロファイリング完了！")

    except Exception as e:
        print(f"❌ プロファイリングエラー: {e}")
        import traceback

        traceback.print_exc()
