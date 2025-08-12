#!/usr/bin/env python3
"""
ML Performance Profiler Test (ASCII Safe Version)
Issue #325: ML Processing Bottleneck Detailed Profiling

ASCII-safe ML component performance analysis and bottleneck identification tool
"""

import cProfile
import io
import logging
import pstats
import sys
import threading
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import ML components
try:
    from src.day_trade.analysis.advanced_technical_indicators import (
        AdvancedTechnicalIndicators,
    )
    from src.day_trade.data.advanced_ml_engine import AdvancedMLEngine
    from src.day_trade.data.lstm_time_series_model import LSTMTimeSeriesModel
    from src.day_trade.utils.logging_config import get_context_logger

    ML_COMPONENTS_AVAILABLE = True
    print("ML components imported successfully")
except ImportError as e:
    print(f"ML components import error: {e}")
    ML_COMPONENTS_AVAILABLE = False
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance measurement results"""

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
    """Component benchmark results"""

    component_name: str
    total_execution_time: float
    total_memory_usage: float
    method_metrics: List[PerformanceMetrics] = field(default_factory=list)
    bottleneck_methods: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)


class ResourceMonitor:
    """Resource usage monitor"""

    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None

    def start_monitoring(self, interval: float = 0.1):
        """Start monitoring"""
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
                    logger.error(f"Resource monitoring error: {e}")
                    break

        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and get results"""
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
    """ML Processing Performance Profiler"""

    def __init__(self):
        """Initialize"""
        self.resource_monitor = ResourceMonitor()
        self.profiling_results = {}
        self.benchmark_results = []

        # Start memory tracking
        tracemalloc.start()

        logger.info("ML Performance Profiler initialized")

    @contextmanager
    def profile_method(self, component_name: str, method_name: str, data_size: int = 0):
        """Method execution profiling context manager"""
        # Start profiling
        profiler = cProfile.Profile()

        # Start resource monitoring
        self.resource_monitor.start_monitoring()

        # Record memory usage
        initial_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        start_time = time.time()

        try:
            profiler.enable()
            yield
            profiler.disable()

            # Calculate execution time
            execution_time = time.time() - start_time

            # Get memory usage
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            current_memory_mb = current_memory / 1024 / 1024
            peak_memory_mb = peak_memory / 1024 / 1024
            memory_usage = current_memory_mb - initial_memory

            # Get resource monitoring results
            resource_stats = self.resource_monitor.stop_monitoring()

            # Get profiling statistics
            stats_io = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_io)
            stats.sort_stats("cumulative")
            function_calls = stats.total_calls

            # Record results
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
                f"Profiling completed: {component_name}.{method_name} "
                f"({execution_time:.2f}s, {memory_usage:.1f}MB)"
            )

        except Exception as e:
            # Record error
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
            logger.error(f"Profiling error: {component_name}.{method_name}: {e}")
            raise

    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record metrics"""
        component_name = metrics.component_name

        if component_name not in self.profiling_results:
            self.profiling_results[component_name] = []

        self.profiling_results[component_name].append(metrics)

    def benchmark_advanced_ml_engine(self, test_data: pd.DataFrame) -> ComponentBenchmark:
        """Benchmark AdvancedMLEngine"""
        print("Starting AdvancedMLEngine benchmark...")
        logger.info("AdvancedMLEngine benchmark starting")

        component_name = "AdvancedMLEngine"
        benchmark_start = time.time()
        total_memory_start = tracemalloc.get_traced_memory()[0] / 1024 / 1024

        try:
            if not ML_COMPONENTS_AVAILABLE:
                raise ImportError("ML components not available")

            engine = AdvancedMLEngine(fast_mode=True)  # Use fast mode for profiling

            # 1. Feature preparation profiling
            print("  - Profiling feature preparation...")
            with self.profile_method(component_name, "prepare_features", len(test_data)):
                features = engine.prepare_features(test_data)

            # 2. Model training profiling (limited)
            if not features.empty and len(features) > 50:  # Only if sufficient data
                print("  - Profiling model training...")
                with self.profile_method(component_name, "train_models", len(features)):
                    results = engine.train_models(features, test_symbol="PROFILE_TEST")

            # 3. Ensemble prediction profiling
            if not features.empty and len(features) > 20:
                print("  - Profiling ensemble prediction...")
                with self.profile_method(component_name, "ensemble_predict", len(features)):
                    predictions = engine.ensemble_predict(features, test_symbol="PROFILE_TEST")

        except Exception as e:
            logger.error(f"AdvancedMLEngine benchmark error: {e}")
            print(f"    [ERROR] {e}")

        # Create benchmark results
        total_time = time.time() - benchmark_start
        total_memory_end = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        total_memory_used = total_memory_end - total_memory_start

        benchmark = ComponentBenchmark(
            component_name=component_name,
            total_execution_time=total_time,
            total_memory_usage=total_memory_used,
            method_metrics=self.profiling_results.get(component_name, []),
        )

        # Bottleneck analysis
        benchmark.bottleneck_methods = self._identify_bottlenecks(benchmark.method_metrics)
        benchmark.optimization_suggestions = self._generate_optimization_suggestions(benchmark)

        self.benchmark_results.append(benchmark)
        logger.info(f"AdvancedMLEngine benchmark completed: {total_time:.2f}s")
        print(f"  Completed in {total_time:.2f}s, Memory: {total_memory_used:.1f}MB")

        return benchmark

    def benchmark_lstm_model(self, test_data: pd.DataFrame) -> ComponentBenchmark:
        """Benchmark LSTM Model"""
        print("Starting LSTMTimeSeriesModel benchmark...")
        logger.info("LSTMTimeSeriesModel benchmark starting")

        component_name = "LSTMTimeSeriesModel"
        benchmark_start = time.time()
        total_memory_start = tracemalloc.get_traced_memory()[0] / 1024 / 1024

        try:
            if not ML_COMPONENTS_AVAILABLE:
                raise ImportError("ML components not available")

            lstm_model = LSTMTimeSeriesModel(
                sequence_length=10,  # Shorter for profiling
                prediction_horizon=2,
                lstm_units=[16, 8],  # Lightweight
            )

            # 1. LSTM feature preparation profiling
            print("  - Profiling LSTM feature preparation...")
            with self.profile_method(component_name, "prepare_lstm_features", len(test_data)):
                features = lstm_model.prepare_lstm_features(test_data)

            # 2. Sequence creation profiling
            if not features.empty and len(features) > 20:
                print("  - Profiling sequence creation...")
                with self.profile_method(component_name, "create_sequences", len(features)):
                    X, y = lstm_model.create_sequences(features)

            # 3. Model building profiling (lightweight)
            if "X" in locals() and len(X) > 0:
                print("  - Profiling model building...")
                with self.profile_method(component_name, "build_lstm_model", X.shape[1]):
                    model = lstm_model.build_lstm_model((X.shape[1], X.shape[2]))

        except Exception as e:
            logger.error(f"LSTMTimeSeriesModel benchmark error: {e}")
            print(f"    [ERROR] {e}")

        # Create benchmark results
        total_time = time.time() - benchmark_start
        total_memory_end = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        total_memory_used = total_memory_end - total_memory_start

        benchmark = ComponentBenchmark(
            component_name=component_name,
            total_execution_time=total_time,
            total_memory_usage=total_memory_used,
            method_metrics=self.profiling_results.get(component_name, []),
        )

        # Bottleneck analysis
        benchmark.bottleneck_methods = self._identify_bottlenecks(benchmark.method_metrics)
        benchmark.optimization_suggestions = self._generate_optimization_suggestions(benchmark)

        self.benchmark_results.append(benchmark)
        logger.info(f"LSTMTimeSeriesModel benchmark completed: {total_time:.2f}s")
        print(f"  Completed in {total_time:.2f}s, Memory: {total_memory_used:.1f}MB")

        return benchmark

    def benchmark_technical_indicators(self, test_data: pd.DataFrame) -> ComponentBenchmark:
        """Benchmark Advanced Technical Indicators"""
        print("Starting AdvancedTechnicalIndicators benchmark...")
        logger.info("AdvancedTechnicalIndicators benchmark starting")

        component_name = "AdvancedTechnicalIndicators"
        benchmark_start = time.time()
        total_memory_start = tracemalloc.get_traced_memory()[0] / 1024 / 1024

        try:
            if not ML_COMPONENTS_AVAILABLE:
                raise ImportError("ML components not available")

            technical_analyzer = AdvancedTechnicalIndicators()

            # 1. Ichimoku Cloud calculation profiling
            print("  - Profiling Ichimoku Cloud calculation...")
            with self.profile_method(component_name, "calculate_ichimoku_cloud", len(test_data)):
                ichimoku_result = technical_analyzer.calculate_ichimoku_cloud(test_data)

            # 2. Fibonacci analysis profiling
            print("  - Profiling Fibonacci analysis...")
            with self.profile_method(component_name, "analyze_fibonacci_levels", len(test_data)):
                fibonacci_result = technical_analyzer.analyze_fibonacci_levels(test_data)

            # 3. Elliott Wave analysis profiling
            print("  - Profiling Elliott Wave analysis...")
            with self.profile_method(component_name, "detect_elliott_waves", len(test_data)):
                elliott_result = technical_analyzer.detect_elliott_waves(test_data)

        except Exception as e:
            logger.error(f"AdvancedTechnicalIndicators benchmark error: {e}")
            print(f"    [ERROR] {e}")

        # Create benchmark results
        total_time = time.time() - benchmark_start
        total_memory_end = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        total_memory_used = total_memory_end - total_memory_start

        benchmark = ComponentBenchmark(
            component_name=component_name,
            total_execution_time=total_time,
            total_memory_usage=total_memory_used,
            method_metrics=self.profiling_results.get(component_name, []),
        )

        # Bottleneck analysis
        benchmark.bottleneck_methods = self._identify_bottlenecks(benchmark.method_metrics)
        benchmark.optimization_suggestions = self._generate_optimization_suggestions(benchmark)

        self.benchmark_results.append(benchmark)
        logger.info(f"AdvancedTechnicalIndicators benchmark completed: {total_time:.2f}s")
        print(f"  Completed in {total_time:.2f}s, Memory: {total_memory_used:.1f}MB")

        return benchmark

    def _identify_bottlenecks(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """Identify bottlenecks"""
        if not metrics:
            return []

        # Sort by execution time
        sorted_metrics = sorted(metrics, key=lambda x: x.execution_time, reverse=True)

        bottlenecks = []
        total_time = sum(m.execution_time for m in metrics if not m.error_occurred)

        for metric in sorted_metrics:
            if metric.error_occurred:
                continue
            # Consider methods taking more than 10% of total time as bottlenecks
            if total_time > 0 and metric.execution_time / total_time > 0.1:
                bottlenecks.append(f"{metric.method_name} ({metric.execution_time:.2f}s)")

        return bottlenecks

    def _generate_optimization_suggestions(self, benchmark: ComponentBenchmark) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []

        # Execution time based suggestions
        if benchmark.total_execution_time > 10:
            suggestions.append("Long execution time - Consider parallel processing")

        # Memory usage based suggestions
        if benchmark.total_memory_usage > 500:  # 500MB+
            suggestions.append("High memory usage - Consider data chunking or cache optimization")

        # Bottleneck specific suggestions
        for bottleneck in benchmark.bottleneck_methods:
            if "prepare_features" in bottleneck:
                suggestions.append(
                    "Feature preparation optimization: Efficient pandas-ta usage, remove unnecessary calculations"
                )
            elif "train_models" in bottleneck:
                suggestions.append(
                    "Model training optimization: Hyperparameter tuning, early stopping implementation"
                )
            elif "lstm" in bottleneck.lower():
                suggestions.append(
                    "LSTM processing optimization: Batch size adjustment, consider GPU usage"
                )
            elif "ichimoku" in bottleneck.lower():
                suggestions.append(
                    "Ichimoku calculation optimization: Vectorization, reduce redundant calculations"
                )
            elif "fibonacci" in bottleneck.lower():
                suggestions.append(
                    "Fibonacci analysis optimization: Caching, mathematical optimization"
                )

        return suggestions

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive report"""
        report = f"""
ML Performance Profiling Report (ASCII Safe)
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

[Overview]
- Components analyzed: {len(self.benchmark_results)}
- Total profiling time: {sum(b.total_execution_time for b in self.benchmark_results):.2f}s
- Total memory usage: {sum(b.total_memory_usage for b in self.benchmark_results):.1f}MB

[Component Details]
"""

        for benchmark in self.benchmark_results:
            report += f"""
{benchmark.component_name}:
  Execution time: {benchmark.total_execution_time:.2f}s
  Memory usage: {benchmark.total_memory_usage:.1f}MB
  Bottlenecks: {', '.join(benchmark.bottleneck_methods[:3]) or 'None'}

  Optimization suggestions:
"""
            for suggestion in benchmark.optimization_suggestions:
                report += f"    - {suggestion}\n"

            report += "\n  Method details:\n"
            for metric in benchmark.method_metrics:
                status = "[ERROR]" if metric.error_occurred else "[OK]"
                report += f"    {metric.method_name}: {metric.execution_time:.2f}s, {metric.memory_usage_mb:.1f}MB {status}\n"

        report += """
[Overall Evaluation]
"""

        # Performance evaluation
        total_time = sum(b.total_execution_time for b in self.benchmark_results)
        total_memory = sum(b.total_memory_usage for b in self.benchmark_results)

        if total_time < 5:
            performance_grade = "A (Excellent)"
        elif total_time < 15:
            performance_grade = "B (Good)"
        elif total_time < 30:
            performance_grade = "C (Average)"
        else:
            performance_grade = "D (Needs Improvement)"

        if total_memory < 200:
            memory_grade = "A (Excellent)"
        elif total_memory < 500:
            memory_grade = "B (Good)"
        elif total_memory < 1000:
            memory_grade = "C (Average)"
        else:
            memory_grade = "D (Needs Improvement)"

        report += f"Performance Rating: {performance_grade}\n"
        report += f"Memory Efficiency Rating: {memory_grade}\n"

        # Priority improvement items
        all_bottlenecks = []
        for benchmark in self.benchmark_results:
            all_bottlenecks.extend(benchmark.bottleneck_methods)

        if all_bottlenecks:
            report += "\n[Priority Improvement Items]\n"
            for bottleneck in all_bottlenecks[:5]:  # Top 5
                report += f"- {bottleneck}\n"

        report += f"\n{'='*60}\n"

        return report

    def save_report(self, filename: str = None):
        """Save report file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ml_performance_report_ascii_{timestamp}.txt"

        report = self.generate_comprehensive_report()

        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Performance report saved: {filename}")
        return filename


def generate_test_data(symbol: str = "TEST", days: int = 200) -> pd.DataFrame:
    """Generate test data"""
    dates = pd.date_range(start="2023-01-01", periods=days)
    np.random.seed(42)

    base_price = 2500
    returns = np.random.normal(0.0005, 0.02, days)
    prices = [base_price]

    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, base_price * 0.5))

    prices = prices[1:]  # Remove first element

    return pd.DataFrame(
        {
            "Open": [p * np.random.uniform(0.995, 1.005) for p in prices],
            "High": [max(o, c) * np.random.uniform(1.000, 1.02) for o, c in zip(prices, prices)],
            "Low": [min(o, c) * np.random.uniform(0.98, 1.000) for o, c in zip(prices, prices)],
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, days),
        },
        index=dates,
    )


if __name__ == "__main__":
    print("=== ML Processing Performance Profiler Test (ASCII Safe) ===")

    try:
        # Initialize profiler
        profiler = MLPerformanceProfiler()

        # Generate test data
        print("1. Generating test data...")
        test_data = generate_test_data(days=100)  # Reduced for faster profiling
        print(f"   Test data: {len(test_data)} days")

        # Execute benchmarks for each component
        print("2. AdvancedMLEngine benchmark...")
        ml_benchmark = profiler.benchmark_advanced_ml_engine(test_data)

        print("3. LSTMTimeSeriesModel benchmark...")
        lstm_benchmark = profiler.benchmark_lstm_model(test_data)

        print("4. AdvancedTechnicalIndicators benchmark...")
        tech_benchmark = profiler.benchmark_technical_indicators(test_data)

        # Generate comprehensive report
        print("5. Generating comprehensive report...")
        report_filename = profiler.save_report()

        # Display summary
        print("\n=== Benchmark Results Summary ===")
        total_time = sum(b.total_execution_time for b in profiler.benchmark_results)
        total_memory = sum(b.total_memory_usage for b in profiler.benchmark_results)

        print(f"Total execution time: {total_time:.2f}s")
        print(f"Total memory usage: {total_memory:.1f}MB")
        print(f"Report saved: {report_filename}")

        # Display top bottlenecks
        all_bottlenecks = []
        for benchmark in profiler.benchmark_results:
            all_bottlenecks.extend(benchmark.bottleneck_methods)

        if all_bottlenecks:
            print("\nTop bottlenecks:")
            for bottleneck in all_bottlenecks[:3]:
                print(f"  - {bottleneck}")

        # Display optimization suggestions
        all_suggestions = []
        for benchmark in profiler.benchmark_results:
            all_suggestions.extend(benchmark.optimization_suggestions)

        if all_suggestions:
            print("\nOptimization suggestions:")
            for suggestion in set(all_suggestions):  # Remove duplicates
                print(f"  - {suggestion}")

        print("\n[OK] ML Processing Performance Profiling Completed!")

    except Exception as e:
        print(f"[ERROR] Profiling error: {e}")
        import traceback

        traceback.print_exc()
