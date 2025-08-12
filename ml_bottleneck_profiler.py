#!/usr/bin/env python3
"""
ML Bottleneck Profiler (Corrected Methods)
Issue #325: ML Processing Bottleneck Detailed Profiling

Actual bottleneck identification with correct method names
"""

import logging
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

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


class SimpleProfiler:
    """Simple performance profiler"""

    def __init__(self):
        self.results = {}
        tracemalloc.start()

    def profile_method(self, component_name: str, method_name: str, func, *args, **kwargs):
        """Profile a single method execution"""
        # Initial memory
        initial_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        initial_process_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Execute and measure
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            success = True
            error_msg = ""
        except Exception as e:
            result = None
            success = False
            error_msg = str(e)

        execution_time = time.time() - start_time

        # Final memory
        final_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        final_process_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        process_memory_increase = final_process_memory - initial_process_memory

        # Store results
        key = f"{component_name}.{method_name}"
        self.results[key] = {
            "execution_time": execution_time,
            "memory_increase": memory_increase,
            "process_memory_increase": process_memory_increase,
            "success": success,
            "error": error_msg,
            "result": result,
        }

        status = "[OK]" if success else "[ERROR]"
        print(f"  {method_name}: {execution_time:.2f}s, {memory_increase:.1f}MB {status}")

        if not success:
            print(f"    Error: {error_msg}")

        return result

    def get_bottlenecks(self, threshold_seconds=1.0):
        """Identify bottleneck methods"""
        bottlenecks = []

        for key, data in self.results.items():
            if data["success"] and data["execution_time"] > threshold_seconds:
                bottlenecks.append((key, data["execution_time"], data["memory_increase"]))

        # Sort by execution time
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        return bottlenecks

    def generate_report(self):
        """Generate profiling report"""
        total_time = sum(data["execution_time"] for data in self.results.values())
        total_memory = sum(
            data["memory_increase"] for data in self.results.values() if data["success"]
        )
        successful_methods = sum(1 for data in self.results.values() if data["success"])
        failed_methods = sum(1 for data in self.results.values() if not data["success"])

        bottlenecks = self.get_bottlenecks()

        report = f"""
ML Components Bottleneck Analysis Report
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTION SUMMARY:
- Total execution time: {total_time:.2f} seconds
- Total memory usage: {total_memory:.1f} MB
- Successful methods: {successful_methods}
- Failed methods: {failed_methods}

TOP BOTTLENECKS (>1.0 second):
"""
        if bottlenecks:
            for i, (method, exec_time, memory) in enumerate(bottlenecks[:5], 1):
                report += f"  {i}. {method}: {exec_time:.2f}s, {memory:.1f}MB\n"
        else:
            report += "  No significant bottlenecks identified\n"

        report += """
DETAILED RESULTS:
"""
        for key, data in sorted(self.results.items()):
            status = "SUCCESS" if data["success"] else "FAILED"
            report += f"  {key}: {data['execution_time']:.2f}s, {data['memory_increase']:.1f}MB [{status}]\n"
            if not data["success"]:
                report += f"    Error: {data['error']}\n"

        report += """
OPTIMIZATION RECOMMENDATIONS:
"""
        if bottlenecks:
            for method, exec_time, memory in bottlenecks[:3]:
                if "prepare_ml_features" in method:
                    report += (
                        "- Feature preparation: Optimize pandas operations, use vectorization\n"
                    )
                elif "train_" in method:
                    report += "- Model training: Consider hyperparameter tuning, early stopping\n"
                elif "lstm" in method.lower():
                    report += "- LSTM processing: Optimize batch size, consider GPU acceleration\n"
                elif "predict_" in method:
                    report += "- Prediction: Implement caching, batch processing\n"
                elif "ichimoku" in method.lower():
                    report += "- Ichimoku calculation: Use vectorized operations, avoid loops\n"
        else:
            report += "- Performance is acceptable, consider monitoring in production\n"

        report += f"\n{'='*60}\n"
        return report


def generate_test_data(days: int = 200) -> pd.DataFrame:
    """Generate realistic test data"""
    dates = pd.date_range(start="2023-01-01", periods=days)
    np.random.seed(42)

    base_price = 2500
    returns = np.random.normal(0.0005, 0.02, days)
    prices = [base_price]

    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, base_price * 0.5))

    prices = prices[1:]

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


def profile_advanced_ml_engine(profiler: SimpleProfiler, test_data: pd.DataFrame):
    """Profile AdvancedMLEngine methods"""
    print("\n1. Profiling AdvancedMLEngine...")

    try:
        engine = AdvancedMLEngine(fast_mode=True)

        # Profile ML feature preparation
        profiler.profile_method(
            "AdvancedMLEngine",
            "prepare_ml_features",
            engine.prepare_ml_features,
            test_data,
        )

        # Profile fast feature preparation
        profiler.profile_method(
            "AdvancedMLEngine",
            "prepare_fast_features",
            engine.prepare_fast_features,
            test_data,
        )

        # Profile technical indicators calculation
        profiler.profile_method(
            "AdvancedMLEngine",
            "calculate_advanced_technical_indicators",
            engine.calculate_advanced_technical_indicators,
            test_data,
        )

        # Profile trend prediction model training (lightweight)
        profiler.profile_method(
            "AdvancedMLEngine",
            "train_trend_prediction_model",
            engine.train_trend_prediction_model,
            test_data,
            "TEST",
        )

        # Profile investment advice generation
        profiler.profile_method(
            "AdvancedMLEngine",
            "generate_fast_investment_advice",
            engine.generate_fast_investment_advice,
            test_data,
            "TEST",
        )

    except Exception as e:
        print(f"  [ERROR] AdvancedMLEngine profiling failed: {e}")


def profile_lstm_model(profiler: SimpleProfiler, test_data: pd.DataFrame):
    """Profile LSTMTimeSeriesModel methods"""
    print("\n2. Profiling LSTMTimeSeriesModel...")

    try:
        # Use lightweight configuration for profiling
        lstm_model = LSTMTimeSeriesModel(
            sequence_length=10, prediction_horizon=2, lstm_units=[16, 8]
        )

        # Profile LSTM feature preparation
        features = profiler.profile_method(
            "LSTMTimeSeriesModel",
            "prepare_lstm_features",
            lstm_model.prepare_lstm_features,
            test_data,
        )

        if features is not None and not features.empty:
            # Profile sequence creation
            sequences = profiler.profile_method(
                "LSTMTimeSeriesModel",
                "create_sequences",
                lstm_model.create_sequences,
                features,
            )

            if sequences is not None and len(sequences) == 2:
                X, y = sequences
                if len(X) > 0:
                    # Profile model building
                    profiler.profile_method(
                        "LSTMTimeSeriesModel",
                        "build_lstm_model",
                        lstm_model.build_lstm_model,
                        (X.shape[1], X.shape[2]),
                    )

                    # Profile LSTM training (very limited)
                    profiler.profile_method(
                        "LSTMTimeSeriesModel",
                        "train_lstm_model",
                        lambda: lstm_model.train_lstm_model(
                            features, "TEST", epochs=2, batch_size=16
                        ),
                    )

    except Exception as e:
        print(f"  [ERROR] LSTMTimeSeriesModel profiling failed: {e}")


def profile_technical_indicators(profiler: SimpleProfiler, test_data: pd.DataFrame):
    """Profile AdvancedTechnicalIndicators methods"""
    print("\n3. Profiling AdvancedTechnicalIndicators...")

    try:
        tech_analyzer = AdvancedTechnicalIndicators()

        # Profile Ichimoku Cloud calculation
        profiler.profile_method(
            "AdvancedTechnicalIndicators",
            "calculate_ichimoku_cloud",
            tech_analyzer.calculate_ichimoku_cloud,
            test_data,
        )

        # Profile Bollinger Bands calculation
        profiler.profile_method(
            "AdvancedTechnicalIndicators",
            "calculate_advanced_bollinger_bands",
            tech_analyzer.calculate_advanced_bollinger_bands,
            test_data,
        )

        # Profile volatility indicators
        profiler.profile_method(
            "AdvancedTechnicalIndicators",
            "calculate_volatility_indicators",
            tech_analyzer.calculate_volatility_indicators,
            test_data,
        )

        # Profile Fibonacci retracements
        profiler.profile_method(
            "AdvancedTechnicalIndicators",
            "detect_fibonacci_retracements",
            tech_analyzer.detect_fibonacci_retracements,
            test_data,
        )

        # Profile Elliott Wave detection
        profiler.profile_method(
            "AdvancedTechnicalIndicators",
            "detect_elliott_wave_patterns",
            tech_analyzer.detect_elliott_wave_patterns,
            test_data,
        )

        # Profile comprehensive signal generation
        profiler.profile_method(
            "AdvancedTechnicalIndicators",
            "generate_comprehensive_signal",
            tech_analyzer.generate_comprehensive_signal,
            test_data,
            "TEST",
        )

    except Exception as e:
        print(f"  [ERROR] AdvancedTechnicalIndicators profiling failed: {e}")


def main():
    """Main profiling execution"""
    print("=" * 60)
    print("ML Components Bottleneck Analysis")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not ML_COMPONENTS_AVAILABLE:
        print("[ERROR] ML components not available for profiling")
        return

    # Initialize profiler
    profiler = SimpleProfiler()

    # Generate test data
    print("\nGenerating test data...")
    test_data = generate_test_data(days=150)  # Medium size for realistic profiling
    print(f"  Test data: {len(test_data)} days")

    # Profile each component
    profile_advanced_ml_engine(profiler, test_data)
    profile_lstm_model(profiler, test_data)
    profile_technical_indicators(profiler, test_data)

    # Generate and save report
    print("\n4. Generating bottleneck analysis report...")
    report = profiler.generate_report()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"ml_bottleneck_analysis_{timestamp}.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved: {report_file}")

    # Display summary
    bottlenecks = profiler.get_bottlenecks()

    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS SUMMARY")
    print("=" * 60)

    if bottlenecks:
        print(f"Top bottlenecks identified: {len(bottlenecks)}")
        for i, (method, exec_time, memory) in enumerate(bottlenecks[:3], 1):
            print(f"  {i}. {method}: {exec_time:.2f}s ({memory:.1f}MB)")
    else:
        print("No significant bottlenecks found (all methods <1.0 second)")

    total_time = sum(data["execution_time"] for data in profiler.results.values())
    successful_methods = sum(1 for data in profiler.results.values() if data["success"])

    print("\nOverall performance:")
    print(f"  Total execution time: {total_time:.2f} seconds")
    print(f"  Successful method calls: {successful_methods}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return bottlenecks


if __name__ == "__main__":
    try:
        bottlenecks = main()
        print("\n[OK] ML Bottleneck Analysis Completed!")
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback

        traceback.print_exc()
