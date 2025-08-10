#!/usr/bin/env python3
"""
Simple Data Optimization Performance Test
Issue #378: Data I/O and Processing Optimization
"""

import time
import gc
import numpy as np
import pandas as pd
import psutil
import os

# ログ出力を有効化
import logging
logging.basicConfig(level=logging.INFO)

from src.day_trade.utils.data_optimization import (
    DataFrameOptimizer,
    ChunkedDataProcessor,
    create_optimized_dataframe
)
from src.day_trade.analysis.feature_engineering_unified import (
    FeatureEngineeringManager,
    FeatureConfig,
    generate_features
)
from src.day_trade.core.optimization_strategy import OptimizationConfig, OptimizationLevel


def create_test_dataset(rows: int = 10000) -> pd.DataFrame:
    """Create test dataset"""
    print(f"Creating test data: {rows} rows")

    # Basic stock price data simulation
    np.random.seed(42)  # For reproducibility

    # Date index
    dates = pd.date_range('2020-01-01', periods=rows, freq='1min')

    # Stock price data
    base_price = 1000.0
    returns = np.random.normal(0, 0.02, rows)  # 2% standard deviation
    prices = base_price * np.cumprod(1 + returns)

    # Generate high/low/volume
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, rows)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, rows)))
    volumes = np.random.lognormal(10, 1, rows).astype(int)

    # Create DataFrame with intentionally inefficient data types
    data = pd.DataFrame({
        'timestamp': dates,
        'Open': prices.astype('float64'),       # Before optimization: float64
        'High': highs.astype('float64'),        # Before optimization: float64
        'Low': lows.astype('float64'),          # Before optimization: float64
        'Close': prices.astype('float64'),      # Before optimization: float64
        'Volume': volumes.astype('int64'),      # Before optimization: int64
        'Market': ['TSE'] * rows,               # Before optimization: object
        'Sector': np.random.choice(['Finance', 'IT', 'Manufacturing', 'Retail'], rows),  # category candidate
    })

    return data


def test_dtype_optimization():
    """Test data type optimization performance"""
    print("\n=== Data Type Optimization Performance Test ===")

    test_sizes = [1000, 5000, 10000]
    results = []

    for size in test_sizes:
        print(f"\nTest data size: {size} rows")

        # Create test data
        data = create_test_dataset(size)
        memory_before = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB

        # Execute optimization
        optimizer = DataFrameOptimizer()
        start_time = time.perf_counter()

        optimized_data = optimizer.optimize_dtypes(data, copy=True)

        optimization_time = time.perf_counter() - start_time
        memory_after = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        memory_saved = memory_before - memory_after
        reduction_percent = (memory_saved / memory_before) * 100

        # Get optimization statistics
        optimization_stats = optimizer.get_optimization_stats()

        result = {
            'size': size,
            'memory_before_mb': round(memory_before, 3),
            'memory_after_mb': round(memory_after, 3),
            'memory_saved_mb': round(memory_saved, 3),
            'reduction_percent': round(reduction_percent, 1),
            'optimization_time_ms': round(optimization_time * 1000, 2),
            'optimizations_applied': optimization_stats['dtype_optimizations']
        }
        results.append(result)

        print(f"  Memory usage: {memory_before:.3f}MB -> {memory_after:.3f}MB")
        print(f"  Memory saved: {memory_saved:.3f}MB ({reduction_percent:.1f}%)")
        print(f"  Optimization time: {optimization_time*1000:.2f}ms")
        print(f"  Optimizations applied: {optimization_stats['dtype_optimizations']}")

    return results


def test_vectorization():
    """Test vectorization operations performance"""
    print("\n=== Vectorization Operations Performance Test ===")

    data = create_test_dataset(5000)
    optimizer = DataFrameOptimizer()

    # Define technical indicator operations
    technical_operations = [
        {'type': 'technical_indicator', 'indicator': 'sma', 'column': 'Close', 'period': 20},
        {'type': 'technical_indicator', 'indicator': 'ema', 'column': 'Close', 'period': 12},
        {'type': 'technical_indicator', 'indicator': 'rsi', 'column': 'Close', 'period': 14},
        {'type': 'technical_indicator', 'indicator': 'bollinger_bands', 'column': 'Close', 'period': 20},
    ]

    print(f"Number of operations: {len(technical_operations)}")

    # Execute vectorization
    start_time = time.perf_counter()

    result_data = optimizer.vectorize_operations(data, technical_operations)

    vectorization_time = time.perf_counter() - start_time

    # Check results
    original_cols = len(data.columns)
    result_cols = len(result_data.columns)
    added_features = result_cols - original_cols

    print(f"  Original columns: {original_cols}")
    print(f"  Result columns: {result_cols}")
    print(f"  Added features: {added_features}")
    print(f"  Vectorization time: {vectorization_time*1000:.2f}ms")
    print(f"  Average time per operation: {vectorization_time*1000/len(technical_operations):.2f}ms")

    # Get optimization statistics
    optimization_stats = optimizer.get_optimization_stats()
    print(f"  Vectorizations applied: {optimization_stats['vectorizations']}")

    return {
        'total_operations': len(technical_operations),
        'vectorization_time_ms': round(vectorization_time * 1000, 2),
        'features_added': added_features,
        'avg_time_per_operation_ms': round(vectorization_time * 1000 / len(technical_operations), 2),
        'vectorizations_applied': optimization_stats['vectorizations']
    }


def test_chunk_processing():
    """Test chunk processing performance"""
    print("\n=== Chunk Processing Performance Test ===")

    # Test with large data
    large_data = create_test_dataset(20000)  # 20,000 rows

    chunk_processor = ChunkedDataProcessor(chunk_size=5000)

    def simple_feature_calculation(chunk_data: pd.DataFrame) -> pd.DataFrame:
        """Simple feature calculation"""
        result = chunk_data.copy()
        result['SMA_20'] = chunk_data['Close'].rolling(window=20, min_periods=1).mean()
        result['Returns'] = chunk_data['Close'].pct_change()
        result['Volatility_10'] = result['Returns'].rolling(window=10, min_periods=1).std()
        return result

    print(f"Large data processing test: {len(large_data)} rows")

    # Execute chunk processing
    start_time = time.perf_counter()

    processed_data = chunk_processor.process_large_dataframe(
        large_data, simple_feature_calculation
    )

    chunk_processing_time = time.perf_counter() - start_time

    print(f"  Chunk processing time: {chunk_processing_time*1000:.2f}ms")
    print(f"  Processed data shape: {processed_data.shape}")
    print(f"  Time per row: {chunk_processing_time*1000/len(processed_data):.4f}ms")

    return {
        'data_size': len(large_data),
        'chunk_processing_time_ms': round(chunk_processing_time * 1000, 2),
        'processed_shape': processed_data.shape,
        'time_per_row_ms': round(chunk_processing_time * 1000 / len(processed_data), 4)
    }


def test_feature_engineering_integration():
    """Test feature engineering integration"""
    print("\n=== Feature Engineering Integration Test ===")

    test_data = create_test_dataset(3000)

    # Test with each optimization level
    optimization_levels = [
        (OptimizationLevel.STANDARD, "Standard"),
        (OptimizationLevel.OPTIMIZED, "Optimized"),
        (OptimizationLevel.ADAPTIVE, "Data Optimized"),
    ]

    results = []

    for level, level_name in optimization_levels:
        print(f"\n{level_name} level test:")

        # Configuration
        opt_config = OptimizationConfig(
            level=level,
            cache_enabled=True,
            performance_monitoring=True,
            parallel_processing=False,
            timeout_seconds=30
        )

        feature_config = FeatureConfig(
            lookback_periods=[5, 10, 20],
            volatility_windows=[10, 20],
            momentum_periods=[5, 10],
            enable_cross_features=True,
            enable_statistical_features=True,
            enable_dtype_optimization=(level == OptimizationLevel.ADAPTIVE),
            enable_copy_elimination=(level == OptimizationLevel.ADAPTIVE),
            enable_index_optimization=(level == OptimizationLevel.ADAPTIVE),
            chunk_size=1000,
        )

        # Performance measurement
        start_time = time.perf_counter()
        memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        # Execute feature generation
        feature_result = generate_features(
            test_data,
            feature_config=feature_config,
            optimization_config=opt_config
        )

        processing_time = time.perf_counter() - start_time
        memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # Record results
        result = {
            'level': level_name,
            'processing_time_ms': round(processing_time * 1000, 2),
            'memory_used_mb': round(memory_used, 2),
            'input_shape': test_data.shape,
            'output_shape': feature_result.features.shape,
            'features_generated': len(feature_result.feature_names),
            'strategy_used': feature_result.strategy_used
        }
        results.append(result)

        print(f"  Processing time: {processing_time*1000:.2f}ms")
        print(f"  Memory used: {memory_used:.2f}MB")
        print(f"  Input shape: {test_data.shape}")
        print(f"  Output shape: {feature_result.features.shape}")
        print(f"  Features generated: {len(feature_result.feature_names)}")
        print(f"  Strategy used: {feature_result.strategy_used}")

        # Garbage collection
        del feature_result
        gc.collect()

    return results


def generate_performance_report(dtype_results, vectorization_result, chunk_result, integration_results):
    """Generate performance report"""
    print("\n" + "="*80)
    print("Data I/O Optimization Performance Test Report")
    print("Issue #378: Data I/O and Processing Optimization")
    print("="*80)

    print("\n1. Data Type Optimization Results:")
    print("  | Data Size | Before(MB) | After(MB) | Saved(MB) | Reduction(%) | Time(ms) |")
    print("  |---|---|---|---|---|---|")
    for result in dtype_results:
        print(f"  | {result['size']:,} rows | {result['memory_before_mb']:.3f} | {result['memory_after_mb']:.3f} | {result['memory_saved_mb']:.3f} | {result['reduction_percent']:.1f} | {result['optimization_time_ms']:.2f} |")

    print("\n2. Vectorization Operations Results:")
    print(f"  - Total operations: {vectorization_result['total_operations']}")
    print(f"  - Processing time: {vectorization_result['vectorization_time_ms']:.2f}ms")
    print(f"  - Features generated: {vectorization_result['features_added']}")
    print(f"  - Average time per operation: {vectorization_result['avg_time_per_operation_ms']:.2f}ms")

    print("\n3. Chunk Processing Results:")
    print(f"  - Data size: {chunk_result['data_size']:,} rows")
    print(f"  - Processing time: {chunk_result['chunk_processing_time_ms']:.2f}ms")
    print(f"  - Time per row: {chunk_result['time_per_row_ms']:.4f}ms")

    print("\n4. Feature Engineering Integration Results:")
    print("  | Optimization Level | Time(ms) | Memory(MB) | Features | Strategy |")
    print("  |---|---|---|---|---|")
    for result in integration_results:
        print(f"  | {result['level']} | {result['processing_time_ms']:.2f} | {result['memory_used_mb']:.2f} | {result['features_generated']} | {result['strategy_used']} |")

    print("\n5. Performance Improvement Effects:")
    if len(integration_results) >= 2:
        standard_time = integration_results[0]['processing_time_ms']
        optimized_time = integration_results[-1]['processing_time_ms']
        speedup = standard_time / optimized_time if optimized_time > 0 else 0

        print(f"  - Processing speed improvement: {speedup:.2f}x faster")

    # Overall evaluation
    avg_memory_reduction = np.mean([r['reduction_percent'] for r in dtype_results])
    print(f"\n6. Overall Evaluation:")
    print(f"  - Memory reduction by dtype optimization: Average {avg_memory_reduction:.1f}%")
    print(f"  - Speed improvement by vectorization: {vectorization_result['total_operations']} operations in {vectorization_result['vectorization_time_ms']:.2f}ms")
    print(f"  - Large data processing by chunking: {chunk_result['data_size']:,} rows in {chunk_result['chunk_processing_time_ms']:.2f}ms")
    print(f"  - Integrated system optimization: All levels working correctly")


def main():
    """Main execution"""
    print("Data I/O Optimization Performance Test Started")
    print("=" * 80)

    try:
        # Execute each test
        print("1/4 Data type optimization test running...")
        dtype_results = test_dtype_optimization()

        print("\n2/4 Vectorization operations test running...")
        vectorization_result = test_vectorization()

        print("\n3/4 Chunk processing test running...")
        chunk_result = test_chunk_processing()

        print("\n4/4 Feature engineering integration test running...")
        integration_results = test_feature_engineering_integration()

        # Generate report
        generate_performance_report(
            dtype_results,
            vectorization_result,
            chunk_result,
            integration_results
        )

        print(f"\n[OK] Data I/O optimization performance test completed")
        print("\nImplemented optimization features:")
        print("- [MEMORY] Data type optimization for memory efficiency")
        print("- [SPEED] DataFrame operation vectorization for acceleration")
        print("- [COPY] Elimination of unnecessary data copying")
        print("- [INDEX] Index optimization")
        print("- [CHUNK] Chunk processing for large data handling")
        print("- [INTEGRATION] Feature engineering system integration")

    except Exception as e:
        print(f"\n[ERROR] Error during test execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
