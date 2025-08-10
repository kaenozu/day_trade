#!/usr/bin/env python3
"""
データI/O最適化パフォーマンステスト
Issue #378: データI/Oとデータ処理の最適化 - データ構造と操作の効率化

実装されたデータ最適化機能のパフォーマンステスト:
1. データ型最適化によるメモリ効率向上
2. DataFrame操作のベクトル化で高速化
3. 不必要なデータコピーの回避
4. インデックスの最適化
5. チャンク処理による大規模データ対応
"""

import time
import gc
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import psutil
import os

# ログ出力を有効化
import logging
logging.basicConfig(level=logging.INFO)

from src.day_trade.utils.data_optimization import (
    DataFrameOptimizer,
    ChunkedDataProcessor,
    create_optimized_dataframe,
    benchmark_dataframe_operations,
    memory_monitor
)
from src.day_trade.analysis.feature_engineering_unified import (
    FeatureEngineeringManager,
    FeatureConfig,
    generate_features
)
from src.day_trade.core.optimization_strategy import OptimizationConfig, OptimizationLevel


def create_test_dataset(rows: int = 10000, add_noise: bool = True) -> pd.DataFrame:
    """テスト用データセット作成"""
    print(f"テストデータ作成中: {rows}行")

    # 基本的な株価データを模擬
    np.random.seed(42)  # 再現性のため

    # 日付インデックス
    dates = pd.date_range('2020-01-01', periods=rows, freq='1min')

    # 株価データ
    base_price = 1000.0
    returns = np.random.normal(0, 0.02, rows)  # 2%の標準偏差
    prices = base_price * np.cumprod(1 + returns)

    # ノイズを追加して実際のデータに近づける
    if add_noise:
        noise = np.random.normal(0, 0.001, rows)
        prices = prices * (1 + noise)

    # 高値・安値・出来高の生成
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, rows)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, rows)))
    volumes = np.random.lognormal(10, 1, rows).astype(int)

    # DataFrame作成（メモリ効率の悪いデータ型を意図的に使用）
    data = pd.DataFrame({
        'timestamp': dates,
        '始値': prices.astype('float64'),      # 最適化前: float64
        '高値': highs.astype('float64'),       # 最適化前: float64
        '安値': lows.astype('float64'),        # 最適化前: float64
        '終値': prices.astype('float64'),      # 最適化前: float64
        '出来高': volumes.astype('int64'),     # 最適化前: int64
        '市場区分': ['東証1部'] * rows,        # 最適化前: object
        'セクター': np.random.choice(['金融', 'IT', '製造', '小売'], rows),  # category候補
    })

    return data


def test_dtype_optimization_performance():
    """データ型最適化のパフォーマンステスト"""
    print("\n=== データ型最適化パフォーマンステスト ===")

    test_sizes = [1000, 5000, 10000, 50000]
    results = []

    for size in test_sizes:
        print(f"\nテストデータ크기: {size}行")

        # テストデータ作成
        data = create_test_dataset(size)
        memory_before = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB

        # 最適化実行
        optimizer = DataFrameOptimizer()
        start_time = time.perf_counter()

        optimized_data = optimizer.optimize_dtypes(data, copy=True)

        optimization_time = time.perf_counter() - start_time
        memory_after = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        memory_saved = memory_before - memory_after
        reduction_percent = (memory_saved / memory_before) * 100

        # 最適化統計の取得
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

        print(f"  メモリ使用量: {memory_before:.3f}MB → {memory_after:.3f}MB")
        print(f"  メモリ削減: {memory_saved:.3f}MB ({reduction_percent:.1f}%)")
        print(f"  最適化時間: {optimization_time*1000:.2f}ms")
        print(f"  最適化適用数: {optimization_stats['dtype_optimizations']}")

    return results


def test_vectorization_performance():
    """ベクトル化操作のパフォーマンステスト"""
    print("\n=== ベクトル化操作パフォーマンステスト ===")

    data = create_test_dataset(10000)
    optimizer = DataFrameOptimizer()

    # テクニカル指標操作の定義
    technical_operations = [
        {'type': 'technical_indicator', 'indicator': 'sma', 'column': '終値', 'period': 20},
        {'type': 'technical_indicator', 'indicator': 'ema', 'column': '終値', 'period': 12},
        {'type': 'technical_indicator', 'indicator': 'rsi', 'column': '終値', 'period': 14},
        {'type': 'technical_indicator', 'indicator': 'bollinger_bands', 'column': '終値', 'period': 20},
        {'type': 'technical_indicator', 'indicator': 'macd', 'column': '終値'},
    ]

    rolling_operations = [
        {'type': 'rolling_calculation', 'column': '終値', 'window': 20, 'calculation': 'std'},
        {'type': 'rolling_calculation', 'column': '終値', 'window': 50, 'calculation': 'min'},
        {'type': 'rolling_calculation', 'column': '終値', 'window': 50, 'calculation': 'max'},
        {'type': 'rolling_calculation', 'column': '出来高', 'window': 20, 'calculation': 'mean'},
    ]

    mathematical_operations = [
        {'type': 'mathematical_operation', 'operation': 'percentage_change', 'columns': ['終値'], 'result_column': '終値_変化率'},
        {'type': 'mathematical_operation', 'operation': 'log_return', 'columns': ['終値'], 'result_column': '終値_対数収益'},
        {'type': 'mathematical_operation', 'operation': 'z_score', 'columns': ['終値'], 'window': 20, 'result_column': '終値_Zスコア'},
    ]

    all_operations = technical_operations + rolling_operations + mathematical_operations

    print(f"操作数: {len(all_operations)}")

    # ベクトル化実行
    start_time = time.perf_counter()

    result_data = optimizer.vectorize_operations(data, all_operations)

    vectorization_time = time.perf_counter() - start_time

    # 結果確認
    original_cols = len(data.columns)
    result_cols = len(result_data.columns)
    added_features = result_cols - original_cols

    print(f"  元のカラム数: {original_cols}")
    print(f"  結果カラム数: {result_cols}")
    print(f"  追加された特徴量: {added_features}")
    print(f"  ベクトル化時間: {vectorization_time*1000:.2f}ms")
    print(f"  操作あたり平均時間: {vectorization_time*1000/len(all_operations):.2f}ms")

    # 最適化統計
    optimization_stats = optimizer.get_optimization_stats()
    print(f"  ベクトル化適用数: {optimization_stats['vectorizations']}")

    return {
        'total_operations': len(all_operations),
        'vectorization_time_ms': round(vectorization_time * 1000, 2),
        'features_added': added_features,
        'avg_time_per_operation_ms': round(vectorization_time * 1000 / len(all_operations), 2),
        'vectorizations_applied': optimization_stats['vectorizations']
    }


def test_chunk_processing_performance():
    """チャンク処理パフォーマンステスト"""
    print("\n=== チャンク処理パフォーマンステスト ===")

    # 大規模データでのテスト
    large_data = create_test_dataset(50000)  # 50,000行

    chunk_processor = ChunkedDataProcessor(chunk_size=10000)

    def simple_feature_calculation(chunk_data: pd.DataFrame) -> pd.DataFrame:
        """シンプルな特徴量計算"""
        result = chunk_data.copy()
        result['SMA_20'] = chunk_data['終値'].rolling(window=20, min_periods=1).mean()
        result['Returns'] = chunk_data['終値'].pct_change()
        result['Volatility_10'] = result['Returns'].rolling(window=10, min_periods=1).std()
        return result

    print(f"大規模データ処理テスト: {len(large_data)}行")

    # チャンク処理実行
    start_time = time.perf_counter()

    processed_data = chunk_processor.process_large_dataframe(
        large_data, simple_feature_calculation
    )

    chunk_processing_time = time.perf_counter() - start_time

    print(f"  チャンク処理時間: {chunk_processing_time*1000:.2f}ms")
    print(f"  処理済みデータ形状: {processed_data.shape}")
    print(f"  行あたり処理時間: {chunk_processing_time*1000/len(processed_data):.4f}ms")

    return {
        'data_size': len(large_data),
        'chunk_processing_time_ms': round(chunk_processing_time * 1000, 2),
        'processed_shape': processed_data.shape,
        'time_per_row_ms': round(chunk_processing_time * 1000 / len(processed_data), 4)
    }


def test_feature_engineering_integration():
    """特徴量エンジニアリング統合テスト"""
    print("\n=== 特徴量エンジニアリング統合テスト ===")

    test_data = create_test_dataset(5000)

    # 各最適化レベルでのテスト
    optimization_levels = [
        (OptimizationLevel.STANDARD, "標準"),
        (OptimizationLevel.OPTIMIZED, "最適化"),
        (OptimizationLevel.ADAPTIVE, "データ最適化"),
    ]

    results = []

    for level, level_name in optimization_levels:
        print(f"\n{level_name}レベルテスト:")

        # 設定
        opt_config = OptimizationConfig(
            enable_caching=True,
            cache_size_limit=1000,
            performance_monitoring=True,
            optimization_level=level
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

        # パフォーマンス測定
        start_time = time.perf_counter()
        memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        # 特徴量生成実行
        feature_result = generate_features(
            test_data,
            feature_config=feature_config,
            optimization_config=opt_config
        )

        processing_time = time.perf_counter() - start_time
        memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # 結果記録
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

        print(f"  処理時間: {processing_time*1000:.2f}ms")
        print(f"  メモリ使用量: {memory_used:.2f}MB")
        print(f"  入力形状: {test_data.shape}")
        print(f"  出力形状: {feature_result.features.shape}")
        print(f"  生成特徴量数: {len(feature_result.feature_names)}")
        print(f"  使用戦略: {feature_result.strategy_used}")

        # ガベージコレクション
        del feature_result
        gc.collect()

    return results


def generate_performance_report(dtype_results, vectorization_result, chunk_result, integration_results):
    """パフォーマンスレポート生成"""
    print("\n" + "="*80)
    print("データI/O最適化パフォーマンステストレポート")
    print("Issue #378: データI/Oとデータ処理の最適化 - データ構造と操作の効率化")
    print("="*80)

    print("\n1. データ型最適化結果:")
    print("  | データサイズ | 最適化前(MB) | 最適化後(MB) | 削減(MB) | 削減率(%) | 処理時間(ms) |")
    print("  |---|---|---|---|---|---|")
    for result in dtype_results:
        print(f"  | {result['size']:,}行 | {result['memory_before_mb']:.3f} | {result['memory_after_mb']:.3f} | {result['memory_saved_mb']:.3f} | {result['reduction_percent']:.1f} | {result['optimization_time_ms']:.2f} |")

    print("\n2. ベクトル化操作結果:")
    print(f"  - 総操作数: {vectorization_result['total_operations']}")
    print(f"  - 処理時間: {vectorization_result['vectorization_time_ms']:.2f}ms")
    print(f"  - 生成特徴量数: {vectorization_result['features_added']}")
    print(f"  - 操作あたり平均時間: {vectorization_result['avg_time_per_operation_ms']:.2f}ms")

    print("\n3. チャンク処理結果:")
    print(f"  - データサイズ: {chunk_result['data_size']:,}行")
    print(f"  - 処理時間: {chunk_result['chunk_processing_time_ms']:.2f}ms")
    print(f"  - 行あたり処理時間: {chunk_result['time_per_row_ms']:.4f}ms")

    print("\n4. 特徴量エンジニアリング統合結果:")
    print("  | 最適化レベル | 処理時間(ms) | メモリ使用(MB) | 特徴量数 | 戦略 |")
    print("  |---|---|---|---|---|")
    for result in integration_results:
        print(f"  | {result['level']} | {result['processing_time_ms']:.2f} | {result['memory_used_mb']:.2f} | {result['features_generated']} | {result['strategy_used']} |")

    print("\n5. パフォーマンス改善効果:")
    if len(integration_results) >= 2:
        standard_time = integration_results[0]['processing_time_ms']
        optimized_time = integration_results[-1]['processing_time_ms']
        speedup = standard_time / optimized_time if optimized_time > 0 else 0

        standard_memory = integration_results[0]['memory_used_mb']
        optimized_memory = integration_results[-1]['memory_used_mb']
        memory_improvement = (standard_memory - optimized_memory) / standard_memory * 100 if standard_memory > 0 else 0

        print(f"  - 処理速度向上: {speedup:.2f}x 高速化")
        print(f"  - メモリ効率向上: {memory_improvement:.1f}% 削減")

    # 総合評価
    avg_memory_reduction = np.mean([r['reduction_percent'] for r in dtype_results])
    print(f"\n6. 総合評価:")
    print(f"  - データ型最適化によるメモリ削減: 平均{avg_memory_reduction:.1f}%")
    print(f"  - ベクトル化操作による高速化: {vectorization_result['total_operations']}操作を{vectorization_result['vectorization_time_ms']:.2f}msで実行")
    print(f"  - チャンク処理による大規模データ対応: {chunk_result['data_size']:,}行を{chunk_result['chunk_processing_time_ms']:.2f}msで処理")
    print(f"  - 統合システムの最適化効果: 全レベルで正常動作確認")


def main():
    """メイン実行"""
    print("データI/O最適化パフォーマンステスト開始")
    print("=" * 80)

    try:
        # 各テストの実行
        print("1/4 データ型最適化テスト実行中...")
        dtype_results = test_dtype_optimization_performance()

        print("\n2/4 ベクトル化操作テスト実行中...")
        vectorization_result = test_vectorization_performance()

        print("\n3/4 チャンク処理テスト実行中...")
        chunk_result = test_chunk_processing_performance()

        print("\n4/4 特徴量エンジニアリング統合テスト実行中...")
        integration_results = test_feature_engineering_integration()

        # レポート生成
        generate_performance_report(
            dtype_results,
            vectorization_result,
            chunk_result,
            integration_results
        )

        print(f"\n✅ データI/O最適化パフォーマンステスト完了")
        print("\n実装された最適化機能:")
        print("- [MEMORY] データ型最適化でメモリ効率向上")
        print("- [SPEED] DataFrame操作のベクトル化で高速化")
        print("- [COPY] 不必要なデータコピーの回避")
        print("- [INDEX] インデックスの最適化")
        print("- [CHUNK] チャンク処理による大規模データ対応")
        print("- [INTEGRATION] 特徴量エンジニアリングシステム統合")

    except Exception as e:
        print(f"\n❌ テスト実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
