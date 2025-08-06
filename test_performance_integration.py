#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
パフォーマンス最適化統合�EチE��トスクリプト

Phase 2: パフォーマンス最適化�Eロジェクト�E動作確誁E既存コンポ�Eネントとパフォーマンス最適化モジュールの統合テスチE"""

import os
import sys
import time
import traceback
from pathlib import Path

# Windows環墁E��のUTF-8エンコーチE��ング対忁Eif sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.utils.performance_config import get_performance_config, set_performance_config
from src.day_trade.utils.optimized_pandas import (
    optimize_dataframe_dtypes,
    vectorized_technical_indicators,
    get_optimized_processor
)
from src.day_trade.models.database import create_database_manager
from src.day_trade.analysis.feature_engineering import AdvancedFeatureEngineer


def create_test_data(rows: int = 10000) -> pd.DataFrame:
    """チE��ト用の価格チE�Eタを生戁E""
    print(f"チE��トデータ生�E中�E�Erows:,}行！E..")

    dates = pd.date_range(start='2020-01-01', periods=rows, freq='D')
    np.random.seed(42)

    # リアルな価格チE�Eタのシミュレーション
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, rows)  # 平坁E.1%、標準偏差2%のリターン
    prices = base_price * np.cumprod(1 + returns)

    # OHLCV チE�Eタ生�E
    high_multiplier = 1 + np.abs(np.random.normal(0, 0.01, rows))
    low_multiplier = 1 - np.abs(np.random.normal(0, 0.01, rows))

    data = pd.DataFrame({
        'Open': prices * np.random.uniform(0.98, 1.02, rows),
        'High': prices * high_multiplier,
        'Low': prices * low_multiplier,
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, rows)
    }, index=dates)

    print(f"チE��トデータ生�E完亁E {data.shape}")
    return data


def test_performance_config():
    """パフォーマンス設定�EチE��チE""
    print("\n=== パフォーマンス設定テスチE===")

    try:
        config = get_performance_config()
        print(f"最適化レベル: {config.optimization_level}")
        print(f"チE�Eタベ�Eス池サイズ: {config.database.pool_size}")
        print(f"計算最大ワーカー数: {config.compute.max_workers}")
        print(f"キャチE��ュL1サイズ: {config.cache.l1_cache_size}")
        print("[OK] パフォーマンス設定テスト�E劁E)
        return True
    except Exception as e:
        print(f"[NG] パフォーマンス設定テスト失敁E {e}")
        traceback.print_exc()
        return False


def test_optimized_pandas():
    """最適化されたpandas処琁E�EチE��チE""
    print("\n=== 最適化pandas処琁E��スチE===")

    try:
        # チE��トデータ作�E
        test_data = create_test_data(5000)
        original_memory = test_data.memory_usage(deep=True).sum()

        # チE�Eタ型最適化テスチE        start_time = time.time()
        optimized_data = optimize_dataframe_dtypes(test_data)
        optimization_time = time.time() - start_time

        optimized_memory = optimized_data.memory_usage(deep=True).sum()
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100

        print(f"チE�Eタ型最適化時閁E {optimization_time:.3f}私E)
        print(f"メモリ使用量削渁E {memory_reduction:.1f}%")

        # ベクトル化テクニカル持E��テスチE        start_time = time.time()
        technical_data = vectorized_technical_indicators(
            optimized_data,
            price_col='Close',
            volume_col='Volume'
        )
        technical_time = time.time() - start_time

        print(f"チE��ニカル持E��計算時閁E {technical_time:.3f}私E)
        print(f"追加された指標数: {len(technical_data.columns) - len(test_data.columns)}")

        # 最適化�EロセチE��ーチE��チE        processor = get_optimized_processor()
        start_time = time.time()
        processed_data = processor.optimize_for_computation(test_data)
        processor_time = time.time() - start_time

        print(f"計算最適化時閁E {processor_time:.3f}私E)
        print("✁E最適化pandas処琁E��スト�E劁E)
        return True

    except Exception as e:
        print(f"✁E最適化pandas処琁E��スト失敁E {e}")
        traceback.print_exc()
        return False


def test_optimized_database():
    """最適化されたチE�Eタベ�Eス処琁E�EチE��チE""
    print("\n=== 最適化データベ�Eス処琁E��スチE===")

    try:
        # パフォーマンス最適化版チE�Eタベ�Eスマネージャー作�E
        db_manager = create_database_manager(use_performance_optimization=True)

        # チE�Eタベ�Eスマネージャーのタイプ確誁E        db_type = type(db_manager).__name__
        print(f"チE�Eタベ�Eスマネージャー: {db_type}")

        # パフォーマンス統計取得（最適化版の場合�Eみ�E�E        if hasattr(db_manager, 'get_performance_stats'):
            stats = db_manager.get_performance_stats()
            print(f"接続�Eール設宁E pool_size={stats['config']['pool_size']}")
            print(f"最適化レベル: {stats['config']['optimization_level']}")

        # 通常版との比輁E        normal_db_manager = create_database_manager(use_performance_optimization=False)
        normal_type = type(normal_db_manager).__name__
        print(f"通常版データベ�Eスマネージャー: {normal_type}")

        print("✁E最適化データベ�Eス処琁E��スト�E劁E)
        return True

    except Exception as e:
        print(f"✁E最適化データベ�Eス処琁E��スト失敁E {e}")
        traceback.print_exc()
        return False


def test_optimized_feature_engineering():
    """最適化された特徴量エンジニアリングのチE��チE""
    print("\n=== 最適化特徴量エンジニアリングチE��チE===")

    try:
        # チE��トデータ作�E
        test_data = create_test_data(3000)

        # 特徴量エンジニア作�E
        feature_engineer = AdvancedFeatureEngineer()

        # 最適化版特徴量生戁E        start_time = time.time()
        features = feature_engineer.generate_all_features(
            price_data=test_data,
            volume_data=test_data['Volume']
        )
        generation_time = time.time() - start_time

        print(f"特徴量生成時閁E {generation_time:.3f}私E)
        print(f"生�Eされた特徴量数: {len(features.columns)}")
        print(f"出力データ形状: {features.shape}")
        print(f"メモリ使用釁E {features.memory_usage(deep=True).sum() / 1024**2:.1f}MB")

        # チE�Eタ品質確誁E        nan_ratio = features.isnull().sum().sum() / (features.shape[0] * features.shape[1]) * 100
        print(f"欠損値比率: {nan_ratio:.2f}%")

        # チャンク処琁E�EチE��チE        if hasattr(feature_engineer, 'generate_features_chunked'):
            print("チャンク処琁E���Eも利用可能")

        print("✁E最適化特徴量エンジニアリングチE��ト�E劁E)
        return True

    except Exception as e:
        print(f"✁E最適化特徴量エンジニアリングチE��ト失敁E {e}")
        traceback.print_exc()
        return False


def run_performance_comparison():
    """パフォーマンス比輁E��スチE""
    print("\n=== パフォーマンス比輁E��スチE===")

    try:
        test_sizes = [1000, 5000, 10000]

        for size in test_sizes:
            print(f"\n--- チE�Eタサイズ: {size:,}衁E---")
            test_data = create_test_data(size)

            # 従来版�E処琁E��間（簡易シミュレーション�E�E            start_time = time.time()

            # 基本皁E��チE��ニカル持E��計算（従来版シミュレーション�E�E            basic_sma = test_data['Close'].rolling(20).mean()
            basic_rsi_delta = test_data['Close'].diff()
            basic_rsi_gain = basic_rsi_delta.where(basic_rsi_delta > 0, 0)
            basic_rsi_loss = -basic_rsi_delta.where(basic_rsi_delta < 0, 0)

            conventional_time = time.time() - start_time

            # 最適化版の処琁E��閁E            start_time = time.time()
            optimized_data = vectorized_technical_indicators(test_data, 'Close', 'Volume')
            optimized_time = time.time() - start_time

            speedup = conventional_time / optimized_time if optimized_time > 0 else 0
            print(f"従来版�E琁E��閁E {conventional_time:.3f}私E)
            print(f"最適化版処琁E��閁E {optimized_time:.3f}私E)
            print(f"高速化倍率: {speedup:.1f}x")

        print("✁Eパフォーマンス比輁E��スト�E劁E)
        return True

    except Exception as e:
        print(f"✁Eパフォーマンス比輁E��スト失敁E {e}")
        traceback.print_exc()
        return False


def main():
    """メインチE��ト実衁E""
    print("パフォーマンス最適化統合テスト開姁E)
    print("=" * 50)

    test_results = []

    # 吁E��スト�E実衁E    test_functions = [
        ("パフォーマンス設宁E, test_performance_config),
        ("最適化pandas処琁E, test_optimized_pandas),
        ("最適化データベ�Eス処琁E, test_optimized_database),
        ("最適化特徴量エンジニアリング", test_optimized_feature_engineering),
        ("パフォーマンス比輁E, run_performance_comparison),
    ]

    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"✁E{test_name}でエラー発甁E {e}")
            test_results.append((test_name, False))
            traceback.print_exc()

    # 結果サマリー
    print("\n" + "=" * 50)
    print("チE��ト結果サマリー")
    print("=" * 50)

    passed = 0
    for test_name, result in test_results:
        status = "✁E成功" if result else "✁E失敁E
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n成功玁E {passed}/{len(test_results)} ({passed/len(test_results)*100:.1f}%)")

    if passed == len(test_results):
        print("\n🎉 すべてのチE��トが成功しました�E�E)
        print("パフォーマンス最適化�E統合が正常に完亁E��てぁE��す、E)
    else:
        print(f"\n⚠�E�E {len(test_results) - passed}個�EチE��トが失敗しました、E)
        print("詳細なエラー惁E��を確認して修正してください、E)

    return passed == len(test_results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
