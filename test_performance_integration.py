#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
パフォーマンス最適化統合テストスクリプト

Phase 2: パフォーマンス最適化プロジェクトの動作確認
既存コンポーネントとパフォーマンス最適化モジュールの統合テスト
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Windows環境でのUTF-8エンコーディング対応
if sys.platform.startswith('win'):
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
    """テスト用の価格データを生成"""
    print(f"テストデータ生成中（{rows:,}行）...")

    dates = pd.date_range(start='2020-01-01', periods=rows, freq='D')
    np.random.seed(42)

    # リアルな価格データのシミュレーション
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, rows)  # 平均0.1%、標準偏差2%のリターン
    prices = base_price * np.cumprod(1 + returns)

    # OHLCV データ生成
    high_multiplier = 1 + np.abs(np.random.normal(0, 0.01, rows))
    low_multiplier = 1 - np.abs(np.random.normal(0, 0.01, rows))

    data = pd.DataFrame({
        'Open': prices * np.random.uniform(0.98, 1.02, rows),
        'High': prices * high_multiplier,
        'Low': prices * low_multiplier,
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, rows)
    }, index=dates)

    print(f"テストデータ生成完了: {data.shape}")
    return data


def test_performance_config():
    """パフォーマンス設定のテスト"""
    print("\n=== パフォーマンス設定テスト ===")

    try:
        config = get_performance_config()
        print(f"最適化レベル: {config.optimization_level}")
        print(f"データベース池サイズ: {config.database.pool_size}")
        print(f"計算最大ワーカー数: {config.compute.max_workers}")
        print(f"キャッシュL1サイズ: {config.cache.l1_cache_size}")
        print("[OK] パフォーマンス設定テスト成功")
        return True
    except Exception as e:
        print(f"[NG] パフォーマンス設定テスト失敗: {e}")
        traceback.print_exc()
        return False


def test_optimized_pandas():
    """最適化されたpandas処理のテスト"""
    print("\n=== 最適化pandas処理テスト ===")

    try:
        # テストデータ作成
        test_data = create_test_data(5000)
        original_memory = test_data.memory_usage(deep=True).sum()

        # データ型最適化テスト
        start_time = time.time()
        optimized_data = optimize_dataframe_dtypes(test_data)
        optimization_time = time.time() - start_time

        optimized_memory = optimized_data.memory_usage(deep=True).sum()
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100

        print(f"データ型最適化時間: {optimization_time:.3f}秒")
        print(f"メモリ使用量削減: {memory_reduction:.1f}%")

        # ベクトル化テクニカル指標テスト
        start_time = time.time()
        technical_data = vectorized_technical_indicators(
            optimized_data,
            price_col='Close',
            volume_col='Volume'
        )
        technical_time = time.time() - start_time

        print(f"テクニカル指標計算時間: {technical_time:.3f}秒")
        print(f"追加された指標数: {len(technical_data.columns) - len(test_data.columns)}")

        # 最適化プロセッサーテスト
        processor = get_optimized_processor()
        start_time = time.time()
        processed_data = processor.optimize_for_computation(test_data)
        processor_time = time.time() - start_time

        print(f"計算最適化時間: {processor_time:.3f}秒")
        print("[OK] 最適化pandas処理テスト成功")
        return True

    except Exception as e:
        print(f"[NG] 最適化pandas処理テスト失敗: {e}")
        traceback.print_exc()
        return False


def test_optimized_database():
    """最適化されたデータベース処理のテスト"""
    print("\n=== 最適化データベース処理テスト ===")

    try:
        # パフォーマンス最適化版データベースマネージャー作成
        db_manager = create_database_manager(use_performance_optimization=True)

        # データベースマネージャーのタイプ確認
        db_type = type(db_manager).__name__
        print(f"データベースマネージャー: {db_type}")

        # パフォーマンス統計取得（最適化版の場合のみ）
        if hasattr(db_manager, 'get_performance_stats'):
            stats = db_manager.get_performance_stats()
            print(f"接続プール設定: pool_size={stats['config']['pool_size']}")
            print(f"最適化レベル: {stats['config']['optimization_level']}")

        # 通常版との比較
        normal_db_manager = create_database_manager(use_performance_optimization=False)
        normal_type = type(normal_db_manager).__name__
        print(f"通常版データベースマネージャー: {normal_type}")

        print("[OK] 最適化データベース処理テスト成功")
        return True

    except Exception as e:
        print(f"[NG] 最適化データベース処理テスト失敗: {e}")
        traceback.print_exc()
        return False


def test_optimized_feature_engineering():
    """最適化された特徴量エンジニアリングのテスト"""
    print("\n=== 最適化特徴量エンジニアリングテスト ===")

    try:
        # テストデータ作成
        test_data = create_test_data(3000)

        # 特徴量エンジニア作成
        feature_engineer = AdvancedFeatureEngineer()

        # 最適化版特徴量生成
        start_time = time.time()
        features = feature_engineer.generate_all_features(
            price_data=test_data,
            volume_data=test_data['Volume']
        )
        generation_time = time.time() - start_time

        print(f"特徴量生成時間: {generation_time:.3f}秒")
        print(f"生成された特徴量数: {len(features.columns)}")
        print(f"出力データ形状: {features.shape}")
        print(f"メモリ使用量: {features.memory_usage(deep=True).sum() / 1024**2:.1f}MB")

        # データ品質確認
        nan_ratio = features.isnull().sum().sum() / (features.shape[0] * features.shape[1]) * 100
        print(f"欠損値比率: {nan_ratio:.2f}%")

        # チャンク処理のテスト
        if hasattr(feature_engineer, 'generate_features_chunked'):
            print("チャンク処理機能も利用可能")

        print("[OK] 最適化特徴量エンジニアリングテスト成功")
        return True

    except Exception as e:
        print(f"[NG] 最適化特徴量エンジニアリングテスト失敗: {e}")
        traceback.print_exc()
        return False


def run_performance_comparison():
    """パフォーマンス比較テスト"""
    print("\n=== パフォーマンス比較テスト ===")

    try:
        test_sizes = [1000, 5000, 10000]

        for size in test_sizes:
            print(f"\n--- データサイズ: {size:,}行 ---")
            test_data = create_test_data(size)

            # 従来版の処理時間（簡易シミュレーション）
            start_time = time.time()

            # 基本的なテクニカル指標計算（従来版シミュレーション）
            basic_sma = test_data['Close'].rolling(20).mean()
            basic_rsi_delta = test_data['Close'].diff()
            basic_rsi_gain = basic_rsi_delta.where(basic_rsi_delta > 0, 0)
            basic_rsi_loss = -basic_rsi_delta.where(basic_rsi_delta < 0, 0)

            conventional_time = time.time() - start_time

            # 最適化版の処理時間
            start_time = time.time()
            optimized_data = vectorized_technical_indicators(test_data, 'Close', 'Volume')
            optimized_time = time.time() - start_time

            speedup = conventional_time / optimized_time if optimized_time > 0 else 0
            print(f"従来版処理時間: {conventional_time:.3f}秒")
            print(f"最適化版処理時間: {optimized_time:.3f}秒")
            print(f"高速化倍率: {speedup:.1f}x")

        print("[OK] パフォーマンス比較テスト成功")
        return True

    except Exception as e:
        print(f"[NG] パフォーマンス比較テスト失敗: {e}")
        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("パフォーマンス最適化統合テスト開始")
    print("=" * 50)

    test_results = []

    # 各テストの実行
    test_functions = [
        ("パフォーマンス設定", test_performance_config),
        ("最適化pandas処理", test_optimized_pandas),
        ("最適化データベース処理", test_optimized_database),
        ("最適化特徴量エンジニアリング", test_optimized_feature_engineering),
        ("パフォーマンス比較", run_performance_comparison),
    ]

    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"[NG] {test_name}でエラー発生: {e}")
            test_results.append((test_name, False))
            traceback.print_exc()

    # 結果サマリー
    print("\n" + "=" * 50)
    print("テスト結果サマリー")
    print("=" * 50)

    passed = 0
    for test_name, result in test_results:
        status = "[OK] 成功" if result else "[NG] 失敗"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n成功率: {passed}/{len(test_results)} ({passed/len(test_results)*100:.1f}%)")

    if passed == len(test_results):
        print("\n🎉 すべてのテストが成功しました！")
        print("パフォーマンス最適化の統合が正常に完了しています。")
    else:
        print(f"\n⚠️  {len(test_results) - passed}個のテストが失敗しました。")
        print("詳細なエラー情報を確認して修正してください。")

    return passed == len(test_results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
