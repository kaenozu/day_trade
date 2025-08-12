#!/usr/bin/env python3
"""
パフォーマンス最適化統合テスト
Phase 2: パフォーマンス最適化プロジェクト
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_performance_config():
    """パフォーマンス設定のテスト"""
    print("\n=== パフォーマンス設定テスト ===")

    try:
        from src.day_trade.utils.performance_config import get_performance_config

        config = get_performance_config()
        print(f"最適化レベル: {config.optimization_level}")
        print(f"データベース池サイズ: {config.database.pool_size}")
        print(f"計算最大ワーカー数: {config.compute.max_workers}")
        print("[OK] パフォーマンス設定テスト成功")
        return True
    except Exception as e:
        print(f"[NG] パフォーマンス設定テスト失敗: {e}")
        return False


def test_optimized_pandas():
    """最適化されたpandas処理のテスト"""
    print("\n=== 最適化pandas処理テスト ===")

    try:
        from src.day_trade.utils.optimized_pandas import (
            optimize_dataframe_dtypes,
            vectorized_technical_indicators,
        )

        # テストデータ作成
        dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 1000))

        test_data = pd.DataFrame(
            {
                "Open": prices * 0.99,
                "High": prices * 1.01,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": np.random.randint(100000, 1000000, 1000),
            },
            index=dates,
        )

        # データ型最適化テスト
        original_memory = test_data.memory_usage(deep=True).sum()
        optimized_data = optimize_dataframe_dtypes(test_data)
        optimized_memory = optimized_data.memory_usage(deep=True).sum()

        memory_reduction = (original_memory - optimized_memory) / original_memory * 100
        print(f"メモリ使用量削減: {memory_reduction:.1f}%")

        # ベクトル化テクニカル指標テスト
        start_time = time.time()
        technical_data = vectorized_technical_indicators(optimized_data, "Close", "Volume")
        calc_time = time.time() - start_time

        print(f"テクニカル指標計算時間: {calc_time:.3f}秒")
        print(f"追加された指標数: {len(technical_data.columns) - len(test_data.columns)}")
        print("[OK] 最適化pandas処理テスト成功")
        return True

    except Exception as e:
        print(f"[NG] 最適化pandas処理テスト失敗: {e}")
        return False


def test_optimized_database():
    """最適化されたデータベース処理のテスト"""
    print("\n=== 最適化データベース処理テスト ===")

    try:
        from src.day_trade.models.database import create_database_manager

        # パフォーマンス最適化版
        db_manager = create_database_manager(use_performance_optimization=True)
        db_type = type(db_manager).__name__
        print(f"最適化データベースマネージャー: {db_type}")

        # 通常版
        normal_db_manager = create_database_manager(use_performance_optimization=False)
        normal_type = type(normal_db_manager).__name__
        print(f"通常版データベースマネージャー: {normal_type}")

        print("[OK] 最適化データベース処理テスト成功")
        return True

    except Exception as e:
        print(f"[NG] 最適化データベース処理テスト失敗: {e}")
        return False


def test_optimized_feature_engineering():
    """最適化された特徴量エンジニアリングのテスト"""
    print("\n=== 最適化特徴量エンジニアリングテスト ===")

    try:
        from src.day_trade.analysis.feature_engineering import AdvancedFeatureEngineer

        # テストデータ作成
        dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 500))

        test_data = pd.DataFrame(
            {
                "Open": prices * 0.99,
                "High": prices * 1.01,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": np.random.randint(100000, 1000000, 500),
            },
            index=dates,
        )

        # 特徴量エンジニア作成
        feature_engineer = AdvancedFeatureEngineer()

        # 最適化版特徴量生成
        start_time = time.time()
        features = feature_engineer.generate_all_features(
            price_data=test_data, volume_data=test_data["Volume"]
        )
        generation_time = time.time() - start_time

        print(f"特徴量生成時間: {generation_time:.3f}秒")
        print(f"生成された特徴量数: {len(features.columns)}")
        print(f"出力データ形状: {features.shape}")

        print("[OK] 最適化特徴量エンジニアリングテスト成功")
        return True

    except Exception as e:
        print(f"[NG] 最適化特徴量エンジニアリングテスト失敗: {e}")
        return False


def main():
    """メインテスト実行"""
    print("パフォーマンス最適化統合テスト開始")
    print("=" * 50)

    test_functions = [
        ("パフォーマンス設定", test_performance_config),
        ("最適化pandas処理", test_optimized_pandas),
        ("最適化データベース処理", test_optimized_database),
        ("最適化特徴量エンジニアリング", test_optimized_feature_engineering),
    ]

    passed = 0

    for test_name, test_func in test_functions:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"[ERROR] {test_name}でエラー発生: {e}")

    print("\n" + "=" * 50)
    print("テスト結果サマリー")
    print("=" * 50)

    print(f"成功率: {passed}/{len(test_functions)} ({passed/len(test_functions)*100:.1f}%)")

    if passed == len(test_functions):
        print("\n[SUCCESS] すべてのテストが成功しました！")
        print("パフォーマンス最適化の統合が正常に完了しています。")
    else:
        print(f"\n[WARNING] {len(test_functions) - passed}個のテストが失敗しました。")

    return passed == len(test_functions)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
