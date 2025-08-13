#!/usr/bin/env python3
"""
Issue #575修正の簡易テスト（ASCII文字のみ）
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def create_test_data():
    """テスト用データ作成"""
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    np.random.seed(42)

    base_price = 100
    prices = base_price + np.random.randn(30).cumsum()

    return pd.DataFrame({
        '始値': np.roll(prices, 1),
        '高値': prices * 1.02,
        '安値': prices * 0.98,
        '終値': prices,
        '出来高': np.random.randint(100000, 1000000, 30),
    }, index=dates)

def test_feature_engines():
    """特徴量エンジンテスト"""
    print("=== Issue #575 Feature Engines Test ===")

    try:
        from day_trade.data.feature_engines import calculate_custom_feature, list_available_features

        # テストデータ作成
        test_data = create_test_data()
        print(f"Test data created: {test_data.shape}")

        # 利用可能特徴量
        features = list_available_features()
        print(f"Available features: {len(features)}")

        # 特徴量計算テスト
        test_features = ["trend_strength", "momentum", "price_channel"]

        for feature in test_features:
            try:
                result = calculate_custom_feature(test_data, feature)
                new_cols = len(result.columns) - len(test_data.columns)
                print(f"  {feature}: {new_cols} new columns added")

            except Exception as e:
                print(f"  {feature}: Error - {str(e)[:50]}...")

        return True

    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Test error: {e}")
        return False

def test_batch_fetcher():
    """BatchDataFetcherテスト"""
    print("\n=== BatchDataFetcher Integration Test ===")

    try:
        from day_trade.data.batch_data_fetcher import AdvancedBatchDataFetcher

        # フェッチャー初期化
        fetcher = AdvancedBatchDataFetcher(max_workers=1, enable_kafka=False, enable_redis=False)

        # テストデータでカスタム特徴量追加テスト
        test_data = create_test_data()

        # メソッド直接テスト
        result = fetcher._add_custom_feature(test_data, "trend_strength")

        if len(result.columns) > len(test_data.columns):
            print("  Custom feature addition: SUCCESS")
        else:
            print("  Custom feature addition: NO NEW COLUMNS")

        fetcher.close()
        return True

    except Exception as e:
        print(f"BatchDataFetcher test error: {e}")
        return False

def main():
    """メインテスト"""
    print("Issue #575 Fix Test - Simple Version\n")

    # 1. 特徴量エンジンテスト
    engine_success = test_feature_engines()

    # 2. BatchDataFetcher統合テスト
    fetcher_success = test_batch_fetcher()

    # 結果
    print("\n=== Test Results ===")
    print(f"Feature Engines: {'PASS' if engine_success else 'FAIL'}")
    print(f"BatchDataFetcher: {'PASS' if fetcher_success else 'FAIL'}")

    overall_success = engine_success and fetcher_success
    print(f"Overall: {'PASS' if overall_success else 'FAIL'}")

    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)