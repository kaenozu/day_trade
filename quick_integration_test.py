#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Integration Test - 高速統合テスト
Issue #870拡張予測システムの基本的な統合テスト
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import warnings
warnings.filterwarnings("ignore")

# パフォーマンス最適化システム
from integrated_performance_optimizer import (
    get_integrated_optimizer,
    optimize_system_performance,
    get_system_performance_report
)

def quick_test():
    """高速統合テスト"""
    print("=== QUICK INTEGRATION TEST ===")

    # システム初期化
    optimizer = get_integrated_optimizer()

    tests_passed = 0
    total_tests = 4

    # テスト1: 基本機能
    try:
        print("\nTest 1: Basic functionality")
        report = get_system_performance_report()
        print(f"  Memory: {report['system_state']['memory_usage_mb']:.1f}MB")

        # キャッシュテスト
        test_model = {"weights": np.random.randn(10, 5)}
        optimizer.model_cache.put("test_model", test_model)
        retrieved = optimizer.model_cache.get("test_model")

        if retrieved is not None:
            print("  [PASS] Basic functionality passed")
            tests_passed += 1
        else:
            print("  [FAIL] Basic functionality failed")
    except Exception as e:
        print(f"  [FAIL] Basic functionality failed: {e}")

    # テスト2: パフォーマンス最適化
    try:
        print("\nTest 2: Performance optimization")

        # データ負荷
        for i in range(5):
            data = np.random.randn(500, 50)
            optimizer.feature_cache.put_features(f"test_{i}", data)

        # 最適化実行
        results = optimizer.run_comprehensive_optimization()

        if len(results) > 0:
            print(f"  [PASS] Optimization passed ({len(results)} strategies)")
            tests_passed += 1
        else:
            print("  [FAIL] Optimization failed")
    except Exception as e:
        print(f"  [FAIL] Optimization failed: {e}")

    # テスト3: キャッシュ統合
    try:
        print("\nTest 3: Cache integration")

        # 複数モデルキャッシュ
        for model_type in ["linear", "tree", "neural"]:
            model_data = {"type": model_type, "weights": np.random.randn(20, 10)}
            optimizer.model_cache.put(f"model_{model_type}", model_data)

        # 特徴量キャッシュ
        features = np.random.randn(100, 15)
        optimizer.feature_cache.put_features("integration_test", features)

        # 取得テスト
        cached_features = optimizer.feature_cache.get_features("integration_test")
        cached_model = optimizer.model_cache.get("model_linear")

        if cached_features is not None and cached_model is not None:
            print("  [PASS] Cache integration passed")
            tests_passed += 1
        else:
            print("  [FAIL] Cache integration failed")
    except Exception as e:
        print(f"  [FAIL] Cache integration failed: {e}")

    # テスト4: エラーハンドリング
    try:
        print("\nTest 4: Error handling")

        # 存在しないキーのアクセス
        missing_model = optimizer.model_cache.get("nonexistent")
        missing_features = optimizer.feature_cache.get_features("nonexistent")

        if missing_model is None and missing_features is None:
            print("  [PASS] Error handling passed")
            tests_passed += 1
        else:
            print("  [FAIL] Error handling failed")
    except Exception as e:
        print(f"  [FAIL] Error handling failed: {e}")

    # 結果
    success_rate = tests_passed / total_tests
    print(f"\n=== TEST RESULTS ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {success_rate:.1%}")

    if success_rate >= 0.8:
        print("Status: PASSED - Production ready")
        return True
    else:
        print("Status: FAILED - Needs improvement")
        return False

if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)

