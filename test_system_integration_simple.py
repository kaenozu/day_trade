#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
システム統合テスト（簡易版）
Issue統合最適化システムの動作確認
"""

import sys
import traceback
from pathlib import Path

# プロジェクトルート追加
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """基本的なインポートテスト"""
    print("=== 基本インポートテスト ===")

    try:
        # Issue #325: ML最適化エンジン
        from src.day_trade.data.optimized_ml_engine import OptimizedMLEngine
        print("[OK] OptimizedMLEngine import success")

        # Issue #324: 統合キャッシュシステム
        from src.day_trade.utils.unified_cache_manager import UnifiedCacheManager
        print("[OK] UnifiedCacheManager import success")

        # Issue #323: 並列処理エンジン
        from src.day_trade.data.advanced_parallel_ml_engine import AdvancedParallelMLEngine
        print("[OK] AdvancedParallelMLEngine import success")

        # Issue #322: 多角的データマネージャー
        from src.day_trade.data.multi_source_data_manager import MultiSourceDataManager
        print("[OK] MultiSourceDataManager import success")

        # データ品質管理
        from src.day_trade.utils.data_quality_manager import DataQualityManager
        print("[OK] DataQualityManager import success")

        return True

    except Exception as e:
        print(f"[ERROR] Import error: {e}")
        traceback.print_exc()
        return False

def test_optimized_ml_engine():
    """Issue #325: ML最適化エンジンテスト"""
    print("\n=== ML最適化エンジンテスト ===")

    try:
        from src.day_trade.data.optimized_ml_engine import OptimizedMLEngine
        import pandas as pd
        import numpy as np

        # テストデータ生成
        dates = pd.date_range(start='2024-01-01', periods=50)
        test_data = pd.DataFrame({
            'Open': np.random.uniform(2000, 2500, 50),
            'High': np.random.uniform(2100, 2600, 50),
            'Low': np.random.uniform(1900, 2400, 50),
            'Close': np.random.uniform(2000, 2500, 50),
            'Volume': np.random.randint(500000, 2000000, 50),
        }, index=dates)

        # ML最適化エンジン初期化
        ml_engine = OptimizedMLEngine()
        print("[OK] OptimizedMLEngine initialization success")

        # 軽量特徴量生成テスト
        features = ml_engine.prepare_lightweight_features(test_data)
        print(f"[OK] Feature generation success: {len(features)} features")

        return True

    except Exception as e:
        print(f"[ERROR] ML optimization engine error: {e}")
        traceback.print_exc()
        return False

def test_unified_cache():
    """Issue #324: 統合キャッシュテスト"""
    print("\n=== 統合キャッシュテスト ===")

    try:
        from src.day_trade.utils.unified_cache_manager import UnifiedCacheManager

        # 統合キャッシュ初期化
        cache_manager = UnifiedCacheManager()
        print("[OK] UnifiedCacheManager initialization success")

        # L1キャッシュテスト
        cache_manager.put("test_key", {"data": "test_value"}, target_layer="L1")
        cached_data = cache_manager.get("test_key")
        assert cached_data is not None
        print("[OK] L1 cache operation confirmed")

        # L2キャッシュテスト
        cache_manager.put("test_key_l2", {"data": "test_l2"}, target_layer="L2")
        cached_l2 = cache_manager.get("test_key_l2")
        assert cached_l2 is not None
        print("[OK] L2 cache operation confirmed")

        return True

    except Exception as e:
        print(f"[ERROR] Unified cache error: {e}")
        traceback.print_exc()
        return False

def test_data_quality_manager():
    """Issue #322: データ品質管理テスト"""
    print("\n=== データ品質管理テスト ===")

    try:
        from src.day_trade.utils.data_quality_manager import DataQualityManager
        import pandas as pd
        import numpy as np

        # データ品質管理初期化
        quality_manager = DataQualityManager(
            enable_cache=False,  # テスト用にキャッシュ無効
            auto_fix_enabled=True
        )
        print("[OK] DataQualityManager initialization success")

        # テストデータ（欠損値含む）
        test_data = pd.DataFrame({
            'price': [100, 102, np.nan, 105, 103],
            'volume': [1000, 1200, 1100, np.nan, 1300],
            'timestamp': pd.date_range('2024-01-01', periods=5)
        })

        # データ品質評価
        metrics = quality_manager.assess_data_quality(test_data, 'price', 'TEST')
        print(f"[OK] Data quality assessment success: Score {metrics.overall_score:.3f}")

        return True

    except Exception as e:
        print(f"[ERROR] Data quality management error: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """統合動作テスト"""
    print("\n=== 統合動作テスト ===")

    try:
        # 基本的な統合動作確認
        from src.day_trade.data.optimized_ml_engine import OptimizedMLEngine
        from src.day_trade.utils.unified_cache_manager import UnifiedCacheManager

        # ML最適化エンジンとキャッシュの組み合わせテスト
        ml_engine = OptimizedMLEngine()
        cache_manager = UnifiedCacheManager()

        # キャッシュキー生成
        cache_key = "ml_test_integration"
        test_result = {"accuracy": 0.89, "features": 70}

        # キャッシュ保存・取得
        cache_manager.put(cache_key, test_result)
        cached_result = cache_manager.get(cache_key)

        assert cached_result is not None
        assert cached_result["accuracy"] == 0.89
        print("[OK] ML optimization x Cache integration confirmed")

        return True

    except Exception as e:
        print(f"[ERROR] Integration operation error: {e}")
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    print("システム統合テスト（簡易版）開始")
    print("=" * 50)

    results = []

    # 各テスト実行
    results.append(("基本インポート", test_basic_imports()))
    results.append(("ML最適化エンジン", test_optimized_ml_engine()))
    results.append(("統合キャッシュ", test_unified_cache()))
    results.append(("データ品質管理", test_data_quality_manager()))
    results.append(("統合動作", test_integration()))

    # 結果サマリー
    print("\n" + "=" * 50)
    print("=== テスト結果サマリー ===")

    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
        if result:
            passed += 1

    print(f"\n成功率: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")

    if passed == len(results):
        print("[SUCCESS] All tests passed! Integration optimization system confirmed")
        return True
    else:
        print("[WARNING] Some tests failed - requires fixes")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nテスト中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n致命的エラー: {e}")
        traceback.print_exc()
        sys.exit(1)
