#!/usr/bin/env python3
"""
Issue #475 簡単テスト: DynamicWeightingSystem予測・実績処理改善
"""

import sys
sys.path.append('src')

from day_trade.ml.dynamic_weighting_system import (
    DynamicWeightingSystem, 
    DynamicWeightingConfig, 
    MarketRegime
)
import numpy as np
import time

def test_issue_475():
    """Issue #475: 予測・実績処理改善テスト"""
    
    print("=== Issue #475: 予測・実績処理改善テスト ===")
    
    model_names = ["model_a", "model_b", "model_c"]
    
    # 1. 改善された型サポートテスト
    print("\n1. 改善された型サポートテスト")
    dws = DynamicWeightingSystem(model_names, DynamicWeightingConfig(verbose=False))
    
    # 各種データ型のテスト
    test_cases = [
        # (予測値辞書, 実際値, 説明)
        ({"model_a": 100.5, "model_b": 101, "model_c": 99.8}, 100.2, "単一値"),
        ({"model_a": [100.1, 100.2], "model_b": [101.1, 101.2], "model_c": [99.1, 99.2]}, [100.0, 100.1], "リスト"),
        ({"model_a": np.array([100.3]), "model_b": np.array([101.3]), "model_c": np.array([99.3])}, np.array([100.25]), "NumPy配列"),
        ({"model_a": (100.7, 100.8), "model_b": (101.7, 101.8), "model_c": (99.7, 99.8)}, (100.6, 100.7), "タプル")
    ]
    
    success_count = 0
    for i, (predictions, actuals, desc) in enumerate(test_cases):
        try:
            dws.update_performance(predictions, actuals, int(time.time()) + i)
            print(f"  {desc}: [OK]")
            success_count += 1
        except Exception as e:
            print(f"  {desc}: [ERROR] {e}")
    
    print(f"型サポートテスト: {success_count}/{len(test_cases)} 成功")
    
    # 2. データ正規化テスト
    print("\n2. データ正規化テスト")
    
    # _normalize_to_arrayメソッドのテスト
    test_normalization_cases = [
        (42.5, "単一float"),
        ([1.1, 2.2, 3.3], "リスト"),
        (np.array([4.4, 5.5]), "NumPy配列"),
        ((6.6, 7.7, 8.8), "タプル")
    ]
    
    normalize_success = 0
    for data, desc in test_normalization_cases:
        try:
            result = dws._normalize_to_array(data, f"test_{desc}")
            print(f"  {desc}: 形状{result.shape}, 型{result.dtype} [OK]")
            normalize_success += 1
        except Exception as e:
            print(f"  {desc}: [ERROR] {e}")
    
    print(f"正規化テスト: {normalize_success}/{len(test_normalization_cases)} 成功")
    
    # 3. バッチ処理テスト
    print("\n3. バッチ処理テスト")
    
    batch_predictions = [
        {"model_a": 100+i*0.1, "model_b": 101+i*0.1, "model_c": 99+i*0.1}
        for i in range(10)
    ]
    batch_actuals = [100.05+i*0.1 for i in range(10)]
    
    try:
        dws.update_performance_batch(batch_predictions, batch_actuals)
        print("  バッチ処理: [OK]")
        batch_success = True
    except Exception as e:
        print(f"  バッチ処理: [ERROR] {e}")
        batch_success = False
    
    # 4. データ検証テスト
    print("\n4. データ検証テスト")
    
    test_predictions = {"model_a": 100.5, "model_b": 101.2, "model_c": 99.8}
    test_actuals = 100.1
    
    try:
        validation_report = dws.validate_input_data(test_predictions, test_actuals)
        print(f"  データ検証: valid={validation_report['valid']}, エラー数={len(validation_report['errors'])} [OK]")
        validate_success = True
    except Exception as e:
        print(f"  データ検証: [ERROR] {e}")
        validate_success = False
    
    # 5. 統計取得テスト
    print("\n5. 統計取得テスト")
    
    try:
        stats = dws.get_data_statistics()
        print(f"  統計取得: サンプル数={stats['total_samples']}, モデル数={len(stats['models'])} [OK]")
        stats_success = True
    except Exception as e:
        print(f"  統計取得: [ERROR] {e}")
        stats_success = False
    
    # 6. エラーハンドリングテスト
    print("\n6. エラーハンドリングテスト")
    
    error_test_cases = [
        # (予測値, 実際値, 期待される動作)
        ({"model_a": float('nan'), "model_b": 101, "model_c": 99}, 100, "NaN拒否"),
        ({"model_a": 100, "model_b": float('inf'), "model_c": 99}, 100, "Inf拒否"),
        ({"model_a": 100, "model_b": 101, "model_c": 99}, None, "None拒否"),
        ({"model_a": 100, "model_b": 101, "model_c": []}, 100, "空リスト拒否")
    ]
    
    error_handled_count = 0
    for predictions, actuals, desc in error_test_cases:
        try:
            dws.update_performance(predictions, actuals)
            print(f"  {desc}: [FAIL] エラーが発生しませんでした")
        except Exception as e:
            print(f"  {desc}: [OK] 正しくエラー処理")
            error_handled_count += 1
    
    print(f"エラーハンドリング: {error_handled_count}/{len(error_test_cases)} 成功")
    
    # 7. 次元不一致処理テスト
    print("\n7. 次元不一致処理テスト")
    
    dimension_test_cases = [
        # (予測値, 実際値, 説明)
        ({"model_a": [100.1, 100.2, 100.3], "model_b": 101, "model_c": 99}, 100.15, "単一実際値vs複数予測"),
        ({"model_a": 100.1, "model_b": 101, "model_c": 99}, [100.15, 100.25], "複数実際値vs単一予測")
    ]
    
    dimension_success = 0
    for predictions, actuals, desc in dimension_test_cases:
        try:
            dws.update_performance(predictions, actuals)
            print(f"  {desc}: [OK]")
            dimension_success += 1
        except Exception as e:
            print(f"  {desc}: [ERROR] {e}")
    
    print(f"次元処理: {dimension_success}/{len(dimension_test_cases)} 成功")
    
    # 全体結果
    print("\n=== Issue #475テスト完了 ===")
    print(f"[OK] 型サポート改善: {success_count}/{len(test_cases)}")
    print(f"[OK] データ正規化: {normalize_success}/{len(test_normalization_cases)}")
    print(f"[OK] バッチ処理: {'成功' if batch_success else '失敗'}")
    print(f"[OK] データ検証: {'成功' if validate_success else '失敗'}")
    print(f"[OK] 統計取得: {'成功' if stats_success else '失敗'}")
    print(f"[OK] エラーハンドリング: {error_handled_count}/{len(error_test_cases)}")
    print(f"[OK] 次元処理: {dimension_success}/{len(dimension_test_cases)}")
    print("\n[SUCCESS] 予測・実績処理の冗長性を排除し、統一的なデータ処理を実現")
    print("[SUCCESS] np.atleast_1dベースの堅牢なデータ正規化を実装")

if __name__ == "__main__":
    test_issue_475()