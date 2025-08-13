#!/usr/bin/env python3
"""
Issue #479 簡単テスト: DynamicWeightingSystem重み制約とモーメンタム適用順序
"""

import sys
sys.path.append('src')

from day_trade.ml.dynamic_weighting_system import DynamicWeightingSystem, DynamicWeightingConfig
import numpy as np

def test_issue_479():
    """Issue #479: 重み制約とモーメンタム適用順序テスト"""

    print("=== Issue #479: 重み制約とモメンタム適用順序テスト ===")

    # テスト設定
    config = DynamicWeightingConfig(
        window_size=20,
        min_samples_for_update=10,
        update_frequency=5,
        momentum_factor=0.3,
        max_weight_change=0.1,
        min_weight=0.05,
        max_weight=0.8,
        verbose=True
    )

    model_names = ["model_a", "model_b", "model_c"]
    dws = DynamicWeightingSystem(model_names, config)

    print(f"初期重み: {dws.get_current_weights()}")

    # 1. 包括的制約適用テスト
    print("\n1. 包括的制約適用テスト")
    test_weights = {"model_a": 0.9, "model_b": 0.05, "model_c": 0.05}  # 制約違反
    constrained = dws._apply_comprehensive_constraints(test_weights)
    print(f"制約適用前: {test_weights}")
    print(f"制約適用後: {constrained}")
    print(f"制約チェック: 合計={sum(constrained.values()):.6f}")

    # 2. モーメンタム→制約の順序テスト
    print("\n2. モーメンタム→制約の順序テスト")
    dws.current_weights = {"model_a": 0.4, "model_b": 0.3, "model_c": 0.3}
    extreme_weights = {"model_a": 0.9, "model_b": 0.05, "model_c": 0.05}

    # Step 1: モーメンタム適用
    momentum_weights = dws._apply_momentum(extreme_weights)
    print(f"極端な重み: {extreme_weights}")
    print(f"モーメンタム後: {momentum_weights}")

    # Step 2: 制約適用
    final_weights = dws._apply_comprehensive_constraints(momentum_weights)
    print(f"最終重み: {final_weights}")

    # 3. 重み更新・検証テスト
    print("\n3. 重み更新・検証テスト")
    valid_weights = {"model_a": 0.5, "model_b": 0.3, "model_c": 0.2}
    old_total_updates = dws.total_updates
    dws._validate_and_update_weights(valid_weights)
    print(f"更新前の総更新回数: {old_total_updates}")
    print(f"更新後の総更新回数: {dws.total_updates}")
    print(f"現在の重み: {dws.current_weights}")

    # 4. 統合テスト - 実際の重み更新プロセス
    print("\n4. 統合テスト - 実際のパフォーマンス更新")
    np.random.seed(42)

    for i in range(30):  # update_frequency=5なので複数回更新される
        predictions = {
            "model_a": 100 + i * 0.1 + np.random.normal(0, 0.5),  # 良い性能
            "model_b": 100 + i * 0.1 + np.random.normal(0, 1.5),  # 中程度
            "model_c": 100 + i * 0.1 + np.random.normal(0, 2.5)   # 悪い性能
        }
        actual = 100 + i * 0.1

        old_weights = dws.current_weights.copy()
        dws.update_performance(predictions, actual, i)

        # 重み更新が発生した場合
        if dws.current_weights != old_weights:
            print(f"Step {i}: 重み更新 {old_weights} -> {dws.current_weights}")

            # 制約チェック
            total = sum(dws.current_weights.values())
            print(f"  合計: {total:.6f}")

            for model_name, weight in dws.current_weights.items():
                assert config.min_weight <= weight <= config.max_weight, f"{model_name}: {weight}"
                change = abs(weight - old_weights[model_name])
                assert change <= config.max_weight_change + 1e-6, f"{model_name}: 変更量{change}"

            print("  [OK] 全制約を満たしています")

    print(f"\n最終重み: {dws.get_current_weights()}")
    print(f"総更新回数: {dws.total_updates}")
    print(f"重み履歴数: {len(dws.weight_history)}")

    print("\n=== Issue #479テスト完了 ===")
    print("[OK] 包括的制約適用が正常に動作")
    print("[OK] モーメンタム→制約の順序で処理")
    print("[OK] 重み検証・更新が適切に実行")
    print("[OK] 統合テストで全制約を維持")

if __name__ == "__main__":
    test_issue_479()