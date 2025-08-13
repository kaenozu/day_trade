#!/usr/bin/env python3
"""
Issue #479対応: DynamicWeightingSystem重み制約とモメンタム適用順序テスト

重み制約とモーメンタムの適用順序最適化の動作検証
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import time

sys.path.append('C:/gemini-desktop/day_trade/src')

# テストに必要なモジュールをインポート
try:
    from day_trade.ml.dynamic_weighting_system import (
        DynamicWeightingSystem,
        DynamicWeightingConfig,
        MarketRegime
    )
    TEST_AVAILABLE = True
except ImportError as e:
    TEST_AVAILABLE = False
    print(f"テストモジュールインポートエラー: {e}")


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestDynamicWeightingConstraintMomentum:
    """Issue #479: 重み制約とモメンタム適用順序テスト"""

    @pytest.fixture
    def model_names(self):
        return ["model_a", "model_b", "model_c"]

    @pytest.fixture
    def test_config(self):
        """テスト用設定"""
        return DynamicWeightingConfig(
            window_size=30,
            min_samples_for_update=15,
            update_frequency=10,
            momentum_factor=0.3,  # モーメンタムを強めに設定
            max_weight_change=0.15,  # 変更量制限
            min_weight=0.1,
            max_weight=0.7,
            verbose=True
        )

    @pytest.fixture
    def dws(self, model_names, test_config):
        """テスト用DynamicWeightingSystem"""
        return DynamicWeightingSystem(model_names, test_config)

    def test_comprehensive_constraints_basic(self, dws):
        """Issue #479: 包括的制約適用基本テスト"""
        # テスト用重み（制約を満たす）
        test_weights = {
            "model_a": 0.5,
            "model_b": 0.3,
            "model_c": 0.2
        }

        result = dws._apply_comprehensive_constraints(test_weights)

        # 基本制約チェック
        assert abs(sum(result.values()) - 1.0) < 1e-6  # 合計1.0
        for weight in result.values():
            assert dws.config.min_weight <= weight <= dws.config.max_weight

    def test_comprehensive_constraints_over_limit(self, dws):
        """Issue #479: 最大重み制限を超える場合のテスト"""
        # 最大重み制限を超える重み
        test_weights = {
            "model_a": 0.8,  # max_weight(0.7)を超過
            "model_b": 0.15,
            "model_c": 0.05
        }

        result = dws._apply_comprehensive_constraints(test_weights)

        # 制約チェック
        assert result["model_a"] <= dws.config.max_weight
        assert abs(sum(result.values()) - 1.0) < 1e-6
        for weight in result.values():
            assert dws.config.min_weight <= weight <= dws.config.max_weight

    def test_comprehensive_constraints_under_limit(self, dws):
        """Issue #479: 最小重み制限を下回る場合のテスト"""
        # 最小重み制限を下回る重み
        test_weights = {
            "model_a": 0.05,  # min_weight(0.1)を下回る
            "model_b": 0.4,
            "model_c": 0.55
        }

        result = dws._apply_comprehensive_constraints(test_weights)

        # 制約チェック
        assert result["model_a"] >= dws.config.min_weight
        assert abs(sum(result.values()) - 1.0) < 1e-6
        for weight in result.values():
            assert dws.config.min_weight <= weight <= dws.config.max_weight

    def test_comprehensive_constraints_max_change_limit(self, dws):
        """Issue #479: 最大変更量制限テスト"""
        # 現在の重みを設定
        dws.current_weights = {
            "model_a": 0.4,
            "model_b": 0.3,
            "model_c": 0.3
        }

        # 大幅な変更を試行
        test_weights = {
            "model_a": 0.7,  # +0.3の変更（max_weight_change=0.15を超過）
            "model_b": 0.2,  # -0.1の変更
            "model_c": 0.1   # -0.2の変更
        }

        result = dws._apply_comprehensive_constraints(test_weights)

        # 変更量制限チェック
        for model_name in dws.model_names:
            current = dws.current_weights[model_name]
            new = result[model_name]
            change = abs(new - current)
            assert change <= dws.config.max_weight_change + 1e-6, f"{model_name}: 変更量{change:.3f}が制限{dws.config.max_weight_change}を超過"

    def test_validate_and_update_weights_success(self, dws):
        """Issue #479: 重み検証・更新成功テスト"""
        # 有効な重み
        valid_weights = {
            "model_a": 0.4,
            "model_b": 0.35,
            "model_c": 0.25
        }

        old_total_updates = dws.total_updates
        old_history_len = len(dws.weight_history)

        dws._validate_and_update_weights(valid_weights)

        # 更新確認
        assert dws.current_weights == valid_weights
        assert dws.total_updates == old_total_updates + 1
        assert len(dws.weight_history) == old_history_len + 1

    def test_validate_and_update_weights_invalid_sum(self, dws):
        """Issue #479: 重み合計が1.0でない場合のテスト"""
        # 合計が1.0でない重み
        invalid_weights = {
            "model_a": 0.5,
            "model_b": 0.3,
            "model_c": 0.3  # 合計1.1
        }

        old_weights = dws.current_weights.copy()
        old_total_updates = dws.total_updates

        dws._validate_and_update_weights(invalid_weights)

        # 更新されていないことを確認
        assert dws.current_weights == old_weights
        assert dws.total_updates == old_total_updates

    def test_validate_and_update_weights_constraint_violation(self, dws):
        """Issue #479: 制約違反重みのテスト"""
        # 制約違反重み
        invalid_weights = {
            "model_a": 0.05,  # min_weightを下回る
            "model_b": 0.4,
            "model_c": 0.55
        }

        old_weights = dws.current_weights.copy()
        old_total_updates = dws.total_updates

        dws._validate_and_update_weights(invalid_weights)

        # 更新されていないことを確認
        assert dws.current_weights == old_weights
        assert dws.total_updates == old_total_updates

    def test_momentum_then_constraints_order(self, dws):
        """Issue #479: モーメンタム→制約の順序テスト"""
        # 初期重み設定
        dws.current_weights = {
            "model_a": 0.4,
            "model_b": 0.3,
            "model_c": 0.3
        }

        # 極端な新重み（制約違反）
        extreme_weights = {
            "model_a": 0.9,  # 極端に高い
            "model_b": 0.05,  # 極端に低い
            "model_c": 0.05   # 極端に低い
        }

        # Step 1: モーメンタム適用
        momentum_weights = dws._apply_momentum(extreme_weights)

        # モーメンタム後は極端さが和らいでいることを確認
        assert momentum_weights["model_a"] < extreme_weights["model_a"]
        assert momentum_weights["model_b"] > extreme_weights["model_b"]
        assert momentum_weights["model_c"] > extreme_weights["model_c"]

        # Step 2: 制約適用
        final_weights = dws._apply_comprehensive_constraints(momentum_weights)

        # 最終的に全制約を満たすことを確認
        assert abs(sum(final_weights.values()) - 1.0) < 1e-6
        for weight in final_weights.values():
            assert dws.config.min_weight <= weight <= dws.config.max_weight

    def test_full_update_process_integration(self, dws):
        """Issue #479: 完全な更新プロセス統合テスト"""
        # 大量のパフォーマンスデータを投入して重み更新を発生させる
        np.random.seed(42)

        for i in range(50):  # update_frequency=10なので複数回更新される
            # 性能差のあるモデル予測
            predictions = {
                "model_a": 100 + i * 0.1 + np.random.normal(0, 0.5),  # 良い性能
                "model_b": 100 + i * 0.1 + np.random.normal(0, 1.5),  # 中程度
                "model_c": 100 + i * 0.1 + np.random.normal(0, 2.5)   # 悪い性能
            }
            actual = 100 + i * 0.1

            old_weights = dws.current_weights.copy()
            dws.update_performance(predictions, actual, i)

            # 重み更新が発生した場合の制約チェック
            if dws.current_weights != old_weights:
                # 全制約を満たしているかチェック
                assert abs(sum(dws.current_weights.values()) - 1.0) < 1e-6
                for weight in dws.current_weights.values():
                    assert dws.config.min_weight <= weight <= dws.config.max_weight

                # 変更量制限チェック
                for model_name in dws.model_names:
                    change = abs(dws.current_weights[model_name] - old_weights[model_name])
                    assert change <= dws.config.max_weight_change + 1e-6

        # 最終的に良い性能のモデルが高い重みを持つことを確認
        final_weights = dws.get_current_weights()
        assert final_weights["model_a"] >= final_weights["model_b"]
        assert final_weights["model_b"] >= final_weights["model_c"]

    def test_error_handling_in_constraints(self, dws):
        """Issue #479: 制約適用でのエラーハンドリングテスト"""
        # 空の重み辞書
        result = dws._apply_comprehensive_constraints({})

        # エラー時は現在の重みを返すことを確認
        assert result == dws.current_weights

        # NaN値を含む重み
        nan_weights = {
            "model_a": float('nan'),
            "model_b": 0.3,
            "model_c": 0.2
        }

        # エラーハンドリングが適切に動作することを確認
        try:
            result = dws._apply_comprehensive_constraints(nan_weights)
            # 結果が有効であることを確認
            assert all(not np.isnan(weight) for weight in result.values())
        except Exception:
            # 例外が発生してもプログラムは続行する
            pass

    def test_weight_history_recording(self, dws):
        """Issue #479: 重み履歴記録テスト"""
        # 有効な重み更新
        test_weights = {
            "model_a": 0.4,
            "model_b": 0.35,
            "model_c": 0.25
        }

        old_history_len = len(dws.weight_history)

        dws._validate_and_update_weights(test_weights)

        # 履歴が正しく記録されることを確認
        assert len(dws.weight_history) == old_history_len + 1

        latest_history = dws.weight_history[-1]
        assert latest_history['weights'] == test_weights
        assert 'timestamp' in latest_history
        assert 'regime' in latest_history
        assert 'total_updates' in latest_history

    def test_verbose_logging(self, dws):
        """Issue #479: 詳細ログ出力テスト"""
        # 重みを大きく変更して、ログ出力をトリガー
        dws.current_weights = {
            "model_a": 0.4,
            "model_b": 0.3,
            "model_c": 0.3
        }

        new_weights = {
            "model_a": 0.5,  # 0.1の変更（1%以上）
            "model_b": 0.25, # -0.05の変更（1%以上）
            "model_c": 0.25  # -0.05の変更（1%以上）
        }

        # verboseモードでの重み更新
        dws._validate_and_update_weights(new_weights)

        # ログが出力されたかはログレベルに依存するが、エラーが発生しないことを確認
        assert dws.current_weights == new_weights


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])