#!/usr/bin/env python3
"""
Issue #492: BaseModelInterface hit_rate計算改善テスト
改善されたhit_rate計算ロジックとエラーハンドリングの機能テスト
"""

import sys
import tempfile
import logging
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def test_hit_rate_calculation_improvements():
    """改善されたhit_rate計算ロジックテスト"""
    print("=== Issue #492 Hit Rate Calculation Improvements Test ===\n")

    try:
        from src.day_trade.ml.base_models.base_model_interface import BaseModelInterface, ModelPrediction

        # テスト用のモックモデルクラス
        class TestModel(BaseModelInterface):
            def fit(self, X, y, validation_data=None):
                self.is_trained = True
                return {"training_time": 0.1}

            def predict(self, X):
                # 簡単な予測：入力に少しノイズを加える
                predictions = X.flatten() + np.random.normal(0, 0.1, len(X.flatten()))
                return ModelPrediction(predictions=predictions, model_name=self.model_name)

            def get_feature_importance(self):
                return {"feature_1": 0.6, "feature_2": 0.4}

        print("改善されたhit_rate計算テスト:")
        print("-" * 60)

        model = TestModel("TestModel")

        results = []

        # テストケース1: 正常なデータでのhit_rate計算
        print("1. 正常なデータでのhit_rate計算テスト:")

        # 上昇トレンドのデータ
        y_true = np.array([100, 101, 102, 101, 103, 104, 102, 105])
        y_pred = np.array([100, 100.8, 102.2, 100.9, 103.1, 104.2, 101.8, 105.1])

        hit_rate = model._calculate_hit_rate(y_true, y_pred)

        normal_data_test = 0.0 <= hit_rate <= 1.0
        results.append(normal_data_test)

        status = "OK PASS" if normal_data_test else "NG FAIL"
        print(f"   正常データでのhit_rate計算: {status}")
        print(f"   hit_rate値: {hit_rate:.3f}")

        # テストケース2: データ不足時の処理
        print("\n2. データ不足時の適切な処理テスト:")

        # 少ないデータ
        y_true_small = np.array([100])
        y_pred_small = np.array([101])

        hit_rate_small = model._calculate_hit_rate(y_true_small, y_pred_small)

        small_data_test = hit_rate_small == 0.5
        results.append(small_data_test)

        status = "OK PASS" if small_data_test else "NG FAIL"
        print(f"   データ不足時のデフォルト値: {status}")
        print(f"   hit_rate値: {hit_rate_small}")

        # テストケース3: 0値（変化なし）の適切な処理
        print("\n3. 0値（変化なし）の適切な処理テスト:")

        # 変化がないデータ
        y_true_flat = np.array([100, 100, 100, 100])
        y_pred_flat = np.array([100.1, 100.05, 99.95, 100.02])

        hit_rate_zero = model._calculate_hit_rate(y_true_flat, y_pred_flat)

        zero_handling_test = 0.0 <= hit_rate_zero <= 1.0  # 0値処理が適切に動作
        results.append(zero_handling_test)

        status = "OK PASS" if zero_handling_test else "NG FAIL"
        print(f"   0値（変化なし）処理: {status}")
        print(f"   hit_rate値: {hit_rate_zero}")

        # テストケース4: _get_direction メソッドのテスト
        print("\n4. 方向性判定メソッドテスト:")

        values = np.array([0.1, -0.1, 0.0001, -0.0001, 1.0, -1.0])
        directions = model._get_direction(values, zero_threshold=1e-3)

        expected_directions = np.array([1, -1, 0, 0, 1, -1])  # 閾値1e-3での期待値
        direction_test = np.array_equal(directions, expected_directions)
        results.append(direction_test)

        status = "OK PASS" if direction_test else "NG FAIL"
        print(f"   方向性判定メソッド: {status}")
        print(f"   実際の方向性: {directions}")
        print(f"   期待方向性: {expected_directions}")

        # テストケース5: 完全一致の場合のhit_rate
        print("\n5. 完全一致ケースのhit_rate計算テスト:")

        # 完全に一致する方向性
        y_true_perfect = np.array([100, 101, 102, 103])
        y_pred_perfect = np.array([100, 101.1, 102.1, 103.1])

        hit_rate_perfect = model._calculate_hit_rate(y_true_perfect, y_pred_perfect)

        perfect_match_test = abs(hit_rate_perfect - 1.0) < 1e-6
        results.append(perfect_match_test)

        status = "OK PASS" if perfect_match_test else "NG FAIL"
        print(f"   完全一致ケース: {status}")
        print(f"   hit_rate値: {hit_rate_perfect}")

        all_passed = all(results)
        print(f"\n改善されたhit_rate計算: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"改善されたhit_rate計算テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling_improvements():
    """エラーハンドリング改善テスト"""
    print("\n=== Issue #492 Error Handling Improvements Test ===\n")

    try:
        from src.day_trade.ml.base_models.base_model_interface import BaseModelInterface, ModelPrediction, ModelMetrics

        # エラーを発生させるテスト用のモックモデル
        class ErrorModel(BaseModelInterface):
            def fit(self, X, y, validation_data=None):
                self.is_trained = True
                return {}

            def predict(self, X):
                # エラーを意図的に発生
                raise ValueError("予測エラーのシミュレーション")

            def get_feature_importance(self):
                return {}

        print("エラーハンドリング改善テスト:")
        print("-" * 60)

        results = []

        # テストケース1: predict()エラー時の適切なフォールバック
        print("1. predict()エラー時のフォールバックテスト:")

        error_model = ErrorModel("ErrorModel")

        # 適当なテストデータ
        X_test = np.array([[1, 2], [3, 4]])
        y_test = np.array([100, 101])

        metrics = error_model.evaluate(X_test, y_test)

        # エラー時のデフォルト値チェック
        error_handling_test = (
            metrics.mse == float('inf') and
            metrics.rmse == float('inf') and
            metrics.mae == float('inf') and
            metrics.r2_score == -1.0 and
            metrics.hit_rate == 0.5
        )

        results.append(error_handling_test)

        status = "OK PASS" if error_handling_test else "NG FAIL"
        print(f"   エラー時のデフォルト値: {status}")
        print(f"   MSE: {metrics.mse}")
        print(f"   Hit Rate: {metrics.hit_rate}")

        # テストケース2: 不正な入力データの処理
        print("\n2. 不正な入力データ処理テスト:")

        # 正常なモデルを作成
        class NormalModel(BaseModelInterface):
            def fit(self, X, y, validation_data=None):
                self.is_trained = True
                return {}

            def predict(self, X):
                return ModelPrediction(predictions=np.array([100, 101]), model_name=self.model_name)

            def get_feature_importance(self):
                return {}

        normal_model = NormalModel("NormalModel")

        # 不正なデータ（異なるサイズ）
        X_invalid = np.array([[1, 2]])  # 1サンプル
        y_invalid = np.array([100, 101, 102])  # 3サンプル

        metrics_invalid = normal_model.evaluate(X_invalid, y_invalid)

        # エラーハンドリングが動作することを確認
        invalid_data_test = isinstance(metrics_invalid, ModelMetrics)
        results.append(invalid_data_test)

        status = "OK PASS" if invalid_data_test else "NG FAIL"
        print(f"   不正データでのMetrics返却: {status}")
        print(f"   返却されたオブジェクト: {type(metrics_invalid)}")

        all_passed = all(results)
        print(f"\nエラーハンドリング改善: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"エラーハンドリング改善テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_actual_data():
    """実データでの統合テスト"""
    print("\n=== Issue #492 Integration with Actual Data Test ===\n")

    try:
        from src.day_trade.ml.base_models.base_model_interface import BaseModelInterface, ModelPrediction

        # 実データに近いテスト用のモックモデル
        class RealDataModel(BaseModelInterface):
            def fit(self, X, y, validation_data=None):
                self.is_trained = True
                return {}

            def predict(self, X):
                # 実データに近い予測（トレンドを若干ずらす）
                if len(X.shape) == 2:
                    base_values = X[:, 0]  # 最初の特徴量をベースとする
                else:
                    base_values = X

                # ランダムな予測誤差を追加
                np.random.seed(42)  # 再現性のため
                predictions = base_values + np.random.normal(0, 0.5, len(base_values))
                return ModelPrediction(predictions=predictions, model_name=self.model_name)

            def get_feature_importance(self):
                return {"price": 0.8, "volume": 0.2}

        print("実データでの統合テスト:")
        print("-" * 60)

        model = RealDataModel("RealDataModel")

        results = []

        # テストケース1: 株価のような実データでのテスト
        print("1. 株価データ類似でのテスト:")

        # 株価のようなデータ（日次変動）
        np.random.seed(42)
        base_prices = 1000.0
        stock_like_data = []
        for i in range(30):
            change = np.random.normal(0, 10)  # 平均0、標準偏差10の変化
            base_prices += change
            stock_like_data.append(base_prices)

        y_stock = np.array(stock_like_data)
        X_stock = y_stock.reshape(-1, 1)  # 特徴量として価格自体を使用

        metrics_stock = model.evaluate(X_stock, y_stock)

        # 実データでのhit_rate計算が正常に動作
        stock_test = (
            0.0 <= metrics_stock.hit_rate <= 1.0 and
            not np.isnan(metrics_stock.hit_rate) and
            not np.isinf(metrics_stock.mse)
        )

        results.append(stock_test)

        status = "OK PASS" if stock_test else "NG FAIL"
        print(f"   株価類似データでのevaluate: {status}")
        print(f"   Hit Rate: {metrics_stock.hit_rate:.3f}")
        print(f"   MSE: {metrics_stock.mse:.2f}")
        print(f"   R2: {metrics_stock.r2_score:.3f}")

        # テストケース2: ボラティリティの高いデータ
        print("\n2. 高ボラティリティデータでのテスト:")

        # 高ボラティリティデータ
        np.random.seed(123)
        volatile_data = np.cumsum(np.random.normal(0, 5, 50))  # 大きな変動

        X_volatile = volatile_data[:-1].reshape(-1, 1)
        y_volatile = volatile_data[1:]

        metrics_volatile = model.evaluate(X_volatile, y_volatile)

        volatile_test = (
            0.0 <= metrics_volatile.hit_rate <= 1.0 and
            not np.isnan(metrics_volatile.hit_rate)
        )

        results.append(volatile_test)

        status = "OK PASS" if volatile_test else "NG FAIL"
        print(f"   高ボラティリティデータでのevaluate: {status}")
        print(f"   Hit Rate: {metrics_volatile.hit_rate:.3f}")

        # テストケース3: 低ボラティリティ（ほぼ平坦）データ
        print("\n3. 低ボラティリティデータでのテスト:")

        # ほぼ平坦なデータ
        flat_data = 1000 + np.random.normal(0, 0.1, 20)  # 微小な変動

        X_flat = flat_data[:-1].reshape(-1, 1)
        y_flat = flat_data[1:]

        metrics_flat = model.evaluate(X_flat, y_flat)

        flat_test = (
            0.0 <= metrics_flat.hit_rate <= 1.0 and
            not np.isnan(metrics_flat.hit_rate)
        )

        results.append(flat_test)

        status = "OK PASS" if flat_test else "NG FAIL"
        print(f"   低ボラティリティデータでのevaluate: {status}")
        print(f"   Hit Rate: {metrics_flat.hit_rate:.3f}")

        all_passed = all(results)
        print(f"\n実データでの統合テスト: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"実データでの統合テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("Issue #492 BaseModelInterface Hit Rate Calculation Improvements Test\n")

    tests = [
        ("改善されたhit_rate計算ロジック", test_hit_rate_calculation_improvements),
        ("エラーハンドリング改善", test_error_handling_improvements),
        ("実データでの統合テスト", test_integration_with_actual_data)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"実行中: {test_name}")
            print('='*60)

            if test_func():
                print(f"OK {test_name}: PASS")
                passed += 1
            else:
                print(f"NG {test_name}: FAIL")
                failed += 1

        except Exception as e:
            print(f"NG {test_name}: ERROR - {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"=== Final Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed == 0:
        print("OK Issue #492 BaseModelInterface Hit Rate Calculation: ALL TESTS PASSED")
        return True
    else:
        print("NG Issue #492 BaseModelInterface Hit Rate Calculation: SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)