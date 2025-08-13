#!/usr/bin/env python3
"""
Quick XGBoost・CatBoost Test - Issue #462対応

最適化なしのクイックテスト（数十秒で完了）
"""

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# プロジェクトルートをパスに追加
from pathlib import Path
import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.ml.ensemble_system import EnsembleSystem, EnsembleConfig


def generate_quick_test_data(n_samples=400):
    """クイックテスト用データ"""
    np.random.seed(42)
    n_features = 10

    X = np.random.randn(n_samples, n_features)

    # シンプルな非線形関係
    y = (
        1.2 * X[:, 0] * X[:, 1] +
        1.0 * np.sin(X[:, 2]) +
        0.8 * X[:, 3] ** 2 +
        np.sum(X[:, 4:8] * 0.3, axis=1) +
        0.1 * np.random.randn(n_samples)
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, y, feature_names


def calculate_simple_accuracy(y_true, y_pred):
    """シンプル精度計算"""
    r2 = max(0, r2_score(y_true, y_pred))

    if len(y_true) > 1:
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        hit_rate = np.mean(true_dir == pred_dir)
    else:
        hit_rate = 0.5

    accuracy = (r2 * 0.6 + hit_rate * 0.4) * 100
    return min(99.9, accuracy)


def run_quick_test():
    """クイックテスト実行"""
    print("=" * 60)
    print("Issue #462: クイック XGBoost・CatBoost効果テスト")
    print("=" * 60)

    X, y, feature_names = generate_quick_test_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    print(f"データサイズ - 訓練: {X_train_sub.shape}, テスト: {X_test.shape}")

    # 完全最適化オフ・高速設定
    configs = [
        ('基本（RF+GBM）', EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=True,
            use_svr=False,
            use_xgboost=False,
            use_catboost=False,
            enable_stacking=False,
            enable_dynamic_weighting=False,
            random_forest_params={'n_estimators': 30, 'max_depth': 5, 'enable_hyperopt': False},
            gradient_boosting_params={'n_estimators': 30, 'learning_rate': 0.1, 'enable_hyperopt': False}
        )),
        ('XGBoost追加', EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=False,  # 重複を避けてXGBoostに注力
            use_svr=False,
            use_xgboost=True,
            use_catboost=False,
            enable_stacking=False,
            enable_dynamic_weighting=False,
            random_forest_params={'n_estimators': 30, 'max_depth': 5, 'enable_hyperopt': False},
            xgboost_params={'n_estimators': 30, 'max_depth': 4, 'learning_rate': 0.1, 'enable_hyperopt': False}
        )),
        ('CatBoost追加', EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=False,
            use_svr=False,
            use_xgboost=False,
            use_catboost=True,
            enable_stacking=False,
            enable_dynamic_weighting=False,
            random_forest_params={'n_estimators': 30, 'max_depth': 5, 'enable_hyperopt': False},
            catboost_params={'iterations': 30, 'learning_rate': 0.1, 'depth': 4, 'enable_hyperopt': False}
        )),
        ('XGB+CatBoost', EnsembleConfig(
            use_random_forest=False,  # RF無効でブースティング系に注力
            use_gradient_boosting=False,
            use_svr=False,
            use_xgboost=True,
            use_catboost=True,
            enable_stacking=False,
            enable_dynamic_weighting=False,
            xgboost_params={'n_estimators': 30, 'max_depth': 4, 'learning_rate': 0.1, 'enable_hyperopt': False},
            catboost_params={'iterations': 30, 'learning_rate': 0.1, 'depth': 4, 'enable_hyperopt': False}
        ))
    ]

    results = []
    best_accuracy = 0

    for name, config in configs:
        print(f"\n{name} テスト中...")

        try:
            start_time = time.time()

            ensemble = EnsembleSystem(config)
            ensemble.fit(X_train_sub, y_train_sub, validation_data=(X_val, y_val), feature_names=feature_names)

            prediction = ensemble.predict(X_test)
            y_pred = prediction.final_predictions

            accuracy = calculate_simple_accuracy(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            elapsed = time.time() - start_time

            print(f"  精度: {accuracy:.2f}%, R2: {r2:.4f}, RMSE: {rmse:.4f}, 時間: {elapsed:.1f}s")

            results.append((name, accuracy, r2, rmse, elapsed))
            best_accuracy = max(best_accuracy, accuracy)

        except Exception as e:
            print(f"  エラー: {e}")
            results.append((name, 0, 0, 999, 0))

    # 結果表示
    print("\n" + "=" * 60)
    print("クイック効果分析結果")
    print("=" * 60)
    print(f"{'設定':<15} {'精度':<8} {'R2':<8} {'RMSE':<8} {'時間':<8}")
    print("-" * 60)

    for name, acc, r2, rmse, time_taken in results:
        if acc > 0:
            print(f"{name:<15} {acc:6.2f}% {r2:6.3f}  {rmse:6.3f}  {time_taken:6.1f}s")
        else:
            print(f"{name:<15} ERROR")

    print(f"\n[RESULT] 最高精度: {best_accuracy:.2f}%")

    # 改善効果分析
    if len(results) >= 2 and results[0][1] > 0:
        base_accuracy = results[0][1]
        improvement = best_accuracy - base_accuracy

        print(f"\n[ANALYSIS] XGBoost・CatBoost効果:")
        print(f"  基本設定: {base_accuracy:.2f}%")
        print(f"  最高設定: {best_accuracy:.2f}%")
        print(f"  改善量: +{improvement:.2f}%")

        if improvement >= 5:
            print("  [EXCELLENT] 明確な改善効果を確認！")
        elif improvement >= 2:
            print("  [GOOD] 着実な改善を確認")
        elif improvement >= 0.5:
            print("  [OK] 微細な改善")
        else:
            print("  [WARNING] 改善効果が不明確")

    print(f"\n[TARGET] 95%まで: {95.0 - best_accuracy:.2f}%")

    if best_accuracy >= 75.0:
        print("[PROGRESS] 基本機能動作確認完了")
        status = "FUNCTIONAL"
    elif best_accuracy >= 60.0:
        print("[OK] 動作確認完了")
        status = "WORKING"
    else:
        print("[WARNING] さらなる調整が必要")
        status = "NEEDS_WORK"

    return best_accuracy, status


if __name__ == "__main__":
    print("Issue #462: XGBoost・CatBoost 基本動作確認")
    print("開始:", time.strftime("%H:%M:%S"))

    accuracy, status = run_quick_test()

    print(f"\n最終結果: {accuracy:.2f}% ({status})")
    print("完了:", time.strftime("%H:%M:%S"))