#!/usr/bin/env python3
"""
Simple XGBoost・CatBoost Test - Issue #462対応

シンプルなXGBoost・CatBoostテスト（最適化なし）
"""

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# プロジェクトルートをパスに追加
from pathlib import Path
import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.ml.ensemble_system import EnsembleSystem, EnsembleConfig


def generate_test_data(n_samples=600):
    """テスト用データ生成"""
    np.random.seed(42)
    n_features = 12

    X = np.random.randn(n_samples, n_features)

    # 複雑な関数
    y = (
        1.2 * X[:, 0] * X[:, 1] +
        1.0 * np.sin(X[:, 2]) * X[:, 3] +
        0.8 * X[:, 4] ** 2 +
        0.6 * np.tanh(X[:, 5]) * X[:, 6] +
        np.sum(X[:, 7:10] * 0.3, axis=1) +
        0.1 * np.random.randn(n_samples)
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, y, feature_names


def calculate_accuracy(y_true, y_pred):
    """精度計算"""
    r2 = max(0, r2_score(y_true, y_pred))

    if len(y_true) > 1:
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        hit_rate = np.mean(true_dir == pred_dir)
    else:
        hit_rate = 0.5

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_norm = max(0, 1 - rmse / (np.std(y_true) + 1e-8))

    accuracy = (r2 * 0.4 + hit_rate * 0.4 + rmse_norm * 0.2) * 100
    return min(99.99, accuracy)


def run_simple_test():
    """シンプルテスト実行"""
    print("=" * 50)
    print("Issue #462: シンプルXGBoost・CatBoostテスト")
    print("=" * 50)

    X, y, feature_names = generate_test_data(600)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # 完全最適化オフ設定
    configs = [
        ('基本（RF+GBM+SVR）', EnsembleConfig(
            use_lstm_transformer=False,
            use_random_forest=True,
            use_gradient_boosting=True,
            use_svr=True,
            use_xgboost=False,
            use_catboost=False,
            enable_stacking=False,
            enable_dynamic_weighting=False,
            random_forest_params={'n_estimators': 50, 'max_depth': 6, 'enable_hyperopt': False},
            gradient_boosting_params={'n_estimators': 50, 'learning_rate': 0.1, 'enable_hyperopt': False},
            svr_params={'kernel': 'rbf', 'enable_hyperopt': False}
        )),
        ('XGBoost追加', EnsembleConfig(
            use_lstm_transformer=False,
            use_random_forest=True,
            use_gradient_boosting=True,
            use_svr=False,  # SVRオフで高速化
            use_xgboost=True,
            use_catboost=False,
            enable_stacking=False,
            enable_dynamic_weighting=False,
            random_forest_params={'n_estimators': 50, 'max_depth': 6, 'enable_hyperopt': False},
            gradient_boosting_params={'n_estimators': 50, 'learning_rate': 0.1, 'enable_hyperopt': False},
            xgboost_params={'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'enable_hyperopt': False}
        )),
        ('CatBoost追加', EnsembleConfig(
            use_lstm_transformer=False,
            use_random_forest=True,
            use_gradient_boosting=True,
            use_svr=False,  # SVRオフで高速化
            use_xgboost=False,
            use_catboost=True,
            enable_stacking=False,
            enable_dynamic_weighting=False,
            random_forest_params={'n_estimators': 50, 'max_depth': 6, 'enable_hyperopt': False},
            gradient_boosting_params={'n_estimators': 50, 'learning_rate': 0.1, 'enable_hyperopt': False},
            catboost_params={'iterations': 50, 'learning_rate': 0.1, 'depth': 4, 'enable_hyperopt': False}
        )),
        ('XGBoost+CatBoost', EnsembleConfig(
            use_lstm_transformer=False,
            use_random_forest=True,
            use_gradient_boosting=False,  # GBMオフでXGBと重複避け
            use_svr=False,
            use_xgboost=True,
            use_catboost=True,
            enable_stacking=False,
            enable_dynamic_weighting=False,
            random_forest_params={'n_estimators': 50, 'max_depth': 6, 'enable_hyperopt': False},
            xgboost_params={'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'enable_hyperopt': False},
            catboost_params={'iterations': 50, 'learning_rate': 0.1, 'depth': 4, 'enable_hyperopt': False}
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

            accuracy = calculate_accuracy(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            elapsed = time.time() - start_time

            print(f"  精度: {accuracy:.2f}%, R²: {r2:.4f}, 時間: {elapsed:.1f}s")

            results.append((name, accuracy, r2, elapsed))
            best_accuracy = max(best_accuracy, accuracy)

        except Exception as e:
            print(f"  エラー: {e}")
            results.append((name, 0, 0, 0))

    # 結果表示
    print("\n" + "=" * 50)
    print("結果サマリー")
    print("=" * 50)
    print(f"{'設定':<20} {'精度':<8} {'R²':<8} {'時間':<8}")
    print("-" * 50)

    for name, acc, r2, time_taken in results:
        if acc > 0:
            print(f"{name:<20} {acc:6.2f}% {r2:6.3f}  {time_taken:6.1f}s")
        else:
            print(f"{name:<20} ERROR")

    print(f"\n[RESULT] 最高精度: {best_accuracy:.2f}%")
    print(f"[TARGET] 95%まで: {95.0 - best_accuracy:.2f}%")

    if best_accuracy >= 95.0:
        print("[SUCCESS] 95%達成！Issue #462完了！")
    elif best_accuracy >= 85.0:
        print("[PROGRESS] 85%超達成！順調な進歩")
    elif best_accuracy >= 75.0:
        print("[OK] 75%超達成！基本的な改善確認")
    else:
        print("[CONTINUE] さらなる改善が必要")

    return best_accuracy


if __name__ == "__main__":
    print("Issue #462: XGBoost・CatBoost効果の簡易検証")
    print("開始:", time.strftime("%H:%M:%S"))

    accuracy = run_simple_test()

    print(f"\n最終精度: {accuracy:.2f}%")
    print("完了:", time.strftime("%H:%M:%S"))