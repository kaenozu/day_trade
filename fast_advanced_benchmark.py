#!/usr/bin/env python3
"""
Fast Advanced Ensemble Benchmark - Issue #462対応

XGBoost・CatBoost追加による95%精度達成テスト（高速版）
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


def generate_test_data(n_samples=800):
    """高速テスト用データ生成"""
    np.random.seed(42)

    n_features = 15

    # より複雑な非線形関係
    X = np.random.randn(n_samples, n_features)

    # 複雑な真の関数関係
    y = (
        1.5 * X[:, 0] * X[:, 1] +           # 交互作用
        1.2 * np.sin(X[:, 2]) * X[:, 3] +  # 非線形
        0.8 * X[:, 4] ** 2 +               # 二次項
        0.6 * np.tanh(X[:, 5]) * X[:, 6] + # 複合非線形
        0.4 * np.sqrt(np.abs(X[:, 7])) * np.sign(X[:, 7]) +
        0.3 * X[:, 8] * X[:, 9] * X[:, 10] + # 3次交互作用
        np.sum(X[:, 11:] * 0.2, axis=1) + # 線形成分
        0.1 * np.random.randn(n_samples)   # ノイズ
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, y, feature_names


def calculate_accuracy_percentage(y_true, y_pred):
    """精度パーセンテージ計算"""
    # R²スコア
    r2 = max(0, r2_score(y_true, y_pred))

    # Hit Rate（方向予測精度）
    if len(y_true) > 1:
        true_directions = np.sign(np.diff(y_true))
        pred_directions = np.sign(np.diff(y_pred))
        hit_rate = np.mean(true_directions == pred_directions)
    else:
        hit_rate = 0.5

    # RMSE正規化スコア
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_normalized = max(0, 1 - rmse / (np.std(y_true) + 1e-8))

    # 総合精度（重み付き平均）
    accuracy = (r2 * 0.4 + hit_rate * 0.35 + rmse_normalized * 0.25) * 100

    return min(99.99, accuracy)


def run_fast_advanced_benchmark():
    """XGBoost・CatBoost搭載の高速アンサンブルベンチマーク"""
    print("=" * 70)
    print("Issue #462: XGBoost・CatBoost 高速ベンチマーク")
    print("=" * 70)

    # データ生成
    print("テストデータ生成中...")
    X, y, feature_names = generate_test_data(n_samples=800)

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 検証用分割
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )

    print(f"訓練: {X_train_sub.shape}, 検証: {X_val.shape}, テスト: {X_test.shape}")

    # 高速設定（ハイパーパラメータ最適化オフ）
    configs = [
        {
            'name': '基本設定（RF+GBM+SVR）',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=True,
                use_xgboost=False,
                use_catboost=False,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                # ハイパーパラメータ最適化オフ
                random_forest_params={'n_estimators': 100, 'max_depth': 10, 'enable_hyperopt': False},
                gradient_boosting_params={'n_estimators': 100, 'learning_rate': 0.1, 'enable_hyperopt': False},
                svr_params={'kernel': 'rbf', 'enable_hyperopt': False}
            )
        },
        {
            'name': 'XGBoost追加版',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=True,
                use_xgboost=True,
                use_catboost=False,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                # ハイパーパラメータ最適化オフ
                random_forest_params={'n_estimators': 80, 'max_depth': 8, 'enable_hyperopt': False},
                gradient_boosting_params={'n_estimators': 80, 'learning_rate': 0.1, 'enable_hyperopt': False},
                svr_params={'kernel': 'rbf', 'enable_hyperopt': False},
                xgboost_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'enable_hyperopt': False}
            )
        },
        {
            'name': 'CatBoost追加版',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=True,
                use_xgboost=False,
                use_catboost=True,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                # ハイパーパラメータ最適化オフ
                random_forest_params={'n_estimators': 80, 'max_depth': 8, 'enable_hyperopt': False},
                gradient_boosting_params={'n_estimators': 80, 'learning_rate': 0.1, 'enable_hyperopt': False},
                svr_params={'kernel': 'rbf', 'enable_hyperopt': False},
                catboost_params={'iterations': 100, 'learning_rate': 0.1, 'depth': 6, 'enable_hyperopt': False}
            )
        },
        {
            'name': 'フル構成（XGB+CatBoost）',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=True,
                use_xgboost=True,
                use_catboost=True,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                # ハイパーパラメータ最適化オフ
                random_forest_params={'n_estimators': 80, 'max_depth': 8, 'enable_hyperopt': False},
                gradient_boosting_params={'n_estimators': 80, 'learning_rate': 0.1, 'enable_hyperopt': False},
                svr_params={'kernel': 'rbf', 'enable_hyperopt': False},
                xgboost_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'enable_hyperopt': False},
                catboost_params={'iterations': 100, 'learning_rate': 0.1, 'depth': 6, 'enable_hyperopt': False}
            )
        },
        {
            'name': 'フル構成+Stacking',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=True,
                use_xgboost=True,
                use_catboost=True,
                enable_stacking=True,
                enable_dynamic_weighting=False,
                # ハイパーパラメータ最適化オフ
                random_forest_params={'n_estimators': 60, 'max_depth': 6, 'enable_hyperopt': False},
                gradient_boosting_params={'n_estimators': 60, 'learning_rate': 0.1, 'enable_hyperopt': False},
                svr_params={'kernel': 'rbf', 'enable_hyperopt': False},
                xgboost_params={'n_estimators': 80, 'max_depth': 5, 'learning_rate': 0.1, 'enable_hyperopt': False},
                catboost_params={'iterations': 80, 'learning_rate': 0.1, 'depth': 5, 'enable_hyperopt': False}
            )
        }
    ]

    results = []
    best_accuracy = 0
    best_config_name = ""

    for i, config_info in enumerate(configs):
        print(f"\n{i+1}. {config_info['name']} テスト中...")

        try:
            start_time = time.time()

            # アンサンブル作成・学習
            ensemble = EnsembleSystem(config_info['config'])
            ensemble.fit(
                X_train_sub, y_train_sub,
                validation_data=(X_val, y_val),
                feature_names=feature_names
            )

            # 予測
            prediction = ensemble.predict(X_test)
            y_pred = prediction.final_predictions

            # 評価
            accuracy = calculate_accuracy_percentage(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            # 方向予測精度
            if len(y_test) > 1:
                true_directions = np.sign(np.diff(y_test))
                pred_directions = np.sign(np.diff(y_pred))
                hit_rate = np.mean(true_directions == pred_directions)
            else:
                hit_rate = 0.5

            elapsed_time = time.time() - start_time

            print(f"  精度: {accuracy:.2f}%")
            print(f"  R²: {r2:.4f}, RMSE: {rmse:.4f}, Hit Rate: {hit_rate:.3f}")
            print(f"  実行時間: {elapsed_time:.2f}秒")

            results.append({
                'name': config_info['name'],
                'accuracy': accuracy,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'hit_rate': hit_rate,
                'time': elapsed_time
            })

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config_name = config_info['name']

        except Exception as e:
            print(f"  エラー: {e}")
            results.append({
                'name': config_info['name'],
                'error': str(e)
            })

    # 結果サマリー
    print("\n" + "=" * 70)
    print("Issue #462: XGBoost・CatBoost効果分析")
    print("=" * 70)

    print(f"{'設定':<30} {'精度':<8} {'R²':<8} {'Hit Rate':<10} {'時間':<8}")
    print("-" * 70)

    for result in results:
        if 'error' not in result:
            print(f"{result['name']:<30} {result['accuracy']:6.2f}% {result['r2']:6.3f}  {result['hit_rate']:6.3f}     {result['time']:6.1f}s")
        else:
            print(f"{result['name']:<30} ERROR")

    print(f"\n[RESULT] 最高精度: {best_accuracy:.2f}%")
    print(f"[BEST] 最優秀設定: {best_config_name}")
    print(f"[TARGET] 95%まで: {95.0 - best_accuracy:.2f}%")

    # 改善効果分析
    if len(results) >= 2 and 'error' not in results[0]:
        base_accuracy = results[0]['accuracy']
        improvement = best_accuracy - base_accuracy

        print(f"\n[ANALYSIS] 改善効果分析:")
        print(f"  基本設定: {base_accuracy:.2f}%")
        print(f"  最高設定: {best_accuracy:.2f}%")
        print(f"  改善量: +{improvement:.2f}%")

        if improvement >= 10:
            print("  [EXCELLENT] XGBoost/CatBoost効果絶大！")
        elif improvement >= 5:
            print("  [GOOD] 明確な改善効果を確認")
        elif improvement >= 2:
            print("  [OK] 着実な改善")
        else:
            print("  [WARNING] 微小な改善")

    # Issue #462評価
    if best_accuracy >= 95.0:
        print(f"\n[SUCCESS] Issue #462完了！95%達成！")
        status = "COMPLETED"
    elif best_accuracy >= 90.0:
        print(f"\n[CLOSE] Issue #462ほぼ完了！90%超達成")
        status = "NEARLY_COMPLETED"
    elif best_accuracy >= 85.0:
        print(f"\n[PROGRESS] Issue #462順調！85%超達成")
        status = "GOOD_PROGRESS"
    else:
        print(f"\n[CONTINUE] Issue #462継続中")
        status = "IN_PROGRESS"

    print("=" * 70)
    return best_accuracy, results, status


if __name__ == "__main__":
    print("Issue #462: 高速XGBoost・CatBoost効果検証")
    print("開始時刻:", time.strftime("%Y-%m-%d %H:%M:%S"))

    accuracy, results, status = run_fast_advanced_benchmark()

    print(f"\n[FINAL] Issue #462 結果: {accuracy:.2f}% ({status})")
    print("完了時刻:", time.strftime("%Y-%m-%d %H:%M:%S"))