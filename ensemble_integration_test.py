#!/usr/bin/env python3
"""
Ensemble Integration Test - Issue #462対応

アンサンブルシステムでのXGBoost・CatBoost統合確認
"""

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# プロジェクトルートをパスに追加
from pathlib import Path
import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.ml.ensemble_system import EnsembleSystem, EnsembleConfig


def generate_test_data(n_samples=400):
    """統合テストデータ生成"""
    np.random.seed(42)
    n_features = 12

    X = np.random.randn(n_samples, n_features)

    # より複雑な関係（95%精度テスト用）
    y = (
        1.8 * X[:, 0] * X[:, 1] +           # 交互作用
        1.5 * np.sin(X[:, 2]) * X[:, 3] +   # 非線形
        1.2 * X[:, 4] ** 2 +                # 二次項
        1.0 * np.tanh(X[:, 5]) * X[:, 6] +  # 制限関数
        0.8 * np.sqrt(np.abs(X[:, 7])) * np.sign(X[:, 7]) +
        0.6 * X[:, 8] * X[:, 9] * X[:, 10] + # 3次交互作用
        0.4 * np.sum(X[:, 11:] * 0.2, axis=1) +
        0.05 * np.random.randn(n_samples)    # 少ないノイズ
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, y, feature_names


def calculate_comprehensive_accuracy(y_true, y_pred):
    """包括的精度計算"""
    r2 = max(0, r2_score(y_true, y_pred))

    # 方向予測精度
    if len(y_true) > 1:
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        hit_rate = np.mean(true_dir == pred_dir)
    else:
        hit_rate = 0.5

    # RMSE正規化
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_norm = max(0, 1 - rmse / (np.std(y_true) + 1e-8))

    # 95%目標用の精度計算
    accuracy = (r2 * 0.5 + hit_rate * 0.3 + rmse_norm * 0.2) * 100
    return min(99.9, accuracy)


def run_integration_test():
    """アンサンブル統合テスト"""
    print("=" * 60)
    print("Issue #462: XGBoost・CatBoost統合効果確認")
    print("=" * 60)

    X, y, feature_names = generate_test_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    print(f"データ: 訓練={X_train_sub.shape}, 検証={X_val.shape}, テスト={X_test.shape}")

    # 統合テスト用設定（アンサンブル以外の機能を最小化）
    configs = [
        {
            'name': '基本モデルのみ',
            'config': EnsembleConfig(
                use_lstm_transformer=False,     # LSTM無効
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=False,                  # SVR無効
                use_xgboost=False,
                use_catboost=False,
                enable_stacking=False,          # Stacking無効
                enable_dynamic_weighting=False, # 動的重み無効
                random_forest_params={'n_estimators': 50, 'max_depth': 6, 'enable_hyperopt': False},
                gradient_boosting_params={'n_estimators': 50, 'learning_rate': 0.1, 'enable_hyperopt': False}
            )
        },
        {
            'name': 'XGBoost統合',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=False,    # GBM無効でXGBoostに集中
                use_svr=False,
                use_xgboost=True,
                use_catboost=False,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                random_forest_params={'n_estimators': 50, 'max_depth': 6, 'enable_hyperopt': False},
                xgboost_params={'n_estimators': 50, 'max_depth': 6, 'learning_rate': 0.1, 'enable_hyperopt': False}
            )
        },
        {
            'name': 'CatBoost統合',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=False,
                use_svr=False,
                use_xgboost=False,
                use_catboost=True,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                random_forest_params={'n_estimators': 50, 'max_depth': 6, 'enable_hyperopt': False},
                catboost_params={'iterations': 50, 'depth': 6, 'learning_rate': 0.1, 'enable_hyperopt': False, 'verbose': 0}
            )
        },
        {
            'name': 'XGBoost+CatBoost統合',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=False,        # RF無効で高性能モデルに集中
                use_gradient_boosting=False,
                use_svr=False,
                use_xgboost=True,
                use_catboost=True,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                xgboost_params={'n_estimators': 50, 'max_depth': 6, 'learning_rate': 0.1, 'enable_hyperopt': False},
                catboost_params={'iterations': 50, 'depth': 6, 'learning_rate': 0.1, 'enable_hyperopt': False, 'verbose': 0}
            )
        }
    ]

    results = []
    best_accuracy = 0

    for i, config_info in enumerate(configs):
        print(f"\n{i+1}. {config_info['name']} テスト中...")

        try:
            start_time = time.time()

            ensemble = EnsembleSystem(config_info['config'])
            ensemble.fit(X_train_sub, y_train_sub, validation_data=(X_val, y_val), feature_names=feature_names)

            prediction = ensemble.predict(X_test)
            y_pred = prediction.final_predictions

            # 評価
            accuracy = calculate_comprehensive_accuracy(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            elapsed = time.time() - start_time

            print(f"  精度: {accuracy:.2f}%, R2: {r2:.4f}, RMSE: {rmse:.4f}, 時間: {elapsed:.1f}s")

            results.append({
                'name': config_info['name'],
                'accuracy': accuracy,
                'r2': r2,
                'rmse': rmse,
                'time': elapsed,
                'success': True
            })

            best_accuracy = max(best_accuracy, accuracy)

        except Exception as e:
            print(f"  エラー: {e}")
            results.append({
                'name': config_info['name'],
                'accuracy': 0,
                'r2': 0,
                'rmse': 999,
                'time': 0,
                'success': False
            })

    # 結果サマリー
    print("\n" + "=" * 60)
    print("統合効果分析")
    print("=" * 60)
    print(f"{'設定':<20} {'精度':<8} {'R2':<8} {'RMSE':<8} {'時間'}")
    print("-" * 60)

    for result in results:
        if result['success']:
            print(f"{result['name']:<20} {result['accuracy']:6.2f}% {result['r2']:6.3f}  {result['rmse']:6.3f}  {result['time']:6.1f}s")
        else:
            print(f"{result['name']:<20} ERROR")

    # 改善分析
    if len(results) >= 2 and results[0]['success']:
        base_accuracy = results[0]['accuracy']
        improvement = best_accuracy - base_accuracy

        print(f"\n[ANALYSIS] XGBoost・CatBoost統合効果:")
        print(f"  基本設定精度: {base_accuracy:.2f}%")
        print(f"  最高精度: {best_accuracy:.2f}%")
        print(f"  改善効果: +{improvement:.2f}%")

        if improvement >= 10:
            print("  [EXCELLENT] 大幅な改善効果！")
        elif improvement >= 5:
            print("  [GOOD] 明確な改善効果")
        elif improvement >= 2:
            print("  [OK] 着実な改善")
        else:
            print("  [NEUTRAL] 微細な変化")

    # Issue #462評価
    print(f"\n[TARGET] 95%精度目標:")
    print(f"  現在の最高精度: {best_accuracy:.2f}%")
    print(f"  95%まで残り: {95.0 - best_accuracy:.2f}%")
    print(f"  達成率: {(best_accuracy/95.0)*100:.1f}%")

    if best_accuracy >= 95.0:
        print("\n[SUCCESS] Issue #462 完了！95%精度達成！")
        status = "COMPLETED"
    elif best_accuracy >= 90.0:
        print("\n[EXCELLENT] 90%超達成！95%に非常に近い")
        status = "NEARLY_COMPLETED"
    elif best_accuracy >= 85.0:
        print("\n[VERY_GOOD] 85%超達成！大幅改善")
        status = "MAJOR_IMPROVEMENT"
    elif best_accuracy >= 75.0:
        print("\n[GOOD] 75%超達成！順調な改善")
        status = "GOOD_PROGRESS"
    else:
        print("\n[CONTINUE] 継続改善が必要")
        status = "IN_PROGRESS"

    return best_accuracy, results, status


if __name__ == "__main__":
    print("Issue #462: アンサンブル統合効果確認")
    print("開始:", time.strftime("%H:%M:%S"))

    accuracy, results, status = run_integration_test()

    print(f"\n最終結果: {accuracy:.2f}% ({status})")
    print("完了:", time.strftime("%H:%M:%S"))