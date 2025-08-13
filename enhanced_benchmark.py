#!/usr/bin/env python3
"""
Enhanced Benchmark - Issue #462対応

高精度設定でのベンチマーク
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


def generate_enhanced_test_data(n_samples=800):
    """高精度テスト用データ生成"""
    np.random.seed(42)

    # より複雑な非線形関係のあるデータ
    n_features = 15
    X = np.random.randn(n_samples, n_features)

    # 非線形目標変数
    y = (
        np.sin(X[:, 0] * 2) * X[:, 1] +
        np.cos(X[:, 2]) * X[:, 3] +
        X[:, 4] ** 2 * 0.5 +
        np.sqrt(np.abs(X[:, 5])) * X[:, 6] +
        0.1 * np.random.randn(n_samples)
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]

    return X, y, feature_names


def calculate_accuracy_percentage(y_true, y_pred):
    """精度パーセンテージ計算"""
    # R2スコア
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
    rmse_normalized = max(0, 1 - rmse / np.std(y_true))

    # 総合精度（重み付き平均）
    accuracy = (r2 * 0.4 + hit_rate * 0.4 + rmse_normalized * 0.2) * 100

    return min(99.99, accuracy)


def run_enhanced_benchmark():
    """高精度ベンチマーク実行"""
    print("=" * 60)
    print("Issue #462: 高精度アンサンブル精度ベンチマーク")
    print("=" * 60)

    # データ生成
    print("高複雑度テストデータ生成中...")
    X, y, feature_names = generate_enhanced_test_data(n_samples=800)

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 高精度設定でテスト
    configs = [
        {
            'name': 'RF + GBM + SVR（最適化無し）',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=True,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                random_forest_params={'n_estimators': 100, 'max_depth': 12, 'enable_hyperopt': False},
                gradient_boosting_params={'n_estimators': 100, 'learning_rate': 0.1, 'enable_hyperopt': False},
                svr_params={'kernel': 'rbf', 'enable_hyperopt': False}
            )
        },
        {
            'name': 'フル設定（RF + GBM + SVR + Stacking）',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=True,
                enable_stacking=True,
                enable_dynamic_weighting=True,
                random_forest_params={'n_estimators': 100, 'max_depth': 12, 'enable_hyperopt': False},
                gradient_boosting_params={'n_estimators': 100, 'learning_rate': 0.1, 'enable_hyperopt': False},
                svr_params={'kernel': 'rbf', 'enable_hyperopt': False}
            )
        },
        {
            'name': 'ハイパーパラメータ最適化有効版',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=True,
                enable_stacking=True,
                enable_dynamic_weighting=True,
                random_forest_params={'n_estimators': 150, 'max_depth': 15, 'enable_hyperopt': True},
                gradient_boosting_params={'n_estimators': 150, 'learning_rate': 0.1, 'enable_hyperopt': True},
                svr_params={'kernel': 'rbf', 'enable_hyperopt': True}
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

            # 検証データを用意（最適化のため）
            val_size = int(len(X_train) * 0.2)
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train_sub = X_train[:-val_size]
            y_train_sub = y_train[:-val_size]

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
            true_directions = np.sign(np.diff(y_test))
            pred_directions = np.sign(np.diff(y_pred))
            hit_rate = np.mean(true_directions == pred_directions)

            elapsed_time = time.time() - start_time

            print(f"  精度: {accuracy:.2f}%")
            print(f"  R2スコア: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Hit Rate: {hit_rate:.3f}")
            print(f"  実行時間: {elapsed_time:.2f}秒")
            print(f"  モデル重み: {prediction.model_weights}")

            results.append({
                'name': config_info['name'],
                'accuracy': accuracy,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'hit_rate': hit_rate,
                'time': elapsed_time,
                'weights': prediction.model_weights
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

    print("\n" + "=" * 60)
    print("高精度ベンチマーク結果サマリー")
    print("=" * 60)

    # 結果比較表示
    print(f"{'設定':<25} {'精度':<8} {'R2':<8} {'Hit Rate':<10} {'時間':<8}")
    print("-" * 60)

    for result in results:
        if 'error' not in result:
            print(f"{result['name']:<25} {result['accuracy']:6.2f}% {result['r2']:6.3f}  {result['hit_rate']:6.3f}     {result['time']:6.1f}s")
        else:
            print(f"{result['name']:<25} {'ERROR':<8}")

    print(f"\n最高精度: {best_accuracy:.2f}%")
    print(f"最適設定: {best_config_name}")
    print(f"95%達成まで: {95.0 - best_accuracy:.2f}%の改善が必要")

    # 95%達成評価
    if best_accuracy >= 95.0:
        print("\n既に95%達成済み！")
        achievement_level = "完璧"
    elif best_accuracy >= 90.0:
        print("\n90%超達成 - あと一歩で95%！")
        achievement_level = "優秀"
        print("最終段階の改善提案:")
        print("1. ハイパーパラメータのより細かい調整")
        print("2. 特徴量の追加・改善")
        print("3. データクリーニングの最適化")
    elif best_accuracy >= 85.0:
        print("\n85%超達成 - 95%が見えてきました")
        achievement_level = "良好"
        print("中盤戦の改善提案:")
        print("1. 深層学習モデル（LSTM-Transformer）の追加")
        print("2. より高度なアンサンブル手法")
        print("3. 特徴量エンジニアリングの強化")
        print("4. クロスバリデーションでの重み最適化")
    else:
        print("\n85%未満 - さらなる基盤強化が必要")
        achievement_level = "改善要"
        print("基盤強化の改善提案:")
        print("1. データ品質の根本的改善")
        print("2. より多様なベースモデルの追加")
        print("3. 特徴量の大幅見直し")
        print("4. 外部データソースの統合")

    print(f"\n現在の達成レベル: {achievement_level}")
    print("=" * 60)

    return best_accuracy, results


if __name__ == "__main__":
    accuracy, results = run_enhanced_benchmark()
    print(f"\n総合評価: {accuracy:.2f}% (95%に対して{95.0-accuracy:.2f}%の改善が必要)")