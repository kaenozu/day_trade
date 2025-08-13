#!/usr/bin/env python3
"""
Quick Accuracy Benchmark - Issue #462対応

現在のアンサンブル精度を迅速に測定
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


def generate_quick_test_data(n_samples=500):
    """高速テスト用データ生成"""
    np.random.seed(42)

    # 簡単な線形関係のある合成データ
    n_features = 10
    X = np.random.randn(n_samples, n_features)

    # 目標変数：特徴量の線形結合 + ノイズ
    true_coeffs = np.random.randn(n_features)
    y = X @ true_coeffs + 0.1 * np.random.randn(n_samples)

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
    rmse_normalized = max(0, 1 - rmse / np.std(y_true))

    # 総合精度（重み付き平均）
    accuracy = (r2 * 0.4 + hit_rate * 0.4 + rmse_normalized * 0.2) * 100

    return min(99.99, accuracy)


def run_quick_benchmark():
    """クイックベンチマーク実行"""
    print("=" * 60)
    print("Issue #462: 高速アンサンブル精度ベンチマーク")
    print("=" * 60)

    # データ生成
    print("テストデータ生成中...")
    X, y, feature_names = generate_quick_test_data(n_samples=500)

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 高速化された設定でテスト
    configs = [
        {
            'name': 'RandomForestのみ',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=False,
                use_svr=False,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                random_forest_params={'n_estimators': 50, 'max_depth': 10, 'enable_hyperopt': False}
            )
        },
        {
            'name': 'RF + GBM（最適化無し）',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=False,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                random_forest_params={'n_estimators': 50, 'max_depth': 10, 'enable_hyperopt': False},
                gradient_boosting_params={'n_estimators': 50, 'learning_rate': 0.1, 'enable_hyperopt': False}
            )
        }
    ]

    best_accuracy = 0
    best_config_name = ""

    for config_info in configs:
        print(f"\n{config_info['name']} テスト中...")

        try:
            start_time = time.time()

            # アンサンブル作成・学習
            ensemble = EnsembleSystem(config_info['config'])
            ensemble.fit(X_train, y_train, feature_names=feature_names)

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

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config_name = config_info['name']

        except Exception as e:
            print(f"  エラー: {e}")

    print("\n" + "=" * 60)
    print("ベンチマーク結果サマリー")
    print("=" * 60)
    print(f"最高精度: {best_accuracy:.2f}%")
    print(f"最適設定: {best_config_name}")
    print(f"95%達成まで: {95.0 - best_accuracy:.2f}%の改善が必要")

    # 95%達成のための提案
    print(f"\n95%精度達成のための改善提案:")
    if best_accuracy >= 95.0:
        print("既に95%達成済み！")
    elif best_accuracy >= 90.0:
        print("1. ハイパーパラメータ最適化を有効化")
        print("2. SVRモデルを追加")
        print("3. スタッキングアンサンブルを有効化")
    elif best_accuracy >= 80.0:
        print("1. より多くのベースモデルを追加")
        print("2. 特徴量エンジニアリングを強化")
        print("3. データ品質を改善")
        print("4. 動的重み調整を有効化")
    else:
        print("1. 根本的なアーキテクチャ見直しが必要")
        print("2. 深層学習モデル（LSTM-Transformer）を追加")
        print("3. より複雑な特徴量を作成")
        print("4. データサイズを大幅に増加")

    print(f"\n現在の精度レベル: {'優秀' if best_accuracy >= 90 else '良好' if best_accuracy >= 80 else '改善要' if best_accuracy >= 70 else '要大幅改善'}")
    print("=" * 60)

    return best_accuracy


if __name__ == "__main__":
    accuracy = run_quick_benchmark()
    print(f"最終精度: {accuracy:.2f}%")