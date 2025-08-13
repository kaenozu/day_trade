#!/usr/bin/env python3
"""
Fast Enhanced Benchmark - Issue #462対応

高速版高精度設定ベンチマーク
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


def generate_enhanced_test_data(n_samples=600):
    """高精度テスト用データ生成"""
    np.random.seed(42)

    # より複雑な非線形関係のあるデータ
    n_features = 12
    X = np.random.randn(n_samples, n_features)

    # 非線形目標変数
    y = (
        np.sin(X[:, 0] * 2) * X[:, 1] +
        np.cos(X[:, 2]) * X[:, 3] +
        X[:, 4] ** 2 * 0.5 +
        np.tanh(X[:, 5]) * X[:, 6] +
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


def run_fast_enhanced_benchmark():
    """高速版高精度ベンチマーク実行"""
    print("=" * 60)
    print("Issue #462: 高速版高精度アンサンブル精度ベンチマーク")
    print("=" * 60)

    # データ生成
    print("高複雑度テストデータ生成中...")
    X, y, feature_names = generate_enhanced_test_data(n_samples=600)

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 高速化された高精度設定
    configs = [
        {
            'name': '基本RF + GBM',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=False,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                random_forest_params={'n_estimators': 100, 'max_depth': 12, 'enable_hyperopt': False},
                gradient_boosting_params={'n_estimators': 100, 'learning_rate': 0.1, 'enable_hyperopt': False}
            )
        },
        {
            'name': 'RF + GBM + SVR (3モデル)',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=True,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                random_forest_params={'n_estimators': 80, 'max_depth': 10, 'enable_hyperopt': False},
                gradient_boosting_params={'n_estimators': 80, 'learning_rate': 0.1, 'enable_hyperopt': False},
                svr_params={'kernel': 'rbf', 'enable_hyperopt': False}
            )
        },
        {
            'name': 'フル設定（Stacking + 動的重み）',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=True,
                enable_stacking=True,
                enable_dynamic_weighting=True,
                random_forest_params={'n_estimators': 80, 'max_depth': 10, 'enable_hyperopt': False},
                gradient_boosting_params={'n_estimators': 80, 'learning_rate': 0.1, 'enable_hyperopt': False},
                svr_params={'kernel': 'rbf', 'enable_hyperopt': False}
            )
        },
        {
            'name': '強化パラメータ版',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=True,
                enable_stacking=True,
                enable_dynamic_weighting=True,
                random_forest_params={
                    'n_estimators': 120,
                    'max_depth': 15,
                    'min_samples_split': 3,
                    'min_samples_leaf': 1,
                    'enable_hyperopt': False
                },
                gradient_boosting_params={
                    'n_estimators': 120,
                    'learning_rate': 0.08,
                    'max_depth': 6,
                    'enable_hyperopt': False
                },
                svr_params={'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale', 'enable_hyperopt': False}
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

            # 検証データを用意（重み最適化のため）
            val_size = int(len(X_train) * 0.2)
            if val_size > 0:
                X_val = X_train[-val_size:]
                y_val = y_train[-val_size:]
                X_train_sub = X_train[:-val_size]
                y_train_sub = y_train[:-val_size]
                validation_data = (X_val, y_val)
            else:
                X_train_sub = X_train
                y_train_sub = y_train
                validation_data = None

            ensemble.fit(
                X_train_sub, y_train_sub,
                validation_data=validation_data,
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

            # モデル重み表示（空でなければ）
            if prediction.model_weights:
                weights_str = ", ".join([f"{k}:{v:.3f}" for k, v in prediction.model_weights.items()])
                print(f"  モデル重み: {weights_str}")

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

    print("\n" + "=" * 75)
    print("高精度ベンチマーク結果サマリー")
    print("=" * 75)

    # 結果比較表示
    print(f"{'設定':<25} {'精度':<8} {'R2':<8} {'Hit Rate':<10} {'時間':<8} {'改善':<8}")
    print("-" * 75)

    baseline_accuracy = None
    for i, result in enumerate(results):
        if 'error' not in result:
            if i == 0:  # 最初の結果をベースライン
                baseline_accuracy = result['accuracy']
                improvement = "+0.00%"
            else:
                improvement = f"+{result['accuracy'] - baseline_accuracy:.2f}%"

            print(f"{result['name']:<25} {result['accuracy']:6.2f}% {result['r2']:6.3f}  {result['hit_rate']:6.3f}     {result['time']:6.1f}s {improvement}")
        else:
            print(f"{result['name']:<25} {'ERROR':<8}")

    print(f"\n最高精度: {best_accuracy:.2f}%")
    print(f"最適設定: {best_config_name}")
    print(f"95%達成まで: {95.0 - best_accuracy:.2f}%の改善が必要")

    # 進捗評価
    progress_to_95 = (best_accuracy / 95.0) * 100
    print(f"95%達成進捗: {progress_to_95:.1f}%")

    # 95%達成評価と次のステップ
    if best_accuracy >= 95.0:
        print("\n🎉 95%達成済み！Issue #462完了！")
        achievement_level = "完璧"
        next_steps = ["更なる精度向上の実験", "実際の株式データでの検証", "本番環境での導入検討"]
    elif best_accuracy >= 90.0:
        print("\n✅ 90%超達成 - 95%まであと少し！")
        achievement_level = "優秀"
        next_steps = [
            "1. ハイパーパラメータ最適化（Optuna等）の有効化",
            "2. より高度な特徴量エンジニアリング",
            "3. LSTM-Transformerモデルの追加",
            "4. クロスバリデーションによる重み最適化"
        ]
    elif best_accuracy >= 85.0:
        print("\n🟡 85%超達成 - 良いペースです！")
        achievement_level = "良好"
        next_steps = [
            "1. より多様なベースモデルの追加",
            "2. データ品質向上とノイズ除去",
            "3. 特徴量選択とエンジニアリング改善",
            "4. アンサンブル手法の高度化"
        ]
    elif best_accuracy >= 80.0:
        print("\n🟠 80%超達成 - 着実に進歩！")
        achievement_level = "まずまず"
        next_steps = [
            "1. データサイズとデータ品質の改善",
            "2. より多くのベースモデル（XGBoost等）",
            "3. 特徴量の根本的見直し",
            "4. 外部データソースの統合検討"
        ]
    else:
        print("\n🔴 80%未満 - さらなる改善が必要")
        achievement_level = "改善要"
        next_steps = [
            "1. データ生成アルゴリズムの見直し",
            "2. モデルアーキテクチャの根本的改善",
            "3. より高度な前処理パイプライン",
            "4. 深層学習アプローチの検討"
        ]

    print(f"\n達成レベル: {achievement_level}")
    print("\n次のステップ:")
    for step in next_steps:
        print(f"  {step}")

    print("\n" + "=" * 75)

    return best_accuracy, results


if __name__ == "__main__":
    accuracy, results = run_fast_enhanced_benchmark()

    print(f"\n🎯 Issue #462 進捗評価:")
    print(f"現在の最高精度: {accuracy:.2f}%")
    print(f"95%達成率: {(accuracy/95.0)*100:.1f}%")

    if accuracy >= 95.0:
        print("🏆 Issue #462 完了！95%精度達成済み！")
    elif accuracy >= 90.0:
        print("🚀 Issue #462 ほぼ完了！もう一歩で95%達成！")
    elif accuracy >= 85.0:
        print("📈 Issue #462 順調に進行中！85%超達成！")
    else:
        print("🔧 Issue #462 継続作業中。更なる改善が必要。")

    print("=" * 75)