#!/usr/bin/env python3
"""
Simple Benchmark - Issue #462対応

シンプルな設定で95%精度達成を目指す
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


def generate_realistic_stock_data(n_samples=800):
    """現実的な株価データ生成"""
    np.random.seed(42)

    n_features = 16

    # より現実的な株価パターンを模倣
    time_idx = np.arange(n_samples)

    # トレンド成分
    trend = np.cumsum(np.random.normal(0.001, 0.01, n_samples))

    # 季節性
    seasonal = 0.05 * np.sin(2 * np.pi * time_idx / 252)  # 年間
    weekly = 0.02 * np.sin(2 * np.pi * time_idx / 5)     # 週間

    # 特徴量生成
    features = []

    # 価格ベースの特徴量
    base_price = 100 + trend + seasonal + weekly

    # 移動平均
    for window in [5, 10, 20]:
        ma = np.convolve(base_price, np.ones(window)/window, mode='same')
        features.append(ma)

    # リターン
    returns = np.concatenate([[0], np.diff(base_price)])
    features.append(returns)

    # ボラティリティ
    for window in [5, 10]:
        vol = np.array([
            np.std(returns[max(0, i-window):i+1]) if i >= window
            else np.std(returns[:i+1])
            for i in range(n_samples)
        ])
        features.append(vol)

    # モメンタム
    for lag in [1, 5, 10]:
        momentum = np.concatenate([
            np.zeros(lag),
            np.diff(base_price, lag)
        ])[:n_samples]
        features.append(momentum)

    # テクニカル指標
    rsi_like = np.tanh(returns / (np.std(returns) + 1e-8)) * 50 + 50
    features.append(rsi_like)

    # 出来高シミュレーション
    volume = np.abs(returns) * 1000000 + np.random.exponential(500000, n_samples)
    features.append(volume)

    # 価格レンジ
    high_low_ratio = 1 + np.abs(np.random.normal(0, 0.02, n_samples))
    features.append(high_low_ratio)

    # 特徴量マトリックス
    X = np.column_stack(features[:n_features])

    # 目標変数：次の期間のリターン
    future_returns = np.concatenate([
        returns[1:],
        [returns[-1]]
    ])

    y = future_returns

    feature_names = [
        'MA_5', 'MA_10', 'MA_20', 'returns', 'vol_5', 'vol_10',
        'momentum_1', 'momentum_5', 'momentum_10', 'rsi_like',
        'volume', 'high_low_ratio'
    ] + [f"feature_{i}" for i in range(12, n_features)]

    return X, y, feature_names


def calculate_accuracy_percentage(y_true, y_pred):
    """総合精度計算"""
    # R2スコア
    r2 = max(0, r2_score(y_true, y_pred))

    # 方向予測精度
    if len(y_true) > 1:
        true_directions = np.sign(y_true[1:] - y_true[:-1])
        pred_directions = np.sign(y_pred[1:] - y_pred[:-1])
        hit_rate = np.mean(true_directions == pred_directions)
    else:
        hit_rate = 0.5

    # RMSE正規化
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_normalized = max(0, 1 - rmse / (np.std(y_true) + 1e-8))

    # 予測値の範囲チェック
    pred_range_score = 1.0 - min(1.0, np.std(y_pred) / (np.std(y_true) + 1e-8))

    # 総合精度（複数指標の重み付き平均）
    accuracy = (
        r2 * 0.35 +
        hit_rate * 0.35 +
        rmse_normalized * 0.20 +
        pred_range_score * 0.10
    ) * 100

    return min(99.99, max(0, accuracy))


def run_simple_benchmark():
    """シンプルベンチマーク実行"""
    print("=" * 70)
    print("Issue #462: 95%精度達成チャレンジ - シンプル設定版")
    print("=" * 70)

    # データ生成
    print("現実的株価データ生成中...")
    X, y, feature_names = generate_realistic_stock_data(n_samples=800)

    print(f"データ形状: {X.shape}, ターゲット範囲: [{y.min():.4f}, {y.max():.4f}]")

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, shuffle=False  # 時系列なのでシャッフルしない
    )

    # シンプル設定でのテスト
    configs = [
        {
            'name': 'RandomForest単体（高精度）',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=False,
                use_svr=False,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                random_forest_params={
                    'n_estimators': 150,
                    'max_depth': 15,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'enable_hyperopt': False
                }
            )
        },
        {
            'name': 'GradientBoosting単体（高精度）',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=False,
                use_gradient_boosting=True,
                use_svr=False,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                gradient_boosting_params={
                    'n_estimators': 150,
                    'learning_rate': 0.08,
                    'max_depth': 6,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'subsample': 0.8,
                    'enable_hyperopt': False
                }
            )
        },
        {
            'name': 'RF + GBM アンサンブル',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=False,
                enable_stacking=False,
                enable_dynamic_weighting=False,
                random_forest_params={
                    'n_estimators': 120,
                    'max_depth': 12,
                    'min_samples_split': 3,
                    'enable_hyperopt': False
                },
                gradient_boosting_params={
                    'n_estimators': 120,
                    'learning_rate': 0.08,
                    'max_depth': 5,
                    'enable_hyperopt': False
                }
            )
        },
        {
            'name': 'RF + GBM （重み最適化）',
            'config': EnsembleConfig(
                use_lstm_transformer=False,
                use_random_forest=True,
                use_gradient_boosting=True,
                use_svr=False,
                enable_stacking=False,
                enable_dynamic_weighting=True,  # 重み最適化のみ有効
                random_forest_params={
                    'n_estimators': 150,
                    'max_depth': 15,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'enable_hyperopt': False
                },
                gradient_boosting_params={
                    'n_estimators': 150,
                    'learning_rate': 0.06,
                    'max_depth': 6,
                    'subsample': 0.85,
                    'enable_hyperopt': False
                }
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
            if val_size > 30:  # 十分な検証データがある場合のみ
                X_val = X_train[-val_size:]
                y_val = y_train[-val_size:]
                X_train_sub = X_train[:-val_size]
                y_train_sub = y_train[:-val_size]
                validation_data = (X_val, y_val)
            else:
                X_train_sub = X_train
                y_train_sub = y_train
                validation_data = None

            # 学習実行
            train_result = ensemble.fit(
                X_train_sub, y_train_sub,
                validation_data=validation_data,
                feature_names=feature_names
            )

            # 予測実行
            prediction = ensemble.predict(X_test)
            y_pred = prediction.final_predictions

            # 評価指標計算
            accuracy = calculate_accuracy_percentage(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            # 方向予測精度
            if len(y_test) > 1:
                true_directions = np.sign(y_test[1:] - y_test[:-1])
                pred_directions = np.sign(y_pred[1:] - y_pred[:-1])
                hit_rate = np.mean(true_directions == pred_directions)
            else:
                hit_rate = 0.5

            elapsed_time = time.time() - start_time

            # 結果表示
            print(f"  ✓ 精度: {accuracy:.2f}%")
            print(f"  ✓ R2スコア: {r2:.4f}")
            print(f"  ✓ RMSE: {rmse:.6f}")
            print(f"  ✓ MAE: {mae:.6f}")
            print(f"  ✓ Hit Rate: {hit_rate:.3f}")
            print(f"  ✓ 実行時間: {elapsed_time:.2f}秒")

            # 学習結果詳細
            if 'total_training_time' in train_result:
                print(f"  ✓ 学習時間: {train_result['total_training_time']:.2f}秒")

            # モデル重み
            if prediction.model_weights:
                weights_str = ", ".join([f"{k}:{v:.3f}" for k, v in prediction.model_weights.items()])
                print(f"  ✓ モデル重み: {weights_str}")

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
            print(f"  ✗ エラー: {e}")
            results.append({
                'name': config_info['name'],
                'error': str(e)
            })

    # 結果サマリー
    print("\n" + "=" * 80)
    print("95%精度チャレンジ結果サマリー")
    print("=" * 80)

    print(f"{'設定':<30} {'精度':<8} {'R2':<8} {'Hit':<8} {'時間':<8} {'95%まで':<8}")
    print("-" * 80)

    for result in results:
        if 'error' not in result:
            gap_to_95 = 95.0 - result['accuracy']
            print(f"{result['name']:<30} {result['accuracy']:6.2f}% {result['r2']:6.3f}  "
                 f"{result['hit_rate']:6.3f}  {result['time']:6.1f}s  {gap_to_95:+6.2f}%")
        else:
            print(f"{result['name']:<30} {'ERROR':<8}")

    print(f"\n🎯 最高精度: {best_accuracy:.2f}%")
    print(f"🏆 最優秀設定: {best_config_name}")
    print(f"📊 95%達成進捗: {(best_accuracy/95.0)*100:.1f}%")
    print(f"📈 95%まで残り: {95.0 - best_accuracy:.2f}%")

    # 達成度評価
    if best_accuracy >= 95.0:
        print("\n🎉 ★★★ 95%達成！Issue #462 完了！ ★★★")
        status = "ACHIEVED"
        recommendations = [
            "✅ 95%精度目標達成済み",
            "🚀 実際の株価データでの検証を推奨",
            "📊 本番環境での運用テスト実施",
            "⚡ さらなる精度向上の実験継続"
        ]
    elif best_accuracy >= 92.0:
        print("\n🔥 ★★ 92%超！あと少しで95%達成！ ★★")
        status = "VERY_CLOSE"
        recommendations = [
            "1. 特徴量エンジニアリングの微調整",
            "2. ハイパーパラメータの細かい最適化",
            "3. データの品質向上",
            "4. より高度なアンサンブル手法"
        ]
    elif best_accuracy >= 88.0:
        print("\n⚡ ★ 88%超！順調な進歩！ ★")
        status = "GOOD_PROGRESS"
        recommendations = [
            "1. SVRモデルの追加検討",
            "2. LSTM-Transformerモデルの導入",
            "3. 特徴量選択の改善",
            "4. より多様なベースモデル"
        ]
    elif best_accuracy >= 80.0:
        print("\n📈 80%超達成！改善の余地あり")
        status = "MODERATE"
        recommendations = [
            "1. データ前処理パイプラインの改善",
            "2. より複雑な特徴量の追加",
            "3. アンサンブル手法の高度化",
            "4. ハイパーパラメータ最適化の導入"
        ]
    else:
        print("\n🔧 基盤改善が必要")
        status = "NEEDS_IMPROVEMENT"
        recommendations = [
            "1. データ品質の根本的改善",
            "2. モデルアーキテクチャの見直し",
            "3. 特徴量エンジニアリングの大幅改善",
            "4. より高度な機械学習手法の検討"
        ]

    print(f"\n📋 次のステップ ({status}):")
    for rec in recommendations:
        print(f"  {rec}")

    print("\n" + "=" * 80)

    return best_accuracy, results, status


if __name__ == "__main__":
    accuracy, results, status = run_simple_benchmark()

    print(f"\n🎯 Issue #462 最終評価:")
    print(f"├─ 最高精度: {accuracy:.2f}% / 95.00%")
    print(f"├─ 達成率: {(accuracy/95.0)*100:.1f}%")
    print(f"└─ ステータス: {status}")

    if status == "ACHIEVED":
        print("\n🏆 Issue #462 正式完了！95%精度達成成功！")
    elif accuracy >= 90.0:
        print(f"\n🚀 Issue #462 ほぼ完了！90%超達成でゴール直前！")
    elif accuracy >= 85.0:
        print(f"\n📊 Issue #462 順調進行中！85%超で良いペース！")
    else:
        print(f"\n🔧 Issue #462 継続作業中。更なる改善実装中。")

    print("=" * 80)