#!/usr/bin/env python3
"""
シーケンシャル実行によるXGBoost・CatBoostモデル統合テスト

並列実行を避けて、順次実行でモデル統合をテスト
Issue #462: 95%精度実現のためのアンサンブルシステム統合確認
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.day_trade.ml.ensemble_system import EnsembleSystem, EnsembleConfig
from src.day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

def create_test_data(n_samples: int = 500, n_features: int = 12):
    """テストデータ生成"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    # 非線形関係を含む目標変数を生成
    y = (np.sum(X[:, :5], axis=1) +
         0.5 * np.sin(X[:, 5]) * X[:, 6] +
         0.1 * np.random.randn(n_samples))
    return X, y

def test_sequential_ensemble():
    """シーケンシャル実行でのアンサンブルテスト"""
    print("=== シーケンシャル実行 XGBoost・CatBoost統合テスト ===")

    # テストデータ生成
    X, y = create_test_data()
    print(f"[データ] 形状: {X.shape}, 目標変数: {y.shape}")

    # データ分割
    split_idx = int(0.75 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # アンサンブル設定（シーケンシャル実行）
    config = EnsembleConfig(
        # 高速テスト用設定
        use_lstm_transformer=False,
        use_random_forest=True,
        use_gradient_boosting=False,
        use_svr=False,
        use_xgboost=True,
        use_catboost=True,

        # 並列化無効
        n_jobs=1,
        verbose=True,

        # ハイパーパラメータ最適化無効（高速化）
        random_forest_params={
            'n_estimators': 50,
            'max_depth': 10,
            'enable_hyperopt': False,
        },
        xgboost_params={
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.1,
            'enable_hyperopt': False,
        },
        catboost_params={
            'iterations': 50,
            'depth': 4,
            'learning_rate': 0.1,
            'enable_hyperopt': False,
            'verbose': 0,
        }
    )

    # アンサンブルシステム初期化
    ensemble = EnsembleSystem(config)
    print(f"[初期化] 使用モデル: {list(ensemble.base_models.keys())}")
    print(f"[初期化] 初期重み: {ensemble.model_weights}")

    # 特徴量名設定
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # 学習実行
    print("\n[学習開始] シーケンシャル実行...")
    try:
        training_results = ensemble.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            feature_names=feature_names
        )

        print(f"[学習完了] 総時間: {training_results['total_training_time']:.2f}秒")
        print(f"[学習完了] 最終重み: {training_results['final_weights']}")

        # 各モデルの学習結果確認
        for model_name, result in training_results['model_results'].items():
            status = result.get('status', '成功')
            print(f"[{model_name}] ステータス: {status}")
            if 'error' in result:
                print(f"[{model_name}] エラー: {result['error']}")

    except Exception as e:
        print(f"[エラー] 学習失敗: {e}")
        return False

    # 予測実行
    print("\n[予測開始]...")
    try:
        ensemble_pred = ensemble.predict(X_test)
        print(f"[予測完了] サンプル数: {len(ensemble_pred.final_predictions)}")
        print(f"[予測完了] 個別モデル予測: {list(ensemble_pred.individual_predictions.keys())}")
        print(f"[予測完了] 最終重み: {ensemble_pred.model_weights}")

        # 精度評価
        from sklearn.metrics import mean_squared_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred.final_predictions))
        r2 = r2_score(y_test, ensemble_pred.final_predictions)

        print(f"\n[結果サマリー]")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Accuracy (%): {max(0, r2 * 100):.2f}%")

        # 95%精度目標との比較
        accuracy_percent = max(0, r2 * 100)
        target_accuracy = 95.0

        if accuracy_percent >= target_accuracy:
            print(f"[SUCCESS] 95%精度目標達成! ({accuracy_percent:.2f}% >= {target_accuracy}%)")
        else:
            print(f"[PROGRESS] 現在の精度: {accuracy_percent:.2f}% (目標: {target_accuracy}%)")
            print(f"[PROGRESS] 改善の余地: {target_accuracy - accuracy_percent:.2f}%")

        return True

    except Exception as e:
        print(f"[エラー] 予測失敗: {e}")
        return False

if __name__ == "__main__":
    success = test_sequential_ensemble()
    print(f"\n=== テスト結果: {'成功' if success else '失敗'} ===")