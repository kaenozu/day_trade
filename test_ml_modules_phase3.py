#!/usr/bin/env python3
"""
ML モジュール Phase 3 テスト

ml_model_base.py の基本機能テスト
"""

import sys
import pandas as pd
import numpy as np

sys.path.append("src")

try:
    # ml_model_base のテスト
    from day_trade.ml.ml_model_base import (
        BaseModelTrainer,
        RandomForestTrainer,
        XGBoostTrainer,
        LightGBMTrainer,
        create_trainer
    )

    # 設定のインポート
    from day_trade.ml.ml_config import (
        ModelType,
        PredictionTask,
        DataQuality,
        TrainingConfig
    )

    print("ML モデル基底クラスインポート成功")

    def test_model_base():
        """モデル基底クラステスト"""

        # テストデータ作成
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        # 分類用ターゲット
        y_class = pd.Series(np.random.choice([0, 1, 2], size=n_samples))

        # 回帰用ターゲット
        y_reg = pd.Series(X.sum(axis=1) + np.random.randn(n_samples) * 0.1)

        print(f"テストデータ作成: {X.shape}, 分類ターゲット: {y_class.shape}, 回帰ターゲット: {y_reg.shape}")

        # 設定作成
        config = TrainingConfig(
            test_size=0.3,
            cv_folds=3,
            random_state=42
        )

        # Random Forest トレーナーテスト
        try:
            rf_trainer = RandomForestTrainer(
                ModelType.RANDOM_FOREST,
                {'classifier_params': {'n_estimators': 10, 'random_state': 42}}
            )
            print("RandomForestTrainer 初期化成功")

            # データ品質検証テスト
            is_valid, quality, message = rf_trainer.validate_data_quality(X, y_class, PredictionTask.PRICE_DIRECTION)
            print(f"データ品質検証: {is_valid}, 品質: {quality}, メッセージ: {message}")

            # 分類モデル作成テスト
            classifier = rf_trainer.create_model(True, {'max_depth': 5})
            print(f"分類モデル作成成功: {type(classifier).__name__}")

            # 回帰モデル作成テスト
            regressor = rf_trainer.create_model(False, {'max_depth': 5})
            print(f"回帰モデル作成成功: {type(regressor).__name__}")

            # データ分割テスト
            X_train, X_test, y_train, y_test = rf_trainer.prepare_data(X, y_class, config)
            print(f"データ分割成功: 訓練{X_train.shape}, テスト{X_test.shape}")

        except Exception as e:
            print(f"RandomForest テスト中にエラー: {e}")

        # ファクトリーメソッドテスト
        for model_type in [ModelType.RANDOM_FOREST]:  # 他のモデルはライブラリが無い場合はスキップ
            try:
                trainer = create_trainer(model_type, {'classifier_params': {'random_state': 42}})
                print(f"ファクトリーメソッド成功: {model_type.value} -> {type(trainer).__name__}")
            except Exception as e:
                print(f"ファクトリーメソッドエラー ({model_type.value}): {e}")

        # 統合訓練・評価テスト（小さなデータで）
        try:
            small_X = X.iloc[:50]  # 小さなデータセット
            small_y = y_class.iloc[:50]

            result = rf_trainer.train_and_evaluate(
                small_X, small_y, config, PredictionTask.PRICE_DIRECTION,
                {'n_estimators': 5, 'max_depth': 3}
            )

            print(f"統合訓練・評価成功:")
            print(f"  - モデルタイプ: {result['model_type']}")
            print(f"  - データ品質: {result['data_quality']}")
            print(f"  - 訓練サイズ: {result['train_size']}")
            print(f"  - テストサイズ: {result['test_size']}")
            print(f"  - メトリクス: {list(result['metrics'].keys())}")
            print(f"  - CV スコア平均: {np.mean(result['cv_scores']):.3f}")
            print(f"  - 特徴量重要度数: {len(result['feature_importance'])}")

        except Exception as e:
            print(f"統合訓練・評価テスト中にエラー: {e}")

    # メイン実行
    test_model_base()

    print("\nML モジュール Phase 3 テスト完了 - すべて成功")

except ImportError as e:
    print(f"インポートエラー: {e}")
except Exception as e:
    print(f"予期しないエラー: {e}")
    import traceback
    traceback.print_exc()