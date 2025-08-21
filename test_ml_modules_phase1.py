#!/usr/bin/env python3
"""
ML モジュール Phase 1 テスト

ml_exceptions.py と ml_config.py の基本機能テスト
"""

import sys
sys.path.append("src")

try:
    # ml_exceptions のテスト
    from day_trade.ml.ml_exceptions import (
        MLPredictionError,
        DataPreparationError,
        ModelTrainingError,
        ModelMetadataError,
        PredictionError
    )

    print("ML例外クラスインポート成功")

    # 例外階層テスト
    try:
        raise DataPreparationError("データ準備エラーのテスト")
    except MLPredictionError as e:
        print(f"例外階層確認成功: {type(e).__name__}: {e}")

    # ml_config のテスト
    from day_trade.ml.ml_config import (
        ModelType,
        PredictionTask,
        DataQuality,
        TrainingConfig
    )

    print("ML設定クラスインポート成功")

    # 列挙型テスト
    model_type = ModelType.RANDOM_FOREST
    print(f"ModelType テスト成功: {model_type.value}")

    task = PredictionTask.PRICE_DIRECTION
    print(f"PredictionTask テスト成功: {task.value}")

    quality = DataQuality.GOOD
    print(f"DataQuality テスト成功: {quality.value}")

    # 設定クラステスト
    config = TrainingConfig()
    print(f"TrainingConfig テスト成功:")
    print(f"  - test_size: {config.test_size}")
    print(f"  - cv_folds: {config.cv_folds}")
    print(f"  - min_data_quality: {config.min_data_quality.value}")

    # カスタム設定テスト
    custom_config = TrainingConfig(
        test_size=0.3,
        cv_folds=10,
        min_data_quality=DataQuality.EXCELLENT
    )
    print(f"カスタム設定テスト成功:")
    print(f"  - test_size: {custom_config.test_size}")
    print(f"  - cv_folds: {custom_config.cv_folds}")
    print(f"  - min_data_quality: {custom_config.min_data_quality.value}")

    print("\nML モジュール Phase 1 テスト完了 - すべて成功")

except ImportError as e:
    print(f"インポートエラー: {e}")
except Exception as e:
    print(f"予期しないエラー: {e}")
    import traceback
    traceback.print_exc()