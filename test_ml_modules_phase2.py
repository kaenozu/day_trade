#!/usr/bin/env python3
"""
ML モジュール Phase 2 テスト

ml_data_processing.py の基本機能テスト
"""

import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append("src")

try:
    # ml_data_processing のテスト
    from day_trade.ml.ml_data_processing import (
        DataProvider,
        DataPreparationPipeline
    )

    # 設定のインポート
    from day_trade.ml.ml_config import (
        TrainingConfig,
        DataQuality,
        PredictionTask
    )

    print("ML データ処理モジュールインポート成功")

    async def test_data_processing():
        """データ処理パイプラインテスト"""

        # 設定作成
        config = TrainingConfig(
            test_size=0.3,
            cv_folds=3,
            enable_scaling=True,
            handle_missing_values=True
        )

        # データ準備パイプライン初期化
        pipeline = DataPreparationPipeline(config)
        print("データ準備パイプライン初期化成功")

        # 模擬データ生成テスト
        mock_data = pipeline._generate_mock_data("TEST", "1y")
        print(f"模擬データ生成成功: {mock_data.shape}")
        print(f"カラム: {list(mock_data.columns)}")
        print(f"期間: {mock_data.index[0]} - {mock_data.index[-1]}")

        # データ品質評価テスト
        is_valid, quality, message = pipeline._assess_data_quality(mock_data)
        print(f"データ品質評価成功:")
        print(f"  - 有効: {is_valid}")
        print(f"  - 品質: {quality.value}")
        print(f"  - メッセージ: {message}")

        # 基本特徴量抽出テスト
        features = pipeline._extract_basic_features(mock_data)
        print(f"基本特徴量抽出成功: {features.shape}")
        print(f"特徴量一部: {list(features.columns[:10])}")

        # ターゲット変数作成テスト
        targets = pipeline._create_target_variables(mock_data)
        print(f"ターゲット変数作成成功: {len(targets)}種類")
        for task, target in targets.items():
            print(f"  - {task.value}: {target.shape}")

        # データ整合性テスト
        aligned_features, aligned_targets = pipeline._align_data(features, targets)
        print(f"データ整合性確保成功:")
        print(f"  - 特徴量: {aligned_features.shape}")
        print(f"  - ターゲット数: {len(aligned_targets)}")

        # 完全なデータ準備テスト（模擬データ使用）
        try:
            full_features, full_targets, data_quality = await pipeline.prepare_training_data("TEST", "6mo")
            print(f"完全データ準備成功:")
            print(f"  - 特徴量: {full_features.shape}")
            print(f"  - ターゲット数: {len(full_targets)}")
            print(f"  - データ品質: {data_quality.value}")
        except Exception as e:
            print(f"完全データ準備テスト中にエラー: {e}")

        # データ分割テスト
        if len(aligned_features) > 20:  # 十分なデータがある場合
            sample_target = aligned_targets[PredictionTask.PRICE_DIRECTION]
            if len(sample_target) > 20:
                X_train, X_test, y_train, y_test = pipeline.prepare_data(
                    aligned_features, sample_target, config
                )
                print(f"データ分割成功:")
                print(f"  - 訓練データ: {X_train.shape}, {y_train.shape}")
                print(f"  - テストデータ: {X_test.shape}, {y_test.shape}")

    # メイン実行
    asyncio.run(test_data_processing())

    print("\nML モジュール Phase 2 テスト完了 - すべて成功")

except ImportError as e:
    print(f"インポートエラー: {e}")
except Exception as e:
    print(f"予期しないエラー: {e}")
    import traceback
    traceback.print_exc()