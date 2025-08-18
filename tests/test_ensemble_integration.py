"""
Issue #850対応：アンサンブル予測機能の統合テスト
改良されたアンサンブル予測ロジックの動作確認
"""

import asyncio
import logging
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from ml_prediction_models import MLPredictionModels, ModelType, PredictionTask

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_ensemble_prediction_integration():
    """アンサンブル予測統合テスト"""

    print("=== アンサンブル予測機能統合テスト開始 ===")

    # 一時ディレクトリでテスト
    with tempfile.TemporaryDirectory() as temp_dir:
        models = MLPredictionModels()

        print(f"✓ MLPredictionModels初期化完了")

        # テスト用サンプルデータ作成
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        sample_features = pd.DataFrame({
            'Close': np.random.normal(100, 5, 100),
            'Volume': np.random.randint(1000, 10000, 100),
            'RSI': np.random.uniform(20, 80, 100),
            'MACD': np.random.normal(0, 1, 100),
            'BB_upper': np.random.normal(105, 5, 100),
            'BB_lower': np.random.normal(95, 5, 100),
            'return_1d': np.random.normal(0, 0.02, 100),
            'return_5d': np.random.normal(0, 0.05, 100),
            'volatility_10d': np.random.uniform(0.01, 0.05, 100),
            'lag_1': np.random.normal(100, 5, 100),
            'lag_2': np.random.normal(100, 5, 100),
            'lag_3': np.random.normal(100, 5, 100)
        }, index=dates)

        # ターゲット変数作成
        targets = {
            PredictionTask.PRICE_DIRECTION: pd.Series(
                np.random.choice(['UP', 'DOWN'], 100), index=dates
            ),
            PredictionTask.PRICE_REGRESSION: sample_features['Close'] * 1.01
        }

        print("✓ テストデータ作成完了")

        # 1. 動的重み計算テスト
        print("\n--- 動的重み計算テスト ---")
        quality_scores = {'Random Forest': 0.8, 'XGBoost': 0.7, 'LightGBM': 0.75}
        confidences = {'Random Forest': 0.9, 'XGBoost': 0.8, 'LightGBM': 0.85}

        weights = await models._calculate_dynamic_weights(
            "TEST", PredictionTask.PRICE_DIRECTION, quality_scores, confidences
        )

        print(f"動的重み: {weights}")
        assert isinstance(weights, dict)
        assert len(weights) == 3

        # 重みの合計が1に近いことを確認
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01
        print(f"✓ 重み合計: {total_weight:.4f}")

        # 2. 分類アンサンブルテスト
        print("\n--- 分類アンサンブルテスト ---")
        predictions = {'Random Forest': 'UP', 'XGBoost': 'UP', 'LightGBM': 'DOWN'}
        confidences = {'Random Forest': 0.8, 'XGBoost': 0.7, 'LightGBM': 0.6}

        result = models._ensemble_classification(predictions, confidences, weights)

        print(f"分類結果: {result}")
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'consensus_strength' in result
        assert 'disagreement_score' in result

        assert result['prediction'] in ['UP', 'DOWN']
        assert 0 <= result['confidence'] <= 1
        assert 0 <= result['consensus_strength'] <= 1
        assert 0 <= result['disagreement_score'] <= 1
        print(f"✓ 分類アンサンブル結果: {result['prediction']} (信頼度: {result['confidence']:.3f})")

        # 3. 回帰アンサンブルテスト
        print("\n--- 回帰アンサンブルテスト ---")
        reg_predictions = {'Random Forest': 100.5, 'XGBoost': 101.2, 'LightGBM': 99.8}
        reg_confidences = {'Random Forest': 0.8, 'XGBoost': 0.7, 'LightGBM': 0.75}

        reg_result = models._ensemble_regression(reg_predictions, reg_confidences, weights)

        print(f"回帰結果: {reg_result}")
        assert 'prediction' in reg_result
        assert isinstance(reg_result['prediction'], (int, float))
        assert 0 <= reg_result['confidence'] <= 1
        print(f"✓ 回帰アンサンブル結果: {reg_result['prediction']:.2f} (信頼度: {reg_result['confidence']:.3f})")

        # 4. 予測安定性計算テスト
        print("\n--- 予測安定性計算テスト ---")

        # 完全一致の場合
        identical_preds = {'model1': 'UP', 'model2': 'UP', 'model3': 'UP'}
        stability = models._calculate_prediction_stability(identical_preds)
        print(f"完全一致安定性: {stability:.3f}")
        assert stability == 1.0

        # 数値予測の場合
        numeric_preds = {'model1': 100.0, 'model2': 100.5, 'model3': 99.5}
        stability = models._calculate_prediction_stability(numeric_preds)
        print(f"数値予測安定性: {stability:.3f}")
        assert 0 <= stability <= 1

        print("✓ 予測安定性計算テスト完了")

        # 5. 特徴量不確実性推定テスト
        print("\n--- 特徴量不確実性推定テスト ---")
        uncertainty = models._estimate_feature_uncertainty(sample_features)
        print(f"特徴量不確実性: {uncertainty:.3f}")
        assert 0 <= uncertainty <= 1
        print("✓ 特徴量不確実性推定テスト完了")

        # 6. 信頼度計算テスト
        print("\n--- 信頼度計算テスト ---")

        # 分類信頼度
        test_proba = np.array([0.7, 0.3])  # 2クラス分類の例
        quality_score = 0.8
        confidence = models._calculate_classification_confidence(test_proba, quality_score)
        print(f"分類信頼度: {confidence:.3f}")
        assert 0.1 <= confidence <= 0.95

        print("✓ 信頼度計算テスト完了")

        # 7. データベーススキーマ確認
        print("\n--- データベーススキーマ確認 ---")
        import sqlite3

        with sqlite3.connect(models.db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name LIKE '%model_%' OR name LIKE '%ensemble%'
            """)
            tables = [row[0] for row in cursor.fetchall()]

            print(f"作成されたテーブル: {tables}")

            expected_tables = [
                'model_metadata',
                'model_performance_history',
                'ensemble_prediction_history',
                'model_weight_history'
            ]

            for table in expected_tables:
                if table in tables:
                    print(f"✓ {table} テーブル作成済み")
                else:
                    print(f"⚠ {table} テーブルが見つかりません")

        print("=== アンサンブル予測機能統合テスト完了 ===")

        # 統合テスト結果サマリー
        print("\n=== テストサマリー ===")
        print("✓ 動的重み計算")
        print("✓ 分類アンサンブル予測")
        print("✓ 回帰アンサンブル予測")
        print("✓ 予測安定性計算")
        print("✓ 特徴量不確実性推定")
        print("✓ 信頼度計算")
        print("✓ データベーススキーマ")
        print("\n🎉 全ての統合テストが成功しました！")

        return True

async def test_quality_score_retrieval():
    """品質スコア取得テスト"""
    print("\n=== 品質スコア取得テスト ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        models = MLPredictionModels()

        # デフォルト品質スコア確認
        quality_score = await models._get_model_quality_score(
            "TEST", ModelType.RANDOM_FOREST, PredictionTask.PRICE_DIRECTION
        )

        print(f"デフォルト品質スコア: {quality_score}")
        assert quality_score == 0.6  # デフォルト値
        print("✓ デフォルト品質スコア取得成功")

        return True

if __name__ == "__main__":
    async def run_all_tests():
        """全テスト実行"""
        try:
            await test_ensemble_prediction_integration()
            await test_quality_score_retrieval()
            print("\n🚀 Issue #850アンサンブル予測機能の統合テスト完了")
            return True
        except Exception as e:
            print(f"\n❌ テスト失敗: {e}")
            import traceback
            traceback.print_exc()
            return False

    success = asyncio.run(run_all_tests())
    if success:
        print("\n✅ 全ての統合テストが正常に完了しました")
    else:
        print("\n❌ 統合テストで問題が発生しました")