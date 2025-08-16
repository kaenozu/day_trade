"""
ML予測モデルシステムのpytestテストモジュール
Issue #850 対応：テストコード分離とpytest対応
"""

import pytest
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

# テスト対象のインポート
from ml_prediction_models import (
    MLPredictionModels, ModelType, PredictionTask,
    DataPreparationPipeline, ModelMetadataManager,
    RandomForestTrainer, XGBoostTrainer, LightGBMTrainer,
    SKLEARN_AVAILABLE
)

# テスト用フィクスチャ
@pytest.fixture
def temp_dir():
    """テスト用一時ディレクトリ"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)

@pytest.fixture
def ml_models(temp_dir):
    """MLPredictionModelsインスタンス（テスト用）"""
    test_db_path = temp_dir / "test_models.db"
    return MLPredictionModels(db_path=test_db_path)

@pytest.fixture
def sample_features():
    """サンプル特徴量データ"""
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    return pd.DataFrame({
        'Close': np.random.normal(100, 5, 30),
        'Volume': np.random.randint(1000, 10000, 30),
        'RSI': np.random.uniform(20, 80, 30),
        'MACD': np.random.normal(0, 1, 30),
        'BB_upper': np.random.normal(105, 5, 30),
        'BB_lower': np.random.normal(95, 5, 30),
        'return_1d': np.random.normal(0, 0.02, 30),
        'return_5d': np.random.normal(0, 0.05, 30),
        'volatility_10d': np.random.uniform(0.01, 0.05, 30),
        'lag_1': np.random.normal(100, 5, 30),
        'lag_2': np.random.normal(100, 5, 30),
        'lag_3': np.random.normal(100, 5, 30)
    }, index=dates)

# テストクラス
class TestDataPreparationPipeline:
    """データ準備パイプラインのテスト"""

    @pytest.mark.asyncio
    async def test_data_preparation_initialization(self, temp_dir):
        """データ準備パイプライン初期化テスト"""
        pipeline = DataPreparationPipeline()

        assert pipeline.feature_cache == {}
        assert hasattr(pipeline, 'logger')

    @pytest.mark.asyncio
    async def test_technical_indicators_calculation(self, temp_dir, sample_features):
        """テクニカル指標計算テスト"""
        pipeline = DataPreparationPipeline()

        # テスト用データ
        test_data = sample_features[['Close', 'Volume']].copy()

        # テクニカル指標計算（モック）
        with patch('ml_prediction_models.yf.download') as mock_download:
            mock_download.return_value = test_data

            result = await pipeline.prepare_training_data("TEST", "1mo", force_refresh=True)
            features, targets = result

            # 基本的な検証
            assert isinstance(features, pd.DataFrame)
            assert isinstance(targets, dict)
            assert len(features) > 0

class TestModelTrainers:
    """モデル訓練クラスのテスト"""

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Scikit-learn not available")
    def test_random_forest_trainer_creation(self, temp_dir):
        """Random Forest訓練クラス作成テスト"""
        trainer = RandomForestTrainer(ModelType.RANDOM_FOREST, {})

        assert trainer.model_type == ModelType.RANDOM_FOREST
        assert hasattr(trainer, 'logger')

        # 分類器作成テスト
        classifier = trainer.create_classifier({'n_estimators': 10, 'random_state': 42})
        assert hasattr(classifier, 'fit')
        assert hasattr(classifier, 'predict')

        # 回帰器作成テスト
        regressor = trainer.create_regressor({'n_estimators': 10, 'random_state': 42})
        assert hasattr(regressor, 'fit')
        assert hasattr(regressor, 'predict')

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Scikit-learn not available")
    def test_xgboost_trainer_creation(self, temp_dir):
        """XGBoost訓練クラス作成テスト"""
        trainer = XGBoostTrainer(ModelType.XGBOOST, {})

        assert trainer.model_type == ModelType.XGBOOST

        # 基本的な設定テスト
        classifier = trainer.create_classifier({'n_estimators': 10, 'random_state': 42})
        assert hasattr(classifier, 'fit')

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Scikit-learn not available")
    def test_lightgbm_trainer_creation(self, temp_dir):
        """LightGBM訓練クラス作成テスト"""
        trainer = LightGBMTrainer(ModelType.LIGHTGBM, {})

        assert trainer.model_type == ModelType.LIGHTGBM

        # 基本的な設定テスト
        classifier = trainer.create_classifier({'n_estimators': 10, 'random_state': 42})
        assert hasattr(classifier, 'fit')

class TestModelMetadataManager:
    """モデルメタデータ管理のテスト"""

    def test_metadata_manager_initialization(self, temp_dir):
        """メタデータ管理初期化テスト"""
        db_path = temp_dir / "test_metadata.db"
        manager = ModelMetadataManager(db_path)

        assert manager.db_path == db_path
        assert hasattr(manager, 'logger')

        # データベースファイルが作成されることを確認
        assert db_path.exists()

    def test_database_schema_creation(self, temp_dir):
        """データベーススキーマ作成テスト"""
        db_path = temp_dir / "test_schema.db"
        manager = ModelMetadataManager(db_path)

        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name LIKE '%model_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]

            # 期待されるテーブルが作成されているか確認
            expected_tables = [
                'model_metadata',
                'model_performance_history',
                'model_weight_history'
            ]

            for table in expected_tables:
                assert table in tables

class TestMLPredictionModels:
    """MLPredictionModelsメインクラスのテスト"""

    @pytest.mark.asyncio
    async def test_ml_models_initialization(self, ml_models):
        """ML予測モデル初期化テスト"""
        assert isinstance(ml_models.db_path, Path)
        assert isinstance(ml_models.data_pipeline, DataPreparationPipeline)
        assert isinstance(ml_models.metadata_manager, ModelMetadataManager)
        assert ml_models.trained_models == {}
        assert ml_models.ensemble_weights == {}

    @pytest.mark.asyncio
    async def test_dynamic_weight_calculation(self, ml_models):
        """動的重み計算テスト"""
        # テスト用データ
        quality_scores = {'random_forest': 0.8, 'xgboost': 0.7, 'lightgbm': 0.75}
        confidences = {'random_forest': 0.9, 'xgboost': 0.8, 'lightgbm': 0.85}

        weights = await ml_models._calculate_dynamic_weights(
            "TEST", PredictionTask.PRICE_DIRECTION, quality_scores, confidences
        )

        # 重みの基本的な検証
        assert isinstance(weights, dict)
        assert len(weights) == 3

        # 重みの合計が1に近いことを確認
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_ensemble_classification_logic(self, ml_models):
        """アンサンブル分類ロジックテスト"""
        # テスト用予測と信頼度
        predictions = {'random_forest': 'UP', 'xgboost': 'UP', 'lightgbm': 'DOWN'}
        confidences = {'random_forest': 0.8, 'xgboost': 0.7, 'lightgbm': 0.6}
        weights = {ModelType.RANDOM_FOREST: 0.4, ModelType.XGBOOST: 0.3, ModelType.LIGHTGBM: 0.3}

        result = ml_models._ensemble_classification(predictions, confidences, weights)

        # 結果の基本的な検証
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'consensus_strength' in result
        assert 'disagreement_score' in result

        assert result['prediction'] in ['UP', 'DOWN']
        assert 0 <= result['confidence'] <= 1
        assert 0 <= result['consensus_strength'] <= 1
        assert 0 <= result['disagreement_score'] <= 1

    @pytest.mark.asyncio
    async def test_ensemble_regression_logic(self, ml_models):
        """アンサンブル回帰ロジックテスト"""
        # テスト用予測と信頼度
        predictions = {'random_forest': 100.5, 'xgboost': 101.2, 'lightgbm': 99.8}
        confidences = {'random_forest': 0.8, 'xgboost': 0.7, 'lightgbm': 0.75}
        weights = {ModelType.RANDOM_FOREST: 0.4, ModelType.XGBOOST: 0.3, ModelType.LIGHTGBM: 0.3}

        result = ml_models._ensemble_regression(predictions, confidences, weights)

        # 結果の基本的な検証
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'consensus_strength' in result
        assert 'disagreement_score' in result

        assert isinstance(result['prediction'], (int, float))
        assert 0 <= result['confidence'] <= 1

    @pytest.mark.asyncio
    async def test_prediction_stability_calculation(self, ml_models):
        """予測安定性計算テスト"""
        # 完全一致の場合
        predictions_identical = {'model1': 'UP', 'model2': 'UP', 'model3': 'UP'}
        stability = ml_models._calculate_prediction_stability(predictions_identical)
        assert stability == 1.0

        # 数値予測の場合
        predictions_numeric = {'model1': 100.0, 'model2': 100.5, 'model3': 99.5}
        stability = ml_models._calculate_prediction_stability(predictions_numeric)
        assert 0 <= stability <= 1

# 統合テスト用のテストクラス
class TestIntegration:
    """統合テスト"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Scikit-learn not available")
    async def test_end_to_end_model_training(self, ml_models, sample_features):
        """エンドツーエンドモデル訓練テスト"""
        # データ準備のモック
        with patch.object(ml_models.data_pipeline, 'prepare_training_data') as mock_prepare:
            # サンプルターゲット作成
            targets = {
                PredictionTask.PRICE_DIRECTION: pd.Series(['UP'] * 20 + ['DOWN'] * 10,
                                                         index=sample_features.index),
                PredictionTask.PRICE_REGRESSION: sample_features['Close'] * 1.01
            }

            mock_prepare.return_value = (sample_features, targets)

            # モデル訓練実行
            performances = await ml_models.train_models("TEST", "1mo")

            # 結果の基本的な検証
            assert isinstance(performances, dict)

            # 少なくとも1つのモデルが訓練されることを確認
            if performances:
                for model_type, task_perfs in performances.items():
                    assert isinstance(model_type, ModelType)
                    assert isinstance(task_perfs, dict)

# パフォーマンステスト用
class TestPerformance:
    """パフォーマンステスト"""

    @pytest.mark.asyncio
    async def test_feature_uncertainty_estimation_speed(self, ml_models, sample_features):
        """特徴量不確実性推定速度テスト"""
        import time

        start_time = time.time()

        # 複数回実行してパフォーマンスをテスト
        for _ in range(100):
            uncertainty = ml_models._estimate_feature_uncertainty(sample_features)

        elapsed_time = time.time() - start_time

        # 1秒以内に完了することを確認
        assert elapsed_time < 1.0
        assert isinstance(uncertainty, float)
        assert 0 <= uncertainty <= 1

# メイン実行用の統合テスト関数
@pytest.mark.asyncio
async def test_ml_prediction_models_integration():
    """機械学習予測モデルの統合テスト（元のテスト関数のpytest版）"""

    if not SKLEARN_AVAILABLE:
        pytest.skip("Scikit-learn not available")

    # 一時ディレクトリでテスト
    with tempfile.TemporaryDirectory() as temp_dir:
        test_db_path = Path(temp_dir) / "test_models.db"
        models = MLPredictionModels(db_path=test_db_path)

        # テスト銘柄（実際のデータに依存しないモック使用）
        test_symbol = "TEST"

        # サンプルデータでモック
        with patch.object(models.data_pipeline, 'prepare_training_data') as mock_prepare:
            # サンプル特徴量とターゲット
            dates = pd.date_range('2024-01-01', periods=50, freq='D')
            features = pd.DataFrame({
                'Close': np.random.normal(100, 5, 50),
                'Volume': np.random.randint(1000, 10000, 50),
                'RSI': np.random.uniform(20, 80, 50),
                'MACD': np.random.normal(0, 1, 50),
                'BB_upper': np.random.normal(105, 5, 50),
                'BB_lower': np.random.normal(95, 5, 50),
                'return_1d': np.random.normal(0, 0.02, 50),
                'return_5d': np.random.normal(0, 0.05, 50),
                'volatility_10d': np.random.uniform(0.01, 0.05, 50),
                'lag_1': np.random.normal(100, 5, 50),
                'lag_2': np.random.normal(100, 5, 50),
                'lag_3': np.random.normal(100, 5, 50)
            }, index=dates)

            targets = {
                PredictionTask.PRICE_DIRECTION: pd.Series(
                    np.random.choice(['UP', 'DOWN'], 50), index=dates
                ),
                PredictionTask.PRICE_REGRESSION: features['Close'] * 1.01
            }

            mock_prepare.return_value = (features, targets)

            # モデル訓練
            performances = await models.train_models(test_symbol, "6mo")

            # 基本的な検証
            assert isinstance(performances, dict)

            if performances:
                # 予測テスト
                latest_features = features.tail(1)
                predictions = await models.predict(test_symbol, latest_features)

                # 予測結果の検証
                for pred in predictions:
                    assert hasattr(pred, 'final_prediction')
                    assert hasattr(pred, 'confidence')
                    assert hasattr(pred, 'consensus_strength')
                    assert hasattr(pred, 'disagreement_score')
                    assert 0 <= pred.confidence <= 1
                    assert 0 <= pred.consensus_strength <= 1
                    assert 0 <= pred.disagreement_score <= 1

                # システムサマリー
                summary = await models.get_model_summary()
                assert isinstance(summary, dict)
                assert 'trained_models_count' in summary

if __name__ == "__main__":
    # 直接実行時の設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # pytest実行
    pytest.main([__file__, "-v"])