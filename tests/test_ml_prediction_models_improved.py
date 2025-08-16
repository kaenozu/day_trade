#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for ML Prediction Models (Improved Version)
改善版MLPredictionModelsのテストケース

pytestフレームワークを使用した構造化テスト
"""

import pytest
import asyncio
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

# テスト対象のインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ml_prediction_models_improved import (
        MLPredictionModels,
        ModelType,
        PredictionTask,
        ModelPerformance,
        PredictionResult,
        EnsemblePrediction,
        ModelMetadata,
        TrainingConfig,
        DataQuality,
        DataPreparationPipeline,
        ModelMetadataManager,
        EnhancedEnsemblePredictor,
        BaseModelTrainer,
        RandomForestTrainer,
        create_improved_ml_prediction_models
    )
    ML_IMPROVED_AVAILABLE = True
except ImportError:
    # フォールバック: 元のモジュール
    from ml_prediction_models import (
        MLPredictionModels,
        ModelType,
        PredictionTask,
        ModelPerformance,
        PredictionResult,
        EnsemblePrediction,
        ModelMetadata,
        TrainingConfig
    )
    ML_IMPROVED_AVAILABLE = False

# 条件付きインポート
try:
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class TestMLPredictionModels:
    """MLPredictionModelsのテストクラス"""

    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリの作成"""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def ml_models(self, temp_dir):
        """テスト用MLPredictionModelsインスタンス"""
        if not SKLEARN_AVAILABLE:
            pytest.skip("Scikit-learn not available")

        # 一時ディレクトリでインスタンス作成
        with patch('ml_prediction_models.MLPredictionModels.data_dir', temp_dir):
            return MLPredictionModels()

    def test_initialization(self, ml_models):
        """初期化テスト"""
        assert ml_models.data_dir is not None
        assert ml_models.models_dir is not None
        assert hasattr(ml_models, 'trained_models')
        assert hasattr(ml_models, 'label_encoders')
        assert hasattr(ml_models, 'model_configs')

    def test_model_metadata_creation(self, ml_models):
        """ModelMetadata作成テスト"""
        metadata = ml_models._create_model_metadata(
            model_type=ModelType.RANDOM_FOREST,
            task=PredictionTask.PRICE_DIRECTION,
            feature_names=['feature1', 'feature2'],
            target_columns=['price_direction'],
            training_period="1y",
            training_samples=1000,
            hyperparameters={'n_estimators': 100},
            performance_metrics={'accuracy': 0.85},
            is_classifier=True
        )

        assert isinstance(metadata, ModelMetadata)
        assert metadata.model_type == ModelType.RANDOM_FOREST
        assert metadata.task == PredictionTask.PRICE_DIRECTION
        assert metadata.is_classifier == True
        assert len(metadata.feature_names) == 2
        assert metadata.training_samples == 1000

    def test_training_config_defaults(self):
        """TrainingConfig デフォルト値テスト"""
        config = TrainingConfig()

        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.cv_folds == 5
        assert config.enable_cross_validation == True
        assert config.save_model == True

    def test_model_instance_creation(self, ml_models):
        """モデルインスタンス作成テスト"""
        # Random Forest 分類器
        rf_classifier = ml_models._create_model_instance(
            ModelType.RANDOM_FOREST, is_classifier=True
        )
        assert isinstance(rf_classifier, RandomForestClassifier)

        # Random Forest 回帰器
        rf_regressor = ml_models._create_model_instance(
            ModelType.RANDOM_FOREST, is_classifier=False
        )
        assert isinstance(rf_regressor, RandomForestRegressor)

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Scikit-learn not available")
    def test_performance_metrics_calculation(self, ml_models):
        """性能指標計算テスト"""
        # 分類メトリクス
        y_true_cls = np.array([0, 1, 0, 1, 1])
        y_pred_cls = np.array([0, 1, 1, 1, 0])

        metrics_cls = ml_models._calculate_performance_metrics(
            y_true_cls, y_pred_cls, is_classifier=True,
            model=None, X_test=None
        )

        assert 'accuracy' in metrics_cls
        assert 'precision' in metrics_cls
        assert 'recall' in metrics_cls
        assert 'f1_score' in metrics_cls
        assert 0 <= metrics_cls['accuracy'] <= 1

        # 回帰メトリクス
        y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_reg = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        metrics_reg = ml_models._calculate_performance_metrics(
            y_true_reg, y_pred_reg, is_classifier=False,
            model=None, X_test=None
        )

        assert 'r2_score' in metrics_reg
        assert 'mse' in metrics_reg
        assert 'rmse' in metrics_reg
        assert 'mae' in metrics_reg

    def test_feature_importance_extraction(self, ml_models):
        """特徴量重要度抽出テスト"""
        # モック特徴量重要度を持つモデル
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.3, 0.5, 0.2])
        feature_names = ['feature1', 'feature2', 'feature3']

        importance = ml_models._get_feature_importance(mock_model, feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == 3
        assert importance['feature2'] == 0.5  # 最高重要度

        # 特徴量重要度がないモデル
        mock_model_no_importance = Mock()
        del mock_model_no_importance.feature_importances_

        importance_empty = ml_models._get_feature_importance(
            mock_model_no_importance, feature_names
        )
        assert importance_empty == {}

    @pytest.mark.asyncio
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Scikit-learn not available")
    async def test_prepare_training_data_fallback(self, ml_models):
        """訓練データ準備（フォールバック）テスト"""
        symbol = "TEST"

        # real_data_provider が利用できない場合のダミーデータ生成
        with patch('ml_prediction_models.REAL_DATA_PROVIDER_AVAILABLE', False):
            features, targets = await ml_models.prepare_training_data(symbol, "6mo")

        assert isinstance(features, pd.DataFrame)
        assert isinstance(targets, dict)
        assert len(features) > 0
        assert PredictionTask.PRICE_DIRECTION in targets
        assert PredictionTask.PRICE_REGRESSION in targets

    @pytest.mark.asyncio
    async def test_model_summary_empty(self, ml_models):
        """空のモデルサマリーテスト"""
        summary = await ml_models.get_model_summary()

        assert isinstance(summary, dict)
        assert 'trained_models_count' in summary
        assert 'recent_performances' in summary
        assert 'recent_predictions' in summary
        assert summary['trained_models_count'] >= 0

    def test_ensemble_prediction_dataclass(self):
        """EnsemblePrediction データクラステスト"""
        prediction = EnsemblePrediction(
            symbol="TEST",
            timestamp=datetime.now(),
            final_prediction="上昇",
            confidence=0.85,
            model_predictions={"RF": "上昇", "XGB": "上昇"},
            model_confidences={"RF": 0.8, "XGB": 0.9},
            model_weights={"RF": 0.4, "XGB": 0.6},
            consensus_strength=0.9,
            disagreement_score=0.1
        )

        assert prediction.symbol == "TEST"
        assert prediction.confidence == 0.85
        assert prediction.consensus_strength == 0.9
        assert len(prediction.model_predictions) == 2

    @pytest.mark.asyncio
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Scikit-learn not available")
    async def test_cross_validation(self, ml_models):
        """クロスバリデーションテスト"""
        # テストデータ作成
        X, y = make_classification(n_samples=100, n_features=5,
                                 n_informative=3, random_state=42)
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)

        # ダミーモデル
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        cv_scores = ml_models._perform_cross_validation(
            model, X_df, y_series, cv_folds=3, is_classifier=True
        )

        assert len(cv_scores) == 3
        assert all(0 <= score <= 1 for score in cv_scores)

class TestDataClasses:
    """データクラスのテスト"""

    def test_model_metadata_serialization(self):
        """ModelMetadata シリアライゼーションテスト"""
        metadata = ModelMetadata(
            model_type=ModelType.RANDOM_FOREST,
            task=PredictionTask.PRICE_DIRECTION,
            version="RF_PRICE_DIRECTION_20250101_120000",
            created_at=datetime(2025, 1, 1, 12, 0, 0),
            feature_names=['price', 'volume'],
            target_columns=['direction'],
            training_period="1y",
            training_samples=1000,
            hyperparameters={'n_estimators': 100},
            preprocessing_info={'scaler': 'StandardScaler'},
            performance_metrics={'accuracy': 0.85},
            is_classifier=True,
            model_size_mb=1.5,
            python_version="3.9.0",
            sklearn_version="1.0.0"
        )

        # 基本属性チェック
        assert metadata.model_type == ModelType.RANDOM_FOREST
        assert metadata.training_samples == 1000
        assert metadata.is_classifier == True
        assert metadata.model_size_mb == 1.5

    def test_training_config_customization(self):
        """TrainingConfig カスタマイズテスト"""
        custom_config = TrainingConfig(
            test_size=0.3,
            cv_folds=10,
            enable_cross_validation=False,
            save_model=False,
            preprocessing={'normalize': True}
        )

        assert custom_config.test_size == 0.3
        assert custom_config.cv_folds == 10
        assert custom_config.enable_cross_validation == False
        assert custom_config.save_model == False
        assert custom_config.preprocessing['normalize'] == True

    def test_prediction_result_creation(self):
        """PredictionResult 作成テスト"""
        result = PredictionResult(
            symbol="7203",
            timestamp=datetime.now(),
            model_type=ModelType.XGBOOST,
            task=PredictionTask.PRICE_REGRESSION,
            prediction=1500.0,
            confidence=0.92,
            probability_distribution={'up': 0.6, 'down': 0.4},
            feature_values={'ma_5': 1480, 'volume': 1000000},
            model_version="v1.0",
            explanation="Strong upward trend detected"
        )

        assert result.symbol == "7203"
        assert result.model_type == ModelType.XGBOOST
        assert result.task == PredictionTask.PRICE_REGRESSION
        assert isinstance(result.prediction, float)
        assert result.confidence == 0.92

# 統合テスト
class TestIntegration:
    """統合テスト"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Scikit-learn not available")
    async def test_full_workflow_simulation(self, temp_dir):
        """全体ワークフローのシミュレーションテスト"""
        # テスト用MLPredictionModels
        with patch('ml_prediction_models.MLPredictionModels.data_dir', temp_dir):
            models = MLPredictionModels()

        # データ準備のテスト
        with patch('ml_prediction_models.REAL_DATA_PROVIDER_AVAILABLE', False):
            features, targets = await models.prepare_training_data("TEST", "3mo")

        # データ検証
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 50  # 十分なサンプル数
        assert len(targets) >= 2   # 複数のタスク

        # 基本的な予測テスト（ダミーデータで）
        latest_features = features.tail(1)

        # モデルが訓練されていない状態での予測
        predictions = await models.predict("TEST", latest_features)

        # 訓練されていないモデルでは空の予測が返される
        assert isinstance(predictions, list)

if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])