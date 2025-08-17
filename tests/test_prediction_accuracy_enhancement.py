#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction Accuracy Enhancement テストスイート
Issue #885対応：包括的な予測精度向上システムのテスト

予測精度向上システムの全機能を検証するテストスイート
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

# テスト対象のインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))

from prediction_accuracy_enhancement import (
    PredictionAccuracyEnhancer,
    AdvancedFeatureEngineer,
    EnsembleModelOptimizer,
    HyperparameterOptimizer,
    ValidationStrategyManager,
    OverfittingDetector,
    AccuracyImprovementType,
    ValidationStrategy,
    FeatureEngineeringResult,
    ModelOptimizationResult,
    EnsembleResult,
    AccuracyImprovementReport
)


@pytest.fixture
def temp_dir():
    """一時ディレクトリフィクスチャ"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_stock_data():
    """株価データサンプル"""
    np.random.seed(42)
    n_samples = 500

    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')

    # 株価データ模擬生成
    price = 1000
    prices = []
    for i in range(n_samples):
        change = np.random.normal(0, 0.02)
        price *= (1 + change)
        prices.append(price)

    data = pd.DataFrame({
        'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
        'High': [p * np.random.uniform(1.00, 1.02) for p in prices],
        'Low': [p * np.random.uniform(0.98, 1.00) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(1000000, 10000000) for _ in range(n_samples)]
    }, index=dates)

    # ターゲット作成（翌日上昇かどうか）
    data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data = data.dropna()

    return data


@pytest.fixture
def test_config(temp_dir):
    """テスト用設定"""
    return {
        'feature_engineering': {
            'include_technical': True,
            'include_statistical': True,
            'include_temporal': True,
            'feature_selection_k': 10
        },
        'model_optimization': {
            'enable_ensemble': True,
            'enable_hyperparameter_tuning': False,  # テスト高速化のため無効
            'models_to_optimize': ['random_forest']
        },
        'validation': {
            'strategy': 'time_series_split',
            'n_splits': 3,  # テスト高速化
            'test_size': 0.2
        },
        'overfitting_prevention': {
            'enable_detection': True,
            'enable_regularization': True,
            'overfitting_threshold': 0.1
        },
        'optimization': {
            'n_trials': 10,  # テスト高速化
            'timeout_seconds': 60
        }
    }


class TestAdvancedFeatureEngineer:
    """高度特徴量エンジニアリングテスト"""

    def test_initialization(self):
        """初期化テスト"""
        engineer = AdvancedFeatureEngineer()
        assert engineer.include_technical == True
        assert engineer.include_statistical == True
        assert engineer.include_temporal == True
        assert engineer.feature_names_ == []

    def test_feature_engineering(self, sample_stock_data):
        """特徴量エンジニアリングテスト"""
        engineer = AdvancedFeatureEngineer()
        original_data = sample_stock_data.drop(columns=['target'])

        # 特徴量エンジニアリング実行
        engineered_data = engineer.fit_transform(original_data)

        # 特徴量数が増加していることを確認
        assert len(engineered_data.columns) > len(original_data.columns)

        # NaN値がないことを確認
        assert not engineered_data.isnull().any().any()

        # 特徴量名が記録されていることを確認
        assert len(engineer.feature_names_) == len(engineered_data.columns)

    def test_technical_indicators(self, sample_stock_data):
        """テクニカル指標テスト"""
        engineer = AdvancedFeatureEngineer(include_technical=True, include_statistical=False, include_temporal=False)
        original_data = sample_stock_data.drop(columns=['target'])

        engineered_data = engineer.fit_transform(original_data)

        # テクニカル指標の存在確認
        expected_columns = ['SMA_5', 'SMA_10', 'EMA_5', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
        for col in expected_columns:
            assert col in engineered_data.columns, f"Missing technical indicator: {col}"

    def test_statistical_features(self, sample_stock_data):
        """統計的特徴量テスト"""
        engineer = AdvancedFeatureEngineer(include_technical=False, include_statistical=True, include_temporal=False)
        original_data = sample_stock_data.drop(columns=['target'])

        engineered_data = engineer.fit_transform(original_data)

        # 統計的特徴量の存在確認
        expected_columns = ['return_1d', 'return_5d', 'rolling_mean_5', 'rolling_std_5', 'zscore_5']
        for col in expected_columns:
            assert col in engineered_data.columns, f"Missing statistical feature: {col}"

    def test_temporal_features(self, sample_stock_data):
        """時系列特徴量テスト"""
        engineer = AdvancedFeatureEngineer(include_technical=False, include_statistical=False, include_temporal=True)
        original_data = sample_stock_data.drop(columns=['target'])

        engineered_data = engineer.fit_transform(original_data)

        # 時系列特徴量の存在確認
        expected_columns = ['day_of_week', 'month', 'close_lag_1', 'close_lag_2']
        for col in expected_columns:
            assert col in engineered_data.columns, f"Missing temporal feature: {col}"


class TestEnsembleModelOptimizer:
    """アンサンブルモデル最適化テスト"""

    def test_initialization(self):
        """初期化テスト"""
        optimizer = EnsembleModelOptimizer()
        assert hasattr(optimizer, 'logger')
        assert optimizer.base_models == {}
        assert optimizer.ensemble_weights == {}

    def test_create_base_models(self):
        """ベースモデル作成テスト"""
        optimizer = EnsembleModelOptimizer()

        # 分類モデル
        classification_models = optimizer.create_base_models('classification')
        assert 'random_forest' in classification_models
        assert 'logistic_regression' in classification_models
        assert 'gradient_boosting' in classification_models

        # 回帰モデル
        regression_models = optimizer.create_base_models('regression')
        assert 'random_forest' in regression_models
        assert 'linear_regression' in regression_models
        assert 'ridge' in regression_models

    def test_ensemble_optimization(self, sample_stock_data):
        """アンサンブル最適化テスト"""
        optimizer = EnsembleModelOptimizer()

        # データ準備
        X = sample_stock_data.drop(columns=['target']).iloc[:100]  # テスト高速化のため少量データ
        y = sample_stock_data['target'].iloc[:100]

        # アンサンブル最適化実行
        result = optimizer.optimize_ensemble(X, y, task_type='classification', cv_folds=3)

        # 結果検証
        assert isinstance(result, EnsembleResult)
        assert result.ensemble_type == "Weighted Voting"
        assert len(result.component_models) > 0
        assert isinstance(result.ensemble_score, (int, float))
        assert isinstance(result.individual_scores, dict)
        assert isinstance(result.weight_distribution, dict)


class TestHyperparameterOptimizer:
    """ハイパーパラメータ最適化テスト"""

    def test_initialization(self):
        """初期化テスト"""
        optimizer = HyperparameterOptimizer(n_trials=10)
        assert optimizer.n_trials == 10
        assert hasattr(optimizer, 'logger')

    def test_model_optimization(self, sample_stock_data):
        """モデル最適化テスト"""
        optimizer = HyperparameterOptimizer(n_trials=5)  # テスト高速化

        # データ準備
        X = sample_stock_data.drop(columns=['target']).iloc[:100]
        y = sample_stock_data['target'].iloc[:100]

        # 最適化実行
        result = optimizer.optimize_model('random_forest', X, y, task_type='classification')

        # 結果検証
        assert isinstance(result, ModelOptimizationResult)
        assert result.model_name == 'random_forest'
        assert isinstance(result.best_score, (int, float))
        assert isinstance(result.best_params, dict)
        assert result.optimization_time >= 0
        assert result.trials_count >= 0


class TestValidationStrategyManager:
    """検証戦略管理テスト"""

    def test_initialization(self):
        """初期化テスト"""
        manager = ValidationStrategyManager()
        assert hasattr(manager, 'logger')

    def test_time_series_validation(self, sample_stock_data):
        """時系列分割検証テスト"""
        from sklearn.ensemble import RandomForestClassifier

        manager = ValidationStrategyManager()
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # データ準備
        X = sample_stock_data.drop(columns=['target']).iloc[:100]
        y = sample_stock_data['target'].iloc[:100]

        # 検証実行
        result = manager.robust_validation(model, X, y, ValidationStrategy.TIME_SERIES_SPLIT)

        # 結果検証
        assert 'mean_score' in result
        assert 'std_score' in result
        assert 'scores' in result
        assert 'strategy' in result
        assert result['strategy'] == 'TimeSeriesSplit'
        assert isinstance(result['mean_score'], (int, float))
        assert isinstance(result['std_score'], (int, float))

    def test_walk_forward_validation(self, sample_stock_data):
        """ウォークフォワード検証テスト"""
        from sklearn.ensemble import RandomForestClassifier

        manager = ValidationStrategyManager()
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # データ準備
        X = sample_stock_data.drop(columns=['target']).iloc[:100]
        y = sample_stock_data['target'].iloc[:100]

        # 検証実行
        result = manager.robust_validation(model, X, y, ValidationStrategy.WALK_FORWARD)

        # 結果検証
        assert 'mean_score' in result
        assert 'strategy' in result
        assert result['strategy'] == 'WalkForward'


class TestOverfittingDetector:
    """過学習検知テスト"""

    def test_initialization(self):
        """初期化テスト"""
        detector = OverfittingDetector()
        assert hasattr(detector, 'logger')

    def test_overfitting_detection(self, sample_stock_data):
        """過学習検知テスト"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        detector = OverfittingDetector()
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # データ準備
        X = sample_stock_data.drop(columns=['target']).iloc[:100]
        y = sample_stock_data['target'].iloc[:100]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        # 過学習検知実行
        result = detector.detect_overfitting(model, X_train, y_train, X_val, y_val)

        # 結果検証
        assert 'overfitting_detected' in result
        assert 'train_score' in result
        assert 'validation_score' in result
        assert 'overfitting_gap' in result
        assert 'learning_curve' in result
        assert isinstance(result['overfitting_detected'], bool)

    def test_regularization_application(self):
        """正則化適用テスト"""
        detector = OverfittingDetector()

        # 初期パラメータ
        params = {'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1}

        # 重度の過学習に対する正則化
        regularized_params = detector.apply_regularization(params, overfitting_severity=0.3)

        # パラメータが制限されていることを確認
        assert regularized_params['max_depth'] <= 5
        assert regularized_params['min_samples_split'] >= 10
        assert regularized_params['min_samples_leaf'] >= 5


class TestPredictionAccuracyEnhancer:
    """予測精度向上統合システムテスト"""

    def test_initialization(self, test_config, temp_dir):
        """初期化テスト"""
        config_path = Path(temp_dir) / "test_config.yaml"

        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(test_config, f)

        enhancer = PredictionAccuracyEnhancer(config_path)

        assert enhancer.config is not None
        assert hasattr(enhancer, 'feature_engineer')
        assert hasattr(enhancer, 'ensemble_optimizer')
        assert hasattr(enhancer, 'hyperparameter_optimizer')
        assert hasattr(enhancer, 'validation_manager')
        assert hasattr(enhancer, 'overfitting_detector')

    @pytest.mark.asyncio
    async def test_feature_engineering_workflow(self, sample_stock_data, test_config, temp_dir):
        """特徴量エンジニアリングワークフローテスト"""
        config_path = Path(temp_dir) / "test_config.yaml"

        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(test_config, f)

        enhancer = PredictionAccuracyEnhancer(config_path)

        # 特徴量エンジニアリング実行
        result = await enhancer._perform_feature_engineering(sample_stock_data, 'target')

        # 結果検証
        assert 'engineered_data' in result
        assert 'summary' in result
        assert isinstance(result['summary'], FeatureEngineeringResult)

        summary = result['summary']
        assert summary.original_features > 0
        assert summary.engineered_features >= summary.original_features
        assert summary.selected_features > 0
        assert len(summary.engineering_methods) > 0

    @pytest.mark.asyncio
    async def test_model_optimization_workflow(self, sample_stock_data, test_config, temp_dir):
        """モデル最適化ワークフローテスト"""
        from sklearn.model_selection import train_test_split

        config_path = Path(temp_dir) / "test_config.yaml"

        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(test_config, f)

        enhancer = PredictionAccuracyEnhancer(config_path)

        # データ準備
        X = sample_stock_data.drop(columns=['target']).iloc[:100]
        y = sample_stock_data['target'].iloc[:100]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # モデル最適化実行
        result = await enhancer._optimize_models(X_train, y_train, X_test, y_test)

        # 結果検証
        assert 'best_model' in result
        assert 'summary' in result
        assert 'all_results' in result
        assert result['best_model'] is not None
        assert isinstance(result['summary'], ModelOptimizationResult)

    @pytest.mark.asyncio
    async def test_full_enhancement_workflow(self, sample_stock_data, test_config, temp_dir):
        """完全な精度向上ワークフローテスト"""
        config_path = Path(temp_dir) / "test_config.yaml"

        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(test_config, f)

        enhancer = PredictionAccuracyEnhancer(config_path)

        # 少量データでテスト（高速化のため）
        test_data = sample_stock_data.iloc[:100].copy()

        # 精度向上実行
        report = await enhancer.enhance_prediction_accuracy('TEST', test_data)

        # レポート検証
        assert isinstance(report, AccuracyImprovementReport)
        assert report.symbol == 'TEST'
        assert isinstance(report.timestamp, datetime)
        assert isinstance(report.baseline_accuracy, (int, float))
        assert isinstance(report.improved_accuracy, (int, float))
        assert isinstance(report.improvement_percentage, (int, float))
        assert len(report.improvement_methods) > 0
        assert isinstance(report.feature_engineering, FeatureEngineeringResult)
        assert isinstance(report.model_optimization, ModelOptimizationResult)
        assert isinstance(report.ensemble_result, EnsembleResult)
        assert isinstance(report.validation_scores, dict)
        assert isinstance(report.overfitting_metrics, dict)
        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) > 0

    def test_default_config_loading(self, temp_dir):
        """デフォルト設定読み込みテスト"""
        # 存在しない設定ファイルパス
        non_existent_config = Path(temp_dir) / "non_existent.yaml"

        # デフォルト設定が使用されることを確認
        enhancer = PredictionAccuracyEnhancer(non_existent_config)

        assert enhancer.config is not None
        assert 'feature_engineering' in enhancer.config
        assert 'model_optimization' in enhancer.config
        assert 'validation' in enhancer.config

    def test_recommendation_generation(self, test_config, temp_dir):
        """推奨事項生成テスト"""
        config_path = Path(temp_dir) / "test_config.yaml"

        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(test_config, f)

        enhancer = PredictionAccuracyEnhancer(config_path)

        # 優秀な改善のケース
        recommendations = enhancer._generate_recommendations(
            improvement_percentage=15.0,
            overfitting_metrics={'overfitting_detected': False},
            validation_scores={'std_score': 0.05}
        )
        assert len(recommendations) > 0
        assert any("優秀な精度向上" in rec for rec in recommendations)

        # 過学習検出のケース
        recommendations = enhancer._generate_recommendations(
            improvement_percentage=5.0,
            overfitting_metrics={'overfitting_detected': True},
            validation_scores={'std_score': 0.05}
        )
        assert any("過学習が検出" in rec for rec in recommendations)


@pytest.mark.asyncio
async def test_integration_with_external_systems():
    """外部システム統合テスト"""
    # データ品質監視システムが利用できない環境でのテスト
    with patch('prediction_accuracy_enhancement.DATA_QUALITY_AVAILABLE', False):
        enhancer = PredictionAccuracyEnhancer()
        assert enhancer.data_quality_monitor is None

        # 性能向上処理が正常に動作することを確認（外部システムなしでも）
        # 実際のテストは他のテストで実行されるため、ここでは初期化のみ確認


def test_accuracy_improvement_types():
    """精度向上タイプのテスト"""
    # 全ての精度向上タイプが定義されていることを確認
    expected_types = [
        'DATA_QUALITY', 'FEATURE_ENGINEERING', 'MODEL_OPTIMIZATION',
        'ENSEMBLE_METHODS', 'HYPERPARAMETER_TUNING', 'VALIDATION_STRATEGY',
        'OVERFITTING_PREVENTION', 'CONCEPT_DRIFT_HANDLING'
    ]

    for type_name in expected_types:
        assert hasattr(AccuracyImprovementType, type_name)


def test_validation_strategies():
    """検証戦略のテスト"""
    # 全ての検証戦略が定義されていることを確認
    expected_strategies = ['TIME_SERIES_SPLIT', 'WALK_FORWARD', 'BLOCKED_CV', 'PURGED_CV']

    for strategy_name in expected_strategies:
        assert hasattr(ValidationStrategy, strategy_name)


if __name__ == "__main__":
    # pytest実行
    import os
    os.system("pytest " + __file__ + " -v")