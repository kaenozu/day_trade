#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for Prediction Accuracy Enhancer
予測精度向上システムのテストケース

Issue #885対応：包括的な予測精度向上アプローチのテスト
"""

import pytest
import asyncio
import tempfile
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

# テスト対象のインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))

from prediction_accuracy_enhancer import (
    PredictionAccuracyEnhancer,
    DataQualityAnalyzer,
    AdvancedFeatureSelector,
    ConceptDriftDetector,
    DataQualityLevel,
    EnhancementConfiguration,
    create_prediction_accuracy_enhancer
)

class TestDataQualityAnalyzer:
    """DataQualityAnalyzerのテストクラス"""

    @pytest.fixture
    def config(self):
        """テスト用設定の作成"""
        return EnhancementConfiguration(
            min_data_quality_threshold=0.7,
            data_quality_check_enabled=True
        )

    @pytest.fixture
    def analyzer(self, config):
        """DataQualityAnalyzerインスタンス"""
        return DataQualityAnalyzer(config)

    @pytest.fixture
    def sample_data(self):
        """サンプルデータの作成"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        data = pd.DataFrame({
            'timestamp': dates,
            'price': np.random.normal(100, 10, 100),
            'volume': np.random.randint(1000, 10000, 100),
            'rsi': np.random.uniform(20, 80, 100),
            'target': np.random.randint(0, 2, 100)  # 0 or 1
        })

        # 一部の欠損値を作成
        data.loc[5:10, 'volume'] = np.nan

        return data

    @pytest.mark.asyncio
    async def test_analyze_data_quality_good_data(self, analyzer, sample_data):
        """良質なデータの品質分析テスト"""
        quality_metrics = await analyzer.analyze_data_quality(sample_data, 'target')

        assert quality_metrics.completeness_score > 0.8  # 欠損が少ない
        assert quality_metrics.overall_quality in [DataQualityLevel.GOOD, DataQualityLevel.EXCELLENT]
        assert isinstance(quality_metrics.issues, list)
        assert isinstance(quality_metrics.recommendations, list)

    @pytest.mark.asyncio
    async def test_analyze_data_quality_poor_data(self, analyzer):
        """低品質データの品質分析テスト"""
        # 低品質データの作成
        poor_data = pd.DataFrame({
            'price': [np.nan] * 50 + [100] * 50,  # 50%が欠損
            'volume': [-100, 0, 1e10] * 33 + [1000],  # 異常値含む
            'target': [1] * 100
        })

        quality_metrics = await analyzer.analyze_data_quality(poor_data, 'target')

        assert quality_metrics.completeness_score < 0.8
        assert quality_metrics.overall_quality in [DataQualityLevel.POOR, DataQualityLevel.FAIR]
        assert len(quality_metrics.issues) > 0

    def test_completeness_calculation(self, analyzer, sample_data):
        """完全性スコア計算テスト"""
        completeness = analyzer._calculate_completeness(sample_data)

        # sample_dataは5行の欠損があるので、95%程度の完全性
        expected_completeness = 1.0 - (6 / sample_data.size)  # 6個の欠損値
        assert abs(completeness - expected_completeness) < 0.01

    def test_consistency_calculation(self, analyzer, sample_data):
        """一貫性スコア計算テスト"""
        consistency = analyzer._calculate_consistency(sample_data)

        # 正常なデータなので高い一貫性スコア
        assert consistency > 0.5

    def test_quality_level_determination(self, analyzer):
        """品質レベル決定テスト"""
        assert analyzer._determine_quality_level(0.95) == DataQualityLevel.EXCELLENT
        assert analyzer._determine_quality_level(0.8) == DataQualityLevel.GOOD
        assert analyzer._determine_quality_level(0.6) == DataQualityLevel.FAIR
        assert analyzer._determine_quality_level(0.3) == DataQualityLevel.POOR


class TestAdvancedFeatureSelector:
    """AdvancedFeatureSelectorのテストクラス"""

    @pytest.fixture
    def config(self):
        """テスト用設定の作成"""
        return EnhancementConfiguration(
            max_features=50,
            feature_selection_enabled=True,
            correlation_threshold=0.95
        )

    @pytest.fixture
    def selector(self, config):
        """AdvancedFeatureSelectorインスタンス"""
        return AdvancedFeatureSelector(config)

    @pytest.fixture
    def sample_features(self):
        """サンプル特徴量データの作成"""
        np.random.seed(42)
        n_samples = 200

        # 相関のある特徴量を含むデータ
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
        })

        # feature4はfeature1と高い相関
        X['feature4'] = X['feature1'] + np.random.normal(0, 0.1, n_samples)

        # feature5はfeature1と完全相関
        X['feature5'] = X['feature1'] * 1.01

        # 追加の独立特徴量
        for i in range(6, 21):
            X[f'feature{i}'] = np.random.normal(0, 1, n_samples)

        # ターゲット変数（feature1と相関）
        y = pd.Series((X['feature1'] + np.random.normal(0, 0.5, n_samples) > 0).astype(int))

        return X, y

    def test_remove_correlated_features(self, selector, sample_features):
        """高相関特徴量除去テスト"""
        X, y = sample_features

        X_filtered = selector._remove_correlated_features(X)

        # 高相関特徴量が除去されているか確認
        assert len(X_filtered.columns) < len(X.columns)
        assert 'feature5' not in X_filtered.columns  # 完全相関のため除去される

    @pytest.mark.asyncio
    async def test_select_features(self, selector, sample_features):
        """特徴量選択の統合テスト"""
        X, y = sample_features

        X_selected, importance_metrics = await selector.select_features(X, y, 'classification')

        assert len(X_selected.columns) <= selector.config.max_features
        assert len(X_selected.columns) > 0
        assert len(importance_metrics) > 0

        # 重要度メトリクスの構造確認
        if importance_metrics:
            first_metric = importance_metrics[0]
            assert hasattr(first_metric, 'feature_name')
            assert hasattr(first_metric, 'importance_score')
            assert first_metric.feature_name in X_selected.columns

    def test_statistical_feature_selection(self, selector, sample_features):
        """統計的特徴量選択テスト"""
        X, y = sample_features

        X_statistical = selector._statistical_feature_selection(X, y, 'classification')

        assert len(X_statistical.columns) <= len(X.columns)
        assert len(X_statistical.columns) > 0


class TestConceptDriftDetector:
    """ConceptDriftDetectorのテストクラス"""

    @pytest.fixture
    def config(self):
        """テスト用設定の作成"""
        return EnhancementConfiguration(
            drift_detection_enabled=True,
            drift_detection_window=50,
            drift_threshold=0.05
        )

    @pytest.fixture
    def detector(self, config):
        """ConceptDriftDetectorインスタンス"""
        return ConceptDriftDetector(config)

    @pytest.fixture
    def reference_data(self):
        """参照データの作成"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(100, 10, 100),
            'feature2': np.random.normal(50, 5, 100),
            'feature3': np.random.uniform(0, 1, 100)
        })

    @pytest.fixture
    def stable_data(self, reference_data):
        """安定したデータ（ドリフトなし）"""
        np.random.seed(43)
        return pd.DataFrame({
            'feature1': np.random.normal(100, 10, 100),
            'feature2': np.random.normal(50, 5, 100),
            'feature3': np.random.uniform(0, 1, 100)
        })

    @pytest.fixture
    def drifted_data(self, reference_data):
        """ドリフトしたデータ"""
        np.random.seed(44)
        return pd.DataFrame({
            'feature1': np.random.normal(120, 15, 100),  # 平均と分散が変化
            'feature2': np.random.normal(30, 8, 100),    # 大きく変化
            'feature3': np.random.uniform(0.2, 0.8, 100) # 分布が変化
        })

    @pytest.mark.asyncio
    async def test_detect_drift_stable_data(self, detector, reference_data, stable_data):
        """安定データでのドリフト検知テスト"""
        # 参照データを設定
        detector.reference_data = reference_data

        result = await detector.detect_drift(stable_data)

        assert 'drift_detected' in result
        assert 'drift_score' in result
        assert 'analysis' in result
        assert result['drift_detected'] == False  # ドリフトは検知されないはず
        assert result['drift_score'] < detector.config.drift_threshold

    @pytest.mark.asyncio
    async def test_detect_drift_drifted_data(self, detector, reference_data, drifted_data):
        """ドリフトデータでのドリフト検知テスト"""
        # 参照データを設定
        detector.reference_data = reference_data

        result = await detector.detect_drift(drifted_data)

        assert 'drift_detected' in result
        assert 'drift_score' in result
        assert result['drift_score'] > 0  # 何らかのドリフトスコア
        # 大きな変化なのでドリフトが検知される可能性が高い

    @pytest.mark.asyncio
    async def test_detect_drift_no_reference(self, detector, stable_data):
        """参照データなしでのドリフト検知テスト"""
        result = await detector.detect_drift(stable_data)

        assert result['drift_detected'] == False
        assert result['drift_score'] == 0.0
        assert "参照データなし" in result['analysis']

    def test_statistical_drift_detection(self, detector, reference_data, drifted_data):
        """統計的ドリフト検知テスト"""
        detector.reference_data = reference_data

        drift_score = detector._statistical_drift_detection(drifted_data)

        assert isinstance(drift_score, float)
        assert drift_score >= 0  # ドリフトスコアは非負

    def test_distribution_drift_detection(self, detector, reference_data, drifted_data):
        """分布ドリフト検知テスト"""
        detector.reference_data = reference_data

        drift_score = detector._distribution_drift_detection(drifted_data)

        assert isinstance(drift_score, float)
        assert drift_score >= 0  # ドリフトスコアは非負


class TestPredictionAccuracyEnhancer:
    """PredictionAccuracyEnhancerのテストクラス"""

    @pytest.fixture
    def temp_config_file(self):
        """一時設定ファイルの作成"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'min_data_quality_threshold': 0.6,
                'data_quality_check_enabled': True,
                'max_features': 20,
                'feature_selection_enabled': True,
                'ensemble_enabled': True,
                'validation_strategy': 'time_series_cv',
                'cv_folds': 3
            }
            yaml.dump(config, f)
            yield Path(f.name)
            # クリーンアップ
            Path(f.name).unlink()

    @pytest.fixture
    def enhancer(self, temp_config_file):
        """PredictionAccuracyEnhancerインスタンス"""
        return PredictionAccuracyEnhancer(config_path=temp_config_file)

    @pytest.fixture
    def sample_training_data(self):
        """サンプル訓練データの作成"""
        np.random.seed(42)
        n_samples = 200

        data = pd.DataFrame({
            'price': np.random.normal(100, 10, n_samples),
            'volume': np.random.randint(1000, 10000, n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
            'macd': np.random.normal(0, 2, n_samples),
            'bb_upper': np.random.normal(110, 5, n_samples),
            'bb_lower': np.random.normal(90, 5, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })

        return data

    def test_initialization(self, enhancer):
        """初期化テスト"""
        assert enhancer.data_quality_analyzer is not None
        assert enhancer.feature_selector is not None
        assert enhancer.drift_detector is not None
        assert isinstance(enhancer.enhancement_history, list)

    def test_configuration_loading(self, temp_config_file):
        """設定読み込みテスト"""
        enhancer = PredictionAccuracyEnhancer(config_path=temp_config_file)

        assert enhancer.config.min_data_quality_threshold == 0.6
        assert enhancer.config.max_features == 20
        assert enhancer.config.ensemble_enabled == True

    @pytest.mark.asyncio
    async def test_enhance_prediction_accuracy_basic(self, enhancer, sample_training_data):
        """基本的な予測精度向上テスト"""
        result = await enhancer.enhance_prediction_accuracy(
            symbol='TEST',
            training_data=sample_training_data,
            target_column='target',
            task_type='classification'
        )

        assert 'symbol' in result
        assert result['symbol'] == 'TEST'
        assert 'timestamp' in result
        assert 'steps_completed' in result
        assert isinstance(result['steps_completed'], list)
        assert len(result['steps_completed']) > 0

        # データ品質分析が実行されているか確認
        assert 'data_quality_analysis' in result['steps_completed']
        assert 'data_quality' in result

    @pytest.mark.asyncio
    async def test_enhance_prediction_accuracy_with_error(self, enhancer):
        """エラー発生時の予測精度向上テスト"""
        # 不正なデータでテスト
        bad_data = pd.DataFrame({'bad_column': [1, 2, 3]})

        result = await enhancer.enhance_prediction_accuracy(
            symbol='ERROR_TEST',
            training_data=bad_data,
            target_column='nonexistent_target',
            task_type='classification'
        )

        assert 'error' in result
        assert result['symbol'] == 'ERROR_TEST'
        assert 'recommendations' in result

    def test_get_enhancement_summary_empty(self, enhancer):
        """空の履歴でのサマリー取得テスト"""
        summary = enhancer.get_enhancement_summary()

        assert 'status' in summary
        assert summary['status'] == "No enhancement history available"

    @pytest.mark.asyncio
    async def test_get_enhancement_summary_with_history(self, enhancer, sample_training_data):
        """履歴ありでのサマリー取得テスト"""
        # 予測精度向上を実行して履歴を作成
        await enhancer.enhance_prediction_accuracy(
            symbol='SUMMARY_TEST',
            training_data=sample_training_data,
            target_column='target',
            task_type='classification'
        )

        summary = enhancer.get_enhancement_summary()

        assert 'total_enhancements' in summary
        assert summary['total_enhancements'] == 1
        assert 'recent_enhancements' in summary
        assert 'system_integrations' in summary

    def test_create_prediction_accuracy_enhancer_function(self, temp_config_file):
        """ファクトリー関数のテスト"""
        enhancer = create_prediction_accuracy_enhancer(str(temp_config_file))

        assert isinstance(enhancer, PredictionAccuracyEnhancer)
        assert enhancer.config.max_features == 20


# インテグレーションテスト
class TestIntegration:
    """統合テスト"""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """エンドツーエンドワークフローテスト"""
        # 一時設定ファイル作成
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "integration_config.yaml"
            config = {
                'min_data_quality_threshold': 0.5,
                'data_quality_check_enabled': True,
                'max_features': 15,
                'feature_selection_enabled': True,
                'ensemble_enabled': False,  # 高速化のため無効
                'validation_strategy': 'time_series_cv',
                'cv_folds': 2,  # 高速化のため少なく
                'drift_detection_enabled': True
            }

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f)

            # エンハンサー作成
            enhancer = PredictionAccuracyEnhancer(config_path=config_path)

            # テストデータ作成
            np.random.seed(42)
            training_data = pd.DataFrame({
                'price': np.random.normal(100, 10, 150),
                'volume': np.random.randint(1000, 10000, 150),
                'rsi': np.random.uniform(20, 80, 150),
                'sma_5': np.random.normal(100, 8, 150),
                'sma_20': np.random.normal(100, 6, 150),
                'volatility': np.random.uniform(0.1, 0.5, 150),
                'target': np.random.randint(0, 2, 150)
            })

            # 予測精度向上実行
            result = await enhancer.enhance_prediction_accuracy(
                symbol='INTEGRATION_TEST',
                training_data=training_data,
                target_column='target',
                task_type='classification'
            )

            # 結果の検証
            assert result['symbol'] == 'INTEGRATION_TEST'
            assert len(result['steps_completed']) >= 3
            assert 'data_quality_analysis' in result['steps_completed']
            assert 'feature_selection' in result['steps_completed']
            assert 'robust_validation' in result['steps_completed']

            # データ品質結果の確認
            assert 'data_quality' in result
            quality = result['data_quality']
            assert hasattr(quality, 'overall_quality')

            # 特徴量選択結果の確認
            assert 'selected_features' in result
            assert len(result['selected_features']) > 0

            # 推奨事項の確認
            assert 'recommendations' in result
            assert isinstance(result['recommendations'], list)

            # サマリーの確認
            summary = enhancer.get_enhancement_summary()
            assert summary['total_enhancements'] == 1


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])