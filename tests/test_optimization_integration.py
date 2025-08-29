#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合最適化システムテスト
Integrated Optimization System Tests
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.day_trade.optimization.integrated_optimization_system import (
    IntegratedOptimizationSystem,
    OptimizationReport,
    SystemHealth
)
from src.day_trade.optimization.prediction_accuracy_enhancer import PredictionAccuracyEnhancer
from src.day_trade.optimization.performance_optimization_engine import PerformanceOptimizationEngine
from src.day_trade.optimization.model_accuracy_improver import ModelAccuracyImprover
from src.day_trade.optimization.response_speed_optimizer import ResponseSpeedOptimizer
from src.day_trade.optimization.memory_efficiency_optimizer import MemoryEfficiencyOptimizer


class TestIntegratedOptimizationSystem:
    """統合最適化システムテストクラス"""
    
    @pytest.fixture
    async def optimization_system(self):
        """最適化システムのフィクスチャ"""
        system = IntegratedOptimizationSystem()
        
        # モックサブシステム
        system.prediction_enhancer = Mock(spec=PredictionAccuracyEnhancer)
        system.performance_engine = Mock(spec=PerformanceOptimizationEngine)
        system.model_improver = Mock(spec=ModelAccuracyImprover)
        system.speed_optimizer = Mock(spec=ResponseSpeedOptimizer)
        system.memory_optimizer = Mock(spec=MemoryEfficiencyOptimizer)
        
        # 非同期メソッドのモック
        system.prediction_enhancer.initialize = AsyncMock()
        system.performance_engine.initialize = AsyncMock()
        system.model_improver.initialize = AsyncMock()
        system.speed_optimizer.initialize = AsyncMock()
        
        await system.initialize()
        return system
        
    @pytest.mark.asyncio
    async def test_system_initialization(self, optimization_system):
        """システム初期化テスト"""
        system = optimization_system
        
        # 初期化が呼び出されたことを確認
        system.prediction_enhancer.initialize.assert_called_once()
        system.performance_engine.initialize.assert_called_once()
        system.model_improver.initialize.assert_called_once()
        system.speed_optimizer.initialize.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_comprehensive_optimization(self, optimization_system):
        """包括的最適化テスト"""
        system = optimization_system
        
        # モックの設定
        system.prediction_enhancer.optimize_feature_engineering = AsyncMock(return_value={"accuracy": 0.85})
        system.prediction_enhancer.optimize_model_selection = AsyncMock(return_value={"best_model": "RandomForest"})
        system.prediction_enhancer.optimize_ensemble_methods = AsyncMock(return_value={"ensemble_score": 0.90})
        
        system.performance_engine.optimize_cpu_performance = AsyncMock(return_value={"cpu_efficiency": 0.8})
        system.performance_engine.optimize_memory_usage = AsyncMock(return_value={"memory_saved": 200})
        system.performance_engine.optimize_io_performance = AsyncMock(return_value={"io_speed": 1.5})
        
        system.model_improver.optimize_hyperparameters = AsyncMock(return_value={"best_params": {}})
        system.model_improver.optimize_ensemble = AsyncMock(return_value={"ensemble_accuracy": 0.92})
        system.model_improver.optimize_feature_selection = AsyncMock(return_value={"selected_features": 50})
        
        system.speed_optimizer.get_performance_metrics = AsyncMock(return_value={"cache_hit_rate": 0.7})
        
        system.memory_optimizer.analyze_memory_usage = Mock(return_value={"total_memory": 1000})
        system.memory_optimizer.detect_memory_leaks = Mock(return_value=[])
        system.memory_optimizer.optimize_memory_usage = Mock(return_value={"optimized": True})
        
        # 最適化実行
        report = await system.run_comprehensive_optimization()
        
        # 結果検証
        assert isinstance(report, OptimizationReport)
        assert report.prediction_accuracy["status"] == "success"
        assert report.performance_metrics["status"] == "success"
        assert report.model_improvements["status"] == "success"
        assert report.response_speed["status"] == "success"
        assert report.memory_efficiency["status"] == "success"
        assert report.overall_score > 0
        assert len(report.recommendations) > 0
        
    @pytest.mark.asyncio
    async def test_service_lifecycle(self, optimization_system):
        """サービスライフサイクルテスト"""
        system = optimization_system
        
        # サービス開始
        await system.start_optimization_service()
        assert system.is_running is True
        
        # 少し待機
        await asyncio.sleep(0.1)
        
        # サービス停止
        await system.stop_optimization_service()
        assert system.is_running is False
        
    @pytest.mark.asyncio
    async def test_health_monitoring(self, optimization_system):
        """ヘルスモニタリングテスト"""
        system = optimization_system
        
        # スピードオプティマイザーのモック設定
        system.speed_optimizer.get_performance_metrics = AsyncMock(return_value={
            "avg_response_time": 0.5,
            "cache_hit_rate": 0.8
        })
        
        # メモリオプティマイザーのモック設定
        system.memory_optimizer.get_memory_metrics = Mock(return_value={
            "current_memory_mb": 500,
            "potential_leaks": []
        })
        
        with patch('psutil.cpu_percent', return_value=20.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # psutilのモック設定
            mock_memory.return_value.percent = 40.0
            mock_disk.return_value.percent = 60.0
            
            health = await system.check_system_health()
            
            assert isinstance(health, SystemHealth)
            assert health.cpu_usage == 20.0
            assert health.memory_usage == 40.0
            assert health.disk_usage == 60.0
            assert health.overall_health in ["excellent", "good", "fair", "poor"]
            
    def test_overall_score_calculation(self, optimization_system):
        """総合スコア計算テスト"""
        system = optimization_system
        
        # テスト用レポート作成
        report = OptimizationReport(timestamp=datetime.now())
        report.prediction_accuracy = {"status": "success"}
        report.performance_metrics = {"status": "success"}
        report.model_improvements = {"status": "success"}
        report.response_speed = {"status": "success", "performance_metrics": {"cache_hit_rate": 0.8}}
        report.memory_efficiency = {"status": "success", "memory_leaks": []}
        
        score = system._calculate_overall_score(report)
        
        assert 0.0 <= score <= 100.0
        assert score > 70.0  # 全て成功の場合は70点以上
        
    def test_recommendations_generation(self, optimization_system):
        """推奨事項生成テスト"""
        system = optimization_system
        
        # 失敗ケースのレポート
        report = OptimizationReport(timestamp=datetime.now())
        report.prediction_accuracy = {"status": "failed"}
        report.performance_metrics = {"status": "failed"}
        report.response_speed = {"performance_metrics": {"cache_hit_rate": 0.3}}
        report.memory_efficiency = {"memory_leaks": ["leak1"]}
        report.overall_score = 60.0
        
        recommendations = system._generate_recommendations(report)
        
        assert len(recommendations) > 0
        assert any("予測精度" in rec for rec in recommendations)
        assert any("パフォーマンス" in rec for rec in recommendations)
        assert any("キャッシュ" in rec for rec in recommendations)
        assert any("メモリリーク" in rec for rec in recommendations)


class TestPredictionAccuracyEnhancer:
    """予測精度向上システムテストクラス"""
    
    @pytest.fixture
    def enhancer(self):
        """エンハンサーのフィクスチャ"""
        return PredictionAccuracyEnhancer()
        
    def test_feature_importance_analysis(self, enhancer):
        """特徴量重要度分析テスト"""
        # テストデータ作成
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(10)]
        
        importance = enhancer.analyze_feature_importance(X, y, feature_names)
        
        assert len(importance) == 10
        assert all(isinstance(score, (int, float)) for score in importance.values())
        
    def test_correlation_analysis(self, enhancer):
        """相関分析テスト"""
        # テストデータフレーム作成
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        correlations = enhancer.analyze_feature_correlations(df, 'target')
        
        assert isinstance(correlations, dict)
        assert 'feature1' in correlations
        assert 'feature2' in correlations


class TestPerformanceOptimizationEngine:
    """パフォーマンス最適化エンジンテストクラス"""
    
    @pytest.fixture
    def engine(self):
        """エンジンのフィクスチャ"""
        return PerformanceOptimizationEngine()
        
    def test_cpu_usage_monitoring(self, engine):
        """CPU使用量監視テスト"""
        with patch('psutil.cpu_percent', return_value=25.0):
            cpu_usage = engine.monitor_cpu_usage()
            assert cpu_usage == 25.0
            
    def test_memory_usage_monitoring(self, engine):
        """メモリ使用量監視テスト"""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 60.0
            mock_memory.return_value.used = 1000000000
            mock_memory.return_value.total = 2000000000
            
            memory_info = engine.monitor_memory_usage()
            assert memory_info['usage_percent'] == 60.0
            assert memory_info['used_mb'] == 1000000000 / 1024 / 1024


class TestModelAccuracyImprover:
    """モデル精度改善システムテストクラス"""
    
    @pytest.fixture
    def improver(self):
        """改善システムのフィクスチャ"""
        return ModelAccuracyImprover()
        
    def test_hyperparameter_search_space_generation(self, improver):
        """ハイパーパラメータ探索空間生成テスト"""
        param_space = improver.generate_hyperparameter_space('RandomForest')
        
        assert isinstance(param_space, dict)
        assert 'n_estimators' in param_space
        assert 'max_depth' in param_space
        
    def test_ensemble_model_creation(self, improver):
        """アンサンブルモデル作成テスト"""
        # テストデータ
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 5)
        
        ensemble = improver.create_voting_ensemble()
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == 20
        assert all(pred in [0, 1] for pred in predictions)


class TestResponseSpeedOptimizer:
    """レスポンス速度最適化システムテストクラス"""
    
    @pytest.fixture
    async def optimizer(self):
        """オプティマイザーのフィクスチャ"""
        optimizer = ResponseSpeedOptimizer()
        await optimizer.initialize()
        return optimizer
        
    @pytest.mark.asyncio
    async def test_cache_functionality(self, optimizer):
        """キャッシュ機能テスト"""
        # キャッシュに値を設定
        await optimizer.cache_manager.set("test_key", "test_value", ttl=60)
        
        # キャッシュから値を取得
        cached_value = await optimizer.cache_manager.get("test_key")
        assert cached_value == "test_value"
        
    @pytest.mark.asyncio
    async def test_compression_functionality(self, optimizer):
        """圧縮機能テスト"""
        test_data = {"large_data": "x" * 2000}  # 2KB のテストデータ
        
        compressed = await optimizer.optimize_api_response(test_data)
        
        # 圧縮されたデータはbytes型
        assert isinstance(compressed, bytes)
        
        # 展開してオリジナルと比較
        decompressed = optimizer.compression.decompress_json(compressed)
        assert decompressed == test_data


class TestMemoryEfficiencyOptimizer:
    """メモリ効率化最適化システムテストクラス"""
    
    @pytest.fixture
    def optimizer(self):
        """オプティマイザーのフィクスチャ"""
        return MemoryEfficiencyOptimizer()
        
    def test_dataframe_memory_optimization(self, optimizer):
        """DataFrameメモリ最適化テスト"""
        # テストデータフレーム作成
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),
            'float_col': np.random.randn(1000),
            'str_col': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = optimizer.optimize_dataframe_memory(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # 最適化後はメモリ使用量が同じまたは少ない
        assert optimized_memory <= original_memory
        
    def test_object_pool_functionality(self, optimizer):
        """オブジェクトプール機能テスト"""
        # リストファクトリでプール作成
        pool = optimizer.get_memory_pool("test_pool", list, max_size=5)
        
        # オブジェクト取得
        obj1 = pool.acquire()
        assert isinstance(obj1, list)
        
        # オブジェクト返却
        pool.release(obj1)
        
        # 再度取得（同じオブジェクトが返されるはず）
        obj2 = pool.acquire()
        assert obj2 is obj1
        
    def test_memory_analysis(self, optimizer):
        """メモリ分析テスト"""
        analysis = optimizer.analyze_memory_usage()
        
        assert isinstance(analysis, dict)
        assert 'system_memory' in analysis
        assert 'process_memory' in analysis
        assert 'gc_stats' in analysis


def run_integration_tests():
    """統合テスト実行"""
    print("Running Optimization System Integration Tests...")
    
    # pytest実行
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])
    
    return exit_code == 0


if __name__ == "__main__":
    success = run_integration_tests()
    if success:
        print("✅ All integration tests passed!")
    else:
        print("❌ Some integration tests failed!")
    sys.exit(0 if success else 1)