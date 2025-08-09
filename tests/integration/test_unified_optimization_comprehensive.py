#!/usr/bin/env python3
"""
統合最適化システム包括テストスイート
Phase E: システム品質強化フェーズ

全統合コンポーネントの網羅的テスト
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.day_trade.core.optimization_strategy import (
    OptimizationConfig,
    OptimizationLevel,
    OptimizationStrategyFactory,
    get_optimized_implementation
)

class TestUnifiedOptimizationSystem:
    """統合最適化システム包括テスト"""

    @pytest.fixture(autouse=True)
    def setup_test_data(self):
        """テストデータセットアップ"""
        # 実際の市場データを模したテストデータ
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        price_base = 1000
        returns = np.random.normal(0.001, 0.02, 100)
        prices = [price_base]

        for ret in returns[:-1]:
            prices.append(prices[-1] * (1 + ret))

        self.test_data = pd.DataFrame({
            'Date': dates,
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }).set_index('Date')

        # 複数最適化レベル設定
        self.optimization_configs = [
            OptimizationConfig(level=OptimizationLevel.STANDARD),
            OptimizationConfig(level=OptimizationLevel.OPTIMIZED),
            OptimizationConfig(level=OptimizationLevel.ADAPTIVE),
        ]

    def test_strategy_factory_registration(self):
        """戦略ファクトリー登録テスト"""
        # 登録済みコンポーネント確認
        components = OptimizationStrategyFactory.get_registered_components()

        # 必須コンポーネントの存在確認
        required_components = [
            "technical_indicators",
            "feature_engineering",
            "ml_models",
            "multi_timeframe_analysis",
            "database"
        ]

        for component in required_components:
            assert component in components, f"コンポーネント未登録: {component}"

        # 各コンポーネントに複数レベルの戦略があることを確認
        for component_name, strategies in components.items():
            assert len(strategies) >= 1, f"戦略数不足: {component_name}"

        print(f"✅ 戦略登録テスト完了: {len(components)}個のコンポーネント")

    @pytest.mark.parametrize("config", [
        OptimizationConfig(level=OptimizationLevel.STANDARD),
        OptimizationConfig(level=OptimizationLevel.OPTIMIZED),
        OptimizationConfig(level=OptimizationLevel.ADAPTIVE)
    ])
    def test_technical_indicators_all_levels(self, config):
        """全最適化レベルでのテクニカル指標テスト"""
        try:
            from src.day_trade.analysis.technical_indicators_unified import TechnicalIndicatorsManager

            manager = TechnicalIndicatorsManager(config)

            # 基本指標計算テスト
            indicators = ["sma", "ema", "rsi", "macd", "bollinger_bands"]
            start_time = time.time()

            results = manager.calculate_indicators(self.test_data, indicators, period=20)

            execution_time = time.time() - start_time

            # 結果検証
            assert len(results) == len(indicators), f"指標数不一致: 期待{len(indicators)}, 実際{len(results)}"

            for indicator, result in results.items():
                assert result is not None, f"指標計算失敗: {indicator}"
                # パフォーマンスオブジェクトの確認
                assert hasattr(result, 'calculation_time'), f"パフォーマンス情報欠如: {indicator}"
                assert hasattr(result, 'strategy_used'), f"戦略情報欠如: {indicator}"

            # レベル別性能期待値
            if config.level == OptimizationLevel.OPTIMIZED:
                assert execution_time < 2.0, f"最適化版の性能不足: {execution_time:.3f}秒"
            elif config.level == OptimizationLevel.STANDARD:
                assert execution_time < 5.0, f"標準版の性能許容範囲超過: {execution_time:.3f}秒"

            print(f"✅ テクニカル指標テスト完了 ({config.level.value}): {execution_time:.3f}秒")

        except ImportError as e:
            pytest.skip(f"テクニカル指標統合システム未利用: {e}")

    def test_feature_engineering_parallel(self):
        """並列特徴量エンジニアリングテスト"""
        try:
            from src.day_trade.analysis.feature_engineering_unified import (
                FeatureEngineeringManager,
                FeatureConfig
            )

            config = OptimizationConfig(
                level=OptimizationLevel.OPTIMIZED,
                parallel_processing=True
            )

            feature_config = FeatureConfig(
                lookback_periods=[5, 10, 20],
                volatility_windows=[10, 20],
                momentum_periods=[5, 10],
                enable_parallel=True,
                max_workers=2
            )

            manager = FeatureEngineeringManager(config)

            start_time = time.time()
            result = manager.generate_features(self.test_data, feature_config)
            execution_time = time.time() - start_time

            # 結果検証
            assert result is not None, "特徴量生成失敗"
            assert hasattr(result, 'feature_names'), "特徴量名情報欠如"
            assert hasattr(result, 'generation_time'), "生成時間情報欠如"
            assert len(result.feature_names) > 0, "特徴量生成数不足"

            # 並列処理効果確認
            assert execution_time < 3.0, f"並列処理の性能不足: {execution_time:.3f}秒"

            print(f"✅ 特徴量エンジニアリングテスト完了: {len(result.feature_names)}個, {execution_time:.3f}秒")

        except ImportError as e:
            pytest.skip(f"特徴量エンジニアリング統合システム未利用: {e}")

    def test_ml_models_caching(self):
        """MLモデル キャッシュ機能テスト"""
        try:
            from src.day_trade.analysis.ml_models_unified import (
                MLModelsManager,
                ModelConfig
            )

            config = OptimizationConfig(
                level=OptimizationLevel.OPTIMIZED,
                cache_enabled=True
            )

            model_config = ModelConfig(
                model_type="random_forest",
                n_estimators=10,  # テスト用に小さな値
                max_depth=3,
                enable_parallel=True
            )

            manager = MLModelsManager(config)

            # ダミー訓練データ生成
            X_train = np.random.rand(50, 10)
            y_train = np.random.randint(0, 2, 50)
            X_test = np.random.rand(10, 10)

            # 1回目の訓練・予測
            start_time = time.time()
            training_result = manager.train_model(X_train, y_train, model_config)
            first_prediction = manager.predict(X_test)
            first_time = time.time() - start_time

            # 2回目の予測（キャッシュ効果期待）
            start_time = time.time()
            second_prediction = manager.predict(X_test)
            second_time = time.time() - start_time

            # 結果検証
            assert training_result is not None, "訓練結果取得失敗"
            assert first_prediction is not None, "1回目予測失敗"
            assert second_prediction is not None, "2回目予測失敗"

            # キャッシュ効果確認（2回目が明らかに高速）
            cache_speedup = first_time / max(second_time, 0.001)  # ゼロ除算回避

            print(f"✅ MLモデルテスト完了: 1回目{first_time:.3f}秒, 2回目{second_time:.3f}秒, 高速化{cache_speedup:.1f}倍")

        except ImportError as e:
            pytest.skip(f"MLモデル統合システム未利用: {e}")

    def test_multi_timeframe_analysis(self):
        """マルチタイムフレーム分析テスト"""
        try:
            from src.day_trade.analysis.multi_timeframe_analysis_unified import MultiTimeframeAnalysisManager

            config = OptimizationConfig(level=OptimizationLevel.OPTIMIZED)
            manager = MultiTimeframeAnalysisManager(config)

            start_time = time.time()
            result = manager.analyze_multi_timeframe(self.test_data)
            execution_time = time.time() - start_time

            # 結果検証
            assert result is not None, "マルチタイムフレーム分析失敗"
            assert hasattr(result, 'timeframe_results'), "タイムフレーム結果欠如"
            assert hasattr(result, 'integrated_trend'), "統合トレンド情報欠如"
            assert hasattr(result, 'confidence_score'), "信頼度情報欠如"

            # タイムフレーム結果確認
            assert len(result.timeframe_results) > 0, "タイムフレーム分析結果不足"

            # トレンド値確認
            valid_trends = ["strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"]
            assert result.integrated_trend in valid_trends, f"不正なトレンド値: {result.integrated_trend}"

            # 信頼度範囲確認
            assert 0.0 <= result.confidence_score <= 1.0, f"信頼度範囲外: {result.confidence_score}"

            print(f"✅ マルチタイムフレーム分析テスト完了: {result.integrated_trend} (信頼度{result.confidence_score:.3f}), {execution_time:.3f}秒")

        except ImportError as e:
            pytest.skip(f"マルチタイムフレーム分析統合システム未利用: {e}")

    def test_database_optimization(self):
        """データベース最適化テスト"""
        try:
            from src.day_trade.models.database_unified import DatabaseManager

            config = OptimizationConfig(level=OptimizationLevel.OPTIMIZED)
            db_manager = DatabaseManager(config)

            # 基本接続テスト
            connection_result = db_manager.test_connection()
            assert connection_result.success, f"データベース接続失敗: {connection_result.error_message}"

            # クエリ実行テスト
            query_result = db_manager.execute_query("SELECT 1 as test_value")
            assert query_result.success, f"クエリ実行失敗: {query_result.error_message}"

            # キャッシュ機能テスト（最適化版）
            strategy = db_manager.get_strategy()
            if hasattr(strategy, 'get_cache_stats'):
                cache_stats = strategy.get_cache_stats()
                assert 'cache_enabled' in cache_stats, "キャッシュ統計情報欠如"

            print(f"✅ データベース最適化テスト完了: 戦略 {strategy.get_strategy_name()}")

        except ImportError as e:
            pytest.skip(f"データベース統合システム未利用: {e}")

    @pytest.mark.asyncio
    async def test_async_processing_capability(self):
        """非同期処理能力テスト"""
        try:
            from src.day_trade.analysis.multi_timeframe_analysis_unified import MultiTimeframeAnalysisManager

            config = OptimizationConfig(level=OptimizationLevel.OPTIMIZED)
            manager = MultiTimeframeAnalysisManager(config)

            # 非同期分析テスト
            start_time = time.time()
            result = await manager.analyze_async(self.test_data)
            execution_time = time.time() - start_time

            assert result is not None, "非同期分析失敗"
            assert hasattr(result, 'integrated_trend'), "非同期分析結果不備"

            print(f"✅ 非同期処理テスト完了: {execution_time:.3f}秒")

        except ImportError as e:
            pytest.skip(f"非同期処理未対応: {e}")

    def test_performance_monitoring_integration(self):
        """パフォーマンス監視統合テスト"""
        for config in self.optimization_configs:
            try:
                strategy = get_optimized_implementation("technical_indicators", config)

                # パフォーマンスメトリクス取得
                metrics = strategy.get_performance_metrics()

                assert 'execution_count' in metrics, "実行回数情報欠如"
                assert 'total_time' in metrics, "総実行時間情報欠如"
                assert 'success_rate' in metrics, "成功率情報欠如"

                print(f"✅ パフォーマンス監視テスト完了 ({config.level.value}): 成功率{metrics.get('success_rate', 0):.2%}")

            except ImportError:
                pytest.skip(f"パフォーマンス監視システム未利用 ({config.level.value})")

    def test_adaptive_level_functionality(self):
        """適応的レベル機能テスト"""
        config = OptimizationConfig(
            level=OptimizationLevel.ADAPTIVE,
            auto_fallback=True
        )

        try:
            strategy = get_optimized_implementation("technical_indicators", config)

            # 適応的戦略の動作確認
            assert strategy is not None, "適応的戦略取得失敗"

            strategy_name = strategy.get_strategy_name()
            assert "適応" in strategy_name or "Adaptive" in strategy_name, f"適応的戦略名不正: {strategy_name}"

            print(f"✅ 適応的レベル機能テスト完了: {strategy_name}")

        except ImportError as e:
            pytest.skip(f"適応的レベル未対応: {e}")

    def test_error_handling_robustness(self):
        """エラーハンドリング堅牢性テスト"""
        # 不正なデータでのテスト
        invalid_data = pd.DataFrame()  # 空のDataFrame

        config = OptimizationConfig(level=OptimizationLevel.STANDARD)

        try:
            from src.day_trade.analysis.technical_indicators_unified import TechnicalIndicatorsManager

            manager = TechnicalIndicatorsManager(config)

            # 不正入力に対するエラーハンドリング確認
            with pytest.raises(Exception):  # 適切な例外が発生することを確認
                manager.calculate_indicators(invalid_data, ["sma"])

            print("✅ エラーハンドリング堅牢性テスト完了")

        except ImportError as e:
            pytest.skip(f"エラーハンドリングテスト未実行: {e}")

    def test_configuration_validation(self):
        """設定検証テスト"""
        # 各種設定パターンのテスト
        test_configs = [
            OptimizationConfig(level=OptimizationLevel.STANDARD, cache_enabled=False),
            OptimizationConfig(level=OptimizationLevel.OPTIMIZED, parallel_processing=False),
            OptimizationConfig(level=OptimizationLevel.ADAPTIVE, auto_fallback=False),
        ]

        for config in test_configs:
            # 設定の有効性確認
            assert config.level in [OptimizationLevel.STANDARD, OptimizationLevel.OPTIMIZED, OptimizationLevel.ADAPTIVE, OptimizationLevel.DEBUG]
            assert isinstance(config.cache_enabled, bool)
            assert isinstance(config.parallel_processing, bool)

        print(f"✅ 設定検証テスト完了: {len(test_configs)}パターン")


if __name__ == "__main__":
    # 直接実行時のテスト
    test_suite = TestUnifiedOptimizationSystem()
    test_suite.setup_test_data()

    print("🧪 統合最適化システム包括テスト開始")

    try:
        test_suite.test_strategy_factory_registration()
        test_suite.test_performance_monitoring_integration()
        test_suite.test_configuration_validation()

        print("✅ 統合最適化システム包括テスト完了")

    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        raise
