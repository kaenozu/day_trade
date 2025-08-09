#!/usr/bin/env python3
"""
統合最適化システム包括テスト

Strategy Pattern実装とPhase A-D統合リファクタリングシステムの動作テスト
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.day_trade.core.optimization_strategy import (
        OptimizationLevel,
        OptimizationConfig,
        OptimizationStrategyFactory,
        get_optimized_implementation
    )
    OPTIMIZATION_STRATEGY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: optimization_strategy import failed: {e}")
    OPTIMIZATION_STRATEGY_AVAILABLE = False

# 統合システムのインポート
try:
    from src.day_trade.analysis.technical_indicators_unified import TechnicalIndicatorsManager
    TECHNICAL_INDICATORS_AVAILABLE = True
except ImportError:
    TECHNICAL_INDICATORS_AVAILABLE = False

try:
    from src.day_trade.analysis.feature_engineering_unified import FeatureEngineeringManager, FeatureConfig
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False

try:
    from src.day_trade.models.database_unified import DatabaseManager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


@pytest.mark.skipif(not OPTIMIZATION_STRATEGY_AVAILABLE, reason="optimization_strategy not available")
class TestUnifiedOptimizationSystem:
    """統合最適化システムテストクラス"""

    @classmethod
    def setup_class(cls):
        """テストクラス初期化"""
        cls.test_data = cls._generate_test_data()
        cls.config_standard = OptimizationConfig(level=OptimizationLevel.STANDARD)
        cls.config_optimized = OptimizationConfig(level=OptimizationLevel.OPTIMIZED)
        cls.config_adaptive = OptimizationConfig(level=OptimizationLevel.ADAPTIVE)

    @staticmethod
    def _generate_test_data() -> pd.DataFrame:
        """テスト用データ生成"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        prices = 1000 + np.cumsum(np.random.randn(200) * 10)

        return pd.DataFrame({
            '終値': prices,
            '高値': prices + np.random.rand(200) * 5,
            '安値': prices - np.random.rand(200) * 5,
            '出来高': np.random.randint(1000, 10000, 200)
        }, index=dates)

    def test_optimization_config_creation(self):
        """最適化設定の作成テスト"""
        # デフォルト設定
        config = OptimizationConfig()
        assert config.level == OptimizationLevel.STANDARD
        assert config.auto_fallback == True
        assert config.cache_enabled == True

        # 環境変数からの設定
        os.environ["DAYTRADE_OPTIMIZATION_LEVEL"] = "optimized"
        config = OptimizationConfig.from_env()
        assert config.level == OptimizationLevel.OPTIMIZED

        # 無効なレベルの処理
        os.environ["DAYTRADE_OPTIMIZATION_LEVEL"] = "invalid"
        config = OptimizationConfig.from_env()
        assert config.level == OptimizationLevel.STANDARD  # フォールバック

        # 環境変数をクリア
        del os.environ["DAYTRADE_OPTIMIZATION_LEVEL"]

    def test_strategy_factory_registration(self):
        """戦略ファクトリーの登録テスト"""
        # 戦略の登録確認
        components = OptimizationStrategyFactory.get_registered_components()
        assert isinstance(components, dict)

        # コンポーネント登録確認
        expected_components = []
        if TECHNICAL_INDICATORS_AVAILABLE:
            expected_components.append("technical_indicators")
        if FEATURE_ENGINEERING_AVAILABLE:
            expected_components.append("feature_engineering")
        if DATABASE_AVAILABLE:
            expected_components.append("database")

        for component in expected_components:
            assert component in components
            assert "standard" in components[component]

    @pytest.mark.skipif(not TECHNICAL_INDICATORS_AVAILABLE, reason="Technical indicators not available")
    def test_technical_indicators_unified(self):
        """統合テクニカル指標テスト"""
        # 標準実装テスト
        manager = TechnicalIndicatorsManager(self.config_standard)
        indicators = ["sma", "bollinger_bands", "rsi"]

        start_time = time.time()
        results = manager.calculate_indicators(self.test_data, indicators, period=20)
        standard_time = time.time() - start_time

        assert len(results) == len(indicators)
        for indicator in indicators:
            assert indicator in results
            assert results[indicator].strategy_used == "標準テクニカル指標"

        # 最適化実装テスト
        manager_opt = TechnicalIndicatorsManager(self.config_optimized)

        start_time = time.time()
        results_opt = manager_opt.calculate_indicators(self.test_data, indicators, period=20)
        optimized_time = time.time() - start_time

        assert len(results_opt) == len(indicators)
        for indicator in indicators:
            assert indicator in results_opt
            assert results_opt[indicator].strategy_used == "最適化テクニカル指標"

        # パフォーマンス比較
        print(f"テクニカル指標処理時間 - 標準: {standard_time:.3f}秒, 最適化: {optimized_time:.3f}秒")

        # パフォーマンス指標確認
        standard_metrics = manager.get_performance_summary()
        optimized_metrics = manager_opt.get_performance_summary()

        assert "execution_count" in standard_metrics
        assert "execution_count" in optimized_metrics
        assert standard_metrics["execution_count"] > 0
        assert optimized_metrics["execution_count"] > 0

    @pytest.mark.skipif(not FEATURE_ENGINEERING_AVAILABLE, reason="Feature engineering not available")
    def test_feature_engineering_unified(self):
        """統合特徴量エンジニアリングテスト"""
        # 標準実装テスト
        manager = FeatureEngineeringManager(self.config_standard)
        feature_config = FeatureConfig.default()

        start_time = time.time()
        result = manager.generate_features(self.test_data, feature_config)
        standard_time = time.time() - start_time

        assert isinstance(result.features, pd.DataFrame)
        assert result.features.shape[0] == len(self.test_data)
        assert len(result.feature_names) > 0
        assert result.strategy_used == "標準特徴量エンジニアリング"

        # 最適化実装テスト
        manager_opt = FeatureEngineeringManager(self.config_optimized)

        start_time = time.time()
        result_opt = manager_opt.generate_features(self.test_data, feature_config)
        optimized_time = time.time() - start_time

        assert isinstance(result_opt.features, pd.DataFrame)
        assert result_opt.features.shape[0] == len(self.test_data)
        assert len(result_opt.feature_names) > 0
        assert result_opt.strategy_used == "最適化特徴量エンジニアリング"

        # パフォーマンス比較
        print(f"特徴量生成処理時間 - 標準: {standard_time:.3f}秒, 最適化: {optimized_time:.3f}秒")

        # 特徴量数の比較
        assert result.features.shape[1] > 0
        assert result_opt.features.shape[1] > 0

    @pytest.mark.skipif(not DATABASE_AVAILABLE, reason="Database not available")
    def test_database_unified(self):
        """統合データベーステスト"""
        # 標準実装テスト
        manager = DatabaseManager(self.config_standard)

        # 簡単なクエリテスト
        result = manager.execute_query("SELECT 1 as test_value")
        assert result.success == True
        assert result.strategy_used == "標準データベース"

        # 最適化実装テスト
        manager_opt = DatabaseManager(self.config_optimized)

        result_opt = manager_opt.execute_query("SELECT datetime('now') as current_time")
        assert result_opt.success == True
        assert result_opt.strategy_used == "最適化データベース"

        # パフォーマンス指標確認
        standard_metrics = manager.get_performance_summary()
        optimized_metrics = manager_opt.get_performance_summary()

        assert "execution_count" in standard_metrics
        assert "execution_count" in optimized_metrics

    def test_adaptive_level_selection(self):
        """適応的レベル選択テスト"""
        config = OptimizationConfig(level=OptimizationLevel.ADAPTIVE)

        # システム状況に基づく適応的選択をテスト
        # （実際の負荷状況に依存するため、戦略取得のみテスト）
        if TECHNICAL_INDICATORS_AVAILABLE:
            strategy = get_optimized_implementation("technical_indicators", config)
            assert strategy is not None
            assert hasattr(strategy, 'get_strategy_name')

    def test_fallback_mechanism(self):
        """フォールバック機能テスト"""
        config = OptimizationConfig(
            level=OptimizationLevel.OPTIMIZED,
            auto_fallback=True
        )

        # 利用不可能なコンポーネントでのフォールバック
        # （実装されていないコンポーネントでのテスト）
        try:
            strategy = get_optimized_implementation("non_existent_component", config)
            # このテストは例外が発生することが期待される
            assert False, "存在しないコンポーネントで例外が発生しませんでした"
        except ValueError:
            pass  # 期待される動作

    def test_performance_monitoring(self):
        """パフォーマンス監視テスト"""
        config = OptimizationConfig(performance_monitoring=True)

        if TECHNICAL_INDICATORS_AVAILABLE:
            manager = TechnicalIndicatorsManager(config)

            # 複数回実行してパフォーマンス指標を蓄積
            for _ in range(3):
                manager.calculate_indicators(self.test_data, ["sma"], period=10)

            metrics = manager.get_performance_summary()
            assert metrics["execution_count"] >= 3
            assert "average_time" in metrics
            assert metrics["average_time"] > 0

    def test_configuration_file_handling(self):
        """設定ファイル処理テスト"""
        # テスト設定ファイルの作成
        test_config = {
            "level": "optimized",
            "auto_fallback": True,
            "performance_monitoring": True,
            "cache_enabled": True,
            "batch_size": 500
        }

        config_file = "test_optimization_config.json"

        try:
            with open(config_file, 'w') as f:
                json.dump(test_config, f)

            # 設定ファイルからの読み込みテスト
            config = OptimizationConfig.from_file(config_file)
            assert config.level == OptimizationLevel.OPTIMIZED
            assert config.auto_fallback == True
            assert config.batch_size == 500

        finally:
            # テストファイルのクリーンアップ
            if os.path.exists(config_file):
                os.remove(config_file)

    def test_memory_and_cache_management(self):
        """メモリとキャッシュ管理テスト"""
        config = OptimizationConfig(
            cache_enabled=True,
            memory_limit_mb=256
        )

        if TECHNICAL_INDICATORS_AVAILABLE:
            manager = TechnicalIndicatorsManager(config)

            # 同一データでの複数回実行（キャッシュ効果確認）
            indicators = ["sma", "rsi"]

            # 初回実行
            start_time = time.time()
            results1 = manager.calculate_indicators(self.test_data, indicators)
            first_time = time.time() - start_time

            # 2回目実行（キャッシュからの取得期待）
            start_time = time.time()
            results2 = manager.calculate_indicators(self.test_data, indicators)
            second_time = time.time() - start_time

            # キャッシュ効果の確認（2回目が著しく高速である必要はないが、実行は成功すべき）
            assert len(results1) == len(results2)
            print(f"キャッシュテスト - 初回: {first_time:.3f}秒, 2回目: {second_time:.3f}秒")


def run_integration_test():
    """統合テストの実行"""
    print("=" * 60)
    print("統合最適化システム包括テスト開始")
    print("=" * 60)

    # システム情報の表示
    print(f"Python版: {sys.version}")
    print(f"NumPy版: {np.__version__}")
    print(f"Pandas版: {pd.__version__}")

    # 利用可能コンポーネントの表示
    available_components = []
    if TECHNICAL_INDICATORS_AVAILABLE:
        available_components.append("テクニカル指標")
    if FEATURE_ENGINEERING_AVAILABLE:
        available_components.append("特徴量エンジニアリング")
    if DATABASE_AVAILABLE:
        available_components.append("データベース")

    print(f"利用可能コンポーネント: {', '.join(available_components)}")

    # テストの実行
    test_instance = TestUnifiedOptimizationSystem()
    test_instance.setup_class()

    test_methods = [
        ("設定作成テスト", test_instance.test_optimization_config_creation),
        ("戦略ファクトリーテスト", test_instance.test_strategy_factory_registration),
        ("設定ファイルテスト", test_instance.test_configuration_file_handling),
        ("パフォーマンス監視テスト", test_instance.test_performance_monitoring),
        ("メモリキャッシュテスト", test_instance.test_memory_and_cache_management),
    ]

    # コンポーネント別テストの追加
    if TECHNICAL_INDICATORS_AVAILABLE:
        test_methods.append(("テクニカル指標統合テスト", test_instance.test_technical_indicators_unified))

    if FEATURE_ENGINEERING_AVAILABLE:
        test_methods.append(("特徴量エンジニアリング統合テスト", test_instance.test_feature_engineering_unified))

    if DATABASE_AVAILABLE:
        test_methods.append(("データベース統合テスト", test_instance.test_database_unified))

    # 追加テスト
    test_methods.extend([
        ("適応的レベル選択テスト", test_instance.test_adaptive_level_selection),
        ("フォールバック機能テスト", test_instance.test_fallback_mechanism),
    ])

    # テスト実行
    passed = 0
    failed = 0

    for test_name, test_method in test_methods:
        try:
            print(f"\n実行中: {test_name}")
            start_time = time.time()
            test_method()
            execution_time = time.time() - start_time
            print(f"[OK] {test_name} - 成功 ({execution_time:.3f}秒)")
            passed += 1
        except Exception as e:
            print(f"[NG] {test_name} - 失敗: {e}")
            failed += 1

    # 結果サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    print(f"成功: {passed}")
    print(f"失敗: {failed}")
    print(f"合計: {passed + failed}")
    print(f"成功率: {passed / (passed + failed) * 100:.1f}%")

    return failed == 0


if __name__ == "__main__":
    try:
        if not OPTIMIZATION_STRATEGY_AVAILABLE:
            print("WARNING: optimization_strategy not available. Skipping tests.")
            sys.exit(0)

        success = run_integration_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Fatal error during test execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
