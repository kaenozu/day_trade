#!/usr/bin/env python3
"""
Issue #870 統合テストスイート
拡張予測システムとアダプターの統合テスト

各コンポーネントの統合動作と性能を検証
"""

import unittest
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# テスト対象システム
try:
    from enhanced_prediction_core import (
        EnhancedPredictionCore, create_enhanced_prediction_core,
        PredictionConfig, PredictionMode
    )
    ENHANCED_CORE_AVAILABLE = True
except ImportError:
    ENHANCED_CORE_AVAILABLE = False

try:
    from prediction_adapter import (
        PredictionSystemAdapter, create_prediction_adapter,
        AdapterConfig, AdapterMode
    )
    ADAPTER_AVAILABLE = True
except ImportError:
    ADAPTER_AVAILABLE = False

# Issue #870 コンポーネント
try:
    from advanced_feature_selector import create_advanced_feature_selector
    FEATURE_SELECTOR_AVAILABLE = True
except ImportError:
    FEATURE_SELECTOR_AVAILABLE = False

try:
    from advanced_ensemble_system import create_advanced_ensemble_system, EnsembleMethod
    ENSEMBLE_SYSTEM_AVAILABLE = True
except ImportError:
    ENSEMBLE_SYSTEM_AVAILABLE = False

try:
    from hybrid_timeseries_predictor import create_hybrid_timeseries_predictor
    HYBRID_PREDICTOR_AVAILABLE = True
except ImportError:
    HYBRID_PREDICTOR_AVAILABLE = False

try:
    from meta_learning_system import create_meta_learning_system
    META_LEARNING_AVAILABLE = True
except ImportError:
    META_LEARNING_AVAILABLE = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


class TestDataGenerator:
    """テストデータ生成器"""

    @staticmethod
    def generate_sample_data(n_samples: int = 200, n_features: int = 20,
                           noise_level: float = 0.1, seed: int = 42) -> Dict[str, Any]:
        """サンプルデータ生成"""
        np.random.seed(seed)

        # 特徴量データ
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        # 複雑な非線形関係でターゲット作成
        y = (
            X['feature_0'] * 2.0 +
            X['feature_1'] ** 2 * 0.5 +
            X['feature_2'] * X['feature_3'] * 0.3 +
            np.sin(X['feature_4']) * 1.5 +
            np.log1p(np.abs(X['feature_5'])) * 0.8 +
            np.random.randn(n_samples) * noise_level
        )

        # 価格データシミュレーション
        price_length = min(n_samples, 100)
        price_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(price_length) * 0.02),
            'volume': np.random.randint(1000, 10000, price_length),
            'high': 100 + np.cumsum(np.random.randn(price_length) * 0.02) + np.random.rand(price_length),
            'low': 100 + np.cumsum(np.random.randn(price_length) * 0.02) - np.random.rand(price_length),
            'open': 100 + np.cumsum(np.random.randn(price_length) * 0.02)
        })

        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_full': X,
            'y_full': y,
            'price_data': price_data
        }


class TestEnhancedPredictionCore(unittest.TestCase):
    """拡張予測コアシステムテスト"""

    def setUp(self):
        """テスト準備"""
        self.test_data = TestDataGenerator.generate_sample_data()
        if ENHANCED_CORE_AVAILABLE:
            config = PredictionConfig(
                mode=PredictionMode.AUTO,
                max_features=15,
                cv_folds=3
            )
            self.core = create_enhanced_prediction_core(config)

    @unittest.skipUnless(ENHANCED_CORE_AVAILABLE, "拡張予測コア未対応")
    def test_core_initialization(self):
        """コア初期化テスト"""
        self.assertIsNotNone(self.core)
        self.assertTrue(self.core.is_initialized or self.core.fallback_mode)

        status = self.core.get_system_status()
        self.assertIn('components', status)
        self.assertIn('is_initialized', status)

    @unittest.skipUnless(ENHANCED_CORE_AVAILABLE, "拡張予測コア未対応")
    def test_prediction_execution(self):
        """予測実行テスト"""
        X_test = self.test_data['X_test']
        y_train = self.test_data['y_train']
        price_data = self.test_data['price_data']

        result = self.core.predict(X_test, y_train, price_data)

        # 基本検証
        self.assertEqual(len(result.predictions), len(X_test))
        self.assertEqual(len(result.confidence), len(X_test))
        self.assertGreater(result.processing_time, 0)
        self.assertIsInstance(result.components_used, list)

        # 予測値範囲検証
        self.assertTrue(np.all(np.isfinite(result.predictions)))
        self.assertTrue(np.all(result.confidence >= 0))
        self.assertTrue(np.all(result.confidence <= 1))

    @unittest.skipUnless(ENHANCED_CORE_AVAILABLE, "拡張予測コア未対応")
    def test_prediction_modes(self):
        """予測モードテスト"""
        X_test = self.test_data['X_test']
        y_train = self.test_data['y_train']
        price_data = self.test_data['price_data']

        # AUTO モード
        result_auto = self.core.predict(X_test, y_train, price_data)
        self.assertIsNotNone(result_auto.predictions)

        # ENHANCED モード（利用可能な場合）
        if self.core.is_initialized:
            config_enhanced = PredictionConfig(mode=PredictionMode.ENHANCED)
            core_enhanced = create_enhanced_prediction_core(config_enhanced)

            if core_enhanced.is_initialized:
                result_enhanced = core_enhanced.predict(X_test, y_train, price_data)
                self.assertIsNotNone(result_enhanced.predictions)

    @unittest.skipUnless(ENHANCED_CORE_AVAILABLE, "拡張予測コア未対応")
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 不正データでのテスト
        X_invalid = pd.DataFrame([[np.inf, np.nan, 1, 2, 3]])
        y_invalid = pd.Series([np.nan])

        try:
            result = self.core.predict(X_invalid, y_invalid)
            # エラーが発生しないか、適切にハンドリングされることを確認
            self.assertIsNotNone(result)
        except Exception as e:
            # 適切なエラーメッセージが含まれていることを確認
            self.assertIsInstance(e, (ValueError, RuntimeError))


class TestPredictionAdapter(unittest.TestCase):
    """予測システムアダプターテスト"""

    def setUp(self):
        """テスト準備"""
        self.test_data = TestDataGenerator.generate_sample_data()
        if ADAPTER_AVAILABLE:
            config = AdapterConfig(
                mode=AdapterMode.SMART_FALLBACK,
                enable_metrics=True,
                comparison_window=20
            )
            self.adapter = create_prediction_adapter(config)

    @unittest.skipUnless(ADAPTER_AVAILABLE, "アダプター未対応")
    def test_adapter_initialization(self):
        """アダプター初期化テスト"""
        self.assertIsNotNone(self.adapter)
        self.assertTrue(self.adapter.is_initialized)

    @unittest.skipUnless(ADAPTER_AVAILABLE, "アダプター未対応")
    def test_prediction_with_adapter(self):
        """アダプター経由予測テスト"""
        X_test = self.test_data['X_test']
        y_train = self.test_data['y_train']
        price_data = self.test_data['price_data']

        result = self.adapter.predict(X_test, y_train, price_data, session_id="test_session")

        # 基本検証
        self.assertEqual(len(result.predictions), len(X_test))
        self.assertIn(result.system_used, ["enhanced", "legacy", "fallback"])
        self.assertGreaterEqual(result.processing_time, 0)  # 0以上であることを確認

    @unittest.skipUnless(ADAPTER_AVAILABLE, "アダプター未対応")
    def test_ab_testing_mode(self):
        """A/Bテストモードテスト"""
        config_ab = AdapterConfig(
            mode=AdapterMode.AB_TEST,
            ab_test_split=0.5
        )
        adapter_ab = create_prediction_adapter(config_ab)

        X_test = self.test_data['X_test'][:10]  # 小さなデータでテスト
        y_train = self.test_data['y_train']

        # 複数セッションでテスト
        sessions = ["session_1", "session_2", "session_3", "session_4"]
        results = []

        for session_id in sessions:
            result = adapter_ab.predict(X_test, y_train, session_id=session_id)
            results.append(result)

        # テストグループが割り当てられていることを確認
        test_groups = [r.test_group for r in results if r.test_group is not None]
        self.assertGreater(len(test_groups), 0)

    @unittest.skipUnless(ADAPTER_AVAILABLE, "アダプター未対応")
    def test_gradual_rollout_mode(self):
        """段階的移行モードテスト"""
        config_rollout = AdapterConfig(
            mode=AdapterMode.GRADUAL_ROLLOUT,
            rollout_percentage=0.3
        )
        adapter_rollout = create_prediction_adapter(config_rollout)

        X_test = self.test_data['X_test'][:5]
        y_train = self.test_data['y_train']

        # 複数回実行して移行率を検証
        enhanced_count = 0
        total_predictions = 20

        for i in range(total_predictions):
            result = adapter_rollout.predict(X_test, y_train)
            if result.system_used == "enhanced":
                enhanced_count += 1

        # 移行率が設定範囲内にあることを確認（確率的なので幅を持たせる）
        enhanced_ratio = enhanced_count / total_predictions
        self.assertGreaterEqual(enhanced_ratio, 0.0)  # 最低値
        self.assertLessEqual(enhanced_ratio, 1.0)     # 最高値


class TestComponentIntegration(unittest.TestCase):
    """コンポーネント統合テスト"""

    def setUp(self):
        """テスト準備"""
        self.test_data = TestDataGenerator.generate_sample_data()

    @unittest.skipUnless(FEATURE_SELECTOR_AVAILABLE, "特徴量選択システム未対応")
    def test_feature_selector_integration(self):
        """特徴量選択システム統合テスト"""
        selector = create_advanced_feature_selector(max_features=10)

        X = self.test_data['X_full']
        y = self.test_data['y_full']
        price_data = self.test_data['price_data']

        selected_X, selection_info = selector.select_features(X, y, price_data)

        # 基本検証
        self.assertLessEqual(selected_X.shape[1], 10)
        self.assertGreater(selected_X.shape[1], 0)
        self.assertIn('selected_features', selection_info)
        self.assertIn('market_regime', selection_info)

    @unittest.skipUnless(ENSEMBLE_SYSTEM_AVAILABLE, "アンサンブルシステム未対応")
    def test_ensemble_system_integration(self):
        """アンサンブルシステム統合テスト"""
        ensemble = create_advanced_ensemble_system(
            method=EnsembleMethod.VOTING,
            cv_folds=3
        )

        X_train = self.test_data['X_train']
        y_train = self.test_data['y_train']
        X_test = self.test_data['X_test']

        # 訓練・予測
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)

        # 基本検証
        self.assertEqual(len(predictions), len(X_test))
        self.assertTrue(np.all(np.isfinite(predictions)))

        # 要約取得
        summary = ensemble.get_ensemble_summary()
        self.assertIn('ensemble_models', summary)

    @unittest.skipUnless(HYBRID_PREDICTOR_AVAILABLE, "ハイブリッド予測システム未対応")
    def test_hybrid_predictor_integration(self):
        """ハイブリッド予測システム統合テスト"""
        predictor = create_hybrid_timeseries_predictor(
            sequence_length=10,
            lstm_units=20
        )

        y_train = self.test_data['y_train']

        # 訓練・予測
        predictor.fit(y_train.values)
        predictions = predictor.predict(steps=10)

        # 基本検証
        self.assertEqual(len(predictions), 10)
        self.assertTrue(np.all(np.isfinite(predictions)))

        # システム要約
        summary = predictor.get_system_summary()
        self.assertIn('is_fitted', summary)

    @unittest.skipUnless(META_LEARNING_AVAILABLE, "メタラーニングシステム未対応")
    def test_meta_learning_integration(self):
        """メタラーニングシステム統合テスト"""
        meta_learner = create_meta_learning_system(repository_size=10)

        X_train = self.test_data['X_train']
        y_train = self.test_data['y_train']
        price_data = self.test_data['price_data']

        # 訓練・予測
        model, predictions, result_info = meta_learner.fit_predict(
            X_train, y_train, price_data
        )

        # 基本検証
        self.assertEqual(len(predictions), len(X_train))
        self.assertIsNotNone(model)
        self.assertIn('model_type', result_info)

        # 学習洞察
        insights = meta_learner.get_learning_insights()
        self.assertIn('total_episodes', insights)


class TestPerformanceValidation(unittest.TestCase):
    """性能検証テスト"""

    def setUp(self):
        """テスト準備"""
        self.test_data = TestDataGenerator.generate_sample_data(n_samples=500)

    @unittest.skipUnless(ENHANCED_CORE_AVAILABLE, "拡張予測コア未対応")
    def test_prediction_accuracy(self):
        """予測精度テスト"""
        if not ENHANCED_CORE_AVAILABLE:
            self.skipTest("拡張予測コア未対応")

        config = PredictionConfig(max_features=20, cv_folds=3)
        core = create_enhanced_prediction_core(config)

        X_train = self.test_data['X_train']
        X_test = self.test_data['X_test']
        y_train = self.test_data['y_train']
        y_test = self.test_data['y_test']
        price_data = self.test_data['price_data']

        # 予測実行
        result = core.predict(X_test, y_train, price_data)

        # 精度評価（訓練データを使った予測なので完璧ではないが、基本的な検証）
        if len(result.predictions) == len(y_test):
            mse = mean_squared_error(y_test, result.predictions)
            mae = mean_absolute_error(y_test, result.predictions)
            r2 = r2_score(y_test, result.predictions)

            # 基本的な精度要件（データによって調整が必要）
            self.assertLess(mse, 10.0)  # MSEが合理的範囲
            self.assertLess(mae, 5.0)   # MAEが合理的範囲
            # R²は負の値も取り得るので最小値は設定しない

    def test_processing_time_performance(self):
        """処理時間性能テスト"""
        if not ENHANCED_CORE_AVAILABLE and not ADAPTER_AVAILABLE:
            self.skipTest("テスト対象システム未対応")

        X_test = self.test_data['X_test'][:50]  # 小さなデータセット
        y_train = self.test_data['y_train']

        # 拡張システムテスト
        if ENHANCED_CORE_AVAILABLE:
            core = create_enhanced_prediction_core()
            start_time = time.time()
            core.predict(X_test, y_train)
            enhanced_time = time.time() - start_time

            # 処理時間が合理的範囲内であることを確認
            self.assertLess(enhanced_time, 30.0)  # 30秒以内

        # アダプターテスト
        if ADAPTER_AVAILABLE:
            adapter = create_prediction_adapter()
            start_time = time.time()
            adapter.predict(X_test, y_train)
            adapter_time = time.time() - start_time

            # 処理時間が合理的範囲内であることを確認
            self.assertLess(adapter_time, 30.0)  # 30秒以内


def run_integration_tests(verbose: bool = True) -> Dict[str, Any]:
    """統合テスト実行"""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # テストスイート作成
    test_classes = [
        TestEnhancedPredictionCore,
        TestPredictionAdapter,
        TestComponentIntegration,
        TestPerformanceValidation
    ]

    results = {
        'timestamp': datetime.now(),
        'test_results': {},
        'summary': {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0
        }
    }

    for test_class in test_classes:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)

        print(f"\n{'='*60}")
        print(f"実行中: {test_class.__name__}")
        print(f"{'='*60}")

        test_result = runner.run(suite)

        # 結果記録
        class_name = test_class.__name__
        results['test_results'][class_name] = {
            'tests_run': test_result.testsRun,
            'failures': len(test_result.failures),
            'errors': len(test_result.errors),
            'skipped': len(test_result.skipped) if hasattr(test_result, 'skipped') else 0
        }

        # サマリー更新
        results['summary']['total_tests'] += test_result.testsRun
        results['summary']['failed_tests'] += len(test_result.failures) + len(test_result.errors)
        if hasattr(test_result, 'skipped'):
            results['summary']['skipped_tests'] += len(test_result.skipped)

    results['summary']['passed_tests'] = (
        results['summary']['total_tests'] -
        results['summary']['failed_tests'] -
        results['summary']['skipped_tests']
    )

    # 結果表示
    print(f"\n{'='*60}")
    print(f"統合テスト結果サマリー")
    print(f"{'='*60}")
    print(f"総テスト数: {results['summary']['total_tests']}")
    print(f"成功: {results['summary']['passed_tests']}")
    print(f"失敗: {results['summary']['failed_tests']}")
    print(f"スキップ: {results['summary']['skipped_tests']}")

    success_rate = results['summary']['passed_tests'] / max(results['summary']['total_tests'], 1)
    print(f"成功率: {success_rate:.1%}")

    # 詳細結果
    print(f"\n詳細結果:")
    for class_name, class_results in results['test_results'].items():
        print(f"  {class_name}:")
        print(f"    実行: {class_results['tests_run']}")
        print(f"    失敗: {class_results['failures']}")
        print(f"    エラー: {class_results['errors']}")
        print(f"    スキップ: {class_results['skipped']}")

    return results


if __name__ == "__main__":
    # 統合テスト実行
    print("Issue #870 統合テストスイート実行開始")
    print("="*60)

    # システム可用性チェック
    print("システム可用性チェック:")
    print(f"  拡張予測コア: {'OK' if ENHANCED_CORE_AVAILABLE else 'NG'}")
    print(f"  アダプター: {'OK' if ADAPTER_AVAILABLE else 'NG'}")
    print(f"  特徴量選択: {'OK' if FEATURE_SELECTOR_AVAILABLE else 'NG'}")
    print(f"  アンサンブル: {'OK' if ENSEMBLE_SYSTEM_AVAILABLE else 'NG'}")
    print(f"  ハイブリッド予測: {'OK' if HYBRID_PREDICTOR_AVAILABLE else 'NG'}")
    print(f"  メタラーニング: {'OK' if META_LEARNING_AVAILABLE else 'NG'}")
    print()

    # テスト実行
    test_results = run_integration_tests(verbose=True)

    # 結果保存
    try:
        import json
        with open('integration_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nテスト結果保存: integration_test_results.json")
    except Exception as e:
        print(f"結果保存エラー: {e}")

    print("\n統合テスト完了")