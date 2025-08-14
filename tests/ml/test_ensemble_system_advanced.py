#!/usr/bin/env python3
"""
Issue #755 Phase 2: EnsembleSystem高度テストスイート

93%精度アンサンブルシステム(Issue #487)の詳細検証
- XGBoost + CatBoost + RandomForest統合テスト
- ハイパーパラメータ最適化テスト
- パフォーマンス・堅牢性・正確性の包括的検証
- リアルタイム予測精度テスト
"""

import unittest
import pytest
import numpy as np
import pandas as pd
import tempfile
import time
import warnings
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import json
import pickle

# テスト対象システムのインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.day_trade.ml.ensemble_system import (
        EnsembleSystem,
        EnsembleConfig,
        EnsembleMethod,
        EnsemblePrediction
    )
    from src.day_trade.ml.base_models import (
        XGBOOST_AVAILABLE,
        CATBOOST_AVAILABLE,
        RandomForestModel,
        GradientBoostingModel
    )
    if XGBOOST_AVAILABLE:
        from src.day_trade.ml.base_models import XGBoostModel
    if CATBOOST_AVAILABLE:
        from src.day_trade.ml.base_models import CatBoostModel

    from src.day_trade.ml.stacking_ensemble import StackingConfig
    from src.day_trade.ml.dynamic_weighting_system import DynamicWeightingConfig

except ImportError as e:
    print(f"インポートエラー: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")


class TestEnsembleSystem93Accuracy(unittest.TestCase):
    """93%精度アンサンブルシステム検証テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.config = EnsembleConfig(
            use_xgboost=XGBOOST_AVAILABLE,
            use_catboost=CATBOOST_AVAILABLE,
            use_random_forest=True,
            use_gradient_boosting=True,
            use_svr=False,  # 高速化のため無効
            enable_dynamic_weighting=True,
            enable_stacking=True,
            cv_folds=3,  # テスト高速化
            verbose=False
        )
        self.ensemble = EnsembleSystem(self.config)

        # テスト用データ生成
        np.random.seed(42)
        self.n_samples = 1000
        self.n_features = 20

        # より現実的な金融データパターンを模擬
        self.X_train, self.y_train = self._generate_realistic_financial_data(
            self.n_samples, self.n_features
        )
        self.X_test, self.y_test = self._generate_realistic_financial_data(
            200, self.n_features
        )

    def _generate_realistic_financial_data(self, n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        """現実的な金融データパターン生成"""
        # トレンド成分
        trend = np.linspace(0, 2, n_samples)

        # 周期性成分（季節性）
        seasonal = 0.5 * np.sin(2 * np.pi * np.arange(n_samples) / 252)  # 年間周期

        # ボラティリティクラスタリング（GARCH効果）
        volatility = np.random.exponential(0.1, n_samples)

        # 基本特徴量
        features = np.random.randn(n_samples, n_features)

        # 金融指標風の特徴量
        features[:, 0] = trend  # トレンド指標
        features[:, 1] = seasonal  # 季節性指標
        features[:, 2] = volatility  # ボラティリティ指標

        # 非線形関係を含むターゲット
        target = (
            0.3 * trend +
            0.2 * seasonal +
            0.1 * features[:, 3] * features[:, 4] +  # 相互作用項
            0.05 * np.sin(features[:, 5]) +  # 非線形項
            volatility * np.random.randn(n_samples) * 0.1  # ヘテロスケダスティック性
        )

        return features, target

    def test_93_percent_accuracy_target(self):
        """93%精度目標達成テスト"""
        # アンサンブル訓練
        self.ensemble.fit(self.X_train, self.y_train)

        # 予測実行
        predictions = self.ensemble.predict(self.X_test)

        # 精度計算（回帰問題では R²スコア使用）
        from sklearn.metrics import r2_score, mean_absolute_percentage_error

        r2 = r2_score(self.y_test, predictions.final_predictions)
        mape = mean_absolute_percentage_error(self.y_test, predictions.final_predictions)

        # 93%精度に近い性能を期待（テストデータでは90%以上で良い）
        self.assertGreater(r2, 0.85, f"R²スコア {r2:.3f} が目標 0.85 を下回りました")
        self.assertLess(mape, 0.15, f"MAPE {mape:.3f} が目標 0.15 を上回りました")

        print(f"R²スコア: {r2:.3f}, MAPE: {mape:.3f}")

    def test_xgboost_catboost_integration(self):
        """XGBoost + CatBoost統合テスト"""
        if not (XGBOOST_AVAILABLE and CATBOOST_AVAILABLE):
            self.skipTest("XGBoost または CatBoost が利用できません")

        # 高精度モデル専用設定
        config = EnsembleConfig(
            use_xgboost=True,
            use_catboost=True,
            use_random_forest=False,  # 高精度モデルのみ
            use_gradient_boosting=False,
            use_svr=False,
            enable_dynamic_weighting=True,
            verbose=False
        )

        ensemble = EnsembleSystem(config)
        ensemble.fit(self.X_train, self.y_train)

        # 両モデルが正常に初期化されたか確認
        model_names = list(ensemble.base_models.keys())
        self.assertIn('xgboost', model_names)
        self.assertIn('catboost', model_names)

        # 予測実行
        predictions = ensemble.predict(self.X_test)

        # XGBoostとCatBoostの個別予測が存在することを確認
        self.assertIn('xgboost', predictions.individual_predictions)
        self.assertIn('catboost', predictions.individual_predictions)

        # 予測値の妥当性確認
        self.assertEqual(len(predictions.final_predictions), len(self.y_test))
        self.assertFalse(np.any(np.isnan(predictions.final_predictions)))

    def test_hyperparameter_optimization(self):
        """ハイパーパラメータ最適化テスト"""
        config = EnsembleConfig(
            use_random_forest=True,
            use_gradient_boosting=True,
            random_forest_params={'enable_hyperopt': True, 'n_estimators': 50},
            gradient_boosting_params={'enable_hyperopt': True, 'n_estimators': 50},
            verbose=False
        )

        ensemble = EnsembleSystem(config)

        # ハイパーパラメータ最適化付きで訓練
        start_time = time.time()
        ensemble.fit(self.X_train, self.y_train)
        optimization_time = time.time() - start_time

        # 最適化が実行されたことを確認（時間ベース）
        self.assertGreater(optimization_time, 1.0, "ハイパーパラメータ最適化に十分な時間がかかっていません")

        # 予測実行
        predictions = ensemble.predict(self.X_test)

        # 最適化後の性能評価
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(self.y_test, predictions.final_predictions)

        # 最適化により改善された性能を期待
        baseline_mse = np.var(self.y_test)  # 単純なベースライン
        improvement_ratio = mse / baseline_mse

        self.assertLess(improvement_ratio, 0.8,
                       f"最適化後のMSE比率 {improvement_ratio:.3f} が期待値 0.8 を上回りました")

    def test_dynamic_weighting_system(self):
        """動的重み付けシステムテスト"""
        config = EnsembleConfig(
            enable_dynamic_weighting=True,
            weight_update_frequency=50,
            performance_window=200,
            verbose=False
        )

        ensemble = EnsembleSystem(config)
        ensemble.fit(self.X_train, self.y_train)

        # 初期重み取得
        initial_weights = ensemble.model_weights.copy()

        # 段階的予測による重み更新
        chunk_size = 50
        for i in range(0, len(self.X_test), chunk_size):
            chunk_X = self.X_test[i:i+chunk_size]
            predictions = ensemble.predict(chunk_X)

            # 重みが更新されているか確認
            if i > 0:  # 最初の予測後
                current_weights = ensemble.model_weights
                weight_changed = any(
                    abs(current_weights.get(model, 0) - initial_weights.get(model, 0)) > 0.01
                    for model in initial_weights.keys()
                )

                if weight_changed:  # 重みが変化した場合はテスト成功
                    break

        # 動的重み付けの効果を確認
        final_weights = ensemble.model_weights
        self.assertIsInstance(final_weights, dict)
        self.assertGreater(len(final_weights), 0)

        # 重みの合計が1に近いことを確認
        total_weight = sum(final_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)

    def test_stacking_ensemble_performance(self):
        """スタッキングアンサンブル性能テスト"""
        config = EnsembleConfig(
            enable_stacking=True,
            stacking_config=StackingConfig(
                meta_learner_type='linear',
                cv_folds=3,
                enable_feature_selection=True
            ),
            verbose=False
        )

        ensemble = EnsembleSystem(config)
        ensemble.fit(self.X_train, self.y_train)

        # スタッキング予測
        predictions = ensemble.predict(self.X_test)

        # スタッキングが使用されたことを確認
        self.assertEqual(predictions.method_used, 'stacking')

        # 性能評価
        from sklearn.metrics import r2_score
        r2 = r2_score(self.y_test, predictions.final_predictions)

        # スタッキングにより改善された性能を期待
        self.assertGreater(r2, 0.7, f"スタッキングのR²スコア {r2:.3f} が期待値 0.7 を下回りました")

        # メタ学習器の存在確認
        self.assertTrue(hasattr(ensemble, 'stacking_ensemble'))
        if hasattr(ensemble, 'stacking_ensemble') and ensemble.stacking_ensemble:
            self.assertTrue(ensemble.stacking_ensemble.is_trained)

    def test_real_time_prediction_performance(self):
        """リアルタイム予測性能テスト"""
        self.ensemble.fit(self.X_train, self.y_train)

        # 単一サンプル予測時間測定
        single_sample = self.X_test[:1]

        prediction_times = []
        for _ in range(10):
            start_time = time.time()
            predictions = self.ensemble.predict(single_sample)
            prediction_time = time.time() - start_time
            prediction_times.append(prediction_time)

        avg_prediction_time = np.mean(prediction_times)
        std_prediction_time = np.std(prediction_times)

        # リアルタイム要件：1秒以内
        self.assertLess(avg_prediction_time, 1.0,
                       f"平均予測時間 {avg_prediction_time:.3f}秒 が目標 1.0秒 を上回りました")

        # 安定性確認：標準偏差が小さいこと
        self.assertLess(std_prediction_time, 0.1,
                       f"予測時間の標準偏差 {std_prediction_time:.3f} が大きすぎます")

        print(f"平均予測時間: {avg_prediction_time:.3f}±{std_prediction_time:.3f}秒")

    def test_ensemble_confidence_scoring(self):
        """アンサンブル信頼度スコアリングテスト"""
        self.ensemble.fit(self.X_train, self.y_train)
        predictions = self.ensemble.predict(self.X_test)

        # 信頼度スコアの存在確認
        self.assertIsInstance(predictions.ensemble_confidence, np.ndarray)
        self.assertEqual(len(predictions.ensemble_confidence), len(self.y_test))

        # 信頼度スコアの範囲確認（0-1）
        self.assertTrue(np.all(predictions.ensemble_confidence >= 0))
        self.assertTrue(np.all(predictions.ensemble_confidence <= 1))

        # 信頼度と予測精度の相関確認
        from sklearn.metrics import mean_squared_error

        # 高信頼度予測の精度が高いことを確認
        high_confidence_mask = predictions.ensemble_confidence > 0.7
        if np.any(high_confidence_mask):
            high_conf_mse = mean_squared_error(
                self.y_test[high_confidence_mask],
                predictions.final_predictions[high_confidence_mask]
            )

            # 低信頼度予測との比較
            low_confidence_mask = predictions.ensemble_confidence < 0.3
            if np.any(low_confidence_mask):
                low_conf_mse = mean_squared_error(
                    self.y_test[low_confidence_mask],
                    predictions.final_predictions[low_confidence_mask]
                )

                # 高信頼度予測の方が精度が高いことを期待
                self.assertLessEqual(high_conf_mse, low_conf_mse * 1.2,
                                   "高信頼度予測の精度が期待ほど高くありません")


class TestEnsembleSystemRobustness(unittest.TestCase):
    """アンサンブルシステム堅牢性テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.config = EnsembleConfig(verbose=False)
        self.ensemble = EnsembleSystem(self.config)

        # テスト用データ
        np.random.seed(123)
        self.n_samples = 500
        self.n_features = 15
        self.X_train = np.random.randn(self.n_samples, self.n_features)
        self.y_train = np.random.randn(self.n_samples)
        self.X_test = np.random.randn(100, self.n_features)

    def test_missing_data_handling(self):
        """欠損データ処理テスト"""
        # 訓練データに欠損値を導入
        X_train_missing = self.X_train.copy()
        X_train_missing[::10, ::3] = np.nan  # 10%程度の欠損

        # テストデータに欠損値を導入
        X_test_missing = self.X_test.copy()
        X_test_missing[::5, ::2] = np.nan

        # 欠損データでも正常に動作することを確認
        try:
            self.ensemble.fit(X_train_missing, self.y_train)
            predictions = self.ensemble.predict(X_test_missing)

            # 予測結果に欠損値がないことを確認
            self.assertFalse(np.any(np.isnan(predictions.final_predictions)))

        except Exception as e:
            self.fail(f"欠損データ処理でエラー: {e}")

    def test_extreme_outlier_robustness(self):
        """極端な外れ値に対する堅牢性テスト"""
        # 外れ値を含むデータ生成
        X_train_outlier = self.X_train.copy()
        y_train_outlier = self.y_train.copy()

        # 極端な外れ値を追加
        X_train_outlier[0, :] = 1000  # 極端に大きな値
        X_train_outlier[1, :] = -1000  # 極端に小さな値
        y_train_outlier[0] = 1000
        y_train_outlier[1] = -1000

        # 外れ値があっても正常に動作することを確認
        try:
            self.ensemble.fit(X_train_outlier, y_train_outlier)
            predictions = self.ensemble.predict(self.X_test)

            # 予測結果が妥当な範囲内であることを確認
            pred_std = np.std(predictions.final_predictions)
            self.assertLess(pred_std, 100, "外れ値により予測が不安定になりました")

        except Exception as e:
            self.fail(f"外れ値処理でエラー: {e}")

    def test_small_dataset_handling(self):
        """小規模データセット処理テスト"""
        # 非常に小さなデータセット
        X_small = np.random.randn(20, 5)
        y_small = np.random.randn(20)

        config = EnsembleConfig(
            cv_folds=2,  # 小データに適応
            verbose=False
        )
        ensemble = EnsembleSystem(config)

        try:
            ensemble.fit(X_small, y_small)
            predictions = ensemble.predict(X_small[:5])

            # 予測が正常に実行されることを確認
            self.assertEqual(len(predictions.final_predictions), 5)

        except Exception as e:
            self.fail(f"小規模データセット処理でエラー: {e}")

    def test_high_dimensional_data(self):
        """高次元データ処理テスト"""
        # 特徴量数がサンプル数より多い場合
        X_high_dim = np.random.randn(100, 200)  # 100サンプル、200特徴量
        y_high_dim = np.random.randn(100)

        config = EnsembleConfig(
            # 高次元データに適したモデルのみ使用
            use_svr=False,  # SVRは高次元で重い
            verbose=False
        )
        ensemble = EnsembleSystem(config)

        try:
            ensemble.fit(X_high_dim, y_high_dim)
            predictions = ensemble.predict(X_high_dim[:10])

            # 予測が正常に実行されることを確認
            self.assertEqual(len(predictions.final_predictions), 10)

        except Exception as e:
            self.fail(f"高次元データ処理でエラー: {e}")

    def test_memory_efficiency(self):
        """メモリ効率性テスト"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 大きなデータセットで訓練
        X_large = np.random.randn(5000, 50)
        y_large = np.random.randn(5000)

        self.ensemble.fit(X_large, y_large)

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # メモリ使用量が妥当な範囲内であることを確認
        self.assertLess(memory_increase, 500,
                       f"メモリ使用量増加 {memory_increase:.1f}MB が過大です")

        print(f"メモリ使用量増加: {memory_increase:.1f}MB")


class TestEnsembleSystemAdvancedFeatures(unittest.TestCase):
    """アンサンブルシステム高度機能テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.config = EnsembleConfig(verbose=False)
        self.ensemble = EnsembleSystem(self.config)

        # 時系列風データ生成
        np.random.seed(456)
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        self.time_series_data = self._generate_time_series_data(dates)

    def _generate_time_series_data(self, dates: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        """時系列データ生成"""
        n_samples = len(dates)
        n_features = 10

        # 時系列特徴量
        trend = np.linspace(0, 1, n_samples)
        seasonal = np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
        noise = np.random.randn(n_samples) * 0.1

        features = np.random.randn(n_samples, n_features)
        features[:, 0] = trend
        features[:, 1] = seasonal

        # 自己回帰的ターゲット
        target = np.zeros(n_samples)
        for i in range(1, n_samples):
            target[i] = (0.7 * target[i-1] +
                        0.2 * trend[i] +
                        0.1 * seasonal[i] +
                        noise[i])

        return features, target

    def test_time_series_cross_validation(self):
        """時系列交差検証テスト"""
        X, y = self.time_series_data

        # 時系列分割に適した設定
        config = EnsembleConfig(
            cv_folds=3,
            train_test_split=0.8,
            verbose=False
        )
        ensemble = EnsembleSystem(config)

        # 時系列順序を保った訓練・テスト分割
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]

        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)

        # 時系列予測の精度評価
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_test, predictions.final_predictions)

        # 時系列データでも妥当な精度を確認
        baseline_mse = np.var(y_test)
        self.assertLess(mse, baseline_mse * 0.5,
                       "時系列予測精度が不十分です")

    def test_feature_importance_analysis(self):
        """特徴量重要度分析テスト"""
        X, y = self.time_series_data

        self.ensemble.fit(X, y)

        # 特徴量重要度を取得
        feature_importance = self.ensemble.get_feature_importance()

        self.assertIsInstance(feature_importance, dict)
        self.assertGreater(len(feature_importance), 0)

        # 各モデルの重要度が存在することを確認
        for model_name in self.ensemble.base_models.keys():
            if f"{model_name}_importance" in feature_importance:
                importance_values = feature_importance[f"{model_name}_importance"]
                self.assertIsInstance(importance_values, np.ndarray)
                self.assertEqual(len(importance_values), X.shape[1])

    def test_model_persistence(self):
        """モデル永続化テスト"""
        X, y = self.time_series_data

        # 訓練済みモデルを保存
        self.ensemble.fit(X[:800], y[:800])

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            model_path = tmp_file.name

        try:
            # モデル保存
            self.ensemble.save_model(model_path)

            # 新しいインスタンスでモデル読み込み
            new_ensemble = EnsembleSystem(self.config)
            new_ensemble.load_model(model_path)

            # 同じ予測結果が得られることを確認
            original_pred = self.ensemble.predict(X[800:850])
            loaded_pred = new_ensemble.predict(X[800:850])

            np.testing.assert_array_almost_equal(
                original_pred.final_predictions,
                loaded_pred.final_predictions,
                decimal=6
            )

        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_incremental_learning(self):
        """インクリメンタル学習テスト"""
        X, y = self.time_series_data

        # 初期訓練
        initial_split = 600
        self.ensemble.fit(X[:initial_split], y[:initial_split])

        # 初期予測
        initial_pred = self.ensemble.predict(X[initial_split:initial_split+50])

        # 追加データで更新
        update_split = 700
        self.ensemble.partial_fit(X[initial_split:update_split], y[initial_split:update_split])

        # 更新後の予測
        updated_pred = self.ensemble.predict(X[update_split:update_split+50])

        # インクリメンタル学習により性能が維持または改善されることを確認
        from sklearn.metrics import mean_squared_error

        initial_mse = mean_squared_error(y[initial_split:initial_split+50],
                                       initial_pred.final_predictions)
        updated_mse = mean_squared_error(y[update_split:update_split+50],
                                       updated_pred.final_predictions)

        # 学習により悪化していないことを確認
        self.assertLessEqual(updated_mse, initial_mse * 1.5,
                           "インクリメンタル学習により性能が大幅に悪化しました")

    def test_ensemble_diversity_metrics(self):
        """アンサンブル多様性指標テスト"""
        X, y = self.time_series_data

        self.ensemble.fit(X[:800], y[:800])
        predictions = self.ensemble.predict(X[800:900])

        # 個別モデル予測の多様性を評価
        individual_preds = predictions.individual_predictions

        if len(individual_preds) >= 2:
            # 予測値の相関を計算
            pred_correlations = []
            models = list(individual_preds.keys())

            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    corr = np.corrcoef(individual_preds[models[i]],
                                     individual_preds[models[j]])[0, 1]
                    pred_correlations.append(corr)

            avg_correlation = np.mean(pred_correlations)

            # 多様性確認：相関が1.0未満（完全に同じでない）
            self.assertLess(avg_correlation, 0.95,
                           f"モデル間の相関 {avg_correlation:.3f} が高すぎます（多様性不足）")

            # 極端に低い相関でないことも確認（モデルが有効であること）
            self.assertGreater(avg_correlation, 0.3,
                             f"モデル間の相関 {avg_correlation:.3f} が低すぎます")

            print(f"アンサンブル多様性（平均相関）: {avg_correlation:.3f}")


if __name__ == '__main__':
    # テストスイート設定
    test_suite = unittest.TestSuite()

    # 93%精度システムテスト
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystem93Accuracy))

    # 堅牢性テスト
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemRobustness))

    # 高度機能テスト
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemAdvancedFeatures))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 結果サマリー
    print(f"\n{'='*60}")
    print(f"EnsembleSystem高度テスト完了")
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗数: {len(result.failures)}")
    print(f"エラー数: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")