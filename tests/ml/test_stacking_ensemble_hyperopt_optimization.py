#!/usr/bin/env python3
"""
StackingEnsemble Hyperparameter Optimization Tests

Issue #692対応: StackingEnsembleハイパーパラメータ最適化効率化テスト
"""

import pytest
import numpy as np
import pandas as pd
import time
from unittest.mock import patch, MagicMock
import sys

sys.path.append('C:/gemini-desktop/day_trade/src')

# テストに必要なモジュールをインポート
try:
    from day_trade.ml.stacking_ensemble import (
        StackingEnsemble,
        StackingConfig
    )
    TEST_AVAILABLE = True
except ImportError as e:
    TEST_AVAILABLE = False
    print(f"テストモジュールインポートエラー: {e}")


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestStackingEnsembleHyperoptOptimization:
    """Issue #692: StackingEnsembleハイパーパラメータ最適化効率化テスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return StackingConfig(
            meta_learner_type="ridge",
            cv_folds=3,
            hyperopt_method="optuna",
            hyperopt_n_trials=10,
            hyperopt_timeout=30,
            hyperopt_early_stopping=True
        )
    
    @pytest.fixture
    def test_data(self):
        """テスト用データ"""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = pd.Series(np.random.randn(n_samples))
        return X, y
    
    @pytest.fixture
    def stacking_ensemble(self, config):
        """テスト用StackingEnsemble"""
        return StackingEnsemble(config)
    
    def test_optuna_import_detection(self):
        """Issue #692: Optuna可用性検出テスト"""
        # Optuna可用性の確認
        try:
            import optuna
            optuna_available = True
        except ImportError:
            optuna_available = False
        
        # モジュール内でのOPTUNA_AVAILABLEフラグとの一致確認
        from day_trade.ml.stacking_ensemble import OPTUNA_AVAILABLE
        assert OPTUNA_AVAILABLE == optuna_available
    
    def test_config_hyperopt_parameters(self, config):
        """Issue #692: ハイパーパラメータ最適化設定テスト"""
        assert config.hyperopt_method == "optuna"
        assert config.hyperopt_n_trials == 10
        assert config.hyperopt_timeout == 30
        assert config.hyperopt_early_stopping == True
    
    @patch('day_trade.ml.stacking_ensemble.OPTUNA_AVAILABLE', True)
    def test_optimize_meta_learner_hyperparams_optuna_ridge(self, stacking_ensemble, test_data):
        """Issue #692: Ridge回帰のOptuna最適化テスト"""
        X, y = test_data
        base_predictions = np.random.randn(len(X), 3)  # 3つのベースモデル予測
        
        with patch('day_trade.ml.stacking_ensemble.optuna') as mock_optuna:
            # モックStudy作成
            mock_study = MagicMock()
            mock_study.best_params = {'alpha': 0.5}
            mock_optuna.create_study.return_value = mock_study
            
            # 最適化実行
            best_params = stacking_ensemble._optimize_meta_learner_hyperparams(
                base_predictions, y.values
            )
            
            # Optunaが呼ばれることを確認
            mock_optuna.create_study.assert_called_once()
            mock_study.optimize.assert_called_once()
            
            # 最適パラメータが返されることを確認
            assert 'alpha' in best_params
            assert best_params['alpha'] == 0.5
    
    @patch('day_trade.ml.stacking_ensemble.OPTUNA_AVAILABLE', True)
    def test_optimize_meta_learner_hyperparams_optuna_lasso(self, test_data):
        """Issue #692: Lasso回帰のOptuna最適化テスト"""
        config = StackingConfig(
            meta_learner_type="lasso",
            hyperopt_method="optuna",
            hyperopt_n_trials=5
        )
        stacking_ensemble = StackingEnsemble(config)
        
        X, y = test_data
        base_predictions = np.random.randn(len(X), 3)
        
        with patch('day_trade.ml.stacking_ensemble.optuna') as mock_optuna:
            mock_study = MagicMock()
            mock_study.best_params = {'alpha': 0.01, 'max_iter': 2000}
            mock_optuna.create_study.return_value = mock_study
            
            best_params = stacking_ensemble._optimize_meta_learner_hyperparams(
                base_predictions, y.values
            )
            
            assert 'alpha' in best_params
            assert 'max_iter' in best_params
    
    @patch('day_trade.ml.stacking_ensemble.OPTUNA_AVAILABLE', True)
    def test_optimize_meta_learner_hyperparams_optuna_elasticnet(self, test_data):
        """Issue #692: ElasticNet回帰のOptuna最適化テスト"""
        config = StackingConfig(
            meta_learner_type="elastic_net",
            hyperopt_method="optuna",
            hyperopt_n_trials=5
        )
        stacking_ensemble = StackingEnsemble(config)
        
        X, y = test_data
        base_predictions = np.random.randn(len(X), 3)
        
        with patch('day_trade.ml.stacking_ensemble.optuna') as mock_optuna:
            mock_study = MagicMock()
            mock_study.best_params = {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 1500}
            mock_optuna.create_study.return_value = mock_study
            
            best_params = stacking_ensemble._optimize_meta_learner_hyperparams(
                base_predictions, y.values
            )
            
            assert 'alpha' in best_params
            assert 'l1_ratio' in best_params
            assert 'max_iter' in best_params
    
    @patch('day_trade.ml.stacking_ensemble.OPTUNA_AVAILABLE', True)
    def test_optimize_meta_learner_hyperparams_optuna_random_forest(self, test_data):
        """Issue #692: RandomForest回帰のOptuna最適化テスト"""
        config = StackingConfig(
            meta_learner_type="random_forest",
            hyperopt_method="optuna",
            hyperopt_n_trials=8
        )
        stacking_ensemble = StackingEnsemble(config)
        
        X, y = test_data
        base_predictions = np.random.randn(len(X), 3)
        
        with patch('day_trade.ml.stacking_ensemble.optuna') as mock_optuna:
            mock_study = MagicMock()
            mock_study.best_params = {
                'n_estimators': 150,
                'max_depth': 8,
                'min_samples_split': 4,
                'min_samples_leaf': 2
            }
            mock_optuna.create_study.return_value = mock_study
            
            best_params = stacking_ensemble._optimize_meta_learner_hyperparams(
                base_predictions, y.values
            )
            
            assert 'n_estimators' in best_params
            assert 'max_depth' in best_params
            assert 'min_samples_split' in best_params
            assert 'min_samples_leaf' in best_params
    
    @patch('day_trade.ml.stacking_ensemble.OPTUNA_AVAILABLE', True)
    def test_optimize_meta_learner_hyperparams_optuna_xgboost(self, test_data):
        """Issue #692: XGBoost回帰のOptuna最適化テスト"""
        config = StackingConfig(
            meta_learner_type="xgboost",
            hyperopt_method="optuna",
            hyperopt_n_trials=6
        )
        stacking_ensemble = StackingEnsemble(config)
        
        X, y = test_data
        base_predictions = np.random.randn(len(X), 3)
        
        with patch('day_trade.ml.stacking_ensemble.optuna') as mock_optuna:
            mock_study = MagicMock()
            mock_study.best_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            mock_optuna.create_study.return_value = mock_study
            
            best_params = stacking_ensemble._optimize_meta_learner_hyperparams(
                base_predictions, y.values
            )
            
            assert 'n_estimators' in best_params
            assert 'max_depth' in best_params
            assert 'learning_rate' in best_params
            assert 'subsample' in best_params
            assert 'colsample_bytree' in best_params
    
    @patch('day_trade.ml.stacking_ensemble.OPTUNA_AVAILABLE', True)
    def test_optimize_meta_learner_hyperparams_optuna_mlp(self, test_data):
        """Issue #692: MLP回帰のOptuna最適化テスト"""
        config = StackingConfig(
            meta_learner_type="mlp",
            hyperopt_method="optuna",
            hyperopt_n_trials=7
        )
        stacking_ensemble = StackingEnsemble(config)
        
        X, y = test_data
        base_predictions = np.random.randn(len(X), 3)
        
        with patch('day_trade.ml.stacking_ensemble.optuna') as mock_optuna:
            mock_study = MagicMock()
            mock_study.best_params = {
                'hidden_layer_sizes': (64, 32),
                'alpha': 0.001,
                'learning_rate_init': 0.01,
                'max_iter': 300
            }
            mock_optuna.create_study.return_value = mock_study
            
            best_params = stacking_ensemble._optimize_meta_learner_hyperparams(
                base_predictions, y.values
            )
            
            assert 'hidden_layer_sizes' in best_params
            assert 'alpha' in best_params
            assert 'learning_rate_init' in best_params
            assert 'max_iter' in best_params
    
    @patch('day_trade.ml.stacking_ensemble.OPTUNA_AVAILABLE', False)
    def test_optimize_meta_learner_hyperparams_grid_search_fallback(self, stacking_ensemble, test_data):
        """Issue #692: GridSearchCVフォールバック最適化テスト"""
        X, y = test_data
        base_predictions = np.random.randn(len(X), 3)
        
        with patch('day_trade.ml.stacking_ensemble.logger') as mock_logger:
            best_params = stacking_ensemble._optimize_meta_learner_hyperparams(
                base_predictions, y.values
            )
            
            # GridSearchCV利用警告が出力されることを確認
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "Optuna未利用" in warning_msg
            assert "GridSearchCV" in warning_msg
            
            # 最適パラメータが返される
            assert isinstance(best_params, dict)
    
    def test_grid_search_method_configuration(self, test_data):
        """Issue #692: grid_search方法明示設定テスト"""
        config = StackingConfig(
            meta_learner_type="ridge",
            hyperopt_method="grid_search",
            cv_folds=3
        )
        stacking_ensemble = StackingEnsemble(config)
        
        X, y = test_data
        base_predictions = np.random.randn(len(X), 3)
        
        with patch('day_trade.ml.stacking_ensemble.logger') as mock_logger:
            best_params = stacking_ensemble._optimize_meta_learner_hyperparams(
                base_predictions, y.values
            )
            
            # GridSearchCV使用ログが出力されることを確認
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("GridSearchCV" in msg for msg in info_calls)
            
            # パラメータが返される
            assert isinstance(best_params, dict)
    
    def test_timeout_configuration(self, test_data):
        """Issue #692: タイムアウト設定テスト"""
        config = StackingConfig(
            meta_learner_type="ridge",
            hyperopt_method="optuna",
            hyperopt_n_trials=100,
            hyperopt_timeout=1  # 1秒タイムアウト
        )
        stacking_ensemble = StackingEnsemble(config)
        
        X, y = test_data
        base_predictions = np.random.randn(len(X), 3)
        
        with patch('day_trade.ml.stacking_ensemble.OPTUNA_AVAILABLE', True):
            with patch('day_trade.ml.stacking_ensemble.optuna') as mock_optuna:
                mock_study = MagicMock()
                mock_study.best_params = {'alpha': 1.0}
                mock_optuna.create_study.return_value = mock_study
                
                start_time = time.time()
                best_params = stacking_ensemble._optimize_meta_learner_hyperparams(
                    base_predictions, y.values
                )
                elapsed_time = time.time() - start_time
                
                # タイムアウト引数が正しく設定されることを確認
                mock_study.optimize.assert_called_once()
                call_args = mock_study.optimize.call_args
                assert 'timeout' in call_args[1]
                assert call_args[1]['timeout'] == 1
    
    def test_early_stopping_configuration(self, test_data):
        """Issue #692: 早期停止設定テスト"""
        config = StackingConfig(
            meta_learner_type="ridge",
            hyperopt_method="optuna",
            hyperopt_n_trials=50,
            hyperopt_early_stopping=True
        )
        stacking_ensemble = StackingEnsemble(config)
        
        X, y = test_data
        base_predictions = np.random.randn(len(X), 3)
        
        with patch('day_trade.ml.stacking_ensemble.OPTUNA_AVAILABLE', True):
            with patch('day_trade.ml.stacking_ensemble.optuna') as mock_optuna:
                mock_study = MagicMock()
                mock_study.best_params = {'alpha': 0.5}
                mock_optuna.create_study.return_value = mock_study
                
                # コールバック作成のモック
                mock_callback = MagicMock()
                mock_optuna.study.MaxTrialsCallback.return_value = mock_callback
                
                best_params = stacking_ensemble._optimize_meta_learner_hyperparams(
                    base_predictions, y.values
                )
                
                # 早期停止コールバックが設定されることを確認
                mock_study.optimize.assert_called_once()
                call_args = mock_study.optimize.call_args
                assert 'callbacks' in call_args[1]
    
    def test_performance_comparison_optuna_vs_gridsearch(self, test_data):
        """Issue #692: Optuna vs GridSearchCV性能比較テスト"""
        X, y = test_data
        base_predictions = np.random.randn(len(X), 4)  # 4つのベースモデル予測
        
        # GridSearchCV設定
        grid_config = StackingConfig(
            meta_learner_type="ridge",
            hyperopt_method="grid_search",
            cv_folds=3
        )
        grid_ensemble = StackingEnsemble(grid_config)
        
        # Optuna設定
        optuna_config = StackingConfig(
            meta_learner_type="ridge",
            hyperopt_method="optuna",
            hyperopt_n_trials=15,
            hyperopt_timeout=10
        )
        optuna_ensemble = StackingEnsemble(optuna_config)
        
        # GridSearchCV実行時間測定
        start_time = time.time()
        grid_params = grid_ensemble._optimize_meta_learner_hyperparams(
            base_predictions, y.values
        )
        grid_time = time.time() - start_time
        
        # Optuna実行時間測定（モック使用）
        with patch('day_trade.ml.stacking_ensemble.OPTUNA_AVAILABLE', True):
            with patch('day_trade.ml.stacking_ensemble.optuna') as mock_optuna:
                mock_study = MagicMock()
                mock_study.best_params = {'alpha': 0.8}
                mock_optuna.create_study.return_value = mock_study
                
                start_time = time.time()
                optuna_params = optuna_ensemble._optimize_meta_learner_hyperparams(
                    base_predictions, y.values
                )
                optuna_time = time.time() - start_time
        
        # 両方のパラメータが取得されることを確認
        assert isinstance(grid_params, dict)
        assert isinstance(optuna_params, dict)
        assert 'alpha' in grid_params
        assert 'alpha' in optuna_params
        
        print(f"性能比較: GridSearch {grid_time:.3f}s, Optuna {optuna_time:.3f}s")
    
    def test_invalid_hyperopt_method_fallback(self, stacking_ensemble, test_data):
        """Issue #692: 無効なハイパーパラメータ最適化方法のフォールバックテスト"""
        # 無効な方法を設定
        stacking_ensemble.config.hyperopt_method = "invalid_method"
        
        X, y = test_data
        base_predictions = np.random.randn(len(X), 3)
        
        with patch('day_trade.ml.stacking_ensemble.logger') as mock_logger:
            best_params = stacking_ensemble._optimize_meta_learner_hyperparams(
                base_predictions, y.values
            )
            
            # 無効方法への警告が出力されることを確認
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "無効なhyperopt_method" in warning_msg
            assert "GridSearchCV" in warning_msg
            
            # GridSearchCVにフォールバック
            assert isinstance(best_params, dict)


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestStackingEnsembleBayesianOptimization:
    """Issue #692: ベイズ最適化詳細テスト"""
    
    def test_optuna_trial_parameter_sampling(self):
        """Issue #692: Optunaトライアルパラメータサンプリングテスト"""
        # モックトライアルでパラメータサンプリングをテスト
        from unittest.mock import Mock
        
        mock_trial = Mock()
        mock_trial.suggest_float.side_effect = lambda name, low, high: (low + high) / 2
        mock_trial.suggest_int.side_effect = lambda name, low, high: (low + high) // 2
        mock_trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        
        # Ridge用パラメータサンプリング例
        alpha = mock_trial.suggest_float('alpha', 0.01, 10.0, log=True)
        assert alpha == (0.01 + 10.0) / 2
        
        # RandomForest用パラメータサンプリング例
        n_estimators = mock_trial.suggest_int('n_estimators', 50, 300)
        max_depth = mock_trial.suggest_int('max_depth', 3, 20)
        
        assert n_estimators == (50 + 300) // 2
        assert max_depth == (3 + 20) // 2
    
    def test_optuna_study_configuration(self):
        """Issue #692: Optunaスタディ設定テスト"""
        config = StackingConfig(
            meta_learner_type="ridge",
            hyperopt_method="optuna",
            hyperopt_n_trials=20
        )
        stacking_ensemble = StackingEnsemble(config)
        
        with patch('day_trade.ml.stacking_ensemble.OPTUNA_AVAILABLE', True):
            with patch('day_trade.ml.stacking_ensemble.optuna') as mock_optuna:
                mock_study = MagicMock()
                mock_study.best_params = {'alpha': 1.0}
                mock_optuna.create_study.return_value = mock_study
                
                # TPESampler設定確認
                mock_sampler = MagicMock()
                mock_optuna.samplers.TPESampler.return_value = mock_sampler
                
                base_predictions = np.random.randn(50, 3)
                y = np.random.randn(50)
                
                stacking_ensemble._optimize_meta_learner_hyperparams(base_predictions, y)
                
                # create_studyが適切なサンプラーで呼ばれることを確認
                mock_optuna.create_study.assert_called_once()
                call_kwargs = mock_optuna.create_study.call_args[1]
                assert 'sampler' in call_kwargs
                assert 'direction' in call_kwargs
                assert call_kwargs['direction'] == 'minimize'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])