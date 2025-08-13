#!/usr/bin/env python3
"""
Model Copy Functionality Tests

Issue #483対応: _copy_modelの堅牢性改善テスト
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

import sys
sys.path.append('C:/gemini-desktop/day_trade/src')

from day_trade.ml.stacking_ensemble import StackingEnsemble, StackingConfig
from day_trade.ml.base_models.base_model_interface import BaseModelInterface, ModelPrediction


class MockModelWithCopy(BaseModelInterface):
    """copyメソッド付きモックモデル"""
    
    def __init__(self, model_name: str, config=None):
        super().__init__(model_name, config)
        self.mock_predictions = None
        self.copy_called = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data=None):
        self.is_trained = True
        self.model = "MockModel"
        return {'training_time': 1.0}
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        predictions = X[:, 0] + 0.1 * np.random.randn(len(X))
        return ModelPrediction(
            predictions=predictions,
            confidence=np.ones(len(X)) * 0.8,
            model_name=self.model_name
        )
    
    def get_feature_importance(self):
        return {}
    
    def copy(self):
        """Issue #483: カスタムコピーメソッド"""
        self.copy_called = True
        new_model = MockModelWithCopy(f"{self.model_name}_copy", self.config)
        new_model.feature_names = self.feature_names.copy()
        return new_model


class MockSklearnModel(BaseModelInterface):
    """scikit-learn互換モデル"""
    
    def __init__(self, model_name: str, config=None):
        super().__init__(model_name, config)
        # scikit-learn風のモックモデル
        self.sklearn_model = Mock()
        self.sklearn_model.get_params.return_value = {'param1': 1, 'param2': 'test'}
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data=None):
        self.is_trained = True
        self.model = self.sklearn_model  # sklearn互換モデル設定
        return {'training_time': 1.0}
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        predictions = X[:, 0] + 0.1 * np.random.randn(len(X))
        return ModelPrediction(
            predictions=predictions,
            confidence=np.ones(len(X)) * 0.8,
            model_name=self.model_name
        )
    
    def get_feature_importance(self):
        return {}


class TestModelCopy:
    """Issue #483: モデルコピー機能テスト"""
    
    @pytest.fixture
    def sample_data(self):
        """テスト用データ"""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.sum(X[:, :2], axis=1) + 0.1 * np.random.randn(50)
        return X, y
    
    def test_base_model_copy_method(self):
        """Issue #483: BaseModelInterfaceのcopyメソッドテスト"""
        # ベースモデル作成
        original_model = MockModelWithCopy("original_model", {'param': 'value'})
        original_model.feature_names = ['f1', 'f2', 'f3']
        original_model.is_trained = True
        original_model.model = "trained_model"
        original_model.training_metrics = {'score': 0.95}
        
        # コピー作成
        copied_model = original_model.copy()
        
        # コピーの妥当性確認
        assert copied_model is not original_model  # 異なるインスタンス
        assert copied_model.model_name == "original_model_copy"
        assert copied_model.feature_names == ['f1', 'f2', 'f3']
        assert not copied_model.is_trained  # 未学習状態
        assert copied_model.model is None  # モデル実体クリア
        assert copied_model.training_metrics == {}  # メトリクスクリア
        assert original_model.copy_called  # カスタムコピーメソッド呼び出し確認
    
    def test_stacking_copy_model_with_custom_copy(self):
        """Issue #483: カスタムcopyメソッド使用テスト"""
        base_models = {
            "custom_model": MockModelWithCopy("custom_model", {'test': True})
        }
        config = StackingConfig(meta_learner_type='linear', enable_hyperopt=False)
        ensemble = StackingEnsemble(base_models, config)
        
        original_model = base_models["custom_model"]
        original_model.feature_names = ['feature_0', 'feature_1']
        
        # _copy_modelメソッドテスト
        copied_model = ensemble._copy_model(original_model)
        
        # カスタムcopyメソッドが呼ばれたことを確認
        assert original_model.copy_called
        assert copied_model.model_name == "custom_model_copy"
        assert copied_model.feature_names == ['feature_0', 'feature_1']
    
    @patch('sklearn.base.clone')
    def test_stacking_copy_model_with_sklearn_clone(self, mock_clone):
        """Issue #483: sklearn.base.clone使用テスト"""
        # モックのsklearnモデル
        sklearn_model = MockSklearnModel("sklearn_model", {'alpha': 1.0})
        sklearn_model.feature_names = ['x1', 'x2', 'x3']
        sklearn_model.is_trained = True
        sklearn_model.training_metrics = {'rmse': 0.1}
        
        # clone設定
        mock_clone.return_value = Mock()
        mock_clone.return_value.get_params.return_value = {'param1': 1}
        
        base_models = {"sklearn_model": sklearn_model}
        config = StackingConfig(meta_learner_type='linear', enable_hyperopt=False)
        ensemble = StackingEnsemble(base_models, config)
        
        # _copy_modelメソッドテスト
        copied_model = ensemble._copy_model(sklearn_model)
        
        # sklearn.base.cloneが呼ばれたことを確認
        mock_clone.assert_called_once()
        assert copied_model is not sklearn_model
        assert copied_model.feature_names == ['x1', 'x2', 'x3']
        assert not copied_model.is_trained  # 未学習状態
        assert copied_model.training_metrics == {}  # メトリクスクリア
    
    def test_stacking_copy_model_fallback(self):
        """Issue #483: フォールバック基本コピーテスト"""
        # copyメソッドなし、sklearn互換でないモデル
        class BasicModel(BaseModelInterface):
            def fit(self, X, y, validation_data=None):
                self.is_trained = True
                self.model = "basic_model"
                return {'time': 1.0}
            
            def predict(self, X) -> ModelPrediction:
                return ModelPrediction(
                    predictions=np.ones(len(X)),
                    model_name=self.model_name
                )
            
            def get_feature_importance(self):
                return {}
        
        basic_model = BasicModel("basic", {'config_val': 42})
        basic_model.feature_names = ['a', 'b']
        basic_model.is_trained = True
        basic_model.training_metrics = {'accuracy': 0.9}
        
        base_models = {"basic": basic_model}
        config = StackingConfig(meta_learner_type='linear', enable_hyperopt=False)
        ensemble = StackingEnsemble(base_models, config)
        
        # フォールバック基本コピーテスト
        copied_model = ensemble._copy_model(basic_model)
        
        assert copied_model is not basic_model
        assert copied_model.model_name == "basic"
        assert copied_model.feature_names == ['a', 'b']
        assert not copied_model.is_trained  # 未学習状態
        assert copied_model.training_metrics == {}  # メトリクスクリア
        assert copied_model.config == {'config_val': 42}  # 設定コピー
    
    def test_copy_model_error_handling(self):
        """Issue #483: コピーエラーハンドリングテスト"""
        # エラーを発生させるモデル
        class ErrorModel(BaseModelInterface):
            def __init__(self, *args, **kwargs):
                raise ValueError("Intentional error")
            
            def fit(self, X, y, validation_data=None):
                pass
            
            def predict(self, X):
                pass
            
            def get_feature_importance(self):
                return {}
        
        # 正常に作成されたモデルだが、コピー時にエラー
        error_model = Mock(spec=BaseModelInterface)
        error_model.model_name = "error_model"
        error_model.__class__ = ErrorModel  # エラーを発生させるクラス
        error_model.config = {}
        
        base_models = {"error_model": error_model}
        config = StackingConfig(meta_learner_type='linear', enable_hyperopt=False)
        ensemble = StackingEnsemble(base_models, config)
        
        # エラーハンドリング確認
        with pytest.raises(ValueError, match="モデル error_model のコピーに失敗しました"):
            ensemble._copy_model(error_model)
    
    def test_copy_preserves_config_deep(self):
        """Issue #483: 設定の深いコピーテスト"""
        # ネストした設定を持つモデル
        nested_config = {
            'level1': {
                'level2': {
                    'param': [1, 2, 3],
                    'dict_param': {'a': 1, 'b': 2}
                }
            },
            'simple_param': 'value'
        }
        
        original_model = MockModelWithCopy("nested_config", nested_config)
        
        base_models = {"nested": original_model}
        config = StackingConfig(meta_learner_type='linear', enable_hyperopt=False)
        ensemble = StackingEnsemble(base_models, config)
        
        # コピー作成
        copied_model = ensemble._copy_model(original_model)
        
        # 深いコピーの確認
        assert copied_model.config is not original_model.config
        assert copied_model.config['level1'] is not original_model.config['level1']
        assert copied_model.config['level1']['level2'] is not original_model.config['level1']['level2']
        assert copied_model.config['level1']['level2']['param'] is not original_model.config['level1']['level2']['param']
        
        # 値の確認
        assert copied_model.config == nested_config
        
        # 変更の独立性確認
        copied_model.config['level1']['level2']['param'][0] = 999
        assert original_model.config['level1']['level2']['param'][0] == 1  # 元の値は変更されない
    
    @patch('sklearn.base.clone', side_effect=ImportError("sklearn not available"))
    def test_copy_sklearn_import_error(self, mock_clone):
        """Issue #483: sklearn未使用環境でのフォールバックテスト"""
        sklearn_model = MockSklearnModel("sklearn_model")
        sklearn_model.feature_names = ['x1', 'x2']
        
        base_models = {"sklearn_model": sklearn_model}
        config = StackingConfig(meta_learner_type='linear', enable_hyperopt=False)
        ensemble = StackingEnsemble(base_models, config)
        
        # ImportErrorが発生してもフォールバック処理される
        copied_model = ensemble._copy_model(sklearn_model)
        
        assert copied_model is not sklearn_model
        assert copied_model.feature_names == ['x1', 'x2']
        assert not copied_model.is_trained
    
    def test_integration_copy_in_cv_training(self, sample_data):
        """Issue #483: CV学習でのコピー統合テスト"""
        X, y = sample_data
        
        # copyメソッドを持つベースモデル
        base_models = {
            "copy_model1": MockModelWithCopy("copy_model1", {'param1': 1}),
            "copy_model2": MockModelWithCopy("copy_model2", {'param2': 2})
        }
        
        config = StackingConfig(
            meta_learner_type='linear',
            cv_folds=3,
            enable_hyperopt=False,
            verbose=False
        )
        
        ensemble = StackingEnsemble(base_models, config)
        
        # CV学習実行（内部でモデルコピーが使用される）
        results = ensemble.fit(X, y)
        
        # 学習が正常完了したことを確認
        assert ensemble.is_fitted
        assert 'training_time' in results
        
        # カスタムcopyメソッドが呼ばれたことを確認
        assert base_models["copy_model1"].copy_called
        assert base_models["copy_model2"].copy_called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])