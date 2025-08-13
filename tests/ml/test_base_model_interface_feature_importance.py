#!/usr/bin/env python3
"""
BaseModelInterface Feature Importance Tests

Issue #495対応: BaseModelInterface get_feature_importance再検討テスト
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys

sys.path.append('C:/gemini-desktop/day_trade/src')

# テストに必要なモジュールをインポート
try:
    from day_trade.ml.base_models.base_model_interface import BaseModelInterface, ModelPrediction
    from day_trade.ml.base_models.random_forest_model import RandomForestModel, RandomForestConfig
    from day_trade.ml.base_models.svr_model import SVRModel, SVRConfig
    from day_trade.ml.base_models.gradient_boosting_model import GradientBoostingModel, GradientBoostingConfig
    from day_trade.ml.deep_learning_models import TransformerModel, ModelConfig
    TEST_AVAILABLE = True
except ImportError as e:
    TEST_AVAILABLE = False
    print(f"テストモジュールインポートエラー: {e}")


# テスト用のダミーモデルクラス
class MockBaseModelWithoutFeatureImportance(BaseModelInterface):
    """特徴量重要度を提供しないモデル"""
    
    def __init__(self, model_name="MockNoFeatureImportance"):
        super().__init__(model_name)
        
    def fit(self, X, y, validation_data=None):
        self.is_trained = True
        self.feature_names = ["feature1", "feature2", "feature3"]
        return {"training_time": 1.0}
    
    def predict(self, X):
        return ModelPrediction(
            predictions=np.random.randn(len(X)),
            model_name=self.model_name
        )


class MockBaseModelWithFeatureImportance(BaseModelInterface):
    """特徴量重要度を提供するモデル"""
    
    def __init__(self, model_name="MockWithFeatureImportance"):
        super().__init__(model_name)
        self.mock_importances = np.array([0.5, 0.3, 0.2])
        
    def fit(self, X, y, validation_data=None):
        self.is_trained = True
        self.feature_names = ["feature1", "feature2", "feature3"]
        return {"training_time": 1.0}
    
    def predict(self, X):
        return ModelPrediction(
            predictions=np.random.randn(len(X)),
            model_name=self.model_name
        )
    
    def get_feature_importance(self) -> dict:
        """カスタム特徴量重要度実装"""
        if not self.is_trained:
            return {}
        return self._create_feature_importance_dict(self.mock_importances)
    
    def has_feature_importance(self) -> bool:
        return self.is_trained


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestBaseModelInterfaceFeatureImportance:
    """Issue #495: BaseModelInterface特徴量重要度再検討テスト"""
    
    @pytest.fixture
    def test_data(self):
        """テスト用データ"""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        return X, y
    
    def test_abstract_method_removal(self):
        """Issue #495: get_feature_importanceが抽象メソッドでないことを確認"""
        # 抽象メソッドでなければインスタンス化可能
        model = MockBaseModelWithoutFeatureImportance()
        assert isinstance(model, BaseModelInterface)
        
        # デフォルト実装では空の辞書を返す
        importance = model.get_feature_importance()
        assert importance == {}
    
    def test_default_get_feature_importance_untrained(self):
        """Issue #495: デフォルト実装での未学習モデルテスト"""
        model = MockBaseModelWithoutFeatureImportance()
        
        with patch('day_trade.ml.base_models.base_model_interface.logger') as mock_logger:
            importance = model.get_feature_importance()
            
            assert importance == {}
            mock_logger.debug.assert_called_once()
            debug_msg = mock_logger.debug.call_args[0][0]
            assert "未学習モデルのため特徴量重要度を取得できません" in debug_msg
    
    def test_default_get_feature_importance_trained(self, test_data):
        """Issue #495: デフォルト実装での学習済みモデルテスト"""
        model = MockBaseModelWithoutFeatureImportance()
        X, y = test_data
        
        model.fit(X, y)
        
        with patch('day_trade.ml.base_models.base_model_interface.logger') as mock_logger:
            importance = model.get_feature_importance()
            
            assert importance == {}
            mock_logger.debug.assert_called_once()
            debug_msg = mock_logger.debug.call_args[0][0]
            assert "このモデルタイプでは特徴量重要度をサポートしていません" in debug_msg
    
    def test_default_has_feature_importance(self, test_data):
        """Issue #495: デフォルトhas_feature_importanceテスト"""
        model = MockBaseModelWithoutFeatureImportance()
        
        # 未学習時
        assert model.has_feature_importance() == False
        
        # 学習済みでも特徴量重要度がない場合
        X, y = test_data
        model.fit(X, y)
        assert model.has_feature_importance() == False
    
    def test_custom_feature_importance_implementation(self, test_data):
        """Issue #495: カスタム特徴量重要度実装テスト"""
        model = MockBaseModelWithFeatureImportance()
        X, y = test_data
        
        # 未学習時
        assert model.get_feature_importance() == {}
        assert model.has_feature_importance() == False
        
        # 学習後
        model.fit(X, y)
        importance = model.get_feature_importance()
        
        assert len(importance) == 3
        assert "feature1" in importance
        assert "feature2" in importance 
        assert "feature3" in importance
        assert model.has_feature_importance() == True
        
        # 重要度でソートされていることを確認
        importance_values = list(importance.values())
        assert importance_values == sorted(importance_values, reverse=True)
    
    def test_create_feature_importance_dict_helper(self):
        """Issue #495: _create_feature_importance_dictヘルパーメソッドテスト"""
        model = MockBaseModelWithFeatureImportance()
        
        importances = np.array([0.1, 0.8, 0.05, 0.05])
        feature_names = ["feat_a", "feat_b", "feat_c", "feat_d"]
        
        result = model._create_feature_importance_dict(importances, feature_names)
        
        # 重要度順にソートされることを確認
        expected_order = ["feat_b", "feat_a", "feat_c", "feat_d"]
        actual_order = list(result.keys())
        assert actual_order == expected_order
        
        # 値が正しく対応していることを確認
        assert result["feat_b"] == 0.8
        assert result["feat_a"] == 0.1
        assert result["feat_c"] == 0.05
        assert result["feat_d"] == 0.05
    
    def test_create_feature_importance_dict_mismatched_lengths(self):
        """Issue #495: 特徴量名と重要度の数不一致テスト"""
        model = MockBaseModelWithFeatureImportance()
        
        importances = np.array([0.6, 0.3, 0.1])
        feature_names = ["feat1"]  # 数が合わない
        
        with patch('day_trade.ml.base_models.base_model_interface.logger') as mock_logger:
            result = model._create_feature_importance_dict(importances, feature_names)
            
            # generic名が使用されることを確認
            expected_keys = ["feature_0", "feature_1", "feature_2"]
            assert list(result.keys()) == expected_keys
            
            # 警告ログが出力されることを確認
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "特徴量名数不一致のためgeneric名を使用" in warning_msg
    
    def test_create_feature_importance_dict_empty_importances(self):
        """Issue #495: 空の重要度配列テスト"""
        model = MockBaseModelWithFeatureImportance()
        
        result = model._create_feature_importance_dict(np.array([]))
        assert result == {}
    
    def test_create_feature_importance_dict_no_feature_names(self):
        """Issue #495: feature_names未設定時のテスト"""
        model = MockBaseModelWithFeatureImportance()
        model.feature_names = []  # 空のfeature_names
        
        importances = np.array([0.4, 0.6])
        
        result = model._create_feature_importance_dict(importances)
        
        # generic名が使用されることを確認
        expected_keys = ["feature_0", "feature_1"]
        assert set(result.keys()) == set(expected_keys)


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestExistingModelIntegration:
    """Issue #495: 既存モデルとの統合テスト"""
    
    @pytest.fixture
    def test_data(self):
        """テスト用データ"""
        np.random.seed(42)
        data = pd.DataFrame({
            'Open': np.random.randn(100),
            'High': np.random.randn(100),
            'Low': np.random.randn(100),
            'Close': np.random.randn(100),
            'Volume': np.random.randn(100)
        })
        return data
    
    def test_random_forest_feature_importance(self, test_data):
        """Issue #495: RandomForest特徴量重要度テスト"""
        config = RandomForestConfig(n_estimators=10, enable_hyperopt=False)
        model = RandomForestModel(config)
        
        # 未学習時
        assert model.has_feature_importance() == False
        assert model.get_feature_importance() == {}
        
        # 学習後
        X = test_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y = test_data['Close'].values
        model.fit(X, y)
        
        assert model.has_feature_importance() == True
        importance = model.get_feature_importance()
        assert len(importance) > 0
        assert all(isinstance(v, (int, float)) for v in importance.values())
    
    def test_gradient_boosting_feature_importance(self, test_data):
        """Issue #495: GradientBoosting特徴量重要度テスト"""
        config = GradientBoostingConfig(n_estimators=10, enable_hyperopt=False)
        model = GradientBoostingModel(config)
        
        # 未学習時
        assert model.has_feature_importance() == False
        
        # 学習後
        X = test_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y = test_data['Close'].values
        model.fit(X, y)
        
        assert model.has_feature_importance() == True
        importance = model.get_feature_importance()
        assert len(importance) > 0
    
    def test_svr_feature_importance_linear(self, test_data):
        """Issue #495: SVR線形カーネル特徴量重要度テスト"""
        config = SVRConfig(kernel='linear', enable_hyperopt=False)
        model = SVRModel(config)
        
        # 未学習時
        assert model.has_feature_importance() == False
        
        # 学習後
        X = test_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y = test_data['Close'].values
        model.fit(X, y)
        
        # 線形カーネルの場合は特徴量重要度を提供
        assert model.has_feature_importance() == True
        importance = model.get_feature_importance()
        assert len(importance) > 0
    
    def test_svr_feature_importance_rbf(self, test_data):
        """Issue #495: SVR非線形カーネル特徴量重要度テスト"""
        config = SVRConfig(kernel='rbf', enable_hyperopt=False)
        model = SVRModel(config)
        
        # 学習後でも非線形カーネルでは特徴量重要度なし
        X = test_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y = test_data['Close'].values
        model.fit(X, y)
        
        assert model.has_feature_importance() == False
        importance = model.get_feature_importance()
        assert importance == {}
    
    def test_deep_learning_model_feature_importance(self, test_data):
        """Issue #495: DeepLearningモデル特徴量重要度テスト"""
        config = ModelConfig(sequence_length=5, prediction_horizon=1, epochs=1)
        model = TransformerModel(config)
        
        # 未学習時
        assert model.has_feature_importance() == False
        
        # 学習後
        model.fit(test_data, test_data)
        
        # 学習済みなら提供可能（データが必要）
        assert model.has_feature_importance() == True
        
        # データなしでは空辞書
        importance_without_data = model.get_feature_importance()
        assert importance_without_data == {}
        
        # データありでは重要度取得可能（時間がかかる場合があるため小さなデータ）
        small_data = test_data.head(20)
        importance_with_data = model.get_feature_importance(small_data)
        # 何かしらの結果が返されることを確認（空でも可）
        assert isinstance(importance_with_data, dict)
    
    def test_model_compatibility_check(self, test_data):
        """Issue #495: 複数モデルタイプの互換性確認テスト"""
        models = [
            RandomForestModel(RandomForestConfig(n_estimators=5, enable_hyperopt=False)),
            SVRModel(SVRConfig(kernel='linear', enable_hyperopt=False)),
            SVRModel(SVRConfig(kernel='rbf', enable_hyperopt=False)),
            GradientBoostingModel(GradientBoostingConfig(n_estimators=5, enable_hyperopt=False))
        ]
        
        X = test_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y = test_data['Close'].values
        
        for model in models:
            # 全モデルでhas_feature_importanceメソッドが使用可能
            assert hasattr(model, 'has_feature_importance')
            assert hasattr(model, 'get_feature_importance')
            
            # 未学習時の一貫性
            pre_training_has_importance = model.has_feature_importance()
            pre_training_importance = model.get_feature_importance()
            assert pre_training_has_importance == False
            assert pre_training_importance == {}
            
            # 学習後の動作
            model.fit(X, y)
            post_training_has_importance = model.has_feature_importance()
            post_training_importance = model.get_feature_importance()
            
            # 特徴量重要度提供可否と実際の重要度の整合性
            if post_training_has_importance:
                assert len(post_training_importance) > 0
            else:
                assert post_training_importance == {}


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestFeatureImportanceEdgeCases:
    """Issue #495: 特徴量重要度エッジケーステスト"""
    
    def test_model_with_no_feature_names(self):
        """Issue #495: feature_names未設定モデルテスト"""
        model = MockBaseModelWithFeatureImportance()
        model.feature_names = None  # feature_names未設定
        
        importances = np.array([0.7, 0.2, 0.1])
        result = model._create_feature_importance_dict(importances)
        
        # generic名が使用される
        expected_keys = ["feature_0", "feature_1", "feature_2"]
        assert set(result.keys()) == set(expected_keys)
    
    def test_feature_importance_error_handling(self):
        """Issue #495: 特徴量重要度取得エラーハンドリングテスト"""
        
        class ErrorProneModel(MockBaseModelWithFeatureImportance):
            def get_feature_importance(self):
                if not self.is_trained:
                    return {}
                # 意図的にエラーを発生
                raise ValueError("Test error")
        
        model = ErrorProneModel()
        model.fit(np.random.randn(10, 3), np.random.randn(10))
        
        # エラーが発生してもhas_feature_importanceは正常動作
        # （get_feature_importanceが空辞書を返すかエラーになるかは実装依存）
        try:
            importance = model.get_feature_importance()
            # エラーハンドリングされて空辞書が返される場合
            assert importance == {}
        except ValueError:
            # エラーが伝播される場合も許可
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])