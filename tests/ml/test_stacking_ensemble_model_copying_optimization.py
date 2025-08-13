#!/usr/bin/env python3
"""
StackingEnsemble Model Copying Optimization Tests

Issue #691対応: StackingEnsembleベースモデルコピー最適化テスト
"""

import pytest
import numpy as np
import pandas as pd
import time
from unittest.mock import patch, MagicMock, Mock
import sys

sys.path.append('C:/gemini-desktop/day_trade/src')

# テストに必要なモジュールをインポート
try:
    from day_trade.ml.stacking_ensemble import (
        StackingEnsemble,
        StackingConfig
    )
    from day_trade.ml.base_models.base_model_interface import BaseModelInterface, ModelPrediction
    TEST_AVAILABLE = True
except ImportError as e:
    TEST_AVAILABLE = False
    print(f"テストモジュールインポートエラー: {e}")


# テスト用のダミーモデルクラス群
class MockBaseModelInterface(BaseModelInterface):
    """テスト用BaseModelInterface実装"""

    def __init__(self, model_name="MockModel", config=None):
        super().__init__(model_name, config)
        self.model = MockSklearnModel()  # sklearn互換モデル
        self.feature_names = ["feature1", "feature2", "feature3"]
        self.training_metrics = {"mse": 0.5, "mae": 0.3}

    def fit(self, X, y, validation_data=None):
        self.is_trained = True
        return {"training_time": 1.0}

    def predict(self, X):
        return ModelPrediction(
            predictions=np.random.randn(len(X)),
            model_name=self.model_name
        )

    def get_feature_importance(self):
        return {"feature1": 0.4, "feature2": 0.3, "feature3": 0.3}


class MockSklearnModel:
    """sklearn互換モックモデル"""

    def __init__(self, param1=1.0, param2="test"):
        self.param1 = param1
        self.param2 = param2

    def get_params(self, deep=True):
        return {"param1": self.param1, "param2": self.param2}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.random.randn(len(X))


class MockSklearnCompatibleBaseModel(BaseModelInterface):
    """sklearn互換のBaseModelInterface（get_params/set_params実装）"""

    def __init__(self, model_name="SklearnCompatibleModel", config=None):
        super().__init__(model_name, config)
        self.param1 = 1.0
        self.param2 = "default"
        self.model = MockSklearnModel()

    def get_params(self, deep=True):
        return {"param1": self.param1, "param2": self.param2, "model_name": self.model_name}

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def fit(self, X, y, validation_data=None):
        self.is_trained = True
        return {"training_time": 2.0}

    def predict(self, X):
        return ModelPrediction(
            predictions=np.random.randn(len(X)),
            model_name=self.model_name
        )

    def get_feature_importance(self):
        return {"feature1": 0.5, "feature2": 0.5}


class MockNonSklearnModel:
    """sklearn非互換モデル"""

    def __init__(self):
        self.weight = 2.0

    def train(self, data):
        pass

    def inference(self, data):
        return np.random.randn(len(data))


class MockNonSklearnBaseModel(BaseModelInterface):
    """sklearn非互換BaseModelInterface"""

    def __init__(self, model_name="NonSklearnModel", config=None):
        super().__init__(model_name, config)
        self.model = MockNonSklearnModel()

    def fit(self, X, y, validation_data=None):
        self.is_trained = True
        return {"training_time": 3.0}

    def predict(self, X):
        return ModelPrediction(
            predictions=np.random.randn(len(X)),
            model_name=self.model_name
        )

    def get_feature_importance(self):
        return {"feature1": 1.0}


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestStackingEnsembleModelCopyingOptimization:
    """Issue #691: StackingEnsembleベースモデルコピー最適化テスト"""

    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return StackingConfig(
            meta_learner_type="ridge",
            cv_folds=2,
            verbose=False
        )

    @pytest.fixture
    def test_data(self):
        """テスト用データ"""
        np.random.seed(42)
        n_samples = 50
        n_features = 3
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = pd.Series(np.random.randn(n_samples))
        return X, y

    @pytest.fixture
    def stacking_ensemble(self, config):
        """テスト用StackingEnsemble"""
        return StackingEnsemble(config)

    def test_copy_model_with_custom_copy_method(self, stacking_ensemble):
        """Issue #691: カスタムcopyメソッド優先テスト"""

        # カスタムcopyメソッド付きモック
        class CustomCopyModel(MockBaseModelInterface):
            def copy(self):
                new_model = CustomCopyModel(self.model_name + "_copied")
                new_model.is_trained = False
                return new_model

        original_model = CustomCopyModel("original")
        original_model.is_trained = True

        with patch('day_trade.ml.stacking_ensemble.logger') as mock_logger:
            copied_model = stacking_ensemble._copy_model(original_model)

            # カスタムcopyメソッドが使用されることを確認
            mock_logger.debug.assert_called_with("original: 独自copyメソッドでコピー")
            assert copied_model.model_name == "original_copied"
            assert copied_model.is_trained == False

    def test_copy_model_sklearn_compatible_base_model(self, stacking_ensemble):
        """Issue #691: sklearn互換BaseModelInterface直接cloneテスト"""
        original_model = MockSklearnCompatibleBaseModel("sklearn_base_model")
        original_model.is_trained = True
        original_model.training_metrics = {"accuracy": 0.95}

        with patch('sklearn.base.clone') as mock_clone:
            # cloneが成功する場合
            cloned_mock = MockSklearnCompatibleBaseModel("sklearn_base_model")
            mock_clone.return_value = cloned_mock

            copied_model = stacking_ensemble._copy_model(original_model)

            # sklearn.base.cloneが呼ばれることを確認
            mock_clone.assert_called_once_with(original_model)
            assert copied_model.is_trained == False
            assert copied_model.training_metrics == {}
            assert copied_model.model is None

    def test_copy_model_sklearn_compatible_internal_model(self, stacking_ensemble):
        """Issue #691: sklearn互換内部モデルcloneテスト"""
        original_model = MockBaseModelInterface("internal_sklearn_model")
        original_model.is_trained = True
        original_model.training_metrics = {"loss": 0.2}

        with patch('sklearn.base.clone') as mock_clone:
            # 内部モデルのclone
            cloned_internal = MockSklearnModel(param1=2.0, param2="cloned")
            mock_clone.return_value = cloned_internal

            copied_model = stacking_ensemble._copy_model(original_model)

            # sklearn.base.cloneが内部モデルで呼ばれることを確認
            mock_clone.assert_called_once_with(original_model.model)
            assert copied_model.model == cloned_internal
            assert copied_model.is_trained == False
            assert copied_model.training_metrics == {}
            assert copied_model.model_name == original_model.model_name

    def test_copy_model_non_sklearn_fallback(self, stacking_ensemble):
        """Issue #691: sklearn非互換モデルのフォールバックテスト"""
        original_model = MockNonSklearnBaseModel("non_sklearn_model")
        original_model.is_trained = True
        original_model.feature_names = ["feature_a", "feature_b"]

        with patch('day_trade.ml.stacking_ensemble.logger') as mock_logger:
            copied_model = stacking_ensemble._copy_model(original_model)

            # 基本コピーにフォールバックすることを確認
            log_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any("基本コピー完了" in msg for msg in log_calls)

            assert copied_model.model_name == original_model.model_name
            assert copied_model.is_trained == False
            assert copied_model.training_metrics == {}
            assert copied_model.model is None
            assert copied_model.feature_names == ["feature_a", "feature_b"]

    def test_create_minimal_model_copy_efficiency(self, stacking_ensemble):
        """Issue #691: 最小限モデルコピーの効率性テスト"""
        original_model = MockBaseModelInterface("efficiency_test")
        original_model.is_trained = True
        original_model.training_metrics = {"epoch": 100, "loss": 0.05}
        original_model.feature_names = ["f1", "f2", "f3", "f4", "f5"]

        cloned_sklearn_model = MockSklearnModel(param1=3.0, param2="efficient")

        # 効率的コピー実行
        copied_model = stacking_ensemble._create_minimal_model_copy(
            original_model, cloned_sklearn_model
        )

        # 効率的コピーの検証
        assert copied_model.model == cloned_sklearn_model
        assert copied_model.model_name == original_model.model_name
        assert copied_model.is_trained == False  # 未学習状態で初期化
        assert copied_model.training_metrics == {}  # 空辞書で初期化
        assert copied_model.feature_names == original_model.feature_names
        assert type(copied_model) == type(original_model)

    def test_create_basic_model_copy_fallback(self, stacking_ensemble):
        """Issue #691: 基本モデルコピーフォールバックテスト"""
        original_model = MockNonSklearnBaseModel("basic_fallback")
        original_model.is_trained = True
        original_model.feature_names = ["basic_f1", "basic_f2"]

        copied_model = stacking_ensemble._create_basic_model_copy(original_model)

        # 基本コピーの検証
        assert copied_model.model_name == original_model.model_name
        assert copied_model.is_trained == False
        assert copied_model.training_metrics == {}
        assert copied_model.model is None
        assert copied_model.feature_names == original_model.feature_names
        assert type(copied_model) == type(original_model)

    def test_copy_model_sklearn_import_error(self, stacking_ensemble):
        """Issue #691: sklearn未インストール環境でのフォールバックテスト"""
        original_model = MockBaseModelInterface("sklearn_unavailable")
        original_model.is_trained = True

        # sklearn.base.cloneのImportErrorをシミュレート
        with patch('day_trade.ml.stacking_ensemble.clone', side_effect=ImportError("sklearn not available")):
            with patch('day_trade.ml.stacking_ensemble.logger') as mock_logger:
                copied_model = stacking_ensemble._copy_model(original_model)

                # sklearn未使用環境の警告が出力されることを確認
                mock_logger.warning.assert_called()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "sklearn未使用環境" in warning_msg

                # 基本コピーにフォールバック
                assert copied_model.model_name == original_model.model_name
                assert copied_model.is_trained == False

    def test_copy_model_performance_comparison(self, stacking_ensemble):
        """Issue #691: コピー性能比較テスト"""
        models_to_test = [
            MockSklearnCompatibleBaseModel("perf_sklearn_compat"),
            MockBaseModelInterface("perf_internal_sklearn"),
            MockNonSklearnBaseModel("perf_non_sklearn")
        ]

        copy_times = []

        for model in models_to_test:
            model.is_trained = True
            model.feature_names = [f"feature_{i}" for i in range(10)]

            start_time = time.time()
            copied_model = stacking_ensemble._copy_model(model)
            copy_time = time.time() - start_time
            copy_times.append(copy_time)

            # 基本的な一致性確認
            assert copied_model.model_name == model.model_name
            assert copied_model.is_trained == False
            assert copied_model.feature_names == model.feature_names

        print(f"コピー性能: sklearn互換 {copy_times[0]:.6f}s, 内部sklearn {copy_times[1]:.6f}s, 非sklearn {copy_times[2]:.6f}s")

        # 全てのコピーが合理的時間内で完了することを確認
        assert all(t < 0.1 for t in copy_times), "コピー時間が長すぎます"

    def test_copy_model_error_handling(self, stacking_ensemble):
        """Issue #691: コピーエラーハンドリングテスト"""

        # 異常なモデル（必要な属性が不足）
        class ProblematicModel:
            # model_nameが存在しない
            pass

        problematic_model = ProblematicModel()

        with pytest.raises(ValueError, match="モデル .* のコピーに失敗しました"):
            stacking_ensemble._copy_model(problematic_model)

    def test_copy_model_config_handling(self, stacking_ensemble):
        """Issue #691: モデル設定コピーテスト"""
        original_model = MockBaseModelInterface("config_test")
        original_model.config = {"learning_rate": 0.01, "batch_size": 32}
        original_model.is_trained = True

        copied_model = stacking_ensemble._copy_model(original_model)

        # 設定が適切にコピーされることを確認
        assert copied_model.config is not original_model.config  # 異なるオブジェクト
        assert copied_model.config == original_model.config  # 同じ内容

    def test_meta_feature_generation_with_optimized_copying(self, stacking_ensemble, test_data):
        """Issue #691: 最適化コピーを使用したメタ特徴量生成統合テスト"""
        X, y = test_data

        # テスト用ベースモデル設定
        stacking_ensemble.base_models = {
            "model1": MockBaseModelInterface("model1"),
            "model2": MockSklearnCompatibleBaseModel("model2"),
            "model3": MockNonSklearnBaseModel("model3")
        }

        # メタ特徴量生成実行（内部で_copy_modelが呼ばれる）
        with patch.object(stacking_ensemble, '_copy_model', wraps=stacking_ensemble._copy_model) as mock_copy:
            meta_features = stacking_ensemble._generate_meta_features(X.values, y.values)

            # _copy_modelが各モデルに対して呼ばれることを確認（CV分割数 * モデル数）
            expected_calls = stacking_ensemble.config.cv_folds * len(stacking_ensemble.base_models)
            assert mock_copy.call_count == expected_calls

            # メタ特徴量が正常に生成されることを確認
            assert meta_features.shape[0] == len(X)
            assert meta_features.shape[1] >= len(stacking_ensemble.base_models)  # 基本予測 + 統計量

    def test_copy_model_memory_efficiency(self, stacking_ensemble):
        """Issue #691: メモリ効率テスト"""
        original_model = MockBaseModelInterface("memory_test")

        # 大きなfeature_namesとtraining_metricsを設定
        original_model.feature_names = [f"feature_{i}" for i in range(1000)]
        original_model.training_metrics = {f"metric_{i}": np.random.random() for i in range(100)}
        original_model.is_trained = True

        copied_model = stacking_ensemble._copy_model(original_model)

        # メモリ効率的なコピーの確認
        assert copied_model.feature_names == original_model.feature_names
        assert copied_model.feature_names is not original_model.feature_names  # 異なるオブジェクト
        assert copied_model.training_metrics == {}  # 空で初期化
        assert copied_model.is_trained == False  # 未学習状態で初期化


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestStackingEnsembleEdgeCases:
    """Issue #691: エッジケーステスト"""

    def test_copy_model_with_none_config(self):
        """Issue #691: config=Noneの場合のテスト"""
        config = StackingConfig()
        stacking_ensemble = StackingEnsemble(config)

        original_model = MockBaseModelInterface("none_config_test")
        original_model.config = None

        copied_model = stacking_ensemble._copy_model(original_model)

        assert copied_model.config is None
        assert copied_model.model_name == original_model.model_name

    def test_copy_model_with_empty_feature_names(self):
        """Issue #691: 空のfeature_namesの場合のテスト"""
        config = StackingConfig()
        stacking_ensemble = StackingEnsemble(config)

        original_model = MockBaseModelInterface("empty_features")
        original_model.feature_names = []

        copied_model = stacking_ensemble._copy_model(original_model)

        assert copied_model.feature_names == []
        assert copied_model.feature_names is not original_model.feature_names

    def test_copy_model_partial_sklearn_compatibility(self):
        """Issue #691: 部分的sklearn互換性テスト"""
        config = StackingConfig()
        stacking_ensemble = StackingEnsemble(config)

        # get_paramsのみ実装（set_paramsなし）
        class PartialSklearnModel:
            def get_params(self, deep=True):
                return {"param": "value"}

            def fit(self, X, y):
                pass

        original_model = MockBaseModelInterface("partial_sklearn")
        original_model.model = PartialSklearnModel()

        with patch('day_trade.ml.stacking_ensemble.logger') as mock_logger:
            copied_model = stacking_ensemble._copy_model(original_model)

            # 基本コピーにフォールバックすることを確認
            log_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any("基本コピー完了" in msg for msg in log_calls)

            assert copied_model.model_name == original_model.model_name


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])