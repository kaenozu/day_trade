#!/usr/bin/env python3
"""
Model Save/Load Optimization Tests

Issue #693対応: BaseModelInterfaceモデル保存/読み込み最適化テスト
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys

sys.path.append('C:/gemini-desktop/day_trade/src')

# テストに必要なモジュールをインポート
try:
    from day_trade.ml.base_models.base_model_interface import BaseModelInterface
    TEST_AVAILABLE = True
except ImportError as e:
    TEST_AVAILABLE = False
    print(f"テストモジュールインポートエラー: {e}")


# テスト用のダミーモデルクラス
class MockBaseModel(BaseModelInterface):
    """テスト用ベースモデル実装"""

    def __init__(self, model_name="TestModel", config=None):
        super().__init__(model_name, config)

    def fit(self, X, y, validation_data=None):
        # ダミー実装
        self.model = MockModel()
        self.is_trained = True
        return {"training_time": 1.0}

    def predict(self, X):
        from day_trade.ml.base_models.base_model_interface import ModelPrediction
        return ModelPrediction(
            predictions=np.random.randn(len(X)),
            model_name=self.model_name
        )

    def get_feature_importance(self):
        return {"feature_1": 0.5, "feature_2": 0.5}


class MockModel:
    """テスト用モデルオブジェクト"""
    def __init__(self, model_type="sklearn"):
        self.model_type = model_type
        if model_type == "sklearn":
            self.__module__ = "sklearn.ensemble"
        elif model_type == "xgboost":
            self.__module__ = "xgboost.sklearn"
        elif model_type == "pytorch":
            self.__module__ = "torch.nn"
        elif model_type == "tensorflow":
            self.__module__ = "tensorflow.keras"

    def get_params(self):
        return {"param1": 1, "param2": 2}

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.random.randn(len(X))

    def save_model(self, filepath):
        # XGBoost用ダミー実装
        pass

    def state_dict(self):
        # PyTorch用ダミー実装
        return {"layer1.weight": np.random.randn(10, 5)}

    def save(self, filepath):
        # TensorFlow用ダミー実装
        pass


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestModelSaveLoadOptimization:
    """Issue #693: モデル保存/読み込み最適化テスト"""

    @pytest.fixture
    def mock_model(self):
        """テスト用モックモデル"""
        return MockBaseModel()

    @pytest.fixture
    def temp_filepath(self):
        """テンポラリファイルパス"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            filepath = tmp.name
        yield filepath
        # クリーンアップ
        try:
            os.unlink(filepath)
        except:
            pass

    def test_detect_optimal_save_method_sklearn(self, mock_model):
        """Issue #693: scikit-learn系モデルの保存方法検出テスト"""
        # scikit-learn系モデル
        sklearn_model = MockModel("sklearn")
        method = mock_model._detect_optimal_save_method(sklearn_model)
        assert method == "joblib"

    def test_detect_optimal_save_method_xgboost(self, mock_model):
        """Issue #693: XGBoost系モデルの保存方法検出テスト"""
        # XGBoost系モデル
        xgb_model = MockModel("xgboost")
        method = mock_model._detect_optimal_save_method(xgb_model)
        assert method == "xgboost"

    def test_detect_optimal_save_method_pytorch(self, mock_model):
        """Issue #693: PyTorch系モデルの保存方法検出テスト"""
        # PyTorch系モデル
        pytorch_model = MockModel("pytorch")
        method = mock_model._detect_optimal_save_method(pytorch_model)
        assert method == "pytorch"

    def test_detect_optimal_save_method_tensorflow(self, mock_model):
        """Issue #693: TensorFlow系モデルの保存方法検出テスト"""
        # TensorFlow系モデル
        tf_model = MockModel("tensorflow")
        method = mock_model._detect_optimal_save_method(tf_model)
        assert method == "tensorflow"

    def test_detect_optimal_save_method_fallback(self, mock_model):
        """Issue #693: フォールバック保存方法テスト"""
        # 不明なモデル
        unknown_model = object()
        method = mock_model._detect_optimal_save_method(unknown_model)
        assert method == "pickle"

        # Noneモデル
        method = mock_model._detect_optimal_save_method(None)
        assert method == "pickle"

    def test_detect_optimal_load_method_extensions(self, mock_model):
        """Issue #693: ファイル拡張子による読み込み方法検出テスト"""
        # joblib拡張子
        assert mock_model._detect_optimal_load_method("model.joblib") == "joblib"
        assert mock_model._detect_optimal_load_method("model.pkl.gz") == "joblib"

        # XGBoost拡張子
        assert mock_model._detect_optimal_load_method("model.json") == "xgboost"
        assert mock_model._detect_optimal_load_method("model_xgb.pkl") == "xgboost"

        # PyTorch拡張子
        assert mock_model._detect_optimal_load_method("model.pt") == "pytorch"
        assert mock_model._detect_optimal_load_method("model.pth") == "pytorch"

        # TensorFlow拡張子
        assert mock_model._detect_optimal_load_method("model.h5") == "tensorflow"
        assert mock_model._detect_optimal_load_method("saved_model.pb") == "tensorflow"

        # デフォルト（pickle）
        assert mock_model._detect_optimal_load_method("model.pkl") == "pickle"

    def test_save_with_pickle_fallback(self, mock_model, temp_filepath):
        """Issue #693: pickle保存（フォールバック）テスト"""
        mock_model.model = MockModel("sklearn")
        mock_model.is_trained = True

        success = mock_model._save_with_pickle(temp_filepath)
        assert success == True
        assert os.path.exists(temp_filepath)

        # ファイルサイズチェック（空でないこと）
        assert os.path.getsize(temp_filepath) > 0

    def test_load_with_pickle_fallback(self, mock_model, temp_filepath):
        """Issue #693: pickle読み込み（フォールバック）テスト"""
        # まず保存
        mock_model.model = MockModel("sklearn")
        mock_model.is_trained = True
        mock_model.config = {"param": "value"}
        mock_model.training_metrics = {"loss": 0.5}
        mock_model.feature_names = ["f1", "f2"]

        success = mock_model._save_with_pickle(temp_filepath)
        assert success

        # 新しいモデルで読み込み
        new_model = MockBaseModel("LoadedModel")
        success = new_model._load_with_pickle(temp_filepath)

        assert success == True
        assert new_model.is_trained == True
        assert new_model.config["param"] == "value"
        assert new_model.training_metrics["loss"] == 0.5
        assert new_model.feature_names == ["f1", "f2"]

    @patch('day_trade.ml.base_models.base_model_interface.joblib')
    def test_save_with_joblib(self, mock_joblib, mock_model, temp_filepath):
        """Issue #693: joblib保存テスト"""
        mock_joblib.dump.return_value = None
        mock_model.model = MockModel("sklearn")
        mock_model.is_trained = True

        success = mock_model._save_with_joblib(temp_filepath, compression=True)
        assert success == True

        # joblib.dump が呼ばれることを確認
        mock_joblib.dump.assert_called_once()

        # 引数確認
        call_args = mock_joblib.dump.call_args
        assert 'compress' in call_args[1]
        assert call_args[1]['compress'] == 3  # 圧縮レベル

    @patch('day_trade.ml.base_models.base_model_interface.joblib')
    def test_save_with_joblib_no_compression(self, mock_joblib, mock_model, temp_filepath):
        """Issue #693: joblib保存（圧縮なし）テスト"""
        mock_joblib.dump.return_value = None
        mock_model.model = MockModel("sklearn")

        success = mock_model._save_with_joblib(temp_filepath, compression=False)
        assert success == True

        call_args = mock_joblib.dump.call_args
        assert call_args[1]['compress'] == 0  # 圧縮なし

    def test_save_with_joblib_import_error_fallback(self, mock_model, temp_filepath):
        """Issue #693: joblib未インストール時のフォールバックテスト"""
        with patch('day_trade.ml.base_models.base_model_interface.joblib', side_effect=ImportError):
            with patch.object(mock_model, '_save_with_pickle', return_value=True) as mock_pickle:
                mock_model.model = MockModel("sklearn")

                success = mock_model._save_with_joblib(temp_filepath)

                # pickle保存にフォールバックすることを確認
                mock_pickle.assert_called_once_with(temp_filepath)
                assert success == True

    @patch('day_trade.ml.base_models.base_model_interface.joblib')
    def test_load_with_joblib(self, mock_joblib, mock_model, temp_filepath):
        """Issue #693: joblib読み込みテスト"""
        # モックデータ
        mock_data = {
            'model': MockModel("sklearn"),
            'model_name': 'TestModel',
            'is_trained': True,
            'config': {'param': 'value'}
        }
        mock_joblib.load.return_value = mock_data

        success = mock_model._load_with_joblib(temp_filepath)
        assert success == True

        # joblib.load が呼ばれることを確認
        mock_joblib.load.assert_called_once_with(temp_filepath)

        # データが正しく復元されることを確認
        assert mock_model.is_trained == True
        assert mock_model.config['param'] == 'value'

    def test_restore_model_data(self, mock_model):
        """Issue #693: モデルデータ復元テスト"""
        test_data = {
            'model': MockModel("sklearn"),
            'model_name': 'RestoredModel',
            'config': {'test': 'config'},
            'is_trained': True,
            'training_metrics': {'accuracy': 0.95},
            'feature_names': ['feature1', 'feature2']
        }

        mock_model._restore_model_data(test_data)

        assert mock_model.model_name == 'RestoredModel'
        assert mock_model.config['test'] == 'config'
        assert mock_model.is_trained == True
        assert mock_model.training_metrics['accuracy'] == 0.95
        assert mock_model.feature_names == ['feature1', 'feature2']

    def test_restore_model_data_invalid_input(self, mock_model):
        """Issue #693: 不正なモデルデータ復元テスト"""
        # 辞書でないデータ
        mock_model._restore_model_data("invalid_data")
        # エラーが発生しないことを確認（ログ警告のみ）

        # 空辞書
        mock_model._restore_model_data({})
        # エラーが発生しないことを確認

    def test_integration_save_and_load_cycle(self, mock_model, temp_filepath):
        """Issue #693: 保存・読み込みサイクル統合テスト"""
        # モデル設定
        mock_model.model = MockModel("sklearn")
        mock_model.is_trained = True
        mock_model.config = {'integration': 'test'}
        mock_model.training_metrics = {'epoch': 10}
        mock_model.feature_names = ['int_feature1', 'int_feature2']

        # 保存テスト
        save_success = mock_model.save_model(temp_filepath)
        assert save_success == True
        assert os.path.exists(temp_filepath)

        # 読み込みテスト
        new_model = MockBaseModel("IntegrationTest")
        load_success = new_model.load_model(temp_filepath)

        assert load_success == True
        assert new_model.is_trained == True
        assert new_model.config['integration'] == 'test'
        assert new_model.training_metrics['epoch'] == 10
        assert new_model.feature_names == ['int_feature1', 'int_feature2']

    def test_save_model_performance_logging(self, mock_model, temp_filepath):
        """Issue #693: 保存時のパフォーマンスログテスト"""
        import time

        mock_model.model = MockModel("sklearn")
        mock_model.is_trained = True

        with patch('day_trade.ml.base_models.base_model_interface.logger') as mock_logger:
            success = mock_model.save_model(temp_filepath)

            # 成功ログが出力されることを確認
            assert success == True
            mock_logger.info.assert_called()

            # ログメッセージに時間が含まれることを確認
            log_message = mock_logger.info.call_args[0][0]
            assert "モデル保存完了" in log_message
            assert "秒" in log_message

    def test_load_model_file_not_found(self, mock_model):
        """Issue #693: 存在しないファイル読み込みエラーテスト"""
        nonexistent_path = "/path/does/not/exist.pkl"

        with patch('day_trade.ml.base_models.base_model_interface.logger') as mock_logger:
            success = mock_model.load_model(nonexistent_path)

            assert success == False
            mock_logger.error.assert_called()

            # ファイル不存在エラーメッセージ確認
            error_message = mock_logger.error.call_args[0][0]
            assert "モデルファイルが見つかりません" in error_message

    def test_save_model_error_handling(self, mock_model):
        """Issue #693: 保存時エラーハンドリングテスト"""
        # 無効なファイルパス
        invalid_path = "/invalid/path/cannot/write.pkl"
        mock_model.model = MockModel("sklearn")

        with patch('day_trade.ml.base_models.base_model_interface.logger') as mock_logger:
            success = mock_model.save_model(invalid_path)

            assert success == False
            mock_logger.error.assert_called()


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestModelTypeDetectionEdgeCases:
    """Issue #693: モデルタイプ検出エッジケーステスト"""

    def test_model_detection_with_exceptions(self):
        """Issue #693: 検出中の例外処理テスト"""
        model = MockBaseModel()

        # 例外を発生させるモックオブジェクト
        class ProblematicModel:
            def __getattr__(self, name):
                raise AttributeError("Test exception")

        problematic_model = ProblematicModel()

        # 例外が発生してもpickleにフォールバックすることを確認
        method = model._detect_optimal_save_method(problematic_model)
        assert method == "pickle"

    def test_complex_model_hierarchy(self):
        """Issue #693: 複雑なモデル階層の検出テスト"""
        model = MockBaseModel()

        # 複数の条件を満たすモデル
        class ComplexModel:
            __module__ = "sklearn.ensemble.forest"

            def get_params(self):
                return {}

            def fit(self, X, y):
                pass

            def predict(self, X):
                return np.random.randn(len(X))

            def save_model(self, filepath):
                pass

        complex_model = ComplexModel()

        # scikit-learnが優先されることを確認（joblibが選択される）
        method = model._detect_optimal_save_method(complex_model)
        assert method == "joblib"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])