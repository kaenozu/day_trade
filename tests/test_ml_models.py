import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.day_trade.analysis.ml_models import MLModelManager, ModelConfig


class TestMLModelManager(unittest.TestCase):
    def setUp(self):
        """テストのセットアップ"""
        self.models_dir = Path("test_models")
        self.models_dir.mkdir(exist_ok=True)
        self.manager = MLModelManager(models_dir=str(self.models_dir))

        # テストデータ
        self.X = pd.DataFrame({"feature1": range(100), "feature2": range(100, 200)})
        self.y = pd.Series(range(100))

    def tearDown(self):
        """テストの後片付け"""
        for f in self.models_dir.glob("*.joblib"):
            f.unlink()
        self.models_dir.rmdir()

    @patch("src.day_trade.analysis.ml_models.MLModelManager.train_model")
    @patch("src.day_trade.analysis.ml_models.MLModelManager.predict")
    def test_train_predict_symbol_specific_model(self, mock_predict, mock_train_model):
        """銘柄固有モデルの訓練と予測をテスト（モック化で高速化）"""
        config = ModelConfig(model_type="linear", task_type="regression")
        self.manager.create_model("return_predictor", config)

        # モックの戻り値を設定（実際の訓練をスキップ）
        mock_train_model.return_value = None
        mock_predict.return_value = np.random.random(100)  # 100個のランダムな予測値

        # 銘柄「7203」のモデルを訓練（モック化済み）
        self.manager.train_model("return_predictor", self.X, self.y)

        # 銘柄「7203」のモデルで予測（モック化済み）
        predictions = self.manager.predict("return_predictor", self.X)
        self.assertEqual(len(predictions), 100)

        # モックが適切に呼び出されたことを確認
        mock_train_model.assert_called_once()
        mock_predict.assert_called_once()

        # 別の銘柄「8306」のモデルは訓練されていないことを確認
        predictions = self.manager.predict("return_predictor", self.X)

    @patch("src.day_trade.analysis.ml_models.MLModelManager.train_model")
    @patch("src.day_trade.analysis.ml_models.MLModelManager.save_model")
    @patch("src.day_trade.analysis.ml_models.MLModelManager.load_model")
    @patch("src.day_trade.analysis.ml_models.MLModelManager.predict")
    @patch("joblib.dump")
    @patch("joblib.load")
    def test_save_load_symbol_specific_model(
        self,
        mock_joblib_load,
        mock_joblib_dump,
        mock_predict,
        mock_load_model,
        mock_save_model,
        mock_train_model,
    ):
        """銘柄固有モデルの保存と読み込みをテスト（モック化で高速化）"""
        config = ModelConfig(model_type="linear", task_type="regression")
        self.manager.create_model("return_predictor", config)

        # モックの戻り値を設定
        mock_train_model.return_value = None
        mock_save_model.return_value = None
        mock_load_model.return_value = None
        mock_predict.return_value = np.random.random(100)
        mock_joblib_dump.return_value = None
        mock_joblib_load.return_value = MagicMock()  # ダミーモデル

        # 銘柄「7203」のモデルを訓練して保存（モック化済み）
        self.manager.train_model("return_predictor", self.X, self.y)
        self.manager.save_model("return_predictor")

        # 新しいマネージャーでモデルを読み込み（モック化済み）
        new_manager = MLModelManager(models_dir=str(self.models_dir))
        new_manager.load_model("return_predictor")

        # 読み込んだモデルで予測（モック化済み）
        predictions = new_manager.predict("return_predictor", self.X)
        self.assertEqual(len(predictions), 100)

        # モック呼び出しの確認
        mock_train_model.assert_called_once()
        mock_save_model.assert_called_once()
        mock_load_model.assert_called_once()
        mock_predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
