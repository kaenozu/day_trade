import unittest
from pathlib import Path

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

    def test_train_predict_symbol_specific_model(self):
        """銘柄固有モデルの訓練と予測をテスト"""
        config = ModelConfig(model_type="linear", task_type="regression")
        self.manager.create_model("return_predictor", config)

        # 銘柄「7203」のモデルを訓練
        self.manager.train_model("return_predictor", self.X, self.y)

        # 銘柄「7203」のモデルで予測
        predictions = self.manager.predict("return_predictor", self.X)
        self.assertEqual(len(predictions), 100)

        # 別の銘柄「8306」のモデルは訓練されていないことを確認
        predictions = self.manager.predict("return_predictor", self.X)

    def test_save_load_symbol_specific_model(self):
        """銘柄固有モデルの保存と読み込みをテスト"""
        config = ModelConfig(model_type="linear", task_type="regression")
        self.manager.create_model("return_predictor", config)

        # 銘柄「7203」のモデルを訓練して保存
        self.manager.train_model("return_predictor", self.X, self.y)
        self.manager.save_model("return_predictor")

        # 新しいマネージャーでモデルを読み込み
        new_manager = MLModelManager(models_dir=str(self.models_dir))
        new_manager.load_model("return_predictor")

        # 読み込んだモデルで予測
        predictions = new_manager.predict("return_predictor", self.X)
        self.assertEqual(len(predictions), 100)


if __name__ == "__main__":
    unittest.main()
