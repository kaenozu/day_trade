# test_overnight_prediction_model.py

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import asyncio

# テスト対象のクラスをインポート
from overnight_prediction_model import OvernightPredictionModel

class TestOvernightPredictionModel(unittest.TestCase):

    def setUp(self):
        """各テストの前に実行されるセットアップ"""
        self.model = OvernightPredictionModel()
        # テスト用のモデルファイルパス
        self.test_model_path = "test_overnight_model.joblib"
        self.model.MODEL_PATH = self.test_model_path

    def tearDown(self):
        """各テストの後に実行されるクリーンアップ"""
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)

    def _run_async(self, coro):
        """非同期コードを同期的に実行するためのヘルパー"""
        return asyncio.run(coro)

    def _create_dummy_yf_dataframe(self, days=50):
        """yfinance.downloadの戻り値全体を模倣するダミーDataFrameを作成"""
        dates = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D'))
        tickers = list(self.model.feature_tickers.keys())

        # MultiIndexカラムを作成
        columns = pd.MultiIndex.from_product([['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], tickers])

        # ダミーデータを作成
        data = np.random.rand(days, len(columns))
        df = pd.DataFrame(data, index=dates, columns=columns)
        return df

    @patch('yfinance.download')
    def test_prepare_data(self, mock_yf_download):
        """_prepare_dataメソッドが正しくデータを整形できるかテスト"""
        print("\n--- test_prepare_data --- ")
        # yfinance.downloadの戻り値を設定
        dummy_df = self._create_dummy_yf_dataframe(days=100)
        mock_yf_download.return_value = dummy_df

        # テスト対象メソッドを実行
        prepared_data = self._run_async(self.model._prepare_data(period="100d"))

        # 検証
        self.assertIsInstance(prepared_data, pd.DataFrame)
        self.assertFalse(prepared_data.empty, "_prepare_data should return a non-empty DataFrame")
        # 目的変数と特徴量カラムが存在するか
        self.assertIn('target_up', prepared_data.columns)
        self.assertIn('NIKKEI_pct_change', prepared_data.columns)
        # NaNが含まれていないか
        self.assertFalse(prepared_data.isnull().values.any())
        print("test_prepare_data: OK")

    @patch('overnight_prediction_model.OvernightPredictionModel._prepare_data')
    def test_train_and_predict(self, mock_prepare_data):
        """trainとpredictのサイクルが正常に動作するかテスト"""
        print("\n--- test_train_and_predict --- ")
        # _prepare_dataの戻り値を設定
        dummy_prepared_df = self._create_dummy_prepared_data(days=100)

        async def async_magic_mock(*args, **kwargs):
            return dummy_prepared_df
        mock_prepare_data.side_effect = async_magic_mock

        # 1. 学習(train)のテスト
        self._run_async(self.model.train())

        # モデルファイルが作成されたか
        self.assertTrue(os.path.exists(self.test_model_path))
        print(f"train: モデルファイル {self.test_model_path} の作成を確認")

        # 2. 予測(predict)のテスト
        # 新しいインスタンスを作成して、モデルがファイルから読み込まれることを確認
        predictor = OvernightPredictionModel()
        predictor.MODEL_PATH = self.test_model_path
        prediction_result = self._run_async(predictor.predict())

        # 予測結果の形式を検証
        self.assertIsInstance(prediction_result, dict)
        self.assertIn('prediction', prediction_result)
        self.assertIn('probability_up', prediction_result)
        self.assertIn('probability_down', prediction_result)
        self.assertIn(prediction_result['prediction'], ['Up', 'Down'])
        self.assertAlmostEqual(prediction_result['probability_up'] + prediction_result['probability_down'], 1.0, places=5)
        print("predict: 予測結果の形式と内容を確認")
        print("test_train_and_predict: OK")

    def _create_dummy_prepared_data(self, days=100):
        """整形済みのダミーデータを作成"""
        dates = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D'))
        # 特徴量カラム名を生成
        feature_cols = []
        for ticker_name in self.model.feature_tickers.values():
            feature_cols.append(f'{ticker_name}_pct_change')
            feature_cols.append(f'{ticker_name}_ma5_divergence')
            feature_cols.append(f'{ticker_name}_ma25_divergence')

        data = {col: np.random.rand(days) for col in feature_cols}
        features_df = pd.DataFrame(data, index=dates)

        # 目的変数
        target = pd.Series(np.random.randint(0, 2, size=days), index=dates, name='target_up')

        return pd.concat([features_df, target], axis=1).dropna()

if __name__ == '__main__':
    unittest.main()