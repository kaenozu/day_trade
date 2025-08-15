import unittest
import sys
from pathlib import Path
import sqlite3
from unittest.mock import patch, AsyncMock, MagicMock, ANY
import pandas as pd
import asyncio
import os
from datetime import datetime
import time
import numpy as np
import importlib

# プロジェクトルート設定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import model_performance_monitor
import ml_model_upgrade_system
import day_trading_engine
import ml_prediction_models
import advanced_ml_prediction_system

class TestIntegrationRetraining(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_dir = Path("test_integration_data")
        self.test_dir.mkdir(exist_ok=True)
        
        # 各テストメソッドでユニークなDBファイル名を使用
        test_id = self.id().split('.')[-1] # テストメソッド名を取得
        self.upgrade_db_path = self.test_dir / f"test_upgrade_system_{test_id}.db"
        self.ml_predictions_db_path = self.test_dir / f"test_ml_predictions_{test_id}.db"
        self.advanced_ml_predictions_db_path = self.test_dir / f"test_advanced_ml_predictions_{test_id}.db"

        # テスト用DBを初期化 (テーブル作成のみ)
        self._init_test_dbs()

        # Reload modules to ensure patches are applied correctly
        importlib.reload(ml_model_upgrade_system)
        importlib.reload(model_performance_monitor)
        importlib.reload(day_trading_engine)
        importlib.reload(ml_prediction_models)
        importlib.reload(advanced_ml_prediction_system)

    def _init_test_dbs(self):
        # upgrade_system.dbの初期化
        conn = sqlite3.connect(self.upgrade_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_thresholds (
                metric_name TEXT PRIMARY KEY,
                threshold_value REAL,
                last_updated TEXT
            )
        """
        )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_integration_config (
                symbol TEXT PRIMARY KEY,
                preferred_system TEXT,
                updated_at TEXT
            )
        """
        )
        conn.commit()
        conn.close()

        # ml_predictions.dbの初期化
        conn = sqlite3.connect(self.ml_predictions_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                task TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                cross_val_mean REAL,
                cross_val_std REAL,
                training_time REAL,
                training_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                model_type TEXT,
                task TEXT,
                prediction TEXT,
                confidence REAL,
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ensemble_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                final_prediction TEXT,
                confidence REAL,
                consensus_strength REAL,
                disagreement_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        conn.close()

        # advanced_ml_predictions.dbの初期化
        conn = sqlite3.connect(self.advanced_ml_predictions_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS advanced_model_performances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                model_type TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                roc_auc REAL,
                cross_val_score REAL,
                feature_importance TEXT,
                created_at TEXT,
                UNIQUE(symbol, model_type)
            )
        """
        )
        conn.commit()
        conn.close()

    async def asyncTearDown(self):
        # テスト用DBファイルを削除
        for db_path in [self.upgrade_db_path, self.ml_predictions_db_path, self.advanced_ml_predictions_db_path]:
            if db_path.exists():
                try:
                    os.remove(db_path)
                except PermissionError:
                    pass # ファイルが使用中の場合はスキップ
        
        if self.test_dir.exists() and not list(self.test_dir.iterdir()): # ディレクトリが空の場合のみ削除
            os.rmdir(self.test_dir)

    @patch('day_trading_engine.REAL_DATA_AVAILABLE', False)
    @patch('day_trading_engine.YFinanceFetcher')
    @patch('ml_model_upgrade_system.hyperparameter_optimizer') # Patch the instance directly
    @patch('ml_model_upgrade_system.MLPredictionModels') # Patch the class directly
    @patch('ml_model_upgrade_system.AdvancedMLPredictionSystem') # Patch the class directly
    @patch('model_performance_monitor.ml_upgrade_system')
    async def test_retraining_improves_accuracy_advanced_integrated(self, mock_ml_upgrade_system, mock_advanced_ml_prediction_system, mock_ml_prediction_models, mock_hyperparameter_optimizer, mock_yfinance_fetcher):
        # シナリオ1: 性能が低下し、再学習によって精度が向上し、高度モデルが統合される
        symbol = "TEST_SYMBOL_ADVANCED"
        initial_accuracy = 0.80
        retrained_accuracy = 0.95
        threshold = 0.85

        # 1. 初期性能を設定 (閾値以下)
        with sqlite3.connect(self.advanced_ml_predictions_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO advanced_model_performances (symbol, model_type, accuracy, created_at) VALUES (?, ?, ?, ?)",
                           (symbol, "ensemble_voting", initial_accuracy, datetime.now().isoformat()))
            conn.commit()

        # 2. 性能閾値を設定
        with sqlite3.connect(self.upgrade_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO performance_thresholds (metric_name, threshold_value, last_updated) VALUES (?, ?, ?)",
                           ("accuracy", threshold, datetime.now().isoformat()))
            conn.commit()

                # 3. ml_upgrade_systemのモック設定
        mock_report = ml_model_upgrade_system.SystemUpgradeReport(
            upgrade_date=datetime.now(),
            total_symbols=1,
            successfully_upgraded=1,
            failed_upgrades=0,
            average_accuracy_before=initial_accuracy,
            average_accuracy_after=retrained_accuracy,
            overall_improvement=((retrained_accuracy - initial_accuracy) / initial_accuracy) * 100,
            individual_results=[
                ml_model_upgrade_system.ModelComparisonResult(
                    symbol=symbol,
                    original_accuracy=initial_accuracy,
                    advanced_accuracy=retrained_accuracy,
                    improvement=((retrained_accuracy - initial_accuracy) / initial_accuracy) * 100,
                    best_model_type="advanced",
                    recommendation="高度システム採用を強く推奨",
                    detailed_metrics={}
                )
            ],
            recommendations=["高度システムへの完全移行を強く推奨します"]
        )
        mock_ml_upgrade_system.run_complete_system_upgrade = AsyncMock(return_value=mock_report)

        async def mock_integrate_and_save(report):
            config = {symbol: "advanced_system"}
            with sqlite3.connect(self.upgrade_db_path) as conn:
                cursor = conn.cursor()
                for s, sys in config.items():
                    cursor.execute(
                        "INSERT OR REPLACE INTO model_integration_config (symbol, preferred_system, updated_at) VALUES (?, ?, ?)",
                        (s, sys, datetime.now().isoformat())
                    )
                conn.commit()
            return config
        mock_ml_upgrade_system.integrate_best_models = AsyncMock(side_effect=mock_integrate_and_save)


        # 4. PersonalDayTradingEngine の初期化と予測実行
        #    これにより、ModelPerformanceMonitor が呼び出され、再学習がトリガーされる
        engine = day_trading_engine.PersonalDayTradingEngine()
        engine.model_performance_monitor.upgrade_db_path = self.upgrade_db_path
        engine.model_performance_monitor.advanced_ml_db_path = self.advanced_ml_predictions_db_path
        engine.model_performance_monitor.load_thresholds()
        engine.model_loader.upgrade_db_path = self.upgrade_db_path
        engine.model_loader.ml_predictions_db_path = self.ml_predictions_db_path
        engine.model_loader.advanced_ml_predictions_db_path = self.advanced_ml_predictions_db_path
        engine.model_loader.model_integration_config = engine.model_loader._load_model_integration_config()
        engine.daytrading_symbols = {symbol: "Test Symbol"}

        # YFinanceFetcherのモック設定
        mock_yfinance_fetcher.return_value.get_historical_data.return_value = pd.DataFrame(
            {
                "始値": np.random.uniform(90, 100, 100),
                "高値": np.random.uniform(100, 110, 100),
                "安値": np.random.uniform(80, 90, 100),
                "終値": np.random.uniform(95, 105, 100),
                "出来高": np.random.randint(1000, 2000, 100),
            },
            index=pd.to_datetime(pd.date_range(start="2023-01-01", periods=100))
        )

        # MLPredictionModelsとAdvancedMLPredictionSystemのモック設定
        mock_ml_prediction_models.return_value.predict.return_value = [
            MagicMock(task=ml_prediction_models.PredictionTask.PRICE_DIRECTION, final_prediction=0, confidence=0.8),
            MagicMock(task=ml_prediction_models.PredictionTask.VOLATILITY, final_prediction=0.5, confidence=0.7)
        ]
        mock_advanced_ml_prediction_system.return_value.predict_with_advanced_models.return_value = MagicMock(prediction=1, confidence=0.9)

        # _analyze_tomorrow_premarket_opportunity または _analyze_daytrading_opportunity を直接呼び出す
        # daytrade.pyのrun_daytrading_modeがModelPerformanceMonitorを呼び出すため、
        # ここでは直接エンジンをテストする
        recommendations = await engine.get_today_daytrading_recommendations(limit=1)

        # 5. 検証
        # 再学習がトリガーされたことを確認
        mock_ml_upgrade_system.run_complete_system_upgrade.assert_called_once()

        # model_integration_config が更新されたことを確認
        with sqlite3.connect(self.upgrade_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT preferred_system FROM model_integration_config WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(result[0], "advanced_system")

        # PersonalDayTradingEngine が統合されたモデルを使用することを確認
        # これは、engine.model_loader.predict が advanced_ml_prediction_system の predict_with_advanced_models を呼び出すことを意味する
        # predict_with_advanced_models がモックされているため、その呼び出しを確認する
        mock_advanced_ml_prediction_system.return_value.predict_with_advanced_models.assert_called_once_with(symbol)
        mock_ml_prediction_models.return_value.predict.assert_not_called()

        # 推奨結果の確認
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0].symbol, symbol)
        self.assertEqual(recommendations[0].signal, day_trading_engine.DayTradingSignal.STRONG_BUY) # モックされた予測結果に基づく
        self.assertEqual(recommendations[0].confidence, 90.0) # モックされた予測結果に基づく

    @patch('day_trading_engine.REAL_DATA_AVAILABLE', False)
    @patch('day_trading_engine.YFinanceFetcher')
    @patch('ml_model_upgrade_system.hyperparameter_optimizer')
    @patch('ml_model_upgrade_system.MLPredictionModels')
    @patch('ml_model_upgrade_system.AdvancedMLPredictionSystem')
    @patch('model_performance_monitor.ml_upgrade_system')
    async def test_retraining_no_improvement_original_integrated(self, mock_ml_upgrade_system, mock_advanced_ml_prediction_system, mock_ml_prediction_models, mock_hyperparameter_optimizer, mock_yfinance_fetcher):
        # シナリオ2: 性能が低下するが、再学習によって精度が向上せず、オリジナルモデルが統合される
        symbol = "TEST_SYMBOL_ORIGINAL"
        initial_accuracy = 0.80
        retrained_accuracy = 0.75 # 改善なし
        threshold = 0.85

        # 1. 初期性能を設定 (閾値以下)
        with sqlite3.connect(self.advanced_ml_predictions_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO advanced_model_performances (symbol, model_type, accuracy, created_at) VALUES (?, ?, ?, ?)",
                           (symbol, "ensemble_voting", initial_accuracy, datetime.now().isoformat()))
            conn.commit()

        # 2. 性能閾値を設定
        with sqlite3.connect(self.upgrade_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO performance_thresholds (metric_name, threshold_value, last_updated) VALUES (?, ?, ?)",
                           ("accuracy", threshold, datetime.now().isoformat()))
            conn.commit()

                # 3. ml_upgrade_systemのモック設定
        mock_report = ml_model_upgrade_system.SystemUpgradeReport(
            upgrade_date=datetime.now(),
            total_symbols=1,
            successfully_upgraded=1,
            failed_upgrades=0,
            average_accuracy_before=initial_accuracy,
            average_accuracy_after=retrained_accuracy,
            overall_improvement=((retrained_accuracy - initial_accuracy) / initial_accuracy) * 100,
            individual_results=[
                ml_model_upgrade_system.ModelComparisonResult(
                    symbol=symbol,
                    original_accuracy=initial_accuracy,
                    advanced_accuracy=retrained_accuracy,
                    improvement=((retrained_accuracy - initial_accuracy) / initial_accuracy) * 100,
                    best_model_type="original",
                    recommendation="既存システム維持",
                    detailed_metrics={}
                )
            ],
            recommendations=["既存システムの維持を推奨します"]
        )
        mock_ml_upgrade_system.run_complete_system_upgrade = AsyncMock(return_value=mock_report)

        async def mock_integrate_and_save(report):
            config = {symbol: "original_system"}
            with sqlite3.connect(self.upgrade_db_path) as conn:
                cursor = conn.cursor()
                for s, sys in config.items():
                    cursor.execute(
                        "INSERT OR REPLACE INTO model_integration_config (symbol, preferred_system, updated_at) VALUES (?, ?, ?)",
                        (s, sys, datetime.now().isoformat())
                    )
                conn.commit()
            return config
        mock_ml_upgrade_system.integrate_best_models = AsyncMock(side_effect=mock_integrate_and_save)


        # 4. PersonalDayTradingEngine の初期化と予測実行
        engine = day_trading_engine.PersonalDayTradingEngine()
        engine.model_performance_monitor.upgrade_db_path = self.upgrade_db_path
        engine.model_performance_monitor.advanced_ml_db_path = self.advanced_ml_predictions_db_path
        engine.model_performance_monitor.load_thresholds()
        engine.model_loader.upgrade_db_path = self.upgrade_db_path
        engine.model_loader.ml_predictions_db_path = self.ml_predictions_db_path
        engine.model_loader.advanced_ml_predictions_db_path = self.advanced_ml_predictions_db_path
        engine.model_loader.model_integration_config = engine.model_loader._load_model_integration_config()
        engine.daytrading_symbols = {symbol: "Test Symbol"}

        # YFinanceFetcherのモック設定
        mock_yfinance_fetcher.return_value.get_historical_data.return_value = pd.DataFrame(
            {
                "始値": np.random.uniform(90, 100, 100),
                "高値": np.random.uniform(100, 110, 100),
                "安値": np.random.uniform(80, 90, 100),
                "終値": np.random.uniform(95, 105, 100),
                "出来高": np.random.randint(1000, 2000, 100),
            },
            index=pd.to_datetime(pd.date_range(start="2023-01-01", periods=100))
        )

        # MLPredictionModelsとAdvancedMLPredictionSystemのモック設定
        mock_ml_prediction_models.return_value.predict.return_value = [
            MagicMock(task=ml_prediction_models.PredictionTask.PRICE_DIRECTION, final_prediction=0, confidence=0.8),
            MagicMock(task=ml_prediction_models.PredictionTask.VOLATILITY, final_prediction=0.5, confidence=0.7)
        ]
        mock_advanced_ml_prediction_system.return_value.predict_with_advanced_models.return_value = MagicMock(prediction=0, confidence=0.7, model_consensus={}, feature_values={}, timestamp=datetime.now())

        recommendations = await engine.get_today_daytrading_recommendations(limit=1)

        # 5. 検証
        # 再学習がトリガーされたことを確認
        mock_ml_upgrade_system.run_complete_system_upgrade.assert_called_once()

        # model_integration_config が更新されたことを確認
        with sqlite3.connect(self.upgrade_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT preferred_system FROM model_integration_config WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(result[0], "original_system")

        # PersonalDayTradingEngine が統合されたモデルを使用することを確認
        # predict がモックされているため、その呼び出しを確認する
        mock_ml_prediction_models.return_value.predict.assert_called_once_with(symbol, ANY)
        mock_advanced_ml_prediction_system.return_value.predict_with_advanced_models.assert_not_called()

        # 推奨結果の確認
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0].symbol, symbol)
        self.assertEqual(recommendations[0].signal, day_trading_engine.DayTradingSignal.WAIT) # モックされた予測結果に基づく
        self.assertEqual(recommendations[0].confidence, 80.0) # モックされた予測結果に基づく

if __name__ == '__main__':
    unittest.main()
