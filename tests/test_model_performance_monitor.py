import unittest
import sys
from pathlib import Path
import sqlite3
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
import os
from datetime import datetime
import time 
import tempfile 

# プロジェクトルート設定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_performance_monitor import ModelPerformanceMonitor

class TestModelPerformanceMonitor(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_dir = Path("test_ml_models_data")
        self.test_dir.mkdir(exist_ok=True)
        
        test_id = self.id().split('.')[-1] 
        self.test_db_path = self.test_dir / f"test_upgrade_system_{test_id}.db"
        self.test_advanced_ml_db_path = self.test_dir / f"test_advanced_ml_predictions_{test_id}.db"

        self._init_test_dbs()

    def _init_test_dbs(self):
        conn = sqlite3.connect(self.test_db_path)
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

        conn = sqlite3.connect(self.test_advanced_ml_db_path)
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
        for db_path in [self.test_db_path, self.test_advanced_ml_db_path]:
            if db_path.exists():
                for i in range(5): 
                    try:
                        os.remove(db_path)
                        break
                    except PermissionError:
                        time.sleep(0.1 * (i + 1)) 
                else:
                    pass 
        
        if self.test_dir.exists() and not list(self.test_dir.iterdir()): 
            os.rmdir(self.test_dir)

    async def test_load_thresholds_existing(self):
        with sqlite3.connect(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO performance_thresholds (metric_name, threshold_value, last_updated) VALUES (?, ?, ?)",
                           ("accuracy", 0.95, datetime.now().isoformat()))
            conn.commit()
        
        monitor = ModelPerformanceMonitor(self.test_db_path, self.test_advanced_ml_db_path)
        monitor.load_thresholds()
        self.assertEqual(monitor.thresholds["accuracy"], 0.95)

    async def test_load_thresholds_default(self):
        monitor = ModelPerformanceMonitor(self.test_db_path, self.test_advanced_ml_db_path)
        monitor.load_thresholds()
        self.assertEqual(monitor.thresholds["accuracy"], 0.90) 

    @patch('model_performance_monitor.PredictionAccuracyValidator')
    async def test_get_latest_model_performance_existing(self, MockPredictionAccuracyValidator):
        # MockPredictionAccuracyValidator は PredictionAccuracyValidator クラスのモック
        # ModelPerformanceMonitor.__init__ で PredictionAccuracyValidator() が呼び出されるので、
        # その戻り値 (つまりインスタンスモック) の validate_current_system_accuracy メソッドを設定
        MockPredictionAccuracyValidator.return_value.validate_current_system_accuracy = AsyncMock(return_value=MagicMock(overall_accuracy=92.0))

        monitor = ModelPerformanceMonitor(self.test_db_path, self.test_advanced_ml_db_path)
        performance = await monitor.get_latest_model_performance()
        self.assertEqual(performance["accuracy"], 92.0) 

    @patch('model_performance_monitor.PredictionAccuracyValidator')
    async def test_get_latest_model_performance_none(self, MockPredictionAccuracyValidator):
        # When no data, overall_accuracy is 0.0, and overall_accuracy is what is returned.
        mock_accuracy_metrics = MagicMock(overall_accuracy=0.0)
        MockPredictionAccuracyValidator.return_value.validate_current_system_accuracy = AsyncMock(return_value=mock_accuracy_metrics)

        monitor = ModelPerformanceMonitor(self.test_db_path, self.test_advanced_ml_db_path)
        performance = await monitor.get_latest_model_performance()
        self.assertEqual(performance, {"accuracy": 0.0}) 

    @patch('model_performance_monitor.ml_upgrade_system')
    @patch('model_performance_monitor.PredictionAccuracyValidator')
    async def test_check_and_trigger_retraining_below_threshold(self, MockPredictionAccuracyValidator, mock_ml_upgrade_system):
        mock_ml_upgrade_system.run_complete_system_upgrade = AsyncMock(return_value=(MagicMock(), MagicMock()))
        
        mock_accuracy_metrics = MagicMock(overall_accuracy=85.0) # 85% < 90%
        MockPredictionAccuracyValidator.return_value.validate_current_system_accuracy = AsyncMock(return_value=mock_accuracy_metrics)

        with sqlite3.connect(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO performance_thresholds (metric_name, threshold_value, last_updated) VALUES (?, ?, ?)",
                           ("accuracy", 90.0, datetime.now().isoformat())) 
            conn.commit()
        
        monitor = ModelPerformanceMonitor(self.test_db_path, self.test_advanced_ml_db_path)
        monitor.load_thresholds() 
        await monitor.check_and_trigger_retraining()
        mock_ml_upgrade_system.run_complete_system_upgrade.assert_called_once()

    @patch('model_performance_monitor.ml_upgrade_system')
    @patch('model_performance_monitor.PredictionAccuracyValidator')
    async def test_check_and_trigger_retraining_above_threshold(self, MockPredictionAccuracyValidator, mock_ml_upgrade_system):
        mock_ml_upgrade_system.run_complete_system_upgrade = AsyncMock(return_value=(MagicMock(), MagicMock()))

        mock_accuracy_metrics = MagicMock(overall_accuracy=92.0) # 92% >= 90%
        MockPredictionAccuracyValidator.return_value.validate_current_system_accuracy = AsyncMock(return_value=mock_accuracy_metrics)

        with sqlite3.connect(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO performance_thresholds (metric_name, threshold_value, last_updated) VALUES (?, ?, ?)",
                           ("accuracy", 90.0, datetime.now().isoformat())) 
            conn.commit()
        
        monitor = ModelPerformanceMonitor(self.test_db_path, self.test_advanced_ml_db_path)
        monitor.load_thresholds() 
        await monitor.check_and_trigger_retraining()
        mock_ml_upgrade_system.run_complete_system_upgrade.assert_not_called()

    @patch('model_performance_monitor.ml_upgrade_system')
    @patch('model_performance_monitor.PredictionAccuracyValidator')
    async def test_check_and_trigger_retraining_no_performance_data(self, MockPredictionAccuracyValidator, mock_ml_upgrade_system):
        mock_ml_upgrade_system.run_complete_system_upgrade = AsyncMock(return_value=(MagicMock(), MagicMock()))

        # Simulate no performance data by making validate_current_system_accuracy raise an exception
        # This will cause get_latest_model_performance to return {}
        MockPredictionAccuracyValidator.return_value.validate_current_system_accuracy.side_effect = Exception("Simulated error for no performance data")

        monitor = ModelPerformanceMonitor(self.test_db_path, self.test_advanced_ml_db_path)
        monitor.load_thresholds() 
        await monitor.check_and_trigger_retraining()
        mock_ml_upgrade_system.run_complete_system_upgrade.assert_not_called()

if __name__ == '__main__':
    unittest.main()