import shutil
import unittest
from pathlib import Path
from unittest.mock import (
    MagicMock,  # 追加
    patch,
)

import numpy as np
import pandas as pd

# MLflowのインポートをモック化
# MLflowがインストールされていない環境でもテストが実行できるように
# ただし、MLflowのログ記録のテストは、実際のMLflow環境が必要
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = MagicMock()  # ダミーのmlflowオブジェクト

# テスト対象のモジュール
from src.day_trade.automation.orchestrator import (
    NextGenAIOrchestrator,
    OrchestrationConfig,
)
from src.day_trade.ml.data_drift_detector import DataDriftDetector

# テスト用のダミーデータパス
TEST_DATA_DIR = Path("test_data")
BASELINE_STATS_PATH = TEST_DATA_DIR / "baseline_stats.json"
MLRUNS_DIR = Path("mlruns")


class TestDataDriftOrchestratorIntegration(unittest.TestCase):
    def setUp(self):
        # テスト用ディレクトリの作成
        TEST_DATA_DIR.mkdir(exist_ok=True)
        if MLRUNS_DIR.exists():
            shutil.rmtree(MLRUNS_DIR)
        MLRUNS_DIR.mkdir(exist_ok=True)

        # ベースラインファイルを削除してクリーンな状態にする
        if BASELINE_STATS_PATH.exists():
            BASELINE_STATS_PATH.unlink()

        # Orchestratorの初期化時にロードされるパスをテスト用に設定
        # これはOrchestratorの内部実装に依存するため、パッチングが必要
        # または、Orchestratorのコンストラクタでパスを渡せるようにする
        # 今回は、Orchestratorの__init__でPath("data/baseline_stats.json")がハードコードされているため、
        # テスト実行前にそのファイルを操作するか、Orchestratorのコードを修正する必要がある。
        # ここでは、テスト実行前にファイルを操作するアプローチを取る。

        # StockFetcherとAdvancedMLEngineをモック化
        self.mock_stock_fetcher = MagicMock()
        self.mock_ml_engine = MagicMock()
        self.mock_batch_fetcher = MagicMock()

        # モックの戻り値を設定
        self.mock_stock_fetcher.fetch_stock_data.return_value = pd.DataFrame(
            {
                "Close": np.random.normal(100, 5, 200),
                "Open": np.random.normal(100, 5, 200),
                "High": np.random.normal(100, 5, 200),
                "Low": np.random.normal(100, 5, 200),
                "Volume": np.random.randint(1000, 5000, 200),
            }
        )

        # AdvancedMLEngineのprepare_dataとpredictのモック
        self.mock_ml_engine.prepare_data.return_value = (
            np.random.rand(1, 100, 5),
            np.random.rand(1, 1),
        )
        self.mock_ml_engine.predict.return_value = MagicMock(
            predictions=np.array([0.5]), confidence=np.array([0.8])
        )

        # AdvancedBatchDataFetcherのモック
        self.mock_batch_fetcher.fetch_batch.return_value = {
            "7203": MagicMock(
                data=self.mock_stock_fetcher.fetch_stock_data.return_value,
                success=True,
                data_quality_score=95.0,
            )
        }

        # Orchestratorの内部依存をパッチ
        self.patcher_stock_fetcher = patch(
            "src.day_trade.automation.orchestrator.StockFetcher",
            return_value=self.mock_stock_fetcher,
        )
        self.patcher_ml_engine_create = patch(
            "src.day_trade.automation.orchestrator.create_advanced_ml_engine",
            return_value=self.mock_ml_engine,
        )
        self.patcher_batch_fetcher = patch(
            "src.day_trade.automation.orchestrator.AdvancedBatchDataFetcher",
            return_value=self.mock_batch_fetcher,
        )
        self.patcher_db_manager = patch(
            "src.day_trade.automation.orchestrator.get_default_database_manager",
            return_value=MagicMock(),
        )
        self.patcher_is_safe_mode = patch(
            "src.day_trade.automation.orchestrator.is_safe_mode", return_value=True
        )

        self.patcher_stock_fetcher.start()
        self.patcher_ml_engine_create.start()
        self.patcher_batch_fetcher.start()
        self.patcher_db_manager.start()
        self.patcher_is_safe_mode.start()

        # MLflowのトラッキングURIをテスト用に設定
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(f"file://{MLRUNS_DIR.absolute()}")
            mlflow.set_experiment("test_orchestrator_drift")

    def tearDown(self):
        self.patcher_stock_fetcher.stop()
        self.patcher_ml_engine_create.stop()
        self.patcher_batch_fetcher.stop()
        self.patcher_db_manager.stop()
        self.patcher_is_safe_mode.stop()

        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR)
        if MLRUNS_DIR.exists():
            shutil.rmtree(MLRUNS_DIR)

    def _create_orchestrator(self):
        # OrchestratorConfigを明示的に渡す
        config = OrchestrationConfig(enable_ml_engine=True, enable_advanced_batch=True)
        return NextGenAIOrchestrator(config=config)

    def test_data_drift_baseline_generation(self):
        """初回実行時にベースラインが生成されることをテスト"""
        orchestrator = self._create_orchestrator()
        symbols = ["7203"]

        # 最初の実行 - ベースラインが生成されるはず
        _report = orchestrator.run_advanced_analysis(symbols=symbols)  # _report に変更

        self.assertTrue(_report.successful_symbols > 0)
        self.assertTrue(BASELINE_STATS_PATH.exists())

        # ベースラインが正しくロードされたかを確認
        loaded_detector = DataDriftDetector()
        loaded_detector.load_baseline(str(BASELINE_STATS_PATH))
        self.assertIn(
            "feature1", loaded_detector.baseline_stats
        )  # feature1はダミーデータから

        # AIAnalysisResultにドリフト結果が含まれていないことを確認（初回なので）
        self.assertIsNotNone(_report.ai_analysis_results)
        self.assertEqual(len(_report.ai_analysis_results), 1)
        self.assertIsNone(_report.ai_analysis_results[0].data_drift_results)

    def test_data_drift_detection_and_alert(self):
        """2回目以降の実行でドリフトが検出され、アラートが生成されることをテスト"""
        orchestrator = self._create_orchestrator()
        symbols = ["7203"]

        # 1回目の実行でベースラインを生成
        orchestrator.run_advanced_analysis(symbols=symbols)
        self.assertTrue(BASELINE_STATS_PATH.exists())

        # 2回目の実行用に、意図的にドリフトするデータをモックに設定
        self.mock_stock_fetcher.fetch_stock_data.return_value = pd.DataFrame(
            {
                "Close": np.random.normal(50, 5, 200),  # 平均を大きくずらす
                "Open": np.random.normal(50, 5, 200),
                "High": np.random.normal(50, 5, 200),
                "Low": np.random.normal(50, 5, 200),
                "Volume": np.random.randint(1000, 5000, 200),
            }
        )
        self.mock_batch_fetcher.fetch_batch.return_value = {
            "7203": MagicMock(
                data=self.mock_stock_fetcher.fetch_stock_data.return_value,
                success=True,
                data_quality_score=95.0,
            )
        }

        # 2回目の実行 - ドリフトが検出されるはず
        report = orchestrator.run_advanced_analysis(symbols=symbols)

        self.assertTrue(report.successful_symbols > 0)
        self.assertIsNotNone(report.ai_analysis_results)
        self.assertEqual(len(report.ai_analysis_results), 1)

        drift_results = report.ai_analysis_results[0].data_drift_results
        self.assertIsNotNone(drift_results)
        self.assertTrue(drift_results["drift_detected"])
        self.assertTrue(drift_results["features"]["Close"]["drift_detected"])

        # アラートが生成されたことを確認
        self.assertIsNotNone(report.triggered_alerts)
        self.assertTrue(
            any(
                alert["type"] == "DATA_DRIFT_DETECTED"
                for alert in report.triggered_alerts
            )
        )

    @unittest.skipUnless(MLFLOW_AVAILABLE, "MLflow is not installed")
    def test_data_drift_mlflow_logging(self):
        """データドリフト結果がMLflowにログされることをテスト"""
        orchestrator = self._create_orchestrator()
        symbols = ["7203"]

        # 1回目の実行でベースラインを生成
        orchestrator.run_advanced_analysis(symbols=symbols)

        # 2回目の実行用に、意図的にドリフトするデータをモックに設定
        self.mock_stock_fetcher.fetch_stock_data.return_value = pd.DataFrame(
            {
                "Close": np.random.normal(50, 5, 200),  # 平均を大きくずらす
                "Open": np.random.normal(50, 5, 200),
                "High": np.random.normal(50, 5, 200),
                "Low": np.random.normal(50, 5, 200),
                "Volume": np.random.randint(1000, 5000, 200),
            }
        )
        self.mock_batch_fetcher.fetch_batch.return_value = {
            "7203": MagicMock(
                data=self.mock_stock_fetcher.fetch_stock_data.return_value,
                success=True,
                data_quality_score=95.0,
            )
        }

        # MLflowのRunを開始
        with mlflow.start_run(run_name="test_drift_logging") as run:
            report = orchestrator.run_advanced_analysis(symbols=symbols)
            # MLflowにログが記録されたことを確認
            # 直接ログの内容を検証するのは難しいので、Runが存在し、アーティファクトが生成されたことを確認
            # MLflowのAPIを使ってRunからアーティファクトをダウンロードして検証することも可能だが、ここではシンプルに
            self.assertTrue(MLRUNS_DIR.exists())
            # run_idに対応するディレクトリが存在することを確認
            run_path = MLRUNS_DIR / run.info.experiment_id / run.info.run_id
            self.assertTrue(run_path.exists())
            # artifactsディレクトリが存在することを確認
            self.assertTrue((run_path / "artifacts").exists())
            # data_drift_results_7203.json が存在することを確認
            self.assertTrue(
                (run_path / "artifacts" / "data_drift_results_7203.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
