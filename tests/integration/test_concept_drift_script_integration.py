import unittest
import subprocess
import shutil
from pathlib import Path
import json
import os

# MLflowのインポートをモック化
# MLflowがインストールされていない環境でもテストが実行できるように
# ただし、MLflowのログ記録のテストは、実際のMLflow環境が必要
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = MagicMock() # ダミーのmlflowオブジェクト

# テスト用のディレクトリ
MLRUNS_DIR = Path("mlruns")

class TestConceptDriftScriptIntegration(unittest.TestCase):

    def setUp(self):
        # mlrunsディレクトリをクリーンアップ
        if MLRUNS_DIR.exists():
            shutil.rmtree(MLRUNS_DIR)
        MLRUNS_DIR.mkdir(exist_ok=True)

        # MLflowのトラッキングURIをテスト用に設定
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(f"file://{MLRUNS_DIR.absolute()}")
            mlflow.set_experiment("test_concept_drift_script")

    def tearDown(self):
        if MLRUNS_DIR.exists():
            shutil.rmtree(MLRUNS_DIR)

    def test_script_execution_and_mlflow_logging(self):
        """コンセプトドリフト検出スクリプトが実行され、MLflowにログが記録されることをテスト"""
        script_path = Path("src/day_trade/ml/run_concept_drift_detection.py")
        
        # スクリプトを実行
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True, text=True, check=False
        )

        # スクリプトがエラーなく終了したことを確認
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")
        self.assertIn("コンセプトドリフト検出が完了しました。", result.stdout)

        # MLflowのRunが作成されたことを確認
        self.assertTrue(MLRUNS_DIR.exists())
        
        # Runが作成されたことをより詳細に確認
        experiments = [f for f in MLRUNS_DIR.iterdir() if f.is_dir() and f.name != ".trash"]
        self.assertTrue(len(experiments) > 0, "No MLflow experiments found.")

        # 少なくとも1つのRunが存在することを確認
        runs_found = False
        for exp_dir in experiments:
            runs_dir = exp_dir / "runs"
            if runs_dir.exists():
                if any(runs_dir.iterdir()):
                    runs_found = True
                    break
        self.assertTrue(runs_found, "No MLflow runs found.")

        # ログされたメトリクスやアーティファクトを検証（オプション）
        # MLflowのSearch APIを使って、特定のメトリクスがログされているか確認
        if MLFLOW_AVAILABLE:
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(
                experiment_ids=client.get_experiment_by_name("concept_drift_detection").experiment_id,
                order_by=["attribute.start_time DESC"],
                max_results=1
            )
            self.assertEqual(len(runs), 1)
            latest_run = runs[0]

            # メトリクスがログされていることを確認
            self.assertIn("latest_mae", latest_run.data.metrics)
            self.assertIn("drift_detected", latest_run.data.metrics)

            # アーティファクトがログされていることを確認
            artifacts = client.list_artifacts(latest_run.info.run_id)
            artifact_names = [a.path for a in artifacts]
            self.assertIn("performance_summary.json", artifact_names)

if __name__ == '__main__':
    unittest.main()
