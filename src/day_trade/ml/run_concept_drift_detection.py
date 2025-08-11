import json  # 追加
import logging
from datetime import datetime, timedelta
from typing import Optional  # 追加

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from concept_drift_detector import ConceptDriftDetector

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MLflow設定
MLFLOW_TRACKING_URI = "./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("concept_drift_detection")


def generate_dummy_data(num_samples: int = 100, drift_point: int = 70) -> pd.DataFrame:
    """
    コンセプトドリフトをシミュレートするためのダミーデータを生成します。
    drift_point以降でモデル性能が低下するように実際の値を調整します。
    """
    np.random.seed(42)
    predictions = np.random.rand(num_samples) * 100  # 0-100の予測値
    actuals = predictions.copy()  # 基本的に予測値と同じ

    # ドリフトをシミュレート: drift_point以降で実際の値にノイズを追加し、MAEを増加させる
    actuals[drift_point:] = (
        actuals[drift_point:] + np.random.rand(num_samples - drift_point) * 20 - 10
    )  # ノイズを追加
    actuals[drift_point:] = (
        actuals[drift_point:]
        + (predictions[drift_point:] - actuals[drift_point:]) * 0.5
    )  # 予測との乖離を大きくする

    # タイムスタンプを生成
    start_date = datetime.now() - timedelta(days=num_samples)
    dates = [start_date + timedelta(days=i) for i in range(num_samples)]

    df = pd.DataFrame(
        {"timestamp": dates, "predictions": predictions, "actuals": actuals}
    )
    return df


def run_concept_drift_detection(data_path: Optional[str] = None):
    """
    コンセプトドリフト検出を実行します。

    Args:
        data_path (Optional[str]): 予測と実際の値を含むCSVファイルのパス。指定しない場合はダミーデータを生成。
    """
    logger.info("コンセプトドリフト検出を開始します。")

    if data_path:
        try:
            df = pd.read_csv(data_path)
            logger.info(f"データファイルをロードしました: {data_path}")
        except FileNotFoundError:
            logger.error(
                f"指定されたデータファイルが見つかりません: {data_path}。ダミーデータを生成します。"
            )
            df = generate_dummy_data()
        except Exception as e:
            logger.error(
                f"データファイルのロード中にエラーが発生しました: {e}。ダミーデータを生成します。"
            )
            df = generate_dummy_data()
    else:
        df = generate_dummy_data()
        logger.info("ダミーデータを生成しました。")

    # ConceptDriftDetectorの初期化
    detector = ConceptDriftDetector(metric_threshold=0.1, window_size=10)

    with mlflow.start_run(run_name="concept_drift_check") as run:
        mlflow.log_param("metric_threshold", detector.metric_threshold)
        mlflow.log_param("window_size", detector.window_size)

        # データをチャンクに分割して処理（実際の運用を想定）
        chunk_size = 10
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size]
            predictions_chunk = chunk["predictions"].values
            actuals_chunk = chunk["actuals"].values
            timestamp_chunk = chunk["timestamp"].iloc[
                -1
            ]  # チャンクの最後のタイムスタンプ

            detector.add_performance_data(
                predictions_chunk, actuals_chunk, timestamp_chunk
            )
            drift_result = detector.detect_drift()

            # MLflowにログを記録
            mlflow.log_metrics(
                {
                    "latest_mae": drift_result.get("latest_mae", 0),
                    "baseline_mae": drift_result.get("baseline_mae", 0),
                    "mae_increase_ratio": drift_result.get("mae_increase_ratio", 0),
                    "drift_detected": 1 if drift_result.get("drift_detected") else 0,
                },
                step=i // chunk_size,
            )

            logger.info(f"ステップ {i // chunk_size}: ドリフト検出結果: {drift_result}")

            # ドリフトが検出されたら、アラートを生成したり、再訓練をトリガーしたりするロジックを追加
            if drift_result.get("drift_detected"):
                logger.critical("!!! コンセプトドリフトが検出されました !!!")
                mlflow.log_text("コンセプトドリフトが検出されました", "drift_alert.txt")
                # ここで再訓練ワークフローをトリガーするなどのアクションを検討

        summary = detector.get_performance_summary()
        mlflow.log_metrics(
            {
                "final_average_mae": summary.get("average_mae", 0),
                "final_max_mae": summary.get("max_mae", 0),
            }
        )
        mlflow.log_text(json.dumps(summary, indent=4), "performance_summary.json")

        logger.info("コンセプトドリフト検出が完了しました。")
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        logger.info(f"MLflow UI: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    # コマンドライン引数からデータパスを受け取ることも可能
    import argparse

    parser = argparse.ArgumentParser(description="Run concept drift detection.")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the CSV file containing predictions and actuals.",
        default=None,
    )
    args = parser.parse_args()

    run_concept_drift_detection(args.data_path)
