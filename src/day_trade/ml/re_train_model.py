import pandas as pd
import numpy as np
import logging
import mlflow
import mlflow.pytorch
from pathlib import Path
from datetime import datetime

# プロジェクト内モジュール
from advanced_ml_engine import AdvancedMLEngine, ModelConfig

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow設定
MLFLOW_TRACKING_URI = "./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("model_re_training")

def generate_dummy_training_data(num_samples: int = 1000) -> pd.DataFrame:
    """
    モデル訓練用のダミーデータを生成します。
    """
    np.random.seed(int(datetime.now().timestamp())) # 毎回異なるデータを生成
    data = {
        'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_samples)),
        'Open': np.random.rand(num_samples) * 100,
        'High': np.random.rand(num_samples) * 100 + 5,
        'Low': np.random.rand(num_samples) * 100 - 5,
        'Close': np.random.rand(num_samples) * 100,
        'Volume': np.random.randint(1000, 5000, num_samples)
    }
    df = pd.DataFrame(data)
    return df

def run_model_re_training():
    """
    モデルの再訓練を実行し、MLflowにログを記録します。
    """
    logger.info("モデルの再訓練を開始します。")

    with mlflow.start_run(run_name=f"re_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        # モデル設定
        model_config = ModelConfig(
            sequence_length=100, # 例: 過去100日間のデータを使用
            num_features=6,      # 例: Open, High, Low, Close, Volume, Date (数値変換後)
            prediction_horizon=1 # 例: 1日後の予測
        )
        ml_engine = AdvancedMLEngine(config=model_config)

        # ダミーデータの生成
        df_train = generate_dummy_training_data()
        
        # 訓練データの準備
        # advanced_ml_engine.pyのprepare_dataは、内部で特徴量エンジニアリングとシーケンス作成を行う
        # ここでは、ダミーデータフレームをそのまま渡す
        # 実際のデータでは、StockFetcherなどから取得したデータフレームを渡す
        try:
            X_train, y_train = ml_engine.prepare_data(df_train, target_column='Close', feature_columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            logger.info(f"訓練データ準備完了: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        except Exception as e:
            logger.error(f"訓練データ準備中にエラーが発生しました: {e}")
            mlflow.log_param("training_status", "failed_data_prep")
            return

        # モデル訓練
        try:
            training_results = ml_engine.train_model(X_train, y_train)
            logger.info(f"モデル訓練完了: {training_results}")
            mlflow.log_param("training_status", "success")
            mlflow.log_metrics(training_results)

            # モデルの保存とMLflowへの登録
            model_path = "./trained_model"
            ml_engine._save_model(model_path + "_final.pth") # 内部メソッドを直接呼び出し
            mlflow.pytorch.log_model(ml_engine.model, "model", registered_model_name="AdvancedMLModel")
            logger.info("訓練済みモデルをMLflowに登録しました。")

        except Exception as e:
            logger.error(f"モデル訓練中にエラーが発生しました: {e}")
            mlflow.log_param("training_status", "failed_training")

        logger.info("モデルの再訓練プロセスが完了しました。")
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        logger.info(f"MLflow UI: {MLFLOW_TRACKING_URI}")

if __name__ == "__main__":
    run_model_re_training()
