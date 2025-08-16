import sqlite3
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

logger = logging.getLogger(__name__)

class IntegratedModelLoader:
    def __init__(self):
        self.upgrade_db_path = Path("ml_models_data/upgrade_system.db")
        self.ml_predictions_db_path = Path("ml_models_data/ml_predictions.db")
        self.advanced_ml_predictions_db_path = Path("ml_models_data/advanced_ml_predictions.db")
        self.model_integration_config = self._load_model_integration_config()
        self.loaded_models = {} # キャッシュされたモデル

        # 必要なモジュールを遅延インポート
        self.ml_prediction_models = None
        self.advanced_ml_prediction_system = None

    def _load_model_integration_config(self) -> Dict[str, str]:
        """モデル統合設定をデータベースから読み込む"""
        config = {}
        try:
            with sqlite3.connect(self.upgrade_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT symbol, preferred_system FROM model_integration_config")
                for row in cursor.fetchall():
                    config[row[0]] = row[1]
            logger.info(f"モデル統合設定を読み込みました: {config}")
        except Exception as e:
            logger.error(f"モデル統合設定の読み込みに失敗しました: {e}")
        return config

    async def _get_ml_prediction_models_instance(self):
        if self.ml_prediction_models is None:
            from ml_prediction_models import MLPredictionModels
            self.ml_prediction_models = MLPredictionModels()
        return self.ml_prediction_models

    async def _get_advanced_ml_prediction_system_instance(self):
        if self.advanced_ml_prediction_system is None:
            from advanced_ml_prediction_system import AdvancedMLPredictionSystem
            self.advanced_ml_prediction_system = AdvancedMLPredictionSystem()
        return self.advanced_ml_prediction_system

    async def get_model_for_symbol(self, symbol: str, task: Any): # taskはPredictionTaskまたはModelType
        """指定された銘柄とタスクに最適なモデルをロードまたは取得する"""
        preferred_system = self.model_integration_config.get(symbol, "original_system")
        model_key_prefix = f"{symbol}_{task.value}" # モデルキーのプレフィックス

        if preferred_system == "advanced_system":
            ml_system = await self._get_advanced_ml_prediction_system_instance()
            # AdvancedMLPredictionSystemは複数のモデルタイプを持つため、タスクとモデルタイプを組み合わせる
            # ここでは、ensemble_votingモデルを優先的にロードする
            model_key = f"{model_key_prefix}_ensemble_voting"
            model = ml_system.load_model(model_key)
            if model:
                logger.info(f"高度システムモデルをロードしました: {model_key}")
                return model, "advanced"
            else:
                logger.warning(f"高度システムモデル {model_key} が見つかりませんでした。オリジナルシステムを試行します。")
                preferred_system = "original_system" # フォールバック

        if preferred_system == "original_system":
            ml_system = await self._get_ml_prediction_models_instance()
            # MLPredictionModelsはRandomForest, XGBoost, LightGBMを持つ
            # ここでは、アンサンブル予測のためにすべてのモデルをロードする必要がある
            # または、特定のモデルタイプをロードする
            # 簡略化のため、ここではRandomForestをロードする例
            model_key = f"{model_key_prefix}_Random Forest" # RandomForestのモデルキー
            model = ml_system.load_model(model_key)
            if model:
                logger.info(f"オリジナルシステムモデルをロードしました: {model_key}")
                return model, "original"
            else:
                logger.warning(f"オリジナルシステムモデル {model_key} が見つかりませんでした。")
                return None, None

        return None, None

    async def predict(self, symbol: str, features: pd.DataFrame) -> Tuple[Any, str]:
        """
        指定された銘柄と特徴量で予測を実行する。
        統合設定に基づいて適切なモデルシステムを使用する。
        """
        preferred_system = self.model_integration_config.get(symbol, "original_system")

        if preferred_system == "advanced_system":
            ml_system = await self._get_advanced_ml_prediction_system_instance()
            # AdvancedMLPredictionSystemのpredict_with_advanced_modelsを呼び出す
            prediction_result = await ml_system.predict_with_advanced_models(symbol)
            return prediction_result, "advanced"
        elif preferred_system == "original_system":
            ml_system = await self._get_ml_prediction_models_instance()
            # MLPredictionModelsのpredictを呼び出す
            prediction_result = await ml_system.predict(symbol, features)
            return prediction_result, "original"
        else:
            logger.warning(f"銘柄 {symbol} の推奨システムが見つかりません。オリジナルシステムで予測します。")
            ml_system = await self._get_ml_prediction_models_instance()
            prediction_result = await ml_system.predict(symbol, features)
            return prediction_result, "original"

# テスト実行
async def test_integrated_model_loader():
    logger.info("IntegratedModelLoader テスト開始")
    loader = IntegratedModelLoader()

    # テスト用のモデル統合設定を挿入 (手動でDBに設定)
    # 例:
    # INSERT OR REPLACE INTO model_integration_config (symbol, preferred_system, updated_at) VALUES ('7203', 'advanced_system', '...');
    # INSERT OR REPLACE INTO model_integration_config (symbol, preferred_system, updated_at) VALUES ('8306', 'original_system', '...');

    # ダミーの予測データ
    dummy_features = pd.DataFrame([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]], columns=[f'feature_{i}' for i in range(10)])

    # 予測テスト
    symbol_to_test = "7203" # advanced_systemが設定されていると仮定
    prediction, system_used = await loader.predict(symbol_to_test, dummy_features)
    logger.info(f"銘柄 {symbol_to_test} の予測結果: {prediction}, 使用システム: {system_used}")

    symbol_to_test = "8306" # original_systemが設定されていると仮定
    prediction, system_used = await loader.predict(symbol_to_test, dummy_features)
    logger.info(f"銘柄 {symbol_to_test} の予測結果: {prediction}, 使用システム: {system_used}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(test_integrated_model_loader())