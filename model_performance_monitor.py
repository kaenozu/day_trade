import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path
import logging
from ml_model_upgrade_system import ml_upgrade_system # Added import
from typing import Optional # Added import
from prediction_accuracy_validator import PredictionAccuracyValidator # Added import

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

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPerformanceMonitor:
    def __init__(self, upgrade_db_path: Optional[Path] = None, advanced_ml_db_path: Optional[Path] = None):
        self.upgrade_db_path = upgrade_db_path or Path("ml_models_data/upgrade_system.db")
        self.advanced_ml_db_path = advanced_ml_db_path or Path("ml_models_data/advanced_ml_predictions.db")
        self.accuracy_validator = PredictionAccuracyValidator() # Initialize the validator
        self.thresholds = {}
        self._init_database() # Call to initialize database
        self.load_thresholds()

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.upgrade_db_path) as conn:
            cursor = conn.cursor()
            # 性能閾値テーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_thresholds (
                    metric_name TEXT PRIMARY KEY,
                    threshold_value REAL,
                    last_updated TEXT
                )
            ''')
            conn.commit()

    def load_thresholds(self):
        """データベースから性能閾値を読み込む"""
        # デフォルト閾値で初期化
        self.thresholds = {"accuracy": 0.90} # 例: 精度90%を下回ったらトリガー

        try:
            with sqlite3.connect(self.upgrade_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT metric_name, threshold_value FROM performance_thresholds")
                for row in cursor.fetchall():
                    self.thresholds[row[0]] = row[1] # DBの値で上書き
            logger.info(f"性能閾値を読み込みました: {self.thresholds}")
        except Exception as e:
            logger.error(f"性能閾値の読み込みに失敗しました: {e}")
            # 失敗した場合はデフォルト値がそのまま使われる

    async def get_latest_model_performance(self) -> dict:
        """最新のモデル性能を取得する (PredictionAccuracyValidatorを使用)"""
        logger.info("PredictionAccuracyValidatorを使用して最新のモデル性能を取得します。")
        # ここでは評価対象のシンボルを仮に設定。必要に応じて外部から与えるか、設定ファイルから読み込む
        test_symbols = ["7203", "8306", "4751"]
        validation_hours = 24 * 7 # 例: 過去7日間のデータで評価

        try:
            metrics = await self.accuracy_validator.validate_current_system_accuracy(test_symbols, validation_hours)
            overall_accuracy = metrics.overall_accuracy # 0-100%の値を直接使用

            logger.info(f"PredictionAccuracyValidatorによる最新の総合精度: {overall_accuracy:.3f}")
            return {"accuracy": overall_accuracy}
        except Exception as e:
            logger.error(f"PredictionAccuracyValidatorによるモデル性能の取得に失敗しました: {e}")
            return {}

    async def check_and_trigger_retraining(self):
        """性能をチェックし、閾値を下回ったら再学習をトリガーする"""
        logger.info("モデル性能監視を開始します。")
        latest_performance = await self.get_latest_model_performance()

        if not latest_performance:
            logger.warning("性能データがないため、再学習トリガーチェックをスキップします。")
            return

        # 精度閾値チェック
        if "accuracy" in self.thresholds and "accuracy" in latest_performance:
            current_accuracy = latest_performance["accuracy"]
            threshold_accuracy = self.thresholds["accuracy"]

            logger.info(f"現在の精度: {current_accuracy:.3f}, 閾値: {threshold_accuracy:.3f}")

            if current_accuracy < threshold_accuracy:
                logger.warning(f"モデル精度が閾値を下回りました ({current_accuracy:.3f} < {threshold_accuracy:.3f})。再学習をトリガーします。")
                await self.trigger_retraining()
            else:
                logger.info("モデル精度は閾値以上です。")
        else:
            logger.info("精度閾値または性能データが設定されていないため、精度チェックをスキップします。")

    async def trigger_retraining(self):
        """再学習プロセスをトリガーする"""
        logger.info("再学習プロセスをトリガーしています...")
        try:
            report = await ml_upgrade_system.run_complete_system_upgrade()
            integration_results = await ml_upgrade_system.integrate_best_models(report)
            logger.info("再学習プロセスが完了しました。")
            logger.info(f"アップグレードレポート: {report.overall_improvement:.2f}% 改善")
            logger.info(f"統合結果: {integration_results}")
        except Exception as e:
            logger.error(f"再学習プロセスのトリガーに失敗しました: {e}")

# テスト実行
async def test_monitor():
    monitor = ModelPerformanceMonitor()
    # 初期閾値を設定（テスト用）
    try:
        with sqlite3.connect(monitor.upgrade_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO performance_thresholds (metric_name, threshold_value, last_updated) VALUES (?, ?, ?)",
                           ("accuracy", 93.0, datetime.now().isoformat()))
            conn.commit()
        logger.info("テスト閾値を設定しました。")
    except Exception as e:
            logger.error(f"テスト閾値の設定に失敗しました: {e}")

    await monitor.check_and_trigger_retraining()

if __name__ == "__main__":
    asyncio.run(test_monitor())