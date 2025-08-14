import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path
import logging

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
    def __init__(self):
        self.upgrade_db_path = Path("ml_models_data/upgrade_system.db")
        self.advanced_ml_db_path = Path("ml_models_data/advanced_ml_predictions.db")
        self.thresholds = {}
        self.load_thresholds()

    def load_thresholds(self):
        """データベースから性能閾値を読み込む"""
        try:
            with sqlite3.connect(self.upgrade_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT metric_name, threshold_value FROM performance_thresholds")
                for row in cursor.fetchall():
                    self.thresholds[row[0]] = row[1]
            logger.info(f"性能閾値を読み込みました: {self.thresholds}")
        except Exception as e:
            logger.error(f"性能閾値の読み込みに失敗しました: {e}")
            # デフォルト閾値
            self.thresholds = {"accuracy": 0.90} # 例: 精度90%を下回ったらトリガー

    async def get_latest_model_performance(self) -> dict:
        """最新のモデル性能を取得する"""
        try:
            with sqlite3.connect(self.advanced_ml_db_path) as conn:
                cursor = conn.cursor()
                # 最新のアンサンブルモデルの精度を取得
                cursor.execute("""
                    SELECT accuracy FROM advanced_model_performances
                    WHERE model_type = 'ensemble_voting'
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                )
                result = cursor.fetchone()
                if result:
                    logger.info(f"最新のアンサンブルモデル精度: {result[0]}")
                    return {"accuracy": result[0]}
                else:
                    logger.warning("最新のアンサンブルモデル性能が見つかりませんでした。")
                    return {}
        except Exception as e:
            logger.error(f"最新モデル性能の取得に失敗しました: {e}")
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
            from ml_model_upgrade_system import ml_upgrade_system
            report, integration_results = await ml_upgrade_system.run_complete_system_upgrade()
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
                           ("accuracy", 0.93, datetime.now().isoformat()))
            conn.commit()
        logger.info("テスト閾値を設定しました。")
    except Exception as e:
            logger.error(f"テスト閾値の設定に失敗しました: {e}")

    await monitor.check_and_trigger_retraining()

if __name__ == "__main__":
    asyncio.run(test_monitor())
