import numpy as np
from collections import deque
import json
from pathlib import Path

class ModelPerformanceMonitor:
    """
    モデルの予測性能をリアルタイムで監視するコンポーネント。
    """
    def __init__(self, window_size: int = 100):
        """
        コンストラクタ。

        Args:
            window_size (int): 性能を計算するための過去のデータポイント数。
        """
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.window_size = window_size
        self._load_config()

    def _load_config(self):
        """
        ml.jsonから性能監視の閾値設定を読み込みます。
        """
        config_path = Path(__file__).parent.parent / "config" / "ml.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            perf_config = config.get("machine_learning", {}).get("performance_monitoring", {})
            self.enabled = perf_config.get("enabled", False)
            self.accuracy_warning_threshold = perf_config.get("accuracy_warning_threshold", 0.85)
            self.accuracy_retrain_threshold = perf_config.get("accuracy_retrain_threshold", 0.75)
            self.min_samples_for_evaluation = perf_config.get("min_samples_for_evaluation", 50)
            print(f"[OK] ModelPerformanceMonitor: 設定を読み込みました (警告閾値: {self.accuracy_warning_threshold}, 再学習閾値: {self.accuracy_retrain_threshold})")
        except Exception as e:
            print(f"[WARNING] ModelPerformanceMonitor: 設定の読み込みに失敗しました: {e}. デフォルト値を使用します。")
            self.enabled = False
            self.accuracy_warning_threshold = 0.85
            self.accuracy_retrain_threshold = 0.75
            self.min_samples_for_evaluation = 50

    def record_prediction(self, predicted_value: float, actual_value: float):
        """
        予測結果と実際の値を記録します。

        Args:
            predicted_value (float): モデルが予測した値。
            actual_value (float): 実際の値。
        """
        self.predictions.append(predicted_value)
        self.actuals.append(actual_value)

    def get_accuracy(self) -> float:
        """
        現在の記録に基づいた予測精度を計算して返します。
        バイナリ分類（0または1）を想定しています。

        Returns:
            float: 予測精度 (0.0から1.0)。データがない場合は0.0。
        """
        if not self.predictions:
            return 0.0

        correct_predictions = 0
        for pred, actual in zip(self.predictions, self.actuals):
            # 簡単のため、予測と実際の値が一致するかどうかで精度を判断
            # 実際の取引システムでは、閾値や方向性の一致など、より複雑なロジックが必要
            if round(pred) == round(actual): # 丸めてバイナリとして比較
                correct_predictions += 1
        return correct_predictions / len(self.predictions)

    def get_metrics(self) -> dict:
        """
        現在の性能メトリクスを辞書形式で返します。

        Returns:
            dict: 性能メトリクスを含む辞書。
        """
        accuracy = self.get_accuracy()
        # 今後、他のメトリクス（例: Precision, Recall, F1-score, RMSEなど）を追加可能
        return {
            "accuracy": accuracy,
            "num_samples": len(self.predictions)
        }

    def check_performance_status(self) -> dict:
        """
        現在のモデル性能が設定された閾値を満たしているかチェックします。

        Returns:
            dict: 性能ステータスと現在のメトリクスを含む辞書。
                  例: {"status": "OK", "current_accuracy": 0.92, "num_samples": 100}
        """
        if not self.enabled:
            return {"status": "DISABLED", "current_accuracy": 0.0, "num_samples": 0}

        current_metrics = self.get_metrics()
        current_accuracy = current_metrics["accuracy"]
        num_samples = current_metrics["num_samples"]

        if num_samples < self.min_samples_for_evaluation:
            return {"status": "INSUFFICIENT_SAMPLES", "current_accuracy": current_accuracy, "num_samples": num_samples}

        if current_accuracy < self.accuracy_retrain_threshold:
            status = "CRITICAL_RETRAIN"
        elif current_accuracy < self.accuracy_warning_threshold:
            status = "WARNING"
        else:
            status = "OK"

        return {"status": status, "current_accuracy": current_accuracy, "num_samples": num_samples}

# 使用例 (テスト用)
if __name__ == "__main__":
    monitor = ModelPerformanceMonitor(window_size=10)

    # ダミーデータで記録
    monitor.record_prediction(0.9, 1.0) # 正解
    monitor.record_prediction(0.1, 0.0) # 正解
    monitor.record_prediction(0.8, 0.0) # 不正解
    monitor.record_prediction(0.6, 1.0) # 不正解
    monitor.record_prediction(0.95, 1.0) # 正解
    monitor.record_prediction(0.05, 0.0) # 正解
    monitor.record_prediction(0.7, 1.0) # 不正解
    monitor.record_prediction(0.2, 0.0) # 正解
    monitor.record_prediction(0.85, 1.0) # 正解
    monitor.record_prediction(0.15, 0.0) # 正解

    metrics = monitor.get_metrics()
    print(f"現在のモデル性能: {metrics}")
    print(f"警告閾値: {monitor.accuracy_warning_threshold}")
    print(f"再学習閾値: {monitor.accuracy_retrain_threshold}")
    print(f"最小評価サンプル数: {monitor.min_samples_for_evaluation}")

    status_report = monitor.check_performance_status()
    print(f"性能ステータス: {status_report}")

    monitor.record_prediction(0.55, 1.0) # 新しいデータ追加
    status_report = monitor.check_performance_status()
    print(f"新しいデータ追加後の性能ステータス: {status_report}")