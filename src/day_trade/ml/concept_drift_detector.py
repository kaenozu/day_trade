import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConceptDriftDetector:
    """
    コンセプトドリフト検出器

    モデルの予測性能の経時的な変化を監視し、性能低下（コンセプトドリフト）を検出します。
    """

    def __init__(self, metric_threshold: float = 0.1, window_size: int = 30):
        """
        初期化

        Args:
            metric_threshold (float): 性能指標の低下を検出するための閾値 (例: 0.1 = 10%の低下)
            window_size (int): 性能指標の履歴を保持する期間 (データポイント数)
        """
        self.performance_history: List[Dict[str, Any]] = []
        self.metric_threshold = metric_threshold
        self.window_size = window_size
        self.baseline_metric: Optional[float] = None

    def add_performance_data(self, predictions: np.ndarray, actuals: np.ndarray, timestamp: datetime = datetime.now()):
        """
        新しい性能データを追加します。

        Args:
            predictions (np.ndarray): モデルの予測値
            actuals (np.ndarray): 実際の値（正解ラベル）
            timestamp (datetime): データが生成されたタイムスタンプ
        """
        if len(predictions) != len(actuals):
            logger.warning("予測値と実際の値の長さが一致しません。性能データをスキップします。")
            return
        if len(predictions) == 0:
            logger.warning("予測値または実際の値が空です。性能データをスキップします。")
            return

        # ここでは例としてMAEを計算
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

        self.performance_history.append({
            "timestamp": timestamp.isoformat(),
            "mae": mae,
            "rmse": rmse,
            "num_samples": len(predictions)
        })

        # 履歴を最新のwindow_sizeに保つ
        self.performance_history = self.performance_history[-self.window_size:]
        logger.info(f"性能データ追加: MAE={mae:.4f}, RMSE={rmse:.4f}, サンプル数={len(predictions)}")

    def detect_drift(self) -> Dict[str, Any]:
        """
        コンセプトドリフトを検出します。

        Returns:
            Dict[str, Any]: ドリフト検出結果
        """
        if len(self.performance_history) < 2:
            return {"drift_detected": False, "reason": "履歴データが不足しています"}

        # 最新の性能指標を取得
        latest_mae = self.performance_history[-1]["mae"]

        # ベースライン性能指標を設定（初回またはリセット時）
        if self.baseline_metric is None:
            # 最初の数データポイントの平均をベースラインとする
            if len(self.performance_history) >= 5:
                self.baseline_metric = np.mean([d["mae"] for d in self.performance_history[:5]])
                logger.info(f"コンセプトドリフト検出器: ベースラインMAEを設定しました: {self.baseline_metric:.4f}")
            else:
                return {"drift_detected": False, "reason": "ベースライン設定のための履歴データが不足しています"}

        # ベースラインからの性能低下をチェック
        # MAEは小さいほど良いので、増加をドリフトとみなす
        if self.baseline_metric > 0: # ゼロ除算回避
            mae_increase_ratio = (latest_mae - self.baseline_metric) / self.baseline_metric
        else:
            mae_increase_ratio = 0 # ベースラインが0の場合はドリフトなし

        drift_detected = mae_increase_ratio > self.metric_threshold

        result = {
            "drift_detected": drift_detected,
            "latest_mae": latest_mae,
            "baseline_mae": self.baseline_metric,
            "mae_increase_ratio": mae_increase_ratio,
            "threshold": self.metric_threshold,
            "history_length": len(self.performance_history)
        }

        if drift_detected:
            logger.warning(f"コンセプトドリフト検出: MAEがベースラインから {mae_increase_ratio:.2%} 増加しました。")
            result["reason"] = f"MAEが閾値 ({self.metric_threshold:.2%}) を超えて増加しました。"
        else:
            logger.info(f"コンセプトドリフトなし: MAE増加率 {mae_increase_ratio:.2%}")
            result["reason"] = "性能は安定しています"

        return result

    def reset_baseline(self):
        """
        ベースライン性能指標をリセットします。
        """
        self.baseline_metric = None
        logger.info("コンセプトドリフト検出器: ベースラインをリセットしました。")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        現在の性能履歴のサマリーを返します。
        """
        if not self.performance_history:
            return {"status": "no_data"}

        maes = [d["mae"] for d in self.performance_history]
        rmses = [d["rmse"] for d in self.performance_history]

        return {
            "latest_mae": maes[-1],
            "average_mae": np.mean(maes),
            "max_mae": np.max(maes),
            "min_mae": np.min(maes),
            "latest_rmse": rmses[-1],
            "average_rmse": np.mean(rmses),
            "history_length": len(self.performance_history),
            "baseline_mae": self.baseline_metric
        }

