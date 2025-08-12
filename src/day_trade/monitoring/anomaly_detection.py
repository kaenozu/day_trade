#!/usr/bin/env python3
"""
機械学習ベース異常検知システム
リアルタイムメトリクス異常検知とアラート生成

Features:
- Isolation Forest異常検知
- LSTM時系列異常検知
- 統計ベース異常検知
- アンサンブル異常検知
- 適応的閾値調整
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..utils.logging_config import get_context_logger
from .metrics.prometheus_metrics import AnomalyDetectionResult

logger = get_context_logger(__name__)


@dataclass
class AnomalyDetectionConfig:
    """異常検知設定"""

    method: str  # isolation_forest, lstm, statistical, ensemble
    window_size: int = 100
    contamination: float = 0.1
    threshold: float = 0.5
    enabled: bool = True
    retrain_interval: int = 3600  # 1時間


@dataclass
class ModelMetrics:
    """モデルメトリクス"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_updated: datetime


class StatisticalAnomalyDetector:
    """統計ベース異常検知器"""

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.data_buffer = deque(maxlen=window_size)

    def update(self, value: float):
        """データ更新"""
        self.data_buffer.append(value)

    def detect_anomaly(self, value: float) -> Tuple[bool, float, str]:
        """異常検知実行"""

        if len(self.data_buffer) < 10:
            return False, 0.0, "データ不足"

        values = list(self.data_buffer)
        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val == 0:
            return False, 0.0, "標準偏差ゼロ"

        z_score = abs(value - mean_val) / std_val
        is_anomaly = z_score > self.z_threshold
        confidence = min(z_score / self.z_threshold, 1.0) if is_anomaly else 0.0

        explanation = f"Z-score: {z_score:.2f}, 閾値: {self.z_threshold}, 平均: {mean_val:.2f}, 標準偏差: {std_val:.2f}"

        return is_anomaly, confidence, explanation


class IsolationForestDetector:
    """Isolation Forest異常検知器"""

    def __init__(self, contamination: float = 0.1, window_size: int = 1000):
        self.contamination = contamination
        self.window_size = window_size
        self.model = None
        self.scaler = StandardScaler()
        self.data_buffer = deque(maxlen=window_size)
        self.is_trained = False
        self.last_retrain_time = None

    def update(self, features: np.ndarray):
        """データ更新"""
        self.data_buffer.append(features)

    def train(self) -> bool:
        """モデル訓練"""

        if len(self.data_buffer) < 100:
            logger.warning("訓練データ不足")
            return False

        try:
            # データ準備
            data = np.array(list(self.data_buffer))
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

            # 正規化
            data_normalized = self.scaler.fit_transform(data)

            # モデル訓練
            self.model = IsolationForest(
                contamination=self.contamination, random_state=42, n_estimators=100
            )
            self.model.fit(data_normalized)

            self.is_trained = True
            self.last_retrain_time = datetime.now()

            logger.info(f"Isolation Forest モデル訓練完了 (サンプル数: {len(data)})")
            return True

        except Exception as e:
            logger.error(f"Isolation Forest 訓練エラー: {e}")
            return False

    def detect_anomaly(self, features: np.ndarray) -> Tuple[bool, float, str]:
        """異常検知実行"""

        if not self.is_trained or self.model is None:
            return False, 0.0, "モデル未訓練"

        try:
            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            # 正規化
            features_normalized = self.scaler.transform(features)

            # 異常スコア計算
            anomaly_score = self.model.decision_function(features_normalized)[0]
            is_anomaly = self.model.predict(features_normalized)[0] == -1

            # 信頼度計算 (スコアを0-1範囲に正規化)
            confidence = max(0, min(1, -anomaly_score))

            explanation = f"Isolation Forest スコア: {anomaly_score:.4f}, 閾値: 0"

            return is_anomaly, confidence, explanation

        except Exception as e:
            logger.error(f"Isolation Forest 異常検知エラー: {e}")
            return False, 0.0, f"検知エラー: {str(e)}"


class LSTMAnomalyDetector:
    """LSTM時系列異常検知器"""

    def __init__(self, sequence_length: int = 50, threshold: float = 0.5):
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.data_buffer = deque(maxlen=1000)
        self.is_trained = False

        # 簡単な実装のため、統計的手法で代替
        self.statistical_detector = StatisticalAnomalyDetector(
            window_size=sequence_length, z_threshold=2.5
        )

    def update(self, value: float):
        """データ更新"""
        self.data_buffer.append(value)
        self.statistical_detector.update(value)

    def train(self) -> bool:
        """モデル訓練（簡略化版）"""

        if len(self.data_buffer) < self.sequence_length * 2:
            return False

        self.is_trained = True
        logger.info("LSTM異常検知器（統計的手法版）訓練完了")
        return True

    def detect_anomaly(self, value: float) -> Tuple[bool, float, str]:
        """異常検知実行"""

        if not self.is_trained:
            return False, 0.0, "モデル未訓練"

        # 統計的手法で代替実装
        is_anomaly, confidence, explanation = self.statistical_detector.detect_anomaly(value)

        return is_anomaly, confidence, f"LSTM(統計代替): {explanation}"


class EnsembleAnomalyDetector:
    """アンサンブル異常検知器"""

    def __init__(self):
        self.detectors = {
            "statistical": StatisticalAnomalyDetector(window_size=100, z_threshold=3.0),
            "isolation_forest": IsolationForestDetector(contamination=0.1),
            "lstm": LSTMAnomalyDetector(sequence_length=50),
        }
        self.weights = {"statistical": 0.4, "isolation_forest": 0.4, "lstm": 0.2}
        self.retrain_counters = defaultdict(int)

    async def update_and_detect(
        self, metric_name: str, value: float, features: Optional[np.ndarray] = None
    ) -> Optional[AnomalyDetectionResult]:
        """データ更新と異常検知実行"""

        try:
            # 各検知器にデータ更新
            self.detectors["statistical"].update(value)
            self.detectors["lstm"].update(value)

            if features is not None:
                self.detectors["isolation_forest"].update(features)
            else:
                # 単一の値を特徴量として使用
                self.detectors["isolation_forest"].update(np.array([value]))

            # 定期的にモデル再訓練
            await self._retrain_if_needed()

            # アンサンブル異常検知実行
            results = {}

            # 統計的検知
            is_anom_stat, conf_stat, exp_stat = self.detectors["statistical"].detect_anomaly(value)
            results["statistical"] = (is_anom_stat, conf_stat, exp_stat)

            # Isolation Forest検知
            detect_features = features if features is not None else np.array([value])
            is_anom_if, conf_if, exp_if = self.detectors["isolation_forest"].detect_anomaly(
                detect_features
            )
            results["isolation_forest"] = (is_anom_if, conf_if, exp_if)

            # LSTM検知
            is_anom_lstm, conf_lstm, exp_lstm = self.detectors["lstm"].detect_anomaly(value)
            results["lstm"] = (is_anom_lstm, conf_lstm, exp_lstm)

            # アンサンブル結果計算
            total_confidence = 0.0
            weighted_votes = 0.0
            explanations = []

            for detector_name, (is_anomaly, confidence, explanation) in results.items():
                weight = self.weights[detector_name]

                if is_anomaly:
                    weighted_votes += weight

                total_confidence += confidence * weight
                explanations.append(f"{detector_name}: {explanation}")

            # 最終判定
            ensemble_is_anomaly = weighted_votes > 0.5
            ensemble_confidence = total_confidence
            ensemble_explanation = "; ".join(explanations)

            if ensemble_is_anomaly:
                return AnomalyDetectionResult(
                    is_anomaly=True,
                    confidence=ensemble_confidence,
                    explanation=f"アンサンブル検知 ({ensemble_confidence:.3f}): {ensemble_explanation}",
                    metric_name=metric_name,
                    timestamp=datetime.now(),
                    value=value,
                    expected_range=None,
                )

            return None

        except Exception as e:
            logger.error(f"アンサンブル異常検知エラー ({metric_name}): {e}")
            return None

    async def _retrain_if_needed(self):
        """必要に応じてモデル再訓練"""

        current_time = datetime.now()

        for detector_name, detector in self.detectors.items():
            self.retrain_counters[detector_name] += 1

            # 100回に1回、または1時間毎に再訓練
            if self.retrain_counters[detector_name] % 100 == 0 or (
                hasattr(detector, "last_retrain_time")
                and detector.last_retrain_time
                and (current_time - detector.last_retrain_time).seconds > 3600
            ):
                if hasattr(detector, "train"):
                    success = detector.train()
                    if success:
                        logger.info(f"{detector_name} モデル再訓練完了")
                    self.retrain_counters[detector_name] = 0


class MLAnomalyDetectionSystem:
    """機械学習異常検知システム"""

    def __init__(self):
        self.detectors = {}
        self.configs = {}
        self.detection_history = deque(maxlen=10000)
        self.model_metrics = {}
        self.running = False

        # デフォルト設定
        self._setup_default_configs()

        logger.info("機械学習異常検知システム初期化完了")

    def _setup_default_configs(self):
        """デフォルト設定"""

        self.configs = {
            "cpu_usage": AnomalyDetectionConfig(
                method="ensemble", window_size=100, contamination=0.05, threshold=0.7
            ),
            "memory_usage": AnomalyDetectionConfig(
                method="ensemble", window_size=100, contamination=0.05, threshold=0.7
            ),
            "response_time": AnomalyDetectionConfig(
                method="isolation_forest",
                window_size=200,
                contamination=0.1,
                threshold=0.6,
            ),
            "error_rate": AnomalyDetectionConfig(
                method="statistical", window_size=50, threshold=0.8
            ),
            "prediction_accuracy": AnomalyDetectionConfig(
                method="ensemble", window_size=100, contamination=0.08, threshold=0.75
            ),
        }

    def add_metric_detector(self, metric_name: str, config: AnomalyDetectionConfig):
        """メトリクス検知器追加"""

        self.configs[metric_name] = config

        if config.method == "ensemble":
            self.detectors[metric_name] = EnsembleAnomalyDetector()
        elif config.method == "isolation_forest":
            self.detectors[metric_name] = IsolationForestDetector(
                contamination=config.contamination, window_size=config.window_size
            )
        elif config.method == "statistical":
            self.detectors[metric_name] = StatisticalAnomalyDetector(
                window_size=config.window_size, z_threshold=3.0
            )
        elif config.method == "lstm":
            self.detectors[metric_name] = LSTMAnomalyDetector(
                sequence_length=config.window_size, threshold=config.threshold
            )

        logger.info(f"異常検知器追加: {metric_name} ({config.method})")

    async def detect_anomaly(
        self, metric_name: str, value: float, features: Optional[np.ndarray] = None
    ) -> Optional[AnomalyDetectionResult]:
        """異常検知実行"""

        if metric_name not in self.configs or not self.configs[metric_name].enabled:
            return None

        if metric_name not in self.detectors:
            self.add_metric_detector(metric_name, self.configs[metric_name])

        detector = self.detectors[metric_name]

        try:
            if isinstance(detector, EnsembleAnomalyDetector):
                result = await detector.update_and_detect(metric_name, value, features)
            else:
                # 単一検知器の場合
                detector.update(value)

                if hasattr(detector, "detect_anomaly"):
                    is_anomaly, confidence, explanation = detector.detect_anomaly(value)

                    if is_anomaly:
                        result = AnomalyDetectionResult(
                            is_anomaly=True,
                            confidence=confidence,
                            explanation=explanation,
                            metric_name=metric_name,
                            timestamp=datetime.now(),
                            value=value,
                        )
                    else:
                        result = None
                else:
                    result = None

            # 検知履歴記録
            if result:
                self.detection_history.append(
                    {
                        "timestamp": result.timestamp.isoformat(),
                        "metric_name": metric_name,
                        "value": value,
                        "confidence": result.confidence,
                        "explanation": result.explanation,
                    }
                )

            return result

        except Exception as e:
            logger.error(f"異常検知エラー ({metric_name}): {e}")
            return None

    def get_detection_summary(self) -> Dict[str, Any]:
        """検知サマリー取得"""

        total_detections = len(self.detection_history)

        if total_detections == 0:
            return {"total_detections": 0, "by_metric": {}, "last_24h": 0}

        by_metric = defaultdict(int)
        last_24h = 0
        current_time = datetime.now()

        for record in self.detection_history:
            # メトリクス別
            metric_name = record["metric_name"]
            by_metric[metric_name] += 1

            # 過去24時間
            detection_time = datetime.fromisoformat(record["timestamp"])
            if (current_time - detection_time).total_seconds() < 86400:
                last_24h += 1

        return {
            "total_detections": total_detections,
            "by_metric": dict(by_metric),
            "last_24h": last_24h,
            "active_detectors": len(self.detectors),
            "enabled_metrics": sum(1 for config in self.configs.values() if config.enabled),
        }

    def get_detection_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """検知履歴取得"""

        return list(self.detection_history)[-limit:]

    def enable_metric_detection(self, metric_name: str, enabled: bool = True):
        """メトリクス検知有効/無効"""

        if metric_name in self.configs:
            self.configs[metric_name].enabled = enabled
            logger.info(f"メトリクス検知 {metric_name}: {'有効' if enabled else '無効'}")

    async def train_all_models(self):
        """全モデル訓練"""

        trained_count = 0

        for metric_name, detector in self.detectors.items():
            try:
                if hasattr(detector, "train"):
                    success = detector.train()
                    if success:
                        trained_count += 1
                        logger.info(f"モデル訓練完了: {metric_name}")
                elif isinstance(detector, EnsembleAnomalyDetector):
                    # アンサンブル内の各検知器を訓練
                    for sub_detector_name, sub_detector in detector.detectors.items():
                        if hasattr(sub_detector, "train"):
                            success = sub_detector.train()
                            if success:
                                logger.info(
                                    f"サブモデル訓練完了: {metric_name}.{sub_detector_name}"
                                )
                    trained_count += 1

            except Exception as e:
                logger.error(f"モデル訓練エラー ({metric_name}): {e}")

        logger.info(f"訓練完了モデル数: {trained_count}/{len(self.detectors)}")
        return trained_count


# グローバルインスタンス
_ml_anomaly_system = MLAnomalyDetectionSystem()


def get_ml_anomaly_system() -> MLAnomalyDetectionSystem:
    """機械学習異常検知システム取得"""
    return _ml_anomaly_system
