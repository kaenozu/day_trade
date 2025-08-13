#!/usr/bin/env python3
"""
Base Model Interface for Ensemble Learning

統一されたインターフェースによりアンサンブル学習での
異なるモデルの管理を簡素化
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class ModelPrediction:
    """モデル予測結果"""
    predictions: np.ndarray
    confidence: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_name: str = "unknown"
    processing_time: float = 0.0


@dataclass
class ModelMetrics:
    """モデル評価指標"""
    mse: float
    rmse: float
    mae: float
    r2_score: float
    hit_rate: float  # 方向性予測精度
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None


class BaseModelInterface(ABC):
    """
    すべてのベースモデルが実装すべき統一インターフェース

    アンサンブル学習での一貫性を保つため、すべてのモデルが
    この基底クラスを継承し、共通のメソッドを実装する必要がある
    """

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            model_name: モデル名
            config: モデル設定辞書
        """
        self.model_name = model_name
        self.config = config or {}
        self.is_trained = False
        self.model = None
        self.training_metrics = {}
        self.feature_names = []

        logger.info(f"{model_name}モデル初期化")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        モデル学習

        Args:
            X: 訓練データの特徴量
            y: 訓練データの目標変数
            validation_data: 検証データ (X_val, y_val)

        Returns:
            学習結果辞書（メトリクス、学習時間等）
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        予測実行

        Args:
            X: 予測対象の特徴量

        Returns:
            ModelPrediction: 予測結果とメタ情報
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度取得

        Returns:
            特徴量名と重要度のマッピング
        """
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """
        モデル評価

        Args:
            X: 評価データの特徴量
            y: 評価データの目標変数

        Returns:
            ModelMetrics: 評価指標
        """
        try:
            prediction = self.predict(X)
            y_pred = prediction.predictions

            # 基本指標計算
            mse = np.mean((y - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y - y_pred))

            # R²スコア
            y_mean = np.mean(y)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            # Hit Rate（方向性予測精度）- Issue #492対応
            hit_rate = self._calculate_hit_rate(y, y_pred)

            return ModelMetrics(
                mse=mse,
                rmse=rmse,
                mae=mae,
                r2_score=r2_score,
                hit_rate=hit_rate
            )

        except Exception as e:
            logger.error(f"{self.model_name}評価エラー: {e}", exc_info=True)

            # エラーの詳細情報をログ出力 - Issue #492対応
            logger.error(
                f"評価データ情報 - X: {X.shape if hasattr(X, 'shape') else type(X)}, "
                f"y: {y.shape if hasattr(y, 'shape') else type(y)}"
            )

            # エラー時はデフォルト値を返す
            return ModelMetrics(
                mse=float('inf'),
                rmse=float('inf'),
                mae=float('inf'),
                r2_score=-1.0,
                hit_rate=0.5
            )

    def save_model(self, filepath: str) -> bool:
        """
        モデル保存

        Args:
            filepath: 保存先パス

        Returns:
            保存成功フラグ
        """
        try:
            import pickle

            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'config': self.config,
                'is_trained': self.is_trained,
                'training_metrics': self.training_metrics,
                'feature_names': self.feature_names
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"{self.model_name}モデル保存完了: {filepath}")
            return True

        except Exception as e:
            logger.error(f"{self.model_name}モデル保存エラー: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        モデル読み込み

        Args:
            filepath: 読み込み元パス

        Returns:
            読み込み成功フラグ
        """
        try:
            import pickle

            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data.get('model')
            self.model_name = model_data.get('model_name', self.model_name)
            self.config = model_data.get('config', {})
            self.is_trained = model_data.get('is_trained', False)
            self.training_metrics = model_data.get('training_metrics', {})
            self.feature_names = model_data.get('feature_names', [])

            logger.info(f"{self.model_name}モデル読み込み完了: {filepath}")
            return True

        except Exception as e:
            logger.error(f"{self.model_name}モデル読み込みエラー: {e}")
            return False

    def _calculate_hit_rate(self, y_true: np.ndarray, y_pred: np.ndarray,
                           min_samples: int = 2, zero_threshold: float = 1e-6) -> float:
        """
        改善されたHit Rate（方向性予測精度）計算 - Issue #492対応

        Args:
            y_true: 実際の値
            y_pred: 予測値
            min_samples: hit_rate計算に必要な最小サンプル数
            zero_threshold: 0値と判定する閾値

        Returns:
            hit_rate: 方向性予測精度（0.0-1.0）

        Note:
            - データ数が少ない場合は適切なデフォルト値を返す
            - np.sign()の0値を適切に処理
            - 連続する変化量での方向性一致率を計算
        """
        try:
            # データが不十分な場合
            if len(y_true) < min_samples or len(y_pred) < min_samples:
                logger.warning(
                    f"Hit rate計算: データ不足 (実際: {len(y_true)}, 予測: {len(y_pred)}, "
                    f"必要: {min_samples}) - デフォルト値0.5を返す"
                )
                return 0.5

            # 前日比（変化量）を計算
            y_diff = np.diff(y_true)
            pred_diff = np.diff(y_pred)

            # 変化量データが不十分な場合
            if len(y_diff) == 0 or len(pred_diff) == 0:
                logger.warning("Hit rate計算: 変化量データなし - デフォルト値0.5を返す")
                return 0.5

            # 方向性を判定（0値の適切な処理）
            actual_direction = self._get_direction(y_diff, zero_threshold)
            predicted_direction = self._get_direction(pred_diff, zero_threshold)

            # 無効なサンプル（どちらも0の場合）を除外
            valid_mask = ~((actual_direction == 0) & (predicted_direction == 0))

            if not np.any(valid_mask):
                logger.warning("Hit rate計算: 有効な方向性データなし - デフォルト値0.5を返す")
                return 0.5

            # 有効なサンプルでの方向性一致率
            valid_actual = actual_direction[valid_mask]
            valid_predicted = predicted_direction[valid_mask]

            # 方向性一致
            direction_match = valid_actual == valid_predicted
            hit_rate = np.mean(direction_match)

            logger.debug(
                f"Hit rate計算完了: {hit_rate:.3f} "
                f"(有効サンプル: {np.sum(valid_mask)}/{len(y_diff)})"
            )

            return float(hit_rate)

        except Exception as e:
            logger.error(f"Hit rate計算エラー: {e}", exc_info=True)
            return 0.5

    def _get_direction(self, values: np.ndarray, zero_threshold: float) -> np.ndarray:
        """
        値の方向性を判定（0値の適切な処理）

        Args:
            values: 判定対象の値
            zero_threshold: 0値と判定する閾値

        Returns:
            direction: 方向性 (1: 上昇, -1: 下降, 0: 変化なし)
        """
        # 絶対値が閾値以下の場合は0（変化なし）とする
        direction = np.zeros_like(values, dtype=int)

        # 上昇判定
        direction[values > zero_threshold] = 1

        # 下降判定
        direction[values < -zero_threshold] = -1

        return direction

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報取得

        Returns:
            モデル情報辞書
        """
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names[:10]  # 最初の10個のみ
        }

    def set_feature_names(self, feature_names: List[str]):
        """特徴量名設定"""
        self.feature_names = feature_names
        logger.debug(f"{self.model_name}特徴量名設定: {len(feature_names)}個")

    def __str__(self) -> str:
        """文字列表現"""
        status = "Trained" if self.is_trained else "Untrained"
        return f"{self.model_name} ({status})"

    def __repr__(self) -> str:
        """詳細文字列表現"""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', trained={self.is_trained})"