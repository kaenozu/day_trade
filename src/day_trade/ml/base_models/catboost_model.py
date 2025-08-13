#!/usr/bin/env python3
"""
CatBoost Model Implementation for Ensemble System

Issue #462対応: より強力なベースモデルとして CatBoost を追加し、
アンサンブル学習システムの予測精度95%超達成を目指す
"""

import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None

from .base_model_interface import BaseModelInterface, ModelPrediction, ModelMetrics
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class CatBoostConfig:
    """CatBoost設定"""
    # 基本パラメータ
    iterations: int = 500
    learning_rate: float = 0.1
    depth: int = 8
    l2_leaf_reg: float = 3.0
    border_count: int = 128

    # 早期停止
    early_stopping_rounds: int = 50
    eval_metric: str = 'RMSE'

    # ハイパーパラメータ最適化
    enable_hyperopt: bool = True
    hyperopt_max_evals: int = 50

    # パフォーマンス
    random_state: int = 42
    thread_count: int = -1
    verbose: int = 0

    # CatBoost特有
    boosting_type: str = 'Plain'  # Plain, Ordered
    bootstrap_type: str = 'Bayesian'  # Bayesian, Bernoulli, MVS
    bagging_temperature: float = 1.0


class CatBoostModel(BaseModelInterface):
    """
    CatBoost実装モデル

    カテゴリ変数に強く、高精度なグラディエントブースティング実装として、
    アンサンブルシステムの精度向上に貢献
    """

    def __init__(self, config: Optional[CatBoostConfig] = None):
        """
        初期化

        Args:
            config: CatBoost設定
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoostが利用できません。インストールしてください: "
                "pip install catboost"
            )

        self.config = config or CatBoostConfig()
        # BaseModelInterface初期化
        super().__init__(model_name="CatBoost", config=self.config.__dict__)

        logger.info(f"CatBoostModel初期化完了: {self.config}")

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        モデル学習

        Args:
            X: 特徴量データ
            y: 目標変数
            validation_data: 検証データ（X_val, y_val）
            feature_names: 特徴量名

        Returns:
            学習結果
        """
        start_time = time.time()
        logger.info(f"CatBoostModel学習開始: {X.shape}")

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        try:
            # ハイパーパラメータ最適化
            enable_hyperopt = getattr(self.config, 'enable_hyperopt', False)
            if enable_hyperopt and validation_data is not None:
                best_params = self._optimize_hyperparameters(X, y, validation_data)
                logger.info(f"最適化されたパラメータ: {best_params}")
            else:
                best_params = self._get_base_params()

            # CatBoostRegressor作成
            self.model = CatBoostRegressor(**best_params)

            # 検証セット準備
            eval_set = None
            if validation_data is not None:
                X_val, y_val = validation_data
                eval_set = (X_val, y_val)

            # モデル学習
            fit_params = {}
            if eval_set is not None:
                fit_params['eval_set'] = eval_set
                fit_params['early_stopping_rounds'] = getattr(self.config, 'early_stopping_rounds', 50)

            self.model.fit(X, y, **fit_params)

            self.is_trained = True
            training_time = time.time() - start_time

            # 学習結果メトリクス
            train_pred = self.model.predict(X)
            train_metrics = self._calculate_metrics(y, train_pred)

            val_metrics = None
            if validation_data is not None:
                val_pred = self.model.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, val_pred)

            results = {
                'model_name': 'CatBoost',
                'training_time': training_time,
                'train_metrics': train_metrics,
                'validation_metrics': val_metrics,
                'feature_importance': self._get_feature_importance(),
                'model_params': best_params,
                'best_iteration': getattr(self.model, 'best_iteration_', None)
            }

            self.training_metrics = results

            logger.info(f"CatBoost学習完了: {training_time:.2f}秒, "
                       f"Train R²={train_metrics.r2_score:.4f}")

            return results

        except Exception as e:
            logger.error(f"CatBoost学習エラー: {e}")
            raise

    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        予測実行

        Args:
            X: 特徴量データ

        Returns:
            予測結果
        """
        if not self.is_trained:
            raise ValueError("モデルが学習されていません")

        start_time = time.time()

        try:
            predictions = self.model.predict(X)
            prediction_time = time.time() - start_time

            # 信頼度計算（特徴量重要度ベース）
            confidence = self._calculate_confidence(X, predictions)

            return ModelPrediction(
                predictions=predictions,
                confidence=confidence,
                processing_time=prediction_time,
                model_name='CatBoost'
            )

        except Exception as e:
            logger.error(f"CatBoost予測エラー: {e}")
            raise

    def _get_base_params(self) -> Dict[str, Any]:
        """基本パラメータ取得"""
        # デフォルト値を設定
        defaults = CatBoostConfig()

        return {
            'iterations': getattr(self.config, 'iterations', defaults.iterations),
            'learning_rate': getattr(self.config, 'learning_rate', defaults.learning_rate),
            'depth': getattr(self.config, 'depth', defaults.depth),
            'l2_leaf_reg': getattr(self.config, 'l2_leaf_reg', defaults.l2_leaf_reg),
            'border_count': getattr(self.config, 'border_count', defaults.border_count),
            'eval_metric': getattr(self.config, 'eval_metric', defaults.eval_metric),
            'random_state': getattr(self.config, 'random_state', defaults.random_state),
            'thread_count': getattr(self.config, 'thread_count', defaults.thread_count),
            'verbose': getattr(self.config, 'verbose', defaults.verbose),
            'boosting_type': getattr(self.config, 'boosting_type', defaults.boosting_type),
            'bootstrap_type': getattr(self.config, 'bootstrap_type', defaults.bootstrap_type),
            'bagging_temperature': getattr(self.config, 'bagging_temperature', defaults.bagging_temperature)
        }

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                                validation_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        ハイパーパラメータ最適化

        Args:
            X: 学習データ
            y: 目標変数
            validation_data: 検証データ

        Returns:
            最適化されたパラメータ
        """
        logger.info("CatBoostハイパーパラメータ最適化開始")

        X_val, y_val = validation_data
        base_params = self._get_base_params()

        # 最適化対象の組み合わせ
        combinations = [
            {'iterations': 400, 'learning_rate': 0.08, 'depth': 6},
            {'iterations': 500, 'learning_rate': 0.1, 'depth': 8},
            {'iterations': 600, 'learning_rate': 0.05, 'depth': 10},
            {'iterations': 500, 'learning_rate': 0.1, 'depth': 8, 'l2_leaf_reg': 5.0},
            {'iterations': 500, 'learning_rate': 0.1, 'depth': 8, 'bagging_temperature': 0.5}
        ]

        best_score = float('inf')
        best_params = base_params.copy()

        for i, params_update in enumerate(combinations):
            try:
                test_params = base_params.copy()
                test_params.update(params_update)

                # テストモデル作成・学習
                test_model = CatBoostRegressor(**test_params)
                test_model.fit(
                    X, y,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=getattr(self.config, 'early_stopping_rounds', 50)
                )

                # 検証スコア
                val_pred = test_model.predict(X_val)
                val_score = np.sqrt(np.mean((y_val - val_pred) ** 2))

                if val_score < best_score:
                    best_score = val_score
                    best_params = test_params.copy()

                logger.debug(f"パラメータテスト {i+1}: RMSE={val_score:.6f}")

            except Exception as e:
                logger.warning(f"パラメータ最適化エラー: {e}")
                continue

        logger.info(f"最適化完了: 最高スコア={best_score:.6f}")
        return best_params

    def _calculate_confidence(self, X: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        予測信頼度計算

        Args:
            X: 特徴量
            predictions: 予測値

        Returns:
            信頼度配列
        """
        try:
            # 特徴量重要度による信頼度計算
            feature_importance = self.model.feature_importances_

            # 各予測の特徴量重要度重み付きスコア
            weighted_features = X * feature_importance
            feature_scores = np.sum(weighted_features, axis=1)

            # 正規化して信頼度に変換
            confidence = np.clip(
                (feature_scores - feature_scores.min()) /
                (feature_scores.max() - feature_scores.min() + 1e-8),
                0.1, 0.99
            )

            return confidence

        except Exception as e:
            logger.warning(f"信頼度計算エラー: {e}")
            return np.full(len(predictions), 0.75)

    def _get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度取得"""
        if not self.is_trained or self.model is None:
            return {}

        importance = self.model.feature_importances_
        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, importance)
        }

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """メトリクス計算"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)

        # 方向予測精度計算
        if len(y_true) > 1:
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            hit_rate = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
        else:
            hit_rate = 0.5

        return ModelMetrics(
            mse=mse,
            mae=mae,
            rmse=rmse,
            r2_score=r2,
            hit_rate=hit_rate
        )

    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        return {
            'model_type': 'CatBoost',
            'is_trained': self.is_trained,
            'config': self.config.__dict__,
            'training_metrics': self.training_metrics,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else None
        }


def create_catboost_model(config: Optional[Dict[str, Any]] = None) -> CatBoostModel:
    """
    CatBoostモデル作成ファクトリ関数

    Args:
        config: モデル設定辞書

    Returns:
        CatBoostModelインスタンス
    """
    if config:
        # 辞書から有効なCatBoostConfigパラメータを抽出
        valid_params = {}
        config_fields = set(CatBoostConfig.__dataclass_fields__.keys())
        for key, value in config.items():
            if key in config_fields:
                valid_params[key] = value
        cb_config = CatBoostConfig(**valid_params)
    else:
        cb_config = CatBoostConfig()

    return CatBoostModel(cb_config)