#!/usr/bin/env python3
"""
XGBoost Model Implementation for Ensemble System

Issue #462対応: より強力なベースモデルとして XGBoost を追加し、
アンサンブル学習システムの予測精度95%超達成を目指す
"""

import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from .base_model_interface import BaseModelInterface, ModelPrediction, ModelMetrics
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class XGBoostConfig:
    """XGBoost設定"""
    # 基本パラメータ
    n_estimators: int = 300
    max_depth: int = 8
    learning_rate: float = 0.08
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0

    # 早期停止
    early_stopping_rounds: int = 50
    eval_metric: str = 'rmse'

    # ハイパーパラメータ最適化
    enable_hyperopt: bool = True
    hyperopt_max_evals: int = 100

    # パフォーマンス
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0


class XGBoostModel(BaseModelInterface):
    """
    XGBoost実装モデル

    グラディエントブースティングの高性能実装として、
    アンサンブルシステムの精度向上に貢献
    """

    def __init__(self, config: Optional[XGBoostConfig] = None):
        """
        初期化

        Args:
            config: XGBoost設定
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoostが利用できません。インストールしてください: "
                "pip install xgboost"
            )

        self.config = config or XGBoostConfig()
        # BaseModelInterface初期化
        super().__init__(model_name="XGBoost", config=self.config.__dict__)

        logger.info(f"XGBoostModel初期化完了: {self.config}")

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
        logger.info(f"XGBoostModel学習開始: {X.shape}")

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        try:
            # ハイパーパラメータ最適化
            enable_hyperopt = getattr(self.config, 'enable_hyperopt', False)
            if enable_hyperopt and validation_data is not None:
                best_params = self._optimize_hyperparameters(X, y, validation_data)
                logger.info(f"最適化されたパラメータ: {best_params}")
            else:
                best_params = self._get_base_params()

            # XGBRegressor作成
            self.model = xgb.XGBRegressor(**best_params)

            # 早期停止用の検証セット準備
            eval_set = None
            if validation_data is not None:
                X_val, y_val = validation_data
                eval_set = [(X_val, y_val)]

            # モデル学習
            self.model.fit(
                X, y,
                eval_set=eval_set,
                early_stopping_rounds=self.config.early_stopping_rounds if eval_set else None,
                verbose=False
            )

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
                'model_name': 'XGBoost',
                'training_time': training_time,
                'train_metrics': train_metrics,
                'validation_metrics': val_metrics,
                'feature_importance': self._get_feature_importance(),
                'model_params': best_params,
                'best_iteration': getattr(self.model, 'best_iteration', None)
            }

            self.training_metrics = results

            logger.info(f"XGBoost学習完了: {training_time:.2f}秒, "
                       f"Train R²={train_metrics['r2_score']:.4f}")

            return results

        except Exception as e:
            logger.error(f"XGBoost学習エラー: {e}")
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

            # 信頼度計算（予測の分散ベース）
            confidence = self._calculate_confidence(X, predictions)

            return ModelPrediction(
                predictions=predictions,
                confidence=confidence,
                processing_time=prediction_time,
                model_name='XGBoost'
            )

        except Exception as e:
            logger.error(f"XGBoost予測エラー: {e}")
            raise

    def _get_base_params(self) -> Dict[str, Any]:
        """基本パラメータ取得"""
        return {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
            'eval_metric': self.config.eval_metric
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
        logger.info("XGBoostハイパーパラメータ最適化開始")

        X_val, y_val = validation_data
        base_params = self._get_base_params()

        # グリッドサーチ的なアプローチ（簡易版）
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.08, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0.0, 0.1, 1.0],
            'reg_lambda': [0.1, 1.0, 10.0]
        }

        best_score = float('inf')
        best_params = base_params.copy()

        # 限定的な組み合わせでテスト（実用的な最適化時間のため）
        combinations = [
            {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.08},
            {'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.05},
            {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1},
            {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.08, 'reg_alpha': 1.0},
            {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.08, 'subsample': 0.8}
        ]

        for i, params_update in enumerate(combinations):
            if i >= 10:  # 最大10回のテスト
                break

            try:
                test_params = base_params.copy()
                test_params.update(params_update)

                # テストモデル作成・学習
                test_model = xgb.XGBRegressor(**test_params)
                test_model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose=False
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
            return np.full(len(predictions), 0.7)

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

        return ModelMetrics(
            mse=mse,
            mae=mae,
            rmse=rmse,
            r2_score=r2,
            accuracy=max(0.0, r2)  # R²スコアを精度として使用
        )

    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        return {
            'model_type': 'XGBoost',
            'is_trained': self.is_trained,
            'config': self.config.__dict__,
            'training_metrics': self.training_metrics,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else None
        }


def create_xgboost_model(config: Optional[Dict[str, Any]] = None) -> XGBoostModel:
    """
    XGBoostモデル作成ファクトリ関数

    Args:
        config: モデル設定辞書

    Returns:
        XGBoostModelインスタンス
    """
    if config:
        # 辞書から有効なXGBoostConfigパラメータを抽出
        valid_params = {}
        config_fields = set(XGBoostConfig.__dataclass_fields__.keys())
        for key, value in config.items():
            if key in config_fields:
                valid_params[key] = value
        xgb_config = XGBoostConfig(**valid_params)
    else:
        xgb_config = XGBoostConfig()

    return XGBoostModel(xgb_config)