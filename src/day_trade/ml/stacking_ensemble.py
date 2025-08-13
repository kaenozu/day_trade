#!/usr/bin/env python3
"""
Stacking Ensemble Implementation

Issue #462: 高度スタッキングアンサンブル実装
メタ学習器による複数モデル統合で最高精度を実現
"""

import time
from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from .base_models.base_model_interface import BaseModelInterface, ModelPrediction
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class StackingConfig:
    """スタッキング設定"""
    # メタ学習器選択
    meta_learner_type: str = "xgboost"  # linear, ridge, lasso, elastic, rf, xgboost, mlp

    # 交差検証設定
    cv_method: str = "timeseries"  # kfold, timeseries
    cv_folds: int = 5

    # メタ学習器パラメータ
    meta_learner_params: Dict[str, Any] = None

    # 特徴量エンジニアリング
    include_base_features: bool = False  # 元特徴量をメタ学習器に含める
    include_prediction_stats: bool = True  # 予測統計量を含める

    # パフォーマンス設定
    enable_hyperopt: bool = True
    normalize_meta_features: bool = True
    verbose: bool = True


class StackingEnsemble:
    """
    スタッキングアンサンブル実装

    2段階学習:
    1. Base Learners: 複数の基底モデルで予測
    2. Meta Learner: Base Learnersの予測を統合
    """

    def __init__(self, base_models: Dict[str, BaseModelInterface],
                 config: Optional[StackingConfig] = None):
        """
        初期化

        Args:
            base_models: ベースモデル辞書
            config: スタッキング設定
        """
        self.base_models = base_models
        self.config = config or StackingConfig()

        # メタ学習器
        self.meta_learner = None
        self.meta_scaler = StandardScaler() if self.config.normalize_meta_features else None

        # 訓練データ保存（メタ学習用）
        self.meta_features_train = None
        self.meta_targets_train = None

        # 学習状態
        self.is_fitted = False
        self.cv_scores = {}
        self.meta_feature_names = []

        logger.info(f"スタッキングアンサンブル初期化: {len(base_models)}個のベースモデル")

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        スタッキングアンサンブル学習

        Args:
            X: 訓練データの特徴量
            y: 訓練データの目標変数
            validation_data: 検証データ（オプション）

        Returns:
            学習結果辞書
        """
        start_time = time.time()
        logger.info(f"スタッキング学習開始: データ形状 {X.shape}")

        try:
            # Step 1: ベースモデルの交差検証学習
            meta_features = self._generate_meta_features(X, y)

            # Step 2: メタ学習器の学習
            self._fit_meta_learner(meta_features, y, X if self.config.include_base_features else None)

            # Step 3: ベースモデルの最終学習（全データ）
            self._fit_base_models_final(X, y)

            # 学習結果
            training_results = {
                'training_time': time.time() - start_time,
                'meta_learner_type': self.config.meta_learner_type,
                'meta_features_shape': meta_features.shape,
                'cv_scores': self.cv_scores,
                'meta_feature_names': self.meta_feature_names
            }

            # 検証データでの評価
            if validation_data:
                X_val, y_val = validation_data
                val_pred = self.predict(X_val)
                val_mse = np.mean((y_val - val_pred.predictions) ** 2)
                val_rmse = np.sqrt(val_mse)

                training_results['validation_rmse'] = val_rmse
                logger.info(f"スタッキング検証RMSE: {val_rmse:.4f}")

            self.is_fitted = True
            logger.info(f"スタッキング学習完了: {time.time() - start_time:.2f}秒")

            return training_results

        except Exception as e:
            logger.error(f"スタッキング学習エラー: {e}")
            raise

    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        スタッキング予測

        Args:
            X: 予測データの特徴量

        Returns:
            ModelPrediction: 予測結果
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません")

        start_time = time.time()

        try:
            # Step 1: ベースモデルからの予測
            base_predictions = {}
            for name, model in self.base_models.items():
                if model.is_trained:
                    pred_result = model.predict(X)
                    base_predictions[name] = pred_result.predictions

            # Step 2: メタ特徴量生成
            meta_features = self._create_meta_features_from_predictions(
                base_predictions, X if self.config.include_base_features else None
            )

            # Step 3: メタ学習器で最終予測
            if self.meta_scaler:
                meta_features_scaled = self.meta_scaler.transform(meta_features)
            else:
                meta_features_scaled = meta_features

            final_predictions = self.meta_learner.predict(meta_features_scaled)

            # 信頼度計算（予測分散ベース）
            confidence = self._calculate_stacking_confidence(base_predictions)

            processing_time = time.time() - start_time

            return ModelPrediction(
                predictions=final_predictions,
                confidence=confidence,
                feature_importance=self._get_meta_feature_importance(),
                model_name="StackingEnsemble",
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"スタッキング予測エラー: {e}")
            raise

    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        交差検証によるメタ特徴量生成

        Args:
            X: 訓練データ特徴量
            y: 訓練データ目標変数

        Returns:
            メタ特徴量配列
        """
        logger.info("メタ特徴量生成開始")

        # 交差検証設定
        if self.config.cv_method == "timeseries":
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        else:
            cv = KFold(n_splits=self.config.cv_folds, shuffle=False)

        # メタ特徴量初期化
        meta_predictions = {name: np.zeros(len(X)) for name in self.base_models.keys()}

        # 交差検証実行
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            logger.info(f"CV Fold {fold + 1}/{self.config.cv_folds}")

            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            # 各ベースモデルの学習・予測
            for name, model in self.base_models.items():
                try:
                    # モデルコピー作成（元モデルを変更しない）
                    model_copy = self._copy_model(model)

                    # Fold学習
                    model_copy.fit(X_fold_train, y_fold_train)

                    # Fold予測
                    pred_result = model_copy.predict(X_fold_val)
                    meta_predictions[name][val_idx] = pred_result.predictions

                    # CV スコア記録
                    fold_mse = np.mean((y_fold_val - pred_result.predictions) ** 2)
                    if name not in self.cv_scores:
                        self.cv_scores[name] = []
                    self.cv_scores[name].append(np.sqrt(fold_mse))

                except Exception as e:
                    logger.warning(f"Fold {fold} {name} エラー: {e}")
                    # エラー時はゼロ埋め
                    meta_predictions[name][val_idx] = 0.0

        # メタ特徴量構築
        meta_features = self._create_meta_features_from_predictions(
            meta_predictions, X if self.config.include_base_features else None
        )

        logger.info(f"メタ特徴量生成完了: {meta_features.shape}")
        return meta_features

    def _create_meta_features_from_predictions(self,
                                             predictions: Dict[str, np.ndarray],
                                             base_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        予測値からメタ特徴量を作成

        Args:
            predictions: モデル名と予測値の辞書
            base_features: 元特徴量（オプション）

        Returns:
            メタ特徴量配列
        """
        feature_list = []
        feature_names = []

        # 1. 基本予測値
        for name, pred in predictions.items():
            feature_list.append(pred.reshape(-1, 1))
            feature_names.append(f"pred_{name}")

        # 2. 予測統計量
        if self.config.include_prediction_stats and len(predictions) > 1:
            pred_array = np.array(list(predictions.values())).T  # (n_samples, n_models)

            # 統計量計算
            pred_mean = np.mean(pred_array, axis=1, keepdims=True)
            pred_std = np.std(pred_array, axis=1, keepdims=True)
            pred_min = np.min(pred_array, axis=1, keepdims=True)
            pred_max = np.max(pred_array, axis=1, keepdims=True)
            pred_median = np.median(pred_array, axis=1, keepdims=True)

            feature_list.extend([pred_mean, pred_std, pred_min, pred_max, pred_median])
            feature_names.extend(['pred_mean', 'pred_std', 'pred_min', 'pred_max', 'pred_median'])

            # ペアワイズ差分
            model_names = list(predictions.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    diff = (pred_array[:, i] - pred_array[:, j]).reshape(-1, 1)
                    feature_list.append(diff)
                    feature_names.append(f"diff_{model_names[i]}_{model_names[j]}")

        # 3. 元特徴量
        if base_features is not None:
            feature_list.append(base_features)
            feature_names.extend([f"base_feature_{i}" for i in range(base_features.shape[1])])

        # 結合
        meta_features = np.concatenate(feature_list, axis=1)
        self.meta_feature_names = feature_names

        return meta_features

    def _fit_meta_learner(self, meta_features: np.ndarray, targets: np.ndarray,
                         base_features: Optional[np.ndarray] = None):
        """
        メタ学習器の学習

        Args:
            meta_features: メタ特徴量
            targets: 目標変数
            base_features: 元特徴量（オプション）
        """
        logger.info(f"メタ学習器学習開始: {self.config.meta_learner_type}")

        # 前処理
        if self.meta_scaler:
            meta_features_scaled = self.meta_scaler.fit_transform(meta_features)
        else:
            meta_features_scaled = meta_features

        # メタ学習器初期化
        self.meta_learner = self._create_meta_learner()

        # 学習
        if hasattr(self.meta_learner, 'fit'):
            self.meta_learner.fit(meta_features_scaled, targets)
        else:
            raise ValueError(f"メタ学習器 {self.config.meta_learner_type} はfitメソッドを持ちません")

        # 学習データ保存
        self.meta_features_train = meta_features_scaled
        self.meta_targets_train = targets

        logger.info("メタ学習器学習完了")

    def _create_meta_learner(self):
        """メタ学習器作成"""
        params = self.config.meta_learner_params or {}

        if self.config.meta_learner_type == "linear":
            return LinearRegression(**params)
        elif self.config.meta_learner_type == "ridge":
            default_params = {'alpha': 1.0}
            return Ridge(**{**default_params, **params})
        elif self.config.meta_learner_type == "lasso":
            default_params = {'alpha': 1.0}
            return Lasso(**{**default_params, **params})
        elif self.config.meta_learner_type == "elastic":
            default_params = {'alpha': 1.0, 'l1_ratio': 0.5}
            return ElasticNet(**{**default_params, **params})
        elif self.config.meta_learner_type == "rf":
            default_params = {'n_estimators': 100, 'random_state': 42}
            return RandomForestRegressor(**{**default_params, **params})
        elif self.config.meta_learner_type == "xgboost":
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'objective': 'reg:squarederror'
            }
            return XGBRegressor(**{**default_params, **params})
        elif self.config.meta_learner_type == "mlp":
            default_params = {
                'hidden_layer_sizes': (100, 50),
                'max_iter': 500,
                'random_state': 42
            }
            return MLPRegressor(**{**default_params, **params})
        else:
            raise ValueError(f"不明なメタ学習器タイプ: {self.config.meta_learner_type}")

    def _fit_base_models_final(self, X: np.ndarray, y: np.ndarray):
        """ベースモデルの最終学習（全データ）"""
        logger.info("ベースモデル最終学習開始")

        for name, model in self.base_models.items():
            try:
                if not model.is_trained:
                    model.fit(X, y)
                logger.info(f"{name} 最終学習完了")
            except Exception as e:
                logger.error(f"{name} 最終学習エラー: {e}")

    def _copy_model(self, model: BaseModelInterface) -> BaseModelInterface:
        """モデルのコピー作成"""
        # 簡単な実装：同じ設定で新しいインスタンス作成
        model_class = model.__class__
        return model_class(model.config)

    def _calculate_stacking_confidence(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """スタッキング信頼度計算"""
        if len(base_predictions) < 2:
            return np.ones(len(list(base_predictions.values())[0])) * 0.5

        # ベース予測の分散ベース信頼度
        pred_array = np.array(list(base_predictions.values())).T
        pred_variance = np.var(pred_array, axis=1)

        # 正規化（分散が小さいほど信頼度が高い）
        max_var = np.max(pred_variance) if np.max(pred_variance) > 0 else 1.0
        confidence = 1.0 - (pred_variance / max_var)

        return confidence

    def _get_meta_feature_importance(self) -> Dict[str, float]:
        """メタ特徴量重要度取得"""
        if not self.is_fitted or not hasattr(self.meta_learner, 'feature_importances_'):
            return {}

        importances = self.meta_learner.feature_importances_

        if len(self.meta_feature_names) == len(importances):
            return dict(zip(self.meta_feature_names, importances))
        else:
            return {}

    def get_cv_results(self) -> pd.DataFrame:
        """交差検証結果取得"""
        if not self.cv_scores:
            return pd.DataFrame()

        results = []
        for model_name, scores in self.cv_scores.items():
            results.append({
                'model': model_name,
                'mean_rmse': np.mean(scores),
                'std_rmse': np.std(scores),
                'min_rmse': np.min(scores),
                'max_rmse': np.max(scores)
            })

        return pd.DataFrame(results).sort_values('mean_rmse')

    def get_stacking_info(self) -> Dict[str, Any]:
        """スタッキング情報取得"""
        return {
            'is_fitted': self.is_fitted,
            'meta_learner_type': self.config.meta_learner_type,
            'n_base_models': len(self.base_models),
            'base_model_names': list(self.base_models.keys()),
            'meta_feature_count': len(self.meta_feature_names),
            'cv_method': self.config.cv_method,
            'cv_folds': self.config.cv_folds
        }


if __name__ == "__main__":
    # テスト実行（プレースホルダー）
    print("=== Stacking Ensemble テスト ===")
    print("実際のテストは統合テストで実行します")