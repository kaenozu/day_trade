"""
ML モデル基底クラスとトレーナー

ml_prediction_models_improved.py からのリファクタリング抽出
抽象基底クラスと具体的なモデルトレーナー実装
"""

import threading
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

# 設定とエラーのインポート
from .ml_config import ModelType, PredictionTask, DataQuality, TrainingConfig
from .ml_exceptions import ModelTrainingError

# 機械学習ライブラリ（フォールバック対応）
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                mean_squared_error, r2_score)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class BaseModelTrainer(ABC):
    """モデル訓練の抽象基底クラス（強化版）"""

    def __init__(self, model_type: ModelType, config: Dict[str, Any], logger=None):
        self.model_type = model_type
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()

    @abstractmethod
    def create_model(self, is_classifier: bool, hyperparameters: Dict[str, Any]):
        """モデルインスタンス作成（抽象メソッド）"""
        pass

    def prepare_data(self, X: pd.DataFrame, y: pd.Series, config: TrainingConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """データ分割の共通処理（強化版）"""
        if not SKLEARN_AVAILABLE:
            raise ModelTrainingError("sklearn が利用できません")

        stratify = y if config.stratify and self._is_classification_task(y) else None

        return train_test_split(
            X, y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify
        )

    def _is_classification_task(self, y: pd.Series) -> bool:
        """分類タスクかどうかの判定"""
        return y.dtype == 'object' or len(y.unique()) <= 10

    def validate_data_quality(self, X: pd.DataFrame, y: pd.Series, task: PredictionTask) -> Tuple[bool, DataQuality, str]:
        """データ品質検証の共通処理（強化版）"""
        try:
            issues = []
            quality_score = 100.0

            # 基本チェック
            if X.empty or y.empty:
                return False, DataQuality.INSUFFICIENT, "データが空です"

            if len(X) != len(y):
                return False, DataQuality.INSUFFICIENT, f"特徴量とターゲットのサイズ不一致: {len(X)} vs {len(y)}"

            # サンプル数チェック
            min_samples = self._get_minimum_samples(task)
            if len(X) < min_samples:
                return False, DataQuality.INSUFFICIENT, f"サンプル数不足: {len(X)} < {min_samples}"

            # 欠損値チェック
            missing_features = X.isnull().sum().sum()
            missing_targets = y.isnull().sum()

            feature_missing_rate = missing_features / (len(X) * len(X.columns))
            target_missing_rate = missing_targets / len(y)

            if feature_missing_rate > 0.2:
                quality_score -= 30
                issues.append(f"特徴量欠損率高: {feature_missing_rate:.1%}")
            elif feature_missing_rate > 0.1:
                quality_score -= 15
                issues.append(f"特徴量欠損率中: {feature_missing_rate:.1%}")

            if target_missing_rate > 0.1:
                quality_score -= 25
                issues.append(f"ターゲット欠損率高: {target_missing_rate:.1%}")

            # 分類タスクのクラス分布チェック
            if task == PredictionTask.PRICE_DIRECTION:
                class_counts = y.value_counts()
                min_class_size = len(y) * 0.05

                if (class_counts < min_class_size).any():
                    quality_score -= 20
                    issues.append(f"クラス不均衡: {class_counts.to_dict()}")

            # 特徴量の分散チェック
            numeric_features = X.select_dtypes(include=[np.number])
            if len(numeric_features.columns) > 0:
                zero_variance_count = (numeric_features.var() == 0).sum()
                zero_variance_rate = zero_variance_count / len(numeric_features.columns)

                if zero_variance_rate > 0.3:
                    quality_score -= 20
                    issues.append(f"分散ゼロ特徴量率高: {zero_variance_rate:.1%}")

            # 品質レベル決定
            if quality_score >= 90:
                quality = DataQuality.EXCELLENT
            elif quality_score >= 75:
                quality = DataQuality.GOOD
            elif quality_score >= 60:
                quality = DataQuality.FAIR
            elif quality_score >= 40:
                quality = DataQuality.POOR
            else:
                quality = DataQuality.INSUFFICIENT

            message = f"品質スコア: {quality_score:.1f}" + (f", 問題: {'; '.join(issues)}" if issues else "")
            success = quality != DataQuality.INSUFFICIENT

            return success, quality, message

        except Exception as e:
            return False, DataQuality.INSUFFICIENT, f"検証エラー: {e}"

    def _get_minimum_samples(self, task: PredictionTask) -> int:
        """タスクに応じた最小サンプル数"""
        return {
            PredictionTask.PRICE_DIRECTION: 100,
            PredictionTask.PRICE_REGRESSION: 50,
            PredictionTask.VOLATILITY: 50,
            PredictionTask.TREND_STRENGTH: 75
        }.get(task, 50)

    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None, is_classifier: bool = True) -> Dict[str, float]:
        """性能指標計算の共通処理（強化版）"""
        if not SKLEARN_AVAILABLE:
            return {}

        metrics = {}

        try:
            if is_classifier:
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

                # 予測確率が利用可能な場合の追加メトリクス
                if y_pred_proba is not None:
                    try:
                        from sklearn.metrics import roc_auc_score, log_loss
                        if len(np.unique(y_true)) == 2:  # 二値分類
                            metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                        metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                    except Exception:
                        pass

            else:  # 回帰
                metrics['r2_score'] = r2_score(y_true, y_pred)
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = np.mean(np.abs(y_true - y_pred))

                # 追加の回帰メトリクス
                metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100

        except Exception as e:
            self.logger.error(f"メトリクス計算エラー: {e}")

        return metrics

    def cross_validate(self, model, X: pd.DataFrame, y: pd.Series, config: TrainingConfig) -> np.ndarray:
        """クロスバリデーションの共通処理（強化版）"""
        if not SKLEARN_AVAILABLE:
            return np.array([])

        scoring = 'accuracy' if self._is_classification_task(y) else 'r2'
        cv_strategy = TimeSeriesSplit(n_splits=config.cv_folds)

        return cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)

    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """特徴量重要度取得の共通処理"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                return dict(zip(feature_names, importance_scores))
            elif hasattr(model, 'coef_'):
                # 線形モデルの場合
                importance_scores = np.abs(model.coef_).flatten()
                return dict(zip(feature_names, importance_scores))
            else:
                return {}
        except Exception as e:
            self.logger.warning(f"特徴量重要度取得失敗: {e}")
            return {}

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, config: TrainingConfig,
                          task: PredictionTask, hyperparameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """訓練と評価の統合メソッド"""
        try:
            with self._lock:
                # データ品質検証
                is_valid, quality, quality_msg = self.validate_data_quality(X, y, task)
                if not is_valid:
                    raise ModelTrainingError(f"データ品質不足: {quality_msg}")

                # データ分割
                X_train, X_test, y_train, y_test = self.prepare_data(X, y, config)

                # モデル作成
                is_classifier = self._is_classification_task(y)
                model = self.create_model(is_classifier, hyperparameters or {})

                # 訓練
                model.fit(X_train, y_train)

                # 予測
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') and is_classifier else None

                # メトリクス計算
                metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba, is_classifier)

                # クロスバリデーション
                cv_scores = self.cross_validate(model, X_train, y_train, config)

                # 特徴量重要度
                feature_importance = self.get_feature_importance(model, X.columns.tolist())

                return {
                    'model': model,
                    'metrics': metrics,
                    'cv_scores': cv_scores,
                    'feature_importance': feature_importance,
                    'data_quality': quality,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'model_type': self.model_type.value
                }

        except Exception as e:
            self.logger.error(f"訓練・評価エラー: {e}")
            raise ModelTrainingError(f"モデル訓練失敗: {e}") from e


class RandomForestTrainer(BaseModelTrainer):
    """Random Forest訓練器"""

    def create_model(self, is_classifier: bool, hyperparameters: Dict[str, Any]):
        """Random Forestモデル作成"""
        if not SKLEARN_AVAILABLE:
            raise ModelTrainingError("sklearn が利用できません")

        base_params = self.config.get('classifier_params' if is_classifier else 'regressor_params', {})
        final_params = {**base_params, **hyperparameters}

        if is_classifier:
            return RandomForestClassifier(**final_params)
        else:
            return RandomForestRegressor(**final_params)


class XGBoostTrainer(BaseModelTrainer):
    """XGBoost訓練器"""

    def create_model(self, is_classifier: bool, hyperparameters: Dict[str, Any]):
        """XGBoostモデル作成"""
        if not XGBOOST_AVAILABLE:
            raise ModelTrainingError("XGBoost is not available")

        base_params = self.config.get('classifier_params' if is_classifier else 'regressor_params', {})
        final_params = {**base_params, **hyperparameters}

        if is_classifier:
            return xgb.XGBClassifier(**final_params)
        else:
            return xgb.XGBRegressor(**final_params)


class LightGBMTrainer(BaseModelTrainer):
    """LightGBM訓練器"""

    def create_model(self, is_classifier: bool, hyperparameters: Dict[str, Any]):
        """LightGBMモデル作成"""
        if not LIGHTGBM_AVAILABLE:
            raise ModelTrainingError("LightGBM is not available")

        base_params = self.config.get('classifier_params' if is_classifier else 'regressor_params', {})
        final_params = {**base_params, **hyperparameters}

        if is_classifier:
            return lgb.LGBMClassifier(**final_params)
        else:
            return lgb.LGBMRegressor(**final_params)


def create_trainer(model_type: ModelType, config: Dict[str, Any] = None, logger=None) -> BaseModelTrainer:
    """モデルタイプに応じたトレーナーファクトリー"""
    config = config or {}

    trainers = {
        ModelType.RANDOM_FOREST: RandomForestTrainer,
        ModelType.XGBOOST: XGBoostTrainer,
        ModelType.LIGHTGBM: LightGBMTrainer,
    }

    trainer_class = trainers.get(model_type)
    if trainer_class is None:
        raise ModelTrainingError(f"未対応のモデルタイプ: {model_type}")

    return trainer_class(model_type, config, logger)