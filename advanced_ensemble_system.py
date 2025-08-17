#!/usr/bin/env python3
"""
高度アンサンブル予測システム
Issue #870: 予測精度向上のための包括的提案

スタッキング・ブレンディング・動的重み調整による15-25%精度向上
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import pickle
from pathlib import Path
import json

from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, cross_val_predict
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor,
    VotingRegressor, BaggingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin, clone
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBRegressor = xgb.XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class EnsembleMethod(Enum):
    """アンサンブル手法"""
    STACKING = "stacking"                    # スタッキング
    BLENDING = "blending"                    # ブレンディング
    VOTING = "voting"                        # 投票
    BAYESIAN_AVERAGING = "bayesian_avg"      # ベイジアン平均
    DYNAMIC_WEIGHTING = "dynamic_weight"     # 動的重み
    HIERARCHICAL = "hierarchical"            # 階層的
    ADAPTIVE = "adaptive"                    # 適応的


class ModelCategory(Enum):
    """モデルカテゴリ"""
    TREE_BASED = "tree_based"               # 木ベース
    LINEAR = "linear"                       # 線形
    NEURAL_NETWORK = "neural_network"       # ニューラルネットワーク
    NEIGHBOR_BASED = "neighbor_based"       # 近傍ベース
    SVM_BASED = "svm_based"                # SVM
    ENSEMBLE = "ensemble"                   # アンサンブル


@dataclass
class ModelPerformance:
    """モデル性能情報"""
    model_name: str
    category: ModelCategory
    mse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    prediction_time: float = 0.0
    training_time: float = 0.0
    stability_score: float = 0.0
    recent_performance: List[float] = field(default_factory=list)
    weight: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class EnsembleConfiguration:
    """アンサンブル設定"""
    method: EnsembleMethod = EnsembleMethod.STACKING
    cv_folds: int = 5
    meta_model_type: str = "ridge"
    blend_ratio: float = 0.8
    min_weight: float = 0.01
    max_weight: float = 0.5
    weight_decay: float = 0.95
    performance_window: int = 20
    update_frequency: int = 10


class BaseModelFactory:
    """ベースモデル工場"""

    @staticmethod
    def create_models() -> Dict[str, Tuple[BaseEstimator, ModelCategory]]:
        """ベースモデル作成"""
        models = {}

        # 木ベースモデル
        models.update({
            'random_forest': (
                RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42,
                    n_jobs=-1, min_samples_split=5
                ), ModelCategory.TREE_BASED
            ),
            'gradient_boosting': (
                GradientBoostingRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, subsample=0.8
                ), ModelCategory.TREE_BASED
            ),
            'extra_trees': (
                ExtraTreesRegressor(
                    n_estimators=100, max_depth=10, random_state=42,
                    n_jobs=-1, min_samples_split=5
                ), ModelCategory.TREE_BASED
            ),
            'ada_boost': (
                AdaBoostRegressor(
                    n_estimators=50, learning_rate=0.1, random_state=42
                ), ModelCategory.ENSEMBLE
            )
        })

        # 線形モデル
        models.update({
            'ridge': (
                Ridge(alpha=1.0, random_state=42),
                ModelCategory.LINEAR
            ),
            'lasso': (
                Lasso(alpha=0.1, random_state=42, max_iter=1000),
                ModelCategory.LINEAR
            ),
            'elastic_net': (
                ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000),
                ModelCategory.LINEAR
            ),
            'bayesian_ridge': (
                BayesianRidge(),
                ModelCategory.LINEAR
            ),
            'huber': (
                HuberRegressor(epsilon=1.35, max_iter=100),
                ModelCategory.LINEAR
            )
        })

        # ニューラルネットワーク
        models.update({
            'mlp': (
                MLPRegressor(
                    hidden_layer_sizes=(100, 50), max_iter=500,
                    random_state=42, alpha=0.01
                ), ModelCategory.NEURAL_NETWORK
            )
        })

        # 近傍ベース
        models.update({
            'knn': (
                KNeighborsRegressor(n_neighbors=5, weights='distance'),
                ModelCategory.NEIGHBOR_BASED
            )
        })

        # SVM
        models.update({
            'svr': (
                SVR(kernel='rbf', C=1.0, gamma='scale'),
                ModelCategory.SVM_BASED
            )
        })

        # XGBoost（利用可能な場合）
        if XGBOOST_AVAILABLE:
            models['xgboost'] = (
                XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1
                ), ModelCategory.TREE_BASED
            )

        # LightGBM（利用可能な場合）
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = (
                lgb.LGBMRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1, verbose=-1
                ), ModelCategory.TREE_BASED
            )

        return models


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """スタッキングアンサンブル"""

    def __init__(self, base_models: Dict[str, BaseEstimator],
                 meta_model: BaseEstimator, cv_folds: int = 5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.trained_base_models = {}
        self.meta_features = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'StackingEnsemble':
        """訓練"""
        # レベル1: ベースモデルの交差検証予測
        meta_features = self._get_meta_features(X, y)

        # レベル1モデルを全データで再訓練
        for name, model in self.base_models.items():
            trained_model = clone(model)
            trained_model.fit(X, y)
            self.trained_base_models[name] = trained_model

        # レベル2: メタモデル訓練
        self.meta_model.fit(meta_features, y)
        self.meta_features = meta_features
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測"""
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")

        # レベル1予測
        level1_predictions = np.column_stack([
            self.trained_base_models[name].predict(X)
            for name in self.base_models.keys()
        ])

        # レベル2予測
        meta_predictions = pd.DataFrame(
            level1_predictions,
            columns=list(self.base_models.keys())
        )

        return self.meta_model.predict(meta_predictions)

    def _get_meta_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """メタ特徴量生成（交差検証予測）"""
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(self.base_models)))

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]

            for i, (name, model) in enumerate(self.base_models.items()):
                # 各フォールドでモデル訓練
                fold_model = clone(model)
                fold_model.fit(X_train, y_train)

                # バリデーション予測
                val_pred = fold_model.predict(X_val)
                meta_features[val_idx, i] = val_pred

        return pd.DataFrame(meta_features, columns=list(self.base_models.keys()))


class BlendingEnsemble(BaseEstimator, RegressorMixin):
    """ブレンディングアンサンブル"""

    def __init__(self, base_models: Dict[str, BaseEstimator],
                 blend_ratio: float = 0.8):
        self.base_models = base_models
        self.blend_ratio = blend_ratio
        self.trained_models = {}
        self.blend_weights = {}
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BlendingEnsemble':
        """訓練"""
        # データ分割
        split_idx = int(len(X) * self.blend_ratio)
        X_blend, X_holdout = X.iloc[:split_idx], X.iloc[split_idx:]
        y_blend, y_holdout = y.iloc[:split_idx], y.iloc[split_idx:]

        # ベースモデル訓練
        holdout_predictions = {}
        for name, model in self.base_models.items():
            trained_model = clone(model)
            trained_model.fit(X_blend, y_blend)
            self.trained_models[name] = trained_model

            # ホールドアウト予測
            holdout_pred = trained_model.predict(X_holdout)
            holdout_predictions[name] = holdout_pred

        # 最適重み計算
        self.blend_weights = self._optimize_blend_weights(
            holdout_predictions, y_holdout
        )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測"""
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")

        # 各モデルの予測
        predictions = {}
        for name, model in self.trained_models.items():
            predictions[name] = model.predict(X)

        # 重み付き予測
        final_prediction = np.zeros(len(X))
        for name, weight in self.blend_weights.items():
            final_prediction += weight * predictions[name]

        return final_prediction

    def _optimize_blend_weights(self, predictions: Dict[str, np.ndarray],
                              y_true: pd.Series) -> Dict[str, float]:
        """最適ブレンド重み計算"""
        from scipy.optimize import minimize

        def objective(weights):
            weights = np.abs(weights)
            weights = weights / np.sum(weights)  # 正規化

            blended = np.zeros(len(y_true))
            for i, (name, pred) in enumerate(predictions.items()):
                blended += weights[i] * pred

            return mean_squared_error(y_true, blended)

        # 初期重み（均等）
        n_models = len(predictions)
        initial_weights = np.ones(n_models) / n_models

        # 制約（重みの和=1）
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1}
        bounds = [(0, 1) for _ in range(n_models)]

        # 最適化
        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraints
        )

        # 結果をディクショナリに変換
        optimized_weights = np.abs(result.x)
        optimized_weights = optimized_weights / np.sum(optimized_weights)

        weight_dict = {}
        for i, name in enumerate(predictions.keys()):
            weight_dict[name] = optimized_weights[i]

        return weight_dict


class DynamicWeightingEnsemble(BaseEstimator, RegressorMixin):
    """動的重み付きアンサンブル"""

    def __init__(self, base_models: Dict[str, BaseEstimator],
                 performance_window: int = 20, weight_decay: float = 0.95):
        self.base_models = base_models
        self.performance_window = performance_window
        self.weight_decay = weight_decay
        self.trained_models = {}
        self.performance_history = {name: [] for name in base_models.keys()}
        self.current_weights = {name: 1.0/len(base_models) for name in base_models.keys()}
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DynamicWeightingEnsemble':
        """訓練"""
        # 各モデルを訓練
        for name, model in self.base_models.items():
            trained_model = clone(model)
            trained_model.fit(X, y)
            self.trained_models[name] = trained_model

        # 初期性能評価
        self._evaluate_initial_performance(X, y)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測"""
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")

        # 各モデルの予測
        predictions = {}
        for name, model in self.trained_models.items():
            predictions[name] = model.predict(X)

        # 動的重み付き予測
        final_prediction = np.zeros(len(X))
        total_weight = sum(self.current_weights.values())

        for name, weight in self.current_weights.items():
            normalized_weight = weight / total_weight
            final_prediction += normalized_weight * predictions[name]

        return final_prediction

    def update_weights(self, X: pd.DataFrame, y: pd.Series) -> None:
        """重み更新"""
        if not self.is_fitted:
            return

        # 各モデルの性能評価
        for name, model in self.trained_models.items():
            try:
                pred = model.predict(X)
                mse = mean_squared_error(y, pred)

                # 性能履歴更新
                self.performance_history[name].append(1.0 / (1.0 + mse))  # 性能スコア

                # ウィンドウサイズ制限
                if len(self.performance_history[name]) > self.performance_window:
                    self.performance_history[name] = self.performance_history[name][-self.performance_window:]

            except Exception as e:
                logging.warning(f"モデル {name} の性能評価失敗: {e}")

        # 重み更新
        self._update_current_weights()

    def _evaluate_initial_performance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """初期性能評価"""
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)

        for name, model in self.trained_models.items():
            scores = []
            for train_idx, val_idx in kfold.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                fold_model = clone(model)
                fold_model.fit(X_train, y_train)
                pred = fold_model.predict(X_val)

                mse = mean_squared_error(y_val, pred)
                scores.append(1.0 / (1.0 + mse))

            self.performance_history[name] = scores

    def _update_current_weights(self) -> None:
        """現在の重み更新"""
        new_weights = {}

        for name in self.base_models.keys():
            if self.performance_history[name]:
                # 最近の性能の指数移動平均
                recent_scores = self.performance_history[name]
                weight = 0
                for i, score in enumerate(reversed(recent_scores)):
                    weight += score * (self.weight_decay ** i)

                new_weights[name] = weight
            else:
                new_weights[name] = 0.1  # デフォルト重み

        # 正規化
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for name in new_weights:
                new_weights[name] /= total_weight

        self.current_weights = new_weights


class AdvancedEnsembleSystem:
    """高度アンサンブルシステム"""

    def __init__(self, config: Optional[EnsembleConfiguration] = None):
        self.config = config or EnsembleConfiguration()
        self.base_models = BaseModelFactory.create_models()
        self.ensemble_models = {}
        self.performance_tracker = {}
        self.scaler = StandardScaler()
        self.best_ensemble = None
        self.ensemble_history = []

        self.logger = logging.getLogger(__name__)
        self.setup_ensemble_models()

    def setup_ensemble_models(self) -> None:
        """アンサンブルモデル設定"""
        # ベースモデル辞書を作成
        models_dict = {name: model for name, (model, _) in self.base_models.items()}

        # メタモデル作成
        meta_models = {
            'ridge': Ridge(alpha=1.0),
            'linear': LinearRegression(),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }

        meta_model = meta_models.get(
            self.config.meta_model_type,
            meta_models['ridge']
        )

        # 各種アンサンブル手法を設定
        if self.config.method == EnsembleMethod.STACKING:
            self.ensemble_models['stacking'] = StackingEnsemble(
                base_models=models_dict,
                meta_model=meta_model,
                cv_folds=self.config.cv_folds
            )

        elif self.config.method == EnsembleMethod.BLENDING:
            self.ensemble_models['blending'] = BlendingEnsemble(
                base_models=models_dict,
                blend_ratio=self.config.blend_ratio
            )

        elif self.config.method == EnsembleMethod.DYNAMIC_WEIGHTING:
            self.ensemble_models['dynamic'] = DynamicWeightingEnsemble(
                base_models=models_dict,
                performance_window=self.config.performance_window,
                weight_decay=self.config.weight_decay
            )

        elif self.config.method == EnsembleMethod.VOTING:
            self.ensemble_models['voting'] = VotingRegressor(
                estimators=list(models_dict.items()),
                n_jobs=-1
            )

        # 全手法を評価する場合
        if self.config.method == EnsembleMethod.ADAPTIVE:
            self.ensemble_models.update({
                'stacking': StackingEnsemble(models_dict, clone(meta_model), self.config.cv_folds),
                'blending': BlendingEnsemble(models_dict, self.config.blend_ratio),
                'dynamic': DynamicWeightingEnsemble(models_dict, self.config.performance_window),
                'voting': VotingRegressor(list(models_dict.items()), n_jobs=-1)
            })

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AdvancedEnsembleSystem':
        """システム訓練"""
        self.logger.info(f"アンサンブルシステム訓練開始: {len(self.ensemble_models)}手法")

        # データ正規化
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # 各アンサンブル手法を訓練
        failed_models = []
        for name, ensemble in list(self.ensemble_models.items()):
            try:
                start_time = datetime.now()
                ensemble.fit(X_scaled, y)
                training_time = (datetime.now() - start_time).total_seconds()

                # 性能追跡初期化
                self.performance_tracker[name] = ModelPerformance(
                    model_name=name,
                    category=ModelCategory.ENSEMBLE,
                    training_time=training_time
                )

                self.logger.info(f"{name}アンサンブル訓練完了: {training_time:.2f}秒")

            except Exception as e:
                self.logger.error(f"{name}アンサンブル訓練失敗: {e}")
                failed_models.append(name)
                continue

        # 失敗したモデルを削除
        for name in failed_models:
            if name in self.ensemble_models:
                del self.ensemble_models[name]

        # 最適手法選択（適応的モードの場合）
        if self.config.method == EnsembleMethod.ADAPTIVE:
            self._select_best_ensemble(X_scaled, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測"""
        # データ正規化
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        if self.best_ensemble:
            # 最適手法で予測
            return self.best_ensemble.predict(X_scaled)
        elif len(self.ensemble_models) == 1:
            # 単一手法で予測
            ensemble = next(iter(self.ensemble_models.values()))
            return ensemble.predict(X_scaled)
        else:
            # 全手法の平均
            predictions = []
            weights = []

            for name, ensemble in self.ensemble_models.items():
                try:
                    pred = ensemble.predict(X_scaled)
                    predictions.append(pred)

                    # 性能ベース重み
                    performance = self.performance_tracker.get(name)
                    weight = performance.r2 if performance and performance.r2 > 0 else 0.1
                    weights.append(weight)

                except Exception as e:
                    self.logger.warning(f"{name}予測失敗: {e}")
                    continue

            if predictions:
                # 重み付き平均
                predictions = np.array(predictions)
                weights = np.array(weights)
                weights = weights / np.sum(weights)

                return np.average(predictions, axis=0, weights=weights)
            else:
                self.logger.error("全アンサンブル手法で予測失敗")
                return np.zeros(len(X))

    def _select_best_ensemble(self, X: pd.DataFrame, y: pd.Series) -> None:
        """最適アンサンブル選択"""
        best_score = -np.inf
        best_name = None

        # 交差検証で各手法を評価
        kfold = TimeSeriesSplit(n_splits=3)

        for name, ensemble in self.ensemble_models.items():
            try:
                scores = []
                for train_idx, val_idx in kfold.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    # モデル訓練・評価
                    ensemble_copy = clone(ensemble)
                    ensemble_copy.fit(X_train, y_train)
                    pred = ensemble_copy.predict(X_val)

                    score = r2_score(y_val, pred)
                    scores.append(score)

                avg_score = np.mean(scores)
                self.performance_tracker[name].r2 = avg_score

                if avg_score > best_score:
                    best_score = avg_score
                    best_name = name

                self.logger.info(f"{name}: R2 = {avg_score:.4f}")

            except Exception as e:
                self.logger.warning(f"{name}評価失敗: {e}")
                continue

        if best_name:
            self.best_ensemble = self.ensemble_models[best_name]
            self.logger.info(f"最適アンサンブル選択: {best_name} (R2={best_score:.4f})")

    def update_ensemble_weights(self, X: pd.DataFrame, y: pd.Series) -> None:
        """アンサンブル重み更新"""
        # 動的重み付きアンサンブルの重み更新
        for name, ensemble in self.ensemble_models.items():
            if isinstance(ensemble, DynamicWeightingEnsemble):
                try:
                    X_scaled = pd.DataFrame(
                        self.scaler.transform(X),
                        columns=X.columns,
                        index=X.index
                    )
                    ensemble.update_weights(X_scaled, y)
                    self.logger.debug(f"{name}重み更新完了")

                except Exception as e:
                    self.logger.warning(f"{name}重み更新失敗: {e}")

    def get_ensemble_summary(self) -> Dict[str, Any]:
        """アンサンブル要約"""
        summary = {
            'timestamp': datetime.now(),
            'config': {
                'method': self.config.method.value,
                'cv_folds': self.config.cv_folds,
                'meta_model': self.config.meta_model_type
            },
            'ensemble_models': list(self.ensemble_models.keys()),
            'best_ensemble': type(self.best_ensemble).__name__ if self.best_ensemble else None,
            'performance': {}
        }

        for name, perf in self.performance_tracker.items():
            summary['performance'][name] = {
                'r2': perf.r2,
                'mse': perf.mse,
                'training_time': perf.training_time
            }

        return summary

    def save_ensemble_system(self, filepath: str) -> None:
        """アンサンブルシステム保存"""
        try:
            save_data = {
                'config': self.config,
                'performance_tracker': self.performance_tracker,
                'ensemble_history': self.ensemble_history,
                'best_ensemble_type': type(self.best_ensemble).__name__ if self.best_ensemble else None
            }

            # モデル保存
            models_path = Path(filepath).parent / 'ensemble_models'
            models_path.mkdir(parents=True, exist_ok=True)

            for name, ensemble in self.ensemble_models.items():
                model_file = models_path / f"{name}_ensemble.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(ensemble, f)

            # スケーラー保存
            scaler_file = models_path / "scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)

            # メタデータ保存
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"アンサンブルシステム保存完了: {filepath}")

        except Exception as e:
            self.logger.error(f"システム保存エラー: {e}")


def create_advanced_ensemble_system(
    method: EnsembleMethod = EnsembleMethod.STACKING,
    cv_folds: int = 5,
    meta_model: str = "ridge"
) -> AdvancedEnsembleSystem:
    """高度アンサンブルシステム作成"""
    config = EnsembleConfiguration(
        method=method,
        cv_folds=cv_folds,
        meta_model_type=meta_model
    )

    return AdvancedEnsembleSystem(config)


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)

    # サンプルデータ作成
    np.random.seed(42)
    n_samples, n_features = 1000, 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # 複雑な非線形関係
    y = (X['feature_0'] * 2 +
         X['feature_1'] ** 2 * 0.5 +
         X['feature_2'] * X['feature_3'] * 0.3 +
         np.sin(X['feature_4']) * 1.5 +
         np.random.randn(n_samples) * 0.1)

    # データ分割
    split_idx = int(n_samples * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # アンサンブルシステムテスト
    ensemble_system = create_advanced_ensemble_system(
        method=EnsembleMethod.ADAPTIVE,
        cv_folds=3
    )

    # 訓練
    ensemble_system.fit(X_train, y_train)

    # 予測
    predictions = ensemble_system.predict(X_test)

    # 評価
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"テスト結果:")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")

    # 要約表示
    summary = ensemble_system.get_ensemble_summary()
    print(f"\nアンサンブル要約:")
    print(f"最適手法: {summary['best_ensemble']}")
    print(f"利用手法数: {len(summary['ensemble_models'])}")

    print("高度アンサンブルシステムのテスト完了")