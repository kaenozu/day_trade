#!/usr/bin/env python3
"""
機械学習自動化システム (AutoML)
ハイパーパラメータ最適化・自動モデル選択・パフォーマンス追跡

Features:
- 自動ハイパーパラメータ最適化
- 複数モデル比較・選択
- 交差検証・パフォーマンス評価
- 自動特徴量選択
- モデルパイプライン構築
- 本番環境デプロイ対応
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# オプショナル依存関係
SKLEARN_AVAILABLE = False
OPTUNA_AVAILABLE = False
XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    from sklearn.svm import SVR

    SKLEARN_AVAILABLE = True
except ImportError:
    pass

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    pass

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    pass


class ModelType(Enum):
    """モデルタイプ"""

    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    SVR = "svr"
    ENSEMBLE = "ensemble"


class OptimizationMethod(Enum):
    """最適化手法"""

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    HYPERBAND = "hyperband"
    OPTUNA = "optuna"


@dataclass
class AutoMLConfig:
    """AutoML設定"""

    # モデル選択
    models_to_try: List[ModelType] = None
    optimization_method: OptimizationMethod = OptimizationMethod.OPTUNA

    # 最適化設定
    max_trials: int = 100
    timeout_seconds: int = 3600  # 1時間
    cv_folds: int = 5
    test_size: float = 0.2

    # 特徴量選択
    feature_selection: bool = True
    max_features: int = 50
    feature_selection_method: str = "mutual_info"

    # 前処理
    scaling_method: str = "standard"  # standard, minmax, robust
    handle_missing: bool = True

    # アンサンブル
    enable_ensemble: bool = True
    ensemble_size: int = 3

    # パフォーマンス
    parallel_jobs: int = -1
    memory_limit_gb: float = 8.0

    # 目標メトリクス
    target_metric: str = "rmse"  # rmse, mae, r2
    early_stopping_patience: int = 20

    def __post_init__(self):
        if self.models_to_try is None:
            self.models_to_try = [
                ModelType.RANDOM_FOREST,
                ModelType.GRADIENT_BOOSTING,
                ModelType.RIDGE,
                ModelType.LASSO,
            ]

            if XGBOOST_AVAILABLE:
                self.models_to_try.append(ModelType.XGBOOST)


@dataclass
class ModelPerformance:
    """モデル性能"""

    model_type: ModelType
    train_score: float
    validation_score: float
    test_score: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    training_time: float
    prediction_time: float
    parameters: Dict[str, Any]
    feature_importance: Dict[str, float] = None


@dataclass
class HyperparameterResults:
    """ハイパーパラメータ最適化結果"""

    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_trials: int
    best_trial_number: int
    optimization_time: float
    convergence_curve: List[float]


class AutoMLSystem:
    """機械学習自動化システム"""

    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()

        # モデル管理
        self.trained_models = {}
        self.model_performances = {}
        self.best_model = None
        self.best_model_type = None

        # 最適化履歴
        self.optimization_history = []
        self.feature_importances = {}

        # パイプライン
        self.preprocessing_pipeline = None
        self.feature_selector = None

        # 状態管理
        self.is_fitted = False
        self.training_start_time = None

        logger.info("AutoMLシステム初期化完了")

    async def auto_train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Tuple[pd.DataFrame, pd.Series] = None,
    ) -> Dict[str, Any]:
        """自動訓練実行"""

        self.training_start_time = datetime.now()
        logger.info("AutoML自動訓練開始")

        try:
            # 1. データ前処理
            X_processed, y_processed = await self._preprocess_data(X, y)

            # 2. 特徴量選択
            if self.config.feature_selection:
                X_processed = await self._select_features(X_processed, y_processed)

            # 3. データ分割
            if validation_data is None:
                from sklearn.model_selection import train_test_split

                X_train, X_val, y_train, y_val = train_test_split(
                    X_processed,
                    y_processed,
                    test_size=self.config.test_size,
                    random_state=42,
                )
            else:
                X_train, y_train = X_processed, y_processed
                X_val, y_val = validation_data

            # 4. モデル訓練・最適化
            training_results = {}

            for model_type in self.config.models_to_try:
                logger.info(f"モデル訓練開始: {model_type.value}")

                try:
                    result = await self._train_single_model(
                        model_type, X_train, y_train, X_val, y_val
                    )
                    training_results[model_type.value] = result

                except Exception as e:
                    logger.error(f"モデル訓練エラー {model_type.value}: {e}")
                    continue

            # 5. 最良モデル選択
            self._select_best_model(training_results)

            # 6. アンサンブル構築
            ensemble_result = None
            if self.config.enable_ensemble and len(training_results) >= 2:
                ensemble_result = await self._build_ensemble(X_train, y_train, X_val, y_val)

            # 7. 最終評価
            final_evaluation = await self._final_evaluation(X_val, y_val)

            self.is_fitted = True
            total_training_time = (datetime.now() - self.training_start_time).total_seconds()

            result = {
                "training_results": training_results,
                "best_model": {
                    "type": self.best_model_type.value if self.best_model_type else None,
                    "performance": (
                        asdict(self.model_performances.get(self.best_model_type.value, {}))
                        if self.best_model_type
                        else None
                    ),
                },
                "ensemble_result": ensemble_result,
                "final_evaluation": final_evaluation,
                "total_training_time": total_training_time,
                "feature_importances": self.feature_importances,
                "models_trained": len(training_results),
                "optimization_method": self.config.optimization_method.value,
            }

            logger.info(f"AutoML自動訓練完了: {total_training_time:.2f}秒")
            return result

        except Exception as e:
            logger.error(f"AutoML自動訓練エラー: {e}")
            raise

    async def _preprocess_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """データ前処理"""
        logger.info("データ前処理開始")

        # 欠損値処理
        if self.config.handle_missing:
            X = X.fillna(X.median())

        # スケーリング
        if self.config.scaling_method == "standard":
            scaler = StandardScaler()
        elif self.config.scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif self.config.scaling_method == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        self.preprocessing_pipeline = scaler

        logger.info(f"前処理完了: {X_scaled.shape}")
        return X_scaled, y

    async def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """特徴量選択"""
        logger.info("特徴量選択開始")

        if len(X.columns) <= self.config.max_features:
            logger.info("特徴量数が制限以下のため、特徴量選択をスキップ")
            return X

        if self.config.feature_selection_method == "mutual_info":
            from sklearn.feature_selection import mutual_info_regression

            selector = SelectKBest(score_func=mutual_info_regression, k=self.config.max_features)
        elif self.config.feature_selection_method == "rfe":
            from sklearn.ensemble import RandomForestRegressor

            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=self.config.max_features)
        else:
            # デフォルト: 分散ベース
            from sklearn.feature_selection import VarianceThreshold

            selector = VarianceThreshold()

        X_selected = pd.DataFrame(selector.fit_transform(X, y), index=X.index)

        # 特徴量名の取得
        if hasattr(selector, "get_support"):
            selected_features = X.columns[selector.get_support()]
            X_selected.columns = selected_features
        else:
            X_selected.columns = [f"feature_{i}" for i in range(X_selected.shape[1])]

        self.feature_selector = selector

        logger.info(f"特徴量選択完了: {X.shape[1]} -> {X_selected.shape[1]}")
        return X_selected

    async def _train_single_model(
        self,
        model_type: ModelType,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> ModelPerformance:
        """単一モデル訓練"""

        start_time = datetime.now()

        # モデル作成
        model = self._create_model(model_type)

        # ハイパーパラメータ最適化
        if self.config.optimization_method == OptimizationMethod.OPTUNA and OPTUNA_AVAILABLE:
            best_params, optimization_result = await self._optuna_optimize(
                model_type, X_train, y_train
            )
        else:
            best_params = self._get_default_params(model_type)
            optimization_result = None

        # 最適パラメータでモデル訓練
        model.set_params(**best_params)
        model.fit(X_train, y_train)

        # 性能評価
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_score = self._calculate_score(y_train, train_pred)
        val_score = self._calculate_score(y_val, val_pred)

        # 交差検証
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=TimeSeriesSplit(n_splits=self.config.cv_folds),
            scoring="neg_mean_squared_error",
        )
        cv_scores = -cv_scores  # 正の値に変換

        # 予測時間測定
        pred_start = datetime.now()
        _ = model.predict(X_val.head(10))
        pred_time = (datetime.now() - pred_start).total_seconds() / 10

        training_time = (datetime.now() - start_time).total_seconds()

        # 特徴量重要度
        feature_importance = self._get_feature_importance(model, X_train.columns)

        performance = ModelPerformance(
            model_type=model_type,
            train_score=train_score,
            validation_score=val_score,
            test_score=0.0,  # テスト時に更新
            cv_scores=cv_scores.tolist(),
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            training_time=training_time,
            prediction_time=pred_time,
            parameters=best_params,
            feature_importance=feature_importance,
        )

        # モデル保存
        self.trained_models[model_type.value] = model
        self.model_performances[model_type.value] = performance

        if feature_importance:
            self.feature_importances[model_type.value] = feature_importance

        return performance

    def _create_model(self, model_type: ModelType):
        """モデル作成"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn が必要です")

        if model_type == ModelType.RANDOM_FOREST:
            return RandomForestRegressor(random_state=42, n_jobs=self.config.parallel_jobs)
        elif model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(random_state=42)
        elif model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(random_state=42, n_jobs=self.config.parallel_jobs)
        elif model_type == ModelType.RIDGE:
            return Ridge(random_state=42)
        elif model_type == ModelType.LASSO:
            return Lasso(random_state=42)
        elif model_type == ModelType.ELASTIC_NET:
            return ElasticNet(random_state=42)
        elif model_type == ModelType.SVR:
            return SVR()
        else:
            return RandomForestRegressor(random_state=42)

    async def _optuna_optimize(
        self, model_type: ModelType, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Optuna最適化"""

        def objective(trial):
            params = self._suggest_params(trial, model_type)
            model = self._create_model(model_type)
            model.set_params(**params)

            # 交差検証
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=TimeSeriesSplit(n_splits=min(3, self.config.cv_folds)),
                scoring="neg_mean_squared_error",
                n_jobs=self.config.parallel_jobs,
            )

            return -scores.mean()

        # Study作成
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_warmup_steps=5),
        )

        # 最適化実行
        study.optimize(
            objective,
            n_trials=self.config.max_trials,
            timeout=self.config.timeout_seconds // len(self.config.models_to_try),
        )

        optimization_result = {
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
            "n_trials": len(study.trials),
        }

        return study.best_params, optimization_result

    def _suggest_params(self, trial, model_type: ModelType) -> Dict[str, Any]:
        """パラメータ提案"""

        if model_type == ModelType.RANDOM_FOREST:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }

        elif model_type == ModelType.GRADIENT_BOOSTING:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }

        elif model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            }

        elif model_type == ModelType.RIDGE:
            return {"alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True)}

        elif model_type == ModelType.LASSO:
            return {"alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True)}

        elif model_type == ModelType.ELASTIC_NET:
            return {
                "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            }

        elif model_type == ModelType.SVR:
            return {
                "C": trial.suggest_float("C", 1e-3, 1000, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
            }

        return {}

    def _get_default_params(self, model_type: ModelType) -> Dict[str, Any]:
        """デフォルトパラメータ取得"""

        defaults = {
            ModelType.RANDOM_FOREST: {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            },
            ModelType.GRADIENT_BOOSTING: {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
            },
            ModelType.XGBOOST: {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
            },
            ModelType.RIDGE: {"alpha": 1.0},
            ModelType.LASSO: {"alpha": 1.0},
            ModelType.ELASTIC_NET: {"alpha": 1.0, "l1_ratio": 0.5},
            ModelType.SVR: {"C": 1.0, "gamma": "scale"},
        }

        return defaults.get(model_type, {})

    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """スコア計算"""
        if self.config.target_metric == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.config.target_metric == "mae":
            return mean_absolute_error(y_true, y_pred)
        elif self.config.target_metric == "r2":
            return r2_score(y_true, y_pred)
        else:
            return np.sqrt(mean_squared_error(y_true, y_pred))

    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """特徴量重要度取得"""
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                return dict(zip(feature_names, importances))
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_)
                return dict(zip(feature_names, importances))
            else:
                return {}
        except:
            return {}

    def _select_best_model(self, training_results: Dict[str, ModelPerformance]):
        """最良モデル選択"""
        if not training_results:
            return

        best_score = float("inf") if self.config.target_metric in ["rmse", "mae"] else float("-inf")
        best_model_name = None

        for model_name, performance in training_results.items():
            score = performance.validation_score

            if self.config.target_metric in ["rmse", "mae"]:
                if score < best_score:
                    best_score = score
                    best_model_name = model_name
            else:  # r2
                if score > best_score:
                    best_score = score
                    best_model_name = model_name

        if best_model_name:
            self.best_model_type = ModelType(best_model_name)
            self.best_model = self.trained_models[best_model_name]

            logger.info(f"最良モデル選択: {best_model_name} (score: {best_score:.4f})")

    async def _build_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """アンサンブル構築"""
        logger.info("アンサンブルモデル構築開始")

        # best_score の定義を追加
        best_score = float("inf") if self.config.target_metric in ["rmse", "mae"] else float("-inf")

        try:
            # 上位モデル選択
            performances = list(self.model_performances.values())
            performances.sort(
                key=lambda x: x.validation_score,
                reverse=(self.config.target_metric == "r2"),
            )

            top_models = performances[: self.config.ensemble_size]
            ensemble_models = []
            ensemble_weights = []

            for perf in top_models:
                model = self.trained_models[perf.model_type.value]
                ensemble_models.append(model)

                # 重み計算（性能ベース）
                if self.config.target_metric in ["rmse", "mae"]:
                    weight = 1.0 / (perf.validation_score + 1e-8)
                else:
                    weight = perf.validation_score

                ensemble_weights.append(weight)

            # 重み正規化
            total_weight = sum(ensemble_weights)
            ensemble_weights = [w / total_weight for w in ensemble_weights]

            # アンサンブル予測
            val_predictions = []
            for model in ensemble_models:
                pred = model.predict(X_val)
                val_predictions.append(pred)

            # 重み付き平均
            ensemble_pred = np.average(val_predictions, axis=0, weights=ensemble_weights)
            ensemble_score = self._calculate_score(y_val, ensemble_pred)

            # アンサンブルモデル作成
            class EnsembleModel:
                def __init__(self, models, weights):
                    self.models = models
                    self.weights = weights

                def predict(self, X):
                    predictions = [model.predict(X) for model in self.models]
                    return np.average(predictions, axis=0, weights=self.weights)

            ensemble_model = EnsembleModel(ensemble_models, ensemble_weights)
            self.trained_models["ensemble"] = ensemble_model

            ensemble_performance = ModelPerformance(
                model_type=ModelType.ENSEMBLE,
                train_score=0.0,
                validation_score=ensemble_score,
                test_score=0.0,
                cv_scores=[],
                cv_mean=0.0,
                cv_std=0.0,
                training_time=0.0,
                prediction_time=0.0,
                parameters={"weights": ensemble_weights},
            )

            self.model_performances["ensemble"] = ensemble_performance

            result = {
                "ensemble_score": ensemble_score,
                "component_models": [perf.model_type.value for perf in top_models],
                "weights": ensemble_weights,
                "improvement_over_best": (
                    best_score - ensemble_score
                    if self.config.target_metric in ["rmse", "mae"]
                    else ensemble_score - best_score
                ),
            }

            logger.info(f"アンサンブル構築完了: score={ensemble_score:.4f}")
            return result

        except Exception as e:
            logger.error(f"アンサンブル構築エラー: {e}")
            return None

    async def _final_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """最終評価"""
        logger.info("最終評価開始")

        evaluation_results = {}

        for model_name, model in self.trained_models.items():
            try:
                pred = model.predict(X_test)
                score = self._calculate_score(y_test, pred)

                evaluation_results[model_name] = {
                    "test_score": score,
                    "predictions": pred[:10].tolist(),  # サンプル予測
                }

                # パフォーマンス更新
                if model_name in self.model_performances:
                    self.model_performances[model_name].test_score = score

            except Exception as e:
                logger.error(f"最終評価エラー {model_name}: {e}")
                continue

        return evaluation_results

    def predict(self, X: pd.DataFrame, use_best_model: bool = True) -> np.ndarray:
        """予測実行"""
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")

        # 前処理
        if self.preprocessing_pipeline:
            X_processed = pd.DataFrame(
                self.preprocessing_pipeline.transform(X),
                columns=X.columns,
                index=X.index,
            )
        else:
            X_processed = X

        # 特徴量選択
        if self.feature_selector:
            X_processed = pd.DataFrame(
                self.feature_selector.transform(X_processed), index=X_processed.index
            )

        # 予測
        if use_best_model and self.best_model:
            return self.best_model.predict(X_processed)
        elif "ensemble" in self.trained_models:
            return self.trained_models["ensemble"].predict(X_processed)
        elif self.trained_models:
            # フォールバック: 最初のモデル
            first_model = next(iter(self.trained_models.values()))
            return first_model.predict(X_processed)
        else:
            raise ValueError("利用可能なモデルがありません")

    def get_model_summary(self) -> Dict[str, Any]:
        """モデル概要取得"""
        if not self.model_performances:
            return {"status": "モデル未訓練"}

        performances = []
        for model_name, perf in self.model_performances.items():
            performances.append(
                {
                    "model": model_name,
                    "validation_score": perf.validation_score,
                    "cv_mean": perf.cv_mean,
                    "cv_std": perf.cv_std,
                    "training_time": perf.training_time,
                }
            )

        # ソート
        performances.sort(
            key=lambda x: x["validation_score"],
            reverse=(self.config.target_metric == "r2"),
        )

        return {
            "total_models": len(self.trained_models),
            "best_model": self.best_model_type.value if self.best_model_type else None,
            "optimization_method": self.config.optimization_method.value,
            "target_metric": self.config.target_metric,
            "model_performances": performances,
            "feature_importances_available": len(self.feature_importances) > 0,
            "ensemble_available": "ensemble" in self.trained_models,
            "is_fitted": self.is_fitted,
        }

    def save_models(self, save_dir: str):
        """モデル保存"""
        import pickle

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # モデル保存
        for model_name, model in self.trained_models.items():
            model_file = save_path / f"model_{model_name}.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

        # 前処理パイプライン保存
        if self.preprocessing_pipeline:
            pipeline_file = save_path / "preprocessing_pipeline.pkl"
            with open(pipeline_file, "wb") as f:
                pickle.dump(self.preprocessing_pipeline, f)

        # 特徴量選択器保存
        if self.feature_selector:
            selector_file = save_path / "feature_selector.pkl"
            with open(selector_file, "wb") as f:
                pickle.dump(self.feature_selector, f)

        # メタデータ保存
        metadata = {
            "model_performances": {k: asdict(v) for k, v in self.model_performances.items()},
            "feature_importances": self.feature_importances,
            "best_model_type": self.best_model_type.value if self.best_model_type else None,
            "config": asdict(self.config),
            "is_fitted": self.is_fitted,
        }

        metadata_file = save_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"モデル保存完了: {save_path}")


# グローバルインスタンス
_automl_system = None


def get_automl_system(config: AutoMLConfig = None) -> AutoMLSystem:
    """AutoMLシステム取得"""
    global _automl_system
    if _automl_system is None:
        _automl_system = AutoMLSystem(config)
    return _automl_system
