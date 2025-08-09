"""
機械学習モデル統合システム（Strategy Pattern実装）

標準ML実装と最適化版を統一し、設定ベースで選択可能なアーキテクチャ
"""

import gc
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..core.optimization_strategy import (
    OptimizationStrategy,
    OptimizationLevel,
    OptimizationConfig,
    optimization_strategy,
    get_optimized_implementation
)
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# オプショナル機械学習依存関係
try:
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn利用可能")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn未利用 - 機械学習機能制限")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logger.info("XGBoost利用可能")
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.info("XGBoost未利用")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    logger.info("LightGBM利用可能")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.info("LightGBM未利用")

try:
    from ..utils.performance_analyzer import profile_performance
    PERFORMANCE_UTILS_AVAILABLE = True
except ImportError:
    PERFORMANCE_UTILS_AVAILABLE = False

warnings.filterwarnings("ignore")


@dataclass
class ModelConfig:
    """統合モデル設定"""
    model_type: str = "random_forest"  # random_forest, gradient_boosting, xgboost, lightgbm
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    test_size: float = 0.2
    cv_folds: int = 5

    # 最適化固有設定
    enable_parallel: bool = True
    enable_caching: bool = True
    max_workers: int = 4
    cache_size: int = 1000
    cache_ttl_hours: int = 1


@dataclass
class ModelPrediction:
    """統合モデル予測結果"""
    prediction: float
    confidence: float = 0.0
    model_name: str = ""
    prediction_time: datetime = field(default_factory=datetime.now)
    computation_time: float = 0.0
    cache_hit: bool = False
    strategy_used: str = ""
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class ModelTrainingResult:
    """モデル訓練結果"""
    model: Any
    training_score: float
    validation_score: float
    training_time: float
    strategy_used: str
    feature_importance: Optional[Dict[str, float]] = None
    cross_validation_scores: Optional[List[float]] = None


class PerformanceCache:
    """高性能キャッシュシステム"""

    def __init__(self, max_size: int = 1000, ttl_hours: int = 1):
        self.cache = {}
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0

    def _generate_key(self, data: pd.DataFrame, model_config: ModelConfig) -> str:
        """キャッシュキー生成"""
        import hashlib
        data_hash = hashlib.md5(str(data.values.tobytes()).encode()).hexdigest()[:16]
        config_hash = hashlib.md5(str(model_config.__dict__).encode()).hexdigest()[:16]
        return f"{data_hash}_{config_hash}"

    def get(self, key: str) -> Optional[Any]:
        """キャッシュから取得"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                self.access_times[key] = datetime.now()
                self.hit_count += 1
                return value
            else:
                # 期限切れエントリの削除
                del self.cache[key]
                del self.access_times[key]

        self.miss_count += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """キャッシュに保存"""
        if len(self.cache) >= self.max_size:
            # LRU削除
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = (value, datetime.now())
        self.access_times[key] = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_size": self.max_size
        }


class MLModelsBase(OptimizationStrategy):
    """機械学習モデルの基底戦略クラス"""

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.model_config = ModelConfig()
        self.trained_models = {}

        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn未利用 - 機械学習機能制限")

    def execute(self, operation: str, *args, **kwargs) -> Any:
        """ML操作の実行"""
        start_time = time.time()

        try:
            if operation == "train":
                result = self._train_model(*args, **kwargs)
            elif operation == "predict":
                result = self._predict(*args, **kwargs)
            elif operation == "evaluate":
                result = self._evaluate_model(*args, **kwargs)
            else:
                raise ValueError(f"未サポート操作: {operation}")

            execution_time = time.time() - start_time
            self.record_execution(execution_time, True)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.record_execution(execution_time, False)
            logger.error(f"ML操作エラー: {e}")
            raise

    def _create_model(self, model_config: ModelConfig):
        """モデル作成"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn未利用")

        if model_config.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=model_config.n_estimators,
                max_depth=model_config.max_depth,
                random_state=42
            )
        elif model_config.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=model_config.n_estimators,
                max_depth=model_config.max_depth,
                learning_rate=model_config.learning_rate,
                random_state=42
            )
        elif model_config.model_type == "xgboost" and XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=model_config.n_estimators,
                max_depth=model_config.max_depth,
                learning_rate=model_config.learning_rate,
                random_state=42
            )
        elif model_config.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            return lgb.LGBMRegressor(
                n_estimators=model_config.n_estimators,
                max_depth=model_config.max_depth,
                learning_rate=model_config.learning_rate,
                random_state=42,
                verbose=-1
            )
        else:
            # フォールバック: Linear Regression
            logger.warning(f"モデル{model_config.model_type}未利用、線形回帰使用")
            return LinearRegression()

    def _train_model(self, X: pd.DataFrame, y: pd.Series, model_config: Optional[ModelConfig] = None) -> ModelTrainingResult:
        """モデル訓練"""
        if model_config:
            self.model_config = model_config

        start_time = time.time()

        # データ分割
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.model_config.test_size, random_state=42
        )

        # 前処理
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # モデル作成・訓練
        model = self._create_model(self.model_config)
        model.fit(X_train_scaled, y_train)

        # 評価
        train_score = model.score(X_train_scaled, y_train)
        val_score = model.score(X_test_scaled, y_test)

        # 特徴重要度
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))

        # クロスバリデーション
        cv_scores = None
        if self.model_config.cv_folds > 1:
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train,
                cv=TimeSeriesSplit(n_splits=self.model_config.cv_folds)
            ).tolist()

        training_time = time.time() - start_time

        # モデル保存
        model_key = f"{self.model_config.model_type}_{hash(str(model_config.__dict__))}"
        self.trained_models[model_key] = (model, scaler)

        return ModelTrainingResult(
            model=model,
            training_score=train_score,
            validation_score=val_score,
            training_time=training_time,
            strategy_used=self.get_strategy_name(),
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores
        )

    def _predict(self, X: pd.DataFrame, model_key: Optional[str] = None) -> ModelPrediction:
        """予測実行"""
        start_time = time.time()

        # モデル選択
        if model_key and model_key in self.trained_models:
            model, scaler = self.trained_models[model_key]
        elif self.trained_models:
            # 最新のモデルを使用
            model, scaler = list(self.trained_models.values())[-1]
        else:
            raise ValueError("訓練済みモデルが存在しません")

        # 予測実行
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)

        # 信頼度計算（可能な場合）
        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_scaled)
            confidence = float(np.max(proba, axis=1).mean())

        computation_time = time.time() - start_time

        return ModelPrediction(
            prediction=float(prediction[0] if len(prediction) == 1 else prediction.mean()),
            confidence=confidence,
            model_name=type(model).__name__,
            computation_time=computation_time,
            strategy_used=self.get_strategy_name()
        )

    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, model_key: Optional[str] = None) -> Dict[str, float]:
        """モデル評価"""
        if model_key and model_key in self.trained_models:
            model, scaler = self.trained_models[model_key]
        elif self.trained_models:
            model, scaler = list(self.trained_models.values())[-1]
        else:
            raise ValueError("訓練済みモデルが存在しません")

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = model.score(X_test_scaled, y_test)

        return {
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2,
            "mean_absolute_error": np.mean(np.abs(y_test - y_pred))
        }


@optimization_strategy("ml_models", OptimizationLevel.STANDARD)
class StandardMLModels(MLModelsBase):
    """標準機械学習モデル実装"""

    def get_strategy_name(self) -> str:
        return "標準MLモデル"


@optimization_strategy("ml_models", OptimizationLevel.OPTIMIZED)
class OptimizedMLModels(MLModelsBase):
    """最適化機械学習モデル実装"""

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.cache = PerformanceCache(
            max_size=self.model_config.cache_size,
            ttl_hours=self.model_config.cache_ttl_hours
        ) if self.model_config.enable_caching else None
        logger.info("最適化MLモデル初期化完了")

    def get_strategy_name(self) -> str:
        return "最適化MLモデル"

    def _train_model(self, X: pd.DataFrame, y: pd.Series, model_config: Optional[ModelConfig] = None) -> ModelTrainingResult:
        """キャッシュ・並列処理機能付き訓練"""
        if model_config:
            self.model_config = model_config

        # キャッシュチェック
        if self.cache and self.model_config.enable_caching:
            cache_key = self.cache._generate_key(X, self.model_config)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug("モデル訓練キャッシュヒット")
                return cached_result

        # 並列処理による高速化（複数モデル同時訓練）
        if self.model_config.enable_parallel and self.model_config.max_workers > 1:
            result = self._train_model_parallel(X, y)
        else:
            result = super()._train_model(X, y, model_config)

        # キャッシュに保存
        if self.cache and self.model_config.enable_caching:
            self.cache.set(cache_key, result)

        # メモリクリーンアップ
        gc.collect()

        return result

    def _train_model_parallel(self, X: pd.DataFrame, y: pd.Series) -> ModelTrainingResult:
        """並列モデル訓練"""
        model_types = ["random_forest", "gradient_boosting"]
        if XGBOOST_AVAILABLE:
            model_types.append("xgboost")
        if LIGHTGBM_AVAILABLE:
            model_types.append("lightgbm")

        best_result = None
        best_score = -np.inf

        with ThreadPoolExecutor(max_workers=self.model_config.max_workers) as executor:
            # 並列訓練タスク投入
            futures = {}
            for model_type in model_types:
                config = ModelConfig(
                    model_type=model_type,
                    n_estimators=self.model_config.n_estimators,
                    max_depth=self.model_config.max_depth,
                    learning_rate=self.model_config.learning_rate
                )

                future = executor.submit(super()._train_model, X, y, config)
                futures[future] = model_type

            # 結果収集
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5分タイムアウト
                    if result.validation_score > best_score:
                        best_score = result.validation_score
                        best_result = result
                        logger.info(f"最良モデル更新: {futures[future]} (score: {best_score:.4f})")
                except Exception as e:
                    logger.warning(f"並列訓練エラー {futures[future]}: {e}")

        if best_result is None:
            # フォールバック: 単一モデル訓練
            logger.warning("並列訓練失敗、標準訓練使用")
            return super()._train_model(X, y)

        return best_result

    def _predict(self, X: pd.DataFrame, model_key: Optional[str] = None) -> ModelPrediction:
        """キャッシュ機能付き予測"""
        start_time = time.time()

        # キャッシュチェック
        cache_key = None
        if self.cache and self.model_config.enable_caching:
            import hashlib
            data_hash = hashlib.md5(str(X.values.tobytes()).encode()).hexdigest()
            cache_key = f"predict_{data_hash}_{model_key or 'default'}"

            cached_result = self.cache.get(cache_key)
            if cached_result:
                cached_result.cache_hit = True
                logger.debug("予測キャッシュヒット")
                return cached_result

        # 予測実行
        result = super()._predict(X, model_key)
        result.strategy_used = self.get_strategy_name()

        # キャッシュに保存
        if self.cache and self.model_config.enable_caching and cache_key:
            self.cache.set(cache_key, result)

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計取得"""
        if self.cache:
            return self.cache.get_stats()
        return {}

    def clear_cache(self) -> None:
        """キャッシュクリア"""
        if self.cache:
            self.cache.cache.clear()
            self.cache.access_times.clear()
            self.cache.hit_count = 0
            self.cache.miss_count = 0
            logger.info("MLモデルキャッシュクリア完了")


# 統合インターフェース
class MLModelsManager:
    """機械学習モデル統合マネージャー"""

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig.from_env()
        self._strategy = None

    def get_strategy(self) -> OptimizationStrategy:
        """現在の戦略を取得"""
        if self._strategy is None:
            self._strategy = get_optimized_implementation("ml_models", self.config)
        return self._strategy

    def train_model(self, X: pd.DataFrame, y: pd.Series, model_config: Optional[ModelConfig] = None) -> ModelTrainingResult:
        """モデル訓練"""
        strategy = self.get_strategy()
        return strategy.execute("train", X, y, model_config)

    def predict(self, X: pd.DataFrame, model_key: Optional[str] = None) -> ModelPrediction:
        """予測実行"""
        strategy = self.get_strategy()
        return strategy.execute("predict", X, model_key)

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, model_key: Optional[str] = None) -> Dict[str, float]:
        """モデル評価"""
        strategy = self.get_strategy()
        return strategy.execute("evaluate", X_test, y_test, model_key)

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要"""
        if self._strategy:
            return self._strategy.get_performance_metrics()
        return {}

    def get_available_models(self) -> List[str]:
        """利用可能モデル一覧"""
        models = ["random_forest", "gradient_boosting", "linear_regression"]
        if XGBOOST_AVAILABLE:
            models.append("xgboost")
        if LIGHTGBM_AVAILABLE:
            models.append("lightgbm")
        return models


# 便利関数
def train_ml_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_config: Optional[ModelConfig] = None,
    optimization_config: Optional[OptimizationConfig] = None
) -> ModelTrainingResult:
    """機械学習モデル訓練のヘルパー関数"""
    manager = MLModelsManager(optimization_config)
    return manager.train_model(X, y, model_config)


def predict_with_ml(
    X: pd.DataFrame,
    model_key: Optional[str] = None,
    optimization_config: Optional[OptimizationConfig] = None
) -> ModelPrediction:
    """機械学習予測のヘルパー関数"""
    manager = MLModelsManager(optimization_config)
    return manager.predict(X, model_key)
