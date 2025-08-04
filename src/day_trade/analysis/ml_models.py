"""
機械学習モデル統合システム
株価予測、方向性予測、リスク予測のための機械学習モデル
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="ml_models")

# オプショナル依存関係のインポート
try:
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        mean_squared_error,
    )
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learnが利用できません。機械学習機能は制限されます。")

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.info("XGBoostが利用できません。")

LIGHTGBM_AVAILABLE = False


@dataclass
class ModelPrediction:
    """機械学習モデル予測結果"""

    prediction: float
    confidence: float
    model_name: str
    metadata: Dict[str, Any] = None


@dataclass
class ModelPerformance:
    """機械学習モデルパフォーマンス"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    metadata: Dict[str, Any] = None


@dataclass
class ModelConfig:
    """機械学習モデル設定"""

    # モデル選択
    model_type: str  # "random_forest", "gradient_boosting", "xgboost", "lightgbm", "svm", "linear"
    task_type: str  # "regression", "classification"

    # 訓練設定
    test_size: float = 0.2
    validation_method: str = "time_series"  # "time_series", "random"
    cv_folds: int = 5

    # モデル固有パラメータ
    model_params: Dict[str, Any] = None

    # 特徴量設定
    feature_selection: bool = True
    max_features: int = 50

    # 前処理設定
    scaling: bool = True
    handle_imbalance: bool = False


class BaseMLModel(ABC):
    """機械学習モデルベースクラス"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False

        if config.scaling and SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()

    @abstractmethod
    def _create_model(self) -> Any:
        """モデルインスタンスを作成"""
        pass

    @abstractmethod
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """モデルを訓練"""
        pass

    @abstractmethod
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """予測を実行"""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        モデルの訓練

        Args:
            X: 特徴量DataFrame
            y: ターゲット変数

        Returns:
            訓練結果の辞書
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learnが必要です")

        logger.info(f"モデル訓練開始: {self.config.model_type}")

        # データの前処理
        X_processed, y_processed = self._preprocess_data(X, y)

        if len(X_processed) < 50:
            raise ValueError("訓練に十分なデータがありません（最低50サンプル必要）")

        # モデルの作成
        self.model = self._create_model()

        # 時系列分割での検証
        if self.config.validation_method == "time_series":
            validation_scores = self._validate_time_series(X_processed, y_processed)
        else:
            validation_scores = self._validate_random(X_processed, y_processed)

        # フルデータでの訓練
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            X_scaled = X_processed

        self._fit_model(X_scaled, y_processed)
        self.feature_names = X.columns.tolist()
        self.is_fitted = True

        # 特徴量重要度の取得
        feature_importance = self._get_feature_importance()

        training_result = {
            "model_type": self.config.model_type,
            "task_type": self.config.task_type,
            "training_samples": len(X_processed),
            "validation_scores": validation_scores,
            "feature_importance": feature_importance,
            "feature_count": len(self.feature_names),
        }

        logger.info(f"モデル訓練完了: {self.config.model_type}")
        return training_result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測実行

        Args:
            X: 特徴量DataFrame

        Returns:
            予測結果
        """
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")

        # 特徴量の整合性チェック
        if self.feature_names and not all(
            col in X.columns for col in self.feature_names
        ):
            missing_features = [
                col for col in self.feature_names if col not in X.columns
            ]
            raise ValueError(f"必要な特徴量が不足しています: {missing_features}")

        # データの前処理
        X_processed = X[self.feature_names].fillna(0)

        X_scaled = self.scaler.transform(X_processed) if self.scaler else X_processed

        # 予測実行
        predictions = self._predict_model(X_scaled)
        return predictions

    def _preprocess_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """データの前処理"""
        # 共通インデックスの取得
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]

        # 欠損値の除去
        valid_mask = ~(X_aligned.isnull().any(axis=1) | y_aligned.isnull())
        X_clean = X_aligned[valid_mask]
        y_clean = y_aligned[valid_mask]

        # 無限値の除去
        inf_mask = ~np.isinf(X_clean).any(axis=1) & ~np.isinf(y_clean)
        X_final = X_clean[inf_mask]
        y_final = y_clean[inf_mask]

        return X_final, y_final

    def _validate_time_series(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """時系列交差検証"""
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # モデルの訓練
            temp_model = self._create_model()
            temp_scaler = StandardScaler() if self.scaler else None

            if temp_scaler:
                X_train_scaled = temp_scaler.fit_transform(X_train)
                X_val_scaled = temp_scaler.transform(X_val)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val

            temp_model.fit(X_train_scaled, y_train)

            # 検証
            val_pred = temp_model.predict(X_val_scaled)

            if self.config.task_type == "regression":
                score = -mean_squared_error(y_val, val_pred)
            else:
                score = accuracy_score(y_val, val_pred)

            scores.append(score)

        return {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "individual_scores": scores,
        }

    def _validate_random(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """ランダム交差検証"""
        scoring = (
            "neg_mean_squared_error"
            if self.config.task_type == "regression"
            else "accuracy"
        )
        scores = cross_val_score(
            self._create_model(), X, y, cv=self.config.cv_folds, scoring=scoring
        )

        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "individual_scores": scores.tolist(),
        }

    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """特徴量重要度の取得"""
        if not hasattr(self.model, "feature_importances_") and not hasattr(
            self.model, "coef_"
        ):
            return None

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(
                self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
            )
        else:
            return None

        if self.feature_names and len(importance) == len(self.feature_names):
            feature_importance = dict(zip(self.feature_names, importance))
            # 重要度順にソート
            return dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

        return None

    def save_model(self, filepath: str) -> None:
        """モデルの保存"""
        if not self.is_fitted:
            raise ValueError("訓練されていないモデルは保存できません")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "config": self.config,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"モデルを保存しました: {filepath}")

    def load_model(self, filepath: str) -> None:
        """モデルの読み込み"""
        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.config = model_data["config"]
        self.is_fitted = True

        logger.info(f"モデルを読み込みました: {filepath}")


class RandomForestModel(BaseMLModel):
    """Random Forestモデル"""

    def _create_model(self) -> Any:
        params = self.config.model_params or {}
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        }
        default_params.update(params)

        if self.config.task_type == "regression":
            return RandomForestRegressor(**default_params)
        else:
            return RandomForestClassifier(**default_params)

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class GradientBoostingModel(BaseMLModel):
    """Gradient Boostingモデル"""

    def _create_model(self) -> Any:
        params = self.config.model_params or {}
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "random_state": 42,
        }
        default_params.update(params)

        if self.config.task_type == "regression":
            return GradientBoostingRegressor(**default_params)
        else:
            # scikit-learn のGradientBoostingClassifierを使用
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(**default_params)

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class XGBoostModel(BaseMLModel):
    """XGBoostモデル"""

    def _create_model(self) -> Any:
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoostがインストールされていません")

        params = self.config.model_params or {}
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "random_state": 42,
        }
        default_params.update(params)

        if self.config.task_type == "regression":
            return xgb.XGBRegressor(**default_params)
        else:
            return xgb.XGBClassifier(**default_params)

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class LinearModel(BaseMLModel):
    """線形モデル"""

    def _create_model(self) -> Any:
        params = self.config.model_params or {}

        if self.config.task_type == "regression":
            return LinearRegression(**params)
        else:
            default_params = {"random_state": 42, "max_iter": 1000}
            default_params.update(params)
            return LogisticRegression(**default_params)

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class MLModelManager:
    """機械学習モデル管理システム"""

    def __init__(self, models_dir: Optional[str] = None):
        """
        Args:
            models_dir: モデル保存ディレクトリ
        """
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.models_dir.mkdir(exist_ok=True)

        self.models: Dict[str, BaseMLModel] = {}
        self.model_configs: Dict[str, ModelConfig] = {}

    def create_model(self, name: str, config: ModelConfig) -> BaseMLModel:
        """
        モデルを作成

        Args:
            name: モデル名
            config: モデル設定

        Returns:
            作成されたモデルインスタンス
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learnが必要です")

        model_type = config.model_type.lower()

        if model_type == "random_forest":
            model = RandomForestModel(config)
        elif model_type == "gradient_boosting":
            model = GradientBoostingModel(config)
        elif model_type == "xgboost":
            model = XGBoostModel(config)
        elif model_type == "linear":
            model = LinearModel(config)
        else:
            raise ValueError(f"サポートされていないモデルタイプ: {model_type}")

        self.models[name] = model
        self.model_configs[name] = config

        logger.info(f"モデルを作成しました: {name} ({model_type})")
        return model

    def train_model(self, name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        モデルを訓練

        Args:
            name: モデル名
            X: 特徴量
            y: ターゲット変数

        Returns:
            訓練結果
        """
        if name not in self.models:
            raise ValueError(f"モデルが見つかりません: {name}")

        return self.models[name].fit(X, y)

    def predict(self, name: str, X: pd.DataFrame) -> np.ndarray:
        """
        予測実行

        Args:
            name: モデル名
            X: 特徴量

        Returns:
            予測結果
        """
        if name not in self.models:
            raise ValueError(f"モデルが見つかりません: {name}")

        return self.models[name].predict(X)

    def save_model(self, name: str) -> None:
        """モデルを保存"""
        if name not in self.models:
            raise ValueError(f"モデルが見つかりません: {name}")

        filepath = self.models_dir / f"{name}.joblib"
        self.models[name].save_model(str(filepath))

    def load_model(self, name: str) -> None:
        """モデルを読み込み"""
        filepath = self.models_dir / f"{name}.joblib"
        if not filepath.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {filepath}")

        # モデルデータから設定を読み込み
        model_data = joblib.load(filepath)
        config = model_data["config"]

        # モデルインスタンスを作成して読み込み
        model = self.create_model(name, config)
        model.load_model(str(filepath))

    def get_model_info(self, name: str) -> Dict[str, Any]:
        """モデル情報を取得"""
        if name not in self.models:
            raise ValueError(f"モデルが見つかりません: {name}")

        model = self.models[name]
        config = self.model_configs[name]

        info = {
            "name": name,
            "model_type": config.model_type,
            "task_type": config.task_type,
            "is_fitted": model.is_fitted,
            "feature_count": len(model.feature_names) if model.feature_names else 0,
            "feature_names": model.feature_names,
            "config": config,
        }

        if model.is_fitted:
            info["feature_importance"] = model._get_feature_importance()

        return info

    def list_models(self) -> List[str]:
        """登録されているモデル名のリストを取得"""
        return list(self.models.keys())


def create_ensemble_predictions(
    models: Dict[str, BaseMLModel],
    X: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    method: str = "average",
) -> np.ndarray:
    """
    アンサンブル予測の実行

    Args:
        models: モデル辞書
        X: 特徴量
        weights: モデル重み
        method: アンサンブル手法（average, weighted_average, voting）

    Returns:
        アンサンブル予測結果
    """
    if not models:
        raise ValueError("予測用のモデルがありません")

    predictions = {}
    for name, model in models.items():
        if model.is_fitted:
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"モデル {name} の予測でエラー: {e}")

    if not predictions:
        raise ValueError("有効な予測結果がありません")

    # アンサンブル計算
    pred_values = list(predictions.values())

    if method == "average":
        ensemble_pred = np.mean(pred_values, axis=0)
    elif method == "weighted_average" and weights:
        weighted_preds = []
        total_weight = 0
        for name, pred in predictions.items():
            weight = weights.get(name, 1.0)
            weighted_preds.append(pred * weight)
            total_weight += weight
        ensemble_pred = np.sum(weighted_preds, axis=0) / total_weight
    elif method == "voting":
        # 分類タスクでの多数決投票
        pred_array = np.array(pred_values)
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=pred_array
        )
    else:
        ensemble_pred = np.mean(pred_values, axis=0)

    return ensemble_pred


# 後方互換性のためのアダプタークラス
def create_default_model_ensemble():
    """デフォルトのモデルアンサンブルを作成"""

    # 簡易的なダミー実装を返す
    class DummyEnsemble:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return []

        def get_model_performances(self):
            return {}

    return DummyEnsemble()


# 後方互換性のためのエイリアス（重複定義を削除）


class MLModelManager:
    """機械学習モデル管理クラス（後方互換性のため）"""

    def __init__(self, models_dir: Optional[str] = None):
        """MLModelManagerの初期化"""
        self.models_dir = models_dir
        self.ensemble = create_default_model_ensemble()
        self.is_trained = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """モデル訓練"""
        self.ensemble.fit(X, y)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> List[ModelPrediction]:
        """予測実行"""
        if not self.is_trained:
            raise ValueError("モデルが訓練されていません")
        return self.ensemble.predict(X)

    def get_performance(self) -> Dict[str, ModelPerformance]:
        """パフォーマンス取得"""
        return self.ensemble.get_model_performances()


def evaluate_prediction_accuracy(
    predictions: List, actual_values: np.ndarray
) -> Dict[str, float]:
    """予測精度評価"""
    if len(predictions) == 0:
        return {"mse": 0.0, "mae": 0.0, "rmse": 0.0}

    pred_values = np.array(predictions)
    if pred_values.ndim > 1:
        pred_values = pred_values.flatten()

    if len(pred_values) != len(actual_values):
        min_len = min(len(pred_values), len(actual_values))
        pred_values = pred_values[:min_len]
        actual_values = actual_values[:min_len]

    mse = np.mean((pred_values - actual_values) ** 2)
    mae = np.mean(np.abs(pred_values - actual_values))

    return {"mse": float(mse), "mae": float(mae), "rmse": float(np.sqrt(mse))}


if __name__ == "__main__":
    # サンプルデータでのテスト
    if SKLEARN_AVAILABLE:
        # テストデータの生成
        np.random.seed(42)
        n_samples = 1000
        n_features = 20

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # 回帰タスクのターゲット
        y_reg = X.iloc[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.1

        # 分類タスクのターゲット
        y_clf = (y_reg > y_reg.median()).astype(int)

        # モデルマネージャーのテスト
        manager = MLModelManager()

        # 回帰モデル
        reg_config = ModelConfig(
            model_type="random_forest", task_type="regression", cv_folds=3
        )

        manager.create_model("rf_regressor", reg_config)
        reg_result = manager.train_model("rf_regressor", X, y_reg)
        print(
            f"回帰モデル訓練結果: {reg_result['validation_scores']['mean_score']:.4f}"
        )

        # 分類モデル
        clf_config = ModelConfig(
            model_type="random_forest", task_type="classification", cv_folds=3
        )

        manager.create_model("rf_classifier", clf_config)
        clf_result = manager.train_model("rf_classifier", X, y_clf)
        print(
            f"分類モデル訓練結果: {clf_result['validation_scores']['mean_score']:.4f}"
        )

        # 予測テスト
        test_x = X.iloc[:10]
        reg_pred = manager.predict("rf_regressor", test_x)
        clf_pred = manager.predict("rf_classifier", test_x)

        print(f"回帰予測例: {reg_pred[:3]}")
        print(f"分類予測例: {clf_pred[:3]}")

        # モデル情報
        print(f"登録モデル: {manager.list_models()}")

    else:
        print("scikit-learnが利用できないため、テストをスキップします。")
