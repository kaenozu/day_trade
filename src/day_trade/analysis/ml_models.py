"""
機械学習予測モデルモジュール

複数の機械学習アルゴリズムを用いて株価予測を行う。
時系列特化型モデルも含む包括的な予測システム。
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger, log_performance_metric

warnings.filterwarnings('ignore', category=FutureWarning)
logger = get_context_logger(__name__)


@dataclass
class ModelPrediction:
    """モデル予測結果"""

    prediction: float
    confidence: float
    prediction_interval: Optional[Tuple[float, float]] = None
    model_name: str = ""
    timestamp: datetime = None
    features_used: List[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.features_used is None:
            self.features_used = []


@dataclass
class ModelPerformance:
    """モデルパフォーマンス評価"""

    model_name: str
    mse: float
    mae: float
    rmse: float
    r2_score: float
    directional_accuracy: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None


class BasePredictionModel(ABC):
    """予測モデルの基底クラス"""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.performance_history = []
        self.kwargs = kwargs

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """モデルの訓練"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> List[ModelPrediction]:
        """予測実行"""
        pass

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelPerformance:
        """モデル性能評価"""
        predictions = self.predict(X)
        pred_values = [p.prediction for p in predictions]

        return self._calculate_performance_metrics(y.values, pred_values)

    def _calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelPerformance:
        """パフォーマンスメトリクス計算"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # 方向性予測精度
        directional_accuracy = np.mean(
            (y_true[1:] > y_true[:-1]) == (y_pred[1:] > y_pred[:-1])
        ) if len(y_true) > 1 else 0.0

        return ModelPerformance(
            model_name=self.name,
            mse=mse,
            mae=mae,
            rmse=rmse,
            r2_score=r2,
            directional_accuracy=directional_accuracy
        )


class LinearRegressionModel(BasePredictionModel):
    """線形回帰モデル"""

    def __init__(self, **kwargs):
        super().__init__("LinearRegression", **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression(**self.kwargs)
        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()
        self.is_trained = True

        logger.info(
            "線形回帰モデル訓練完了",
            section="model_training",
            model_name=self.name,
            features_count=len(self.feature_names),
            samples_count=len(X)
        )

    def predict(self, X: pd.DataFrame) -> List[ModelPrediction]:
        if not self.is_trained:
            raise ValueError("モデルが訓練されていません")

        predictions = self.model.predict(X)

        # 信頼度は係数の重要度に基づいて簡易計算
        feature_importance = np.abs(self.model.coef_)
        avg_importance = np.mean(feature_importance)
        confidence = min(avg_importance * 100, 100.0)

        return [
            ModelPrediction(
                prediction=pred,
                confidence=confidence,
                model_name=self.name,
                features_used=self.feature_names
            ) for pred in predictions
        ]


class RandomForestModel(BasePredictionModel):
    """ランダムフォレストモデル"""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, **kwargs):
        super().__init__("RandomForest", n_estimators=n_estimators, max_depth=max_depth, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(
            n_estimators=self.kwargs.get('n_estimators', 100),
            max_depth=self.kwargs.get('max_depth', 10),
            random_state=42,
            **{k: v for k, v in self.kwargs.items() if k not in ['n_estimators', 'max_depth']}
        )
        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()
        self.is_trained = True

        logger.info(
            "ランダムフォレストモデル訓練完了",
            section="model_training",
            model_name=self.name,
            features_count=len(self.feature_names),
            samples_count=len(X),
            n_estimators=self.model.n_estimators
        )

    def predict(self, X: pd.DataFrame) -> List[ModelPrediction]:
        if not self.is_trained:
            raise ValueError("モデルが訓練されていません")

        predictions = self.model.predict(X)

        # 各決定木の予測の分散を信頼度として使用
        tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        prediction_std = np.std(tree_predictions, axis=0)
        confidence = np.maximum(0, 100 - prediction_std * 100)

        return [
            ModelPrediction(
                prediction=pred,
                confidence=conf,
                model_name=self.name,
                features_used=self.feature_names
            ) for pred, conf in zip(predictions, confidence)
        ]

    def get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度取得"""
        if not self.is_trained:
            return {}

        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


class GradientBoostingModel(BasePredictionModel):
    """勾配ブースティングモデル"""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, **kwargs):
        super().__init__("GradientBoosting", n_estimators=n_estimators, learning_rate=learning_rate, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        self.model = GradientBoostingRegressor(
            n_estimators=self.kwargs.get('n_estimators', 100),
            learning_rate=self.kwargs.get('learning_rate', 0.1),
            random_state=42,
            **{k: v for k, v in self.kwargs.items() if k not in ['n_estimators', 'learning_rate']}
        )
        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()
        self.is_trained = True

        logger.info(
            "勾配ブースティングモデル訓練完了",
            section="model_training",
            model_name=self.name,
            features_count=len(self.feature_names),
            samples_count=len(X)
        )

    def predict(self, X: pd.DataFrame) -> List[ModelPrediction]:
        if not self.is_trained:
            raise ValueError("モデルが訓練されていません")

        predictions = self.model.predict(X)

        # 段階的予測における改善度を信頼度として使用
        staged_predictions = list(self.model.staged_predict(X))
        if len(staged_predictions) > 1:
            improvement = np.abs(staged_predictions[-1] - staged_predictions[-2])
            confidence = np.maximum(0, 100 - improvement * 1000)
        else:
            confidence = np.full(len(predictions), 50.0)

        return [
            ModelPrediction(
                prediction=pred,
                confidence=conf,
                model_name=self.name,
                features_used=self.feature_names
            ) for pred, conf in zip(predictions, confidence)
        ]


class LSTMModel(BasePredictionModel):
    """LSTM時系列予測モデル"""

    def __init__(self, sequence_length: int = 20, lstm_units: int = 50, **kwargs):
        super().__init__("LSTM", sequence_length=sequence_length, lstm_units=lstm_units, **kwargs)
        self.sequence_length = sequence_length
        self.scaler = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from sklearn.preprocessing import MinMaxScaler

            # データの正規化
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)

            # シーケンスデータの準備
            X_seq, y_seq = self._create_sequences(X_scaled, y.values)

            # モデル構築
            self.model = Sequential([
                LSTM(self.kwargs.get('lstm_units', 50), return_sequences=True, input_shape=(self.sequence_length, X.shape[1])),
                Dropout(0.2),
                LSTM(self.kwargs.get('lstm_units', 50), return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])

            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # 訓練
            self.model.fit(
                X_seq, y_seq,
                batch_size=32,
                epochs=self.kwargs.get('epochs', 50),
                validation_split=0.2,
                verbose=0
            )

            self.feature_names = X.columns.tolist()
            self.is_trained = True

            logger.info(
                "LSTMモデル訓練完了",
                section="model_training",
                model_name=self.name,
                sequence_length=self.sequence_length,
                features_count=len(self.feature_names),
                samples_count=len(X_seq)
            )

        except ImportError:
            logger.error(
                "TensorFlowが利用できません",
                section="model_training",
                model_name=self.name
            )
            raise ImportError("TensorFlow が必要です: pip install tensorflow")

    def predict(self, X: pd.DataFrame) -> List[ModelPrediction]:
        if not self.is_trained:
            raise ValueError("モデルが訓練されていません")

        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))

        if len(X_seq) == 0:
            return []

        predictions = self.model.predict(X_seq, verbose=0)

        # LSTM の場合、予測の一貫性を信頼度として使用
        confidence = np.full(len(predictions), 70.0)  # 固定値（実際はより複雑な計算が可能）

        return [
            ModelPrediction(
                prediction=pred[0],
                confidence=conf,
                model_name=self.name,
                features_used=self.feature_names
            ) for pred, conf in zip(predictions, confidence)
        ]

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """時系列シーケンスデータ作成"""
        X_seq, y_seq = [], []

        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)


class ARIMAModel(BasePredictionModel):
    """ARIMA時系列モデル"""

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), **kwargs):
        super().__init__("ARIMA", order=order, **kwargs)
        self.order = order

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            from statsmodels.tsa.arima.model import ARIMA

            # ARIMA は単変量時系列モデルなので、主要な特徴量のみ使用
            if len(X.columns) > 0:
                primary_feature = X.iloc[:, 0]  # 最初の特徴量を使用
            else:
                primary_feature = y  # 特徴量がない場合はターゲット自体

            self.model = ARIMA(primary_feature, order=self.order)
            self.model_fit = self.model.fit()
            self.feature_names = [X.columns[0]] if len(X.columns) > 0 else ['target']
            self.is_trained = True

            logger.info(
                "ARIMAモデル訓練完了",
                section="model_training",
                model_name=self.name,
                order=self.order,
                aic=self.model_fit.aic
            )

        except ImportError:
            logger.error(
                "statsmodels が利用できません",
                section="model_training",
                model_name=self.name
            )
            raise ImportError("statsmodels が必要です: pip install statsmodels")

    def predict(self, X: pd.DataFrame) -> List[ModelPrediction]:
        if not self.is_trained:
            raise ValueError("モデルが訓練されていません")

        forecast = self.model_fit.forecast(steps=len(X))
        confidence_intervals = self.model_fit.get_forecast(steps=len(X)).conf_int()

        predictions = []
        for i, (pred, conf_int) in enumerate(zip(forecast, confidence_intervals.values)):
            confidence = max(0, min(100, (1 - (conf_int[1] - conf_int[0]) / abs(pred)) * 100))

            predictions.append(ModelPrediction(
                prediction=pred,
                confidence=confidence,
                prediction_interval=(conf_int[0], conf_int[1]),
                model_name=self.name,
                features_used=self.feature_names
            ))

        return predictions


class EnsemblePredictor:
    """アンサンブル予測システム"""

    def __init__(self, models: List[BasePredictionModel], voting_method: str = 'weighted'):
        self.models = models
        self.voting_method = voting_method
        self.model_weights = {}
        self.performance_history = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """全モデルの訓練"""
        logger.info(
            "アンサンブルモデル訓練開始",
            section="ensemble_training",
            models_count=len(self.models)
        )

        for model in self.models:
            try:
                model.fit(X, y)

                # 性能評価とウェイト設定
                performance = model.evaluate(X, y)
                self.performance_history[model.name] = performance

                # R²スコアベースのウェイト計算
                weight = max(0, performance.r2_score)
                self.model_weights[model.name] = weight

                logger.info(
                    f"{model.name}モデル訓練完了",
                    section="ensemble_training",
                    r2_score=performance.r2_score,
                    weight=weight
                )

            except Exception as e:
                logger.error(
                    f"{model.name}モデル訓練エラー",
                    section="ensemble_training",
                    error=str(e)
                )
                self.model_weights[model.name] = 0

        # ウェイトの正規化
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}

        logger.info(
            "アンサンブルモデル訓練完了",
            section="ensemble_training",
            final_weights=self.model_weights
        )

    def predict(self, X: pd.DataFrame) -> List[ModelPrediction]:
        """アンサンブル予測"""
        all_predictions = {}

        # 各モデルから予測取得
        for model in self.models:
            if model.is_trained and self.model_weights.get(model.name, 0) > 0:
                try:
                    predictions = model.predict(X)
                    all_predictions[model.name] = predictions
                except Exception as e:
                    logger.warning(
                        f"{model.name}予測エラー",
                        section="ensemble_prediction",
                        error=str(e)
                    )

        if not all_predictions:
            return []

        # アンサンブル予測計算
        ensemble_predictions = []
        num_samples = len(list(all_predictions.values())[0])

        for i in range(num_samples):
            weighted_pred = 0
            weighted_conf = 0
            total_weight = 0

            for model_name, predictions in all_predictions.items():
                if i < len(predictions):
                    weight = self.model_weights.get(model_name, 0)
                    weighted_pred += predictions[i].prediction * weight
                    weighted_conf += predictions[i].confidence * weight
                    total_weight += weight

            if total_weight > 0:
                final_pred = weighted_pred / total_weight
                final_conf = weighted_conf / total_weight
            else:
                final_pred = 0
                final_conf = 0

            ensemble_predictions.append(ModelPrediction(
                prediction=final_pred,
                confidence=final_conf,
                model_name="Ensemble",
                features_used=X.columns.tolist()
            ))

        return ensemble_predictions

    def get_model_performances(self) -> Dict[str, ModelPerformance]:
        """モデル性能履歴取得"""
        return self.performance_history

    def update_weights_by_performance(self, recent_performance: Dict[str, float]) -> None:
        """性能に基づくウェイト更新"""
        for model_name, performance in recent_performance.items():
            if model_name in self.model_weights:
                # 指数移動平均でウェイト更新
                alpha = 0.3
                current_weight = self.model_weights[model_name]
                self.model_weights[model_name] = alpha * performance + (1 - alpha) * current_weight

        # 再正規化
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}


# ユーティリティ関数

def create_default_model_ensemble() -> EnsemblePredictor:
    """デフォルトモデルアンサンブル作成"""
    models = [
        LinearRegressionModel(),
        RandomForestModel(n_estimators=50, max_depth=8),
        GradientBoostingModel(n_estimators=50, learning_rate=0.1)
    ]

    # 高度なモデルは optional
    try:
        models.append(LSTMModel(sequence_length=10, lstm_units=25, epochs=20))
    except ImportError:
        logger.info("TensorFlow利用不可のため、LSTMモデルをスキップ")

    try:
        models.append(ARIMAModel(order=(1, 1, 1)))
    except ImportError:
        logger.info("statsmodels利用不可のため、ARIMAモデルをスキップ")

    return EnsemblePredictor(models)


class MLModelManager:
    """機械学習モデル管理クラス（後方互換性のため）"""

    def __init__(self):
        """MLModelManagerの初期化"""
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
    predictions: List[ModelPrediction],
    actual_values: np.ndarray
) -> Dict[str, float]:
    """予測精度評価"""
    pred_values = np.array([p.prediction for p in predictions])

    mse = np.mean((pred_values - actual_values) ** 2)
    mae = np.mean(np.abs(pred_values - actual_values))
    rmse = np.sqrt(mse)

    # 方向性予測精度
    if len(actual_values) > 1:
        actual_direction = actual_values[1:] > actual_values[:-1]
        pred_direction = pred_values[1:] > pred_values[:-1]
        directional_accuracy = np.mean(actual_direction == pred_direction)
    else:
        directional_accuracy = 0.0

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'directional_accuracy': directional_accuracy
    }


# 使用例とデモ
if __name__ == "__main__":
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 1000

    # 模擬特徴量データ
    features = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples)
    })

    # 模擬ターゲットデータ（特徴量に基づく）
    target = (features['feature_1'] * 0.5 +
              features['feature_2'] * 0.3 +
              features['feature_3'] * 0.2 +
              np.random.randn(n_samples) * 0.1)

    # 訓練・テストデータ分割
    split_idx = int(n_samples * 0.8)
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = target[:split_idx], target[split_idx:]

    # アンサンブルモデル作成・訓練
    ensemble = create_default_model_ensemble()
    ensemble.fit(X_train, y_train)

    # 予測実行
    predictions = ensemble.predict(X_test)

    # 精度評価
    accuracy_metrics = evaluate_prediction_accuracy(predictions, y_test.values)

    logger.info(
        "機械学習モデルデモ完了",
        section="demo",
        models_used=len(ensemble.models),
        accuracy_metrics=accuracy_metrics,
        model_weights=ensemble.model_weights
    )
