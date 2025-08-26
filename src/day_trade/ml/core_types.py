import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Protocol
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')

# Issue #850-1: カスタム例外クラス
class MLPredictionError(Exception):
    """ML予測システムの基底例外"""
    pass

class DataPreparationError(MLPredictionError):
    """データ準備エラー"""
    pass

class ModelTrainingError(MLPredictionError):
    """モデル訓練エラー"""
    pass

class ModelMetadataError(MLPredictionError):
    """モデルメタデータエラー"""
    pass

class PredictionError(MLPredictionError):
    """予測実行エラー"""
    pass


# Issue #850-1: 列挙型定義の強化
class ModelType(Enum):
    """モデルタイプ（拡張版）"""
    RANDOM_FOREST = "Random Forest"
    XGBOOST = "XGBoost"
    LIGHTGBM = "LightGBM"
    ENSEMBLE = "Ensemble"

class PredictionTask(Enum):
    """予測タスク（拡張版）"""
    PRICE_DIRECTION = "価格方向予測"
    PRICE_REGRESSION = "価格回帰予測"
    VOLATILITY = "ボラティリティ予測"
    TREND_STRENGTH = "トレンド強度予測"

class DataQuality(Enum):
    """データ品質レベル"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INSUFFICIENT = "insufficient"


# Issue #850-3: 強化されたメタデータ管理
@dataclass
class ModelMetadata:
    """モデルメタデータ（強化版）"""
    model_id: str
    model_type: ModelType
    task: PredictionTask
    symbol: str
    version: str
    created_at: datetime
    updated_at: datetime

    # 訓練情報
    feature_columns: List[str]
    target_info: Dict[str, Any]
    training_samples: int
    training_period: str
    data_quality: DataQuality

    # モデル設定
    hyperparameters: Dict[str, Any]
    preprocessing_config: Dict[str, Any]
    feature_selection_config: Dict[str, Any]

    # 性能メトリクス
    performance_metrics: Dict[str, float]
    cross_validation_scores: List[float]
    feature_importance: Dict[str, float]

    # システム情報
    is_classifier: bool
    model_size_mb: float
    training_time_seconds: float
    python_version: str
    sklearn_version: str
    framework_versions: Dict[str, str]

    # 品質管理
    validation_status: str = "pending"
    deployment_status: str = "development"
    performance_threshold_met: bool = False
    data_drift_detected: bool = False

    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            **asdict(self),
            'model_type': self.model_type.value,
            'task': self.task.value,
            'data_quality': self.data_quality.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

@dataclass
class ModelPerformance:
    """モデル性能（強化版）"""
    model_id: str
    symbol: str
    task: PredictionTask
    model_type: ModelType

    # 基本性能指標
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    # クロスバリデーション
    cross_val_mean: float
    cross_val_std: float
    cross_val_scores: List[float]

    # 回帰指標（該当する場合）
    r2_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None

    # 詳細分析
    feature_importance: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict] = None

    # 時間指標
    training_time: float = 0.0
    prediction_time: float = 0.0

    # 品質指標
    prediction_stability: float = 0.0
    confidence_calibration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['task'] = self.task.value
        result['model_type'] = self.model_type.value
        if self.confusion_matrix is not None:
            result['confusion_matrix'] = self.confusion_matrix.tolist()
        return result


# Issue #850-1: データプロバイダープロトコル
class DataProvider(Protocol):
    """データプロバイダーのインターフェース"""

    async def get_stock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """株価データ取得"""
        ...

    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, DataQuality, str]:
        """データ品質評価"""
        ...


# Issue #850-2: 抽象化されたモデル訓練器基底クラス
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

    def prepare_data(self, X: pd.DataFrame, y: pd.Series, config: 'TrainingConfig') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """データ分割の共通処理（強化版）"""
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

    def cross_validate(self, model, X: pd.DataFrame, y: pd.Series, config: 'TrainingConfig') -> np.ndarray:
        """クロスバリデーションの共通処理（強化版）"""
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
