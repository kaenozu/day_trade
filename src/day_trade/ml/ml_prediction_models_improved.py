#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Prediction Models - 機械学習予測モデル (Issue #850対応改善版)

Issue #850対応: データ頑健性、訓練最適化、メタデータ管理の強化
- データ準備と特徴量エンジニアリングの頑健化
- モデル訓練ロジックの重複排除と抽象化
- モデルの永続化とメタデータ管理の強化
- アンサンブル予測ロジックの洗練
- データベーススキーマとデータ管理の改善
- テストコードの分離とフレームワーク統合
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Protocol
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import json
import pickle
import sqlite3
import warnings
import hashlib
import threading
from abc import ABC, abstractmethod
warnings.filterwarnings('ignore')
import joblib

# 共通ユーティリティ
try:
    from src.day_trade.utils.encoding_utils import setup_windows_encoding
    setup_windows_encoding()
except ImportError:
    # フォールバック: 統合Windows環境対応
    import sys
    import os

    def setup_windows_encoding():
        """Windows環境でのエンコーディング設定（統合版）"""
        if sys.platform == 'win32':
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            try:
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            except (AttributeError, OSError):
                try:
                    import codecs
                    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
                    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
                except Exception:
                    pass

    setup_windows_encoding()

# 機械学習ライブラリ
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression
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

# 外部システムインポート
try:
    from enhanced_feature_engineering import enhanced_feature_engineer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False

try:
    from real_data_provider import RealDataProvider
    REAL_DATA_PROVIDER_AVAILABLE = True
except ImportError:
    REAL_DATA_PROVIDER_AVAILABLE = False


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


# Issue #850-2: 具体的なモデル訓練器クラス
class RandomForestTrainer(BaseModelTrainer):
    """Random Forest訓練器"""

    def create_model(self, is_classifier: bool, hyperparameters: Dict[str, Any]):
        """Random Forestモデル作成"""
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
            raise ImportError("XGBoost is not available")

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
            raise ImportError("LightGBM is not available")

        base_params = self.config.get('classifier_params' if is_classifier else 'regressor_params', {})
        final_params = {**base_params, **hyperparameters}

        if is_classifier:
            return lgb.LGBMClassifier(**final_params)
        else:
            return lgb.LGBMRegressor(**final_params)


@dataclass
class TrainingConfig:
    """訓練設定（強化版）"""
    # データ分割
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    stratify: bool = True

    # クロスバリデーション
    cv_folds: int = 5
    enable_cross_validation: bool = True

    # 特徴量選択
    feature_selection: bool = False
    feature_selection_method: str = "SelectKBest"
    max_features: Optional[int] = None

    # モデル保存
    save_model: bool = True
    save_metadata: bool = True

    # ハイパーパラメータ最適化
    use_optimized_params: bool = True
    optimization_method: str = "grid_search"
    optimization_budget: int = 50

    # 前処理
    enable_scaling: bool = True
    handle_missing_values: bool = True
    outlier_detection: bool = False

    # 品質管理
    min_data_quality: DataQuality = DataQuality.FAIR
    performance_threshold: float = 0.6

    # その他
    n_jobs: int = -1
    verbose: bool = False


@dataclass
class PredictionResult:
    """予測結果（強化版）"""
    symbol: str
    timestamp: datetime
    model_type: ModelType
    task: PredictionTask
    model_version: str

    # 予測結果
    prediction: Union[str, float]
    confidence: float
    prediction_interval: Optional[Tuple[float, float]] = None

    # 詳細情報
    probability_distribution: Dict[str, float] = field(default_factory=dict)
    feature_values: Dict[str, float] = field(default_factory=dict)
    feature_importance_contribution: Dict[str, float] = field(default_factory=dict)

    # メタ情報
    model_performance_history: Dict[str, float] = field(default_factory=dict)
    data_quality_assessment: Optional[DataQuality] = None
    explanation: str = ""

    # 品質指標
    prediction_stability_score: float = 0.0
    confidence_calibration_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['model_type'] = self.model_type.value
        result['task'] = self.task.value
        result['timestamp'] = self.timestamp.isoformat()
        if self.data_quality_assessment:
            result['data_quality_assessment'] = self.data_quality_assessment.value
        return result


@dataclass
class EnsemblePrediction:
    """アンサンブル予測結果（強化版）"""
    symbol: str
    timestamp: datetime

    # 最終予測
    final_prediction: Union[str, float]
    confidence: float
    prediction_interval: Optional[Tuple[float, float]] = None

    # 個別モデル情報
    model_predictions: Dict[str, Any] = field(default_factory=dict)
    model_confidences: Dict[str, float] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)
    model_quality_scores: Dict[str, float] = field(default_factory=dict)

    # アンサンブル品質
    consensus_strength: float = 0.0
    disagreement_score: float = 0.0
    prediction_stability: float = 0.0
    diversity_score: float = 0.0

    # メタ情報
    total_models_used: int = 0
    excluded_models: List[str] = field(default_factory=list)
    ensemble_method: str = "weighted_average"

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


# Issue #850-3 & #850-5: モデルメタデータ管理システム
class ModelMetadataManager:
    """モデルメタデータ管理システム（強化版）"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._init_metadata_database()

    def _init_metadata_database(self):
        """メタデータ用データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # モデルメタデータテーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    task TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    feature_columns TEXT NOT NULL,
                    target_info TEXT NOT NULL,
                    training_samples INTEGER,
                    training_period TEXT,
                    data_quality TEXT,
                    hyperparameters TEXT,
                    preprocessing_config TEXT,
                    feature_selection_config TEXT,
                    performance_metrics TEXT,
                    cross_validation_scores TEXT,
                    feature_importance TEXT,
                    is_classifier BOOLEAN,
                    model_size_mb REAL,
                    training_time_seconds REAL,
                    python_version TEXT,
                    sklearn_version TEXT,
                    framework_versions TEXT,
                    validation_status TEXT DEFAULT 'pending',
                    deployment_status TEXT DEFAULT 'development',
                    performance_threshold_met BOOLEAN DEFAULT FALSE,
                    data_drift_detected BOOLEAN DEFAULT FALSE
                )
            """)

            # モデル性能履歴テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    evaluation_date TEXT NOT NULL,
                    dataset_type TEXT NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    r2_score REAL,
                    mse REAL,
                    rmse REAL,
                    mae REAL,
                    cross_val_mean REAL,
                    cross_val_std REAL,
                    prediction_stability REAL,
                    confidence_calibration REAL,
                    feature_drift_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES model_metadata (model_id)
                )
            """)

            # アンサンブル予測履歴テーブル（強化版）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ensemble_prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    task TEXT NOT NULL,
                    final_prediction TEXT,
                    confidence REAL,
                    consensus_strength REAL,
                    disagreement_score REAL,
                    prediction_stability REAL,
                    diversity_score REAL,
                    model_count INTEGER,
                    avg_model_quality REAL,
                    confidence_variance REAL,
                    model_predictions TEXT,
                    model_weights TEXT,
                    model_quality_scores TEXT,
                    excluded_models TEXT,
                    ensemble_method TEXT,
                    quality_metrics TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 予測精度追跡テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_accuracy_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    predicted_value TEXT,
                    actual_value TEXT,
                    prediction_date TEXT,
                    evaluation_date TEXT,
                    confidence_at_prediction REAL,
                    accuracy_score REAL,
                    error_magnitude REAL,
                    model_used TEXT,
                    task TEXT,
                    was_correct BOOLEAN,
                    error_category TEXT,
                    market_conditions TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # モデル重み履歴テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_weight_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    task TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    static_weight REAL,
                    dynamic_weight REAL,
                    performance_contribution REAL,
                    confidence_contribution REAL,
                    quality_contribution REAL,
                    weight_change_reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def save_metadata(self, metadata: ModelMetadata) -> bool:
        """メタデータ保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO model_metadata VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.model_id,
                    metadata.model_type.value,
                    metadata.task.value,
                    metadata.symbol,
                    metadata.version,
                    metadata.created_at.isoformat(),
                    metadata.updated_at.isoformat(),
                    json.dumps(metadata.feature_columns),
                    json.dumps(metadata.target_info),
                    metadata.training_samples,
                    metadata.training_period,
                    metadata.data_quality.value,
                    json.dumps(metadata.hyperparameters),
                    json.dumps(metadata.preprocessing_config),
                    json.dumps(metadata.feature_selection_config),
                    json.dumps(metadata.performance_metrics),
                    json.dumps(metadata.cross_validation_scores),
                    json.dumps(metadata.feature_importance),
                    metadata.is_classifier,
                    metadata.model_size_mb,
                    metadata.training_time_seconds,
                    metadata.python_version,
                    metadata.sklearn_version,
                    json.dumps(metadata.framework_versions),
                    metadata.validation_status,
                    metadata.deployment_status,
                    metadata.performance_threshold_met,
                    metadata.data_drift_detected
                ))
            return True
        except Exception as e:
            self.logger.error(f"メタデータ保存エラー: {e}")
            return False

    def load_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """メタデータ読み込み"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM model_metadata WHERE model_id = ?
                """, (model_id,))
                row = cursor.fetchone()

                if row:
                    return self._row_to_metadata(row)
                return None

        except Exception as e:
            self.logger.error(f"メタデータ読み込みエラー: {e}")
            return None

    def _row_to_metadata(self, row) -> ModelMetadata:
        """データベース行をModelMetadataに変換"""
        return ModelMetadata(
            model_id=row[0],
            model_type=ModelType(row[1]),
            task=PredictionTask(row[2]),
            symbol=row[3],
            version=row[4],
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
            feature_columns=json.loads(row[7]),
            target_info=json.loads(row[8]),
            training_samples=row[9],
            training_period=row[10],
            data_quality=DataQuality(row[11]),
            hyperparameters=json.loads(row[12]),
            preprocessing_config=json.loads(row[13]),
            feature_selection_config=json.loads(row[14]),
            performance_metrics=json.loads(row[15]),
            cross_validation_scores=json.loads(row[16]),
            feature_importance=json.loads(row[17]),
            is_classifier=bool(row[18]),
            model_size_mb=row[19],
            training_time_seconds=row[20],
            python_version=row[21],
            sklearn_version=row[22],
            framework_versions=json.loads(row[23]),
            validation_status=row[24],
            deployment_status=row[25],
            performance_threshold_met=bool(row[26]),
            data_drift_detected=bool(row[27])
        )

    def get_model_versions(self, symbol: str, model_type: ModelType, task: PredictionTask) -> List[str]:
        """モデルバージョン一覧取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT version FROM model_metadata
                    WHERE symbol = ? AND model_type = ? AND task = ?
                    ORDER BY created_at DESC
                """, (symbol, model_type.value, task.value))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"バージョン一覧取得エラー: {e}")
            return []


# Issue #850-1 & #850-5: 強化されたデータ準備パイプライン
class DataPreparationPipeline:
    """データ準備パイプライン（強化版）"""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler() if self.config.enable_scaling else None
        self.feature_selector = None

    async def prepare_training_data(self, symbol: str, period: str = "1y",
                                  data_provider: Optional[DataProvider] = None) -> Tuple[pd.DataFrame, Dict[PredictionTask, pd.Series], DataQuality]:
        """訓練データ準備（強化版）"""
        try:
            self.logger.info(f"データ準備開始: {symbol}")

            # データ取得
            data = await self._fetch_data(symbol, period, data_provider)

            # データ品質評価
            is_valid, quality, quality_message = self._assess_data_quality(data)
            if not is_valid or quality < self.config.min_data_quality:
                raise DataPreparationError(f"データ品質不足: {quality_message}")

            self.logger.info(f"データ品質評価: {quality.value} - {quality_message}")

            # 特徴量エンジニアリング
            features = await self._engineer_features(symbol, data)

            # 特徴量後処理
            features = self._postprocess_features(features)

            # ターゲット変数作成
            targets = self._create_target_variables(data)

            # データ整合性チェック
            features, targets = self._align_data(features, targets)

            self.logger.info(f"データ準備完了: features={features.shape}, quality={quality.value}")

            return features, targets, quality

        except Exception as e:
            self.logger.error(f"データ準備エラー: {e}")
            raise DataPreparationError(f"データ準備失敗: {e}") from e

    async def _fetch_data(self, symbol: str, period: str, data_provider: Optional[DataProvider]) -> pd.DataFrame:
        """データ取得（強化版）"""
        if data_provider and REAL_DATA_PROVIDER_AVAILABLE:
            # 実データプロバイダー使用
            try:
                data = await data_provider.get_stock_data(symbol, period)
                if not data.empty:
                    return data
                else:
                    self.logger.warning("実データプロバイダーが空データを返しました")
            except Exception as e:
                self.logger.error(f"実データ取得失敗: {e}")

        # フォールバック: 模擬データ生成（開発・テスト環境のみ）
        self.logger.warning("模擬データを生成します（本番環境では推奨されません）")
        return self._generate_mock_data(symbol, period)

    def _generate_mock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """模擬データ生成（改良版）"""
        # より現実的な模擬データを生成
        np.random.seed(hash(symbol) % (2**32))

        if period == "1y":
            days = 252
        elif period == "6mo":
            days = 126
        elif period == "3mo":
            days = 63
        else:
            days = 252

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # より現実的な価格変動シミュレーション
        initial_price = 1000 + np.random.randint(-500, 500)
        returns = np.random.normal(0.0005, 0.02, days)  # 日次リターン
        prices = [initial_price]

        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 10))  # 最小価格制限

        # OHLC価格生成
        high_multiplier = np.random.uniform(1.005, 1.03, days)
        low_multiplier = np.random.uniform(0.97, 0.995, days)

        data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.995, 1.005) for p in prices],
            'High': [p * mult for p, mult in zip(prices, high_multiplier)],
            'Low': [p * mult for p, mult in zip(prices, low_multiplier)],
            'Close': prices,
            'Volume': np.random.lognormal(10, 0.5, days).astype(int)
        }, index=dates)

        # 価格整合性修正
        for i in range(len(data)):
            data.loc[data.index[i], 'High'] = max(data.iloc[i]['High'],
                                                  data.iloc[i]['Open'],
                                                  data.iloc[i]['Close'])
            data.loc[data.index[i], 'Low'] = min(data.iloc[i]['Low'],
                                                 data.iloc[i]['Open'],
                                                 data.iloc[i]['Close'])

        return data

    def _assess_data_quality(self, data: pd.DataFrame) -> Tuple[bool, DataQuality, str]:
        """データ品質評価（詳細版）"""
        try:
            issues = []
            quality_score = 100.0

            # 基本チェック
            if data.empty:
                return False, DataQuality.INSUFFICIENT, "データが空"

            if len(data) < 30:
                return False, DataQuality.INSUFFICIENT, f"データ不足: {len(data)}行"

            # 必須カラムチェック
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                quality_score -= 50
                issues.append(f"必須カラム不足: {missing_columns}")

            # 欠損値チェック
            missing_rate = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_rate > 0.1:
                quality_score -= 20
                issues.append(f"欠損値率高: {missing_rate:.1%}")
            elif missing_rate > 0.05:
                quality_score -= 10
                issues.append(f"欠損値率中: {missing_rate:.1%}")

            # 価格データ整合性
            if 'Open' in data.columns and 'High' in data.columns and 'Low' in data.columns and 'Close' in data.columns:
                invalid_ohlc = ((data['High'] < data['Low']) |
                               (data['High'] < data['Open']) |
                               (data['High'] < data['Close']) |
                               (data['Low'] > data['Open']) |
                               (data['Low'] > data['Close'])).sum()

                if invalid_ohlc > 0:
                    invalid_rate = invalid_ohlc / len(data)
                    quality_score -= min(30, invalid_rate * 100)
                    issues.append(f"OHLC不整合: {invalid_ohlc}件")

            # ゼロまたは負の価格
            if 'Close' in data.columns:
                invalid_prices = (data['Close'] <= 0).sum()
                if invalid_prices > 0:
                    quality_score -= 25
                    issues.append(f"無効価格: {invalid_prices}件")

            # データの連続性（週末除く）
            date_gaps = self._detect_date_gaps(data.index)
            if date_gaps > len(data) * 0.1:
                quality_score -= 15
                issues.append(f"日付ギャップ多: {date_gaps}件")

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

            success = quality_score >= 40
            message = f"スコア: {quality_score:.1f}" + (f" - {'; '.join(issues)}" if issues else "")

            return success, quality, message

        except Exception as e:
            return False, DataQuality.INSUFFICIENT, f"評価エラー: {e}"

    def _detect_date_gaps(self, dates: pd.DatetimeIndex) -> int:
        """日付ギャップ検出"""
        try:
            # 営業日ベースでギャップを検出
            business_days = pd.bdate_range(start=dates.min(), end=dates.max())
            expected_count = len(business_days)
            actual_count = len(dates)
            return max(0, expected_count - actual_count)
        except Exception:
            return 0

    async def _engineer_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """特徴量エンジニアリング（強化版）"""
        try:
            if FEATURE_ENGINEERING_AVAILABLE:
                # 既存の特徴量エンジニアリングシステムを使用
                feature_set = await enhanced_feature_engineer.extract_comprehensive_features(symbol, data)

                if hasattr(feature_set, 'to_dataframe'):
                    features = feature_set.to_dataframe()
                else:
                    features = self._convert_featureset_to_dataframe(feature_set, data)

                # 品質チェック
                if features.empty or len(features.columns) < 5:
                    self.logger.warning("高度特徴量エンジニアリング結果不十分、基本特徴量を使用")
                    features = self._extract_basic_features(data)
                else:
                    self.logger.info(f"高度特徴量エンジニアリング完了: {len(features.columns)}特徴量")

            else:
                # 基本特徴量のみ
                features = self._extract_basic_features(data)
                self.logger.info(f"基本特徴量エンジニアリング完了: {len(features.columns)}特徴量")

            return features

        except Exception as e:
            self.logger.error(f"特徴量エンジニアリングエラー: {e}")
            # フォールバック
            return self._extract_basic_features(data)

    def _extract_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """基本特徴量抽出（改良版）"""
        features = pd.DataFrame(index=data.index)

        try:
            # 価格系特徴量
            features['returns'] = data['Close'].pct_change()
            features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
            features['price_range'] = (data['High'] - data['Low']) / data['Close']
            features['body_size'] = abs(data['Close'] - data['Open']) / data['Close']
            features['upper_shadow'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / data['Close']
            features['lower_shadow'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / data['Close']

            # 移動平均とその比率
            for window in [5, 10, 20, 50]:
                sma = data['Close'].rolling(window).mean()
                features[f'sma_{window}'] = sma
                features[f'sma_ratio_{window}'] = data['Close'] / sma
                features[f'sma_slope_{window}'] = sma.diff() / sma.shift(1)

            # ボラティリティ
            for window in [5, 10, 20]:
                features[f'volatility_{window}'] = features['returns'].rolling(window).std()
                features[f'volatility_ratio_{window}'] = (features[f'volatility_{window}'] /
                                                         features[f'volatility_{window}'].rolling(window*2).mean())

            # 出来高特徴量
            if 'Volume' in data.columns:
                features['volume_ma_20'] = data['Volume'].rolling(20).mean()
                features['volume_ratio'] = data['Volume'] / features['volume_ma_20']
                features['volume_price_trend'] = (data['Volume'] * features['returns']).rolling(5).mean()

            # テクニカル指標
            features = self._add_technical_indicators(features, data)

            # 欠損値処理
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

            return features

        except Exception as e:
            self.logger.error(f"基本特徴量抽出エラー: {e}")
            # 最小限の特徴量
            min_features = pd.DataFrame(index=data.index)
            min_features['returns'] = data['Close'].pct_change().fillna(0)
            min_features['sma_20'] = data['Close'].rolling(20).mean().fillna(method='ffill').fillna(data['Close'])
            return min_features

    def _add_technical_indicators(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標追加"""
        try:
            # RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            features['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']

            # ボリンジャーバンド
            sma_20 = data['Close'].rolling(20).mean()
            std_20 = data['Close'].rolling(20).std()
            features['bb_upper'] = sma_20 + (std_20 * 2)
            features['bb_lower'] = sma_20 - (std_20 * 2)
            features['bb_position'] = (data['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

            return features

        except Exception as e:
            self.logger.warning(f"テクニカル指標追加エラー: {e}")
            return features

    def _convert_featureset_to_dataframe(self, feature_set, data: pd.DataFrame) -> pd.DataFrame:
        """FeatureSetをDataFrameに変換（改良版）"""
        try:
            if isinstance(feature_set, list):
                # 時系列FeatureSetリストの場合
                feature_rows = []
                timestamps = []

                for fs in feature_set:
                    row_features = {}
                    # 各カテゴリの特徴量を統合
                    for category in ['price_features', 'technical_features', 'volume_features',
                                   'momentum_features', 'volatility_features', 'pattern_features',
                                   'market_features', 'statistical_features']:
                        if hasattr(fs, category):
                            features_dict = getattr(fs, category)
                            if isinstance(features_dict, dict):
                                # プレフィックスを追加して名前衝突を回避
                                prefixed_features = {f"{category}_{k}": v for k, v in features_dict.items()}
                                row_features.update(prefixed_features)

                    if row_features:
                        feature_rows.append(row_features)
                        timestamps.append(getattr(fs, 'timestamp', data.index[len(timestamps)]))

                if feature_rows:
                    features_df = pd.DataFrame(feature_rows, index=timestamps[:len(feature_rows)])
                else:
                    features_df = self._extract_basic_features(data)

            else:
                # 単一FeatureSetの場合
                all_features = {}
                for category in ['price_features', 'technical_features', 'volume_features',
                               'momentum_features', 'volatility_features', 'pattern_features',
                               'market_features', 'statistical_features']:
                    if hasattr(feature_set, category):
                        features_dict = getattr(feature_set, category)
                        if isinstance(features_dict, dict):
                            prefixed_features = {f"{category}_{k}": v for k, v in features_dict.items()}
                            all_features.update(prefixed_features)

                if all_features:
                    timestamp = getattr(feature_set, 'timestamp', data.index[-1])
                    features_df = pd.DataFrame([all_features], index=[timestamp])
                else:
                    features_df = self._extract_basic_features(data)

            # 欠損値処理
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

            return features_df

        except Exception as e:
            self.logger.error(f"FeatureSet変換エラー: {e}")
            return self._extract_basic_features(data)

    def _postprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """特徴量後処理"""
        try:
            # 無限値、異常値の処理
            features = features.replace([np.inf, -np.inf], np.nan)

            # 数値列の特定
            numeric_columns = features.select_dtypes(include=[np.number]).columns

            # 外れ値処理（IQR方式）
            if self.config.outlier_detection:
                for col in numeric_columns:
                    Q1 = features[col].quantile(0.25)
                    Q3 = features[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    features[col] = features[col].clip(lower_bound, upper_bound)

            # 欠損値処理
            if self.config.handle_missing_values:
                features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

            # スケーリング
            if self.config.enable_scaling and self.scaler:
                features[numeric_columns] = self.scaler.fit_transform(features[numeric_columns])

            return features

        except Exception as e:
            self.logger.error(f"特徴量後処理エラー: {e}")
            return features

    def _create_target_variables(self, data: pd.DataFrame) -> Dict[PredictionTask, pd.Series]:
        """ターゲット変数作成（改良版）"""
        targets = {}

        try:
            # 価格方向予測（翌日の価格変動方向）
            returns = data['Close'].pct_change().shift(-1)  # 翌日のリターン

            # 閾値を動的に調整（ボラティリティベース）
            volatility = returns.rolling(20).std().fillna(returns.std())
            threshold = volatility * 0.5  # ボラティリティの半分を閾値とする

            direction = pd.Series(index=data.index, dtype='int')
            direction[returns > threshold] = 1   # 上昇
            direction[returns < -threshold] = -1  # 下落
            direction[(returns >= -threshold) & (returns <= threshold)] = 0  # 横ばい

            targets[PredictionTask.PRICE_DIRECTION] = direction

            # 価格回帰予測（翌日の終値）
            targets[PredictionTask.PRICE_REGRESSION] = data['Close'].shift(-1)

            # ボラティリティ予測（翌日の変動率）
            high_low_range = (data['High'] - data['Low']) / data['Close']
            targets[PredictionTask.VOLATILITY] = high_low_range.shift(-1)

            # トレンド強度予測
            trend_strength = abs(returns) / volatility
            targets[PredictionTask.TREND_STRENGTH] = trend_strength.shift(-1)

            return targets

        except Exception as e:
            self.logger.error(f"ターゲット変数作成エラー: {e}")
            # 最小限のターゲット
            simple_direction = pd.Series(0, index=data.index)
            simple_direction[data['Close'].pct_change().shift(-1) > 0] = 1
            simple_direction[data['Close'].pct_change().shift(-1) < 0] = -1
            return {PredictionTask.PRICE_DIRECTION: simple_direction}

    def _align_data(self, features: pd.DataFrame, targets: Dict[PredictionTask, pd.Series]) -> Tuple[pd.DataFrame, Dict[PredictionTask, pd.Series]]:
        """データ整合性確保"""
        try:
            # 共通のインデックスを取得
            common_index = features.index
            for target in targets.values():
                common_index = common_index.intersection(target.index)

            # 最後の行を除外（未来の値が不明）
            common_index = common_index[:-1]

            # データを共通インデックスに合わせる
            aligned_features = features.loc[common_index]
            aligned_targets = {}

            for task, target in targets.items():
                aligned_targets[task] = target.loc[common_index].dropna()

            self.logger.info(f"データ整合完了: 共通サンプル数={len(common_index)}")

            return aligned_features, aligned_targets

        except Exception as e:
            self.logger.error(f"データ整合エラー: {e}")
            return features, targets


# Issue #850-2 & #850-3 & #850-4: メインのMLPredictionModelsクラス（統合改善版）
class MLPredictionModels:
    """機械学習予測モデルシステム（Issue #850対応強化版）"""

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required")

        # ディレクトリ初期化
        self.data_dir = Path("ml_models_data_improved")
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir = self.data_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        # 設定読み込み
        self.config = self._load_config(config_path)

        # データベース初期化
        self.db_path = self.data_dir / "ml_predictions_improved.db"

        # メタデータ管理システム
        self.metadata_manager = ModelMetadataManager(self.db_path)

        # データ準備パイプライン
        self.data_pipeline = DataPreparationPipeline(TrainingConfig())

        # モデル訓練器
        self.trainers = self._init_trainers()

        # 訓練済みモデル
        self.trained_models: Dict[str, Any] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}

        # アンサンブル重み
        self.ensemble_weights: Dict[str, Dict[PredictionTask, Dict[ModelType, float]]] = {}

        # リアルタイム性能追跡（後で実装）
        # self.performance_tracker = ModelPerformanceTracker(self.db_path)

        # 強化されたアンサンブル予測器
        self.ensemble_predictor = EnhancedEnsemblePredictor(self)

        # 訓練済みモデルのロード
        self._load_existing_models()

        self.logger.info("ML prediction models (improved) initialized successfully")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        default_config = {
            'model_configs': {
                ModelType.RANDOM_FOREST: {
                    'classifier_params': {
                        'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10,
                        'min_samples_leaf': 5, 'max_features': 'sqrt', 'random_state': 42,
                        'n_jobs': -1, 'class_weight': 'balanced'
                    },
                    'regressor_params': {
                        'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10,
                        'min_samples_leaf': 5, 'max_features': 'sqrt', 'random_state': 42,
                        'n_jobs': -1
                    }
                },
                ModelType.XGBOOST: {
                    'classifier_params': {
                        'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1,
                        'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
                        'n_jobs': -1, 'eval_metric': 'mlogloss'
                    },
                    'regressor_params': {
                        'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1,
                        'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
                        'n_jobs': -1, 'eval_metric': 'rmse'
                    }
                },
                ModelType.LIGHTGBM: {
                    'classifier_params': {
                        'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1,
                        'random_state': 42, 'n_jobs': -1, 'verbose': -1
                    },
                    'regressor_params': {
                        'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1,
                        'random_state': 42, 'n_jobs': -1, 'verbose': -1
                    }
                }
            },
            'training_config': {
                'performance_threshold': 0.6,
                'min_data_quality': 'fair',
                'enable_cross_validation': True,
                'save_metadata': True
            }
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.endswith('.yaml'):
                        import yaml
                        loaded_config = yaml.safe_load(f)
                    else:
                        loaded_config = json.load(f)

                # 設定をマージ
                for key, value in loaded_config.items():
                    if key in default_config and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value

                self.logger.info(f"設定ファイル読み込み完了: {config_path}")
            except Exception as e:
                self.logger.warning(f"設定ファイル読み込み失敗: {e}, デフォルト設定を使用")

        return default_config

    def _init_trainers(self) -> Dict[ModelType, BaseModelTrainer]:
        """モデル訓練器初期化"""
        trainers = {}
        model_configs = self.config.get('model_configs', {})

        # Random Forest
        trainers[ModelType.RANDOM_FOREST] = RandomForestTrainer(
            ModelType.RANDOM_FOREST,
            model_configs.get(ModelType.RANDOM_FOREST, {}),
            self.logger
        )

        # XGBoost
        if XGBOOST_AVAILABLE:
            trainers[ModelType.XGBOOST] = XGBoostTrainer(
                ModelType.XGBOOST,
                model_configs.get(ModelType.XGBOOST, {}),
                self.logger
            )

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            trainers[ModelType.LIGHTGBM] = LightGBMTrainer(
                ModelType.LIGHTGBM,
                model_configs.get(ModelType.LIGHTGBM, {}),
                self.logger
            )

        self.logger.info(f"訓練器初期化完了: {list(trainers.keys())}")
        return trainers

    def _load_existing_models(self):
        """既存モデルの読み込み"""
        try:
            loaded_count = 0
            for model_file in self.models_dir.glob("*.joblib"):
                try:
                    model_data = joblib.load(model_file)
                    model_key = model_file.stem

                    self.trained_models[model_key] = model_data['model']

                    if 'label_encoder' in model_data:
                        self.label_encoders[model_key] = model_data['label_encoder']

                    if 'metadata' in model_data:
                        self.model_metadata[model_key] = model_data['metadata']

                    loaded_count += 1

                except Exception as e:
                    self.logger.error(f"モデル読み込み失敗 {model_file.name}: {e}")

            self.logger.info(f"既存モデル読み込み完了: {loaded_count}件")

        except Exception as e:
            self.logger.error(f"モデル読み込み処理エラー: {e}")

    async def train_models(self, symbol: str, period: str = "1y",
                          config: Optional[TrainingConfig] = None,
                          optimized_params: Optional[Dict[str, Any]] = None) -> Dict[ModelType, Dict[PredictionTask, ModelPerformance]]:
        """モデル訓練（統合改善版）"""

        config = config or TrainingConfig()
        self.logger.info(f"モデル訓練開始: {symbol}")

        try:
            # データ準備
            features, targets, data_quality = await self.data_pipeline.prepare_training_data(symbol, period)

            # データ品質チェック
            if data_quality < config.min_data_quality:
                raise DataPreparationError(f"データ品質不足: {data_quality.value}")

            # 有効なインデックス取得
            valid_idx = features.index[:-1]  # 最後の行は未来の値が不明
            X = features.loc[valid_idx]

            performances = {}

            # 各モデルタイプで訓練
            for model_type, trainer in self.trainers.items():
                self.logger.info(f"{model_type.value} 訓練開始")
                performances[model_type] = {}

                # 各予測タスクで訓練
                for task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
                    if task not in targets:
                        continue

                    try:
                        perf = await self._train_single_model(
                            model_type, task, trainer, X, targets, symbol,
                            valid_idx, config, data_quality, optimized_params
                        )
                        performances[model_type][task] = perf

                    except Exception as e:
                        self.logger.error(f"{model_type.value}-{task.value} 訓練失敗: {e}")

            # アンサンブル重み計算
            self._calculate_ensemble_weights(performances, symbol)

            # 性能結果保存
            await self._save_training_results(performances, symbol)

            self.logger.info(f"モデル訓練完了: {symbol}")
            return performances

        except Exception as e:
            self.logger.error(f"モデル訓練エラー: {e}")
            raise ModelTrainingError(f"モデル訓練失敗: {e}") from e

    async def _train_single_model(self, model_type: ModelType, task: PredictionTask,
                                trainer: BaseModelTrainer, X: pd.DataFrame,
                                targets: Dict[PredictionTask, pd.Series], symbol: str,
                                valid_idx: pd.Index, config: TrainingConfig,
                                data_quality: DataQuality,
                                optimized_params: Optional[Dict[str, Any]] = None) -> ModelPerformance:
        """単一モデル訓練（共通化された処理）"""

        start_time = datetime.now()

        # ターゲット準備
        y = targets[task].loc[valid_idx].dropna()
        X_clean = X.loc[y.index]

        # データ品質再チェック
        is_valid, quality, message = trainer.validate_data_quality(X_clean, y, task)
        if not is_valid:
            raise ModelTrainingError(f"データ品質チェック失敗: {message}")

        # データ分割
        X_train, X_test, y_train, y_test = trainer.prepare_data(X_clean, y, config)

        # モデル作成
        is_classifier = task == PredictionTask.PRICE_DIRECTION
        hyperparams = optimized_params.get(model_type.value, {}) if optimized_params else {}
        model = trainer.create_model(is_classifier, hyperparams)

        # ラベルエンコーダー（分類の場合）
        label_encoder = None
        if is_classifier:
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
        else:
            y_train_encoded = y_train
            y_test_encoded = y_test

        # モデル訓練
        model.fit(X_train, y_train_encoded)

        # 予測と評価
        y_pred = model.predict(X_test)
        y_pred_proba = None
        if is_classifier and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)

        metrics = trainer.calculate_metrics(y_test_encoded, y_pred, y_pred_proba, is_classifier)

        # クロスバリデーション
        cv_scores = []
        if config.enable_cross_validation:
            cv_scores = trainer.cross_validate(model, X_clean, y_train_encoded if is_classifier else y, config)
            metrics.update({
                'cross_val_mean': np.mean(cv_scores),
                'cross_val_std': np.std(cv_scores)
            })

        # 特徴量重要度
        feature_importance = trainer.get_feature_importance(model, list(X_clean.columns))

        # モデル保存（メタデータ付き）
        if config.save_model:
            await self._save_model_with_enhanced_metadata(
                model, model_type, task, symbol, X_clean.columns, targets[task],
                hyperparams, metrics, cv_scores, feature_importance, is_classifier,
                data_quality, start_time, label_encoder
            )

        training_time = (datetime.now() - start_time).total_seconds()

        # ModelPerformance作成
        return ModelPerformance(
            model_id=f"{symbol}_{model_type.value}_{task.value}",
            symbol=symbol,
            task=task,
            model_type=model_type,
            accuracy=metrics.get('accuracy', 0.0),
            precision=metrics.get('precision', 0.0),
            recall=metrics.get('recall', 0.0),
            f1_score=metrics.get('f1_score', 0.0),
            cross_val_mean=metrics.get('cross_val_mean', 0.0),
            cross_val_std=metrics.get('cross_val_std', 0.0),
            cross_val_scores=cv_scores.tolist() if len(cv_scores) > 0 else [],
            r2_score=metrics.get('r2_score'),
            mse=metrics.get('mse'),
            rmse=metrics.get('rmse'),
            mae=metrics.get('mae'),
            feature_importance=feature_importance,
            training_time=training_time
        )

    async def _save_model_with_enhanced_metadata(self, model, model_type: ModelType, task: PredictionTask,
                                               symbol: str, feature_columns: pd.Index, target_series: pd.Series,
                                               hyperparams: Dict[str, Any], metrics: Dict[str, float],
                                               cv_scores: np.ndarray, feature_importance: Dict[str, float],
                                               is_classifier: bool, data_quality: DataQuality,
                                               start_time: datetime, label_encoder=None):
        """強化されたメタデータ付きモデル保存"""

        try:
            # モデルID生成
            model_id = f"{symbol}_{model_type.value}_{task.value}"
            model_key = model_id

            # バージョン生成
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # フレームワークバージョン情報
            framework_versions = {'sklearn': '1.0.0'}  # 実際のバージョンを取得
            if model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
                import xgboost as xgb
                framework_versions['xgboost'] = xgb.__version__
            elif model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
                import lightgbm as lgb
                framework_versions['lightgbm'] = lgb.__version__

            # メタデータ作成
            metadata = ModelMetadata(
                model_id=model_id,
                model_type=model_type,
                task=task,
                symbol=symbol,
                version=version,
                created_at=start_time,
                updated_at=datetime.now(),
                feature_columns=list(feature_columns),
                target_info={'name': task.value, 'type': 'classification' if is_classifier else 'regression'},
                training_samples=len(target_series.dropna()),
                training_period="1y",  # 設定可能にする
                data_quality=data_quality,
                hyperparameters=hyperparams,
                preprocessing_config={},
                feature_selection_config={},
                performance_metrics=metrics,
                cross_validation_scores=cv_scores.tolist() if len(cv_scores) > 0 else [],
                feature_importance=feature_importance,
                is_classifier=is_classifier,
                model_size_mb=0.0,  # 後で更新
                training_time_seconds=(datetime.now() - start_time).total_seconds(),
                python_version=sys.version,
                sklearn_version='1.0.0',  # 実際のバージョンを取得
                framework_versions=framework_versions
            )

            # モデル保存
            file_path = self.models_dir / f"{model_key}.joblib"
            model_data = {
                'model': model,
                'metadata': metadata
            }
            if label_encoder:
                model_data['label_encoder'] = label_encoder

            joblib.dump(model_data, file_path)

            # ファイルサイズ更新
            metadata.model_size_mb = file_path.stat().st_size / (1024 * 1024)

            # メタデータ保存
            self.metadata_manager.save_metadata(metadata)

            # メモリに保存
            self.trained_models[model_key] = model
            self.model_metadata[model_key] = metadata
            if label_encoder:
                self.label_encoders[model_key] = label_encoder

            self.logger.info(f"モデル保存完了: {model_key} (サイズ: {metadata.model_size_mb:.2f}MB)")

        except Exception as e:
            self.logger.error(f"モデル保存エラー: {e}")
            raise ModelMetadataError(f"モデル保存失敗: {e}") from e

    def _calculate_ensemble_weights(self, performances: Dict[ModelType, Dict[PredictionTask, ModelPerformance]], symbol: str):
        """アンサンブル重み計算（改良版）"""

        if symbol not in self.ensemble_weights:
            self.ensemble_weights[symbol] = {}

        for task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
            task_performances = []
            model_types = []

            for model_type, task_perfs in performances.items():
                if task in task_perfs:
                    perf = task_perfs[task]
                    # 適切なメトリクスを選択
                    if task == PredictionTask.PRICE_DIRECTION:
                        score = perf.f1_score  # 分類ではF1スコア
                    else:
                        score = perf.r2_score or 0.0  # 回帰ではR2スコア

                    task_performances.append(score)
                    model_types.append(model_type)

            if task_performances:
                # 性能に基づくソフトマックス重み計算
                performances_array = np.array(task_performances)
                # 負の値を避けるために最小値を0にシフト
                performances_array = performances_array - np.min(performances_array) + 0.1
                exp_performances = np.exp(performances_array * 5)  # スケーリング
                weights = exp_performances / exp_performances.sum()

                self.ensemble_weights[symbol][task] = dict(zip(model_types, weights))

                self.logger.info(f"アンサンブル重み計算完了 {symbol}-{task.value}: {dict(zip(model_types, weights))}")

    async def _save_training_results(self, performances: Dict[ModelType, Dict[PredictionTask, ModelPerformance]], symbol: str):
        """訓練結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for model_type, task_perfs in performances.items():
                    for task, perf in task_perfs.items():
                        # model_performance_historyテーブルに保存
                        conn.execute("""
                            INSERT INTO model_performance_history
                            (model_id, evaluation_date, dataset_type, accuracy, precision_score,
                             recall_score, f1_score, r2_score, mse, rmse, mae, cross_val_mean, cross_val_std)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            perf.model_id,
                            datetime.now().isoformat(),
                            'training',
                            perf.accuracy,
                            perf.precision,
                            perf.recall,
                            perf.f1_score,
                            perf.r2_score,
                            perf.mse,
                            perf.rmse,
                            perf.mae,
                            perf.cross_val_mean,
                            perf.cross_val_std
                        ))

            self.logger.info(f"訓練結果保存完了: {symbol}")

        except Exception as e:
            self.logger.error(f"訓練結果保存エラー: {e}")

    def get_model_summary(self) -> Dict[str, Any]:
        """モデルサマリー取得（強化版）"""
        try:
            summary = {
                'total_models': len(self.trained_models),
                'model_types': list(self.trainers.keys()),
                'symbols_covered': [],
                'tasks_covered': [],
                'recent_training_activity': [],
                'performance_summary': {}
            }

            # 銘柄とタスクの分析
            symbols = set()
            tasks = set()

            for model_key in self.trained_models.keys():
                parts = model_key.split('_')
                if len(parts) >= 3:
                    symbol = parts[0]
                    task = '_'.join(parts[1:-1])
                    symbols.add(symbol)
                    tasks.add(task)

            summary['symbols_covered'] = list(symbols)
            summary['tasks_covered'] = list(tasks)

            # データベースから最新情報取得
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT model_id, accuracy, f1_score, r2_score, created_at
                        FROM model_performance_history
                        ORDER BY created_at DESC LIMIT 10
                    """)

                    for row in cursor.fetchall():
                        summary['recent_training_activity'].append({
                            'model_id': row[0],
                            'accuracy': row[1],
                            'f1_score': row[2],
                            'r2_score': row[3],
                            'date': row[4]
                        })
            except Exception as db_error:
                self.logger.warning(f"データベース取得エラー: {db_error}")
                summary['recent_training_activity'] = []

            return summary

        except Exception as e:
            self.logger.error(f"サマリー取得エラー: {e}")
            return {'error': str(e)}

    # 統一された予測インターフェース
    async def predict(self, symbol: str, features: pd.DataFrame) -> Dict[PredictionTask, EnsemblePrediction]:
        """統一された予測インターフェース（Issue #850-4対応）"""
        return await self.ensemble_predictor.predict(symbol, features)

    async def predict_list(self, symbol: str, features: pd.DataFrame) -> List[EnsemblePrediction]:
        """リスト形式予測（後方互換性）"""
        predictions_dict = await self.predict(symbol, features)
        return list(predictions_dict.values())


# Issue #850-4: 強化されたアンサンブル予測システム
class EnhancedEnsemblePredictor:
    """強化されたアンサンブル予測システム"""

    def __init__(self, ml_models):
        self.ml_models = ml_models
        self.logger = logging.getLogger(__name__)

    async def predict(self, symbol: str, features: pd.DataFrame) -> Dict[PredictionTask, EnsemblePrediction]:
        """統一された予測インターフェース（強化版）"""

        predictions = {}

        for task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
            try:
                ensemble_pred = await self._make_enhanced_ensemble_prediction(symbol, features, task)
                if ensemble_pred:
                    predictions[task] = ensemble_pred
            except Exception as e:
                self.logger.error(f"予測失敗 {task.value}: {e}")

        return predictions

    async def _make_enhanced_ensemble_prediction(self, symbol: str, features: pd.DataFrame,
                                               task: PredictionTask) -> Optional[EnsemblePrediction]:
        """強化されたアンサンブル予測"""

        model_predictions = {}
        model_confidences = {}
        model_quality_scores = {}
        excluded_models = []

        # 各モデルで予測
        for model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM]:
            model_key = f"{symbol}_{model_type.value}_{task.value}"

            if model_key not in self.ml_models.trained_models:
                excluded_models.append(f"{model_type.value} (not trained)")
                continue

            try:
                model = self.ml_models.trained_models[model_key]
                metadata = self.ml_models.model_metadata.get(model_key)

                # モデル品質スコア
                quality_score = await self._get_model_quality_score(model_key, metadata)
                model_quality_scores[model_type.value] = quality_score

                # 品質が低すぎる場合は除外
                if quality_score < 0.3:
                    excluded_models.append(f"{model_type.value} (low quality: {quality_score:.2f})")
                    continue

                if task == PredictionTask.PRICE_DIRECTION:
                    # 分類予測
                    pred_proba = model.predict_proba(features)
                    pred_class = model.predict(features)[0]

                    # ラベルエンコーダーで逆変換
                    if model_key in self.ml_models.label_encoders:
                        le = self.ml_models.label_encoders[model_key]
                        pred_class = le.inverse_transform([pred_class])[0]

                    confidence = self._calculate_enhanced_classification_confidence(
                        pred_proba[0], quality_score, metadata
                    )

                    model_predictions[model_type.value] = pred_class
                    model_confidences[model_type.value] = confidence

                else:  # 回帰予測
                    pred_value = model.predict(features)[0]
                    confidence = self._calculate_enhanced_regression_confidence(
                        model, features, pred_value, quality_score, metadata
                    )

                    model_predictions[model_type.value] = pred_value
                    model_confidences[model_type.value] = confidence

            except Exception as e:
                self.logger.error(f"予測失敗 {model_key}: {e}")
                excluded_models.append(f"{model_type.value} (error: {str(e)[:50]})")

        if not model_predictions:
            return None

        # 動的重み計算
        dynamic_weights = await self._calculate_enhanced_dynamic_weights(
            symbol, task, model_quality_scores, model_confidences
        )

        # アンサンブル統合
        if task == PredictionTask.PRICE_DIRECTION:
            ensemble_result = self._enhanced_ensemble_classification(
                model_predictions, model_confidences, dynamic_weights
            )
        else:
            ensemble_result = self._enhanced_ensemble_regression(
                model_predictions, model_confidences, dynamic_weights
            )

        # 品質メトリクス計算
        diversity_score = self._calculate_diversity_score(model_predictions)

        return EnsemblePrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            final_prediction=ensemble_result['prediction'],
            confidence=ensemble_result['confidence'],
            prediction_interval=ensemble_result.get('prediction_interval'),
            model_predictions=model_predictions,
            model_confidences=model_confidences,
            model_weights={ModelType(k): v for k, v in dynamic_weights.items()},
            model_quality_scores=model_quality_scores,
            consensus_strength=ensemble_result['consensus_strength'],
            disagreement_score=ensemble_result['disagreement_score'],
            prediction_stability=ensemble_result.get('prediction_stability', 0.0),
            diversity_score=diversity_score,
            total_models_used=len(model_predictions),
            excluded_models=excluded_models,
            ensemble_method="enhanced_weighted_ensemble"
        )

    async def _get_model_quality_score(self, model_key: str, metadata: Optional[ModelMetadata]) -> float:
        """モデル品質スコア取得（強化版）"""
        try:
            if metadata:
                # メタデータから品質スコア計算
                performance_metrics = metadata.performance_metrics

                # 複数メトリクスの重み付き平均
                if metadata.is_classifier:
                    score = (0.4 * performance_metrics.get('accuracy', 0.5) +
                            0.3 * performance_metrics.get('f1_score', 0.5) +
                            0.3 * performance_metrics.get('cross_val_mean', 0.5))
                else:
                    score = (0.5 * max(0, performance_metrics.get('r2_score', 0.0)) +
                            0.3 * performance_metrics.get('cross_val_mean', 0.5) +
                            0.2 * (1.0 - min(1.0, performance_metrics.get('rmse', 1.0))))

                # データ品質による調整
                quality_multiplier = {
                    DataQuality.EXCELLENT: 1.0,
                    DataQuality.GOOD: 0.9,
                    DataQuality.FAIR: 0.8,
                    DataQuality.POOR: 0.6
                }.get(metadata.data_quality, 0.5)

                return np.clip(score * quality_multiplier, 0.0, 1.0)
            else:
                return 0.6  # デフォルト値

        except Exception as e:
            self.logger.error(f"品質スコア計算エラー: {e}")
            return 0.5

    def _calculate_enhanced_classification_confidence(self, pred_proba: np.ndarray,
                                                    quality_score: float,
                                                    metadata: Optional[ModelMetadata]) -> float:
        """強化された分類信頼度計算"""
        # 確率の最大値
        max_prob = np.max(pred_proba)

        # エントロピーベースの不確実性
        entropy = -np.sum(pred_proba * np.log(pred_proba + 1e-15))
        normalized_entropy = entropy / np.log(len(pred_proba))
        certainty = 1.0 - normalized_entropy

        # メタデータベースの調整
        metadata_adjustment = 1.0
        if metadata:
            cv_std = metadata.performance_metrics.get('cross_val_std', 0.1)
            metadata_adjustment = max(0.5, 1.0 - cv_std)  # CV標準偏差が低いほど信頼度高

        # 最終信頼度計算
        confidence = (0.4 * max_prob + 0.3 * certainty + 0.2 * quality_score + 0.1 * metadata_adjustment)

        return np.clip(confidence, 0.1, 0.95)

    def _calculate_enhanced_regression_confidence(self, model, features: pd.DataFrame,
                                                prediction: float, quality_score: float,
                                                metadata: Optional[ModelMetadata]) -> float:
        """強化された回帰信頼度計算"""
        try:
            # アンサンブルモデルの場合、個別予測のばらつきから信頼度推定
            if hasattr(model, 'estimators_'):
                individual_predictions = []
                for estimator in model.estimators_[:min(10, len(model.estimators_))]:
                    try:
                        pred = estimator.predict(features)[0]
                        individual_predictions.append(pred)
                    except:
                        continue

                if individual_predictions:
                    pred_std = np.std(individual_predictions)
                    pred_mean = np.mean(individual_predictions)
                    if pred_mean != 0:
                        cv = abs(pred_std / pred_mean)
                        prediction_consistency = max(0.1, 1.0 - cv)
                    else:
                        prediction_consistency = 0.5
                else:
                    prediction_consistency = 0.5
            else:
                prediction_consistency = 0.7  # デフォルト値

            # メタデータベースの調整
            metadata_adjustment = 1.0
            if metadata:
                r2_score = metadata.performance_metrics.get('r2_score', 0.5)
                metadata_adjustment = max(0.3, r2_score)

            # 最終信頼度計算
            confidence = (0.4 * quality_score + 0.3 * prediction_consistency + 0.3 * metadata_adjustment)

            return np.clip(confidence, 0.1, 0.95)

        except Exception as e:
            self.logger.error(f"回帰信頼度計算エラー: {e}")
            return quality_score * 0.8

    async def _calculate_enhanced_dynamic_weights(self, symbol: str, task: PredictionTask,
                                                quality_scores: Dict[str, float],
                                                confidences: Dict[str, float]) -> Dict[str, float]:
        """強化された動的重み計算"""
        try:
            # 基本重み（過去の性能ベース）
            base_weights = self.ml_models.ensemble_weights.get(symbol, {}).get(task, {})

            dynamic_weights = {}
            total_score = 0

            for model_name in quality_scores.keys():
                try:
                    model_type = ModelType(model_name)
                except ValueError:
                    continue

                # 各要素の重み
                quality = quality_scores[model_name]
                confidence = confidences[model_name]
                base_weight = base_weights.get(model_type, 1.0)

                # 時間減衰ファクター（新しいモデルほど重視）
                time_decay = 1.0  # 実装可能: モデルの新しさに基づく重み

                # 動的重み計算
                dynamic_weight = (0.3 * quality + 0.3 * confidence + 0.2 * base_weight + 0.2 * time_decay)
                dynamic_weights[model_name] = dynamic_weight
                total_score += dynamic_weight

            # 正規化
            if total_score > 0:
                for model_name in dynamic_weights:
                    dynamic_weights[model_name] /= total_score

            return dynamic_weights

        except Exception as e:
            self.logger.error(f"動的重み計算エラー: {e}")
            # フォールバック: 均等重み
            num_models = len(quality_scores)
            return {name: 1.0/num_models for name in quality_scores.keys()}

    def _enhanced_ensemble_classification(self, predictions: Dict[str, Any],
                                        confidences: Dict[str, float],
                                        weights: Dict[str, float]) -> Dict[str, Any]:
        """強化された分類アンサンブル"""
        weighted_votes = {}
        total_weight = 0
        prediction_values = list(predictions.values())

        # 重み付き投票
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.0)
            confidence = confidences[model_name]

            vote_strength = weight * confidence
            if prediction not in weighted_votes:
                weighted_votes[prediction] = 0
            weighted_votes[prediction] += vote_strength
            total_weight += vote_strength

        # 最終予測
        final_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]

        # アンサンブル信頼度
        if total_weight > 0:
            ensemble_confidence = weighted_votes[final_prediction] / total_weight
        else:
            ensemble_confidence = 0.5

        # コンセンサス強度
        unique_predictions = len(set(prediction_values))
        consensus_strength = 1.0 - (unique_predictions - 1) / max(1, len(predictions) - 1)

        # 予測安定性
        prediction_stability = self._calculate_prediction_stability_classification(predictions)

        return {
            'prediction': final_prediction,
            'confidence': np.clip(ensemble_confidence, 0.1, 0.95),
            'consensus_strength': consensus_strength,
            'disagreement_score': 1.0 - consensus_strength,
            'prediction_stability': prediction_stability
        }

    def _enhanced_ensemble_regression(self, predictions: Dict[str, float],
                                    confidences: Dict[str, float],
                                    weights: Dict[str, float]) -> Dict[str, Any]:
        """強化された回帰アンサンブル"""
        weighted_sum = 0
        total_weight = 0
        pred_values = list(predictions.values())

        # 重み付き平均
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.0)
            confidence = confidences[model_name]

            adjusted_weight = weight * confidence
            weighted_sum += prediction * adjusted_weight
            total_weight += adjusted_weight

        final_prediction = weighted_sum / total_weight if total_weight > 0 else np.mean(pred_values)

        # 予測区間推定
        prediction_std = np.std(pred_values)
        prediction_interval = (final_prediction - 1.96 * prediction_std,
                             final_prediction + 1.96 * prediction_std)

        # アンサンブル信頼度
        avg_confidence = np.mean(list(confidences.values()))
        prediction_variance = np.var(pred_values)
        prediction_mean = np.mean(pred_values)

        if prediction_mean != 0:
            cv = prediction_variance / abs(prediction_mean)
            consistency_factor = max(0.1, 1.0 - np.tanh(cv))
        else:
            consistency_factor = 0.5

        ensemble_confidence = avg_confidence * consistency_factor

        # コンセンサス強度
        if prediction_mean != 0:
            consensus_strength = max(0.0, 1.0 - (prediction_std / abs(prediction_mean)))
        else:
            consensus_strength = max(0.0, 1.0 - prediction_std)

        return {
            'prediction': final_prediction,
            'confidence': np.clip(ensemble_confidence, 0.1, 0.95),
            'prediction_interval': prediction_interval,
            'consensus_strength': np.clip(consensus_strength, 0.0, 1.0),
            'disagreement_score': 1.0 - np.clip(consensus_strength, 0.0, 1.0),
            'prediction_stability': self._calculate_prediction_stability_regression(predictions)
        }

    def _calculate_diversity_score(self, predictions: Dict[str, Any]) -> float:
        """予測多様性スコア計算"""
        try:
            pred_values = list(predictions.values())
            if len(set(str(p) for p in pred_values)) == 1:
                return 0.0  # 完全一致

            if all(isinstance(p, (int, float)) for p in pred_values):
                # 数値予測の多様性
                normalized_std = np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-8)
                return min(1.0, normalized_std)
            else:
                # カテゴリ予測の多様性
                unique_count = len(set(str(p) for p in pred_values))
                return (unique_count - 1) / max(1, len(pred_values) - 1)

        except Exception:
            return 0.5

    def _calculate_prediction_stability_classification(self, predictions: Dict[str, Any]) -> float:
        """分類予測安定性計算"""
        pred_values = list(predictions.values())
        if len(set(str(p) for p in pred_values)) == 1:
            return 1.0  # 完全一致

        # 最も多い予測の割合
        from collections import Counter
        counts = Counter(str(p) for p in pred_values)
        most_common_count = counts.most_common(1)[0][1]
        stability = most_common_count / len(pred_values)

        return stability

    def _calculate_prediction_stability_regression(self, predictions: Dict[str, float]) -> float:
        """回帰予測安定性計算"""
        pred_values = list(predictions.values())
        if len(pred_values) <= 1:
            return 1.0

        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)

        if mean_pred != 0:
            cv = std_pred / abs(mean_pred)
            stability = max(0.0, 1.0 - cv)
        else:
            stability = max(0.0, 1.0 - std_pred)

        return np.clip(stability, 0.0, 1.0)




# Issue #850-6: ファクトリー関数とユーティリティ
def create_improved_ml_prediction_models(config_path: Optional[str] = None) -> MLPredictionModels:
    """改善版MLPredictionModelsの作成"""
    return MLPredictionModels(config_path)

# グローバルインスタンス（後方互換性）
try:
    ml_prediction_models_improved = MLPredictionModels()
except Exception as e:
    logging.getLogger(__name__).error(f"改善版MLモデル初期化失敗: {e}")
    ml_prediction_models_improved = None


# Issue #850-6: テストコードの分離完了
if __name__ == "__main__":
    # 基本的な動作確認のみ
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logger = logging.getLogger(__name__)
    logger.info("ML Prediction Models (Issue #850対応改善版)")
    logger.info("主要改善点:")
    logger.info("- データ準備と特徴量エンジニアリングの頑健化")
    logger.info("- モデル訓練ロジックの重複排除と抽象化")
    logger.info("- モデルの永続化とメタデータ管理の強化")
    logger.info("- アンサンブル予測ロジックの洗練")
    logger.info("- データベーススキーマとデータ管理の改善")
    logger.info("- テストコードの分離とフレームワーク統合")
    logger.info("")
    logger.info("詳細なテストは tests/test_ml_prediction_models_improved.py を実行してください")

    try:
        models = MLPredictionModels()
        logger.info(f"✓ 初期化成功: {models.data_dir}")
        logger.info(f"✓ データベース: {models.db_path}")
        logger.info(f"✓ 利用可能訓練器: {list(models.trainers.keys())}")

        # 基本設定確認
        summary = models.get_model_summary()
        logger.info(f"✓ モデルサマリー: {summary.get('total_models', 0)}個のモデル")

    except Exception as e:
        logger.error(f"✗ 初期化エラー: {e}")