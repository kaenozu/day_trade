#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Prediction Models - 機械学習予測モデル

Random Forest・XGBoost・LightGBMによる高精度予測システム
Issue #796-2実装：Random Forest・XGBoost導入
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import pickle
import sqlite3
import warnings
warnings.filterwarnings('ignore')
import joblib # Added for model saving/loading

# 共通ユーティリティ
try:
    from src.day_trade.utils.encoding_utils import setup_windows_encoding
    setup_windows_encoding()
except ImportError:
    # フォールバック: 簡易Windows対応
    import sys
    import os
    if sys.platform == 'win32':
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass

# 機械学習ライブラリ
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.metrics import mean_squared_error, r2_score
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

# 既存システムのインポート
try:
    from enhanced_feature_engineering import enhanced_feature_engineer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False

try:
    from real_data_provider_v2 import real_data_provider
    REAL_DATA_PROVIDER_AVAILABLE = True
except ImportError:
    REAL_DATA_PROVIDER_AVAILABLE = False

class ModelType(Enum):
    """モデルタイプ"""
    RANDOM_FOREST = "Random Forest"
    XGBOOST = "XGBoost"
    LIGHTGBM = "LightGBM"
    ENSEMBLE = "Ensemble"

class PredictionTask(Enum):
    """予測タスク"""
    PRICE_DIRECTION = "価格方向予測"      # 上昇/下落/横ばい
    PRICE_REGRESSION = "価格回帰予測"     # 具体的価格予測
    VOLATILITY = "ボラティリティ予測"     # 変動率予測
    TREND_STRENGTH = "トレンド強度予測"   # トレンドの強さ

@dataclass
class ModelPerformance:
    """モデル性能"""
    model_name: str
    task: PredictionTask
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cross_val_mean: float
    cross_val_std: float
    feature_importance: Dict[str, float]
    confusion_matrix: np.ndarray
    training_time: float
    prediction_time: float

@dataclass
class ModelMetadata:
    """モデルメタデータ"""
    model_type: ModelType
    task: PredictionTask
    version: str
    created_at: datetime
    feature_names: List[str]
    target_columns: List[str]
    training_period: str
    training_samples: int
    hyperparameters: Dict[str, Any]
    preprocessing_info: Dict[str, Any]
    performance_metrics: Dict[str, float]
    is_classifier: bool
    model_size_mb: float
    python_version: str
    sklearn_version: str

    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)

@dataclass
class TrainingConfig:
    """訓練設定"""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    enable_cross_validation: bool = True
    save_model: bool = True
    use_optimized_params: bool = True
    feature_selection: bool = False
    preprocessing: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionResult:
    """予測結果"""
    symbol: str
    timestamp: datetime
    model_type: ModelType
    task: PredictionTask
    prediction: Union[str, float]
    confidence: float
    probability_distribution: Dict[str, float]
    feature_values: Dict[str, float]
    model_version: str
    explanation: str

@dataclass
class EnsemblePrediction:
    """アンサンブル予測結果"""
    symbol: str
    timestamp: datetime
    final_prediction: Union[str, float]
    confidence: float
    model_predictions: Dict[str, Any]
    model_weights: Dict[str, float]
    consensus_strength: float
    disagreement_score: float

class MLPredictionModels:
    """機械学習予測モデルシステム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required")

        # データディレクトリ初期化
        self.data_dir = Path("ml_models_data")
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        # モデル設定
        self.model_configs = {
            ModelType.RANDOM_FOREST: {
                'classifier_params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1,
                    'class_weight': 'balanced'
                },
                'regressor_params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            ModelType.XGBOOST: {
                'classifier_params': {
                    'n_estimators': 300,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1,
                    'eval_metric': 'mlogloss'
                },
                'regressor_params': {
                    'n_estimators': 300,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1,
                    'eval_metric': 'rmse'
                }
            }
        }

        # 訓練済みモデル
        self.trained_models = {}
        self.scalers = {}
        self.label_encoders = {}

        # パフォーマンス記録
        self.model_performances = {}

        # アンサンブル重み（性能ベース）
        self.ensemble_weights = {}

        # 訓練済みモデルのロード
        self._load_trained_models()

        # データベース初期化
        self.db_path = self.data_dir / "ml_predictions.db"
        self._init_database()

        self.logger.info("ML prediction models initialized")

    def _load_trained_models(self):
        """保存されたモデルをロードする (joblibを使用) """
        for model_file in self.models_dir.glob("*.joblib"):
            try:
                model_data = joblib.load(model_file)
                model_key = model_file.stem
                self.trained_models[model_key] = model_data['model']
                if 'label_encoder' in model_data:
                    self.label_encoders[model_key] = model_data['label_encoder']
                self.logger.info(f"Loaded model: {model_key}")
            except Exception as e:
                self.logger.error(f"Failed to load model {model_file.name}: {e}")

    def _save_model(self, model, model_key: str, label_encoder=None):
        """モデルをファイルに保存する (joblibを使用) """
        file_path = self.models_dir / f"{model_key}.joblib"
        try:
            model_data = {'model': model}
            if label_encoder:
                model_data['label_encoder'] = label_encoder
            joblib.dump(model_data, file_path)
            self.logger.info(f"Saved model: {model_key}.joblib")
        except Exception as e:
            self.logger.error(f"Failed to save model {model_key}: {e}")

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # モデル性能テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    task TEXT NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    cross_val_mean REAL,
                    cross_val_std REAL,
                    training_time REAL,
                    training_date TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 予測結果テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_type TEXT,
                    task TEXT,
                    prediction TEXT,
                    confidence REAL,
                    model_version TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # アンサンブル予測テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ensemble_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    final_prediction TEXT,
                    confidence REAL,
                    consensus_strength REAL,
                    disagreement_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _create_model_metadata(self, model_type: ModelType, task: PredictionTask,
                              feature_names: List[str], target_columns: List[str],
                              training_period: str, training_samples: int,
                              hyperparameters: Dict[str, Any], performance_metrics: Dict[str, float],
                              is_classifier: bool) -> ModelMetadata:
        """モデルメタデータを作成"""
        import sklearn
        import sys

        return ModelMetadata(
            model_type=model_type,
            task=task,
            version=f"{model_type.value}_{task.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now(),
            feature_names=feature_names,
            target_columns=target_columns,
            training_period=training_period,
            training_samples=training_samples,
            hyperparameters=hyperparameters,
            preprocessing_info={'scaler': 'StandardScaler', 'encoding': 'LabelEncoder'},
            performance_metrics=performance_metrics,
            is_classifier=is_classifier,
            model_size_mb=0.0,  # 実際のファイルサイズで更新
            python_version=sys.version,
            sklearn_version=sklearn.__version__
        )

    def _save_model_with_metadata(self, model, model_key: str, metadata: ModelMetadata,
                                 label_encoder=None):
        """メタデータ付きでモデルを保存"""
        file_path = self.models_dir / f"{model_key}.joblib"
        metadata_path = self.models_dir / f"{model_key}_metadata.json"

        try:
            # モデル保存
            model_data = {'model': model, 'metadata': metadata}
            if label_encoder:
                model_data['label_encoder'] = label_encoder
            joblib.dump(model_data, file_path)

            # ファイルサイズ更新
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            metadata.model_size_mb = file_size_mb

            # メタデータ保存（JSON形式）
            metadata_dict = {
                'model_type': metadata.model_type.value,
                'task': metadata.task.value,
                'version': metadata.version,
                'created_at': metadata.created_at.isoformat(),
                'feature_names': metadata.feature_names,
                'target_columns': metadata.target_columns,
                'training_period': metadata.training_period,
                'training_samples': metadata.training_samples,
                'hyperparameters': metadata.hyperparameters,
                'preprocessing_info': metadata.preprocessing_info,
                'performance_metrics': metadata.performance_metrics,
                'is_classifier': metadata.is_classifier,
                'model_size_mb': metadata.model_size_mb,
                'python_version': metadata.python_version,
                'sklearn_version': metadata.sklearn_version
            }

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved model with metadata: {model_key}")

        except Exception as e:
            self.logger.error(f"Failed to save model with metadata {model_key}: {e}")

    def _train_model_common(self, model_type: ModelType, task: PredictionTask,
                           X: pd.DataFrame, targets: Dict[PredictionTask, pd.Series],
                           symbol: str, valid_idx: pd.Index, config: TrainingConfig,
                           optimized_params: Optional[Dict[str, Any]] = None) -> ModelPerformance:
        """共通のモデル訓練ロジック"""

        start_time = datetime.now()
        y = targets[task].loc[valid_idx]
        X_clean = X.loc[y.index]

        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, test_size=config.test_size,
            random_state=config.random_state, stratify=y if task == PredictionTask.PRICE_DIRECTION else None
        )

        # モデル初期化
        is_classifier = task == PredictionTask.PRICE_DIRECTION
        model = self._create_model_instance(model_type, is_classifier, optimized_params)

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
        performance_metrics = self._calculate_performance_metrics(
            y_test_encoded, y_pred, is_classifier, model, X_test
        )

        # クロスバリデーション
        if config.enable_cross_validation:
            cv_scores = self._perform_cross_validation(
                model, X_clean, y_train_encoded if is_classifier else y,
                config.cv_folds, is_classifier
            )
            performance_metrics.update({
                'cross_val_mean': np.mean(cv_scores),
                'cross_val_std': np.std(cv_scores)
            })

        # モデル保存
        if config.save_model:
            model_key = f"{symbol}_{model_type.value}_{task.value}"

            # メタデータ作成
            metadata = self._create_model_metadata(
                model_type, task, list(X_clean.columns), [task.value],
                "training_period", len(X_clean),
                optimized_params or {}, performance_metrics, is_classifier
            )

            # 保存
            self._save_model_with_metadata(model, model_key, metadata, label_encoder)

            # メモリに保存
            self.trained_models[model_key] = model
            if label_encoder:
                self.label_encoders[model_key] = label_encoder

        # 特徴量重要度
        feature_importance = self._get_feature_importance(model, X_clean.columns)

        training_time = (datetime.now() - start_time).total_seconds()

        return ModelPerformance(
            model_name=f"{model_type.value}_{task.value}",
            task=task,
            accuracy=performance_metrics.get('accuracy', 0.0),
            precision=performance_metrics.get('precision', 0.0),
            recall=performance_metrics.get('recall', 0.0),
            f1_score=performance_metrics.get('f1_score', 0.0),
            cross_val_mean=performance_metrics.get('cross_val_mean', 0.0),
            cross_val_std=performance_metrics.get('cross_val_std', 0.0),
            feature_importance=feature_importance,
            confusion_matrix=performance_metrics.get('confusion_matrix', np.array([])),
            training_time=training_time,
            prediction_time=0.0
        )

    def _create_model_instance(self, model_type: ModelType, is_classifier: bool,
                              optimized_params: Optional[Dict[str, Any]] = None):
        """モデルインスタンスを作成"""
        params = optimized_params or {}

        if model_type == ModelType.RANDOM_FOREST:
            base_params = self.model_configs[ModelType.RANDOM_FOREST]
            base_params = base_params['classifier_params' if is_classifier else 'regressor_params']
            final_params = {**base_params, **params}

            if is_classifier:
                return RandomForestClassifier(**final_params)
            else:
                return RandomForestRegressor(**final_params)

        elif model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            base_params = self.model_configs[ModelType.XGBOOST]
            base_params = base_params['classifier_params' if is_classifier else 'regressor_params']
            final_params = {**base_params, **params}

            if is_classifier:
                return xgb.XGBClassifier(**final_params)
            else:
                return xgb.XGBRegressor(**final_params)

        elif model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
            base_params = self.model_configs[ModelType.LIGHTGBM]
            base_params = base_params['classifier_params' if is_classifier else 'regressor_params']
            final_params = {**base_params, **params}

            if is_classifier:
                return lgb.LGBMClassifier(**final_params)
            else:
                return lgb.LGBMRegressor(**final_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _calculate_performance_metrics(self, y_true, y_pred, is_classifier: bool,
                                     model, X_test) -> Dict[str, Any]:
        """性能指標を計算"""
        metrics = {}

        if is_classifier:
            from sklearn.metrics import precision_score, recall_score, f1_score
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        else:
            metrics['r2_score'] = r2_score(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))

        return metrics

    def _perform_cross_validation(self, model, X, y, cv_folds: int, is_classifier: bool) -> np.ndarray:
        """クロスバリデーション実行"""
        scoring = 'accuracy' if is_classifier else 'r2'
        cv_strategy = TimeSeriesSplit(n_splits=cv_folds)
        return cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring)

    def _get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        """特徴量重要度を取得"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                return dict(zip(feature_names, importance_scores))
            else:
                return {}
        except Exception as e:
            self.logger.warning(f"Failed to get feature importance: {e}")
            return {}

    def _estimate_regression_confidence(self, model, features: pd.DataFrame,
                                      prediction: float, task: PredictionTask) -> float:
        """回帰モデルの信頼度推定"""
        try:
            # 方法1: 予測値の安定性チェック（特徴量を少し変動させて予測の一貫性を確認）
            if len(features) > 0:
                feature_variations = []
                base_features = features.copy()

                # 特徴量を少し変動させた予測を複数回実行
                for i in range(5):
                    varied_features = base_features.copy()
                    # 各特徴量に小さなノイズを追加（1%以下）
                    noise_factor = 0.01 * (i + 1)
                    for col in varied_features.columns:
                        if varied_features[col].dtype in ['float64', 'int64']:
                            varied_features[col] *= (1 + np.random.normal(0, noise_factor))

                    try:
                        varied_pred = model.predict(varied_features)[0]
                        feature_variations.append(varied_pred)
                    except:
                        continue

                if feature_variations:
                    # 予測値の分散を信頼度に変換
                    pred_std = np.std(feature_variations)
                    pred_mean = np.mean(feature_variations)

                    # 相対標準偏差を信頼度に変換（低い分散 = 高い信頼度）
                    if pred_mean != 0:
                        cv = abs(pred_std / pred_mean)  # 変動係数
                        confidence = max(0.1, min(0.95, 1.0 - cv))
                    else:
                        confidence = 0.5
                else:
                    confidence = 0.5

            else:
                confidence = 0.5

            # 方法2: モデルタイプに基づく基準信頼度調整
            if hasattr(model, 'n_estimators'):  # Random Forest, XGBoost
                # アンサンブルモデルは一般的に高い信頼度
                confidence *= 1.1

            # 信頼度を0.1-0.95の範囲に制限
            confidence = max(0.1, min(0.95, confidence))

            return float(confidence)

        except Exception as e:
            self.logger.warning(f"Failed to estimate regression confidence: {e}")
            return 0.6  # デフォルト値

    async def prepare_training_data(self, symbol: str, period: str = "1y") -> Tuple[pd.DataFrame, Dict[PredictionTask, pd.Series]]:
        """訓練データの準備"""

        self.logger.info(f"Preparing training data for {symbol}")

        # 履歴データ取得
        if REAL_DATA_PROVIDER_AVAILABLE:
            data = await real_data_provider.get_stock_data(symbol, period)
        else:
            # ダミーデータ
            dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
            np.random.seed(42)
            prices = [1000]
            for _ in range(len(dates)-1):
                change = np.random.normal(0, 0.02)
                prices.append(prices[-1] * (1 + change))

            data = pd.DataFrame({
                'Open': [p * 0.99 for p in prices],
                'High': [p * 1.02 for p in prices],
                'Low': [p * 0.98 for p in prices],
                'Close': prices,
                'Volume': np.random.randint(10000, 100000, len(dates))
            }, index=dates)

        if data.empty:
            raise ValueError(f"No data available for {symbol}")

        # 特徴量エンジニアリング
        if FEATURE_ENGINEERING_AVAILABLE:
            try:
                feature_set = await enhanced_feature_engineer.extract_comprehensive_features(symbol, data)
                # FeatureSetをDataFrameに変換
                if hasattr(feature_set, 'to_dataframe'):
                    features = feature_set.to_dataframe()
                else:
                    # フォールバック：FeatureSetから手動でDataFrame作成
                    features = self._convert_featureset_to_dataframe(feature_set, data)

                # 特徴量が不十分な場合は基本特徴量にフォールバック
                if features.empty or len(features) < 10:
                    self.logger.warning("Feature engineering returned insufficient data, falling back to basic features")
                    features = self._extract_basic_features(data)

            except Exception as e:
                self.logger.error(f"Feature engineering failed: {e}")
                features = self._extract_basic_features(data)
        else:
            # 基本的な特徴量
            features = self._extract_basic_features(data)

        # ターゲット変数作成
        targets = self._create_target_variables(data)

        self.logger.info(f"Features shape: {features.shape}, Data shape: {data.shape}")
        self.logger.info(f"Feature columns: {list(features.columns)[:10]}")  # 最初の10列

        for task, target in targets.items():
            valid_targets = target.dropna()
            self.logger.info(f"Target {task.value}: {len(valid_targets)} valid samples")

        return features, targets

    def _extract_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """基本特徴量抽出（フォールバック）"""

        features = pd.DataFrame(index=data.index)

        # 価格系特徴量
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features['volatility'] = features['returns'].rolling(20).std()
        features['price_range'] = (data['High'] - data['Low']) / data['Close']

        # 移動平均
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = data['Close'].rolling(window).mean()
            features[f'sma_ratio_{window}'] = data['Close'] / features[f'sma_{window}']

        # 出来高特徴量
        if 'Volume' in data.columns:
            features['volume_ma'] = data['Volume'].rolling(20).mean()
            features['volume_ratio'] = data['Volume'] / features['volume_ma']
        else:
            features['volume_ratio'] = 1.0

        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        features['rsi'] = 100 - (100 / (1 + rs))

        # 欠損値処理
        features = features.fillna(method='ffill').fillna(0)

        return features

    def _convert_featureset_to_dataframe(self, feature_set, data: pd.DataFrame) -> pd.DataFrame:
        """FeatureSetをDataFrameに変換"""

        try:
            if isinstance(feature_set, list):
                # 複数のFeatureSetのリスト（時系列）の場合
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
                                row_features.update(features_dict)

                    if row_features:
                        feature_rows.append(row_features)
                        timestamps.append(fs.timestamp)

                if feature_rows:
                    features_df = pd.DataFrame(feature_rows, index=timestamps)
                else:
                    features_df = self._extract_basic_features(data)

            else:
                # 単一のFeatureSetの場合
                all_features = {}
                for category in ['price_features', 'technical_features', 'volume_features',
                               'momentum_features', 'volatility_features', 'pattern_features',
                               'market_features', 'statistical_features']:
                    if hasattr(feature_set, category):
                        features_dict = getattr(feature_set, category)
                        if isinstance(features_dict, dict):
                            all_features.update(features_dict)

                if all_features:
                    features_df = pd.DataFrame([all_features], index=[feature_set.timestamp])
                else:
                    features_df = self._extract_basic_features(data)

            # 欠損値処理
            features_df = features_df.fillna(method='ffill').fillna(0)

            return features_df

        except Exception as e:
            self.logger.error(f"Failed to convert FeatureSet to DataFrame: {e}")
            # フォールバック：基本特徴量を使用
            return self._extract_basic_features(data)

    def _create_target_variables(self, data: pd.DataFrame) -> Dict[PredictionTask, pd.Series]:
        """ターゲット変数作成"""

        targets = {}

        # 価格方向予測（翌日の価格変動方向）
        returns = data['Close'].pct_change().shift(-1)  # 翌日のリターン

        # 3クラス分類：上昇(1)、横ばい(0)、下落(-1)
        direction = pd.Series(index=data.index, dtype='int')
        direction[returns > 0.01] = 1   # 1%以上上昇
        direction[returns < -0.01] = -1 # 1%以上下落
        direction[(returns >= -0.01) & (returns <= 0.01)] = 0  # 横ばい

        targets[PredictionTask.PRICE_DIRECTION] = direction

        # 価格回帰予測（翌日の終値）
        targets[PredictionTask.PRICE_REGRESSION] = data['Close'].shift(-1)

        # ボラティリティ予測（翌日の変動率）
        high_low_range = (data['High'] - data['Low']) / data['Close']
        targets[PredictionTask.VOLATILITY] = high_low_range.shift(-1)

        return targets

    async def train_models(self, symbol: str, period: str = "1y", optimized_params: Optional[Dict[str, Any]] = None) -> Dict[ModelType, Dict[PredictionTask, ModelPerformance]]:
        """モデル訓練"""

        self.logger.info(f"Training models for {symbol}")

        # 訓練データ準備
        features, targets = await self.prepare_training_data(symbol, period)

        # 欠損値除去（最後の行は未来の値が不明）
        valid_idx = features.index[:-1]
        X = features.loc[valid_idx]

        if optimized_params is None:
            optimized_params = {}
        performances = {}

        # Random Forest
        if ModelType.RANDOM_FOREST not in performances:
            performances[ModelType.RANDOM_FOREST] = {}

        rf_perf = await self._train_random_forest(X, targets, symbol, valid_idx, optimized_params)
        performances[ModelType.RANDOM_FOREST] = rf_perf

        # XGBoost
        if XGBOOST_AVAILABLE:
            if ModelType.XGBOOST not in performances:
                performances[ModelType.XGBOOST] = {}

            xgb_perf = await self._train_xgboost(X, targets, symbol, valid_idx, optimized_params)
            performances[ModelType.XGBOOST] = xgb_perf

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            if ModelType.LIGHTGBM not in performances:
                performances[ModelType.LIGHTGBM] = {}

            lgb_perf = await self._train_lightgbm(X, targets, symbol, valid_idx, optimized_params)
            performances[ModelType.LIGHTGBM] = lgb_perf

        # 性能結果保存
        await self._save_model_performances(performances, symbol)

        # アンサンブル重み計算
        self._calculate_ensemble_weights(performances, symbol)

        return performances

    async def _train_random_forest(self, X: pd.DataFrame, targets: Dict[PredictionTask, pd.Series],
                                 symbol: str, valid_idx: pd.Index, optimized_params: Optional[Dict[str, Any]] = None) -> Dict[PredictionTask, ModelPerformance]:
        """Random Forest訓練"""

        performances = {}

        for task, target_series in targets.items():
            if task not in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
                continue

            self.logger.info(f"Training Random Forest for {task.value}")

            start_time = datetime.now()

            # ターゲット準備
            y = target_series.loc[valid_idx].dropna()
            X_clean = X.loc[y.index]

            self.logger.info(f"Sample count for {task.value}: {len(y)}")

            if len(y) < 20:  # 最小サンプル数チェック（現実的な値に調整）
                self.logger.warning(f"Insufficient samples for {task.value}: {len(y)} < 20")
                continue

            # 訓練・テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y, test_size=0.3, random_state=42,
                stratify=y if task == PredictionTask.PRICE_DIRECTION else None
            )

            try:
                if task == PredictionTask.PRICE_DIRECTION:
                    # 分類モデル
                    model_params = self.model_configs[ModelType.RANDOM_FOREST]['classifier_params']
                    if optimized_params and ModelType.RANDOM_FOREST.value in optimized_params:
                        model_params.update(optimized_params[ModelType.RANDOM_FOREST.value].get(task.value, {})) # .get(task.value, {})を追加
                    model = RandomForestClassifier(**model_params)

                    # ラベルエンコーダー
                    le = LabelEncoder()
                    y_train_encoded = le.fit_transform(y_train)
                    y_test_encoded = le.transform(y_test)

                    model.fit(X_train, y_train_encoded)
                    y_pred = model.predict(X_test)

                    # 性能計算
                    accuracy = accuracy_score(y_test_encoded, y_pred)
                    report = classification_report(y_test_encoded, y_pred, output_dict=True)
                    cm = confusion_matrix(y_test_encoded, y_pred)

                    # クロスバリデーション
                    cv_scores = cross_val_score(model, X_train, y_train_encoded,
                                              cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')

                    # 特徴量重要度
                    feature_importance = dict(zip(X_train.columns, model.feature_importances_))

                    # モデル保存
                    model_key = f"{symbol}_{task.value}_{ModelType.RANDOM_FOREST.value}"
                    self.trained_models[model_key] = model
                    self.label_encoders[model_key] = le
                    self._save_model(model, model_key, le)

                    performance = ModelPerformance(
                        model_name=f"RandomForest_{task.value}",
                        task=task,
                        accuracy=accuracy,
                        precision=report['weighted avg']['precision'],
                        recall=report['weighted avg']['recall'],
                        f1_score=report['weighted avg']['f1-score'],
                        cross_val_mean=cv_scores.mean(),
                        cross_val_std=cv_scores.std(),
                        feature_importance=feature_importance,
                        confusion_matrix=cm,
                        training_time=(datetime.now() - start_time).total_seconds(),
                        prediction_time=0.0
                    )

                    performances[task] = performance

                else:  # PredictionTask.PRICE_REGRESSION
                    # 回帰モデル
                    model_params = self.model_configs[ModelType.RANDOM_FOREST]['regressor_params']
                    if optimized_params and ModelType.RANDOM_FOREST.value in optimized_params:
                        model_params.update(optimized_params[ModelType.RANDOM_FOREST.value].get(task.value, {}))
                    model = RandomForestRegressor(**model_params)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # 性能計算
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)

                    # クロスバリデーション
                    cv_scores = cross_val_score(model, X_train, y_train,
                                              cv=TimeSeriesSplit(n_splits=5), scoring='r2')

                    # 特徴量重要度
                    feature_importance = dict(zip(X_train.columns, model.feature_importances_))

                    # モデル保存
                    model_key = f"{symbol}_{task.value}_{ModelType.RANDOM_FOREST.value}"
                    self.trained_models[model_key] = model
                    self._save_model(model, model_key)

                    performance = ModelPerformance(
                        model_name=f"RandomForest_{task.value}",
                        task=task,
                        accuracy=r2,  # 回帰ではR²を精度として使用
                        precision=1.0 - (rmse / y_test.std()),  # 正規化RMSE
                        recall=0.0,
                        f1_score=0.0,
                        cross_val_mean=cv_scores.mean(),
                        cross_val_std=cv_scores.std(),
                        feature_importance=feature_importance,
                        confusion_matrix=np.array([]),
                        training_time=(datetime.now() - start_time).total_seconds(),
                        prediction_time=0.0
                    )

                    performances[task] = performance

            except Exception as e:
                self.logger.error(f"Random Forest training failed for {task.value}: {e}")

        return performances

    async def _train_xgboost(self, X: pd.DataFrame, targets: Dict[PredictionTask, pd.Series],
                           symbol: str, valid_idx: pd.Index, optimized_params: Optional[Dict[str, Any]] = None) -> Dict[PredictionTask, ModelPerformance]:
        """XGBoost訓練"""

        performances = {}

        for task, target_series in targets.items():
            if task not in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
                continue

            self.logger.info(f"Training XGBoost for {task.value}")

            start_time = datetime.now()

            # ターゲット準備
            y = target_series.loc[valid_idx].dropna()
            X_clean = X.loc[y.index]

            self.logger.info(f"XGBoost sample count for {task.value}: {len(y)}")

            if len(y) < 20:
                self.logger.warning(f"XGBoost insufficient samples for {task.value}: {len(y)} < 20")
                continue

            # 訓練・テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y, test_size=0.3, random_state=42,
                stratify=y if task == PredictionTask.PRICE_DIRECTION else None
            )

            try:
                if task == PredictionTask.PRICE_DIRECTION:
                    # 分類モデル
                    # XGBoostは-1, 0, 1を0, 1, 2にマッピング
                    le = LabelEncoder()
                    y_train_encoded = le.fit_transform(y_train)
                    y_test_encoded = le.transform(y_test)

                    model_params = self.model_configs[ModelType.XGBOOST]['classifier_params']
                    if optimized_params and ModelType.XGBOOST.value in optimized_params:
                        model_params.update(optimized_params[ModelType.XGBOOST.value].get(task.value, {}))
                    model = xgb.XGBClassifier(**model_params)

                    model.fit(X_train, y_train_encoded)
                    y_pred = model.predict(X_test)

                    # 性能計算
                    accuracy = accuracy_score(y_test_encoded, y_pred)
                    report = classification_report(y_test_encoded, y_pred, output_dict=True)
                    cm = confusion_matrix(y_test_encoded, y_pred)

                    # クロスバリデーション
                    cv_scores = cross_val_score(model, X_train, y_train_encoded,
                                              cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')

                    # 特徴量重要度
                    feature_importance = dict(zip(X_train.columns, model.feature_importances_))

                    # モデル保存
                    model_key = f"{symbol}_{task.value}_{ModelType.XGBOOST.value}"
                    self.trained_models[model_key] = model
                    self.label_encoders[model_key] = le
                    self._save_model(model, model_key, le)

                    performance = ModelPerformance(
                        model_name=f"XGBoost_{task.value}",
                        task=task,
                        accuracy=accuracy,
                        precision=report['weighted avg']['precision'],
                        recall=report['weighted avg']['recall'],
                        f1_score=report['weighted avg']['f1-score'],
                        cross_val_mean=cv_scores.mean(),
                        cross_val_std=cv_scores.std(),
                        feature_importance=feature_importance,
                        confusion_matrix=cm,
                        training_time=(datetime.now() - start_time).total_seconds(),
                        prediction_time=0.0
                    )

                    performances[task] = performance

                else:  # PredictionTask.PRICE_REGRESSION
                    # 回帰モデル
                    model_params = self.model_configs[ModelType.XGBOOST]['regressor_params']
                    if optimized_params and ModelType.XGBOOST.value in optimized_params:
                        model_params.update(optimized_params[ModelType.XGBOOST.value].get(task.value, {}))
                    model = xgb.XGBRegressor(**model_params)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # 性能計算
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)

                    # クロスバリデーション
                    cv_scores = cross_val_score(model, X_train, y_train,
                                              cv=TimeSeriesSplit(n_splits=5), scoring='r2')

                    # 特徴量重要度
                    feature_importance = dict(zip(X_train.columns, model.feature_importances_))

                    # モデル保存
                    model_key = f"{symbol}_{task.value}_{ModelType.XGBOOST.value}"
                    self.trained_models[model_key] = model
                    self._save_model(model, model_key)

                    performance = ModelPerformance(
                        model_name=f"XGBoost_{task.value}",
                        task=task,
                        accuracy=r2,
                        precision=1.0 - (rmse / y_test.std()),
                        recall=0.0,
                        f1_score=0.0,
                        cross_val_mean=cv_scores.mean(),
                        cross_val_std=cv_scores.std(),
                        feature_importance=feature_importance,
                        confusion_matrix=np.array([]),
                        training_time=(datetime.now() - start_time).total_seconds(),
                        prediction_time=0.0
                    )

                    performances[task] = performance

            except Exception as e:
                self.logger.error(f"XGBoost training failed for {task.value}: {e}")

    async def _train_lightgbm(self, X: pd.DataFrame, targets: Dict[PredictionTask, pd.Series],
                            symbol: str, valid_idx: pd.Index, optimized_params: Optional[Dict[str, Any]] = None) -> Dict[PredictionTask, ModelPerformance]:
        """LightGBM訓練"""

        performances = {}

        for task, target_series in targets.items():
            if task not in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
                continue

            self.logger.info(f"Training LightGBM for {task.value}")

            start_time = datetime.now()

            # ターゲット準備
            y = target_series.loc[valid_idx].dropna()
            X_clean = X.loc[y.index]

            if len(y) < 50:
                continue

            # 訓練・テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y, test_size=0.3, random_state=42,
                stratify=y if task == PredictionTask.PRICE_DIRECTION else None
            )

            try:
                if task == PredictionTask.PRICE_DIRECTION:
                    # 分類モデル
                    le = LabelEncoder()
                    y_train_encoded = le.fit_transform(y_train)
                    y_test_encoded = le.transform(y_test)

                    model_params = {
                        'n_estimators': 200,
                        'max_depth': 10,
                        'learning_rate': 0.1,
                        'random_state': 42,
                        'n_jobs': -1,
                        'verbose': -1
                    }
                    if optimized_params and ModelType.LIGHTGBM.value in optimized_params:
                        model_params.update(optimized_params[ModelType.LIGHTGBM.value].get(task.value, {}))
                    model = lgb.LGBMClassifier(**model_params)

                    model.fit(X_train, y_train_encoded)
                    y_pred = model.predict(X_test)

                    # 性能計算
                    accuracy = accuracy_score(y_test_encoded, y_pred)
                    report = classification_report(y_test_encoded, y_pred, output_dict=True)
                    cm = confusion_matrix(y_test_encoded, y_pred)

                    # クロスバリデーション
                    cv_scores = cross_val_score(model, X_train, y_train_encoded,
                                              cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')

                    # 特徴量重要度
                    feature_importance = dict(zip(X_train.columns, model.feature_importances_))

                    # モデル保存
                    model_key = f"{symbol}_{task.value}_{ModelType.LIGHTGBM.value}"
                    self.trained_models[model_key] = model
                    self.label_encoders[model_key] = le
                    self._save_model(model, model_key, le)

                    performance = ModelPerformance(
                        model_name=f"LightGBM_{task.value}",
                        task=task,
                        accuracy=accuracy,
                        precision=report['weighted avg']['precision'],
                        recall=report['weighted avg']['recall'],
                        f1_score=report['weighted avg']['f1-score'],
                        cross_val_mean=cv_scores.mean(),
                        cross_val_std=cv_scores.std(),
                        feature_importance=feature_importance,
                        confusion_matrix=cm,
                        training_time=(datetime.now() - start_time).total_seconds(),
                        prediction_time=0.0
                    )

                    performances[task] = performance

            except Exception as e:
                self.logger.error(f"LightGBM training failed for {task.value}: {e}")

        return performances

    def _calculate_ensemble_weights(self, performances: Dict[ModelType, Dict[PredictionTask, ModelPerformance]],
                                  symbol: str):
        """アンサンブル重み計算"""

        if symbol not in self.ensemble_weights:
            self.ensemble_weights[symbol] = {}

        for task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
            task_performances = []
            model_names = []

            for model_type, task_perfs in performances.items():
                if task in task_perfs:
                    task_performances.append(task_perfs[task].accuracy)
                    model_names.append(model_type)

            if task_performances:
                # 精度に基づく重み計算（ソフトマックス）
                accuracies = np.array(task_performances)
                exp_accuracies = np.exp(accuracies * 10)  # スケーリング
                weights = exp_accuracies / exp_accuracies.sum()

                self.ensemble_weights[symbol][task] = dict(zip(model_names, weights))

    async def predict(self, symbol: str, features: pd.DataFrame) -> Dict[PredictionTask, EnsemblePrediction]:
        """統一された予測インターフェース"""

        predictions = {}

        for task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
            try:
                ensemble_pred = await self._make_ensemble_prediction(symbol, features, task)
                if ensemble_pred:
                    predictions[task] = ensemble_pred
            except Exception as e:
                self.logger.error(f"Prediction failed for {task.value}: {e}")

        return predictions

    async def predict_list(self, symbol: str, features: pd.DataFrame) -> List[EnsemblePrediction]:
        """リスト形式での予測（後方互換性のため）"""
        predictions_dict = await self.predict(symbol, features)
        return list(predictions_dict.values())

    async def _make_ensemble_prediction(self, symbol: str, features: pd.DataFrame,
                                      task: PredictionTask) -> Optional[EnsemblePrediction]:
        """アンサンブル予測"""

        model_predictions = {}
        model_confidences = {}

        # 各モデルで予測
        for model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM]:
            model_key = f"{symbol}_{task.value}_{model_type.value}"

            if model_key not in self.trained_models:
                continue

            try:
                model = self.trained_models[model_key]

                if task == PredictionTask.PRICE_DIRECTION:
                    # 分類予測
                    pred_proba = model.predict_proba(features)
                    pred_class = model.predict(features)[0]

                    # ラベルエンコーダーで元のラベルに変換
                    if model_key in self.label_encoders:
                        le = self.label_encoders[model_key]
                        pred_class = le.inverse_transform([pred_class])[0]

                    confidence = np.max(pred_proba[0])
                    model_predictions[model_type.value] = pred_class
                    model_confidences[model_type.value] = confidence

                else:  # PredictionTask.PRICE_REGRESSION
                    # 回帰予測
                    pred_value = model.predict(features)[0]

                    # 回帰信頼度推定（予測区間ベース）
                    confidence = self._estimate_regression_confidence(
                        model, features, pred_value, task
                    )

                    model_predictions[model_type.value] = pred_value
                    model_confidences[model_type.value] = confidence

            except Exception as e:
                self.logger.error(f"Prediction failed for {model_key}: {e}")

        if not model_predictions:
            return None

        # アンサンブル統合
        weights = self.ensemble_weights.get(symbol, {}).get(task, {})

        if task == PredictionTask.PRICE_DIRECTION:
            # 多数決 + 重み付け
            weighted_votes = {}
            total_weight = 0

            for model_name, prediction in model_predictions.items():
                model_type = ModelType(model_name)
                weight = weights.get(model_type, 1.0)
                confidence = model_confidences[model_name]

                if prediction not in weighted_votes:
                    weighted_votes[prediction] = 0
                weighted_votes[prediction] += weight * confidence
                total_weight += weight

            # 最も重み付きスコアが高い予測を選択
            final_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]
            confidence = weighted_votes[final_prediction] / total_weight if total_weight > 0 else 0.5

            # 合意強度（予測の一致度）
            unique_predictions = len(set(model_predictions.values()))
            consensus_strength = 1.0 - (unique_predictions - 1) / max(1, len(model_predictions) - 1)

        else:  # PredictionTask.PRICE_REGRESSION
            # 重み付き平均
            weighted_sum = 0
            total_weight = 0

            for model_name, prediction in model_predictions.items():
                model_type = ModelType(model_name)
                weight = weights.get(model_type, 1.0)

                weighted_sum += prediction * weight
                total_weight += weight

            final_prediction = weighted_sum / total_weight if total_weight > 0 else 0
            confidence = np.mean(list(model_confidences.values()))
            consensus_strength = 1.0 - np.std(list(model_predictions.values())) / np.mean(list(model_predictions.values()))

        # 意見不一致スコア
        prediction_values = list(model_predictions.values())
        if len(set(str(p) for p in prediction_values)) == 1:
            disagreement_score = 0.0
        else:
            disagreement_score = 1.0 - consensus_strength

        ensemble_prediction = EnsemblePrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            final_prediction=final_prediction,
            confidence=confidence,
            model_predictions=model_predictions,
            model_weights=weights,
            consensus_strength=consensus_strength,
            disagreement_score=disagreement_score
        )

        # 予測結果保存
        await self._save_prediction_result(ensemble_prediction, task)

        return ensemble_prediction

    async def _save_model_performances(self, performances: Dict[ModelType, Dict[PredictionTask, ModelPerformance]],
                                     symbol: str):
        """モデル性能保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                for model_type, task_perfs in performances.items():
                    for task, perf in task_perfs.items():
                        conn.execute("""
                            INSERT INTO model_performances
                            (model_name, task, accuracy, precision_score, recall_score, f1_score,
                             cross_val_mean, cross_val_std, training_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            perf.model_name,
                            task.value,
                            perf.accuracy,
                            perf.precision,
                            perf.recall,
                            perf.f1_score,
                            perf.cross_val_mean,
                            perf.cross_val_std,
                            perf.training_time
                        ))

        except Exception as e:
            self.logger.error(f"Failed to save model performances: {e}")

    async def _save_prediction_result(self, prediction: EnsemblePrediction, task: PredictionTask):
        """予測結果保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO ensemble_predictions
                    (symbol, timestamp, final_prediction, confidence, consensus_strength, disagreement_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    prediction.symbol,
                    prediction.timestamp.isoformat(),
                    str(prediction.final_prediction),
                    prediction.confidence,
                    prediction.consensus_strength,
                    prediction.disagreement_score
                ))

        except Exception as e:
            self.logger.error(f"Failed to save prediction result: {e}")

    async def get_model_summary(self) -> Dict[str, Any]:
        """モデルサマリー取得"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                # 最新のモデル性能
                cursor = conn.execute("""
                    SELECT model_name, task, accuracy, cross_val_mean, training_time
                    FROM model_performances
                    ORDER BY training_date DESC
                    LIMIT 20
                """)

                performances = cursor.fetchall()

                # 最新の予測結果
                cursor = conn.execute("""
                    SELECT symbol, final_prediction, confidence, consensus_strength
                    FROM ensemble_predictions
                    ORDER BY created_at DESC
                    LIMIT 10
                """)

                predictions = cursor.fetchall()

                return {
                    'trained_models_count': len(self.trained_models),
                    'recent_performances': [
                        {
                            'model': p[0],
                            'task': p[1],
                            'accuracy': p[2],
                            'cv_score': p[3],
                            'training_time': p[4]
                        } for p in performances
                    ],
                    'recent_predictions': [
                        {
                            'symbol': p[0],
                            'prediction': p[1],
                            'confidence': p[2],
                            'consensus': p[3]
                        } for p in predictions
                    ]
                }

        except Exception as e:
            self.logger.error(f"Failed to get model summary: {e}")
            return {
                'trained_models_count': 0,
                'recent_performances': [],
                'recent_predictions': []
            }

# ファクトリー関数
def create_ml_prediction_models() -> MLPredictionModels:
    """
    MLPredictionModelsインスタンスの作成

    Returns:
        MLPredictionModelsインスタンス
    """
    return MLPredictionModels()

# グローバルインスタンス（後方互換性のため）
try:
    ml_prediction_models = MLPredictionModels()
except Exception:
    # 初期化に失敗した場合のフォールバック
    ml_prediction_models = None

if __name__ == "__main__":
    # 基本動作確認は別ファイルに分離済み
    pass