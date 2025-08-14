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

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

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

        # データベース初期化
        self.db_path = self.data_dir / "ml_predictions.db"
        self._init_database()

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

        self.logger.info("ML prediction models initialized")

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

    async def prepare_training_data(self, symbol: str, period: str = "1y") -> Tuple[pd.DataFrame, pd.Series]:
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

    def _create_target_variables(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
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

    async def train_models(self, symbol: str, period: str = "1y") -> Dict[ModelType, Dict[PredictionTask, ModelPerformance]]:
        """モデル訓練"""

        self.logger.info(f"Training models for {symbol}")

        # 訓練データ準備
        features, targets = await self.prepare_training_data(symbol, period)

        # 欠損値除去（最後の行は未来の値が不明）
        valid_idx = features.index[:-1]
        X = features.loc[valid_idx]

        performances = {}

        # Random Forest
        if ModelType.RANDOM_FOREST not in performances:
            performances[ModelType.RANDOM_FOREST] = {}

        rf_perf = await self._train_random_forest(X, targets, symbol, valid_idx)
        performances[ModelType.RANDOM_FOREST] = rf_perf

        # XGBoost
        if XGBOOST_AVAILABLE:
            if ModelType.XGBOOST not in performances:
                performances[ModelType.XGBOOST] = {}

            xgb_perf = await self._train_xgboost(X, targets, symbol, valid_idx)
            performances[ModelType.XGBOOST] = xgb_perf

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            if ModelType.LIGHTGBM not in performances:
                performances[ModelType.LIGHTGBM] = {}

            lgb_perf = await self._train_lightgbm(X, targets, symbol, valid_idx)
            performances[ModelType.LIGHTGBM] = lgb_perf

        # 性能結果保存
        await self._save_model_performances(performances, symbol)

        # アンサンブル重み計算
        self._calculate_ensemble_weights(performances, symbol)

        return performances

    async def _train_random_forest(self, X: pd.DataFrame, targets: Dict[PredictionTask, pd.Series],
                                 symbol: str, valid_idx: pd.Index) -> Dict[PredictionTask, ModelPerformance]:
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
                    model = RandomForestClassifier(**self.model_configs[ModelType.RANDOM_FOREST]['classifier_params'])

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
                    model = RandomForestRegressor(**self.model_configs[ModelType.RANDOM_FOREST]['regressor_params'])

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
                           symbol: str, valid_idx: pd.Index) -> Dict[PredictionTask, ModelPerformance]:
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

                    model = xgb.XGBClassifier(**self.model_configs[ModelType.XGBOOST]['classifier_params'])

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
                    model = xgb.XGBRegressor(**self.model_configs[ModelType.XGBOOST]['regressor_params'])

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

        return performances

    async def _train_lightgbm(self, X: pd.DataFrame, targets: Dict[PredictionTask, pd.Series],
                            symbol: str, valid_idx: pd.Index) -> Dict[PredictionTask, ModelPerformance]:
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

                    model = lgb.LGBMClassifier(
                        n_estimators=200,
                        max_depth=10,
                        learning_rate=0.1,
                        random_state=42,
                        n_jobs=-1,
                        verbose=-1
                    )

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

    async def predict(self, symbol: str, features: pd.DataFrame) -> List[EnsemblePrediction]:
        """予測実行"""

        predictions = []

        for task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
            try:
                ensemble_pred = await self._make_ensemble_prediction(symbol, features, task)
                if ensemble_pred:
                    predictions.append(ensemble_pred)
            except Exception as e:
                self.logger.error(f"Prediction failed for {task.value}: {e}")

        return predictions

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
                    model_predictions[model_type.value] = pred_value
                    model_confidences[model_type.value] = 0.8  # デフォルト信頼度

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

# グローバルインスタンス
ml_prediction_models = MLPredictionModels()

# テスト関数
async def test_ml_prediction_models():
    """機械学習予測モデルのテスト"""

    print("=== 機械学習予測モデル テスト ===")

    if not SKLEARN_AVAILABLE:
        print("❌ Scikit-learn not available")
        return

    models = MLPredictionModels()

    # テスト銘柄
    test_symbols = ["7203", "8306"]

    print(f"\n[ {len(test_symbols)}銘柄でのモデル訓練 ]")

    for symbol in test_symbols:
        print(f"\n--- {symbol} モデル訓練 ---")

        try:
            # モデル訓練
            performances = await models.train_models(symbol, "6mo")

            print(f"訓練完了: {len(performances)} モデル")

            for model_type, task_perfs in performances.items():
                print(f"\n{model_type.value}:")
                if not task_perfs:
                    print("  訓練データ不足または失敗")
                for task, perf in task_perfs.items():
                    print(f"  {task.value}:")
                    print(f"    精度: {perf.accuracy:.3f}")
                    print(f"    CV平均: {perf.cross_val_mean:.3f}")
                    print(f"    F1スコア: {perf.f1_score:.3f}")
                    print(f"    訓練時間: {perf.training_time:.1f}秒")

                    # 特徴量重要度（上位5つ）
                    top_features = sorted(perf.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    if top_features:
                        print(f"    重要特徴量:")
                        for feat, importance in top_features:
                            print(f"      {feat}: {importance:.3f}")

            # 予測テスト
            print(f"\n[ {symbol} 予測テスト ]")

            # テスト特徴量データ（最新データ）
            features, _ = await models.prepare_training_data(symbol, "1mo")
            latest_features = features.tail(1)

            predictions = await models.predict(symbol, latest_features)

            for pred in predictions:
                print(f"タスク: {pred.final_prediction}")
                print(f"信頼度: {pred.confidence:.3f}")
                print(f"合意強度: {pred.consensus_strength:.3f}")
                print(f"意見不一致: {pred.disagreement_score:.3f}")
                print(f"モデル予測: {pred.model_predictions}")
                print()

        except Exception as e:
            print(f"❌ エラー: {e}")
            import traceback
            traceback.print_exc()

    # システムサマリー
    print(f"\n[ システムサマリー ]")
    summary = await models.get_model_summary()

    print(f"訓練済みモデル数: {summary['trained_models_count']}")
    print(f"最新性能（上位3）:")
    for perf in summary['recent_performances'][:3]:
        print(f"  {perf['model']} ({perf['task']}): {perf['accuracy']:.3f}")

    print(f"\n=== 機械学習予測モデル テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_ml_prediction_models())