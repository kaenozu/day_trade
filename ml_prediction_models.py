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
class ModelMetadata:
    """モデルメタデータ"""
    model_id: str
    model_type: ModelType
    task: PredictionTask
    symbol: str
    version: str
    created_at: datetime
    parameters: Dict[str, Any]
    feature_columns: List[str]
    target_info: Dict[str, Any]
    training_samples: int

@dataclass
class ModelPerformance:
    """モデル性能"""
    model_id: str
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
    validation_metrics: Dict[str, float] = field(default_factory=dict)

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

class BaseModelTrainer:
    """基底モデル訓練クラス"""

    def __init__(self, model_type: ModelType, config: Dict[str, Any]):
        self.model_type = model_type
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{model_type.value}")

    def create_classifier(self, params: Dict[str, Any]):
        """分類器作成（サブクラスで実装）"""
        raise NotImplementedError

    def create_regressor(self, params: Dict[str, Any]):
        """回帰器作成（サブクラスで実装）"""
        raise NotImplementedError

    async def train_model(self, X: pd.DataFrame, y: pd.Series, task: PredictionTask,
                         symbol: str, optimized_params: Optional[Dict[str, Any]] = None) -> Optional[ModelPerformance]:
        """統一されたモデル訓練インターフェース"""

        start_time = datetime.now()

        # データ検証
        if len(y) < 20:
            self.logger.warning(f"Insufficient samples for {task.value}: {len(y)} < 20")
            return None

        # 訓練・テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42,
            stratify=y if task == PredictionTask.PRICE_DIRECTION else None
        )

        try:
            # パラメータ準備
            params = self._get_model_params(task, optimized_params)

            # モデル作成・訓練
            if task == PredictionTask.PRICE_DIRECTION:
                model, le, performance = await self._train_classifier(X_train, X_test, y_train, y_test, params)
            else:
                model, le, performance = await self._train_regressor(X_train, X_test, y_train, y_test, params)

            if model is None:
                return None

            # モデルID生成
            model_id = f"{symbol}_{task.value}_{self.model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 性能情報更新
            performance.model_id = model_id
            performance.training_time = (datetime.now() - start_time).total_seconds()

            # 特徴量重要度
            if hasattr(model, 'feature_importances_'):
                performance.feature_importance = dict(zip(X_train.columns, model.feature_importances_))

            return performance

        except Exception as e:
            self.logger.error(f"Training failed for {task.value}: {e}")
            return None

    def _get_model_params(self, task: PredictionTask, optimized_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """モデルパラメータ取得"""
        param_key = 'classifier_params' if task == PredictionTask.PRICE_DIRECTION else 'regressor_params'
        params = self.config.get(param_key, {}).copy()

        if optimized_params and self.model_type.value in optimized_params:
            params.update(optimized_params[self.model_type.value].get(task.value, {}))

        return params

    async def _train_classifier(self, X_train, X_test, y_train, y_test, params) -> Tuple[Any, Any, ModelPerformance]:
        """分類器訓練"""
        # ラベルエンコーダー
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)

        # モデル作成・訓練
        model = self.create_classifier(params)
        model.fit(X_train, y_train_encoded)
        y_pred = model.predict(X_test)

        # 性能計算
        accuracy = accuracy_score(y_test_encoded, y_pred)
        report = classification_report(y_test_encoded, y_pred, output_dict=True)
        cm = confusion_matrix(y_test_encoded, y_pred)

        # クロスバリデーション
        cv_scores = cross_val_score(model, X_train, y_train_encoded,
                                  cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')

        performance = ModelPerformance(
            model_id="",  # 後で設定
            accuracy=accuracy,
            precision=report['weighted avg']['precision'],
            recall=report['weighted avg']['recall'],
            f1_score=report['weighted avg']['f1-score'],
            cross_val_mean=cv_scores.mean(),
            cross_val_std=cv_scores.std(),
            feature_importance={},  # 後で設定
            confusion_matrix=cm,
            training_time=0.0,  # 後で設定
            prediction_time=0.0
        )

        return model, le, performance

    async def _train_regressor(self, X_train, X_test, y_train, y_test, params) -> Tuple[Any, Any, ModelPerformance]:
        """回帰器訓練"""
        # モデル作成・訓練
        model = self.create_regressor(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 性能計算
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # クロスバリデーション
        cv_scores = cross_val_score(model, X_train, y_train,
                                  cv=TimeSeriesSplit(n_splits=5), scoring='r2')

        performance = ModelPerformance(
            model_id="",  # 後で設定
            accuracy=r2,  # 回帰ではR²を精度として使用
            precision=1.0 - (rmse / y_test.std()) if y_test.std() > 0 else 0.0,
            recall=0.0,
            f1_score=0.0,
            cross_val_mean=cv_scores.mean(),
            cross_val_std=cv_scores.std(),
            feature_importance={},  # 後で設定
            confusion_matrix=np.array([]),
            training_time=0.0,  # 後で設定
            prediction_time=0.0,
            validation_metrics={'mse': mse, 'rmse': rmse, 'r2': r2}
        )

        return model, None, performance

class RandomForestTrainer(BaseModelTrainer):
    """Random Forest訓練クラス"""

    def create_classifier(self, params: Dict[str, Any]):
        return RandomForestClassifier(**params)

    def create_regressor(self, params: Dict[str, Any]):
        return RandomForestRegressor(**params)

class XGBoostTrainer(BaseModelTrainer):
    """XGBoost訓練クラス"""

    def create_classifier(self, params: Dict[str, Any]):
        return xgb.XGBClassifier(**params)

    def create_regressor(self, params: Dict[str, Any]):
        return xgb.XGBRegressor(**params)

class LightGBMTrainer(BaseModelTrainer):
    """LightGBM訓練クラス"""

    def create_classifier(self, params: Dict[str, Any]):
        return lgb.LGBMClassifier(**params)

    def create_regressor(self, params: Dict[str, Any]):
        return lgb.LGBMRegressor(**params)

class ModelMetadataManager:
    """モデルメタデータ管理クラス"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.MetadataManager")
        self._init_metadata_tables()

    def _init_metadata_tables(self):
        """改良されたメタデータテーブル初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # モデルメタデータテーブル（拡張）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    task TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    parameters TEXT,
                    feature_columns TEXT,
                    target_info TEXT,
                    training_samples INTEGER,
                    status TEXT DEFAULT 'active',
                    last_trained TEXT,
                    hyperparameter_hash TEXT,
                    data_fingerprint TEXT
                )
            """)

            # モデル性能履歴テーブル（拡張）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    cross_val_mean REAL,
                    cross_val_std REAL,
                    training_time REAL,
                    validation_metrics TEXT,
                    quality_score REAL,
                    stability_score REAL,
                    feature_importance TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES model_metadata (model_id)
                )
            """)

            # アンサンブル予測履歴テーブル（新規）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ensemble_prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    task TEXT NOT NULL,
                    final_prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    consensus_strength REAL NOT NULL,
                    disagreement_score REAL NOT NULL,
                    model_count INTEGER NOT NULL,
                    avg_model_quality REAL,
                    confidence_variance REAL,
                    prediction_stability REAL,
                    model_predictions TEXT,
                    model_weights TEXT,
                    quality_metrics TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # モデル重み履歴テーブル（新規）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_weight_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    task TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    static_weight REAL,
                    dynamic_weight REAL,
                    quality_contribution REAL,
                    confidence_contribution REAL,
                    performance_trend REAL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 予測精度追跡テーブル（新規）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_accuracy_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    predicted_value TEXT NOT NULL,
                    actual_value TEXT,
                    prediction_date TEXT NOT NULL,
                    verification_date TEXT,
                    accuracy_score REAL,
                    confidence_at_prediction REAL,
                    model_used TEXT,
                    task TEXT NOT NULL
                )
            """)

            # インデックス作成
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_model_metadata_symbol_task ON model_metadata (symbol, task)",
                "CREATE INDEX IF NOT EXISTS idx_performance_history_model_created ON model_performance_history (model_id, created_at)",
                "CREATE INDEX IF NOT EXISTS idx_ensemble_predictions_symbol_timestamp ON ensemble_prediction_history (symbol, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_weight_history_symbol_task ON model_weight_history (symbol, task, updated_at)",
                "CREATE INDEX IF NOT EXISTS idx_accuracy_tracking_symbol_date ON prediction_accuracy_tracking (symbol, prediction_date)"
            ]

            for index_sql in indexes:
                conn.execute(index_sql)

    def save_model_metadata(self, metadata: ModelMetadata) -> bool:
        """モデルメタデータ保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO model_metadata
                    (model_id, model_type, task, symbol, version, created_at,
                     parameters, feature_columns, target_info, training_samples)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.model_id,
                    metadata.model_type.value,
                    metadata.task.value,
                    metadata.symbol,
                    metadata.version,
                    metadata.created_at.isoformat(),
                    json.dumps(metadata.parameters),
                    json.dumps(metadata.feature_columns),
                    json.dumps(metadata.target_info),
                    metadata.training_samples
                ))
                return True
        except Exception as e:
            self.logger.error(f"Failed to save model metadata: {e}")
            return False

    def save_model_performance(self, performance: ModelPerformance) -> bool:
        """モデル性能保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO model_performance_history
                    (model_id, accuracy, precision_score, recall_score, f1_score,
                     cross_val_mean, cross_val_std, training_time, validation_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    performance.model_id,
                    performance.accuracy,
                    performance.precision,
                    performance.recall,
                    performance.f1_score,
                    performance.cross_val_mean,
                    performance.cross_val_std,
                    performance.training_time,
                    json.dumps(performance.validation_metrics)
                ))
                return True
        except Exception as e:
            self.logger.error(f"Failed to save model performance: {e}")
            return False

    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """モデルメタデータ取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT model_id, model_type, task, symbol, version, created_at,
                           parameters, feature_columns, target_info, training_samples
                    FROM model_metadata WHERE model_id = ?
                """, (model_id,))

                row = cursor.fetchone()
                if row:
                    return ModelMetadata(
                        model_id=row[0],
                        model_type=ModelType(row[1]),
                        task=PredictionTask(row[2]),
                        symbol=row[3],
                        version=row[4],
                        created_at=datetime.fromisoformat(row[5]),
                        parameters=json.loads(row[6]) if row[6] else {},
                        feature_columns=json.loads(row[7]) if row[7] else [],
                        target_info=json.loads(row[8]) if row[8] else {},
                        training_samples=row[9]
                    )
        except Exception as e:
            self.logger.error(f"Failed to get model metadata: {e}")
        return None

    def list_models(self, symbol: str = None, model_type: ModelType = None,
                   task: PredictionTask = None) -> List[ModelMetadata]:
        """モデル一覧取得"""
        try:
            query = "SELECT model_id FROM model_metadata WHERE status = 'active'"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if model_type:
                query += " AND model_type = ?"
                params.append(model_type.value)
            if task:
                query += " AND task = ?"
                params.append(task.value)

            query += " ORDER BY created_at DESC"

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                model_ids = [row[0] for row in cursor.fetchall()]

                return [self.get_model_metadata(mid) for mid in model_ids if self.get_model_metadata(mid)]
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []

class DataPreparationPipeline:
    """データ準備パイプライン"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DataPipeline")
        self.feature_cache = {}

    async def prepare_training_data(self, symbol: str, period: str = "1y",
                                  force_refresh: bool = False) -> Tuple[pd.DataFrame, Dict[PredictionTask, pd.Series]]:
        """統合された訓練データ準備"""

        self.logger.info(f"Preparing training data for {symbol} ({period})")

        # キャッシュキー
        cache_key = f"{symbol}_{period}"
        if not force_refresh and cache_key in self.feature_cache:
            self.logger.info("Using cached features")
            return self.feature_cache[cache_key]

        try:
            # データ取得
            raw_data = await self._fetch_market_data(symbol, period)

            # データ品質検証
            validated_data = self._validate_and_clean_data(raw_data)

            # 特徴量エンジニアリング
            features = await self._engineer_features(symbol, validated_data)

            # ターゲット変数作成
            targets = self._create_robust_targets(validated_data)

            # 最終検証
            features, targets = self._final_validation(features, targets)

            # キャッシュに保存
            result = (features, targets)
            self.feature_cache[cache_key] = result

            self.logger.info(f"Data preparation completed: Features {features.shape}, Targets: {len(targets)}")
            return result

        except Exception as e:
            self.logger.error(f"Data preparation failed for {symbol}: {e}")
            # フォールバックで基本データを返す
            return await self._fallback_data_preparation(symbol, period)

    async def _fetch_market_data(self, symbol: str, period: str) -> pd.DataFrame:
        """市場データ取得"""
        if REAL_DATA_PROVIDER_AVAILABLE:
            try:
                data = await real_data_provider.get_stock_data(symbol, period)
                if not data.empty:
                    return data
            except Exception as e:
                self.logger.warning(f"Real data provider failed: {e}")

        # フォールバック: ダミーデータ生成
        return self._generate_synthetic_data(symbol, period)

    def _generate_synthetic_data(self, symbol: str, period: str) -> pd.DataFrame:
        """シンセティックデータ生成"""

        # 期間に応じたデータポイント数を計算
        period_days = {
            '1mo': 30, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825
        }
        days = period_days.get(period, 365)

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(hash(symbol) % 2**32)  # シンボル固有のシード

        # よりリアルな価格変動をシミュレート
        base_price = 1000 + (hash(symbol) % 5000)  # シンボル固有の基準価格
        prices = [base_price]

        for i in range(len(dates)-1):
            # トレンド、ボラティリティ、ランダムウォークを組み合わせ
            trend = 0.0001 * np.sin(i / 30)  # 季節性トレンド
            volatility = 0.02 * (1 + 0.5 * np.sin(i / 10))  # 可変ボラティリティ
            random_change = np.random.normal(trend, volatility)

            new_price = prices[-1] * (1 + random_change)
            prices.append(max(new_price, 10))  # 最低価格制限

        # OHLCVデータ作成
        data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.995, 1.005) for p in prices],
            'High': [p * np.random.uniform(1.01, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 0.99) for p in prices],
            'Close': prices,
            'Volume': np.random.lognormal(10, 1, len(dates)).astype(int)
        }, index=dates)

        # 現実的な制約を適用
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))

        return data

    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """データ品質検証とクリーニング"""

        if data.empty:
            raise ValueError("Empty dataset provided")

        # 必要な列の確認
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # データ型の確認と変換
        for col in required_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        if 'Volume' in data.columns:
            data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
        else:
            data['Volume'] = 1000000  # デフォルト出来高

        # 非現実的な値の修正
        data = data[(data['High'] >= data['Low']) &
                   (data['High'] >= data['Open']) &
                   (data['High'] >= data['Close']) &
                   (data['Low'] <= data['Open']) &
                   (data['Low'] <= data['Close'])]

        # 異常値の除去（極端な価格変動）
        for col in ['Open', 'High', 'Low', 'Close']:
            pct_change = data[col].pct_change().abs()
            outliers = pct_change > 0.5  # 50%以上の変動
            if outliers.any():
                self.logger.warning(f"Removing {outliers.sum()} outliers from {col}")
                data = data[~outliers]

        # 欠損値の前方補間
        data = data.fillna(method='ffill').dropna()

        if len(data) < 50:
            raise ValueError(f"Insufficient data after cleaning: {len(data)} rows")

        return data

    async def _engineer_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """特徴量エンジニアリング"""

        try:
            if FEATURE_ENGINEERING_AVAILABLE:
                # 高度な特徴量エンジニアリングを試みる
                feature_set = await enhanced_feature_engineer.extract_comprehensive_features(symbol, data)

                if hasattr(feature_set, 'to_dataframe'):
                    features = feature_set.to_dataframe()
                else:
                    features = self._convert_featureset_to_dataframe(feature_set, data)

                # 特徴量の品質チェック
                if not features.empty and len(features.columns) >= 10:
                    return self._enhance_basic_features(features, data)
                else:
                    self.logger.warning("Enhanced feature engineering insufficient, using basic features")

        except Exception as e:
            self.logger.warning(f"Enhanced feature engineering failed: {e}")

        # 基本特徴量にフォールバック
        return self._extract_comprehensive_basic_features(data)

    def _extract_comprehensive_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """包括的な基本特徴量抽出"""

        features = pd.DataFrame(index=data.index)

        # 価格関連特徴量
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features['price_range'] = (data['High'] - data['Low']) / data['Close']
        features['upper_shadow'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / data['Close']
        features['lower_shadow'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / data['Close']
        features['body_size'] = abs(data['Close'] - data['Open']) / data['Close']

        # 移動平均と偏差
        for window in [5, 10, 20, 50, 100]:
            if len(data) > window:
                ma = data['Close'].rolling(window).mean()
                features[f'sma_{window}'] = ma
                features[f'sma_ratio_{window}'] = data['Close'] / ma
                features[f'sma_distance_{window}'] = (data['Close'] - ma) / ma
                features[f'volatility_{window}'] = features['returns'].rolling(window).std()

        # テクニカル指標
        features = self._add_technical_indicators(features, data)

        # 出来高特徴量
        if 'Volume' in data.columns:
            features['volume_ma'] = data['Volume'].rolling(20).mean()
            features['volume_ratio'] = data['Volume'] / features['volume_ma']
            features['volume_price_trend'] = features['returns'] * np.log(data['Volume'] + 1)

        # 時系列特徴量
        features = self._add_time_series_features(features, data)

        # 欠損値処理
        features = features.fillna(method='ffill').fillna(0)

        # 無限大やNaNのチェック
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        return features

    def _add_technical_indicators(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標を追加"""

        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        for period in [14, 30]:
            if len(data) > period:
                avg_gain = gain.rolling(period).mean()
                avg_loss = loss.rolling(period).mean()
                rs = avg_gain / avg_loss
                features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = data['Close'].ewm(span=12).mean()
        ema26 = data['Close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # ボリンジャーバンド
        sma20 = data['Close'].rolling(20).mean()
        std20 = data['Close'].rolling(20).std()
        features['bb_upper'] = sma20 + (std20 * 2)
        features['bb_lower'] = sma20 - (std20 * 2)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma20
        features['bb_position'] = (data['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

        # ストキャスティック
        if len(data) > 14:
            low_14 = data['Low'].rolling(14).min()
            high_14 = data['High'].rolling(14).max()
            features['stoch_k'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)
            features['stoch_d'] = features['stoch_k'].rolling(3).mean()

        return features

    def _add_time_series_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """時系列特徴量を追加"""

        # ラグ特徴量
        for lag in [1, 2, 3, 5, 10]:
            if len(data) > lag:
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                features[f'volume_lag_{lag}'] = data['Volume'].shift(lag) if 'Volume' in data.columns else 0

        # ローリング統計
        for window in [5, 10, 20]:
            if len(data) > window:
                features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
                features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
                features[f'returns_skew_{window}'] = features['returns'].rolling(window).skew()
                features[f'returns_kurt_{window}'] = features['returns'].rolling(window).kurt()

        # モメンタム特徴量
        for window in [10, 20]:
            if len(data) > window:
                features[f'momentum_{window}'] = data['Close'] / data['Close'].shift(window) - 1

        return features

    def _create_robust_targets(self, data: pd.DataFrame) -> Dict[PredictionTask, pd.Series]:
        """頑健なターゲット変数作成"""

        targets = {}

        # 価格方向予測（改善版）
        returns = data['Close'].pct_change().shift(-1)  # 翌日のリターン

        # 閾値をデータ依存に調整
        volatility = returns.rolling(20).std()
        threshold_up = volatility.quantile(0.7)  # 上位30%のボラティリティ
        threshold_down = -threshold_up

        direction = pd.Series(index=data.index, dtype='int')
        direction[returns > threshold_up] = 1      # 上昇
        direction[returns < threshold_down] = -1   # 下落
        direction[(returns >= threshold_down) & (returns <= threshold_up)] = 0  # 横ばい

        targets[PredictionTask.PRICE_DIRECTION] = direction

        # 価格回帰予測
        targets[PredictionTask.PRICE_REGRESSION] = data['Close'].shift(-1)

        # ボラティリティ予測
        high_low_range = (data['High'] - data['Low']) / data['Close']
        targets[PredictionTask.VOLATILITY] = high_low_range.shift(-1)

        # トレンド強度予測
        trend_strength = abs(returns.rolling(5).mean()) / returns.rolling(5).std()
        targets[PredictionTask.TREND_STRENGTH] = trend_strength.shift(-1)

        return targets

    def _final_validation(self, features: pd.DataFrame, targets: Dict[PredictionTask, pd.Series]) -> Tuple[pd.DataFrame, Dict[PredictionTask, pd.Series]]:
        """最終検証とクリーニング"""

        # 特徴量のフィルタリング
        # 分散がゼロの特徴量を除去
        feature_variance = features.var()
        valid_features = feature_variance[feature_variance > 1e-10].index
        features = features[valid_features]

        # 相関が高すぎる特徴量を除去（簡易版）
        correlation_matrix = features.corr().abs()
        high_corr_pairs = np.where(np.triu(correlation_matrix, k=1) > 0.95)
        features_to_drop = set()

        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            # より多くの特徴量と相関が高い方を除去
            col_i_corr_count = (correlation_matrix.iloc[i] > 0.8).sum()
            col_j_corr_count = (correlation_matrix.iloc[j] > 0.8).sum()

            if col_i_corr_count > col_j_corr_count:
                features_to_drop.add(features.columns[i])
            else:
                features_to_drop.add(features.columns[j])

        if features_to_drop:
            self.logger.info(f"Removing {len(features_to_drop)} highly correlated features")
            features = features.drop(columns=list(features_to_drop))

        # ターゲットのクリーニング
        cleaned_targets = {}
        for task, target in targets.items():
            # 最後の一つの値は不明なので除去
            clean_target = target[:-1].dropna()

            if len(clean_target) > 50:  # 十分なサンプル数がある場合のみ
                cleaned_targets[task] = clean_target
            else:
                self.logger.warning(f"Insufficient samples for {task.value}: {len(clean_target)}")

        return features, cleaned_targets

    async def _fallback_data_preparation(self, symbol: str, period: str) -> Tuple[pd.DataFrame, Dict[PredictionTask, pd.Series]]:
        """フォールバックデータ準備"""

        # 最小限のデータで続行
        try:
            synthetic_data = self._generate_synthetic_data(symbol, period)
            basic_features = self._extract_comprehensive_basic_features(synthetic_data)
            basic_targets = self._create_robust_targets(synthetic_data)

            return self._final_validation(basic_features, basic_targets)

        except Exception as e:
            self.logger.error(f"Fallback data preparation also failed: {e}")
            raise ValueError(f"Cannot prepare any data for {symbol}")

    def _convert_featureset_to_dataframe(self, feature_set, data: pd.DataFrame) -> pd.DataFrame:
        """FeatureSetをDataFrameに変換（簡易版）"""
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
                    features_df = self._extract_comprehensive_basic_features(data)

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
                    features_df = self._extract_comprehensive_basic_features(data)

            # 欠損値処理
            features_df = features_df.fillna(method='ffill').fillna(0)

            return features_df

        except Exception as e:
            self.logger.error(f"Failed to convert FeatureSet to DataFrame: {e}")
            # フォールバック：基本特徴量を使用
            return self._extract_comprehensive_basic_features(data)

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
            },
            ModelType.LIGHTGBM: {
                'classifier_params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                },
                'regressor_params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
            }
        }

        # 訓練済みモデル
        self.trained_models = {}
        self.label_encoders = {}

        # メタデータ管理
        self.metadata_manager = ModelMetadataManager(self.data_dir / "ml_predictions.db")

        # モデル訓練器
        self.trainers = {
            ModelType.RANDOM_FOREST: RandomForestTrainer(ModelType.RANDOM_FOREST, self.model_configs[ModelType.RANDOM_FOREST]),
            ModelType.XGBOOST: XGBoostTrainer(ModelType.XGBOOST, self.model_configs[ModelType.XGBOOST]) if XGBOOST_AVAILABLE else None,
            ModelType.LIGHTGBM: LightGBMTrainer(ModelType.LIGHTGBM, self.model_configs[ModelType.LIGHTGBM]) if LIGHTGBM_AVAILABLE else None
        }
        # Noneを除去
        self.trainers = {k: v for k, v in self.trainers.items() if v is not None}

        # アンサンブル重み（性能ベース）
        self.ensemble_weights = {}

        # 訓練済みモデルのロード
        self._load_trained_models()

        # データベース初期化
        self.db_path = self.data_dir / "ml_predictions.db"
        self._init_database()

        # データ準備パイプライン
        self.data_pipeline = DataPreparationPipeline()

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
        """改良されたアンサンブル予測"""

        model_predictions = {}
        model_confidences = {}
        model_quality_scores = {}

        # 各モデルで予測
        for model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM]:
            model_key = f"{symbol}_{task.value}_{model_type.value}"

            if model_key not in self.trained_models:
                continue

            try:
                model = self.trained_models[model_key]

                # モデル品質スコア取得（過去の性能に基づく）
                quality_score = await self._get_model_quality_score(symbol, model_type, task)
                model_quality_scores[model_type.value] = quality_score

                if task == PredictionTask.PRICE_DIRECTION:
                    # 分類予測の改良
                    pred_proba = model.predict_proba(features)
                    pred_class = model.predict(features)[0]

                    # ラベルエンコーダーで元のラベルに変換
                    if model_key in self.label_encoders:
                        le = self.label_encoders[model_key]
                        pred_class = le.inverse_transform([pred_class])[0]

                    # 改良された信頼度計算
                    confidence = self._calculate_classification_confidence(pred_proba[0], quality_score)
                    model_predictions[model_type.value] = pred_class
                    model_confidences[model_type.value] = confidence

                else:  # PredictionTask.PRICE_REGRESSION
                    # 回帰予測の改良
                    pred_value = model.predict(features)[0]

                    # 回帰信頼度計算（特徴量の不確実性を考慮）
                    confidence = self._calculate_regression_confidence(model, features, quality_score)

                    model_predictions[model_type.value] = pred_value
                    model_confidences[model_type.value] = confidence

            except Exception as e:
                self.logger.error(f"Prediction failed for {model_key}: {e}")

        if not model_predictions:
            return None

        # 動的重み調整（性能ベース）
        dynamic_weights = await self._calculate_dynamic_weights(
            symbol, task, model_quality_scores, model_confidences
        )

        # アンサンブル統合の改良
        if task == PredictionTask.PRICE_DIRECTION:
            # 改良された多数決+重み付け+信頼度調整
            ensemble_result = self._ensemble_classification(
                model_predictions, model_confidences, dynamic_weights
            )
        else:  # PredictionTask.PRICE_REGRESSION
            # 改良された重み付き平均+不確実性考慮
            ensemble_result = self._ensemble_regression(
                model_predictions, model_confidences, dynamic_weights
            )

        final_prediction = ensemble_result['prediction']
        confidence = ensemble_result['confidence']
        consensus_strength = ensemble_result['consensus_strength']
        disagreement_score = ensemble_result['disagreement_score']

        # 追加の品質メトリクス
        quality_metrics = {
            'model_count': len(model_predictions),
            'avg_model_quality': np.mean(list(model_quality_scores.values())),
            'confidence_variance': np.var(list(model_confidences.values())),
            'prediction_stability': self._calculate_prediction_stability(model_predictions)
        }

        ensemble_prediction = EnsemblePrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            final_prediction=final_prediction,
            confidence=confidence,
            model_predictions=model_predictions,
            model_weights=dynamic_weights,
            consensus_strength=consensus_strength,
            disagreement_score=disagreement_score
        )

        # 品質メトリクスを追加属性として保存
        ensemble_prediction.quality_metrics = quality_metrics

        # 予測結果保存
        await self._save_prediction_result(ensemble_prediction, task)

        return ensemble_prediction

    async def _get_model_quality_score(self, symbol: str, model_type: ModelType,
                                     task: PredictionTask) -> float:
        """過去の性能に基づくモデル品質スコア取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT accuracy, f1_score, cross_val_mean
                    FROM model_performance_history mph
                    JOIN model_metadata mm ON mph.model_id = mm.model_id
                    WHERE mm.symbol = ? AND mm.model_type = ? AND mm.task = ?
                    ORDER BY mph.created_at DESC LIMIT 5
                """, (symbol, model_type.value, task.value))

                results = cursor.fetchall()
                if not results:
                    return 0.6  # デフォルト品質スコア

                # 直近5回の平均性能から品質スコア計算
                recent_scores = []
                for accuracy, f1_score, cross_val_mean in results:
                    # 複数メトリクスの重み付き平均
                    score = (0.4 * (accuracy or 0.5) +
                            0.4 * (f1_score or 0.5) +
                            0.2 * (cross_val_mean or 0.5))
                    recent_scores.append(score)

                return np.mean(recent_scores)

        except Exception as e:
            self.logger.error(f"Failed to get model quality score: {e}")
            return 0.6

    def _calculate_classification_confidence(self, pred_proba: np.ndarray,
                                           quality_score: float) -> float:
        """改良された分類信頼度計算"""
        # 確率の最大値
        max_prob = np.max(pred_proba)

        # エントロピーベースの不確実性
        entropy = -np.sum(pred_proba * np.log(pred_proba + 1e-15))
        normalized_entropy = entropy / np.log(len(pred_proba))
        certainty = 1.0 - normalized_entropy

        # 品質スコアと確率情報の組み合わせ
        confidence = (0.5 * max_prob + 0.3 * certainty + 0.2 * quality_score)

        return np.clip(confidence, 0.1, 0.95)

    def _calculate_regression_confidence(self, model, features: pd.DataFrame,
                                       quality_score: float) -> float:
        """回帰予測の信頼度計算"""
        try:
            # 特徴量の代表性チェック（学習データとの類似性）
            feature_uncertainty = self._estimate_feature_uncertainty(features)

            # モデル固有の予測分散（可能な場合）
            prediction_variance = 0.1  # デフォルト値
            if hasattr(model, 'predict') and hasattr(model, 'estimators_'):
                # Random Forestの場合、各木の予測のばらつきから分散推定
                try:
                    predictions = [tree.predict(features)[0] for tree in model.estimators_[:10]]
                    prediction_variance = np.var(predictions)
                except:
                    pass

            # 信頼度計算
            uncertainty = feature_uncertainty + prediction_variance
            confidence = quality_score * (1.0 - np.tanh(uncertainty))

            return np.clip(confidence, 0.1, 0.95)

        except Exception as e:
            self.logger.error(f"Failed to calculate regression confidence: {e}")
            return quality_score * 0.8

    def _estimate_feature_uncertainty(self, features: pd.DataFrame) -> float:
        """特徴量の不確実性推定"""
        try:
            # 特徴量の統計的性質をチェック
            uncertainty_factors = []

            for col in features.columns:
                if features[col].dtype in ['float64', 'int64']:
                    # 数値特徴量の変動性
                    col_std = features[col].std()
                    col_mean = abs(features[col].mean())
                    if col_mean > 0:
                        cv = col_std / col_mean  # 変動係数
                        uncertainty_factors.append(min(cv, 1.0))

            return np.mean(uncertainty_factors) if uncertainty_factors else 0.2

        except Exception:
            return 0.2

    async def _calculate_dynamic_weights(self, symbol: str, task: PredictionTask,
                                       quality_scores: Dict[str, float],
                                       confidences: Dict[str, float]) -> Dict[ModelType, float]:
        """動的重み計算"""
        try:
            # 基本重み（設定値）
            base_weights = self.ensemble_weights.get(symbol, {}).get(task, {})

            dynamic_weights = {}
            total_score = 0

            for model_name in quality_scores.keys():
                model_type = ModelType(model_name)

                # 品質スコア、信頼度、基本重みの組み合わせ
                quality = quality_scores[model_name]
                confidence = confidences[model_name]
                base_weight = base_weights.get(model_type, 1.0)

                # 動的重み計算式
                dynamic_weight = (0.4 * quality + 0.3 * confidence + 0.3 * base_weight)
                dynamic_weights[model_type] = dynamic_weight
                total_score += dynamic_weight

            # 正規化
            if total_score > 0:
                for model_type in dynamic_weights:
                    dynamic_weights[model_type] /= total_score

            return dynamic_weights

        except Exception as e:
            self.logger.error(f"Failed to calculate dynamic weights: {e}")
            # フォールバック：均等重み
            num_models = len(quality_scores)
            return {ModelType(name): 1.0/num_models for name in quality_scores.keys()}

    def _ensemble_classification(self, predictions: Dict[str, Any],
                               confidences: Dict[str, float],
                               weights: Dict[ModelType, float]) -> Dict[str, Any]:
        """改良された分類アンサンブル"""
        weighted_votes = {}
        total_weighted_confidence = 0

        for model_name, prediction in predictions.items():
            model_type = ModelType(model_name)
            weight = weights.get(model_type, 1.0)
            confidence = confidences[model_name]

            # 重み付き投票
            vote_strength = weight * confidence
            if prediction not in weighted_votes:
                weighted_votes[prediction] = 0
            weighted_votes[prediction] += vote_strength
            total_weighted_confidence += vote_strength

        # 最終予測選択
        final_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]

        # アンサンブル信頼度
        if total_weighted_confidence > 0:
            ensemble_confidence = weighted_votes[final_prediction] / total_weighted_confidence
        else:
            ensemble_confidence = 0.5

        # コンセンサス強度
        unique_predictions = len(set(predictions.values()))
        consensus_strength = 1.0 - (unique_predictions - 1) / max(1, len(predictions) - 1)

        # 不一致スコア
        disagreement_score = 1.0 - consensus_strength

        return {
            'prediction': final_prediction,
            'confidence': ensemble_confidence,
            'consensus_strength': consensus_strength,
            'disagreement_score': disagreement_score
        }

    def _ensemble_regression(self, predictions: Dict[str, float],
                           confidences: Dict[str, float],
                           weights: Dict[ModelType, float]) -> Dict[str, Any]:
        """改良された回帰アンサンブル"""
        weighted_sum = 0
        total_weight = 0
        pred_values = list(predictions.values())

        # 重み付き平均計算
        for model_name, prediction in predictions.items():
            model_type = ModelType(model_name)
            weight = weights.get(model_type, 1.0)
            confidence = confidences[model_name]

            adjusted_weight = weight * confidence
            weighted_sum += prediction * adjusted_weight
            total_weight += adjusted_weight

        final_prediction = weighted_sum / total_weight if total_weight > 0 else np.mean(pred_values)

        # 予測値の分散ベースの信頼度
        prediction_std = np.std(pred_values)
        prediction_mean = np.mean(pred_values)

        if prediction_mean != 0:
            coefficient_of_variation = prediction_std / abs(prediction_mean)
        else:
            coefficient_of_variation = prediction_std

        # 信頼度：分散が小さいほど高い
        ensemble_confidence = np.mean(list(confidences.values())) * (1.0 - np.tanh(coefficient_of_variation))

        # コンセンサス強度
        if prediction_mean != 0:
            consensus_strength = 1.0 - (prediction_std / abs(prediction_mean))
        else:
            consensus_strength = 1.0 - prediction_std
        consensus_strength = np.clip(consensus_strength, 0.0, 1.0)

        # 不一致スコア
        disagreement_score = 1.0 - consensus_strength

        return {
            'prediction': final_prediction,
            'confidence': np.clip(ensemble_confidence, 0.1, 0.95),
            'consensus_strength': consensus_strength,
            'disagreement_score': disagreement_score
        }

    def _calculate_prediction_stability(self, predictions: Dict[str, Any]) -> float:
        """予測安定性計算"""
        try:
            pred_values = list(predictions.values())
            if len(set(str(p) for p in pred_values)) == 1:
                return 1.0  # 完全一致

            # 数値予測の場合
            if all(isinstance(p, (int, float)) for p in pred_values):
                mean_pred = np.mean(pred_values)
                std_pred = np.std(pred_values)
                if mean_pred != 0:
                    stability = 1.0 - (std_pred / abs(mean_pred))
                else:
                    stability = 1.0 - std_pred
                return np.clip(stability, 0.0, 1.0)

            # カテゴリ予測の場合
            unique_count = len(set(str(p) for p in pred_values))
            stability = 1.0 - (unique_count - 1) / max(1, len(pred_values) - 1)
            return np.clip(stability, 0.0, 1.0)

        except Exception:
            return 0.5

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
        """改良された予測結果保存"""

        try:
            # 品質メトリクスの取得（デフォルト値付き）
            quality_metrics = getattr(prediction, 'quality_metrics', {})

            with sqlite3.connect(self.db_path) as conn:
                # アンサンブル予測履歴に保存
                cursor = conn.execute("""
                    INSERT INTO ensemble_prediction_history
                    (symbol, timestamp, task, final_prediction, confidence,
                     consensus_strength, disagreement_score, model_count,
                     avg_model_quality, confidence_variance, prediction_stability,
                     model_predictions, model_weights, quality_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction.symbol,
                    prediction.timestamp.isoformat(),
                    task.value,
                    str(prediction.final_prediction),
                    prediction.confidence,
                    prediction.consensus_strength,
                    prediction.disagreement_score,
                    quality_metrics.get('model_count', len(prediction.model_predictions)),
                    quality_metrics.get('avg_model_quality', 0.0),
                    quality_metrics.get('confidence_variance', 0.0),
                    quality_metrics.get('prediction_stability', 0.0),
                    json.dumps(prediction.model_predictions),
                    json.dumps({k.value if hasattr(k, 'value') else str(k): v
                               for k, v in prediction.model_weights.items()}),
                    json.dumps(quality_metrics)
                ))

                prediction_id = cursor.lastrowid

                # 予測精度追跡テーブルにエントリ作成（後で実際値と比較用）
                conn.execute("""
                    INSERT INTO prediction_accuracy_tracking
                    (prediction_id, symbol, predicted_value, prediction_date,
                     confidence_at_prediction, model_used, task)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(prediction_id),
                    prediction.symbol,
                    str(prediction.final_prediction),
                    prediction.timestamp.isoformat(),
                    prediction.confidence,
                    'ensemble',
                    task.value
                ))

                # モデル重み履歴の更新
                await self._update_model_weights_history(
                    prediction.symbol, task, prediction.model_weights, conn
                )

        except Exception as e:
            self.logger.error(f"Failed to save prediction result: {e}")

    async def _update_model_weights_history(self, symbol: str, task: PredictionTask,
                                          weights: Dict, conn):
        """モデル重み履歴更新"""
        try:
            for model_type, weight in weights.items():
                model_type_str = model_type.value if hasattr(model_type, 'value') else str(model_type)

                conn.execute("""
                    INSERT INTO model_weight_history
                    (symbol, task, model_type, dynamic_weight)
                    VALUES (?, ?, ?, ?)
                """, (symbol, task.value, model_type_str, weight))

        except Exception as e:
            self.logger.error(f"Failed to update model weights history: {e}")

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

# このファイルは本体コードのみ。テストはtest_ml_prediction_models.pyにて実行。