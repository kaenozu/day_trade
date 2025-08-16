#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Timeframe Predictor - マルチタイムフレーム予測システム

複数の時間軸（日次、週次、月次）での価格予測を統合
Issue #882対応：デイトレード以外の取引機能実装
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
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
    from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
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
    from ml_prediction_models import MLPredictionModels, ModelType, PredictionTask
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False

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

class TimeFrame(Enum):
    """タイムフレーム列挙"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class MultiTimeframePredictionTask(Enum):
    """マルチタイムフレーム予測タスク"""
    PRICE_DIRECTION = "price_direction"
    VOLATILITY = "volatility"
    MOMENTUM_STRENGTH = "momentum_strength"
    TREND_CONTINUATION = "trend_continuation"
    TREND_STRENGTH = "trend_strength"
    MARKET_REGIME = "market_regime"

@dataclass
class TimeFrameConfig:
    """タイムフレーム設定"""
    name: str
    prediction_horizon_days: int
    data_period: str
    min_training_samples: int
    enabled: bool
    description: str

@dataclass
class MultiTimeframePrediction:
    """マルチタイムフレーム予測結果"""
    symbol: str
    timestamp: datetime
    timeframe: TimeFrame
    task: MultiTimeframePredictionTask
    prediction: Union[str, float]
    confidence: float
    model_predictions: Dict[str, Any]
    feature_importance: Dict[str, float]
    explanation: str

@dataclass
class IntegratedPrediction:
    """統合予測結果"""
    symbol: str
    timestamp: datetime
    timeframe_predictions: Dict[TimeFrame, List[MultiTimeframePrediction]]
    integrated_direction: str
    integrated_confidence: float
    consistency_score: float
    risk_assessment: str
    recommendation: str

class MultiTimeframePredictor:
    """マルチタイムフレーム予測システム"""

    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)

        # 設定読み込み
        self.config_path = config_path or Path("config/multi_timeframe_config.yaml")
        self.config = self._load_configuration()

        # データディレクトリ初期化
        self.data_dir = Path("multi_timeframe_data")
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        # データベース初期化
        self.db_path = self.data_dir / "multi_timeframe_predictions.db"
        self._init_database()

        # タイムフレーム設定
        self.timeframes = self._init_timeframes()

        # モデル管理
        self.trained_models = {}  # {timeframe: {task: {model_type: model}}}
        self.scalers = {}
        self.label_encoders = {}

        # 性能追跡
        self.model_performances = {}

        # 既存システム統合
        if ML_MODELS_AVAILABLE:
            self.ml_models = MLPredictionModels()
        else:
            self.ml_models = None

        self.logger.info("Multi-timeframe predictor initialized")

    def _load_configuration(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'timeframes': {
                'daily': {
                    'name': 'デイトレード',
                    'prediction_horizon_days': 1,
                    'data_period': '1y',
                    'min_training_samples': 100,
                    'enabled': True,
                    'description': '翌日の価格変動予測'
                },
                'weekly': {
                    'name': '週間予測',
                    'prediction_horizon_days': 7,
                    'data_period': '2y',
                    'min_training_samples': 200,
                    'enabled': True,
                    'description': '1週間後の価格変動予測'
                }
            }
        }

    def _init_timeframes(self) -> Dict[TimeFrame, TimeFrameConfig]:
        """タイムフレーム設定初期化"""
        timeframes = {}

        for tf_name, tf_config in self.config.get('timeframes', {}).items():
            if tf_config.get('enabled', False):
                try:
                    timeframe = TimeFrame(tf_name)
                    config = TimeFrameConfig(
                        name=tf_config.get('name', tf_name),
                        prediction_horizon_days=tf_config.get('prediction_horizon_days', 1),
                        data_period=tf_config.get('data_period', '1y'),
                        min_training_samples=tf_config.get('min_training_samples', 100),
                        enabled=tf_config.get('enabled', True),
                        description=tf_config.get('description', '')
                    )
                    timeframes[timeframe] = config
                except ValueError:
                    self.logger.warning(f"Unknown timeframe: {tf_name}")

        return timeframes

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # マルチタイムフレーム予測テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS multi_timeframe_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    task TEXT NOT NULL,
                    prediction TEXT,
                    confidence REAL,
                    model_predictions TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 統合予測テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS integrated_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    integrated_direction TEXT,
                    integrated_confidence REAL,
                    consistency_score REAL,
                    risk_assessment TEXT,
                    recommendation TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # モデル性能テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS multi_timeframe_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timeframe TEXT NOT NULL,
                    task TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    cross_val_mean REAL,
                    training_time REAL,
                    training_date TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def prepare_timeframe_data(self, symbol: str, timeframe: TimeFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """タイムフレーム別データ準備"""

        config = self.timeframes[timeframe]
        self.logger.info(f"Preparing {timeframe.value} data for {symbol}")

        # 履歴データ取得
        if REAL_DATA_PROVIDER_AVAILABLE:
            data = await real_data_provider.get_stock_data(symbol, config.data_period)
        else:
            # ダミーデータ生成
            data = self._generate_dummy_data(config.data_period)

        if data.empty:
            raise ValueError(f"No data available for {symbol}")

        # タイムフレーム別特徴量抽出
        features = await self._extract_timeframe_features(data, timeframe)

        # タイムフレーム別ターゲット変数作成
        targets = self._create_timeframe_targets(data, timeframe, config.prediction_horizon_days)

        self.logger.info(f"{timeframe.value} features shape: {features.shape}")

        return features, targets

    async def _extract_timeframe_features(self, data: pd.DataFrame, timeframe: TimeFrame) -> pd.DataFrame:
        """タイムフレーム別特徴量抽出"""

        features = pd.DataFrame(index=data.index)

        # 基本価格特徴量
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))

        # タイムフレーム別の移動平均期間
        if timeframe == TimeFrame.DAILY:
            sma_periods = [5, 10, 20, 50]
            rsi_period = 14
        elif timeframe == TimeFrame.WEEKLY:
            sma_periods = [10, 20, 50, 100]
            rsi_period = 14
        elif timeframe == TimeFrame.MONTHLY:
            sma_periods = [20, 50, 100, 200]
            rsi_period = 20
        else:
            sma_periods = [5, 10, 20]
            rsi_period = 14

        # 移動平均特徴量
        for period in sma_periods:
            features[f'sma_{period}'] = data['Close'].rolling(period).mean()
            features[f'sma_ratio_{period}'] = data['Close'] / features[f'sma_{period}']

        # ボラティリティ
        volatility_window = min(20, len(data) // 4)
        features['volatility'] = features['returns'].rolling(volatility_window).std()

        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(rsi_period).mean()
        avg_loss = loss.rolling(rsi_period).mean()
        rs = avg_gain / avg_loss
        features['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = data['Close'].ewm(span=12).mean()
        ema26 = data['Close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # 出来高特徴量
        if 'Volume' in data.columns:
            volume_window = min(20, len(data) // 4)
            features['volume_ma'] = data['Volume'].rolling(volume_window).mean()
            features['volume_ratio'] = data['Volume'] / features['volume_ma']
        else:
            features['volume_ratio'] = 1.0

        # タイムフレーム特有特徴量
        if timeframe == TimeFrame.WEEKLY:
            # 週間トレンド強度
            features['weekly_trend'] = features['returns'].rolling(7).mean()
            features['weekly_volatility'] = features['returns'].rolling(7).std()

        elif timeframe == TimeFrame.MONTHLY:
            # 月間モメンタム
            features['monthly_momentum'] = data['Close'] / data['Close'].shift(30) - 1
            features['monthly_max_drawdown'] = self._calculate_max_drawdown(data['Close'], 30)

        # 欠損値処理
        features = features.fillna(method='ffill').fillna(0)

        return features

    def _calculate_max_drawdown(self, prices: pd.Series, window: int) -> pd.Series:
        """最大ドローダウン計算"""
        rolling_max = prices.rolling(window).max()
        drawdown = (prices - rolling_max) / rolling_max
        return drawdown.rolling(window).min()

    def _create_timeframe_targets(self, data: pd.DataFrame, timeframe: TimeFrame, horizon: int) -> Dict[str, pd.Series]:
        """タイムフレーム別ターゲット変数作成"""

        targets = {}

        # 設定から閾値取得
        task_configs = self.config.get('prediction_tasks', {}).get(timeframe.value, {})

        # 価格方向予測
        if 'price_direction' in task_configs:
            direction_config = task_configs['price_direction']
            threshold = direction_config.get('threshold_percent', 1.0) / 100.0

            # horizon日後のリターン
            future_returns = data['Close'].pct_change(horizon).shift(-horizon)

            direction = pd.Series(index=data.index, dtype='object')
            direction[future_returns > threshold] = '上昇'
            direction[future_returns < -threshold] = '下落'
            direction[(future_returns >= -threshold) & (future_returns <= threshold)] = '横ばい'

            targets[MultiTimeframePredictionTask.PRICE_DIRECTION.value] = direction

        # ボラティリティ予測
        if 'volatility' in task_configs:
            volatility_window = min(horizon * 2, 20)
            future_volatility = data['Close'].pct_change().rolling(volatility_window).std().shift(-horizon)
            targets[MultiTimeframePredictionTask.VOLATILITY.value] = future_volatility

        # トレンド継続予測（週間以上）
        if timeframe in [TimeFrame.WEEKLY, TimeFrame.MONTHLY] and 'trend_continuation' in task_configs:
            current_trend = data['Close'].pct_change(horizon)
            future_trend = data['Close'].pct_change(horizon).shift(-horizon)

            trend_continuation = pd.Series(index=data.index, dtype='object')
            # 同じ方向なら継続、逆なら反転
            same_direction = (current_trend * future_trend) > 0
            trend_continuation[same_direction] = '継続'
            trend_continuation[~same_direction] = '反転'

            targets[MultiTimeframePredictionTask.TREND_CONTINUATION.value] = trend_continuation

        return targets

    def _generate_dummy_data(self, period: str) -> pd.DataFrame:
        """ダミーデータ生成"""

        # 期間解析
        if period.endswith('y'):
            years = int(period[:-1])
            days = years * 365
        elif period.endswith('mo'):
            months = int(period[:-2])
            days = months * 30
        else:
            days = 365

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)

        prices = [1000]
        for _ in range(len(dates)-1):
            change = np.random.normal(0, 0.02)
            prices.append(prices[-1] * (1 + change))

        return pd.DataFrame({
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(10000, 100000, len(dates))
        }, index=dates)

    async def train_timeframe_models(self, symbol: str, timeframe: TimeFrame) -> Dict[str, Dict[str, Any]]:
        """タイムフレーム別モデル訓練"""

        self.logger.info(f"Training {timeframe.value} models for {symbol}")

        # データ準備
        features, targets = await self.prepare_timeframe_data(symbol, timeframe)

        # 有効なインデックス（最後のhorizon日分は除外）
        config = self.timeframes[timeframe]
        valid_idx = features.index[:-config.prediction_horizon_days]
        X = features.loc[valid_idx]

        performances = {}

        # タスク別モデル訓練
        for task_name, target_series in targets.items():
            if task_name not in performances:
                performances[task_name] = {}

            try:
                task_performance = await self._train_task_models(
                    X, target_series, symbol, timeframe, task_name, valid_idx
                )
                performances[task_name] = task_performance

            except Exception as e:
                self.logger.error(f"Failed to train {task_name} for {timeframe.value}: {e}")

        # 性能結果保存
        await self._save_timeframe_performances(performances, symbol, timeframe)

        return performances

    async def _train_task_models(self, X: pd.DataFrame, target_series: pd.Series,
                                symbol: str, timeframe: TimeFrame, task_name: str,
                                valid_idx: pd.Index) -> Dict[str, Any]:
        """タスク別モデル訓練"""

        # ターゲット準備
        y = target_series.loc[valid_idx].dropna()
        X_clean = X.loc[y.index]

        if len(y) < self.timeframes[timeframe].min_training_samples:
            raise ValueError(f"Insufficient samples: {len(y)} < {self.timeframes[timeframe].min_training_samples}")

        # 訓練・テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, test_size=0.2, random_state=42,
            stratify=y if task_name.endswith('direction') or task_name.endswith('continuation') else None
        )

        performances = {}

        # Random Forest
        try:
            rf_perf = await self._train_random_forest_timeframe(
                X_train, X_test, y_train, y_test, symbol, timeframe, task_name
            )
            performances['RandomForest'] = rf_perf
        except Exception as e:
            self.logger.error(f"Random Forest training failed: {e}")

        # XGBoost
        if XGBOOST_AVAILABLE:
            try:
                xgb_perf = await self._train_xgboost_timeframe(
                    X_train, X_test, y_train, y_test, symbol, timeframe, task_name
                )
                performances['XGBoost'] = xgb_perf
            except Exception as e:
                self.logger.error(f"XGBoost training failed: {e}")

        return performances

    async def _train_random_forest_timeframe(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                           y_train: pd.Series, y_test: pd.Series,
                                           symbol: str, timeframe: TimeFrame, task_name: str) -> Dict[str, Any]:
        """Random Forest タイムフレーム訓練"""

        start_time = datetime.now()

        # モデル設定取得
        model_config = self.config.get('models', {}).get(timeframe.value, {}).get('random_forest', {})

        if task_name.endswith('direction') or task_name.endswith('continuation'):
            # 分類タスク
            model = RandomForestClassifier(
                n_estimators=model_config.get('n_estimators', 200),
                max_depth=model_config.get('max_depth', 15),
                min_samples_split=model_config.get('min_samples_split', 10),
                class_weight=model_config.get('class_weight', 'balanced'),
                random_state=42,
                n_jobs=-1
            )

            # ラベルエンコーディング
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)

            model.fit(X_train, y_train_encoded)
            y_pred = model.predict(X_test)

            # 性能計算
            accuracy = accuracy_score(y_test_encoded, y_pred)
            report = classification_report(y_test_encoded, y_pred, output_dict=True)

            # モデル保存
            model_key = f"{symbol}_{timeframe.value}_{task_name}_RandomForest"
            if timeframe not in self.trained_models:
                self.trained_models[timeframe] = {}
            if task_name not in self.trained_models[timeframe]:
                self.trained_models[timeframe][task_name] = {}

            self.trained_models[timeframe][task_name]['RandomForest'] = model
            self.label_encoders[model_key] = le

            performance = {
                'model_type': 'RandomForest',
                'task_type': 'classification',
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
                'training_time': (datetime.now() - start_time).total_seconds()
            }

        else:
            # 回帰タスク
            model = RandomForestRegressor(
                n_estimators=model_config.get('n_estimators', 200),
                max_depth=model_config.get('max_depth', 15),
                min_samples_split=model_config.get('min_samples_split', 10),
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # 性能計算
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # モデル保存
            if timeframe not in self.trained_models:
                self.trained_models[timeframe] = {}
            if task_name not in self.trained_models[timeframe]:
                self.trained_models[timeframe][task_name] = {}

            self.trained_models[timeframe][task_name]['RandomForest'] = model

            performance = {
                'model_type': 'RandomForest',
                'task_type': 'regression',
                'mse': mse,
                'r2_score': r2,
                'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
                'training_time': (datetime.now() - start_time).total_seconds()
            }

        return performance

    async def _train_xgboost_timeframe(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                     y_train: pd.Series, y_test: pd.Series,
                                     symbol: str, timeframe: TimeFrame, task_name: str) -> Dict[str, Any]:
        """XGBoost タイムフレーム訓練"""

        start_time = datetime.now()

        # モデル設定取得
        model_config = self.config.get('models', {}).get(timeframe.value, {}).get('xgboost', {})

        if task_name.endswith('direction') or task_name.endswith('continuation'):
            # 分類タスク
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)

            model = xgb.XGBClassifier(
                n_estimators=model_config.get('n_estimators', 300),
                max_depth=model_config.get('max_depth', 8),
                learning_rate=model_config.get('learning_rate', 0.1),
                subsample=model_config.get('subsample', 0.8),
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )

            model.fit(X_train, y_train_encoded)
            y_pred = model.predict(X_test)

            # 性能計算
            accuracy = accuracy_score(y_test_encoded, y_pred)
            report = classification_report(y_test_encoded, y_pred, output_dict=True)

            # モデル保存
            model_key = f"{symbol}_{timeframe.value}_{task_name}_XGBoost"
            if timeframe not in self.trained_models:
                self.trained_models[timeframe] = {}
            if task_name not in self.trained_models[timeframe]:
                self.trained_models[timeframe][task_name] = {}

            self.trained_models[timeframe][task_name]['XGBoost'] = model
            self.label_encoders[model_key] = le

            performance = {
                'model_type': 'XGBoost',
                'task_type': 'classification',
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
                'training_time': (datetime.now() - start_time).total_seconds()
            }

        else:
            # 回帰タスク
            model = xgb.XGBRegressor(
                n_estimators=model_config.get('n_estimators', 300),
                max_depth=model_config.get('max_depth', 8),
                learning_rate=model_config.get('learning_rate', 0.1),
                subsample=model_config.get('subsample', 0.8),
                random_state=42,
                n_jobs=-1,
                eval_metric='rmse'
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # 性能計算
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # モデル保存
            if timeframe not in self.trained_models:
                self.trained_models[timeframe] = {}
            if task_name not in self.trained_models[timeframe]:
                self.trained_models[timeframe][task_name] = {}

            self.trained_models[timeframe][task_name]['XGBoost'] = model

            performance = {
                'model_type': 'XGBoost',
                'task_type': 'regression',
                'mse': mse,
                'r2_score': r2,
                'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
                'training_time': (datetime.now() - start_time).total_seconds()
            }

        return performance

    async def predict_all_timeframes(self, symbol: str) -> IntegratedPrediction:
        """全タイムフレーム予測実行"""

        self.logger.info(f"Predicting all timeframes for {symbol}")

        timeframe_predictions = {}

        # 各タイムフレームで予測
        for timeframe in self.timeframes:
            try:
                predictions = await self._predict_timeframe(symbol, timeframe)
                if predictions:
                    timeframe_predictions[timeframe] = predictions
            except Exception as e:
                self.logger.error(f"Prediction failed for {timeframe.value}: {e}")

        # 統合予測
        integrated = await self._integrate_predictions(symbol, timeframe_predictions)

        # 結果保存
        await self._save_integrated_prediction(integrated)

        return integrated

    async def _predict_timeframe(self, symbol: str, timeframe: TimeFrame) -> List[MultiTimeframePrediction]:
        """タイムフレーム別予測"""

        # データ準備
        features, _ = await self.prepare_timeframe_data(symbol, timeframe)
        latest_features = features.tail(1)

        predictions = []

        # 各タスクで予測
        if timeframe in self.trained_models:
            for task_name, task_models in self.trained_models[timeframe].items():
                try:
                    # モデル予測統合
                    model_predictions = {}

                    for model_type, model in task_models.items():
                        pred = model.predict(latest_features)[0]

                        # ラベルデコーディング
                        model_key = f"{symbol}_{timeframe.value}_{task_name}_{model_type}"
                        if model_key in self.label_encoders:
                            le = self.label_encoders[model_key]
                            pred = le.inverse_transform([pred])[0]

                        model_predictions[model_type] = pred

                    # 多数決または平均
                    if task_name.endswith('direction') or task_name.endswith('continuation'):
                        # 分類：多数決
                        final_prediction = max(set(model_predictions.values()),
                                             key=list(model_predictions.values()).count)
                    else:
                        # 回帰：平均
                        final_prediction = np.mean(list(model_predictions.values()))

                    # 信頼度計算
                    confidence = self._calculate_prediction_confidence(model_predictions, task_name)

                    prediction = MultiTimeframePrediction(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        task=MultiTimeframePredictionTask(task_name),
                        prediction=final_prediction,
                        confidence=confidence,
                        model_predictions=model_predictions,
                        feature_importance={},  # 省略
                        explanation=f"{timeframe.value}_{task_name}予測"
                    )

                    predictions.append(prediction)

                except Exception as e:
                    self.logger.error(f"Task prediction failed for {task_name}: {e}")

        return predictions

    def _calculate_prediction_confidence(self, model_predictions: Dict[str, Any], task_name: str) -> float:
        """予測信頼度計算"""

        if not model_predictions:
            return 0.0

        values = list(model_predictions.values())

        if task_name.endswith('direction') or task_name.endswith('continuation'):
            # 分類：一致度ベース
            unique_values = len(set(str(v) for v in values))
            consistency = 1.0 - (unique_values - 1) / max(1, len(values) - 1)
            return max(0.5, consistency)  # 最低50%
        else:
            # 回帰：変動係数ベース
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else 1.0
                confidence = max(0.5, 1.0 - min(1.0, cv))
                return confidence
            else:
                return 0.8

    async def _integrate_predictions(self, symbol: str,
                                   timeframe_predictions: Dict[TimeFrame, List[MultiTimeframePrediction]]) -> IntegratedPrediction:
        """予測統合"""

        # 価格方向予測の統合
        direction_predictions = {}
        for timeframe, predictions in timeframe_predictions.items():
            for pred in predictions:
                if pred.task == MultiTimeframePredictionTask.PRICE_DIRECTION:
                    direction_predictions[timeframe] = pred.prediction

        # 統合方向決定
        if direction_predictions:
            # 長期予測により重みを付ける
            weighted_votes = {}
            total_weight = 0

            timeframe_weights = {
                TimeFrame.DAILY: 1.0,
                TimeFrame.WEEKLY: 2.0,
                TimeFrame.MONTHLY: 3.0,
                TimeFrame.QUARTERLY: 4.0
            }

            for timeframe, direction in direction_predictions.items():
                weight = timeframe_weights.get(timeframe, 1.0)
                if direction not in weighted_votes:
                    weighted_votes[direction] = 0
                weighted_votes[direction] += weight
                total_weight += weight

            integrated_direction = max(weighted_votes.items(), key=lambda x: x[1])[0]
            integrated_confidence = weighted_votes[integrated_direction] / total_weight if total_weight > 0 else 0.5
        else:
            integrated_direction = "不明"
            integrated_confidence = 0.0

        # 一貫性スコア計算
        consistency_score = self._calculate_consistency_score(direction_predictions)

        # リスク評価
        risk_assessment = self._assess_risk(timeframe_predictions, consistency_score)

        # 推奨事項
        recommendation = self._generate_recommendation(integrated_direction, integrated_confidence, consistency_score)

        return IntegratedPrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            timeframe_predictions=timeframe_predictions,
            integrated_direction=integrated_direction,
            integrated_confidence=integrated_confidence,
            consistency_score=consistency_score,
            risk_assessment=risk_assessment,
            recommendation=recommendation
        )

    def _calculate_consistency_score(self, direction_predictions: Dict[TimeFrame, str]) -> float:
        """一貫性スコア計算"""

        if len(direction_predictions) <= 1:
            return 1.0

        # 全ての予測が一致している場合は1.0
        directions = list(direction_predictions.values())
        unique_directions = len(set(directions))

        if unique_directions == 1:
            return 1.0
        else:
            return 1.0 - (unique_directions - 1) / max(1, len(directions) - 1)

    def _assess_risk(self, timeframe_predictions: Dict[TimeFrame, List[MultiTimeframePrediction]],
                    consistency_score: float) -> str:
        """リスク評価"""

        if consistency_score >= 0.8:
            risk_level = "低"
        elif consistency_score >= 0.6:
            risk_level = "中"
        else:
            risk_level = "高"

        # ボラティリティ考慮
        volatility_predictions = []
        for predictions in timeframe_predictions.values():
            for pred in predictions:
                if pred.task == MultiTimeframePredictionTask.VOLATILITY and isinstance(pred.prediction, (int, float)):
                    volatility_predictions.append(pred.prediction)

        if volatility_predictions:
            avg_volatility = np.mean(volatility_predictions)
            if avg_volatility > 0.03:  # 3%以上
                risk_level = "高" if risk_level != "高" else "非常に高い"

        return f"リスク: {risk_level}"

    def _generate_recommendation(self, direction: str, confidence: float, consistency: float) -> str:
        """推奨事項生成"""

        if confidence < 0.5 or consistency < 0.5:
            return "予測の信頼性が低いため、慎重な判断を推奨"

        if direction == "上昇":
            if confidence >= 0.8 and consistency >= 0.8:
                return "強い上昇予測 - 買いポジション検討"
            else:
                return "上昇予測 - 小さなポジションでの買い検討"
        elif direction == "下落":
            if confidence >= 0.8 and consistency >= 0.8:
                return "強い下落予測 - 売りポジション検討"
            else:
                return "下落予測 - ポジション削減検討"
        else:
            return "横ばい予測 - 様子見を推奨"

    async def _save_timeframe_performances(self, performances: Dict[str, Dict[str, Any]],
                                         symbol: str, timeframe: TimeFrame):
        """タイムフレーム性能保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                for task_name, task_perfs in performances.items():
                    for model_type, perf in task_perfs.items():
                        conn.execute("""
                            INSERT INTO multi_timeframe_performance
                            (timeframe, task, model_type, accuracy, precision_score, recall_score,
                             f1_score, training_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            timeframe.value,
                            task_name,
                            model_type,
                            perf.get('accuracy', perf.get('r2_score', 0)),
                            perf.get('precision', 0),
                            perf.get('recall', 0),
                            perf.get('f1_score', 0),
                            perf.get('training_time', 0)
                        ))
        except Exception as e:
            self.logger.error(f"Failed to save performances: {e}")

    async def _save_integrated_prediction(self, prediction: IntegratedPrediction):
        """統合予測保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO integrated_predictions
                    (symbol, timestamp, integrated_direction, integrated_confidence,
                     consistency_score, risk_assessment, recommendation)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction.symbol,
                    prediction.timestamp.isoformat(),
                    prediction.integrated_direction,
                    prediction.integrated_confidence,
                    prediction.consistency_score,
                    prediction.risk_assessment,
                    prediction.recommendation
                ))
        except Exception as e:
            self.logger.error(f"Failed to save integrated prediction: {e}")

    async def get_system_summary(self) -> Dict[str, Any]:
        """システムサマリー取得"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                # 最新性能
                cursor = conn.execute("""
                    SELECT timeframe, task, model_type, accuracy
                    FROM multi_timeframe_performance
                    ORDER BY training_date DESC
                    LIMIT 10
                """)
                performances = cursor.fetchall()

                # 最新予測
                cursor = conn.execute("""
                    SELECT symbol, integrated_direction, integrated_confidence, consistency_score
                    FROM integrated_predictions
                    ORDER BY created_at DESC
                    LIMIT 5
                """)
                predictions = cursor.fetchall()

                return {
                    'enabled_timeframes': [tf.value for tf in self.timeframes.keys()],
                    'trained_models_count': sum(
                        len(tasks) for tasks in self.trained_models.values()
                    ),
                    'recent_performances': [
                        {
                            'timeframe': p[0],
                            'task': p[1],
                            'model': p[2],
                            'accuracy': p[3]
                        } for p in performances
                    ],
                    'recent_predictions': [
                        {
                            'symbol': p[0],
                            'direction': p[1],
                            'confidence': p[2],
                            'consistency': p[3]
                        } for p in predictions
                    ]
                }
        except Exception as e:
            self.logger.error(f"Failed to get summary: {e}")
            return {
                'enabled_timeframes': [],
                'trained_models_count': 0,
                'recent_performances': [],
                'recent_predictions': []
            }

# グローバルインスタンス
multi_timeframe_predictor = MultiTimeframePredictor()

# テスト関数
async def test_multi_timeframe_predictor():
    """マルチタイムフレーム予測システムのテスト"""

    print("=== マルチタイムフレーム予測システム テスト ===")

    if not SKLEARN_AVAILABLE:
        print("❌ Scikit-learn not available")
        return

    predictor = MultiTimeframePredictor()

    test_symbols = ["7203", "8306"]

    print(f"\n[ 有効タイムフレーム: {list(predictor.timeframes.keys())} ]")

    for symbol in test_symbols:
        print(f"\n--- {symbol} マルチタイムフレーム訓練 ---")

        try:
            # 各タイムフレームで訓練
            for timeframe in predictor.timeframes:
                print(f"\n{timeframe.value} 訓練開始...")
                performances = await predictor.train_timeframe_models(symbol, timeframe)

                print(f"{timeframe.value} 訓練完了: {len(performances)} タスク")

                for task_name, task_perfs in performances.items():
                    print(f"  {task_name}:")
                    for model_type, perf in task_perfs.items():
                        if 'accuracy' in perf:
                            print(f"    {model_type}: 精度 {perf['accuracy']:.3f}")
                        elif 'r2_score' in perf:
                            print(f"    {model_type}: R² {perf['r2_score']:.3f}")

            # 統合予測テスト
            print(f"\n[ {symbol} 統合予測テスト ]")
            integrated = await predictor.predict_all_timeframes(symbol)

            print(f"統合方向: {integrated.integrated_direction}")
            print(f"統合信頼度: {integrated.integrated_confidence:.3f}")
            print(f"一貫性スコア: {integrated.consistency_score:.3f}")
            print(f"リスク評価: {integrated.risk_assessment}")
            print(f"推奨事項: {integrated.recommendation}")

            # タイムフレーム別詳細
            for timeframe, predictions in integrated.timeframe_predictions.items():
                print(f"\n{timeframe.value} 予測:")
                for pred in predictions:
                    print(f"  {pred.task.value}: {pred.prediction} (信頼度: {pred.confidence:.3f})")

        except Exception as e:
            print(f"❌ エラー: {e}")
            import traceback
            traceback.print_exc()

    # システムサマリー
    print(f"\n[ システムサマリー ]")
    summary = await predictor.get_system_summary()

    print(f"有効タイムフレーム: {summary['enabled_timeframes']}")
    print(f"訓練済みモデル数: {summary['trained_models_count']}")

    if summary['recent_performances']:
        print(f"最新性能（上位3）:")
        for perf in summary['recent_performances'][:3]:
            print(f"  {perf['timeframe']}_{perf['task']}_{perf['model']}: {perf['accuracy']:.3f}")

    print(f"\n=== マルチタイムフレーム予測システム テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_multi_timeframe_predictor())