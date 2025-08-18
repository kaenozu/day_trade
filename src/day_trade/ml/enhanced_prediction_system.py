#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Prediction System - 改良予測システム

Issue #810対応：予測精度向上システム
より多くの学習データ、改良された特徴量エンジニアリング、高度なアンサンブル学習
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# 機械学習ライブラリ
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

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

@dataclass
class EnhancedPrediction:
    """改良予測結果"""
    symbol: str
    prediction: int  # -1: 下落, 0: 横ばい, 1: 上昇
    confidence: float
    probability_distribution: Dict[int, float]  # {-1: prob, 0: prob, 1: prob}
    model_consensus: Dict[str, int]  # 各モデルの予測
    feature_importance: Dict[str, float]
    timestamp: datetime
    data_quality_score: float
    prediction_horizon: str = "1day"

@dataclass
class ModelPerformanceMetrics:
    """モデル性能メトリクス"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_samples: int
    feature_count: int
    last_updated: datetime

class AdvancedFeatureEngineering:
    """高度特徴量エンジニアリング"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """包括的特徴量作成"""

        if len(data) < 50:  # 最低限のデータ数
            self.logger.warning(f"データ不足: {len(data)}レコード（最低50必要）")

        features = pd.DataFrame(index=data.index)

        # 1. 価格ベース特徴量（改良版）
        features.update(self._create_price_features(data))

        # 2. 技術指標（拡張版）
        features.update(self._create_technical_indicators(data))

        # 3. ボラティリティ特徴量
        features.update(self._create_volatility_features(data))

        # 4. 出来高分析特徴量
        features.update(self._create_volume_features(data))

        # 5. 時系列特徴量
        features.update(self._create_temporal_features(data))

        # 6. 相対強度特徴量
        features.update(self._create_momentum_features(data))

        # 7. 統計的特徴量
        features.update(self._create_statistical_features(data))

        # 8. パターン認識特徴量
        features.update(self._create_pattern_features(data))

        # 欠損値処理（改良版、非推奨警告対応）
        features = features.ffill().bfill()
        features = features.replace([np.inf, -np.inf], 0)

        self.logger.info(f"包括的特徴量作成完了: {len(features.columns)}特徴量, {len(features)}サンプル")

        return features

    def _create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """価格ベース特徴量"""

        features = pd.DataFrame(index=data.index)

        if 'Close' in data.columns:
            close = data['Close']

            # 基本リターン
            features['return_1d'] = close.pct_change(1)
            features['return_3d'] = close.pct_change(3)
            features['return_5d'] = close.pct_change(5)
            features['return_10d'] = close.pct_change(10)
            features['return_20d'] = close.pct_change(20)

            # 対数リターン
            features['log_return_1d'] = np.log(close / close.shift(1))
            features['log_return_5d'] = np.log(close / close.shift(5))

            # 価格相対位置
            features['price_rank_5d'] = close.rolling(5).rank() / 5
            features['price_rank_10d'] = close.rolling(10).rank() / 10
            features['price_rank_20d'] = close.rolling(20).rank() / 20

            # 移動平均からの乖離率
            for period in [5, 10, 20, 50]:
                sma = close.rolling(period).mean()
                features[f'sma_deviation_{period}d'] = (close - sma) / sma

        # OHLC特徴量
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            features['hl_ratio'] = (data['High'] - data['Low']) / data['Close']
            features['oc_ratio'] = (data['Close'] - data['Open']) / data['Open']
            features['body_ratio'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low'] + 1e-10)
            features['upper_shadow'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / data['Close']
            features['lower_shadow'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / data['Close']

        return features

    def _create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """技術指標（拡張版）"""

        features = pd.DataFrame(index=data.index)

        if 'Close' not in data.columns:
            return features

        close = data['Close']

        # 移動平均系
        for period in [5, 10, 20, 50]:
            sma = close.rolling(period).mean()
            ema = close.ewm(span=period).mean()

            features[f'sma_{period}'] = sma / close
            features[f'ema_{period}'] = ema / close
            features[f'sma_slope_{period}'] = sma.diff() / sma
            features[f'ema_slope_{period}'] = ema.diff() / ema

        # RSI（複数期間）
        for period in [9, 14, 21]:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # ボリンジャーバンド
        for period in [10, 20]:
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f'bb_upper_{period}'] = (sma + 2*std) / close
            features[f'bb_lower_{period}'] = (sma - 2*std) / close
            features[f'bb_width_{period}'] = (4*std) / sma
            features[f'bb_position_{period}'] = (close - sma) / (2*std + 1e-10)

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        features['macd'] = macd_line / close
        features['macd_signal'] = signal_line / close
        features['macd_histogram'] = (macd_line - signal_line) / close

        # ストキャスティクス
        if all(col in data.columns for col in ['High', 'Low']):
            for period in [9, 14]:
                lowest_low = data['Low'].rolling(period).min()
                highest_high = data['High'].rolling(period).max()
                k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
                features[f'stoch_k_{period}'] = k_percent
                features[f'stoch_d_{period}'] = k_percent.rolling(3).mean()

        # Williams %R
        if all(col in data.columns for col in ['High', 'Low']):
            for period in [14, 21]:
                highest_high = data['High'].rolling(period).max()
                lowest_low = data['Low'].rolling(period).min()
                features[f'williams_r_{period}'] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

        return features

    def _create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ特徴量"""

        features = pd.DataFrame(index=data.index)

        if 'Close' not in data.columns:
            return features

        returns = data['Close'].pct_change()

        # 歴史的ボラティリティ（複数期間）
        for period in [5, 10, 20, 30]:
            vol = returns.rolling(period).std() * np.sqrt(252)
            features[f'volatility_{period}d'] = vol
            features[f'volatility_rank_{period}d'] = vol.rolling(60).rank() / 60

        # GARCH風ボラティリティ
        features['ewm_volatility'] = returns.ewm(span=30).std() * np.sqrt(252)

        # True Range
        if all(col in data.columns for col in ['High', 'Low', 'Close']):
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))

            # ATR
            for period in [14, 20]:
                atr = true_range.rolling(period).mean()
                features[f'atr_{period}'] = atr / data['Close']
                features[f'atr_ratio_{period}'] = atr / atr.rolling(60).mean()

        # パーキンソン推定値
        if all(col in data.columns for col in ['High', 'Low']):
            features['parkinson_vol'] = np.sqrt(
                0.361 * (np.log(data['High'] / data['Low'])) ** 2
            )

        return features

    def _create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """出来高分析特徴量"""

        features = pd.DataFrame(index=data.index)

        if 'Volume' not in data.columns:
            return features

        volume = data['Volume']

        # 出来高移動平均
        for period in [5, 10, 20]:
            vol_sma = volume.rolling(period).mean()
            features[f'volume_sma_{period}'] = volume / vol_sma
            features[f'volume_trend_{period}'] = vol_sma.pct_change()

        # 出来高ランク
        features['volume_rank_20d'] = volume.rolling(20).rank() / 20
        features['volume_rank_60d'] = volume.rolling(60).rank() / 60

        # 価格・出来高関係
        if 'Close' in data.columns:
            price_change = data['Close'].pct_change()
            features['pv_trend'] = price_change * volume
            features['volume_price_correlation'] = price_change.rolling(20).corr(volume.pct_change())

        # OBV（On Balance Volume）
        if 'Close' in data.columns:
            price_change = data['Close'].diff()
            obv = (np.sign(price_change) * volume).cumsum()
            features['obv'] = obv / obv.rolling(20).mean()
            features['obv_slope'] = obv.pct_change()

        return features

    def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """時系列特徴量"""

        features = pd.DataFrame(index=data.index)

        # 曜日・月効果
        if hasattr(data.index, 'dayofweek'):
            features['day_of_week'] = data.index.dayofweek
            features['is_monday'] = (data.index.dayofweek == 0).astype(int)
            features['is_friday'] = (data.index.dayofweek == 4).astype(int)
            features['month'] = data.index.month

        # 期間効果
        features['quarter'] = data.index.quarter if hasattr(data.index, 'quarter') else 1

        # ラグ特徴量
        if 'Close' in data.columns:
            returns = data['Close'].pct_change()
            for lag in [1, 2, 3, 5]:
                features[f'return_lag_{lag}'] = returns.shift(lag)

        return features

    def _create_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """相対強度特徴量"""

        features = pd.DataFrame(index=data.index)

        if 'Close' not in data.columns:
            return features

        close = data['Close']

        # Rate of Change (ROC)
        for period in [5, 10, 15, 20]:
            features[f'roc_{period}'] = (close - close.shift(period)) / close.shift(period)

        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = close - close.shift(period)

        # Price Oscillator
        features['price_osc'] = (close.ewm(span=12).mean() - close.ewm(span=26).mean()) / close

        return features

    def _create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """統計的特徴量"""

        features = pd.DataFrame(index=data.index)

        if 'Close' not in data.columns:
            return features

        close = data['Close']
        returns = close.pct_change()

        # 統計的モーメント
        for period in [10, 20]:
            features[f'skewness_{period}'] = returns.rolling(period).skew()
            features[f'kurtosis_{period}'] = returns.rolling(period).kurt()

        # エントロピー
        for period in [10, 20]:
            price_bins = pd.cut(close.rolling(period).apply(lambda x: x.iloc[-1]), bins=5, labels=False)
            features[f'entropy_{period}'] = price_bins

        return features

    def _create_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """パターン認識特徴量"""

        features = pd.DataFrame(index=data.index)

        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            return features

        # ローソク足パターン
        body = data['Close'] - data['Open']
        upper_shadow = data['High'] - np.maximum(data['Open'], data['Close'])
        lower_shadow = np.minimum(data['Open'], data['Close']) - data['Low']

        # Doji
        features['is_doji'] = (abs(body) / (data['High'] - data['Low'] + 1e-10) < 0.1).astype(int)

        # Hammer/Hanging Man
        features['is_hammer'] = ((lower_shadow > 2 * abs(body)) & (upper_shadow < abs(body))).astype(int)

        # Gap
        features['gap_up'] = (data['Open'] > data['High'].shift()).astype(int)
        features['gap_down'] = (data['Open'] < data['Low'].shift()).astype(int)

        return features

class EnhancedModelEnsemble:
    """改良アンサンブルモデル"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}

        # モデル設定
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaler': False
            },
            'xgboost': {
                'model': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='mlogloss'
                ),
                'use_scaler': True
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ),
                'use_scaler': True
            },
            'logistic_regression': {
                'model': LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    multi_class='ovr'
                ),
                'use_scaler': True
            },
            'svm': {
                'model': SVC(
                    probability=True,
                    random_state=42
                ),
                'use_scaler': True
            }
        }

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """モデル学習"""

        training_results = {}

        # ターゲット変数の変換（3クラス分類：-1, 0, 1）
        y_classes = self._create_target_classes(y)

        # 時系列分割
        tscv = TimeSeriesSplit(n_splits=3)

        for model_name, config in self.model_configs.items():
            try:
                model = config['model']
                use_scaler = config['use_scaler']

                # データ前処理
                X_processed = X.copy()
                if use_scaler:
                    scaler = RobustScaler()
                    X_processed = pd.DataFrame(
                        scaler.fit_transform(X_processed),
                        columns=X_processed.columns,
                        index=X_processed.index
                    )
                    self.scalers[model_name] = scaler

                # クロスバリデーション
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_processed):
                    X_train_cv, X_val_cv = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
                    y_train_cv, y_val_cv = y_classes.iloc[train_idx], y_classes.iloc[val_idx]

                    model_cv = model.__class__(**model.get_params())
                    model_cv.fit(X_train_cv, y_train_cv)
                    y_pred_cv = model_cv.predict(X_val_cv)
                    cv_scores.append(accuracy_score(y_val_cv, y_pred_cv))

                # 全データで最終学習
                model.fit(X_processed, y_classes)
                self.models[model_name] = model

                # 特徴量重要度
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(X.columns, model.feature_importances_))
                    self.feature_importance[model_name] = importance

                # 性能メトリクス
                y_pred = model.predict(X_processed)
                metrics = ModelPerformanceMetrics(
                    model_name=model_name,
                    accuracy=accuracy_score(y_classes, y_pred),
                    precision=precision_score(y_classes, y_pred, average='weighted', zero_division=0),
                    recall=recall_score(y_classes, y_pred, average='weighted', zero_division=0),
                    f1_score=f1_score(y_classes, y_pred, average='weighted', zero_division=0),
                    training_samples=len(X_processed),
                    feature_count=len(X.columns),
                    last_updated=datetime.now()
                )
                self.performance_metrics[model_name] = metrics

                training_results[model_name] = {
                    'cv_accuracy': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'final_accuracy': metrics.accuracy,
                    'feature_count': len(X.columns)
                }

                self.logger.info(f"{model_name} 学習完了: CV精度={np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}")

            except Exception as e:
                self.logger.error(f"{model_name} 学習エラー: {e}")
                training_results[model_name] = {'error': str(e)}

        return training_results

    def predict(self, X: pd.DataFrame) -> EnhancedPrediction:
        """アンサンブル予測"""

        if not self.models:
            raise ValueError("モデルが学習されていません")

        predictions = {}
        probabilities = {}

        # 各モデルで予測
        for model_name, model in self.models.items():
            try:
                X_processed = X.copy()

                # スケーリング
                if model_name in self.scalers:
                    scaler = self.scalers[model_name]
                    X_processed = pd.DataFrame(
                        scaler.transform(X_processed),
                        columns=X_processed.columns,
                        index=X_processed.index
                    )

                # 予測実行
                pred = model.predict(X_processed.iloc[-1:].values)[0]
                prob = model.predict_proba(X_processed.iloc[-1:].values)[0]

                predictions[model_name] = pred
                probabilities[model_name] = prob

            except Exception as e:
                self.logger.error(f"{model_name} 予測エラー: {e}")
                predictions[model_name] = 0
                probabilities[model_name] = np.array([0.33, 0.34, 0.33])

        # アンサンブル予測
        final_prediction, final_confidence, prob_dist = self._ensemble_prediction(predictions, probabilities)

        # 特徴量重要度（平均）
        avg_importance = {}
        if self.feature_importance:
            all_features = set()
            for importance_dict in self.feature_importance.values():
                all_features.update(importance_dict.keys())

            for feature in all_features:
                importance_values = [
                    importance_dict.get(feature, 0)
                    for importance_dict in self.feature_importance.values()
                ]
                avg_importance[feature] = np.mean(importance_values)

        return EnhancedPrediction(
            symbol=X.index[-1] if hasattr(X.index[-1], 'strftime') else "unknown",
            prediction=final_prediction,
            confidence=final_confidence,
            probability_distribution=prob_dist,
            model_consensus=predictions,
            feature_importance=dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]),
            timestamp=datetime.now(),
            data_quality_score=95.0  # デフォルト値
        )

    def _create_target_classes(self, returns: pd.Series, threshold: float = 0.01) -> pd.Series:
        """ターゲットクラス作成"""

        future_returns = returns.shift(-1)  # 翌日のリターン

        conditions = [
            future_returns < -threshold,  # 下落（-1）
            future_returns > threshold,   # 上昇（1）
        ]
        choices = [-1, 1]

        y_classes = np.select(conditions, choices, default=0)  # 横ばい（0）

        return pd.Series(y_classes, index=returns.index)

    def _ensemble_prediction(self, predictions: Dict[str, int],
                           probabilities: Dict[str, np.ndarray]) -> Tuple[int, float, Dict[int, float]]:
        """アンサンブル予測"""

        if not predictions:
            return 0, 0.0, {-1: 0.33, 0: 0.34, 1: 0.33}

        # 重み（性能に基づく）
        weights = {}
        for model_name in predictions.keys():
            if model_name in self.performance_metrics:
                weights[model_name] = self.performance_metrics[model_name].accuracy
            else:
                weights[model_name] = 0.5

        # 重みの正規化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            weights = {k: 1/len(predictions) for k in predictions.keys()}

        # 重み付き投票
        weighted_probs = np.zeros(3)  # [-1, 0, 1]に対応

        for model_name, prob in probabilities.items():
            if model_name in weights:
                weighted_probs += prob * weights[model_name]

        # 最終予測
        class_mapping = {0: -1, 1: 0, 2: 1}  # インデックス -> クラス
        final_class_idx = np.argmax(weighted_probs)
        final_prediction = class_mapping[final_class_idx]
        final_confidence = weighted_probs[final_class_idx]

        prob_distribution = {
            -1: weighted_probs[0],
            0: weighted_probs[1],
            1: weighted_probs[2]
        }

        return final_prediction, final_confidence, prob_distribution

class EnhancedPredictionSystem:
    """改良予測システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_engineer = AdvancedFeatureEngineering()
        self.model_ensemble = EnhancedModelEnsemble()

        # データベース設定
        self.db_path = Path("enhanced_prediction_data/models_and_performance.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

        self.logger.info("Enhanced prediction system initialized")

    def _init_database(self):
        """データベース初期化"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        accuracy REAL NOT NULL,
                        precision_score REAL NOT NULL,
                        recall_score REAL NOT NULL,
                        f1_score REAL NOT NULL,
                        training_samples INTEGER NOT NULL,
                        feature_count INTEGER NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        prediction INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        data_quality_score REAL,
                        actual_result INTEGER
                    )
                ''')

                conn.commit()

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")

    async def train_enhanced_models(self, symbol: str) -> Dict[str, Any]:
        """改良モデル学習"""

        try:
            # より多くの学習データ取得（6ヶ月）
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, "6mo")

            if data is None or len(data) < 100:
                return {"error": f"学習に十分なデータがありません: {len(data) if data else 0}件"}

            # データ品質評価
            from data_quality_manager import data_quality_manager
            quality_result = await data_quality_manager.evaluate_data_quality(symbol)
            data_quality_score = quality_result.get('overall_score', 50)

            if data_quality_score < 70:
                self.logger.warning(f"データ品質が低い: {data_quality_score:.1f}/100")

            # 改良特徴量作成
            features = self.feature_engineer.create_comprehensive_features(data)

            # ターゲット変数作成
            returns = data['Close'].pct_change().dropna()

            # データ整列
            common_index = features.index.intersection(returns.index)
            X = features.loc[common_index].dropna()
            y = returns.loc[X.index]

            if len(X) < 50:
                return {"error": f"学習サンプル不足: {len(X)}件（最低50件必要）"}

            # モデル学習
            training_results = self.model_ensemble.train_models(X, y)

            # 結果保存
            await self._save_performance_metrics(symbol, training_results)

            # 学習結果サマリー
            avg_accuracy = np.mean([
                result.get('final_accuracy', 0)
                for result in training_results.values()
                if 'error' not in result
            ])

            return {
                "symbol": symbol,
                "training_samples": len(X),
                "feature_count": len(X.columns),
                "data_quality_score": data_quality_score,
                "avg_model_accuracy": avg_accuracy,
                "model_results": training_results,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"改良モデル学習エラー {symbol}: {e}")
            return {"error": str(e)}

    async def predict_with_enhanced_models(self, symbol: str) -> Optional[EnhancedPrediction]:
        """改良予測実行"""

        try:
            # データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, "6mo")

            if data is None or len(data) < 50:
                self.logger.error(f"予測用データ不足: {len(data) if data else 0}件")
                return None

            # データ品質評価
            from data_quality_manager import data_quality_manager
            quality_result = await data_quality_manager.evaluate_data_quality(symbol)
            data_quality_score = quality_result.get('overall_score', 50)

            # 特徴量作成
            features = self.feature_engineer.create_comprehensive_features(data)

            if len(features.columns) == 0:
                self.logger.error("特徴量が作成されませんでした")
                return None

            # 予測実行
            prediction = self.model_ensemble.predict(features)
            prediction.symbol = symbol
            prediction.data_quality_score = data_quality_score

            # 予測ログ保存
            await self._save_prediction_log(prediction)

            self.logger.info(f"改良予測完了: {symbol}, 予測={prediction.prediction}, 信頼度={prediction.confidence:.3f}")

            return prediction

        except Exception as e:
            self.logger.error(f"改良予測エラー {symbol}: {e}")
            return None

    async def _save_performance_metrics(self, symbol: str, training_results: Dict[str, Any]):
        """性能メトリクス保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                timestamp = datetime.now().isoformat()

                for model_name, model_metrics in self.model_ensemble.performance_metrics.items():
                    cursor.execute('''
                        INSERT INTO model_performance
                        (symbol, model_name, accuracy, precision_score, recall_score,
                         f1_score, training_samples, feature_count, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        model_name,
                        model_metrics.accuracy,
                        model_metrics.precision,
                        model_metrics.recall,
                        model_metrics.f1_score,
                        model_metrics.training_samples,
                        model_metrics.feature_count,
                        timestamp
                    ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"性能メトリクス保存エラー: {e}")

    async def _save_prediction_log(self, prediction: EnhancedPrediction):
        """予測ログ保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO predictions_log
                    (symbol, prediction, confidence, timestamp, data_quality_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    prediction.symbol,
                    prediction.prediction,
                    prediction.confidence,
                    prediction.timestamp.isoformat(),
                    prediction.data_quality_score
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"予測ログ保存エラー: {e}")

# グローバルインスタンス
enhanced_prediction_system = EnhancedPredictionSystem()

# テスト実行
async def run_enhanced_prediction_test():
    """改良予測システムテスト"""

    print("=== 🚀 改良予測システムテスト ===")

    test_symbols = ["7203", "8306"]

    for symbol in test_symbols:
        print(f"\n--- {symbol} 改良予測テスト ---")

        # モデル学習
        print("モデル学習中...")
        training_result = await enhanced_prediction_system.train_enhanced_models(symbol)

        if 'error' in training_result:
            print(f"❌ 学習失敗: {training_result['error']}")
            continue

        print(f"✅ 学習完了:")
        print(f"  学習サンプル数: {training_result['training_samples']}")
        print(f"  特徴量数: {training_result['feature_count']}")
        print(f"  データ品質: {training_result['data_quality_score']:.1f}/100")
        print(f"  平均精度: {training_result['avg_model_accuracy']:.1%}")

        # 予測実行
        print("予測実行中...")
        prediction = await enhanced_prediction_system.predict_with_enhanced_models(symbol)

        if prediction:
            print(f"✅ 予測完了:")
            print(f"  予測: {prediction.prediction} ({'上昇' if prediction.prediction > 0 else '下落' if prediction.prediction < 0 else '横ばい'})")
            print(f"  信頼度: {prediction.confidence:.1%}")
            print(f"  データ品質スコア: {prediction.data_quality_score:.1f}/100")

            # 確率分布表示
            print(f"  確率分布:")
            for direction, prob in prediction.probability_distribution.items():
                direction_name = {-1: '下落', 0: '横ばい', 1: '上昇'}[direction]
                print(f"    {direction_name}: {prob:.1%}")

            # モデル合意表示
            print(f"  モデル合意:")
            for model, pred in prediction.model_consensus.items():
                pred_name = {-1: '下落', 0: '横ばい', 1: '上昇'}[pred]
                print(f"    {model}: {pred_name}")
        else:
            print(f"❌ 予測失敗")

    print(f"\n✅ 改良予測システムテスト完了")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(run_enhanced_prediction_test())