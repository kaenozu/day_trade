#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Model Data Integration System - AIモデル・データ統合システム

Issue #801実装：AIモデルとデータ統合の実施
複数のAIモデルとデータソースの統合による高度予測システム
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
import warnings
warnings.filterwarnings('ignore')

# 機械学習ライブラリ
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib

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

class ModelType(Enum):
    """モデルタイプ"""
    TREND_FOLLOWING = "trend_following"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    ENSEMBLE = "ensemble"

class DataSource(Enum):
    """データソース"""
    PRICE_DATA = "price_data"
    VOLUME_DATA = "volume_data"
    TECHNICAL_INDICATORS = "technical_indicators"
    MARKET_SENTIMENT = "market_sentiment"
    ECONOMIC_DATA = "economic_data"
    NEWS_DATA = "news_data"

@dataclass
class ModelPrediction:
    """モデル予測結果"""
    model_type: ModelType
    symbol: str
    prediction: int  # 0: 下降, 1: 上昇
    confidence: float
    feature_importance: Dict[str, float]
    prediction_details: Dict[str, Any]
    timestamp: datetime

@dataclass
class IntegratedPrediction:
    """統合予測結果"""
    symbol: str
    final_prediction: int
    confidence: float
    model_consensus: Dict[str, Dict[str, Any]]
    feature_contributions: Dict[str, float]
    risk_assessment: Dict[str, float]
    explanation: List[str]
    timestamp: datetime

class AdvancedFeatureEngineering:
    """高度特徴量エンジニアリング"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {}

    def create_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """包括的特徴量作成"""

        if len(data) < 50:
            raise ValueError("データが不足しています（最低50データポイント必要）")

        features = pd.DataFrame(index=data.index)

        # 基本価格データ
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        open_price = data['Open']

        try:
            # 1. 価格系特徴量
            features = self._add_price_features(features, open_price, high, low, close)

            # 2. ボリューム系特徴量
            features = self._add_volume_features(features, volume, close)

            # 3. 技術指標特徴量
            features = self._add_technical_features(features, high, low, close, volume)

            # 4. 統計的特徴量
            features = self._add_statistical_features(features, close)

            # 5. トレンド特徴量
            features = self._add_trend_features(features, close)

            # 6. ボラティリティ特徴量
            features = self._add_volatility_features(features, high, low, close)

            # 7. 時間系特徴量
            features = self._add_temporal_features(features, data.index)

            # データクリーニング
            features = self._clean_features(features)

            self.logger.info(f"包括特徴量作成完了: {features.shape[1]}特徴量, {features.shape[0]}サンプル")

        except Exception as e:
            self.logger.error(f"特徴量作成エラー: {e}")
            raise

        return features

    def _add_price_features(self, features: pd.DataFrame, open_price: pd.Series,
                          high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
        """価格系特徴量追加"""

        # 基本価格特徴量
        features['returns'] = close.pct_change()
        features['log_returns'] = np.log(close / close.shift(1))
        features['price_range'] = (high - low) / close
        features['gap'] = (open_price - close.shift(1)) / close.shift(1)
        features['body_size'] = abs(close - open_price) / close

        # 価格位置特徴量
        features['high_close_ratio'] = (high - close) / (high - low + 1e-10)
        features['low_close_ratio'] = (close - low) / (high - low + 1e-10)
        features['open_close_ratio'] = (close - open_price) / (high - low + 1e-10)

        # 複数期間のリターン
        for period in [2, 3, 5, 10]:
            if len(close) > period:
                features[f'return_{period}d'] = close.pct_change(period)
                features[f'return_vol_{period}d'] = features['returns'].rolling(period).std()

        return features

    def _add_volume_features(self, features: pd.DataFrame, volume: pd.Series, close: pd.Series) -> pd.DataFrame:
        """ボリューム系特徴量追加"""

        # ボリューム基本統計
        for period in [5, 10, 20]:
            if len(volume) > period:
                vol_ma = volume.rolling(period).mean()
                features[f'volume_ratio_{period}'] = volume / vol_ma
                features[f'volume_zscore_{period}'] = (volume - vol_ma) / volume.rolling(period).std()

        # 価格・ボリューム関係
        features['price_volume'] = close * volume
        features['volume_price_trend'] = (features['returns'] * (volume / volume.rolling(10).mean())).rolling(5).mean()

        # On-Balance Volume
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        features['obv'] = obv
        features['obv_trend'] = obv.pct_change(5)

        return features

    def _add_technical_features(self, features: pd.DataFrame, high: pd.Series,
                              low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.DataFrame:
        """技術指標特徴量追加"""

        # 移動平均系
        for period in [5, 10, 20, 50]:
            if len(close) > period:
                sma = close.rolling(period).mean()
                ema = close.ewm(span=period).mean()

                features[f'sma_{period}'] = sma
                features[f'sma_distance_{period}'] = (close - sma) / sma
                features[f'ema_distance_{period}'] = (close - ema) / ema
                features[f'price_position_{period}'] = (close - sma) / close.rolling(period).std()

        # MACD
        if len(close) > 26:
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd - macd_signal
            features['macd_trend'] = macd.pct_change(3)

        # RSI
        for period in [9, 14, 21]:
            if len(close) > period:
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / (loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                features[f'rsi_{period}'] = rsi
                features[f'rsi_trend_{period}'] = rsi.pct_change(3)

        # ボリンジャーバンド
        if len(close) > 20:
            sma_20 = close.rolling(20).mean()
            std_20 = close.rolling(20).std()
            bb_upper = sma_20 + (std_20 * 2)
            bb_lower = sma_20 - (std_20 * 2)

            features['bb_width'] = (bb_upper - bb_lower) / sma_20
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
            features['bb_squeeze'] = features['bb_width'] < features['bb_width'].rolling(10).mean()

        # ATR
        for period in [10, 14, 20]:
            if len(high) > period:
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = true_range.rolling(period).mean()
                features[f'atr_{period}'] = atr
                features[f'atr_ratio_{period}'] = atr / close

        # Stochastic
        if len(high) > 14:
            period = 14
            lowest_low = low.rolling(period).min()
            highest_high = high.rolling(period).max()
            k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
            d_percent = k_percent.rolling(3).mean()
            features['stoch_k'] = k_percent
            features['stoch_d'] = d_percent
            features['stoch_divergence'] = k_percent - d_percent

        return features

    def _add_statistical_features(self, features: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """統計的特徴量追加"""

        returns = close.pct_change()

        for period in [5, 10, 20]:
            if len(returns) > period:
                ret_window = returns.rolling(period)
                features[f'skewness_{period}'] = ret_window.skew()
                features[f'kurtosis_{period}'] = ret_window.kurt()
                features[f'var_{period}'] = ret_window.var()
                features[f'sharpe_{period}'] = ret_window.mean() / (ret_window.std() + 1e-10) * np.sqrt(252)

        # 価格分布特徴量
        for period in [10, 20]:
            if len(close) > period:
                close_window = close.rolling(period)
                features[f'price_percentile_{period}'] = close.rolling(period).rank(pct=True)
                features[f'price_zscore_{period}'] = (close - close_window.mean()) / close_window.std()

        return features

    def _add_trend_features(self, features: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """トレンド特徴量追加"""

        # 単純トレンド
        for period in [5, 10, 20]:
            if len(close) > period:
                features[f'trend_strength_{period}'] = (close > close.shift(period)).rolling(5).mean()
                features[f'trend_consistency_{period}'] = (close.diff() > 0).rolling(period).mean()

        # 線形回帰トレンド
        for period in [10, 20]:
            if len(close) > period:
                def calc_trend_slope(x):
                    if len(x) < 3:
                        return 0
                    y = np.arange(len(x))
                    slope = np.polyfit(y, x, 1)[0]
                    return slope

                features[f'trend_slope_{period}'] = close.rolling(period).apply(calc_trend_slope, raw=True)

        # ADX（トレンド強度）
        if len(close) > 14:
            high_diff = close.shift(1).rolling(2).max() - close.shift(1)
            low_diff = close.shift(1) - close.shift(1).rolling(2).min()

            plus_dm = high_diff.where(high_diff > low_diff, 0)
            minus_dm = low_diff.where(low_diff > high_diff, 0)

            # 簡易版ADX
            dm_sum = plus_dm.rolling(14).sum() + minus_dm.rolling(14).sum()
            features['trend_intensity'] = dm_sum / close * 100

        return features

    def _add_volatility_features(self, features: pd.DataFrame, high: pd.Series,
                               low: pd.Series, close: pd.Series) -> pd.DataFrame:
        """ボラティリティ特徴量追加"""

        returns = close.pct_change()

        # 実現ボラティリティ
        for period in [5, 10, 20]:
            if len(returns) > period:
                vol = returns.rolling(period).std() * np.sqrt(252)
                features[f'volatility_{period}'] = vol
                features[f'vol_rank_{period}'] = vol.rolling(50).rank(pct=True)

        # Parkinson推定量（高値・安値を使用）
        for period in [10, 20]:
            if len(high) > period:
                parkinson = np.log(high / low) ** 2
                features[f'parkinson_vol_{period}'] = parkinson.rolling(period).mean()

        # Garman-Klass推定量
        if len(high) > 10:
            gk = 0.5 * (np.log(high / low)) ** 2 - (2 * np.log(2) - 1) * (np.log(close / close.shift(1))) ** 2
            features['garman_klass_vol'] = gk.rolling(10).mean()

        # ボラティリティクラスター
        vol_5 = returns.rolling(5).std()
        features['vol_cluster'] = (vol_5 > vol_5.rolling(20).mean()).astype(int)

        return features

    def _add_temporal_features(self, features: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        """時間系特徴量追加"""

        if isinstance(index, pd.DatetimeIndex):
            features['day_of_week'] = index.dayofweek
            features['month'] = index.month
            features['quarter'] = index.quarter
            features['is_month_end'] = index.is_month_end.astype(int)
            features['is_quarter_end'] = index.is_quarter_end.astype(int)

        # 季節性
        features['seasonal_trend'] = np.sin(2 * np.pi * np.arange(len(features)) / 252)  # 年次季節性
        features['monthly_cycle'] = np.sin(2 * np.pi * np.arange(len(features)) / 21)   # 月次サイクル

        return features

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """特徴量クリーニング"""

        # 無限大値処理
        features = features.replace([np.inf, -np.inf], np.nan)

        # 欠損値処理
        features = features.fillna(method='ffill').fillna(method='bfill')
        features = features.fillna(0)

        # 異常値処理（3シグマルール）
        for col in features.select_dtypes(include=[np.number]).columns:
            mean = features[col].mean()
            std = features[col].std()
            if std > 0:
                features[col] = features[col].clip(mean - 3*std, mean + 3*std)

        return features

class SpecializedAIModels:
    """専門AIモデル群"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}

    def create_trend_model(self) -> RandomForestClassifier:
        """トレンド追従モデル"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced'
        )

    def create_momentum_model(self) -> RandomForestClassifier:
        """モメンタムモデル"""
        return RandomForestClassifier(
            n_estimators=80,
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=43,
            class_weight='balanced'
        )

    def create_volatility_model(self) -> RandomForestClassifier:
        """ボラティリティモデル"""
        return RandomForestClassifier(
            n_estimators=60,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=44,
            class_weight='balanced'
        )

    async def train_specialized_models(self, symbol: str, features: pd.DataFrame,
                                     targets: pd.Series) -> Dict[ModelType, Any]:
        """専門モデル群訓練"""

        if len(features) < 30:
            raise ValueError("訓練データが不足しています")

        trained_models = {}

        # 特徴量選択とスケーリング
        scaler = StandardScaler()
        features_scaled = pd.DataFrame(
            scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )

        self.scalers[symbol] = scaler

        # トレンド追従モデル
        trend_features = self._select_trend_features(features_scaled)
        trend_model = self.create_trend_model()
        trend_model.fit(trend_features, targets)
        trained_models[ModelType.TREND_FOLLOWING] = {
            'model': trend_model,
            'features': trend_features.columns.tolist(),
            'accuracy': self._calculate_accuracy(trend_model, trend_features, targets)
        }

        # モメンタムモデル
        momentum_features = self._select_momentum_features(features_scaled)
        momentum_model = self.create_momentum_model()
        momentum_model.fit(momentum_features, targets)
        trained_models[ModelType.MOMENTUM] = {
            'model': momentum_model,
            'features': momentum_features.columns.tolist(),
            'accuracy': self._calculate_accuracy(momentum_model, momentum_features, targets)
        }

        # ボラティリティモデル
        vol_features = self._select_volatility_features(features_scaled)
        vol_model = self.create_volatility_model()
        vol_model.fit(vol_features, targets)
        trained_models[ModelType.VOLATILITY] = {
            'model': vol_model,
            'features': vol_features.columns.tolist(),
            'accuracy': self._calculate_accuracy(vol_model, vol_features, targets)
        }

        # アンサンブルモデル
        ensemble_model = VotingClassifier([
            ('trend', trend_model),
            ('momentum', momentum_model),
            ('volatility', vol_model)
        ], voting='soft')

        ensemble_model.fit(features_scaled, targets)
        trained_models[ModelType.ENSEMBLE] = {
            'model': ensemble_model,
            'features': features_scaled.columns.tolist(),
            'accuracy': self._calculate_accuracy(ensemble_model, features_scaled, targets)
        }

        self.models[symbol] = trained_models

        return trained_models

    def _select_trend_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """トレンド関連特徴量選択"""
        trend_columns = [col for col in features.columns if any(
            keyword in col.lower() for keyword in [
                'sma', 'ema', 'trend', 'macd', 'adx', 'slope'
            ]
        )]
        return features[trend_columns[:20]]  # 上位20特徴量

    def _select_momentum_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """モメンタム関連特徴量選択"""
        momentum_columns = [col for col in features.columns if any(
            keyword in col.lower() for keyword in [
                'rsi', 'stoch', 'momentum', 'roc', 'return'
            ]
        )]
        return features[momentum_columns[:20]]

    def _select_volatility_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ関連特徴量選択"""
        vol_columns = [col for col in features.columns if any(
            keyword in col.lower() for keyword in [
                'vol', 'atr', 'bb', 'range', 'parkinson', 'garman'
            ]
        )]
        return features[vol_columns[:20]]

    def _calculate_accuracy(self, model, features: pd.DataFrame, targets: pd.Series) -> float:
        """モデル精度計算"""
        try:
            cv_scores = cross_val_score(model, features, targets, cv=3, scoring='accuracy')
            return cv_scores.mean()
        except:
            return 0.5

    async def predict_with_models(self, symbol: str, current_features: pd.DataFrame) -> Dict[ModelType, ModelPrediction]:
        """モデル群で予測"""

        if symbol not in self.models:
            raise ValueError(f"モデルが訓練されていません: {symbol}")

        predictions = {}
        scaler = self.scalers.get(symbol)

        if scaler is None:
            raise ValueError(f"スケーラーが見つかりません: {symbol}")

        # 特徴量スケーリング
        features_scaled = pd.DataFrame(
            scaler.transform(current_features),
            columns=current_features.columns,
            index=current_features.index
        )

        for model_type, model_info in self.models[symbol].items():
            try:
                model = model_info['model']
                required_features = model_info['features']

                # 必要な特徴量を選択
                X = features_scaled[required_features]

                # 予測実行
                prediction = model.predict(X.iloc[-1:].values)[0]
                confidence = model.predict_proba(X.iloc[-1:].values)[0].max()

                # 特徴量重要度
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(required_features, model.feature_importances_))
                else:
                    importance = {}

                predictions[model_type] = ModelPrediction(
                    model_type=model_type,
                    symbol=symbol,
                    prediction=int(prediction),
                    confidence=float(confidence),
                    feature_importance=importance,
                    prediction_details={
                        'accuracy': model_info['accuracy'],
                        'feature_count': len(required_features)
                    },
                    timestamp=datetime.now()
                )

            except Exception as e:
                self.logger.error(f"モデル予測エラー {model_type}: {e}")
                continue

        return predictions

class AIModelDataIntegration:
    """AIモデル・データ統合システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # コンポーネント初期化
        self.feature_engineering = AdvancedFeatureEngineering()
        self.ai_models = SpecializedAIModels()

        # データベース設定
        self.db_path = Path("ai_integration_data/integrated_predictions.db")
        self.db_path.parent.mkdir(exist_ok=True)

        self._init_database()
        self.logger.info("AI model data integration system initialized")

    def _init_database(self):
        """データベース初期化"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 統合予測テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS integrated_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        final_prediction INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        model_consensus TEXT,
                        feature_contributions TEXT,
                        risk_assessment TEXT,
                        explanation TEXT,
                        timestamp TEXT NOT NULL
                    )
                ''')

                # モデル性能テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        accuracy REAL,
                        training_samples INTEGER,
                        feature_count INTEGER,
                        timestamp TEXT NOT NULL
                    )
                ''')

                conn.commit()

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")

    async def train_integrated_system(self, symbol: str, period: str = "6mo") -> Dict[str, Any]:
        """統合システム訓練"""

        self.logger.info(f"統合システム訓練開始: {symbol}")

        try:
            # データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, period)

            if data is None or len(data) < 100:
                raise ValueError("十分な訓練データがありません")

            # 包括的特徴量エンジニアリング
            features = self.feature_engineering.create_comprehensive_features(data)

            # ターゲット作成
            targets = self._create_targets(data['Close'].iloc[-len(features):])

            # データ同期
            min_length = min(len(features), len(targets))
            features = features.iloc[:min_length]
            targets = targets[:min_length]

            if len(features) < 50:
                raise ValueError("訓練に十分なデータがありません")

            # 専門モデル群訓練
            trained_models = await self.ai_models.train_specialized_models(symbol, features, targets)

            # 訓練結果記録
            await self._save_model_performance(symbol, trained_models, len(targets))

            self.logger.info(f"統合システム訓練完了: {len(trained_models)}モデル")

            return {
                'symbol': symbol,
                'models_trained': len(trained_models),
                'training_samples': len(targets),
                'feature_count': len(features.columns),
                'model_accuracies': {
                    str(model_type): model_info['accuracy']
                    for model_type, model_info in trained_models.items()
                }
            }

        except Exception as e:
            self.logger.error(f"統合システム訓練エラー: {e}")
            raise

    def _create_targets(self, prices: pd.Series) -> np.ndarray:
        """ターゲット作成"""
        # 翌日の価格変動
        returns = prices.pct_change().shift(-1)

        # 閾値以上の上昇を1、それ以外を0
        threshold = 0.001  # 0.1%以上の変動
        targets = (returns > threshold).astype(int)

        return targets.values[:-1]  # 最後の要素（NaN）を除去

    async def predict_integrated(self, symbol: str, force_retrain: bool = False) -> IntegratedPrediction:
        """統合予測実行"""

        try:
            # 必要に応じてモデル再訓練
            if force_retrain or symbol not in self.ai_models.models:
                await self.train_integrated_system(symbol)

            # 最新データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, "3mo")

            if data is None or len(data) < 50:
                raise ValueError("予測に十分なデータがありません")

            # 特徴量エンジニアリング
            features = self.feature_engineering.create_comprehensive_features(data)

            # 各専門モデルで予測
            model_predictions = await self.ai_models.predict_with_models(symbol, features)

            # 統合予測計算
            integrated_result = self._integrate_predictions(symbol, model_predictions)

            # 結果保存
            await self._save_integrated_prediction(integrated_result)

            return integrated_result

        except Exception as e:
            self.logger.error(f"統合予測エラー: {e}")
            # デフォルト予測を返す
            return IntegratedPrediction(
                symbol=symbol,
                final_prediction=1,
                confidence=0.5,
                model_consensus={},
                feature_contributions={},
                risk_assessment={'risk_level': 'medium'},
                explanation=['予測エラーが発生しました'],
                timestamp=datetime.now()
            )

    def _integrate_predictions(self, symbol: str, model_predictions: Dict[ModelType, ModelPrediction]) -> IntegratedPrediction:
        """予測統合"""

        if not model_predictions:
            return IntegratedPrediction(
                symbol=symbol,
                final_prediction=1,
                confidence=0.5,
                model_consensus={},
                feature_contributions={},
                risk_assessment={'risk_level': 'high'},
                explanation=['利用可能なモデル予測がありません'],
                timestamp=datetime.now()
            )

        # モデル重み（精度ベース）
        model_weights = {}
        total_accuracy = sum(pred.prediction_details.get('accuracy', 0.5) for pred in model_predictions.values())

        for model_type, prediction in model_predictions.items():
            accuracy = prediction.prediction_details.get('accuracy', 0.5)
            model_weights[model_type] = accuracy / total_accuracy if total_accuracy > 0 else 1.0 / len(model_predictions)

        # 重み付き投票
        weighted_sum = sum(
            prediction.prediction * model_weights[model_type]
            for model_type, prediction in model_predictions.items()
        )

        final_prediction = 1 if weighted_sum >= 0.5 else 0

        # 信頼度計算
        confidence_scores = [pred.confidence for pred in model_predictions.values()]
        final_confidence = np.mean(confidence_scores)

        # モデル合意度計算
        predictions_list = [pred.prediction for pred in model_predictions.values()]
        consensus_rate = predictions_list.count(final_prediction) / len(predictions_list)

        # 特徴量貢献度統合
        feature_contributions = {}
        for prediction in model_predictions.values():
            for feature, importance in prediction.feature_importance.items():
                if feature not in feature_contributions:
                    feature_contributions[feature] = 0
                feature_contributions[feature] += importance * model_weights[prediction.model_type]

        # リスク評価
        risk_assessment = self._assess_prediction_risk(
            final_confidence, consensus_rate, len(model_predictions)
        )

        # 説明生成
        explanation = self._generate_explanation(
            final_prediction, final_confidence, model_predictions, feature_contributions
        )

        # モデル合意情報
        model_consensus = {}
        for model_type, prediction in model_predictions.items():
            model_consensus[model_type.value] = {
                'prediction': prediction.prediction,
                'confidence': prediction.confidence,
                'weight': model_weights[model_type],
                'accuracy': prediction.prediction_details.get('accuracy', 0.5)
            }

        return IntegratedPrediction(
            symbol=symbol,
            final_prediction=final_prediction,
            confidence=final_confidence,
            model_consensus=model_consensus,
            feature_contributions=feature_contributions,
            risk_assessment=risk_assessment,
            explanation=explanation,
            timestamp=datetime.now()
        )

    def _assess_prediction_risk(self, confidence: float, consensus_rate: float, model_count: int) -> Dict[str, float]:
        """予測リスク評価"""

        # 信頼度リスク
        confidence_risk = 1.0 - confidence

        # 合意度リスク
        consensus_risk = 1.0 - consensus_rate

        # モデル数リスク
        model_diversity_risk = max(0, (5 - model_count) / 5)

        # 総合リスク
        overall_risk = (confidence_risk * 0.5 + consensus_risk * 0.3 + model_diversity_risk * 0.2)

        risk_level = "low" if overall_risk < 0.3 else "medium" if overall_risk < 0.6 else "high"

        return {
            'overall_risk': overall_risk,
            'confidence_risk': confidence_risk,
            'consensus_risk': consensus_risk,
            'model_diversity_risk': model_diversity_risk,
            'risk_level': risk_level
        }

    def _generate_explanation(self, final_prediction: int, confidence: float,
                            model_predictions: Dict[ModelType, ModelPrediction],
                            feature_contributions: Dict[str, float]) -> List[str]:
        """説明生成"""

        explanation = []

        # 基本予測情報
        direction = "上昇" if final_prediction == 1 else "下降"
        explanation.append(f"統合予測: {direction} (信頼度: {confidence:.1%})")

        # モデル合意情報
        agree_count = sum(1 for pred in model_predictions.values() if pred.prediction == final_prediction)
        total_count = len(model_predictions)
        explanation.append(f"モデル合意: {agree_count}/{total_count}モデルが{direction}予測")

        # 主要モデル情報
        best_model = max(model_predictions.values(), key=lambda x: x.confidence)
        explanation.append(f"最高信頼度モデル: {best_model.model_type.value} ({best_model.confidence:.1%})")

        # 主要特徴量
        if feature_contributions:
            top_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            feature_names = [f"{name}({contribution:.3f})" for name, contribution in top_features]
            explanation.append(f"主要要因: {', '.join(feature_names)}")

        # リスク警告
        if confidence < 0.6:
            explanation.append("⚠️ 低信頼度予測のため注意が必要")

        return explanation

    async def _save_model_performance(self, symbol: str, trained_models: Dict[ModelType, Any], sample_count: int):
        """モデル性能保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for model_type, model_info in trained_models.items():
                    cursor.execute('''
                        INSERT INTO model_performance
                        (symbol, model_type, accuracy, training_samples, feature_count, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        model_type.value,
                        model_info['accuracy'],
                        sample_count,
                        len(model_info['features']),
                        datetime.now().isoformat()
                    ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"モデル性能保存エラー: {e}")

    async def _save_integrated_prediction(self, prediction: IntegratedPrediction):
        """統合予測保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO integrated_predictions
                    (symbol, final_prediction, confidence, model_consensus,
                     feature_contributions, risk_assessment, explanation, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction.symbol,
                    prediction.final_prediction,
                    prediction.confidence,
                    json.dumps(prediction.model_consensus),
                    json.dumps(prediction.feature_contributions),
                    json.dumps(prediction.risk_assessment),
                    json.dumps(prediction.explanation),
                    prediction.timestamp.isoformat()
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"統合予測保存エラー: {e}")

# グローバルインスタンス
ai_model_data_integration = AIModelDataIntegration()

# テスト実行
async def run_ai_integration_test():
    """AI統合システムテスト実行"""

    print("=== 🤖 AIモデル・データ統合システムテスト ===")

    test_symbols = ["7203", "8306"]

    for symbol in test_symbols:
        print(f"\n🔬 {symbol} 統合システム訓練・テスト")

        try:
            # 統合システム訓練
            training_result = await ai_model_data_integration.train_integrated_system(symbol)

            print(f"  ✅ 訓練完了:")
            print(f"    訓練モデル数: {training_result['models_trained']}")
            print(f"    訓練サンプル数: {training_result['training_samples']}")
            print(f"    特徴量数: {training_result['feature_count']}")

            # モデル精度表示
            for model_type, accuracy in training_result['model_accuracies'].items():
                print(f"    {model_type}: {accuracy:.3f}")

            # 統合予測テスト
            prediction = await ai_model_data_integration.predict_integrated(symbol)

            print(f"  🎯 統合予測結果:")
            print(f"    予測: {'上昇' if prediction.final_prediction else '下降'}")
            print(f"    信頼度: {prediction.confidence:.1%}")
            print(f"    リスクレベル: {prediction.risk_assessment['risk_level']}")
            print(f"    モデル数: {len(prediction.model_consensus)}")

            # 説明表示
            print(f"    説明:")
            for explanation in prediction.explanation[:3]:
                print(f"      - {explanation}")

        except Exception as e:
            print(f"  ❌ {symbol} エラー: {e}")

    print(f"\n✅ AI統合システムテスト完了")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(run_ai_integration_test())