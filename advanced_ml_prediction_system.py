#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced ML Prediction System - 高度機械学習予測システム

予測精度改善のための包括的システム
Issue #800-2-1実装：予測精度改善計画
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
import joblib # Added for model saving/loading

# 機械学習ライブラリ
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 技術指標ライブラリ
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARNING: talib not available - using fallback technical indicators")

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
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE_VOTING = "ensemble_voting"

class ScalerType(Enum):
    """スケーラータイプ"""
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    NONE = "none"

class FeatureSelectionType(Enum):
    """特徴選択タイプ"""
    K_BEST_F = "k_best_f"
    K_BEST_MUTUAL = "k_best_mutual"
    NONE = "none"

@dataclass
class ModelConfig:
    """モデル設定"""
    model_type: ModelType
    scaler_type: ScalerType
    feature_selection: FeatureSelectionType
    feature_selection_k: int = 50
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelPerformance:
    """モデル性能"""
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cross_val_score: float
    feature_importance: Dict[str, float] = field(default_factory=dict)

@dataclass
class PredictionResult:
    """予測結果"""
    symbol: str
    prediction: int  # 0: 下降, 1: 上昇
    confidence: float
    model_consensus: Dict[str, int]  # 各モデルの予測
    feature_values: Dict[str, float]
    timestamp: datetime

class AdvancedFeatureEngineering:
    """高度特徴量エンジニアリング"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """高度特徴量作成"""

        if len(data) < 100:
            raise ValueError("データが不足しています（最低100データポイント必要）")

        features = pd.DataFrame(index=data.index)

        # 基本価格データ
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        volume = data['Volume'].values
        open_price = data['Open'].values

        try:
            # 1. トレンド指標（16種類）
            features['sma_5'] = talib.SMA(close, timeperiod=5)
            features['sma_10'] = talib.SMA(close, timeperiod=10)
            features['sma_20'] = talib.SMA(close, timeperiod=20)
            features['sma_50'] = talib.SMA(close, timeperiod=50)
            features['ema_5'] = talib.EMA(close, timeperiod=5)
            features['ema_10'] = talib.EMA(close, timeperiod=10)
            features['ema_20'] = talib.EMA(close, timeperiod=20)
            features['wma_10'] = talib.WMA(close, timeperiod=10)
            features['tema_10'] = talib.TEMA(close, timeperiod=10)
            features['dema_10'] = talib.DEMA(close, timeperiod=10)
            features['kama_10'] = talib.KAMA(close, timeperiod=10)
            features['trima_10'] = talib.TRIMA(close, timeperiod=10)
            features['t3_10'] = talib.T3(close, timeperiod=10)
            features['mama'], features['fama'] = talib.MAMA(close, fastlimit=0.5, slowlimit=0.05)
            features['ht_trendline'] = talib.HT_TRENDLINE(close)

            # 2. モメンタム指標（20種類）
            features['rsi_14'] = talib.RSI(close, timeperiod=14)
            features['rsi_7'] = talib.RSI(close, timeperiod=7)
            features['rsi_21'] = talib.RSI(close, timeperiod=21)
            features['mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14)
            features['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            features['cci_14'] = talib.CCI(high, low, close, timeperiod=14)
            features['stoch_k'], features['stoch_d'] = talib.STOCH(high, low, close)
            features['stochf_k'], features['stochf_d'] = talib.STOCHF(high, low, close)
            features['stochrsi_k'], features['stochrsi_d'] = talib.STOCHRSI(close)
            features['ultimate_osc'] = talib.ULTOSC(high, low, close)
            features['roc_10'] = talib.ROC(close, timeperiod=10)
            features['roc_5'] = talib.ROC(close, timeperiod=5)
            features['rocp_10'] = talib.ROCP(close, timeperiod=10)
            features['rocr_10'] = talib.ROCR(close, timeperiod=10)
            features['trix_14'] = talib.TRIX(close, timeperiod=14)
            features['apo'] = talib.APO(close)
            features['ppo'] = talib.PPO(close)
            features['cmo_14'] = talib.CMO(close, timeperiod=14)
            features['dx_14'] = talib.DX(high, low, close, timeperiod=14)
            features['adx_14'] = talib.ADX(high, low, close, timeperiod=14)

            # 3. ボラティリティ指標（8種類）
            features['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            features['atr_7'] = talib.ATR(high, low, close, timeperiod=7)
            features['natr_14'] = talib.NATR(high, low, close, timeperiod=14)
            features['trange'] = talib.TRANGE(high, low, close)
            features['ht_dcperiod'] = talib.HT_DCPERIOD(close)
            features['ht_dcphase'] = talib.HT_DCPHASE(close)
            features['inphase'], features['quadrature'] = talib.HT_PHASOR(close)
            features['ht_sine'], features['ht_leadsine'] = talib.HT_SINE(close)

            # 4. ボリューム指標（10種類）
            features['ad'] = talib.AD(high, low, close, volume)
            features['adosc'] = talib.ADOSC(high, low, close, volume)
            features['obv'] = talib.OBV(close, volume)
            features['cmf'] = features['ad'] / volume  # Chaikin Money Flow
            features['volume_sma_10'] = talib.SMA(volume, timeperiod=10)
            features['volume_ratio'] = volume / features['volume_sma_10']
            features['price_volume'] = close * volume
            features['vwap'] = (features['price_volume'].rolling(20).sum() /
                              pd.Series(volume).rolling(20).sum())
            features['volume_oscillator'] = (talib.SMA(volume, 5) - talib.SMA(volume, 10)) / talib.SMA(volume, 10) * 100
            features['ease_of_movement'] = ((high + low) / 2 - (high.shift(1) + low.shift(1)) / 2) / (volume / ((high - low) * 1000000))

            # 5. オーバーレイ指標（12種類）
            upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=20)
            features['bb_upper'] = upperband
            features['bb_middle'] = middleband
            features['bb_lower'] = lowerband
            features['bb_width'] = (upperband - lowerband) / middleband
            features['bb_position'] = (close - lowerband) / (upperband - lowerband)

            features['sar'] = talib.SAR(high, low)
            features['sar_ext'] = talib.SAREXT(high, low)

            macd, macdsignal, macdhist = talib.MACD(close)
            features['macd'] = macd
            features['macd_signal'] = macdsignal
            features['macd_hist'] = macdhist
            features['macd_ratio'] = macd / macdsignal

            # 6. 価格パターン指標（10種類）
            features['cdl_doji'] = talib.CDLDOJI(open_price, high, low, close)
            features['cdl_hammer'] = talib.CDLHAMMER(open_price, high, low, close)
            features['cdl_engulfing'] = talib.CDLENGULFING(open_price, high, low, close)
            features['cdl_harami'] = talib.CDLHARAMI(open_price, high, low, close)
            features['cdl_spinning_top'] = talib.CDLSPINNINGTOP(open_price, high, low, close)
            features['cdl_marubozu'] = talib.CDLMARUBOZU(open_price, high, low, close)
            features['cdl_shooting_star'] = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
            features['cdl_hanging_man'] = talib.CDLHANGINGMAN(open_price, high, low, close)
            features['cdl_morning_star'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
            features['cdl_evening_star'] = talib.CDLEVENINGSTAR(open_price, high, low, close)

            # 7. 統計的指標（8種類）
            features['returns'] = close.pct_change()
            features['log_returns'] = np.log(close / close.shift(1))
            features['volatility_5'] = features['returns'].rolling(5).std()
            features['volatility_20'] = features['returns'].rolling(20).std()
            features['skewness_20'] = features['returns'].rolling(20).skew()
            features['kurtosis_20'] = features['returns'].rolling(20).kurt()
            features['var_95'] = features['returns'].rolling(20).quantile(0.05)
            features['sharpe_ratio_20'] = features['returns'].rolling(20).mean() / features['volatility_20']

            # 8. カスタム複合指標（6種類）
            features['price_position'] = (close - talib.SMA(close, 20)) / talib.SMA(close, 20)
            features['volume_price_trend'] = features['volume_ratio'] * features['returns']
            features['momentum_divergence'] = features['rsi_14'] - (close / close.shift(14) - 1) * 100
            features['trend_strength'] = abs(features['sma_5'] - features['sma_20']) / features['sma_20']
            features['volatility_breakout'] = features['atr_14'] / close
            features['composite_momentum'] = (features['rsi_14'] / 100 + features['stoch_k'] / 100 +
                                            (features['williams_r'] + 100) / 100) / 3

        except Exception as e:
            self.logger.error(f"特徴量計算エラー: {e}")
            raise

        # NaN値を前方向に補間
        features = features.fillna(method='ffill').fillna(method='bfill')

        # 最初の100行を削除（指標計算のため）
        features = features.iloc[100:].copy()

        self.logger.info(f"高度特徴量作成完了: {features.shape[1]}特徴量")
        return features

class AdvancedMLPredictionSystem:
    """高度機械学習予測システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_engineering = AdvancedFeatureEngineering()

        # データベース設定
        self.db_path = Path("ml_models_data/advanced_ml_predictions.db")
        self.db_path.parent.mkdir(exist_ok=True)

        # モデル設定
        self.model_configs = self._create_model_configs()

        # 訓練されたモデル
        self.trained_models: Dict[str, Any] = {}

        self.logger.info("Advanced ML prediction system initialized")

    def _create_model_configs(self) -> List[ModelConfig]:
        """モデル設定作成"""

        configs = [
            # Random Forest 配置
            ModelConfig(
                model_type=ModelType.RANDOM_FOREST,
                scaler_type=ScalerType.NONE,
                feature_selection=FeatureSelectionType.K_BEST_F,
                feature_selection_k=40,
                hyperparameters={
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            ),

            # Gradient Boosting 配置
            ModelConfig(
                model_type=ModelType.GRADIENT_BOOSTING,
                scaler_type=ScalerType.STANDARD,
                feature_selection=FeatureSelectionType.K_BEST_MUTUAL,
                feature_selection_k=35,
                hyperparameters={
                    'n_estimators': 150,
                    'learning_rate': 0.1,
                    'max_depth': 8,
                    'min_samples_split': 4,
                    'random_state': 42
                }
            ),

            # Logistic Regression 配置
            ModelConfig(
                model_type=ModelType.LOGISTIC_REGRESSION,
                scaler_type=ScalerType.STANDARD,
                feature_selection=FeatureSelectionType.K_BEST_F,
                feature_selection_k=30,
                hyperparameters={
                    'C': 1.0,
                    'penalty': 'l2',
                    'max_iter': 1000,
                    'random_state': 42
                }
            ),

            # SVM 配置
            ModelConfig(
                model_type=ModelType.SVM,
                scaler_type=ScalerType.ROBUST,
                feature_selection=FeatureSelectionType.K_BEST_F,
                feature_selection_k=25,
                hyperparameters={
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'probability': True,
                    'random_state': 42
                }
            ),

            # Neural Network 配置
            ModelConfig(
                model_type=ModelType.NEURAL_NETWORK,
                scaler_type=ScalerType.MINMAX,
                feature_selection=FeatureSelectionType.K_BEST_MUTUAL,
                feature_selection_k=45,
                hyperparameters={
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'learning_rate': 'adaptive',
                    'max_iter': 500,
                    'random_state': 42
                }
            )
        ]

        return configs

    async def train_advanced_models(self, symbol: str, period: str = "6mo", hyperparameters: Optional[Dict[ModelType, Dict[str, Any]]] = None) -> Dict[ModelType, ModelPerformance]:
        """高度モデル訓練"""

        self.logger.info(f"高度モデル訓練開始: {symbol}")

        try:
            # データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, period)

            if data is None or len(data) < 150:
                raise ValueError("訓練に十分なデータがありません")

            # 特徴量エンジニアリング
            features = await self.feature_engineering.create_advanced_features(data)

            # ターゲット作成（翌日の上昇/下降）
            targets = self._create_targets(data.iloc[100:]['Close'])  # 特徴量と同じ期間

            # データ同期
            min_length = min(len(features), len(targets))
            features = features.iloc[:min_length]
            targets = targets[:min_length]

            if len(features) < 50:
                raise ValueError("訓練データが不足しています")

            # 各モデル訓練
            if hyperparameters is None:
                hyperparameters = {}
            performances = {}

            for config in self.model_configs:
                try:
                    # ここでハイパーパラメータ最適化を呼び出す
                    optimized_params = hyperparameters.get(config.model_type)

                    performance = await self._train_single_model(
                        features, targets, config, symbol, optimized_params
                    )
                    performances[config.model_type] = performance

                except Exception as e:
                    self.logger.error(f"モデル訓練失敗 {config.model_type.value}: {e}")
                    continue

            # アンサンブルモデル作成
            if len(performances) >= 3:
                ensemble_performance = await self._create_ensemble_model(
                    features, targets, symbol
                )
                performances[ModelType.ENSEMBLE_VOTING] = ensemble_performance

            # 結果保存
            await self._save_model_performances(symbol, performances)

            self.logger.info(f"高度モデル訓練完了: {len(performances)}モデル")
            return performances

        except Exception as e:
            self.logger.error(f"高度モデル訓練エラー: {e}")
            raise

    def _create_targets(self, prices: pd.Series) -> np.ndarray:
        """ターゲット作成"""

        # 翌日の価格変動率
        returns = prices.pct_change().shift(-1)  # 翌日のリターン

        # 閾値以上の上昇を1、それ以外を0
        threshold = 0.005  # 0.5%以上の上昇
        targets = (returns > threshold).astype(int)

        return targets.values[:-1]  # 最後の要素（NaN）を除去

    async def _train_single_model(self, features: pd.DataFrame, targets: np.ndarray,
                                config: ModelConfig, symbol: str, hyperparameters: Optional[Dict[str, Any]] = None) -> ModelPerformance:
        """単一モデル訓練"""

        X = features.copy()
        y = targets.copy()

        # 特徴選択
        if config.feature_selection != FeatureSelectionType.NONE:
            X = self._apply_feature_selection(X, y, config)

        # スケーリング
        if config.scaler_type != ScalerType.NONE:
            X = self._apply_scaling(X, config.scaler_type)

        # モデル作成
        model = self._create_model(config)

        # 訓練
        model.fit(X, y)

        # 性能評価
        predictions = model.predict(X)
        prediction_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else predictions

        # クロスバリデーション
        cv_scores = cross_val_score(model, X, y, cv=5)

        # 特徴重要度
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for i, importance in enumerate(model.feature_importances_):
                feature_importance[X.columns[i]] = float(importance)

        performance = ModelPerformance(
            model_type=config.model_type,
            accuracy=accuracy_score(y, predictions),
            precision=precision_score(y, predictions, average='weighted', zero_division=0),
            recall=recall_score(y, predictions, average='weighted', zero_division=0),
            f1_score=f1_score(y, predictions, average='weighted', zero_division=0),
            roc_auc=roc_auc_score(y, prediction_proba) if len(np.unique(y)) > 1 else 0.5,
            cross_val_score=float(cv_scores.mean()),
            feature_importance=feature_importance
        )

        # モデル保存
        model_key = f"{symbol}_{config.model_type.value}"
        self.trained_models[model_key] = {
            'model': model,
            'config': config,
            'feature_columns': X.columns.tolist(),
            'performance': performance
        }
        self.save_model(model, model_key) # Save the model

        return performance

    def _apply_feature_selection(self, X: pd.DataFrame, y: np.ndarray,
                                config: ModelConfig) -> pd.DataFrame:
        """特徴選択適用"""

        k = min(config.feature_selection_k, X.shape[1])

        if config.feature_selection == FeatureSelectionType.K_BEST_F:
            selector = SelectKBest(score_func=f_classif, k=k)
        elif config.feature_selection == FeatureSelectionType.K_BEST_MUTUAL:
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            return X

        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]

        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    def _apply_scaling(self, X: pd.DataFrame, scaler_type: ScalerType) -> pd.DataFrame:
        """スケーリング適用"""

        if scaler_type == ScalerType.STANDARD:
            scaler = StandardScaler()
        elif scaler_type == ScalerType.ROBUST:
            scaler = RobustScaler()
        elif scaler_type == ScalerType.MINMAX:
            scaler = MinMaxScaler()
        else:
            return X

        X_scaled = scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def _create_model(self, config: ModelConfig):
        """モデル作成"""

        params = config.hyperparameters.copy()
        if hyperparameters:
            params.update(hyperparameters)

        if config.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(**params)
        elif config.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(**params)
        elif config.model_type == ModelType.LOGISTIC_REGRESSION:
            return LogisticRegression(**params)
        elif config.model_type == ModelType.SVM:
            return SVC(**params)
        elif config.model_type == ModelType.NEURAL_NETWORK:
            return MLPClassifier(**params)
        else:
            raise ValueError(f"未サポートのモデルタイプ: {config.model_type}")

    async def _create_ensemble_model(self, features: pd.DataFrame, targets: np.ndarray,
                                   symbol: str) -> ModelPerformance:
        """アンサンブルモデル作成"""

        # 最高性能のモデルを選択
        best_models = []
        for model_key, model_data in self.trained_models.items():
            if symbol in model_key and model_data['performance'].accuracy > 0.55:
                best_models.append((
                    model_data['config'].model_type.value,
                    model_data['model']
                ))

        if len(best_models) < 2:
            # フォールバック：全モデル使用
            best_models = [
                (model_data['config'].model_type.value, model_data['model'])
                for model_key, model_data in self.trained_models.items()
                if symbol in model_key
            ]

        # Voting Classifier作成
        voting_classifier = VotingClassifier(
            estimators=best_models,
            voting='soft' if all(hasattr(model, 'predict_proba') for _, model in best_models) else 'hard'
        )

        # 特徴選択とスケーリング（Random Forestの設定を使用）
        X = features.copy()
        y = targets.copy()

        # 最も基本的な前処理のみ適用
        k = min(50, X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

        # 訓練
        voting_classifier.fit(X, y)

        # 性能評価
        predictions = voting_classifier.predict(X)
        prediction_proba = voting_classifier.predict_proba(X)[:, 1] if hasattr(voting_classifier, 'predict_proba') else predictions

        cv_scores = cross_val_score(voting_classifier, X, y, cv=3)  # 高速化のためCV=3

        performance = ModelPerformance(
            model_type=ModelType.ENSEMBLE_VOTING,
            accuracy=accuracy_score(y, predictions),
            precision=precision_score(y, predictions, average='weighted', zero_division=0),
            recall=recall_score(y, predictions, average='weighted', zero_division=0),
            f1_score=f1_score(y, predictions, average='weighted', zero_division=0),
            roc_auc=roc_auc_score(y, prediction_proba) if len(np.unique(y)) > 1 else 0.5,
            cross_val_score=float(cv_scores.mean())
        )

        # モデル保存
        model_key = f"{symbol}_{ModelType.ENSEMBLE_VOTING.value}"
        self.trained_models[model_key] = {
            'model': voting_classifier,
            'config': None,
            'feature_columns': X.columns.tolist(),
            'performance': performance
        }
        self.save_model(voting_classifier, model_key) # Save the model

        return performance

    async def _save_model_performances(self, symbol: str,
                                     performances: Dict[ModelType, ModelPerformance]):
        """モデル性能保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # テーブル作成
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS advanced_model_performances (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        f1_score REAL,
                        roc_auc REAL,
                        cross_val_score REAL,
                        feature_importance TEXT,
                        created_at TEXT,
                        UNIQUE(symbol, model_type)
                    )
                ''')

                # データ挿入
                for model_type, performance in performances.items():
                    cursor.execute('''
                        INSERT OR REPLACE INTO advanced_model_performances
                        (symbol, model_type, accuracy, precision_score, recall_score,
                         f1_score, roc_auc, cross_val_score, feature_importance, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        model_type.value,
                        performance.accuracy,
                        performance.precision,
                        performance.recall,
                        performance.f1_score,
                        performance.roc_auc,
                        performance.cross_val_score,
                        json.dumps(performance.feature_importance),
                        datetime.now().isoformat()
                    ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"性能保存エラー: {e}")

    def save_model(self, model, model_key: str):
        """モデルをファイルに保存する"""
        model_path = self.db_path.parent / "trained_advanced_models" / f"{model_key}.joblib"
        model_path.parent.mkdir(exist_ok=True)
        try:
            joblib.dump(model, model_path)
            self.logger.info(f"モデルを保存しました: {model_path}")
        except Exception as e:
            self.logger.error(f"モデルの保存に失敗しました {model_key}: {e}")

    def load_model(self, model_key: str):
        """ファイルからモデルをロードする"""
        model_path = self.db_path.parent / "trained_advanced_models" / f"{model_key}.joblib"
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                self.logger.info(f"モデルをロードしました: {model_path}")
                return model
            except Exception as e:
                self.logger.error(f"モデルのロードに失敗しました {model_key}: {e}")
        return None

    async def predict_with_advanced_models(self, symbol: str) -> PredictionResult:
        """高度モデルによる予測"""

        try:
            # 最新データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, "2mo")

            # 特徴量作成
            features = await self.feature_engineering.create_advanced_features(data)
            latest_features = features.iloc[-1:].copy()

            # 各モデルで予測
            model_predictions = {}
            confidences = []

            for model_key, model_data in self.trained_models.items():
                if symbol in model_key:
                    try:
                        model = model_data['model']
                        feature_columns = model_data['feature_columns']

                        # 特徴量を合わせる
                        X = latest_features[feature_columns].copy()

                        # 予測
                        prediction = model.predict(X)[0]
                        confidence = model.predict_proba(X)[0].max() if hasattr(model, 'predict_proba') else 0.7

                        model_type = model_data['config'].model_type.value if model_data['config'] else ModelType.ENSEMBLE_VOTING.value
                        model_predictions[model_type] = int(prediction)
                        confidences.append(confidence)

                    except Exception as e:
                        self.logger.warning(f"予測失敗 {model_key}: {e}")
                        continue

            # 最終予測（多数決）
            if model_predictions:
                final_prediction = int(np.mean(list(model_predictions.values())) >= 0.5)
                final_confidence = float(np.mean(confidences))
            else:
                final_prediction = 1  # デフォルト予測
                final_confidence = 0.5

            # 特徴量値
            feature_values = {col: float(val) for col, val in latest_features.iloc[0].items()}

            return PredictionResult(
                symbol=symbol,
                prediction=final_prediction,
                confidence=final_confidence,
                model_consensus=model_predictions,
                feature_values=feature_values,
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"予測エラー: {e}")
            # デフォルト予測を返す
            return PredictionResult(
                symbol=symbol,
                prediction=1,
                confidence=0.5,
                model_consensus={},
                feature_values={},
                timestamp=datetime.now()
            )

# グローバルインスタンス
advanced_ml_system = AdvancedMLPredictionSystem()

# テスト実行
async def test_advanced_ml_system():
    """高度ML系统测试"""

    print("=== 高度機械学習予測システムテスト ===")

    test_symbol = "7203"

    # モデル訓練
    print(f"\n🤖 {test_symbol} 高度モデル訓練開始...")
    performances = await advanced_ml_system.train_advanced_models(test_symbol, "4mo")

    print(f"\n📊 訓練結果:")
    for model_type, performance in performances.items():
        print(f"  {model_type.value}: 精度{performance.accuracy:.3f} F1{performance.f1_score:.3f}")

    # 予測実行
    print(f"\n🔮 予測実行...")
    prediction = await advanced_ml_system.predict_with_advanced_models(test_symbol)

    print(f"\n📈 予測結果:")
    print(f"  予測: {'上昇' if prediction.prediction else '下降'}")
    print(f"  信頼度: {prediction.confidence:.3f}")
    print(f"  モデル合意: {prediction.model_consensus}")

    print(f"\n✅ 高度機械学習システムテスト完了")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_advanced_ml_system())