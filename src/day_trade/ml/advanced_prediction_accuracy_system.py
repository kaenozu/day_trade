#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Prediction Accuracy System - 高度予測精度向上システム

予測精度の大幅な向上を実現する包括的なシステム
- 高度な特徴量エンジニアリング
- 多層アンサンブル学習
- 動的モデル選択
- リアルタイム精度監視
- 自動的な改善提案
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from collections import defaultdict, deque
import hashlib
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, classification_report
)
import optuna
from optuna.samplers import TPESampler
from scipy import stats

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

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """モデルタイプ列挙型"""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    SVM = "svm"
    ENSEMBLE_VOTING = "ensemble_voting"
    ENSEMBLE_STACKING = "ensemble_stacking"
    ADAPTIVE_ENSEMBLE = "adaptive_ensemble"

class PredictionConfidence(Enum):
    """予測信頼度レベル"""
    VERY_HIGH = "very_high"    # 90%以上
    HIGH = "high"              # 80-90%
    MEDIUM = "medium"          # 70-80%
    LOW = "low"               # 60-70%
    VERY_LOW = "very_low"     # 60%未満

@dataclass
class AccuracyMetrics:
    """精度メトリクス"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    confidence: PredictionConfidence = PredictionConfidence.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PredictionResult:
    """予測結果"""
    symbol: str
    prediction: float
    probability: float
    confidence: PredictionConfidence
    model_type: ModelType
    features: Dict[str, float]
    accuracy_metrics: AccuracyMetrics
    explanation: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ModelPerformance:
    """モデルパフォーマンス"""
    model_type: ModelType
    symbol: str
    accuracy_metrics: AccuracyMetrics
    feature_importance: Dict[str, float]
    training_time: float
    prediction_time: float
    memory_usage: float
    last_updated: datetime = field(default_factory=datetime.now)

class AdvancedFeatureEngineering:
    """高度な特徴量エンジニアリング"""
    
    def __init__(self):
        self.feature_cache = {}
        self.feature_importance_history = defaultdict(list)
        
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標特徴量作成"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # 基本的な価格特徴量
        df['price_change'] = df['close'].pct_change()
        df['price_volatility'] = df['price_change'].rolling(window=20).std()
        df['volume_change'] = df['volume'].pct_change()
        df['volume_volatility'] = df['volume_change'].rolling(window=20).std()
        
        # 移動平均系
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # ボリンジャーバンド
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 高度なモメンタム指標
        df['stoch_k'] = self._calculate_stochastic_k(df)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        df['williams_r'] = self._calculate_williams_r(df)
        df['cci'] = self._calculate_cci(df)
        
        return df
    
    def create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """マーケット特徴量作成"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # 時間的特徴量
        df['hour'] = pd.to_datetime(df.index).hour if hasattr(df.index, 'hour') else 9
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek if hasattr(df.index, 'dayofweek') else 0
        df['month'] = pd.to_datetime(df.index).month if hasattr(df.index, 'month') else 1
        df['quarter'] = pd.to_datetime(df.index).quarter if hasattr(df.index, 'quarter') else 1
        
        # 市場体制特徴量
        df['market_regime'] = self._identify_market_regime(df)
        df['volatility_regime'] = self._identify_volatility_regime(df)
        
        # 流動性特徴量
        df['bid_ask_spread'] = 0.001  # Mock値
        df['market_impact'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # センチメント特徴量（Mock）
        df['market_sentiment'] = np.random.normal(0, 1, len(df))  # 実際にはニュースAPIから取得
        df['fear_greed_index'] = np.random.uniform(0, 100, len(df))
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高度な特徴量作成"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # フラクタル次元
        df['fractal_dimension'] = self._calculate_fractal_dimension(df['close'])
        
        # ハースト指数
        df['hurst_exponent'] = self._calculate_hurst_exponent(df['close'])
        
        # エントロピー
        df['entropy'] = self._calculate_entropy(df['close'])
        
        # 相関特徴量
        df['price_volume_corr'] = self._rolling_correlation(df['close'], df['volume'], 20)
        
        # ウェーブレット特徴量
        df['wavelet_energy'] = self._calculate_wavelet_energy(df['close'])
        
        # 非線形特徴量
        df['lyapunov_exponent'] = self._calculate_lyapunov_exponent(df['close'])
        
        return df
    
    def _calculate_stochastic_k(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ストキャスティクス%K計算"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        return k_percent
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ウィリアムズ%R計算"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        wr = -100 * ((high_max - df['close']) / (high_max - low_min))
        return wr
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """CCI（商品チャネル指数）計算"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (tp - sma) / (0.015 * mad)
        return cci
    
    def _identify_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """マーケット体制識別"""
        # 簡単な体制識別（トレンド/レンジ）
        price_change = df['close'].pct_change()
        volatility = price_change.rolling(window=20).std()
        trend_strength = abs(df['close'].rolling(window=20).mean().pct_change())
        
        # 0: レンジ相場, 1: 上昇トレンド, 2: 下降トレンド
        regime = np.where(trend_strength > volatility * 2, 
                         np.where(price_change > 0, 1, 2), 0)
        return pd.Series(regime, index=df.index)
    
    def _identify_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """ボラティリティ体制識別"""
        volatility = df['close'].pct_change().rolling(window=20).std()
        vol_percentile = volatility.rolling(window=100).rank(pct=True)
        
        # 0: 低ボラティリティ, 1: 通常, 2: 高ボラティリティ
        regime = np.where(vol_percentile < 0.33, 0,
                         np.where(vol_percentile > 0.67, 2, 1))
        return pd.Series(regime, index=df.index)
    
    def _calculate_fractal_dimension(self, series: pd.Series, max_k: int = 10) -> pd.Series:
        """フラクタル次元計算"""
        def _fd_single(data):
            if len(data) < max_k * 2:
                return 1.5  # デフォルト値
            
            ns = []
            rs = []
            
            for k in range(1, max_k + 1):
                n = len(data) // k
                if n < 2:
                    continue
                    
                segments = [data[i*k:(i+1)*k] for i in range(n)]
                r = np.mean([np.max(seg) - np.min(seg) for seg in segments if len(seg) > 0])
                
                if r > 0:
                    ns.append(n)
                    rs.append(r)
            
            if len(ns) < 2:
                return 1.5
                
            try:
                slope, _, _, _, _ = stats.linregress(np.log(ns), np.log(rs))
                return max(1.0, min(2.0, -slope))  # 1-2の範囲に制限
            except:
                return 1.5
        
        return series.rolling(window=50).apply(lambda x: _fd_single(x.values))
    
    def _calculate_hurst_exponent(self, series: pd.Series) -> pd.Series:
        """ハースト指数計算"""
        def _hurst_single(data):
            if len(data) < 20:
                return 0.5  # デフォルト値
            
            try:
                lags = range(2, min(20, len(data) // 2))
                tau = [np.sqrt(np.var(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
                
                if len(tau) < 2 or any(t <= 0 for t in tau):
                    return 0.5
                
                slope, _, _, _, _ = stats.linregress(np.log(lags), np.log(tau))
                return max(0.0, min(1.0, slope))  # 0-1の範囲に制限
            except:
                return 0.5
        
        return series.rolling(window=50).apply(lambda x: _hurst_single(x.values))
    
    def _calculate_entropy(self, series: pd.Series, bins: int = 10) -> pd.Series:
        """エントロピー計算"""
        def _entropy_single(data):
            if len(data) < bins:
                return 0.0
            
            try:
                hist, _ = np.histogram(data, bins=bins)
                hist = hist[hist > 0]  # ゼロを除去
                prob = hist / hist.sum()
                entropy = -np.sum(prob * np.log2(prob))
                return entropy
            except:
                return 0.0
        
        return series.rolling(window=50).apply(lambda x: _entropy_single(x.values))
    
    def _rolling_correlation(self, series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
        """ローリング相関係数計算"""
        return series1.rolling(window=window).corr(series2.rolling(window=window))
    
    def _calculate_wavelet_energy(self, series: pd.Series) -> pd.Series:
        """ウェーブレット エネルギー計算（簡易版）"""
        # 簡易的なウェーブレット変換として高周波成分のエネルギーを計算
        def _energy_single(data):
            if len(data) < 10:
                return 0.0
            try:
                diff = np.diff(data)
                energy = np.sum(diff ** 2)
                return energy
            except:
                return 0.0
        
        return series.rolling(window=20).apply(lambda x: _energy_single(x.values))
    
    def _calculate_lyapunov_exponent(self, series: pd.Series) -> pd.Series:
        """リアプノフ指数計算（簡易版）"""
        def _lyapunov_single(data):
            if len(data) < 20:
                return 0.0
            
            try:
                # 簡易的なカオス性の指標
                returns = np.diff(np.log(data + 1e-10))
                if len(returns) < 10:
                    return 0.0
                
                # 隣接点間の距離の発散度を測定
                divergence = 0
                count = 0
                
                for i in range(len(returns) - 10):
                    for j in range(i + 1, min(i + 10, len(returns))):
                        if abs(returns[i]) > 1e-10:
                            div = abs(returns[j] - returns[i]) / abs(returns[i])
                            divergence += np.log(max(div, 1e-10))
                            count += 1
                
                return divergence / max(count, 1) if count > 0 else 0.0
            except:
                return 0.0
        
        return series.rolling(window=50).apply(lambda x: _lyapunov_single(x.values))

class MultiLayerEnsemble:
    """多層アンサンブル学習システム"""
    
    def __init__(self):
        self.base_models = {}
        self.meta_models = {}
        self.ensemble_weights = {}
        self.performance_history = defaultdict(list)
        
    def create_base_models(self) -> Dict[str, Any]:
        """ベースモデル作成"""
        from sklearn.ensemble import (
            RandomForestClassifier, GradientBoostingClassifier,
            ExtraTreesClassifier, AdaBoostClassifier
        )
        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        
        try:
            import xgboost as xgb
            import lightgbm as lgb
            has_xgb = True
            has_lgb = True
        except ImportError:
            has_xgb = False
            has_lgb = False
            logger.warning("XGBoost and/or LightGBM not available")
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=1000
            ),
            'ridge_classifier': RidgeClassifier(
                random_state=42
            ),
            'svm_rbf': SVC(
                kernel='rbf', probability=True, random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=42, max_iter=500
            )
        }
        
        if has_xgb:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100, random_state=42, n_jobs=-1,
                eval_metric='logloss'
            )
        
        if has_lgb:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100, random_state=42, n_jobs=-1,
                verbose=-1
            )
        
        return models
    
    def create_meta_models(self) -> Dict[str, Any]:
        """メタモデル作成"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        
        return {
            'meta_logistic': LogisticRegression(random_state=42),
            'meta_rf': RandomForestClassifier(n_estimators=50, random_state=42),
            'meta_mlp': MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=300)
        }
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """アンサンブル訓練"""
        logger.info("アンサンブル学習を開始します")
        
        # ベースモデル訓練
        base_models = self.create_base_models()
        base_predictions = {}
        
        for name, model in base_models.items():
            try:
                logger.info(f"ベースモデル {name} を訓練中...")
                model.fit(X_train, y_train)
                
                # 検証データでの予測
                if X_val is not None:
                    pred_proba = model.predict_proba(X_val)[:, 1]
                    base_predictions[name] = pred_proba
                    
                    # パフォーマンス記録
                    pred_class = model.predict(X_val)
                    accuracy = accuracy_score(y_val, pred_class)
                    self.performance_history[name].append(accuracy)
                
                self.base_models[name] = model
                logger.info(f"ベースモデル {name} 訓練完了")
                
            except Exception as e:
                logger.error(f"ベースモデル {name} の訓練エラー: {e}")
                continue
        
        # メタモデル訓練
        if X_val is not None and base_predictions:
            meta_X = pd.DataFrame(base_predictions)
            meta_models = self.create_meta_models()
            
            for name, model in meta_models.items():
                try:
                    logger.info(f"メタモデル {name} を訓練中...")
                    model.fit(meta_X, y_val)
                    self.meta_models[name] = model
                    logger.info(f"メタモデル {name} 訓練完了")
                except Exception as e:
                    logger.error(f"メタモデル {name} の訓練エラー: {e}")
                    continue
        
        # 動的重み更新
        self._update_ensemble_weights()
        
        return {
            'base_models': len(self.base_models),
            'meta_models': len(self.meta_models),
            'training_completed': True
        }
    
    def predict_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """アンサンブル予測"""
        if not self.base_models:
            raise ValueError("モデルが訓練されていません")
        
        base_predictions = {}
        base_probabilities = {}
        
        # ベースモデル予測
        for name, model in self.base_models.items():
            try:
                pred_class = model.predict(X)
                pred_proba = model.predict_proba(X)[:, 1]
                
                base_predictions[name] = pred_class
                base_probabilities[name] = pred_proba
                
            except Exception as e:
                logger.error(f"ベースモデル {name} 予測エラー: {e}")
                continue
        
        if not base_predictions:
            raise RuntimeError("予測可能なモデルがありません")
        
        # メタモデル予測
        if self.meta_models:
            meta_X = pd.DataFrame(base_probabilities)
            meta_predictions = {}
            
            for name, model in self.meta_models.items():
                try:
                    pred_proba = model.predict_proba(meta_X)[:, 1]
                    meta_predictions[name] = pred_proba
                except Exception as e:
                    logger.error(f"メタモデル {name} 予測エラー: {e}")
                    continue
            
            # メタモデルの平均を最終予測とする
            if meta_predictions:
                final_probabilities = np.mean(list(meta_predictions.values()), axis=0)
            else:
                # メタモデルが使用不可の場合は重み付き平均
                final_probabilities = self._weighted_average_prediction(base_probabilities)
        else:
            # メタモデルがない場合は重み付き平均
            final_probabilities = self._weighted_average_prediction(base_probabilities)
        
        final_predictions = (final_probabilities > 0.5).astype(int)
        
        # 信頼度計算
        confidence_scores = self._calculate_prediction_confidence(base_probabilities)
        
        return final_predictions, final_probabilities, confidence_scores
    
    def _weighted_average_prediction(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """重み付き平均予測"""
        if not self.ensemble_weights:
            # 等重みで平均
            return np.mean(list(predictions.values()), axis=0)
        
        weighted_sum = np.zeros_like(list(predictions.values())[0])
        weight_sum = 0.0
        
        for name, pred in predictions.items():
            weight = self.ensemble_weights.get(name, 1.0)
            weighted_sum += weight * pred
            weight_sum += weight
        
        return weighted_sum / max(weight_sum, 1e-10)
    
    def _update_ensemble_weights(self):
        """アンサンブル重み更新"""
        for name, performances in self.performance_history.items():
            if performances:
                # 最近のパフォーマンスに基づく重み計算
                recent_perf = performances[-min(10, len(performances)):]
                weight = np.mean(recent_perf)
                self.ensemble_weights[name] = max(0.1, weight)  # 最低重み保証
    
    def _calculate_prediction_confidence(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """予測信頼度計算"""
        if not predictions:
            return {}
        
        pred_array = np.array(list(predictions.values()))
        
        # 予測の一致度
        prediction_std = np.std(pred_array, axis=0)
        avg_confidence = 1.0 - np.mean(prediction_std)
        
        # モデル数による信頼度補正
        model_count_factor = min(1.0, len(predictions) / 5.0)
        
        # 総合信頼度
        overall_confidence = avg_confidence * model_count_factor
        
        return {
            'overall_confidence': float(overall_confidence),
            'prediction_std': float(np.mean(prediction_std)),
            'model_agreement': float(1.0 - np.mean(prediction_std)),
            'model_count': len(predictions)
        }

class DynamicModelSelection:
    """動的モデル選択システム"""
    
    def __init__(self):
        self.model_performance = {}
        self.selection_history = deque(maxlen=1000)
        self.context_performance = defaultdict(lambda: defaultdict(list))
        
    def select_best_model(self, symbol: str, market_context: Dict[str, Any],
                         available_models: Dict[str, Any]) -> List[str]:
        """最適モデル選択"""
        if not available_models:
            return []
        
        # コンテキスト特徴量抽出
        context_key = self._extract_context_key(symbol, market_context)
        
        # 各モデルのコンテキスト別パフォーマンス評価
        model_scores = {}
        
        for model_name in available_models.keys():
            # 履歴からのスコア計算
            if context_key in self.context_performance:
                performances = self.context_performance[context_key][model_name]
                if performances:
                    # 最近のパフォーマンスを重視
                    weights = np.exp(np.linspace(0, 1, len(performances)))
                    weighted_score = np.average(performances, weights=weights)
                else:
                    weighted_score = 0.5  # デフォルトスコア
            else:
                weighted_score = 0.5
            
            # 全体的なパフォーマンスも考慮
            global_perf = self.model_performance.get(model_name, {}).get('accuracy', 0.5)
            
            # 最終スコア
            model_scores[model_name] = 0.7 * weighted_score + 0.3 * global_perf
        
        # スコア順にソート
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 上位モデルを選択（最大3つ）
        selected_models = [name for name, score in sorted_models[:3]]
        
        # 選択履歴記録
        self.selection_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'context': context_key,
            'selected_models': selected_models,
            'scores': model_scores
        })
        
        return selected_models
    
    def update_model_performance(self, model_name: str, symbol: str,
                                market_context: Dict[str, Any], 
                                performance_metrics: AccuracyMetrics):
        """モデルパフォーマンス更新"""
        # 全体的なパフォーマンス更新
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {
                'accuracy_history': [],
                'precision_history': [],
                'recall_history': [],
                'f1_history': []
            }
        
        perf = self.model_performance[model_name]
        perf['accuracy_history'].append(performance_metrics.accuracy)
        perf['precision_history'].append(performance_metrics.precision)
        perf['recall_history'].append(performance_metrics.recall)
        perf['f1_history'].append(performance_metrics.f1_score)
        
        # 最新の平均を保存
        perf['accuracy'] = np.mean(perf['accuracy_history'][-20:])  # 最新20回の平均
        perf['precision'] = np.mean(perf['precision_history'][-20:])
        perf['recall'] = np.mean(perf['recall_history'][-20:])
        perf['f1_score'] = np.mean(perf['f1_history'][-20:])
        
        # コンテキスト別パフォーマンス更新
        context_key = self._extract_context_key(symbol, market_context)
        self.context_performance[context_key][model_name].append(performance_metrics.accuracy)
        
        # 履歴制限（メモリ効率のため）
        if len(self.context_performance[context_key][model_name]) > 50:
            self.context_performance[context_key][model_name] = \
                self.context_performance[context_key][model_name][-30:]
    
    def _extract_context_key(self, symbol: str, market_context: Dict[str, Any]) -> str:
        """コンテキストキー抽出"""
        # 市場コンテキストから特徴的な情報を抽出してキー化
        volatility = market_context.get('volatility', 'normal')
        trend = market_context.get('trend', 'neutral')
        volume = market_context.get('volume_regime', 'normal')
        time_of_day = market_context.get('time_of_day', 'regular')
        
        # セクター情報（銘柄コードから推定）
        sector = self._infer_sector(symbol)
        
        return f"{sector}_{volatility}_{trend}_{volume}_{time_of_day}"
    
    def _infer_sector(self, symbol: str) -> str:
        """セクター推定（簡易版）"""
        try:
            code = int(symbol.replace('.T', ''))
            if 1000 <= code <= 1999:
                return "mining"
            elif 2000 <= code <= 2999:
                return "food"
            elif 3000 <= code <= 3999:
                return "textiles"
            elif 4000 <= code <= 4999:
                return "chemicals"
            elif 5000 <= code <= 5999:
                return "pharmaceutical"
            elif 6000 <= code <= 6999:
                return "basic_materials"
            elif 7000 <= code <= 7999:
                return "machinery"
            elif 8000 <= code <= 8999:
                return "electrical"
            elif 9000 <= code <= 9999:
                return "services"
            else:
                return "other"
        except:
            return "other"

class RealtimeAccuracyMonitor:
    """リアルタイム精度監視システム"""
    
    def __init__(self):
        self.accuracy_buffer = defaultdict(lambda: deque(maxlen=100))
        self.alert_thresholds = {
            'accuracy_drop': 0.1,  # 10%以上の精度低下でアラート
            'prediction_drift': 0.15  # 15%以上の予測ドリフトでアラート
        }
        self.monitoring_active = True
        self.last_alert_time = {}
        
    def record_prediction(self, model_name: str, symbol: str, 
                         prediction: float, actual: float = None,
                         features: Dict[str, float] = None):
        """予測記録"""
        if not self.monitoring_active:
            return
        
        record = {
            'timestamp': datetime.now(),
            'model_name': model_name,
            'symbol': symbol,
            'prediction': prediction,
            'actual': actual,
            'features': features or {}
        }
        
        key = f"{model_name}_{symbol}"
        self.accuracy_buffer[key].append(record)
        
        # 精度チェック（実際の値が利用可能な場合）
        if actual is not None:
            self._check_accuracy_drift(key)
    
    def _check_accuracy_drift(self, key: str):
        """精度ドリフトチェック"""
        records = list(self.accuracy_buffer[key])
        
        if len(records) < 20:  # 最低20件の記録が必要
            return
        
        # 最新の精度計算
        recent_records = [r for r in records[-20:] if r['actual'] is not None]
        if len(recent_records) < 10:
            return
        
        recent_accuracy = np.mean([
            1.0 if abs(r['prediction'] - r['actual']) < 0.1 else 0.0
            for r in recent_records
        ])
        
        # 過去の精度計算
        if len(records) >= 40:
            past_records = [r for r in records[-40:-20] if r['actual'] is not None]
            if len(past_records) >= 10:
                past_accuracy = np.mean([
                    1.0 if abs(r['prediction'] - r['actual']) < 0.1 else 0.0
                    for r in past_records
                ])
                
                # 精度低下チェック
                accuracy_drop = past_accuracy - recent_accuracy
                if accuracy_drop > self.alert_thresholds['accuracy_drop']:
                    self._trigger_alert('accuracy_drop', key, {
                        'past_accuracy': past_accuracy,
                        'recent_accuracy': recent_accuracy,
                        'drop': accuracy_drop
                    })
    
    def _trigger_alert(self, alert_type: str, key: str, details: Dict[str, Any]):
        """アラート発生"""
        current_time = datetime.now()
        last_alert = self.last_alert_time.get(f"{alert_type}_{key}")
        
        # アラート頻度制限（30分間隔）
        if last_alert and (current_time - last_alert).seconds < 1800:
            return
        
        logger.warning(f"精度監視アラート: {alert_type} - {key}")
        logger.warning(f"詳細: {details}")
        
        self.last_alert_time[f"{alert_type}_{key}"] = current_time
        
        # 改善提案生成
        suggestions = self._generate_improvement_suggestions(alert_type, details)
        logger.info(f"改善提案: {suggestions}")
    
    def _generate_improvement_suggestions(self, alert_type: str, 
                                        details: Dict[str, Any]) -> List[str]:
        """改善提案生成"""
        suggestions = []
        
        if alert_type == 'accuracy_drop':
            suggestions.extend([
                "特徴量エンジニアリングの見直し",
                "モデルの再訓練",
                "アンサンブル構成の変更",
                "データ品質の確認",
                "市場体制変化の分析"
            ])
        
        return suggestions

class AdvancedPredictionAccuracySystem:
    """高度予測精度向上システム（メインクラス）"""
    
    def __init__(self, db_path: str = "advanced_prediction_system.db"):
        self.db_path = Path(db_path)
        self.feature_engineering = AdvancedFeatureEngineering()
        self.ensemble_system = MultiLayerEnsemble()
        self.model_selector = DynamicModelSelection()
        self.accuracy_monitor = RealtimeAccuracyMonitor()
        
        self.setup_database()
        
        # パフォーマンス統計
        self.performance_stats = {
            'predictions_made': 0,
            'accuracy_improvements': 0,
            'model_updates': 0,
            'feature_additions': 0
        }
        
        logger.info("高度予測精度向上システムを初期化しました")
    
    def setup_database(self):
        """データベース初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        symbol TEXT,
                        model_type TEXT,
                        prediction REAL,
                        probability REAL,
                        confidence TEXT,
                        actual_result REAL,
                        features TEXT,
                        accuracy_metrics TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        model_name TEXT,
                        symbol TEXT,
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        f1_score REAL,
                        training_time REAL,
                        prediction_time REAL,
                        feature_importance TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS system_stats (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        stat_type TEXT,
                        stat_value REAL,
                        details TEXT
                    )
                ''')
                
                conn.commit()
                logger.info("データベースを初期化しました")
        except Exception as e:
            logger.error(f"データベース初期化エラー: {e}")
    
    async def train_models(self, symbol: str, data: pd.DataFrame, 
                          target_column: str = 'target') -> Dict[str, Any]:
        """モデル訓練（非同期）"""
        try:
            logger.info(f"{symbol} のモデル訓練を開始します")
            
            # 特徴量エンジニアリング
            enhanced_data = await self._create_enhanced_features(data)
            
            # 訓練/検証データ分割
            split_idx = int(len(enhanced_data) * 0.8)
            train_data = enhanced_data.iloc[:split_idx]
            val_data = enhanced_data.iloc[split_idx:]
            
            # 特徴量とターゲット分離
            feature_columns = [col for col in enhanced_data.columns if col != target_column]
            X_train = train_data[feature_columns].fillna(0)
            y_train = train_data[target_column]
            X_val = val_data[feature_columns].fillna(0)
            y_val = val_data[target_column]
            
            # アンサンブル学習実行
            training_result = self.ensemble_system.train_ensemble(
                X_train, y_train, X_val, y_val
            )
            
            # パフォーマンス評価
            predictions, probabilities, confidence = self.ensemble_system.predict_ensemble(X_val)
            accuracy_metrics = self._calculate_metrics(y_val, predictions, probabilities)
            
            # パフォーマンス記録
            await self._save_model_performance(symbol, 'ensemble', accuracy_metrics)
            
            self.performance_stats['model_updates'] += 1
            
            logger.info(f"{symbol} のモデル訓練が完了しました。精度: {accuracy_metrics.accuracy:.3f}")
            
            return {
                'symbol': symbol,
                'training_result': training_result,
                'accuracy_metrics': accuracy_metrics,
                'feature_count': len(feature_columns),
                'training_samples': len(train_data),
                'validation_samples': len(val_data)
            }
            
        except Exception as e:
            logger.error(f"モデル訓練エラー ({symbol}): {e}")
            raise
    
    async def predict(self, symbol: str, data: pd.DataFrame, 
                     market_context: Dict[str, Any] = None) -> PredictionResult:
        """予測実行"""
        try:
            # 特徴量作成
            enhanced_data = await self._create_enhanced_features(data)
            feature_columns = [col for col in enhanced_data.columns 
                             if not col.startswith('target')]
            X = enhanced_data[feature_columns].fillna(0).iloc[-1:]  # 最新データのみ
            
            # 市場コンテキスト設定
            if market_context is None:
                market_context = self._extract_market_context(enhanced_data.iloc[-20:])
            
            # 動的モデル選択
            available_models = self.ensemble_system.base_models
            selected_models = self.model_selector.select_best_model(
                symbol, market_context, available_models
            )
            
            # 予測実行
            predictions, probabilities, confidence_scores = \
                self.ensemble_system.predict_ensemble(X)
            
            prediction = float(predictions[0])
            probability = float(probabilities[0])
            
            # 信頼度判定
            confidence_level = self._determine_confidence_level(
                probability, confidence_scores
            )
            
            # 精度メトリクス（予測時点では仮の値）
            accuracy_metrics = AccuracyMetrics(
                confidence=confidence_level
            )
            
            # 予測結果作成
            result = PredictionResult(
                symbol=symbol,
                prediction=prediction,
                probability=probability,
                confidence=confidence_level,
                model_type=ModelType.ADAPTIVE_ENSEMBLE,
                features=X.iloc[0].to_dict(),
                accuracy_metrics=accuracy_metrics,
                explanation={
                    'selected_models': selected_models,
                    'market_context': market_context,
                    'confidence_scores': confidence_scores
                }
            )
            
            # リアルタイム監視記録
            self.accuracy_monitor.record_prediction(
                'ensemble', symbol, prediction, features=result.features
            )
            
            # 予測記録保存
            await self._save_prediction_result(result)
            
            self.performance_stats['predictions_made'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"予測エラー ({symbol}): {e}")
            raise
    
    async def _create_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """拡張特徴量作成"""
        try:
            # 基本特徴量
            enhanced_data = data.copy()
            
            # テクニカル特徴量
            enhanced_data = self.feature_engineering.create_technical_features(enhanced_data)
            
            # 市場特徴量
            enhanced_data = self.feature_engineering.create_market_features(enhanced_data)
            
            # 高度な特徴量
            enhanced_data = self.feature_engineering.create_advanced_features(enhanced_data)
            
            # 無限値・欠損値処理
            enhanced_data = enhanced_data.replace([np.inf, -np.inf], np.nan)
            enhanced_data = enhanced_data.fillna(method='ffill').fillna(0)
            
            self.performance_stats['feature_additions'] += len(enhanced_data.columns) - len(data.columns)
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"特徴量作成エラー: {e}")
            return data
    
    def _extract_market_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """市場コンテキスト抽出"""
        try:
            # ボラティリティ
            returns = data['close'].pct_change().dropna()
            volatility = returns.std()
            
            if volatility > 0.03:
                vol_regime = 'high'
            elif volatility < 0.01:
                vol_regime = 'low'
            else:
                vol_regime = 'normal'
            
            # トレンド
            price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
            if price_change > 0.05:
                trend = 'bullish'
            elif price_change < -0.05:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            # 出来高
            avg_volume = data['volume'].mean()
            recent_volume = data['volume'].iloc[-5:].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio > 1.5:
                volume_regime = 'high'
            elif volume_ratio < 0.5:
                volume_regime = 'low'
            else:
                volume_regime = 'normal'
            
            # 時間帯
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 11:
                time_of_day = 'morning'
            elif 12 <= current_hour <= 14:
                time_of_day = 'afternoon'
            else:
                time_of_day = 'irregular'
            
            return {
                'volatility': vol_regime,
                'trend': trend,
                'volume_regime': volume_regime,
                'time_of_day': time_of_day,
                'volatility_value': volatility,
                'trend_value': price_change,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            logger.error(f"市場コンテキスト抽出エラー: {e}")
            return {}
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_proba: np.ndarray) -> AccuracyMetrics:
        """メトリクス計算"""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # AUC計算（バイナリ分類の場合）
            try:
                auc = roc_auc_score(y_true, y_proba)
            except:
                auc = 0.0
            
            # MSE, MAE計算
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            # 信頼度判定
            confidence = self._determine_confidence_level(accuracy, {'overall_confidence': accuracy})
            
            return AccuracyMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_score=auc,
                mse=mse,
                mae=mae,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"メトリクス計算エラー: {e}")
            return AccuracyMetrics()
    
    def _determine_confidence_level(self, probability: float, 
                                   confidence_scores: Dict[str, float]) -> PredictionConfidence:
        """信頼度レベル判定"""
        overall_conf = confidence_scores.get('overall_confidence', 0.5)
        
        # 確率と信頼度スコアを組み合わせた総合評価
        combined_score = 0.6 * overall_conf + 0.4 * abs(probability - 0.5) * 2
        
        if combined_score >= 0.9:
            return PredictionConfidence.VERY_HIGH
        elif combined_score >= 0.8:
            return PredictionConfidence.HIGH
        elif combined_score >= 0.7:
            return PredictionConfidence.MEDIUM
        elif combined_score >= 0.6:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW
    
    async def _save_prediction_result(self, result: PredictionResult):
        """予測結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO predictions (
                        timestamp, symbol, model_type, prediction, probability,
                        confidence, features, accuracy_metrics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.timestamp.isoformat(),
                    result.symbol,
                    result.model_type.value,
                    result.prediction,
                    result.probability,
                    result.confidence.value,
                    json.dumps(result.features),
                    json.dumps(result.accuracy_metrics.__dict__, default=str)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"予測結果保存エラー: {e}")
    
    async def _save_model_performance(self, symbol: str, model_name: str, 
                                    metrics: AccuracyMetrics):
        """モデルパフォーマンス保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO model_performance (
                        timestamp, model_name, symbol, accuracy, precision_score,
                        recall_score, f1_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    model_name,
                    symbol,
                    metrics.accuracy,
                    metrics.precision,
                    metrics.recall,
                    metrics.f1_score
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"パフォーマンス保存エラー: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポート取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 最新の精度統計
                accuracy_query = '''
                    SELECT AVG(accuracy), AVG(precision_score), AVG(recall_score), AVG(f1_score)
                    FROM model_performance 
                    WHERE timestamp > datetime('now', '-7 days')
                '''
                accuracy_stats = conn.execute(accuracy_query).fetchone()
                
                # 予測数統計
                prediction_query = '''
                    SELECT COUNT(*), AVG(probability) 
                    FROM predictions 
                    WHERE timestamp > datetime('now', '-7 days')
                '''
                prediction_stats = conn.execute(prediction_query).fetchone()
                
                # 銘柄別統計
                symbol_query = '''
                    SELECT symbol, COUNT(*), AVG(accuracy) 
                    FROM model_performance 
                    WHERE timestamp > datetime('now', '-7 days')
                    GROUP BY symbol 
                    ORDER BY AVG(accuracy) DESC
                '''
                symbol_stats = conn.execute(symbol_query).fetchall()
                
            return {
                'system_stats': self.performance_stats,
                'accuracy_stats': {
                    'avg_accuracy': accuracy_stats[0] or 0.0,
                    'avg_precision': accuracy_stats[1] or 0.0,
                    'avg_recall': accuracy_stats[2] or 0.0,
                    'avg_f1': accuracy_stats[3] or 0.0
                },
                'prediction_stats': {
                    'total_predictions': prediction_stats[0] or 0,
                    'avg_probability': prediction_stats[1] or 0.0
                },
                'top_symbols': [
                    {'symbol': row[0], 'predictions': row[1], 'accuracy': row[2]}
                    for row in symbol_stats[:10]
                ],
                'ensemble_models': len(self.ensemble_system.base_models),
                'meta_models': len(self.ensemble_system.meta_models),
                'monitoring_active': self.accuracy_monitor.monitoring_active
            }
            
        except Exception as e:
            logger.error(f"パフォーマンスレポート取得エラー: {e}")
            return {'error': str(e)}

# 使用例とテスト用の関数
async def demo_prediction_system():
    """予測システムデモ"""
    try:
        # システム初期化
        system = AdvancedPredictionAccuracySystem("demo_prediction.db")
        
        # サンプルデータ作成
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')
        sample_data = pd.DataFrame({
            'close': np.cumsum(np.random.randn(1000) * 0.02) + 100,
            'volume': np.random.randint(1000, 10000, 1000),
            'high': np.cumsum(np.random.randn(1000) * 0.02) + 102,
            'low': np.cumsum(np.random.randn(1000) * 0.02) + 98,
            'target': np.random.randint(0, 2, 1000)  # バイナリターゲット
        }, index=dates)
        
        # モデル訓練
        training_result = await system.train_models('7203.T', sample_data)
        print(f"訓練結果: {training_result}")
        
        # 予測実行
        prediction_data = sample_data.iloc[-30:].copy()  # 最新30日分
        prediction_result = await system.predict('7203.T', prediction_data)
        
        print(f"予測結果:")
        print(f"  銘柄: {prediction_result.symbol}")
        print(f"  予測値: {prediction_result.prediction}")
        print(f"  確率: {prediction_result.probability:.3f}")
        print(f"  信頼度: {prediction_result.confidence.value}")
        print(f"  モデル: {prediction_result.model_type.value}")
        
        # パフォーマンスレポート
        report = system.get_performance_report()
        print(f"パフォーマンスレポート: {report}")
        
        return system
        
    except Exception as e:
        logger.error(f"デモ実行エラー: {e}")
        raise

if __name__ == "__main__":
    # デモ実行
    asyncio.run(demo_prediction_system())