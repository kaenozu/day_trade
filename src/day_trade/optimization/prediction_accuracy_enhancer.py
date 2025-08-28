#!/usr/bin/env python3
"""
Prediction Accuracy Enhancement System
予測精度向上システム

This module implements advanced techniques to improve prediction accuracy
through ensemble methods, feature engineering, and model optimization.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
import joblib
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, StackingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

from ..utils.error_handling import TradingResult


class AccuracyMetric(Enum):
    """予測精度評価メトリクス"""
    MSE = "mean_squared_error"
    RMSE = "root_mean_squared_error"
    MAE = "mean_absolute_error"
    MAPE = "mean_absolute_percentage_error"
    R2_SCORE = "r2_score"
    DIRECTIONAL_ACCURACY = "directional_accuracy"
    SHARPE_RATIO = "sharpe_ratio"
    INFORMATION_RATIO = "information_ratio"


class ModelType(Enum):
    """モデルタイプ"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NEURAL_NETWORK = "neural_network"
    SUPPORT_VECTOR = "support_vector"
    LINEAR_REGRESSION = "linear_regression"
    ENSEMBLE = "ensemble"


class FeatureImportanceMethod(Enum):
    """特徴量重要度算出方法"""
    TREE_BASED = "tree_based"
    PERMUTATION = "permutation"
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    LASSO_COEF = "lasso_coef"
    RFE = "recursive_feature_elimination"


@dataclass
class PredictionMetrics:
    """予測メトリクス"""
    mse: float
    rmse: float
    mae: float
    mape: float
    r2_score: float
    directional_accuracy: float
    sharpe_ratio: float
    information_ratio: float
    prediction_confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelPerformance:
    """モデルパフォーマンス"""
    model_id: str
    model_type: ModelType
    metrics: PredictionMetrics
    training_time: float
    prediction_time: float
    memory_usage: float
    feature_importance: Dict[str, float]
    hyperparameters: Dict[str, Any]
    validation_scores: List[float]
    cross_validation_score: float


@dataclass
class FeatureEngineeringConfig:
    """特徴量エンジニアリング設定"""
    technical_indicators: List[str]
    statistical_features: List[str]
    time_windows: List[int]
    lag_features: int
    rolling_features: List[str]
    momentum_features: bool
    volatility_features: bool
    seasonality_features: bool
    interaction_features: bool
    polynomial_features: int


class AdvancedFeatureEngine:
    """高度な特徴量エンジニアリング"""
    
    def __init__(self):
        self.feature_cache = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.logger = logging.getLogger(__name__)
        
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標の生成"""
        try:
            df = data.copy()
            
            # 移動平均
            for window in [5, 10, 20, 50]:
                df[f'sma_{window}'] = df['close'].rolling(window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
                
            # ボリンジャーバンド
            for window in [20, 50]:
                rolling_mean = df['close'].rolling(window).mean()
                rolling_std = df['close'].rolling(window).std()
                df[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
                df[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
                df[f'bb_width_{window}'] = df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']
                df[f'bb_ratio_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / df[f'bb_width_{window}']
                
            # RSI
            for window in [14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
                
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr_14'] = true_range.rolling(14).mean()
            
            # Stochastic Oscillator
            lowest_low = df['low'].rolling(14).min()
            highest_high = df['high'].rolling(14).max()
            df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_sma_20'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma_20']
                df['price_volume'] = df['close'] * df['volume']
                df['vwap'] = df['price_volume'].cumsum() / df['volume'].cumsum()
                
            return df
            
        except Exception as e:
            self.logger.error(f"Technical indicators creation failed: {e}")
            return data
    
    def create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """統計的特徴量の生成"""
        try:
            df = data.copy()
            
            # リターン系特徴量
            for lag in [1, 2, 5, 10]:
                df[f'return_{lag}d'] = df['close'].pct_change(lag)
                df[f'log_return_{lag}d'] = np.log(df['close'] / df['close'].shift(lag))
                
            # 統計的特徴量
            for window in [5, 10, 20]:
                returns = df['close'].pct_change()
                df[f'volatility_{window}d'] = returns.rolling(window).std()
                df[f'skewness_{window}d'] = returns.rolling(window).skew()
                df[f'kurtosis_{window}d'] = returns.rolling(window).kurt()
                df[f'max_drawdown_{window}d'] = self._calculate_max_drawdown(df['close'], window)
                
            # 価格位置特徴量
            for window in [10, 20, 50]:
                df[f'price_position_{window}d'] = (
                    (df['close'] - df['close'].rolling(window).min()) /
                    (df['close'].rolling(window).max() - df['close'].rolling(window).min())
                )
                
            # トレンド特徴量
            for window in [5, 10, 20]:
                df[f'trend_strength_{window}d'] = (
                    df['close'].rolling(window).apply(lambda x: stats.linregress(range(len(x)), x)[0])
                )
                
            return df
            
        except Exception as e:
            self.logger.error(f"Statistical features creation failed: {e}")
            return data
    
    def create_lag_features(self, data: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
        """ラグ特徴量の生成"""
        try:
            df = data.copy()
            
            target_cols = ['close', 'high', 'low', 'volume'] if 'volume' in df.columns else ['close', 'high', 'low']
            
            for col in target_cols:
                for lag in range(1, lags + 1):
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
            return df
            
        except Exception as e:
            self.logger.error(f"Lag features creation failed: {e}")
            return data
    
    def create_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """モメンタム特徴量の生成"""
        try:
            df = data.copy()
            
            # 価格モメンタム
            for window in [3, 5, 10, 20]:
                df[f'momentum_{window}d'] = df['close'] / df['close'].shift(window) - 1
                df[f'momentum_smoothed_{window}d'] = df[f'momentum_{window}d'].rolling(5).mean()
                
            # 加速度（モメンタムの変化率）
            for window in [5, 10]:
                momentum = df['close'].pct_change(window)
                df[f'acceleration_{window}d'] = momentum.diff()
                
            # 相対的強さ
            for window in [10, 20]:
                df[f'relative_strength_{window}d'] = (
                    df['close'].rolling(window).apply(
                        lambda x: (x > x.shift(1)).sum() / len(x)
                    )
                )
                
            return df
            
        except Exception as e:
            self.logger.error(f"Momentum features creation failed: {e}")
            return data
    
    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ特徴量の生成"""
        try:
            df = data.copy()
            
            returns = df['close'].pct_change()
            
            # 各種ボラティリティ指標
            for window in [5, 10, 20, 30]:
                # 単純標準偏差
                df[f'volatility_{window}d'] = returns.rolling(window).std()
                
                # EWMA volatility
                df[f'ewma_volatility_{window}d'] = returns.ewm(span=window).std()
                
                # Parkinson volatility (高値・安値を使用)
                if 'high' in df.columns and 'low' in df.columns:
                    parkinson_vol = np.sqrt(
                        np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))
                    ).rolling(window).mean()
                    df[f'parkinson_volatility_{window}d'] = parkinson_vol
                
                # Volatility of volatility
                df[f'vol_of_vol_{window}d'] = df[f'volatility_{window}d'].rolling(window).std()
                
            # GARCH-like features
            for window in [10, 20]:
                squared_returns = returns ** 2
                df[f'garch_like_{window}d'] = squared_returns.ewm(span=window).mean()
                
            return df
            
        except Exception as e:
            self.logger.error(f"Volatility features creation failed: {e}")
            return data
    
    def create_seasonality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """季節性特徴量の生成"""
        try:
            df = data.copy()
            
            if df.index.dtype != 'datetime64[ns]':
                df.index = pd.to_datetime(df.index)
                
            # 時間系特徴量
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['year'] = df.index.year
            
            # 周期性特徴量
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # 取引時間フラグ
            df['is_trading_hour'] = ((df['hour'] >= 9) & (df['hour'] < 15)).astype(int)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Seasonality features creation failed: {e}")
            return data
    
    def create_interaction_features(self, data: pd.DataFrame, max_features: int = 50) -> pd.DataFrame:
        """交互作用特徴量の生成"""
        try:
            df = data.copy()
            
            # 主要な特徴量を選択
            main_features = ['close', 'high', 'low', 'volume'] if 'volume' in df.columns else ['close', 'high', 'low']
            
            # 既存の数値特徴量から上位を選択
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            correlation_with_close = df[numeric_cols].corrwith(df['close']).abs().sort_values(ascending=False)
            top_features = correlation_with_close.head(min(10, len(correlation_with_close))).index.tolist()
            
            interaction_count = 0
            for i, feat1 in enumerate(top_features):
                if interaction_count >= max_features:
                    break
                for feat2 in top_features[i+1:]:
                    if interaction_count >= max_features:
                        break
                    
                    # 乗算交互作用
                    df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                    interaction_count += 1
                    
                    # 比率交互作用（ゼロ除算回避）
                    if (df[feat2] != 0).all():
                        df[f'{feat1}_div_{feat2}'] = df[feat1] / df[feat2]
                        interaction_count += 1
                    
            return df
            
        except Exception as e:
            self.logger.error(f"Interaction features creation failed: {e}")
            return data
    
    def _calculate_max_drawdown(self, prices: pd.Series, window: int) -> pd.Series:
        """最大ドローダウンの計算"""
        try:
            rolling_max = prices.rolling(window).max()
            drawdown = (prices - rolling_max) / rolling_max
            return drawdown.rolling(window).min()
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'correlation', k: int = 50) -> List[str]:
        """特徴量選択"""
        try:
            if method == 'correlation':
                # 相関ベースの選択
                correlations = X.corrwith(y).abs().sort_values(ascending=False)
                return correlations.head(k).index.tolist()
                
            elif method == 'mutual_info':
                from sklearn.feature_selection import mutual_info_regression
                mi_scores = mutual_info_regression(X.fillna(0), y)
                feature_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
                return feature_scores.head(k).index.tolist()
                
            elif method == 'rfe':
                from sklearn.linear_model import LinearRegression
                estimator = LinearRegression()
                rfe = RFE(estimator, n_features_to_select=k)
                rfe.fit(X.fillna(0), y)
                return X.columns[rfe.support_].tolist()
                
            elif method == 'lasso':
                from sklearn.linear_model import LassoCV
                lasso = LassoCV(cv=5, random_state=42)
                lasso.fit(X.fillna(0), y)
                feature_importance = pd.Series(np.abs(lasso.coef_), index=X.columns)
                return feature_importance.sort_values(ascending=False).head(k).index.tolist()
                
            else:
                # デフォルトは分散ベース
                variances = X.var().sort_values(ascending=False)
                return variances.head(k).index.tolist()
                
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            return X.columns.tolist()[:k]


class AdvancedModelOptimizer:
    """高度なモデル最適化"""
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.optimization_history = []
        self.logger = logging.getLogger(__name__)
        
    def create_base_models(self) -> Dict[str, Any]:
        """ベースモデルの作成"""
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            ),
            'support_vector': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            ),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5)
        }
        return models
    
    def optimize_hyperparameters(self, model_name: str, X_train: pd.DataFrame, 
                                y_train: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """ハイパーパラメータ最適化"""
        try:
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5, 10]
                },
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'lightgbm': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'neural_network': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                    'activation': ['relu', 'tanh'],
                    'learning_rate_init': [0.001, 0.01, 0.1],
                    'alpha': [0.0001, 0.001, 0.01]
                },
                'support_vector': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'poly', 'sigmoid']
                },
                'ridge': {
                    'alpha': [0.1, 1, 10, 100, 1000]
                },
                'lasso': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10]
                },
                'elastic_net': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            }
            
            if model_name not in param_grids:
                return {}
            
            base_model = self.create_base_models()[model_name]
            param_grid = param_grids[model_name]
            
            # Time Series Cross Validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            # 計算時間を考慮してRandomizedSearchを使用
            if len(param_grid) > 20:
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=20,
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    random_state=42
                )
            else:
                search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
            
            search.fit(X_train.fillna(0), y_train)
            
            self.best_params[model_name] = search.best_params_
            
            return {
                'best_params': search.best_params_,
                'best_score': -search.best_score_,
                'cv_results': search.cv_results_
            }
            
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed for {model_name}: {e}")
            return {}
    
    def create_ensemble_model(self, base_models: Dict[str, Any], method: str = 'voting') -> Any:
        """アンサンブルモデルの作成"""
        try:
            if method == 'voting':
                return VotingRegressor(
                    estimators=list(base_models.items()),
                    n_jobs=-1
                )
            elif method == 'stacking':
                return StackingRegressor(
                    estimators=list(base_models.items()),
                    final_estimator=Ridge(),
                    n_jobs=-1
                )
            else:
                return VotingRegressor(
                    estimators=list(base_models.items()),
                    n_jobs=-1
                )
                
        except Exception as e:
            self.logger.error(f"Ensemble model creation failed: {e}")
            return None


class PredictionAccuracyEnhancer:
    """予測精度向上システム"""
    
    def __init__(self):
        self.feature_engine = AdvancedFeatureEngine()
        self.model_optimizer = AdvancedModelOptimizer()
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.performance_history = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
        
    async def enhance_prediction_system(self, 
                                      data: pd.DataFrame, 
                                      target_column: str,
                                      config: FeatureEngineeringConfig) -> TradingResult[Dict[str, Any]]:
        """予測システムの包括的強化"""
        try:
            self.logger.info("Starting prediction system enhancement...")
            
            # データ準備
            enhanced_data = await self._prepare_enhanced_features(data, config)
            
            if enhanced_data.empty:
                return TradingResult.failure("Feature engineering failed")
            
            # ターゲット変数の準備
            if target_column not in enhanced_data.columns:
                return TradingResult.failure(f"Target column {target_column} not found")
            
            # 訓練・テストデータ分割
            train_size = int(len(enhanced_data) * 0.8)
            train_data = enhanced_data[:train_size]
            test_data = enhanced_data[train_size:]
            
            # 特徴量選択
            feature_cols = [col for col in enhanced_data.columns if col != target_column]
            X_train = train_data[feature_cols]
            y_train = train_data[target_column]
            X_test = test_data[feature_cols]
            y_test = test_data[target_column]
            
            # 特徴量選択の実行
            selected_features = self.feature_engine.select_features(
                X_train, y_train, method='correlation', k=min(50, len(feature_cols))
            )
            
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            # データ正規化
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train_selected.fillna(0)),
                columns=selected_features,
                index=X_train_selected.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test_selected.fillna(0)),
                columns=selected_features,
                index=X_test_selected.index
            )
            
            # モデル訓練と最適化
            model_performances = await self._train_and_optimize_models(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            
            # 最良モデルの選択
            best_model_info = max(model_performances, key=lambda x: x.metrics.r2_score)
            
            # アンサンブルモデルの作成
            ensemble_performance = await self._create_ensemble_model(
                model_performances, X_train_scaled, y_train, X_test_scaled, y_test
            )
            
            # 結果の集計
            enhancement_results = {
                'original_features': len(feature_cols),
                'selected_features': len(selected_features),
                'feature_names': selected_features,
                'model_performances': [
                    {
                        'model_type': perf.model_type.value,
                        'metrics': {
                            'mse': perf.metrics.mse,
                            'rmse': perf.metrics.rmse,
                            'mae': perf.metrics.mae,
                            'r2_score': perf.metrics.r2_score,
                            'directional_accuracy': perf.metrics.directional_accuracy
                        },
                        'training_time': perf.training_time,
                        'cross_validation_score': perf.cross_validation_score
                    }
                    for perf in model_performances
                ],
                'best_model': {
                    'model_type': best_model_info.model_type.value,
                    'metrics': {
                        'mse': best_model_info.metrics.mse,
                        'rmse': best_model_info.metrics.rmse,
                        'mae': best_model_info.metrics.mae,
                        'r2_score': best_model_info.metrics.r2_score,
                        'directional_accuracy': best_model_info.metrics.directional_accuracy
                    },
                    'feature_importance': dict(list(best_model_info.feature_importance.items())[:10])
                },
                'ensemble_performance': ensemble_performance,
                'improvement_summary': self._calculate_improvement_metrics(model_performances)
            }
            
            self.logger.info("Prediction system enhancement completed successfully")
            return TradingResult.success(enhancement_results)
            
        except Exception as e:
            self.logger.error(f"Prediction enhancement failed: {e}")
            return TradingResult.failure(f"Enhancement error: {e}")
    
    async def _prepare_enhanced_features(self, data: pd.DataFrame, 
                                       config: FeatureEngineeringConfig) -> pd.DataFrame:
        """強化された特徴量の準備"""
        try:
            df = data.copy()
            
            # テクニカル指標
            if config.technical_indicators:
                df = self.feature_engine.create_technical_indicators(df)
            
            # 統計的特徴量
            if config.statistical_features:
                df = self.feature_engine.create_statistical_features(df)
            
            # ラグ特徴量
            if config.lag_features > 0:
                df = self.feature_engine.create_lag_features(df, config.lag_features)
            
            # モメンタム特徴量
            if config.momentum_features:
                df = self.feature_engine.create_momentum_features(df)
            
            # ボラティリティ特徴量
            if config.volatility_features:
                df = self.feature_engine.create_volatility_features(df)
            
            # 季節性特徴量
            if config.seasonality_features:
                df = self.feature_engine.create_seasonality_features(df)
            
            # 交互作用特徴量
            if config.interaction_features:
                df = self.feature_engine.create_interaction_features(df)
            
            # NaN値の処理
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # 無限値の処理
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Enhanced features preparation failed: {e}")
            return pd.DataFrame()
    
    async def _train_and_optimize_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                       X_test: pd.DataFrame, y_test: pd.Series) -> List[ModelPerformance]:
        """モデル訓練と最適化"""
        try:
            base_models = self.model_optimizer.create_base_models()
            performances = []
            
            for model_name, model in base_models.items():
                self.logger.info(f"Training and optimizing {model_name}...")
                
                start_time = time.time()
                
                # ハイパーパラメータ最適化
                optimization_result = self.model_optimizer.optimize_hyperparameters(
                    model_name, X_train, y_train, cv_folds=5
                )
                
                # 最適化されたモデルで訓練
                if optimization_result and 'best_params' in optimization_result:
                    model.set_params(**optimization_result['best_params'])
                
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # 予測
                start_pred_time = time.time()
                y_pred = model.predict(X_test)
                prediction_time = time.time() - start_pred_time
                
                # メトリクス計算
                metrics = self._calculate_metrics(y_test, y_pred)
                
                # 特徴量重要度
                feature_importance = self._get_feature_importance(model, X_train.columns)
                
                # クロスバリデーション
                cv_score = optimization_result.get('best_score', 0) if optimization_result else 0
                
                performance = ModelPerformance(
                    model_id=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_type=ModelType(model_name) if model_name in [mt.value for mt in ModelType] else ModelType.LINEAR_REGRESSION,
                    metrics=metrics,
                    training_time=training_time,
                    prediction_time=prediction_time,
                    memory_usage=0.0,  # 実装時に計算
                    feature_importance=feature_importance,
                    hyperparameters=optimization_result.get('best_params', {}) if optimization_result else {},
                    validation_scores=[],
                    cross_validation_score=cv_score
                )
                
                performances.append(performance)
                
            return performances
            
        except Exception as e:
            self.logger.error(f"Model training and optimization failed: {e}")
            return []
    
    async def _create_ensemble_model(self, model_performances: List[ModelPerformance],
                                   X_train: pd.DataFrame, y_train: pd.Series,
                                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """アンサンブルモデルの作成"""
        try:
            # 上位モデルを選択
            top_models = sorted(model_performances, key=lambda x: x.metrics.r2_score, reverse=True)[:5]
            
            # ベースモデルを再構築
            base_models = {}
            for perf in top_models:
                model_type = perf.model_type.value
                base_model = self.model_optimizer.create_base_models()[model_type]
                base_model.set_params(**perf.hyperparameters)
                base_model.fit(X_train, y_train)
                base_models[model_type] = base_model
            
            # アンサンブルモデル作成
            ensemble_model = self.model_optimizer.create_ensemble_model(base_models, method='voting')
            
            start_time = time.time()
            ensemble_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # 予測
            y_pred_ensemble = ensemble_model.predict(X_test)
            
            # メトリクス計算
            ensemble_metrics = self._calculate_metrics(y_test, y_pred_ensemble)
            
            return {
                'metrics': {
                    'mse': ensemble_metrics.mse,
                    'rmse': ensemble_metrics.rmse,
                    'mae': ensemble_metrics.mae,
                    'r2_score': ensemble_metrics.r2_score,
                    'directional_accuracy': ensemble_metrics.directional_accuracy
                },
                'training_time': training_time,
                'component_models': [perf.model_type.value for perf in top_models],
                'model_weights': 'equal'  # Voting regressorは等重み
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble model creation failed: {e}")
            return {}
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> PredictionMetrics:
        """予測メトリクスの計算"""
        try:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # R² Score
            r2 = r2_score(y_true, y_pred)
            
            # Directional Accuracy
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
            
            # Sharpe Ratio (returns as proxy)
            returns = y_pred / np.roll(y_pred, 1) - 1
            returns = returns[1:]  # Remove first NaN
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
            
            # Information Ratio
            excess_returns = y_pred - y_true
            information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0
            
            return PredictionMetrics(
                mse=mse,
                rmse=rmse,
                mae=mae,
                mape=mape,
                r2_score=r2,
                directional_accuracy=directional_accuracy,
                sharpe_ratio=sharpe_ratio,
                information_ratio=information_ratio,
                prediction_confidence=r2  # R²を信頼度の代理として使用
            )
            
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            return PredictionMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """特徴量重要度の取得"""
        try:
            importance_dict = {}
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                for name, importance in zip(feature_names, importances):
                    importance_dict[name] = float(importance)
                    
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = np.abs(model.coef_)
                for name, coef in zip(feature_names, coefficients):
                    importance_dict[name] = float(coef)
                    
            else:
                # Default: equal importance
                for name in feature_names:
                    importance_dict[name] = 1.0 / len(feature_names)
            
            # Normalize to sum to 1
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
            
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {e}")
            return {}
    
    def _calculate_improvement_metrics(self, performances: List[ModelPerformance]) -> Dict[str, Any]:
        """改善メトリクスの計算"""
        try:
            if not performances:
                return {}
            
            r2_scores = [p.metrics.r2_score for p in performances]
            mae_scores = [p.metrics.mae for p in performances]
            directional_accuracies = [p.metrics.directional_accuracy for p in performances]
            
            return {
                'best_r2_score': max(r2_scores),
                'average_r2_score': np.mean(r2_scores),
                'best_mae': min(mae_scores),
                'average_mae': np.mean(mae_scores),
                'best_directional_accuracy': max(directional_accuracies),
                'average_directional_accuracy': np.mean(directional_accuracies),
                'model_count': len(performances),
                'training_time_total': sum(p.training_time for p in performances)
            }
            
        except Exception as e:
            self.logger.error(f"Improvement metrics calculation failed: {e}")
            return {}


# Global instance
prediction_enhancer = PredictionAccuracyEnhancer()


async def enhance_prediction_accuracy(data: pd.DataFrame, 
                                    target_column: str,
                                    config: Optional[FeatureEngineeringConfig] = None) -> TradingResult[Dict[str, Any]]:
    """予測精度向上の実行"""
    if config is None:
        config = FeatureEngineeringConfig(
            technical_indicators=['sma', 'ema', 'rsi', 'macd', 'bb'],
            statistical_features=['volatility', 'returns', 'momentum'],
            time_windows=[5, 10, 20, 50],
            lag_features=10,
            rolling_features=['mean', 'std', 'min', 'max'],
            momentum_features=True,
            volatility_features=True,
            seasonality_features=True,
            interaction_features=True,
            polynomial_features=2
        )
    
    return await prediction_enhancer.enhance_prediction_system(data, target_column, config)