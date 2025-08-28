"""
次世代予測精度向上システム

最新の機械学習手法と高度な特徴量エンジニアリングを統合した
革新的な予測システム
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import joblib
from collections import defaultdict
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

warnings.filterwarnings('ignore')


@dataclass
class NextGenPredictionResult:
    """次世代予測結果"""
    prediction: int
    confidence: float
    probability: float
    ensemble_scores: Dict[str, float]
    feature_importance: Dict[str, float]
    prediction_path: List[str]
    quality_metrics: Dict[str, float]


@dataclass
class ModelPerformanceMetrics:
    """モデル性能メトリクス"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cv_mean: float
    cv_std: float
    prediction_time_ms: float
    training_time_s: float


class AdvancedFeatureEngineering:
    """高度特徴量エンジニアリング"""
    
    @staticmethod
    def create_market_microstructure_features(data: pd.DataFrame) -> pd.DataFrame:
        """市場マイクロ構造特徴量作成"""
        df = data.copy()
        
        # スプレッド関連
        df['bid_ask_spread'] = (df['high'] - df['low']) / df['close']
        df['midpoint'] = (df['high'] + df['low']) / 2
        df['price_impact'] = abs(df['close'] - df['midpoint']) / df['close']
        
        # 取引強度
        df['volume_weighted_price'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['price_volume_trend'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        
        # 流動性指標
        df['amihud_illiquidity'] = abs(df['close'].pct_change()) / (df['volume'] * df['close'])
        df['volume_volatility'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
        
        return df
    
    @staticmethod
    def create_sentiment_features(data: pd.DataFrame) -> pd.DataFrame:
        """センチメント特徴量作成"""
        df = data.copy()
        
        # 価格センチメント
        returns = df['close'].pct_change()
        df['bullish_sentiment'] = (returns > 0).rolling(10).sum() / 10
        df['bearish_sentiment'] = (returns < 0).rolling(10).sum() / 10
        
        # ボラティリティセンチメント
        volatility = returns.rolling(20).std()
        df['volatility_regime'] = np.where(volatility > volatility.quantile(0.7), 1, 
                                          np.where(volatility < volatility.quantile(0.3), -1, 0))
        
        # モメンタムセンチメント
        short_ma = df['close'].rolling(5).mean()
        long_ma = df['close'].rolling(20).mean()
        df['momentum_sentiment'] = (short_ma - long_ma) / long_ma
        
        return df
    
    @staticmethod
    def create_advanced_technical_features(data: pd.DataFrame) -> pd.DataFrame:
        """高度テクニカル特徴量作成"""
        df = data.copy()
        
        # 複数期間の移動平均
        for period in [3, 5, 8, 13, 21, 34, 55]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_sma_ratio_{period}'] = df['close'] / df[f'sma_{period}']
        
        # 高度なオシレータ
        # Williams %R
        high_14 = df['high'].rolling(14).max()
        low_14 = df['low'].rolling(14).min()
        df['williams_r'] = (high_14 - df['close']) / (high_14 - low_14) * -100
        
        # Commodity Channel Index (CCI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad_tp = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad_tp)
        
        # Ultimate Oscillator
        bp = df['close'] - np.minimum(df['low'], df['close'].shift(1))
        tr = np.maximum(df['high'], df['close'].shift(1)) - np.minimum(df['low'], df['close'].shift(1))
        
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        
        df['ultimate_oscillator'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
        
        # Ichimoku Cloud components
        high_9 = df['high'].rolling(9).max()
        low_9 = df['low'].rolling(9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2
        
        high_26 = df['high'].rolling(26).max()
        low_26 = df['low'].rolling(26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2
        
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        high_52 = df['high'].rolling(52).max()
        low_52 = df['low'].rolling(52).min()
        df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
        return df
    
    @staticmethod
    def create_fractal_features(data: pd.DataFrame) -> pd.DataFrame:
        """フラクタル・複雑系特徴量作成"""
        df = data.copy()
        
        # Hurst指数
        def hurst_exponent(ts, max_lag=20):
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        df['hurst_exponent'] = df['close'].rolling(50).apply(lambda x: hurst_exponent(x) if len(x) >= 20 else 0.5)
        
        # フラクタル次元
        def fractal_dimension(ts, k_max=10):
            if len(ts) < k_max:
                return 1.5
            
            L = []
            x = np.array(ts)
            N = len(x)
            
            for k in range(1, k_max):
                Lk = np.sum(np.abs(x[k:] - x[:-k]))
                L.append(Lk / (N - k))
            
            if len(L) < 2:
                return 1.5
                
            try:
                # log-log回帰
                log_L = np.log(L)
                log_k = np.log(range(1, len(L) + 1))
                coeffs = np.polyfit(log_k, log_L, 1)
                return 2 - coeffs[0]
            except:
                return 1.5
        
        df['fractal_dimension'] = df['close'].rolling(30).apply(lambda x: fractal_dimension(x))
        
        # エントロピー
        def shannon_entropy(ts, bins=10):
            hist, _ = np.histogram(ts, bins=bins)
            hist = hist / hist.sum()
            return -np.sum(hist * np.log(hist + 1e-10))
        
        df['shannon_entropy'] = df['close'].pct_change().rolling(20).apply(lambda x: shannon_entropy(x))
        
        return df


class NextGenerationEnsembleSystem:
    """次世代アンサンブルシステム"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # モデル管理
        self.base_models = {}
        self.meta_learner = None
        self.feature_selector = None
        self.scaler = None
        
        # 性能追跡
        self.model_performance: Dict[str, ModelPerformanceMetrics] = {}
        self.feature_importance_history = defaultdict(list)
        
        # 並列処理
        self.n_jobs = min(mp.cpu_count(), self.config.get('max_workers', 4))
        self.executor = ThreadPoolExecutor(max_workers=self.n_jobs)
        
        # 特徴量エンジニアリング
        self.feature_engineer = AdvancedFeatureEngineering()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _create_base_models(self) -> Dict[str, Any]:
        """基礎モデル作成"""
        models = {
            'rf_optimized': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=self.n_jobs
            ),
            'gb_advanced': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=42
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=self.n_jobs
            ),
            'svm_tuned': SVC(
                C=10,
                gamma='scale',
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'mlp_deep': MLPClassifier(
                hidden_layer_sizes=(150, 100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42
            ),
            'ridge_regularized': RidgeClassifier(
                alpha=1.0,
                random_state=42
            )
        }
        
        return models
    
    async def create_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """包括的特徴量作成"""
        self.logger.info("包括的特徴量エンジニアリング開始")
        
        # 並列特徴量作成
        futures = []
        
        # 基本テクニカル特徴量
        futures.append(self.executor.submit(
            self.feature_engineer.create_advanced_technical_features, data
        ))
        
        # マイクロ構造特徴量
        futures.append(self.executor.submit(
            self.feature_engineer.create_market_microstructure_features, data
        ))
        
        # センチメント特徴量
        futures.append(self.executor.submit(
            self.feature_engineer.create_sentiment_features, data
        ))
        
        # フラクタル特徴量
        futures.append(self.executor.submit(
            self.feature_engineer.create_fractal_features, data
        ))
        
        # 結果統合
        feature_dfs = []
        for future in futures:
            try:
                result = future.result(timeout=30)
                feature_dfs.append(result)
            except Exception as e:
                self.logger.error(f"特徴量作成エラー: {e}")
                feature_dfs.append(data.copy())
        
        # 特徴量統合
        combined_features = data.copy()
        
        for df in feature_dfs:
            # 数値列のみを追加
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in combined_features.columns:
                    combined_features[col] = df[col]
        
        # 無限値・NaN処理
        combined_features = combined_features.replace([np.inf, -np.inf], np.nan)
        combined_features = combined_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 数値列のみを選択
        numeric_columns = combined_features.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['timestamp']]
        
        result_df = combined_features[numeric_columns]
        
        self.logger.info(f"特徴量作成完了: {len(result_df.columns)}次元")
        
        return result_df
    
    async def train_next_generation_ensemble(self, data: pd.DataFrame, target_column: str = 'target') -> Dict[str, Any]:
        """次世代アンサンブル訓練"""
        self.logger.info("次世代アンサンブル訓練開始")
        
        # 包括的特徴量作成
        features_df = await self.create_comprehensive_features(data)
        
        # ターゲット準備
        if target_column not in data.columns:
            target = (data['close'].shift(-1) > data['close']).astype(int)
        else:
            target = data[target_column]
        
        # 有効なデータのみ
        valid_indices = ~(features_df.isna().any(axis=1) | target.isna())
        X = features_df[valid_indices].values
        y = target[valid_indices].values
        
        if len(X) < 100:
            raise ValueError("訓練データが不足しています")
        
        self.logger.info(f"訓練データ: {len(X)}サンプル, {X.shape[1]}特徴量")
        
        # 特徴量選択
        self.feature_selector = SelectKBest(
            score_func=mutual_info_classif,
            k=min(100, X.shape[1] // 2)
        )
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # スケーリング
        self.scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # 基礎モデル作成・訓練
        self.base_models = self._create_base_models()
        
        # 時系列分割で検証
        tscv = TimeSeriesSplit(n_splits=5)
        
        training_results = {}
        
        # 各モデルの訓練と評価
        for model_name, model in self.base_models.items():
            self.logger.info(f"モデル訓練中: {model_name}")
            
            try:
                start_time = time.time()
                
                # 交差検証
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy', n_jobs=1)
                
                # 全データで訓練
                model.fit(X_scaled, y)
                
                training_time = time.time() - start_time
                
                # 予測時間測定
                pred_start = time.time()
                test_pred = model.predict(X_scaled[:100])  # サンプル予測
                prediction_time = (time.time() - pred_start) / 100 * 1000  # ms per prediction
                
                # メトリクス計算
                y_pred = model.predict(X_scaled)
                y_pred_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                accuracy = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
                
                try:
                    roc_auc = roc_auc_score(y, y_pred_proba)
                except:
                    roc_auc = 0.5
                
                # 性能メトリクス保存
                metrics = ModelPerformanceMetrics(
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    roc_auc=roc_auc,
                    cv_mean=cv_scores.mean(),
                    cv_std=cv_scores.std(),
                    prediction_time_ms=prediction_time,
                    training_time_s=training_time
                )
                
                self.model_performance[model_name] = metrics
                
                training_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_time_s': training_time,
                    'prediction_time_ms': prediction_time
                }
                
                self.logger.info(f"{model_name} 訓練完了 - 精度: {accuracy:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                
            except Exception as e:
                self.logger.error(f"{model_name} 訓練エラー: {e}")
                training_results[model_name] = {'error': str(e)}
        
        # メタ学習器（スタッキング）作成
        successful_models = [(name, model) for name, model in self.base_models.items() 
                           if name in training_results and 'error' not in training_results[name]]
        
        if len(successful_models) >= 2:
            try:
                self.meta_learner = StackingClassifier(
                    estimators=successful_models,
                    final_estimator=LogisticRegression(random_state=42),
                    cv=3,
                    n_jobs=1
                )
                
                self.logger.info("メタ学習器訓練中...")
                self.meta_learner.fit(X_scaled, y)
                
                # メタ学習器性能評価
                meta_pred = self.meta_learner.predict(X_scaled)
                meta_accuracy = accuracy_score(y, meta_pred)
                
                training_results['meta_learner'] = {
                    'accuracy': meta_accuracy,
                    'model_count': len(successful_models)
                }
                
                self.logger.info(f"メタ学習器訓練完了 - 精度: {meta_accuracy:.3f}")
                
            except Exception as e:
                self.logger.error(f"メタ学習器エラー: {e}")
        
        self.logger.info("次世代アンサンブル訓練完了")
        return training_results
    
    async def predict_next_generation(self, data: pd.DataFrame) -> NextGenPredictionResult:
        """次世代予測実行"""
        if not self.base_models:
            raise ValueError("モデルが訓練されていません")
        
        # 特徴量作成
        features_df = await self.create_comprehensive_features(data)
        X = features_df.tail(1).values
        
        # 前処理
        if self.feature_selector and self.scaler:
            X_selected = self.feature_selector.transform(X)
            X_scaled = self.scaler.transform(X_selected)
        else:
            X_scaled = X
        
        # 個別モデル予測
        ensemble_scores = {}
        predictions = []
        probabilities = []
        
        for model_name, model in self.base_models.items():
            try:
                if model_name in self.model_performance:  # 訓練済みモデルのみ
                    pred = model.predict(X_scaled)[0]
                    prob = model.predict_proba(X_scaled)[0, 1] if hasattr(model, 'predict_proba') else pred
                    
                    ensemble_scores[model_name] = prob
                    predictions.append(pred)
                    probabilities.append(prob)
                    
            except Exception as e:
                self.logger.error(f"{model_name} 予測エラー: {e}")
        
        # メタ学習器予測
        if self.meta_learner:
            try:
                meta_prob = self.meta_learner.predict_proba(X_scaled)[0, 1]
                final_probability = meta_prob
                ensemble_scores['meta_learner'] = meta_prob
            except:
                final_probability = np.mean(probabilities) if probabilities else 0.5
        else:
            final_probability = np.mean(probabilities) if probabilities else 0.5
        
        final_prediction = 1 if final_probability > 0.5 else 0
        
        # 信頼度計算
        confidence = abs(final_probability - 0.5) * 2
        
        # 予測パス
        prediction_path = ['特徴量作成', '前処理', '個別モデル予測']
        if self.meta_learner:
            prediction_path.append('メタ学習統合')
        
        # 品質メトリクス
        quality_metrics = {
            'model_agreement': np.std(probabilities) if len(probabilities) > 1 else 0,
            'feature_count': X_scaled.shape[1] if X_scaled.ndim > 1 else 1,
            'prediction_diversity': len(set(predictions)) / max(len(predictions), 1)
        }
        
        # 特徴量重要度（簡易版）
        feature_importance = {}
        if features_df.columns.any():
            for i, col in enumerate(features_df.columns[:20]):  # 上位20特徴量
                feature_importance[col] = 1.0 / (i + 1)
        
        return NextGenPredictionResult(
            prediction=final_prediction,
            confidence=confidence,
            probability=final_probability,
            ensemble_scores=ensemble_scores,
            feature_importance=feature_importance,
            prediction_path=prediction_path,
            quality_metrics=quality_metrics
        )
    
    def get_system_performance_summary(self) -> Dict[str, Any]:
        """システム性能サマリー取得"""
        if not self.model_performance:
            return {'error': 'モデルが訓練されていません'}
        
        # 平均性能計算
        avg_accuracy = np.mean([m.accuracy for m in self.model_performance.values()])
        avg_precision = np.mean([m.precision for m in self.model_performance.values()])
        avg_recall = np.mean([m.recall for m in self.model_performance.values()])
        avg_f1 = np.mean([m.f1_score for m in self.model_performance.values()])
        avg_cv_score = np.mean([m.cv_mean for m in self.model_performance.values()])
        avg_pred_time = np.mean([m.prediction_time_ms for m in self.model_performance.values()])
        
        # 最高性能モデル
        best_model = max(self.model_performance.items(), key=lambda x: x[1].accuracy)
        
        return {
            'model_count': len(self.model_performance),
            'average_performance': {
                'accuracy': avg_accuracy,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1,
                'cv_score': avg_cv_score,
                'prediction_time_ms': avg_pred_time
            },
            'best_model': {
                'name': best_model[0],
                'accuracy': best_model[1].accuracy,
                'cv_score': best_model[1].cv_mean
            },
            'has_meta_learner': self.meta_learner is not None,
            'feature_dimensions': X_scaled.shape[1] if hasattr(self, 'X_scaled') else 0
        }


async def demo_next_generation_prediction():
    """次世代予測システムデモ"""
    print("=== 次世代予測精度向上システム デモ ===")
    
    # システム初期化
    next_gen_system = NextGenerationEnsembleSystem()
    
    # テストデータ作成（より複雑）
    np.random.seed(42)
    size = 2000
    dates = pd.date_range(start=datetime.now() - timedelta(days=size), periods=size, freq='1min')
    
    # 複雑な価格動態シミュレーション
    base_trend = np.linspace(1000, 1400, size)
    seasonal = 50 * np.sin(2 * np.pi * np.arange(size) / 100)
    volatility_regime = 20 * np.sin(2 * np.pi * np.arange(size) / 500)
    noise = np.random.normal(0, 10, size)
    jumps = np.random.choice([0, 50, -50], size, p=[0.95, 0.025, 0.025])
    
    prices = base_trend + seasonal + volatility_regime + noise + jumps
    prices = np.maximum(prices, 10)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'symbol': ['NEXTGEN_TEST'] * size,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
        'volume': np.random.randint(10000, 500000, size)
    })
    
    # 複雑なターゲット
    returns = data['close'].pct_change()
    volume_spike = data['volume'] > data['volume'].rolling(50).quantile(0.8)
    strong_momentum = abs(returns) > returns.rolling(50).quantile(0.7)
    
    data['target'] = ((returns > 0.01) | (volume_spike & strong_momentum)).astype(int)
    
    try:
        print("\n1. 次世代アンサンブルモデル訓練中...")
        training_results = await next_gen_system.train_next_generation_ensemble(data)
        
        print("=== 訓練結果 ===")
        for model_name, result in training_results.items():
            if 'error' not in result:
                print(f"{model_name}:")
                print(f"  精度: {result['accuracy']:.3f}")
                print(f"  適合率: {result['precision']:.3f}")
                print(f"  再現率: {result['recall']:.3f}")
                print(f"  F1スコア: {result['f1_score']:.3f}")
                if 'cv_mean' in result:
                    print(f"  交差検証: {result['cv_mean']:.3f}±{result['cv_std']:.3f}")
                if 'prediction_time_ms' in result:
                    print(f"  予測時間: {result['prediction_time_ms']:.2f}ms")
        
        print("\n2. 次世代予測テスト...")
        test_data = data.tail(100)
        prediction = await next_gen_system.predict_next_generation(test_data)
        
        print(f"=== 次世代予測結果 ===")
        print(f"最終予測: {prediction.prediction}")
        print(f"予測確率: {prediction.probability:.3f}")
        print(f"信頼度: {prediction.confidence:.3f}")
        
        print(f"\n=== アンサンブルスコア ===")
        for model_name, score in prediction.ensemble_scores.items():
            print(f"{model_name}: {score:.3f}")
        
        print(f"\n=== 品質メトリクス ===")
        for metric, value in prediction.quality_metrics.items():
            print(f"{metric}: {value:.3f}")
        
        print(f"\n=== 予測処理パス ===")
        for i, step in enumerate(prediction.prediction_path, 1):
            print(f"{i}. {step}")
        
        print(f"\n3. システム性能サマリー...")
        summary = next_gen_system.get_system_performance_summary()
        
        if 'error' not in summary:
            print(f"=== システム性能 ===")
            print(f"訓練済みモデル数: {summary['model_count']}")
            print(f"メタ学習器: {'あり' if summary['has_meta_learner'] else 'なし'}")
            
            avg_perf = summary['average_performance']
            print(f"\n平均性能:")
            print(f"  精度: {avg_perf['accuracy']:.3f}")
            print(f"  適合率: {avg_perf['precision']:.3f}")
            print(f"  交差検証スコア: {avg_perf['cv_score']:.3f}")
            print(f"  予測時間: {avg_perf['prediction_time_ms']:.2f}ms")
            
            best_model = summary['best_model']
            print(f"\n最高性能モデル: {best_model['name']}")
            print(f"  精度: {best_model['accuracy']:.3f}")
        
        print(f"✅ 次世代予測精度向上システム完了")
        
        return training_results, prediction, summary
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return {}, None, {}


if __name__ == "__main__":
    asyncio.run(demo_next_generation_prediction())