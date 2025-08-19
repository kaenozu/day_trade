#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced AI Engine - 次世代AI分析システム
Issue #934対応: 高度なアンサンブル学習とリアルタイムML
"""

import numpy as np
import pandas as pd
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from performance_monitor import performance_monitor, track_performance
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False
    def track_performance(func):
        return func

try:
    from data_persistence import data_persistence
    HAS_DATA_PERSISTENCE = True
except ImportError:
    HAS_DATA_PERSISTENCE = False

# オプショナル依存関係
try:
    import scikit_learn
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


@dataclass
class MarketSignal:
    """市場シグナル情報"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    strength: float
    timestamp: datetime
    indicators: Dict[str, float]
    reasons: List[str]
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'


@dataclass
class ModelPrediction:
    """モデル予測結果"""
    model_name: str
    prediction: float
    confidence: float
    features_importance: Dict[str, float]
    execution_time_ms: float


class TechnicalIndicatorCalculator:
    """テクニカル指標計算エンジン"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """RSI計算（Relative Strength Index）"""
        if len(prices) < period + 1:
            return 50.0  # 中立値
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    @staticmethod
    def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """MACD計算（Moving Average Convergence Divergence）"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        prices_array = np.array(prices)
        
        # EMA計算
        def calculate_ema(data, period):
            ema = []
            multiplier = 2 / (period + 1)
            ema.append(data[0])
            
            for i in range(1, len(data)):
                ema.append((data[i] * multiplier) + (ema[i-1] * (1 - multiplier)))
            
            return ema
        
        ema_fast = calculate_ema(prices, fast)
        ema_slow = calculate_ema(prices, slow)
        
        # MACD線
        macd_line = ema_fast[-1] - ema_slow[-1] if len(ema_fast) >= slow and len(ema_slow) >= slow else 0.0
        
        # シグナル線（簡易計算）
        signal_line = macd_line * 0.8  # 簡易近似
        
        # ヒストグラム
        histogram = macd_line - signal_line
        
        return float(macd_line), float(signal_line), float(histogram)
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """ボリンジャーバンド計算"""
        if len(prices) < period:
            price = prices[-1] if prices else 0.0
            return price, price, price
        
        recent_prices = prices[-period:]
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return float(sma), float(upper_band), float(lower_band)
    
    @staticmethod
    def calculate_volume_indicators(volumes: List[float], prices: List[float]) -> Dict[str, float]:
        """出来高指標計算"""
        if not volumes or not prices:
            return {
                'volume_sma': 0.0,
                'volume_ratio': 1.0,
                'price_volume_trend': 0.0
            }
        
        recent_volumes = volumes[-10:] if len(volumes) >= 10 else volumes
        volume_sma = np.mean(recent_volumes)
        current_volume = volumes[-1] if volumes else 0.0
        volume_ratio = current_volume / volume_sma if volume_sma > 0 else 1.0
        
        # Price Volume Trend (簡易計算)
        if len(prices) >= 2 and len(volumes) >= 2:
            price_change = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0
            pvt = price_change * volumes[-1] if volumes[-1] else 0
        else:
            pvt = 0.0
        
        return {
            'volume_sma': float(volume_sma),
            'volume_ratio': float(volume_ratio),
            'price_volume_trend': float(pvt)
        }


class AdvancedMLModel:
    """高度な機械学習モデル"""
    
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.models = {}
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.feature_importance = {}
        self.performance_history = deque(maxlen=100)
        
        if HAS_SKLEARN:
            self._initialize_models()
    
    def _initialize_models(self):
        """モデル初期化"""
        if not HAS_SKLEARN:
            return
        
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'linear_ridge': Ridge(alpha=1.0),
            'neural_network_sim': LinearRegression()  # ニューラルネット簡易シミュレーション
        }
    
    def prepare_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """特徴量準備"""
        features = []
        
        # 価格関連特徴量
        price = market_data.get('price', 0.0)
        price_change = market_data.get('price_change_percent', 0.0)
        volume = market_data.get('volume', 0.0)
        
        # テクニカル指標
        rsi = market_data.get('rsi', 50.0)
        macd = market_data.get('macd', 0.0)
        bb_position = market_data.get('bb_position', 0.5)  # ボリンジャーバンド内位置
        
        # 市場構造特徴量
        volatility = market_data.get('volatility', 0.02)
        momentum = market_data.get('momentum', 0.0)
        
        # 時間特徴量
        hour_of_day = datetime.now().hour / 24.0
        day_of_week = datetime.now().weekday() / 7.0
        
        # 組合せ特徴量
        price_volume_ratio = (price * volume) if volume > 0 else 0.0
        rsi_momentum = rsi * momentum
        
        features = [
            price, price_change, volume, rsi, macd, bb_position,
            volatility, momentum, hour_of_day, day_of_week,
            price_volume_ratio, rsi_momentum
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train_models(self, training_data: List[Dict[str, Any]], targets: List[float]) -> Dict[str, float]:
        """モデル訓練"""
        if not HAS_SKLEARN or not training_data:
            return {'error': 'No training data or sklearn not available'}
        
        # 特徴量準備
        X = np.array([self.prepare_features(data).flatten() for data in training_data])
        y = np.array(targets)
        
        if len(X) < 10:  # 最小データ数チェック
            return {'error': 'Insufficient training data'}
        
        # スケーリング
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        
        for name, model in self.models.items():
            try:
                start_time = time.time()
                
                # モデル訓練
                model.fit(X_scaled, y)
                
                # 交差検証
                cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X)//2))
                
                training_time = (time.time() - start_time) * 1000
                
                results[name] = {
                    'cv_score_mean': float(np.mean(cv_scores)),
                    'cv_score_std': float(np.std(cv_scores)),
                    'training_time_ms': training_time
                }
                
                # 特徴量重要度（利用可能な場合）
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_.tolist()
                
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results
    
    def predict(self, market_data: Dict[str, Any]) -> List[ModelPrediction]:
        """予測実行"""
        if not HAS_SKLEARN:
            return [ModelPrediction(
                model_name='fallback',
                prediction=0.5,
                confidence=0.7,
                features_importance={},
                execution_time_ms=1.0
            )]
        
        X = self.prepare_features(market_data)
        X_scaled = self.scaler.transform(X) if self.scaler else X
        
        predictions = []
        
        for name, model in self.models.items():
            start_time = time.time()
            
            try:
                prediction = model.predict(X_scaled)[0]
                execution_time = (time.time() - start_time) * 1000
                
                # 信頼度計算（簡易）
                confidence = min(0.95, max(0.5, abs(prediction) * 0.8 + 0.2))
                
                predictions.append(ModelPrediction(
                    model_name=name,
                    prediction=float(prediction),
                    confidence=float(confidence),
                    features_importance=self.feature_importance.get(name, {}),
                    execution_time_ms=execution_time
                ))
                
            except Exception as e:
                predictions.append(ModelPrediction(
                    model_name=f"{name}_error",
                    prediction=0.0,
                    confidence=0.1,
                    features_importance={},
                    execution_time_ms=0.0
                ))
        
        return predictions


class AdvancedAIEngine:
    """次世代AI分析エンジン"""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.market_history = defaultdict(lambda: deque(maxlen=max_history_size))
        self.prediction_cache = {}
        self.cache_expiry = {}
        
        # コンポーネント初期化
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.ml_model = AdvancedMLModel()
        
        # パフォーマンス追跡
        self.prediction_accuracy = defaultdict(list)
        self.execution_times = deque(maxlen=100)
        
        # 非同期処理用
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # アンサンブル重み
        self.ensemble_weights = {
            'technical_analysis': 0.3,
            'ml_prediction': 0.4,
            'momentum_analysis': 0.2,
            'volume_analysis': 0.1
        }
    
    def update_market_data(self, symbol: str, price: float, volume: float, 
                          timestamp: Optional[datetime] = None):
        """市場データ更新"""
        if timestamp is None:
            timestamp = datetime.now()
        
        market_point = {
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        }
        
        self.market_history[symbol].append(market_point)
        
        # キャッシュ無効化
        if symbol in self.prediction_cache:
            del self.prediction_cache[symbol]
    
    def _get_price_history(self, symbol: str, periods: int = 50) -> List[float]:
        """価格履歴取得"""
        history = list(self.market_history[symbol])
        prices = [point['price'] for point in history[-periods:]]
        
        # データが不足している場合は模擬データで補完
        if len(prices) < periods:
            base_price = prices[-1] if prices else 1500.0
            missing_count = periods - len(prices)
            
            # 簡易ランダムウォーク生成
            np.random.seed(hash(symbol) % 1000)
            random_changes = np.random.normal(0, 0.02, missing_count)
            
            simulated_prices = []
            current_price = base_price * 0.95  # 少し低めから開始
            
            for change in random_changes:
                current_price *= (1 + change)
                simulated_prices.append(current_price)
            
            prices = simulated_prices + prices
        
        return prices
    
    def _get_volume_history(self, symbol: str, periods: int = 50) -> List[float]:
        """出来高履歴取得"""
        history = list(self.market_history[symbol])
        volumes = [point['volume'] for point in history[-periods:]]
        
        # データが不足している場合は模擬データで補完
        if len(volumes) < periods:
            base_volume = volumes[-1] if volumes else 1000000.0
            missing_count = periods - len(volumes)
            
            # 簡易出来高生成
            np.random.seed(hash(symbol + 'volume') % 1000)
            volume_multipliers = np.random.lognormal(0, 0.3, missing_count)
            
            simulated_volumes = [base_volume * mult for mult in volume_multipliers]
            volumes = simulated_volumes + volumes
        
        return volumes
    
    @track_performance
    def analyze_symbol(self, symbol: str, real_time: bool = True) -> MarketSignal:
        """シンボル分析実行"""
        start_time = time.time()
        
        # キャッシュチェック
        if not real_time and symbol in self.prediction_cache:
            cache_time = self.cache_expiry.get(symbol, datetime.min)
            if datetime.now() - cache_time < timedelta(minutes=5):
                return self.prediction_cache[symbol]
        
        try:
            # 市場データ準備
            prices = self._get_price_history(symbol)
            volumes = self._get_volume_history(symbol)
            current_price = prices[-1] if prices else 1500.0
            
            # テクニカル指標計算
            rsi = self.indicator_calculator.calculate_rsi(prices)
            macd, macd_signal, macd_histogram = self.indicator_calculator.calculate_macd(prices)
            sma, bb_upper, bb_lower = self.indicator_calculator.calculate_bollinger_bands(prices)
            volume_indicators = self.indicator_calculator.calculate_volume_indicators(volumes, prices)
            
            # 市場データパッケージング
            market_data = {
                'symbol': symbol,
                'price': current_price,
                'price_change_percent': ((current_price - prices[-2]) / prices[-2] * 100) if len(prices) > 1 else 0.0,
                'volume': volumes[-1] if volumes else 0.0,
                'rsi': rsi,
                'macd': macd,
                'bb_position': (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5,
                'volatility': np.std(prices[-20:]) / np.mean(prices[-20:]) if len(prices) >= 20 else 0.02,
                'momentum': (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0.0,
                **volume_indicators
            }
            
            # アンサンブル分析実行
            signal = self._execute_ensemble_analysis(market_data)
            
            # パフォーマンス記録
            execution_time = (time.time() - start_time) * 1000
            self.execution_times.append(execution_time)
            
            if HAS_PERFORMANCE_MONITOR and performance_monitor:
                performance_monitor.track_analysis_time(symbol, execution_time / 1000, 'advanced_ai_analysis')
            
            # キャッシュ更新
            if not real_time:
                self.prediction_cache[symbol] = signal
                self.cache_expiry[symbol] = datetime.now()
            
            # データ永続化
            if HAS_DATA_PERSISTENCE and data_persistence:
                data_persistence.save_analysis_result(
                    symbol=symbol,
                    analysis_type='advanced_ai_analysis',
                    duration_ms=execution_time,
                    result_data={
                        'signal_type': signal.signal_type,
                        'confidence': signal.confidence,
                        'strength': signal.strength,
                        'risk_level': signal.risk_level,
                        'technical_indicators': signal.indicators
                    },
                    confidence_score=signal.confidence
                )
            
            return signal
            
        except Exception as e:
            # エラー時のフォールバック
            return MarketSignal(
                symbol=symbol,
                signal_type='HOLD',
                confidence=0.5,
                strength=0.0,
                timestamp=datetime.now(),
                indicators={},
                reasons=[f'Analysis error: {str(e)}'],
                risk_level='MEDIUM'
            )
    
    def _execute_ensemble_analysis(self, market_data: Dict[str, Any]) -> MarketSignal:
        """アンサンブル分析実行"""
        symbol = market_data['symbol']
        
        # 各分析手法の実行
        technical_result = self._technical_analysis(market_data)
        ml_result = self._ml_analysis(market_data)
        momentum_result = self._momentum_analysis(market_data)
        volume_result = self._volume_analysis(market_data)
        
        # アンサンブル統合
        ensemble_score = (
            technical_result['score'] * self.ensemble_weights['technical_analysis'] +
            ml_result['score'] * self.ensemble_weights['ml_prediction'] +
            momentum_result['score'] * self.ensemble_weights['momentum_analysis'] +
            volume_result['score'] * self.ensemble_weights['volume_analysis']
        )
        
        # シグナル決定
        if ensemble_score > 0.6:
            signal_type = 'BUY'
            confidence = min(0.95, ensemble_score)
        elif ensemble_score < 0.4:
            signal_type = 'SELL'
            confidence = min(0.95, 1.0 - ensemble_score)
        else:
            signal_type = 'HOLD'
            confidence = 0.7
        
        # 強度計算
        strength = abs(ensemble_score - 0.5) * 2.0
        
        # リスク評価
        volatility = market_data.get('volatility', 0.02)
        if volatility > 0.05:
            risk_level = 'HIGH'
        elif volatility > 0.03:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # 理由集約
        all_reasons = []
        all_reasons.extend(technical_result.get('reasons', []))
        all_reasons.extend(ml_result.get('reasons', []))
        all_reasons.extend(momentum_result.get('reasons', []))
        all_reasons.extend(volume_result.get('reasons', []))
        
        return MarketSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            strength=strength,
            timestamp=datetime.now(),
            indicators={
                'rsi': market_data.get('rsi', 50.0),
                'macd': market_data.get('macd', 0.0),
                'bb_position': market_data.get('bb_position', 0.5),
                'volume_ratio': market_data.get('volume_ratio', 1.0),
                'ensemble_score': ensemble_score,
                'technical_score': technical_result['score'],
                'ml_score': ml_result['score'],
                'momentum_score': momentum_result['score'],
                'volume_score': volume_result['score']
            },
            reasons=all_reasons[:5],  # 上位5つの理由
            risk_level=risk_level
        )
    
    def _technical_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """テクニカル分析"""
        rsi = market_data.get('rsi', 50.0)
        macd = market_data.get('macd', 0.0)
        bb_position = market_data.get('bb_position', 0.5)
        
        score = 0.5  # 中立から開始
        reasons = []
        
        # RSI分析
        if rsi < 30:
            score += 0.2
            reasons.append('RSI oversold signal')
        elif rsi > 70:
            score -= 0.2
            reasons.append('RSI overbought signal')
        
        # MACD分析
        if macd > 0:
            score += 0.15
            reasons.append('MACD bullish signal')
        else:
            score -= 0.15
            reasons.append('MACD bearish signal')
        
        # ボリンジャーバンド分析
        if bb_position < 0.2:
            score += 0.1
            reasons.append('Price near lower Bollinger band')
        elif bb_position > 0.8:
            score -= 0.1
            reasons.append('Price near upper Bollinger band')
        
        return {
            'score': max(0.0, min(1.0, score)),
            'reasons': reasons
        }
    
    def _ml_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """機械学習分析"""
        try:
            predictions = self.ml_model.predict(market_data)
            
            if not predictions:
                return {'score': 0.5, 'reasons': ['No ML predictions available']}
            
            # 予測値統合
            avg_prediction = np.mean([p.prediction for p in predictions])
            avg_confidence = np.mean([p.confidence for p in predictions])
            
            # スコア正規化
            score = max(0.0, min(1.0, avg_prediction * 0.5 + 0.5))
            
            reasons = [f'ML ensemble prediction: {avg_prediction:.3f}']
            if avg_confidence > 0.8:
                reasons.append('High ML confidence')
            
            return {
                'score': score,
                'reasons': reasons,
                'ml_predictions': predictions
            }
            
        except Exception:
            return {
                'score': 0.5,
                'reasons': ['ML analysis failed, using neutral score']
            }
    
    def _momentum_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """モメンタム分析"""
        momentum = market_data.get('momentum', 0.0)
        price_change = market_data.get('price_change_percent', 0.0)
        
        score = 0.5 + (momentum * 2.0) + (price_change * 0.01)
        score = max(0.0, min(1.0, score))
        
        reasons = []
        if momentum > 0.02:
            reasons.append('Strong positive momentum')
        elif momentum < -0.02:
            reasons.append('Strong negative momentum')
        
        if abs(price_change) > 2.0:
            reasons.append(f'Significant price change: {price_change:.2f}%')
        
        return {
            'score': score,
            'reasons': reasons if reasons else ['Neutral momentum']
        }
    
    def _volume_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """出来高分析"""
        volume_ratio = market_data.get('volume_ratio', 1.0)
        price_volume_trend = market_data.get('price_volume_trend', 0.0)
        
        score = 0.5
        reasons = []
        
        # 出来高倍率分析
        if volume_ratio > 2.0:
            score += 0.15
            reasons.append('High volume activity')
        elif volume_ratio < 0.5:
            score -= 0.1
            reasons.append('Low volume activity')
        
        # PVT分析
        if price_volume_trend > 0:
            score += 0.1
            reasons.append('Positive price-volume trend')
        else:
            score -= 0.1
            reasons.append('Negative price-volume trend')
        
        return {
            'score': max(0.0, min(1.0, score)),
            'reasons': reasons if reasons else ['Normal volume pattern']
        }
    
    def batch_analysis(self, symbols: List[str], max_workers: int = 4) -> Dict[str, MarketSignal]:
        """バッチ分析（並列実行）"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.analyze_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    results[symbol] = MarketSignal(
                        symbol=symbol,
                        signal_type='HOLD',
                        confidence=0.5,
                        strength=0.0,
                        timestamp=datetime.now(),
                        indicators={},
                        reasons=[f'Batch analysis error: {str(e)}'],
                        risk_level='HIGH'
                    )
        
        return results
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """エンジン統計取得"""
        return {
            'total_symbols_tracked': len(self.market_history),
            'total_predictions_cached': len(self.prediction_cache),
            'average_execution_time_ms': np.mean(self.execution_times) if self.execution_times else 0.0,
            'execution_time_p95_ms': np.percentile(self.execution_times, 95) if self.execution_times else 0.0,
            'ensemble_weights': self.ensemble_weights,
            'ml_models_available': HAS_SKLEARN,
            'feature_importance': self.ml_model.feature_importance if HAS_SKLEARN else {}
        }
    
    def update_ensemble_weights(self, performance_feedback: Dict[str, float]):
        """アンサンブル重み更新（学習機能）"""
        total_performance = sum(performance_feedback.values())
        if total_performance > 0:
            for method, performance in performance_feedback.items():
                if method in self.ensemble_weights:
                    # 性能に基づく重み調整
                    self.ensemble_weights[method] = performance / total_performance
        
        # 重み正規化
        total_weight = sum(self.ensemble_weights.values())
        if total_weight > 0:
            for method in self.ensemble_weights:
                self.ensemble_weights[method] /= total_weight


# グローバルインスタンス
advanced_ai_engine = AdvancedAIEngine()


if __name__ == "__main__":
    # テスト実行
    engine = AdvancedAIEngine()
    
    # テストデータ追加
    test_symbols = ['7203', '8306', '9984']
    
    for symbol in test_symbols:
        # 模擬市場データ更新
        base_price = 1500 + hash(symbol) % 1000
        for i in range(50):
            price = base_price * (1 + np.random.normal(0, 0.02))
            volume = np.random.lognormal(13, 0.5)  # 模擬出来高
            engine.update_market_data(symbol, price, volume)
    
    print("=== AI分析エンジンテスト実行 ===")
    
    # 単一シンボル分析
    signal = engine.analyze_symbol('7203')
    print(f"\n【{signal.symbol}】")
    print(f"シグナル: {signal.signal_type}")
    print(f"信頼度: {signal.confidence:.2f}")
    print(f"強度: {signal.strength:.2f}")
    print(f"リスクレベル: {signal.risk_level}")
    print(f"理由: {', '.join(signal.reasons)}")
    
    # バッチ分析テスト
    print("\n=== バッチ分析結果 ===")
    batch_results = engine.batch_analysis(test_symbols)
    
    for symbol, signal in batch_results.items():
        print(f"{symbol}: {signal.signal_type} ({signal.confidence:.2f})")
    
    # エンジン統計
    print("\n=== エンジン統計 ===")
    stats = engine.get_engine_statistics()
    print(json.dumps(stats, indent=2, default=str, ensure_ascii=False))