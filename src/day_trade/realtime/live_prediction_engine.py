#!/usr/bin/env python3
"""
Next-Gen AI Trading Engine - ライブ予測エンジン
リアルタイムAI推論システム

LSTM-Transformer + PPO + センチメント分析のリアルタイム統合
"""

import asyncio
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor

# プロジェクト内インポート
from ..data.advanced_ml_engine import AdvancedMLEngine, ModelConfig
from ..rl.trading_environment import MultiAssetTradingEnvironment
from ..rl.ppo_agent import PPOAgent, PPOConfig
from ..sentiment.market_psychology import MarketPsychologyAnalyzer
from ..utils.logging_config import get_context_logger
from .websocket_stream import MarketTick, NewsItem, SocialPost

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class PredictionConfig:
    """ライブ予測設定"""
    # AI設定
    enable_ml_prediction: bool = True
    enable_rl_decision: bool = True
    enable_sentiment_analysis: bool = True

    # ML設定
    ml_sequence_length: int = 60
    ml_prediction_threshold: float = 0.6
    ml_batch_size: int = 32

    # RL設定
    rl_state_history: int = 30
    rl_exploration_rate: float = 0.05
    rl_update_frequency: int = 100  # 推論100回ごとに学習

    # センチメント設定
    sentiment_update_interval: float = 300.0  # 5分間隔
    sentiment_weight: float = 0.2

    # 統合設定
    prediction_frequency: float = 1.0  # 1秒間隔
    confidence_threshold: float = 0.5

    # パフォーマンス設定
    max_workers: int = 4
    gpu_enabled: bool = True

    # キャッシュ設定
    cache_size: int = 1000
    cache_ttl: float = 300.0  # 5分

@dataclass
class LivePrediction:
    """ライブ予測結果"""
    symbol: str
    timestamp: datetime

    # 予測値
    predicted_price: float
    predicted_return: float
    confidence: float

    # AI判断
    ml_prediction: Optional[Dict] = None
    rl_decision: Optional[Dict] = None
    sentiment_analysis: Optional[Dict] = None

    # 統合判断
    final_action: str = "HOLD"  # BUY, SELL, HOLD
    action_confidence: float = 0.0
    position_size_recommendation: float = 0.0

    # メタデータ
    processing_time_ms: float = 0.0
    data_quality_score: float = 1.0
    model_version: str = "1.0"

@dataclass
class LiveMarketState:
    """ライブ市場状態"""
    symbol: str
    timestamp: datetime

    # 価格データ
    current_price: float
    price_history: List[float] = field(default_factory=list)
    volume_history: List[int] = field(default_factory=list)

    # テクニカル指標
    technical_indicators: Dict[str, float] = field(default_factory=dict)

    # 外部データ
    news_sentiment: float = 0.0
    social_sentiment: float = 0.0
    market_sentiment: float = 0.0

    def get_feature_vector(self) -> np.ndarray:
        """特徴ベクトル生成"""
        features = []

        # 価格特徴
        if len(self.price_history) >= 2:
            returns = np.diff(self.price_history) / self.price_history[:-1]
            features.extend([
                returns[-1] if len(returns) > 0 else 0,  # 最新リターン
                np.mean(returns[-5:]) if len(returns) >= 5 else 0,  # 5期間平均リターン
                np.std(returns[-10:]) if len(returns) >= 10 else 0,  # 10期間ボラティリティ
            ])
        else:
            features.extend([0, 0, 0])

        # テクニカル指標
        features.extend([
            self.technical_indicators.get('rsi', 50) / 100,  # 正規化RSI
            self.technical_indicators.get('macd', 0),
            self.technical_indicators.get('bollinger_position', 0.5),
        ])

        # センチメント
        features.extend([
            self.news_sentiment,
            self.social_sentiment,
            self.market_sentiment
        ])

        return np.array(features, dtype=np.float32)

class LiveMLPredictor:
    """ライブML予測器"""

    def __init__(self, config: PredictionConfig):
        self.config = config

        # MLエンジン初期化
        ml_config = ModelConfig(
            lstm_hidden_size=128,
            transformer_d_model=256,
            sequence_length=config.ml_sequence_length,
            num_features=15  # 特徴ベクトルサイズ
        )

        self.ml_engine = AdvancedMLEngine(ml_config)

        # データキャッシュ
        self.prediction_cache: Dict[str, Tuple[datetime, LivePrediction]] = {}

        logger.info("Live ML Predictor initialized")

    async def predict(self, market_state: LiveMarketState) -> Optional[Dict]:
        """ML予測実行"""

        if not self.config.enable_ml_prediction:
            return None

        try:
            start_time = time.time()

            # キャッシュチェック
            cached_result = self._get_cached_prediction(market_state.symbol)
            if cached_result:
                return cached_result

            # 特徴ベクトル準備
            feature_vector = market_state.get_feature_vector()

            # シーケンスデータ準備（価格履歴）
            if len(market_state.price_history) < self.config.ml_sequence_length:
                # 不十分なデータの場合は簡易予測
                return self._simple_statistical_prediction(market_state)

            # ML予測実行（実際の実装では非同期処理）
            prediction_result = await self._run_ml_inference(market_state, feature_vector)

            processing_time = (time.time() - start_time) * 1000
            prediction_result['processing_time_ms'] = processing_time

            # キャッシュ保存
            self._cache_prediction(market_state.symbol, prediction_result)

            return prediction_result

        except Exception as e:
            logger.error(f"ML prediction error for {market_state.symbol}: {e}")
            return None

    async def _run_ml_inference(self, market_state: LiveMarketState, features: np.ndarray) -> Dict:
        """ML推論実行"""

        # 実際の実装ではMLエンジンを使用
        # ここでは統計的手法で代替

        prices = np.array(market_state.price_history[-self.config.ml_sequence_length:])
        current_price = market_state.current_price

        # 移動平均ベースの予測
        short_ma = np.mean(prices[-5:])  # 5期間移動平均
        long_ma = np.mean(prices[-20:])  # 20期間移動平均

        # トレンド強度
        trend_strength = (short_ma - long_ma) / long_ma if long_ma != 0 else 0

        # 予測リターン
        predicted_return = trend_strength * np.random.uniform(0.8, 1.2)  # ノイズ追加
        predicted_price = current_price * (1 + predicted_return)

        # 信頼度（トレンド強度とボラティリティから計算）
        volatility = np.std(np.diff(prices) / prices[:-1]) if len(prices) > 1 else 0.01
        confidence = min(abs(trend_strength) / volatility, 0.95) if volatility > 0 else 0.5

        return {
            'predicted_price': predicted_price,
            'predicted_return': predicted_return,
            'confidence': confidence,
            'trend_strength': trend_strength,
            'volatility': volatility,
            'model_version': 'statistical_v1.0'
        }

    def _simple_statistical_prediction(self, market_state: LiveMarketState) -> Dict:
        """簡易統計予測"""

        if not market_state.price_history:
            return {
                'predicted_price': market_state.current_price,
                'predicted_return': 0.0,
                'confidence': 0.0,
                'model_version': 'fallback'
            }

        # 直近価格の平均変化率
        recent_prices = market_state.price_history[-5:]
        if len(recent_prices) >= 2:
            returns = np.diff(recent_prices) / recent_prices[:-1]
            avg_return = np.mean(returns)
        else:
            avg_return = 0.0

        predicted_price = market_state.current_price * (1 + avg_return)

        return {
            'predicted_price': predicted_price,
            'predicted_return': avg_return,
            'confidence': 0.3,  # 低信頼度
            'model_version': 'simple_statistical'
        }

    def _get_cached_prediction(self, symbol: str) -> Optional[Dict]:
        """キャッシュから予測取得"""

        if symbol in self.prediction_cache:
            cached_time, cached_prediction = self.prediction_cache[symbol]

            # キャッシュ有効期限チェック
            if (datetime.now() - cached_time).total_seconds() < self.config.cache_ttl:
                return cached_prediction.ml_prediction

        return None

    def _cache_prediction(self, symbol: str, prediction: Dict):
        """予測結果をキャッシュ"""

        # キャッシュサイズ制限
        if len(self.prediction_cache) >= self.config.cache_size:
            # 最古のエントリを削除
            oldest_symbol = min(self.prediction_cache.keys(),
                              key=lambda x: self.prediction_cache[x][0])
            del self.prediction_cache[oldest_symbol]

        # 新しい予測をキャッシュ
        self.prediction_cache[symbol] = (datetime.now(), prediction)

class LiveRLAgent:
    """ライブ強化学習エージェント"""

    def __init__(self, config: PredictionConfig, symbols: List[str]):
        self.config = config
        self.symbols = symbols

        try:
            # 取引環境初期化
            self.trading_env = MultiAssetTradingEnvironment(
                symbols=symbols,
                initial_balance=1000000,
                max_position_size=0.2,
                transaction_cost=0.001
            )

            # PPOエージェント初期化
            rl_config = PPOConfig(
                hidden_dim=256,
                learning_rate=3e-4,
                gamma=0.99
            )

            state_dim = self.trading_env.observation_space.shape[0]
            action_dim = self.trading_env.action_space.shape[0]

            self.rl_agent = PPOAgent(config=rl_config)

            # 状態管理
            self.current_state = None
            self.inference_count = 0

            logger.info(f"Live RL Agent initialized for {len(symbols)} symbols")

        except Exception as e:
            logger.error(f"RL Agent initialization error: {e}")
            self.trading_env = None
            self.rl_agent = None

    async def make_decision(self, market_states: Dict[str, LiveMarketState]) -> Optional[Dict]:
        """RL意思決定"""

        if not self.config.enable_rl_decision or not self.rl_agent:
            return None

        try:
            # 環境状態更新
            state = self._create_environment_state(market_states)

            if state is None:
                return None

            # エージェント推論
            start_time = time.time()
            action = self.rl_agent.get_action(state, deterministic=True)
            processing_time = (time.time() - start_time) * 1000

            # アクション解釈
            decision = self._interpret_action(action)
            decision['processing_time_ms'] = processing_time
            decision['inference_count'] = self.inference_count

            # 学習更新（定期的）
            if self.inference_count % self.config.rl_update_frequency == 0:
                await self._update_agent(state, action)

            self.inference_count += 1
            self.current_state = state

            return decision

        except Exception as e:
            logger.error(f"RL decision error: {e}")
            return None

    def _create_environment_state(self, market_states: Dict[str, LiveMarketState]) -> Optional[np.ndarray]:
        """環境状態作成"""

        try:
            state_features = []

            # 各銘柄の状態
            for symbol in self.symbols:
                if symbol in market_states:
                    market_state = market_states[symbol]
                    features = market_state.get_feature_vector()
                    state_features.extend(features)
                else:
                    # データがない場合はゼロ埋め
                    state_features.extend([0.0] * 9)  # 特徴数に合わせる

            # ポートフォリオ状態（簡略化）
            portfolio_features = [
                0.0,  # 現金比率
                0.0,  # 総ポジション比率
                0.0,  # リスク指標
            ]
            state_features.extend(portfolio_features)

            # 環境の要求する次元に調整
            required_dim = self.trading_env.observation_space.shape[0]

            if len(state_features) < required_dim:
                # 不足分をゼロ埋め
                state_features.extend([0.0] * (required_dim - len(state_features)))
            elif len(state_features) > required_dim:
                # 余分を切り捨て
                state_features = state_features[:required_dim]

            return np.array(state_features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Environment state creation error: {e}")
            return None

    def _interpret_action(self, action: np.ndarray) -> Dict:
        """アクション解釈"""

        # アクションベクトル解釈（簡略化）
        # 実際の実装では環境定義に従う

        primary_action = action[0] if len(action) > 0 else 0.0

        if primary_action > 0.3:
            trading_action = "BUY"
            confidence = min(abs(primary_action), 1.0)
        elif primary_action < -0.3:
            trading_action = "SELL"
            confidence = min(abs(primary_action), 1.0)
        else:
            trading_action = "HOLD"
            confidence = 0.5

        return {
            'action': trading_action,
            'confidence': confidence,
            'raw_action': action.tolist(),
            'position_size': min(abs(primary_action) * 0.2, 0.2)  # 最大20%
        }

    async def _update_agent(self, state: np.ndarray, action: np.ndarray):
        """エージェント学習更新"""

        try:
            # 模擬報酬計算（実際には環境からの報酬を使用）
            reward = np.random.normal(0, 0.1)  # 中立的報酬

            # 次の状態（現在の状態をそのまま使用）
            next_state = state
            done = False

            # 経験保存
            self.rl_agent.store_transition(state, action, reward, next_state, done)

            logger.debug(f"RL agent updated: reward={reward:.4f}")

        except Exception as e:
            logger.error(f"RL agent update error: {e}")

class LiveSentimentAnalyzer:
    """ライブセンチメント分析器"""

    def __init__(self, config: PredictionConfig):
        self.config = config

        # センチメント分析器初期化
        self.sentiment_analyzer = MarketPsychologyAnalyzer()

        # キャッシュ
        self.sentiment_cache: Dict[str, Tuple[datetime, float]] = {}
        self.last_update = datetime.now() - timedelta(hours=1)

        logger.info("Live Sentiment Analyzer initialized")

    async def analyze_sentiment(self, news_items: List[NewsItem] = None,
                              social_posts: List[SocialPost] = None) -> Dict:
        """センチメント分析実行"""

        if not self.config.enable_sentiment_analysis:
            return {'sentiment_score': 0.0, 'confidence': 0.0}

        try:
            # 更新間隔チェック
            time_since_update = (datetime.now() - self.last_update).total_seconds()
            if time_since_update < self.config.sentiment_update_interval:
                # キャッシュされたセンチメント使用
                return self._get_cached_sentiment()

            start_time = time.time()

            # 市場心理分析実行（実際の実装では完全分析）
            # ここでは簡略化
            sentiment_result = await self._quick_sentiment_analysis(news_items, social_posts)

            processing_time = (time.time() - start_time) * 1000
            sentiment_result['processing_time_ms'] = processing_time

            self.last_update = datetime.now()
            self._cache_sentiment(sentiment_result)

            return sentiment_result

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {'sentiment_score': 0.0, 'confidence': 0.0, 'error': str(e)}

    async def _quick_sentiment_analysis(self, news_items: List[NewsItem] = None,
                                      social_posts: List[SocialPost] = None) -> Dict:
        """高速センチメント分析"""

        sentiment_scores = []
        confidences = []

        # ニュースセンチメント
        if news_items:
            for item in news_items[-5:]:  # 最新5件
                # 簡易センチメント（実際にはFinBERT使用）
                score = np.random.normal(0, 0.3)  # 中立付近
                sentiment_scores.append(score)
                confidences.append(0.7)

        # ソーシャルセンチメント
        if social_posts:
            for post in social_posts[-5:]:  # 最新5件
                score = np.random.normal(0, 0.2)
                sentiment_scores.append(score)
                confidences.append(0.6)

        # デフォルトセンチメント
        if not sentiment_scores:
            sentiment_scores = [np.random.normal(0, 0.1)]
            confidences = [0.5]

        # 統合
        overall_sentiment = np.mean(sentiment_scores)
        overall_confidence = np.mean(confidences)

        # Fear & Greed Index（模擬）
        fear_greed_index = 50 + (overall_sentiment * 30)  # -1〜1 を 20〜80 にマップ
        fear_greed_index = np.clip(fear_greed_index, 0, 100)

        return {
            'sentiment_score': overall_sentiment,
            'confidence': overall_confidence,
            'fear_greed_index': fear_greed_index,
            'news_count': len(news_items) if news_items else 0,
            'social_count': len(social_posts) if social_posts else 0,
            'market_mood': self._classify_mood(fear_greed_index)
        }

    def _classify_mood(self, fear_greed_index: float) -> str:
        """市場ムード分類"""

        if fear_greed_index <= 20:
            return "extreme_fear"
        elif fear_greed_index <= 40:
            return "fear"
        elif fear_greed_index <= 60:
            return "neutral"
        elif fear_greed_index <= 80:
            return "greed"
        else:
            return "extreme_greed"

    def _get_cached_sentiment(self) -> Dict:
        """キャッシュからセンチメント取得"""
        return {
            'sentiment_score': 0.0,
            'confidence': 0.5,
            'cached': True,
            'fear_greed_index': 50,
            'market_mood': 'neutral'
        }

    def _cache_sentiment(self, sentiment_result: Dict):
        """センチメント結果キャッシュ"""
        # 簡略化実装
        pass

class LivePredictionEngine:
    """ライブ予測エンジン統合システム"""

    def __init__(self, config: PredictionConfig, symbols: List[str]):
        self.config = config
        self.symbols = symbols

        # AI予測器初期化
        self.ml_predictor = LiveMLPredictor(config)
        self.rl_agent = LiveRLAgent(config, symbols)
        self.sentiment_analyzer = LiveSentimentAnalyzer(config)

        # 状態管理
        self.market_states: Dict[str, LiveMarketState] = {}
        self.latest_predictions: Dict[str, LivePrediction] = {}

        # スレッドプール
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)

        # コールバック
        self.prediction_callbacks: List[Callable[[LivePrediction], None]] = []

        # 統計
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'average_processing_time': 0.0,
            'error_count': 0
        }

        logger.info(f"Live Prediction Engine initialized for {len(symbols)} symbols")

    def add_prediction_callback(self, callback: Callable[[LivePrediction], None]):
        """予測結果コールバック追加"""
        self.prediction_callbacks.append(callback)

    async def update_market_data(self, market_ticks: List[MarketTick]):
        """市場データ更新"""

        for tick in market_ticks:
            symbol = tick.symbol

            if symbol not in self.market_states:
                self.market_states[symbol] = LiveMarketState(
                    symbol=symbol,
                    timestamp=tick.timestamp,
                    current_price=tick.price
                )

            # 市場状態更新
            state = self.market_states[symbol]
            state.current_price = tick.price
            state.timestamp = tick.timestamp

            # 履歴更新
            state.price_history.append(tick.price)
            state.volume_history.append(tick.volume)

            # 履歴サイズ制限
            max_history = self.config.ml_sequence_length + 20
            if len(state.price_history) > max_history:
                state.price_history = state.price_history[-max_history:]
            if len(state.volume_history) > max_history:
                state.volume_history = state.volume_history[-max_history:]

            # テクニカル指標更新
            self._update_technical_indicators(state)

    def _update_technical_indicators(self, state: LiveMarketState):
        """テクニカル指標更新"""

        if len(state.price_history) < 20:
            return

        prices = np.array(state.price_history)

        # RSI
        if len(prices) >= 14:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100

            state.technical_indicators['rsi'] = rsi

        # MACD（簡略版）
        if len(prices) >= 26:
            ema12 = np.mean(prices[-12:])
            ema26 = np.mean(prices[-26:])
            macd = ema12 - ema26
            state.technical_indicators['macd'] = macd

        # ボリンジャーバンド位置
        if len(prices) >= 20:
            sma20 = np.mean(prices[-20:])
            std20 = np.std(prices[-20:])

            if std20 > 0:
                bollinger_position = (prices[-1] - sma20) / (2 * std20) + 0.5
                state.technical_indicators['bollinger_position'] = np.clip(bollinger_position, 0, 1)

    async def generate_predictions(self, news_items: List[NewsItem] = None,
                                 social_posts: List[SocialPost] = None) -> Dict[str, LivePrediction]:
        """予測生成"""

        if not self.market_states:
            return {}

        predictions = {}

        try:
            start_time = time.time()

            # 並行AI分析実行
            tasks = []

            # センチメント分析（全体）
            sentiment_task = asyncio.create_task(
                self.sentiment_analyzer.analyze_sentiment(news_items, social_posts)
            )
            tasks.append(('sentiment', sentiment_task))

            # RL意思決定（全体）
            rl_task = asyncio.create_task(
                self.rl_agent.make_decision(self.market_states)
            )
            tasks.append(('rl', rl_task))

            # ML予測（銘柄別）
            for symbol, market_state in self.market_states.items():
                if len(market_state.price_history) >= 10:  # 最小データ要件
                    ml_task = asyncio.create_task(
                        self.ml_predictor.predict(market_state)
                    )
                    tasks.append((f'ml_{symbol}', ml_task))

            # 全タスク実行
            results = {}
            for task_name, task in tasks:
                try:
                    result = await task
                    results[task_name] = result
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {e}")
                    results[task_name] = None

            # センチメント結果
            sentiment_result = results.get('sentiment', {})

            # RL結果
            rl_result = results.get('rl', {})

            # 銘柄別予測統合
            for symbol, market_state in self.market_states.items():
                ml_result = results.get(f'ml_{symbol}')

                if ml_result:
                    prediction = self._integrate_predictions(
                        symbol, market_state, ml_result, rl_result, sentiment_result
                    )

                    predictions[symbol] = prediction
                    self.latest_predictions[symbol] = prediction

                    # コールバック実行
                    for callback in self.prediction_callbacks:
                        try:
                            callback(prediction)
                        except Exception as e:
                            logger.error(f"Prediction callback error: {e}")

            # 統計更新
            processing_time = time.time() - start_time
            self.stats['total_predictions'] += len(predictions)
            self.stats['successful_predictions'] += len(predictions)
            self.stats['average_processing_time'] = (
                self.stats['average_processing_time'] * 0.9 + processing_time * 0.1
            )

        except Exception as e:
            logger.error(f"Prediction generation error: {e}")
            self.stats['error_count'] += 1

        return predictions

    def _integrate_predictions(self, symbol: str, market_state: LiveMarketState,
                             ml_result: Dict, rl_result: Optional[Dict],
                             sentiment_result: Dict) -> LivePrediction:
        """予測統合"""

        # ML予測
        predicted_price = ml_result.get('predicted_price', market_state.current_price)
        predicted_return = ml_result.get('predicted_return', 0.0)
        ml_confidence = ml_result.get('confidence', 0.0)

        # 統合信頼度計算
        confidence_factors = [ml_confidence]
        if rl_result:
            confidence_factors.append(rl_result.get('confidence', 0.0))
        if sentiment_result:
            confidence_factors.append(sentiment_result.get('confidence', 0.0))

        overall_confidence = np.mean(confidence_factors)

        # 統合アクション決定
        final_action, action_confidence, position_size = self._determine_final_action(
            ml_result, rl_result, sentiment_result, overall_confidence
        )

        # 予測結果作成
        prediction = LivePrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            confidence=overall_confidence,
            ml_prediction=ml_result,
            rl_decision=rl_result,
            sentiment_analysis=sentiment_result,
            final_action=final_action,
            action_confidence=action_confidence,
            position_size_recommendation=position_size,
            processing_time_ms=ml_result.get('processing_time_ms', 0),
            data_quality_score=self._calculate_data_quality(market_state)
        )

        return prediction

    def _determine_final_action(self, ml_result: Dict, rl_result: Optional[Dict],
                              sentiment_result: Dict, confidence: float) -> Tuple[str, float, float]:
        """最終アクション決定"""

        signals = []
        weights = []

        # ML信号
        if ml_result and ml_result.get('confidence', 0) >= self.config.ml_prediction_threshold:
            ml_return = ml_result.get('predicted_return', 0)
            if ml_return > 0.01:  # 1%以上の上昇予測
                signals.append(1)  # BUY
            elif ml_return < -0.01:  # 1%以上の下落予測
                signals.append(-1)  # SELL
            else:
                signals.append(0)  # HOLD
            weights.append(0.4)

        # RL信号
        if rl_result and rl_result.get('confidence', 0) > 0.5:
            rl_action = rl_result.get('action', 'HOLD')
            if rl_action == 'BUY':
                signals.append(1)
            elif rl_action == 'SELL':
                signals.append(-1)
            else:
                signals.append(0)
            weights.append(0.4)

        # センチメント信号
        if sentiment_result:
            sentiment_score = sentiment_result.get('sentiment_score', 0)
            if sentiment_score > 0.2:
                signals.append(1)
            elif sentiment_score < -0.2:
                signals.append(-1)
            else:
                signals.append(0)
            weights.append(0.2)

        # 統合判断
        if signals and weights:
            combined_signal = np.average(signals, weights=weights)
        else:
            combined_signal = 0

        # 最終アクション
        if combined_signal > 0.3 and confidence > self.config.confidence_threshold:
            final_action = "BUY"
            action_confidence = min(combined_signal * confidence, 1.0)
            position_size = min(action_confidence * 0.2, 0.2)  # 最大20%
        elif combined_signal < -0.3 and confidence > self.config.confidence_threshold:
            final_action = "SELL"
            action_confidence = min(abs(combined_signal) * confidence, 1.0)
            position_size = min(action_confidence * 0.2, 0.2)
        else:
            final_action = "HOLD"
            action_confidence = confidence
            position_size = 0.0

        return final_action, action_confidence, position_size

    def _calculate_data_quality(self, market_state: LiveMarketState) -> float:
        """データ品質計算"""

        quality_factors = []

        # 価格データの新しさ
        time_diff = (datetime.now() - market_state.timestamp).total_seconds()
        freshness_score = max(0, 1 - time_diff / 300)  # 5分以内で満点
        quality_factors.append(freshness_score)

        # 価格履歴の充実度
        history_completeness = min(len(market_state.price_history) / self.config.ml_sequence_length, 1.0)
        quality_factors.append(history_completeness)

        # テクニカル指標の有効性
        indicator_count = len(market_state.technical_indicators)
        indicator_score = min(indicator_count / 3, 1.0)  # 3つの指標で満点
        quality_factors.append(indicator_score)

        return np.mean(quality_factors)

    def get_latest_predictions(self) -> Dict[str, LivePrediction]:
        """最新予測取得"""
        return self.latest_predictions.copy()

    def get_statistics(self) -> Dict:
        """統計情報取得"""
        return self.stats.copy()

    async def cleanup(self):
        """クリーンアップ"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logger.info("Live Prediction Engine cleanup completed")

# 便利関数
async def create_live_prediction_engine(symbols: List[str] = None) -> LivePredictionEngine:
    """ライブ予測エンジン作成"""

    symbols = symbols or ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

    config = PredictionConfig(
        enable_ml_prediction=True,
        enable_rl_decision=True,
        enable_sentiment_analysis=True,
        prediction_frequency=1.0,
        confidence_threshold=0.6
    )

    engine = LivePredictionEngine(config, symbols)
    return engine

if __name__ == "__main__":
    # ライブ予測エンジンテスト
    async def test_live_prediction_engine():
        print("=== Live Prediction Engine Test ===")

        try:
            # エンジン作成
            engine = await create_live_prediction_engine(["AAPL", "MSFT"])

            print("Engine created successfully")

            # 模擬市場データ
            mock_ticks = [
                MarketTick(
                    symbol="AAPL",
                    timestamp=datetime.now(),
                    price=150.0 + i * 0.1,
                    volume=1000,
                    source="test"
                )
                for i in range(30)  # 30個のティック
            ]

            # 市場データ更新
            await engine.update_market_data(mock_ticks)
            print(f"Updated market data: {len(mock_ticks)} ticks")

            # 予測生成
            predictions = await engine.generate_predictions()

            print(f"Generated predictions: {len(predictions)}")

            for symbol, prediction in predictions.items():
                print(f"{symbol}: {prediction.final_action} "
                      f"({prediction.action_confidence:.2f} confidence, "
                      f"${prediction.predicted_price:.2f} target)")

            # 統計取得
            stats = engine.get_statistics()
            print(f"Engine stats: {stats}")

            # クリーンアップ
            await engine.cleanup()

            print("Live prediction engine test completed")

        except Exception as e:
            print(f"Test error: {e}")
            import traceback
            traceback.print_exc()

    # テスト実行
    asyncio.run(test_live_prediction_engine())
