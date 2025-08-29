#!/usr/bin/env python3
"""
Emotional Intelligence System
感情知能システム

This module implements emotional intelligence for conscious AI trading.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from datetime import datetime, timedelta
import asyncio
import numpy as np
from collections import defaultdict, deque
import logging

from ..utils.error_handling import TradingResult


class EmotionType(Enum):
    """Core emotion types"""
    FEAR = "fear"
    GREED = "greed"
    CONFIDENCE = "confidence"
    ANXIETY = "anxiety"
    EUPHORIA = "euphoria"
    REGRET = "regret"
    HOPE = "hope"
    EXCITEMENT = "excitement"
    FRUSTRATION = "frustration"
    SATISFACTION = "satisfaction"
    CURIOSITY = "curiosity"
    DETERMINATION = "determination"


class MarketEmotionState(Enum):
    """Market emotional states"""
    PANIC_SELLING = "panic_selling"
    EUPHORIC_BUYING = "euphoric_buying"
    CAUTIOUS_OPTIMISM = "cautious_optimism"
    FEARFUL_UNCERTAINTY = "fearful_uncertainty"
    GREEDY_OVERCONFIDENCE = "greedy_overconfidence"
    BALANCED_RATIONAL = "balanced_rational"
    FRUSTRATED_STAGNATION = "frustrated_stagnation"
    HOPEFUL_RECOVERY = "hopeful_recovery"


class EmotionalIntensity(IntEnum):
    """Emotional intensity levels"""
    MINIMAL = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    EXTREME = 5


class SentimentPolarity(Enum):
    """Sentiment polarity"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


@dataclass
class EmotionalState:
    """Represents an emotional state"""
    emotion_type: EmotionType
    intensity: EmotionalIntensity
    valence: float  # -1.0 to 1.0 (negative to positive)
    arousal: float  # 0.0 to 1.0 (calm to excited)
    timestamp: datetime
    duration: float = 0.0
    triggers: List[str] = field(default_factory=list)
    physiological_markers: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarketSentiment:
    """Market sentiment analysis"""
    sentiment_id: str
    polarity: SentimentPolarity
    confidence: float
    sentiment_score: float  # -1.0 to 1.0
    market_fear_index: float
    greed_index: float
    volatility_sentiment: float
    news_sentiment: float
    social_sentiment: float
    timestamp: datetime
    contributing_factors: List[str] = field(default_factory=list)
    sector_sentiments: Dict[str, float] = field(default_factory=dict)


@dataclass
class EmotionalMarketProfile:
    """Emotional profile of market conditions"""
    profile_id: str
    dominant_emotions: List[EmotionType]
    emotional_volatility: float
    sentiment_momentum: float
    fear_greed_balance: float
    crowd_psychology_state: MarketEmotionState
    emotional_contagion_level: float
    market_mood_stability: float
    timestamp: datetime
    historical_patterns: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntuitiveInsight:
    """Intuitive trading insight"""
    insight_id: str
    insight_type: str
    market_symbols: List[str]
    intuitive_direction: str  # bullish, bearish, neutral
    conviction_level: float
    emotional_basis: List[EmotionType]
    pattern_recognition: Dict[str, Any]
    subconscious_signals: List[str]
    timestamp: datetime
    confidence_score: float = 0.0
    risk_assessment: str = ""


class EmotionalProcessor:
    """Processes emotional data and market sentiment"""
    
    def __init__(self):
        self.emotion_history = deque(maxlen=1000)
        self.market_emotion_patterns = {}
        self.sentiment_models = {}
        self.emotional_triggers = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        
    async def analyze_market_emotions(self, market_data: Dict[str, Any], 
                                    news_data: List[str] = None) -> MarketSentiment:
        """Analyze market emotional sentiment"""
        try:
            # Calculate base sentiment from market data
            market_sentiment_score = await self._calculate_market_sentiment(market_data)
            
            # Analyze news sentiment
            news_sentiment_score = await self._analyze_news_sentiment(news_data or [])
            
            # Calculate fear and greed indices
            fear_index = await self._calculate_fear_index(market_data)
            greed_index = await self._calculate_greed_index(market_data)
            
            # Analyze volatility sentiment
            volatility_sentiment = await self._analyze_volatility_sentiment(market_data)
            
            # Social sentiment analysis (simulated)
            social_sentiment_score = await self._analyze_social_sentiment()
            
            # Combined sentiment score
            combined_score = (
                market_sentiment_score * 0.4 +
                news_sentiment_score * 0.25 +
                social_sentiment_score * 0.2 +
                (greed_index - fear_index) * 0.15
            )
            
            # Determine polarity
            polarity = self._determine_sentiment_polarity(combined_score)
            
            # Calculate confidence
            confidence = await self._calculate_sentiment_confidence(
                market_data, combined_score
            )
            
            sentiment = MarketSentiment(
                sentiment_id=f"sent_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                polarity=polarity,
                confidence=confidence,
                sentiment_score=combined_score,
                market_fear_index=fear_index,
                greed_index=greed_index,
                volatility_sentiment=volatility_sentiment,
                news_sentiment=news_sentiment_score,
                social_sentiment=social_sentiment_score,
                timestamp=datetime.now(),
                contributing_factors=self._identify_contributing_factors(
                    market_data, news_data
                ),
                sector_sentiments=await self._analyze_sector_sentiments(market_data)
            )
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Market emotion analysis failed: {e}")
            return None
    
    async def _calculate_market_sentiment(self, market_data: Dict[str, Any]) -> float:
        """Calculate market sentiment from price data"""
        try:
            price_change = market_data.get('price_change_percent', 0)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            rsi = market_data.get('rsi', 50)
            
            # Price momentum sentiment
            price_sentiment = np.tanh(price_change / 2.0)
            
            # Volume confirmation
            volume_factor = min(2.0, volume_ratio) / 2.0
            
            # RSI sentiment
            rsi_sentiment = (rsi - 50) / 50.0
            
            # Combined market sentiment
            market_sentiment = (price_sentiment * 0.5 + rsi_sentiment * 0.3) * volume_factor
            
            return np.clip(market_sentiment, -1.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Market sentiment calculation failed: {e}")
            return 0.0
    
    async def _analyze_news_sentiment(self, news_items: List[str]) -> float:
        """Analyze news sentiment"""
        if not news_items:
            return 0.0
            
        # Simulated news sentiment analysis
        # In real implementation, would use NLP models
        sentiment_scores = []
        
        positive_keywords = [
            'growth', 'positive', 'strong', 'bullish', 'optimistic',
            'rally', 'gains', 'recovery', 'surge', 'breakthrough'
        ]
        
        negative_keywords = [
            'decline', 'negative', 'weak', 'bearish', 'pessimistic',
            'crash', 'losses', 'recession', 'plunge', 'crisis'
        ]
        
        for news_item in news_items:
            news_lower = news_item.lower()
            positive_count = sum(1 for word in positive_keywords if word in news_lower)
            negative_count = sum(1 for word in negative_keywords if word in news_lower)
            
            if positive_count + negative_count > 0:
                score = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                score = 0.0
                
            sentiment_scores.append(score)
        
        return np.mean(sentiment_scores) if sentiment_scores else 0.0
    
    async def _calculate_fear_index(self, market_data: Dict[str, Any]) -> float:
        """Calculate market fear index"""
        volatility = market_data.get('volatility', 0.2)
        drawdown = abs(market_data.get('max_drawdown', 0))
        volume_spike = max(0, market_data.get('volume_ratio', 1.0) - 1.0)
        
        # Higher volatility, drawdown, and volume spikes indicate fear
        fear_score = (volatility * 2 + drawdown + volume_spike * 0.5) / 3.5
        
        return min(1.0, fear_score)
    
    async def _calculate_greed_index(self, market_data: Dict[str, Any]) -> float:
        """Calculate market greed index"""
        price_momentum = max(0, market_data.get('price_change_percent', 0))
        rsi = market_data.get('rsi', 50)
        volume_confirmation = market_data.get('volume_ratio', 1.0)
        
        # Strong positive momentum with high RSI indicates greed
        momentum_factor = np.tanh(price_momentum / 5.0)
        overbought_factor = max(0, (rsi - 70) / 30) if rsi > 70 else 0
        volume_factor = min(1.0, (volume_confirmation - 1.0) * 2)
        
        greed_score = (momentum_factor + overbought_factor + volume_factor) / 3.0
        
        return min(1.0, greed_score)
    
    async def _analyze_volatility_sentiment(self, market_data: Dict[str, Any]) -> float:
        """Analyze sentiment from volatility patterns"""
        current_vol = market_data.get('volatility', 0.2)
        historical_vol = market_data.get('historical_volatility', 0.2)
        
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        # High volatility typically indicates negative sentiment
        if vol_ratio > 1.5:
            return -0.7  # High fear/uncertainty
        elif vol_ratio > 1.2:
            return -0.3  # Moderate uncertainty
        elif vol_ratio < 0.8:
            return 0.4   # Low volatility = complacency/confidence
        else:
            return 0.0   # Normal volatility
    
    async def _analyze_social_sentiment(self) -> float:
        """Analyze social media sentiment (simulated)"""
        # In real implementation, would analyze Twitter, Reddit, etc.
        return np.random.uniform(-0.3, 0.3)  # Simulated social sentiment
    
    def _determine_sentiment_polarity(self, score: float) -> SentimentPolarity:
        """Determine sentiment polarity from score"""
        if score >= 0.6:
            return SentimentPolarity.VERY_POSITIVE
        elif score >= 0.2:
            return SentimentPolarity.POSITIVE
        elif score <= -0.6:
            return SentimentPolarity.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentPolarity.NEGATIVE
        else:
            return SentimentPolarity.NEUTRAL
    
    async def _calculate_sentiment_confidence(self, market_data: Dict[str, Any], 
                                            score: float) -> float:
        """Calculate confidence in sentiment analysis"""
        # Factors that increase confidence
        volume_confirmation = min(1.0, market_data.get('volume_ratio', 1.0))
        price_consistency = 1.0 - abs(market_data.get('price_volatility', 0.5))
        data_quality = market_data.get('data_quality', 0.8)
        
        # Extreme scores are typically more confident
        score_confidence = abs(score)
        
        confidence = (volume_confirmation + price_consistency + 
                     data_quality + score_confidence) / 4.0
        
        return min(1.0, confidence)
    
    def _identify_contributing_factors(self, market_data: Dict[str, Any], 
                                     news_data: List[str] = None) -> List[str]:
        """Identify factors contributing to sentiment"""
        factors = []
        
        if market_data.get('price_change_percent', 0) > 2.0:
            factors.append('strong_price_momentum')
        elif market_data.get('price_change_percent', 0) < -2.0:
            factors.append('negative_price_action')
            
        if market_data.get('volume_ratio', 1.0) > 1.5:
            factors.append('high_volume_activity')
            
        if market_data.get('volatility', 0.2) > 0.3:
            factors.append('elevated_volatility')
            
        if news_data:
            factors.append('news_influence')
            
        return factors
    
    async def _analyze_sector_sentiments(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze sentiment by market sector"""
        # Simulated sector sentiment analysis
        sectors = ['technology', 'finance', 'healthcare', 'energy', 'consumer']
        base_sentiment = market_data.get('price_change_percent', 0) / 100.0
        
        sector_sentiments = {}
        for sector in sectors:
            # Add sector-specific noise
            sector_sentiment = base_sentiment + np.random.uniform(-0.1, 0.1)
            sector_sentiments[sector] = np.clip(sector_sentiment, -1.0, 1.0)
            
        return sector_sentiments


class EmotionalMarketAnalysis:
    """Advanced emotional market analysis"""
    
    def __init__(self):
        self.emotion_patterns = {}
        self.market_psychology_models = {}
        self.crowd_behavior_analyzer = CrowdPsychologyAnalyzer()
        self.logger = logging.getLogger(__name__)
        
    async def create_emotional_market_profile(self, market_data: Dict[str, Any], 
                                            sentiment: MarketSentiment) -> EmotionalMarketProfile:
        """Create comprehensive emotional market profile"""
        try:
            # Identify dominant emotions
            dominant_emotions = await self._identify_dominant_emotions(
                market_data, sentiment
            )
            
            # Calculate emotional volatility
            emotional_volatility = await self._calculate_emotional_volatility(
                market_data, sentiment
            )
            
            # Analyze sentiment momentum
            sentiment_momentum = await self._calculate_sentiment_momentum(sentiment)
            
            # Calculate fear-greed balance
            fear_greed_balance = self._calculate_fear_greed_balance(sentiment)
            
            # Determine crowd psychology state
            crowd_state = await self._determine_crowd_psychology_state(
                sentiment, market_data
            )
            
            # Measure emotional contagion
            contagion_level = await self._measure_emotional_contagion(market_data)
            
            # Assess market mood stability
            mood_stability = await self._assess_mood_stability(sentiment)
            
            profile = EmotionalMarketProfile(
                profile_id=f"emp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                dominant_emotions=dominant_emotions,
                emotional_volatility=emotional_volatility,
                sentiment_momentum=sentiment_momentum,
                fear_greed_balance=fear_greed_balance,
                crowd_psychology_state=crowd_state,
                emotional_contagion_level=contagion_level,
                market_mood_stability=mood_stability,
                timestamp=datetime.now(),
                historical_patterns=await self._analyze_historical_patterns(market_data)
            )
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Emotional market profile creation failed: {e}")
            return None
    
    async def _identify_dominant_emotions(self, market_data: Dict[str, Any], 
                                        sentiment: MarketSentiment) -> List[EmotionType]:
        """Identify dominant market emotions"""
        emotions = []
        
        # Fear indicators
        if sentiment.market_fear_index > 0.6:
            emotions.append(EmotionType.FEAR)
        if sentiment.market_fear_index > 0.8:
            emotions.append(EmotionType.ANXIETY)
            
        # Greed indicators
        if sentiment.greed_index > 0.6:
            emotions.append(EmotionType.GREED)
        if sentiment.greed_index > 0.8:
            emotions.append(EmotionType.EUPHORIA)
            
        # Confidence/optimism
        if sentiment.sentiment_score > 0.4 and sentiment.confidence > 0.7:
            emotions.append(EmotionType.CONFIDENCE)
            
        # Frustration from stagnation
        volatility = market_data.get('volatility', 0.2)
        price_change = abs(market_data.get('price_change_percent', 0))
        if volatility > 0.3 and price_change < 0.5:
            emotions.append(EmotionType.FRUSTRATION)
            
        # Hope during recovery
        if sentiment.sentiment_score > 0.0 and sentiment.market_fear_index > 0.3:
            emotions.append(EmotionType.HOPE)
            
        return emotions[:3]  # Return top 3 dominant emotions
    
    async def _calculate_emotional_volatility(self, market_data: Dict[str, Any], 
                                            sentiment: MarketSentiment) -> float:
        """Calculate emotional volatility"""
        price_volatility = market_data.get('volatility', 0.2)
        sentiment_extremity = abs(sentiment.sentiment_score)
        fear_greed_imbalance = abs(sentiment.market_fear_index - sentiment.greed_index)
        
        emotional_vol = (price_volatility + sentiment_extremity + fear_greed_imbalance) / 3.0
        return min(1.0, emotional_vol)
    
    async def _calculate_sentiment_momentum(self, sentiment: MarketSentiment) -> float:
        """Calculate sentiment momentum"""
        # In real implementation, would compare with historical sentiment
        current_strength = abs(sentiment.sentiment_score) * sentiment.confidence
        return current_strength
    
    def _calculate_fear_greed_balance(self, sentiment: MarketSentiment) -> float:
        """Calculate fear-greed balance (-1 = all fear, 1 = all greed)"""
        return sentiment.greed_index - sentiment.market_fear_index
    
    async def _determine_crowd_psychology_state(self, sentiment: MarketSentiment, 
                                              market_data: Dict[str, Any]) -> MarketEmotionState:
        """Determine crowd psychology state"""
        fear_index = sentiment.market_fear_index
        greed_index = sentiment.greed_index
        sentiment_score = sentiment.sentiment_score
        volatility = market_data.get('volatility', 0.2)
        
        if fear_index > 0.7 and sentiment_score < -0.5:
            return MarketEmotionState.PANIC_SELLING
        elif greed_index > 0.7 and sentiment_score > 0.5:
            return MarketEmotionState.EUPHORIC_BUYING
        elif sentiment_score > 0.2 and fear_index < 0.3:
            return MarketEmotionState.CAUTIOUS_OPTIMISM
        elif fear_index > 0.5 and abs(sentiment_score) < 0.3:
            return MarketEmotionState.FEARFUL_UNCERTAINTY
        elif greed_index > 0.6 and volatility < 0.2:
            return MarketEmotionState.GREEDY_OVERCONFIDENCE
        elif abs(sentiment_score) < 0.2 and volatility < 0.25:
            return MarketEmotionState.BALANCED_RATIONAL
        elif volatility > 0.3 and abs(sentiment_score) < 0.3:
            return MarketEmotionState.FRUSTRATED_STAGNATION
        elif sentiment_score > 0.0 and fear_index > 0.4:
            return MarketEmotionState.HOPEFUL_RECOVERY
        else:
            return MarketEmotionState.BALANCED_RATIONAL
    
    async def _measure_emotional_contagion(self, market_data: Dict[str, Any]) -> float:
        """Measure emotional contagion level"""
        volume_surge = max(0, market_data.get('volume_ratio', 1.0) - 1.0)
        price_momentum = abs(market_data.get('price_change_percent', 0))
        
        # High volume + strong price moves = emotional contagion
        contagion = (volume_surge + price_momentum / 5.0) / 2.0
        return min(1.0, contagion)
    
    async def _assess_mood_stability(self, sentiment: MarketSentiment) -> float:
        """Assess market mood stability"""
        # Higher confidence and balanced fear/greed = more stable
        balance_stability = 1.0 - abs(sentiment.greed_index - sentiment.market_fear_index)
        confidence_stability = sentiment.confidence
        
        stability = (balance_stability + confidence_stability) / 2.0
        return stability
    
    async def _analyze_historical_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze historical emotional patterns"""
        return {
            'similar_periods': [],  # Would identify similar market conditions
            'pattern_strength': 0.0,  # Strength of recurring patterns
            'seasonal_emotions': {},  # Seasonal emotional patterns
            'cycle_phase': 'unknown'  # Current cycle phase
        }


class CrowdPsychologyAnalyzer:
    """Analyzes crowd psychology and herd behavior"""
    
    def __init__(self):
        self.herd_indicators = {}
        self.contrarian_signals = []
        self.crowd_sentiment_history = deque(maxlen=500)
        
    async def analyze_herd_behavior(self, market_data: Dict[str, Any], 
                                  sentiment: MarketSentiment) -> Dict[str, Any]:
        """Analyze herd behavior patterns"""
        # Identify herd following indicators
        volume_surge = market_data.get('volume_ratio', 1.0) > 2.0
        momentum_following = abs(market_data.get('price_change_percent', 0)) > 3.0
        sentiment_extreme = abs(sentiment.sentiment_score) > 0.7
        
        herd_following_score = sum([volume_surge, momentum_following, sentiment_extreme]) / 3.0
        
        # Identify contrarian opportunities
        contrarian_score = await self._identify_contrarian_opportunities(sentiment)
        
        return {
            'herd_following_score': herd_following_score,
            'contrarian_opportunities': contrarian_score,
            'crowd_behavior_type': self._classify_crowd_behavior(
                herd_following_score, sentiment
            )
        }
    
    async def _identify_contrarian_opportunities(self, sentiment: MarketSentiment) -> float:
        """Identify contrarian trading opportunities"""
        # Extreme sentiment often precedes reversals
        sentiment_extremity = abs(sentiment.sentiment_score)
        
        # High fear can indicate buying opportunity
        fear_opportunity = sentiment.market_fear_index if sentiment.sentiment_score < -0.5 else 0
        
        # High greed can indicate selling opportunity  
        greed_warning = sentiment.greed_index if sentiment.sentiment_score > 0.5 else 0
        
        contrarian_score = (sentiment_extremity + fear_opportunity + greed_warning) / 3.0
        return min(1.0, contrarian_score)
    
    def _classify_crowd_behavior(self, herd_score: float, 
                               sentiment: MarketSentiment) -> str:
        """Classify current crowd behavior"""
        if herd_score > 0.7:
            if sentiment.sentiment_score > 0.5:
                return "euphoric_herding"
            else:
                return "panic_herding"
        elif herd_score > 0.4:
            return "moderate_following"
        else:
            return "rational_behavior"


class IntuitiveDecisionMaking:
    """Intuitive decision making based on emotional intelligence"""
    
    def __init__(self):
        self.intuitive_models = {}
        self.pattern_recognition_engine = PatternRecognitionEngine()
        self.subconscious_analyzer = SubconsciousSignalAnalyzer()
        self.logger = logging.getLogger(__name__)
        
    async def generate_intuitive_insights(self, market_data: Dict[str, Any],
                                        emotional_profile: EmotionalMarketProfile,
                                        consciousness_state: Dict[str, Any] = None) -> List[IntuitiveInsight]:
        """Generate intuitive trading insights"""
        try:
            insights = []
            
            # Pattern-based intuition
            pattern_insights = await self._generate_pattern_insights(
                market_data, emotional_profile
            )
            insights.extend(pattern_insights)
            
            # Emotional-based intuition
            emotional_insights = await self._generate_emotional_insights(
                emotional_profile, market_data
            )
            insights.extend(emotional_insights)
            
            # Subconscious signal insights
            subconscious_insights = await self._generate_subconscious_insights(
                market_data, emotional_profile, consciousness_state
            )
            insights.extend(subconscious_insights)
            
            # Rank insights by conviction level
            insights.sort(key=lambda x: x.conviction_level, reverse=True)
            
            return insights[:5]  # Return top 5 insights
            
        except Exception as e:
            self.logger.error(f"Intuitive insight generation failed: {e}")
            return []
    
    async def _generate_pattern_insights(self, market_data: Dict[str, Any],
                                       emotional_profile: EmotionalMarketProfile) -> List[IntuitiveInsight]:
        """Generate insights based on pattern recognition"""
        insights = []
        
        # Market structure patterns
        if await self._detect_reversal_pattern(market_data, emotional_profile):
            insight = IntuitiveInsight(
                insight_id=f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                insight_type="reversal_pattern",
                market_symbols=list(market_data.get('symbols', ['USDJPY'])),
                intuitive_direction="reversal",
                conviction_level=0.7,
                emotional_basis=[EmotionType.INTUITION, EmotionType.PATTERN_RECOGNITION],
                pattern_recognition={
                    'pattern_type': 'emotional_reversal',
                    'confidence': 0.75,
                    'supporting_indicators': ['extreme_sentiment', 'volume_divergence']
                },
                subconscious_signals=['market_exhaustion', 'sentiment_extremity'],
                timestamp=datetime.now(),
                confidence_score=0.75,
                risk_assessment="moderate"
            )
            insights.append(insight)
            
        return insights
    
    async def _generate_emotional_insights(self, emotional_profile: EmotionalMarketProfile,
                                         market_data: Dict[str, Any]) -> List[IntuitiveInsight]:
        """Generate insights based on emotional analysis"""
        insights = []
        
        # Emotional imbalance insights
        if emotional_profile.fear_greed_balance > 0.6:  # Excessive greed
            insight = IntuitiveInsight(
                insight_id=f"emotion_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                insight_type="emotional_warning",
                market_symbols=list(market_data.get('symbols', ['USDJPY'])),
                intuitive_direction="bearish",
                conviction_level=0.6,
                emotional_basis=[EmotionType.GREED, EmotionType.EUPHORIA],
                pattern_recognition={
                    'emotional_imbalance': emotional_profile.fear_greed_balance,
                    'crowd_psychology': emotional_profile.crowd_psychology_state.value
                },
                subconscious_signals=['market_euphoria', 'greed_extreme'],
                timestamp=datetime.now(),
                confidence_score=0.6,
                risk_assessment="high"
            )
            insights.append(insight)
            
        return insights
    
    async def _generate_subconscious_insights(self, market_data: Dict[str, Any],
                                            emotional_profile: EmotionalMarketProfile,
                                            consciousness_state: Dict[str, Any] = None) -> List[IntuitiveInsight]:
        """Generate insights from subconscious processing"""
        if not consciousness_state:
            return []
            
        insights = []
        
        # Subconscious pattern detection
        subconscious_strength = consciousness_state.get('subconscious_activity', 0.5)
        if subconscious_strength > 0.7:
            insight = IntuitiveInsight(
                insight_id=f"subconscious_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                insight_type="subconscious_pattern",
                market_symbols=list(market_data.get('symbols', ['USDJPY'])),
                intuitive_direction=self._determine_subconscious_direction(consciousness_state),
                conviction_level=subconscious_strength,
                emotional_basis=[EmotionType.CURIOSITY, EmotionType.INTUITION],
                pattern_recognition={
                    'subconscious_processing': subconscious_strength,
                    'pattern_complexity': 'high'
                },
                subconscious_signals=['deep_pattern_recognition', 'non_conscious_processing'],
                timestamp=datetime.now(),
                confidence_score=subconscious_strength,
                risk_assessment="speculative"
            )
            insights.append(insight)
            
        return insights
    
    async def _detect_reversal_pattern(self, market_data: Dict[str, Any],
                                     emotional_profile: EmotionalMarketProfile) -> bool:
        """Detect potential reversal patterns"""
        # Emotional extremity
        emotional_extreme = emotional_profile.emotional_volatility > 0.8
        
        # Sentiment extremity
        if hasattr(emotional_profile, 'sentiment_score'):
            sentiment_extreme = abs(getattr(emotional_profile, 'sentiment_score', 0)) > 0.7
        else:
            sentiment_extreme = False
            
        # Volume divergence
        volume_divergence = market_data.get('volume_ratio', 1.0) < 0.7
        
        return emotional_extreme and sentiment_extreme and volume_divergence
    
    def _determine_subconscious_direction(self, consciousness_state: Dict[str, Any]) -> str:
        """Determine trading direction from subconscious signals"""
        subconscious_bias = consciousness_state.get('subconscious_bias', 0.0)
        
        if subconscious_bias > 0.3:
            return "bullish"
        elif subconscious_bias < -0.3:
            return "bearish"
        else:
            return "neutral"


class PatternRecognitionEngine:
    """Advanced pattern recognition for emotional trading"""
    
    def __init__(self):
        self.pattern_library = {}
        self.learning_models = {}
        
    async def recognize_emotional_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recognize emotional patterns in market data"""
        # Simulated pattern recognition
        return {
            'patterns_detected': ['reversal_pattern', 'continuation_pattern'],
            'confidence_scores': [0.75, 0.65],
            'pattern_strength': 0.7
        }


class SubconsciousSignalAnalyzer:
    """Analyzes subconscious signals and non-conscious processing"""
    
    def __init__(self):
        self.signal_processors = {}
        self.pattern_detectors = {}
        
    async def analyze_subconscious_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze subconscious trading signals"""
        # Simulated subconscious analysis
        return {
            'subconscious_activity': np.random.uniform(0.3, 0.9),
            'subconscious_bias': np.random.uniform(-0.5, 0.5),
            'pattern_complexity': 'medium'
        }


class EmotionalIntelligence:
    """Main emotional intelligence system"""
    
    def __init__(self):
        self.emotional_processor = EmotionalProcessor()
        self.market_analyzer = EmotionalMarketAnalysis()
        self.intuitive_engine = IntuitiveDecisionMaking()
        self.emotional_state = EmotionalState(
            emotion_type=EmotionType.CURIOSITY,
            intensity=EmotionalIntensity.MODERATE,
            valence=0.3,
            arousal=0.5,
            timestamp=datetime.now()
        )
        self.logger = logging.getLogger(__name__)
        
    async def analyze_market_emotions(self, market_data: Dict[str, Any],
                                    additional_data: Dict[str, Any] = None) -> TradingResult[Dict[str, Any]]:
        """Complete emotional market analysis"""
        try:
            # Analyze market sentiment
            market_sentiment = await self.emotional_processor.analyze_market_emotions(
                market_data, additional_data.get('news_data', []) if additional_data else []
            )
            
            if not market_sentiment:
                return TradingResult.failure("Market sentiment analysis failed")
            
            # Create emotional market profile
            emotional_profile = await self.market_analyzer.create_emotional_market_profile(
                market_data, market_sentiment
            )
            
            if not emotional_profile:
                return TradingResult.failure("Emotional profile creation failed")
            
            # Generate intuitive insights
            consciousness_state = additional_data.get('consciousness_state') if additional_data else None
            intuitive_insights = await self.intuitive_engine.generate_intuitive_insights(
                market_data, emotional_profile, consciousness_state
            )
            
            # Compile comprehensive analysis
            analysis_result = {
                'market_sentiment': {
                    'polarity': market_sentiment.polarity.value,
                    'score': market_sentiment.sentiment_score,
                    'confidence': market_sentiment.confidence,
                    'fear_index': market_sentiment.market_fear_index,
                    'greed_index': market_sentiment.greed_index,
                    'contributing_factors': market_sentiment.contributing_factors
                },
                'emotional_profile': {
                    'dominant_emotions': [e.value for e in emotional_profile.dominant_emotions],
                    'emotional_volatility': emotional_profile.emotional_volatility,
                    'crowd_psychology_state': emotional_profile.crowd_psychology_state.value,
                    'fear_greed_balance': emotional_profile.fear_greed_balance,
                    'mood_stability': emotional_profile.market_mood_stability
                },
                'intuitive_insights': [
                    {
                        'type': insight.insight_type,
                        'direction': insight.intuitive_direction,
                        'conviction': insight.conviction_level,
                        'symbols': insight.market_symbols,
                        'emotional_basis': [e.value for e in insight.emotional_basis],
                        'risk_assessment': insight.risk_assessment
                    }
                    for insight in intuitive_insights
                ],
                'trading_recommendations': await self._generate_emotional_trading_recommendations(
                    market_sentiment, emotional_profile, intuitive_insights
                ),
                'risk_assessment': await self._assess_emotional_risks(
                    market_sentiment, emotional_profile
                )
            }
            
            return TradingResult.success(analysis_result)
            
        except Exception as e:
            self.logger.error(f"Emotional intelligence analysis failed: {e}")
            return TradingResult.failure(f"Analysis error: {e}")
    
    async def _generate_emotional_trading_recommendations(self, sentiment: MarketSentiment,
                                                        profile: EmotionalMarketProfile,
                                                        insights: List[IntuitiveInsight]) -> List[Dict[str, Any]]:
        """Generate trading recommendations based on emotional analysis"""
        recommendations = []
        
        # Fear-based opportunities
        if sentiment.market_fear_index > 0.7 and profile.crowd_psychology_state == MarketEmotionState.PANIC_SELLING:
            recommendations.append({
                'type': 'contrarian_buy',
                'rationale': 'extreme_fear_opportunity',
                'confidence': 0.7,
                'risk_level': 'moderate'
            })
        
        # Greed-based warnings
        if sentiment.greed_index > 0.7 and profile.crowd_psychology_state == MarketEmotionState.EUPHORIC_BUYING:
            recommendations.append({
                'type': 'take_profits',
                'rationale': 'excessive_greed_warning',
                'confidence': 0.8,
                'risk_level': 'high'
            })
        
        # Intuitive insights
        for insight in insights:
            if insight.conviction_level > 0.6:
                recommendations.append({
                    'type': 'intuitive_trade',
                    'direction': insight.intuitive_direction,
                    'rationale': insight.insight_type,
                    'confidence': insight.conviction_level,
                    'risk_level': insight.risk_assessment
                })
        
        return recommendations
    
    async def _assess_emotional_risks(self, sentiment: MarketSentiment,
                                    profile: EmotionalMarketProfile) -> Dict[str, Any]:
        """Assess emotional risks in current market"""
        risk_factors = []
        overall_risk = "low"
        
        # Sentiment extremity risks
        if abs(sentiment.sentiment_score) > 0.8:
            risk_factors.append("extreme_sentiment")
            overall_risk = "high"
        
        # Emotional volatility risks
        if profile.emotional_volatility > 0.7:
            risk_factors.append("high_emotional_volatility")
            if overall_risk == "low":
                overall_risk = "moderate"
        
        # Crowd psychology risks
        if profile.crowd_psychology_state in [MarketEmotionState.PANIC_SELLING, MarketEmotionState.EUPHORIC_BUYING]:
            risk_factors.append("extreme_crowd_behavior")
            overall_risk = "high"
        
        # Fear-greed imbalance
        if abs(profile.fear_greed_balance) > 0.8:
            risk_factors.append("fear_greed_imbalance")
            if overall_risk == "low":
                overall_risk = "moderate"
        
        return {
            'overall_risk_level': overall_risk,
            'risk_factors': risk_factors,
            'emotional_stability': profile.market_mood_stability,
            'recommended_position_size': self._calculate_emotional_position_size(overall_risk),
            'monitoring_frequency': self._recommend_monitoring_frequency(overall_risk)
        }
    
    def _calculate_emotional_position_size(self, risk_level: str) -> float:
        """Calculate recommended position size based on emotional risk"""
        size_multipliers = {
            'low': 1.0,
            'moderate': 0.7,
            'high': 0.4
        }
        return size_multipliers.get(risk_level, 0.5)
    
    def _recommend_monitoring_frequency(self, risk_level: str) -> str:
        """Recommend monitoring frequency based on emotional risk"""
        frequencies = {
            'low': 'hourly',
            'moderate': '30_minutes',
            'high': '15_minutes'
        }
        return frequencies.get(risk_level, '30_minutes')


# Global emotional intelligence instance
emotional_intelligence = EmotionalIntelligence()


async def analyze_market_emotions(market_data: Dict[str, Any], 
                                additional_data: Dict[str, Any] = None) -> TradingResult[Dict[str, Any]]:
    """Analyze market emotions using global EI system"""
    return await emotional_intelligence.analyze_market_emotions(market_data, additional_data)


async def get_emotional_state() -> EmotionalState:
    """Get current emotional state of the AI"""
    return emotional_intelligence.emotional_state