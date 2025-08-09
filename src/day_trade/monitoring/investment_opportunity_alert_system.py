#!/usr/bin/env python3
"""
投資機会アラートシステム
Issue #318: 監視・アラートシステム - Phase 4

投資機会検出・利益機会分析・リスク評価・投資推奨アラート
- テクニカル分析シグナル
- ファンダメンタル分析アラート
- ポートフォリオ最適化推奨
- 市場異常検知・投資機会発見
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class OpportunityType(Enum):
    """投資機会タイプ"""

    TECHNICAL_BREAKOUT = "technical_breakout"
    MOMENTUM_SIGNAL = "momentum_signal"
    REVERSAL_PATTERN = "reversal_pattern"
    VOLUME_ANOMALY = "volume_anomaly"
    PRICE_UNDERVALUATION = "price_undervaluation"
    EARNINGS_SURPRISE = "earnings_surprise"
    DIVIDEND_OPPORTUNITY = "dividend_opportunity"
    SECTOR_ROTATION = "sector_rotation"
    PAIRS_TRADING = "pairs_trading"
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"
    NEWS_SENTIMENT = "news_sentiment"
    VOLATILITY_SQUEEZE = "volatility_squeeze"


class OpportunitySeverity(Enum):
    """機会重要度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradingAction(Enum):
    """トレーディングアクション"""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE = "reduce"
    INCREASE = "increase"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class TimeHorizon(Enum):
    """投資期間"""

    INTRADAY = "intraday"
    SHORT_TERM = "short_term"  # 1週間以内
    MEDIUM_TERM = "medium_term"  # 1ヶ月以内
    LONG_TERM = "long_term"  # 3ヶ月以上


@dataclass
class OpportunityConfig:
    """機会検出設定"""

    rule_id: str
    rule_name: str
    opportunity_type: OpportunityType
    symbols: List[str]
    time_horizon: TimeHorizon

    # 検出しきい値
    confidence_threshold: float = 0.7
    profit_potential_threshold: float = 5.0  # %
    risk_reward_ratio: float = 2.0

    # テクニカル指標設定
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    bollinger_deviation: float = 2.0
    macd_signal_threshold: float = 0
    volume_spike_threshold: float = 2.0

    # ファンダメンタル設定
    pe_ratio_threshold: float = 15.0
    earnings_growth_threshold: float = 20.0
    dividend_yield_threshold: float = 3.0

    # リスク管理設定
    max_position_size: float = 0.1  # 10%
    stop_loss_percentage: float = 5.0  # 5%
    take_profit_percentage: float = 10.0  # 10%

    enabled: bool = True
    check_interval_minutes: int = 15


@dataclass
class InvestmentOpportunity:
    """投資機会"""

    opportunity_id: str
    timestamp: datetime
    symbol: str
    opportunity_type: OpportunityType
    severity: OpportunitySeverity

    # 機会詳細
    recommended_action: TradingAction
    target_price: Optional[float]
    current_price: float
    profit_potential: float  # %
    confidence_score: float
    time_horizon: TimeHorizon

    # リスク・リワード
    risk_level: str
    risk_reward_ratio: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]

    # 分析データ
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    fundamental_data: Dict[str, Any] = field(default_factory=dict)

    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    executed: bool = False
    executed_at: Optional[datetime] = None


@dataclass
class MarketCondition:
    """市場状況"""

    timestamp: datetime
    market_trend: str  # bull, bear, sideways
    volatility_level: str  # low, medium, high
    volume_trend: str  # increasing, decreasing, stable
    sector_performance: Dict[str, float]
    market_sentiment: float  # -1 to 1
    fear_greed_index: Optional[float] = None


@dataclass
class AlertConfig:
    """アラート設定"""

    # 基本アラート設定
    enable_email_notifications: bool = False
    enable_push_notifications: bool = False
    enable_trade_execution: bool = False

    # 機会フィルタリング
    min_confidence_score: float = 0.6
    min_profit_potential: float = 3.0
    min_risk_reward_ratio: float = 1.5

    # 実行設定
    max_opportunities_per_hour: int = 10
    opportunity_cooldown_minutes: int = 30
    alert_history_retention_days: int = 30

    # 市場条件フィルター
    enable_market_condition_filter: bool = True
    avoid_high_volatility_periods: bool = True
    min_market_sentiment: float = -0.3


class InvestmentOpportunityAlertSystem:
    """投資機会アラートシステム"""

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.opportunity_configs: Dict[str, OpportunityConfig] = {}
        self.opportunities: List[InvestmentOpportunity] = []
        self.market_condition: Optional[MarketCondition] = None

        # アラートハンドラー
        self.alert_handlers: Dict[OpportunitySeverity, List[Callable]] = {
            severity: [] for severity in OpportunitySeverity
        }

        # 価格データ・指標データ（模擬）
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.technical_indicators: Dict[str, Dict[str, float]] = {}

        # アラート抑制
        self.recent_opportunities: Dict[str, datetime] = {}

        # 監視タスク
        self.monitor_task: Optional[asyncio.Task] = None
        self._is_running = False

        # デフォルト設定
        self._setup_default_opportunity_configs()

    def _setup_default_opportunity_configs(self) -> None:
        """デフォルト機会検出設定"""
        default_configs = [
            # テクニカル分析ブレイクアウト
            OpportunityConfig(
                rule_id="technical_breakout_topix",
                rule_name="TOPIX テクニカルブレイクアウト",
                opportunity_type=OpportunityType.TECHNICAL_BREAKOUT,
                symbols=["TOPIX", "N225"],
                time_horizon=TimeHorizon.SHORT_TERM,
                confidence_threshold=0.8,
                profit_potential_threshold=5.0,
                check_interval_minutes=15,
            ),
            # モメンタム分析
            OpportunityConfig(
                rule_id="momentum_large_cap",
                rule_name="大型株モメンタム",
                opportunity_type=OpportunityType.MOMENTUM_SIGNAL,
                symbols=["7203", "8306", "9984", "4502"],  # トヨタ、三菱UFJ、SBG、武田
                time_horizon=TimeHorizon.MEDIUM_TERM,
                confidence_threshold=0.75,
                profit_potential_threshold=7.0,
                rsi_oversold=25,
                rsi_overbought=75,
            ),
            # リバーサルパターン
            OpportunityConfig(
                rule_id="reversal_pattern_detection",
                rule_name="反転パターン検出",
                opportunity_type=OpportunityType.REVERSAL_PATTERN,
                symbols=["7182", "6501", "8058"],  # ゆうちょ、日立、三菱商事
                time_horizon=TimeHorizon.SHORT_TERM,
                confidence_threshold=0.7,
                profit_potential_threshold=6.0,
                risk_reward_ratio=2.5,
            ),
            # 出来高異常
            OpportunityConfig(
                rule_id="volume_anomaly_alert",
                rule_name="出来高異常検出",
                opportunity_type=OpportunityType.VOLUME_ANOMALY,
                symbols=["*"],  # 全銘柄
                time_horizon=TimeHorizon.INTRADAY,
                confidence_threshold=0.8,
                volume_spike_threshold=3.0,
                check_interval_minutes=5,
            ),
            # 株価割安検出
            OpportunityConfig(
                rule_id="undervaluation_screening",
                rule_name="割安株スクリーニング",
                opportunity_type=OpportunityType.PRICE_UNDERVALUATION,
                symbols=["*"],
                time_horizon=TimeHorizon.LONG_TERM,
                confidence_threshold=0.85,
                pe_ratio_threshold=12.0,
                profit_potential_threshold=15.0,
            ),
            # 配当機会
            OpportunityConfig(
                rule_id="dividend_opportunity",
                rule_name="高配当投資機会",
                opportunity_type=OpportunityType.DIVIDEND_OPPORTUNITY,
                symbols=["8306", "7182", "9437"],  # 三菱UFJ、ゆうちょ、NTTドコモ
                time_horizon=TimeHorizon.LONG_TERM,
                dividend_yield_threshold=4.0,
                confidence_threshold=0.7,
            ),
            # ボラティリティスクイーズ
            OpportunityConfig(
                rule_id="volatility_squeeze",
                rule_name="ボラティリティスクイーズ",
                opportunity_type=OpportunityType.VOLATILITY_SQUEEZE,
                symbols=["N225", "TOPIX"],
                time_horizon=TimeHorizon.SHORT_TERM,
                confidence_threshold=0.75,
                bollinger_deviation=1.5,
            ),
        ]

        for config in default_configs:
            self.add_opportunity_config(config)

    def add_opportunity_config(self, config: OpportunityConfig) -> None:
        """機会検出設定追加"""
        self.opportunity_configs[config.rule_id] = config
        logger.info(f"投資機会検出設定追加: {config.rule_id}")

    def remove_opportunity_config(self, rule_id: str) -> None:
        """機会検出設定削除"""
        if rule_id in self.opportunity_configs:
            del self.opportunity_configs[rule_id]
            logger.info(f"投資機会検出設定削除: {rule_id}")

    def register_alert_handler(
        self, severity: OpportunitySeverity, handler: Callable
    ) -> None:
        """アラートハンドラー登録"""
        self.alert_handlers[severity].append(handler)
        logger.info(f"機会アラートハンドラー登録: {severity.value}")

    async def start_monitoring(self) -> None:
        """監視開始"""
        if self._is_running:
            logger.warning("投資機会監視は既に実行中です")
            return

        self._is_running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("投資機会アラート監視開始")

    async def stop_monitoring(self) -> None:
        """監視停止"""
        self._is_running = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("投資機会アラート監視停止")

    async def _monitoring_loop(self) -> None:
        """監視ループ"""
        while self._is_running:
            try:
                # 市場状況更新
                await self._update_market_condition()

                # 価格データ更新
                await self._update_price_data()

                # テクニカル指標計算
                await self._calculate_technical_indicators()

                # 機会検出実行
                await self._detect_opportunities()

                # 履歴クリーンアップ
                await self._cleanup_opportunity_history()

                # 最短チェック間隔で実行（デフォルト5分）
                min_interval = min(
                    [
                        config.check_interval_minutes
                        for config in self.opportunity_configs.values()
                    ],
                    default=5,
                )
                await asyncio.sleep(min_interval * 60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"投資機会監視ループエラー: {e}")
                await asyncio.sleep(60)

    async def _update_market_condition(self) -> None:
        """市場状況更新"""
        # 模擬的な市場状況データ
        import random

        market_trends = ["bull", "bear", "sideways"]
        volatility_levels = ["low", "medium", "high"]
        volume_trends = ["increasing", "decreasing", "stable"]

        # 市場センチメント（-1から1の間）
        market_sentiment = random.uniform(-0.5, 0.8)

        # セクターパフォーマンス
        sectors = ["technology", "finance", "healthcare", "energy", "consumer"]
        sector_performance = {sector: random.uniform(-3.0, 5.0) for sector in sectors}

        self.market_condition = MarketCondition(
            timestamp=datetime.now(),
            market_trend=random.choice(market_trends),
            volatility_level=random.choice(volatility_levels),
            volume_trend=random.choice(volume_trends),
            sector_performance=sector_performance,
            market_sentiment=market_sentiment,
            fear_greed_index=random.uniform(20, 80),
        )

    async def _update_price_data(self) -> None:
        """価格データ更新"""
        # 模擬的な価格データ生成
        symbols = [
            "TOPIX",
            "N225",
            "7203",
            "8306",
            "9984",
            "4502",
            "7182",
            "6501",
            "8058",
        ]

        for symbol in symbols:
            # 過去30日分の価格データ生成
            dates = pd.date_range(end=datetime.now(), periods=30, freq="D")

            # ランダムウォーク価格生成
            base_price = 1000 + hash(symbol) % 5000
            returns = np.random.normal(0.001, 0.02, 30)  # 平均0.1%、標準偏差2%
            prices = [base_price]

            for return_rate in returns[1:]:
                prices.append(prices[-1] * (1 + return_rate))

            # OHLCV データ生成
            data = []
            for i, (date, close) in enumerate(zip(dates, prices)):
                open_price = close * random.uniform(0.99, 1.01)
                high = max(open_price, close) * random.uniform(1.0, 1.02)
                low = min(open_price, close) * random.uniform(0.98, 1.0)
                volume = random.randint(100000, 1000000)

                data.append(
                    {
                        "Date": date,
                        "Open": open_price,
                        "High": high,
                        "Low": low,
                        "Close": close,
                        "Volume": volume,
                    }
                )

            self.price_data[symbol] = pd.DataFrame(data).set_index("Date")

    async def _calculate_technical_indicators(self) -> None:
        """テクニカル指標計算"""
        for symbol, df in self.price_data.items():
            if len(df) < 20:  # 最低限必要なデータポイント
                continue

            try:
                indicators = {}

                # 移動平均
                indicators["SMA_5"] = df["Close"].rolling(5).mean().iloc[-1]
                indicators["SMA_20"] = df["Close"].rolling(20).mean().iloc[-1]
                indicators["EMA_12"] = df["Close"].ewm(span=12).mean().iloc[-1]
                indicators["EMA_26"] = df["Close"].ewm(span=26).mean().iloc[-1]

                # RSI
                delta = df["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                indicators["RSI"] = (100 - (100 / (1 + rs))).iloc[-1]

                # MACD
                indicators["MACD"] = indicators["EMA_12"] - indicators["EMA_26"]
                macd_signal = pd.Series([indicators["MACD"]]).ewm(span=9).mean()
                indicators["MACD_Signal"] = macd_signal.iloc[-1]
                indicators["MACD_Histogram"] = (
                    indicators["MACD"] - indicators["MACD_Signal"]
                )

                # ボリンジャーバンド
                bb_period = 20
                sma = df["Close"].rolling(bb_period).mean()
                std = df["Close"].rolling(bb_period).std()
                indicators["BB_Upper"] = (sma + 2 * std).iloc[-1]
                indicators["BB_Lower"] = (sma - 2 * std).iloc[-1]
                indicators["BB_Middle"] = sma.iloc[-1]

                # 現在価格のボリンジャーバンド位置
                current_price = df["Close"].iloc[-1]
                bb_width = indicators["BB_Upper"] - indicators["BB_Lower"]
                if bb_width > 0:
                    indicators["BB_Position"] = (
                        current_price - indicators["BB_Lower"]
                    ) / bb_width
                else:
                    indicators["BB_Position"] = 0.5

                # 出来高指標
                indicators["Volume_SMA"] = df["Volume"].rolling(20).mean().iloc[-1]
                indicators["Volume_Ratio"] = (
                    df["Volume"].iloc[-1] / indicators["Volume_SMA"]
                )

                # 価格変化率
                indicators["Price_Change_1D"] = (
                    df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1
                ) * 100
                indicators["Price_Change_5D"] = (
                    df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1
                ) * 100

                self.technical_indicators[symbol] = indicators

            except Exception as e:
                logger.error(f"テクニカル指標計算エラー {symbol}: {e}")

    async def _detect_opportunities(self) -> None:
        """投資機会検出"""
        for rule_id, config in self.opportunity_configs.items():
            if not config.enabled:
                continue

            try:
                # チェック間隔判定
                if not await self._should_run_opportunity_check(config):
                    continue

                # 機会検出実行
                opportunities = await self._analyze_opportunity(config)

                for opportunity in opportunities:
                    # 市場状況フィルター
                    if self.config.enable_market_condition_filter:
                        if not await self._passes_market_filter(opportunity):
                            continue

                    # 品質フィルター
                    if not await self._passes_quality_filter(opportunity):
                        continue

                    # アラート生成
                    await self._generate_opportunity_alert(opportunity)

            except Exception as e:
                logger.error(f"機会検出エラー {rule_id}: {e}")

    async def _should_run_opportunity_check(self, config: OpportunityConfig) -> bool:
        """機会チェック実行判定"""
        check_key = f"last_check_{config.rule_id}"

        if check_key not in self.recent_opportunities:
            return True

        last_check = self.recent_opportunities[check_key]
        elapsed_minutes = (datetime.now() - last_check).total_seconds() / 60

        return elapsed_minutes >= config.check_interval_minutes

    async def _analyze_opportunity(
        self, config: OpportunityConfig
    ) -> List[InvestmentOpportunity]:
        """機会分析"""
        opportunities = []

        # 対象銘柄取得
        target_symbols = config.symbols
        if "*" in target_symbols:
            target_symbols = list(self.price_data.keys())

        for symbol in target_symbols:
            if symbol not in self.technical_indicators:
                continue

            try:
                opportunity = await self._analyze_symbol_opportunity(symbol, config)
                if opportunity:
                    opportunities.append(opportunity)
            except Exception as e:
                logger.error(f"銘柄機会分析エラー {symbol}: {e}")

        return opportunities

    async def _analyze_symbol_opportunity(
        self, symbol: str, config: OpportunityConfig
    ) -> Optional[InvestmentOpportunity]:
        """銘柄機会分析"""

        indicators = self.technical_indicators.get(symbol, {})
        price_data = self.price_data.get(symbol)

        if not indicators or price_data is None or len(price_data) == 0:
            return None

        current_price = price_data["Close"].iloc[-1]

        # 機会タイプ別分析
        if config.opportunity_type == OpportunityType.TECHNICAL_BREAKOUT:
            return await self._analyze_technical_breakout(
                symbol, config, indicators, current_price
            )
        elif config.opportunity_type == OpportunityType.MOMENTUM_SIGNAL:
            return await self._analyze_momentum_signal(
                symbol, config, indicators, current_price
            )
        elif config.opportunity_type == OpportunityType.REVERSAL_PATTERN:
            return await self._analyze_reversal_pattern(
                symbol, config, indicators, current_price
            )
        elif config.opportunity_type == OpportunityType.VOLUME_ANOMALY:
            return await self._analyze_volume_anomaly(
                symbol, config, indicators, current_price
            )
        elif config.opportunity_type == OpportunityType.VOLATILITY_SQUEEZE:
            return await self._analyze_volatility_squeeze(
                symbol, config, indicators, current_price
            )
        else:
            # その他の機会タイプは基本的な分析
            return await self._analyze_generic_opportunity(
                symbol, config, indicators, current_price
            )

    async def _analyze_technical_breakout(
        self,
        symbol: str,
        config: OpportunityConfig,
        indicators: Dict[str, float],
        current_price: float,
    ) -> Optional[InvestmentOpportunity]:
        """テクニカルブレイクアウト分析"""

        # ブレイクアウト条件
        sma_20 = indicators.get("SMA_20", current_price)
        bb_upper = indicators.get("BB_Upper", current_price * 1.02)
        volume_ratio = indicators.get("Volume_Ratio", 1.0)

        # 上方ブレイクアウト判定
        price_above_sma = current_price > sma_20 * 1.02  # 2%以上上
        volume_confirmation = volume_ratio > 1.5  # 出来高1.5倍以上
        near_bb_upper = current_price > bb_upper * 0.98  # ボリンジャー上限付近

        if price_above_sma and volume_confirmation and near_bb_upper:
            confidence = 0.6 + (volume_ratio - 1.5) * 0.1  # 出来高に応じて信頼度調整
            confidence = min(confidence, 0.95)

            profit_potential = ((bb_upper * 1.05 - current_price) / current_price) * 100

            if (
                confidence >= config.confidence_threshold
                and profit_potential >= config.profit_potential_threshold
            ):
                return InvestmentOpportunity(
                    opportunity_id=f"breakout_{symbol}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=symbol,
                    opportunity_type=config.opportunity_type,
                    severity=OpportunitySeverity.HIGH,
                    recommended_action=TradingAction.BUY,
                    target_price=bb_upper * 1.05,
                    current_price=current_price,
                    profit_potential=profit_potential,
                    confidence_score=confidence,
                    time_horizon=config.time_horizon,
                    risk_level="medium",
                    risk_reward_ratio=profit_potential / config.stop_loss_percentage,
                    stop_loss_price=current_price
                    * (1 - config.stop_loss_percentage / 100),
                    take_profit_price=current_price * (1 + profit_potential / 100),
                    technical_indicators=indicators,
                    message=f"{symbol} 上方ブレイクアウト機会検出",
                )

        return None

    async def _analyze_momentum_signal(
        self,
        symbol: str,
        config: OpportunityConfig,
        indicators: Dict[str, float],
        current_price: float,
    ) -> Optional[InvestmentOpportunity]:
        """モメンタム分析"""

        rsi = indicators.get("RSI", 50)
        macd = indicators.get("MACD", 0)
        macd_signal = indicators.get("MACD_Signal", 0)
        price_change_5d = indicators.get("Price_Change_5D", 0)

        # 強い上昇モメンタム判定
        rsi_bullish = 50 < rsi < config.rsi_overbought
        macd_bullish = macd > macd_signal and macd > 0
        positive_momentum = price_change_5d > 2.0  # 5日間で2%以上上昇

        if rsi_bullish and macd_bullish and positive_momentum:
            confidence = 0.65 + (price_change_5d / 20.0)  # モメンタムに応じて調整
            confidence = min(confidence, 0.9)

            # 利益目標をモメンタムに基づいて設定
            profit_potential = price_change_5d * 1.5  # 現在のモメンタムの1.5倍

            if (
                confidence >= config.confidence_threshold
                and profit_potential >= config.profit_potential_threshold
            ):
                return InvestmentOpportunity(
                    opportunity_id=f"momentum_{symbol}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=symbol,
                    opportunity_type=config.opportunity_type,
                    severity=OpportunitySeverity.MEDIUM,
                    recommended_action=TradingAction.BUY,
                    target_price=current_price * (1 + profit_potential / 100),
                    current_price=current_price,
                    profit_potential=profit_potential,
                    confidence_score=confidence,
                    time_horizon=config.time_horizon,
                    risk_level="medium",
                    risk_reward_ratio=profit_potential / config.stop_loss_percentage,
                    stop_loss_price=current_price
                    * (1 - config.stop_loss_percentage / 100),
                    take_profit_price=current_price * (1 + profit_potential / 100),
                    technical_indicators=indicators,
                    message=f"{symbol} 強い上昇モメンタム検出",
                )

        return None

    async def _analyze_reversal_pattern(
        self,
        symbol: str,
        config: OpportunityConfig,
        indicators: Dict[str, float],
        current_price: float,
    ) -> Optional[InvestmentOpportunity]:
        """リバーサルパターン分析"""

        rsi = indicators.get("RSI", 50)
        bb_position = indicators.get("BB_Position", 0.5)
        price_change_1d = indicators.get("Price_Change_1D", 0)

        # 過売り反転パターン
        oversold_rsi = rsi < config.rsi_oversold
        near_bb_lower = bb_position < 0.2  # ボリンジャー下限付近
        recent_decline = price_change_1d < -2.0  # 直近で2%以上下落

        if oversold_rsi and near_bb_lower and recent_decline:
            # RSIが低いほど反転の可能性が高い
            confidence = 0.5 + ((config.rsi_oversold - rsi) / config.rsi_oversold) * 0.3
            confidence = min(confidence, 0.85)

            # 反転による利益目標
            profit_potential = abs(price_change_1d) * 2.0  # 下落分の2倍の反転

            if (
                confidence >= config.confidence_threshold
                and profit_potential >= config.profit_potential_threshold
            ):
                return InvestmentOpportunity(
                    opportunity_id=f"reversal_{symbol}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=symbol,
                    opportunity_type=config.opportunity_type,
                    severity=OpportunitySeverity.MEDIUM,
                    recommended_action=TradingAction.BUY,
                    target_price=current_price * (1 + profit_potential / 100),
                    current_price=current_price,
                    profit_potential=profit_potential,
                    confidence_score=confidence,
                    time_horizon=config.time_horizon,
                    risk_level="high",  # 反転狙いはリスクが高い
                    risk_reward_ratio=profit_potential / config.stop_loss_percentage,
                    stop_loss_price=current_price
                    * (1 - config.stop_loss_percentage / 100),
                    take_profit_price=current_price * (1 + profit_potential / 100),
                    technical_indicators=indicators,
                    message=f"{symbol} 過売り反転パターン検出",
                )

        return None

    async def _analyze_volume_anomaly(
        self,
        symbol: str,
        config: OpportunityConfig,
        indicators: Dict[str, float],
        current_price: float,
    ) -> Optional[InvestmentOpportunity]:
        """出来高異常分析"""

        volume_ratio = indicators.get("Volume_Ratio", 1.0)
        price_change_1d = indicators.get("Price_Change_1D", 0)

        # 異常出来高判定
        volume_spike = volume_ratio >= config.volume_spike_threshold
        significant_price_move = abs(price_change_1d) > 1.0  # 1%以上の価格変動

        if volume_spike and significant_price_move:
            confidence = 0.6 + (volume_ratio - config.volume_spike_threshold) * 0.05
            confidence = min(confidence, 0.9)

            # 価格変動方向に基づくアクション決定
            if price_change_1d > 0:
                action = TradingAction.BUY
                profit_potential = price_change_1d * 1.2  # 現在の上昇の1.2倍
            else:
                action = TradingAction.SELL
                profit_potential = abs(price_change_1d) * 1.2

            if (
                confidence >= config.confidence_threshold
                and profit_potential >= config.profit_potential_threshold
            ):
                return InvestmentOpportunity(
                    opportunity_id=f"volume_{symbol}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=symbol,
                    opportunity_type=config.opportunity_type,
                    severity=OpportunitySeverity.HIGH,
                    recommended_action=action,
                    target_price=current_price
                    * (
                        1
                        + profit_potential
                        / 100
                        * (1 if action == TradingAction.BUY else -1)
                    ),
                    current_price=current_price,
                    profit_potential=profit_potential,
                    confidence_score=confidence,
                    time_horizon=config.time_horizon,
                    risk_level="high",  # 出来高異常は高リスク
                    risk_reward_ratio=profit_potential / config.stop_loss_percentage,
                    stop_loss_price=current_price
                    * (1 - config.stop_loss_percentage / 100),
                    technical_indicators=indicators,
                    message=f"{symbol} 異常出来高検出 ({volume_ratio:.1f}倍)",
                )

        return None

    async def _analyze_volatility_squeeze(
        self,
        symbol: str,
        config: OpportunityConfig,
        indicators: Dict[str, float],
        current_price: float,
    ) -> Optional[InvestmentOpportunity]:
        """ボラティリティスクイーズ分析"""

        bb_upper = indicators.get("BB_Upper", current_price * 1.02)
        bb_lower = indicators.get("BB_Lower", current_price * 0.98)
        bb_position = indicators.get("BB_Position", 0.5)

        # ボリンジャーバンド幅
        bb_width = (bb_upper - bb_lower) / current_price

        # ボラティリティスクイーズ判定（バンド幅が狭い）
        narrow_bands = bb_width < 0.04  # 4%以下の狭いバンド
        central_position = 0.3 < bb_position < 0.7  # 中央付近

        if narrow_bands and central_position:
            confidence = 0.7 + (0.04 - bb_width) * 10  # バンドが狭いほど信頼度高い
            confidence = min(confidence, 0.85)

            # スクイーズ後の拡張を予想した利益目標
            profit_potential = bb_width * 150  # バンド幅の1.5倍の拡張予想

            if (
                confidence >= config.confidence_threshold
                and profit_potential >= config.profit_potential_threshold
            ):
                return InvestmentOpportunity(
                    opportunity_id=f"squeeze_{symbol}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=symbol,
                    opportunity_type=config.opportunity_type,
                    severity=OpportunitySeverity.MEDIUM,
                    recommended_action=TradingAction.HOLD,  # ブレイクアウト方向待ち
                    target_price=None,  # 方向が不明のため未設定
                    current_price=current_price,
                    profit_potential=profit_potential,
                    confidence_score=confidence,
                    time_horizon=config.time_horizon,
                    risk_level="medium",
                    risk_reward_ratio=2.0,  # デフォルト値
                    technical_indicators=indicators,
                    message=f"{symbol} ボラティリティスクイーズ検出",
                )

        return None

    async def _analyze_generic_opportunity(
        self,
        symbol: str,
        config: OpportunityConfig,
        indicators: Dict[str, float],
        current_price: float,
    ) -> Optional[InvestmentOpportunity]:
        """汎用機会分析"""

        # 基本的な強気/弱気判定
        rsi = indicators.get("RSI", 50)
        price_change_5d = indicators.get("Price_Change_5D", 0)

        # 模擬的な機会検出
        import random

        if random.random() < 0.1:  # 10%の確率で機会検出
            confidence = random.uniform(0.5, 0.9)
            profit_potential = random.uniform(3.0, 15.0)

            if (
                confidence >= config.confidence_threshold
                and profit_potential >= config.profit_potential_threshold
            ):
                action = (
                    TradingAction.BUY if price_change_5d > 0 else TradingAction.SELL
                )

                return InvestmentOpportunity(
                    opportunity_id=f"generic_{symbol}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=symbol,
                    opportunity_type=config.opportunity_type,
                    severity=OpportunitySeverity.LOW,
                    recommended_action=action,
                    target_price=current_price * (1 + profit_potential / 100),
                    current_price=current_price,
                    profit_potential=profit_potential,
                    confidence_score=confidence,
                    time_horizon=config.time_horizon,
                    risk_level="medium",
                    risk_reward_ratio=profit_potential / 5.0,
                    technical_indicators=indicators,
                    message=f"{symbol} 投資機会検出 ({config.opportunity_type.value})",
                )

        return None

    async def _passes_market_filter(self, opportunity: InvestmentOpportunity) -> bool:
        """市場状況フィルター"""
        if not self.market_condition:
            return True

        # 高ボラティリティ期間を避ける
        if (
            self.config.avoid_high_volatility_periods
            and self.market_condition.volatility_level == "high"
        ):
            return False

        # 市場センチメントフィルター
        if self.market_condition.market_sentiment < self.config.min_market_sentiment:
            return False

        return True

    async def _passes_quality_filter(self, opportunity: InvestmentOpportunity) -> bool:
        """品質フィルター"""
        # 最低信頼度チェック
        if opportunity.confidence_score < self.config.min_confidence_score:
            return False

        # 最低利益可能性チェック
        if opportunity.profit_potential < self.config.min_profit_potential:
            return False

        # 最低リスクリワード比チェック
        if opportunity.risk_reward_ratio < self.config.min_risk_reward_ratio:
            return False

        return True

    async def _generate_opportunity_alert(
        self, opportunity: InvestmentOpportunity
    ) -> None:
        """機会アラート生成"""
        alert_key = f"{opportunity.symbol}_{opportunity.opportunity_type.value}"

        # アラート抑制チェック
        if not await self._should_generate_opportunity_alert(alert_key):
            return

        # 機会追加
        self.opportunities.append(opportunity)

        # アラートハンドラー実行
        await self._execute_opportunity_handlers(opportunity)

        # アラート抑制記録
        self.recent_opportunities[alert_key] = datetime.now()

        logger.info(f"投資機会アラート: {opportunity.message}")

    async def _should_generate_opportunity_alert(self, alert_key: str) -> bool:
        """機会アラート生成判定"""
        if alert_key in self.recent_opportunities:
            last_alert = self.recent_opportunities[alert_key]
            elapsed_minutes = (datetime.now() - last_alert).total_seconds() / 60
            if elapsed_minutes < self.config.opportunity_cooldown_minutes:
                return False

        # 時間あたりアラート数制限
        current_hour = datetime.now().hour
        hourly_count = len(
            [
                opp
                for opp in self.opportunities
                if opp.timestamp.hour == current_hour
                and opp.timestamp.date() == datetime.now().date()
            ]
        )

        if hourly_count >= self.config.max_opportunities_per_hour:
            return False

        return True

    async def _execute_opportunity_handlers(
        self, opportunity: InvestmentOpportunity
    ) -> None:
        """機会ハンドラー実行"""
        handlers = self.alert_handlers.get(opportunity.severity, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(opportunity)
                else:
                    handler(opportunity)
            except Exception as e:
                logger.error(f"機会アラートハンドラー実行エラー: {e}")

    async def _cleanup_opportunity_history(self) -> None:
        """機会履歴クリーンアップ"""
        cutoff_time = datetime.now() - timedelta(
            days=self.config.alert_history_retention_days
        )

        # 機会履歴クリーンアップ
        self.opportunities = [
            opp for opp in self.opportunities if opp.timestamp > cutoff_time
        ]

        # 最近の機会記録クリーンアップ
        recent_cutoff = datetime.now() - timedelta(
            minutes=self.config.opportunity_cooldown_minutes * 2
        )
        self.recent_opportunities = {
            key: timestamp
            for key, timestamp in self.recent_opportunities.items()
            if timestamp > recent_cutoff
        }

    async def get_active_opportunities(
        self,
        symbol: Optional[str] = None,
        opportunity_type: Optional[OpportunityType] = None,
        severity: Optional[OpportunitySeverity] = None,
    ) -> List[InvestmentOpportunity]:
        """アクティブ機会取得"""
        active_opportunities = [opp for opp in self.opportunities if not opp.executed]

        if symbol:
            active_opportunities = [
                opp for opp in active_opportunities if opp.symbol == symbol
            ]

        if opportunity_type:
            active_opportunities = [
                opp
                for opp in active_opportunities
                if opp.opportunity_type == opportunity_type
            ]

        if severity:
            active_opportunities = [
                opp for opp in active_opportunities if opp.severity == severity
            ]

        return sorted(active_opportunities, key=lambda o: o.timestamp, reverse=True)

    async def get_opportunity_summary(self) -> Dict[str, Any]:
        """機会概要取得"""
        active_opportunities = await self.get_active_opportunities()

        # 重要度別集計
        severity_counts = {}
        for severity in OpportunitySeverity:
            severity_counts[severity.value] = len(
                [opp for opp in active_opportunities if opp.severity == severity]
            )

        # 機会タイプ別集計
        type_counts = {}
        for opp_type in OpportunityType:
            type_counts[opp_type.value] = len(
                [
                    opp
                    for opp in active_opportunities
                    if opp.opportunity_type == opp_type
                ]
            )

        # 銘柄別集計
        symbol_counts = {}
        for opp in active_opportunities:
            symbol_counts[opp.symbol] = symbol_counts.get(opp.symbol, 0) + 1

        # 統計情報
        if active_opportunities:
            avg_confidence = mean(
                [opp.confidence_score for opp in active_opportunities]
            )
            avg_profit_potential = mean(
                [opp.profit_potential for opp in active_opportunities]
            )
        else:
            avg_confidence = 0
            avg_profit_potential = 0

        return {
            "timestamp": datetime.now().isoformat(),
            "total_active_opportunities": len(active_opportunities),
            "severity_breakdown": severity_counts,
            "type_breakdown": type_counts,
            "symbol_breakdown": dict(
                sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),  # トップ10
            "statistics": {
                "average_confidence_score": avg_confidence,
                "average_profit_potential": avg_profit_potential,
                "total_opportunities_generated": len(self.opportunities),
                "executed_opportunities": len(
                    [opp for opp in self.opportunities if opp.executed]
                ),
            },
            "market_condition": {
                "market_trend": self.market_condition.market_trend
                if self.market_condition
                else None,
                "volatility_level": self.market_condition.volatility_level
                if self.market_condition
                else None,
                "market_sentiment": self.market_condition.market_sentiment
                if self.market_condition
                else None,
            },
            "monitoring_status": {
                "is_running": self._is_running,
                "active_configs": len(
                    [
                        config
                        for config in self.opportunity_configs.values()
                        if config.enabled
                    ]
                ),
                "total_configs": len(self.opportunity_configs),
            },
        }

    async def execute_opportunity(
        self, opportunity_id: str, execution_note: Optional[str] = None
    ) -> bool:
        """機会実行"""
        for opportunity in self.opportunities:
            if (
                opportunity.opportunity_id == opportunity_id
                and not opportunity.executed
            ):
                opportunity.executed = True
                opportunity.executed_at = datetime.now()
                if execution_note:
                    opportunity.details["execution_note"] = execution_note

                logger.info(f"投資機会実行: {opportunity_id}")
                return True

        return False


# 標準アラートハンドラー


async def log_opportunity_handler(opportunity: InvestmentOpportunity) -> None:
    """ログ機会ハンドラー"""
    logger.info(
        f"投資機会: {opportunity.message} (信頼度: {opportunity.confidence_score:.2f}, 利益可能性: {opportunity.profit_potential:.1f}%)"
    )


def console_opportunity_handler(opportunity: InvestmentOpportunity) -> None:
    """コンソール機会ハンドラー"""
    print(
        f"[{opportunity.timestamp.strftime('%H:%M:%S')}] {opportunity.severity.value.upper()}: {opportunity.message}"
    )
    print(
        f"  アクション: {opportunity.recommended_action.value}, 現在価格: {opportunity.current_price:.2f}"
    )
    print(
        f"  利益可能性: {opportunity.profit_potential:.1f}%, 信頼度: {opportunity.confidence_score:.2f}"
    )


async def file_opportunity_handler(opportunity: InvestmentOpportunity) -> None:
    """ファイル機会ハンドラー"""
    alert_file = Path("alerts") / "investment_opportunities.log"
    alert_file.parent.mkdir(exist_ok=True)

    opportunity_data = {
        "timestamp": opportunity.timestamp.isoformat(),
        "opportunity_id": opportunity.opportunity_id,
        "symbol": opportunity.symbol,
        "opportunity_type": opportunity.opportunity_type.value,
        "severity": opportunity.severity.value,
        "recommended_action": opportunity.recommended_action.value,
        "current_price": opportunity.current_price,
        "target_price": opportunity.target_price,
        "profit_potential": opportunity.profit_potential,
        "confidence_score": opportunity.confidence_score,
        "time_horizon": opportunity.time_horizon.value,
        "risk_level": opportunity.risk_level,
        "message": opportunity.message,
        "technical_indicators": opportunity.technical_indicators,
    }

    with open(alert_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(opportunity_data, ensure_ascii=False) + "\n")


# 使用例・テスト関数


async def setup_investment_opportunity_monitoring() -> InvestmentOpportunityAlertSystem:
    """投資機会監視セットアップ"""
    config = AlertConfig(
        min_confidence_score=0.7,
        min_profit_potential=5.0,
        min_risk_reward_ratio=2.0,
        max_opportunities_per_hour=5,
    )

    alert_system = InvestmentOpportunityAlertSystem(config)

    # アラートハンドラー登録
    alert_system.register_alert_handler(
        OpportunitySeverity.MEDIUM, console_opportunity_handler
    )
    alert_system.register_alert_handler(
        OpportunitySeverity.HIGH, log_opportunity_handler
    )
    alert_system.register_alert_handler(
        OpportunitySeverity.CRITICAL, file_opportunity_handler
    )

    return alert_system


if __name__ == "__main__":

    async def main():
        alert_system = await setup_investment_opportunity_monitoring()

        # 監視開始
        await alert_system.start_monitoring()

        # 15秒間監視実行
        await asyncio.sleep(15)

        # 機会概要取得
        summary = await alert_system.get_opportunity_summary()
        print(f"アクティブ投資機会数: {summary['total_active_opportunities']}")
        print(f"平均信頼度: {summary['statistics']['average_confidence_score']:.2f}")
        print(
            f"平均利益可能性: {summary['statistics']['average_profit_potential']:.1f}%"
        )

        # アクティブ機会表示
        active_opportunities = await alert_system.get_active_opportunities()
        for opp in active_opportunities[:5]:  # 上位5件
            print(
                f"機会: {opp.message} ({opp.symbol}, 信頼度: {opp.confidence_score:.2f})"
            )

        # 監視停止
        await alert_system.stop_monitoring()

    asyncio.run(main())
