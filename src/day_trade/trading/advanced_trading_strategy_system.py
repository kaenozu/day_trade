#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Trading Strategy System - 高度取引戦略システム

最適化予測システムと統合した高度取引戦略
Issue #800-2-2実装：戦略最適化
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

class StrategyType(Enum):
    """戦略タイプ"""
    ML_ENHANCED_MOMENTUM = "ml_enhanced_momentum"
    ADAPTIVE_VOLATILITY = "adaptive_volatility"
    MULTI_SIGNAL_FUSION = "multi_signal_fusion"
    DYNAMIC_STOP_LOSS = "dynamic_stop_loss"
    VOLUME_PRICE_ACTION = "volume_price_action"

class SignalStrength(Enum):
    """シグナル強度"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NO_SIGNAL = "no_signal"

@dataclass
class TradingSignal:
    """取引シグナル"""
    symbol: str
    signal_type: str  # "BUY", "SELL", "HOLD"
    strength: SignalStrength
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reasons: List[str]
    timestamp: datetime

@dataclass
class StrategyPerformance:
    """戦略性能"""
    strategy_type: StrategyType
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    volatility: float
    total_trades: int

@dataclass
class BacktestResult:
    """バックテスト結果"""
    symbol: str
    strategy_type: StrategyType
    performance: StrategyPerformance
    equity_curve: pd.Series
    trades: List[Dict]
    signals: List[TradingSignal]

class AdvancedTradingStrategySystem:
    """高度取引戦略システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データベース設定
        self.db_path = Path("ml_models_data/advanced_trading_strategies.db")
        self.db_path.parent.mkdir(exist_ok=True)

        # 戦略設定
        self.strategies = {
            StrategyType.ML_ENHANCED_MOMENTUM: {
                "confidence_threshold": 0.65,
                "momentum_periods": [5, 10, 20],
                "volatility_adjustment": True
            },
            StrategyType.ADAPTIVE_VOLATILITY: {
                "volatility_lookback": 20,
                "position_size_factor": 0.02,
                "max_position_size": 0.1
            },
            StrategyType.MULTI_SIGNAL_FUSION: {
                "min_signals": 3,
                "signal_weights": {
                    "ml_prediction": 0.4,
                    "momentum": 0.25,
                    "volatility": 0.2,
                    "volume": 0.15
                }
            },
            StrategyType.DYNAMIC_STOP_LOSS: {
                "initial_stop_pct": 0.02,
                "trailing_stop_pct": 0.015,
                "volatility_multiplier": 1.5
            },
            StrategyType.VOLUME_PRICE_ACTION: {
                "volume_threshold": 1.5,
                "price_momentum_threshold": 0.005,
                "confirmation_periods": 3
            }
        }

        self.logger.info("Advanced trading strategy system initialized")

    async def generate_trading_signals(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """取引シグナル生成"""

        self.logger.info(f"取引シグナル生成開始: {symbol}")

        signals = []

        try:
            # ML予測取得
            ml_predictions = await self._get_ml_predictions(symbol, data)

            # 各戦略でシグナル生成
            for strategy_type in StrategyType:
                strategy_signals = await self._generate_strategy_signals(
                    strategy_type, symbol, data, ml_predictions
                )
                signals.extend(strategy_signals)

            # シグナル統合・最適化
            optimized_signals = self._optimize_signals(signals)

            self.logger.info(f"取引シグナル生成完了: {len(optimized_signals)}シグナル")
            return optimized_signals

        except Exception as e:
            self.logger.error(f"シグナル生成エラー: {e}")
            return []

    async def _get_ml_predictions(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """ML予測取得"""

        try:
            from optimized_prediction_system import optimized_prediction_system

            # 最新の予測結果取得
            prediction_result = await optimized_prediction_system.predict_with_optimized_models(symbol)

            return {
                "prediction": prediction_result.prediction,
                "confidence": prediction_result.confidence,
                "model_consensus": prediction_result.model_consensus,
                "timestamp": prediction_result.timestamp
            }

        except Exception as e:
            self.logger.warning(f"ML予測取得失敗 {symbol}: {e}")
            return {
                "prediction": 1,
                "confidence": 0.5,
                "model_consensus": {},
                "timestamp": datetime.now()
            }

    async def _generate_strategy_signals(self, strategy_type: StrategyType, symbol: str,
                                       data: pd.DataFrame, ml_predictions: Dict) -> List[TradingSignal]:
        """戦略別シグナル生成"""

        if strategy_type == StrategyType.ML_ENHANCED_MOMENTUM:
            return await self._ml_enhanced_momentum_strategy(symbol, data, ml_predictions)
        elif strategy_type == StrategyType.ADAPTIVE_VOLATILITY:
            return await self._adaptive_volatility_strategy(symbol, data, ml_predictions)
        elif strategy_type == StrategyType.MULTI_SIGNAL_FUSION:
            return await self._multi_signal_fusion_strategy(symbol, data, ml_predictions)
        elif strategy_type == StrategyType.DYNAMIC_STOP_LOSS:
            return await self._dynamic_stop_loss_strategy(symbol, data, ml_predictions)
        elif strategy_type == StrategyType.VOLUME_PRICE_ACTION:
            return await self._volume_price_action_strategy(symbol, data, ml_predictions)
        else:
            return []

    async def _ml_enhanced_momentum_strategy(self, symbol: str, data: pd.DataFrame,
                                           ml_predictions: Dict) -> List[TradingSignal]:
        """ML強化モメンタム戦略"""

        signals = []
        config = self.strategies[StrategyType.ML_ENHANCED_MOMENTUM]

        # 価格データ
        close = data['Close']
        current_price = close.iloc[-1]

        # ML予測チェック
        ml_confidence = ml_predictions["confidence"]
        ml_prediction = ml_predictions["prediction"]

        if ml_confidence < config["confidence_threshold"]:
            return signals  # 信頼度が低い場合はスキップ

        # モメンタム計算
        momentum_scores = []
        for period in config["momentum_periods"]:
            if len(close) > period:
                momentum = (close.iloc[-1] - close.iloc[-period-1]) / close.iloc[-period-1]
                momentum_scores.append(momentum)

        avg_momentum = np.mean(momentum_scores) if momentum_scores else 0

        # ボラティリティ調整
        if config["volatility_adjustment"]:
            volatility = close.pct_change().rolling(20).std().iloc[-1]
            momentum_threshold = 0.005 + volatility * 2
        else:
            momentum_threshold = 0.005

        # シグナル生成
        reasons = []
        signal_type = "HOLD"
        strength = SignalStrength.NO_SIGNAL

        if ml_prediction == 1 and avg_momentum > momentum_threshold:
            signal_type = "BUY"
            strength = SignalStrength.STRONG if ml_confidence > 0.75 else SignalStrength.MODERATE
            reasons.extend([
                f"ML予測: 上昇 (信頼度: {ml_confidence:.3f})",
                f"モメンタム: {avg_momentum:.3f} > {momentum_threshold:.3f}"
            ])
        elif ml_prediction == 0 and avg_momentum < -momentum_threshold:
            signal_type = "SELL"
            strength = SignalStrength.STRONG if ml_confidence > 0.75 else SignalStrength.MODERATE
            reasons.extend([
                f"ML予測: 下降 (信頼度: {ml_confidence:.3f})",
                f"モメンタム: {avg_momentum:.3f} < {-momentum_threshold:.3f}"
            ])

        if signal_type != "HOLD":
            # ストップロスとテイクプロフィット計算
            atr = self._calculate_atr(data, 14)
            current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02

            if signal_type == "BUY":
                stop_loss = current_price - (current_atr * 2)
                take_profit = current_price + (current_atr * 3)
            else:  # SELL
                stop_loss = current_price + (current_atr * 2)
                take_profit = current_price - (current_atr * 3)

            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=ml_confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasons=reasons,
                timestamp=datetime.now()
            )
            signals.append(signal)

        return signals

    async def _adaptive_volatility_strategy(self, symbol: str, data: pd.DataFrame,
                                          ml_predictions: Dict) -> List[TradingSignal]:
        """適応ボラティリティ戦略"""

        signals = []
        config = self.strategies[StrategyType.ADAPTIVE_VOLATILITY]

        close = data['Close']
        current_price = close.iloc[-1]

        # ボラティリティ計算
        returns = close.pct_change()
        volatility = returns.rolling(config["volatility_lookback"]).std().iloc[-1]

        # 平均ボラティリティとの比較
        avg_volatility = returns.rolling(60).std().iloc[-1] if len(returns) > 60 else volatility
        volatility_ratio = volatility / avg_volatility if avg_volatility > 0 else 1

        # ML予測とボラティリティに基づくシグナル
        ml_prediction = ml_predictions["prediction"]
        ml_confidence = ml_predictions["confidence"]

        reasons = []
        signal_type = "HOLD"
        strength = SignalStrength.NO_SIGNAL

        # 低ボラティリティ環境での強いシグナル
        if volatility_ratio < 0.8 and ml_confidence > 0.6:
            if ml_prediction == 1:
                signal_type = "BUY"
                strength = SignalStrength.STRONG
                reasons.append(f"低ボラティリティ環境({volatility_ratio:.2f}) + ML上昇予測")
            else:
                signal_type = "SELL"
                strength = SignalStrength.MODERATE
                reasons.append(f"低ボラティリティ環境({volatility_ratio:.2f}) + ML下降予測")

        # 高ボラティリティ環境での慎重なシグナル
        elif volatility_ratio > 1.5 and ml_confidence > 0.8:
            if ml_prediction == 1:
                signal_type = "BUY"
                strength = SignalStrength.MODERATE
                reasons.append(f"高ボラティリティ環境({volatility_ratio:.2f}) + 高信頼度ML予測")
            # 高ボラティリティでのSELLは避ける

        if signal_type != "HOLD":
            # ボラティリティ調整されたストップロス
            volatility_stop = current_price * volatility * 3

            if signal_type == "BUY":
                stop_loss = current_price - volatility_stop
                take_profit = current_price + (volatility_stop * 2)
            else:
                stop_loss = current_price + volatility_stop
                take_profit = current_price - (volatility_stop * 2)

            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=ml_confidence * (1 / volatility_ratio),  # ボラティリティ調整
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasons=reasons,
                timestamp=datetime.now()
            )
            signals.append(signal)

        return signals

    async def _multi_signal_fusion_strategy(self, symbol: str, data: pd.DataFrame,
                                          ml_predictions: Dict) -> List[TradingSignal]:
        """マルチシグナル統合戦略"""

        signals = []
        config = self.strategies[StrategyType.MULTI_SIGNAL_FUSION]

        close = data['Close']
        volume = data['Volume']
        current_price = close.iloc[-1]

        # 各シグナルの評価
        signal_scores = {}

        # 1. ML予測シグナル
        ml_prediction = ml_predictions["prediction"]
        ml_confidence = ml_predictions["confidence"]
        signal_scores["ml_prediction"] = (ml_prediction * 2 - 1) * ml_confidence  # -1 to 1 scale

        # 2. モメンタムシグナル
        momentum_5 = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] if len(close) > 5 else 0
        momentum_10 = (close.iloc[-1] - close.iloc[-11]) / close.iloc[-11] if len(close) > 10 else 0
        avg_momentum = (momentum_5 + momentum_10) / 2
        signal_scores["momentum"] = np.tanh(avg_momentum * 100)  # -1 to 1 scale

        # 3. ボラティリティシグナル
        volatility = close.pct_change().rolling(20).std().iloc[-1]
        avg_volatility = close.pct_change().rolling(60).std().iloc[-1] if len(close) > 60 else volatility
        volatility_ratio = volatility / avg_volatility if avg_volatility > 0 else 1
        # 低ボラティリティを好む
        signal_scores["volatility"] = max(-1, min(1, 2 - volatility_ratio))

        # 4. ボリュームシグナル
        vol_sma = volume.rolling(20).mean().iloc[-1] if len(volume) > 20 else volume.iloc[-1]
        volume_ratio = volume.iloc[-1] / vol_sma if vol_sma > 0 else 1
        # 高ボリュームを好む（ただし過度は避ける）
        signal_scores["volume"] = np.tanh((volume_ratio - 1) * 2)

        # 重み付き統合スコア
        weighted_score = sum(
            signal_scores[signal] * config["signal_weights"][signal]
            for signal in signal_scores
        )

        # シグナル強度決定
        abs_score = abs(weighted_score)
        if abs_score > 0.7:
            strength = SignalStrength.VERY_STRONG
        elif abs_score > 0.5:
            strength = SignalStrength.STRONG
        elif abs_score > 0.3:
            strength = SignalStrength.MODERATE
        elif abs_score > 0.1:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NO_SIGNAL

        # シグナル生成
        if abs_score > 0.3:  # 最小閾値
            signal_type = "BUY" if weighted_score > 0 else "SELL"

            reasons = [
                f"統合スコア: {weighted_score:.3f}",
                f"ML: {signal_scores['ml_prediction']:.3f}",
                f"モメンタム: {signal_scores['momentum']:.3f}",
                f"ボラティリティ: {signal_scores['volatility']:.3f}",
                f"ボリューム: {signal_scores['volume']:.3f}"
            ]

            # リスク調整されたストップロス
            risk_factor = min(1.5, 1 + volatility)
            base_stop = current_price * 0.02 * risk_factor

            if signal_type == "BUY":
                stop_loss = current_price - base_stop
                take_profit = current_price + (base_stop * 2.5)
            else:
                stop_loss = current_price + base_stop
                take_profit = current_price - (base_stop * 2.5)

            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=abs_score,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasons=reasons,
                timestamp=datetime.now()
            )
            signals.append(signal)

        return signals

    async def _dynamic_stop_loss_strategy(self, symbol: str, data: pd.DataFrame,
                                        ml_predictions: Dict) -> List[TradingSignal]:
        """動的ストップロス戦略"""

        signals = []
        config = self.strategies[StrategyType.DYNAMIC_STOP_LOSS]

        close = data['Close']
        current_price = close.iloc[-1]

        ml_prediction = ml_predictions["prediction"]
        ml_confidence = ml_predictions["confidence"]

        if ml_confidence < 0.6:
            return signals  # 低信頼度ではスキップ

        # ATRベースの動的ストップロス
        atr = self._calculate_atr(data, 14)
        current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02

        # ボラティリティ調整
        volatility = close.pct_change().rolling(20).std().iloc[-1]
        volatility_multiplier = config["volatility_multiplier"] * (1 + volatility * 10)

        # 初期ストップロス
        initial_stop_distance = max(
            current_price * config["initial_stop_pct"],
            current_atr * volatility_multiplier
        )

        # トレーリングストップ
        trailing_stop_distance = max(
            current_price * config["trailing_stop_pct"],
            current_atr * volatility_multiplier * 0.8
        )

        signal_type = "BUY" if ml_prediction == 1 else "SELL"
        strength = SignalStrength.STRONG if ml_confidence > 0.75 else SignalStrength.MODERATE

        if signal_type == "BUY":
            stop_loss = current_price - initial_stop_distance
            take_profit = current_price + (initial_stop_distance * 3)
        else:
            stop_loss = current_price + initial_stop_distance
            take_profit = current_price - (initial_stop_distance * 3)

        reasons = [
            f"ML予測: {signal_type} (信頼度: {ml_confidence:.3f})",
            f"動的ストップロス: {initial_stop_distance/current_price*100:.2f}%",
            f"トレーリング: {trailing_stop_distance/current_price*100:.2f}%"
        ]

        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=ml_confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasons=reasons,
            timestamp=datetime.now()
        )
        signals.append(signal)

        return signals

    async def _volume_price_action_strategy(self, symbol: str, data: pd.DataFrame,
                                          ml_predictions: Dict) -> List[TradingSignal]:
        """ボリューム・価格アクション戦略"""

        signals = []
        config = self.strategies[StrategyType.VOLUME_PRICE_ACTION]

        close = data['Close']
        volume = data['Volume']
        current_price = close.iloc[-1]

        # ボリューム分析
        vol_sma = volume.rolling(20).mean()
        volume_ratio = volume.iloc[-1] / vol_sma.iloc[-1] if len(vol_sma) > 0 and vol_sma.iloc[-1] > 0 else 1

        # 価格モメンタム
        price_momentum = close.pct_change().iloc[-1]

        # ML予測
        ml_prediction = ml_predictions["prediction"]
        ml_confidence = ml_predictions["confidence"]

        # シグナル条件
        volume_confirmation = volume_ratio > config["volume_threshold"]
        price_confirmation = abs(price_momentum) > config["price_momentum_threshold"]
        ml_confirmation = ml_confidence > 0.6

        if volume_confirmation and price_confirmation and ml_confirmation:
            if ml_prediction == 1 and price_momentum > 0:
                signal_type = "BUY"
                strength = SignalStrength.STRONG
            elif ml_prediction == 0 and price_momentum < 0:
                signal_type = "SELL"
                strength = SignalStrength.STRONG
            else:
                return signals  # 矛盾するシグナル

            reasons = [
                f"高ボリューム確認: {volume_ratio:.2f}x",
                f"価格モメンタム: {price_momentum:.3f}",
                f"ML予測合致: {signal_type} (信頼度: {ml_confidence:.3f})"
            ]

            # ボリューム加重ストップロス
            base_stop = current_price * 0.015
            volume_adjustment = min(1.5, volume_ratio / 2)
            adjusted_stop = base_stop * volume_adjustment

            if signal_type == "BUY":
                stop_loss = current_price - adjusted_stop
                take_profit = current_price + (adjusted_stop * 2.5)
            else:
                stop_loss = current_price + adjusted_stop
                take_profit = current_price - (adjusted_stop * 2.5)

            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=ml_confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasons=reasons,
                timestamp=datetime.now()
            )
            signals.append(signal)

        return signals

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR（平均トゥルーレンジ）計算"""

        high = data['High']
        low = data['Low']
        close = data['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()

        return atr

    def _optimize_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """シグナル最適化"""

        if not signals:
            return signals

        # 同一銘柄の複数シグナルを統合
        symbol_signals = {}
        for signal in signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)

        optimized_signals = []

        for symbol, symbol_signal_list in symbol_signals.items():
            if len(symbol_signal_list) == 1:
                optimized_signals.extend(symbol_signal_list)
            else:
                # 複数シグナルの統合
                optimized_signal = self._merge_signals(symbol_signal_list)
                if optimized_signal:
                    optimized_signals.append(optimized_signal)

        return optimized_signals

    def _merge_signals(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """複数シグナルの統合"""

        if not signals:
            return None

        # BUY/SELL/HOLDの投票
        buy_votes = sum(1 for s in signals if s.signal_type == "BUY")
        sell_votes = sum(1 for s in signals if s.signal_type == "SELL")

        # 強度重み付き投票
        buy_strength = sum(self._strength_to_weight(s.strength) for s in signals if s.signal_type == "BUY")
        sell_strength = sum(self._strength_to_weight(s.strength) for s in signals if s.signal_type == "SELL")

        # 最終判定
        if buy_strength > sell_strength and buy_votes > sell_votes:
            final_signal_type = "BUY"
        elif sell_strength > buy_strength and sell_votes > buy_votes:
            final_signal_type = "SELL"
        else:
            return None  # 矛盾するシグナル

        # 統合された属性
        avg_confidence = np.mean([s.confidence for s in signals])
        avg_entry_price = np.mean([s.entry_price for s in signals])

        # 最も保守的なストップロス
        if final_signal_type == "BUY":
            best_stop_loss = max([s.stop_loss for s in signals if s.signal_type == "BUY"])
            best_take_profit = min([s.take_profit for s in signals if s.signal_type == "BUY"])
        else:
            best_stop_loss = min([s.stop_loss for s in signals if s.signal_type == "SELL"])
            best_take_profit = max([s.take_profit for s in signals if s.signal_type == "SELL"])

        # 統合強度
        total_strength = max(buy_strength, sell_strength)
        if total_strength > 3:
            final_strength = SignalStrength.VERY_STRONG
        elif total_strength > 2:
            final_strength = SignalStrength.STRONG
        elif total_strength > 1:
            final_strength = SignalStrength.MODERATE
        else:
            final_strength = SignalStrength.WEAK

        # 理由統合
        all_reasons = []
        for signal in signals:
            all_reasons.extend(signal.reasons)

        return TradingSignal(
            symbol=signals[0].symbol,
            signal_type=final_signal_type,
            strength=final_strength,
            confidence=avg_confidence,
            entry_price=avg_entry_price,
            stop_loss=best_stop_loss,
            take_profit=best_take_profit,
            reasons=list(set(all_reasons)),  # 重複除去
            timestamp=datetime.now()
        )

    def _strength_to_weight(self, strength: SignalStrength) -> float:
        """シグナル強度を重みに変換"""

        weights = {
            SignalStrength.VERY_STRONG: 2.0,
            SignalStrength.STRONG: 1.5,
            SignalStrength.MODERATE: 1.0,
            SignalStrength.WEAK: 0.5,
            SignalStrength.NO_SIGNAL: 0.0
        }
        return weights.get(strength, 0.0)

    async def run_strategy_backtest(self, symbol: str, strategy_type: StrategyType,
                                  period: str = "6mo") -> BacktestResult:
        """戦略バックテスト実行"""

        self.logger.info(f"戦略バックテスト開始: {symbol} - {strategy_type.value}")

        try:
            # データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, period)

            if data is None or len(data) < 50:
                raise ValueError("バックテストに十分なデータがありません")

            # シミュレーション実行
            equity_curve, trades, all_signals = await self._run_backtest_simulation(
                symbol, strategy_type, data
            )

            # 性能計算
            performance = self._calculate_strategy_performance(equity_curve, trades)

            result = BacktestResult(
                symbol=symbol,
                strategy_type=strategy_type,
                performance=performance,
                equity_curve=equity_curve,
                trades=trades,
                signals=all_signals
            )

            self.logger.info(f"戦略バックテスト完了: {symbol} - リターン{performance.total_return:.2f}%")
            return result

        except Exception as e:
            self.logger.error(f"戦略バックテストエラー: {e}")
            raise

    async def _run_backtest_simulation(self, symbol: str, strategy_type: StrategyType,
                                     data: pd.DataFrame) -> Tuple[pd.Series, List[Dict], List[TradingSignal]]:
        """バックテストシミュレーション実行"""

        # 初期設定
        initial_capital = 100000  # 10万円
        current_capital = initial_capital
        position = 0  # 0: ノーポジション, 1: ロング, -1: ショート
        entry_price = 0
        stop_loss = 0
        take_profit = 0

        equity_curve = []
        trades = []
        all_signals = []

        # 日次シミュレーション
        for i in range(50, len(data)):  # 50日後から開始（指標計算のため）
            current_data = data.iloc[:i+1]
            current_price = current_data['Close'].iloc[-1]

            # シグナル生成（現在の戦略のみ）
            ml_predictions = await self._get_ml_predictions(symbol, current_data)
            signals = await self._generate_strategy_signals(strategy_type, symbol, current_data, ml_predictions)

            # ポジション管理
            if position == 0 and signals:  # エントリー
                signal = signals[0]
                if signal.signal_type in ["BUY", "SELL"]:
                    position = 1 if signal.signal_type == "BUY" else -1
                    entry_price = signal.entry_price
                    stop_loss = signal.stop_loss
                    take_profit = signal.take_profit

                    all_signals.append(signal)

            elif position != 0:  # エグジットチェック
                exit_reason = None

                # ストップロス/テイクプロフィットチェック
                if position == 1:  # ロングポジション
                    if current_price <= stop_loss:
                        exit_reason = "stop_loss"
                    elif current_price >= take_profit:
                        exit_reason = "take_profit"
                else:  # ショートポジション
                    if current_price >= stop_loss:
                        exit_reason = "stop_loss"
                    elif current_price <= take_profit:
                        exit_reason = "take_profit"

                # 反対シグナルチェック
                if not exit_reason and signals:
                    signal = signals[0]
                    if (position == 1 and signal.signal_type == "SELL") or \
                       (position == -1 and signal.signal_type == "BUY"):
                        exit_reason = "signal_reversal"

                # エグジット実行
                if exit_reason:
                    if position == 1:
                        trade_return = (current_price - entry_price) / entry_price
                    else:
                        trade_return = (entry_price - current_price) / entry_price

                    current_capital *= (1 + trade_return)

                    trades.append({
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "position": position,
                        "return": trade_return,
                        "exit_reason": exit_reason,
                        "date": current_data.index[-1]
                    })

                    position = 0

            # 現在の資産価値計算
            if position == 0:
                current_value = current_capital
            else:
                if position == 1:
                    unrealized_return = (current_price - entry_price) / entry_price
                else:
                    unrealized_return = (entry_price - current_price) / entry_price
                current_value = current_capital * (1 + unrealized_return)

            equity_curve.append(current_value)

        # 最後のポジション強制クローズ
        if position != 0:
            current_price = data['Close'].iloc[-1]
            if position == 1:
                trade_return = (current_price - entry_price) / entry_price
            else:
                trade_return = (entry_price - current_price) / entry_price

            trades.append({
                "entry_price": entry_price,
                "exit_price": current_price,
                "position": position,
                "return": trade_return,
                "exit_reason": "forced_close",
                "date": data.index[-1]
            })

        equity_series = pd.Series(equity_curve, index=data.index[50:50+len(equity_curve)])

        return equity_series, trades, all_signals

    def _calculate_strategy_performance(self, equity_curve: pd.Series, trades: List[Dict]) -> StrategyPerformance:
        """戦略性能計算"""

        if len(equity_curve) == 0:
            return StrategyPerformance(
                strategy_type=StrategyType.ML_ENHANCED_MOMENTUM,
                total_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                profit_factor=0,
                avg_trade_return=0,
                volatility=0,
                total_trades=0
            )

        # 総リターン
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

        # 日次リターン
        daily_returns = equity_curve.pct_change().dropna()

        # シャープレシオ
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # 最大ドローダウン
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100

        # トレード分析
        if trades:
            winning_trades = [t for t in trades if t["return"] > 0]
            losing_trades = [t for t in trades if t["return"] < 0]

            win_rate = len(winning_trades) / len(trades) * 100

            total_profit = sum(t["return"] for t in winning_trades)
            total_loss = abs(sum(t["return"] for t in losing_trades))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            avg_trade_return = np.mean([t["return"] for t in trades]) * 100
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_return = 0

        # ボラティリティ
        volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else 0

        return StrategyPerformance(
            strategy_type=StrategyType.ML_ENHANCED_MOMENTUM,  # 仮値
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            volatility=volatility,
            total_trades=len(trades)
        )

# グローバルインスタンス
advanced_trading_system = AdvancedTradingStrategySystem()

# テスト実行
async def test_advanced_trading_strategies():
    """高度取引戦略テスト"""

    print("=== 高度取引戦略システムテスト ===")

    test_symbols = ["7203", "8306"]
    test_strategies = [StrategyType.ML_ENHANCED_MOMENTUM, StrategyType.MULTI_SIGNAL_FUSION]

    for symbol in test_symbols:
        print(f"\n🔍 {symbol} 戦略テスト開始...")

        # データ取得
        from real_data_provider_v2 import real_data_provider
        data = await real_data_provider.get_stock_data(symbol, "6mo")

        if data is not None:
            # シグナル生成テスト
            signals = await advanced_trading_system.generate_trading_signals(symbol, data)
            print(f"📊 {symbol} 生成シグナル: {len(signals)}個")

            for signal in signals[:3]:  # 上位3個表示
                print(f"  {signal.signal_type} - {signal.strength.value} (信頼度: {signal.confidence:.3f})")

            # バックテストテスト
            for strategy in test_strategies:
                try:
                    result = await advanced_trading_system.run_strategy_backtest(symbol, strategy, "6mo")
                    print(f"📈 {strategy.value}: リターン{result.performance.total_return:.2f}% "
                          f"シャープ{result.performance.sharpe_ratio:.3f} "
                          f"勝率{result.performance.win_rate:.1f}%")
                except Exception as e:
                    print(f"❌ {strategy.value}: エラー - {e}")
        else:
            print(f"❌ {symbol}: データ取得失敗")

    print(f"\n✅ 高度取引戦略システムテスト完了")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_advanced_trading_strategies())