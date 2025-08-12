#!/usr/bin/env python3
"""
取引戦略実行エンジン

Phase 4: 高度な取引戦略とシグナル管理
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class SignalType(Enum):
    """シグナル種別"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class StrategyType(Enum):
    """戦略種別"""

    ML_BASED = "ml_based"  # ML助言ベース
    MOMENTUM = "momentum"  # モメンタム戦略
    MEAN_REVERSION = "mean_reversion"  # 平均回帰戦略
    PORTFOLIO_OPTIMIZED = "portfolio_optimized"  # ポートフォリオ最適化
    HYBRID = "hybrid"  # ハイブリッド戦略


@dataclass
class TradingSignal:
    """取引シグナル"""

    symbol: str
    signal_type: SignalType
    confidence: float
    price: float
    quantity: int
    strategy: StrategyType
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StrategyParameters:
    """戦略パラメータ"""

    strategy_type: StrategyType
    risk_tolerance: float = 0.5  # 0.0(保守) - 1.0(積極)
    max_position_size: float = 0.1  # 1銘柄最大10%
    stop_loss_pct: float = 0.05  # 5%損切り
    take_profit_pct: float = 0.15  # 15%利確
    rebalance_frequency: int = 5  # 5日毎リバランス
    min_confidence_threshold: float = 0.7  # 最小信頼度70%

    # ML戦略固有
    ml_weight: float = 0.6
    technical_weight: float = 0.4

    # モメンタム戦略固有
    momentum_period: int = 20
    momentum_threshold: float = 0.02

    # 平均回帰戦略固有
    mean_reversion_period: int = 50
    deviation_threshold: float = 2.0


class BaseStrategy(ABC):
    """戦略基底クラス"""

    def __init__(self, parameters: StrategyParameters):
        self.parameters = parameters
        self.strategy_type = parameters.strategy_type
        logger.info(f"{self.strategy_type.value}戦略初期化")

    @abstractmethod
    def generate_signals(
        self, symbol: str, data: pd.DataFrame, ml_recommendation: Dict = None
    ) -> List[TradingSignal]:
        """シグナル生成"""
        pass

    @abstractmethod
    def calculate_position_size(
        self, signal: TradingSignal, current_capital: float, current_price: float
    ) -> int:
        """ポジションサイズ計算"""
        pass


class MLBasedStrategy(BaseStrategy):
    """ML助言ベース戦略"""

    def __init__(self, parameters: StrategyParameters):
        super().__init__(parameters)
        self.last_signals = {}

    def generate_signals(
        self, symbol: str, data: pd.DataFrame, ml_recommendation: Dict = None
    ) -> List[TradingSignal]:
        """ML助言に基づくシグナル生成"""
        try:
            signals = []

            if not ml_recommendation:
                return signals

            advice = ml_recommendation.get("advice", "HOLD")
            confidence = ml_recommendation.get("confidence", 50) / 100.0
            current_price = data["Close"].iloc[-1]

            # 信頼度フィルタリング
            if confidence < self.parameters.min_confidence_threshold:
                logger.debug(
                    f"信頼度不足: {symbol} {confidence:.1%} < {self.parameters.min_confidence_threshold:.1%}"
                )
                return signals

            # シグナル生成
            if advice == "BUY" and confidence >= 0.7:
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    price=current_price,
                    quantity=0,  # calculate_position_sizeで計算
                    strategy=StrategyType.ML_BASED,
                    timestamp=datetime.now(),
                    metadata={
                        "ml_advice": advice,
                        "ml_confidence": confidence * 100,
                        "risk_level": ml_recommendation.get("risk_level", "MEDIUM"),
                        "reason": ml_recommendation.get("reason", "ML推奨"),
                    },
                )
                signals.append(signal)

            elif advice == "SELL" and confidence >= 0.6:
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    price=current_price,
                    quantity=0,
                    strategy=StrategyType.ML_BASED,
                    timestamp=datetime.now(),
                    metadata={
                        "ml_advice": advice,
                        "ml_confidence": confidence * 100,
                        "risk_level": ml_recommendation.get("risk_level", "HIGH"),
                        "reason": ml_recommendation.get("reason", "ML売却推奨"),
                    },
                )
                signals.append(signal)

            # テクニカル分析補強
            technical_signals = self._generate_technical_signals(symbol, data, current_price)
            signals.extend(technical_signals)

            return signals

        except Exception as e:
            logger.error(f"MLベース戦略シグナル生成エラー {symbol}: {e}")
            return []

    def _generate_technical_signals(
        self, symbol: str, data: pd.DataFrame, current_price: float
    ) -> List[TradingSignal]:
        """テクニカル分析シグナル"""
        try:
            signals = []

            # RSI（単純版）
            price_changes = data["Close"].diff()
            gains = price_changes.where(price_changes > 0, 0).rolling(14).mean()
            losses = (-price_changes.where(price_changes < 0, 0)).rolling(14).mean()
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            # RSIシグナル
            if current_rsi < 30:  # 過売り
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=0.6,
                    price=current_price,
                    quantity=0,
                    strategy=StrategyType.ML_BASED,
                    timestamp=datetime.now(),
                    metadata={
                        "technical_indicator": "RSI",
                        "rsi_value": current_rsi,
                        "signal_reason": "RSI過売りシグナル",
                    },
                )
                signals.append(signal)

            elif current_rsi > 70:  # 過買い
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=0.6,
                    price=current_price,
                    quantity=0,
                    strategy=StrategyType.ML_BASED,
                    timestamp=datetime.now(),
                    metadata={
                        "technical_indicator": "RSI",
                        "rsi_value": current_rsi,
                        "signal_reason": "RSI過買いシグナル",
                    },
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.debug(f"テクニカル分析エラー {symbol}: {e}")
            return []

    def calculate_position_size(
        self, signal: TradingSignal, current_capital: float, current_price: float
    ) -> int:
        """ポジションサイズ計算（ML信頼度ベース）"""
        try:
            # 基本投資額（信頼度調整）
            confidence_multiplier = signal.confidence
            risk_adjusted_size = self.parameters.max_position_size * confidence_multiplier

            # 投資金額
            investment_amount = current_capital * risk_adjusted_size

            # 株数計算
            quantity = int(investment_amount // current_price)

            # 最小単位チェック（100株単位）
            quantity = (quantity // 100) * 100

            return max(0, quantity)

        except Exception as e:
            logger.error(f"ポジションサイズ計算エラー: {e}")
            return 0


class MomentumStrategy(BaseStrategy):
    """モメンタム戦略"""

    def generate_signals(
        self, symbol: str, data: pd.DataFrame, ml_recommendation: Dict = None
    ) -> List[TradingSignal]:
        """モメンタムベースシグナル生成"""
        try:
            signals = []
            current_price = data["Close"].iloc[-1]

            # モメンタム計算
            period = self.parameters.momentum_period
            if len(data) < period + 1:
                return signals

            momentum = (current_price - data["Close"].iloc[-(period + 1)]) / data["Close"].iloc[
                -(period + 1)
            ]

            # 出来高確認
            avg_volume = data["Volume"].rolling(20).mean().iloc[-1]
            current_volume = data["Volume"].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            # モメンタムシグナル
            if momentum > self.parameters.momentum_threshold and volume_ratio > 1.2:
                confidence = min(0.9, abs(momentum) * 10)  # モメンタム強度に応じた信頼度

                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    price=current_price,
                    quantity=0,
                    strategy=StrategyType.MOMENTUM,
                    timestamp=datetime.now(),
                    metadata={
                        "momentum": momentum,
                        "volume_ratio": volume_ratio,
                        "period": period,
                    },
                )
                signals.append(signal)

            elif momentum < -self.parameters.momentum_threshold:
                confidence = min(0.9, abs(momentum) * 10)

                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    price=current_price,
                    quantity=0,
                    strategy=StrategyType.MOMENTUM,
                    timestamp=datetime.now(),
                    metadata={
                        "momentum": momentum,
                        "volume_ratio": volume_ratio,
                        "period": period,
                    },
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"モメンタム戦略エラー {symbol}: {e}")
            return []

    def calculate_position_size(
        self, signal: TradingSignal, current_capital: float, current_price: float
    ) -> int:
        """モメンタム強度ベースサイジング"""
        momentum = abs(signal.metadata.get("momentum", 0))
        momentum_multiplier = min(2.0, 1 + momentum * 5)  # モメンタム強度で調整

        base_size = self.parameters.max_position_size * signal.confidence
        adjusted_size = base_size * momentum_multiplier

        investment_amount = current_capital * min(
            adjusted_size, self.parameters.max_position_size * 2
        )
        quantity = int(investment_amount // current_price)

        return max(0, (quantity // 100) * 100)


class HybridStrategy(BaseStrategy):
    """ハイブリッド戦略（ML + テクニカル + ポートフォリオ最適化）"""

    def __init__(self, parameters: StrategyParameters):
        super().__init__(parameters)
        self.ml_strategy = MLBasedStrategy(parameters)
        self.momentum_strategy = MomentumStrategy(parameters)

    def generate_signals(
        self, symbol: str, data: pd.DataFrame, ml_recommendation: Dict = None
    ) -> List[TradingSignal]:
        """複合シグナル生成"""
        try:
            all_signals = []

            # 各戦略からシグナル取得
            ml_signals = self.ml_strategy.generate_signals(symbol, data, ml_recommendation)
            momentum_signals = self.momentum_strategy.generate_signals(symbol, data)

            # シグナル統合
            combined_signal = self._combine_signals(
                symbol, data, ml_signals + momentum_signals, ml_recommendation
            )

            if combined_signal:
                all_signals.append(combined_signal)

            return all_signals

        except Exception as e:
            logger.error(f"ハイブリッド戦略エラー {symbol}: {e}")
            return []

    def _combine_signals(
        self,
        symbol: str,
        data: pd.DataFrame,
        signals: List[TradingSignal],
        ml_recommendation: Dict,
    ) -> Optional[TradingSignal]:
        """シグナル統合ロジック"""
        if not signals:
            return None

        try:
            current_price = data["Close"].iloc[-1]

            # シグナル集計
            buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
            sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]

            # 統合判定
            if len(buy_signals) > len(sell_signals):
                # 買いシグナル統合
                combined_confidence = np.mean([s.confidence for s in buy_signals])

                # ML重み調整
                ml_buy_signals = [s for s in buy_signals if s.strategy == StrategyType.ML_BASED]
                if ml_buy_signals:
                    ml_confidence = np.mean([s.confidence for s in ml_buy_signals])
                    combined_confidence = (
                        combined_confidence * (1 - self.parameters.ml_weight)
                        + ml_confidence * self.parameters.ml_weight
                    )

                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=combined_confidence,
                    price=current_price,
                    quantity=0,
                    strategy=StrategyType.HYBRID,
                    timestamp=datetime.now(),
                    metadata={
                        "component_signals": len(buy_signals),
                        "ml_signals": len(ml_buy_signals),
                        "momentum_signals": len(
                            [s for s in buy_signals if s.strategy == StrategyType.MOMENTUM]
                        ),
                        "ml_recommendation": ml_recommendation,
                    },
                )

            elif len(sell_signals) > len(buy_signals):
                # 売りシグナル統合
                combined_confidence = np.mean([s.confidence for s in sell_signals])

                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=combined_confidence,
                    price=current_price,
                    quantity=0,
                    strategy=StrategyType.HYBRID,
                    timestamp=datetime.now(),
                    metadata={
                        "component_signals": len(sell_signals),
                        "ml_recommendation": ml_recommendation,
                    },
                )

            return None

        except Exception as e:
            logger.error(f"シグナル統合エラー {symbol}: {e}")
            return None

    def calculate_position_size(
        self, signal: TradingSignal, current_capital: float, current_price: float
    ) -> int:
        """ハイブリッドポジションサイズ計算"""
        try:
            # 基本サイズ
            base_size = self.parameters.max_position_size * signal.confidence

            # ML信頼度調整
            ml_rec = signal.metadata.get("ml_recommendation", {})
            ml_confidence = ml_rec.get("confidence", 50) / 100.0 if ml_rec else 0.5

            # 複合シグナル強度
            component_signals = signal.metadata.get("component_signals", 1)
            signal_strength = min(2.0, 1 + (component_signals - 1) * 0.3)

            # 最終サイズ計算
            final_size = base_size * ml_confidence * signal_strength
            final_size = min(final_size, self.parameters.max_position_size * 1.5)

            investment_amount = current_capital * final_size
            quantity = int(investment_amount // current_price)

            return max(0, (quantity // 100) * 100)

        except Exception as e:
            logger.error(f"ハイブリッドサイジングエラー: {e}")
            return 0


class StrategyExecutor:
    """
    戦略実行管理クラス

    複数戦略の統合実行とシグナル管理
    """

    def __init__(self, strategy_parameters: StrategyParameters):
        """初期化"""
        self.parameters = strategy_parameters
        self.strategy = self._create_strategy(strategy_parameters.strategy_type)
        self.active_signals: List[TradingSignal] = []
        self.signal_history: List[TradingSignal] = []

        logger.info(f"戦略実行エンジン初期化: {strategy_parameters.strategy_type.value}")

    def _create_strategy(self, strategy_type: StrategyType) -> BaseStrategy:
        """戦略インスタンス生成"""
        if strategy_type == StrategyType.ML_BASED:
            return MLBasedStrategy(self.parameters)
        elif strategy_type == StrategyType.MOMENTUM:
            return MomentumStrategy(self.parameters)
        elif strategy_type == StrategyType.HYBRID:
            return HybridStrategy(self.parameters)
        else:
            logger.warning(f"未サポート戦略: {strategy_type}, MLベース戦略を使用")
            return MLBasedStrategy(self.parameters)

    def execute_strategy(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        ml_recommendations: Dict[str, Dict],
        current_capital: float,
    ) -> List[TradingSignal]:
        """
        戦略実行

        Args:
            symbols_data: 銘柄データ辞書
            ml_recommendations: ML推奨辞書
            current_capital: 現在資金

        Returns:
            生成されたシグナルリスト
        """
        try:
            new_signals = []

            for symbol, data in symbols_data.items():
                if data.empty:
                    continue

                # ML推奨取得
                ml_rec = ml_recommendations.get(symbol, {})

                # シグナル生成
                signals = self.strategy.generate_signals(symbol, data, ml_rec)

                for signal in signals:
                    # ポジションサイズ計算
                    signal.quantity = self.strategy.calculate_position_size(
                        signal, current_capital, signal.price
                    )

                    if signal.quantity > 0:
                        new_signals.append(signal)

            # シグナル管理
            self.active_signals = new_signals
            self.signal_history.extend(new_signals)

            # 古い履歴削除（メモリ管理）
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-800:]

            logger.info(f"戦略実行完了: {len(new_signals)}シグナル生成")
            return new_signals

        except Exception as e:
            logger.error(f"戦略実行エラー: {e}")
            return []

    def get_signal_summary(self) -> Dict:
        """シグナル統計"""
        if not self.active_signals:
            return {"total": 0, "by_type": {}, "avg_confidence": 0.0}

        by_type = {}
        for signal in self.active_signals:
            signal_type = signal.signal_type.value
            by_type[signal_type] = by_type.get(signal_type, 0) + 1

        avg_confidence = np.mean([s.confidence for s in self.active_signals])

        return {
            "total": len(self.active_signals),
            "by_type": by_type,
            "avg_confidence": avg_confidence,
            "strategy": self.parameters.strategy_type.value,
        }

    def clear_signals(self):
        """アクティブシグナルクリア"""
        self.active_signals.clear()
        logger.debug("アクティブシグナルクリア")


if __name__ == "__main__":
    # テスト実行
    params = StrategyParameters(
        strategy_type=StrategyType.HYBRID,
        risk_tolerance=0.7,
        max_position_size=0.08,
        min_confidence_threshold=0.65,
    )

    executor = StrategyExecutor(params)

    # サンプルデータ
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    sample_data = pd.DataFrame(
        {
            "Close": 1000 + np.cumsum(np.random.normal(0, 10, 60)),
            "Volume": np.random.randint(100000, 1000000, 60),
        },
        index=dates,
    )

    sample_ml_rec = {
        "advice": "BUY",
        "confidence": 85,
        "risk_level": "MEDIUM",
        "reason": "上昇トレンド検出",
    }

    # テスト実行
    signals = executor.execute_strategy({"TEST": sample_data}, {"TEST": sample_ml_rec}, 1000000)

    print("=== 戦略テスト結果 ===")
    print(f"生成シグナル数: {len(signals)}")

    if signals:
        signal = signals[0]
        print(f"シグナル: {signal.signal_type.value}")
        print(f"信頼度: {signal.confidence:.1%}")
        print(f"推奨株数: {signal.quantity}")
        print(f"戦略: {signal.strategy.value}")

    summary = executor.get_signal_summary()
    print(f"シグナル統計: {summary}")
