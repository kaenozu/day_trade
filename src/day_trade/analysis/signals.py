"""
売買シグナル生成エンジン
テクニカル指標とチャートパターンを組み合わせてシグナルを生成する
"""

import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .indicators import TechnicalIndicators
from .patterns import ChartPatternRecognizer

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """シグナルタイプ"""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class SignalStrength(Enum):
    """シグナル強度"""

    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"


@dataclass
class TradingSignal:
    """売買シグナル情報"""

    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0-100
    reasons: List[str]
    conditions_met: Dict[str, bool]
    timestamp: pd.Timestamp
    price: float


class SignalRule:
    """シグナルルールの基底クラス"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    def evaluate(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> Tuple[bool, float]:
        """
        ルールを評価

        Returns:
            条件が満たされたか、信頼度スコア
        """
        raise NotImplementedError


class RSIOversoldRule(SignalRule):
    """RSI過売りルール"""

    def __init__(self, threshold: float = 30, weight: float = 1.0):
        super().__init__("RSI Oversold", weight)
        self.threshold = threshold

    def evaluate(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> Tuple[bool, float]:
        if "RSI" not in indicators.columns:
            return False, 0.0

        latest_rsi = indicators["RSI"].iloc[-1]
        if pd.isna(latest_rsi):
            return False, 0.0

        if latest_rsi < self.threshold:
            # RSIが低いほど信頼度が高い
            confidence = (self.threshold - latest_rsi) / self.threshold * 100
            return True, min(confidence * 2, 100)

        return False, 0.0


class RSIOverboughtRule(SignalRule):
    """RSI過買いルール"""

    def __init__(self, threshold: float = 70, weight: float = 1.0):
        super().__init__("RSI Overbought", weight)
        self.threshold = threshold

    def evaluate(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> Tuple[bool, float]:
        if "RSI" not in indicators.columns:
            return False, 0.0

        latest_rsi = indicators["RSI"].iloc[-1]
        if pd.isna(latest_rsi):
            return False, 0.0

        if latest_rsi > self.threshold:
            # RSIが高いほど信頼度が高い
            confidence = (latest_rsi - self.threshold) / (100 - self.threshold) * 100
            return True, min(confidence * 2, 100)

        return False, 0.0


class MACDCrossoverRule(SignalRule):
    """MACDクロスオーバールール"""

    def __init__(self, lookback: int = 2, weight: float = 1.5):
        super().__init__("MACD Crossover", weight)
        self.lookback = lookback

    def evaluate(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> Tuple[bool, float]:
        if "MACD" not in indicators.columns or "MACD_Signal" not in indicators.columns:
            return False, 0.0

        macd = indicators["MACD"].iloc[-self.lookback :]
        signal = indicators["MACD_Signal"].iloc[-self.lookback :]

        if len(macd) < self.lookback or macd.isna().any() or signal.isna().any():
            return False, 0.0

        # ゴールデンクロスをチェック
        crossed_above = (macd.iloc[-2] <= signal.iloc[-2]) and (
            macd.iloc[-1] > signal.iloc[-1]
        )

        if crossed_above:
            # クロスの角度から信頼度を計算
            angle = abs(macd.iloc[-1] - signal.iloc[-1]) / df["Close"].iloc[-1] * 100
            confidence = min(angle * 20, 100)
            return True, confidence

        return False, 0.0


class MACDDeathCrossRule(SignalRule):
    """MACDデスクロスルール"""

    def __init__(self, lookback: int = 2, weight: float = 1.5):
        super().__init__("MACD Death Cross", weight)
        self.lookback = lookback

    def evaluate(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> Tuple[bool, float]:
        if "MACD" not in indicators.columns or "MACD_Signal" not in indicators.columns:
            return False, 0.0

        macd = indicators["MACD"].iloc[-self.lookback :]
        signal = indicators["MACD_Signal"].iloc[-self.lookback :]

        if len(macd) < self.lookback or macd.isna().any() or signal.isna().any():
            return False, 0.0

        # デスクロスをチェック
        crossed_below = (macd.iloc[-2] >= signal.iloc[-2]) and (
            macd.iloc[-1] < signal.iloc[-1]
        )

        if crossed_below:
            # クロスの角度から信頼度を計算
            angle = abs(signal.iloc[-1] - macd.iloc[-1]) / df["Close"].iloc[-1] * 100
            confidence = min(angle * 20, 100)
            return True, confidence

        return False, 0.0


class BollingerBandRule(SignalRule):
    """ボリンジャーバンドルール"""

    def __init__(self, position: str = "lower", weight: float = 1.0):
        """
        Args:
            position: "lower" (買いシグナル) or "upper" (売りシグナル)
        """
        super().__init__(f"Bollinger Band {position}", weight)
        self.position = position

    def evaluate(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> Tuple[bool, float]:
        if "BB_Upper" not in indicators.columns or "BB_Lower" not in indicators.columns:
            return False, 0.0

        close_price = df["Close"].iloc[-1]

        if self.position == "lower":
            bb_lower = indicators["BB_Lower"].iloc[-1]
            if pd.isna(bb_lower):
                return False, 0.0

            if close_price <= bb_lower:
                # バンドからの乖離率を信頼度に
                deviation = (bb_lower - close_price) / close_price * 100
                confidence = min(deviation * 10, 100)
                return True, confidence

        elif self.position == "upper":
            bb_upper = indicators["BB_Upper"].iloc[-1]
            if pd.isna(bb_upper):
                return False, 0.0

            if close_price >= bb_upper:
                # バンドからの乖離率を信頼度に
                deviation = (close_price - bb_upper) / close_price * 100
                confidence = min(deviation * 10, 100)
                return True, confidence

        return False, 0.0


class PatternBreakoutRule(SignalRule):
    """パターンブレイクアウトルール"""

    def __init__(self, direction: str = "upward", weight: float = 2.0):
        """
        Args:
            direction: "upward" or "downward"
        """
        super().__init__(f"{direction.capitalize()} Breakout", weight)
        self.direction = direction

    def evaluate(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> Tuple[bool, float]:
        if "breakouts" not in patterns or len(patterns["breakouts"]) == 0:
            return False, 0.0

        breakouts = patterns["breakouts"]

        if self.direction == "upward":
            if "Upward_Breakout" in breakouts.columns:
                latest_breakout = breakouts["Upward_Breakout"].iloc[-1]
                confidence = breakouts["Upward_Confidence"].iloc[-1]

                if latest_breakout and confidence > 0:
                    return True, confidence

        elif self.direction == "downward":
            if "Downward_Breakout" in breakouts.columns:
                latest_breakout = breakouts["Downward_Breakout"].iloc[-1]
                confidence = breakouts["Downward_Confidence"].iloc[-1]

                if latest_breakout and confidence > 0:
                    return True, confidence

        return False, 0.0


class GoldenCrossRule(SignalRule):
    """ゴールデンクロスルール"""

    def __init__(self, weight: float = 2.0):
        super().__init__("Golden Cross", weight)

    def evaluate(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> Tuple[bool, float]:
        if "crosses" not in patterns or len(patterns["crosses"]) == 0:
            return False, 0.0

        crosses = patterns["crosses"]

        if "Golden_Cross" in crosses.columns:
            # 最新とその前をチェック（直近でクロスが発生したか）
            recent_crosses = crosses["Golden_Cross"].iloc[-3:]
            if recent_crosses.any():
                # 最も新しいクロスの信頼度を取得
                idx = recent_crosses[recent_crosses].index[-1]
                confidence = crosses.loc[idx, "Golden_Confidence"]
                return True, confidence

        return False, 0.0


class DeadCrossRule(SignalRule):
    """デッドクロスルール"""

    def __init__(self, weight: float = 2.0):
        super().__init__("Dead Cross", weight)

    def evaluate(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> Tuple[bool, float]:
        if "crosses" not in patterns or len(patterns["crosses"]) == 0:
            return False, 0.0

        crosses = patterns["crosses"]

        if "Dead_Cross" in crosses.columns:
            # 最新とその前をチェック（直近でクロスが発生したか）
            recent_crosses = crosses["Dead_Cross"].iloc[-3:]
            if recent_crosses.any():
                # 最も新しいクロスの信頼度を取得
                idx = recent_crosses[recent_crosses].index[-1]
                confidence = crosses.loc[idx, "Dead_Confidence"]
                return True, confidence

        return False, 0.0


class TradingSignalGenerator:
    """売買シグナル生成クラス"""

    def __init__(self):
        """デフォルトルールセットで初期化"""
        self.buy_rules: List[SignalRule] = [
            RSIOversoldRule(threshold=30, weight=1.0),
            MACDCrossoverRule(weight=1.5),
            BollingerBandRule(position="lower", weight=1.0),
            PatternBreakoutRule(direction="upward", weight=2.0),
            GoldenCrossRule(weight=2.0),
        ]

        self.sell_rules: List[SignalRule] = [
            RSIOverboughtRule(threshold=70, weight=1.0),
            MACDDeathCrossRule(weight=1.5),
            BollingerBandRule(position="upper", weight=1.0),
            PatternBreakoutRule(direction="downward", weight=2.0),
            DeadCrossRule(weight=2.0),
        ]

    def add_buy_rule(self, rule: SignalRule):
        """買いルールを追加"""
        self.buy_rules.append(rule)

    def add_sell_rule(self, rule: SignalRule):
        """売りルールを追加"""
        self.sell_rules.append(rule)

    def clear_rules(self):
        """すべてのルールをクリア"""
        self.buy_rules.clear()
        self.sell_rules.clear()

    def generate_signal(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> Optional[TradingSignal]:
        """
        売買シグナルを生成

        Args:
            df: 価格データのDataFrame
            indicators: テクニカル指標のDataFrame
            patterns: チャートパターン認識結果

        Returns:
            TradingSignal or None
        """
        try:
            # データの検証
            if df.empty or len(df) < 20:
                logger.warning("データが不足しています")
                return None

            # indicatorsとpatternsはgenerate_signals_seriesから提供される前提
            # そのため、ここではNoneチェックや再計算は行わない

            # 買いシグナルの評価
            buy_conditions = {}
            buy_confidences = []
            buy_reasons = []
            total_buy_weight = 0

            for rule in self.buy_rules:
                met, confidence = rule.evaluate(df, indicators, patterns)
                buy_conditions[rule.name] = met

                if met:
                    weighted_confidence = confidence * rule.weight
                    buy_confidences.append(weighted_confidence)
                    buy_reasons.append(f"{rule.name} (信頼度: {confidence:.0f}%)")
                    total_buy_weight += rule.weight

            # 売りシグナルの評価
            sell_conditions = {}
            sell_confidences = []
            sell_reasons = []
            total_sell_weight = 0

            for rule in self.sell_rules:
                met, confidence = rule.evaluate(df, indicators, patterns)
                sell_conditions[rule.name] = met

                if met:
                    weighted_confidence = confidence * rule.weight
                    sell_confidences.append(weighted_confidence)
                    sell_reasons.append(f"{rule.name} (信頼度: {confidence:.0f}%)")
                    total_sell_weight += rule.weight

            # シグナルの決定
            buy_score = (
                sum(buy_confidences) / total_buy_weight if total_buy_weight > 0 else 0
            )
            sell_score = (
                sum(sell_confidences) / total_sell_weight
                if total_sell_weight > 0
                else 0
            )

            # 最新の価格とタイムスタンプ
            latest_price = df["Close"].iloc[-1]
            latest_timestamp = df.index[-1]

            # シグナルタイプと強度の決定
            if buy_score > sell_score and buy_score > 0:
                # 買いシグナル
                signal_type = SignalType.BUY
                confidence = buy_score
                reasons = buy_reasons
                conditions_met = {**buy_conditions, **sell_conditions}

                # 強度の判定
                active_rules = sum(1 for v in buy_conditions.values() if v)
                if confidence >= 70 and active_rules >= 3:
                    strength = SignalStrength.STRONG
                elif confidence >= 40 and active_rules >= 2:
                    strength = SignalStrength.MEDIUM
                else:
                    strength = SignalStrength.WEAK

            elif sell_score > buy_score and sell_score > 0:
                # 売りシグナル
                signal_type = SignalType.SELL
                confidence = sell_score
                reasons = sell_reasons
                conditions_met = {**buy_conditions, **sell_conditions}

                # 強度の判定
                active_rules = sum(1 for v in sell_conditions.values() if v)
                if confidence >= 70 and active_rules >= 3:
                    strength = SignalStrength.STRONG
                elif confidence >= 40 and active_rules >= 2:
                    strength = SignalStrength.MEDIUM
                else:
                    strength = SignalStrength.WEAK

            else:
                # ホールド
                signal_type = SignalType.HOLD
                confidence = 0
                strength = SignalStrength.WEAK
                reasons = ["明確なシグナルなし"]
                conditions_met = {**buy_conditions, **sell_conditions}

            return TradingSignal(
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                reasons=reasons,
                conditions_met=conditions_met,
                timestamp=latest_timestamp,
                price=latest_price,
            )

        except Exception as e:
            logger.error(f"シグナル生成エラー: {e}")
            return None

    def generate_signals_series(
        self, df: pd.DataFrame, lookback_window: int = 50
    ) -> pd.DataFrame:
        """
        時系列でシグナルを生成

        Args:
            df: 価格データのDataFrame
            lookback_window: 各時点での計算に使用する過去データ数

        Returns:
            シグナル情報を含むDataFrame
        """
        try:
            signals = []

            # 最低限必要なデータ数
            min_required = max(lookback_window, 60)  # MACDなどの計算に必要な期間を考慮

            if len(df) < min_required:
                logger.warning(
                    f"データが不足しています。最低 {min_required} 日分のデータが必要です。"
                )
                return pd.DataFrame()

            # 全期間の指標とパターンを事前に計算
            # これにより、ループ内での再計算を避ける
            all_indicators = TechnicalIndicators.calculate_all(df)
            all_patterns = ChartPatternRecognizer.detect_all_patterns(df)

            for i in range(min_required - 1, len(df)):
                # 現在のウィンドウのデータと、対応する指標・パターンをスライス
                window_df = df.iloc[i - lookback_window + 1 : i + 1].copy()
                current_indicators = all_indicators.iloc[
                    i - lookback_window + 1 : i + 1
                ].copy()

                # patternsは辞書なので、直接スライスするのではなく、必要な情報だけ抽出して渡す
                # 現在の設計ではpatternsの各要素も時系列データを持つ可能性があるので注意
                # ここではpatternsが各時点のシグナル情報ではなく、全体に対するパターン結果を持つと仮定
                # もしpatternsの要素が時系列データなら、同様にスライスする必要がある

                # たとえば、crossesはDataFrameなので、ilocでスライス
                current_crosses = (
                    all_patterns["crosses"].iloc[i - lookback_window + 1 : i + 1].copy()
                    if "crosses" in all_patterns
                    and isinstance(all_patterns["crosses"], pd.DataFrame)
                    else pd.DataFrame()
                )

                # breakoutesもDataFrame
                current_breakouts = (
                    all_patterns["breakouts"]
                    .iloc[i - lookback_window + 1 : i + 1]
                    .copy()
                    if "breakouts" in all_patterns
                    and isinstance(all_patterns["breakouts"], pd.DataFrame)
                    else pd.DataFrame()
                )

                # levels, trends, overall_confidence は単一の結果と仮定
                current_levels = (
                    all_patterns["levels"] if "levels" in all_patterns else {}
                )
                current_trends = (
                    all_patterns["trends"] if "trends" in all_patterns else {}
                )
                current_overall_confidence = (
                    all_patterns["overall_confidence"]
                    if "overall_confidence" in all_patterns
                    else 0
                )
                current_latest_signal = (
                    all_patterns["latest_signal"]
                    if "latest_signal" in all_patterns
                    else None
                )

                current_patterns = {
                    "crosses": current_crosses,
                    "breakouts": current_breakouts,
                    "levels": current_levels,
                    "trends": current_trends,
                    "overall_confidence": current_overall_confidence,
                    "latest_signal": current_latest_signal,
                }

                # シグナルを生成
                # indicatorsとpatternsを必須引数に変更
                signal = self.generate_signal(
                    window_df, current_indicators, current_patterns
                )

                if signal:
                    signals.append(
                        {
                            "Date": signal.timestamp,
                            "Signal": signal.signal_type.value,
                            "Strength": signal.strength.value,
                            "Confidence": signal.confidence,
                            "Price": signal.price,
                            "Reasons": ", ".join(signal.reasons),
                        }
                    )

            if signals:
                signals_df = pd.DataFrame(signals)
                signals_df.set_index("Date", inplace=True)
                return signals_df
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"時系列シグナル生成エラー: {e}")
            return pd.DataFrame()

    def validate_signal(
        self,
        signal: TradingSignal,
        historical_performance: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        シグナルの有効性を検証

        Args:
            signal: 検証するシグナル
            historical_performance: 過去のパフォーマンスデータ

        Returns:
            有効性スコア（0-100）
        """
        try:
            base_score = signal.confidence

            # 強度による調整
            if signal.strength == SignalStrength.STRONG:
                base_score *= 1.2
            elif signal.strength == SignalStrength.WEAK:
                base_score *= 0.8

            # 複数条件の組み合わせによるボーナス
            active_conditions = sum(1 for v in signal.conditions_met.values() if v)
            if active_conditions >= 3:
                base_score *= 1.1

            # 過去のパフォーマンスがある場合は参考にする
            if historical_performance is not None and len(historical_performance) > 0:
                # 同じタイプのシグナルの成功率を計算
                same_type = historical_performance[
                    historical_performance["Signal"] == signal.signal_type.value
                ]

                if len(same_type) > 0:
                    # 仮の成功率（実際の取引結果が必要）
                    # ここでは簡単のため、強度が高いほど成功率が高いと仮定
                    success_rate = same_type[same_type["Strength"] == "strong"].shape[
                        0
                    ] / len(same_type)
                    base_score *= 0.7 + 0.3 * success_rate

            return min(base_score, 100)

        except Exception as e:
            logger.error(f"シグナル検証エラー: {e}")
            return 0.0


# カスタムルールの例
class VolumeSpikeBuyRule(SignalRule):
    """出来高急増買いルール"""

    def __init__(
        self,
        volume_factor: float = 2.0,
        price_change: float = 0.02,
        weight: float = 1.5,
    ):
        super().__init__("Volume Spike Buy", weight)
        self.volume_factor = volume_factor
        self.price_change = price_change

    def evaluate(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> Tuple[bool, float]:
        if len(df) < 20:
            return False, 0.0

        # 平均出来高
        avg_volume = df["Volume"].iloc[-20:-1].mean()
        latest_volume = df["Volume"].iloc[-1]

        # 価格変化
        price_change = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[
            -2
        ]

        # 出来高が平均の2倍以上かつ価格が2%以上上昇
        if (
            latest_volume > avg_volume * self.volume_factor
            and price_change > self.price_change
        ):
            volume_ratio = latest_volume / avg_volume
            confidence = min((volume_ratio - self.volume_factor) * 20 + 50, 100)
            return True, confidence

        return False, 0.0


# 使用例
if __name__ == "__main__":
    import numpy as np
    from datetime import datetime

    # サンプルデータ作成
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    np.random.seed(42)

    # トレンドのあるデータを生成
    trend = np.linspace(100, 120, 100)
    noise = np.random.randn(100) * 2
    close_prices = trend + noise

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close_prices + np.random.randn(100) * 0.5,
            "High": close_prices + np.abs(np.random.randn(100)) * 2,
            "Low": close_prices - np.abs(np.random.randn(100)) * 2,
            "Close": close_prices,
            "Volume": np.random.randint(1000000, 5000000, 100),
        }
    )
    df.set_index("Date", inplace=True)

    # シグナル生成
    generator = TradingSignalGenerator()

    # カスタムルールを追加
    generator.add_buy_rule(VolumeSpikeBuyRule())

    # 最新のシグナルを生成
    signal = generator.generate_signal(df)

    if signal:
        print(f"シグナル: {signal.signal_type.value.upper()}")
        print(f"強度: {signal.strength.value}")
        print(f"信頼度: {signal.confidence:.1f}%")
        print(f"価格: {signal.price:.2f}")
        print("\n理由:")
        for reason in signal.reasons:
            print(f"  - {reason}")

        print("\n条件の状態:")
        for condition, met in signal.conditions_met.items():
            status = "✓" if met else "✗"
            print(f"  {status} {condition}")

        # シグナルの検証
        validity = generator.validate_signal(signal)
        print(f"\n有効性スコア: {validity:.1f}%")
    else:
        print("シグナルなし")

    # 時系列シグナル生成
    print("\n=== 時系列シグナル ===")
    signals_series = generator.generate_signals_series(df, lookback_window=30)

    if not signals_series.empty:
        # 買い/売りシグナルのみ表示
        active_signals = signals_series[signals_series["Signal"] != "hold"]
        print(active_signals.tail(10))
