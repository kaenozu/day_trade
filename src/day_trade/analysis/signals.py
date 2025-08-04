"""
売買シグナル生成エンジン
テクニカル指標とチャートパターンを組み合わせてシグナルを生成する
"""

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_config import (
    get_context_logger,
    log_business_event,
)
from .indicators import TechnicalIndicators
from .patterns import ChartPatternRecognizer

logger = get_context_logger(__name__)


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
    timestamp: datetime
    price: Decimal
    symbol: Optional[str] = None  # 銘柄コード


class SignalRulesConfig:
    """シグナルルール設定管理クラス"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # プロジェクトルートからの絶対パス設定
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "signal_rules.json"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(self.config_path, encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"シグナル設定ファイルが見つかりません: {self.config_path}")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"シグナル設定ファイル読み込みエラー: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            "default_buy_rules": [],
            "default_sell_rules": [],
            "signal_generation_settings": {
                "min_data_period": 60,
                "confidence_multipliers": {
                    "rsi_oversold": 2.0,
                    "rsi_overbought": 2.0,
                    "macd_angle": 20.0,
                    "bollinger_deviation": 10.0
                },
                "strength_thresholds": {
                    "strong": {"confidence": 70, "min_active_rules": 3},
                    "medium": {"confidence": 40, "min_active_rules": 2}
                }
            }
        }

    def get_buy_rules_config(self) -> List[Dict[str, Any]]:
        """買いルール設定を取得"""
        return self.config.get("default_buy_rules", [])

    def get_sell_rules_config(self) -> List[Dict[str, Any]]:
        """売りルール設定を取得"""
        return self.config.get("default_sell_rules", [])

    def get_signal_settings(self) -> Dict[str, Any]:
        """シグナル生成設定を取得"""
        return self.config.get("signal_generation_settings", {})

    def get_confidence_multiplier(self, rule_type: str, default: float = 1.0) -> float:
        """信頼度乗数を取得"""
        multipliers = self.get_signal_settings().get("confidence_multipliers", {})
        return multipliers.get(rule_type, default)

    def get_strength_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """強度閾値を取得"""
        return self.get_signal_settings().get("strength_thresholds", {})


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

    def __init__(self, threshold: float = 30, weight: float = 1.0,
                 confidence_multiplier: float = 2.0):
        super().__init__("RSI Oversold", weight)
        self.threshold = threshold
        self.confidence_multiplier = confidence_multiplier

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
            return True, min(confidence * self.confidence_multiplier, 100)

        return False, 0.0


class RSIOverboughtRule(SignalRule):
    """RSI過買いルール"""

    def __init__(self, threshold: float = 70, weight: float = 1.0,
                 confidence_multiplier: float = 2.0):
        super().__init__("RSI Overbought", weight)
        self.threshold = threshold
        self.confidence_multiplier = confidence_multiplier

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
            return True, min(confidence * self.confidence_multiplier, 100)

        return False, 0.0


class MACDCrossoverRule(SignalRule):
    """MACDクロスオーバールール"""

    def __init__(self, lookback: int = 2, weight: float = 1.5,
                 angle_multiplier: float = 20.0):
        super().__init__("MACD Crossover", weight)
        self.lookback = lookback
        self.angle_multiplier = angle_multiplier

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
            confidence = min(angle * self.angle_multiplier, 100)
            return True, confidence

        return False, 0.0


class MACDDeathCrossRule(SignalRule):
    """MACDデスクロスルール"""

    def __init__(self, lookback: int = 2, weight: float = 1.5,
                 angle_multiplier: float = 20.0):
        super().__init__("MACD Death Cross", weight)
        self.lookback = lookback
        self.angle_multiplier = angle_multiplier

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
            confidence = min(angle * self.angle_multiplier, 100)
            return True, confidence

        return False, 0.0


class BollingerBandRule(SignalRule):
    """ボリンジャーバンドルール"""

    def __init__(self, position: str = "lower", weight: float = 1.0,
                 deviation_multiplier: float = 10.0):
        """
        Args:
            position: "lower" (買いシグナル) or "upper" (売りシグナル)
            deviation_multiplier: 乖離率に対する信頼度乗数
        """
        super().__init__(f"Bollinger Band {position}", weight)
        self.position = position
        self.deviation_multiplier = deviation_multiplier

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
                confidence = min(deviation * self.deviation_multiplier, 100)
                return True, confidence

        elif self.position == "upper":
            bb_upper = indicators["BB_Upper"].iloc[-1]
            if pd.isna(bb_upper):
                return False, 0.0

            if close_price >= bb_upper:
                # バンドからの乖離率を信頼度に
                deviation = (close_price - bb_upper) / close_price * 100
                confidence = min(deviation * self.deviation_multiplier, 100)
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

        elif self.direction == "downward" and "Downward_Breakout" in breakouts.columns:
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

    def __init__(self, config_path: Optional[str] = None):
        """設定ファイルからルールセットを初期化"""
        self.config = SignalRulesConfig(config_path)
        self.buy_rules: List[SignalRule] = []
        self.sell_rules: List[SignalRule] = []

        # 設定からルールを動的に読み込み
        self._load_rules_from_config()

    def _load_rules_from_config(self):
        """設定ファイルからルールを読み込み"""
        # 買いルールの読み込み
        for rule_config in self.config.get_buy_rules_config():
            rule = self._create_rule_from_config(rule_config)
            if rule:
                self.buy_rules.append(rule)

        # 売りルールの読み込み
        for rule_config in self.config.get_sell_rules_config():
            rule = self._create_rule_from_config(rule_config)
            if rule:
                self.sell_rules.append(rule)

        # 設定が空の場合はデフォルトルールを使用
        if not self.buy_rules:
            self._load_default_buy_rules()
        if not self.sell_rules:
            self._load_default_sell_rules()

    def _create_rule_from_config(self, rule_config: Dict[str, Any]) -> Optional[SignalRule]:
        """設定からルールオブジェクトを作成"""
        rule_type = rule_config.get("type")
        parameters = rule_config.get("parameters", {})

        rule_map = {
            "RSIOversoldRule": RSIOversoldRule,
            "RSIOverboughtRule": RSIOverboughtRule,
            "MACDCrossoverRule": MACDCrossoverRule,
            "MACDDeathCrossRule": MACDDeathCrossRule,
            "BollingerBandRule": BollingerBandRule,
            "PatternBreakoutRule": PatternBreakoutRule,
            "GoldenCrossRule": GoldenCrossRule,
            "DeadCrossRule": DeadCrossRule,
            "VolumeSpikeBuyRule": VolumeSpikeBuyRule,
        }

        if rule_type in rule_map:
            try:
                return rule_map[rule_type](**parameters)
            except Exception as e:
                logger.error(f"ルール作成エラー {rule_type}: {e}")
                return None
        else:
            logger.warning(f"未知のルールタイプ: {rule_type}")
            return None

    def _load_default_buy_rules(self):
        """デフォルト買いルールを読み込み"""
        multiplier = self.config.get_confidence_multiplier("rsi_oversold", 2.0)
        self.buy_rules = [
            RSIOversoldRule(threshold=30, weight=1.0, confidence_multiplier=multiplier),
            MACDCrossoverRule(weight=1.5, angle_multiplier=self.config.get_confidence_multiplier("macd_angle", 20.0)),
            BollingerBandRule(position="lower", weight=1.0, deviation_multiplier=self.config.get_confidence_multiplier("bollinger_deviation", 10.0)),
            PatternBreakoutRule(direction="upward", weight=2.0),
            GoldenCrossRule(weight=2.0),
        ]

    def _load_default_sell_rules(self):
        """デフォルト売りルールを読み込み"""
        multiplier = self.config.get_confidence_multiplier("rsi_overbought", 2.0)
        self.sell_rules = [
            RSIOverboughtRule(threshold=70, weight=1.0, confidence_multiplier=multiplier),
            MACDDeathCrossRule(weight=1.5, angle_multiplier=self.config.get_confidence_multiplier("macd_angle", 20.0)),
            BollingerBandRule(position="upper", weight=1.0, deviation_multiplier=self.config.get_confidence_multiplier("bollinger_deviation", 10.0)),
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
            latest_price = Decimal(str(df["Close"].iloc[-1]))
            latest_timestamp = pd.Timestamp(df.index[-1]).to_pydatetime()

            # 強度閾値を設定から取得
            thresholds = self.config.get_strength_thresholds()
            strong_threshold = thresholds.get("strong", {}).get("confidence", 70)
            strong_min_rules = thresholds.get("strong", {}).get("min_active_rules", 3)
            medium_threshold = thresholds.get("medium", {}).get("confidence", 40)
            medium_min_rules = thresholds.get("medium", {}).get("min_active_rules", 2)

            # シグナルタイプと強度の決定
            if buy_score > sell_score and buy_score > 0:
                # 買いシグナル
                signal_type = SignalType.BUY
                confidence = buy_score
                reasons = buy_reasons
                # コンディションの安全な結合（同名キー警告付き）
                conditions_met = self._merge_conditions_safely(
                    buy_conditions, sell_conditions
                )

                # 強度の判定
                active_rules = sum(1 for v in buy_conditions.values() if v)
                if confidence >= strong_threshold and active_rules >= strong_min_rules:
                    strength = SignalStrength.STRONG
                elif confidence >= medium_threshold and active_rules >= medium_min_rules:
                    strength = SignalStrength.MEDIUM
                else:
                    strength = SignalStrength.WEAK

            elif sell_score > buy_score and sell_score > 0:
                # 売りシグナル
                signal_type = SignalType.SELL
                confidence = sell_score
                reasons = sell_reasons
                # コンディションの安全な結合（同名キー警告付き）
                conditions_met = self._merge_conditions_safely(
                    buy_conditions, sell_conditions
                )

                # 強度の判定
                active_rules = sum(1 for v in sell_conditions.values() if v)
                if confidence >= strong_threshold and active_rules >= strong_min_rules:
                    strength = SignalStrength.STRONG
                elif confidence >= medium_threshold and active_rules >= medium_min_rules:
                    strength = SignalStrength.MEDIUM
                else:
                    strength = SignalStrength.WEAK

            else:
                # ホールド
                signal_type = SignalType.HOLD
                confidence = 0
                strength = SignalStrength.WEAK
                reasons = ["明確なシグナルなし"]
                # コンディションの安全な結合（同名キー警告付き）
                conditions_met = self._merge_conditions_safely(
                    buy_conditions, sell_conditions
                )

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
            logger.error(
                f"売買シグナルの生成中に予期せぬエラーが発生しました。入力データまたはルール設定を確認してください。詳細: {e}"
            )
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

            # 最低限必要なデータ数を設定から取得
            config_min_period = self.config.get_signal_settings().get("min_data_period", 60)
            min_required = max(lookback_window, config_min_period)

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
                    all_patterns.get("levels", {})
                )
                current_trends = (
                    all_patterns.get("trends", {})
                )
                current_overall_confidence = (
                    all_patterns.get("overall_confidence", 0)
                )
                current_latest_signal = (
                    all_patterns.get("latest_signal", None)
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
                            "Price": float(signal.price),  # DataFrameではDecimalよりfloatが扱いやすい
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
            logger.error(
                f"時系列売買シグナルの生成中に予期せぬエラーが発生しました。入力データまたは計算ロジックを確認してください。詳細: {e}"
            )
            return pd.DataFrame()

    def validate_signal(
        self,
        signal: TradingSignal,
        historical_performance: Optional[pd.DataFrame] = None,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        シグナルの有効性を検証

        Args:
            signal: 検証するシグナル
            historical_performance: 過去のパフォーマンスデータ
            market_context: 市場環境情報（ボラティリティ、トレンド等）

        Returns:
            有効性スコア（0-100）
        """
        try:
            base_score = signal.confidence

            # 強度による調整係数を設定から取得
            strength_multipliers = {
                SignalStrength.STRONG: 1.2,
                SignalStrength.MEDIUM: 1.0,
                SignalStrength.WEAK: 0.8
            }
            base_score *= strength_multipliers.get(signal.strength, 1.0)

            # 複数条件の組み合わせによるボーナス
            active_conditions = sum(1 for v in signal.conditions_met.values() if v)
            if active_conditions >= 3:
                base_score *= 1.15  # より高いボーナス
            elif active_conditions >= 2:
                base_score *= 1.05

            # 市場環境を考慮した調整
            if market_context:
                volatility = market_context.get("volatility", 0.0)
                trend_direction = market_context.get("trend_direction", "neutral")

                # 高ボラティリティ時は信頼度を下げる
                if volatility > 0.05:  # 5%以上の高ボラティリティ
                    base_score *= 0.9

                # トレンドとシグナルの整合性チェック
                if trend_direction == "upward" and signal.signal_type == SignalType.BUY or trend_direction == "downward" and signal.signal_type == SignalType.SELL:
                    base_score *= 1.1  # トレンドフォロー
                elif (trend_direction != "neutral" and signal.signal_type != SignalType.HOLD and
                      ((trend_direction == "upward" and signal.signal_type == SignalType.SELL) or
                       (trend_direction == "downward" and signal.signal_type == SignalType.BUY))):
                    # トレンドに逆らうシグナルは信頼度を下げる
                    base_score *= 0.85

            # 過去のパフォーマンスデータによる調整
            if historical_performance is not None and len(historical_performance) > 0:
                same_type_signals = historical_performance[
                    historical_performance["Signal"] == signal.signal_type.value
                ]

                if len(same_type_signals) > 0:
                    # 同じシグナルタイプの成功率を計算
                    if "Success" in historical_performance.columns:
                        # 実際の成功/失敗データがある場合
                        success_rate = same_type_signals["Success"].mean()
                    else:
                        # 成功データがない場合は強度で代用
                        strong_signals = same_type_signals[same_type_signals["Strength"] == "strong"]
                        success_rate = len(strong_signals) / len(same_type_signals) * 0.7 + 0.3

                    # 成功率による調整（0.6-1.3の範囲）
                    performance_multiplier = 0.6 + 0.7 * success_rate
                    base_score *= performance_multiplier

                    # 同じ強度の過去シグナルの成功率も考慮
                    same_strength = same_type_signals[
                        same_type_signals["Strength"] == signal.strength.value
                    ]
                    if len(same_strength) > 0 and "Success" in historical_performance.columns:
                        strength_success_rate = same_strength["Success"].mean()
                        # 強度別成功率による微調整
                        base_score *= (0.9 + 0.2 * strength_success_rate)

            # シグナルの新鮮度（タイムスタンプからの経過時間）による調整
            if hasattr(signal, 'timestamp') and signal.timestamp:
                from datetime import datetime, timezone
                current_time = datetime.now(timezone.utc)
                signal_time = signal.timestamp
                if signal_time.tzinfo is None:
                    signal_time = signal_time.replace(tzinfo=timezone.utc)

                time_diff = (current_time - signal_time).total_seconds() / 3600  # 時間差
                if time_diff > 24:  # 24時間以上古い
                    base_score *= 0.9
                elif time_diff > 72:  # 72時間以上古い
                    base_score *= 0.8

            return min(max(base_score, 0), 100)  # 0-100の範囲に制限

        except Exception as e:
            logger.error(
                f"シグナルの有効性検証中に予期せぬエラーが発生しました。詳細: {e}"
            )
            return 0.0

    def _merge_conditions_safely(
        self, buy_conditions: Dict[str, bool], sell_conditions: Dict[str, bool]
    ) -> Dict[str, bool]:
        """コンディションを安全に結合し、同名キーの衝突を警告"""
        merged = buy_conditions.copy()

        # 衝突チェック
        overlapping_keys = set(buy_conditions.keys()) & set(sell_conditions.keys())
        if overlapping_keys:
            logger.warning(
                f"買い・売りコンディションで同名キーが検出されました: {overlapping_keys}"
            )
            # 売り条件を優先（より安全）
            for key in overlapping_keys:
                merged[f"buy_{key}"] = buy_conditions[key]
                merged[f"sell_{key}"] = sell_conditions[key]
                del merged[key]

        # 残りの売り条件を追加
        for key, value in sell_conditions.items():
            if key not in overlapping_keys:
                merged[key] = value

        return merged


# カスタムルールの例
class VolumeSpikeBuyRule(SignalRule):
    """出来高急増買いルール"""

    def __init__(
        self,
        volume_factor: float = 2.0,
        price_change: float = 0.02,
        weight: float = 1.5,
        confidence_base: float = 50.0,
        confidence_multiplier: float = 20.0,
    ):
        super().__init__("Volume Spike Buy", weight)
        self.volume_factor = volume_factor
        self.price_change = price_change
        self.confidence_base = confidence_base
        self.confidence_multiplier = confidence_multiplier

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

        # 出来高が平均の指定倍数以上かつ価格が指定%以上上昇
        if (
            latest_volume > avg_volume * self.volume_factor
            and price_change > self.price_change
        ):
            volume_ratio = latest_volume / avg_volume
            confidence = min(
                (volume_ratio - self.volume_factor) * self.confidence_multiplier + self.confidence_base,
                100
            )
            return True, confidence

        return False, 0.0


# 使用例
if __name__ == "__main__":
    from datetime import datetime

    import numpy as np

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

    # テクニカル指標とパターンを計算
    indicators = TechnicalIndicators.calculate_all(df)
    patterns = ChartPatternRecognizer.detect_all_patterns(df)

    # 最新のシグナルを生成
    signal = generator.generate_signal(df, indicators, patterns)

    if signal:
        # シグナル情報の構造化ログ出力
        signal_data = {
            "signal_type": signal.signal_type.value.upper(),
            "strength": signal.strength.value,
            "confidence": signal.confidence,
            "price": float(signal.price),
            "timestamp": signal.timestamp.isoformat(),
            "reasons": signal.reasons,
            "conditions_met": signal.conditions_met
        }

        # 市場環境情報も含めて検証
        market_context = {
            "volatility": float(df["Close"].pct_change().std()),
            "trend_direction": "upward" if df["Close"].iloc[-1] > df["Close"].iloc[-20] else "downward"
        }
        validity = generator.validate_signal(signal, market_context=market_context)
        signal_data["validity_score"] = validity
        signal_data["market_context"] = market_context

        logger.info("シグナル生成完了", **signal_data)

        # ビジネスイベントとして記録
        log_business_event("signal_generated", **signal_data)
    else:
        logger.info("シグナル生成結果", section="signal_generation", signal_type="none", result="シグナルなし")

    # 時系列シグナル生成
    logger.info("時系列シグナル生成開始", lookback_window=30)
    signals_series = generator.generate_signals_series(df, lookback_window=30)

    if not signals_series.empty:
        # 買い/売りシグナルのみ表示
        active_signals = signals_series[signals_series["Signal"] != "hold"]

        # 構造化ログ出力
        series_summary = {
            "total_signals": len(signals_series),
            "active_signals": len(active_signals),
            "buy_signals": len(active_signals[active_signals["Signal"] == "buy"]),
            "sell_signals": len(active_signals[active_signals["Signal"] == "sell"]),
            "latest_signals": active_signals.tail(5).to_dict("records") if len(active_signals) > 0 else []
        }

        logger.info("時系列シグナル生成完了", **series_summary)
        log_business_event("time_series_signals_generated", **series_summary)

        # デモ用表示ログ
        logger.info("時系列シグナル表示",
                   signal_count=len(active_signals),
                   latest_signals=active_signals.tail(10).to_dict("records") if len(active_signals) > 0 else [])
    else:
        logger.info("時系列シグナル生成結果", result="no_active_signals")


# SignalRulesConfigクラスに安全なマージメソッドを追加
class TradingSignalGeneratorExtended(TradingSignalGenerator):
    """拡張版シグナル生成器"""

    def _merge_conditions_safely(
        self, buy_conditions: Dict[str, bool], sell_conditions: Dict[str, bool]
    ) -> Dict[str, bool]:
        """コンディションを安全に結合し、同名キーの衝突を警告"""
        merged = buy_conditions.copy()

        # 衝突チェック
        overlapping_keys = set(buy_conditions.keys()) & set(sell_conditions.keys())
        if overlapping_keys:
            logger.warning(
                f"買い・売りコンディションで同名キーが検出されました: {overlapping_keys}"
            )
            # 売り条件を優先（より安全）
            for key in overlapping_keys:
                merged[f"buy_{key}"] = buy_conditions[key]
                merged[f"sell_{key}"] = sell_conditions[key]
                del merged[key]

        # 残りの売り条件を追加
        for key, value in sell_conditions.items():
            if key not in overlapping_keys:
                merged[key] = value

        return merged
