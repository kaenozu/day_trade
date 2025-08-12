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
            with open(self.config_path, encoding="utf-8") as f:
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
                    "bollinger_deviation": 10.0,
                },
                "strength_thresholds": {
                    "strong": {"confidence": 70, "min_active_rules": 3},
                    "medium": {"confidence": 40, "min_active_rules": 2},
                },
            },
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

    def get_min_data_for_generation(self) -> int:
        """シグナル生成用最小データ数を取得"""
        return self.get_signal_settings().get("min_data_for_generation", 20)

    def get_volume_calculation_period(self) -> int:
        """出来高計算期間を取得"""
        return self.get_signal_settings().get("volume_calculation_period", 20)

    def get_trend_lookback_period(self) -> int:
        """トレンド判定期間を取得"""
        return self.get_signal_settings().get("trend_lookback_period", 20)

    def get_signal_freshness(self) -> Dict[str, int]:
        """シグナル新鮮度設定を取得"""
        return self.get_signal_settings().get(
            "signal_freshness", {"warning_hours": 24, "stale_hours": 72}
        )

    def get_high_volatility_threshold(self) -> float:
        """高ボラティリティ判定閾値を取得"""
        return (
            self.get_signal_settings()
            .get("validation_adjustments", {})
            .get("high_volatility_threshold", 0.05)
        )

    def get_volume_spike_settings(self) -> Dict[str, float]:
        """出来高急増設定を取得"""
        return self.config.get(
            "volume_spike_settings",
            {
                "volume_factor": 2.0,
                "price_change_threshold": 0.02,
                "confidence_base": 50.0,
                "confidence_multiplier": 20.0,
            },
        )

    def get_rsi_thresholds(self) -> Dict[str, float]:
        """RSI閾値を取得"""
        return self.config.get(
            "rsi_default_thresholds", {"oversold": 30, "overbought": 70}
        )

    def get_macd_settings(self) -> Dict[str, float]:
        """MACD設定を取得"""
        return self.config.get(
            "macd_default_settings", {"lookback_period": 2, "default_weight": 1.5}
        )

    def get_pattern_breakout_settings(self) -> Dict[str, float]:
        """パターンブレイクアウト設定を取得"""
        return self.config.get("pattern_breakout_settings", {"default_weight": 2.0})

    def get_cross_settings(self) -> Dict[str, float]:
        """ゴールデン/デッドクロス設定を取得"""
        return self.config.get("golden_dead_cross_settings", {"default_weight": 2.0})


class SignalRule:
    """シグナルルールの基底クラス"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional["SignalRulesConfig"] = None,
    ) -> Tuple[bool, float]:
        """
        ルールを評価

        Returns:
            条件が満たされたか、信頼度スコア
        """
        raise NotImplementedError


class RSIOversoldRule(SignalRule):
    """RSI過売りルール"""

    def __init__(
        self,
        threshold: float = 30,
        weight: float = 1.0,
        confidence_multiplier: float = 2.0,
    ):
        super().__init__("RSI Oversold", weight)
        self.threshold = threshold
        self.confidence_multiplier = confidence_multiplier

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional["SignalRulesConfig"] = None,
    ) -> Tuple[bool, float]:
        if config is None:
            config = SignalRulesConfig()

        threshold = config.get_rsi_thresholds().get("oversold", self.threshold)
        confidence_multiplier = config.get_confidence_multiplier(
            "rsi_oversold", self.confidence_multiplier
        )

        if "RSI" not in indicators.columns or indicators["RSI"].empty:
            return False, 0.0

        latest_rsi = indicators["RSI"].iloc[-1]
        if pd.isna(latest_rsi):
            return False, 0.0

        if latest_rsi < threshold:
            # RSIが低いほど信頼度が高い
            confidence = (threshold - latest_rsi) / threshold * 100
            return True, min(confidence * confidence_multiplier, 100)

        return False, 0.0


class RSIOverboughtRule(SignalRule):
    """RSI過買いルール"""

    def __init__(
        self,
        threshold: float = 70,
        weight: float = 1.0,
        confidence_multiplier: float = 2.0,
    ):
        super().__init__("RSI Overbought", weight)
        self.threshold = threshold
        self.confidence_multiplier = confidence_multiplier

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional["SignalRulesConfig"] = None,
    ) -> Tuple[bool, float]:
        if config is None:
            config = SignalRulesConfig()

        threshold = config.get_rsi_thresholds().get("overbought", self.threshold)
        confidence_multiplier = config.get_confidence_multiplier(
            "rsi_overbought", self.confidence_multiplier
        )

        if "RSI" not in indicators.columns or indicators["RSI"].empty:
            return False, 0.0

        latest_rsi = indicators["RSI"].iloc[-1]
        if pd.isna(latest_rsi):
            return False, 0.0

        if latest_rsi > threshold:
            # RSIが高いほど信頼度が高い
            confidence = (latest_rsi - threshold) / (100 - threshold) * 100
            return True, min(confidence * confidence_multiplier, 100)

        return False, 0.0


class MACDCrossoverRule(SignalRule):
    """MACDクロスオーバールール"""

    def __init__(
        self, lookback: int = 2, weight: float = 1.5, angle_multiplier: float = 20.0
    ):
        super().__init__("MACD Crossover", weight)
        self.lookback = lookback
        self.angle_multiplier = angle_multiplier

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional["SignalRulesConfig"] = None,
    ) -> Tuple[bool, float]:
        if config is None:
            config = SignalRulesConfig()

        lookback = config.get_macd_settings().get("lookback_period", self.lookback)
        angle_multiplier = config.get_confidence_multiplier(
            "macd_angle", self.angle_multiplier
        )

        if (
            "MACD" not in indicators.columns
            or "MACD_Signal" not in indicators.columns
            or indicators["MACD"].empty
            or indicators["MACD_Signal"].empty
        ):
            return False, 0.0

        macd = indicators["MACD"].iloc[-lookback:]
        signal = indicators["MACD_Signal"].iloc[-lookback:]

        if len(macd) < lookback or macd.isna().any() or signal.isna().any():
            return False, 0.0

        # ゴールデンクロスをチェック
        crossed_above = (macd.iloc[-2] <= signal.iloc[-2]) and (
            macd.iloc[-1] > signal.iloc[-1]
        )

        if crossed_above:
            # クロスの角度から信頼度を計算
            angle = abs(macd.iloc[-1] - signal.iloc[-1]) / df["Close"].iloc[-1] * 100
            confidence = min(angle * angle_multiplier, 100)
            return True, confidence

        return False, 0.0


class MACDDeathCrossRule(SignalRule):
    """MACDデスクロスルール"""

    def __init__(
        self, lookback: int = 2, weight: float = 1.5, angle_multiplier: float = 20.0
    ):
        super().__init__("MACD Death Cross", weight)
        self.lookback = lookback
        self.angle_multiplier = angle_multiplier

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional["SignalRulesConfig"] = None,
    ) -> Tuple[bool, float]:
        if config is None:
            config = SignalRulesConfig()

        lookback = config.get_macd_settings().get("lookback_period", self.lookback)
        angle_multiplier = config.get_confidence_multiplier(
            "macd_angle", self.angle_multiplier
        )

        if (
            "MACD" not in indicators.columns
            or "MACD_Signal" not in indicators.columns
            or indicators["MACD"].empty
            or indicators["MACD_Signal"].empty
        ):
            return False, 0.0

        macd = indicators["MACD"].iloc[-lookback:]
        signal = indicators["MACD_Signal"].iloc[-lookback:]

        if len(macd) < lookback or macd.isna().any() or signal.isna().any():
            return False, 0.0

        # デスクロスをチェック
        crossed_below = (macd.iloc[-2] >= signal.iloc[-2]) and (
            macd.iloc[-1] < signal.iloc[-1]
        )

        if crossed_below:
            # クロスの角度から信頼度を計算
            angle = abs(signal.iloc[-1] - macd.iloc[-1]) / df["Close"].iloc[-1] * 100
            confidence = min(angle * angle_multiplier, 100)
            return True, confidence

        return False, 0.0


class BollingerBandRule(SignalRule):
    """ボリンジャーバンドルール"""

    def __init__(
        self,
        position: str = "lower",
        weight: float = 1.0,
        deviation_multiplier: float = 10.0,
    ):
        """
        Args:
            position: "lower" (買いシグナル) or "upper" (売りシグナル)
            deviation_multiplier: 乖離率に対する信頼度乗数
        """
        super().__init__(f"Bollinger Band {position}", weight)
        self.position = position
        self.deviation_multiplier = deviation_multiplier

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional["SignalRulesConfig"] = None,
    ) -> Tuple[bool, float]:
        if config is None:
            config = SignalRulesConfig()

        deviation_multiplier = config.get_confidence_multiplier(
            "bollinger_deviation", self.deviation_multiplier
        )

        if (
            "BB_Upper" not in indicators.columns
            or "BB_Lower" not in indicators.columns
            or indicators["BB_Upper"].empty
            or indicators["BB_Lower"].empty
        ):
            return False, 0.0

        close_price = df["Close"].iloc[-1]

        if self.position == "lower":
            bb_lower = indicators["BB_Lower"].iloc[-1]
            if pd.isna(bb_lower):
                return False, 0.0

            if close_price <= bb_lower:
                # バンドからの乖離率を信頼度に
                deviation = (bb_lower - close_price) / close_price * 100
                confidence = min(deviation * deviation_multiplier, 100)
                return True, confidence

        elif self.position == "upper":
            bb_upper = indicators["BB_Upper"].iloc[-1]
            if pd.isna(bb_upper):
                return False, 0.0

            if close_price >= bb_upper:
                # バンドからの乖離率を信頼度に
                deviation = (close_price - bb_upper) / close_price * 100
                confidence = min(deviation * deviation_multiplier, 100)
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
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional["SignalRulesConfig"] = None,
    ) -> Tuple[bool, float]:
        breakouts = patterns.get("breakouts", pd.DataFrame())

        if not isinstance(breakouts, pd.DataFrame) or breakouts.empty:
            return False, 0.0

        if self.direction == "upward":
            if (
                "Upward_Breakout" in breakouts.columns
                and "Upward_Confidence" in breakouts.columns
            ):
                latest_breakout = breakouts["Upward_Breakout"].iloc[-1]
                confidence = breakouts["Upward_Confidence"].iloc[-1]

                if latest_breakout and confidence > 0:
                    return True, confidence

        elif (
            self.direction == "downward"
            and "Downward_Breakout" in breakouts.columns
            and "Downward_Confidence" in breakouts.columns
        ):
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
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional["SignalRulesConfig"] = None,
    ) -> Tuple[bool, float]:
        crosses = patterns.get("crosses", pd.DataFrame())

        if not isinstance(crosses, pd.DataFrame) or crosses.empty:
            logger.debug(
                f"GoldenCrossRule: 'crosses' is not a DataFrame or is empty. type: {type(crosses)}"
            )
            return False, 0.0

        if "Golden_Cross" in crosses.columns and "Golden_Confidence" in crosses.columns:
            # Look for the most recent golden cross (within last periods from config)
            if config is None:
                config = SignalRulesConfig()
            recent_signal_lookback = config.get_signal_settings().get(
                "recent_signal_lookback", 5
            )
            lookback_window = min(recent_signal_lookback, len(crosses))
            recent_crosses = crosses["Golden_Cross"].iloc[-lookback_window:]
            recent_confidences = crosses["Golden_Confidence"].iloc[-lookback_window:]

            for i in range(len(recent_crosses) - 1, -1, -1):
                if recent_crosses.iloc[i] and recent_confidences.iloc[i] > 0:
                    return True, recent_confidences.iloc[i]
        return False, 0.0


class DeadCrossRule(SignalRule):
    """デッドクロスルール"""

    def __init__(self, weight: float = 2.0):
        super().__init__("Dead Cross", weight)

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional["SignalRulesConfig"] = None,
    ) -> Tuple[bool, float]:
        crosses = patterns.get("crosses", pd.DataFrame())

        if not isinstance(crosses, pd.DataFrame) or crosses.empty:
            logger.debug(
                f"DeadCrossRule: 'crosses' is not a DataFrame or is empty. type: {type(crosses)}"
            )
            return False, 0.0

        if "Dead_Cross" in crosses.columns and "Dead_Confidence" in crosses.columns:
            # Look for the most recent dead cross (within last periods from config)
            if config is None:
                config = SignalRulesConfig()
            recent_signal_lookback = config.get_signal_settings().get(
                "recent_signal_lookback", 5
            )
            lookback_window = min(recent_signal_lookback, len(crosses))
            recent_crosses = crosses["Dead_Cross"].iloc[-lookback_window:]
            recent_confidences = crosses["Dead_Confidence"].iloc[-lookback_window:]

            for i in range(len(recent_crosses) - 1, -1, -1):
                if recent_crosses.iloc[i] and recent_confidences.iloc[i] > 0:
                    return True, recent_confidences.iloc[i]
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

    def _create_rule_from_config(
        self, rule_config: Dict[str, Any]
    ) -> Optional[SignalRule]:
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
        # 設定ファイルからデフォルト値を取得
        rsi_thresholds = self.config.get_rsi_thresholds()
        macd_settings = self.config.get_macd_settings()
        pattern_settings = self.config.get_pattern_breakout_settings()
        cross_settings = self.config.get_cross_settings()

        multiplier = self.config.get_confidence_multiplier("rsi_oversold", 2.0)
        self.buy_rules = [
            RSIOversoldRule(
                threshold=rsi_thresholds["oversold"],
                weight=1.0,
                confidence_multiplier=multiplier,
            ),
            MACDCrossoverRule(
                lookback=int(macd_settings["lookback_period"]),
                weight=macd_settings["default_weight"],
                angle_multiplier=self.config.get_confidence_multiplier(
                    "macd_angle", 20.0
                ),
            ),
            BollingerBandRule(
                position="lower",
                weight=1.0,
                deviation_multiplier=self.config.get_confidence_multiplier(
                    "bollinger_deviation", 10.0
                ),
            ),
            PatternBreakoutRule(
                direction="upward", weight=pattern_settings["default_weight"]
            ),
            GoldenCrossRule(weight=cross_settings["default_weight"]),
        ]

    def _load_default_sell_rules(self):
        """デフォルト売りルールを読み込み"""
        # 設定ファイルからデフォルト値を取得
        rsi_thresholds = self.config.get_rsi_thresholds()
        macd_settings = self.config.get_macd_settings()
        pattern_settings = self.config.get_pattern_breakout_settings()
        cross_settings = self.config.get_cross_settings()

        multiplier = self.config.get_confidence_multiplier("rsi_overbought", 2.0)
        self.sell_rules = [
            RSIOverboughtRule(
                threshold=rsi_thresholds["overbought"],
                weight=1.0,
                confidence_multiplier=multiplier,
            ),
            MACDDeathCrossRule(
                lookback=int(macd_settings["lookback_period"]),
                weight=macd_settings["default_weight"],
                angle_multiplier=self.config.get_confidence_multiplier(
                    "macd_angle", 20.0
                ),
            ),
            BollingerBandRule(
                position="upper",
                weight=1.0,
                deviation_multiplier=self.config.get_confidence_multiplier(
                    "bollinger_deviation", 10.0
                ),
            ),
            PatternBreakoutRule(
                direction="downward", weight=pattern_settings["default_weight"]
            ),
            DeadCrossRule(weight=cross_settings["default_weight"]),
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

    def get_high_volatility_threshold(self) -> float:
        """高ボラティリティ判定闾値を取得"""
        return self.config.get_high_volatility_threshold()

    def get_signal_settings(self) -> Dict[str, Any]:
        """シグナル生成設定を取得"""
        return self.config.get_signal_settings()

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
            min_data = self.config.get_min_data_for_generation()
            if df.empty or len(df) < min_data:
                logger.warning(f"データが不足しています（最低{min_data}日分必要）")
                return None

            # patternsがNoneの場合に空の辞書を代入
            if patterns is None:
                patterns = {}

            # indicatorsとpatternsはgenerate_signals_seriesから提供される前提
            # そのため、ここではNoneチェックや再計算は行わない

            # 買いシグナルの評価
            buy_conditions = {}
            buy_confidences = []
            buy_reasons = []
            total_buy_weight = 0

            for rule in self.buy_rules:
                met, confidence = rule.evaluate(df, indicators, patterns, self.config)
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
                met, confidence = rule.evaluate(df, indicators, patterns, self.config)
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
                conditions_met = self._merge_conditions_safely(
                    buy_conditions, sell_conditions
                )

                # 強度の判定
                active_rules = sum(1 for v in buy_conditions.values() if v)
                if confidence >= strong_threshold and active_rules >= strong_min_rules:
                    strength = SignalStrength.STRONG
                elif (
                    confidence >= medium_threshold and active_rules >= medium_min_rules
                ):
                    strength = SignalStrength.MEDIUM
                else:
                    strength = SignalStrength.WEAK

            elif sell_score > buy_score and sell_score > 0:
                # 売りシグナル
                signal_type = SignalType.SELL
                confidence = sell_score
                reasons = sell_reasons
                conditions_met = self._merge_conditions_safely(
                    buy_conditions, sell_conditions
                )

                # 強度の判定
                active_rules = sum(1 for v in sell_conditions.values() if v)
                if confidence >= strong_threshold and active_rules >= strong_min_rules:
                    strength = SignalStrength.STRONG
                elif (
                    confidence >= medium_threshold and active_rules >= medium_min_rules
                ):
                    strength = SignalStrength.MEDIUM
                else:
                    strength = SignalStrength.WEAK

            else:
                # ホールド
                signal_type = SignalType.HOLD
                confidence = 0
                strength = SignalStrength.WEAK
                reasons = ["明確なシグナルなし"]
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
            config_min_period = self.config.get_signal_settings().get(
                "min_data_period", 60
            )
            min_required = max(lookback_window, config_min_period)

            if len(df) < min_required:
                logger.warning(
                    f"データが不足しています。最低 {min_required} 日分のデータが必要です。"
                )
                return pd.DataFrame()

            # 全期間の指標とパターンを事前に計算
            # これにより、ループ内での再計算を避ける
            all_indicators = TechnicalIndicators.calculate_all(df)
            pattern_recognizer = ChartPatternRecognizer()
            all_patterns = pattern_recognizer.detect_all_patterns(df)

            for i in range(min_required - 1, len(df)):
                # 現在のウィンドウのデータと、対応する指標をスライス
                window_df = df.iloc[i - lookback_window + 1 : i + 1].copy()
                current_indicators = all_indicators.iloc[
                    i - lookback_window + 1 : i + 1
                ].copy()

                # パターンデータの簡素化されたスライス処理
                # 改善されたpatterns.pyの構造により、シンプルな処理が可能
                current_patterns = self._slice_patterns(
                    all_patterns, i, lookback_window
                )

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
                            "Price": float(
                                signal.price
                            ),  # DataFrameではDecimalよりfloatが扱いやすい
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
            market_context: 市場環境情報(ボラティリティ,トレンド等)

        Returns:
            有効性スコア(0-100)
        """
        try:
            base_score = signal.confidence

            # 強度による調整係数を設定から取得
            strength_config = self.config.get_signal_settings().get(
                "strength_multipliers", {}
            )
            strength_multipliers = {
                SignalStrength.STRONG: strength_config.get("strong", 1.2),
                SignalStrength.MEDIUM: strength_config.get("medium", 1.0),
                SignalStrength.WEAK: strength_config.get("weak", 0.8),
            }
            base_score *= strength_multipliers.get(signal.strength, 1.0)

            # 複数条件の組み合わせによるボーナス
            validation_config = self.config.get_signal_settings().get(
                "validation_adjustments", {}
            )
            multi_condition_bonus = validation_config.get("multi_condition_bonus", 1.15)

            active_conditions = sum(1 for v in signal.conditions_met.values() if v)
            if active_conditions >= 3:
                base_score *= multi_condition_bonus
            elif active_conditions >= 2:
                base_score *= (multi_condition_bonus + 1.0) / 2  # より控えめなボーナス

            # 市場環境を考慮した調整
            if market_context:
                volatility = market_context.get("volatility", 0.0)
                trend_direction = market_context.get("trend_direction", "neutral")

                # 高ボラティリティ時は信頼度を下げる
                high_vol_threshold = self.config.get_high_volatility_threshold()
                if volatility > high_vol_threshold:
                    volatile_penalty = validation_config.get(
                        "volatile_market_penalty", 0.85
                    )
                    base_score *= volatile_penalty

                # トレンドとシグナルの整合性チェック
                if (
                    trend_direction == "upward"
                    and signal.signal_type == SignalType.BUY
                    or trend_direction == "downward"
                    and signal.signal_type == SignalType.SELL
                ):
                    base_score *= 1.1  # トレンドフォロー
                elif (
                    trend_direction != "neutral"
                    and signal.signal_type != SignalType.HOLD
                    and (
                        (
                            trend_direction == "upward"
                            and signal.signal_type == SignalType.SELL
                        )
                        or (
                            trend_direction == "downward"
                            and signal.signal_type == SignalType.BUY
                        )
                    )
                ):
                    # トレンドに逆らうシグナルは信頼度を下げる
                    trend_penalty = validation_config.get(
                        "trend_continuation_penalty", 0.85
                    )
                    base_score *= trend_penalty

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
                        strong_signals = same_type_signals[
                            same_type_signals["Strength"] == "strong"
                        ]
                        success_rate = (
                            len(strong_signals) / len(same_type_signals) * 0.7 + 0.3
                        )

                    # 成功率による調整（設定から取得）
                    performance_config = validation_config.get(
                        "performance_multipliers", {}
                    )
                    multiplier_base = performance_config.get("base", 0.6)
                    multiplier_range = performance_config.get("range", 0.7)
                    performance_multiplier = (
                        multiplier_base + multiplier_range * success_rate
                    )
                    base_score *= performance_multiplier

                    # 同じ強度の過去シグナルの成功率も考慮
                    same_strength = same_type_signals[
                        same_type_signals["Strength"] == signal.strength.value
                    ]
                    if (
                        len(same_strength) > 0
                        and "Success" in historical_performance.columns
                    ):
                        strength_success_rate = same_strength["Success"].mean()
                        # 強度別成功率による微調整（設定から取得）
                        strength_config = performance_config.get(
                            "strength_adjustment", {}
                        )
                        strength_base = strength_config.get("base", 0.9)
                        strength_range = strength_config.get("range", 0.2)
                        base_score *= (
                            strength_base + strength_range * strength_success_rate
                        )

            # シグナルの新鮮度(タイムスタンプからの経過時間)による調整
            if hasattr(signal, "timestamp") and signal.timestamp:
                from datetime import datetime, timezone

                current_time = datetime.now(timezone.utc)
                signal_time = signal.timestamp
                if signal_time.tzinfo is None:
                    signal_time = signal_time.replace(tzinfo=timezone.utc)

                time_diff = (
                    current_time - signal_time
                ).total_seconds() / 3600  # 時間差

                freshness = self.config.get_signal_freshness()
                warning_hours = freshness.get("warning_hours", 24)
                stale_hours = freshness.get("stale_hours", 72)

                freshness_multipliers = validation_config.get("time_multipliers", {})
                warning_multiplier = freshness_multipliers.get("warning", 0.9)
                stale_multiplier = freshness_multipliers.get("stale", 0.8)

                if time_diff > stale_hours:
                    base_score *= stale_multiplier
                elif time_diff > warning_hours:
                    base_score *= warning_multiplier

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
            # 売り条件を優先(より安全)
            for key in overlapping_keys:
                merged[f"buy_{key}"] = buy_conditions[key]
                merged[f"sell_{key}"] = sell_conditions[key]
                del merged[key]

        # 残りの売り条件を追加
        for key, value in sell_conditions.items():
            if key not in overlapping_keys:
                merged[key] = value

        return merged

    def _slice_patterns(
        self, all_patterns: Dict[str, Any], current_index: int, lookback_window: int
    ) -> Dict[str, Any]:
        """
        パターンデータをスライスして現在のウィンドウ用に調整
        改善されたpatterns.pyの構造により簡素化

        Args:
            all_patterns: 全パターンデータ
            current_index: 現在のインデックス
            lookback_window: ルックバックウィンドウ

        Returns:
            スライスされたパターンデータ
        """
        try:
            sliced_patterns = {}
            start_idx = current_index - lookback_window + 1
            end_idx = current_index + 1

            # DataFrameのスライス処理
            for key in ["crosses", "breakouts"]:
                if key in all_patterns and isinstance(all_patterns[key], pd.DataFrame):
                    df = all_patterns[key]
                    if not df.empty:
                        sliced_patterns[key] = df.iloc[start_idx:end_idx].copy()
                    else:
                        sliced_patterns[key] = pd.DataFrame()
                else:
                    sliced_patterns[key] = pd.DataFrame()

            # 辞書データ（levels, trends）はそのまま使用
            # これらは全体に対する分析結果なので、時系列でスライスする必要がない
            sliced_patterns["levels"] = all_patterns.get("levels", {})
            sliced_patterns["trends"] = all_patterns.get("trends", {})

            # 最新シグナルと総合信頼度はそのまま使用
            sliced_patterns["latest_signal"] = all_patterns.get("latest_signal")
            sliced_patterns["overall_confidence"] = all_patterns.get(
                "overall_confidence", 0
            )

            return sliced_patterns

        except Exception as e:
            logger.debug(f"パターンスライス処理エラー: {e}")
            return {
                "crosses": pd.DataFrame(),
                "breakouts": pd.DataFrame(),
                "levels": {},
                "trends": {},
                "latest_signal": None,
                "overall_confidence": 0,
            }


# カスタムルールの例
class VolumeSpikeBuyRule(SignalRule):
    """出来高急増買いルール"""

    def __init__(
        self,
        volume_factor: Optional[float] = None,
        price_change: Optional[float] = None,
        weight: float = 1.5,
        confidence_base: Optional[float] = None,
        confidence_multiplier: Optional[float] = None,
        config: Optional["SignalRulesConfig"] = None,
    ):
        super().__init__("Volume Spike Buy", weight)

        # 設定ファイルからデフォルト値を取得
        if config is None:
            config = SignalRulesConfig()

        volume_settings = config.get_volume_spike_settings()

        self.volume_factor = (
            volume_factor
            if volume_factor is not None
            else volume_settings["volume_factor"]
        )
        self.price_change = (
            price_change
            if price_change is not None
            else volume_settings["price_change_threshold"]
        )
        self.confidence_base = (
            confidence_base
            if confidence_base is not None
            else volume_settings["confidence_base"]
        )
        self.confidence_multiplier = (
            confidence_multiplier
            if confidence_multiplier is not None
            else volume_settings["confidence_multiplier"]
        )

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional["SignalRulesConfig"] = None,
    ) -> Tuple[bool, float]:
        if config is None:
            config = SignalRulesConfig()

        volume_settings = config.get_volume_spike_settings()
        volume_factor = volume_settings["volume_factor"]
        price_change_threshold = volume_settings["price_change_threshold"]
        confidence_base = volume_settings["confidence_base"]
        confidence_multiplier = volume_settings["confidence_multiplier"]

        min_data = config.get_min_data_for_generation()
        volume_period = config.get_volume_calculation_period()

        if len(df) < min_data:
            return False, 0.0

        # 平均出来高（設定値に基づく期間）
        avg_volume = df["Volume"].iloc[-volume_period:-1].mean()
        latest_volume = df["Volume"].iloc[-1]

        # 価格変化
        price_change = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[
            -2
        ]

        # 出来高が平均の指定倍数以上かつ価格が指定%以上上昇
        if (
            latest_volume > avg_volume * volume_factor
            and price_change > price_change_threshold
        ):
            volume_ratio = latest_volume / avg_volume
            confidence = min(
                (volume_ratio - volume_factor) * confidence_multiplier
                + confidence_base,
                100,
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
    pattern_recognizer = ChartPatternRecognizer()
    patterns = pattern_recognizer.detect_all_patterns(df)

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
            "conditions_met": signal.conditions_met,
        }

        # 市場環境情報も含めて検証
        config = SignalRulesConfig()
        trend_period = config.get_trend_lookback_period()

        market_context = {
            "volatility": float(df["Close"].pct_change().std()),
            "trend_direction": (
                "upward"
                if df["Close"].iloc[-1] > df["Close"].iloc[-trend_period]
                else "downward"
            ),
        }
        validity = generator.validate_signal(signal, market_context=market_context)
        signal_data["validity_score"] = validity
        signal_data["market_context"] = market_context

        logger.info("シグナル生成完了", **signal_data)

        # ビジネスイベントとして記録
        log_business_event("signal_generated", **signal_data)
    else:
        logger.info(
            "シグナル生成結果",
            section="signal_generation",
            signal_type="none",
            result="シグナルなし",
        )

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
            "latest_signals": (
                active_signals.tail(5).to_dict("records")
                if len(active_signals) > 0
                else []
            ),
        }

        logger.info("時系列シグナル生成完了", **series_summary)
        log_business_event("time_series_signals_generated", **series_summary)

        # デモ用表示ログ
        logger.info(
            "時系列シグナル表示",
            signal_count=len(active_signals),
            latest_signals=(
                active_signals.tail(10).to_dict("records")
                if len(active_signals) > 0
                else []
            ),
        )
    else:
        logger.info("時系列シグナル生成結果", result="no_active_signals")
