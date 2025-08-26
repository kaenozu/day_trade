"""
売買シグナル生成エンジン
"""

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_rules import SignalRule
from .band_rules import BollingerBandRule
from .config import SignalRulesConfig
from .macd_rules import MACDCrossoverRule, MACDDeathCrossRule
from .pattern_rules import DeadCrossRule, GoldenCrossRule, PatternBreakoutRule
from .rsi_rules import RSIOversoldRule, RSIOverboughtRule
from .types import SignalStrength, SignalType, TradingSignal
from .volume_rules import VolumeSpikeBuyRule
from ..indicators import TechnicalIndicators
from ..patterns import ChartPatternRecognizer
from ...utils.logging_config import get_context_logger, log_business_event

logger = get_context_logger(__name__)


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

        # Issue #656対応: デフォルトルール読み込みの統合
        if not self.buy_rules:
            self._load_default_rules("buy")
        if not self.sell_rules:
            self._load_default_rules("sell")

    def _create_rule_from_config(
        self, rule_config: Dict[str, Any]
    ) -> Optional[SignalRule]:
        """設定からルールオブジェクトを作成"""
        rule_type = rule_config.get("type")
        parameters = rule_config.get("parameters", {})

        rule_map = self._get_rule_mapping()

        if rule_type in rule_map:
            try:
                return rule_map[rule_type](**parameters)
            except Exception as e:
                logger.error(f"ルール作成エラー {rule_type}: {e}")
                return None
        else:
            logger.warning(f"未知のルールタイプ: {rule_type}")
            return None

    def _get_rule_mapping(self) -> Dict[str, type]:
        """
        ルールタイプとクラスのマッピングを取得（Issue #655対応）

        Returns:
            ルール名とクラスのマッピング辞書
        """
        return {
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

    def _load_default_rules(self, rule_type: str):
        """
        Issue #656対応: デフォルトルール読み込みの統合

        Args:
            rule_type: "buy" または "sell"
        """
        # 設定ファイルからデフォルト値を取得
        rsi_thresholds = self.config.get_rsi_thresholds()
        macd_settings = self.config.get_macd_settings()
        pattern_settings = self.config.get_pattern_breakout_settings()
        cross_settings = self.config.get_cross_settings()

        if rule_type == "buy":
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

        elif rule_type == "sell":
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
        else:
            logger.warning(f"Unknown rule type: {rule_type}. Expected 'buy' or 'sell'")

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
        """高ボラティリティ判定閾値を取得"""
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
            # データの基本検証 - Issue #657対応
            if not self._validate_input_data(df, indicators, patterns):
                return None

            # patternsがNoneの場合に空の辞書を代入
            if patterns is None:
                patterns = {}

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

            # 最新の価格とタイムスタンプ - Issue #657対応: Decimal型処理改善
            latest_price = self._safe_decimal_conversion(df["Close"].iloc[-1])
            if latest_price is None:
                logger.error("最新価格の取得に失敗しました")
                return None

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

            # Issue #658対応: lookback_window処理の改善
            # データ長とlookback_windowの妥当性チェック
            if lookback_window < 1:
                logger.warning(f"Invalid lookback_window: {lookback_window}. Using default value 50.")
                lookback_window = 50

            # 最低限必要なデータ数を設定から取得
            config_min_period = self.config.get_signal_settings().get(
                "min_data_period", 60
            )
            min_required = max(lookback_window, config_min_period)

            # データ長に対するlookback_windowの調整
            if len(df) < min_required:
                if len(df) < config_min_period:
                    logger.warning(
                        f"データが不足しています。最低 {config_min_period} 日分のデータが必要です。現在: {len(df)}日"
                    )
                    return pd.DataFrame()
                else:
                    # データ長に合わせてlookback_windowを調整
                    adjusted_window = min(lookback_window, len(df) - config_min_period // 2)
                    logger.debug(f"lookback_window調整: {lookback_window} -> {adjusted_window}")
                    lookback_window = max(adjusted_window, 20)  # 最小20に設定
                    min_required = max(lookback_window, config_min_period)

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
                    # Decimal型を安全にfloatに変換 - Issue #657対応
                    safe_price = self._safe_decimal_to_float(signal.price)
                    if safe_price is not None:
                        signals.append(
                            {
                                "Date": signal.timestamp,
                                "Signal": signal.signal_type.value,
                                "Strength": signal.strength.value,
                                "Confidence": signal.confidence,
                                "Price": safe_price,
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
        validation_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Issue #659対応: パラメータ化されたシグナル検証ロジック

        Args:
            signal: 検証するシグナル
            historical_performance: 過去のパフォーマンスデータ
            market_context: 市場環境情報(ボラティリティ,トレンド等)
            validation_params: 検証パラメータのオーバーライド

        Returns:
            有効性スコア(0-100)
        """
        try:
            base_score = signal.confidence

            # Issue #659対応: パラメータ化された検証ロジック
            # デフォルト設定を取得し、validation_paramsでオーバーライド可能
            default_config = self.config.get_signal_settings()
            validation_config = default_config.get("validation_adjustments", {})

            if validation_params:
                # validation_paramsが提供された場合は設定をオーバーライド
                validation_config.update(validation_params)

            # 強度による調整係数
            strength_multipliers = validation_config.get("strength_multipliers", {
                "strong": 1.2,
                "medium": 1.0,
                "weak": 0.8
            })
            strength_key = signal.strength.value if hasattr(signal.strength, 'value') else str(signal.strength)
            base_score *= strength_multipliers.get(strength_key, 1.0)

            # 複数条件の組み合わせによるボーナス
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
        """
        Issue #660対応: コンディション結合の衝突解決を簡素化
        プレフィックス付きで明確に分離することで衝突を回避
        """
        merged = {}

        # プレフィックス付きで明確に分離
        for key, value in buy_conditions.items():
            merged[f"buy_{key}"] = value

        for key, value in sell_conditions.items():
            merged[f"sell_{key}"] = value

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

    def _validate_input_data(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> bool:
        """
        入力データの完全性を検証 - Issue #657対応

        Args:
            df: 価格データのDataFrame
            indicators: テクニカル指標のDataFrame
            patterns: チャートパターン認識結果

        Returns:
            データが有効かどうか
        """
        try:
            # 価格データの基本検証
            min_data = self.config.get_min_data_for_generation()
            if df.empty or len(df) < min_data:
                logger.warning(f"価格データが不足しています（最低{min_data}日分必要）")
                return False

            # 必須カラムの存在確認
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"必須カラムが不足しています: {missing_columns}")
                return False

            # 最新データのNaN値チェック
            latest_row = df.iloc[-1]
            if latest_row[required_columns].isna().any():
                nan_columns = latest_row[required_columns][latest_row[required_columns].isna()].index.tolist()
                logger.error(f"最新データにNaN値があります: {nan_columns}")
                return False

            # 価格データの妥当性チェック
            if not self._validate_price_data(latest_row):
                return False

            # 指標データの検証
            if not self._validate_indicators_data(indicators):
                return False

            # パターンデータの検証
            if not self._validate_patterns_data(patterns):
                return False

            return True

        except Exception as e:
            logger.error(f"入力データ検証中にエラーが発生: {e}")
            return False

    def _validate_price_data(self, latest_row: pd.Series) -> bool:
        """
        価格データの妥当性を検証

        Args:
            latest_row: 最新の価格データ行

        Returns:
            価格データが妥当かどうか
        """
        try:
            # 正の値チェック
            price_columns = ["Open", "High", "Low", "Close", "Volume"]
            for col in price_columns:
                if latest_row[col] <= 0:
                    logger.error(f"{col}が0以下の値です: {latest_row[col]}")
                    return False

            # 価格の論理的整合性チェック (High >= Low, Close/Open が High-Low 範囲内)
            high, low, open_price, close = latest_row["High"], latest_row["Low"], latest_row["Open"], latest_row["Close"]

            if high < low:
                logger.error(f"高値({high})が安値({low})より低い値です")
                return False

            if not (low <= open_price <= high):
                logger.error(f"始値({open_price})が高安値範囲({low}-{high})外です")
                return False

            if not (low <= close <= high):
                logger.error(f"終値({close})が高安値範囲({low}-{high})外です")
                return False

            # 極端な価格変動チェック（前日比で99%以上の変動は異常とみなす）
            return True

        except Exception as e:
            logger.error(f"価格データ検証中にエラーが発生: {e}")
            return False

    def _validate_indicators_data(self, indicators: pd.DataFrame) -> bool:
        """
        テクニカル指標データの完全性を検証

        Args:
            indicators: テクニカル指標のDataFrame

        Returns:
            指標データが有効かどうか
        """
        try:
            if indicators.empty:
                logger.warning("テクニカル指標データが空です")
                return True  # 空でも処理は継続可能

            # 最新行の重要指標のNaN値チェック
            critical_indicators = ["RSI", "MACD", "MACD_Signal"]
            latest_indicators = indicators.iloc[-1]

            for indicator in critical_indicators:
                if indicator in indicators.columns:
                    if pd.isna(latest_indicators[indicator]):
                        logger.debug(f"重要指標{indicator}の最新値がNaNです")
                        # NaNでも処理は継続（各ルールで個別にハンドリング）

            # 指標値の範囲チェック（RSIは0-100範囲など）
            if "RSI" in indicators.columns and not indicators["RSI"].empty:
                rsi_latest = latest_indicators["RSI"]
                if not pd.isna(rsi_latest) and not (0 <= rsi_latest <= 100):
                    logger.warning(f"RSI値が範囲外です: {rsi_latest}")

            return True

        except Exception as e:
            logger.error(f"指標データ検証中にエラーが発生: {e}")
            return False

    def _validate_patterns_data(self, patterns: Dict) -> bool:
        """
        パターンデータの完全性を検証

        Args:
            patterns: チャートパターン認識結果

        Returns:
            パターンデータが有効かどうか
        """
        try:
            if not patterns:
                logger.debug("パターンデータが空です")
                return True  # 空でも処理は継続可能

            # DataFrameの構造チェック
            for key in ["crosses", "breakouts"]:
                if key in patterns:
                    pattern_data = patterns[key]
                    if not isinstance(pattern_data, pd.DataFrame):
                        logger.warning(f"パターンデータ{key}がDataFrameではありません: {type(pattern_data)}")
                        # 型が違っても処理は継続

            return True

        except Exception as e:
            logger.error(f"パターンデータ検証中にエラーが発生: {e}")
            return False

    def _safe_decimal_conversion(self, value: Any) -> Optional[Decimal]:
        """
        安全なDecimal型変換 - Issue #657対応

        Args:
            value: 変換する値

        Returns:
            Decimal値またはNone（変換失敗時）
        """
        try:
            if pd.isna(value):
                logger.error("NaN値をDecimalに変換しようとしました")
                return None

            # numpy型やpandas型も含めて安全に文字列化
            str_value = str(float(value))

            # 無限大や非数値のチェック
            if str_value.lower() in ['inf', '-inf', 'nan']:
                logger.error(f"無効な数値をDecimalに変換しようとしました: {str_value}")
                return None

            decimal_value = Decimal(str_value)

            # 負の価格チェック
            if decimal_value <= 0:
                logger.error(f"負または0の価格値です: {decimal_value}")
                return None

            return decimal_value

        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(f"Decimal変換エラー: {value} -> {e}")
            return None
        except Exception as e:
            logger.error(f"予期しないDecimal変換エラー: {value} -> {e}")
            return None

    def _safe_decimal_to_float(self, decimal_value: Decimal) -> Optional[float]:
        """
        Decimal型を安全にfloatに変換 - Issue #657対応

        Args:
            decimal_value: 変換するDecimal値

        Returns:
            float値またはNone（変換失敗時）
        """
        try:
            if decimal_value is None:
                return None

            float_value = float(decimal_value)

            # 無限大や非数値のチェック
            if not (float('inf') > float_value > float('-inf')):
                logger.error(f"Decimal->float変換で無効な値: {decimal_value}")
                return None

            return float_value

        except (ValueError, OverflowError) as e:
            logger.error(f"Decimal->float変換エラー: {decimal_value} -> {e}")
            return None
        except Exception as e:
            logger.error(f"予期しないDecimal->float変換エラー: {decimal_value} -> {e}")
            return None