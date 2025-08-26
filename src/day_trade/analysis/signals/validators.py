"""
検証・評価機能

シグナルの検証とデータ整合性チェック機能
"""

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

import pandas as pd

from ...utils.logging_config import get_context_logger
from .config import SignalRulesConfig
from .types import SignalType, TradingSignal

logger = get_context_logger(__name__)


class SignalValidator:
    """シグナル検証クラス"""

    def __init__(self, config: Optional[SignalRulesConfig] = None):
        self.config = config or SignalRulesConfig()

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
                base_score = self._apply_market_context_adjustments(
                    base_score, signal, market_context, validation_config
                )

            # 過去のパフォーマンスデータによる調整
            if historical_performance is not None and len(historical_performance) > 0:
                base_score = self._apply_historical_performance_adjustments(
                    base_score, signal, historical_performance, validation_config
                )

            # シグナルの新鮮度(タイムスタンプからの経過時間)による調整
            if hasattr(signal, "timestamp") and signal.timestamp:
                base_score = self._apply_freshness_adjustments(
                    base_score, signal, validation_config
                )

            return min(max(base_score, 0), 100)  # 0-100の範囲に制限

        except Exception as e:
            logger.error(
                f"シグナルの有効性検証中に予期せぬエラーが発生しました。詳細: {e}"
            )
            return 0.0

    def _apply_market_context_adjustments(
        self,
        base_score: float,
        signal: TradingSignal,
        market_context: Dict[str, Any],
        validation_config: Dict[str, Any],
    ) -> float:
        """市場環境を考慮した調整を適用"""
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

        return base_score

    def _apply_historical_performance_adjustments(
        self,
        base_score: float,
        signal: TradingSignal,
        historical_performance: pd.DataFrame,
        validation_config: Dict[str, Any],
    ) -> float:
        """過去のパフォーマンスデータによる調整を適用"""
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

        return base_score

    def _apply_freshness_adjustments(
        self,
        base_score: float,
        signal: TradingSignal,
        validation_config: Dict[str, Any],
    ) -> float:
        """シグナル新鮮度による調整を適用"""
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

        return base_score


class DataValidator:
    """データ検証クラス"""

    def __init__(self, config: Optional[SignalRulesConfig] = None):
        self.config = config or SignalRulesConfig()

    def validate_input_data(
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
        """価格データの妥当性を検証"""
        try:
            # 正の値チェック
            price_columns = ["Open", "High", "Low", "Close", "Volume"]
            for col in price_columns:
                if latest_row[col] <= 0:
                    logger.error(f"{col}が0以下の値です: {latest_row[col]}")
                    return False

            # 価格の論理的整合性チェック
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

            return True

        except Exception as e:
            logger.error(f"価格データ検証中にエラーが発生: {e}")
            return False

    def _validate_indicators_data(self, indicators: pd.DataFrame) -> bool:
        """テクニカル指標データの完全性を検証"""
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
        """パターンデータの完全性を検証"""
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

    def safe_decimal_conversion(self, value: Any) -> Optional[Decimal]:
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

    def safe_decimal_to_float(self, decimal_value: Decimal) -> Optional[float]:
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