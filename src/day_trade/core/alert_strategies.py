"""
アラート評価戦略パターン
各アラートタイプに対応した評価ロジックを分離
"""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from ..models.enums import AlertType
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class AlertEvaluationStrategy(ABC):
    """アラート評価戦略の基底クラス"""

    @abstractmethod
    def evaluate(
        self,
        condition_value: Any,
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any],
        comparison_operator: str = ">",
        custom_parameters: dict = None
    ) -> tuple[bool, str, Any]:
        """
        アラート条件を評価

        Returns:
            (条件満足フラグ, メッセージ, 現在値) のタプル
        """
        pass


class PriceAboveStrategy(AlertEvaluationStrategy):
    """価格上昇アラート戦略"""

    def evaluate(
        self,
        condition_value: Any,
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any],
        comparison_operator: str = ">",
        custom_parameters: dict = None
    ) -> tuple[bool, str, Any]:
        try:
            target_price = Decimal(str(condition_value))
            is_triggered = self._compare_values(current_price, target_price, comparison_operator)

            if is_triggered:
                message = f"価格が {target_price} を上回りました (現在価格: ¥{current_price:,})"
                return True, message, current_price

        except (ValueError, TypeError) as e:
            logger.error(f"価格上昇アラート評価エラー: {e}")

        return False, "", current_price

    def _compare_values(self, current: Decimal, target: Decimal, operator: str) -> bool:
        if operator == ">":
            return current > target
        elif operator == ">=":
            return current >= target
        elif operator == "==":
            return abs(current - target) < Decimal("0.01")
        return False


class PriceBelowStrategy(AlertEvaluationStrategy):
    """価格下落アラート戦略"""

    def evaluate(
        self,
        condition_value: Any,
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any],
        comparison_operator: str = "<",
        custom_parameters: dict = None
    ) -> tuple[bool, str, Any]:
        try:
            target_price = Decimal(str(condition_value))
            is_triggered = self._compare_values(current_price, target_price, comparison_operator)

            if is_triggered:
                message = f"価格が {target_price} を下回りました (現在価格: ¥{current_price:,})"
                return True, message, current_price

        except (ValueError, TypeError) as e:
            logger.error(f"価格下落アラート評価エラー: {e}")

        return False, "", current_price

    def _compare_values(self, current: Decimal, target: Decimal, operator: str) -> bool:
        if operator == "<":
            return current < target
        elif operator == "<=":
            return current <= target
        elif operator == "==":
            return abs(current - target) < Decimal("0.01")
        return False


class ChangePercentUpStrategy(AlertEvaluationStrategy):
    """上昇率アラート戦略"""

    def evaluate(
        self,
        condition_value: Any,
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any],
        comparison_operator: str = ">=",
        custom_parameters: dict = None
    ) -> tuple[bool, str, Any]:
        try:
            target_percent = float(condition_value)
            is_triggered = change_percent >= target_percent

            if is_triggered:
                message = f"上昇率が {target_percent}% を超えました (現在: {change_percent:.2f}%)"
                return True, message, change_percent

        except (ValueError, TypeError) as e:
            logger.error(f"上昇率アラート評価エラー: {e}")

        return False, "", change_percent


class ChangePercentDownStrategy(AlertEvaluationStrategy):
    """下落率アラート戦略"""

    def evaluate(
        self,
        condition_value: Any,
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any],
        comparison_operator: str = "<=",
        custom_parameters: dict = None
    ) -> tuple[bool, str, Any]:
        try:
            target_percent = float(condition_value)
            is_triggered = change_percent <= target_percent

            if is_triggered:
                message = f"下落率が {abs(target_percent)}% を超えました (現在: {change_percent:.2f}%)"
                return True, message, change_percent

        except (ValueError, TypeError) as e:
            logger.error(f"下落率アラート評価エラー: {e}")

        return False, "", change_percent


class VolumeSpikeStrategy(AlertEvaluationStrategy):
    """出来高急増アラート戦略"""

    def evaluate(
        self,
        condition_value: Any,
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any],
        comparison_operator: str = ">=",
        custom_parameters: dict = None
    ) -> tuple[bool, str, Any]:
        try:
            if historical_data is None or historical_data.empty:
                return False, "", 1.0

            target_ratio = float(condition_value)

            # 過去20日の平均出来高を計算
            window_size = custom_parameters.get("volume_window", 20) if custom_parameters else 20
            avg_volume = historical_data["Volume"].rolling(window=window_size).mean().iloc[-1]

            if avg_volume <= 0:
                return False, "", 1.0

            volume_ratio = volume / avg_volume
            is_triggered = volume_ratio >= target_ratio

            if is_triggered:
                message = f"出来高急増を検出 (平均の {volume_ratio:.1f}倍: {volume:,})"
                return True, message, volume_ratio

        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"出来高急増アラート評価エラー: {e}")

        return False, "", 1.0


class RSIOverBoughtStrategy(AlertEvaluationStrategy):
    """RSI買われすぎアラート戦略"""

    def __init__(self, technical_indicators):
        self.technical_indicators = technical_indicators

    def evaluate(
        self,
        condition_value: Any,
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any],
        comparison_operator: str = ">=",
        custom_parameters: dict = None
    ) -> tuple[bool, str, Any]:
        try:
            if historical_data is None or historical_data.empty:
                return False, "", 50.0

            target_rsi = float(condition_value)

            # RSI期間をカスタムパラメーターから取得
            rsi_period = custom_parameters.get("rsi_period", 14) if custom_parameters else 14

            rsi = self.technical_indicators.calculate_rsi(
                historical_data["Close"], period=rsi_period
            )

            if rsi.empty:
                return False, "", 50.0

            current_rsi = rsi.iloc[-1]
            is_triggered = current_rsi >= target_rsi

            if is_triggered:
                message = f"RSI買われすぎ水準 (RSI: {current_rsi:.1f})"
                return True, message, current_rsi

        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"RSI買われすぎアラート評価エラー: {e}")

        return False, "", 50.0


class RSIOverSoldStrategy(AlertEvaluationStrategy):
    """RSI売られすぎアラート戦略"""

    def __init__(self, technical_indicators):
        self.technical_indicators = technical_indicators

    def evaluate(
        self,
        condition_value: Any,
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any],
        comparison_operator: str = "<=",
        custom_parameters: dict = None
    ) -> tuple[bool, str, Any]:
        try:
            if historical_data is None or historical_data.empty:
                return False, "", 50.0

            target_rsi = float(condition_value)

            # RSI期間をカスタムパラメーターから取得
            rsi_period = custom_parameters.get("rsi_period", 14) if custom_parameters else 14

            rsi = self.technical_indicators.calculate_rsi(
                historical_data["Close"], period=rsi_period
            )

            if rsi.empty:
                return False, "", 50.0

            current_rsi = rsi.iloc[-1]
            is_triggered = current_rsi <= target_rsi

            if is_triggered:
                message = f"RSI売られすぎ水準 (RSI: {current_rsi:.1f})"
                return True, message, current_rsi

        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"RSI売られすぎアラート評価エラー: {e}")

        return False, "", 50.0


class CustomConditionStrategy(AlertEvaluationStrategy):
    """カスタム条件アラート戦略"""

    def evaluate(
        self,
        condition_value: Any,
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any],
        comparison_operator: str = "==",
        custom_parameters: dict = None
    ) -> tuple[bool, str, Any]:
        try:
            # カスタムパラメーターからカスタム関数を取得
            if not custom_parameters or "custom_function" not in custom_parameters:
                return False, "", "Custom"

            custom_function = custom_parameters["custom_function"]

            if not callable(custom_function):
                return False, "", "Custom"

            # カスタム関数を実行
            result = custom_function(
                symbol=custom_parameters.get("symbol", ""),
                price=current_price,
                volume=volume,
                change_pct=change_percent,
                historical_data=historical_data,
                params=custom_parameters
            )

            if result:
                description = custom_parameters.get("description", "カスタム条件")
                message = f"カスタム条件が満たされました: {description}"
                return True, message, "Custom"

        except Exception as e:
            logger.error(f"カスタム条件アラート評価エラー: {e}")

        return False, "", "Custom"


class AlertStrategyFactory:
    """アラート戦略ファクトリー"""

    def __init__(self, technical_indicators):
        self.technical_indicators = technical_indicators
        self._strategies = {
            AlertType.PRICE_ABOVE: PriceAboveStrategy(),
            AlertType.PRICE_BELOW: PriceBelowStrategy(),
            AlertType.CHANGE_PERCENT_UP: ChangePercentUpStrategy(),
            AlertType.CHANGE_PERCENT_DOWN: ChangePercentDownStrategy(),
            AlertType.VOLUME_SPIKE: VolumeSpikeStrategy(),
            AlertType.RSI_OVERBOUGHT: RSIOverBoughtStrategy(technical_indicators),
            AlertType.RSI_OVERSOLD: RSIOverSoldStrategy(technical_indicators),
            AlertType.CUSTOM_CONDITION: CustomConditionStrategy(),
        }

    def get_strategy(self, alert_type: AlertType) -> Optional[AlertEvaluationStrategy]:
        """指定されたアラートタイプに対応する戦略を取得"""
        return self._strategies.get(alert_type)

    def add_strategy(self, alert_type: AlertType, strategy: AlertEvaluationStrategy):
        """新しい戦略を追加"""
        self._strategies[alert_type] = strategy

    def remove_strategy(self, alert_type: AlertType):
        """戦略を削除"""
        if alert_type in self._strategies:
            del self._strategies[alert_type]
