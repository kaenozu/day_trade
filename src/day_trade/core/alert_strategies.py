"""
アラート戦略パターン実装
各AlertTypeに対応する評価戦略を定義
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple, Union

from ..models.enums import AlertType
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class AlertStrategy(ABC):
    """アラート評価戦略の基底クラス"""

    @abstractmethod
    def evaluate(
        self,
        condition_value: Union[Decimal, float, str],
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any] = None,
        comparison_operator: str = ">",
        custom_parameters: Optional[Dict] = None
    ) -> Tuple[bool, str, Any]:
        """
        アラート条件を評価

        Returns:
            Tuple[bool, str, Any]: (トリガー判定, メッセージ, 現在値)
        """
        pass


class PriceAboveStrategy(AlertStrategy):
    """価格上昇アラート戦略"""

    def evaluate(
        self,
        condition_value: Union[Decimal, float, str],
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any] = None,
        comparison_operator: str = ">",
        custom_parameters: Optional[Dict] = None
    ) -> Tuple[bool, str, Any]:
        try:
            threshold = Decimal(str(condition_value))
            if self._compare_values(current_price, threshold, comparison_operator):
                message = f"価格が {condition_value} を上回りました (現在価格: ¥{current_price:,})"
                return True, message, current_price
            return False, "", current_price
        except Exception as e:
            logger.error(f"価格上昇アラート評価エラー: {e}")
            return False, "", current_price

    def _compare_values(self, current: Decimal, target: Decimal, operator: str) -> bool:
        if operator == ">":
            return current > target
        elif operator == ">=":
            return current >= target
        return False


class PriceBelowStrategy(AlertStrategy):
    """価格下落アラート戦略"""

    def evaluate(
        self,
        condition_value: Union[Decimal, float, str],
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any] = None,
        comparison_operator: str = "<",
        custom_parameters: Optional[Dict] = None
    ) -> Tuple[bool, str, Any]:
        try:
            threshold = Decimal(str(condition_value))
            if self._compare_values(current_price, threshold, comparison_operator):
                message = f"価格が {condition_value} を下回りました (現在価格: ¥{current_price:,})"
                return True, message, current_price
            return False, "", current_price
        except Exception as e:
            logger.error(f"価格下落アラート評価エラー: {e}")
            return False, "", current_price

    def _compare_values(self, current: Decimal, target: Decimal, operator: str) -> bool:
        if operator == "<":
            return current < target
        elif operator == "<=":
            return current <= target
        return False


class ChangePercentUpStrategy(AlertStrategy):
    """上昇率アラート戦略"""

    def evaluate(
        self,
        condition_value: Union[Decimal, float, str],
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any] = None,
        comparison_operator: str = ">",
        custom_parameters: Optional[Dict] = None
    ) -> Tuple[bool, str, Any]:
        try:
            threshold = float(condition_value)
            if change_percent >= threshold:
                message = f"上昇率が {condition_value}% を超えました (現在: {change_percent:.2f}%)"
                return True, message, change_percent
            return False, "", change_percent
        except Exception as e:
            logger.error(f"上昇率アラート評価エラー: {e}")
            return False, "", change_percent


class ChangePercentDownStrategy(AlertStrategy):
    """下落率アラート戦略"""

    def evaluate(
        self,
        condition_value: Union[Decimal, float, str],
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any] = None,
        comparison_operator: str = "<",
        custom_parameters: Optional[Dict] = None
    ) -> Tuple[bool, str, Any]:
        try:
            threshold = float(condition_value)
            if change_percent <= threshold:
                message = f"下落率が {abs(threshold)}% を超えました (現在: {change_percent:.2f}%)"
                return True, message, change_percent
            return False, "", change_percent
        except Exception as e:
            logger.error(f"下落率アラート評価エラー: {e}")
            return False, "", change_percent


class VolumeSpikeStrategy(AlertStrategy):
    """出来高急増アラート戦略"""

    def evaluate(
        self,
        condition_value: Union[Decimal, float, str],
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any] = None,
        comparison_operator: str = ">",
        custom_parameters: Optional[Dict] = None
    ) -> Tuple[bool, str, Any]:
        try:
            if historical_data is None or historical_data.empty:
                return False, "", 1.0

            threshold = float(condition_value)
            avg_volume = historical_data["Volume"].rolling(window=20).mean().iloc[-1]
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1

            if volume_ratio >= threshold:
                message = f"出来高急増を検出 (平均の {volume_ratio:.1f}倍: {volume:,})"
                return True, message, volume_ratio
            return False, "", volume_ratio
        except Exception as e:
            logger.error(f"出来高急増アラート評価エラー: {e}")
            return False, "", 1.0


class RSIOverboughtStrategy(AlertStrategy):
    """RSI買われすぎアラート戦略"""

    def __init__(self, technical_indicators):
        self.technical_indicators = technical_indicators

    def evaluate(
        self,
        condition_value: Union[Decimal, float, str],
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any] = None,
        comparison_operator: str = ">",
        custom_parameters: Optional[Dict] = None
    ) -> Tuple[bool, str, Any]:
        try:
            if historical_data is None or historical_data.empty:
                return False, "", 0

            threshold = float(condition_value)
            rsi = self.technical_indicators.calculate_rsi(historical_data["Close"])
            if rsi.empty:
                return False, "", 0

            current_rsi = rsi.iloc[-1]
            if current_rsi >= threshold:
                message = f"RSI買われすぎ水準 (RSI: {current_rsi:.1f})"
                return True, message, current_rsi
            return False, "", current_rsi
        except Exception as e:
            logger.error(f"RSI買われすぎアラート評価エラー: {e}")
            return False, "", 0


class RSIOversoldStrategy(AlertStrategy):
    """RSI売られすぎアラート戦略"""

    def __init__(self, technical_indicators):
        self.technical_indicators = technical_indicators

    def evaluate(
        self,
        condition_value: Union[Decimal, float, str],
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any] = None,
        comparison_operator: str = "<",
        custom_parameters: Optional[Dict] = None
    ) -> Tuple[bool, str, Any]:
        try:
            if historical_data is None or historical_data.empty:
                return False, "", 0

            threshold = float(condition_value)
            rsi = self.technical_indicators.calculate_rsi(historical_data["Close"])
            if rsi.empty:
                return False, "", 0

            current_rsi = rsi.iloc[-1]
            if current_rsi <= threshold:
                message = f"RSI売られすぎ水準 (RSI: {current_rsi:.1f})"
                return True, message, current_rsi
            return False, "", current_rsi
        except Exception as e:
            logger.error(f"RSI売られすぎアラート評価エラー: {e}")
            return False, "", 0


class CustomConditionStrategy(AlertStrategy):
    """カスタム条件アラート戦略"""

    def evaluate(
        self,
        condition_value: Union[Decimal, float, str],
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any] = None,
        comparison_operator: str = ">",
        custom_parameters: Optional[Dict] = None
    ) -> Tuple[bool, str, Any]:
        try:
            if not custom_parameters or 'custom_function' not in custom_parameters:
                return False, "", "Custom"

            custom_function = custom_parameters['custom_function']
            symbol = custom_parameters.get('symbol', '')
            description = custom_parameters.get('description', 'カスタム条件')

            result = custom_function(
                symbol,
                current_price,
                volume,
                change_percent,
                historical_data,
                custom_parameters,
            )

            if result:
                message = f"カスタム条件が満たされました: {description}"
                return True, message, "Custom"
            return False, "", "Custom"
        except Exception as e:
            logger.error(f"カスタム条件アラート評価エラー: {e}")
            return False, "", "Custom"


class AlertStrategyFactory:
    """アラート戦略ファクトリー"""

    def __init__(self, technical_indicators=None):
        self.technical_indicators = technical_indicators
        self._strategies = {
            AlertType.PRICE_ABOVE: PriceAboveStrategy(),
            AlertType.PRICE_BELOW: PriceBelowStrategy(),
            AlertType.CHANGE_PERCENT_UP: ChangePercentUpStrategy(),
            AlertType.CHANGE_PERCENT_DOWN: ChangePercentDownStrategy(),
            AlertType.VOLUME_SPIKE: VolumeSpikeStrategy(),
            AlertType.CUSTOM_CONDITION: CustomConditionStrategy(),
        }

        # テクニカル指標が必要な戦略は後から設定
        if technical_indicators:
            self._strategies[AlertType.RSI_OVERBOUGHT] = RSIOverboughtStrategy(technical_indicators)
            self._strategies[AlertType.RSI_OVERSOLD] = RSIOversoldStrategy(technical_indicators)

    def get_strategy(self, alert_type: AlertType) -> Optional[AlertStrategy]:
        """指定されたアラートタイプに対応する戦略を取得"""
        return self._strategies.get(alert_type)

    def register_strategy(self, alert_type: AlertType, strategy: AlertStrategy):
        """カスタム戦略を登録"""
        self._strategies[alert_type] = strategy
