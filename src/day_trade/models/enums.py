
import enum


class AlertType(enum.Enum):
    """アラートタイプ"""

    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    CHANGE_PERCENT_UP = "change_percent_up"
    CHANGE_PERCENT_DOWN = "change_percent_down"
    VOLUME_SPIKE = "volume_spike"
    RSI_OVERBOUGHT = "rsi_overbought"
    RSI_OVERSOLD = "rsi_oversold"
    MACD_SIGNAL = "macd_signal"
    BOLLINGER_BREAKOUT = "bollinger_breakout"
    PATTERN_DETECTED = "pattern_detected"
    CUSTOM_CONDITION = "custom_condition"


class TradeType(enum.Enum):
    BUY = "buy"
    SELL = "sell"
