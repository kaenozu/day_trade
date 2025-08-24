"""
取引記録管理パッケージ

このパッケージは巨大なtrade_manager.pyファイルを機能別に分割したものです。

構成:
- types.py: 取引関連の列挙型
- models.py: 取引関連のデータクラス
- manager.py: TradeManagerクラス
"""

from .types import TradeType, TradeStatus
from .models import Trade, Position, RealizedPnL
from .manager import TradeManager

__all__ = [
    "TradeType",
    "TradeStatus", 
    "Trade",
    "Position",
    "RealizedPnL",
    "TradeManager",
]