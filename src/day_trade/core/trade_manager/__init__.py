"""
取引記録管理パッケージ

このパッケージは巨大なmanager.pyファイルを機能別に分割し、
以下のモジュール構成で再構築されています。

構成:
- types.py: 取引関連の列挙型
- models.py: 取引関連のデータクラス
- trade_operations.py: 基本取引管理機能
- database_integration.py: データベース連携機能
- stock_trading.py: 株式売買機能
- data_management.py: データ管理・I/O機能
- portfolio_analytics.py: ポートフォリオ分析・税務計算機能
- manager_new.py: 統合TradeManagerクラス（各モジュールを組み合わせ）

後方互換性のため、元のインターフェースは維持されています。
"""

from .types import TradeType, TradeStatus
from .models import Trade, Position, RealizedPnL
from .manager import TradeManager

# 分割されたモジュールも必要に応じてインポート可能
from .trade_operations import TradeOperations
from .database_integration import DatabaseIntegration
from .database_loader import DatabaseLoader
from .database_operations import DatabaseOperations
from .stock_trading import StockTrading
from .buy_operations import BuyOperations
from .sell_operations import SellOperations
from .data_management import DataManagement
from .portfolio_analytics import PortfolioAnalytics

__all__ = [
    "TradeType",
    "TradeStatus",
    "Trade",
    "Position",
    "RealizedPnL",
    "TradeManager",
    # 分割されたモジュールクラス
    "TradeOperations",
    "DatabaseIntegration",
    "DatabaseLoader",
    "DatabaseOperations",
    "StockTrading",
    "BuyOperations",
    "SellOperations",
    "DataManagement",
    "PortfolioAnalytics",
]