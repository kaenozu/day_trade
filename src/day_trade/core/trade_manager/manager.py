"""
取引記録管理機能（モジュール分割版）
各機能モジュールを組み合わせた統合クラス
"""

from decimal import Decimal
from typing import Dict, List, Optional

from ...utils.logging_config import get_context_logger
from .database_integration import DatabaseIntegration
from .data_management import DataManagement
from .models import Position, RealizedPnL, Trade
from .portfolio_analytics import PortfolioAnalytics
from .stock_trading import StockTrading
from .trade_operations import TradeOperations

logger = get_context_logger(__name__)


class TradeManager:
    """取引記録管理クラス（モジュール分割版）"""
    
    def __init__(
        self,
        commission_rate: Decimal = Decimal("0.001"),
        tax_rate: Decimal = Decimal("0.2"),
        load_from_db: bool = False,
    ):
        """
        初期化
        
        Args:
            commission_rate: 手数料率（デフォルト0.1%）
            tax_rate: 税率（デフォルト20%）
            load_from_db: データベースから取引履歴を読み込むかどうか
        """
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.realized_pnl: List[RealizedPnL] = []
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self._trade_counter = 0
        
        # ロガーを初期化
        self.logger = get_context_logger(__name__)
        
        # 各機能モジュールを初期化
        self.trade_operations = TradeOperations(self)
        self.database_integration = DatabaseIntegration(self)
        self.stock_trading = StockTrading(self)
        self.data_management = DataManagement(self)
        self.portfolio_analytics = PortfolioAnalytics(self)
        
        if load_from_db:
            self.database_integration.load_trades_from_db()
    
    # ===== 基本取引管理機能の委譲 =====
    def add_trade(
        self,
        symbol: str,
        trade_type,
        quantity: int,
        price: Decimal,
        timestamp=None,
        commission: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> str:
        """取引を追加"""
        return self.trade_operations.add_trade(
            symbol=symbol,
            trade_type=trade_type,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            commission=commission,
            notes=notes,
            persist_to_db=persist_to_db,
        )
    
    def _generate_trade_id(self) -> str:
        """取引IDを生成（後方互換性のため）"""
        return self.trade_operations.generate_trade_id()
    
    def _calculate_commission(self, price: Decimal, quantity: int) -> Decimal:
        """手数料を計算（後方互換性のため）"""
        return self.trade_operations.calculate_commission(price, quantity)
    
    def _update_position(self, trade: Trade):
        """ポジションを更新（後方互換性のため）"""
        return self.trade_operations.update_position(trade)
    
    def _get_earliest_buy_date(self, symbol: str):
        """最も古い買い取引の日付を取得（後方互換性のため）"""
        return self.trade_operations.get_earliest_buy_date(symbol)
    
    # ===== データベース連携機能の委譲 =====
    def _load_trades_from_db(self):
        """データベースから取引履歴を読み込み（後方互換性のため）"""
        return self.database_integration.load_trades_from_db()
    
    def sync_with_db(self):
        """データベースとの同期を実行"""
        return self.database_integration.sync_with_db()
    
    def add_trades_batch(
        self, trades_data: List[Dict], persist_to_db: bool = True
    ) -> List[str]:
        """複数の取引を一括追加"""
        return self.database_integration.add_trades_batch(trades_data, persist_to_db)
    
    def clear_all_data(self, persist_to_db: bool = True):
        """すべての取引データを削除"""
        return self.database_integration.clear_all_data(persist_to_db)
    
    # ===== 株式売買機能の委譲 =====
    def buy_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        current_market_price: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ):
        """株式買い注文を実行"""
        return self.stock_trading.buy_stock(
            symbol=symbol,
            quantity=quantity,
            price=price,
            current_market_price=current_market_price,
            notes=notes,
            persist_to_db=persist_to_db,
        )
    
    def sell_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        current_market_price: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ):
        """株式売り注文を実行"""
        return self.stock_trading.sell_stock(
            symbol=symbol,
            quantity=quantity,
            price=price,
            current_market_price=current_market_price,
            notes=notes,
            persist_to_db=persist_to_db,
        )
    
    def execute_trade_order(self, trade_order: Dict, persist_to_db: bool = True):
        """取引注文を実行"""
        return self.stock_trading.execute_trade_order(trade_order, persist_to_db)
    
    # ===== データ管理機能の委譲 =====
    def get_position(self, symbol: str) -> Optional[Position]:
        """ポジション情報を取得"""
        return self.data_management.get_position(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """全ポジション情報を取得"""
        return self.data_management.get_all_positions()
    
    def update_current_prices(self, prices: Dict[str, Decimal]):
        """現在価格を更新"""
        return self.data_management.update_current_prices(prices)
    
    def get_trade_history(self, symbol: Optional[str] = None) -> List[Trade]:
        """取引履歴を取得"""
        return self.data_management.get_trade_history(symbol)
    
    def get_realized_pnl_history(
        self, symbol: Optional[str] = None
    ) -> List[RealizedPnL]:
        """実現損益履歴を取得"""
        return self.data_management.get_realized_pnl_history(symbol)
    
    def export_to_csv(self, filepath: str, data_type: str = "trades"):
        """CSVファイルにエクスポート"""
        return self.data_management.export_to_csv(filepath, data_type)
    
    def save_to_json(self, filepath: str):
        """JSON形式で保存"""
        return self.data_management.save_to_json(filepath)
    
    def load_from_json(self, filepath: str):
        """JSON形式から読み込み"""
        return self.data_management.load_from_json(filepath)
    
    # ===== ポートフォリオ分析機能の委譲 =====
    def get_portfolio_summary(self) -> Dict:
        """ポートフォリオサマリーを取得"""
        return self.portfolio_analytics.get_portfolio_summary()
    
    def calculate_tax_implications(self, year: int) -> Dict:
        """税務計算"""
        return self.portfolio_analytics.calculate_tax_implications(year)