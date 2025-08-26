"""
データベース連携機能（統合クラス）
DatabaseLoaderとDatabaseOperationsを組み合わせた統合クラス
"""

from typing import Dict, List

from ...utils.logging_config import get_context_logger
from .database_loader import DatabaseLoader
from .database_operations import DatabaseOperations

logger = get_context_logger(__name__)


class DatabaseIntegration:
    """データベース連携操作を管理する統合クラス"""
    
    def __init__(self, trade_manager_ref):
        """
        初期化
        
        Args:
            trade_manager_ref: TradeManagerオブジェクトへの参照
        """
        self.trade_manager = trade_manager_ref
        self.logger = get_context_logger(__name__)
        
        # サブモジュールの初期化
        self.loader = DatabaseLoader(trade_manager_ref)
        self.operations = DatabaseOperations(trade_manager_ref)
    
    def load_trades_from_db(self):
        """データベースから取引履歴を読み込み"""
        return self.loader.load_trades_from_db()
    
    def sync_with_db(self):
        """データベースとの同期を実行"""
        return self.loader.sync_with_db()
    
    def add_trades_batch(
        self, trades_data: List[Dict], persist_to_db: bool = True
    ) -> List[str]:
        """複数の取引を一括追加"""
        return self.operations.add_trades_batch(trades_data, persist_to_db)
    
    def clear_all_data(self, persist_to_db: bool = True):
        """すべての取引データを削除"""
        return self.operations.clear_all_data(persist_to_db)