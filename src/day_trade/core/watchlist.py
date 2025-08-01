"""
ウォッチリスト管理モジュール
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..models import db_manager, WatchlistItem, Stock
from ..data.stock_fetcher import StockFetcher


class WatchlistManager:
    """ウォッチリスト管理クラス"""
    
    def __init__(self):
        self.fetcher = StockFetcher()
    
    def add_stock(self, stock_code: str, group_name: str = "default", memo: str = "") -> bool:
        """
        銘柄をウォッチリストに追加
        
        Args:
            stock_code: 証券コード
            group_name: グループ名
            memo: メモ
            
        Returns:
            追加に成功した場合True
        """
        try:
            with db_manager.session_scope() as session:
                # 重複チェック
                existing = session.query(WatchlistItem).filter(
                    and_(
                        WatchlistItem.stock_code == stock_code,
                        WatchlistItem.group_name == group_name
                    )
                ).first()
                
                if existing:
                    return False  # 既に存在
                
                # 銘柄マスタにない場合は作成
                stock = session.query(Stock).filter(Stock.code == stock_code).first()
                if not stock:
                    # 企業情報を取得して銘柄マスタに追加
                    company_info = self.fetcher.get_company_info(stock_code)
                    if company_info:
                        stock = Stock(
                            code=stock_code,
                            name=company_info.get('name', stock_code),
                            sector=company_info.get('sector'),
                            industry=company_info.get('industry')
                        )
                        session.add(stock)
                        session.flush()  # IDを取得するため
                
                # ウォッチリストに追加
                watchlist_item = WatchlistItem(
                    stock_code=stock_code,
                    group_name=group_name,
                    memo=memo
                )
                session.add(watchlist_item)
                
                return True
                
        except Exception as e:
            print(f"ウォッチリスト追加エラー: {e}")
            return False
    
    def remove_stock(self, stock_code: str, group_name: str = "default") -> bool:
        """
        銘柄をウォッチリストから削除
        
        Args:
            stock_code: 証券コード
            group_name: グループ名
            
        Returns:
            削除に成功した場合True
        """
        try:
            with db_manager.session_scope() as session:
                item = session.query(WatchlistItem).filter(
                    and_(
                        WatchlistItem.stock_code == stock_code,
                        WatchlistItem.group_name == group_name
                    )
                ).first()
                
                if item:
                    session.delete(item)
                    return True
                else:
                    return False
                    
        except Exception as e:
            print(f"ウォッチリスト削除エラー: {e}")
            return False
    
    def get_watchlist(self, group_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        ウォッチリストを取得
        
        Args:
            group_name: グループ名（指定しない場合は全て）
            
        Returns:
            ウォッチリストアイテムのリスト
        """
        try:
            with db_manager.session_scope() as session:
                query = session.query(WatchlistItem).join(Stock)
                
                if group_name:
                    query = query.filter(WatchlistItem.group_name == group_name)
                
                items = query.all()
                
                result = []
                for item in items:
                    result.append({
                        'stock_code': item.stock_code,
                        'stock_name': item.stock.name if item.stock else item.stock_code,
                        'group_name': item.group_name,
                        'memo': item.memo,
                        'added_date': item.created_at
                    })
                
                return result
                
        except Exception as e:
            print(f"ウォッチリスト取得エラー: {e}")
            return []
    
    def get_groups(self) -> List[str]:
        """
        ウォッチリストのグループ一覧を取得
        
        Returns:
            グループ名のリスト
        """
        try:
            with db_manager.session_scope() as session:
                groups = session.query(WatchlistItem.group_name).distinct().all()
                return [group[0] for group in groups]
                
        except Exception as e:
            print(f"グループ取得エラー: {e}")
            return []
    
    def get_watchlist_with_prices(self, group_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        価格情報付きのウォッチリストを取得
        
        Args:
            group_name: グループ名（指定しない場合は全て）
            
        Returns:
            銘柄コードをキーとした価格情報付きデータ
        """
        watchlist = self.get_watchlist(group_name)
        if not watchlist:
            return {}
        
        # 銘柄コードを抽出
        stock_codes = [item['stock_code'] for item in watchlist]
        
        # 価格情報を取得
        price_data = self.fetcher.get_realtime_data(stock_codes)
        
        # ウォッチリスト情報と価格情報をマージ
        result = {}
        for item in watchlist:
            code = item['stock_code']
            result[code] = {
                **item,
                **price_data.get(code, {})
            }
        
        return result
    
    def update_memo(self, stock_code: str, group_name: str, memo: str) -> bool:
        """
        メモを更新
        
        Args:
            stock_code: 証券コード
            group_name: グループ名
            memo: 新しいメモ
            
        Returns:
            更新に成功した場合True
        """
        try:
            with db_manager.session_scope() as session:
                item = session.query(WatchlistItem).filter(
                    and_(
                        WatchlistItem.stock_code == stock_code,
                        WatchlistItem.group_name == group_name
                    )
                ).first()
                
                if item:
                    item.memo = memo
                    return True
                else:
                    return False
                    
        except Exception as e:
            print(f"メモ更新エラー: {e}")
            return False
    
    def move_to_group(self, stock_code: str, from_group: str, to_group: str) -> bool:
        """
        銘柄を別のグループに移動
        
        Args:
            stock_code: 証券コード
            from_group: 移動元グループ
            to_group: 移動先グループ
            
        Returns:
            移動に成功した場合True
        """
        try:
            with db_manager.session_scope() as session:
                item = session.query(WatchlistItem).filter(
                    and_(
                        WatchlistItem.stock_code == stock_code,
                        WatchlistItem.group_name == from_group
                    )
                ).first()
                
                if item:
                    # 移動先に同じ銘柄が既に存在するかチェック
                    existing = session.query(WatchlistItem).filter(
                        and_(
                            WatchlistItem.stock_code == stock_code,
                            WatchlistItem.group_name == to_group
                        )
                    ).first()
                    
                    if existing:
                        return False  # 移動先に既に存在
                    
                    item.group_name = to_group
                    return True
                else:
                    return False
                    
        except Exception as e:
            print(f"グループ移動エラー: {e}")
            return False