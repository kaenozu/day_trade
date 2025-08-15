#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Time Manager - 市場時間管理システム
東証の営業時間を管理し、クローズ時の自動更新を停止
"""

from datetime import datetime, time
from typing import Optional
from enum import Enum


class MarketStatus(Enum):
    """市場状態"""
    CLOSED = "closed"
    OPEN = "open"
    PRE_MARKET = "pre_market"
    LUNCH_BREAK = "lunch_break"
    POST_MARKET = "post_market"


class MarketTimeManager:
    """東証市場時間管理"""
    
    def __init__(self):
        # 東証営業時間設定
        self.market_open_morning = time(9, 0)     # 9:00
        self.market_close_morning = time(11, 30)  # 11:30
        self.market_open_afternoon = time(12, 30) # 12:30
        self.market_close_afternoon = time(15, 0) # 15:00
        
        # プリマーケット・アフターマーケット
        self.pre_market_start = time(8, 0)        # 8:00
        self.post_market_end = time(16, 0)        # 16:00
    
    def get_current_time(self) -> datetime:
        """現在時刻を取得"""
        return datetime.now()
    
    def get_market_status(self, target_time: Optional[datetime] = None) -> MarketStatus:
        """市場状態を取得"""
        if target_time is None:
            target_time = self.get_current_time()
        
        # 土日は休場
        if target_time.weekday() in [5, 6]:  # Saturday=5, Sunday=6
            return MarketStatus.CLOSED
        
        current_time = target_time.time()
        
        # 営業時間前
        if current_time < self.pre_market_start:
            return MarketStatus.CLOSED
        
        # プリマーケット
        if self.pre_market_start <= current_time < self.market_open_morning:
            return MarketStatus.PRE_MARKET
        
        # 前場
        if self.market_open_morning <= current_time < self.market_close_morning:
            return MarketStatus.OPEN
        
        # 昼休み
        if self.market_close_morning <= current_time < self.market_open_afternoon:
            return MarketStatus.LUNCH_BREAK
        
        # 後場
        if self.market_open_afternoon <= current_time < self.market_close_afternoon:
            return MarketStatus.OPEN
        
        # アフターマーケット
        if self.market_close_afternoon <= current_time < self.post_market_end:
            return MarketStatus.POST_MARKET
        
        # 営業時間外
        return MarketStatus.CLOSED
    
    def is_market_open(self, target_time: Optional[datetime] = None) -> bool:
        """市場がオープンしているかチェック"""
        return self.get_market_status(target_time) == MarketStatus.OPEN
    
    def should_auto_update(self, target_time: Optional[datetime] = None) -> bool:
        """自動更新が必要かチェック"""
        status = self.get_market_status(target_time)
        
        # 市場オープン中または昼休み中は自動更新
        if status in [MarketStatus.OPEN, MarketStatus.LUNCH_BREAK]:
            return True
        
        # プリマーケット・アフターマーケットは低頻度更新
        if status in [MarketStatus.PRE_MARKET, MarketStatus.POST_MARKET]:
            return True
        
        # 完全クローズ時は更新不要
        return False
    
    def get_update_interval(self, target_time: Optional[datetime] = None) -> int:
        """更新間隔（秒）を取得"""
        status = self.get_market_status(target_time)
        
        if status == MarketStatus.OPEN:
            return 30  # 30秒間隔（活発な取引時間）
        elif status == MarketStatus.LUNCH_BREAK:
            return 300  # 5分間隔（昼休み）
        elif status in [MarketStatus.PRE_MARKET, MarketStatus.POST_MARKET]:
            return 600  # 10分間隔（プリ・アフター）
        else:
            return 3600  # 1時間間隔（クローズ時）
    
    def get_market_status_message(self, target_time: Optional[datetime] = None) -> str:
        """市場状態メッセージを取得"""
        status = self.get_market_status(target_time)
        
        messages = {
            MarketStatus.CLOSED: "🔴 市場クローズ",
            MarketStatus.OPEN: "🟢 市場オープン",
            MarketStatus.PRE_MARKET: "🟡 プリマーケット",
            MarketStatus.LUNCH_BREAK: "🟠 昼休み",
            MarketStatus.POST_MARKET: "🟡 アフターマーケット"
        }
        
        return messages.get(status, "❓ 不明")


# グローバルインスタンス
_market_time_manager = None


def get_market_manager() -> MarketTimeManager:
    """グローバル市場時間管理システムを取得"""
    global _market_time_manager
    if _market_time_manager is None:
        _market_time_manager = MarketTimeManager()
    return _market_time_manager


def is_market_open() -> bool:
    """市場がオープンしているかチェック"""
    return get_market_manager().is_market_open()


def should_auto_update() -> bool:
    """自動更新が必要かチェック"""
    return get_market_manager().should_auto_update()


def get_update_interval() -> int:
    """更新間隔を取得"""
    return get_market_manager().get_update_interval()


def get_market_status_message() -> str:
    """市場状態メッセージを取得"""
    return get_market_manager().get_market_status_message()


if __name__ == "__main__":
    manager = MarketTimeManager()
    print("🕐 市場時間管理システムテスト")
    print(f"現在の市場状態: {manager.get_market_status_message()}")
    print(f"自動更新: {manager.should_auto_update()}")
    print(f"更新間隔: {manager.get_update_interval()}秒")
