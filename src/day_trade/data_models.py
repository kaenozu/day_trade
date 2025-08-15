#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Models for Day Trading
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class RiskLevel(Enum):
    """リスクレベル"""
    VERY_LOW = "超低リスク"
    LOW = "低リスク"
    MEDIUM = "中リスク"
    HIGH = "高リスク"
    VERY_HIGH = "超高リスク"

class PositionStatus(Enum):
    """ポジション状態"""
    OPEN = "建玉中"
    CLOSED = "決済済"
    STOP_LOSS = "損切り"
    TAKE_PROFIT = "利確"
    TIME_STOP = "時間切れ決済"

class AlertLevel(Enum):
    """アラートレベル"""
    INFO = "情報"
    WARNING = "警告"
    CRITICAL = "緊急"

@dataclass
class Position:
    """ポジション情報"""
    symbol: str
    name: str
    entry_price: float
    quantity: int
    entry_time: datetime
    stop_loss: float
    take_profit: float
    current_price: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    pnl: float = 0.0
    pnl_percent: float = 0.0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    max_holding_time: int = 240  # 分（4時間）

    def update_current_price(self, price: float):
        """現在価格更新とPnL計算"""
        self.current_price = price
        self.pnl = (price - self.entry_price) * self.quantity
        if self.entry_price > 0:
            self.pnl_percent = ((price - self.entry_price) / self.entry_price) * 100

    @property
    def holding_minutes(self) -> int:
        """保有時間（分）"""
        return int((datetime.now() - self.entry_time).total_seconds() / 60)

    @property
    def is_profitable(self) -> bool:
        """含み益判定"""
        return self.pnl > 0

    @property
    def should_stop_loss(self) -> bool:
        """損切り判定"""
        return self.current_price <= self.stop_loss

    @property
    def should_take_profit(self) -> bool:
        """利確判定"""
        return self.current_price >= self.take_profit

    @property
    def should_time_stop(self) -> bool:
        """時間切れ判定"""
        return self.holding_minutes >= self.max_holding_time

@dataclass
class RiskAlert:
    """リスクアラート"""
    timestamp: datetime
    symbol: str
    level: AlertLevel
    message: str
    current_price: float = 0.0
    recommended_action: str = ""
