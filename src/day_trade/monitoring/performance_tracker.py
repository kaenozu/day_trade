#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Tracker - 包括的パフォーマンス追跡システム
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import statistics
import uuid
import numpy as np

from sqlalchemy import Column, Integer, String, Float, DateTime, select, func, case
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# データベースモデル
Base = declarative_base()

class DBTrade(Base):
    """トレード記録テーブル"""
    __tablename__ = 'trades'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    quantity = Column(Integer, nullable=False)
    entry_date = Column(DateTime, nullable=False)
    exit_date = Column(DateTime)
    profit_loss = Column(Float)
    trade_type = Column(String, nullable=False)  # 'BUY' or 'SELL'
    trade_result = Column(String)  # 'PROFIT', 'LOSS', 'BREAKEVEN'

# Enums
class TradeType:
    BUY = "BUY"
    SELL = "SELL"

class TradeResult:
    PROFIT = "PROFIT"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"

class PerformanceTracker:
    # ... (__init__ and other methods as before) ...

    async def calculate_performance_metrics(self, days_back: int = 30) -> Dict[str, Any]:
        period_start = datetime.now() - timedelta(days=days_back)
        async with self.AsyncSession() as session:
            stmt = select(DBTrade).where(DBTrade.exit_date >= period_start).order_by(DBTrade.exit_date)
            trades = (await session.execute(stmt)).scalars().all()

        if not trades:
            return {"error": "No closed trades found for the period."}

        returns_pct = [t.profit_loss_pct for t in trades if t.profit_loss_pct is not None]
        if not returns_pct:
            return {"error": "No trades with returns found for the period."}

        # PnL and returns
        pnl_values = [t.profit_loss for t in trades if t.profit_loss is not None]
        total_pnl = sum(pnl_values)
        daily_returns = pd.Series(pnl_values, index=[t.exit_date for t in trades]).resample('D').sum()

        # Metrics calculation
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.profit_loss > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        avg_win = statistics.mean([t.profit_loss for t in trades if t.profit_loss > 0]) if winning_trades > 0 else 0
        avg_loss = statistics.mean([t.profit_loss for t in trades if t.profit_loss < 0]) if total_trades > winning_trades else 0
        profit_factor = sum([t.profit_loss for t in trades if t.profit_loss > 0]) / abs(sum([t.profit_loss for t in trades if t.profit_loss < 0])) if avg_loss != 0 else float('inf')

        # Risk & Return metrics
        volatility = daily_returns.std() * np.sqrt(365) # Annualized
        sharpe_ratio = (daily_returns.mean() * 365) / volatility if volatility > 0 else 0

        # Drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "avg_win_pnl": avg_win,
            "avg_loss_pnl": avg_loss,
            "volatility_annualized": volatility,
        }

    # ... (other methods) ...