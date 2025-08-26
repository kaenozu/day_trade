"""
高度なバックテストエンジン - データ構造とタイプ定義

バックテストシステムで使用する主要なデータ構造を定義。
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from day_trade.analysis.events import TradeRecord

warnings.filterwarnings("ignore")


@dataclass
class TradingCosts:
    """取引コスト設定"""

    commission_rate: float = 0.001  # 手数料率 (0.1%)
    min_commission: float = 0.0  # 最小手数料
    max_commission: float = float("inf")  # 最大手数料
    bid_ask_spread_rate: float = 0.001  # ビッドアスクスプレッド率
    slippage_rate: float = 0.0005  # スリッページ率
    market_impact_rate: float = 0.0002  # マーケットインパクト率


@dataclass
class Position:
    """ポジション情報"""

    symbol: str
    quantity: int = 0
    average_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""

    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar_ratio: float = 0.0

    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_duration: float = 0.0

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    total_commission: float = 0.0
    total_slippage: float = 0.0


class PerformanceAnalyzer:
    """パフォーマンス分析ユーティリティクラス"""

    @staticmethod
    def calculate_basic_metrics(
        equity_curve: List[float], daily_returns: List[float]
    ) -> Dict[str, float]:
        """基本的なパフォーマンス指標を計算"""
        if not equity_curve or len(equity_curve) < 2:
            return {}

        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

        if len(daily_returns) > 0:
            daily_mean = np.mean(daily_returns)
            annual_return = (1 + daily_mean) ** 252 - 1
            volatility = np.std(daily_returns) * np.sqrt(252)
        else:
            annual_return = 0.0
            volatility = 0.0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
        }

    @staticmethod
    def calculate_drawdown_series(equity_curve: List[float]) -> List[float]:
        """ドローダウン系列を計算"""
        if not equity_curve:
            return []

        drawdown_series = []
        peak = equity_curve[0]

        for value in equity_curve:
            peak = max(peak, value)
            drawdown = (value - peak) / peak
            drawdown_series.append(drawdown)

        return drawdown_series

    @staticmethod
    def calculate_trade_metrics(trade_history: List[TradeRecord]) -> Dict[str, float]:
        """取引関連指標を計算"""
        if not trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "total_commission": 0.0,
                "total_slippage": 0.0,
            }

        profits = []
        losses = []

        for trade in trade_history:
            if trade.is_closed:
                pnl = trade.realized_pnl
                if pnl > 0:
                    profits.append(pnl)
                else:
                    losses.append(abs(pnl))

        total_trades = len(trade_history)
        winning_trades = len(profits)
        losing_trades = len(losses)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean(profits) if profits else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        profit_factor = (
            sum(profits) / sum(losses)
            if losses and sum(losses) > 0
            else (float("inf") if profits else 0.0)
        )

        total_commission = sum(trade.total_commission for trade in trade_history)
        total_slippage = sum(trade.total_slippage for trade in trade_history)

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_commission": total_commission,
            "total_slippage": total_slippage,
        }