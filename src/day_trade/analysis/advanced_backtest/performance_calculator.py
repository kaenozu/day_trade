"""
高度なバックテストエンジン - パフォーマンス計算

バックテスト結果のパフォーマンス指標を計算。
"""

import warnings
from typing import Dict, List

import numpy as np

from day_trade.analysis.events import TradeRecord
from day_trade.utils.logging_config import get_context_logger
from .data_structures import PerformanceMetrics, PerformanceAnalyzer

warnings.filterwarnings("ignore")
logger = get_context_logger(__name__)


class PerformanceCalculator:
    """パフォーマンス計算システム"""

    def __init__(self, risk_free_rate: float = 0.01):
        """パフォーマンス計算システムの初期化"""
        self.risk_free_rate = risk_free_rate
        self.analyzer = PerformanceAnalyzer()

    def calculate_performance_metrics(
        self,
        equity_curve: List[float],
        daily_returns: List[float],
        trade_history: List[TradeRecord],
        drawdown_series: List[float] = None,
    ) -> PerformanceMetrics:
        """包括的なパフォーマンス指標を計算"""
        if not equity_curve or len(equity_curve) < 2:
            return PerformanceMetrics()

        # 基本指標を計算
        basic_metrics = self.analyzer.calculate_basic_metrics(equity_curve, daily_returns)
        
        # リスク調整指標を計算
        risk_metrics = self._calculate_risk_adjusted_metrics(
            basic_metrics.get("annual_return", 0.0),
            basic_metrics.get("volatility", 0.0),
            daily_returns
        )

        # ドローダウン分析
        if drawdown_series is None:
            drawdown_series = self.analyzer.calculate_drawdown_series(equity_curve)
        
        drawdown_metrics = self._calculate_drawdown_metrics(drawdown_series)

        # 取引分析
        trade_metrics = self.analyzer.calculate_trade_metrics(trade_history)

        # 全指標を統合
        return PerformanceMetrics(
            total_return=basic_metrics.get("total_return", 0.0),
            annual_return=basic_metrics.get("annual_return", 0.0),
            volatility=basic_metrics.get("volatility", 0.0),
            sharpe_ratio=risk_metrics.get("sharpe_ratio", 0.0),
            sortino_ratio=risk_metrics.get("sortino_ratio", 0.0),
            max_drawdown=drawdown_metrics.get("max_drawdown", 0.0),
            max_drawdown_duration=drawdown_metrics.get("max_drawdown_duration", 0),
            calmar_ratio=risk_metrics.get("calmar_ratio", 0.0),
            win_rate=trade_metrics.get("win_rate", 0.0),
            profit_factor=trade_metrics.get("profit_factor", 0.0),
            avg_win=trade_metrics.get("avg_win", 0.0),
            avg_loss=trade_metrics.get("avg_loss", 0.0),
            total_trades=trade_metrics.get("total_trades", 0),
            winning_trades=trade_metrics.get("winning_trades", 0),
            losing_trades=trade_metrics.get("losing_trades", 0),
            total_commission=trade_metrics.get("total_commission", 0.0),
            total_slippage=trade_metrics.get("total_slippage", 0.0),
        )

    def _calculate_risk_adjusted_metrics(
        self, annual_return: float, volatility: float, daily_returns: List[float]
    ) -> Dict[str, float]:
        """リスク調整指標の計算"""
        metrics = {}

        # シャープレシオ
        metrics["sharpe_ratio"] = self._calculate_sharpe_ratio(annual_return, volatility)

        # ソルティーノレシオ
        metrics["sortino_ratio"] = self._calculate_sortino_ratio(annual_return, daily_returns)

        return metrics

    def _calculate_sharpe_ratio(self, annual_return: float, volatility: float) -> float:
        """シャープレシオの計算"""
        if volatility <= 0:
            return 0.0
        return (annual_return - self.risk_free_rate) / volatility

    def _calculate_sortino_ratio(self, annual_return: float, daily_returns: List[float]) -> float:
        """ソルティーノレシオの計算"""
        if not daily_returns:
            return 0.0

        negative_returns = [r for r in daily_returns if r < 0]
        if not negative_returns:
            return 0.0

        downside_deviation = np.std(negative_returns) * np.sqrt(252)
        
        if downside_deviation <= 1e-10:
            return 0.0

        return (annual_return - self.risk_free_rate) / downside_deviation

    def _calculate_drawdown_metrics(self, drawdown_series: List[float]) -> Dict[str, float]:
        """ドローダウン関連指標の計算"""
        if not drawdown_series:
            return {"max_drawdown": 0.0, "max_drawdown_duration": 0}

        max_drawdown = min(drawdown_series)
        
        # 最大ドローダウン期間の計算
        max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown_series)

        # カルマーレシオ（別途計算する場合）
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
        }

    def _calculate_max_drawdown_duration(self, drawdown_series: List[float]) -> int:
        """最大ドローダウン期間の計算"""
        if not drawdown_series:
            return 0

        max_duration = 0
        current_duration = 0
        
        for drawdown in drawdown_series:
            if drawdown < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """カルマーレシオの計算"""
        if max_drawdown == 0 or abs(max_drawdown) <= 1e-10:
            return 0.0
        return annual_return / abs(max_drawdown)

    def calculate_information_ratio(
        self, portfolio_returns: List[float], benchmark_returns: List[float]
    ) -> float:
        """情報レシオの計算"""
        if len(portfolio_returns) != len(benchmark_returns) or not portfolio_returns:
            return 0.0

        excess_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
        
        if not excess_returns:
            return 0.0

        excess_mean = np.mean(excess_returns)
        excess_std = np.std(excess_returns)

        if excess_std <= 1e-10:
            return 0.0

        return excess_mean / excess_std * np.sqrt(252)

    def calculate_maximum_adverse_excursion(
        self, trade_history: List[TradeRecord]
    ) -> Dict[str, float]:
        """最大逆行幅（MAE）の計算"""
        if not trade_history:
            return {"avg_mae": 0.0, "max_mae": 0.0}

        maes = []
        for trade in trade_history:
            if trade.is_closed and trade.entry_price > 0:
                # 簡易的なMAE計算（実際にはtick-by-tickデータが必要）
                mae = abs(trade.entry_price - trade.exit_price) / trade.entry_price
                maes.append(mae)

        if not maes:
            return {"avg_mae": 0.0, "max_mae": 0.0}

        return {
            "avg_mae": np.mean(maes),
            "max_mae": max(maes),
        }

    def calculate_maximum_favorable_excursion(
        self, trade_history: List[TradeRecord]
    ) -> Dict[str, float]:
        """最大順行幅（MFE）の計算"""
        if not trade_history:
            return {"avg_mfe": 0.0, "max_mfe": 0.0}

        mfes = []
        for trade in trade_history:
            if trade.is_closed and trade.realized_pnl > 0:
                # 簡易的なMFE計算
                mfe = trade.realized_pnl / (trade.entry_price * trade.entry_quantity)
                mfes.append(mfe)

        if not mfes:
            return {"avg_mfe": 0.0, "max_mfe": 0.0}

        return {
            "avg_mfe": np.mean(mfes),
            "max_mfe": max(mfes),
        }

    def calculate_var_cvar(
        self, daily_returns: List[float], confidence_level: float = 0.05
    ) -> Dict[str, float]:
        """VaR（Value at Risk）とCVaR（Conditional VaR）の計算"""
        if not daily_returns:
            return {"var": 0.0, "cvar": 0.0}

        returns_array = np.array(daily_returns)
        
        # VaR計算
        var = np.percentile(returns_array, confidence_level * 100)
        
        # CVaR計算（VaR以下のリターンの平均）
        tail_returns = returns_array[returns_array <= var]
        cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var

        return {
            "var": var,
            "cvar": cvar,
        }

    def create_performance_summary(self, metrics: PerformanceMetrics) -> Dict[str, str]:
        """パフォーマンスサマリの作成"""
        return {
            "総リターン": f"{metrics.total_return:.2%}",
            "年率リターン": f"{metrics.annual_return:.2%}",
            "ボラティリティ": f"{metrics.volatility:.2%}",
            "シャープレシオ": f"{metrics.sharpe_ratio:.3f}",
            "ソルティーノレシオ": f"{metrics.sortino_ratio:.3f}",
            "最大ドローダウン": f"{metrics.max_drawdown:.2%}",
            "カルマーレシオ": f"{metrics.calmar_ratio:.3f}",
            "勝率": f"{metrics.win_rate:.2%}",
            "プロフィットファクター": f"{metrics.profit_factor:.2f}",
            "総取引数": f"{metrics.total_trades}",
            "総手数料": f"¥{metrics.total_commission:,.0f}",
            "総スリッページ": f"¥{metrics.total_slippage:,.0f}",
        }