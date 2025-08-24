from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from day_trade.core.trade_manager import TradeType


class BacktestMode(Enum):
    """バックテストモード"""

    SINGLE_SYMBOL = "single_symbol"
    PORTFOLIO = "portfolio"
    STRATEGY_COMPARISON = "comparison"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    PARAMETER_OPTIMIZATION = "optimization"
    ML_ENSEMBLE = "ml_ensemble"


class OptimizationObjective(Enum):
    """最適化目標"""

    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"


@dataclass
class BacktestConfig:
    """バックテスト設定"""

    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    commission: Decimal = Decimal("0.001")
    slippage: Decimal = Decimal("0.001")
    max_position_size: Decimal = Decimal("0.2")
    rebalance_frequency: str = "monthly"

    walk_forward_window: int = 252
    walk_forward_step: int = 21
    min_training_samples: int = 100

    optimization_objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO
    max_optimization_iterations: int = 100
    optimization_method: str = "differential_evolution"

    monte_carlo_runs: int = 1000
    confidence_interval: float = 0.95

    max_drawdown_limit: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None

    enable_ml_features: bool = False
    retrain_frequency: int = 50
    ml_models_to_use: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": str(self.initial_capital),
            "commission": str(self.commission),
            "slippage": str(self.slippage),
            "max_position_size": str(self.max_position_size),
            "rebalance_frequency": self.rebalance_frequency,
            "walk_forward_window": self.walk_forward_window,
            "walk_forward_step": self.walk_forward_step,
            "min_training_samples": self.min_training_samples,
            "optimization_objective": self.optimization_objective.value,
            "max_optimization_iterations": self.max_optimization_iterations,
            "optimization_method": self.optimization_method,
            "monte_carlo_runs": self.monte_carlo_runs,
            "confidence_interval": self.confidence_interval,
            "max_drawdown_limit": self.max_drawdown_limit,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "enable_ml_features": self.enable_ml_features,
            "retrain_frequency": self.retrain_frequency,
            "ml_models_to_use": self.ml_models_to_use,
        }


@dataclass
class Trade:
    """取引記録"""

    timestamp: datetime
    symbol: str
    action: TradeType
    quantity: int
    price: Decimal
    commission: Decimal
    total_cost: Decimal


@dataclass
class Position:
    """ポジション情報"""

    symbol: str
    quantity: int
    average_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    weight: Decimal


@dataclass
class WalkForwardResult:
    """Walk-Forward分析結果"""

    period_start: datetime
    period_end: datetime
    training_start: datetime
    training_end: datetime
    out_of_sample_return: float
    in_sample_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    parameters_used: Dict[str, Any]


@dataclass
class OptimizationResult:
    """パラメータ最適化結果"""

    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    iterations: int
    convergence_achieved: bool
    backtest_result: "BacktestResult"


@dataclass
class MonteCarloResult:
    """モンテカルロシミュレーション結果"""

    mean_return: float
    std_return: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    percentiles: Dict[str, float]
    probability_of_loss: float
    var_95: float
    cvar_95: float
    simulation_runs: int


@dataclass
class BacktestResult:
    """バックテスト結果"""

    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    duration_days: int

    total_return: Decimal
    annualized_return: Decimal
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

    total_trades: int
    profitable_trades: int
    losing_trades: int
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: float

    trades: List[Trade]
    daily_returns: pd.Series
    portfolio_value: pd.Series
    positions_history: List[Dict[str, Position]]

    walk_forward_results: Optional[List[WalkForwardResult]] = None
    optimization_result: Optional[OptimizationResult] = None
    monte_carlo_result: Optional[MonteCarloResult] = None

    calmar_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    information_ratio: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "duration_days": self.duration_days,
            "total_return": str(self.total_return),
            "annualized_return": str(self.annualized_return),
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "profitable_trades": self.profitable_trades,
            "losing_trades": self.losing_trades,
            "avg_win": str(self.avg_win),
            "avg_loss": str(self.avg_loss),
            "profit_factor": self.profit_factor,
            "trades_count": len(self.trades),
            "portfolio_final_value": (
                str(self.portfolio_value.iloc[-1])
                if not self.portfolio_value.empty
                else "0"
            ),
        }
