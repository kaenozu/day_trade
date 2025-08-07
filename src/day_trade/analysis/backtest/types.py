"""
バックテスト関連の型定義とデータクラス
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.day_trade.core.trade_manager import TradeType


class BacktestMode(Enum):
    """バックテストモード"""

    SINGLE_SYMBOL = "single_symbol"  # 単一銘柄
    PORTFOLIO = "portfolio"  # ポートフォリオ
    STRATEGY_COMPARISON = "comparison"  # 戦略比較
    WALK_FORWARD = "walk_forward"  # Walk-Forward分析
    MONTE_CARLO = "monte_carlo"  # モンテカルロシミュレーション
    PARAMETER_OPTIMIZATION = "optimization"  # パラメータ最適化
    ML_ENSEMBLE = "ml_ensemble"  # 機械学習アンサンブル


class OptimizationObjective(Enum):
    """最適化目標"""

    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"  # リターン/最大ドローダウン
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"


@dataclass
class BacktestConfig:
    """バックテスト設定"""

    start_date: datetime
    end_date: datetime
    initial_capital: Decimal  # 初期資金
    commission: Decimal = Decimal("0.001")  # 手数料率（0.1%）
    slippage: Decimal = Decimal("0.001")  # スリッページ（0.1%）
    max_position_size: Decimal = Decimal("0.2")  # 最大ポジションサイズ（20%）
    rebalance_frequency: str = "monthly"  # リバランス頻度

    # Walk-Forward分析設定
    walk_forward_window: int = 252  # 訓練期間（日数）
    walk_forward_step: int = 21  # ステップサイズ（日数）
    min_training_samples: int = 100  # 最小訓練サンプル数

    # 最適化設定
    optimization_objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO
    max_optimization_iterations: int = 100
    optimization_method: str = "differential_evolution"  # scipy.optimize手法

    # モンテカルロ設定
    monte_carlo_runs: int = 1000
    confidence_interval: float = 0.95

    # リスク管理設定
    max_drawdown_limit: Optional[float] = None  # 最大許容ドローダウン
    stop_loss_pct: Optional[float] = None  # ストップロス率
    take_profit_pct: Optional[float] = None  # テイクプロフィット率

    # 機械学習設定
    enable_ml_features: bool = False
    retrain_frequency: int = 50  # ML再訓練頻度（日数）
    ml_models_to_use: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
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
    action: TradeType  # BUY or SELL
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
    weight: Decimal  # ポートフォリオ内での重み


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
    confidence_intervals: Dict[str, Tuple[float, float]]  # 信頼区間
    percentiles: Dict[str, float]  # パーセンタイル
    probability_of_loss: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%


@dataclass
class BacktestResult:
    """バックテスト結果"""

    config: BacktestConfig
    start_date: datetime
    end_date: datetime

    # 基本パフォーマンス指標
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # 取引統計
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float

    # リスク指標
    value_at_risk: float
    conditional_var: float

    # 詳細データ
    trades: List[Trade]
    positions: List[Position]
    daily_returns: List[float]
    portfolio_value_history: List[Decimal]
    drawdown_history: List[float]

    # オプションフィールドは最後に配置
    beta: Optional[float] = None
    alpha: Optional[float] = None

    # Walk-Forward結果（該当する場合）
    walk_forward_results: Optional[List[WalkForwardResult]] = None

    # 最適化結果（該当する場合）
    optimization_result: Optional[OptimizationResult] = None

    # モンテカルロ結果（該当する場合）
    monte_carlo_result: Optional[MonteCarloResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "config": self.config.to_dict(),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "value_at_risk": self.value_at_risk,
            "conditional_var": self.conditional_var,
            "beta": self.beta,
            "alpha": self.alpha,
            "daily_returns": self.daily_returns,
            "portfolio_value_history": [str(v) for v in self.portfolio_value_history],
            "drawdown_history": self.drawdown_history,
        }
