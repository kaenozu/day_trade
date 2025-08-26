import logging

import numpy as np
import pandas as pd

from .datastructures import RiskMetrics
from .enums import MarketDirection


class RiskManager:
    """リスク管理システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_risk_metrics(self, data: pd.DataFrame, returns: pd.Series = None) -> RiskMetrics:
        """リスク指標計算"""
        if returns is None and 'Close' in data.columns:
            returns = data['Close'].pct_change().dropna()

        if returns is None or len(returns) < 20:
            return self._default_risk_metrics()

        try:
            # ボラティリティ（年率）
            volatility = returns.std() * np.sqrt(252)

            # VaR 95%
            var_95 = returns.quantile(0.05)

            # 期待ショートフォール
            expected_shortfall = returns[returns <= var_95].mean()

            # 最大ドローダウン
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            maximum_drawdown = drawdown.min()

            # シャープレシオ（リスクフリーレート2%と仮定）
            excess_returns = returns.mean() - 0.02/252
            sharpe_ratio = excess_returns / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

            # ベータ値（模擬的に計算）
            beta = np.random.uniform(0.8, 1.2)

            # 市場相関（模擬的に計算）
            correlation_market = np.random.uniform(0.6, 0.9)

            return RiskMetrics(
                volatility=volatility,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                maximum_drawdown=maximum_drawdown,
                sharpe_ratio=sharpe_ratio,
                beta=beta,
                correlation_market=correlation_market
            )

        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return self._default_risk_metrics()

    def _default_risk_metrics(self) -> RiskMetrics:
        """デフォルトリスク指標"""
        return RiskMetrics(
            volatility=0.25,
            var_95=-0.03,
            expected_shortfall=-0.05,
            maximum_drawdown=-0.15,
            sharpe_ratio=0.5,
            beta=1.0,
            correlation_market=0.7
        )

    def calculate_position_size(self, account_balance: float, risk_tolerance: float,
                              volatility: float, max_loss_percent: float = 2.0) -> float:
        """ポジションサイズ計算"""
        try:
            # リスク許容度に基づく基本ポジションサイズ
            base_position_size = account_balance * risk_tolerance

            # ボラティリティ調整
            volatility_adjusted_size = base_position_size / (volatility + 0.1)

            # 最大損失制限
            max_loss_amount = account_balance * (max_loss_percent / 100)
            max_allowed_position = max_loss_amount / (volatility * 2)  # 2σ想定

            # 最小値を採用
            position_size = min(volatility_adjusted_size, max_allowed_position)

            # ポートフォリオ比率として返す
            return min(position_size / account_balance, 0.1)  # 最大10%制限

        except Exception as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return 0.02  # デフォルト2%

    def calculate_stop_loss(self, entry_price: float, direction: MarketDirection,
                          volatility: float, risk_multiplier: float = 2.0) -> float:
        """損切り価格計算"""
        try:
            # ATR的な概念でボラティリティベースの損切り設定
            volatility_based_stop = entry_price * volatility * risk_multiplier

            if direction in [MarketDirection.BULLISH, MarketDirection.STRONG_BULLISH]:
                # 買いポジション
                stop_loss = entry_price - volatility_based_stop
            else:
                # 売りポジション
                stop_loss = entry_price + volatility_based_stop

            return max(stop_loss, 0)  # 負の価格防止

        except Exception as e:
            self.logger.error(f"Stop loss calculation failed: {e}")
            return entry_price * 0.95  # デフォルト5%下

    def calculate_target_price(self, entry_price: float, direction: MarketDirection,
                             expected_return: float, risk_reward_ratio: float = 2.0) -> float:
        """目標価格計算"""
        try:
            # リスクリワード比を考慮した目標設定
            if direction in [MarketDirection.BULLISH, MarketDirection.STRONG_BULLISH]:
                # 買いポジション
                target_return = abs(expected_return) * risk_reward_ratio
                target_price = entry_price * (1 + target_return)
            else:
                # 売りポジション
                target_return = abs(expected_return) * risk_reward_ratio
                target_price = entry_price * (1 - target_return)

            return max(target_price, 0)

        except Exception as e:
            self.logger.error(f"Target price calculation failed: {e}")
            return entry_price * 1.05  # デフォルト5%上
