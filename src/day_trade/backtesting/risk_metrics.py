#!/usr/bin/env python3
"""
リスクメトリクス計算機

Issue #323: 実データバックテスト機能開発
リスク・リターン指標の計算とリスク分析機能
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class RiskMetrics:
    """リスクメトリクス"""

    # リターン指標
    total_return: float
    annualized_return: float
    cumulative_return: float

    # リスク指標
    volatility: float
    downside_deviation: float
    maximum_drawdown: float
    maximum_drawdown_duration: int

    # リスク調整済みリターン指標
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float

    # VaR指標
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # 95% Conditional VaR

    # その他の指標
    skewness: float
    kurtosis: float
    win_rate: float
    profit_factor: float

    # ベータ・相関（ベンチマーク比較）
    beta: Optional[float] = None
    alpha: Optional[float] = None
    correlation: Optional[float] = None
    tracking_error: Optional[float] = None


class RiskMetricsCalculator:
    """リスクメトリクス計算機"""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        初期化

        Args:
            risk_free_rate: リスクフリーレート（年率）
        """
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(
        self,
        returns: List[float],
        portfolio_values: List[float],
        benchmark_returns: Optional[List[float]] = None,
    ) -> RiskMetrics:
        """
        総合的なリスクメトリクス計算

        Args:
            returns: 日次リターン
            portfolio_values: ポートフォリオ価値
            benchmark_returns: ベンチマークリターン

        Returns:
            リスクメトリクス
        """
        returns_array = np.array(returns)
        values_array = np.array(portfolio_values)

        # 基本リターン指標
        total_return = self._calculate_total_return(values_array)
        annualized_return = self._calculate_annualized_return(returns_array)
        cumulative_return = self._calculate_cumulative_return(returns_array)

        # リスク指標
        volatility = self._calculate_volatility(returns_array)
        downside_deviation = self._calculate_downside_deviation(returns_array)
        max_dd, max_dd_duration = self._calculate_maximum_drawdown(values_array)

        # リスク調整済みリターン指標
        sharpe_ratio = self._calculate_sharpe_ratio(returns_array)
        sortino_ratio = self._calculate_sortino_ratio(returns_array)
        calmar_ratio = self._calculate_calmar_ratio(annualized_return, max_dd)

        # VaR指標
        var_95 = self._calculate_var(returns_array, 0.05)
        var_99 = self._calculate_var(returns_array, 0.01)
        cvar_95 = self._calculate_cvar(returns_array, 0.05)

        # 分布の特徴
        skewness = self._calculate_skewness(returns_array)
        kurtosis = self._calculate_kurtosis(returns_array)

        # 取引統計
        win_rate = self._calculate_win_rate(returns_array)
        profit_factor = self._calculate_profit_factor(returns_array)

        # ベンチマーク比較（オプション）
        beta = alpha = correlation = tracking_error = information_ratio = None
        if benchmark_returns is not None:
            beta, alpha = self._calculate_beta_alpha(returns_array, np.array(benchmark_returns))
            correlation = self._calculate_correlation(returns_array, np.array(benchmark_returns))
            tracking_error = self._calculate_tracking_error(
                returns_array, np.array(benchmark_returns)
            )
            information_ratio = self._calculate_information_ratio(
                returns_array, np.array(benchmark_returns)
            )
        else:
            information_ratio = 0.0

        return RiskMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=cumulative_return,
            volatility=volatility,
            downside_deviation=downside_deviation,
            maximum_drawdown=max_dd,
            maximum_drawdown_duration=max_dd_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            win_rate=win_rate,
            profit_factor=profit_factor,
            beta=beta,
            alpha=alpha,
            correlation=correlation,
            tracking_error=tracking_error,
        )

    def _calculate_total_return(self, values: np.ndarray) -> float:
        """総リターン計算"""
        if len(values) < 2:
            return 0.0
        return (values[-1] - values[0]) / values[0]

    def _calculate_annualized_return(self, returns: np.ndarray) -> float:
        """年率リターン計算"""
        if len(returns) == 0:
            return 0.0

        # 複利計算
        cumulative_return = (1 + returns).prod() - 1
        trading_days = len(returns)
        years = trading_days / 252

        if years > 0 and (1 + cumulative_return) > 0:
            return (1 + cumulative_return) ** (1 / years) - 1
        return 0.0

    def _calculate_cumulative_return(self, returns: np.ndarray) -> float:
        """累積リターン計算"""
        if len(returns) == 0:
            return 0.0
        return (1 + returns).prod() - 1

    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """ボラティリティ計算（年率）"""
        if len(returns) < 2:
            return 0.0
        return np.std(returns) * np.sqrt(252)

    def _calculate_downside_deviation(self, returns: np.ndarray) -> float:
        """下方偏差計算（年率）"""
        if len(returns) < 2:
            return 0.0

        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 0.0

        return np.std(negative_returns) * np.sqrt(252)

    def _calculate_maximum_drawdown(self, values: np.ndarray) -> Tuple[float, int]:
        """最大ドローダウンと期間計算"""
        if len(values) < 2:
            return 0.0, 0

        # 累積最高値
        peak = np.maximum.accumulate(values)

        # ドローダウン計算
        drawdowns = (values - peak) / peak
        max_drawdown = np.min(drawdowns)

        # 最大ドローダウン期間計算
        max_dd_duration = 0
        current_duration = 0

        for dd in drawdowns:
            if dd < 0:
                current_duration += 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                current_duration = 0

        return max_drawdown, max_dd_duration

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """シャープレシオ計算"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / 252  # 日次リスクフリーレート
        return (
            np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            if np.std(excess_returns) > 0
            else 0.0
        )

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """ソルティノレシオ計算"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)

    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """カルマーレシオ計算"""
        if max_drawdown == 0:
            return 0.0
        return annualized_return / abs(max_drawdown)

    def _calculate_var(self, returns: np.ndarray, alpha: float) -> float:
        """VaR (Value at Risk) 計算"""
        if len(returns) < 10:
            return 0.0
        return np.percentile(returns, alpha * 100)

    def _calculate_cvar(self, returns: np.ndarray, alpha: float) -> float:
        """CVaR (Conditional Value at Risk) 計算"""
        if len(returns) < 10:
            return 0.0

        var = self._calculate_var(returns, alpha)
        tail_returns = returns[returns <= var]

        return np.mean(tail_returns) if len(tail_returns) > 0 else 0.0

    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """歪度計算"""
        if len(returns) < 3:
            return 0.0
        return stats.skew(returns)

    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """尖度計算"""
        if len(returns) < 4:
            return 0.0
        return stats.kurtosis(returns)

    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """勝率計算"""
        if len(returns) == 0:
            return 0.0
        positive_returns = returns[returns > 0]
        return len(positive_returns) / len(returns)

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """プロフィットファクター計算"""
        if len(returns) == 0:
            return 0.0

        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        return profits / losses if losses > 0 else float("inf") if profits > 0 else 0.0

    def _calculate_beta_alpha(
        self, returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> Tuple[float, float]:
        """ベータ・アルファ計算"""
        if len(returns) != len(benchmark_returns) or len(returns) < 10:
            return 0.0, 0.0

        # 超過リターン
        excess_returns = returns - self.risk_free_rate / 252
        excess_benchmark = benchmark_returns - self.risk_free_rate / 252

        # ベータ計算
        covariance = np.cov(excess_returns, excess_benchmark)[0, 1]
        benchmark_variance = np.var(excess_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0

        # アルファ計算（年率）
        portfolio_return = np.mean(excess_returns) * 252
        benchmark_return = np.mean(excess_benchmark) * 252
        alpha = portfolio_return - beta * benchmark_return

        return beta, alpha

    def _calculate_correlation(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """相関係数計算"""
        if len(returns) != len(benchmark_returns) or len(returns) < 10:
            return 0.0
        return np.corrcoef(returns, benchmark_returns)[0, 1]

    def _calculate_tracking_error(
        self, returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> float:
        """トラッキングエラー計算"""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0

        active_returns = returns - benchmark_returns
        return np.std(active_returns) * np.sqrt(252)

    def _calculate_information_ratio(
        self, returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> float:
        """情報比率計算"""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0

        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(252)

        if tracking_error == 0:
            return 0.0

        active_return = np.mean(active_returns) * 252
        return active_return / tracking_error

    def generate_risk_report(self, metrics: RiskMetrics, benchmark_name: str = None) -> str:
        """リスク分析レポート生成"""
        report = []
        report.append("=" * 60)
        report.append("リスク・リターン分析レポート")
        report.append("=" * 60)

        # リターン分析
        report.append("\n【リターン分析】")
        report.append(f"  総リターン: {metrics.total_return:.2%}")
        report.append(f"  年率リターン: {metrics.annualized_return:.2%}")
        report.append(f"  累積リターン: {metrics.cumulative_return:.2%}")

        # リスク分析
        report.append("\n【リスク分析】")
        report.append(f"  年率ボラティリティ: {metrics.volatility:.2%}")
        report.append(f"  下方偏差: {metrics.downside_deviation:.2%}")
        report.append(f"  最大ドローダウン: {metrics.maximum_drawdown:.2%}")
        report.append(f"  最大DD期間: {metrics.maximum_drawdown_duration}日")

        # リスク調整済みリターン
        report.append("\n【リスク調整済みリターン】")
        report.append(f"  シャープレシオ: {metrics.sharpe_ratio:.3f}")
        report.append(f"  ソルティノレシオ: {metrics.sortino_ratio:.3f}")
        report.append(f"  カルマーレシオ: {metrics.calmar_ratio:.3f}")
        if metrics.information_ratio is not None:
            report.append(f"  情報比率: {metrics.information_ratio:.3f}")

        # VaR分析
        report.append("\n【VaR分析】")
        report.append(f"  95% VaR: {metrics.var_95:.2%}")
        report.append(f"  99% VaR: {metrics.var_99:.2%}")
        report.append(f"  95% CVaR: {metrics.cvar_95:.2%}")

        # 分布特性
        report.append("\n【分布特性】")
        report.append(f"  歪度: {metrics.skewness:.3f}")
        report.append(f"  尖度: {metrics.kurtosis:.3f}")

        # 取引統計
        report.append("\n【取引統計】")
        report.append(f"  勝率: {metrics.win_rate:.2%}")
        report.append(f"  プロフィットファクター: {metrics.profit_factor:.2f}")

        # ベンチマーク比較（利用可能な場合）
        if metrics.beta is not None and benchmark_name:
            report.append(f"\n【{benchmark_name}比較】")
            report.append(f"  ベータ: {metrics.beta:.3f}")
            report.append(f"  アルファ: {metrics.alpha:.2%}")
            report.append(f"  相関係数: {metrics.correlation:.3f}")
            report.append(f"  トラッキングエラー: {metrics.tracking_error:.2%}")

        # リスク評価
        report.append("\n【総合リスク評価】")
        risk_level = self._assess_risk_level(metrics)
        report.append(f"  リスクレベル: {risk_level}")

        return "\n".join(report)

    def _assess_risk_level(self, metrics: RiskMetrics) -> str:
        """総合リスクレベル評価"""
        risk_factors = []

        # ボラティリティ評価
        if metrics.volatility > 0.30:
            risk_factors.append("高ボラティリティ")
        elif metrics.volatility > 0.20:
            risk_factors.append("中程度ボラティリティ")

        # ドローダウン評価
        if abs(metrics.maximum_drawdown) > 0.30:
            risk_factors.append("高ドローダウン")
        elif abs(metrics.maximum_drawdown) > 0.20:
            risk_factors.append("中程度ドローダウン")

        # シャープレシオ評価
        if metrics.sharpe_ratio < 0.5:
            risk_factors.append("低リスク調整後リターン")

        # VaR評価
        if abs(metrics.var_95) > 0.05:
            risk_factors.append("高VaR")

        if len(risk_factors) >= 3:
            return "高リスク"
        elif len(risk_factors) >= 1:
            return "中リスク"
        else:
            return "低リスク"


if __name__ == "__main__":
    # テスト実行
    print("リスクメトリクス計算機テスト")
    print("=" * 50)

    # サンプルデータ生成
    np.random.seed(42)
    n_days = 252
    returns = np.random.normal(0.0008, 0.015, n_days)  # 年率20%リターン、15%ボラティリティ
    portfolio_values = [1000000]

    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    # ベンチマーク（市場平均）
    benchmark_returns = np.random.normal(0.0005, 0.012, n_days)

    calculator = RiskMetricsCalculator()
    metrics = calculator.calculate_metrics(
        returns.tolist(), portfolio_values, benchmark_returns.tolist()
    )

    report = calculator.generate_risk_report(metrics, "TOPIX")
    print(report)
