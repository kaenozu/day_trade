#!/usr/bin/env python3
"""
高度パフォーマンス指標とリスク分析

Issue #753対応: バックテスト機能強化
包括的なリスク指標、パフォーマンス分析、統計指標を提供
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


@dataclass
class AdvancedRiskMetrics:
    """高度リスク指標"""

    # Value at Risk (VaR)
    var_1: float  # 1日VaR (95%)
    var_5: float  # 5日VaR (95%)
    var_10: float # 10日VaR (95%)

    # Expected Shortfall (Conditional VaR)
    cvar_1: float  # 1日CVaR (95%)
    cvar_5: float  # 5日CVaR (95%)
    cvar_10: float # 10日CVaR (95%)

    # ドローダウン分析
    max_drawdown: float
    max_drawdown_duration: int  # 日数
    average_drawdown: float
    recovery_factor: float  # リターン/最大ドローダウン

    # テール分析
    skewness: float  # 歪度
    excess_kurtosis: float  # 超過尖度
    jarque_bera_stat: float  # 正規性検定統計量
    jarque_bera_pvalue: float

    # ダウンサイドリスク
    downside_deviation: float
    sortino_ratio: float
    pain_index: float  # 平均ドローダウン
    ulcer_index: float  # ドローダウンの二乗平均平方根


@dataclass
class AdvancedReturnMetrics:
    """高度リターン指標"""

    # リターン分析
    total_return: float
    annualized_return: float
    geometric_mean_return: float
    arithmetic_mean_return: float

    # リスク調整後リターン
    sharpe_ratio: float
    information_ratio: float
    calmar_ratio: float  # 年率リターン/最大ドローダウン
    sterling_ratio: float  # 年率リターン/平均最大ドローダウン

    # 勝率・損益分析
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float  # 総利益/総損失
    payoff_ratio: float  # 平均利益/平均損失
    expectancy: float  # 期待値

    # 取引効率
    trade_efficiency: float  # 有効な取引の割合
    maximum_consecutive_wins: int
    maximum_consecutive_losses: int


@dataclass
class MarketRegimeMetrics:
    """市場レジーム別分析"""

    bull_market_performance: Dict[str, float]
    bear_market_performance: Dict[str, float]
    sideways_market_performance: Dict[str, float]
    high_volatility_performance: Dict[str, float]
    low_volatility_performance: Dict[str, float]

    regime_detection_accuracy: float
    regime_transition_analysis: Dict[str, Any]


class AdvancedBacktestAnalyzer:
    """高度バックテスト分析エンジン"""

    def __init__(self, confidence_level: float = 0.95):
        """
        初期化

        Args:
            confidence_level: VaR/CVaR計算の信頼水準
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def calculate_advanced_risk_metrics(self,
                                      returns: pd.Series,
                                      portfolio_values: pd.Series) -> AdvancedRiskMetrics:
        """
        高度リスク指標の計算

        Args:
            returns: 日次リターン系列
            portfolio_values: ポートフォリオ価値系列

        Returns:
            AdvancedRiskMetrics: 高度リスク指標
        """
        # Value at Risk計算
        var_1 = self._calculate_var(returns, days=1)
        var_5 = self._calculate_var(returns, days=5)
        var_10 = self._calculate_var(returns, days=10)

        # Expected Shortfall計算
        cvar_1 = self._calculate_cvar(returns, days=1)
        cvar_5 = self._calculate_cvar(returns, days=5)
        cvar_10 = self._calculate_cvar(returns, days=10)

        # ドローダウン分析
        drawdowns = self._calculate_drawdowns(portfolio_values)
        max_dd = drawdowns.min()
        max_dd_duration = self._calculate_max_drawdown_duration(drawdowns)
        avg_dd = drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0
        recovery_factor = returns.mean() * 252 / abs(max_dd) if max_dd != 0 else float('inf')

        # 分布統計
        skewness = returns.skew()
        excess_kurtosis = returns.kurtosis()
        jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna()) if len(returns.dropna()) > 7 else (0, 1)

        # ダウンサイドリスク
        downside_dev = self._calculate_downside_deviation(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        pain_index = abs(drawdowns.mean())
        ulcer_index = np.sqrt((drawdowns ** 2).mean())

        return AdvancedRiskMetrics(
            var_1=var_1, var_5=var_5, var_10=var_10,
            cvar_1=cvar_1, cvar_5=cvar_5, cvar_10=cvar_10,
            max_drawdown=max_dd, max_drawdown_duration=max_dd_duration,
            average_drawdown=avg_dd, recovery_factor=recovery_factor,
            skewness=skewness, excess_kurtosis=excess_kurtosis,
            jarque_bera_stat=jb_stat, jarque_bera_pvalue=jb_pvalue,
            downside_deviation=downside_dev, sortino_ratio=sortino_ratio,
            pain_index=pain_index, ulcer_index=ulcer_index
        )

    def calculate_advanced_return_metrics(self,
                                        returns: pd.Series,
                                        trades: List[Any],
                                        benchmark_returns: Optional[pd.Series] = None) -> AdvancedReturnMetrics:
        """
        高度リターン指標の計算

        Args:
            returns: 日次リターン系列
            trades: 取引記録
            benchmark_returns: ベンチマークリターン（オプション）

        Returns:
            AdvancedReturnMetrics: 高度リターン指標
        """
        # 基本リターン
        total_ret = (1 + returns).prod() - 1
        annualized_ret = (1 + returns.mean()) ** 252 - 1
        geo_mean_ret = stats.gmean(1 + returns) - 1 if len(returns) > 0 else 0
        arith_mean_ret = returns.mean()

        # リスク調整後リターン
        sharpe = self._calculate_sharpe_ratio(returns)
        info_ratio = self._calculate_information_ratio(returns, benchmark_returns)

        # ドローダウンベースの比率
        max_dd = self._calculate_drawdowns(pd.Series((1 + returns).cumprod())).min()
        calmar = annualized_ret / abs(max_dd) if max_dd != 0 else float('inf')
        sterling = self._calculate_sterling_ratio(returns)

        # 取引分析
        if trades:
            trade_returns = [float(trade.total_cost) for trade in trades if hasattr(trade, 'total_cost')]
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]

            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')
            payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
            expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

            # 連続勝敗
            max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_trades(trade_returns)
            trade_efficiency = len([r for r in trade_returns if abs(r) > 0.001]) / len(trades) if trades else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = payoff_ratio = expectancy = 0
            max_consecutive_wins = max_consecutive_losses = trade_efficiency = 0

        return AdvancedReturnMetrics(
            total_return=total_ret, annualized_return=annualized_ret,
            geometric_mean_return=geo_mean_ret, arithmetic_mean_return=arith_mean_ret,
            sharpe_ratio=sharpe, information_ratio=info_ratio,
            calmar_ratio=calmar, sterling_ratio=sterling,
            win_rate=win_rate, average_win=avg_win, average_loss=avg_loss,
            profit_factor=profit_factor, payoff_ratio=payoff_ratio, expectancy=expectancy,
            trade_efficiency=trade_efficiency, maximum_consecutive_wins=max_consecutive_wins,
            maximum_consecutive_losses=max_consecutive_losses
        )

    def analyze_market_regimes(self,
                             returns: pd.Series,
                             market_data: pd.DataFrame) -> MarketRegimeMetrics:
        """
        市場レジーム別パフォーマンス分析

        Args:
            returns: 戦略リターン
            market_data: 市場データ（価格、ボラティリティ等）

        Returns:
            MarketRegimeMetrics: 市場レジーム別指標
        """
        # 市場レジーム分類
        regimes = self._classify_market_regimes(market_data)

        # レジーム別パフォーマンス計算
        bull_perf = self._calculate_regime_performance(returns, regimes == 'bull')
        bear_perf = self._calculate_regime_performance(returns, regimes == 'bear')
        sideways_perf = self._calculate_regime_performance(returns, regimes == 'sideways')
        high_vol_perf = self._calculate_regime_performance(returns, regimes == 'high_vol')
        low_vol_perf = self._calculate_regime_performance(returns, regimes == 'low_vol')

        # レジーム検出精度（簡易版）
        regime_accuracy = self._estimate_regime_detection_accuracy(market_data, regimes)

        # レジーム遷移分析
        transition_analysis = self._analyze_regime_transitions(regimes, returns)

        return MarketRegimeMetrics(
            bull_market_performance=bull_perf,
            bear_market_performance=bear_perf,
            sideways_market_performance=sideways_perf,
            high_volatility_performance=high_vol_perf,
            low_volatility_performance=low_vol_perf,
            regime_detection_accuracy=regime_accuracy,
            regime_transition_analysis=transition_analysis
        )

    def _calculate_var(self, returns: pd.Series, days: int = 1) -> float:
        """Value at Risk計算"""
        if len(returns) < 2:
            return 0.0

        # 複数日VaRの場合はスケーリング
        daily_var = np.percentile(returns.dropna(), self.alpha * 100)
        return daily_var * np.sqrt(days)

    def _calculate_cvar(self, returns: pd.Series, days: int = 1) -> float:
        """Conditional Value at Risk計算"""
        if len(returns) < 2:
            return 0.0

        var = self._calculate_var(returns, 1)  # 1日VaR
        tail_returns = returns[returns <= var]
        daily_cvar = tail_returns.mean() if len(tail_returns) > 0 else var
        return daily_cvar * np.sqrt(days)

    def _calculate_drawdowns(self, portfolio_values: pd.Series) -> pd.Series:
        """ドローダウン計算"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown

    def _calculate_max_drawdown_duration(self, drawdowns: pd.Series) -> int:
        """最大ドローダウン継続期間計算"""
        underwater = drawdowns < 0
        periods = []
        start = None

        for i, is_underwater in enumerate(underwater):
            if is_underwater and start is None:
                start = i
            elif not is_underwater and start is not None:
                periods.append(i - start)
                start = None

        if start is not None:  # 期間終了時もドローダウン中
            periods.append(len(underwater) - start)

        return max(periods) if periods else 0

    def _calculate_downside_deviation(self, returns: pd.Series, target: float = 0.0) -> float:
        """ダウンサイド偏差計算"""
        downside_returns = returns[returns < target]
        return np.sqrt((downside_returns ** 2).mean()) if len(downside_returns) > 0 else 0.0

    def _calculate_sortino_ratio(self, returns: pd.Series, target: float = 0.0) -> float:
        """ソルティーノレシオ計算"""
        excess_return = returns.mean() - target / 252  # 年率を日率に変換
        downside_dev = self._calculate_downside_deviation(returns, target / 252)
        return (excess_return * 252) / (downside_dev * np.sqrt(252)) if downside_dev != 0 else float('inf')

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """シャープレシオ計算"""
        excess_return = returns.mean() - risk_free_rate / 252
        return (excess_return * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0.0

    def _calculate_information_ratio(self, returns: pd.Series, benchmark_returns: Optional[pd.Series]) -> float:
        """インフォメーションレシオ計算"""
        if benchmark_returns is None or len(benchmark_returns) != len(returns):
            return 0.0

        active_returns = returns - benchmark_returns
        return (active_returns.mean() * 252) / (active_returns.std() * np.sqrt(252)) if active_returns.std() != 0 else 0.0

    def _calculate_sterling_ratio(self, returns: pd.Series) -> float:
        """スターリングレシオ計算"""
        # 簡易実装：年率リターン / 平均最大ドローダウン
        portfolio_values = (1 + returns).cumprod()
        drawdowns = self._calculate_drawdowns(portfolio_values)
        avg_max_dd = abs(drawdowns.min())  # 簡易版
        annualized_return = (1 + returns.mean()) ** 252 - 1
        return annualized_return / avg_max_dd if avg_max_dd != 0 else float('inf')

    def _calculate_consecutive_trades(self, trade_returns: List[float]) -> Tuple[int, int]:
        """連続勝敗の計算"""
        if not trade_returns:
            return 0, 0

        max_wins = max_losses = 0
        current_wins = current_losses = 0

        for ret in trade_returns:
            if ret > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif ret < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = current_losses = 0

        return max_wins, max_losses

    def _classify_market_regimes(self, market_data: pd.DataFrame) -> pd.Series:
        """市場レジーム分類（簡易版）"""
        if 'Close' not in market_data.columns:
            return pd.Series(['unknown'] * len(market_data), index=market_data.index)

        # 価格トレンド判定
        prices = market_data['Close']
        ma_20 = prices.rolling(20).mean()
        ma_50 = prices.rolling(50).mean()

        # ボラティリティ計算
        returns = prices.pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        vol_threshold = volatility.median()

        regimes = pd.Series(index=market_data.index, dtype=str)

        for i in range(len(market_data)):
            if pd.isna(ma_20.iloc[i]) or pd.isna(ma_50.iloc[i]):
                regimes.iloc[i] = 'unknown'
            elif ma_20.iloc[i] > ma_50.iloc[i] * 1.02:  # 上昇トレンド
                regimes.iloc[i] = 'bull'
            elif ma_20.iloc[i] < ma_50.iloc[i] * 0.98:  # 下降トレンド
                regimes.iloc[i] = 'bear'
            else:  # 横ばい
                regimes.iloc[i] = 'sideways'

        # ボラティリティレジーム
        for i in range(len(market_data)):
            if not pd.isna(volatility.iloc[i]):
                if volatility.iloc[i] > vol_threshold:
                    if regimes.iloc[i] != 'unknown':
                        regimes.iloc[i] = 'high_vol'
                else:
                    if regimes.iloc[i] != 'unknown':
                        regimes.iloc[i] = 'low_vol'

        return regimes

    def _calculate_regime_performance(self, returns: pd.Series, regime_mask: pd.Series) -> Dict[str, float]:
        """レジーム別パフォーマンス計算"""
        if not regime_mask.any():
            return {'return': 0.0, 'volatility': 0.0, 'sharpe': 0.0, 'count': 0}

        regime_returns = returns[regime_mask]

        if len(regime_returns) == 0:
            return {'return': 0.0, 'volatility': 0.0, 'sharpe': 0.0, 'count': 0}

        total_return = (1 + regime_returns).prod() - 1
        volatility = regime_returns.std() * np.sqrt(252)
        sharpe = self._calculate_sharpe_ratio(regime_returns)

        return {
            'return': total_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'count': len(regime_returns)
        }

    def _estimate_regime_detection_accuracy(self, market_data: pd.DataFrame, regimes: pd.Series) -> float:
        """レジーム検出精度の推定（簡易版）"""
        # 実装簡略化：一貫性チェック
        if 'Close' not in market_data.columns:
            return 0.0

        prices = market_data['Close']
        returns = prices.pct_change()

        # レジームと実際のリターンの一貫性をチェック
        consistency_score = 0
        total_checks = 0

        for i in range(1, len(regimes)):
            if regimes.iloc[i] != 'unknown' and not pd.isna(returns.iloc[i]):
                total_checks += 1

                if regimes.iloc[i] == 'bull' and returns.iloc[i] > 0:
                    consistency_score += 1
                elif regimes.iloc[i] == 'bear' and returns.iloc[i] < 0:
                    consistency_score += 1
                elif regimes.iloc[i] == 'sideways' and abs(returns.iloc[i]) < 0.01:
                    consistency_score += 1

        return consistency_score / total_checks if total_checks > 0 else 0.0

    def _analyze_regime_transitions(self, regimes: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """レジーム遷移分析"""
        transitions = {}
        regime_changes = 0

        for i in range(1, len(regimes)):
            if regimes.iloc[i] != regimes.iloc[i-1]:
                regime_changes += 1
                transition = f"{regimes.iloc[i-1]}_to_{regimes.iloc[i]}"
                if transition not in transitions:
                    transitions[transition] = []

                # 遷移時のリターン記録
                if not pd.isna(returns.iloc[i]):
                    transitions[transition].append(returns.iloc[i])

        # 遷移統計計算
        transition_stats = {}
        for transition, rets in transitions.items():
            if rets:
                transition_stats[transition] = {
                    'count': len(rets),
                    'avg_return': np.mean(rets),
                    'volatility': np.std(rets)
                }

        return {
            'total_regime_changes': regime_changes,
            'transition_frequency': regime_changes / len(regimes) if len(regimes) > 0 else 0,
            'transition_statistics': transition_stats
        }


class MultiTimeframeAnalyzer:
    """マルチタイムフレーム分析"""

    def __init__(self):
        self.supported_timeframes = ['1min', '5min', '15min', '1h', '4h', '1d', '1w', '1m']

    def analyze_multiple_timeframes(self,
                                  data: pd.DataFrame,
                                  timeframes: List[str]) -> Dict[str, Any]:
        """
        複数タイムフレームでの分析

        Args:
            data: 分足データ
            timeframes: 分析対象タイムフレーム

        Returns:
            タイムフレーム別分析結果
        """
        results = {}

        for tf in timeframes:
            if tf in self.supported_timeframes:
                resampled_data = self._resample_data(data, tf)
                if len(resampled_data) > 0:
                    results[tf] = self._analyze_timeframe(resampled_data)

        return results

    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """データのリサンプリング"""
        if not isinstance(data.index, pd.DatetimeIndex):
            return pd.DataFrame()

        # リサンプリング規則
        resample_rules = {
            '1min': '1T', '5min': '5T', '15min': '15T',
            '1h': '1H', '4h': '4H', '1d': '1D',
            '1w': '1W', '1m': '1M'
        }

        rule = resample_rules.get(timeframe, '1D')

        try:
            resampled = data.resample(rule).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

            return resampled
        except Exception:
            return pd.DataFrame()

    def _analyze_timeframe(self, data: pd.DataFrame) -> Dict[str, Any]:
        """タイムフレーム別分析"""
        if len(data) < 2:
            return {}

        returns = data['Close'].pct_change().dropna()

        return {
            'total_return': (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1),
            'volatility': returns.std() * np.sqrt(len(returns)),
            'max_drawdown': self._calculate_simple_drawdown(data['Close']),
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
            'data_points': len(data)
        }

    def _calculate_simple_drawdown(self, prices: pd.Series) -> float:
        """簡易ドローダウン計算"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()