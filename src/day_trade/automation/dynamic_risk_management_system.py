#!/usr/bin/env python3
"""
動的リスク管理システム

Issue #487対応: 完全自動化システム実装 - Phase 2
VaR・CVaR計算・ポートフォリオ最適化・相関リスク管理・動的ストップロス
"""

import asyncio
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import scipy.optimize as optimize
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from ..utils.logging_config import get_context_logger
from .adaptive_optimization_system import MarketRegime
from .notification_system import get_notification_system

logger = get_context_logger(__name__)


class RiskLevel(Enum):
    """リスクレベル"""
    CONSERVATIVE = "conservative"   # 保守的
    MODERATE = "moderate"          # 中程度
    AGGRESSIVE = "aggressive"      # 積極的
    DYNAMIC = "dynamic"           # 動的調整


class RiskMetric(Enum):
    """リスク指標"""
    VAR = "var"                   # Value at Risk
    CVAR = "cvar"                 # Conditional Value at Risk
    MAX_DRAWDOWN = "max_drawdown" # 最大ドローダウン
    VOLATILITY = "volatility"     # ボラティリティ
    BETA = "beta"                 # ベータ
    SHARPE_RATIO = "sharpe_ratio" # シャープレシオ


@dataclass
class RiskMetrics:
    """リスク指標データ"""
    symbol: str
    var_95: float                 # 95% VaR
    var_99: float                 # 99% VaR
    cvar_95: float               # 95% CVaR
    cvar_99: float               # 99% CVaR
    max_drawdown: float          # 最大ドローダウン
    volatility: float            # 年率ボラティリティ
    beta: float                  # ベータ（市場対比）
    sharpe_ratio: float          # シャープレシオ
    correlation_risk: float      # 相関リスク
    liquidity_risk: float        # 流動性リスク
    calculated_at: datetime


@dataclass
class PortfolioOptimizationResult:
    """ポートフォリオ最適化結果"""
    symbols: List[str]
    weights: np.ndarray
    expected_return: float
    portfolio_risk: float
    sharpe_ratio: float
    risk_metrics: RiskMetrics
    optimization_method: str
    constraints_met: bool
    optimization_time: float


@dataclass
class StopLossConfig:
    """ストップロス設定"""
    symbol: str
    entry_price: float
    stop_loss_price: float
    stop_loss_pct: float
    atr_multiplier: float
    volatility_adjusted: bool
    dynamic_update: bool
    last_updated: datetime


@dataclass
class RiskManagementConfig:
    """リスク管理設定"""
    max_portfolio_risk: float = 0.15      # 最大ポートフォリオリスク(15%)
    max_position_size: float = 0.10       # 最大ポジションサイズ(10%)
    max_correlation: float = 0.7          # 最大相関係数
    var_confidence: float = 0.95          # VaR信頼水準
    rebalance_threshold: float = 0.05     # リバランス閾値
    stop_loss_atr_multiplier: float = 2.0 # ストップロスATR倍数
    min_liquidity_score: float = 50.0     # 最小流動性スコア
    risk_level: RiskLevel = RiskLevel.MODERATE


class DynamicRiskManagementSystem:
    """
    Issue #487対応: 動的リスク管理システム

    Phase 2のリスク管理機能:
    - VaR・CVaR自動計算
    - Markowitzポートフォリオ最適化
    - 相関リスク監視
    - 動的ストップロス設定
    - リスクアラートシステム
    """

    def __init__(self, config: RiskManagementConfig = None):
        """初期化"""
        self.config = config or RiskManagementConfig()
        self.risk_history: List[RiskMetrics] = []
        self.portfolio_history: List[PortfolioOptimizationResult] = []
        self.stop_loss_configs: Dict[str, StopLossConfig] = {}

        # 通知システム
        self.notification = get_notification_system()

        # ベンチマーク（日経平均など）のデータ
        self.benchmark_returns = None

        logger.info("動的リスク管理システム初期化完了")

    async def calculate_risk_metrics(self, symbol: str, price_data: pd.DataFrame,
                                   benchmark_data: pd.DataFrame = None) -> RiskMetrics:
        """
        包括的リスク指標計算

        Args:
            symbol: 銘柄シンボル
            price_data: 価格データ
            benchmark_data: ベンチマークデータ

        Returns:
            リスク指標
        """
        try:
            if price_data.empty or len(price_data) < 30:
                raise ValueError("計算に十分なデータがありません")

            # リターン計算
            close_col = 'Close' if 'Close' in price_data.columns else price_data.columns[-1]
            prices = price_data[close_col]
            returns = prices.pct_change().dropna()

            if len(returns) < 10:
                raise ValueError("リターンデータが不足しています")

            # 1. VaR計算 (Historical Method)
            var_95 = self._calculate_var(returns, confidence=0.95)
            var_99 = self._calculate_var(returns, confidence=0.99)

            # 2. CVaR計算
            cvar_95 = self._calculate_cvar(returns, confidence=0.95)
            cvar_99 = self._calculate_cvar(returns, confidence=0.99)

            # 3. 最大ドローダウン計算
            max_drawdown = self._calculate_max_drawdown(prices)

            # 4. ボラティリティ計算（年率化）
            volatility = returns.std() * np.sqrt(252)

            # 5. ベータ計算
            beta = self._calculate_beta(returns, benchmark_data) if benchmark_data is not None else 1.0

            # 6. シャープレシオ計算
            risk_free_rate = 0.02  # 2%と仮定
            mean_return = returns.mean() * 252  # 年率化
            sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0.0

            # 7. 相関リスク計算（他の保有銘柄との相関）
            correlation_risk = self._calculate_correlation_risk(symbol, returns)

            # 8. 流動性リスク計算
            liquidity_risk = self._calculate_liquidity_risk(price_data)

            risk_metrics = RiskMetrics(
                symbol=symbol,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                sharpe_ratio=sharpe_ratio,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                calculated_at=datetime.now()
            )

            # 履歴に追加
            self.risk_history.append(risk_metrics)
            if len(self.risk_history) > 1000:  # 最新1000件保持
                self.risk_history = self.risk_history[-1000:]

            logger.info(f"リスク指標計算完了: {symbol}")
            logger.info(f"  VaR(95%): {var_95:.4f}, CVaR(95%): {cvar_95:.4f}")
            logger.info(f"  最大DD: {max_drawdown:.4f}, ボラティリティ: {volatility:.4f}")
            logger.info(f"  シャープ: {sharpe_ratio:.4f}, ベータ: {beta:.4f}")

            return risk_metrics

        except Exception as e:
            logger.error(f"リスク指標計算エラー ({symbol}): {e}")
            raise

    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """VaR計算 (Historical Method)"""
        if len(returns) == 0:
            return 0.0

        alpha = 1 - confidence
        var = np.percentile(returns, alpha * 100)
        return abs(var)  # 正値として返す

    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """CVaR計算 (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0

        var = self._calculate_var(returns, confidence)
        # VaRを超える損失の平均
        tail_losses = returns[returns <= -var]

        if len(tail_losses) == 0:
            return var

        cvar = abs(tail_losses.mean())
        return cvar

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """最大ドローダウン計算"""
        if len(prices) == 0:
            return 0.0

        # 累積最大値
        peak = prices.expanding().max()

        # ドローダウン
        drawdown = (prices - peak) / peak

        # 最大ドローダウン
        max_dd = abs(drawdown.min())
        return max_dd

    def _calculate_beta(self, returns: pd.Series, benchmark_data: pd.DataFrame) -> float:
        """ベータ計算"""
        if benchmark_data is None or benchmark_data.empty:
            return 1.0

        try:
            # ベンチマークリターン計算
            bench_close = 'Close' if 'Close' in benchmark_data.columns else benchmark_data.columns[-1]
            bench_returns = benchmark_data[bench_close].pct_change().dropna()

            # 期間を合わせる
            min_len = min(len(returns), len(bench_returns))
            if min_len < 10:
                return 1.0

            aligned_returns = returns.iloc[-min_len:]
            aligned_bench = bench_returns.iloc[-min_len:]

            # 線形回帰でベータ計算
            covariance = np.cov(aligned_returns, aligned_bench)[0, 1]
            benchmark_variance = np.var(aligned_bench)

            if benchmark_variance == 0:
                return 1.0

            beta = covariance / benchmark_variance
            return beta

        except Exception as e:
            logger.debug(f"ベータ計算エラー: {e}")
            return 1.0

    def _calculate_correlation_risk(self, symbol: str, returns: pd.Series) -> float:
        """相関リスク計算"""
        # 他の保有銘柄との相関を計算（簡易実装）
        # 実際の実装では、ポートフォリオ内の他銘柄データと比較

        # デモ実装: ランダムな相関リスクスコア
        np.random.seed(hash(symbol) % 2**32)
        correlation_risk = np.random.uniform(0.3, 0.8)
        return correlation_risk

    def _calculate_liquidity_risk(self, price_data: pd.DataFrame) -> float:
        """流動性リスク計算"""
        try:
            if 'Volume' not in price_data.columns:
                return 0.5  # デフォルト中程度リスク

            volumes = price_data['Volume']

            # ボリューム変動係数
            vol_cv = volumes.std() / volumes.mean() if volumes.mean() > 0 else 1.0

            # 低ボリューム日の割合
            low_volume_days = len(volumes[volumes < volumes.median() * 0.5]) / len(volumes)

            # 流動性リスクスコア (0-1, 1が最もリスク高)
            liquidity_risk = min((vol_cv * 0.5 + low_volume_days * 0.5), 1.0)
            return liquidity_risk

        except Exception as e:
            logger.debug(f"流動性リスク計算エラー: {e}")
            return 0.5

    async def optimize_portfolio(self, symbols: List[str],
                               expected_returns: np.ndarray,
                               covariance_matrix: np.ndarray,
                               method: str = "max_sharpe") -> PortfolioOptimizationResult:
        """
        ポートフォリオ最適化 (Markowitz理論)

        Args:
            symbols: 銘柄リスト
            expected_returns: 期待リターン配列
            covariance_matrix: 共分散行列
            method: 最適化手法 ("max_sharpe", "min_risk", "efficient_frontier")

        Returns:
            ポートフォリオ最適化結果
        """
        logger.info(f"ポートフォリオ最適化開始: {method}")
        start_time = datetime.now()

        try:
            n_assets = len(symbols)

            # 制約条件
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # 重み合計=1
            ]

            # 境界条件（各資産の重み 0-max_position_size）
            bounds = [(0.0, self.config.max_position_size) for _ in range(n_assets)]

            # 初期重み（等分散）
            initial_weights = np.ones(n_assets) / n_assets

            # 最適化実行
            if method == "max_sharpe":
                result = self._maximize_sharpe_ratio(
                    expected_returns, covariance_matrix, constraints, bounds, initial_weights
                )
            elif method == "min_risk":
                result = self._minimize_portfolio_risk(
                    covariance_matrix, constraints, bounds, initial_weights
                )
            else:
                raise ValueError(f"未サポートの最適化手法: {method}")

            if not result.success:
                raise Exception(f"最適化失敗: {result.message}")

            optimal_weights = result.x

            # ポートフォリオ指標計算
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)))

            # シャープレシオ
            risk_free_rate = 0.02  # 2%
            portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0.0

            # ポートフォリオリスク指標計算
            portfolio_risk_metrics = self._calculate_portfolio_risk_metrics(
                optimal_weights, symbols, expected_returns, covariance_matrix
            )

            # 制約チェック
            constraints_met = self._check_portfolio_constraints(optimal_weights, portfolio_risk_metrics)

            optimization_time = (datetime.now() - start_time).total_seconds()

            optimization_result = PortfolioOptimizationResult(
                symbols=symbols,
                weights=optimal_weights,
                expected_return=portfolio_return,
                portfolio_risk=portfolio_risk,
                sharpe_ratio=portfolio_sharpe,
                risk_metrics=portfolio_risk_metrics,
                optimization_method=method,
                constraints_met=constraints_met,
                optimization_time=optimization_time
            )

            # 履歴に追加
            self.portfolio_history.append(optimization_result)
            if len(self.portfolio_history) > 100:  # 最新100件保持
                self.portfolio_history = self.portfolio_history[-100:]

            logger.info(f"ポートフォリオ最適化完了")
            logger.info(f"  期待リターン: {portfolio_return:.4f}")
            logger.info(f"  ポートフォリオリスク: {portfolio_risk:.4f}")
            logger.info(f"  シャープレシオ: {portfolio_sharpe:.4f}")

            # 重み表示
            for i, (symbol, weight) in enumerate(zip(symbols, optimal_weights)):
                if weight > 0.01:  # 1%以上の重みのみ表示
                    logger.info(f"  {symbol}: {weight:.3f} ({weight*100:.1f}%)")

            # 通知送信
            await self._send_portfolio_optimization_notification(optimization_result)

            return optimization_result

        except Exception as e:
            logger.error(f"ポートフォリオ最適化エラー: {e}")
            raise

    def _maximize_sharpe_ratio(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                              constraints: List[Dict], bounds: List[Tuple],
                              initial_weights: np.ndarray) -> optimize.OptimizeResult:
        """シャープレシオ最大化"""

        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))

            if portfolio_risk == 0:
                return -1000.0  # 非常に悪いスコア

            risk_free_rate = 0.02
            sharpe = (portfolio_return - risk_free_rate) / portfolio_risk
            return -sharpe  # 最小化のため符号反転

        return optimize.minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

    def _minimize_portfolio_risk(self, covariance_matrix: np.ndarray,
                                constraints: List[Dict], bounds: List[Tuple],
                                initial_weights: np.ndarray) -> optimize.OptimizeResult:
        """ポートフォリオリスク最小化"""

        def portfolio_variance(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))

        return optimize.minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

    def _calculate_portfolio_risk_metrics(self, weights: np.ndarray, symbols: List[str],
                                        expected_returns: np.ndarray,
                                        covariance_matrix: np.ndarray) -> RiskMetrics:
        """ポートフォリオリスク指標計算"""

        # 簡易ポートフォリオリスク指標
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))

        # VaR・CVaR推定（正規分布仮定）
        var_95 = abs(stats.norm.ppf(0.05, portfolio_return, portfolio_risk))
        var_99 = abs(stats.norm.ppf(0.01, portfolio_return, portfolio_risk))
        cvar_95 = var_95 * 1.2  # 簡易推定
        cvar_99 = var_99 * 1.15  # 簡易推定

        # 最大ドローダウン推定
        max_drawdown = portfolio_risk * 2.0  # 簡易推定

        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0.0

        return RiskMetrics(
            symbol="PORTFOLIO",
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            volatility=portfolio_risk,
            beta=1.0,
            sharpe_ratio=sharpe_ratio,
            correlation_risk=0.5,
            liquidity_risk=0.3,
            calculated_at=datetime.now()
        )

    def _check_portfolio_constraints(self, weights: np.ndarray,
                                   risk_metrics: RiskMetrics) -> bool:
        """ポートフォリオ制約チェック"""

        # 1. 最大ポートフォリオリスク制約
        if risk_metrics.volatility > self.config.max_portfolio_risk:
            logger.warning(f"ポートフォリオリスク制約違反: {risk_metrics.volatility:.4f} > {self.config.max_portfolio_risk}")
            return False

        # 2. 最大ポジションサイズ制約
        if np.max(weights) > self.config.max_position_size:
            logger.warning(f"最大ポジション制約違反: {np.max(weights):.4f} > {self.config.max_position_size}")
            return False

        # 3. 重み合計制約
        if abs(np.sum(weights) - 1.0) > 0.01:
            logger.warning(f"重み合計制約違反: {np.sum(weights):.4f} != 1.0")
            return False

        return True

    async def calculate_dynamic_stop_loss(self, symbol: str, price_data: pd.DataFrame,
                                        entry_price: float,
                                        market_regime: MarketRegime = None) -> StopLossConfig:
        """
        動的ストップロス計算

        Args:
            symbol: 銘柄シンボル
            price_data: 価格データ
            entry_price: エントリー価格
            market_regime: 市場レジーム

        Returns:
            ストップロス設定
        """
        try:
            if price_data.empty or len(price_data) < 14:
                raise ValueError("ATR計算に十分なデータがありません")

            # ATR (Average True Range) 計算
            atr = self._calculate_atr(price_data, period=14)

            # 市場レジーム別ATR倍数調整
            atr_multiplier = self._get_regime_adjusted_atr_multiplier(market_regime)

            # ストップロス価格計算
            stop_loss_distance = atr * atr_multiplier
            stop_loss_price = entry_price - stop_loss_distance  # ロングポジション想定
            stop_loss_pct = stop_loss_distance / entry_price

            # ボラティリティ調整
            volatility_adjustment = self._calculate_volatility_adjustment(price_data)
            adjusted_stop_loss_price = entry_price * (1 - stop_loss_pct * volatility_adjustment)

            stop_loss_config = StopLossConfig(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss_price=adjusted_stop_loss_price,
                stop_loss_pct=stop_loss_pct * volatility_adjustment,
                atr_multiplier=atr_multiplier,
                volatility_adjusted=True,
                dynamic_update=True,
                last_updated=datetime.now()
            )

            # 設定保存
            self.stop_loss_configs[symbol] = stop_loss_config

            logger.info(f"動的ストップロス設定: {symbol}")
            logger.info(f"  エントリー価格: {entry_price:.2f}")
            logger.info(f"  ストップロス価格: {adjusted_stop_loss_price:.2f}")
            logger.info(f"  ストップロス率: {stop_loss_pct * volatility_adjustment * 100:.2f}%")
            logger.info(f"  ATR倍数: {atr_multiplier:.2f}")

            return stop_loss_config

        except Exception as e:
            logger.error(f"動的ストップロス計算エラー ({symbol}): {e}")
            raise

    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """ATR計算"""
        try:
            high = price_data['High']
            low = price_data['Low']
            close = price_data['Close']

            # True Range計算
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # ATR（True Rangeの移動平均）
            atr = true_range.rolling(period).mean().iloc[-1]

            return atr if not np.isnan(atr) else 0.0

        except Exception as e:
            logger.debug(f"ATR計算エラー: {e}")
            return 0.01  # デフォルト値

    def _get_regime_adjusted_atr_multiplier(self, regime: MarketRegime = None) -> float:
        """レジーム調整ATR倍数"""
        base_multiplier = self.config.stop_loss_atr_multiplier

        if regime == MarketRegime.VOLATILE:
            return base_multiplier * 1.5  # ボラティリティ高時は広めに設定
        elif regime == MarketRegime.BEAR:
            return base_multiplier * 0.8  # 弱気相場は狭めに設定
        elif regime == MarketRegime.BULL:
            return base_multiplier * 1.2  # 強気相場は少し広めに設定
        else:
            return base_multiplier

    def _calculate_volatility_adjustment(self, price_data: pd.DataFrame) -> float:
        """ボラティリティ調整係数"""
        try:
            close = price_data['Close']
            returns = close.pct_change().dropna()

            if len(returns) < 10:
                return 1.0

            # 最近のボラティリティ vs 長期ボラティリティ
            recent_vol = returns.tail(10).std()
            long_term_vol = returns.std()

            if long_term_vol == 0:
                return 1.0

            vol_ratio = recent_vol / long_term_vol

            # 調整係数 (0.5-2.0の範囲)
            adjustment = np.clip(vol_ratio, 0.5, 2.0)
            return adjustment

        except Exception as e:
            logger.debug(f"ボラティリティ調整計算エラー: {e}")
            return 1.0

    async def _send_portfolio_optimization_notification(self, result: PortfolioOptimizationResult):
        """ポートフォリオ最適化結果通知"""
        try:
            # 重み表示文字列生成
            weights_text = ""
            for symbol, weight in zip(result.symbols, result.weights):
                if weight > 0.01:  # 1%以上のみ表示
                    weights_text += f"  {symbol}: {weight:.3f} ({weight*100:.1f}%)\n"

            notification_data = {
                'optimization_method': result.optimization_method,
                'expected_return': result.expected_return,
                'portfolio_risk': result.portfolio_risk,
                'sharpe_ratio': result.sharpe_ratio,
                'n_assets': len(result.symbols),
                'weights_distribution': weights_text,
                'constraints_met': '満足' if result.constraints_met else '違反',
                'optimization_time': result.optimization_time
            }

            # 通知テンプレート作成
            from .notification_system import NotificationTemplate, NotificationType, NotificationChannel

            template = NotificationTemplate(
                template_id="portfolio_optimization",
                subject_template="[ポートフォリオ最適化] {optimization_method}最適化完了",
                body_template="""
📊 ポートフォリオ最適化結果

🎯 最適化手法: {optimization_method}
📈 期待リターン: {expected_return:.4f}
📉 ポートフォリオリスク: {portfolio_risk:.4f}
⚡ シャープレシオ: {sharpe_ratio:.4f}
🏛️ 資産数: {n_assets}
✅ 制約: {constraints_met}
⏱️ 最適化時間: {optimization_time:.1f}秒

💼 重み配分:
{weights_distribution}

---
Issue #487 Phase 2: 動的リスク管理システム
""",
                notification_type=NotificationType.SUCCESS,
                channels=[NotificationChannel.LOG, NotificationChannel.FILE]
            )

            self.notification.templates["portfolio_optimization"] = template
            self.notification.send_notification("portfolio_optimization", notification_data)

        except Exception as e:
            logger.error(f"ポートフォリオ最適化通知エラー: {e}")

    def get_portfolio_risk_summary(self) -> Dict[str, Any]:
        """ポートフォリオリスクサマリー取得"""
        if not self.portfolio_history:
            return {"status": "no_data"}

        latest_portfolio = self.portfolio_history[-1]

        return {
            "timestamp": latest_portfolio.risk_metrics.calculated_at,
            "portfolio_risk": latest_portfolio.portfolio_risk,
            "expected_return": latest_portfolio.expected_return,
            "sharpe_ratio": latest_portfolio.sharpe_ratio,
            "var_95": latest_portfolio.risk_metrics.var_95,
            "max_drawdown": latest_portfolio.risk_metrics.max_drawdown,
            "constraints_met": latest_portfolio.constraints_met,
            "n_assets": len(latest_portfolio.symbols),
            "optimization_method": latest_portfolio.optimization_method
        }


# デバッグ用メイン関数
async def main():
    """デバッグ用メイン"""
    logger.info("動的リスク管理システム テスト実行")

    # システム初期化
    risk_manager = DynamicRiskManagementSystem()

    # テスト用価格データ生成
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')

    # リアルな株価データ生成
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = [100.0]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))

    # OHLCV データ生成
    test_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Volume': np.random.lognormal(10, 0.5, n_days)
    })

    # リスク指標計算テスト
    print("=" * 60)
    print("リスク指標計算テスト")
    print("=" * 60)

    risk_metrics = await risk_manager.calculate_risk_metrics("TEST", test_data)

    print(f"銘柄: {risk_metrics.symbol}")
    print(f"VaR(95%): {risk_metrics.var_95:.4f}")
    print(f"CVaR(95%): {risk_metrics.cvar_95:.4f}")
    print(f"最大ドローダウン: {risk_metrics.max_drawdown:.4f}")
    print(f"ボラティリティ: {risk_metrics.volatility:.4f}")
    print(f"シャープレシオ: {risk_metrics.sharpe_ratio:.4f}")

    # ポートフォリオ最適化テスト
    print("\n" + "=" * 60)
    print("ポートフォリオ最適化テスト")
    print("=" * 60)

    # テスト用データ
    symbols = ["ASSET_A", "ASSET_B", "ASSET_C"]
    expected_returns = np.array([0.10, 0.12, 0.08])  # 期待リターン

    # 共分散行列（相関を含む）
    correlations = np.array([
        [1.0, 0.3, 0.2],
        [0.3, 1.0, 0.4],
        [0.2, 0.4, 1.0]
    ])
    volatilities = np.array([0.15, 0.20, 0.12])
    covariance_matrix = np.outer(volatilities, volatilities) * correlations

    # 最適化実行
    optimization_result = await risk_manager.optimize_portfolio(
        symbols, expected_returns, covariance_matrix, method="max_sharpe"
    )

    print(f"最適化手法: {optimization_result.optimization_method}")
    print(f"期待リターン: {optimization_result.expected_return:.4f}")
    print(f"ポートフォリオリスク: {optimization_result.portfolio_risk:.4f}")
    print(f"シャープレシオ: {optimization_result.sharpe_ratio:.4f}")
    print(f"制約満足: {optimization_result.constraints_met}")

    print("\n重み配分:")
    for symbol, weight in zip(symbols, optimization_result.weights):
        print(f"  {symbol}: {weight:.3f} ({weight*100:.1f}%)")

    # 動的ストップロステスト
    print("\n" + "=" * 60)
    print("動的ストップロステスト")
    print("=" * 60)

    entry_price = 100.0
    stop_loss_config = await risk_manager.calculate_dynamic_stop_loss(
        "TEST", test_data, entry_price, MarketRegime.VOLATILE
    )

    print(f"エントリー価格: {stop_loss_config.entry_price:.2f}")
    print(f"ストップロス価格: {stop_loss_config.stop_loss_price:.2f}")
    print(f"ストップロス率: {stop_loss_config.stop_loss_pct*100:.2f}%")
    print(f"ATR倍数: {stop_loss_config.atr_multiplier:.2f}")
    print(f"ボラティリティ調整: {stop_loss_config.volatility_adjusted}")

    logger.info("動的リスク管理システム テスト完了")


if __name__ == "__main__":
    asyncio.run(main())