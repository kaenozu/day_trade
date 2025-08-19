#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Management AI - AIポートフォリオ最適化・リスク管理システム
Issue #943対応: 高度なリスク分析 + ポートフォリオ最適化 + VaR計算
"""

import numpy as np
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import asyncio

# 数値計算・最適化ライブラリ
try:
    import scipy.optimize as optimize
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

# 機械学習ライブラリ
try:
    from sklearn.covariance import LedoitWolf
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# 統合モジュール
try:
    from advanced_ai_engine import advanced_ai_engine, MarketSignal
    HAS_AI_ENGINE = True
except ImportError:
    HAS_AI_ENGINE = False

try:
    from quantum_ai_engine import quantum_ai_engine
    HAS_QUANTUM_AI = True
except ImportError:
    HAS_QUANTUM_AI = False

try:
    from performance_monitor import performance_monitor
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False


@dataclass
class RiskMetrics:
    """リスクメトリクス"""
    symbol: str
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    cvar_95: float  # 95% Conditional VaR
    max_drawdown: float  # 最大ドローダウン
    volatility: float  # ボラティリティ
    beta: float  # ベータ値
    sharpe_ratio: float  # シャープレシオ
    sortino_ratio: float  # ソルティノレシオ
    calmar_ratio: float  # カルマーレシオ
    risk_score: float  # 総合リスクスコア（0-100）
    timestamp: datetime


@dataclass
class PortfolioAllocation:
    """ポートフォリオ配分"""
    symbol: str
    weight: float  # 配分比率（0-1）
    target_amount: float  # 目標金額
    current_amount: float  # 現在金額
    risk_contribution: float  # リスク貢献度
    expected_return: float  # 期待収益率
    confidence: float  # 信頼度


@dataclass
class PortfolioOptimization:
    """ポートフォリオ最適化結果"""
    portfolio_id: str
    optimization_method: str  # 'mean_variance', 'risk_parity', 'black_litterman', 'ai_enhanced'
    allocations: List[PortfolioAllocation]
    expected_return: float  # ポートフォリオ期待収益率
    expected_volatility: float  # ポートフォリオ期待ボラティリティ
    sharpe_ratio: float  # ポートフォリオシャープレシオ
    var_95: float  # ポートフォリオVaR
    optimization_score: float  # 最適化スコア
    rebalance_frequency: str  # リバランス頻度
    timestamp: datetime


@dataclass
class RiskAlert:
    """リスクアラート"""
    alert_id: str
    alert_type: str  # 'VAR_BREACH', 'CONCENTRATION', 'CORRELATION', 'VOLATILITY'
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    symbol: Optional[str]
    message: str
    current_value: float
    threshold_value: float
    recommendation: str
    timestamp: datetime


class VaRCalculator:
    """VaR計算器"""
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
        self.methods = ['historical', 'parametric', 'monte_carlo']
    
    def calculate_historical_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """ヒストリカルVaR計算"""
        if len(returns) == 0:
            return 0.0
        
        percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, percentile)
        return abs(var)
    
    def calculate_parametric_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """パラメトリックVaR計算"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        z_score = stats.norm.ppf(1 - confidence_level)
        var = mean_return + z_score * std_return
        return abs(var)
    
    def calculate_monte_carlo_var(self, returns: np.ndarray, confidence_level: float = 0.95, 
                                n_simulations: int = 10000) -> float:
        """モンテカルロVaR計算"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # モンテカルロシミュレーション
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        percentile = (1 - confidence_level) * 100
        var = np.percentile(simulated_returns, percentile)
        return abs(var)
    
    def calculate_conditional_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """条件付きVaR（CVaR）計算"""
        var = self.calculate_historical_var(returns, confidence_level)
        
        # VaRを超える損失の平均
        tail_returns = returns[returns <= -var]
        if len(tail_returns) > 0:
            cvar = abs(np.mean(tail_returns))
        else:
            cvar = var
        
        return cvar


class PortfolioOptimizer:
    """ポートフォリオ最適化エンジン"""
    
    def __init__(self):
        self.optimization_methods = [
            'mean_variance', 'risk_parity', 'minimum_variance', 
            'maximum_sharpe', 'black_litterman', 'ai_enhanced'
        ]
    
    def mean_variance_optimization(self, expected_returns: np.ndarray, 
                                 covariance_matrix: np.ndarray,
                                 risk_aversion: float = 1.0) -> np.ndarray:
        """平均分散最適化"""
        if not HAS_SCIPY:
            logging.warning("Scipy not available, using equal weights")
            return np.ones(len(expected_returns)) / len(expected_returns)
        
        n = len(expected_returns)
        
        # 目的関数: リターン最大化 - リスクペナルティ
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
        
        # 制約条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 重み合計 = 1
        ]
        
        bounds = [(0, 1) for _ in range(n)]  # 各重みは0-1
        
        # 初期解
        x0 = np.ones(n) / n
        
        result = optimize.minimize(objective, x0, method='SLSQP', 
                                 bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def risk_parity_optimization(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """リスクパリティ最適化"""
        if not HAS_SCIPY:
            logging.warning("Scipy not available, using equal weights")
            return np.ones(covariance_matrix.shape[0]) / covariance_matrix.shape[0]
        
        n = covariance_matrix.shape[0]
        
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # 各資産のリスク貢献度を均等にする
            target_contrib = portfolio_vol / n
            return np.sum((contrib - target_contrib) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.01, 1) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = optimize.minimize(risk_budget_objective, x0, method='SLSQP',
                                 bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def maximum_sharpe_optimization(self, expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   risk_free_rate: float = 0.02) -> np.ndarray:
        """最大シャープレシオ最適化"""
        if not HAS_SCIPY:
            logging.warning("Scipy not available, using equal weights")
            return np.ones(len(expected_returns)) / len(expected_returns)
        
        n = len(expected_returns)
        excess_returns = expected_returns - risk_free_rate
        
        def neg_sharpe_ratio(weights):
            portfolio_return = np.dot(weights, excess_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            return -portfolio_return / max(portfolio_vol, 1e-6)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = optimize.minimize(neg_sharpe_ratio, x0, method='SLSQP',
                                 bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def minimum_variance_optimization(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """最小分散最適化"""
        if not HAS_SCIPY:
            logging.warning("Scipy not available, using equal weights")
            return np.ones(covariance_matrix.shape[0]) / covariance_matrix.shape[0]
        
        n = covariance_matrix.shape[0]
        
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = optimize.minimize(portfolio_variance, x0, method='SLSQP',
                                 bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def ai_enhanced_optimization(self, symbols: List[str],
                               expected_returns: np.ndarray,
                               covariance_matrix: np.ndarray) -> np.ndarray:
        """AI強化最適化"""
        if not HAS_AI_ENGINE or not advanced_ai_engine:
            return self.mean_variance_optimization(expected_returns, covariance_matrix)
        
        # AI信号を取得して期待収益率を調整
        ai_adjustments = []
        for symbol in symbols:
            try:
                signal = advanced_ai_engine.analyze_symbol(symbol)
                
                # AI信号に基づく調整係数
                if signal.signal_type == 'BUY':
                    adjustment = 1.0 + signal.confidence * signal.strength
                elif signal.signal_type == 'SELL':
                    adjustment = 1.0 - signal.confidence * signal.strength
                else:
                    adjustment = 1.0
                
                ai_adjustments.append(adjustment)
                
            except Exception as e:
                logging.warning(f"AI analysis failed for {symbol}: {e}")
                ai_adjustments.append(1.0)
        
        # AI調整を適用した期待収益率
        adjusted_returns = expected_returns * np.array(ai_adjustments)
        
        return self.maximum_sharpe_optimization(adjusted_returns, covariance_matrix)


class RiskManagementAI:
    """リスク管理AIシステム"""
    
    def __init__(self):
        self.var_calculator = VaRCalculator()
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # データ保存
        self.risk_metrics_history: Dict[str, List[RiskMetrics]] = defaultdict(list)
        self.portfolio_history: List[PortfolioOptimization] = []
        self.alert_history: List[RiskAlert] = []
        
        # 設定
        self.risk_thresholds = {
            'var_95': 0.05,  # 5%
            'var_99': 0.10,  # 10%
            'max_drawdown': 0.20,  # 20%
            'concentration': 0.30,  # 30%
            'correlation': 0.80  # 80%
        }
        
        # パフォーマンス統計
        self.total_risk_calculations = 0
        self.total_optimizations = 0
        self.total_alerts_generated = 0
        self.start_time = datetime.now()
        
        # 市場データキャッシュ
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        self.cache_lock = threading.Lock()
    
    def generate_market_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """市場データ生成（シミュレーション）"""
        with self.cache_lock:
            cache_key = f"{symbol}_{days}"
            
            if cache_key in self.market_data_cache:
                return self.market_data_cache[cache_key]
            
            # 模擬市場データ生成
            np.random.seed(hash(symbol) % 2**32)
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            # 価格データ生成（幾何ブラウン運動）
            initial_price = 1000 + (hash(symbol) % 1000)
            mu = 0.08 / 252  # 年率8%のドリフト
            sigma = 0.20 / np.sqrt(252)  # 年率20%のボラティリティ
            
            returns = np.random.normal(mu, sigma, days)
            prices = [initial_price]
            
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            df = pd.DataFrame({
                'date': dates,
                'price': prices[1:],
                'returns': returns,
                'volume': np.random.randint(100000, 1000000, days)
            })
            
            self.market_data_cache[cache_key] = df
            return df
    
    def calculate_risk_metrics(self, symbol: str, lookback_days: int = 252) -> RiskMetrics:
        """個別銘柄のリスクメトリクス計算"""
        try:
            # 市場データ取得
            market_data = self.generate_market_data(symbol, lookback_days)
            returns = market_data['returns'].values
            prices = market_data['price'].values
            
            # VaR計算
            var_95 = self.var_calculator.calculate_historical_var(returns, 0.95)
            var_99 = self.var_calculator.calculate_historical_var(returns, 0.99)
            cvar_95 = self.var_calculator.calculate_conditional_var(returns, 0.95)
            
            # その他のリスクメトリクス
            max_drawdown = self._calculate_max_drawdown(prices)
            volatility = np.std(returns) * np.sqrt(252)  # 年率ボラティリティ
            
            # ベータ値（日経平均との相関を仮定）
            market_returns = np.random.normal(0.08/252, 0.15/np.sqrt(252), len(returns))
            beta = np.cov(returns, market_returns)[0, 1] / np.var(market_returns)
            
            # パフォーマンス指標
            mean_return = np.mean(returns) * 252  # 年率リターン
            risk_free_rate = 0.02  # リスクフリーレート2%
            
            sharpe_ratio = (mean_return - risk_free_rate) / max(volatility, 1e-6)
            
            # ソルティノレシオ（下方偏差）
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = (mean_return - risk_free_rate) / max(downside_deviation, 1e-6)
            
            # カルマーレシオ
            calmar_ratio = mean_return / max(max_drawdown, 1e-6)
            
            # 総合リスクスコア（0-100、低いほど安全）
            risk_score = min(100, (var_95 * 50 + max_drawdown * 30 + volatility * 20))
            
            risk_metrics = RiskMetrics(
                symbol=symbol,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                risk_score=risk_score,
                timestamp=datetime.now()
            )
            
            # 履歴に保存
            self.risk_metrics_history[symbol].append(risk_metrics)
            if len(self.risk_metrics_history[symbol]) > 100:
                self.risk_metrics_history[symbol].pop(0)
            
            self.total_risk_calculations += 1
            return risk_metrics
            
        except Exception as e:
            logging.error(f"Risk metrics calculation failed for {symbol}: {e}")
            return RiskMetrics(
                symbol=symbol, var_95=0.0, var_99=0.0, cvar_95=0.0,
                max_drawdown=0.0, volatility=0.0, beta=1.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                risk_score=100.0, timestamp=datetime.now()
            )
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """最大ドローダウン計算"""
        if len(prices) == 0:
            return 0.0
        
        peak = prices[0]
        max_drawdown = 0.0
        
        for price in prices:
            if price > peak:
                peak = price
            
            drawdown = (peak - price) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def optimize_portfolio(self, symbols: List[str], method: str = 'ai_enhanced',
                         portfolio_value: float = 1000000.0) -> PortfolioOptimization:
        """ポートフォリオ最適化実行"""
        try:
            # 各銘柄の期待収益率とリスクメトリクス取得
            expected_returns = []
            risk_metrics_list = []
            
            for symbol in symbols:
                risk_metrics = self.calculate_risk_metrics(symbol)
                risk_metrics_list.append(risk_metrics)
                
                # シャープレシオから期待収益率を推定
                expected_return = max(0.02, risk_metrics.sharpe_ratio * risk_metrics.volatility + 0.02)
                expected_returns.append(expected_return)
            
            expected_returns = np.array(expected_returns)
            
            # 共分散行列生成
            covariance_matrix = self._estimate_covariance_matrix(symbols)
            
            # 最適化実行
            if method == 'mean_variance':
                weights = self.portfolio_optimizer.mean_variance_optimization(
                    expected_returns, covariance_matrix
                )
            elif method == 'risk_parity':
                weights = self.portfolio_optimizer.risk_parity_optimization(covariance_matrix)
            elif method == 'minimum_variance':
                weights = self.portfolio_optimizer.minimum_variance_optimization(covariance_matrix)
            elif method == 'maximum_sharpe':
                weights = self.portfolio_optimizer.maximum_sharpe_optimization(
                    expected_returns, covariance_matrix
                )
            elif method == 'ai_enhanced':
                weights = self.portfolio_optimizer.ai_enhanced_optimization(
                    symbols, expected_returns, covariance_matrix
                )
            else:
                weights = np.ones(len(symbols)) / len(symbols)  # 等重量
            
            # ポートフォリオアロケーション構築
            allocations = []
            total_risk_contribution = 0.0
            
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_variance)
            
            for i, symbol in enumerate(symbols):
                risk_contrib = weights[i] * (covariance_matrix[i] @ weights) / max(portfolio_vol, 1e-6)
                total_risk_contribution += risk_contrib
                
                allocation = PortfolioAllocation(
                    symbol=symbol,
                    weight=weights[i],
                    target_amount=portfolio_value * weights[i],
                    current_amount=portfolio_value * weights[i],  # 初期は同じ
                    risk_contribution=risk_contrib,
                    expected_return=expected_returns[i],
                    confidence=risk_metrics_list[i].sharpe_ratio / max(1.0, abs(risk_metrics_list[i].sharpe_ratio))
                )
                allocations.append(allocation)
            
            # ポートフォリオ統計
            portfolio_expected_return = np.dot(weights, expected_returns)
            portfolio_expected_vol = portfolio_vol
            portfolio_sharpe = (portfolio_expected_return - 0.02) / max(portfolio_expected_vol, 1e-6)
            
            # ポートフォリオVaR
            portfolio_returns = []
            for i, symbol in enumerate(symbols):
                market_data = self.generate_market_data(symbol, 252)
                portfolio_returns.append(market_data['returns'].values * weights[i])
            
            portfolio_returns_series = np.sum(portfolio_returns, axis=0)
            portfolio_var = self.var_calculator.calculate_historical_var(portfolio_returns_series, 0.95)
            
            optimization_result = PortfolioOptimization(
                portfolio_id=f"portfolio_{int(time.time())}",
                optimization_method=method,
                allocations=allocations,
                expected_return=portfolio_expected_return,
                expected_volatility=portfolio_expected_vol,
                sharpe_ratio=portfolio_sharpe,
                var_95=portfolio_var,
                optimization_score=min(100.0, max(0.0, portfolio_sharpe * 20 + 50)),
                rebalance_frequency='monthly',
                timestamp=datetime.now()
            )
            
            self.portfolio_history.append(optimization_result)
            if len(self.portfolio_history) > 50:
                self.portfolio_history.pop(0)
            
            self.total_optimizations += 1
            return optimization_result
            
        except Exception as e:
            logging.error(f"Portfolio optimization failed: {e}")
            # フォールバック：等重量ポートフォリオ
            equal_weight = 1.0 / len(symbols)
            allocations = [
                PortfolioAllocation(
                    symbol=symbol,
                    weight=equal_weight,
                    target_amount=portfolio_value * equal_weight,
                    current_amount=portfolio_value * equal_weight,
                    risk_contribution=equal_weight,
                    expected_return=0.08,
                    confidence=0.5
                ) for symbol in symbols
            ]
            
            return PortfolioOptimization(
                portfolio_id=f"fallback_{int(time.time())}",
                optimization_method=method,
                allocations=allocations,
                expected_return=0.08,
                expected_volatility=0.15,
                sharpe_ratio=0.4,
                var_95=0.05,
                optimization_score=50.0,
                rebalance_frequency='monthly',
                timestamp=datetime.now()
            )
    
    def _estimate_covariance_matrix(self, symbols: List[str]) -> np.ndarray:
        """共分散行列推定"""
        returns_matrix = []
        
        for symbol in symbols:
            market_data = self.generate_market_data(symbol, 252)
            returns_matrix.append(market_data['returns'].values)
        
        returns_matrix = np.array(returns_matrix).T
        
        if HAS_SKLEARN:
            # Ledoit-Wolf収束推定器を使用
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_matrix).covariance_
        else:
            # 標準的な標本共分散行列
            cov_matrix = np.cov(returns_matrix.T)
        
        return cov_matrix
    
    def generate_risk_alerts(self, symbols: List[str]) -> List[RiskAlert]:
        """リスクアラート生成"""
        alerts = []
        
        for symbol in symbols:
            risk_metrics = self.calculate_risk_metrics(symbol)
            
            # VaRアラート
            if risk_metrics.var_95 > self.risk_thresholds['var_95']:
                severity = 'HIGH' if risk_metrics.var_95 > 0.10 else 'MEDIUM'
                alert = RiskAlert(
                    alert_id=f"var_{symbol}_{int(time.time())}",
                    alert_type='VAR_BREACH',
                    severity=severity,
                    symbol=symbol,
                    message=f"{symbol}のVaR(95%)が閾値を超過",
                    current_value=risk_metrics.var_95,
                    threshold_value=self.risk_thresholds['var_95'],
                    recommendation="ポジションサイズの縮小を検討",
                    timestamp=datetime.now()
                )
                alerts.append(alert)
            
            # ボラティリティアラート
            if risk_metrics.volatility > 0.40:  # 40%超
                alert = RiskAlert(
                    alert_id=f"vol_{symbol}_{int(time.time())}",
                    alert_type='VOLATILITY',
                    severity='HIGH',
                    symbol=symbol,
                    message=f"{symbol}のボラティリティが異常に高い",
                    current_value=risk_metrics.volatility,
                    threshold_value=0.30,
                    recommendation="リスク許容度の再評価が必要",
                    timestamp=datetime.now()
                )
                alerts.append(alert)
            
            # ドローダウンアラート
            if risk_metrics.max_drawdown > self.risk_thresholds['max_drawdown']:
                alert = RiskAlert(
                    alert_id=f"dd_{symbol}_{int(time.time())}",
                    alert_type='DRAWDOWN',
                    severity='CRITICAL',
                    symbol=symbol,
                    message=f"{symbol}で大きなドローダウンが発生",
                    current_value=risk_metrics.max_drawdown,
                    threshold_value=self.risk_thresholds['max_drawdown'],
                    recommendation="ストップロス設定の検討",
                    timestamp=datetime.now()
                )
                alerts.append(alert)
        
        self.alert_history.extend(alerts)
        self.total_alerts_generated += len(alerts)
        
        return alerts
    
    def stress_test_portfolio(self, portfolio: PortfolioOptimization,
                           stress_scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """ポートフォリオストレステスト"""
        stress_results = []
        
        default_scenarios = [
            {'name': '市場クラッシュ', 'market_shock': -0.20, 'vol_increase': 2.0},
            {'name': '金利上昇', 'rate_shock': 0.02, 'duration_risk': 1.5},
            {'name': 'デフレーション', 'inflation_shock': -0.03, 'real_rate_change': 0.01},
            {'name': '地政学リスク', 'market_shock': -0.15, 'vol_increase': 1.8},
            {'name': 'リーマンショック級', 'market_shock': -0.40, 'vol_increase': 3.0}
        ]
        
        scenarios = stress_scenarios if stress_scenarios else default_scenarios
        
        for scenario in scenarios:
            scenario_loss = 0.0
            scenario_details = []
            
            for allocation in portfolio.allocations:
                # シナリオ適用
                market_shock = scenario.get('market_shock', 0.0)
                vol_increase = scenario.get('vol_increase', 1.0)
                
                # 個別損失計算
                position_loss = allocation.target_amount * market_shock
                vol_adjusted_loss = position_loss * vol_increase
                
                scenario_loss += vol_adjusted_loss
                scenario_details.append({
                    'symbol': allocation.symbol,
                    'position_value': allocation.target_amount,
                    'loss': vol_adjusted_loss,
                    'loss_percent': vol_adjusted_loss / allocation.target_amount
                })
            
            stress_results.append({
                'scenario_name': scenario.get('name', 'Unnamed Scenario'),
                'total_loss': scenario_loss,
                'loss_percent': scenario_loss / sum(a.target_amount for a in portfolio.allocations),
                'details': scenario_details
            })
        
        return {
            'portfolio_id': portfolio.portfolio_id,
            'stress_test_results': stress_results,
            'worst_case_loss': min(result['total_loss'] for result in stress_results),
            'worst_case_scenario': min(stress_results, key=lambda x: x['total_loss'])['scenario_name'],
            'test_timestamp': datetime.now().isoformat()
        }
    
    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """リスクダッシュボードデータ取得"""
        # 最新のリスクメトリクス
        latest_metrics = {}
        for symbol, metrics_list in self.risk_metrics_history.items():
            if metrics_list:
                latest_metrics[symbol] = asdict(metrics_list[-1])
        
        # アクティブアラート
        active_alerts = [
            asdict(alert) for alert in self.alert_history[-20:]
        ]
        
        # ポートフォリオ統計
        portfolio_stats = {}
        if self.portfolio_history:
            latest_portfolio = self.portfolio_history[-1]
            portfolio_stats = {
                'expected_return': latest_portfolio.expected_return,
                'expected_volatility': latest_portfolio.expected_volatility,
                'sharpe_ratio': latest_portfolio.sharpe_ratio,
                'var_95': latest_portfolio.var_95,
                'optimization_score': latest_portfolio.optimization_score
            }
        
        return {
            'risk_metrics': latest_metrics,
            'active_alerts': active_alerts,
            'portfolio_stats': portfolio_stats,
            'system_stats': {
                'total_risk_calculations': self.total_risk_calculations,
                'total_optimizations': self.total_optimizations,
                'total_alerts_generated': self.total_alerts_generated,
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
            },
            'last_update': datetime.now().isoformat()
        }
    
    def get_risk_report(self, symbols: List[str]) -> Dict[str, Any]:
        """包括的リスクレポート生成"""
        risk_summary = {}
        
        for symbol in symbols:
            risk_metrics = self.calculate_risk_metrics(symbol)
            risk_summary[symbol] = asdict(risk_metrics)
        
        # ポートフォリオ最適化
        portfolio = self.optimize_portfolio(symbols, 'ai_enhanced')
        
        # リスクアラート
        alerts = self.generate_risk_alerts(symbols)
        
        # ストレステスト
        stress_test = self.stress_test_portfolio(portfolio, [])
        
        return {
            'report_id': f"risk_report_{int(time.time())}",
            'symbols': symbols,
            'individual_risk_metrics': risk_summary,
            'portfolio_optimization': asdict(portfolio),
            'risk_alerts': [asdict(alert) for alert in alerts],
            'stress_test_results': stress_test,
            'recommendations': self._generate_risk_recommendations(risk_summary, alerts),
            'report_timestamp': datetime.now().isoformat()
        }
    
    def _generate_risk_recommendations(self, risk_summary: Dict[str, Any],
                                    alerts: List[RiskAlert]) -> List[str]:
        """リスク推奨事項生成"""
        recommendations = []
        
        # 高リスク銘柄の識別
        high_risk_symbols = []
        for symbol, metrics in risk_summary.items():
            if metrics['risk_score'] > 70:
                high_risk_symbols.append(symbol)
        
        if high_risk_symbols:
            recommendations.append(
                f"高リスク銘柄 ({', '.join(high_risk_symbols)}) の比重見直しを推奨"
            )
        
        # アラートベースの推奨
        alert_counts = defaultdict(int)
        for alert in alerts:
            alert_counts[alert.alert_type] += 1
        
        if alert_counts['VAR_BREACH'] > 0:
            recommendations.append("VaR超過銘柄あり：ポジションサイズの調整が必要")
        
        if alert_counts['VOLATILITY'] > 0:
            recommendations.append("高ボラティリティ警告：リスク許容度の再評価を推奨")
        
        # ポートフォリオレベルの推奨
        recommendations.append("月次リバランシングの実行を推奨")
        recommendations.append("ストレステスト結果に基づくヘッジ戦略の検討")
        
        return recommendations


# グローバルインスタンス
risk_management_ai = RiskManagementAI()


async def test_risk_management_ai():
    """リスク管理AIテスト"""
    print("=== Risk Management AI Test ===")
    
    # テスト銘柄
    test_symbols = ['7203', '8306', '9984', '6758', '4689']
    
    # 個別リスク分析
    print("1. Individual Risk Analysis:")
    for symbol in test_symbols:
        risk_metrics = risk_management_ai.calculate_risk_metrics(symbol)
        print(f"{symbol}: VaR95={risk_metrics.var_95:.3f}, "
              f"Vol={risk_metrics.volatility:.3f}, Score={risk_metrics.risk_score:.1f}")
    
    # ポートフォリオ最適化
    print("\n2. Portfolio Optimization:")
    portfolio = risk_management_ai.optimize_portfolio(test_symbols, 'ai_enhanced', 1000000)
    print(f"Method: {portfolio.optimization_method}")
    print(f"Expected Return: {portfolio.expected_return:.3f}")
    print(f"Expected Vol: {portfolio.expected_volatility:.3f}")
    print(f"Sharpe Ratio: {portfolio.sharpe_ratio:.3f}")
    
    print("Allocations:")
    for allocation in portfolio.allocations:
        print(f"  {allocation.symbol}: {allocation.weight:.3f} ({allocation.target_amount:,.0f}円)")
    
    # リスクアラート
    print("\n3. Risk Alerts:")
    alerts = risk_management_ai.generate_risk_alerts(test_symbols)
    for alert in alerts[:3]:  # 最初の3つのアラート
        print(f"  {alert.severity}: {alert.message}")
    
    # ストレステスト
    print("\n4. Stress Test:")
    stress_test = risk_management_ai.stress_test_portfolio(portfolio, [])
    print(f"Worst Case Loss: {stress_test['worst_case_loss']:,.0f}円")
    print(f"Worst Scenario: {stress_test['worst_case_scenario']}")
    
    # システム統計
    print(f"\n5. System Statistics:")
    dashboard_data = risk_management_ai.get_risk_dashboard_data()
    stats = dashboard_data['system_stats']
    print(f"Risk Calculations: {stats['total_risk_calculations']}")
    print(f"Optimizations: {stats['total_optimizations']}")
    print(f"Alerts Generated: {stats['total_alerts_generated']}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(test_risk_management_ai())