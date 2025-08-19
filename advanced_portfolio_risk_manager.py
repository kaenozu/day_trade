#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Portfolio Risk Manager - 高度ポートフォリオリスク管理システム
Issue #941 対応: リスク管理強化、ポートフォリオ最適化とパフォーマンス分析
"""

import time
import json
import numpy as np
import pandas as pd
import sqlite3
import threading
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import uuid
import warnings

# 数値計算・最適化
try:
    from scipy.optimize import minimize, NonlinearConstraint
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# カスタムモジュール
try:
    from swing_trading_scheduler import swing_trading_scheduler, HoldingStatus, PurchaseStrategy
    HAS_SWING_SCHEDULER = True
except ImportError:
    HAS_SWING_SCHEDULER = False

try:
    from enhanced_ml_ensemble_system import enhanced_ml_ensemble
    HAS_ML_ENSEMBLE = True
except ImportError:
    HAS_ML_ENSEMBLE = False

try:
    from advanced_feature_engineering_system import advanced_feature_engineering
    HAS_FEATURE_ENGINEERING = True
except ImportError:
    HAS_FEATURE_ENGINEERING = False

try:
    from audit_logger import audit_logger
    HAS_AUDIT_LOGGER = True
except ImportError:
    HAS_AUDIT_LOGGER = False

warnings.filterwarnings('ignore')


class RiskLevel(Enum):
    """リスクレベル"""
    LOW = 1      # 低リスク
    MODERATE = 2 # 中リスク
    HIGH = 3     # 高リスク
    EXTREME = 4  # 極高リスク


class OptimizationObjective(Enum):
    """最適化目標"""
    MAX_RETURN = "max_return"           # リターン最大化
    MIN_RISK = "min_risk"              # リスク最小化
    MAX_SHARPE = "max_sharpe"          # シャープレシオ最大化
    MAX_DIVERSIFICATION = "max_div"     # 分散化最大化


@dataclass
class RiskMetrics:
    """リスクメトリクス"""
    portfolio_value: float
    var_1d_95: float           # 1日95%VaR
    var_5d_95: float           # 5日95%VaR
    expected_shortfall: float   # 条件付きVaR
    max_drawdown: float        # 最大ドローダウン
    volatility: float          # ボラティリティ
    beta: float               # ベータ値
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sector_concentration: Dict[str, float] = field(default_factory=dict)
    calculated_at: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioOptimizationResult:
    """ポートフォリオ最適化結果"""
    current_allocation: Dict[str, float]
    optimal_allocation: Dict[str, float]
    rebalance_recommendations: List[Dict[str, Any]]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    optimization_objective: OptimizationObjective
    constraints_satisfied: bool
    optimization_date: datetime = field(default_factory=datetime.now)


@dataclass
class StopLossRecommendation:
    """ストップロス推奨"""
    symbol: str
    current_stop_loss: float
    recommended_stop_loss: float
    reason: str
    urgency: int  # 1-5 (高-低)
    potential_loss: float


@dataclass
class PositionSizingRecommendation:
    """ポジションサイジング推奨"""
    symbol: str
    current_position: float
    recommended_position: float
    reason: str
    risk_adjusted: bool


class AdvancedPortfolioRiskManager:
    """高度ポートフォリオリスク管理システム"""
    
    def __init__(self, db_path: str = "data/portfolio_risk.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # データベース初期化
        self._initialize_database()
        
        # 設定
        self.max_position_weight = 0.20      # 最大ポジション比重 (20%)
        self.max_sector_weight = 0.30        # 最大セクター比重 (30%)
        self.max_total_risk = 0.15          # 最大ポートフォリオリスク (15%)
        self.target_sharpe_ratio = 1.0       # 目標シャープレシオ
        
        # キャッシュ
        self.risk_metrics_cache = {}
        self.optimization_cache = {}
        
        # ロック
        self._lock = threading.Lock()
        
        print("Advanced Portfolio Risk Manager initialized")
    
    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # リスクメトリクス履歴テーブル
            conn.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics_history (
                    id TEXT PRIMARY KEY,
                    portfolio_value REAL NOT NULL,
                    var_1d_95 REAL NOT NULL,
                    var_5d_95 REAL NOT NULL,
                    expected_shortfall REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    volatility REAL NOT NULL,
                    beta REAL NOT NULL,
                    sector_concentration TEXT,
                    correlation_matrix TEXT,
                    calculated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 最適化結果履歴テーブル
            conn.execute('''
                CREATE TABLE IF NOT EXISTS optimization_history (
                    id TEXT PRIMARY KEY,
                    current_allocation TEXT NOT NULL,
                    optimal_allocation TEXT NOT NULL,
                    rebalance_recommendations TEXT NOT NULL,
                    expected_return REAL NOT NULL,
                    expected_risk REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    optimization_objective TEXT NOT NULL,
                    constraints_satisfied BOOLEAN NOT NULL,
                    optimization_date DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # リスク調整推奨テーブル
            conn.execute('''
                CREATE TABLE IF NOT EXISTS risk_recommendations (
                    id TEXT PRIMARY KEY,
                    recommendation_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    current_value REAL,
                    recommended_value REAL,
                    reason TEXT NOT NULL,
                    urgency INTEGER NOT NULL,
                    potential_impact REAL,
                    is_implemented BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # インデックス作成
            conn.execute('CREATE INDEX IF NOT EXISTS idx_risk_metrics_date ON risk_metrics_history(calculated_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_optimization_date ON optimization_history(optimization_date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_recommendations_urgency ON risk_recommendations(urgency)')
            
            conn.commit()
    
    def calculate_portfolio_risk_metrics(self) -> RiskMetrics:
        """ポートフォリオリスクメトリクスを計算"""
        try:
            if not HAS_SWING_SCHEDULER:
                raise ValueError("Swing Trading Scheduler not available")
            
            # アクティブポジション取得
            positions = swing_trading_scheduler.get_monitoring_list()
            active_positions = [p for p in positions if p['status'] not in ['sold']]
            
            if not active_positions:
                return RiskMetrics(
                    portfolio_value=0.0,
                    var_1d_95=0.0,
                    var_5d_95=0.0,
                    expected_shortfall=0.0,
                    max_drawdown=0.0,
                    volatility=0.0,
                    beta=0.0
                )
            
            # ポートフォリオ価値
            total_value = sum(p['current_value'] for p in active_positions)
            
            # 個別銘柄データ取得と分析
            position_data = {}
            returns_data = []
            
            for position in active_positions:
                symbol = position['symbol']
                weight = position['current_value'] / total_value if total_value > 0 else 0
                
                # 過去データシミュレーション（実際の実装では株価APIから取得）
                returns = self._simulate_returns(symbol, days=30)
                
                position_data[symbol] = {
                    'weight': weight,
                    'returns': returns,
                    'current_value': position['current_value'],
                    'volatility': np.std(returns),
                    'sector': self._get_sector(symbol)  # セクター情報
                }
                
                returns_data.append(returns)
            
            # ポートフォリオリターン計算
            portfolio_returns = np.zeros(len(returns_data[0]))
            for symbol, data in position_data.items():
                portfolio_returns += data['returns'] * data['weight']
            
            # リスクメトリクス計算
            portfolio_volatility = np.std(portfolio_returns)
            
            # VaR計算 (95%信頼区間)
            var_1d_95 = np.percentile(portfolio_returns, 5) * total_value
            var_5d_95 = var_1d_95 * np.sqrt(5)  # スケーリング
            
            # 条件付きVaR (Expected Shortfall)
            es_threshold = np.percentile(portfolio_returns, 5)
            expected_shortfall = np.mean(portfolio_returns[portfolio_returns <= es_threshold]) * total_value
            
            # 最大ドローダウン計算
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # ベータ計算（市場との相関、ここでは簡易計算）
            market_returns = self._simulate_market_returns(len(portfolio_returns))
            beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
            
            # 相関行列計算
            correlation_matrix = {}
            symbols = list(position_data.keys())
            for i, symbol1 in enumerate(symbols):
                correlation_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i <= j:
                        corr = np.corrcoef(position_data[symbol1]['returns'], 
                                         position_data[symbol2]['returns'])[0, 1]
                        correlation_matrix[symbol1][symbol2] = float(corr)
                        if i != j:
                            if symbol2 not in correlation_matrix:
                                correlation_matrix[symbol2] = {}
                            correlation_matrix[symbol2][symbol1] = float(corr)
            
            # セクター集中度
            sector_concentration = {}
            for symbol, data in position_data.items():
                sector = data['sector']
                if sector not in sector_concentration:
                    sector_concentration[sector] = 0
                sector_concentration[sector] += data['weight']
            
            # リスクメトリクス作成
            risk_metrics = RiskMetrics(
                portfolio_value=total_value,
                var_1d_95=var_1d_95,
                var_5d_95=var_5d_95,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                volatility=portfolio_volatility,
                beta=beta,
                correlation_matrix=correlation_matrix,
                sector_concentration=sector_concentration
            )
            
            # データベースに保存
            self._save_risk_metrics(risk_metrics)
            
            # キャッシュ更新
            self.risk_metrics_cache[datetime.now().date()] = risk_metrics
            
            return risk_metrics
            
        except Exception as e:
            print(f"リスクメトリクス計算エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"context": "risk_metrics_calculation"})
            raise
    
    def _simulate_returns(self, symbol: str, days: int = 30) -> np.ndarray:
        """株式リターンをシミュレート（実際の実装では過去データを使用）"""
        # シンボルベースの疑似リターン生成
        np.random.seed(hash(symbol) % 2**32)
        
        # 基本パラメータ
        annual_return = 0.08  # 8%年間リターン
        annual_volatility = 0.20  # 20%年間ボラティリティ
        
        # シンボル固有の調整
        symbol_factor = (hash(symbol) % 100) / 100 - 0.5  # -0.5 to 0.5
        annual_return += symbol_factor * 0.1  # ±5%調整
        annual_volatility += abs(symbol_factor) * 0.1  # ボラティリティ調整
        
        # 日次リターン生成
        daily_return = annual_return / 252
        daily_volatility = annual_volatility / np.sqrt(252)
        
        returns = np.random.normal(daily_return, daily_volatility, days)
        
        # トレンド追加（時系列相関）
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]  # 軽い自己相関
        
        return returns
    
    def _simulate_market_returns(self, days: int) -> np.ndarray:
        """市場リターンをシミュレート"""
        np.random.seed(42)  # 固定シード
        
        annual_return = 0.06  # 6%年間リターン（市場）
        annual_volatility = 0.15  # 15%年間ボラティリティ
        
        daily_return = annual_return / 252
        daily_volatility = annual_volatility / np.sqrt(252)
        
        return np.random.normal(daily_return, daily_volatility, days)
    
    def _get_sector(self, symbol: str) -> str:
        """セクター情報を取得（簡易実装）"""
        # 実際の実装では企業データベースから取得
        sectors = [
            "Technology", "Finance", "Healthcare", "Consumer", 
            "Industrial", "Energy", "Materials", "Utilities"
        ]
        
        sector_index = hash(symbol) % len(sectors)
        return sectors[sector_index]
    
    def _save_risk_metrics(self, metrics: RiskMetrics):
        """リスクメトリクスを保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO risk_metrics_history 
                (id, portfolio_value, var_1d_95, var_5d_95, expected_shortfall,
                 max_drawdown, volatility, beta, sector_concentration, 
                 correlation_matrix, calculated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                metrics.portfolio_value,
                metrics.var_1d_95,
                metrics.var_5d_95,
                metrics.expected_shortfall,
                metrics.max_drawdown,
                metrics.volatility,
                metrics.beta,
                json.dumps(metrics.sector_concentration, ensure_ascii=False),
                json.dumps(metrics.correlation_matrix, ensure_ascii=False),
                metrics.calculated_at
            ))
            conn.commit()
    
    def optimize_portfolio(self, 
                          objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
                          risk_tolerance: float = 0.15) -> PortfolioOptimizationResult:
        """ポートフォリオ最適化"""
        try:
            if not HAS_SWING_SCHEDULER:
                raise ValueError("Swing Trading Scheduler not available")
            
            if not HAS_SCIPY:
                print("SciPy not available - using simplified optimization")
                return self._simplified_optimization(objective)
            
            # 現在のポジション取得
            positions = swing_trading_scheduler.get_monitoring_list()
            active_positions = [p for p in positions if p['status'] not in ['sold']]
            
            if len(active_positions) < 2:
                raise ValueError("最適化には2つ以上のポジションが必要です")
            
            # 現在のアロケーション
            total_value = sum(p['current_value'] for p in active_positions)
            current_allocation = {}
            symbols = []
            returns_data = []
            
            for position in active_positions:
                symbol = position['symbol']
                weight = position['current_value'] / total_value if total_value > 0 else 0
                current_allocation[symbol] = weight
                symbols.append(symbol)
                
                # リターンデータ
                returns = self._simulate_returns(symbol, days=60)
                returns_data.append(returns)
            
            # リターン・リスク行列
            returns_matrix = np.array(returns_data).T
            mean_returns = np.mean(returns_matrix, axis=0)
            cov_matrix = np.cov(returns_matrix.T)
            
            # 最適化実行
            n_assets = len(symbols)
            
            # 初期値（等重み）
            x0 = np.ones(n_assets) / n_assets
            
            # 制約条件
            constraints = []
            
            # 重み和が1
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1.0
            })
            
            # 個別重み制限
            bounds = [(0.01, self.max_position_weight) for _ in range(n_assets)]
            
            # 目的関数
            if objective == OptimizationObjective.MAX_RETURN:
                obj_func = lambda x: -np.dot(x, mean_returns)
            elif objective == OptimizationObjective.MIN_RISK:
                obj_func = lambda x: np.dot(x.T, np.dot(cov_matrix, x))
            elif objective == OptimizationObjective.MAX_SHARPE:
                obj_func = lambda x: -(np.dot(x, mean_returns) / np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))))
            else:  # MAX_DIVERSIFICATION
                obj_func = lambda x: -self._diversification_ratio(x, cov_matrix)
            
            # 最適化実行
            result = minimize(
                obj_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                optimal_allocation = {symbols[i]: float(optimal_weights[i]) for i in range(n_assets)}
                
                # リバランス推奨
                rebalance_recommendations = []
                for i, symbol in enumerate(symbols):
                    current_weight = current_allocation[symbol]
                    optimal_weight = optimal_weights[i]
                    difference = optimal_weight - current_weight
                    
                    if abs(difference) > 0.05:  # 5%以上の差異
                        rebalance_recommendations.append({
                            'symbol': symbol,
                            'current_weight': current_weight,
                            'optimal_weight': optimal_weight,
                            'action': 'increase' if difference > 0 else 'decrease',
                            'amount_change': abs(difference),
                            'value_change': abs(difference) * total_value
                        })
                
                # 期待リターン・リスク計算
                expected_return = np.dot(optimal_weights, mean_returns) * 252  # 年率化
                expected_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))) * np.sqrt(252)
                sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0
                
                optimization_result = PortfolioOptimizationResult(
                    current_allocation=current_allocation,
                    optimal_allocation=optimal_allocation,
                    rebalance_recommendations=rebalance_recommendations,
                    expected_return=expected_return,
                    expected_risk=expected_risk,
                    sharpe_ratio=sharpe_ratio,
                    optimization_objective=objective,
                    constraints_satisfied=True
                )
                
                # データベースに保存
                self._save_optimization_result(optimization_result)
                
                return optimization_result
                
            else:
                raise ValueError(f"最適化に失敗しました: {result.message}")
                
        except Exception as e:
            print(f"ポートフォリオ最適化エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"context": "portfolio_optimization"})
            raise
    
    def _diversification_ratio(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """分散化比率を計算"""
        weighted_vol = np.sum(weights * np.sqrt(np.diag(cov_matrix)))
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return weighted_vol / portfolio_vol if portfolio_vol > 0 else 0
    
    def _simplified_optimization(self, objective: OptimizationObjective) -> PortfolioOptimizationResult:
        """簡易最適化（SciPyが利用できない場合）"""
        positions = swing_trading_scheduler.get_monitoring_list()
        active_positions = [p for p in positions if p['status'] not in ['sold']]
        
        total_value = sum(p['current_value'] for p in active_positions)
        current_allocation = {}
        
        for position in active_positions:
            symbol = position['symbol']
            current_allocation[symbol] = position['current_value'] / total_value
        
        # 簡易的な等重み最適化
        n_positions = len(active_positions)
        optimal_weight = 1.0 / n_positions
        optimal_allocation = {p['symbol']: optimal_weight for p in active_positions}
        
        # リバランス推奨
        rebalance_recommendations = []
        for position in active_positions:
            symbol = position['symbol']
            current_weight = current_allocation[symbol]
            difference = optimal_weight - current_weight
            
            if abs(difference) > 0.05:
                rebalance_recommendations.append({
                    'symbol': symbol,
                    'current_weight': current_weight,
                    'optimal_weight': optimal_weight,
                    'action': 'increase' if difference > 0 else 'decrease',
                    'amount_change': abs(difference),
                    'value_change': abs(difference) * total_value
                })
        
        return PortfolioOptimizationResult(
            current_allocation=current_allocation,
            optimal_allocation=optimal_allocation,
            rebalance_recommendations=rebalance_recommendations,
            expected_return=0.08,  # 仮定値
            expected_risk=0.15,    # 仮定値
            sharpe_ratio=0.53,     # 仮定値
            optimization_objective=objective,
            constraints_satisfied=True
        )
    
    def _save_optimization_result(self, result: PortfolioOptimizationResult):
        """最適化結果を保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO optimization_history 
                (id, current_allocation, optimal_allocation, rebalance_recommendations,
                 expected_return, expected_risk, sharpe_ratio, optimization_objective,
                 constraints_satisfied, optimization_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                json.dumps(result.current_allocation, ensure_ascii=False),
                json.dumps(result.optimal_allocation, ensure_ascii=False),
                json.dumps(result.rebalance_recommendations, ensure_ascii=False, default=str),
                result.expected_return,
                result.expected_risk,
                result.sharpe_ratio,
                result.optimization_objective.value,
                result.constraints_satisfied,
                result.optimization_date
            ))
            conn.commit()
    
    def generate_stop_loss_recommendations(self) -> List[StopLossRecommendation]:
        """動的ストップロス推奨を生成"""
        try:
            if not HAS_SWING_SCHEDULER:
                return []
            
            recommendations = []
            positions = swing_trading_scheduler.get_monitoring_list()
            
            for position in positions:
                if position['status'] in ['sold']:
                    continue
                
                symbol = position['symbol']
                purchase_record = swing_trading_scheduler.get_purchase_record(position['purchase_id'])
                
                if not purchase_record:
                    continue
                
                current_price = position['current_price'] or purchase_record.purchase_price
                current_stop_loss = purchase_record.stop_loss_percent / 100
                
                # ボラティリティベースのストップロス計算
                returns = self._simulate_returns(symbol, days=20)
                volatility = np.std(returns)
                
                # 動的ストップロス（ボラティリティの2倍）
                dynamic_stop_loss = -2.5 * volatility
                
                # 現在の損益状況を考慮
                current_pl_percent = (current_price - purchase_record.purchase_price) / purchase_record.purchase_price
                
                if current_pl_percent > 0.10:  # 10%以上の利益
                    # 利益確保のためトレーリングストップ
                    recommended_stop_loss = max(dynamic_stop_loss, current_pl_percent * 0.5 - 0.05)
                    reason = "利益確保のためトレーリングストップ"
                    urgency = 3
                elif current_pl_percent < current_stop_loss:
                    # 損失拡大の恐れ
                    recommended_stop_loss = max(dynamic_stop_loss, current_stop_loss * 0.8)
                    reason = "損失拡大防止のため損切りライン引き上げ"
                    urgency = 2
                else:
                    # 通常のボラティリティ調整
                    recommended_stop_loss = dynamic_stop_loss
                    reason = "ボラティリティに基づく調整"
                    urgency = 4
                
                # 大きな変更がある場合のみ推奨
                if abs(recommended_stop_loss - current_stop_loss) > 0.02:  # 2%以上の変更
                    potential_loss = (recommended_stop_loss * current_price * position['current_shares'])
                    
                    recommendations.append(StopLossRecommendation(
                        symbol=symbol,
                        current_stop_loss=current_stop_loss,
                        recommended_stop_loss=recommended_stop_loss,
                        reason=reason,
                        urgency=urgency,
                        potential_loss=potential_loss
                    ))
            
            # データベースに保存
            for rec in recommendations:
                self._save_recommendation("stop_loss", rec.symbol, rec.current_stop_loss, 
                                        rec.recommended_stop_loss, rec.reason, rec.urgency, 
                                        rec.potential_loss)
            
            return recommendations
            
        except Exception as e:
            print(f"ストップロス推奨生成エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"context": "stop_loss_recommendations"})
            return []
    
    def generate_position_sizing_recommendations(self) -> List[PositionSizingRecommendation]:
        """ポジションサイジング推奨を生成"""
        try:
            if not HAS_SWING_SCHEDULER:
                return []
            
            recommendations = []
            positions = swing_trading_scheduler.get_monitoring_list()
            total_value = sum(p['current_value'] for p in positions if p['status'] not in ['sold'])
            
            if total_value == 0:
                return recommendations
            
            for position in positions:
                if position['status'] in ['sold']:
                    continue
                
                symbol = position['symbol']
                current_weight = position['current_value'] / total_value
                
                # リスク調整ポジションサイズ計算
                returns = self._simulate_returns(symbol, days=30)
                volatility = np.std(returns)
                
                # Kelly criterionベースの最適サイズ
                expected_return = np.mean(returns)
                if volatility > 0:
                    kelly_fraction = expected_return / (volatility ** 2)
                    # 保守的に1/4ケリーを使用
                    optimal_weight = max(0.05, min(0.25, kelly_fraction * 0.25))
                else:
                    optimal_weight = 0.10  # デフォルト10%
                
                # 制約チェック
                if optimal_weight > self.max_position_weight:
                    optimal_weight = self.max_position_weight
                    reason = "最大ポジション制限による調整"
                elif abs(optimal_weight - current_weight) < 0.03:
                    continue  # 小さな変更は無視
                elif optimal_weight > current_weight:
                    reason = "リスク調整後の増加推奨"
                else:
                    reason = "リスク調整後の減少推奨"
                
                recommendations.append(PositionSizingRecommendation(
                    symbol=symbol,
                    current_position=current_weight,
                    recommended_position=optimal_weight,
                    reason=reason,
                    risk_adjusted=True
                ))
                
                # データベースに保存
                self._save_recommendation("position_sizing", symbol, current_weight, 
                                        optimal_weight, reason, 3, 0.0)
            
            return recommendations
            
        except Exception as e:
            print(f"ポジションサイジング推奨生成エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"context": "position_sizing_recommendations"})
            return []
    
    def _save_recommendation(self, rec_type: str, symbol: str, current_value: float,
                           recommended_value: float, reason: str, urgency: int,
                           potential_impact: float):
        """推奨をデータベースに保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO risk_recommendations 
                (id, recommendation_type, symbol, current_value, recommended_value,
                 reason, urgency, potential_impact, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                rec_type,
                symbol,
                current_value,
                recommended_value,
                reason,
                urgency,
                potential_impact,
                datetime.now()
            ))
            conn.commit()
    
    def get_portfolio_risk_summary(self) -> Dict[str, Any]:
        """ポートフォリオリスクサマリーを取得"""
        try:
            # 最新リスクメトリクス
            risk_metrics = self.calculate_portfolio_risk_metrics()
            
            # リスクレベル判定
            risk_level = self._assess_risk_level(risk_metrics)
            
            # 推奨アクション生成
            stop_loss_recs = self.generate_stop_loss_recommendations()
            position_sizing_recs = self.generate_position_sizing_recommendations()
            
            # 最新最適化結果
            try:
                optimization_result = self.optimize_portfolio()
            except Exception:
                optimization_result = None
            
            return {
                'risk_summary': {
                    'risk_level': risk_level.name,
                    'portfolio_value': risk_metrics.portfolio_value,
                    'daily_var_95': risk_metrics.var_1d_95,
                    'weekly_var_95': risk_metrics.var_5d_95,
                    'expected_shortfall': risk_metrics.expected_shortfall,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'volatility': risk_metrics.volatility,
                    'beta': risk_metrics.beta,
                    'sector_concentration': risk_metrics.sector_concentration
                },
                'recommendations': {
                    'stop_loss_adjustments': len(stop_loss_recs),
                    'position_sizing_adjustments': len(position_sizing_recs),
                    'high_priority_actions': len([r for r in stop_loss_recs if r.urgency <= 2]) + 
                                           len([r for r in position_sizing_recs])
                },
                'optimization': {
                    'available': optimization_result is not None,
                    'rebalance_needed': len(optimization_result.rebalance_recommendations) > 0 
                                      if optimization_result else False,
                    'expected_sharpe_improvement': optimization_result.sharpe_ratio - 0.5 
                                                 if optimization_result else 0  # 現在値仮定
                },
                'alerts': {
                    'high_concentration_warning': max(risk_metrics.sector_concentration.values()) > 0.4,
                    'high_correlation_warning': self._check_high_correlation(risk_metrics.correlation_matrix),
                    'excessive_risk_warning': risk_level in [RiskLevel.HIGH, RiskLevel.EXTREME]
                }
            }
            
        except Exception as e:
            print(f"リスクサマリー取得エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"context": "portfolio_risk_summary"})
            return {'error': str(e)}
    
    def _assess_risk_level(self, risk_metrics: RiskMetrics) -> RiskLevel:
        """リスクレベルを評価"""
        risk_score = 0
        
        # ボラティリティチェック
        if risk_metrics.volatility > 0.25:
            risk_score += 2
        elif risk_metrics.volatility > 0.15:
            risk_score += 1
        
        # VaRチェック
        var_ratio = abs(risk_metrics.var_1d_95) / risk_metrics.portfolio_value if risk_metrics.portfolio_value > 0 else 0
        if var_ratio > 0.05:
            risk_score += 2
        elif var_ratio > 0.02:
            risk_score += 1
        
        # セクター集中度チェック
        max_sector_weight = max(risk_metrics.sector_concentration.values()) if risk_metrics.sector_concentration else 0
        if max_sector_weight > 0.5:
            risk_score += 2
        elif max_sector_weight > 0.3:
            risk_score += 1
        
        # ドローダウンチェック
        if risk_metrics.max_drawdown < -0.20:
            risk_score += 2
        elif risk_metrics.max_drawdown < -0.10:
            risk_score += 1
        
        # リスクレベル判定
        if risk_score >= 6:
            return RiskLevel.EXTREME
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _check_high_correlation(self, correlation_matrix: Dict[str, Dict[str, float]]) -> bool:
        """高相関警告をチェック"""
        if not correlation_matrix:
            return False
        
        symbols = list(correlation_matrix.keys())
        high_corr_count = 0
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                corr = correlation_matrix.get(symbol1, {}).get(symbol2, 0)
                if abs(corr) > 0.8:  # 80%以上の相関
                    high_corr_count += 1
        
        return high_corr_count > len(symbols) * 0.3  # 30%以上のペアが高相関


# グローバルインスタンス
portfolio_risk_manager = AdvancedPortfolioRiskManager()


if __name__ == "__main__":
    # テスト実行
    print("Advanced Portfolio Risk Manager テスト開始")
    
    # リスクマネージャーインスタンス作成
    risk_manager = AdvancedPortfolioRiskManager()
    
    print("\n1. リスクメトリクス計算テスト")
    try:
        risk_metrics = risk_manager.calculate_portfolio_risk_metrics()
        print(f"   ポートフォリオ価値: ¥{risk_metrics.portfolio_value:,.0f}")
        print(f"   日次VaR(95%): ¥{risk_metrics.var_1d_95:,.0f}")
        print(f"   ボラティリティ: {risk_metrics.volatility:.2%}")
        print(f"   ベータ: {risk_metrics.beta:.2f}")
        print(f"   セクター集中度: {risk_metrics.sector_concentration}")
    except Exception as e:
        print(f"   エラー: {e}")
    
    print("\n2. ポートフォリオ最適化テスト")
    try:
        optimization = risk_manager.optimize_portfolio(OptimizationObjective.MAX_SHARPE)
        print(f"   期待リターン: {optimization.expected_return:.2%}")
        print(f"   期待リスク: {optimization.expected_risk:.2%}")
        print(f"   シャープレシオ: {optimization.sharpe_ratio:.2f}")
        print(f"   リバランス推奨数: {len(optimization.rebalance_recommendations)}")
    except Exception as e:
        print(f"   エラー: {e}")
    
    print("\n3. ストップロス推奨テスト")
    try:
        stop_loss_recs = risk_manager.generate_stop_loss_recommendations()
        print(f"   ストップロス調整推奨数: {len(stop_loss_recs)}")
        for rec in stop_loss_recs[:3]:  # 最初の3件
            print(f"   {rec.symbol}: {rec.current_stop_loss:.1%} -> {rec.recommended_stop_loss:.1%} ({rec.reason})")
    except Exception as e:
        print(f"   エラー: {e}")
    
    print("\n4. ポジションサイジング推奨テスト")
    try:
        position_recs = risk_manager.generate_position_sizing_recommendations()
        print(f"   ポジション調整推奨数: {len(position_recs)}")
        for rec in position_recs[:3]:  # 最初の3件
            print(f"   {rec.symbol}: {rec.current_position:.1%} -> {rec.recommended_position:.1%}")
    except Exception as e:
        print(f"   エラー: {e}")
    
    print("\n5. ポートフォリオリスクサマリーテスト")
    try:
        risk_summary = risk_manager.get_portfolio_risk_summary()
        print(json.dumps(risk_summary, ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        print(f"   エラー: {e}")
    
    print("テスト完了 ✅")