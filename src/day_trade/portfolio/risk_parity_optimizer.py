#!/usr/bin/env python3
"""
リスクパリティ最適化システム
等リスク寄与度ポートフォリオ構築・リスクバジェット配分

Features:
- 等リスク寄与度最適化
- カスタムリスクバジェット
- 階層リスクパリティ
- レバレッジ制御
- リスク寄与度分析
- 動的リバランシング
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# オプショナル依存関係
SCIPY_AVAILABLE = False
try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.linalg import sqrtm
    from scipy.optimize import minimize, minimize_scalar
    from scipy.spatial.distance import squareform

    SCIPY_AVAILABLE = True
except ImportError:
    pass


class RiskParityMethod(Enum):
    """リスクパリティ手法"""

    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"
    RISK_BUDGETING = "risk_budgeting"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    CONSTRAINED_RISK_PARITY = "constrained_risk_parity"


class OptimizationObjective(Enum):
    """最適化目標"""

    MINIMIZE_CONCENTRATION = "minimize_concentration"
    MAXIMIZE_DIVERSIFICATION = "maximize_diversification"
    TARGET_VOLATILITY = "target_volatility"
    MAXIMIZE_RISK_ADJUSTED_RETURN = "maximize_risk_adjusted_return"


@dataclass
class RiskParityConfig:
    """リスクパリティ設定"""

    # 基本設定
    method: RiskParityMethod = RiskParityMethod.EQUAL_RISK_CONTRIBUTION
    objective: OptimizationObjective = OptimizationObjective.MINIMIZE_CONCENTRATION

    # 制約条件
    min_weight: float = 0.01  # 1%
    max_weight: float = 0.50  # 50%
    target_volatility: float = 0.10  # 10%
    max_leverage: float = 1.0  # レバレッジなし

    # リスクバジェット（カスタム配分）
    risk_budgets: Dict[str, float] = None

    # 階層クラスタリング設定
    clustering_method: str = "ward"
    distance_metric: str = "euclidean"
    n_clusters: int = 5

    # 最適化設定
    tolerance: float = 1e-8
    max_iterations: int = 1000

    # ボラティリティ推定
    volatility_window: int = 252
    correlation_window: int = 252

    # 動的設定
    rebalancing_threshold: float = 0.05  # 5%
    lookback_period: int = 252


@dataclass
class RiskBudgetAllocation:
    """リスクバジェット配分"""

    asset: str
    risk_budget: float
    actual_risk_contribution: float
    weight: float
    volatility: float
    marginal_risk_contribution: float


@dataclass
class OptimizationResult:
    """最適化結果"""

    weights: Dict[str, float]
    risk_contributions: Dict[str, float]
    risk_budgets: Dict[str, float]
    portfolio_volatility: float
    diversification_ratio: float
    concentration_index: float
    optimization_success: bool
    iterations: int
    objective_value: float

    # 詳細メトリクス
    effective_number_assets: float
    risk_concentration_hhi: float
    largest_risk_contribution: float
    smallest_risk_contribution: float


class RiskParityOptimizer:
    """リスクパリティ最適化システム"""

    def __init__(self, config: RiskParityConfig = None):
        self.config = config or RiskParityConfig()

        # 状態管理
        self.current_weights = None
        self.current_covariance = None
        self.last_optimization = None

        # 履歴管理
        self.optimization_history = []
        self.rebalancing_history = []

        # パフォーマンス追跡
        self.performance_metrics = {
            "optimization_times": [],
            "convergence_history": [],
            "risk_concentration_history": [],
        }

        logger.info("リスクパリティ最適化システム初期化完了")

    def optimize(
        self,
        returns_data: pd.DataFrame,
        expected_returns: pd.Series = None,
        custom_covariance: pd.DataFrame = None,
    ) -> OptimizationResult:
        """リスクパリティ最適化実行"""

        start_time = datetime.now()
        logger.info(f"リスクパリティ最適化開始: {self.config.method.value}")

        try:
            # 共分散行列推定
            if custom_covariance is not None:
                cov_matrix = custom_covariance.values
                asset_names = custom_covariance.columns.tolist()
            else:
                cov_matrix = self._estimate_covariance_matrix(returns_data)
                asset_names = returns_data.columns.tolist()

            # 最適化実行
            if self.config.method == RiskParityMethod.EQUAL_RISK_CONTRIBUTION:
                result = self._equal_risk_contribution_optimization(
                    cov_matrix, asset_names
                )
            elif self.config.method == RiskParityMethod.RISK_BUDGETING:
                result = self._risk_budgeting_optimization(cov_matrix, asset_names)
            elif self.config.method == RiskParityMethod.HIERARCHICAL_RISK_PARITY:
                result = self._hierarchical_risk_parity(returns_data, asset_names)
            else:
                result = self._equal_risk_contribution_optimization(
                    cov_matrix, asset_names
                )

            # 結果検証・調整
            result = self._validate_and_adjust_result(result, cov_matrix, asset_names)

            # 履歴更新
            optimization_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics["optimization_times"].append(optimization_time)
            self.optimization_history.append(result)

            self.current_weights = result.weights
            self.current_covariance = cov_matrix
            self.last_optimization = datetime.now()

            logger.info(f"リスクパリティ最適化完了: {optimization_time:.3f}秒")
            return result

        except Exception as e:
            logger.error(f"リスクパリティ最適化エラー: {e}")
            raise

    def _estimate_covariance_matrix(self, returns_data: pd.DataFrame) -> np.ndarray:
        """共分散行列推定"""

        # 基本共分散行列
        window = min(self.config.correlation_window, len(returns_data))
        recent_returns = returns_data.tail(window)
        cov_matrix = recent_returns.cov().values

        # 年率化
        cov_matrix *= 252

        # 正定値性確保
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return cov_matrix

    def _equal_risk_contribution_optimization(
        self, cov_matrix: np.ndarray, asset_names: List[str]
    ) -> OptimizationResult:
        """等リスク寄与度最適化"""

        if not SCIPY_AVAILABLE:
            # フォールバック: 等ウェイト
            n_assets = len(asset_names)
            weights = np.ones(n_assets) / n_assets
            return self._create_result(weights, cov_matrix, asset_names, False)

        n_assets = len(asset_names)

        def risk_budget_objective(weights):
            """リスクバジェット目的関数"""
            weights = np.maximum(weights, 1e-8)  # ゼロ回避

            # ポートフォリオボラティリティ
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

            if portfolio_vol < 1e-8:
                return 1e10

            # 限界リスク寄与度
            marginal_contrib = (cov_matrix @ weights) / portfolio_vol

            # リスク寄与度
            risk_contrib = weights * marginal_contrib / portfolio_vol

            # 目標リスク寄与度（等分）
            target_contrib = 1.0 / n_assets

            # 目的関数：二乗誤差の合計
            return np.sum((risk_contrib - target_contrib) ** 2)

        # 制約条件
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]  # 重み合計=1

        # 境界条件
        bounds = [
            (self.config.min_weight, self.config.max_weight) for _ in range(n_assets)
        ]

        # 初期値
        initial_weights = np.ones(n_assets) / n_assets

        # 最適化実行
        result = minimize(
            risk_budget_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": self.config.max_iterations,
                "ftol": self.config.tolerance,
            },
        )

        return self._create_result(
            result.x, cov_matrix, asset_names, result.success, result.nit, result.fun
        )

    def _risk_budgeting_optimization(
        self, cov_matrix: np.ndarray, asset_names: List[str]
    ) -> OptimizationResult:
        """リスクバジェット最適化"""

        if not SCIPY_AVAILABLE:
            return self._equal_risk_contribution_optimization(cov_matrix, asset_names)

        # カスタムリスクバジェット取得
        if self.config.risk_budgets is None:
            # デフォルト: 等分
            n_assets = len(asset_names)
            risk_budgets = {name: 1.0 / n_assets for name in asset_names}
        else:
            risk_budgets = self.config.risk_budgets.copy()

            # 正規化
            total_budget = sum(risk_budgets.values())
            risk_budgets = {k: v / total_budget for k, v in risk_budgets.items()}

        # 目標リスク寄与度配列
        target_contrib = np.array(
            [risk_budgets.get(name, 1.0 / len(asset_names)) for name in asset_names]
        )

        def risk_budget_objective(weights):
            """カスタムリスクバジェット目的関数"""
            weights = np.maximum(weights, 1e-8)

            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            if portfolio_vol < 1e-8:
                return 1e10

            marginal_contrib = (cov_matrix @ weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib / portfolio_vol

            return np.sum((risk_contrib - target_contrib) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [
            (self.config.min_weight, self.config.max_weight)
            for _ in range(len(asset_names))
        ]

        initial_weights = target_contrib / np.sum(target_contrib)

        result = minimize(
            risk_budget_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": self.config.max_iterations,
                "ftol": self.config.tolerance,
            },
        )

        optimization_result = self._create_result(
            result.x, cov_matrix, asset_names, result.success, result.nit, result.fun
        )
        optimization_result.risk_budgets = risk_budgets

        return optimization_result

    def _hierarchical_risk_parity(
        self, returns_data: pd.DataFrame, asset_names: List[str]
    ) -> OptimizationResult:
        """階層リスクパリティ"""

        if not SCIPY_AVAILABLE:
            # フォールバック
            n_assets = len(asset_names)
            weights = np.ones(n_assets) / n_assets
            cov_matrix = self._estimate_covariance_matrix(returns_data)
            return self._create_result(weights, cov_matrix, asset_names, False)

        try:
            # 相関行列計算
            corr_matrix = returns_data.corr().values

            # 距離行列作成
            distance_matrix = np.sqrt((1 - corr_matrix) / 2)

            # 階層クラスタリング
            condensed_distances = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(
                condensed_distances, method=self.config.clustering_method
            )

            # クラスタ配分
            clusters = fcluster(
                linkage_matrix, self.config.n_clusters, criterion="maxclust"
            )

            # 各クラスタ内で等リスク寄与度配分
            cov_matrix = self._estimate_covariance_matrix(returns_data)
            final_weights = np.zeros(len(asset_names))

            # クラスタ間重み配分（等ウェイト）
            cluster_weights = 1.0 / self.config.n_clusters

            for cluster_id in range(1, self.config.n_clusters + 1):
                cluster_assets = [i for i, c in enumerate(clusters) if c == cluster_id]

                if len(cluster_assets) == 1:
                    final_weights[cluster_assets[0]] = cluster_weights
                else:
                    # クラスタ内共分散行列
                    cluster_cov = cov_matrix[np.ix_(cluster_assets, cluster_assets)]

                    # クラスタ内等リスク寄与度最適化
                    cluster_result = self._equal_risk_contribution_optimization(
                        cluster_cov, [asset_names[i] for i in cluster_assets]
                    )

                    # クラスタ重みで調整
                    for i, asset_idx in enumerate(cluster_assets):
                        asset_name = asset_names[asset_idx]
                        final_weights[asset_idx] = (
                            cluster_result.weights[asset_name] * cluster_weights
                        )

            # 重み正規化
            final_weights /= np.sum(final_weights)

            return self._create_result(final_weights, cov_matrix, asset_names, True)

        except Exception as e:
            logger.error(f"階層リスクパリティエラー: {e}")
            # フォールバック
            return self._equal_risk_contribution_optimization(
                self._estimate_covariance_matrix(returns_data), asset_names
            )

    def _create_result(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: List[str],
        success: bool = True,
        iterations: int = 0,
        objective_value: float = 0.0,
    ) -> OptimizationResult:
        """最適化結果作成"""

        # 重み辞書作成
        weights_dict = {name: weight for name, weight in zip(asset_names, weights)}

        # ポートフォリオボラティリティ
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

        # リスク寄与度計算
        if portfolio_vol > 1e-8:
            marginal_contrib = (cov_matrix @ weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib / portfolio_vol
        else:
            risk_contrib = np.zeros_like(weights)

        risk_contributions = {
            name: contrib for name, contrib in zip(asset_names, risk_contrib)
        }

        # デフォルトリスクバジェット
        n_assets = len(asset_names)
        risk_budgets = {name: 1.0 / n_assets for name in asset_names}

        # 分散化比率
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_avg_vol = np.sum(weights * individual_vols)
        diversification_ratio = (
            weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        )

        # 集中度指標
        concentration_index = np.sum(weights**2)  # ハーフィンダール指数
        effective_number_assets = 1.0 / concentration_index

        # リスク集中度
        risk_concentration_hhi = np.sum(risk_contrib**2)
        largest_risk_contrib = np.max(risk_contrib)
        smallest_risk_contrib = np.min(risk_contrib)

        return OptimizationResult(
            weights=weights_dict,
            risk_contributions=risk_contributions,
            risk_budgets=risk_budgets,
            portfolio_volatility=portfolio_vol,
            diversification_ratio=diversification_ratio,
            concentration_index=concentration_index,
            optimization_success=success,
            iterations=iterations,
            objective_value=objective_value,
            effective_number_assets=effective_number_assets,
            risk_concentration_hhi=risk_concentration_hhi,
            largest_risk_contribution=largest_risk_contrib,
            smallest_risk_contribution=smallest_risk_contrib,
        )

    def _validate_and_adjust_result(
        self, result: OptimizationResult, cov_matrix: np.ndarray, asset_names: List[str]
    ) -> OptimizationResult:
        """結果検証・調整"""

        weights_array = np.array([result.weights[name] for name in asset_names])

        # 制約チェック
        if not (abs(np.sum(weights_array) - 1.0) < 1e-6):
            logger.warning("重み合計が1.0から逸脱しています。正規化します。")
            weights_array /= np.sum(weights_array)

        # 最小・最大重み制約
        weights_array = np.clip(
            weights_array, self.config.min_weight, self.config.max_weight
        )
        weights_array /= np.sum(weights_array)  # 再正規化

        # 結果更新
        result.weights = {
            name: weight for name, weight in zip(asset_names, weights_array)
        }

        # リスク寄与度再計算
        portfolio_vol = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
        if portfolio_vol > 1e-8:
            marginal_contrib = (cov_matrix @ weights_array) / portfolio_vol
            risk_contrib = weights_array * marginal_contrib / portfolio_vol
            result.risk_contributions = {
                name: contrib for name, contrib in zip(asset_names, risk_contrib)
            }
            result.portfolio_volatility = portfolio_vol

        return result

    def calculate_rebalancing_need(
        self, current_weights: Dict[str, float], target_weights: Dict[str, float]
    ) -> Tuple[bool, float]:
        """リバランシング必要性判定"""

        if not current_weights or not target_weights:
            return True, 1.0

        # 共通資産のみ比較
        common_assets = set(current_weights.keys()) & set(target_weights.keys())

        if not common_assets:
            return True, 1.0

        # 重み差の計算
        total_deviation = 0.0
        for asset in common_assets:
            current_w = current_weights.get(asset, 0.0)
            target_w = target_weights.get(asset, 0.0)
            total_deviation += abs(current_w - target_w)

        # 平均偏差
        avg_deviation = total_deviation / len(common_assets)

        need_rebalancing = avg_deviation > self.config.rebalancing_threshold

        return need_rebalancing, avg_deviation

    def analyze_risk_contributions(self, result: OptimizationResult) -> Dict[str, Any]:
        """リスク寄与度分析"""

        risk_contribs = list(result.risk_contributions.values())
        weights = list(result.weights.values())

        analysis = {
            "risk_contribution_stats": {
                "mean": np.mean(risk_contribs),
                "std": np.std(risk_contribs),
                "min": np.min(risk_contribs),
                "max": np.max(risk_contribs),
                "range": np.max(risk_contribs) - np.min(risk_contribs),
            },
            "weight_stats": {
                "mean": np.mean(weights),
                "std": np.std(weights),
                "min": np.min(weights),
                "max": np.max(weights),
                "range": np.max(weights) - np.min(weights),
            },
            "diversification_metrics": {
                "portfolio_volatility": result.portfolio_volatility,
                "diversification_ratio": result.diversification_ratio,
                "effective_number_assets": result.effective_number_assets,
                "concentration_hhi": result.concentration_index,
                "risk_concentration_hhi": result.risk_concentration_hhi,
            },
            "risk_budget_deviation": {},
        }

        # リスクバジェット偏差
        for asset, actual_contrib in result.risk_contributions.items():
            target_budget = result.risk_budgets.get(
                asset, 1.0 / len(result.risk_budgets)
            )
            deviation = abs(actual_contrib - target_budget)
            analysis["risk_budget_deviation"][asset] = {
                "actual": actual_contrib,
                "target": target_budget,
                "deviation": deviation,
                "relative_deviation": (
                    deviation / target_budget if target_budget > 0 else 0
                ),
            }

        return analysis

    def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化概要取得"""

        if not self.optimization_history:
            return {"status": "最適化未実行"}

        latest_result = self.optimization_history[-1]

        # パフォーマンス統計
        avg_optimization_time = np.mean(self.performance_metrics["optimization_times"])
        total_optimizations = len(self.optimization_history)

        # 成功率
        successful_optimizations = sum(
            1 for r in self.optimization_history if r.optimization_success
        )
        success_rate = (
            successful_optimizations / total_optimizations
            if total_optimizations > 0
            else 0
        )

        return {
            "latest_optimization": {
                "portfolio_volatility": latest_result.portfolio_volatility,
                "diversification_ratio": latest_result.diversification_ratio,
                "effective_number_assets": latest_result.effective_number_assets,
                "optimization_success": latest_result.optimization_success,
                "largest_weight": max(latest_result.weights.values()),
                "smallest_weight": min(latest_result.weights.values()),
            },
            "performance_stats": {
                "total_optimizations": total_optimizations,
                "success_rate": success_rate,
                "avg_optimization_time": avg_optimization_time,
                "method": self.config.method.value,
                "last_optimization": (
                    self.last_optimization.isoformat()
                    if self.last_optimization
                    else None
                ),
            },
            "configuration": {
                "min_weight": self.config.min_weight,
                "max_weight": self.config.max_weight,
                "target_volatility": self.config.target_volatility,
                "rebalancing_threshold": self.config.rebalancing_threshold,
            },
            "current_weights": self.current_weights,
        }


# グローバルインスタンス
_risk_parity_optimizer = None


def get_risk_parity_optimizer(config: RiskParityConfig = None) -> RiskParityOptimizer:
    """リスクパリティ最適化システム取得"""
    global _risk_parity_optimizer
    if _risk_parity_optimizer is None:
        _risk_parity_optimizer = RiskParityOptimizer(config)
    return _risk_parity_optimizer
