#!/usr/bin/env python3
"""
AI駆動ポートフォリオマネージャー
マルチアセット・ポートフォリオ自動構築・最適化システム

Features:
- AI駆動資産配分最適化
- リアルタイム リバランシング
- リスク調整リターン最大化
- マルチファクター分析
- 機械学習予測統合
- ESG要因考慮
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..data.advanced_ml_engine import create_advanced_ml_engine
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class AssetClass(Enum):
    """資産クラス"""

    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    REAL_ESTATE = "real_estate"
    CRYPTO = "crypto"
    CASH = "cash"
    ALTERNATIVE = "alternative"


class OptimizationMethod(Enum):
    """最適化手法"""

    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    AI_ENHANCED = "ai_enhanced"
    ESG_INTEGRATED = "esg_integrated"


@dataclass
class PortfolioConfig:
    """ポートフォリオ設定"""

    # 基本設定
    target_return: float = 0.08  # 年間8%
    risk_tolerance: float = 0.15  # 年間15%ボラティリティ
    investment_horizon_months: int = 12

    # 最適化設定
    optimization_method: OptimizationMethod = OptimizationMethod.AI_ENHANCED
    rebalancing_frequency_days: int = 30
    minimum_allocation: float = 0.01  # 1%
    maximum_allocation: float = 0.40  # 40%

    # AI設定
    use_ml_predictions: bool = True
    prediction_confidence_threshold: float = 0.7
    ensemble_models: bool = True

    # リスク設定
    maximum_drawdown: float = 0.20  # 20%
    var_confidence_level: float = 0.05  # 95% VaR
    expected_shortfall_level: float = 0.05  # 95% ES

    # ESG設定
    include_esg_factors: bool = False
    esg_minimum_score: float = 3.0  # 1-5スケール

    # 制約条件
    asset_class_constraints: Dict[AssetClass, Tuple[float, float]] = None

    def __post_init__(self):
        if self.asset_class_constraints is None:
            # デフォルト制約
            self.asset_class_constraints = {
                AssetClass.EQUITY: (0.20, 0.80),
                AssetClass.BOND: (0.10, 0.50),
                AssetClass.COMMODITY: (0.00, 0.20),
                AssetClass.REAL_ESTATE: (0.00, 0.15),
                AssetClass.CRYPTO: (0.00, 0.10),
                AssetClass.CASH: (0.00, 0.20),
            }


@dataclass
class AssetAllocation:
    """資産配分"""

    symbol: str
    asset_class: AssetClass
    weight: float
    expected_return: float
    risk: float
    confidence: float
    last_updated: datetime


@dataclass
class PortfolioMetrics:
    """ポートフォリオメトリクス"""

    # リターン指標
    expected_annual_return: float
    realized_return: float
    cumulative_return: float

    # リスク指標
    annual_volatility: float
    maximum_drawdown: float
    var_95: float
    expected_shortfall_95: float

    # リスク調整指標
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float

    # 分散化指標
    diversification_ratio: float
    concentration_herfindahl: float
    effective_number_assets: float

    # AI予測指標
    prediction_accuracy: float
    model_confidence: float
    forecast_horizon_days: int


@dataclass
class OptimizationResults:
    """最適化結果"""

    allocations: List[AssetAllocation]
    portfolio_metrics: PortfolioMetrics
    optimization_method: OptimizationMethod
    optimization_time: float
    constraints_satisfied: bool
    convergence_achieved: bool
    objective_value: float

    # 詳細結果
    covariance_matrix: np.ndarray
    correlation_matrix: np.ndarray
    expected_returns: np.ndarray
    risk_contributions: np.ndarray


class PortfolioManager:
    """AI駆動ポートフォリオマネージャー"""

    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()

        # AI/ML コンポーネント
        self.ml_engine = create_advanced_ml_engine()
        self.prediction_models = {}

        # データ管理
        self.asset_universe = {}
        self.historical_data = {}
        self.current_allocations = []

        # パフォーマンス追跡
        self.performance_history = []
        self.rebalancing_history = []

        # 状態管理
        self.last_optimization = None
        self.is_initialized = False

        logger.info("AI ポートフォリオマネージャー初期化完了")

    async def initialize_portfolio(
        self,
        asset_universe: Dict[str, Dict[str, Any]],
        historical_data: Dict[str, pd.DataFrame],
    ) -> bool:
        """ポートフォリオ初期化"""
        try:
            logger.info("ポートフォリオ初期化開始")

            # 資産ユニバース設定
            self.asset_universe = asset_universe
            self.historical_data = historical_data

            # AIモデル訓練
            if self.config.use_ml_predictions:
                await self._train_prediction_models()

            # 初期最適化実行
            optimization_result = await self.optimize_portfolio()

            if optimization_result.convergence_achieved:
                self.current_allocations = optimization_result.allocations
                self.last_optimization = datetime.now()
                self.is_initialized = True

                logger.info("ポートフォリオ初期化成功")
                return True
            else:
                logger.error("初期最適化が収束しませんでした")
                return False

        except Exception as e:
            logger.error(f"ポートフォリオ初期化エラー: {e}")
            return False

    async def optimize_portfolio(self) -> OptimizationResults:
        """ポートフォリオ最適化"""
        start_time = datetime.now()

        try:
            logger.info(
                f"ポートフォリオ最適化開始: {self.config.optimization_method.value}"
            )

            # 期待リターン予測
            expected_returns = await self._predict_expected_returns()

            # 共分散行列推定
            covariance_matrix = await self._estimate_covariance_matrix()

            # 最適化実行
            if self.config.optimization_method == OptimizationMethod.AI_ENHANCED:
                weights = await self._ai_enhanced_optimization(
                    expected_returns, covariance_matrix
                )
            elif self.config.optimization_method == OptimizationMethod.MEAN_VARIANCE:
                weights = self._mean_variance_optimization(
                    expected_returns, covariance_matrix
                )
            elif self.config.optimization_method == OptimizationMethod.RISK_PARITY:
                weights = self._risk_parity_optimization(covariance_matrix)
            elif self.config.optimization_method == OptimizationMethod.BLACK_LITTERMAN:
                weights = self._black_litterman_optimization(
                    expected_returns, covariance_matrix
                )
            else:
                raise ValueError(
                    f"未対応の最適化手法: {self.config.optimization_method}"
                )

            # 配分結果作成
            allocations = self._create_allocations(weights, expected_returns)

            # メトリクス計算
            portfolio_metrics = self._calculate_portfolio_metrics(
                weights, expected_returns, covariance_matrix
            )

            # リスク寄与度計算
            risk_contributions = self._calculate_risk_contributions(
                weights, covariance_matrix
            )

            # 制約チェック
            constraints_satisfied = self._check_constraints(weights)

            optimization_time = (datetime.now() - start_time).total_seconds()

            result = OptimizationResults(
                allocations=allocations,
                portfolio_metrics=portfolio_metrics,
                optimization_method=self.config.optimization_method,
                optimization_time=optimization_time,
                constraints_satisfied=constraints_satisfied,
                convergence_achieved=True,
                objective_value=self._calculate_objective_value(
                    weights, expected_returns, covariance_matrix
                ),
                covariance_matrix=covariance_matrix,
                correlation_matrix=np.corrcoef(covariance_matrix),
                expected_returns=expected_returns,
                risk_contributions=risk_contributions,
            )

            logger.info(f"ポートフォリオ最適化完了: {optimization_time:.2f}秒")
            return result

        except Exception as e:
            logger.error(f"ポートフォリオ最適化エラー: {e}")
            raise

    async def _train_prediction_models(self):
        """予測モデル訓練"""
        logger.info("AI予測モデル訓練開始")

        try:
            for symbol, data in self.historical_data.items():
                if len(data) > 252:  # 1年以上のデータが必要
                    # 特徴量準備
                    X, y = self.ml_engine.prepare_data(data, target_column="終値")

                    if len(X) > 50:  # 最低限の学習データ
                        # モデル訓練
                        train_result = self.ml_engine.train_model(
                            X[:-20],  # 訓練用
                            y[:-20],
                            X[-20:],  # 検証用
                            y[-20:],
                        )

                        self.prediction_models[symbol] = {
                            "model": self.ml_engine,
                            "performance": train_result,
                            "last_updated": datetime.now(),
                        }

                        logger.info(f"モデル訓練完了: {symbol}")

            logger.info(f"予測モデル訓練完了: {len(self.prediction_models)}銘柄")

        except Exception as e:
            logger.error(f"予測モデル訓練エラー: {e}")

    async def _predict_expected_returns(self) -> np.ndarray:
        """期待リターン予測"""
        expected_returns = []

        for symbol in self.asset_universe.keys():
            if symbol in self.prediction_models:
                # AI予測使用
                model_info = self.prediction_models[symbol]
                recent_data = self.historical_data[symbol].tail(
                    self.ml_engine.config.sequence_length
                )

                X, _ = self.ml_engine.prepare_data(recent_data, target_column="終値")
                if len(X) > 0:
                    prediction = self.ml_engine.predict(X[-1:])
                    expected_return = (
                        prediction.predictions[0]
                        if len(prediction.predictions) > 0
                        else 0.08
                    )
                else:
                    expected_return = 0.08
            else:
                # 過去データ基準
                returns = self.historical_data[symbol]["終値"].pct_change().dropna()
                expected_return = (
                    returns.mean() * 252 if len(returns) > 0 else 0.08
                )  # 年率化

            expected_returns.append(expected_return)

        return np.array(expected_returns)

    async def _estimate_covariance_matrix(self) -> np.ndarray:
        """共分散行列推定"""
        returns_data = []

        for symbol in self.asset_universe.keys():
            if symbol in self.historical_data:
                returns = self.historical_data[symbol]["終値"].pct_change().dropna()
                returns_data.append(returns)

        if returns_data:
            # データ長を揃える
            min_length = min(len(r) for r in returns_data)
            returns_aligned = [r.tail(min_length) for r in returns_data]
            returns_df = pd.concat(returns_aligned, axis=1)

            # 共分散行列計算（年率化）
            cov_matrix = returns_df.cov().values * 252

            # 正定値性確保
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)
            cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

            return cov_matrix
        else:
            # フォールバック: 単位行列
            n_assets = len(self.asset_universe)
            return np.eye(n_assets) * 0.04  # 20%ボラティリティ想定

    async def _ai_enhanced_optimization(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        """AI強化最適化"""
        from scipy.optimize import minimize

        n_assets = len(expected_returns)

        # 目的関数：シャープレシオ最大化 + AI信頼度重み付け
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            if portfolio_risk == 0:
                return -1e10

            sharpe_ratio = portfolio_return / portfolio_risk

            # AI信頼度ボーナス
            confidence_bonus = 0
            for i, symbol in enumerate(self.asset_universe.keys()):
                if symbol in self.prediction_models:
                    confidence = self.prediction_models[symbol]["performance"].get(
                        "confidence", 0.5
                    )
                    confidence_bonus += weights[i] * confidence

            return -(sharpe_ratio + confidence_bonus * 0.1)

        # 制約条件
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}  # 合計100%
        ]

        # 境界条件
        bounds = [
            (self.config.minimum_allocation, self.config.maximum_allocation)
            for _ in range(n_assets)
        ]

        # 初期値
        initial_weights = np.ones(n_assets) / n_assets

        # 最適化実行
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            return result.x
        else:
            logger.warning("AI強化最適化が収束しませんでした。等ウェイトを使用")
            return initial_weights

    def _mean_variance_optimization(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        """平均-分散最適化"""
        from scipy.optimize import minimize

        n_assets = len(expected_returns)

        # 目的関数：効用最大化
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            # 効用 = リターン - (リスク回避度/2) * 分散
            utility = (
                portfolio_return
                - (1 / (2 * self.config.risk_tolerance)) * portfolio_variance
            )
            return -utility  # 最小化のため符号反転

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [
            (self.config.minimum_allocation, self.config.maximum_allocation)
            for _ in range(n_assets)
        ]

        initial_weights = np.ones(n_assets) / n_assets

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else initial_weights

    def _risk_parity_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """リスクパリティ最適化"""
        from scipy.optimize import minimize

        n_assets = len(cov_matrix)

        def risk_budget_objective(weights):
            """リスクバジェット目的関数"""
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [
            (self.config.minimum_allocation, self.config.maximum_allocation)
            for _ in range(n_assets)
        ]

        initial_weights = np.ones(n_assets) / n_assets

        result = minimize(
            risk_budget_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else initial_weights

    def _black_litterman_optimization(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Black-Litterman最適化（簡易版）"""
        from scipy.optimize import minimize

        n_assets = len(expected_returns)

        # 市場インプライドリターン（CAPM想定）
        market_cap_weights = np.ones(n_assets) / n_assets  # 簡易版：等ウェイト
        risk_aversion = 1 / self.config.risk_tolerance
        implied_returns = risk_aversion * np.dot(cov_matrix, market_cap_weights)

        # Black-Litterman調整（簡易版）
        tau = 0.1  # 不確実性パラメータ
        adjusted_cov = cov_matrix * (1 + tau)
        adjusted_returns = (implied_returns + tau * expected_returns) / (1 + tau)

        # 平均分散最適化
        def objective(weights):
            portfolio_return = np.dot(weights, adjusted_returns)
            portfolio_variance = np.dot(weights.T, np.dot(adjusted_cov, weights))
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [
            (self.config.minimum_allocation, self.config.maximum_allocation)
            for _ in range(n_assets)
        ]

        initial_weights = np.ones(n_assets) / n_assets

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else initial_weights

    def _create_allocations(
        self, weights: np.ndarray, expected_returns: np.ndarray
    ) -> List[AssetAllocation]:
        """配分結果作成"""
        allocations = []

        for i, (symbol, asset_info) in enumerate(self.asset_universe.items()):
            # 信頼度計算
            confidence = 0.7  # デフォルト
            if symbol in self.prediction_models:
                model_performance = self.prediction_models[symbol]["performance"]
                confidence = 1.0 - model_performance.get("final_val_loss", 0.3)
                confidence = max(0.1, min(0.99, confidence))

            allocation = AssetAllocation(
                symbol=symbol,
                asset_class=AssetClass(asset_info.get("asset_class", "equity")),
                weight=weights[i],
                expected_return=expected_returns[i],
                risk=np.sqrt(
                    self.historical_data[symbol]["終値"].pct_change().var() * 252
                )
                if symbol in self.historical_data
                else 0.15,
                confidence=confidence,
                last_updated=datetime.now(),
            )
            allocations.append(allocation)

        return allocations

    def _calculate_portfolio_metrics(
        self, weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> PortfolioMetrics:
        """ポートフォリオメトリクス計算"""
        # 基本指標
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        # リスクフリーレート（仮に2%）
        risk_free_rate = 0.02

        # シャープレシオ
        sharpe_ratio = (
            (portfolio_return - risk_free_rate) / portfolio_volatility
            if portfolio_volatility > 0
            else 0
        )

        # その他の指標（簡易計算）
        diversification_ratio = (
            np.sum(weights * np.sqrt(np.diag(cov_matrix))) / portfolio_volatility
        )
        concentration_herfindahl = np.sum(weights**2)
        effective_number_assets = 1 / concentration_herfindahl

        # VaR計算（正規分布想定）
        var_95 = -1.645 * portfolio_volatility / np.sqrt(252)  # 日次VaR
        expected_shortfall_95 = -2.33 * portfolio_volatility / np.sqrt(252)  # 日次ES

        # AI予測精度（平均）
        prediction_accuracy = 0.7
        model_confidence = 0.7
        if self.prediction_models:
            accuracies = []
            confidences = []
            for model_info in self.prediction_models.values():
                perf = model_info["performance"]
                acc = 1.0 - perf.get("final_val_loss", 0.3)
                accuracies.append(max(0.1, min(0.99, acc)))
                confidences.append(acc)

            prediction_accuracy = np.mean(accuracies)
            model_confidence = np.mean(confidences)

        return PortfolioMetrics(
            expected_annual_return=portfolio_return,
            realized_return=0.0,  # 実績は後で更新
            cumulative_return=0.0,
            annual_volatility=portfolio_volatility,
            maximum_drawdown=0.0,  # 実績は後で更新
            var_95=var_95,
            expected_shortfall_95=expected_shortfall_95,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sharpe_ratio * 1.1,  # 簡易近似
            calmar_ratio=0.0,  # 実績は後で更新
            information_ratio=0.0,  # ベンチマーク比較時に計算
            diversification_ratio=diversification_ratio,
            concentration_herfindahl=concentration_herfindahl,
            effective_number_assets=effective_number_assets,
            prediction_accuracy=prediction_accuracy,
            model_confidence=model_confidence,
            forecast_horizon_days=30,
        )

    def _calculate_risk_contributions(
        self, weights: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        """リスク寄与度計算"""
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib / portfolio_vol
        return risk_contrib

    def _check_constraints(self, weights: np.ndarray) -> bool:
        """制約条件チェック"""
        # 基本制約
        if not (abs(np.sum(weights) - 1.0) < 1e-6):
            return False

        if np.any(weights < self.config.minimum_allocation - 1e-6):
            return False

        return not (np.any(weights > self.config.maximum_allocation + 1e-6))

    def _calculate_objective_value(
        self, weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> float:
        """目的関数値計算"""
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

    def get_current_allocations(self) -> List[AssetAllocation]:
        """現在の配分取得"""
        return self.current_allocations.copy()

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要取得"""
        if not self.current_allocations:
            return {"status": "ポートフォリオ未初期化"}

        total_weight = sum(alloc.weight for alloc in self.current_allocations)
        avg_confidence = (
            sum(alloc.confidence * alloc.weight for alloc in self.current_allocations)
            / total_weight
            if total_weight > 0
            else 0
        )

        asset_class_distribution = {}
        for alloc in self.current_allocations:
            asset_class = alloc.asset_class.value
            asset_class_distribution[asset_class] = (
                asset_class_distribution.get(asset_class, 0) + alloc.weight
            )

        return {
            "total_assets": len(self.current_allocations),
            "total_weight": total_weight,
            "average_confidence": avg_confidence,
            "asset_class_distribution": asset_class_distribution,
            "last_optimization": self.last_optimization.isoformat()
            if self.last_optimization
            else None,
            "optimization_method": self.config.optimization_method.value,
            "ml_models_count": len(self.prediction_models),
            "initialization_status": self.is_initialized,
        }


# グローバルインスタンス
_portfolio_manager = None


def get_portfolio_manager(config: PortfolioConfig = None) -> PortfolioManager:
    """ポートフォリオマネージャー取得"""
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = PortfolioManager(config)
    return _portfolio_manager
