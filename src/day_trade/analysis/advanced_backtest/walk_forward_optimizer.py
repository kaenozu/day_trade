"""
高度なバックテストエンジン - ウォークフォワード最適化

戦略パラメータの動的最適化と検証を行う。
"""

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from day_trade.utils.logging_config import get_context_logger
from .data_structures import PerformanceMetrics

warnings.filterwarnings("ignore")
logger = get_context_logger(__name__)


class WalkForwardOptimizer:
    """ウォークフォワード最適化システム"""

    def __init__(
        self,
        backtest_engine,
        optimization_window: int = 252,  # 1年
        rebalance_frequency: int = 63,  # 四半期
        parameter_grid: Optional[Dict[str, List]] = None,
        min_optimization_samples: int = 100,
        stability_threshold: float = 0.7,
    ):
        """ウォークフォワード最適化の初期化"""
        self.backtest_engine = backtest_engine
        self.optimization_window = optimization_window
        self.rebalance_frequency = rebalance_frequency
        self.parameter_grid = parameter_grid or {}
        self.min_optimization_samples = min_optimization_samples
        self.stability_threshold = stability_threshold

        # 最適化履歴
        self.optimization_history: List[Dict] = []

    def optimize(
        self,
        data: pd.DataFrame,
        strategy_func,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """
        ウォークフォワード最適化実行

        Args:
            data: 価格データ
            strategy_func: 戦略関数
            start_date: 開始日
            end_date: 終了日

        Returns:
            最適化結果
        """
        logger.info(
            "ウォークフォワード最適化開始",
            section="walk_forward",
            optimization_window=self.optimization_window,
            rebalance_frequency=self.rebalance_frequency,
        )

        results = []
        current_date = start_date

        while current_date < end_date:
            # 最適化期間の設定
            opt_start = current_date - timedelta(days=self.optimization_window)
            opt_end = current_date

            # テスト期間の設定
            test_start = current_date
            test_end = min(
                current_date + timedelta(days=self.rebalance_frequency), end_date
            )

            # データが十分にあるかチェック
            if opt_start < data.index[0]:
                current_date = test_end
                continue

            opt_data = data.loc[opt_start:opt_end]
            test_data = data.loc[test_start:test_end]

            if len(opt_data) < self.min_optimization_samples:
                logger.warning(
                    f"最適化期間のデータが不十分: {len(opt_data)}サンプル",
                    section="walk_forward"
                )
                current_date = test_end
                continue

            # 最適化実行
            best_params, optimization_results = self._optimize_parameters(
                opt_data, strategy_func
            )

            # テスト実行
            test_performance = self._test_parameters(
                test_data, strategy_func, best_params
            )

            period_result = {
                "optimization_period": (opt_start, opt_end),
                "test_period": (test_start, test_end),
                "best_parameters": best_params,
                "optimization_results": optimization_results,
                "test_performance": test_performance,
                "data_quality": self._assess_data_quality(opt_data),
                "parameter_stability": self._assess_parameter_stability(best_params),
            }

            results.append(period_result)
            self.optimization_history.append(period_result)

            current_date = test_end

        # 結果分析
        analysis = self._analyze_walk_forward_results(results)

        logger.info(
            "ウォークフォワード最適化完了",
            section="walk_forward",
            periods_tested=len(results),
            avg_sharpe=analysis.get("avg_sharpe", 0),
            stability_score=analysis.get("stability_score", 0),
        )

        return analysis

    def _optimize_parameters(
        self, data: pd.DataFrame, strategy_func
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """パラメータ最適化"""
        if not self.parameter_grid:
            return {}, {}

        optimization_results = {
            "tested_combinations": 0,
            "successful_tests": 0,
            "best_sharpe": -float("inf"),
            "parameter_performance": {}
        }

        best_params = {}
        best_sharpe = -float("inf")

        # グリッドサーチ実行
        for param_combination in self._generate_parameter_combinations():
            optimization_results["tested_combinations"] += 1

            try:
                # 戦略実行
                signals = strategy_func(data, **param_combination)
                
                if signals is None or len(signals) == 0:
                    continue

                # バックテスト実行
                performance = self.backtest_engine.run_backtest(data, signals)
                optimization_results["successful_tests"] += 1

                # パフォーマンス記録
                param_key = str(sorted(param_combination.items()))
                optimization_results["parameter_performance"][param_key] = {
                    "sharpe_ratio": performance.sharpe_ratio,
                    "total_return": performance.total_return,
                    "max_drawdown": performance.max_drawdown,
                    "total_trades": performance.total_trades,
                }

                if performance.sharpe_ratio > best_sharpe:
                    best_sharpe = performance.sharpe_ratio
                    best_params = param_combination.copy()
                    optimization_results["best_sharpe"] = best_sharpe

            except Exception as e:
                logger.warning(
                    f"パラメータ最適化エラー: {param_combination}",
                    section="parameter_optimization",
                    error=str(e),
                )

        return best_params, optimization_results

    def _generate_parameter_combinations(self):
        """パラメータの全組み合わせを生成"""
        if not self.parameter_grid:
            yield {}
            return

        # 単一パラメータの場合の簡易実装
        if len(self.parameter_grid) == 1:
            param_name, param_values = next(iter(self.parameter_grid.items()))
            for value in param_values:
                yield {param_name: value}
        else:
            # 複数パラメータの組み合わせ（より高度な実装が必要）
            # ここでは簡易的に各パラメータを個別にテスト
            for param_name, param_values in self.parameter_grid.items():
                for value in param_values:
                    yield {param_name: value}

    def _test_parameters(
        self, data: pd.DataFrame, strategy_func, params: Dict[str, Any]
    ) -> PerformanceMetrics:
        """パラメータテスト"""
        try:
            signals = strategy_func(data, **params)
            if signals is None or len(signals) == 0:
                return PerformanceMetrics()
            
            return self.backtest_engine.run_backtest(data, signals)
        except Exception as e:
            logger.error(
                f"パラメータテストエラー: {params}",
                section="parameter_test",
                error=str(e),
            )
            return PerformanceMetrics()

    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """データ品質の評価"""
        if data.empty:
            return {"quality_score": 0.0, "issues": ["empty_data"]}

        quality_metrics = {}
        issues = []

        # データ完全性チェック
        missing_data_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        quality_metrics["missing_data_ratio"] = missing_data_ratio

        if missing_data_ratio > 0.05:  # 5%以上の欠損
            issues.append("high_missing_data")

        # ボラティリティチェック
        if "Close" in data.columns or ("Close" in str(data.columns)):
            try:
                # マルチインデックス対応
                close_series = data["Close"] if "Close" in data.columns else data.iloc[:, data.columns.str.contains("Close")].iloc[:, 0]
                daily_returns = close_series.pct_change().dropna()
                
                volatility = daily_returns.std()
                quality_metrics["volatility"] = volatility
                
                if volatility < 0.001:  # 極端に低いボラティリティ
                    issues.append("low_volatility")
                elif volatility > 0.1:  # 極端に高いボラティリティ
                    issues.append("high_volatility")
                    
            except Exception:
                issues.append("volatility_calculation_error")

        # 品質スコア計算
        quality_score = 1.0
        quality_score -= min(missing_data_ratio * 2, 0.5)  # 欠損データペナルティ
        quality_score -= len(issues) * 0.1  # 問題数ペナルティ

        return {
            "quality_score": max(quality_score, 0.0),
            "issues": issues,
            "metrics": quality_metrics
        }

    def _assess_parameter_stability(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータ安定性の評価"""
        if not self.optimization_history or not params:
            return {"stability_score": 1.0, "change_frequency": 0.0}

        # 過去のパラメータと比較
        recent_history = self.optimization_history[-5:]  # 直近5期間
        parameter_changes = {}

        for historical_result in recent_history:
            historical_params = historical_result.get("best_parameters", {})
            
            for param_name, current_value in params.items():
                if param_name not in parameter_changes:
                    parameter_changes[param_name] = []
                
                historical_value = historical_params.get(param_name)
                if historical_value is not None:
                    parameter_changes[param_name].append(
                        current_value != historical_value
                    )

        # 安定性スコア計算
        stability_scores = {}
        overall_stability = 1.0

        for param_name, changes in parameter_changes.items():
            if changes:
                change_rate = sum(changes) / len(changes)
                param_stability = 1.0 - change_rate
                stability_scores[param_name] = param_stability
                overall_stability = min(overall_stability, param_stability)

        return {
            "stability_score": overall_stability,
            "parameter_stability": stability_scores,
            "change_frequency": 1.0 - overall_stability
        }

    def _analyze_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """ウォークフォワード結果分析"""
        if not results:
            return {}

        # パフォーマンス統計
        test_performances = [r["test_performance"] for r in results]
        sharpe_ratios = [p.sharpe_ratio for p in test_performances]
        returns = [p.total_return for p in test_performances]
        max_drawdowns = [p.max_drawdown for p in test_performances]

        # 安定性分析
        stability_scores = [
            r.get("parameter_stability", {}).get("stability_score", 1.0) 
            for r in results
        ]

        # データ品質分析
        quality_scores = [
            r.get("data_quality", {}).get("quality_score", 1.0) 
            for r in results
        ]

        analysis = {
            "summary_statistics": {
                "total_periods": len(results),
                "avg_sharpe": np.mean(sharpe_ratios),
                "std_sharpe": np.std(sharpe_ratios),
                "avg_return": np.mean(returns),
                "std_return": np.std(returns),
                "avg_max_drawdown": np.mean(max_drawdowns),
                "worst_drawdown": min(max_drawdowns) if max_drawdowns else 0.0,
            },
            "stability_analysis": {
                "avg_stability_score": np.mean(stability_scores),
                "stability_trend": self._calculate_trend(stability_scores),
                "parameter_consistency": self._analyze_parameter_consistency(results),
            },
            "quality_analysis": {
                "avg_data_quality": np.mean(quality_scores),
                "quality_trend": self._calculate_trend(quality_scores),
            },
            "robustness_score": self._calculate_robustness_score(
                sharpe_ratios, stability_scores, quality_scores
            ),
            "recommendations": self._generate_recommendations(results),
        }

        return analysis

    def _calculate_trend(self, values: List[float]) -> str:
        """値の傾向を計算"""
        if len(values) < 2:
            return "insufficient_data"
            
        slope = np.polyfit(range(len(values)), values, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "deteriorating"
        else:
            return "stable"

    def _analyze_parameter_consistency(self, results: List[Dict]) -> Dict[str, float]:
        """パラメータ一貫性の分析"""
        if len(results) < 2:
            return {}

        parameter_values = {}
        
        for result in results:
            params = result.get("best_parameters", {})
            for param_name, value in params.items():
                if param_name not in parameter_values:
                    parameter_values[param_name] = []
                parameter_values[param_name].append(value)

        consistency_scores = {}
        
        for param_name, values in parameter_values.items():
            if len(set(values)) == 1:
                consistency_scores[param_name] = 1.0  # 完全に一貫
            else:
                # 値の分散に基づく一貫性スコア
                if isinstance(values[0], (int, float)):
                    cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 1
                    consistency_scores[param_name] = max(0, 1 - cv)
                else:
                    # カテゴリカル変数の場合
                    mode_count = max(values.count(val) for val in set(values))
                    consistency_scores[param_name] = mode_count / len(values)

        return consistency_scores

    def _calculate_robustness_score(
        self, 
        sharpe_ratios: List[float], 
        stability_scores: List[float], 
        quality_scores: List[float]
    ) -> float:
        """ロバストネススコアの計算"""
        if not sharpe_ratios:
            return 0.0

        # 各要素の重み
        performance_weight = 0.5
        stability_weight = 0.3
        quality_weight = 0.2

        # 正規化されたスコア
        performance_score = max(0, min(1, np.mean(sharpe_ratios) / 2 + 0.5))
        stability_score = np.mean(stability_scores)
        quality_score = np.mean(quality_scores)

        robustness = (
            performance_score * performance_weight +
            stability_score * stability_weight +
            quality_score * quality_weight
        )

        return robustness

    def _generate_recommendations(self, results: List[Dict]) -> List[str]:
        """最適化結果に基づく推奨事項を生成"""
        recommendations = []

        if not results:
            return ["十分なデータがありません"]

        # パフォーマンス分析
        avg_sharpe = np.mean([r["test_performance"].sharpe_ratio for r in results])
        if avg_sharpe < 0.5:
            recommendations.append("シャープレシオが低いため、戦略の見直しを検討してください")

        # 安定性分析
        avg_stability = np.mean([
            r.get("parameter_stability", {}).get("stability_score", 1.0) 
            for r in results
        ])
        if avg_stability < self.stability_threshold:
            recommendations.append("パラメータの安定性が低いため、より安定したパラメータ設定を検討してください")

        # データ品質分析
        quality_issues = []
        for result in results:
            issues = result.get("data_quality", {}).get("issues", [])
            quality_issues.extend(issues)
        
        if quality_issues:
            recommendations.append(f"データ品質の問題が検出されました: {', '.join(set(quality_issues))}")

        if not recommendations:
            recommendations.append("最適化結果は良好です")

        return recommendations