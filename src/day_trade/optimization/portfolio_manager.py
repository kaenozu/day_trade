#!/usr/bin/env python3
"""
ポートフォリオ統合管理システム

ポートフォリオ最適化、リスク管理、セクター分析を統合した
包括的なポートフォリオ管理システム
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..data.advanced_ml_engine import AdvancedMLEngine
from ..data.batch_data_fetcher import BatchDataFetcher
from ..utils.logging_config import get_context_logger
from .portfolio_optimizer import PortfolioOptimizer
from .risk_manager import RiskManager
from .sector_analyzer import SectorAnalyzer

logger = get_context_logger(__name__)


class PortfolioManager:
    """
    統合ポートフォリオ管理システム

    最適化、リスク管理、セクター分析、ML投資助言を統合
    """

    def __init__(
        self,
        investment_amount: float = 1000000,
        risk_tolerance: float = 0.5,
        rebalancing_threshold: float = 0.05,
    ):
        """
        初期化

        Args:
            investment_amount: 投資金額
            risk_tolerance: リスク許容度 (0-1)
            rebalancing_threshold: リバランシング閾値
        """
        self.investment_amount = investment_amount
        self.risk_tolerance = risk_tolerance
        self.rebalancing_threshold = rebalancing_threshold

        # コンポーネント初期化
        self.optimizer = PortfolioOptimizer(
            risk_tolerance=risk_tolerance,
            max_position_size=0.15,  # 1銘柄最大15%
            min_position_size=0.01,  # 最小1%
        )
        self.risk_manager = RiskManager(
            confidence_level=0.05,
            max_correlation=0.7,
        )
        self.sector_analyzer = SectorAnalyzer()
        self.ml_engine = AdvancedMLEngine()
        self.data_fetcher = BatchDataFetcher(max_workers=5)

        logger.info("ポートフォリオマネージャー初期化:")
        logger.info(f"  - 投資金額: {investment_amount:,.0f}円")
        logger.info(f"  - リスク許容度: {risk_tolerance:.1f}")
        logger.info(f"  - リバランシング閾値: {rebalancing_threshold:.1%}")

    def generate_comprehensive_portfolio(
        self, symbols: List[str], use_ml_signals: bool = True
    ) -> Dict:
        """
        包括的ポートフォリオ生成

        Args:
            symbols: 対象銘柄リスト
            use_ml_signals: ML投資助言の使用

        Returns:
            包括的ポートフォリオ推奨
        """
        logger.info(f"包括的ポートフォリオ生成開始: {len(symbols)}銘柄")

        try:
            # 1. データ取得
            logger.info("マーケットデータ取得中...")
            price_data = self.data_fetcher.fetch_multiple_symbols(
                symbols, period="60d", use_parallel=True
            )

            if len(price_data) < 5:
                logger.error(f"データ不足: {len(price_data)}銘柄のみ取得")
                return self._get_fallback_portfolio(symbols)

            logger.info(f"データ取得完了: {len(price_data)}銘柄")

            # 2. ML投資助言生成（オプション）
            ml_signals = {}
            if use_ml_signals:
                logger.info("ML投資助言生成中...")
                try:
                    for symbol, data in price_data.items():
                        if not data.empty:
                            features = self.ml_engine.prepare_ml_features(data)
                            advice = self.ml_engine.generate_investment_advice(
                                symbol, data, features
                            )
                            ml_signals[symbol] = advice
                    logger.info(f"ML助言生成完了: {len(ml_signals)}銘柄")
                except Exception as e:
                    logger.warning(f"ML助言生成エラー: {e}")
                    ml_signals = {}

            # 3. ポートフォリオ最適化
            logger.info("ポートフォリオ最適化実行中...")
            portfolio_recommendation = self.optimizer.generate_portfolio_recommendation(
                price_data, self.investment_amount
            )

            # 4. セクター制約適用
            logger.info("セクター制約チェック中...")
            initial_weights = portfolio_recommendation["optimal_portfolio"][
                "allocations"
            ]
            weights_dict = {k: v["weight"] for k, v in initial_weights.items()}

            sector_optimization = self.sector_analyzer.optimize_sector_allocation(
                weights_dict, target_return=0.10, risk_tolerance=self.risk_tolerance
            )

            # 最適化後のウェイトを使用
            optimized_weights = sector_optimization["optimized_weights"]

            # 5. リスク分析
            logger.info("リスク分析実行中...")
            returns_data = self._calculate_returns_data(price_data)
            risk_report = self.risk_manager.generate_risk_report(
                optimized_weights, returns_data, self.investment_amount
            )

            # 6. 最終ポートフォリオ構築
            final_allocations = self._build_final_allocations(
                optimized_weights, ml_signals, price_data
            )

            # 7. 包括的推奨生成
            comprehensive_portfolio = {
                "timestamp": datetime.now(),
                "investment_amount": self.investment_amount,
                "portfolio_allocations": final_allocations,
                "optimization_results": {
                    "initial_optimization": portfolio_recommendation,
                    "sector_optimization": sector_optimization,
                    "final_weights": optimized_weights,
                },
                "risk_analysis": risk_report,
                "ml_signals": ml_signals,
                "portfolio_metrics": self._calculate_portfolio_metrics(
                    final_allocations, price_data
                ),
                "recommendations": self._generate_portfolio_recommendations(
                    final_allocations, risk_report, ml_signals
                ),
                "execution_plan": self._generate_execution_plan(final_allocations),
            }

            logger.info("包括的ポートフォリオ生成完了")
            return comprehensive_portfolio

        except Exception as e:
            logger.error(f"ポートフォリオ生成エラー: {e}")
            return self._get_fallback_portfolio(symbols)

    def analyze_current_portfolio(self, current_holdings: Dict[str, float]) -> Dict:
        """
        現在のポートフォリオ分析

        Args:
            current_holdings: 現在の保有ウェイト

        Returns:
            ポートフォリオ分析結果
        """
        logger.info("現在ポートフォリオ分析開始")

        try:
            symbols = list(current_holdings.keys())

            # データ取得
            price_data = self.data_fetcher.fetch_multiple_symbols(
                symbols, period="30d", use_parallel=True
            )

            # リスク分析
            returns_data = self._calculate_returns_data(price_data)
            risk_analysis = self.risk_manager.generate_risk_report(
                current_holdings, returns_data, self.investment_amount
            )

            # セクター分析
            sector_analysis = self.sector_analyzer.analyze_sector_allocation(
                current_holdings
            )

            # パフォーマンス分析
            performance_metrics = self._analyze_portfolio_performance(
                current_holdings, price_data
            )

            portfolio_analysis = {
                "timestamp": datetime.now(),
                "current_holdings": current_holdings,
                "risk_analysis": risk_analysis,
                "sector_analysis": sector_analysis,
                "performance_metrics": performance_metrics,
                "health_score": self._calculate_portfolio_health_score(
                    risk_analysis, sector_analysis, performance_metrics
                ),
                "improvement_suggestions": self._generate_improvement_suggestions(
                    current_holdings, risk_analysis, sector_analysis
                ),
            }

            logger.info("現在ポートフォリオ分析完了")
            return portfolio_analysis

        except Exception as e:
            logger.error(f"ポートフォリオ分析エラー: {e}")
            return {"error": str(e)}

    def generate_rebalancing_plan(
        self,
        current_holdings: Dict[str, float],
        target_symbols: Optional[List[str]] = None,
    ) -> Dict:
        """
        リバランシング計画生成

        Args:
            current_holdings: 現在の保有
            target_symbols: 目標銘柄リスト（None時は現在保有を基準）

        Returns:
            リバランシング計画
        """
        logger.info("リバランシング計画生成開始")

        try:
            # 目標ポートフォリオ生成
            if target_symbols is None:
                target_symbols = list(current_holdings.keys())

            target_portfolio = self.generate_comprehensive_portfolio(target_symbols)
            target_weights = target_portfolio["portfolio_allocations"]
            target_weights_dict = {k: v["weight"] for k, v in target_weights.items()}

            # リバランシング提案
            rebalancing_proposal = self.sector_analyzer.generate_rebalancing_proposal(
                current_holdings,
                target_weights_dict,
                self.rebalancing_threshold,
            )

            # リバランシング必要性評価
            deviation_analysis = self._analyze_weight_deviations(
                current_holdings, target_weights_dict
            )

            rebalancing_plan = {
                "timestamp": datetime.now(),
                "current_holdings": current_holdings,
                "target_portfolio": target_weights_dict,
                "rebalancing_proposal": rebalancing_proposal,
                "deviation_analysis": deviation_analysis,
                "rebalancing_necessity": {
                    "is_required": deviation_analysis["max_deviation"]
                    > self.rebalancing_threshold,
                    "urgency_level": self._assess_rebalancing_urgency(
                        deviation_analysis
                    ),
                    "estimated_benefit": self._estimate_rebalancing_benefit(
                        current_holdings, target_weights_dict
                    ),
                },
                "execution_recommendations": self._generate_execution_recommendations(
                    rebalancing_proposal
                ),
            }

            logger.info(
                f"リバランシング計画完了: 取引{rebalancing_proposal['summary']['total_trades']}件"
            )
            return rebalancing_plan

        except Exception as e:
            logger.error(f"リバランシング計画エラー: {e}")
            return {"error": str(e)}

    def save_portfolio_report(
        self, portfolio_data: Dict, output_dir: str = "reports"
    ) -> str:
        """
        ポートフォリオレポート保存

        Args:
            portfolio_data: ポートフォリオデータ
            output_dir: 出力ディレクトリ

        Returns:
            保存先ファイルパス
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_report_{timestamp}.json"
            filepath = output_path / filename

            # JSON保存用にタイムスタンプを文字列に変換
            save_data = self._prepare_data_for_json(portfolio_data)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            logger.info(f"ポートフォリオレポート保存: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"レポート保存エラー: {e}")
            return ""

    def _calculate_returns_data(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """価格データから収益率データを計算"""
        returns_dict = {}
        for symbol, data in price_data.items():
            if not data.empty and "Close" in data.columns:
                daily_returns = data["Close"].pct_change().dropna()
                if len(daily_returns) > 5:
                    returns_dict[symbol] = daily_returns

        returns_df = pd.DataFrame(returns_dict)

        # インデックスのタイムゾーン情報を除去して統一
        if hasattr(returns_df.index, "tz") and returns_df.index.tz is not None:
            returns_df.index = returns_df.index.tz_localize(None)

        return returns_df.fillna(method="ffill").fillna(method="bfill")

    def _build_final_allocations(
        self,
        weights: Dict[str, float],
        ml_signals: Dict,
        price_data: Dict[str, pd.DataFrame],
    ) -> Dict:
        """最終配分構築"""
        allocations = {}

        for symbol, weight in weights.items():
            if weight >= 0.01:  # 最小ポジション以上
                allocation_data = {
                    "symbol": symbol,
                    "weight": weight,
                    "amount": weight * self.investment_amount,
                }

                # ML信号情報追加
                if symbol in ml_signals:
                    ml_signal = ml_signals[symbol]
                    allocation_data.update(
                        {
                            "ml_advice": ml_signal.get("advice", "HOLD"),
                            "ml_confidence": ml_signal.get("confidence", 50.0),
                            "ml_risk_level": ml_signal.get("risk_level", "MEDIUM"),
                        }
                    )

                # 価格情報追加
                if symbol in price_data and not price_data[symbol].empty:
                    current_price = price_data[symbol]["Close"].iloc[-1]
                    allocation_data["current_price"] = current_price
                    allocation_data["target_shares"] = int(
                        allocation_data["amount"] / current_price
                    )

                allocations[symbol] = allocation_data

        return allocations

    def _calculate_portfolio_metrics(
        self, allocations: Dict, price_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """ポートフォリオ指標計算"""
        try:
            symbols = list(allocations.keys())
            weights = [allocations[s]["weight"] for s in symbols]

            # 基本統計
            n_positions = len(allocations)
            max_position = max(weights) if weights else 0
            herfindahl_index = sum(w**2 for w in weights)
            effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 0

            # 期待リターン・リスク計算
            returns_data = self._calculate_returns_data(price_data)
            available_symbols = [s for s in symbols if s in returns_data.columns]

            if len(available_symbols) >= 2:
                portfolio_returns = returns_data[available_symbols].mean() * 252

                weights_available = [
                    allocations[s]["weight"] for s in available_symbols
                ]
                weights_normalized = [
                    w / sum(weights_available) for w in weights_available
                ]

                expected_return = sum(
                    w * portfolio_returns.loc[s]
                    for w, s in zip(weights_normalized, available_symbols)
                )
                portfolio_volatility = (
                    returns_data[available_symbols] @ weights_normalized
                ).std() * (252**0.5)

                sharpe_ratio = (
                    expected_return / portfolio_volatility
                    if portfolio_volatility > 0
                    else 0
                )
            else:
                expected_return = 0.08
                portfolio_volatility = 0.15
                sharpe_ratio = 0.5

            return {
                "n_positions": n_positions,
                "max_position_weight": max_position,
                "herfindahl_index": herfindahl_index,
                "effective_positions": effective_positions,
                "expected_return": expected_return,
                "portfolio_volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
            }

        except Exception as e:
            logger.error(f"ポートフォリオ指標計算エラー: {e}")
            return {"error": str(e)}

    def _generate_portfolio_recommendations(
        self, allocations: Dict, risk_report: Dict, ml_signals: Dict
    ) -> List[str]:
        """ポートフォリオ推奨事項生成"""
        recommendations = []

        # 分散性評価
        n_positions = len(allocations)
        if n_positions >= 15:
            recommendations.append("適切に分散されたポートフォリオです")
        elif n_positions >= 10:
            recommendations.append("良好な分散レベルです")
        else:
            recommendations.append("更なる分散化を検討してください")

        # リスク評価
        if "overall_risk_level" in risk_report:
            risk_level = risk_report["overall_risk_level"]
            if risk_level == "LOW_RISK":
                recommendations.append("低リスクポートフォリオです")
            elif risk_level == "HIGH_RISK":
                recommendations.append("高リスクです。リスク管理を強化してください")

        # ML信号評価
        if ml_signals:
            buy_signals = len(
                [s for s in ml_signals.values() if s.get("advice") == "BUY"]
            )
            total_signals = len(ml_signals)
            buy_ratio = buy_signals / total_signals if total_signals > 0 else 0

            if buy_ratio > 0.6:
                recommendations.append("多くの銘柄でBUY信号が出ています")
            elif buy_ratio < 0.3:
                recommendations.append("慎重な投資環境です")

        if not recommendations:
            recommendations.append("バランスの取れたポートフォリオです")

        return recommendations

    def _generate_execution_plan(self, allocations: Dict) -> Dict:
        """実行計画生成"""
        symbols = list(allocations.keys())
        n_symbols = len(symbols)

        # 実行時間推定
        estimated_execution_time = n_symbols * 30  # 1銘柄30秒
        batch_size = min(5, n_symbols)  # 最大5銘柄ずつ実行

        return {
            "total_positions": n_symbols,
            "estimated_execution_time_seconds": estimated_execution_time,
            "batch_size": batch_size,
            "n_batches": (n_symbols + batch_size - 1) // batch_size,
            "execution_order": "market_cap_descending",  # 時価総額降順
        }

    def _analyze_portfolio_performance(
        self, holdings: Dict[str, float], price_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """ポートフォリオパフォーマンス分析"""
        try:
            returns_data = self._calculate_returns_data(price_data)
            available_symbols = [s for s in holdings if s in returns_data.columns]

            if len(available_symbols) < 2:
                return {"error": "パフォーマンス分析に十分なデータがありません"}

            # ポートフォリオリターン計算
            weights = [holdings[s] for s in available_symbols]
            weights_normalized = [w / sum(weights) for w in weights]

            portfolio_returns = (
                returns_data[available_symbols] * weights_normalized
            ).sum(axis=1)

            # パフォーマンス指標
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * (252**0.5)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

            # ドローダウン計算
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            return {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "data_period_days": len(portfolio_returns),
            }

        except Exception as e:
            logger.error(f"パフォーマンス分析エラー: {e}")
            return {"error": str(e)}

    def _calculate_portfolio_health_score(
        self, risk_analysis: Dict, sector_analysis: Dict, performance_metrics: Dict
    ) -> Dict:
        """ポートフォリオ健全性スコア計算"""
        try:
            scores = {}
            weights = {}

            # リスク評価スコア
            if "overall_risk_level" in risk_analysis:
                risk_level = risk_analysis["overall_risk_level"]
                if risk_level == "LOW_RISK":
                    scores["risk"] = 90
                elif risk_level == "MEDIUM_RISK":
                    scores["risk"] = 70
                else:
                    scores["risk"] = 40
                weights["risk"] = 0.3

            # セクター分散スコア
            if "diversification_metrics" in sector_analysis:
                effective_sectors = sector_analysis["diversification_metrics"].get(
                    "effective_sectors", 1
                )
                sector_score = min(100, effective_sectors * 15)  # 7セクター=100点
                scores["diversification"] = sector_score
                weights["diversification"] = 0.25

            # パフォーマンススコア
            if "sharpe_ratio" in performance_metrics:
                sharpe = performance_metrics["sharpe_ratio"]
                if sharpe > 1.0:
                    perf_score = 90
                elif sharpe > 0.5:
                    perf_score = 70
                elif sharpe > 0:
                    perf_score = 50
                else:
                    perf_score = 20
                scores["performance"] = perf_score
                weights["performance"] = 0.25

            # 制約適合スコア
            if "compliance" in sector_analysis:
                compliance_score = (
                    90 if sector_analysis["compliance"]["is_compliant"] else 50
                )
                scores["compliance"] = compliance_score
                weights["compliance"] = 0.2

            # 総合スコア計算
            if scores and weights:
                total_weight = sum(weights.values())
                weighted_score = (
                    sum(scores[k] * weights[k] for k in scores if k in weights)
                    / total_weight
                )

                # スコアレベル判定
                if weighted_score >= 80:
                    level = "EXCELLENT"
                elif weighted_score >= 70:
                    level = "GOOD"
                elif weighted_score >= 60:
                    level = "FAIR"
                else:
                    level = "POOR"

                return {
                    "overall_score": weighted_score,
                    "score_level": level,
                    "component_scores": scores,
                    "component_weights": weights,
                }
            else:
                return {"error": "スコア計算に必要なデータが不足"}

        except Exception as e:
            logger.error(f"健全性スコア計算エラー: {e}")
            return {"error": str(e)}

    def _generate_improvement_suggestions(
        self, holdings: Dict, risk_analysis: Dict, sector_analysis: Dict
    ) -> List[str]:
        """改善提案生成"""
        suggestions = []

        # セクター集中の改善
        if "constraint_violations" in sector_analysis:
            violations = sector_analysis["constraint_violations"]
            if len(violations) > 0:
                suggestions.append(
                    f"{len(violations)}個のセクターで制約違反があります。分散を見直してください"
                )

        # リスク改善
        if (
            "overall_risk_level" in risk_analysis
            and risk_analysis["overall_risk_level"] == "HIGH_RISK"
        ):
            suggestions.append(
                "リスクレベルが高いため、ポジションサイズの見直しを検討してください"
            )

        # 分散改善
        n_positions = len(holdings)
        if n_positions < 8:
            suggestions.append("更なる銘柄分散でリスク低減効果が期待できます")

        # ポジションサイズ改善
        max_weight = max(holdings.values()) if holdings else 0
        if max_weight > 0.20:
            suggestions.append(
                "単一銘柄への集中度が高いため、分散投資を検討してください"
            )

        if not suggestions:
            suggestions.append("現在のポートフォリオ構成は良好です")

        return suggestions

    def _analyze_weight_deviations(
        self, current: Dict[str, float], target: Dict[str, float]
    ) -> Dict:
        """ウェイト乖離分析"""
        all_symbols = set(current.keys()) | set(target.keys())
        deviations = {}
        absolute_deviations = []

        for symbol in all_symbols:
            current_weight = current.get(symbol, 0)
            target_weight = target.get(symbol, 0)
            deviation = target_weight - current_weight

            deviations[symbol] = {
                "current": current_weight,
                "target": target_weight,
                "deviation": deviation,
                "absolute_deviation": abs(deviation),
            }
            absolute_deviations.append(abs(deviation))

        return {
            "symbol_deviations": deviations,
            "max_deviation": max(absolute_deviations) if absolute_deviations else 0,
            "average_deviation": (
                sum(absolute_deviations) / len(absolute_deviations)
                if absolute_deviations
                else 0
            ),
            "total_absolute_deviation": sum(absolute_deviations),
        }

    def _assess_rebalancing_urgency(self, deviation_analysis: Dict) -> str:
        """リバランシング緊急度評価"""
        max_dev = deviation_analysis["max_deviation"]
        avg_dev = deviation_analysis["average_deviation"]

        if max_dev > 0.10 or avg_dev > 0.05:
            return "HIGH"
        elif max_dev > 0.07 or avg_dev > 0.03:
            return "MEDIUM"
        else:
            return "LOW"

    def _estimate_rebalancing_benefit(
        self, current: Dict[str, float], target: Dict[str, float]
    ) -> Dict:
        """リバランシング効果推定"""
        # 簡易的な効果推定
        deviation_reduction = sum(
            abs(target.get(s, 0) - current.get(s, 0))
            for s in set(current.keys()) | set(target.keys())
        )

        expected_risk_reduction = min(
            0.05, deviation_reduction * 0.1
        )  # 最大5%リスク削減
        expected_return_improvement = min(
            0.02, deviation_reduction * 0.05
        )  # 最大2%リターン改善

        return {
            "expected_risk_reduction": expected_risk_reduction,
            "expected_return_improvement": expected_return_improvement,
            "deviation_reduction": deviation_reduction,
        }

    def _generate_execution_recommendations(
        self, rebalancing_proposal: Dict
    ) -> List[str]:
        """実行推奨事項生成"""
        recommendations = []
        summary = rebalancing_proposal.get("summary", {})

        total_trades = summary.get("total_trades", 0)
        if total_trades == 0:
            recommendations.append("リバランシングは不要です")
            return recommendations

        # 実行順序推奨
        recommendations.append("売り注文を先に実行してください")
        recommendations.append("市場の流動性が高い時間帯での実行を推奨します")

        # 分割実行推奨
        if total_trades > 10:
            recommendations.append("大口取引は複数回に分割して執行してください")

        # コスト管理
        cost_estimate = rebalancing_proposal.get("cost_estimate", {})
        if cost_estimate.get("cost_percentage", 0) > 0.5:
            recommendations.append(
                "取引コストが高いため、実行タイミングを検討してください"
            )

        return recommendations

    def _prepare_data_for_json(self, data: Dict) -> Dict:
        """JSON保存用データ変換"""
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                if isinstance(v, (pd.Timestamp, datetime)):
                    result[k] = v.isoformat()
                elif isinstance(v, (pd.Series, pd.DataFrame)):
                    result[k] = v.to_dict()
                elif isinstance(v, dict):
                    result[k] = self._prepare_data_for_json(v)
                elif isinstance(v, list):
                    result[k] = [
                        (
                            self._prepare_data_for_json(item)
                            if isinstance(item, dict)
                            else item
                        )
                        for item in v
                    ]
                else:
                    result[k] = v
            return result
        return data

    def _get_fallback_portfolio(self, symbols: List[str]) -> Dict:
        """フォールバックポートフォリオ"""
        n_symbols = min(len(symbols), 10)
        equal_weight = 1.0 / n_symbols

        fallback_allocations = {}
        for symbol in symbols[:n_symbols]:
            fallback_allocations[symbol] = {
                "symbol": symbol,
                "weight": equal_weight,
                "amount": equal_weight * self.investment_amount,
                "ml_advice": "HOLD",
                "ml_confidence": 50.0,
            }

        return {
            "timestamp": datetime.now(),
            "investment_amount": self.investment_amount,
            "portfolio_allocations": fallback_allocations,
            "error": "データ不足のため等ウェイトポートフォリオを生成",
            "portfolio_metrics": {
                "n_positions": n_symbols,
                "expected_return": 0.08,
                "portfolio_volatility": 0.20,
            },
        }


if __name__ == "__main__":
    # 使用例
    print("ポートフォリオマネージャーテスト")

    # テスト用銘柄
    test_symbols = [
        "7203",
        "8306",
        "9984",
        "6758",
        "4689",  # 主要5銘柄
        "4563",
        "4592",
        "3655",
        "4382",
        "4475",  # 新興5銘柄
    ]

    try:
        # ポートフォリオマネージャー初期化
        manager = PortfolioManager(
            investment_amount=1000000,  # 100万円
            risk_tolerance=0.6,  # やや積極的
            rebalancing_threshold=0.05,  # 5%閾値
        )

        print("包括的ポートフォリオ生成テスト実行中...")

        # 包括的ポートフォリオ生成
        comprehensive_portfolio = manager.generate_comprehensive_portfolio(
            test_symbols, use_ml_signals=True
        )

        if "error" not in comprehensive_portfolio:
            print("\n=== ポートフォリオ配分 ===")
            for symbol, allocation in comprehensive_portfolio[
                "portfolio_allocations"
            ].items():
                print(
                    f"{symbol}: {allocation['weight']:.1%} ({allocation['amount']:,.0f}円)"
                )

            print("\n=== ポートフォリオ指標 ===")
            metrics = comprehensive_portfolio["portfolio_metrics"]
            print(f"ポジション数: {metrics['n_positions']}")
            print(f"期待リターン: {metrics['expected_return']:.2%}")
            print(f"ボラティリティ: {metrics['portfolio_volatility']:.2%}")
            print(f"シャープレシオ: {metrics['sharpe_ratio']:.2f}")

            print("\n=== 推奨事項 ===")
            for rec in comprehensive_portfolio["recommendations"]:
                print(f"• {rec}")

            # レポート保存
            report_path = manager.save_portfolio_report(comprehensive_portfolio)
            if report_path:
                print(f"\nレポート保存: {report_path}")

        else:
            print(f"エラー: {comprehensive_portfolio['error']}")

    except Exception as e:
        print(f"テストエラー: {e}")
