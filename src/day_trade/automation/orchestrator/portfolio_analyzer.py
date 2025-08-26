"""
ポートフォリオ・システム分析
ポートフォリオ統計とシステムヘルス分析を担当
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ...core.portfolio import PortfolioManager
from ...models.database import get_default_database_manager
from ...utils.logging_config import get_context_logger
from .types import AIAnalysisResult

logger = get_context_logger(__name__)


class PortfolioAnalyzer:
    """ポートフォリオ・システム分析クラス"""

    def __init__(self, core, config):
        """
        初期化

        Args:
            core: オーケストレーターコア
            config: 設定オブジェクト
        """
        self.core = core
        self.config = config

    def generate_portfolio_analysis(
        self, ai_results: List[AIAnalysisResult]
    ) -> Dict[str, Any]:
        """
        ポートフォリオ分析生成

        Args:
            ai_results: AI分析結果リスト

        Returns:
            ポートフォリオ分析結果
        """
        if not ai_results:
            return {
                "status": "analysis_only",
                "trading_disabled": True,
                "analyzed_symbols": 0,
                "message": "分析対象データなし",
            }

        try:
            # 総合統計
            total_symbols = len(ai_results)
            high_confidence_count = sum(
                1
                for r in ai_results
                if r.confidence_scores.get("overall", 0)
                > self.config.confidence_threshold
            )

            # 推奨分布
            recommendations = [r.recommendation for r in ai_results]
            recommendation_counts = {}
            for rec in set(recommendations):
                recommendation_counts[rec] = recommendations.count(rec)

            # 平均データ品質
            avg_data_quality = np.mean([r.data_quality for r in ai_results])

            # リスク分布
            risk_levels = [
                r.risk_assessment.get("risk_level", "unknown") for r in ai_results
            ]
            risk_distribution = {}
            for risk in set(risk_levels):
                risk_distribution[risk] = risk_levels.count(risk)

            # 予測分布
            prediction_stats = self._analyze_predictions(ai_results)

            # セクター分析
            sector_analysis = self._analyze_sectors(ai_results)

            # 相関分析
            correlation_analysis = self._analyze_correlations(ai_results)

            return {
                "status": "analysis_only",
                "trading_disabled": True,
                "analyzed_symbols": total_symbols,
                "high_confidence_predictions": high_confidence_count,
                "recommendation_distribution": recommendation_counts,
                "average_data_quality": avg_data_quality,
                "risk_distribution": risk_distribution,
                "prediction_statistics": prediction_stats,
                "sector_analysis": sector_analysis,
                "correlation_analysis": correlation_analysis,
                "portfolio_metrics": {
                    "total_analysis_value": "N/A (分析専用)",
                    "confidence_weighted_score": np.mean(
                        [r.confidence_scores.get("overall", 0) for r in ai_results]
                    ),
                    "risk_weighted_score": np.mean(
                        [
                            r.risk_assessment.get("overall_risk_score", 0.5)
                            for r in ai_results
                        ]
                    ),
                    "data_quality_score": avg_data_quality,
                },
                "alerts": self._generate_portfolio_alerts(ai_results),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"ポートフォリオ分析エラー: {e}")
            return {
                "status": "analysis_only",
                "trading_disabled": True,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _analyze_predictions(self, ai_results: List[AIAnalysisResult]) -> Dict[str, Any]:
        """予測分析"""
        predictions = []
        confidences = []

        for result in ai_results:
            if "predicted_change" in result.predictions:
                predictions.append(result.predictions["predicted_change"])
                confidences.append(result.confidence_scores.get("overall", 0))

        if not predictions:
            return {"available": False, "reason": "予測データなし"}

        return {
            "available": True,
            "total_predictions": len(predictions),
            "average_change": np.mean(predictions),
            "std_change": np.std(predictions),
            "max_change": np.max(predictions),
            "min_change": np.min(predictions),
            "positive_predictions": sum(1 for p in predictions if p > 0),
            "negative_predictions": sum(1 for p in predictions if p < 0),
            "average_confidence": np.mean(confidences),
            "high_confidence_predictions": sum(
                1 for c in confidences if c > self.config.confidence_threshold
            ),
        }

    def _analyze_sectors(self, ai_results: List[AIAnalysisResult]) -> Dict[str, Any]:
        """セクター分析（銘柄コードベース）"""
        try:
            sector_map = {
                "7203": "自動車",  # トヨタ
                "8306": "銀行",    # 三菱UFJ
                "9984": "通信",    # ソフトバンク
                "6758": "電機",    # ソニー
                "4689": "通信",    # Yahoo
            }

            sector_stats = {}
            
            for result in ai_results:
                sector = sector_map.get(result.symbol, "その他")
                
                if sector not in sector_stats:
                    sector_stats[sector] = {
                        "symbols": [],
                        "recommendations": [],
                        "confidences": [],
                        "risks": [],
                    }
                
                sector_stats[sector]["symbols"].append(result.symbol)
                sector_stats[sector]["recommendations"].append(result.recommendation)
                sector_stats[sector]["confidences"].append(
                    result.confidence_scores.get("overall", 0)
                )
                sector_stats[sector]["risks"].append(
                    result.risk_assessment.get("overall_risk_score", 0.5)
                )

            # セクター別統計計算
            sector_summary = {}
            for sector, data in sector_stats.items():
                sector_summary[sector] = {
                    "symbol_count": len(data["symbols"]),
                    "average_confidence": np.mean(data["confidences"]),
                    "average_risk": np.mean(data["risks"]),
                    "dominant_recommendation": max(
                        set(data["recommendations"]),
                        key=data["recommendations"].count
                    ),
                }

            return {
                "available": True,
                "sectors_analyzed": len(sector_summary),
                "sector_breakdown": sector_summary,
                "most_confident_sector": max(
                    sector_summary.items(),
                    key=lambda x: x[1]["average_confidence"]
                )[0] if sector_summary else None,
                "highest_risk_sector": max(
                    sector_summary.items(),
                    key=lambda x: x[1]["average_risk"]
                )[0] if sector_summary else None,
            }

        except Exception as e:
            logger.error(f"セクター分析エラー: {e}")
            return {
                "available": False,
                "error": str(e),
            }

    def _analyze_correlations(self, ai_results: List[AIAnalysisResult]) -> Dict[str, Any]:
        """相関分析"""
        try:
            if len(ai_results) < 2:
                return {
                    "available": False,
                    "reason": "分析には2銘柄以上必要",
                }

            # 予測値の相関
            predictions = []
            confidences = []
            risks = []

            for result in ai_results:
                predictions.append(result.predictions.get("predicted_change", 0))
                confidences.append(result.confidence_scores.get("overall", 0))
                risks.append(result.risk_assessment.get("overall_risk_score", 0.5))

            # 相関係数計算
            pred_conf_corr = np.corrcoef(predictions, confidences)[0, 1] if len(predictions) > 1 else 0
            pred_risk_corr = np.corrcoef(predictions, risks)[0, 1] if len(predictions) > 1 else 0
            conf_risk_corr = np.corrcoef(confidences, risks)[0, 1] if len(confidences) > 1 else 0

            return {
                "available": True,
                "prediction_confidence_correlation": pred_conf_corr,
                "prediction_risk_correlation": pred_risk_corr,
                "confidence_risk_correlation": conf_risk_corr,
                "correlation_strength": self._interpret_correlation(pred_conf_corr),
                "analysis_summary": {
                    "prediction_range": [min(predictions), max(predictions)],
                    "confidence_range": [min(confidences), max(confidences)],
                    "risk_range": [min(risks), max(risks)],
                },
            }

        except Exception as e:
            logger.error(f"相関分析エラー: {e}")
            return {
                "available": False,
                "error": str(e),
            }

    def _interpret_correlation(self, correlation: float) -> str:
        """相関の強度を解釈"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            return "強い相関"
        elif abs_corr >= 0.4:
            return "中程度の相関"
        elif abs_corr >= 0.2:
            return "弱い相関"
        else:
            return "相関なし"

    def _generate_portfolio_alerts(self, ai_results: List[AIAnalysisResult]) -> List[Dict[str, Any]]:
        """ポートフォリオアラート生成"""
        alerts = []

        try:
            # 高リスク銘柄アラート
            high_risk_count = sum(
                1 for r in ai_results 
                if r.risk_assessment.get("risk_level") == "high"
            )
            
            if high_risk_count > len(ai_results) * 0.5:  # 50%以上が高リスク
                alerts.append({
                    "type": "PORTFOLIO_HIGH_RISK",
                    "message": f"ポートフォリオの{high_risk_count}/{len(ai_results)}銘柄が高リスク",
                    "severity": "high",
                    "timestamp": datetime.now().isoformat(),
                    "recommendation": "リスク分散を検討",
                })

            # 低信頼度アラート
            low_confidence_count = sum(
                1 for r in ai_results 
                if r.confidence_scores.get("overall", 0) < self.config.confidence_threshold
            )
            
            if low_confidence_count > len(ai_results) * 0.7:  # 70%以上が低信頼度
                alerts.append({
                    "type": "PORTFOLIO_LOW_CONFIDENCE",
                    "message": f"ポートフォリオの{low_confidence_count}/{len(ai_results)}銘柄が低信頼度",
                    "severity": "medium",
                    "timestamp": datetime.now().isoformat(),
                    "recommendation": "追加データ収集を推奨",
                })

            # データ品質アラート
            avg_quality = np.mean([r.data_quality for r in ai_results])
            if avg_quality < self.config.data_quality_threshold:
                alerts.append({
                    "type": "PORTFOLIO_DATA_QUALITY",
                    "message": f"ポートフォリオ平均データ品質: {avg_quality:.1f}%",
                    "severity": "medium",
                    "timestamp": datetime.now().isoformat(),
                    "recommendation": "データソースの確認",
                })

        except Exception as e:
            logger.error(f"ポートフォリオアラート生成エラー: {e}")
            alerts.append({
                "type": "PORTFOLIO_ALERT_ERROR",
                "message": f"アラート生成エラー: {e}",
                "severity": "low",
                "timestamp": datetime.now().isoformat(),
            })

        return alerts

    def analyze_system_health(self) -> Dict[str, Any]:
        """システムヘルス分析"""
        try:
            health = {"overall_status": "healthy", "components": {}}

            # MLエンジンヘルス
            if self.core.ml_engine:
                try:
                    ml_summary = self.core.ml_engine.get_model_summary()
                    health["components"]["ml_engine"] = {
                        "status": "operational",
                        "model_loaded": ml_summary.get("status") != "モデル未初期化",
                        "device": ml_summary.get("device", "unknown"),
                    }
                except Exception as e:
                    health["components"]["ml_engine"] = {
                        "status": "error",
                        "error": str(e),
                    }

            # バッチフェッチャーヘルス
            if self.core.batch_fetcher:
                try:
                    batch_stats = self.core.batch_fetcher.get_pipeline_stats()
                    health["components"]["batch_fetcher"] = {
                        "status": "operational",
                        "throughput": batch_stats.throughput_rps,
                        "success_rate": (
                            batch_stats.successful_requests / batch_stats.total_requests
                            if batch_stats.total_requests > 0
                            else 1.0
                        ),
                    }
                except Exception as e:
                    health["components"]["batch_fetcher"] = {
                        "status": "error",
                        "error": str(e),
                    }

            # パフォーマンス監視ヘルス
            if self.core.performance_monitor:
                try:
                    health["components"]["performance_monitor"] = {
                        "status": "operational",
                        "monitoring_active": True,
                    }
                except Exception as e:
                    health["components"]["performance_monitor"] = {
                        "status": "error",
                        "error": str(e),
                    }

            # 並列マネージャーヘルス
            if self.core.parallel_manager:
                try:
                    health["components"]["parallel_manager"] = {
                        "status": "operational",
                        "thread_workers": self.config.max_thread_workers,
                        "process_workers": self.config.max_process_workers,
                    }
                except Exception as e:
                    health["components"]["parallel_manager"] = {
                        "status": "error",
                        "error": str(e),
                    }

            # 全体ステータス判定
            component_statuses = [
                comp.get("status") for comp in health["components"].values()
            ]
            if "error" in component_statuses:
                health["overall_status"] = "degraded"

            health["timestamp"] = datetime.now().isoformat()
            health["components_count"] = len(health["components"])
            health["healthy_components"] = sum(
                1 for comp in health["components"].values() 
                if comp.get("status") == "operational"
            )

            return health

        except Exception as e:
            return {
                "overall_status": "error", 
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def calculate_performance_stats(self, start_time: datetime) -> Dict[str, Any]:
        """パフォーマンス統計計算"""
        try:
            execution_time = (datetime.now() - start_time).total_seconds()

            stats = {
                "execution_time_seconds": execution_time,
                "timestamp": datetime.now().isoformat(),
                "performance_grade": self._get_performance_grade(execution_time),
            }

            # バッチフェッチャー統計
            if self.core.batch_fetcher:
                batch_stats = self.core.batch_fetcher.get_pipeline_stats()
                stats["batch_fetcher"] = {
                    "total_requests": batch_stats.total_requests,
                    "success_rate": (
                        batch_stats.successful_requests / batch_stats.total_requests
                        if batch_stats.total_requests > 0
                        else 0
                    ),
                    "avg_fetch_time": batch_stats.avg_fetch_time,
                    "throughput_rps": batch_stats.throughput_rps,
                }

            # MLエンジン統計
            if (self.core.ml_engine and 
                hasattr(self.core.ml_engine, 'performance_history') and
                self.core.ml_engine.performance_history):
                avg_inference_time = np.mean(
                    [p["inference_time"] for p in self.core.ml_engine.performance_history]
                )
                stats["ml_engine"] = {
                    "predictions_made": len(self.core.ml_engine.performance_history),
                    "avg_inference_time": avg_inference_time,
                    "model_version": self.core.ml_engine.model_metadata.get("version", "unknown"),
                }

            # 並列マネージャー統計
            if self.core.parallel_manager:
                perf_stats = self.core.parallel_manager.get_performance_stats()
                stats["parallel_execution"] = {
                    "executors": len(perf_stats),
                    "avg_task_time": np.mean([
                        s["average_time_ms"] for s in perf_stats.values()
                    ]) if perf_stats else 0,
                    "overall_success_rate": np.mean([
                        s["success_rate"] for s in perf_stats.values()
                    ]) if perf_stats else 0,
                }

            return stats

        except Exception as e:
            return {"error": str(e), "execution_time_seconds": 0}

    def _get_performance_grade(self, execution_time: float) -> str:
        """パフォーマンス評価"""
        if execution_time < 30:
            return "excellent"
        elif execution_time < 60:
            return "good"
        elif execution_time < 120:
            return "fair"
        else:
            return "poor"