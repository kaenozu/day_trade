"""
システム監視・リスク評価モジュール
システムヘルス監視、パフォーマンス統計、ポートフォリオ分析を担当
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ...utils.logging_config import get_context_logger
from .config import AIAnalysisResult, CI_MODE, OrchestrationConfig

logger = get_context_logger(__name__)

# 条件付きインポート
if not CI_MODE:
    try:
        from ...data.advanced_ml_engine import AdvancedMLEngine
        from ...data.batch_data_fetcher import AdvancedBatchDataFetcher
        from ...utils.performance_monitor import PerformanceMonitor
    except ImportError:
        AdvancedMLEngine = None
        AdvancedBatchDataFetcher = None
        PerformanceMonitor = None
        logger.warning("監視関連モジュールのインポートに失敗")
else:
    AdvancedMLEngine = None
    AdvancedBatchDataFetcher = None
    PerformanceMonitor = None


class SystemMonitor:
    """
    システム監視・リスク評価エンジン
    
    システムヘルス、パフォーマンス、ポートフォリオを
    統合的に監視・評価します。
    """

    def __init__(self, config: OrchestrationConfig):
        """
        初期化
        
        Args:
            config: オーケストレーション設定
        """
        self.config = config
        self.ml_engine: Optional[AdvancedMLEngine] = None
        self.batch_fetcher: Optional[AdvancedBatchDataFetcher] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None

    def set_ml_engine(self, ml_engine: Optional[AdvancedMLEngine]) -> None:
        """MLエンジンを設定"""
        self.ml_engine = ml_engine

    def set_batch_fetcher(self, batch_fetcher: Optional[AdvancedBatchDataFetcher]) -> None:
        """バッチフェッチャーを設定"""
        self.batch_fetcher = batch_fetcher

    def set_performance_monitor(self, performance_monitor: Optional[PerformanceMonitor]) -> None:
        """パフォーマンス監視を設定"""
        self.performance_monitor = performance_monitor

    def analyze_system_health(self) -> Dict[str, Any]:
        """
        システムヘルス分析
        
        Returns:
            Dict[str, Any]: システムヘルス情報
        """
        try:
            health = {"overall_status": "healthy", "components": {}}

            # MLエンジンヘルス
            if self.ml_engine:
                try:
                    ml_summary = self.ml_engine.get_model_summary()
                    health["components"]["ml_engine"] = {
                        "status": "operational",
                        "model_loaded": ml_summary.get("status") != "モデル未初期化",
                        "device": ml_summary.get("device", "unknown"),
                        "model_version": ml_summary.get("version", "unknown"),
                        "last_prediction": ml_summary.get("last_prediction_time"),
                    }
                except Exception as e:
                    health["components"]["ml_engine"] = {
                        "status": "error",
                        "error": str(e),
                    }
            else:
                health["components"]["ml_engine"] = {
                    "status": "disabled",
                    "reason": "CI mode or not initialized",
                }

            # バッチフェッチャーヘルス
            if self.batch_fetcher:
                try:
                    batch_stats = self.batch_fetcher.get_pipeline_stats()
                    success_rate = (
                        batch_stats.successful_requests / batch_stats.total_requests
                        if batch_stats.total_requests > 0
                        else 1.0
                    )
                    
                    health["components"]["batch_fetcher"] = {
                        "status": "operational" if success_rate > 0.8 else "degraded",
                        "throughput": batch_stats.throughput_rps,
                        "success_rate": success_rate,
                        "total_requests": batch_stats.total_requests,
                        "avg_fetch_time": batch_stats.avg_fetch_time,
                    }
                except Exception as e:
                    health["components"]["batch_fetcher"] = {
                        "status": "error",
                        "error": str(e),
                    }
            else:
                health["components"]["batch_fetcher"] = {
                    "status": "disabled",
                    "reason": "CI mode or not initialized",
                }

            # パフォーマンス監視ヘルス
            if self.performance_monitor:
                try:
                    health["components"]["performance_monitor"] = {
                        "status": "operational",
                        "monitoring_active": True,
                        "uptime": self.performance_monitor.get_uptime(),
                    }
                except Exception as e:
                    health["components"]["performance_monitor"] = {
                        "status": "error",
                        "error": str(e),
                    }
            else:
                health["components"]["performance_monitor"] = {
                    "status": "disabled",
                    "reason": "CI mode or not initialized",
                }

            # リソース使用状況
            health["resources"] = self._get_resource_usage()

            # 全体ステータス判定
            component_statuses = [
                comp.get("status") for comp in health["components"].values()
            ]
            
            error_count = component_statuses.count("error")
            degraded_count = component_statuses.count("degraded")
            
            if error_count > 1:
                health["overall_status"] = "critical"
            elif error_count > 0 or degraded_count > 0:
                health["overall_status"] = "degraded"
            else:
                health["overall_status"] = "healthy"

            # ヘルススコア計算
            health["health_score"] = self._calculate_health_score(health["components"])

            return health

        except Exception as e:
            return {
                "overall_status": "error",
                "health_score": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def calculate_performance_stats(self, start_time: datetime) -> Dict[str, Any]:
        """
        パフォーマンス統計計算
        
        Args:
            start_time: 実行開始時刻
            
        Returns:
            Dict[str, Any]: パフォーマンス統計
        """
        try:
            execution_time = (datetime.now() - start_time).total_seconds()

            stats = {
                "execution_time_seconds": execution_time,
                "timestamp": datetime.now().isoformat(),
                "start_time": start_time.isoformat(),
                "system_performance": {
                    "avg_processing_time": execution_time,
                    "throughput_estimate": 1.0 / execution_time if execution_time > 0 else 0,
                }
            }

            # バッチフェッチャー統計
            if self.batch_fetcher:
                try:
                    batch_stats = self.batch_fetcher.get_pipeline_stats()
                    stats["batch_fetcher"] = {
                        "total_requests": batch_stats.total_requests,
                        "success_rate": (
                            batch_stats.successful_requests / batch_stats.total_requests
                            if batch_stats.total_requests > 0
                            else 0
                        ),
                        "avg_fetch_time": batch_stats.avg_fetch_time,
                        "throughput_rps": batch_stats.throughput_rps,
                        "failed_requests": (
                            batch_stats.total_requests - batch_stats.successful_requests
                        ),
                    }
                except Exception as e:
                    stats["batch_fetcher"] = {"error": str(e)}

            # MLエンジン統計
            if self.ml_engine and hasattr(self.ml_engine, 'performance_history'):
                try:
                    perf_history = self.ml_engine.performance_history
                    if perf_history:
                        avg_inference_time = np.mean(
                            [p.get("inference_time", 0) for p in perf_history]
                        )
                        stats["ml_engine"] = {
                            "predictions_made": len(perf_history),
                            "avg_inference_time": avg_inference_time,
                            "model_version": self.ml_engine.model_metadata.get("version", "unknown"),
                            "total_inference_time": sum(
                                p.get("inference_time", 0) for p in perf_history
                            ),
                        }
                    else:
                        stats["ml_engine"] = {"predictions_made": 0}
                except Exception as e:
                    stats["ml_engine"] = {"error": str(e)}

            # パフォーマンス監視統計
            if self.performance_monitor:
                try:
                    monitor_stats = self.performance_monitor.get_performance_summary()
                    stats["performance_monitor"] = monitor_stats
                except Exception as e:
                    stats["performance_monitor"] = {"error": str(e)}

            return stats

        except Exception as e:
            return {
                "error": str(e),
                "execution_time_seconds": 0,
                "timestamp": datetime.now().isoformat(),
            }

    def generate_portfolio_analysis(
        self, ai_results: List[AIAnalysisResult]
    ) -> Dict[str, Any]:
        """
        ポートフォリオ分析生成
        
        Args:
            ai_results: AI分析結果リスト
            
        Returns:
            Dict[str, Any]: ポートフォリオ分析結果
        """
        if not ai_results:
            return {
                "status": "analysis_only",
                "trading_disabled": True,
                "analyzed_symbols": 0,
                "timestamp": datetime.now().isoformat(),
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

            # パフォーマンス統計
            analysis_times = [
                r.performance_metrics.get("analysis_time", 0) for r in ai_results
            ]
            
            # 予測統計
            predictions = [
                r.predictions.get("predicted_change", 0) 
                for r in ai_results 
                if "predicted_change" in r.predictions
            ]

            portfolio_analysis = {
                "status": "analysis_only",
                "trading_disabled": True,
                "timestamp": datetime.now().isoformat(),
                "analyzed_symbols": total_symbols,
                "high_confidence_predictions": high_confidence_count,
                "recommendation_distribution": recommendation_counts,
                "average_data_quality": avg_data_quality,
                "risk_distribution": risk_distribution,
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
                    "avg_analysis_time": np.mean(analysis_times) if analysis_times else 0,
                    "total_analysis_time": sum(analysis_times) if analysis_times else 0,
                },
                "prediction_summary": {
                    "predictions_count": len(predictions),
                    "avg_predicted_change": np.mean(predictions) if predictions else 0,
                    "max_predicted_change": max(predictions) if predictions else 0,
                    "min_predicted_change": min(predictions) if predictions else 0,
                    "positive_predictions": sum(1 for p in predictions if p > 0),
                    "negative_predictions": sum(1 for p in predictions if p < 0),
                },
                "quality_metrics": {
                    "high_quality_analyses": sum(
                        1 for r in ai_results 
                        if r.data_quality >= self.config.data_quality_threshold
                    ),
                    "low_risk_analyses": sum(
                        1 for r in ai_results
                        if r.risk_assessment.get("overall_risk_score", 1.0) < 0.4
                    ),
                    "actionable_signals": sum(
                        1 for r in ai_results
                        if r.recommendation in ["STRONG_BUY_SIGNAL", "STRONG_SELL_SIGNAL"]
                    ),
                }
            }

            return portfolio_analysis

        except Exception as e:
            logger.error(f"ポートフォリオ分析エラー: {e}")
            return {
                "status": "analysis_only",
                "trading_disabled": True,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _get_resource_usage(self) -> Dict[str, Any]:
        """
        リソース使用状況取得
        
        Returns:
            Dict[str, Any]: リソース使用情報
        """
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "memory_used_mb": memory.used / (1024 * 1024),
                "timestamp": datetime.now().isoformat(),
            }
            
        except ImportError:
            return {
                "status": "unavailable",
                "reason": "psutil not available",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _calculate_health_score(self, components: Dict[str, Any]) -> float:
        """
        ヘルススコア計算
        
        Args:
            components: コンポーネント情報
            
        Returns:
            float: ヘルススコア (0.0-1.0)
        """
        try:
            scores = []
            
            for comp_name, comp_info in components.items():
                status = comp_info.get("status", "unknown")
                
                if status == "operational":
                    scores.append(1.0)
                elif status == "degraded":
                    scores.append(0.5)
                elif status == "disabled":
                    scores.append(0.7)  # 無効化は意図的なので中程度
                elif status == "error":
                    scores.append(0.0)
                else:
                    scores.append(0.3)  # 不明な状態
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"ヘルススコア計算エラー: {e}")
            return 0.0

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        監視サマリー取得
        
        Returns:
            Dict[str, Any]: 監視サマリー情報
        """
        return {
            "components": {
                "ml_engine": self.ml_engine is not None,
                "batch_fetcher": self.batch_fetcher is not None,
                "performance_monitor": self.performance_monitor is not None,
            },
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "data_quality_threshold": self.config.data_quality_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "retry_attempts": self.config.retry_attempts,
            },
            "ci_mode": CI_MODE,
            "timestamp": datetime.now().isoformat(),
        }

    def cleanup(self) -> Dict[str, Any]:
        """
        監視システムのクリーンアップ
        
        Returns:
            Dict[str, Any]: クリーンアップ結果サマリー
        """
        cleanup_summary = {
            "performance_monitor": False,
            "errors": []
        }

        try:
            # パフォーマンスモニタークリーンアップ
            if hasattr(self, 'performance_monitor') and self.performance_monitor:
                try:
                    if hasattr(self.performance_monitor, "stop"):
                        self.performance_monitor.stop()
                    if hasattr(self.performance_monitor, "close"):
                        self.performance_monitor.close()
                    self.performance_monitor = None
                    cleanup_summary["performance_monitor"] = True
                    logger.debug("パフォーマンスモニター クリーンアップ完了")
                except Exception as e:
                    error_msg = f"パフォーマンスモニター クリーンアップエラー: {e}"
                    logger.warning(error_msg)
                    cleanup_summary["errors"].append(error_msg)

            # 参照をクリア（実際のオブジェクトは他で管理）
            self.ml_engine = None
            self.batch_fetcher = None

        except Exception as e:
            error_msg = f"SystemMonitor クリーンアップ致命的エラー: {e}"
            logger.error(error_msg)
            cleanup_summary["errors"].append(error_msg)

        return cleanup_summary