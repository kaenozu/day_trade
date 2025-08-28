#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Optimization System
統合最適化システム - 第4世代完全統合最適化技術
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path

# 最適化モジュール
from .prediction_accuracy_enhancer import PredictionAccuracyEnhancer
from .performance_optimization_engine import PerformanceOptimizationEngine
from .model_accuracy_improver import ModelAccuracyImprover
from .response_speed_optimizer import ResponseSpeedOptimizer
from .memory_efficiency_optimizer import MemoryEfficiencyOptimizer

logger = logging.getLogger(__name__)


@dataclass
class OptimizationReport:
    """最適化レポート"""
    timestamp: datetime
    prediction_accuracy: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    model_improvements: Dict[str, Any] = field(default_factory=dict)
    response_speed: Dict[str, Any] = field(default_factory=dict)
    memory_efficiency: Dict[str, Any] = field(default_factory=dict)
    overall_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SystemHealth:
    """システムヘルス"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    prediction_accuracy: float
    model_performance: float
    response_time: float
    error_rate: float
    overall_health: str  # excellent, good, fair, poor


class IntegratedOptimizationSystem:
    """統合最適化システム"""
    
    def __init__(self, config_path: Optional[str] = None):
        # サブシステム初期化
        self.prediction_enhancer = PredictionAccuracyEnhancer()
        self.performance_engine = PerformanceOptimizationEngine()
        self.model_improver = ModelAccuracyImprover()
        self.speed_optimizer = ResponseSpeedOptimizer()
        self.memory_optimizer = MemoryEfficiencyOptimizer()
        
        # システム状態
        self.is_running = False
        self.optimization_history: List[OptimizationReport] = []
        self.health_history: List[SystemHealth] = []
        
        # 設定
        self.auto_optimization_interval = 300  # 5分
        self.health_check_interval = 60       # 1分
        
        # スケジューラ
        self.optimization_scheduler = None
        self.health_monitor = None
        
    async def initialize(self):
        """システム初期化"""
        logger.info("Initializing Integrated Optimization System...")
        
        # サブシステム初期化
        await self.prediction_enhancer.initialize()
        await self.performance_engine.initialize()
        await self.model_improver.initialize()
        await self.speed_optimizer.initialize()
        
        logger.info("All optimization subsystems initialized successfully")
        
    async def start_optimization_service(self):
        """最適化サービス開始"""
        if self.is_running:
            logger.warning("Optimization service is already running")
            return
            
        self.is_running = True
        
        # 自動最適化タスク開始
        self.optimization_scheduler = asyncio.create_task(
            self._auto_optimization_loop()
        )
        
        # ヘルスモニタリングタスク開始
        self.health_monitor = asyncio.create_task(
            self._health_monitoring_loop()
        )
        
        logger.info("Integrated Optimization Service started")
        
    async def stop_optimization_service(self):
        """最適化サービス停止"""
        self.is_running = False
        
        if self.optimization_scheduler:
            self.optimization_scheduler.cancel()
            try:
                await self.optimization_scheduler
            except asyncio.CancelledError:
                pass
                
        if self.health_monitor:
            self.health_monitor.cancel()
            try:
                await self.health_monitor
            except asyncio.CancelledError:
                pass
                
        logger.info("Integrated Optimization Service stopped")
        
    async def run_comprehensive_optimization(self) -> OptimizationReport:
        """包括的最適化実行"""
        start_time = time.time()
        logger.info("Starting comprehensive optimization...")
        
        report = OptimizationReport(timestamp=datetime.now())
        
        try:
            # 1. 予測精度向上
            logger.info("Running prediction accuracy enhancement...")
            prediction_results = await self._optimize_prediction_accuracy()
            report.prediction_accuracy = prediction_results
            
            # 2. パフォーマンス最適化
            logger.info("Running performance optimization...")
            performance_results = await self._optimize_performance()
            report.performance_metrics = performance_results
            
            # 3. モデル精度改善
            logger.info("Running model accuracy improvement...")
            model_results = await self._improve_model_accuracy()
            report.model_improvements = model_results
            
            # 4. レスポンス速度最適化
            logger.info("Running response speed optimization...")
            speed_results = await self._optimize_response_speed()
            report.response_speed = speed_results
            
            # 5. メモリ効率化
            logger.info("Running memory efficiency optimization...")
            memory_results = await self._optimize_memory_efficiency()
            report.memory_efficiency = memory_results
            
            # 総合スコア計算
            report.overall_score = self._calculate_overall_score(report)
            
            # 推奨事項生成
            report.recommendations = self._generate_recommendations(report)
            
            # レポート保存
            self.optimization_history.append(report)
            
            execution_time = time.time() - start_time
            logger.info(f"Comprehensive optimization completed in {execution_time:.2f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Comprehensive optimization failed: {e}")
            report.recommendations.append(f"Optimization failed: {str(e)}")
            return report
            
    async def _optimize_prediction_accuracy(self) -> Dict[str, Any]:
        """予測精度最適化"""
        try:
            # フィーチャーエンジニアリング最適化
            feature_results = await self.prediction_enhancer.optimize_feature_engineering()
            
            # モデル選択最適化
            model_results = await self.prediction_enhancer.optimize_model_selection()
            
            # アンサンブル最適化
            ensemble_results = await self.prediction_enhancer.optimize_ensemble_methods()
            
            return {
                "feature_engineering": feature_results,
                "model_selection": model_results,
                "ensemble_methods": ensemble_results,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Prediction accuracy optimization failed: {e}")
            return {"status": "failed", "error": str(e)}
            
    async def _optimize_performance(self) -> Dict[str, Any]:
        """パフォーマンス最適化"""
        try:
            # CPU最適化
            cpu_results = await self.performance_engine.optimize_cpu_performance()
            
            # メモリ最適化
            memory_results = await self.performance_engine.optimize_memory_usage()
            
            # I/O最適化
            io_results = await self.performance_engine.optimize_io_performance()
            
            return {
                "cpu_optimization": cpu_results,
                "memory_optimization": memory_results,
                "io_optimization": io_results,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            return {"status": "failed", "error": str(e)}
            
    async def _improve_model_accuracy(self) -> Dict[str, Any]:
        """モデル精度改善"""
        try:
            # ハイパーパラメータ最適化
            hyperopt_results = await self.model_improver.optimize_hyperparameters()
            
            # アンサンブル最適化
            ensemble_results = await self.model_improver.optimize_ensemble()
            
            # 特徴量選択最適化
            feature_results = await self.model_improver.optimize_feature_selection()
            
            return {
                "hyperparameter_optimization": hyperopt_results,
                "ensemble_optimization": ensemble_results,
                "feature_selection": feature_results,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Model accuracy improvement failed: {e}")
            return {"status": "failed", "error": str(e)}
            
    async def _optimize_response_speed(self) -> Dict[str, Any]:
        """レスポンス速度最適化"""
        try:
            # パフォーマンスメトリクス取得
            metrics = await self.speed_optimizer.get_performance_metrics()
            
            # キャッシュ最適化
            cache_results = {"cache_hit_rate": metrics.get("cache_hit_rate", 0.0)}
            
            return {
                "performance_metrics": metrics,
                "cache_optimization": cache_results,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Response speed optimization failed: {e}")
            return {"status": "failed", "error": str(e)}
            
    async def _optimize_memory_efficiency(self) -> Dict[str, Any]:
        """メモリ効率化最適化"""
        try:
            # メモリ使用量分析
            analysis = self.memory_optimizer.analyze_memory_usage()
            
            # メモリリーク検出
            leaks = self.memory_optimizer.detect_memory_leaks()
            
            # 最適化実行
            optimization = self.memory_optimizer.optimize_memory_usage()
            
            return {
                "memory_analysis": analysis,
                "memory_leaks": leaks,
                "optimization_results": optimization,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Memory efficiency optimization failed: {e}")
            return {"status": "failed", "error": str(e)}
            
    def _calculate_overall_score(self, report: OptimizationReport) -> float:
        """総合スコア計算"""
        scores = []
        weights = []
        
        # 予測精度スコア
        if report.prediction_accuracy.get("status") == "success":
            scores.append(85.0)  # ベーススコア
            weights.append(0.3)
            
        # パフォーマンススコア
        if report.performance_metrics.get("status") == "success":
            scores.append(80.0)
            weights.append(0.2)
            
        # モデル改善スコア
        if report.model_improvements.get("status") == "success":
            scores.append(90.0)
            weights.append(0.25)
            
        # レスポンス速度スコア
        if report.response_speed.get("status") == "success":
            cache_hit_rate = report.response_speed.get("performance_metrics", {}).get("cache_hit_rate", 0.0)
            score = 70.0 + (cache_hit_rate * 30.0)
            scores.append(score)
            weights.append(0.15)
            
        # メモリ効率スコア
        if report.memory_efficiency.get("status") == "success":
            has_leaks = len(report.memory_efficiency.get("memory_leaks", [])) > 0
            score = 60.0 if has_leaks else 85.0
            scores.append(score)
            weights.append(0.1)
            
        if not scores:
            return 0.0
            
        # 重み付き平均
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
            
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return round(weighted_score, 2)
        
    def _generate_recommendations(self, report: OptimizationReport) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        # 予測精度関連
        if report.prediction_accuracy.get("status") != "success":
            recommendations.append("予測精度の向上が必要です。特徴量エンジニアリングとモデル選択を見直してください。")
            
        # パフォーマンス関連
        if report.performance_metrics.get("status") != "success":
            recommendations.append("システムパフォーマンスの最適化が必要です。CPU・メモリ・I/O使用量を確認してください。")
            
        # モデル精度関連
        if report.model_improvements.get("status") != "success":
            recommendations.append("モデル精度の改善が必要です。ハイパーパラメータとアンサンブル手法を最適化してください。")
            
        # レスポンス速度関連
        cache_hit_rate = report.response_speed.get("performance_metrics", {}).get("cache_hit_rate", 0.0)
        if cache_hit_rate < 0.5:
            recommendations.append("キャッシュヒット率が低いです。キャッシュ戦略を見直してください。")
            
        # メモリ効率関連
        memory_leaks = report.memory_efficiency.get("memory_leaks", [])
        if memory_leaks:
            recommendations.append("メモリリークが検出されました。メモリ管理を見直してください。")
            
        # 総合スコア関連
        if report.overall_score < 70.0:
            recommendations.append("システム全体の最適化が必要です。包括的な見直しを行ってください。")
        elif report.overall_score < 85.0:
            recommendations.append("システムは良好ですが、さらなる改善の余地があります。")
        else:
            recommendations.append("システムは最適な状態です。定期的な監視を続けてください。")
            
        return recommendations
        
    async def _auto_optimization_loop(self):
        """自動最適化ループ"""
        while self.is_running:
            try:
                await asyncio.sleep(self.auto_optimization_interval)
                
                if self.is_running:
                    logger.info("Running scheduled optimization...")
                    await self.run_comprehensive_optimization()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto optimization error: {e}")
                
    async def _health_monitoring_loop(self):
        """ヘルスモニタリングループ"""
        while self.is_running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if self.is_running:
                    health = await self.check_system_health()
                    self.health_history.append(health)
                    
                    # 古いヘルス記録を削除（24時間分保持）
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    self.health_history = [
                        h for h in self.health_history 
                        if h.timestamp > cutoff_time
                    ]
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                
    async def check_system_health(self) -> SystemHealth:
        """システムヘルスチェック"""
        import psutil
        
        # システムリソース取得
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # パフォーマンスメトリクス取得
        speed_metrics = await self.speed_optimizer.get_performance_metrics()
        memory_metrics = self.memory_optimizer.get_memory_metrics()
        
        # ヘルススコア計算
        response_time = speed_metrics.get("avg_response_time", 1.0)
        error_rate = 0.0  # TODO: 実際のエラーレート取得
        
        # 総合ヘルス判定
        health_score = 100.0
        health_score -= min(cpu_usage, 50)  # CPU使用率による減点
        health_score -= min(memory.percent, 50)  # メモリ使用率による減点
        health_score -= min(response_time * 10, 30)  # レスポンス時間による減点
        health_score -= error_rate * 100  # エラー率による減点
        
        if health_score >= 90:
            overall_health = "excellent"
        elif health_score >= 75:
            overall_health = "good"
        elif health_score >= 60:
            overall_health = "fair"
        else:
            overall_health = "poor"
            
        return SystemHealth(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            network_latency=0.0,  # TODO: 実際のネットワーク遅延測定
            prediction_accuracy=85.0,  # TODO: 実際の予測精度取得
            model_performance=80.0,   # TODO: 実際のモデル性能取得
            response_time=response_time,
            error_rate=error_rate,
            overall_health=overall_health
        )
        
    def get_optimization_history(self, limit: int = 10) -> List[OptimizationReport]:
        """最適化履歴取得"""
        return self.optimization_history[-limit:]
        
    def get_health_history(self, hours: int = 24) -> List[SystemHealth]:
        """ヘルス履歴取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [h for h in self.health_history if h.timestamp > cutoff_time]
        
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        latest_health = self.health_history[-1] if self.health_history else None
        latest_report = self.optimization_history[-1] if self.optimization_history else None
        
        return {
            "service_running": self.is_running,
            "latest_health": latest_health.__dict__ if latest_health else None,
            "latest_optimization": {
                "timestamp": latest_report.timestamp.isoformat() if latest_report else None,
                "overall_score": latest_report.overall_score if latest_report else None,
                "recommendations": latest_report.recommendations if latest_report else []
            } if latest_report else None,
            "subsystems": {
                "prediction_enhancer": "active",
                "performance_engine": "active", 
                "model_improver": "active",
                "speed_optimizer": "active",
                "memory_optimizer": "active"
            }
        }


# グローバルインスタンス
integrated_optimizer = IntegratedOptimizationSystem()


async def initialize_optimization_system():
    """統合最適化システム初期化"""
    await integrated_optimizer.initialize()


async def start_optimization_service():
    """最適化サービス開始"""
    await integrated_optimizer.start_optimization_service()


async def stop_optimization_service():
    """最適化サービス停止"""
    await integrated_optimizer.stop_optimization_service()


async def run_optimization():
    """最適化実行"""
    return await integrated_optimizer.run_comprehensive_optimization()