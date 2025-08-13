#!/usr/bin/env python3
"""
パフォーマンス最適化・SLA監視システム
リアルタイムパフォーマンス監視と自動最適化

Features:
- リアルタイムパフォーマンス監視
- SLA(Service Level Agreement)監視
- 自動パフォーマンス最適化
- リソース使用量最適化
- ボトルネック検知と対策
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import psutil

from ..utils.logging_config import get_context_logger
from .metrics.prometheus_metrics import get_metrics_collector

logger = get_context_logger(__name__)


class SLAStatus(Enum):
    """SLA状態"""

    HEALTHY = "healthy"
    WARNING = "warning"
    BREACH = "breach"
    CRITICAL = "critical"


class OptimizationAction(Enum):
    """最適化アクション"""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CACHE_CLEAR = "cache_clear"
    RESTART_SERVICE = "restart_service"
    ADJUST_BATCH_SIZE = "adjust_batch_size"
    OPTIMIZE_QUERY = "optimize_query"


@dataclass
class SLATarget:
    """SLA目標"""

    service_name: str
    availability_target: float  # %
    response_time_p95: float  # ms
    response_time_p99: float  # ms
    error_rate_threshold: float  # %
    throughput_minimum: float  # req/s


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""

    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    throughput: float
    error_rate: float
    active_connections: int


@dataclass
class SLAViolation:
    """SLA違反"""

    service_name: str
    violation_type: str
    threshold: float
    current_value: float
    severity: SLAStatus
    timestamp: datetime
    duration: timedelta
    description: str


@dataclass
class OptimizationRecommendation:
    """最適化推奨"""

    action: OptimizationAction
    target_component: str
    description: str
    expected_impact: str
    confidence: float
    priority: int


class PerformanceMonitor:
    """パフォーマンス監視器"""

    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.baseline_metrics = None
        self.monitoring_active = False
        self.collection_interval = 5  # 秒

    async def start_monitoring(self):
        """監視開始"""

        if self.monitoring_active:
            return

        self.monitoring_active = True
        logger.info("パフォーマンス監視開始")

        while self.monitoring_active:
            try:
                metrics = await self._collect_performance_metrics()
                self.metrics_history.append(metrics)

                # ベースライン設定（初回100サンプル後）
                if len(self.metrics_history) == 100 and self.baseline_metrics is None:
                    self._calculate_baseline()

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"パフォーマンス監視エラー: {e}")
                await asyncio.sleep(1)

    def stop_monitoring(self):
        """監視停止"""

        self.monitoring_active = False
        logger.info("パフォーマンス監視停止")

    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """パフォーマンスメトリクス収集"""

        # システムメトリクス収集
        cpu_usage = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        network = psutil.net_io_counters()

        # アプリケーションメトリクス（Prometheusから取得）
        metrics_collector = get_metrics_collector()

        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=(disk.used / disk.total) * 100 if disk.total > 0 else 0,
            network_io=network.bytes_sent + network.bytes_recv if network else 0,
            response_time_avg=0.0,  # 実際の実装では計算
            response_time_p95=0.0,  # 実際の実装では計算
            response_time_p99=0.0,  # 実際の実装では計算
            throughput=0.0,  # 実際の実装では計算
            error_rate=0.0,  # 実際の実装では計算
            active_connections=len(psutil.net_connections()),
        )

    def _calculate_baseline(self):
        """ベースラインメトリクス計算"""

        if len(self.metrics_history) < 50:
            return

        recent_metrics = list(self.metrics_history)[-100:]

        self.baseline_metrics = {
            "cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            "memory_usage": sum(m.memory_usage for m in recent_metrics)
            / len(recent_metrics),
            "disk_usage": sum(m.disk_usage for m in recent_metrics)
            / len(recent_metrics),
            "response_time_avg": sum(m.response_time_avg for m in recent_metrics)
            / len(recent_metrics),
            "throughput": sum(m.throughput for m in recent_metrics)
            / len(recent_metrics),
            "error_rate": sum(m.error_rate for m in recent_metrics)
            / len(recent_metrics),
        }

        logger.info("ベースラインメトリクス計算完了")

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """現在のメトリクス取得"""

        if not self.metrics_history:
            return None

        return self.metrics_history[-1]

    def get_metrics_history(
        self, duration_minutes: int = 60
    ) -> List[PerformanceMetrics]:
        """メトリクス履歴取得"""

        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)

        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]


class SLAMonitor:
    """SLA監視器"""

    def __init__(self):
        self.sla_targets = {}
        self.violations = deque(maxlen=1000)
        self.violation_history = defaultdict(list)
        self.current_violations = {}

    def add_sla_target(self, target: SLATarget):
        """SLA目標追加"""

        self.sla_targets[target.service_name] = target
        logger.info(f"SLA目標設定: {target.service_name}")

    def check_sla_compliance(
        self, service_name: str, metrics: PerformanceMetrics
    ) -> List[SLAViolation]:
        """SLA遵守チェック"""

        if service_name not in self.sla_targets:
            return []

        target = self.sla_targets[service_name]
        violations = []
        current_time = datetime.now()

        # レスポンス時間チェック
        if metrics.response_time_p95 > target.response_time_p95:
            violations.append(
                SLAViolation(
                    service_name=service_name,
                    violation_type="response_time_p95",
                    threshold=target.response_time_p95,
                    current_value=metrics.response_time_p95,
                    severity=(
                        SLAStatus.BREACH
                        if metrics.response_time_p95 > target.response_time_p95 * 1.5
                        else SLAStatus.WARNING
                    ),
                    timestamp=current_time,
                    duration=timedelta(seconds=0),  # 実際の実装では継続時間を計算
                    description=f"P95レスポンス時間が目標値 {target.response_time_p95}ms を超過: {metrics.response_time_p95:.2f}ms",
                )
            )

        if metrics.response_time_p99 > target.response_time_p99:
            violations.append(
                SLAViolation(
                    service_name=service_name,
                    violation_type="response_time_p99",
                    threshold=target.response_time_p99,
                    current_value=metrics.response_time_p99,
                    severity=(
                        SLAStatus.CRITICAL
                        if metrics.response_time_p99 > target.response_time_p99 * 2
                        else SLAStatus.BREACH
                    ),
                    timestamp=current_time,
                    duration=timedelta(seconds=0),
                    description=f"P99レスポンス時間が目標値 {target.response_time_p99}ms を超過: {metrics.response_time_p99:.2f}ms",
                )
            )

        # エラー率チェック
        if metrics.error_rate > target.error_rate_threshold:
            violations.append(
                SLAViolation(
                    service_name=service_name,
                    violation_type="error_rate",
                    threshold=target.error_rate_threshold,
                    current_value=metrics.error_rate,
                    severity=(
                        SLAStatus.CRITICAL
                        if metrics.error_rate > target.error_rate_threshold * 2
                        else SLAStatus.BREACH
                    ),
                    timestamp=current_time,
                    duration=timedelta(seconds=0),
                    description=f"エラー率が目標値 {target.error_rate_threshold}% を超過: {metrics.error_rate:.2f}%",
                )
            )

        # スループットチェック
        if metrics.throughput < target.throughput_minimum:
            violations.append(
                SLAViolation(
                    service_name=service_name,
                    violation_type="throughput",
                    threshold=target.throughput_minimum,
                    current_value=metrics.throughput,
                    severity=(
                        SLAStatus.BREACH
                        if metrics.throughput < target.throughput_minimum * 0.5
                        else SLAStatus.WARNING
                    ),
                    timestamp=current_time,
                    duration=timedelta(seconds=0),
                    description=f"スループットが最低値 {target.throughput_minimum} req/s を下回る: {metrics.throughput:.2f} req/s",
                )
            )

        # 違反記録
        for violation in violations:
            self.violations.append(violation)
            self.violation_history[service_name].append(violation)

            violation_key = f"{service_name}_{violation.violation_type}"
            self.current_violations[violation_key] = violation

        return violations

    def get_sla_status(self, service_name: str) -> Dict[str, Any]:
        """SLA状態取得"""

        if service_name not in self.sla_targets:
            return {"error": "SLA target not found"}

        current_violations = [
            v
            for v in self.current_violations.values()
            if v.service_name == service_name
        ]

        if not current_violations:
            overall_status = SLAStatus.HEALTHY
        else:
            severities = [v.severity for v in current_violations]
            if SLAStatus.CRITICAL in severities:
                overall_status = SLAStatus.CRITICAL
            elif SLAStatus.BREACH in severities:
                overall_status = SLAStatus.BREACH
            else:
                overall_status = SLAStatus.WARNING

        return {
            "service_name": service_name,
            "overall_status": overall_status.value,
            "active_violations": len(current_violations),
            "violations": [asdict(v) for v in current_violations],
        }


class PerformanceOptimizer:
    """パフォーマンス最適化器"""

    def __init__(self):
        self.optimization_history = deque(maxlen=500)
        self.active_optimizations = {}

    def analyze_performance_issues(
        self, metrics: PerformanceMetrics, baseline: Dict[str, float] = None
    ) -> List[OptimizationRecommendation]:
        """パフォーマンス問題分析"""

        recommendations = []

        # CPU使用率分析
        if metrics.cpu_usage > 80:
            recommendations.append(
                OptimizationRecommendation(
                    action=OptimizationAction.SCALE_UP,
                    target_component="cpu",
                    description=f"CPU使用率が高い: {metrics.cpu_usage:.1f}%",
                    expected_impact="レスポンス時間改善",
                    confidence=0.8,
                    priority=1 if metrics.cpu_usage > 95 else 2,
                )
            )

        # メモリ使用率分析
        if metrics.memory_usage > 85:
            recommendations.append(
                OptimizationRecommendation(
                    action=OptimizationAction.CACHE_CLEAR,
                    target_component="memory",
                    description=f"メモリ使用率が高い: {metrics.memory_usage:.1f}%",
                    expected_impact="メモリ解放",
                    confidence=0.7,
                    priority=1 if metrics.memory_usage > 95 else 3,
                )
            )

        # ディスク使用率分析
        if metrics.disk_usage > 90:
            recommendations.append(
                OptimizationRecommendation(
                    action=OptimizationAction.OPTIMIZE_QUERY,
                    target_component="storage",
                    description=f"ディスク使用率が高い: {metrics.disk_usage:.1f}%",
                    expected_impact="ディスク容量確保",
                    confidence=0.6,
                    priority=2,
                )
            )

        # レスポンス時間分析（ベースラインとの比較）
        if baseline and "response_time_avg" in baseline:
            if metrics.response_time_avg > baseline["response_time_avg"] * 1.5:
                recommendations.append(
                    OptimizationRecommendation(
                        action=OptimizationAction.ADJUST_BATCH_SIZE,
                        target_component="application",
                        description=f"レスポンス時間がベースラインの1.5倍: {metrics.response_time_avg:.2f}ms",
                        expected_impact="レスポンス時間短縮",
                        confidence=0.75,
                        priority=1,
                    )
                )

        # エラー率分析
        if metrics.error_rate > 5.0:  # 5%以上
            recommendations.append(
                OptimizationRecommendation(
                    action=OptimizationAction.RESTART_SERVICE,
                    target_component="application",
                    description=f"エラー率が高い: {metrics.error_rate:.2f}%",
                    expected_impact="エラー率低減",
                    confidence=0.6,
                    priority=1 if metrics.error_rate > 10 else 2,
                )
            )

        # 優先度でソート
        recommendations.sort(key=lambda x: x.priority)

        return recommendations

    async def apply_optimization(
        self, recommendation: OptimizationRecommendation
    ) -> bool:
        """最適化適用"""

        optimization_id = f"{recommendation.target_component}_{int(time.time())}"

        try:
            logger.info(f"最適化実行開始: {recommendation.description}")

            success = False

            if recommendation.action == OptimizationAction.SCALE_UP:
                success = await self._scale_up_resources(
                    recommendation.target_component
                )
            elif recommendation.action == OptimizationAction.CACHE_CLEAR:
                success = await self._clear_cache()
            elif recommendation.action == OptimizationAction.RESTART_SERVICE:
                success = await self._restart_service(recommendation.target_component)
            elif recommendation.action == OptimizationAction.ADJUST_BATCH_SIZE:
                success = await self._adjust_batch_size()
            elif recommendation.action == OptimizationAction.OPTIMIZE_QUERY:
                success = await self._optimize_queries()

            # 最適化履歴記録
            self.optimization_history.append(
                {
                    "id": optimization_id,
                    "timestamp": datetime.now().isoformat(),
                    "action": recommendation.action.value,
                    "target": recommendation.target_component,
                    "description": recommendation.description,
                    "success": success,
                }
            )

            if success:
                logger.info(f"最適化完了: {recommendation.description}")
            else:
                logger.warning(f"最適化失敗: {recommendation.description}")

            return success

        except Exception as e:
            logger.error(f"最適化エラー ({recommendation.action.value}): {e}")
            return False

    async def _scale_up_resources(self, component: str) -> bool:
        """リソーススケールアップ"""
        # 実際の実装では、クラウドAPIやKubernetes APIを呼び出し
        logger.info(f"リソーススケールアップ: {component}")
        await asyncio.sleep(1)  # シミュレート
        return True

    async def _clear_cache(self) -> bool:
        """キャッシュクリア"""
        # 実際の実装では、Redisやメモリキャッシュをクリア
        logger.info("キャッシュクリア実行")
        await asyncio.sleep(0.5)  # シミュレート
        return True

    async def _restart_service(self, component: str) -> bool:
        """サービス再起動"""
        # 実際の実装では、アプリケーションサービス再起動
        logger.info(f"サービス再起動: {component}")
        await asyncio.sleep(2)  # シミュレート
        return True

    async def _adjust_batch_size(self) -> bool:
        """バッチサイズ調整"""
        # 実際の実装では、処理バッチサイズ動的調整
        logger.info("バッチサイズ調整")
        await asyncio.sleep(0.1)  # シミュレート
        return True

    async def _optimize_queries(self) -> bool:
        """クエリ最適化"""
        # 実際の実装では、データベースクエリ最適化
        logger.info("クエリ最適化実行")
        await asyncio.sleep(1)  # シミュレート
        return True


class IntegratedPerformanceSystem:
    """統合パフォーマンスシステム"""

    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.sla_monitor = SLAMonitor()
        self.performance_optimizer = PerformanceOptimizer()
        self.running = False

        # デフォルトSLA設定
        self._setup_default_sla_targets()

        logger.info("統合パフォーマンスシステム初期化完了")

    def _setup_default_sla_targets(self):
        """デフォルトSLA目標設定"""

        # トレーディングサービス
        self.sla_monitor.add_sla_target(
            SLATarget(
                service_name="trading",
                availability_target=99.9,
                response_time_p95=100.0,  # 100ms
                response_time_p99=200.0,  # 200ms
                error_rate_threshold=0.1,  # 0.1%
                throughput_minimum=10.0,  # 10 req/s
            )
        )

        # AIエンジン
        self.sla_monitor.add_sla_target(
            SLATarget(
                service_name="ai_engine",
                availability_target=99.5,
                response_time_p95=500.0,  # 500ms
                response_time_p99=1000.0,  # 1秒
                error_rate_threshold=0.5,  # 0.5%
                throughput_minimum=5.0,  # 5 req/s
            )
        )

        # データ取得サービス
        self.sla_monitor.add_sla_target(
            SLATarget(
                service_name="data_fetcher",
                availability_target=99.8,
                response_time_p95=200.0,  # 200ms
                response_time_p99=500.0,  # 500ms
                error_rate_threshold=0.2,  # 0.2%
                throughput_minimum=20.0,  # 20 req/s
            )
        )

    async def start(self):
        """システム開始"""

        if self.running:
            return

        self.running = True
        logger.info("統合パフォーマンスシステム開始")

        # パフォーマンス監視開始
        monitoring_task = asyncio.create_task(
            self.performance_monitor.start_monitoring()
        )

        # 定期SLAチェック＆最適化ループ
        optimization_task = asyncio.create_task(self._optimization_loop())

        await asyncio.gather(monitoring_task, optimization_task)

    async def _optimization_loop(self):
        """最適化ループ"""

        while self.running:
            try:
                current_metrics = self.performance_monitor.get_current_metrics()

                if current_metrics:
                    # 各サービスのSLAチェック
                    for service_name in self.sla_monitor.sla_targets.keys():
                        violations = self.sla_monitor.check_sla_compliance(
                            service_name, current_metrics
                        )

                        if violations:
                            logger.warning(
                                f"SLA違反検出 ({service_name}): {len(violations)}件"
                            )

                    # パフォーマンス問題分析
                    recommendations = (
                        self.performance_optimizer.analyze_performance_issues(
                            current_metrics, self.performance_monitor.baseline_metrics
                        )
                    )

                    # 高優先度の最適化を実行
                    for rec in recommendations[:2]:  # 最大2つまで
                        if rec.priority == 1:  # 最優先のみ自動実行
                            await self.performance_optimizer.apply_optimization(rec)

                await asyncio.sleep(30)  # 30秒間隔で実行

            except Exception as e:
                logger.error(f"最適化ループエラー: {e}")
                await asyncio.sleep(5)

    def stop(self):
        """システム停止"""

        self.running = False
        self.performance_monitor.stop_monitoring()
        logger.info("統合パフォーマンスシステム停止")

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""

        current_metrics = self.performance_monitor.get_current_metrics()

        sla_statuses = {}
        for service_name in self.sla_monitor.sla_targets.keys():
            sla_statuses[service_name] = self.sla_monitor.get_sla_status(service_name)

        return {
            "system_running": self.running,
            "current_metrics": asdict(current_metrics) if current_metrics else None,
            "baseline_metrics": self.performance_monitor.baseline_metrics,
            "sla_statuses": sla_statuses,
            "recent_optimizations": list(
                self.performance_optimizer.optimization_history
            )[-10:],
            "metrics_history_size": len(self.performance_monitor.metrics_history),
        }


# グローバルインスタンス
_performance_system = IntegratedPerformanceSystem()


def get_performance_system() -> IntegratedPerformanceSystem:
    """統合パフォーマンスシステム取得"""
    return _performance_system
