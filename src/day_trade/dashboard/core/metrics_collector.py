"""
システムメトリクス収集器

システム全体のパフォーマンス、健全性、使用状況メトリクスを収集
"""

import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import psutil

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class MetricsCollector:
    """システムメトリクス収集クラス"""

    def __init__(self, collection_interval: float = 1.0):
        """
        メトリクス収集器の初期化

        Args:
            collection_interval: メトリクス収集間隔（秒）
        """
        self.collection_interval = collection_interval
        self.metrics_buffer: List[Dict] = []
        self.is_collecting = False
        self._collection_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

        # システム情報キャッシュ
        self.system_info = self._collect_static_system_info()

    def _collect_static_system_info(self) -> Dict:
        """静的システム情報収集"""
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "platform": psutil.WINDOWS if psutil.WINDOWS else "unix",
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            }
        except Exception as e:
            logger.error(f"静的システム情報取得エラー: {e}")
            return {}

    async def start_collection(self) -> None:
        """メトリクス収集開始"""
        if self.is_collecting:
            logger.warning("メトリクス収集は既に開始されています")
            return

        self.is_collecting = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info(
            f"メトリクス収集を開始しました (間隔: {self.collection_interval}秒)"
        )

    async def stop_collection(self) -> None:
        """メトリクス収集停止"""
        if not self.is_collecting:
            return

        self.is_collecting = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        logger.info("メトリクス収集を停止しました")

    async def _collection_loop(self) -> None:
        """メトリクス収集メインループ"""
        while self.is_collecting:
            try:
                metrics = self._collect_current_metrics()

                with self._lock:
                    self.metrics_buffer.append(metrics)
                    # バッファサイズ制限（直近3600秒分のみ保持）
                    max_size = int(3600 / self.collection_interval)
                    if len(self.metrics_buffer) > max_size:
                        self.metrics_buffer = self.metrics_buffer[-max_size:]

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"メトリクス収集でエラーが発生: {e}")
                await asyncio.sleep(self.collection_interval)

    def _collect_current_metrics(self) -> Dict:
        """現在のシステムメトリクス収集"""
        timestamp = datetime.now()

        try:
            # CPU メトリクス
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_freq = psutil.cpu_freq()
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)

            # メモリメトリクス
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # ディスクメトリクス
            disk_usage = psutil.disk_usage(".")
            disk_io = psutil.disk_io_counters()

            # ネットワークメトリクス
            network_io = psutil.net_io_counters()

            # プロセス情報
            process_count = len(psutil.pids())

            # 現在のプロセス情報
            current_process = psutil.Process()
            process_memory = current_process.memory_info()
            process_cpu = current_process.cpu_percent()

            return {
                "timestamp": timestamp.isoformat(),
                "epoch": timestamp.timestamp(),
                # CPU メトリクス
                "cpu": {
                    "usage_percent": cpu_percent,
                    "frequency_mhz": cpu_freq.current if cpu_freq else 0,
                    "cores_usage": cpu_per_core,
                    "load_average": (
                        psutil.getloadavg()
                        if hasattr(psutil, "getloadavg")
                        else [0, 0, 0]
                    ),
                },
                # メモリメトリクス
                "memory": {
                    "total_bytes": memory.total,
                    "available_bytes": memory.available,
                    "used_bytes": memory.used,
                    "usage_percent": memory.percent,
                    "cached_bytes": getattr(memory, "cached", 0),
                    "buffers_bytes": getattr(memory, "buffers", 0),
                },
                # スワップメトリクス
                "swap": {
                    "total_bytes": swap.total,
                    "used_bytes": swap.used,
                    "usage_percent": swap.percent,
                },
                # ディスクメトリクス
                "disk": {
                    "total_bytes": disk_usage.total,
                    "used_bytes": disk_usage.used,
                    "free_bytes": disk_usage.free,
                    "usage_percent": (disk_usage.used / disk_usage.total) * 100,
                    "io_read_bytes": disk_io.read_bytes if disk_io else 0,
                    "io_write_bytes": disk_io.write_bytes if disk_io else 0,
                    "io_read_count": disk_io.read_count if disk_io else 0,
                    "io_write_count": disk_io.write_count if disk_io else 0,
                },
                # ネットワークメトリクス
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv,
                    "errors_in": network_io.errin,
                    "errors_out": network_io.errout,
                },
                # プロセスメトリクス
                "processes": {
                    "total_count": process_count,
                    "current_memory_rss": process_memory.rss,
                    "current_memory_vms": process_memory.vms,
                    "current_cpu_percent": process_cpu,
                },
            }

        except Exception as e:
            logger.error(f"メトリクス収集エラー: {e}")
            return {
                "timestamp": timestamp.isoformat(),
                "epoch": timestamp.timestamp(),
                "error": str(e),
            }

    def get_current_metrics(self) -> Dict:
        """現在のメトリクス取得"""
        with self._lock:
            if self.metrics_buffer:
                return self.metrics_buffer[-1]
            else:
                return self._collect_current_metrics()

    def get_metrics_history(self, minutes: int = 10) -> List[Dict]:
        """指定分間のメトリクス履歴取得"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        with self._lock:
            filtered_metrics = []
            for metric in self.metrics_buffer:
                if "error" in metric:
                    continue

                metric_time = datetime.fromisoformat(metric["timestamp"])
                if metric_time >= cutoff_time:
                    filtered_metrics.append(metric)

            return filtered_metrics

    def get_aggregated_metrics(self, minutes: int = 5) -> Dict:
        """集約されたメトリクス取得"""
        history = self.get_metrics_history(minutes)

        if not history:
            return {}

        # 各メトリクスの平均、最大、最小値計算
        cpu_usage = [m["cpu"]["usage_percent"] for m in history if "cpu" in m]
        memory_usage = [m["memory"]["usage_percent"] for m in history if "memory" in m]
        disk_usage = [m["disk"]["usage_percent"] for m in history if "disk" in m]

        return {
            "period_minutes": minutes,
            "samples_count": len(history),
            "timestamp_range": {
                "start": history[0]["timestamp"] if history else None,
                "end": history[-1]["timestamp"] if history else None,
            },
            "cpu": {
                "avg_usage_percent": (
                    round(sum(cpu_usage) / len(cpu_usage), 2) if cpu_usage else 0
                ),
                "max_usage_percent": round(max(cpu_usage), 2) if cpu_usage else 0,
                "min_usage_percent": round(min(cpu_usage), 2) if cpu_usage else 0,
            },
            "memory": {
                "avg_usage_percent": (
                    round(sum(memory_usage) / len(memory_usage), 2)
                    if memory_usage
                    else 0
                ),
                "max_usage_percent": round(max(memory_usage), 2) if memory_usage else 0,
                "min_usage_percent": round(min(memory_usage), 2) if memory_usage else 0,
                "avg_usage_gb": (
                    round(
                        (sum(m["memory"]["used_bytes"] for m in history) / len(history))
                        / (1024**3),
                        2,
                    )
                    if history
                    else 0
                ),
            },
            "disk": {
                "avg_usage_percent": (
                    round(sum(disk_usage) / len(disk_usage), 2) if disk_usage else 0
                ),
                "max_usage_percent": round(max(disk_usage), 2) if disk_usage else 0,
                "min_usage_percent": round(min(disk_usage), 2) if disk_usage else 0,
            },
            "performance_indicators": self._calculate_performance_indicators(history),
        }

    def _calculate_performance_indicators(self, history: List[Dict]) -> Dict:
        """パフォーマンス指標計算"""
        if not history:
            return {}

        indicators = {}

        # CPU負荷レベル判定
        cpu_values = [m["cpu"]["usage_percent"] for m in history if "cpu" in m]
        if cpu_values:
            avg_cpu = sum(cpu_values) / len(cpu_values)
            if avg_cpu < 30:
                cpu_level = "low"
            elif avg_cpu < 70:
                cpu_level = "normal"
            elif avg_cpu < 90:
                cpu_level = "high"
            else:
                cpu_level = "critical"

            indicators["cpu_load_level"] = cpu_level

        # メモリ負荷レベル判定
        memory_values = [m["memory"]["usage_percent"] for m in history if "memory" in m]
        if memory_values:
            avg_memory = sum(memory_values) / len(memory_values)
            if avg_memory < 50:
                memory_level = "low"
            elif avg_memory < 80:
                memory_level = "normal"
            elif avg_memory < 95:
                memory_level = "high"
            else:
                memory_level = "critical"

            indicators["memory_load_level"] = memory_level

        # システム健全性スコア (0-100)
        health_score = 100
        if cpu_values:
            avg_cpu = sum(cpu_values) / len(cpu_values)
            if avg_cpu > 90:
                health_score -= 30
            elif avg_cpu > 70:
                health_score -= 15

        if memory_values:
            avg_memory = sum(memory_values) / len(memory_values)
            if avg_memory > 95:
                health_score -= 25
            elif avg_memory > 80:
                health_score -= 10

        indicators["system_health_score"] = max(0, health_score)

        # パフォーマンス安定性 (変動係数)
        if cpu_values and len(cpu_values) > 1:
            cpu_std = (
                sum((x - sum(cpu_values) / len(cpu_values)) ** 2 for x in cpu_values)
                / len(cpu_values)
            ) ** 0.5
            cpu_cv = (
                cpu_std / (sum(cpu_values) / len(cpu_values))
                if sum(cpu_values) > 0
                else 0
            )
            indicators["cpu_stability"] = "stable" if cpu_cv < 0.3 else "unstable"

        return indicators

    def get_system_info(self) -> Dict:
        """システム情報取得"""
        current = self.get_current_metrics()
        return {
            **self.system_info,
            "current_status": {
                "collection_active": self.is_collecting,
                "metrics_buffer_size": len(self.metrics_buffer),
                "last_collection": current.get("timestamp", "unknown"),
            },
        }

    def generate_health_report(self) -> Dict:
        """システム健全性レポート生成"""
        current = self.get_current_metrics()
        aggregated = self.get_aggregated_metrics(10)  # 直近10分

        # アラートチェック
        alerts = []
        if current.get("cpu", {}).get("usage_percent", 0) > 90:
            alerts.append(
                {
                    "level": "critical",
                    "message": "CPU使用率が90%を超えています",
                    "value": current["cpu"]["usage_percent"],
                }
            )

        if current.get("memory", {}).get("usage_percent", 0) > 95:
            alerts.append(
                {
                    "level": "critical",
                    "message": "メモリ使用率が95%を超えています",
                    "value": current["memory"]["usage_percent"],
                }
            )

        if current.get("disk", {}).get("usage_percent", 0) > 90:
            alerts.append(
                {
                    "level": "warning",
                    "message": "ディスク使用率が90%を超えています",
                    "value": current["disk"]["usage_percent"],
                }
            )

        # 推奨アクション
        recommendations = []
        if aggregated.get("cpu", {}).get("avg_usage_percent", 0) > 80:
            recommendations.append(
                "CPU負荷が高くなっています。不要なプロセスの停止を検討してください"
            )

        if aggregated.get("memory", {}).get("avg_usage_percent", 0) > 85:
            recommendations.append(
                "メモリ使用量が多くなっています。メモリ使用量の最適化を検討してください"
            )

        return {
            "report_timestamp": datetime.now().isoformat(),
            "overall_health": aggregated.get("performance_indicators", {}).get(
                "system_health_score", 100
            ),
            "current_metrics": current,
            "aggregated_metrics": aggregated,
            "alerts": alerts,
            "recommendations": recommendations,
            "system_info": self.system_info,
        }
