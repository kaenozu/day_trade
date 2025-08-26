#!/usr/bin/env python3
"""
SLAメトリクス管理モジュール
SLA計算、追跡、レポート機能を提供
"""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np

from .database import DatabaseManager
from .models import DataSourceConfig, FreshnessStatus, SLAMetrics


class SLAMetricsManager:
    """SLAメトリクス管理クラス"""

    def __init__(self, database_manager: DatabaseManager):
        self.logger = logging.getLogger(__name__)
        self.database_manager = database_manager

    async def calculate_hourly_sla_metrics(self, data_sources: Dict[str, DataSourceConfig]):
        """時間別SLAメトリクス計算"""
        try:
            current_time = datetime.now(timezone.utc)
            period_start = current_time.replace(
                minute=0, second=0, microsecond=0
            ) - timedelta(hours=1)
            period_end = current_time.replace(minute=0, second=0, microsecond=0)

            for source_id, config in data_sources.items():
                sla_metrics = await self._calculate_sla_for_source(
                    source_id, config, period_start, period_end
                )
                if sla_metrics:
                    await self.database_manager.save_sla_metrics(sla_metrics)

        except Exception as e:
            self.logger.error(f"SLAメトリクス計算エラー: {e}")

    async def _calculate_sla_for_source(
        self,
        source_id: str,
        config: DataSourceConfig,
        period_start: datetime,
        period_end: datetime,
    ) -> SLAMetrics:
        """特定データソースのSLA計算"""
        try:
            with sqlite3.connect(self.database_manager.db_path) as conn:
                # 該当期間の鮮度チェック結果取得
                cursor = conn.execute(
                    """
                    SELECT status, age_seconds FROM freshness_checks
                    WHERE source_id = ? AND timestamp BETWEEN ? AND ?
                """,
                    (source_id, period_start.isoformat(), period_end.isoformat()),
                )

                checks = cursor.fetchall()

                if not checks:
                    return None

                # SLAメトリクス計算
                total_checks = len(checks)
                fresh_checks = sum(
                    1
                    for status, _ in checks
                    if status == FreshnessStatus.FRESH.value
                )

                availability_percent = (fresh_checks / total_checks) * 100
                
                # 平均応答時間（無限大の値を除外）
                valid_ages = [age for _, age in checks if age != float("inf")]
                average_response_time = np.mean(valid_ages) if valid_ages else float("inf")
                
                error_count = sum(
                    1
                    for status, _ in checks
                    if status == FreshnessStatus.EXPIRED.value
                )

                # アップタイム・ダウンタイム計算（1時間 = 3600秒）
                uptime_seconds = (fresh_checks / total_checks) * 3600
                downtime_seconds = 3600 - uptime_seconds

                # SLA違反判定
                sla_target = config.sla_target
                sla_violations = 1 if availability_percent < sla_target else 0

                return SLAMetrics(
                    source_id=source_id,
                    period_start=period_start,
                    period_end=period_end,
                    availability_percent=availability_percent,
                    average_response_time=average_response_time,
                    error_count=error_count,
                    total_requests=total_checks,
                    uptime_seconds=uptime_seconds,
                    downtime_seconds=downtime_seconds,
                    sla_violations=sla_violations,
                )

        except Exception as e:
            self.logger.error(f"SLA計算エラー ({source_id}): {e}")
            return None

    async def get_sla_summary(self, hours: int = 24) -> Dict[str, any]:
        """SLAサマリー取得"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            with sqlite3.connect(self.database_manager.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT source_id, AVG(availability_percent), AVG(average_response_time),
                           SUM(sla_violations) as total_violations, COUNT(*) as periods
                    FROM sla_metrics
                    WHERE period_start >= ?
                    GROUP BY source_id
                """,
                    (start_time.isoformat(),),
                )

                summary_data = []
                total_violations = 0

                for row in cursor.fetchall():
                    source_summary = {
                        "source_id": row[0],
                        "average_availability": round(row[1], 2) if row[1] else None,
                        "average_response_time": round(row[2], 3) if row[2] else None,
                        "sla_violations": row[3] or 0,
                        "monitoring_periods": row[4] or 0,
                    }
                    
                    summary_data.append(source_summary)
                    total_violations += (row[3] or 0)

                return {
                    "time_period_hours": hours,
                    "total_sources": len(summary_data),
                    "total_sla_violations": total_violations,
                    "sources": summary_data,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            self.logger.error(f"SLAサマリー取得エラー: {e}")
            return {"error": str(e)}

    async def get_sla_trend(self, source_id: str, days: int = 7) -> Dict[str, any]:
        """SLAトレンド取得"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(days=days)

            with sqlite3.connect(self.database_manager.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT period_start, availability_percent, average_response_time, sla_violations
                    FROM sla_metrics
                    WHERE source_id = ? AND period_start >= ?
                    ORDER BY period_start
                """,
                    (source_id, start_time.isoformat()),
                )

                trend_data = []
                for row in cursor.fetchall():
                    trend_data.append({
                        "timestamp": row[0],
                        "availability_percent": row[1],
                        "average_response_time": row[2],
                        "sla_violations": row[3],
                    })

                # 統計計算
                if trend_data:
                    availabilities = [d["availability_percent"] for d in trend_data]
                    response_times = [d["average_response_time"] for d in trend_data if d["average_response_time"] != float("inf")]
                    
                    stats = {
                        "min_availability": min(availabilities),
                        "max_availability": max(availabilities),
                        "avg_availability": np.mean(availabilities),
                        "avg_response_time": np.mean(response_times) if response_times else None,
                        "total_violations": sum(d["sla_violations"] for d in trend_data),
                    }
                else:
                    stats = {}

                return {
                    "source_id": source_id,
                    "days": days,
                    "data_points": len(trend_data),
                    "trend_data": trend_data,
                    "statistics": stats,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            self.logger.error(f"SLAトレンド取得エラー ({source_id}): {e}")
            return {"error": str(e)}

    def calculate_sla_compliance_rate(self, sla_metrics: List[SLAMetrics]) -> float:
        """SLA遵守率計算"""
        if not sla_metrics:
            return 0.0

        total_periods = len(sla_metrics)
        violation_periods = sum(1 for metrics in sla_metrics if metrics.sla_violations > 0)
        
        compliance_rate = ((total_periods - violation_periods) / total_periods) * 100
        return round(compliance_rate, 2)

    def get_worst_performing_sources(
        self, sla_summary: Dict[str, any], limit: int = 5
    ) -> List[Dict[str, any]]:
        """パフォーマンスの悪いデータソース取得"""
        sources = sla_summary.get("sources", [])
        if not sources:
            return []

        # SLA違反数と可用性の低さで並び替え
        sorted_sources = sorted(
            sources,
            key=lambda x: (x.get("sla_violations", 0), -(x.get("average_availability", 100))),
            reverse=True
        )

        return sorted_sources[:limit]

    def format_sla_report(self, sla_summary: Dict[str, any]) -> str:
        """SLAレポートフォーマット"""
        try:
            report_lines = [
                f"=== SLAレポート ===",
                f"期間: {sla_summary['time_period_hours']}時間",
                f"総データソース数: {sla_summary['total_sources']}",
                f"総SLA違反数: {sla_summary['total_sla_violations']}",
                "",
                "データソース別詳細:",
            ]

            for source in sla_summary["sources"]:
                availability = source.get("average_availability", "N/A")
                response_time = source.get("average_response_time", "N/A")
                violations = source.get("sla_violations", 0)

                report_lines.append(
                    f"  {source['source_id']}: "
                    f"可用性 {availability}%, "
                    f"応答時間 {response_time}s, "
                    f"違反 {violations}回"
                )

            # ワーストパフォーマー
            worst_sources = self.get_worst_performing_sources(sla_summary, 3)
            if worst_sources:
                report_lines.extend([
                    "",
                    "要注意データソース:",
                ])
                for source in worst_sources:
                    report_lines.append(f"  - {source['source_id']} (違反: {source.get('sla_violations', 0)}回)")

            return "\n".join(report_lines)

        except Exception as e:
            self.logger.error(f"SLAレポートフォーマットエラー: {e}")
            return "SLAレポート生成エラー"

    def is_sla_violation(self, metrics: SLAMetrics, target: float) -> bool:
        """SLA違反判定"""
        return metrics.availability_percent < target

    def calculate_downtime_minutes(self, metrics: SLAMetrics) -> float:
        """ダウンタイム（分）計算"""
        return metrics.downtime_seconds / 60

    def calculate_availability_trend(self, sla_metrics: List[SLAMetrics]) -> str:
        """可用性トレンド計算"""
        if len(sla_metrics) < 2:
            return "stable"

        # 直近の可用性と過去の平均を比較
        recent = sla_metrics[-1].availability_percent
        historical = np.mean([m.availability_percent for m in sla_metrics[:-1]])

        if recent > historical + 1:
            return "improving"
        elif recent < historical - 1:
            return "degrading"
        else:
            return "stable"