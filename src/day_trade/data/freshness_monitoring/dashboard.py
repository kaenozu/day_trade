#!/usr/bin/env python3
"""
ダッシュボードモジュール
監視ダッシュボードデータの生成とレポート機能を提供
"""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from .database import DatabaseManager
from .models import DataSourceConfig


class DashboardManager:
    """ダッシュボード管理クラス"""

    def __init__(self, database_manager: DatabaseManager):
        self.logger = logging.getLogger(__name__)
        self.database_manager = database_manager

    async def get_monitoring_dashboard(
        self,
        data_sources: Dict[str, DataSourceConfig],
        monitoring_stats: Dict[str, int],
        monitoring_active: bool,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """監視ダッシュボードデータ取得"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            dashboard_data = {
                "overview": {
                    "total_sources": len(data_sources),
                    "active_monitoring": monitoring_active,
                    "monitoring_stats": monitoring_stats,
                    "time_range_hours": hours,
                },
                "source_summary": await self._get_source_summary(data_sources, start_time),
                "recent_alerts": await self._get_recent_alerts(start_time),
                "sla_summary": await self._get_sla_summary(start_time),
                "health_overview": await self._get_health_overview(data_sources),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            return dashboard_data

        except Exception as e:
            self.logger.error(f"ダッシュボードデータ取得エラー: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

    async def _get_source_summary(
        self, data_sources: Dict[str, DataSourceConfig], start_time: datetime
    ) -> List[Dict[str, Any]]:
        """データソースサマリー取得"""
        source_summary = []

        for source_id, config in data_sources.items():
            try:
                with sqlite3.connect(self.database_manager.db_path) as conn:
                    # 最新ステータス
                    cursor = conn.execute(
                        """
                        SELECT current_status, consecutive_failures, last_success
                        FROM data_source_states WHERE source_id = ?
                    """,
                        (source_id,),
                    )

                    state = cursor.fetchone()

                    # 最新鮮度チェック
                    cursor = conn.execute(
                        """
                        SELECT status, age_seconds, quality_score, timestamp
                        FROM freshness_checks WHERE source_id = ?
                        ORDER BY timestamp DESC LIMIT 1
                    """,
                        (source_id,),
                    )

                    latest_check = cursor.fetchone()

                    # 期間内のアラート数
                    cursor = conn.execute(
                        """
                        SELECT COUNT(*) FROM data_alerts
                        WHERE source_id = ? AND timestamp >= ? AND NOT resolved
                    """,
                        (source_id, start_time.isoformat()),
                    )

                    active_alerts = cursor.fetchone()[0]

                    # チェック成功率計算
                    cursor = conn.execute(
                        """
                        SELECT COUNT(*) as total,
                               SUM(CASE WHEN status = 'fresh' THEN 1 ELSE 0 END) as fresh
                        FROM freshness_checks
                        WHERE source_id = ? AND timestamp >= ?
                    """,
                        (source_id, start_time.isoformat()),
                    )

                    check_stats = cursor.fetchone()
                    total_checks = check_stats[0] if check_stats else 0
                    fresh_checks = check_stats[1] if check_stats else 0
                    success_rate = (
                        (fresh_checks / total_checks * 100) if total_checks > 0 else 0
                    )

                    summary = {
                        "source_id": source_id,
                        "source_type": config.source_type.value,
                        "monitoring_level": config.monitoring_level.value,
                        "current_status": state[0] if state else "unknown",
                        "consecutive_failures": state[1] if state else 0,
                        "last_success": state[2] if state else None,
                        "active_alerts": active_alerts,
                        "success_rate_percent": round(success_rate, 1),
                        "total_checks": total_checks,
                        "sla_target": config.sla_target,
                    }

                    if latest_check:
                        summary.update(
                            {
                                "latest_freshness_status": latest_check[0],
                                "latest_age_seconds": latest_check[1],
                                "latest_quality_score": latest_check[2],
                                "last_check": latest_check[3],
                            }
                        )

                    source_summary.append(summary)

            except Exception as e:
                self.logger.error(f"ソースサマリー取得エラー ({source_id}): {e}")

        return source_summary

    async def _get_recent_alerts(self, start_time: datetime) -> List[Dict[str, Any]]:
        """最新アラート取得"""
        try:
            with sqlite3.connect(self.database_manager.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT alert_id, source_id, severity, alert_type, message, timestamp
                    FROM data_alerts
                    WHERE timestamp >= ? AND NOT resolved
                    ORDER BY timestamp DESC LIMIT 20
                """,
                    (start_time.isoformat(),),
                )

                alerts = []
                for row in cursor.fetchall():
                    alerts.append(
                        {
                            "alert_id": row[0],
                            "source_id": row[1],
                            "severity": row[2],
                            "alert_type": row[3],
                            "message": row[4],
                            "timestamp": row[5],
                        }
                    )

                return alerts

        except Exception as e:
            self.logger.error(f"最新アラート取得エラー: {e}")
            return []

    async def _get_sla_summary(self, start_time: datetime) -> List[Dict[str, Any]]:
        """SLAサマリー取得"""
        try:
            with sqlite3.connect(self.database_manager.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT source_id, AVG(availability_percent), AVG(average_response_time),
                           SUM(sla_violations) as total_violations
                    FROM sla_metrics
                    WHERE period_start >= ?
                    GROUP BY source_id
                """,
                    (start_time.isoformat(),),
                )

                sla_summary = []
                for row in cursor.fetchall():
                    sla_summary.append(
                        {
                            "source_id": row[0],
                            "average_availability": (
                                round(row[1], 2) if row[1] else None
                            ),
                            "average_response_time": (
                                round(row[2], 3) if row[2] else None
                            ),
                            "sla_violations": row[3] or 0,
                        }
                    )

                return sla_summary

        except Exception as e:
            self.logger.error(f"SLAサマリー取得エラー: {e}")
            return []

    async def _get_health_overview(
        self, data_sources: Dict[str, DataSourceConfig]
    ) -> Dict[str, Any]:
        """ヘルス概要取得"""
        try:
            health_counts = {"healthy": 0, "warning": 0, "critical": 0, "unknown": 0}

            for source_id in data_sources.keys():
                try:
                    with sqlite3.connect(self.database_manager.db_path) as conn:
                        # 最新状態取得
                        cursor = conn.execute(
                            """
                            SELECT current_status, consecutive_failures
                            FROM data_source_states WHERE source_id = ?
                        """,
                            (source_id,),
                        )

                        state = cursor.fetchone()
                        if not state:
                            health_counts["unknown"] += 1
                            continue

                        current_status = state[0]
                        consecutive_failures = state[1]

                        # ヘルス判定
                        if current_status == "healthy" and consecutive_failures == 0:
                            health_counts["healthy"] += 1
                        elif consecutive_failures <= 2:
                            health_counts["warning"] += 1
                        else:
                            health_counts["critical"] += 1

                except Exception as e:
                    self.logger.error(f"ヘルス状態取得エラー ({source_id}): {e}")
                    health_counts["unknown"] += 1

            # 全体ヘルススコア計算
            total = sum(health_counts.values())
            if total > 0:
                health_score = (health_counts["healthy"] / total) * 100
            else:
                health_score = 0

            return {
                "health_counts": health_counts,
                "total_sources": total,
                "health_score_percent": round(health_score, 1),
                "status": self._determine_overall_status(health_counts),
            }

        except Exception as e:
            self.logger.error(f"ヘルス概要取得エラー: {e}")
            return {"error": str(e)}

    def _determine_overall_status(self, health_counts: Dict[str, int]) -> str:
        """全体ステータス判定"""
        if health_counts["critical"] > 0:
            return "critical"
        elif health_counts["warning"] > 0:
            return "warning"
        elif health_counts["healthy"] > 0:
            return "healthy"
        else:
            return "unknown"

    async def generate_summary_report(
        self, dashboard_data: Dict[str, Any]
    ) -> str:
        """サマリーレポート生成"""
        try:
            overview = dashboard_data.get("overview", {})
            health = dashboard_data.get("health_overview", {})
            sources = dashboard_data.get("source_summary", [])
            alerts = dashboard_data.get("recent_alerts", [])

            report_lines = [
                "=== データ鮮度監視システム サマリーレポート ===",
                f"生成日時: {dashboard_data.get('generated_at', 'N/A')}",
                f"監視期間: {overview.get('time_range_hours', 'N/A')}時間",
                "",
                "システム概要:",
                f"  総データソース数: {overview.get('total_sources', 0)}",
                f"  監視アクティブ: {'はい' if overview.get('active_monitoring') else 'いいえ'}",
                f"  実行チェック数: {overview.get('monitoring_stats', {}).get('total_checks_performed', 0)}",
                "",
                "ヘルス状況:",
                f"  正常: {health.get('health_counts', {}).get('healthy', 0)}",
                f"  警告: {health.get('health_counts', {}).get('warning', 0)}",
                f"  重要: {health.get('health_counts', {}).get('critical', 0)}",
                f"  不明: {health.get('health_counts', {}).get('unknown', 0)}",
                f"  ヘルススコア: {health.get('health_score_percent', 0)}%",
                "",
                f"アクティブアラート数: {len(alerts)}",
            ]

            # 問題のあるデータソース
            problem_sources = [
                s for s in sources
                if s.get("consecutive_failures", 0) > 0 or s.get("active_alerts", 0) > 0
            ]

            if problem_sources:
                report_lines.extend([
                    "",
                    "要注意データソース:",
                ])
                for source in problem_sources[:5]:  # 上位5個
                    report_lines.append(
                        f"  - {source['source_id']}: "
                        f"失敗 {source.get('consecutive_failures', 0)}回, "
                        f"アラート {source.get('active_alerts', 0)}件"
                    )

            # 高重要度アラート
            critical_alerts = [a for a in alerts if a.get("severity") == "critical"]
            if critical_alerts:
                report_lines.extend([
                    "",
                    "クリティカルアラート:",
                ])
                for alert in critical_alerts[:3]:  # 上位3個
                    report_lines.append(f"  - {alert['source_id']}: {alert['message']}")

            return "\n".join(report_lines)

        except Exception as e:
            self.logger.error(f"サマリーレポート生成エラー: {e}")
            return "サマリーレポート生成に失敗しました"

    def format_dashboard_for_console(self, dashboard_data: Dict[str, Any]) -> str:
        """コンソール表示用フォーマット"""
        try:
            overview = dashboard_data.get("overview", {})
            health = dashboard_data.get("health_overview", {})

            lines = [
                "╔══════════════════════════════════════════════════════════════╗",
                "║                   データ鮮度監視システム                     ║",
                "╚══════════════════════════════════════════════════════════════╝",
                "",
                f"監視状態: {'🟢 アクティブ' if overview.get('active_monitoring') else '🔴 停止中'}",
                f"データソース数: {overview.get('total_sources', 0)}",
                f"ヘルススコア: {health.get('health_score_percent', 0):.1f}%",
                "",
                "ヘルス分布:",
                f"  🟢 正常: {health.get('health_counts', {}).get('healthy', 0)}",
                f"  🟡 警告: {health.get('health_counts', {}).get('warning', 0)}",
                f"  🔴 重要: {health.get('health_counts', {}).get('critical', 0)}",
                f"  ⚪ 不明: {health.get('health_counts', {}).get('unknown', 0)}",
            ]

            return "\n".join(lines)

        except Exception as e:
            self.logger.error(f"コンソールフォーマットエラー: {e}")
            return "ダッシュボード表示エラー"