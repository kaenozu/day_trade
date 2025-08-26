#!/usr/bin/env python3
"""
セキュリティ管制センター - セキュリティダッシュボード
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from .enums import SecurityLevel, IncidentStatus
from .models import SecurityMetrics

try:
    import psutil
    MONITORING_LIBS_AVAILABLE = True
except ImportError:
    MONITORING_LIBS_AVAILABLE = False


class SecurityDashboard:
    """セキュリティダッシュボード"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def get_dashboard_data(
        self, active_threats, incidents, component_status
    ) -> Dict[str, Any]:
        """セキュリティダッシュボードデータ取得"""
        try:
            dashboard = {
                "overview": {
                    "system_name": "Day Trade Security Control Center",
                    "version": "1.0.0",
                    "status": "active",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                },
                "threat_summary": await self._get_threat_summary(active_threats),
                "incident_summary": await self._get_incident_summary(incidents),
                "security_metrics": await self._calculate_security_metrics(
                    active_threats, incidents
                ),
                "recent_activities": await self._get_recent_activities(
                    active_threats, incidents
                ),
                "component_status": component_status,
                "recommendations": await self._generate_security_recommendations(
                    active_threats
                ),
            }

            return dashboard

        except Exception as e:
            self.logger.error(f"ダッシュボード取得エラー: {e}")
            return {"error": str(e)}

    async def _get_threat_summary(self, active_threats) -> Dict[str, Any]:
        """脅威サマリー取得"""
        threats = list(active_threats.values())

        summary = {
            "total": len(threats),
            "by_severity": {
                "critical": len(
                    [t for t in threats if t.severity == SecurityLevel.CRITICAL]
                ),
                "high": len([t for t in threats if t.severity == SecurityLevel.HIGH]),
                "medium": len(
                    [t for t in threats if t.severity == SecurityLevel.MEDIUM]
                ),
                "low": len([t for t in threats if t.severity == SecurityLevel.LOW]),
                "info": len([t for t in threats if t.severity == SecurityLevel.INFO]),
            },
            "by_category": {},
            "recent": [
                {
                    "id": t.id,
                    "title": t.title,
                    "severity": t.severity.value,
                    "detected_at": t.detected_at.isoformat(),
                }
                for t in sorted(threats, key=lambda x: x.detected_at, reverse=True)[:5]
            ],
        }

        # カテゴリ別統計
        for threat in threats:
            category = threat.category.value
            summary["by_category"][category] = (
                summary["by_category"].get(category, 0) + 1
            )

        return summary

    async def _get_incident_summary(self, incidents) -> Dict[str, Any]:
        """インシデントサマリー取得"""
        incidents_list = list(incidents.values())

        return {
            "total": len(incidents_list),
            "open": len([i for i in incidents_list if i.status == IncidentStatus.OPEN]),
            "investigating": len(
                [i for i in incidents_list if i.status == IncidentStatus.INVESTIGATING]
            ),
            "resolved": len(
                [i for i in incidents_list if i.status == IncidentStatus.RESOLVED]
            ),
            "recent": [
                {
                    "id": i.id,
                    "title": i.title,
                    "severity": i.severity.value,
                    "status": i.status.value,
                    "created_at": i.created_at.isoformat(),
                }
                for i in sorted(
                    incidents_list, key=lambda x: x.created_at, reverse=True
                )[:5]
            ],
        }

    async def _calculate_security_metrics(
        self, active_threats, incidents
    ) -> SecurityMetrics:
        """セキュリティメトリクス計算"""
        threats = list(active_threats.values())
        incidents_list = list(incidents.values())

        # 脅威統計
        critical_threats = len(
            [t for t in threats if t.severity == SecurityLevel.CRITICAL]
        )
        high_threats = len([t for t in threats if t.severity == SecurityLevel.HIGH])
        medium_threats = len([t for t in threats if t.severity == SecurityLevel.MEDIUM])
        low_threats = len([t for t in threats if t.severity == SecurityLevel.LOW])

        # インシデント統計
        open_incidents = len(
            [
                i
                for i in incidents_list
                if i.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]
            ]
        )
        resolved_incidents = len(
            [i for i in incidents_list if i.status == IncidentStatus.RESOLVED]
        )

        # 解決時間計算
        resolved_incidents_with_time = [
            i
            for i in incidents_list
            if i.status == IncidentStatus.RESOLVED and i.resolved_at
        ]

        if resolved_incidents_with_time:
            resolution_times = [
                (i.resolved_at - i.created_at).total_seconds() / 3600  # 時間単位
                for i in resolved_incidents_with_time
            ]
            mean_resolution_time = sum(resolution_times) / len(resolution_times)
        else:
            mean_resolution_time = 0.0

        # セキュリティスコア計算（100点満点）
        security_score = 100.0

        # 脅威によるスコア減点
        security_score -= critical_threats * 20  # 重大脅威1つにつき20点減点
        security_score -= high_threats * 10  # 高脅威1つにつき10点減点
        security_score -= medium_threats * 5  # 中脅威1つにつき5点減点
        security_score -= low_threats * 1  # 低脅威1つにつき1点減点

        # オープンインシデントによる減点
        security_score -= open_incidents * 15  # オープンインシデント1つにつき15点減点

        security_score = max(0.0, security_score)  # 0未満にはならない

        return SecurityMetrics(
            total_threats=len(threats),
            critical_threats=critical_threats,
            high_threats=high_threats,
            medium_threats=medium_threats,
            low_threats=low_threats,
            open_incidents=open_incidents,
            resolved_incidents=resolved_incidents,
            mean_resolution_time=mean_resolution_time,
            security_score=security_score,
            compliance_score=85.0,  # コンプライアンススコア（仮の値）
        )

    async def _get_recent_activities(
        self, active_threats, incidents
    ) -> List[Dict[str, Any]]:
        """最近のアクティビティ取得"""
        activities = []

        # 最近の脅威
        recent_threats = sorted(
            active_threats.values(), key=lambda x: x.detected_at, reverse=True
        )[:10]

        for threat in recent_threats:
            activities.append(
                {
                    "type": "threat_detected",
                    "title": f"脅威検出: {threat.title}",
                    "severity": threat.severity.value,
                    "timestamp": threat.detected_at.isoformat(),
                    "details": {
                        "threat_id": threat.id,
                        "category": threat.category.value,
                        "source": threat.source,
                    },
                }
            )

        # 最近のインシデント
        recent_incidents = sorted(
            incidents.values(), key=lambda x: x.created_at, reverse=True
        )[:5]

        for incident in recent_incidents:
            activities.append(
                {
                    "type": "incident_created",
                    "title": f"インシデント作成: {incident.title}",
                    "severity": incident.severity.value,
                    "timestamp": incident.created_at.isoformat(),
                    "details": {
                        "incident_id": incident.id,
                        "status": incident.status.value,
                    },
                }
            )

        # 時系列でソート
        activities.sort(key=lambda x: x["timestamp"], reverse=True)

        return activities[:15]  # 最新15件

    async def _generate_security_recommendations(
        self, active_threats
    ) -> List[str]:
        """セキュリティ推奨事項を生成"""
        recommendations = []

        # アクティブ脅威に基づく推奨事項
        critical_threats = [
            t
            for t in active_threats.values()
            if t.severity == SecurityLevel.CRITICAL
        ]
        high_threats = [
            t for t in active_threats.values() if t.severity == SecurityLevel.HIGH
        ]

        if critical_threats:
            recommendations.append(
                "重大な脅威が検出されています。即座の対応が必要です。"
            )
            recommendations.append(
                "システム管理者に緊急連絡を取り、対応を開始してください。"
            )

        if high_threats:
            recommendations.append("高リスク脅威への対応を優先してください。")
            recommendations.append(
                "セキュリティパッチの適用とシステム更新を実施してください。"
            )

        # 一般的な推奨事項
        recommendations.extend(
            [
                "定期的なセキュリティスキャンを継続してください。",
                "セキュリティログの監視体制を強化してください。",
                "スタッフに対するセキュリティ教育を実施してください。",
                "インシデント対応プロセスの見直しと改善を行ってください。",
                "バックアップシステムの動作確認を定期的に実施してください。",
            ]
        )

        return recommendations

    async def get_component_status(
        self, vulnerability_manager, coding_enforcer, data_protection,
        access_auditor, security_tester
    ) -> Dict[str, Any]:
        """コンポーネント状態取得"""
        status = {}

        # セキュリティコンポーネントの状態チェック
        components = [
            ("vulnerability_manager", vulnerability_manager),
            ("secure_coding_enforcer", coding_enforcer),
            ("data_protection_manager", data_protection),
            ("access_control_auditor", access_auditor),
            ("security_test_coordinator", security_tester),
        ]

        for name, component in components:
            if component:
                status[name] = {
                    "status": "active",
                    "last_check": datetime.now(timezone.utc).isoformat(),
                }
            else:
                status[name] = {"status": "unavailable", "last_check": None}

        # システムリソース状態
        if MONITORING_LIBS_AVAILABLE:
            try:
                import psutil

                status["system_resources"] = {
                    "status": "active",
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": (
                        psutil.disk_usage("/").percent
                        if hasattr(psutil.disk_usage("/"), "percent")
                        else 0
                    ),
                }
            except Exception:
                status["system_resources"] = {"status": "error"}
        else:
            status["system_resources"] = {"status": "unavailable"}

        return status