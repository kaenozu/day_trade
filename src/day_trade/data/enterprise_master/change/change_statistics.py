#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理 - 変更統計

変更管理に関する統計情報と分析機能を提供
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from ..enums import ApprovalStatus, ChangeType
from ..models import DataChangeRequest


class ChangeStatistics:
    """変更統計クラス"""

    def __init__(self, change_requests: Dict[str, DataChangeRequest]):
        self.logger = logging.getLogger(__name__)
        self._change_requests = change_requests

    async def get_change_statistics(self) -> Dict[str, Any]:
        """変更統計情報取得"""
        try:
            stats = {
                "total_requests": len(self._change_requests),
                "by_status": {status.value: 0 for status in ApprovalStatus},
                "by_change_type": {change_type.value: 0 for change_type in ChangeType},
                "by_risk_level": {},
                "average_approval_time_hours": 0.0,
                "pending_high_risk": 0,
                "approval_rate": 0.0,
                "rejection_rate": 0.0,
            }

            approval_times = []
            total_processed = 0
            approved_count = 0
            rejected_count = 0

            for request in self._change_requests.values():
                # 状態別集計
                stats["by_status"][request.approval_status.value] += 1

                # 変更タイプ別集計
                stats["by_change_type"][request.change_type.value] += 1

                # リスクレベル別集計
                risk_level = request.impact_assessment.get("risk_level", "unknown")
                if risk_level not in stats["by_risk_level"]:
                    stats["by_risk_level"][risk_level] = 0
                stats["by_risk_level"][risk_level] += 1

                # 高リスクで承認待ちの件数
                if (
                    request.approval_status == ApprovalStatus.PENDING
                    and risk_level in ["high", "critical"]
                ):
                    stats["pending_high_risk"] += 1

                # 承認時間計算
                if request.approved_at:
                    if request.approval_status == ApprovalStatus.APPROVED:
                        approved_count += 1
                    elif request.approval_status == ApprovalStatus.REJECTED:
                        rejected_count += 1

                    total_processed += 1
                    approval_time = (
                        request.approved_at - request.requested_at
                    ).total_seconds() / 3600
                    approval_times.append(approval_time)

            # 平均承認時間
            if approval_times:
                stats["average_approval_time_hours"] = sum(approval_times) / len(
                    approval_times
                )

            # 承認率・却下率
            if total_processed > 0:
                stats["approval_rate"] = approved_count / total_processed
                stats["rejection_rate"] = rejected_count / total_processed

            return stats

        except Exception as e:
            self.logger.error(f"変更統計情報取得エラー: {e}")
            return {"error": str(e)}

    async def get_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """変更トレンド分析"""
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            daily_stats = {}
            for i in range(days):
                current_date = start_date + timedelta(days=i)
                date_key = current_date.strftime("%Y-%m-%d")
                daily_stats[date_key] = {
                    "total_requests": 0,
                    "approved": 0,
                    "rejected": 0,
                    "pending": 0,
                    "high_risk_requests": 0,
                }

            for request in self._change_requests.values():
                request_date = request.requested_at.strftime("%Y-%m-%d")

                if request_date in daily_stats:
                    daily_stats[request_date]["total_requests"] += 1

                    if request.approval_status == ApprovalStatus.APPROVED:
                        daily_stats[request_date]["approved"] += 1
                    elif request.approval_status == ApprovalStatus.REJECTED:
                        daily_stats[request_date]["rejected"] += 1
                    elif request.approval_status == ApprovalStatus.PENDING:
                        daily_stats[request_date]["pending"] += 1

                    risk_level = request.impact_assessment.get("risk_level", "low")
                    if risk_level in ["high", "critical"]:
                        daily_stats[request_date]["high_risk_requests"] += 1

            return {
                "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "daily_stats": daily_stats,
            }

        except Exception as e:
            self.logger.error(f"変更トレンド分析エラー: {e}")
            return {"error": str(e)}

    async def get_approval_performance(self) -> Dict[str, Any]:
        """承認パフォーマンス分析"""
        try:
            performance = {
                "approvers": {},
                "fastest_approver": None,
                "slowest_approver": None,
                "average_approval_time_by_risk": {},
                "sla_compliance": {"within_sla": 0, "exceeds_sla": 0, "sla_rate": 0.0},
            }

            approval_times_by_approver = {}
            approval_times_by_risk = {}
            sla_threshold_hours = 24  # SLA閾値: 24時間

            for request in self._change_requests.values():
                if request.approved_at and request.approved_by:
                    approver = request.approved_by
                    approval_time = (
                        request.approved_at - request.requested_at
                    ).total_seconds() / 3600

                    # 承認者別統計
                    if approver not in approval_times_by_approver:
                        approval_times_by_approver[approver] = []
                    approval_times_by_approver[approver].append(approval_time)

                    # リスクレベル別承認時間
                    risk_level = request.impact_assessment.get("risk_level", "low")
                    if risk_level not in approval_times_by_risk:
                        approval_times_by_risk[risk_level] = []
                    approval_times_by_risk[risk_level].append(approval_time)

                    # SLA準拠
                    if approval_time <= sla_threshold_hours:
                        performance["sla_compliance"]["within_sla"] += 1
                    else:
                        performance["sla_compliance"]["exceeds_sla"] += 1

            # 承認者統計計算
            for approver, times in approval_times_by_approver.items():
                performance["approvers"][approver] = {
                    "total_approvals": len(times),
                    "average_time_hours": sum(times) / len(times),
                    "fastest_approval_hours": min(times),
                    "slowest_approval_hours": max(times),
                }

            # 最速・最遅承認者
            if approval_times_by_approver:
                avg_times = {
                    approver: sum(times) / len(times)
                    for approver, times in approval_times_by_approver.items()
                }
                performance["fastest_approver"] = min(avg_times, key=avg_times.get)
                performance["slowest_approver"] = max(avg_times, key=avg_times.get)

            # リスクレベル別平均承認時間
            for risk_level, times in approval_times_by_risk.items():
                performance["average_approval_time_by_risk"][risk_level] = sum(times) / len(
                    times
                )

            # SLA準拠率
            total_sla = (
                performance["sla_compliance"]["within_sla"]
                + performance["sla_compliance"]["exceeds_sla"]
            )
            if total_sla > 0:
                performance["sla_compliance"]["sla_rate"] = (
                    performance["sla_compliance"]["within_sla"] / total_sla
                )

            return performance

        except Exception as e:
            self.logger.error(f"承認パフォーマンス分析エラー: {e}")
            return {"error": str(e)}

    async def get_rejection_analysis(self) -> Dict[str, Any]:
        """却下分析"""
        try:
            rejected_requests = [
                req
                for req in self._change_requests.values()
                if req.approval_status == ApprovalStatus.REJECTED
            ]

            analysis = {
                "total_rejections": len(rejected_requests),
                "rejection_reasons": {},
                "by_change_type": {},
                "by_risk_level": {},
                "common_rejection_patterns": [],
            }

            for request in rejected_requests:
                # 却下理由分類
                reason = request.rejection_reason or "未指定"
                if reason not in analysis["rejection_reasons"]:
                    analysis["rejection_reasons"][reason] = 0
                analysis["rejection_reasons"][reason] += 1

                # 変更タイプ別却下
                change_type = request.change_type.value
                if change_type not in analysis["by_change_type"]:
                    analysis["by_change_type"][change_type] = 0
                analysis["by_change_type"][change_type] += 1

                # リスクレベル別却下
                risk_level = request.impact_assessment.get("risk_level", "unknown")
                if risk_level not in analysis["by_risk_level"]:
                    analysis["by_risk_level"][risk_level] = 0
                analysis["by_risk_level"][risk_level] += 1

            # 共通却下パターンの特定
            analysis["common_rejection_patterns"] = self._identify_rejection_patterns(
                rejected_requests
            )

            return analysis

        except Exception as e:
            self.logger.error(f"却下分析エラー: {e}")
            return {"error": str(e)}

    async def get_workload_analysis(self) -> Dict[str, Any]:
        """ワークロード分析"""
        try:
            now = datetime.now(timezone.utc)
            pending_requests = [
                req
                for req in self._change_requests.values()
                if req.approval_status == ApprovalStatus.PENDING
            ]

            workload = {
                "current_pending": len(pending_requests),
                "aging_analysis": {"within_24h": 0, "1_3_days": 0, "3_7_days": 0, "over_7_days": 0},
                "priority_queue": {"critical": 0, "high": 0, "medium": 0, "low": 0},
                "estimated_approval_time": 0.0,
            }

            total_age_hours = 0

            for request in pending_requests:
                # エージング分析
                age_hours = (now - request.requested_at).total_seconds() / 3600

                if age_hours <= 24:
                    workload["aging_analysis"]["within_24h"] += 1
                elif age_hours <= 72:
                    workload["aging_analysis"]["1_3_days"] += 1
                elif age_hours <= 168:
                    workload["aging_analysis"]["3_7_days"] += 1
                else:
                    workload["aging_analysis"]["over_7_days"] += 1

                total_age_hours += age_hours

                # 優先度キュー
                risk_level = request.impact_assessment.get("risk_level", "low")
                if risk_level in workload["priority_queue"]:
                    workload["priority_queue"][risk_level] += 1

            # 推定承認時間（履歴ベース）
            stats = await self.get_change_statistics()
            workload["estimated_approval_time"] = stats.get("average_approval_time_hours", 0)

            return workload

        except Exception as e:
            self.logger.error(f"ワークロード分析エラー: {e}")
            return {"error": str(e)}

    def _identify_rejection_patterns(
        self, rejected_requests: List[DataChangeRequest]
    ) -> List[Dict[str, Any]]:
        """共通却下パターンの特定"""
        patterns = []

        try:
            # データ品質関連の却下
            quality_related = sum(
                1
                for req in rejected_requests
                if req.rejection_reason
                and ("品質" in req.rejection_reason or "quality" in req.rejection_reason.lower())
            )

            if quality_related > 0:
                patterns.append({
                    "pattern": "データ品質関連",
                    "count": quality_related,
                    "recommendation": "事前品質チェックプロセスの強化が必要",
                })

            # 承認権限関連の却下
            authority_related = sum(
                1
                for req in rejected_requests
                if req.rejection_reason
                and ("権限" in req.rejection_reason or "authority" in req.rejection_reason.lower())
            )

            if authority_related > 0:
                patterns.append({
                    "pattern": "承認権限関連",
                    "count": authority_related,
                    "recommendation": "承認フローの見直しと権限設定の明確化が必要",
                })

            # 高リスク変更の却下
            high_risk_rejected = sum(
                1
                for req in rejected_requests
                if req.impact_assessment.get("risk_level") in ["high", "critical"]
            )

            if high_risk_rejected > len(rejected_requests) * 0.5:
                patterns.append({
                    "pattern": "高リスク変更",
                    "count": high_risk_rejected,
                    "recommendation": "リスク評価基準の見直しと事前協議プロセスの導入を検討",
                })

        except Exception as e:
            self.logger.error(f"却下パターン分析エラー: {e}")

        return patterns