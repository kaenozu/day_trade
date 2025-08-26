#!/usr/bin/env python3
"""
アクセス制御システム - セキュリティレポート

このモジュールは、セキュリティレポート生成と統計機能を提供します。
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List


class ReportingMixin:
    """
    レポート機能のミックスイン
    
    AccessControlManagerにレポート関連機能を追加します。
    """

    def get_security_report(self) -> Dict[str, Any]:
        """
        セキュリティレポート生成
        
        Returns:
            Dict[str, Any]: 包括的なセキュリティレポート
        """
        now = datetime.utcnow()

        # ユーザー統計
        user_stats = {
            "total_users": len(self.users),
            "active_users": sum(1 for u in self.users.values() if u.is_active),
            "locked_users": sum(1 for u in self.users.values() if u.is_locked),
            "mfa_enabled_users": sum(1 for u in self.users.values() if u.mfa_enabled),
            "password_expiry_soon": sum(
                1
                for u in self.users.values()
                if u.password_changed_at
                and now - u.password_changed_at
                > timedelta(days=self.password_expiry_days - 7)
            ),
        }

        # セッション統計
        active_sessions = [s for s in self.sessions.values() if s.is_valid()]
        session_stats = {
            "total_sessions": len(self.sessions),
            "active_sessions": len(active_sessions),
            "high_risk_sessions": sum(1 for s in active_sessions if s.risk_score > 0.5),
        }

        # ログ統計（過去24時間）
        recent_logs = [
            log for log in self.access_logs if log.timestamp > now - timedelta(hours=24)
        ]

        log_stats = {
            "total_events_24h": len(recent_logs),
            "failed_logins_24h": sum(
                1 for log in recent_logs if log.action == "login" and not log.success
            ),
            "successful_logins_24h": sum(
                1 for log in recent_logs if log.action == "login" and log.success
            ),
        }

        report = {
            "report_id": f"access-control-report-{int(time.time())}",
            "generated_at": now.isoformat(),
            "user_statistics": user_stats,
            "session_statistics": session_stats,
            "log_statistics": log_stats,
            "recommendations": self._generate_security_recommendations(
                user_stats, session_stats, log_stats
            ),
        }

        return report

    def _generate_security_recommendations(
        self,
        user_stats: Dict[str, Any],
        session_stats: Dict[str, Any],
        log_stats: Dict[str, Any],
    ) -> List[str]:
        """
        セキュリティ推奨事項生成
        
        Args:
            user_stats: ユーザー統計
            session_stats: セッション統計
            log_stats: ログ統計
            
        Returns:
            List[str]: 推奨事項のリスト
        """
        recommendations = []

        if user_stats["locked_users"] > 0:
            recommendations.append(
                f"🔒 {user_stats['locked_users']}個のアカウントがロックされています。調査してください。"
            )

        mfa_ratio = user_stats["mfa_enabled_users"] / max(user_stats["total_users"], 1)
        if mfa_ratio < 0.8:
            recommendations.append(
                f"🔐 MFA有効化率が{mfa_ratio:.1%}です。全ユーザーでのMFA有効化を推奨します。"
            )

        if user_stats["password_expiry_soon"] > 0:
            recommendations.append(
                f"⏰ {user_stats['password_expiry_soon']}ユーザーのパスワード期限が近づいています。"
            )

        if session_stats["high_risk_sessions"] > 0:
            recommendations.append(
                f"⚠️ {session_stats['high_risk_sessions']}個の高リスクセッションが検出されています。"
            )

        if log_stats["failed_logins_24h"] > 10:
            recommendations.append(
                f"🚨 過去24時間で{log_stats['failed_logins_24h']}回のログイン失敗が発生しています。攻撃の可能性があります。"
            )

        if not recommendations:
            recommendations.append("✅ アクセス制御システムは正常に稼働しています。")

        return recommendations

    def get_user_activity_summary(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        ユーザーアクティビティサマリー取得
        
        Args:
            user_id: ユーザーID
            days: 取得日数
            
        Returns:
            Dict[str, Any]: アクティビティサマリー
        """
        user = self.users.get(user_id)
        if not user:
            return {}

        cutoff_date = datetime.utcnow() - timedelta(days=days)
        user_logs = [
            log for log in self.access_logs
            if log.user_id == user_id and log.timestamp > cutoff_date
        ]

        login_attempts = [log for log in user_logs if log.action == "login"]
        successful_logins = [log for log in login_attempts if log.success]
        failed_logins = [log for log in login_attempts if not log.success]

        return {
            "user_id": user_id,
            "username": user.username,
            "period_days": days,
            "total_activities": len(user_logs),
            "login_attempts": len(login_attempts),
            "successful_logins": len(successful_logins),
            "failed_logins": len(failed_logins),
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "account_status": {
                "is_active": user.is_active,
                "is_locked": user.is_locked,
                "mfa_enabled": user.mfa_enabled,
                "failed_attempts": user.failed_login_attempts,
            },
            "recent_activities": [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "action": log.action,
                    "resource": log.resource,
                    "success": log.success,
                    "ip_address": log.ip_address,
                }
                for log in user_logs[-10:]  # 最新10件
            ],
        }

    def get_session_statistics(self) -> Dict[str, Any]:
        """
        セッション統計取得
        
        Returns:
            Dict[str, Any]: セッション統計情報
        """
        active_sessions = [s for s in self.sessions.values() if s.is_valid()]
        
        # リスクレベル別分類
        risk_levels = {"low": 0, "medium": 0, "high": 0}
        for session in active_sessions:
            if session.risk_score < 0.3:
                risk_levels["low"] += 1
            elif session.risk_score < 0.7:
                risk_levels["medium"] += 1
            else:
                risk_levels["high"] += 1

        # ロール別統計
        role_sessions = {}
        for session in active_sessions:
            user = self.users.get(session.user_id)
            if user:
                role = user.role.value
                role_sessions[role] = role_sessions.get(role, 0) + 1

        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len(active_sessions),
            "risk_distribution": risk_levels,
            "role_distribution": role_sessions,
            "average_risk_score": (
                sum(s.risk_score for s in active_sessions) / len(active_sessions)
                if active_sessions else 0.0
            ),
        }

    def get_security_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        セキュリティイベント取得
        
        Args:
            hours: 取得する時間範囲
            
        Returns:
            List[Dict[str, Any]]: セキュリティイベントのリスト
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        security_events = []

        for log in self.access_logs:
            if log.timestamp < cutoff_time:
                continue

            # セキュリティ関連イベントを抽出
            is_security_event = (
                not log.success
                or log.action in ["mfa_failure", "permission_denied", "suspicious_activity"]
                or "failed_attempts" in log.details
            )

            if is_security_event:
                security_events.append({
                    "timestamp": log.timestamp.isoformat(),
                    "user_id": log.user_id,
                    "action": log.action,
                    "resource": log.resource,
                    "success": log.success,
                    "ip_address": log.ip_address,
                    "details": log.details,
                    "severity": self._calculate_event_severity(log),
                })

        # 重要度でソート
        security_events.sort(key=lambda x: x["severity"], reverse=True)
        return security_events

    def _calculate_event_severity(self, log_entry) -> int:
        """
        イベントの重要度計算
        
        Args:
            log_entry: ログエントリ
            
        Returns:
            int: 重要度スコア（高いほど重要）
        """
        severity = 1

        if not log_entry.success:
            severity += 2

        if log_entry.action == "login" and not log_entry.success:
            severity += 1

        if log_entry.action == "mfa_failure":
            severity += 3

        if "failed_attempts" in log_entry.details:
            attempts = log_entry.details.get("failed_attempts", 0)
            severity += min(attempts, 5)

        if log_entry.details.get("reason") == "ip_restricted":
            severity += 4

        return severity