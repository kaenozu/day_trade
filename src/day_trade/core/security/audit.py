#!/usr/bin/env python3
"""
セキュリティ監査サービス実装
Issue #918 項目9対応: セキュリティ強化

セキュリティイベント監査、脅威分析、レポート生成機能
"""

import json
import secrets
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any

from ..dependency_injection import (
    ILoggingService, IConfigurationService, injectable, singleton
)
from .interfaces import ISecurityAuditService
from .types import SecurityEvent, ThreatLevel


@singleton(ISecurityAuditService)
@injectable
class SecurityAuditService(ISecurityAuditService):
    """セキュリティ監査サービス実装"""

    def __init__(self,
                 logging_service: ILoggingService,
                 config_service: IConfigurationService):
        self.logging_service = logging_service
        self.config_service = config_service
        self.logger = logging_service.get_logger(__name__, "SecurityAuditService")

        # イベントストレージ
        self._events: List[SecurityEvent] = []
        self._event_lock = threading.RLock()

        # 設定
        config = config_service.get_config()
        security_config = config.get('security', {})
        self._max_events = security_config.get('max_audit_events', 10000)
        self._audit_log_file = security_config.get('audit_log_file', 'security_audit.log')

    def log_security_event(self, event: SecurityEvent) -> str:
        """セキュリティイベントログ"""
        try:
            with self._event_lock:
                # イベントID生成
                if not event.event_id:
                    event.event_id = f"evt_{int(time.time())}_{secrets.token_hex(4)}"

                # イベント追加
                self._events.append(event)

                # イベント数制限
                if len(self._events) > self._max_events:
                    self._events = self._events[-self._max_events:]

                # ログ出力
                self._write_audit_log(event)

                self.logger.info(f"Security event logged: {event.event_id}")
                return event.event_id

        except Exception as e:
            self.logger.error(f"Security event logging error: {e}")
            return ""

    def get_security_events(self, start_time: datetime = None,
                          end_time: datetime = None,
                          threat_level: ThreatLevel = None) -> List[SecurityEvent]:
        """セキュリティイベント取得"""
        try:
            with self._event_lock:
                filtered_events = self._events.copy()

                # 時間範囲フィルター
                if start_time:
                    filtered_events = [e for e in filtered_events if e.timestamp >= start_time]

                if end_time:
                    filtered_events = [e for e in filtered_events if e.timestamp <= end_time]

                # 脅威レベルフィルター
                if threat_level:
                    filtered_events = [e for e in filtered_events if e.threat_level == threat_level]

                return filtered_events

        except Exception as e:
            self.logger.error(f"Security event retrieval error: {e}")
            return []

    def analyze_threats(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """脅威分析"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_window_hours)

            events = self.get_security_events(start_time, end_time)

            # 統計計算
            total_events = len(events)
            threat_counts = {}
            event_type_counts = {}
            blocked_count = 0
            source_ips = set()

            for event in events:
                # 脅威レベル集計
                threat_level = event.threat_level.value
                threat_counts[threat_level] = threat_counts.get(threat_level, 0) + 1

                # イベント種別集計
                event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1

                # ブロック数集計
                if event.blocked:
                    blocked_count += 1

                # ソースIP集計
                if event.source_ip:
                    source_ips.add(event.source_ip)

            # リスク評価
            risk_score = self._calculate_risk_score(threat_counts, total_events)

            return {
                'analysis_period_hours': time_window_hours,
                'total_events': total_events,
                'threat_level_distribution': threat_counts,
                'event_type_distribution': event_type_counts,
                'blocked_events': blocked_count,
                'unique_source_ips': len(source_ips),
                'risk_score': risk_score,
                'risk_level': self._get_risk_level(risk_score),
                'top_threat_sources': list(source_ips)[:10]
            }

        except Exception as e:
            self.logger.error(f"Threat analysis error: {e}")
            return {}

    def generate_security_report(self) -> Dict[str, Any]:
        """セキュリティレポート生成"""
        try:
            # 24時間、7日間、30日間の分析
            analysis_24h = self.analyze_threats(24)
            analysis_7d = self.analyze_threats(24 * 7)
            analysis_30d = self.analyze_threats(24 * 30)

            # 全体統計
            with self._event_lock:
                total_events_all_time = len(self._events)

            return {
                'report_generated_at': datetime.now().isoformat(),
                'total_events_all_time': total_events_all_time,
                'analysis_24_hours': analysis_24h,
                'analysis_7_days': analysis_7d,
                'analysis_30_days': analysis_30d,
                'recommendations': self._generate_recommendations(analysis_24h, analysis_7d)
            }

        except Exception as e:
            self.logger.error(f"Security report generation error: {e}")
            return {}

    def _write_audit_log(self, event: SecurityEvent):
        """監査ログ書き込み"""
        try:
            log_entry = {
                'timestamp': event.timestamp.isoformat(),
                'event_id': event.event_id,
                'event_type': event.event_type,
                'threat_level': event.threat_level.value,
                'source_ip': event.source_ip,
                'user_id': event.user_id,
                'action': event.action.value if event.action else None,
                'resource': event.resource,
                'details': event.details,
                'blocked': event.blocked
            }

            # ファイルに追記（実装では外部ログシステムを使用することを推奨）
            log_line = json.dumps(log_entry, ensure_ascii=False)

            # 簡易ファイル書き込み（実装では適切なログローテーションを実装）
            with open(self._audit_log_file, 'a', encoding='utf-8') as f:
                f.write(log_line + '\n')

        except Exception as e:
            self.logger.error(f"Audit log write error: {e}")

    def _calculate_risk_score(self, threat_counts: Dict[str, int], total_events: int) -> float:
        """リスクスコア計算"""
        if total_events == 0:
            return 0.0

        # 脅威レベル重み
        weights = {
            'info': 0.1,
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 1.0
        }

        weighted_score = 0.0
        for threat_level, count in threat_counts.items():
            weight = weights.get(threat_level, 0.5)
            weighted_score += (count / total_events) * weight

        return min(weighted_score * 100, 100.0)  # 0-100スケール

    def _get_risk_level(self, risk_score: float) -> str:
        """リスクレベル取得"""
        if risk_score >= 80:
            return 'CRITICAL'
        elif risk_score >= 60:
            return 'HIGH'
        elif risk_score >= 40:
            return 'MEDIUM'
        elif risk_score >= 20:
            return 'LOW'
        else:
            return 'MINIMAL'

    def _generate_recommendations(self, analysis_24h: Dict[str, Any],
                                analysis_7d: Dict[str, Any]) -> List[str]:
        """推奨事項生成"""
        recommendations = []

        try:
            # 24時間の高脅威イベント数チェック
            high_threats_24h = analysis_24h.get('threat_level_distribution', {})
            high_count = high_threats_24h.get('high', 0) + high_threats_24h.get('critical', 0)

            if high_count > 10:
                recommendations.append("高脅威イベントが頻発しています。セキュリティ対策の見直しを推奨します。")

            # ブロック率チェック
            total_24h = analysis_24h.get('total_events', 0)
            blocked_24h = analysis_24h.get('blocked_events', 0)

            if total_24h > 0 and (blocked_24h / total_24h) > 0.1:
                recommendations.append("ブロック率が高くなっています。攻撃者の活動が活発化している可能性があります。")

            # IP多様性チェック
            unique_ips = analysis_24h.get('unique_source_ips', 0)
            if unique_ips > 100:
                recommendations.append("多数のIPアドレスからのアクセスが検出されています。分散攻撃の可能性を調査してください。")

            # 長期トレンドチェック
            events_24h = analysis_24h.get('total_events', 0)
            events_7d = analysis_7d.get('total_events', 0)

            if events_24h > 0 and events_7d > 0:
                daily_avg_7d = events_7d / 7
                if events_24h > daily_avg_7d * 2:
                    recommendations.append("イベント数が平常時より大幅に増加しています。")

            if not recommendations:
                recommendations.append("現在、セキュリティ状況は安定しています。定期的な監視を継続してください。")

        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
            recommendations.append("推奨事項の生成中にエラーが発生しました。")

        return recommendations