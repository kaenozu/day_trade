#!/usr/bin/env python3
"""
ダッシュボード・レポート機能
監視データの可視化とレポート生成を提供します
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from .database_operations import DatabaseOperations
from .models import DashboardData, DataSourceConfig


class DashboardManager:
    """ダッシュボード・レポート管理システム
    
    監視システムのデータを集約し、ダッシュボード表示用のデータや
    各種レポートを生成します。
    """
    
    def __init__(self, db_operations: DatabaseOperations):
        """ダッシュボード管理システムを初期化
        
        Args:
            db_operations: データベース操作インスタンス
        """
        self.logger = logging.getLogger(__name__)
        self.db_operations = db_operations
        self.db_path = db_operations.db_path
    
    async def get_monitoring_dashboard(
        self,
        data_sources: Dict[str, DataSourceConfig],
        hours: int = 24
    ) -> DashboardData:
        """監視ダッシュボードデータ取得
        
        Args:
            data_sources: データソース設定の辞書
            hours: 表示対象期間（時間）
            
        Returns:
            ダッシュボードデータ
        """
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            current_time = datetime.now(timezone.utc)
            
            # 概要情報
            overview = await self._generate_overview(data_sources, hours)
            
            # データソース別サマリー
            source_summary = await self._generate_source_summary(
                data_sources, start_time
            )
            
            # 最新アラート
            recent_alerts = await self._get_recent_alerts(start_time)
            
            # SLAサマリー
            sla_summary = await self._generate_sla_summary(
                data_sources, start_time
            )
            
            return DashboardData(
                overview=overview,
                source_summary=source_summary,
                recent_alerts=recent_alerts,
                sla_summary=sla_summary,
                generated_at=current_time.isoformat(),
                time_range_hours=hours,
            )
            
        except Exception as e:
            self.logger.error(f"ダッシュボードデータ取得エラー: {e}")
            return DashboardData(
                overview={},
                source_summary=[],
                recent_alerts=[],
                sla_summary=[],
                generated_at=datetime.now(timezone.utc).isoformat(),
                time_range_hours=hours,
                error=str(e),
            )
    
    async def _generate_overview(
        self,
        data_sources: Dict[str, DataSourceConfig],
        hours: int
    ) -> Dict[str, Any]:
        """概要情報生成
        
        Args:
            data_sources: データソース設定の辞書
            hours: 表示対象期間（時間）
            
        Returns:
            概要情報
        """
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                # 総チェック数
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM freshness_checks WHERE timestamp >= ?",
                    (start_time.isoformat(),)
                )
                total_checks = cursor.fetchone()[0]
                
                # 鮮度違反数
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM freshness_checks WHERE timestamp >= ? AND status != 'fresh'",
                    (start_time.isoformat(),)
                )
                freshness_violations = cursor.fetchone()[0]
                
                # 整合性違反数
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM integrity_checks WHERE timestamp >= ? AND passed = 0",
                    (start_time.isoformat(),)
                )
                integrity_violations = cursor.fetchone()[0]
                
                # アクティブアラート数
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM data_alerts WHERE timestamp >= ? AND resolved = 0",
                    (start_time.isoformat(),)
                )
                active_alerts = cursor.fetchone()[0]
                
                # SLA違反数
                cursor = conn.execute(
                    "SELECT SUM(sla_violations) FROM sla_metrics WHERE period_start >= ?",
                    (start_time.isoformat(),)
                )
                sla_violations = cursor.fetchone()[0] or 0
            
            return {
                "total_sources": len(data_sources),
                "active_monitoring": True,  # 実際の監視状態を反映
                "time_range_hours": hours,
                "monitoring_stats": {
                    "total_checks_performed": total_checks,
                    "freshness_violations": freshness_violations,
                    "integrity_violations": integrity_violations,
                    "active_alerts": active_alerts,
                    "sla_violations": sla_violations,
                },
                "health_score": self._calculate_overall_health_score(
                    total_checks, freshness_violations, integrity_violations
                ),
            }
            
        except Exception as e:
            self.logger.error(f"概要情報生成エラー: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_health_score(
        self,
        total_checks: int,
        freshness_violations: int,
        integrity_violations: int
    ) -> float:
        """全体健全性スコア計算
        
        Args:
            total_checks: 総チェック数
            freshness_violations: 鮮度違反数
            integrity_violations: 整合性違反数
            
        Returns:
            健全性スコア（0-100）
        """
        if total_checks == 0:
            return 100.0
        
        violation_rate = (freshness_violations + integrity_violations) / total_checks
        health_score = max(0.0, (1.0 - violation_rate) * 100)
        
        return round(health_score, 1)
    
    async def _generate_source_summary(
        self,
        data_sources: Dict[str, DataSourceConfig],
        start_time: datetime
    ) -> List[Dict[str, Any]]:
        """データソース別サマリー生成
        
        Args:
            data_sources: データソース設定の辞書
            start_time: 開始時刻
            
        Returns:
            データソース別サマリーのリスト
        """
        summary_list = []
        
        for source_id, config in data_sources.items():
            try:
                summary = await self._generate_single_source_summary(
                    source_id, config, start_time
                )
                summary_list.append(summary)
                
            except Exception as e:
                self.logger.error(f"ソースサマリー生成エラー ({source_id}): {e}")
                summary_list.append({
                    "source_id": source_id,
                    "error": str(e),
                })
        
        return summary_list
    
    async def _generate_single_source_summary(
        self,
        source_id: str,
        config: DataSourceConfig,
        start_time: datetime
    ) -> Dict[str, Any]:
        """個別データソースサマリー生成
        
        Args:
            source_id: データソースID
            config: データソース設定
            start_time: 開始時刻
            
        Returns:
            個別ソースサマリー
        """
        with sqlite3.connect(self.db_path) as conn:
            # 最新ステータス
            source_state = self.db_operations.get_source_state(source_id)
            
            # 最新鮮度チェック
            cursor = conn.execute(
                """
                SELECT status, age_seconds, quality_score, timestamp
                FROM freshness_checks WHERE source_id = ?
                ORDER BY timestamp DESC LIMIT 1
            """,
                (source_id,)
            )
            latest_check = cursor.fetchone()
            
            # 期間内のアラート数
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM data_alerts
                WHERE source_id = ? AND timestamp >= ? AND NOT resolved
            """,
                (source_id, start_time.isoformat())
            )
            active_alerts = cursor.fetchone()[0]
            
            # 期間内のチェック統計
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_checks,
                    SUM(CASE WHEN status = 'fresh' THEN 1 ELSE 0 END) as fresh_count,
                    AVG(age_seconds) as avg_age
                FROM freshness_checks 
                WHERE source_id = ? AND timestamp >= ?
            """,
                (source_id, start_time.isoformat())
            )
            check_stats = cursor.fetchone()
            
            summary = {
                "source_id": source_id,
                "source_type": config.source_type.value,
                "monitoring_level": config.monitoring_level.value,
                "active_alerts": active_alerts,
                "check_statistics": {
                    "total_checks": check_stats[0] or 0,
                    "fresh_checks": check_stats[1] or 0,
                    "average_age_seconds": round(check_stats[2] or 0, 1),
                    "freshness_rate": (
                        round((check_stats[1] or 0) / (check_stats[0] or 1) * 100, 1)
                    ),
                },
            }
            
            # データソース状態情報
            if source_state:
                summary.update({
                    "current_status": source_state.current_status,
                    "consecutive_failures": source_state.consecutive_failures,
                    "last_success": (
                        source_state.last_success.isoformat()
                        if source_state.last_success else None
                    ),
                    "recovery_attempts": source_state.recovery_attempts,
                })
            else:
                summary.update({
                    "current_status": "unknown",
                    "consecutive_failures": 0,
                    "last_success": None,
                    "recovery_attempts": 0,
                })
            
            # 最新チェック情報
            if latest_check:
                summary.update({
                    "latest_freshness_status": latest_check[0],
                    "latest_age_seconds": latest_check[1],
                    "latest_quality_score": latest_check[2],
                    "last_check": latest_check[3],
                })
            
            return summary
    
    async def _get_recent_alerts(
        self,
        start_time: datetime,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """最新アラート取得
        
        Args:
            start_time: 開始時刻
            limit: 取得件数制限
            
        Returns:
            最新アラートのリスト
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT alert_id, source_id, severity, alert_type, message, 
                           timestamp, resolved, resolved_at
                    FROM data_alerts
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC LIMIT ?
                """,
                    (start_time.isoformat(), limit)
                )
                
                alerts = []
                for row in cursor.fetchall():
                    alerts.append({
                        "alert_id": row[0],
                        "source_id": row[1],
                        "severity": row[2],
                        "alert_type": row[3],
                        "message": row[4],
                        "timestamp": row[5],
                        "resolved": bool(row[6]),
                        "resolved_at": row[7],
                    })
                
                return alerts
                
        except Exception as e:
            self.logger.error(f"最新アラート取得エラー: {e}")
            return []
    
    async def _generate_sla_summary(
        self,
        data_sources: Dict[str, DataSourceConfig],
        start_time: datetime
    ) -> List[Dict[str, Any]]:
        """SLAサマリー生成
        
        Args:
            data_sources: データソース設定の辞書
            start_time: 開始時刻
            
        Returns:
            SLAサマリーのリスト
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT 
                        source_id,
                        AVG(availability_percent) as avg_availability,
                        AVG(average_response_time) as avg_response_time,
                        SUM(sla_violations) as total_violations,
                        COUNT(*) as measurement_count
                    FROM sla_metrics
                    WHERE period_start >= ?
                    GROUP BY source_id
                """,
                    (start_time.isoformat(),)
                )
                
                sla_data = []
                for row in cursor.fetchall():
                    source_id, avg_avail, avg_resp, violations, count = row
                    
                    # SLA目標取得
                    sla_target = data_sources.get(source_id, {}).sla_target if source_id in data_sources else 99.9
                    
                    sla_data.append({
                        "source_id": source_id,
                        "average_availability": round(avg_avail, 2),
                        "average_response_time": round(avg_resp, 3),
                        "sla_violations": violations or 0,
                        "measurement_periods": count,
                        "sla_target": sla_target,
                        "sla_compliance": avg_avail >= sla_target,
                        "violation_rate": round((violations or 0) / count * 100, 2),
                    })
                
                return sla_data
                
        except Exception as e:
            self.logger.error(f"SLAサマリー生成エラー: {e}")
            return []
    
    async def generate_health_report(
        self,
        data_sources: Dict[str, DataSourceConfig],
        period_days: int = 7
    ) -> Dict[str, Any]:
        """健全性レポート生成
        
        Args:
            data_sources: データソース設定の辞書
            period_days: レポート期間（日数）
            
        Returns:
            健全性レポート
        """
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=period_days)
            
            # 期間統計
            period_stats = await self._calculate_period_statistics(
                start_time, end_time
            )
            
            # データソース別健全性
            source_health = await self._calculate_source_health_scores(
                data_sources, start_time, end_time
            )
            
            # トレンド分析
            trend_analysis = await self._calculate_health_trends(
                start_time, end_time
            )
            
            # 推奨アクション
            recommendations = await self._generate_recommendations(
                source_health, trend_analysis
            )
            
            return {
                "report_type": "health_report",
                "period_days": period_days,
                "report_period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "period_statistics": period_stats,
                "source_health_scores": source_health,
                "trend_analysis": trend_analysis,
                "recommendations": recommendations,
            }
            
        except Exception as e:
            self.logger.error(f"健全性レポート生成エラー: {e}")
            return {
                "report_type": "health_report",
                "period_days": period_days,
                "error": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
    
    async def _calculate_period_statistics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """期間統計計算
        
        Args:
            start_time: 開始時刻
            end_time: 終了時刻
            
        Returns:
            期間統計情報
        """
        with sqlite3.connect(self.db_path) as conn:
            # 鮮度統計
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_checks,
                    SUM(CASE WHEN status = 'fresh' THEN 1 ELSE 0 END) as fresh_count,
                    SUM(CASE WHEN status = 'stale' THEN 1 ELSE 0 END) as stale_count,
                    SUM(CASE WHEN status = 'expired' THEN 1 ELSE 0 END) as expired_count,
                    AVG(age_seconds) as avg_age,
                    AVG(quality_score) as avg_quality
                FROM freshness_checks 
                WHERE timestamp BETWEEN ? AND ?
            """,
                (start_time.isoformat(), end_time.isoformat())
            )
            
            freshness_stats = cursor.fetchone()
            
            # 整合性統計
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_checks,
                    SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed_count
                FROM integrity_checks 
                WHERE timestamp BETWEEN ? AND ?
            """,
                (start_time.isoformat(), end_time.isoformat())
            )
            
            integrity_stats = cursor.fetchone()
            
            # アラート統計
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_alerts,
                    SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_alerts,
                    SUM(CASE WHEN severity = 'error' THEN 1 ELSE 0 END) as error_alerts,
                    SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as resolved_alerts
                FROM data_alerts 
                WHERE timestamp BETWEEN ? AND ?
            """,
                (start_time.isoformat(), end_time.isoformat())
            )
            
            alert_stats = cursor.fetchone()
            
            return {
                "freshness": {
                    "total_checks": freshness_stats[0] or 0,
                    "fresh_rate": (
                        round((freshness_stats[1] or 0) / (freshness_stats[0] or 1) * 100, 2)
                    ),
                    "stale_count": freshness_stats[2] or 0,
                    "expired_count": freshness_stats[3] or 0,
                    "average_age_seconds": round(freshness_stats[4] or 0, 1),
                    "average_quality_score": round(freshness_stats[5] or 0, 1),
                },
                "integrity": {
                    "total_checks": integrity_stats[0] or 0,
                    "pass_rate": (
                        round((integrity_stats[1] or 0) / (integrity_stats[0] or 1) * 100, 2)
                    ),
                },
                "alerts": {
                    "total_alerts": alert_stats[0] or 0,
                    "critical_alerts": alert_stats[1] or 0,
                    "error_alerts": alert_stats[2] or 0,
                    "resolved_alerts": alert_stats[3] or 0,
                    "resolution_rate": (
                        round((alert_stats[3] or 0) / (alert_stats[0] or 1) * 100, 2)
                    ),
                },
            }
    
    async def _calculate_source_health_scores(
        self,
        data_sources: Dict[str, DataSourceConfig],
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """データソース別健全性スコア計算
        
        Args:
            data_sources: データソース設定の辞書
            start_time: 開始時刻
            end_time: 終了時刻
            
        Returns:
            データソース別健全性スコアのリスト
        """
        health_scores = []
        
        for source_id in data_sources.keys():
            with sqlite3.connect(self.db_path) as conn:
                # 鮮度スコア
                cursor = conn.execute(
                    """
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'fresh' THEN 1 ELSE 0 END) as fresh
                    FROM freshness_checks 
                    WHERE source_id = ? AND timestamp BETWEEN ? AND ?
                """,
                    (source_id, start_time.isoformat(), end_time.isoformat())
                )
                
                freshness_data = cursor.fetchone()
                freshness_score = (
                    (freshness_data[1] or 0) / (freshness_data[0] or 1) * 100
                )
                
                # 整合性スコア
                cursor = conn.execute(
                    """
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed
                    FROM integrity_checks 
                    WHERE source_id = ? AND timestamp BETWEEN ? AND ?
                """,
                    (source_id, start_time.isoformat(), end_time.isoformat())
                )
                
                integrity_data = cursor.fetchone()
                integrity_score = (
                    (integrity_data[1] or 0) / (integrity_data[0] or 1) * 100
                )
                
                # アラートペナルティ
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM data_alerts WHERE source_id = ? AND timestamp BETWEEN ? AND ?",
                    (source_id, start_time.isoformat(), end_time.isoformat())
                )
                alert_count = cursor.fetchone()[0]
                alert_penalty = min(alert_count * 5, 50)  # 最大50ポイント減点
                
                # 総合健全性スコア
                overall_score = max(0, (freshness_score + integrity_score) / 2 - alert_penalty)
                
                health_scores.append({
                    "source_id": source_id,
                    "health_score": round(overall_score, 1),
                    "freshness_score": round(freshness_score, 1),
                    "integrity_score": round(integrity_score, 1),
                    "alert_count": alert_count,
                    "alert_penalty": alert_penalty,
                    "health_grade": self._get_health_grade(overall_score),
                })
        
        return sorted(health_scores, key=lambda x: x["health_score"], reverse=True)
    
    def _get_health_grade(self, score: float) -> str:
        """健全性グレード判定
        
        Args:
            score: 健全性スコア
            
        Returns:
            健全性グレード
        """
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    async def _calculate_health_trends(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """健全性トレンド計算
        
        Args:
            start_time: 開始時刻
            end_time: 終了時刻
            
        Returns:
            トレンド分析結果
        """
        # 簡略化されたトレンド分析
        # 実際の実装では、より詳細な時系列分析を行う
        return {
            "trend_direction": "stable",
            "improvement_areas": [],
            "warning_areas": [],
        }
    
    async def _generate_recommendations(
        self,
        source_health: List[Dict],
        trend_analysis: Dict
    ) -> List[str]:
        """推奨アクション生成
        
        Args:
            source_health: データソース別健全性スコア
            trend_analysis: トレンド分析結果
            
        Returns:
            推奨アクションのリスト
        """
        recommendations = []
        
        # 低健全性スコアのデータソースに対する推奨
        for source in source_health:
            if source["health_score"] < 80:
                recommendations.append(
                    f"{source['source_id']}: 健全性スコア {source['health_score']} - "
                    f"監視設定の見直しまたは回復戦略の強化を検討してください"
                )
        
        # アラート多発データソースに対する推奨
        high_alert_sources = [s for s in source_health if s["alert_count"] > 10]
        for source in high_alert_sources:
            recommendations.append(
                f"{source['source_id']}: アラート多発 ({source['alert_count']}件) - "
                f"データソースの安定性改善を検討してください"
            )
        
        if not recommendations:
            recommendations.append("現在、特別な対応を要する問題は検出されていません")
        
        return recommendations