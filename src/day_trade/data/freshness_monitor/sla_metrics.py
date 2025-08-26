#!/usr/bin/env python3
"""
SLAメトリクス計算機能
Service Level Agreement メトリクスの計算、追跡、評価を行います
"""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np

from .database_operations import DatabaseOperations
from .enums import FreshnessStatus
from .models import DataSourceConfig, MonitoringStats, SLAMetrics


class SLAMetricsCalculator:
    """SLAメトリクス計算システム
    
    データソースのSLA（Service Level Agreement）メトリクスを計算し、
    可用性、応答時間、エラー率などのKPIを追跡します。
    """
    
    def __init__(self, db_operations: DatabaseOperations):
        """SLAメトリクス計算システムを初期化
        
        Args:
            db_operations: データベース操作インスタンス
        """
        self.logger = logging.getLogger(__name__)
        self.db_operations = db_operations
        self.db_path = db_operations.db_path
        
        # SLAメトリクス保存用（インメモリ）
        self.sla_history: Dict[str, List[SLAMetrics]] = {}
        
        # 統計情報
        self.stats = MonitoringStats()
    
    async def calculate_hourly_sla_metrics(
        self, 
        data_sources: Dict[str, DataSourceConfig]
    ):
        """時間別SLAメトリクス計算
        
        各データソースの過去1時間のパフォーマンスを評価し、
        SLAメトリクスを計算・保存します。
        
        Args:
            data_sources: データソース設定の辞書
        """
        try:
            current_time = datetime.now(timezone.utc)
            period_start = current_time.replace(
                minute=0, second=0, microsecond=0
            ) - timedelta(hours=1)
            period_end = current_time.replace(minute=0, second=0, microsecond=0)
            
            self.logger.info(
                f"SLAメトリクス計算開始: {period_start.isoformat()} - {period_end.isoformat()}"
            )
            
            for source_id, config in data_sources.items():
                try:
                    metrics = await self._calculate_source_sla_metrics(
                        source_id, config, period_start, period_end
                    )
                    
                    if metrics:
                        # データベース保存
                        await self.db_operations.save_sla_metrics(metrics)
                        
                        # インメモリ履歴更新
                        if source_id not in self.sla_history:
                            self.sla_history[source_id] = []
                        self.sla_history[source_id].append(metrics)
                        
                        # 履歴サイズ制限（直近100件まで）
                        if len(self.sla_history[source_id]) > 100:
                            self.sla_history[source_id] = self.sla_history[source_id][-100:]
                        
                        # SLA違反チェック
                        if metrics.sla_violations > 0:
                            self.stats.sla_violations += metrics.sla_violations
                            self.logger.warning(
                                f"SLA違反検出: {source_id} - 可用性 {metrics.availability_percent:.2f}%"
                            )
                    
                except Exception as e:
                    self.logger.error(f"SLAメトリクス計算エラー ({source_id}): {e}")
            
            self.logger.info("SLAメトリクス計算完了")
            
        except Exception as e:
            self.logger.error(f"時間別SLAメトリクス計算エラー: {e}")
    
    async def _calculate_source_sla_metrics(
        self,
        source_id: str,
        config: DataSourceConfig,
        period_start: datetime,
        period_end: datetime
    ) -> SLAMetrics:
        """個別データソースのSLAメトリクス計算
        
        Args:
            source_id: データソースID
            config: データソース設定
            period_start: 計算期間開始時刻
            period_end: 計算期間終了時刻
            
        Returns:
            SLAメトリクス、データがない場合はNone
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 該当期間の鮮度チェック結果取得
                cursor = conn.execute(
                    """
                    SELECT status, age_seconds FROM freshness_checks
                    WHERE source_id = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """,
                    (source_id, period_start.isoformat(), period_end.isoformat()),
                )
                
                checks = cursor.fetchall()
                
                if not checks:
                    return None
                
                # 基本統計計算
                total_checks = len(checks)
                fresh_checks = sum(
                    1 for status, _ in checks
                    if status == FreshnessStatus.FRESH.value
                )
                stale_checks = sum(
                    1 for status, _ in checks
                    if status == FreshnessStatus.STALE.value
                )
                expired_checks = sum(
                    1 for status, _ in checks
                    if status == FreshnessStatus.EXPIRED.value
                )
                
                # 可用性計算（FRESH状態の割合）
                availability_percent = (fresh_checks / total_checks) * 100
                
                # 応答時間計算（age_secondsの平均、ただしinf除外）
                valid_ages = [
                    age for _, age in checks 
                    if age != float("inf") and age is not None
                ]
                
                if valid_ages:
                    average_response_time = np.mean(valid_ages)
                else:
                    average_response_time = 0.0
                
                # アップタイム・ダウンタイム計算（1時間 = 3600秒）
                period_seconds = 3600.0
                uptime_seconds = (fresh_checks / total_checks) * period_seconds
                downtime_seconds = period_seconds - uptime_seconds
                
                # SLA違反判定
                sla_target = config.sla_target
                sla_violations = 1 if availability_percent < sla_target else 0
                
                # エラー数（EXPIRED状態のチェック）
                error_count = expired_checks
                
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
            self.logger.error(
                f"個別SLAメトリクス計算エラー ({source_id}): {e}"
            )
            return None
    
    async def calculate_daily_sla_summary(
        self,
        source_id: str,
        date: datetime = None
    ) -> Dict:
        """日別SLAサマリー計算
        
        Args:
            source_id: データソースID
            date: 計算対象日（デフォルトは昨日）
            
        Returns:
            日別SLAサマリー
        """
        if date is None:
            date = datetime.now(timezone.utc).date() - timedelta(days=1)
        
        try:
            start_of_day = datetime.combine(date, datetime.min.time()).replace(tzinfo=timezone.utc)
            end_of_day = start_of_day + timedelta(days=1)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT 
                        AVG(availability_percent) as avg_availability,
                        AVG(average_response_time) as avg_response_time,
                        SUM(error_count) as total_errors,
                        SUM(total_requests) as total_requests,
                        SUM(uptime_seconds) as total_uptime,
                        SUM(downtime_seconds) as total_downtime,
                        SUM(sla_violations) as total_violations,
                        COUNT(*) as hourly_samples
                    FROM sla_metrics
                    WHERE source_id = ? AND period_start BETWEEN ? AND ?
                """,
                    (source_id, start_of_day.isoformat(), end_of_day.isoformat()),
                )
                
                result = cursor.fetchone()
                
                if not result or result[0] is None:
                    return {
                        "source_id": source_id,
                        "date": date.isoformat(),
                        "no_data": True,
                    }
                
                return {
                    "source_id": source_id,
                    "date": date.isoformat(),
                    "average_availability": round(result[0], 2),
                    "average_response_time": round(result[1], 3),
                    "total_errors": result[2] or 0,
                    "total_requests": result[3] or 0,
                    "total_uptime_hours": round((result[4] or 0) / 3600, 2),
                    "total_downtime_hours": round((result[5] or 0) / 3600, 2),
                    "sla_violations": result[6] or 0,
                    "hourly_samples": result[7] or 0,
                }
                
        except Exception as e:
            self.logger.error(f"日別SLAサマリー計算エラー ({source_id}): {e}")
            return {
                "source_id": source_id,
                "date": date.isoformat(),
                "error": str(e),
            }
    
    async def get_sla_trend_analysis(
        self,
        source_id: str,
        days: int = 7
    ) -> Dict:
        """SLAトレンド分析
        
        Args:
            source_id: データソースID
            days: 分析対象日数
            
        Returns:
            トレンド分析結果
        """
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT 
                        DATE(period_start) as date,
                        AVG(availability_percent) as daily_availability,
                        AVG(average_response_time) as daily_response_time,
                        SUM(sla_violations) as daily_violations
                    FROM sla_metrics
                    WHERE source_id = ? AND period_start BETWEEN ? AND ?
                    GROUP BY DATE(period_start)
                    ORDER BY date
                """,
                    (source_id, start_date.isoformat(), end_date.isoformat()),
                )
                
                daily_data = cursor.fetchall()
                
                if not daily_data:
                    return {
                        "source_id": source_id,
                        "analysis_period_days": days,
                        "no_data": True,
                    }
                
                # データ準備
                availabilities = [row[1] for row in daily_data if row[1] is not None]
                response_times = [row[2] for row in daily_data if row[2] is not None]
                violations = [row[3] or 0 for row in daily_data]
                
                # トレンド計算
                availability_trend = self._calculate_trend(availabilities)
                response_time_trend = self._calculate_trend(response_times)
                
                return {
                    "source_id": source_id,
                    "analysis_period_days": days,
                    "data_points": len(daily_data),
                    "availability_stats": {
                        "mean": round(np.mean(availabilities), 2),
                        "std": round(np.std(availabilities), 2),
                        "min": round(np.min(availabilities), 2),
                        "max": round(np.max(availabilities), 2),
                        "trend": availability_trend,
                    },
                    "response_time_stats": {
                        "mean": round(np.mean(response_times), 3),
                        "std": round(np.std(response_times), 3),
                        "min": round(np.min(response_times), 3),
                        "max": round(np.max(response_times), 3),
                        "trend": response_time_trend,
                    },
                    "violations_total": sum(violations),
                    "violation_days": len([v for v in violations if v > 0]),
                }
                
        except Exception as e:
            self.logger.error(f"SLAトレンド分析エラー ({source_id}): {e}")
            return {
                "source_id": source_id,
                "analysis_period_days": days,
                "error": str(e),
            }
    
    def _calculate_trend(self, values: List[float]) -> Dict:
        """トレンド計算
        
        Args:
            values: 時系列データ
            
        Returns:
            トレンド情報
        """
        if len(values) < 2:
            return {"trend": "insufficient_data"}
        
        # 線形回帰による傾き計算
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # トレンド判定
        if abs(slope) < np.std(values) * 0.1:  # 標準偏差の10%未満は「安定」
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        return {
            "trend": trend_direction,
            "slope": round(slope, 4),
            "slope_significance": abs(slope) / (np.std(values) + 1e-6),
        }
    
    def get_sla_compliance_report(
        self,
        source_ids: List[str] = None,
        period_days: int = 30
    ) -> Dict:
        """SLAコンプライアンスレポート
        
        Args:
            source_ids: レポート対象のソースIDリスト（Noneの場合は全て）
            period_days: レポート期間（日数）
            
        Returns:
            SLAコンプライアンスレポート
        """
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=period_days)
            
            with sqlite3.connect(self.db_path) as conn:
                # クエリ構築
                where_clause = "WHERE period_start BETWEEN ? AND ?"
                params = [start_date.isoformat(), end_date.isoformat()]
                
                if source_ids:
                    placeholders = ",".join("?" * len(source_ids))
                    where_clause += f" AND source_id IN ({placeholders})"
                    params.extend(source_ids)
                
                cursor = conn.execute(
                    f"""
                    SELECT 
                        source_id,
                        AVG(availability_percent) as avg_availability,
                        AVG(average_response_time) as avg_response_time,
                        SUM(sla_violations) as total_violations,
                        COUNT(*) as measurement_count
                    FROM sla_metrics
                    {where_clause}
                    GROUP BY source_id
                """,
                    params,
                )
                
                results = cursor.fetchall()
                
                compliance_data = []
                for row in results:
                    source_id, avg_avail, avg_resp, violations, count = row
                    
                    compliance_data.append({
                        "source_id": source_id,
                        "average_availability": round(avg_avail, 2),
                        "average_response_time": round(avg_resp, 3),
                        "sla_violations": violations or 0,
                        "measurement_periods": count,
                        "violation_rate": round((violations or 0) / count * 100, 2),
                    })
                
                # 全体統計
                if compliance_data:
                    all_availabilities = [d["average_availability"] for d in compliance_data]
                    all_violations = [d["sla_violations"] for d in compliance_data]
                    
                    summary = {
                        "total_sources": len(compliance_data),
                        "overall_avg_availability": round(np.mean(all_availabilities), 2),
                        "total_violations": sum(all_violations),
                        "sources_with_violations": len([v for v in all_violations if v > 0]),
                    }
                else:
                    summary = {
                        "total_sources": 0,
                        "no_data": True,
                    }
                
                return {
                    "report_period_days": period_days,
                    "report_generated_at": datetime.now(timezone.utc).isoformat(),
                    "summary": summary,
                    "source_details": compliance_data,
                }
                
        except Exception as e:
            self.logger.error(f"SLAコンプライアンスレポートエラー: {e}")
            return {
                "report_period_days": period_days,
                "error": str(e),
                "report_generated_at": datetime.now(timezone.utc).isoformat(),
            }