#!/usr/bin/env python3
"""
整合性チェックモジュール
データの整合性チェック、ベースライン比較、異常検出を担当
"""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from .database import DatabaseManager
from .models import DataSourceConfig, IntegrityCheck, MonitoringLevel


class IntegrityCheckManager:
    """整合性チェック管理クラス"""

    def __init__(self, database_manager: DatabaseManager):
        self.logger = logging.getLogger(__name__)
        self.database_manager = database_manager

    async def perform_integrity_checks(
        self, config: DataSourceConfig, data_info: Optional[Dict[str, Any]] = None
    ) -> List[IntegrityCheck]:
        """整合性チェック実行"""
        checks = []
        current_time = datetime.now(timezone.utc)

        try:
            if not data_info:
                return [
                    IntegrityCheck(
                        source_id=config.source_id,
                        check_type="data_availability",
                        timestamp=current_time,
                        passed=False,
                        issues_found=["データが取得できません"],
                        metadata={"check_status": "failed"},
                    )
                ]

            # 基本チェック実行
            checks.extend(await self._perform_basic_checks(config, data_info, current_time))

            # 高度チェック（レベルに応じて）
            if config.monitoring_level in [
                MonitoringLevel.COMPREHENSIVE,
                MonitoringLevel.CRITICAL,
            ]:
                checks.extend(
                    await self._perform_advanced_checks(config, data_info, current_time)
                )

            return checks

        except Exception as e:
            self.logger.error(f"整合性チェックエラー ({config.source_id}): {e}")
            return [
                IntegrityCheck(
                    source_id=config.source_id,
                    check_type="integrity_error",
                    timestamp=current_time,
                    passed=False,
                    issues_found=[f"整合性チェック実行エラー: {str(e)}"],
                )
            ]

    async def _perform_basic_checks(
        self, config: DataSourceConfig, data_info: Dict[str, Any], current_time: datetime
    ) -> List[IntegrityCheck]:
        """基本チェック実行"""
        checks = []

        # 1. レコード数チェック
        record_count = data_info.get("record_count", 0)
        record_count_check = IntegrityCheck(
            source_id=config.source_id,
            check_type="record_count",
            timestamp=current_time,
            passed=record_count > 0,
            metrics={"record_count": record_count},
        )

        if record_count == 0:
            record_count_check.issues_found.append("レコード数が0です")

        checks.append(record_count_check)

        # 2. データ品質チェック（品質スコア基準）
        quality_score = data_info.get("quality_score")
        if quality_score is not None:
            quality_check = IntegrityCheck(
                source_id=config.source_id,
                check_type="data_quality",
                timestamp=current_time,
                passed=quality_score >= config.quality_threshold,
                metrics={
                    "quality_score": quality_score,
                    "threshold": config.quality_threshold,
                },
            )

            if quality_score < config.quality_threshold:
                quality_check.issues_found.append(
                    f"品質スコア {quality_score:.1f} が閾値 {config.quality_threshold} を下回っています"
                )

            checks.append(quality_check)

        return checks

    async def _perform_advanced_checks(
        self, config: DataSourceConfig, data_info: Dict[str, Any], current_time: datetime
    ) -> List[IntegrityCheck]:
        """高度チェック実行"""
        checks = []

        # ベースライン比較
        baseline_check = await self._perform_baseline_comparison(
            config, data_info, current_time
        )
        if baseline_check:
            checks.append(baseline_check)

        # データ一貫性チェック
        consistency_check = await self._perform_consistency_check(
            config, data_info, current_time
        )
        if consistency_check:
            checks.append(consistency_check)

        # 異常値検出
        if config.monitoring_level == MonitoringLevel.CRITICAL:
            anomaly_check = await self._perform_anomaly_detection(
                config, data_info, current_time
            )
            if anomaly_check:
                checks.append(anomaly_check)

        return checks

    async def _perform_baseline_comparison(
        self, config: DataSourceConfig, data_info: Dict[str, Any], current_time: datetime
    ) -> Optional[IntegrityCheck]:
        """ベースライン比較実行"""
        try:
            # 過去のデータを取得してベースラインと比較
            historical_data = await self._get_historical_data(config.source_id)

            if len(historical_data) < 3:
                return None  # データ不足

            current_count = data_info.get("record_count", 0)
            historical_counts = [row[0] for row in historical_data if row[0] is not None]

            avg_count = np.mean(historical_counts)
            std_count = np.std(historical_counts)

            # 統計的異常判定 (3σ外れ値)
            if std_count > 0:
                z_score = abs(current_count - avg_count) / std_count
                is_anomaly = z_score > 3.0
            else:
                is_anomaly = False

            baseline_check = IntegrityCheck(
                source_id=config.source_id,
                check_type="baseline_comparison",
                timestamp=current_time,
                passed=not is_anomaly,
                metrics={
                    "current_count": current_count,
                    "baseline_avg": avg_count,
                    "baseline_std": std_count,
                    "z_score": z_score if std_count > 0 else 0,
                },
                baseline_comparison={
                    "historical_samples": len(historical_counts),
                    "deviation_percent": (
                        ((current_count - avg_count) / avg_count * 100)
                        if avg_count > 0
                        else 0
                    ),
                },
            )

            if is_anomaly:
                baseline_check.issues_found.append(
                    f"レコード数 {current_count} がベースライン {avg_count:.0f}±{std_count:.0f} から大きく逸脱しています (Z-score: {z_score:.2f})"
                )

            return baseline_check

        except Exception as e:
            self.logger.error(f"ベースライン比較エラー ({config.source_id}): {e}")
            return None

    async def _perform_consistency_check(
        self, config: DataSourceConfig, data_info: Dict[str, Any], current_time: datetime
    ) -> Optional[IntegrityCheck]:
        """データ一貫性チェック"""
        try:
            issues = []
            metrics = {}

            # メタデータの一貫性チェック
            metadata = data_info.get("metadata", {})
            
            # API応答時間チェック（APIソースの場合）
            if "api_response_time" in metadata:
                response_time = metadata["api_response_time"]
                if response_time > 5.0:  # 5秒以上は異常
                    issues.append(f"API応答時間が異常に長い: {response_time:.2f}秒")
                metrics["api_response_time"] = response_time

            # ファイルサイズチェック（ファイルソースの場合）
            if "file_size" in metadata:
                file_size = metadata["file_size"]
                if file_size == 0:
                    issues.append("ファイルサイズが0バイトです")
                metrics["file_size"] = file_size

            # クエリ時間チェック（データベースソースの場合）
            if "query_time" in metadata:
                query_time = metadata["query_time"]
                if query_time > 10.0:  # 10秒以上は異常
                    issues.append(f"クエリ実行時間が異常に長い: {query_time:.2f}秒")
                metrics["query_time"] = query_time

            consistency_check = IntegrityCheck(
                source_id=config.source_id,
                check_type="consistency",
                timestamp=current_time,
                passed=len(issues) == 0,
                issues_found=issues,
                metrics=metrics,
            )

            return consistency_check

        except Exception as e:
            self.logger.error(f"一貫性チェックエラー ({config.source_id}): {e}")
            return None

    async def _perform_anomaly_detection(
        self, config: DataSourceConfig, data_info: Dict[str, Any], current_time: datetime
    ) -> Optional[IntegrityCheck]:
        """異常値検出"""
        try:
            issues = []
            metrics = {}

            # 時系列異常検出
            record_count = data_info.get("record_count", 0)
            historical_data = await self._get_recent_historical_data(config.source_id, 24)  # 24時間

            if len(historical_data) >= 10:
                counts = [row[0] for row in historical_data if row[0] is not None]
                
                # 移動平均との乖離
                if len(counts) >= 5:
                    moving_avg = np.mean(counts[-5:])  # 直近5回の平均
                    deviation_percent = abs((record_count - moving_avg) / moving_avg * 100) if moving_avg > 0 else 0
                    
                    if deviation_percent > 50:  # 50%以上の乖離
                        issues.append(f"レコード数が直近平均から大きく乖離: {deviation_percent:.1f}%")
                    
                    metrics.update({
                        "moving_average": moving_avg,
                        "deviation_percent": deviation_percent,
                    })

                # 急激な増減の検出
                if len(counts) >= 2:
                    prev_count = counts[-1]
                    change_percent = abs((record_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
                    
                    if change_percent > 100:  # 100%以上の変化
                        issues.append(f"前回から急激な変化: {change_percent:.1f}%")
                    
                    metrics["change_percent"] = change_percent

            anomaly_check = IntegrityCheck(
                source_id=config.source_id,
                check_type="anomaly_detection",
                timestamp=current_time,
                passed=len(issues) == 0,
                issues_found=issues,
                metrics=metrics,
            )

            return anomaly_check

        except Exception as e:
            self.logger.error(f"異常検出エラー ({config.source_id}): {e}")
            return None

    async def _get_historical_data(self, source_id: str, hours: int = 24) -> List[tuple]:
        """履歴データ取得"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            with sqlite3.connect(self.database_manager.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT record_count FROM freshness_checks
                    WHERE source_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC LIMIT 10
                """,
                    (source_id, start_time.isoformat()),
                )
                return cursor.fetchall()

        except Exception as e:
            self.logger.error(f"履歴データ取得エラー ({source_id}): {e}")
            return []

    async def _get_recent_historical_data(self, source_id: str, hours: int) -> List[tuple]:
        """直近履歴データ取得"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            with sqlite3.connect(self.database_manager.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT record_count FROM freshness_checks
                    WHERE source_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC LIMIT 50
                """,
                    (source_id, start_time.isoformat()),
                )
                return cursor.fetchall()

        except Exception as e:
            self.logger.error(f"直近履歴データ取得エラー ({source_id}): {e}")
            return []

    def has_integrity_issues(self, checks: List[IntegrityCheck]) -> bool:
        """整合性問題があるかどうかを判定"""
        return any(not check.passed for check in checks)

    def get_failed_checks(self, checks: List[IntegrityCheck]) -> List[IntegrityCheck]:
        """失敗したチェック一覧を取得"""
        return [check for check in checks if not check.passed]

    def get_check_by_type(
        self, checks: List[IntegrityCheck], check_type: str
    ) -> Optional[IntegrityCheck]:
        """指定タイプのチェック結果を取得"""
        for check in checks:
            if check.check_type == check_type:
                return check
        return None