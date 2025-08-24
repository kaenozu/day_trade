#!/usr/bin/env python3
"""
データ整合性チェック機能
データの整合性検証、ベースライン比較、統計的異常検出を実行します
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from .database_operations import DatabaseOperations
from .enums import MonitoringLevel
from .models import DataSourceConfig, IntegrityCheck


class IntegrityChecker:
    """データ整合性チェッカー
    
    データの品質、レコード数の妥当性、ベースライン比較、
    統計的異常検出などの整合性チェックを実行します。
    """
    
    def __init__(self, db_operations: DatabaseOperations):
        """整合性チェッカーを初期化
        
        Args:
            db_operations: データベース操作インスタンス
        """
        self.logger = logging.getLogger(__name__)
        self.db_operations = db_operations
    
    async def check_integrity(
        self, 
        config: DataSourceConfig
    ) -> List[IntegrityCheck]:
        """整合性チェック実行
        
        監視レベルに応じて適切な整合性チェックを実行し、
        結果をIntegrityCheckオブジェクトのリストで返します。
        
        Args:
            config: データソース設定
            
        Returns:
            整合性チェック結果のリスト
        """
        checks = []
        current_time = datetime.now(timezone.utc)
        
        try:
            # データ取得（鮮度チェッカーと同じロジックを使用）
            from .freshness_checker import FreshnessChecker
            freshness_checker = FreshnessChecker()
            _, data_info = await freshness_checker._get_latest_data_info(config)
            
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
            
            # 基本チェック（全監視レベル対象）
            checks.extend(await self._perform_basic_checks(config, data_info, current_time))
            
            # 詳細チェック（COMPREHENSIVE, CRITICAL レベル対象）
            if config.monitoring_level in [MonitoringLevel.COMPREHENSIVE, MonitoringLevel.CRITICAL]:
                checks.extend(await self._perform_detailed_checks(config, data_info, current_time))
            
            # 高度チェック（CRITICAL レベルのみ対象）
            if config.monitoring_level == MonitoringLevel.CRITICAL:
                checks.extend(await self._perform_advanced_checks(config, data_info, current_time))
            
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
                    metadata={"error_details": str(e)},
                )
            ]
    
    async def _perform_basic_checks(
        self, 
        config: DataSourceConfig, 
        data_info: Dict[str, Any], 
        current_time: datetime
    ) -> List[IntegrityCheck]:
        """基本整合性チェック実行
        
        Args:
            config: データソース設定
            data_info: データ情報
            current_time: 現在時刻
            
        Returns:
            基本チェック結果のリスト
        """
        checks = []
        
        # 1. レコード数チェック
        record_count_check = self._check_record_count(
            config, data_info, current_time
        )
        checks.append(record_count_check)
        
        # 2. データ品質スコアチェック
        quality_check = self._check_quality_score(
            config, data_info, current_time
        )
        if quality_check:
            checks.append(quality_check)
        
        return checks
    
    async def _perform_detailed_checks(
        self, 
        config: DataSourceConfig, 
        data_info: Dict[str, Any], 
        current_time: datetime
    ) -> List[IntegrityCheck]:
        """詳細整合性チェック実行
        
        Args:
            config: データソース設定
            data_info: データ情報
            current_time: 現在時刻
            
        Returns:
            詳細チェック結果のリスト
        """
        checks = []
        
        # 3. ベースライン比較チェック
        baseline_check = await self._perform_baseline_comparison(
            config, data_info, current_time
        )
        if baseline_check:
            checks.append(baseline_check)
        
        # 4. メタデータ整合性チェック
        metadata_check = self._check_metadata_integrity(
            config, data_info, current_time
        )
        if metadata_check:
            checks.append(metadata_check)
        
        return checks
    
    async def _perform_advanced_checks(
        self, 
        config: DataSourceConfig, 
        data_info: Dict[str, Any], 
        current_time: datetime
    ) -> List[IntegrityCheck]:
        """高度整合性チェック実行
        
        Args:
            config: データソース設定
            data_info: データ情報
            current_time: 現在時刻
            
        Returns:
            高度チェック結果のリスト
        """
        checks = []
        
        # 5. 統計的異常検出
        anomaly_check = await self._perform_anomaly_detection(
            config, data_info, current_time
        )
        if anomaly_check:
            checks.append(anomaly_check)
        
        # 6. トレンド分析
        trend_check = await self._perform_trend_analysis(
            config, data_info, current_time
        )
        if trend_check:
            checks.append(trend_check)
        
        return checks
    
    def _check_record_count(
        self, 
        config: DataSourceConfig, 
        data_info: Dict[str, Any], 
        current_time: datetime
    ) -> IntegrityCheck:
        """レコード数チェック
        
        Args:
            config: データソース設定
            data_info: データ情報
            current_time: 現在時刻
            
        Returns:
            レコード数チェック結果
        """
        record_count = data_info.get("record_count", 0)
        
        check = IntegrityCheck(
            source_id=config.source_id,
            check_type="record_count",
            timestamp=current_time,
            passed=record_count > 0,
            metrics={"record_count": record_count},
        )
        
        if record_count == 0:
            check.issues_found.append("レコード数が0です")
        elif record_count < 10:  # 最小レコード数の閾値
            check.issues_found.append(f"レコード数が少なすぎます: {record_count}")
            check.passed = False
        
        return check
    
    def _check_quality_score(
        self, 
        config: DataSourceConfig, 
        data_info: Dict[str, Any], 
        current_time: datetime
    ) -> Optional[IntegrityCheck]:
        """品質スコアチェック
        
        Args:
            config: データソース設定
            data_info: データ情報
            current_time: 現在時刻
            
        Returns:
            品質スコアチェック結果、該当しない場合はNone
        """
        quality_score = data_info.get("quality_score")
        if quality_score is None:
            return None
        
        check = IntegrityCheck(
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
            check.issues_found.append(
                f"品質スコア {quality_score:.1f} が閾値 {config.quality_threshold} を下回っています"
            )
        
        return check
    
    async def _perform_baseline_comparison(
        self, 
        config: DataSourceConfig, 
        data_info: Dict[str, Any], 
        current_time: datetime
    ) -> Optional[IntegrityCheck]:
        """ベースライン比較チェック
        
        過去のデータとの統計的比較を行います。
        
        Args:
            config: データソース設定
            data_info: データ情報
            current_time: 現在時刻
            
        Returns:
            ベースライン比較結果、データ不足の場合はNone
        """
        try:
            # 過去24時間のレコード数履歴を取得
            historical_counts = self.db_operations.get_historical_record_counts(
                config.source_id, hours=24, limit=10
            )
            
            if len(historical_counts) < 3:
                return None  # データ不足
            
            current_count = data_info.get("record_count", 0)
            avg_count = np.mean(historical_counts)
            std_count = np.std(historical_counts)
            
            # 統計的異常判定 (3σ外れ値)
            if std_count > 0:
                z_score = abs(current_count - avg_count) / std_count
                is_anomaly = z_score > 3.0
            else:
                is_anomaly = abs(current_count - avg_count) > avg_count * 0.5  # 50%以上の偏差
            
            check = IntegrityCheck(
                source_id=config.source_id,
                check_type="baseline_comparison",
                timestamp=current_time,
                passed=not is_anomaly,
                metrics={
                    "current_count": current_count,
                    "baseline_avg": float(avg_count),
                    "baseline_std": float(std_count),
                    "z_score": float(z_score) if std_count > 0 else 0.0,
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
                check.issues_found.append(
                    f"レコード数 {current_count} がベースライン {avg_count:.0f}±{std_count:.0f} から大きく逸脱しています"
                )
                if std_count > 0:
                    check.issues_found.append(f"Z-score: {z_score:.2f}")
            
            return check
            
        except Exception as e:
            self.logger.error(f"ベースライン比較エラー ({config.source_id}): {e}")
            return None
    
    def _check_metadata_integrity(
        self, 
        config: DataSourceConfig, 
        data_info: Dict[str, Any], 
        current_time: datetime
    ) -> Optional[IntegrityCheck]:
        """メタデータ整合性チェック
        
        Args:
            config: データソース設定
            data_info: データ情報
            current_time: 現在時刻
            
        Returns:
            メタデータ整合性チェック結果
        """
        metadata = data_info.get("metadata", {})
        
        check = IntegrityCheck(
            source_id=config.source_id,
            check_type="metadata_integrity",
            timestamp=current_time,
            passed=True,
            metrics={"metadata_fields": len(metadata)},
        )
        
        # 必須メタデータフィールドの確認
        required_fields = self._get_required_metadata_fields(config.source_type)
        missing_fields = [field for field in required_fields if field not in metadata]
        
        if missing_fields:
            check.passed = False
            check.issues_found.extend([
                f"必須メタデータフィールド不足: {', '.join(missing_fields)}"
            ])
        
        # データ型チェック
        type_violations = self._validate_metadata_types(metadata, config.source_type)
        if type_violations:
            check.passed = False
            check.issues_found.extend(type_violations)
        
        return check if not check.passed or missing_fields else None
    
    def _get_required_metadata_fields(self, source_type) -> List[str]:
        """データソース種別ごとの必須メタデータフィールドを取得
        
        Args:
            source_type: データソース種別
            
        Returns:
            必須フィールドのリスト
        """
        from .enums import DataSourceType
        
        field_map = {
            DataSourceType.API: ["api_response_time", "endpoint"],
            DataSourceType.DATABASE: ["query_time", "connection_params"],
            DataSourceType.FILE: ["file_size", "file_path"],
            DataSourceType.STREAM: ["stream_lag", "partition_info"],
            DataSourceType.EXTERNAL_FEED: ["feed_provider", "feed_type"],
        }
        
        return field_map.get(source_type, [])
    
    def _validate_metadata_types(self, metadata: Dict[str, Any], source_type) -> List[str]:
        """メタデータの型チェック
        
        Args:
            metadata: メタデータ
            source_type: データソース種別
            
        Returns:
            型違反のリスト
        """
        violations = []
        
        # 数値型であるべきフィールド
        numeric_fields = {
            "api_response_time", "query_time", "file_size", 
            "stream_lag", "offset"
        }
        
        for field in numeric_fields:
            if field in metadata and not isinstance(metadata[field], (int, float)):
                violations.append(f"{field} は数値である必要があります")
        
        # 文字列型であるべきフィールド
        string_fields = {
            "endpoint", "file_path", "partition_info", 
            "feed_provider", "feed_type"
        }
        
        for field in string_fields:
            if field in metadata and not isinstance(metadata[field], str):
                violations.append(f"{field} は文字列である必要があります")
        
        return violations
    
    async def _perform_anomaly_detection(
        self, 
        config: DataSourceConfig, 
        data_info: Dict[str, Any], 
        current_time: datetime
    ) -> Optional[IntegrityCheck]:
        """統計的異常検出チェック
        
        Args:
            config: データソース設定
            data_info: データ情報
            current_time: 現在時刻
            
        Returns:
            異常検出結果
        """
        try:
            # より長期間の履歴データを取得（1週間分）
            historical_counts = self.db_operations.get_historical_record_counts(
                config.source_id, hours=168, limit=50  # 1週間 = 168時間
            )
            
            if len(historical_counts) < 20:  # 最低20サンプル必要
                return None
            
            current_count = data_info.get("record_count", 0)
            
            # IQR（四分位範囲）による異常検出
            q1 = np.percentile(historical_counts, 25)
            q3 = np.percentile(historical_counts, 75)
            iqr = q3 - q1
            
            # 異常値の閾値（IQR * 1.5）
            lower_threshold = q1 - (iqr * 1.5)
            upper_threshold = q3 + (iqr * 1.5)
            
            is_outlier = current_count < lower_threshold or current_count > upper_threshold
            
            check = IntegrityCheck(
                source_id=config.source_id,
                check_type="anomaly_detection",
                timestamp=current_time,
                passed=not is_outlier,
                metrics={
                    "current_count": current_count,
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower_threshold": float(lower_threshold),
                    "upper_threshold": float(upper_threshold),
                },
            )
            
            if is_outlier:
                outlier_type = "下限" if current_count < lower_threshold else "上限"
                check.issues_found.append(
                    f"統計的異常値検出: {current_count} は{outlier_type}閾値を超えています "
                    f"(範囲: {lower_threshold:.0f} - {upper_threshold:.0f})"
                )
            
            return check
            
        except Exception as e:
            self.logger.error(f"異常検出エラー ({config.source_id}): {e}")
            return None
    
    async def _perform_trend_analysis(
        self, 
        config: DataSourceConfig, 
        data_info: Dict[str, Any], 
        current_time: datetime
    ) -> Optional[IntegrityCheck]:
        """トレンド分析チェック
        
        Args:
            config: データソース設定
            data_info: データ情報
            current_time: 現在時刻
            
        Returns:
            トレンド分析結果
        """
        try:
            # 過去24時間の詳細履歴を取得
            historical_counts = self.db_operations.get_historical_record_counts(
                config.source_id, hours=24, limit=24
            )
            
            if len(historical_counts) < 10:
                return None
            
            # 線形回帰によるトレンド分析
            x = np.arange(len(historical_counts))
            y = np.array(historical_counts)
            
            # 最小二乗法で傾きを計算
            slope = np.polyfit(x, y, 1)[0]
            
            # 変動係数（CV）を計算
            cv = np.std(y) / np.mean(y) if np.mean(y) > 0 else float('inf')
            
            check = IntegrityCheck(
                source_id=config.source_id,
                check_type="trend_analysis",
                timestamp=current_time,
                passed=True,
                metrics={
                    "trend_slope": float(slope),
                    "coefficient_of_variation": float(cv),
                    "data_points": len(historical_counts),
                    "mean_value": float(np.mean(y)),
                },
            )
            
            # トレンドの警告判定
            if abs(slope) > np.mean(y) * 0.1:  # 平均値の10%以上の変化
                trend_direction = "増加" if slope > 0 else "減少"
                check.issues_found.append(
                    f"急激な{trend_direction}トレンドを検出: 傾き {slope:.2f}"
                )
                check.passed = False
            
            # 変動の大きさの警告
            if cv > 0.5:  # 変動係数が50%以上
                check.issues_found.append(
                    f"データの変動が大きすぎます: CV {cv:.2f}"
                )
                check.passed = False
            
            return check if not check.passed else None
            
        except Exception as e:
            self.logger.error(f"トレンド分析エラー ({config.source_id}): {e}")
            return None