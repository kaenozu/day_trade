#!/usr/bin/env python3
"""
鮮度チェックモジュール
データソースの鮮度チェックとデータ情報取得を担当
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd

try:
    from ...utils.unified_cache_manager import UnifiedCacheManager
    from ..comprehensive_data_quality_system import ComprehensiveDataQualitySystem

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

from .models import DataSourceConfig, DataSourceType, FreshnessCheck, FreshnessStatus


class FreshnessCheckManager:
    """鮮度チェック管理クラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_system = None
        self.cache_manager = None

        if DEPENDENCIES_AVAILABLE:
            try:
                self.quality_system = ComprehensiveDataQualitySystem()
                self.cache_manager = UnifiedCacheManager()
            except Exception as e:
                self.logger.warning(f"品質システム初期化エラー: {e}")

    async def perform_freshness_check(
        self, config: DataSourceConfig
    ) -> FreshnessCheck:
        """鮮度チェック実行"""
        current_time = datetime.now(timezone.utc)

        try:
            # データソースから最新データ取得
            last_update, data_info = await self._get_latest_data_info(config)

            if last_update is None:
                # データ取得失敗
                return FreshnessCheck(
                    source_id=config.source_id,
                    timestamp=current_time,
                    last_update=current_time,
                    age_seconds=float("inf"),
                    status=FreshnessStatus.EXPIRED,
                    metadata={"error": "データ取得失敗"},
                )

            # 経過時間計算
            age_seconds = (current_time - last_update).total_seconds()

            # ステータス判定
            status = self._determine_freshness_status(
                age_seconds, config.expected_frequency, config.freshness_threshold
            )

            # 品質スコア計算（データが利用可能な場合）
            quality_score = await self._calculate_quality_score(
                config, data_info
            )

            return FreshnessCheck(
                source_id=config.source_id,
                timestamp=current_time,
                last_update=last_update,
                age_seconds=age_seconds,
                status=status,
                quality_score=quality_score,
                record_count=data_info.get("record_count") if data_info else None,
                metadata=data_info.get("metadata", {}) if data_info else {},
            )

        except Exception as e:
            self.logger.error(f"鮮度チェックエラー ({config.source_id}): {e}")
            return FreshnessCheck(
                source_id=config.source_id,
                timestamp=current_time,
                last_update=current_time,
                age_seconds=float("inf"),
                status=FreshnessStatus.EXPIRED,
                metadata={"error": str(e)},
            )

    def _determine_freshness_status(
        self, age_seconds: float, expected_frequency: int, freshness_threshold: int
    ) -> FreshnessStatus:
        """鮮度ステータス判定"""
        if age_seconds <= expected_frequency:
            return FreshnessStatus.FRESH
        elif age_seconds <= freshness_threshold:
            return FreshnessStatus.STALE
        else:
            return FreshnessStatus.EXPIRED

    async def _calculate_quality_score(
        self, config: DataSourceConfig, data_info: Optional[Dict[str, Any]]
    ) -> Optional[float]:
        """品質スコア計算"""
        if not data_info or not self.quality_system:
            return None

        try:
            if isinstance(data_info.get("data"), pd.DataFrame):
                quality_report = await self.quality_system.process_dataset(
                    data_info["data"], config.source_id
                )
                return quality_report.overall_score
        except Exception as e:
            self.logger.warning(f"品質スコア計算エラー ({config.source_id}): {e}")

        return None

    async def _get_latest_data_info(
        self, config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """最新データ情報取得"""
        try:
            if config.source_type == DataSourceType.API:
                return await self._get_api_data_info(config)
            elif config.source_type == DataSourceType.DATABASE:
                return await self._get_database_data_info(config)
            elif config.source_type == DataSourceType.FILE:
                return await self._get_file_data_info(config)
            elif config.source_type == DataSourceType.STREAM:
                return await self._get_stream_data_info(config)
            elif config.source_type == DataSourceType.EXTERNAL_FEED:
                return await self._get_external_feed_data_info(config)
            else:
                # 不明なソースタイプ
                return await self._get_generic_data_info(config)

        except Exception as e:
            self.logger.error(f"データ情報取得エラー ({config.source_id}): {e}")
            return None, None

    async def _get_api_data_info(
        self, config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """API データ情報取得（モック実装）"""
        # 実際の実装では、APIエンドポイントを呼び出してデータを取得
        # ここではモック応答を返す
        return datetime.now(timezone.utc) - timedelta(seconds=30), {
            "record_count": 100,
            "metadata": {"api_response_time": 0.15, "endpoint": config.endpoint_url},
        }

    async def _get_database_data_info(
        self, config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """データベース データ情報取得（モック実装）"""
        # 実際の実装では、データベースに接続してクエリを実行
        # ここではモック応答を返す
        return datetime.now(timezone.utc) - timedelta(seconds=60), {
            "record_count": 1500,
            "metadata": {"query_time": 0.25, "connection_params": config.connection_params},
        }

    async def _get_file_data_info(
        self, config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """ファイル データ情報取得（モック実装）"""
        # 実際の実装では、ファイルシステムをチェック
        # ここではモック応答を返す
        return datetime.now(timezone.utc) - timedelta(seconds=120), {
            "record_count": 500,
            "metadata": {"file_size": 1024000, "file_path": config.endpoint_url},
        }

    async def _get_stream_data_info(
        self, config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """ストリーム データ情報取得（モック実装）"""
        # 実際の実装では、ストリームの最新メッセージを確認
        # ここではモック応答を返す
        return datetime.now(timezone.utc) - timedelta(seconds=5), {
            "record_count": 50,
            "metadata": {"stream_lag": 0.05, "stream_endpoint": config.endpoint_url},
        }

    async def _get_external_feed_data_info(
        self, config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """外部フィード データ情報取得（モック実装）"""
        # 実際の実装では、外部フィードのステータスを確認
        # ここではモック応答を返す
        return datetime.now(timezone.utc) - timedelta(seconds=15), {
            "record_count": 75,
            "metadata": {"feed_status": "active", "feed_url": config.endpoint_url},
        }

    async def _get_generic_data_info(
        self, config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """汎用 データ情報取得（モック実装）"""
        # 汎用的なモック応答
        return datetime.now(timezone.utc) - timedelta(seconds=45), {
            "record_count": 200,
            "metadata": {"source_type": config.source_type.value},
        }

    def is_data_fresh(self, check: FreshnessCheck) -> bool:
        """データが新鮮かどうかを判定"""
        return check.status == FreshnessStatus.FRESH

    def is_data_stale(self, check: FreshnessCheck) -> bool:
        """データが古いかどうかを判定"""
        return check.status == FreshnessStatus.STALE

    def is_data_expired(self, check: FreshnessCheck) -> bool:
        """データが期限切れかどうかを判定"""
        return check.status == FreshnessStatus.EXPIRED

    def get_quality_score(self, check: FreshnessCheck) -> Optional[float]:
        """品質スコア取得"""
        return check.quality_score

    def has_sufficient_quality(
        self, check: FreshnessCheck, threshold: float
    ) -> bool:
        """品質が十分かどうかを判定"""
        if check.quality_score is None:
            return True  # スコア不明の場合は合格とみなす
        return check.quality_score >= threshold