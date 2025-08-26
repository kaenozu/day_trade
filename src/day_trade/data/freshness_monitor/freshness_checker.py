#!/usr/bin/env python3
"""
データ鮮度チェック機能
データソースの鮮度を監視し、チェック結果を返します
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .enums import DataSourceType, FreshnessStatus
from .models import DataSourceConfig, FreshnessCheck

# 依存コンポーネントのインポート
try:
    from ...comprehensive_data_quality_system import ComprehensiveDataQualitySystem
    QUALITY_SYSTEM_AVAILABLE = True
except ImportError:
    QUALITY_SYSTEM_AVAILABLE = False


class FreshnessChecker:
    """データ鮮度チェッカー
    
    各種データソースの鮮度をチェックし、
    設定された閾値に基づいてステータスを判定します。
    """
    
    def __init__(self):
        """鮮度チェッカーを初期化"""
        self.logger = logging.getLogger(__name__)
        self.quality_system = None
        
        # 品質システムの初期化
        if QUALITY_SYSTEM_AVAILABLE:
            try:
                self.quality_system = ComprehensiveDataQualitySystem()
                self.logger.info("品質システム初期化完了")
            except Exception as e:
                self.logger.warning(f"品質システム初期化エラー: {e}")
    
    async def check_freshness(
        self, 
        config: DataSourceConfig
    ) -> FreshnessCheck:
        """鮮度チェック実行
        
        指定されたデータソース設定に基づいて鮮度チェックを実行し、
        結果をFreshnessCheckオブジェクトで返します。
        
        Args:
            config: データソース設定
            
        Returns:
            鮮度チェック結果
        """
        current_time = datetime.now(timezone.utc)
        
        try:
            # データソースから最新データ情報を取得
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
            status = self._determine_freshness_status(age_seconds, config)
            
            # 品質スコア計算（データが利用可能な場合）
            quality_score = await self._calculate_quality_score(
                data_info, config
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
        self, 
        age_seconds: float, 
        config: DataSourceConfig
    ) -> FreshnessStatus:
        """鮮度ステータス判定
        
        Args:
            age_seconds: 経過時間（秒）
            config: データソース設定
            
        Returns:
            鮮度ステータス
        """
        if age_seconds <= config.expected_frequency:
            return FreshnessStatus.FRESH
        elif age_seconds <= config.freshness_threshold:
            return FreshnessStatus.STALE
        else:
            return FreshnessStatus.EXPIRED
    
    async def _get_latest_data_info(
        self, 
        config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """最新データ情報取得
        
        データソースの種別に応じて最新データ情報を取得します。
        実際の実装では、各データソースに対応したクライアントを使用します。
        
        Args:
            config: データソース設定
            
        Returns:
            (最終更新時刻, データ情報)のタプル
        """
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
                return await self._get_generic_data_info(config)
                
        except Exception as e:
            self.logger.error(f"データ情報取得エラー ({config.source_id}): {e}")
            return None, None
    
    async def _get_api_data_info(
        self, 
        config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """API データ情報取得
        
        Args:
            config: データソース設定
            
        Returns:
            (最終更新時刻, データ情報)のタプル
        """
        # モック実装 - 実際の実装では、HTTPクライアントを使用してAPIを呼び出し
        # レスポンスヘッダーやボディから最終更新時刻を取得する
        return datetime.now(timezone.utc) - timedelta(seconds=30), {
            "record_count": 100,
            "metadata": {
                "api_response_time": 0.15,
                "endpoint": config.endpoint_url,
                "method": "GET"
            },
        }
    
    async def _get_database_data_info(
        self, 
        config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """データベース データ情報取得
        
        Args:
            config: データソース設定
            
        Returns:
            (最終更新時刻, データ情報)のタプル
        """
        # モック実装 - 実際の実装では、データベースクライアントを使用して
        # MAX(updated_at)などで最新更新時刻を取得する
        return datetime.now(timezone.utc) - timedelta(seconds=60), {
            "record_count": 1500,
            "metadata": {
                "query_time": 0.25,
                "connection_params": config.connection_params,
                "table_info": "main_data_table"
            },
        }
    
    async def _get_file_data_info(
        self, 
        config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """ファイル データ情報取得
        
        Args:
            config: データソース設定
            
        Returns:
            (最終更新時刻, データ情報)のタプル
        """
        # モック実装 - 実際の実装では、ファイルシステムAPIを使用して
        # ファイルの更新時刻とサイズを取得する
        return datetime.now(timezone.utc) - timedelta(seconds=120), {
            "record_count": 500,
            "metadata": {
                "file_size": 1024000,
                "file_path": config.connection_params.get("file_path", "unknown"),
                "encoding": "utf-8"
            },
        }
    
    async def _get_stream_data_info(
        self, 
        config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """ストリーム データ情報取得
        
        Args:
            config: データソース設定
            
        Returns:
            (最終更新時刻, データ情報)のタプル
        """
        # モック実装 - 実際の実装では、ストリーミングサービスから
        # 最新メッセージのタイムスタンプを取得する
        return datetime.now(timezone.utc) - timedelta(seconds=10), {
            "record_count": 25,
            "metadata": {
                "stream_lag": 10,
                "partition_info": "partition-0",
                "offset": 12345
            },
        }
    
    async def _get_external_feed_data_info(
        self, 
        config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """外部フィード データ情報取得
        
        Args:
            config: データソース設定
            
        Returns:
            (最終更新時刻, データ情報)のタプル
        """
        # モック実装 - 実際の実装では、外部データプロバイダーのAPIを使用して
        # フィードの最新更新時刻を取得する
        return datetime.now(timezone.utc) - timedelta(seconds=45), {
            "record_count": 200,
            "metadata": {
                "feed_provider": "external_provider",
                "feed_type": "market_data",
                "update_frequency": "real_time"
            },
        }
    
    async def _get_generic_data_info(
        self, 
        config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """汎用 データ情報取得
        
        Args:
            config: データソース設定
            
        Returns:
            (最終更新時刻, データ情報)のタプル
        """
        # 汎用的なフォールバック実装
        return datetime.now(timezone.utc) - timedelta(seconds=90), {
            "record_count": 150,
            "metadata": {
                "source_type": config.source_type.value,
                "check_type": "generic"
            },
        }
    
    async def _calculate_quality_score(
        self, 
        data_info: Optional[Dict[str, Any]], 
        config: DataSourceConfig
    ) -> Optional[float]:
        """品質スコア計算
        
        Args:
            data_info: データ情報
            config: データソース設定
            
        Returns:
            品質スコア（0-100）、計算できない場合はNone
        """
        if not data_info or not self.quality_system:
            return None
        
        try:
            # データが DataFrame として利用可能な場合の品質チェック
            if isinstance(data_info.get("data"), pd.DataFrame):
                quality_report = await self.quality_system.process_dataset(
                    data_info["data"], config.source_id
                )
                return quality_report.overall_score
            
            # データフレームが利用できない場合は、基本的な品質指標を計算
            return self._calculate_basic_quality_score(data_info, config)
            
        except Exception as e:
            self.logger.warning(
                f"品質スコア計算エラー ({config.source_id}): {e}"
            )
            return None
    
    def _calculate_basic_quality_score(
        self, 
        data_info: Dict[str, Any], 
        config: DataSourceConfig
    ) -> float:
        """基本品質スコア計算
        
        データフレームが利用できない場合の基本的な品質評価
        
        Args:
            data_info: データ情報
            config: データソース設定
            
        Returns:
            基本品質スコア（0-100）
        """
        score = 100.0
        
        # レコード数チェック
        record_count = data_info.get("record_count", 0)
        if record_count == 0:
            score -= 50.0  # レコードが0の場合は大幅減点
        elif record_count < 10:
            score -= 20.0  # レコード数が少ない場合は減点
        
        # メタデータの情報量チェック
        metadata = data_info.get("metadata", {})
        if len(metadata) < 2:
            score -= 10.0  # メタデータが不足している場合は減点
        
        # API応答時間チェック（APIの場合）
        if config.source_type == DataSourceType.API:
            response_time = metadata.get("api_response_time", 0)
            if response_time > 1.0:  # 1秒以上の場合は減点
                score -= min(20.0, response_time * 10)
        
        # クエリ時間チェック（データベースの場合）
        if config.source_type == DataSourceType.DATABASE:
            query_time = metadata.get("query_time", 0)
            if query_time > 0.5:  # 0.5秒以上の場合は減点
                score -= min(15.0, query_time * 20)
        
        # ストリーム遅延チェック（ストリームの場合）
        if config.source_type == DataSourceType.STREAM:
            stream_lag = metadata.get("stream_lag", 0)
            if stream_lag > config.expected_frequency:
                score -= min(30.0, stream_lag / config.expected_frequency * 10)
        
        return max(0.0, min(100.0, score))

    # 統合版メソッド - advanced_data_freshness_monitor.pyとの互換性
    async def perform_freshness_check(
        self, 
        config: DataSourceConfig,
        quality_system: Optional[Any] = None
    ) -> FreshnessCheck:
        """鮮度チェック実行（統合版）
        
        元のadvanced_data_freshness_monitor.pyの_perform_freshness_checkメソッドと
        互換性を持つ統合版実装です。
        
        Args:
            config: データソース設定
            quality_system: 品質システム（オプション）
            
        Returns:
            鮮度チェック結果
        """
        # 外部品質システムが渡された場合は一時的に使用
        original_quality_system = self.quality_system
        if quality_system:
            self.quality_system = quality_system
        
        try:
            result = await self.check_freshness(config)
            return result
        finally:
            # 元の品質システムを復元
            self.quality_system = original_quality_system