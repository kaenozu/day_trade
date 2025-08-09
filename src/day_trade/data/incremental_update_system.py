#!/usr/bin/env python3
"""
増分更新システム
Issue #317: 高速データ管理システム - Phase 3

差分データのみの効率取得・リアルタイムストリーミングシステム
- 差分データ検出・取得
- リアルタイムストリーミング
- 変更データキャプチャ（CDC）
- 重複データ排除
"""

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class ChangeType(Enum):
    """変更タイプ"""

    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    UPSERT = "upsert"


class DataSource(Enum):
    """データソース"""

    DATABASE = "database"
    API = "api"
    FILE = "file"
    STREAM = "stream"


@dataclass
class ChangeRecord:
    """変更レコード"""

    change_id: str
    change_type: ChangeType
    table_name: str
    primary_key: str
    old_data: Optional[Dict[str, Any]]
    new_data: Optional[Dict[str, Any]]
    timestamp: datetime
    checksum: str
    data_source: DataSource
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IncrementalConfig:
    """増分更新設定"""

    # CDC設定
    enable_cdc: bool = True
    cdc_poll_interval_seconds: int = 1
    batch_size: int = 1000
    max_concurrent_streams: int = 10

    # 重複排除設定
    enable_deduplication: bool = True
    dedup_window_minutes: int = 60
    dedup_key_fields: List[str] = field(default_factory=lambda: ["timestamp", "symbol"])

    # ストリーミング設定
    stream_buffer_size: int = 10000
    stream_timeout_seconds: int = 30
    enable_backpressure_control: bool = True

    # データ整合性設定
    enable_checksum_validation: bool = True
    enable_transaction_log: bool = True
    checkpoint_interval_minutes: int = 5


@dataclass
class StreamingMetrics:
    """ストリーミングメトリクス"""

    total_records_processed: int = 0
    records_per_second: float = 0.0
    duplicates_filtered: int = 0
    errors_count: int = 0
    last_checkpoint: Optional[datetime] = None
    backpressure_events: int = 0
    lag_seconds: float = 0.0


class IncrementalUpdateSystem:
    """増分更新システム"""

    def __init__(self, config: Optional[IncrementalConfig] = None):
        self.config = config or IncrementalConfig()
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_streams
        )

        # 状態管理
        self.change_log: List[ChangeRecord] = []
        self.deduplication_cache: Dict[str, datetime] = {}
        self.streaming_metrics = StreamingMetrics()
        self.last_sync_timestamps: Dict[str, datetime] = {}
        self.active_streams: Dict[str, asyncio.Task] = {}

        # バッファ管理
        self.stream_buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._is_running = False

    async def start_system(self) -> None:
        """システム開始"""
        self._is_running = True

        # CDC監視開始
        if self.config.enable_cdc:
            await self._start_cdc_monitoring()

        # チェックポイント管理開始
        if self.config.enable_transaction_log:
            await self._start_checkpoint_management()

        logger.info("増分更新システム開始")

    async def stop_system(self) -> None:
        """システム停止"""
        self._is_running = False

        # アクティブストリーム停止
        for stream_id, task in self.active_streams.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.active_streams.clear()

        logger.info("増分更新システム停止")

    async def detect_changes(
        self,
        data_source: DataSource,
        table_name: str,
        current_data: pd.DataFrame,
        previous_data: Optional[pd.DataFrame] = None,
        primary_key_column: str = "id",
    ) -> List[ChangeRecord]:
        """データ変更検出"""
        changes = []

        try:
            if previous_data is None:
                # 初回取得：全てINSERT
                for idx, row in current_data.iterrows():
                    change = ChangeRecord(
                        change_id=self._generate_change_id(
                            table_name, row[primary_key_column]
                        ),
                        change_type=ChangeType.INSERT,
                        table_name=table_name,
                        primary_key=str(row[primary_key_column]),
                        old_data=None,
                        new_data=row.to_dict(),
                        timestamp=datetime.now(),
                        checksum=self._calculate_record_checksum(row.to_dict()),
                        data_source=data_source,
                    )
                    changes.append(change)

            else:
                # 差分検出
                current_keys = set(current_data[primary_key_column].values)
                previous_keys = set(previous_data[primary_key_column].values)

                # 新規データ（INSERT）
                new_keys = current_keys - previous_keys
                for key in new_keys:
                    row = current_data[current_data[primary_key_column] == key].iloc[0]
                    change = ChangeRecord(
                        change_id=self._generate_change_id(table_name, key),
                        change_type=ChangeType.INSERT,
                        table_name=table_name,
                        primary_key=str(key),
                        old_data=None,
                        new_data=row.to_dict(),
                        timestamp=datetime.now(),
                        checksum=self._calculate_record_checksum(row.to_dict()),
                        data_source=data_source,
                    )
                    changes.append(change)

                # 削除データ（DELETE）
                deleted_keys = previous_keys - current_keys
                for key in deleted_keys:
                    old_row = previous_data[
                        previous_data[primary_key_column] == key
                    ].iloc[0]
                    change = ChangeRecord(
                        change_id=self._generate_change_id(table_name, key),
                        change_type=ChangeType.DELETE,
                        table_name=table_name,
                        primary_key=str(key),
                        old_data=old_row.to_dict(),
                        new_data=None,
                        timestamp=datetime.now(),
                        checksum=self._calculate_record_checksum(old_row.to_dict()),
                        data_source=data_source,
                    )
                    changes.append(change)

                # 更新データ（UPDATE）
                common_keys = current_keys & previous_keys
                for key in common_keys:
                    current_row = current_data[
                        current_data[primary_key_column] == key
                    ].iloc[0]
                    previous_row = previous_data[
                        previous_data[primary_key_column] == key
                    ].iloc[0]

                    # レコード比較
                    current_checksum = self._calculate_record_checksum(
                        current_row.to_dict()
                    )
                    previous_checksum = self._calculate_record_checksum(
                        previous_row.to_dict()
                    )

                    if current_checksum != previous_checksum:
                        change = ChangeRecord(
                            change_id=self._generate_change_id(table_name, key),
                            change_type=ChangeType.UPDATE,
                            table_name=table_name,
                            primary_key=str(key),
                            old_data=previous_row.to_dict(),
                            new_data=current_row.to_dict(),
                            timestamp=datetime.now(),
                            checksum=current_checksum,
                            data_source=data_source,
                        )
                        changes.append(change)

            # 重複排除
            if self.config.enable_deduplication:
                changes = await self._deduplicate_changes(changes)

            # 変更ログ記録
            self.change_log.extend(changes)

            logger.info(f"変更検出完了: {table_name}, {len(changes)}件の変更")
            return changes

        except Exception as e:
            logger.error(f"変更検出エラー: {table_name}, {e}")
            return []

    async def _deduplicate_changes(
        self, changes: List[ChangeRecord]
    ) -> List[ChangeRecord]:
        """重複排除"""
        deduplicated = []
        current_time = datetime.now()

        # 古いキャッシュエントリクリーンアップ
        cutoff_time = current_time - timedelta(minutes=self.config.dedup_window_minutes)
        self.deduplication_cache = {
            k: v for k, v in self.deduplication_cache.items() if v > cutoff_time
        }

        for change in changes:
            # 重複キー生成
            dedup_key = self._generate_dedup_key(change)

            if dedup_key not in self.deduplication_cache:
                # 新規変更
                self.deduplication_cache[dedup_key] = current_time
                deduplicated.append(change)
            else:
                # 重複検出
                self.streaming_metrics.duplicates_filtered += 1
                logger.debug(f"重複データ除外: {dedup_key}")

        return deduplicated

    def _generate_dedup_key(self, change: ChangeRecord) -> str:
        """重複排除キー生成"""
        key_parts = [change.table_name, change.primary_key, change.change_type.value]

        # 設定されたフィールドを追加
        if change.new_data:
            for field in self.config.dedup_key_fields:
                if field in change.new_data:
                    key_parts.append(str(change.new_data[field]))

        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    async def start_real_time_stream(
        self,
        stream_id: str,
        data_source: DataSource,
        callback: callable,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """リアルタイムストリーム開始"""
        if stream_id in self.active_streams:
            logger.warning(f"ストリーム既に実行中: {stream_id}")
            return False

        try:
            # ストリームタスク作成
            stream_task = asyncio.create_task(
                self._run_real_time_stream(
                    stream_id, data_source, callback, filter_conditions
                )
            )

            self.active_streams[stream_id] = stream_task
            self.stream_buffers[stream_id] = []

            logger.info(f"リアルタイムストリーム開始: {stream_id}")
            return True

        except Exception as e:
            logger.error(f"ストリーム開始エラー: {stream_id}, {e}")
            return False

    async def _run_real_time_stream(
        self,
        stream_id: str,
        data_source: DataSource,
        callback: callable,
        filter_conditions: Optional[Dict[str, Any]],
    ) -> None:
        """リアルタイムストリーム実行"""
        buffer = self.stream_buffers[stream_id]

        try:
            while self._is_running and stream_id in self.active_streams:
                try:
                    # データポーリング
                    new_data = await self._poll_data_source(
                        data_source, filter_conditions
                    )

                    if new_data:
                        # バッファ管理
                        buffer.extend(new_data)

                        # バックプレッシャー制御
                        if (
                            self.config.enable_backpressure_control
                            and len(buffer) > self.config.stream_buffer_size
                        ):
                            self.streaming_metrics.backpressure_events += 1
                            logger.warning(f"バックプレッシャー検出: {stream_id}")

                            # バッファフラッシュ強制実行
                            await self._flush_stream_buffer(stream_id, callback)

                        # バッチサイズに達した場合
                        elif len(buffer) >= self.config.batch_size:
                            await self._flush_stream_buffer(stream_id, callback)

                    # メトリクス更新
                    self._update_streaming_metrics()

                    # ポーリング間隔
                    await asyncio.sleep(self.config.cdc_poll_interval_seconds)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"ストリーム処理エラー: {stream_id}, {e}")
                    self.streaming_metrics.errors_count += 1
                    await asyncio.sleep(1)  # エラー時は少し待機

        finally:
            # 残りバッファ処理
            if buffer:
                await self._flush_stream_buffer(stream_id, callback)

            # クリーンアップ
            if stream_id in self.stream_buffers:
                del self.stream_buffers[stream_id]

            logger.info(f"リアルタイムストリーム終了: {stream_id}")

    async def _flush_stream_buffer(self, stream_id: str, callback: callable) -> None:
        """ストリームバッファフラッシュ"""
        buffer = self.stream_buffers.get(stream_id, [])

        if buffer:
            try:
                # コールバック実行
                await callback(stream_id, buffer.copy())

                # バッファクリア
                buffer.clear()
                self.streaming_metrics.total_records_processed += len(buffer)

            except Exception as e:
                logger.error(f"バッファフラッシュエラー: {stream_id}, {e}")
                self.streaming_metrics.errors_count += 1

    async def _poll_data_source(
        self, data_source: DataSource, filter_conditions: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """データソースポーリング"""
        # データソース別の実装（実際の使用時に拡張）

        if data_source == DataSource.DATABASE:
            return await self._poll_database(filter_conditions)
        elif data_source == DataSource.API:
            return await self._poll_api(filter_conditions)
        elif data_source == DataSource.FILE:
            return await self._poll_file(filter_conditions)
        elif data_source == DataSource.STREAM:
            return await self._poll_stream(filter_conditions)
        else:
            return []

    async def _poll_database(
        self, filter_conditions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """データベースポーリング"""
        # 実装例（実際のDBアクセスに置換）
        return []

    async def _poll_api(
        self, filter_conditions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """APIポーリング"""
        # 実装例（実際のAPI呼び出しに置換）
        return []

    async def _poll_file(
        self, filter_conditions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """ファイルポーリング"""
        # 実装例（実際のファイル監視に置換）
        return []

    async def _poll_stream(
        self, filter_conditions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """ストリームポーリング"""
        # 実装例（実際のストリーミングに置換）
        return []

    def _update_streaming_metrics(self) -> None:
        """ストリーミングメトリクス更新"""
        current_time = datetime.now()

        # records_per_second計算
        if self.streaming_metrics.last_checkpoint:
            time_diff = (
                current_time - self.streaming_metrics.last_checkpoint
            ).total_seconds()
            if time_diff > 0:
                # 簡易的な計算（実際の実装では移動平均などを使用）
                self.streaming_metrics.records_per_second = (
                    self.config.batch_size / time_diff
                )

        self.streaming_metrics.last_checkpoint = current_time

    async def _start_cdc_monitoring(self) -> None:
        """CDC監視開始"""
        # CDC実装（実際の使用時に拡張）
        logger.info("CDC監視開始")

    async def _start_checkpoint_management(self) -> None:
        """チェックポイント管理開始"""
        asyncio.create_task(self._checkpoint_manager())

    async def _checkpoint_manager(self) -> None:
        """チェックポイント管理"""
        while self._is_running:
            try:
                await asyncio.sleep(self.config.checkpoint_interval_minutes * 60)

                # チェックポイント実行
                await self._create_checkpoint()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"チェックポイント管理エラー: {e}")

    async def _create_checkpoint(self) -> None:
        """チェックポイント作成"""
        checkpoint_data = {
            "timestamp": datetime.now(),
            "change_log_size": len(self.change_log),
            "active_streams": list(self.active_streams.keys()),
            "streaming_metrics": self.streaming_metrics,
            "last_sync_timestamps": self.last_sync_timestamps,
        }

        # チェックポイント保存（実際の実装では永続化）
        logger.info(f"チェックポイント作成: {checkpoint_data['timestamp']}")

    def _generate_change_id(self, table_name: str, primary_key: Any) -> str:
        """変更ID生成"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(
            f"{table_name}_{primary_key}_{timestamp}".encode()
        ).hexdigest()

    def _calculate_record_checksum(self, record: Dict[str, Any]) -> str:
        """レコードチェックサム計算"""
        # ソート済みJSON文字列からハッシュ計算
        import json

        record_str = json.dumps(record, sort_keys=True, default=str)
        return hashlib.sha256(record_str.encode()).hexdigest()

    async def get_change_statistics(self) -> Dict[str, Any]:
        """変更統計取得"""
        if not self.change_log:
            return {}

        change_type_counts = {}
        table_counts = {}

        for change in self.change_log:
            change_type = change.change_type.value
            table_name = change.table_name

            change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1
            table_counts[table_name] = table_counts.get(table_name, 0) + 1

        return {
            "total_changes": len(self.change_log),
            "change_type_distribution": change_type_counts,
            "table_distribution": table_counts,
            "streaming_metrics": {
                "total_records_processed": self.streaming_metrics.total_records_processed,
                "records_per_second": self.streaming_metrics.records_per_second,
                "duplicates_filtered": self.streaming_metrics.duplicates_filtered,
                "errors_count": self.streaming_metrics.errors_count,
                "backpressure_events": self.streaming_metrics.backpressure_events,
            },
            "active_streams": len(self.active_streams),
            "buffer_sizes": {
                stream_id: len(buffer)
                for stream_id, buffer in self.stream_buffers.items()
            },
        }

    async def replay_changes(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        table_filter: Optional[List[str]] = None,
    ) -> List[ChangeRecord]:
        """変更履歴リプレイ"""
        end_time = end_time or datetime.now()

        filtered_changes = []
        for change in self.change_log:
            if start_time <= change.timestamp <= end_time:
                if table_filter is None or change.table_name in table_filter:
                    filtered_changes.append(change)

        return filtered_changes

    async def cleanup(self) -> None:
        """システムクリーンアップ"""
        await self.stop_system()
        self.executor.shutdown(wait=True)
        logger.info("増分更新システムクリーンアップ完了")


# テスト用データ生成
async def test_incremental_system():
    """増分システムテスト"""
    system = IncrementalUpdateSystem()
    await system.start_system()

    # テストデータ生成
    old_data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "price": [100.0, 200.0, 300.0],
            "timestamp": pd.date_range("2024-01-01", periods=3),
        }
    )

    new_data = pd.DataFrame(
        {
            "id": [1, 2, 4],  # 3削除、4追加
            "price": [110.0, 200.0, 400.0],  # 1更新
            "timestamp": pd.date_range("2024-01-01", periods=3),
        }
    )

    # 変更検出テスト
    changes = await system.detect_changes(
        DataSource.DATABASE, "test_table", new_data, old_data, "id"
    )

    # 統計取得
    stats = await system.get_change_statistics()

    await system.cleanup()

    return changes, stats


if __name__ == "__main__":

    async def main():
        changes, stats = await test_incremental_system()

        print(f"検出された変更: {len(changes)}件")
        for change in changes:
            print(f"  {change.change_type.value}: {change.primary_key}")

        print("\n統計情報:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    asyncio.run(main())
