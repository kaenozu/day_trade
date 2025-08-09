#!/usr/bin/env python3
"""
データ圧縮・アーカイブシステム
Issue #317: 高速データ管理システム - Phase 2

長期データの効率的保存・管理システム
- データ圧縮アルゴリズム最適化
- 自動ライフサイクル管理
- コールドストレージ移行
- データ復元機能
"""

import asyncio
import gzip
import json
import lzma
import pickle
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class CompressionAlgorithm(Enum):
    """圧縮アルゴリズム種類"""

    GZIP = "gzip"
    LZMA = "lzma"
    ZLIB = "zlib"
    PICKLE = "pickle"
    PARQUET = "parquet"
    CUSTOM_HYBRID = "custom_hybrid"


class DataLifecycleStage(Enum):
    """データライフサイクルステージ"""

    HOT = "hot"  # ホットデータ（高頻度アクセス）
    WARM = "warm"  # ウォームデータ（中頻度アクセス）
    COLD = "cold"  # コールドデータ（低頻度アクセス）
    ARCHIVED = "archived"  # アーカイブ済み（最低頻度アクセス）
    DELETED = "deleted"  # 削除済み


@dataclass
class CompressionConfig:
    """圧縮設定"""

    # 圧縮設定
    default_algorithm: CompressionAlgorithm = CompressionAlgorithm.LZMA
    compression_level: int = 6  # 1-9（高いほど圧縮率高、時間長）
    chunk_size_mb: int = 100  # チャンクサイズ

    # ライフサイクル設定
    hot_retention_days: int = 30  # ホット期間
    warm_retention_days: int = 180  # ウォーム期間
    cold_retention_days: int = 730  # コールド期間
    archive_retention_years: int = 7  # アーカイブ期間

    # ストレージ設定
    hot_storage_path: str = "data/hot"
    warm_storage_path: str = "data/warm"
    cold_storage_path: str = "data/cold"
    archive_storage_path: str = "data/archive"

    # パフォーマンス設定
    max_concurrent_operations: int = 4
    enable_integrity_check: bool = True
    enable_deduplication: bool = True


@dataclass
class CompressionResult:
    """圧縮結果"""

    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm: CompressionAlgorithm
    compression_time_ms: float
    file_path: str
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ArchiveMetadata:
    """アーカイブメタデータ"""

    data_id: str
    original_size: int
    compressed_size: int
    compression_algorithm: CompressionAlgorithm
    lifecycle_stage: DataLifecycleStage
    creation_date: datetime
    last_access_date: datetime
    access_count: int
    checksum: str
    tags: List[str] = field(default_factory=list)
    retention_until: Optional[datetime] = None


class DataCompressionArchiveSystem:
    """データ圧縮・アーカイブシステム"""

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_operations
        )
        self.compression_stats: List[CompressionResult] = []
        self.archive_catalog: Dict[str, ArchiveMetadata] = {}
        self._initialize_storage_paths()

    def _initialize_storage_paths(self) -> None:
        """ストレージパス初期化"""
        storage_paths = [
            self.config.hot_storage_path,
            self.config.warm_storage_path,
            self.config.cold_storage_path,
            self.config.archive_storage_path,
        ]

        for path in storage_paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    async def compress_data(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], bytes],
        data_id: str,
        algorithm: Optional[CompressionAlgorithm] = None,
        lifecycle_stage: DataLifecycleStage = DataLifecycleStage.HOT,
    ) -> CompressionResult:
        """データ圧縮"""
        algorithm = algorithm or self.config.default_algorithm
        start_time = time.time()

        try:
            # データシリアライズ
            if isinstance(data, pd.DataFrame):
                serialized_data = self._serialize_dataframe(data)
            elif isinstance(data, dict):
                serialized_data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            else:
                serialized_data = data

            original_size = len(serialized_data)

            # 圧縮実行
            compressed_data, compression_time = await self._compress_with_algorithm(
                serialized_data, algorithm
            )

            compressed_size = len(compressed_data)
            compression_ratio = (
                compressed_size / original_size if original_size > 0 else 1.0
            )

            # ストレージパス決定
            storage_path = self._get_storage_path(lifecycle_stage)
            file_path = storage_path / f"{data_id}.{algorithm.value}"

            # ファイル保存
            with open(file_path, "wb") as f:
                f.write(compressed_data)

            # チェックサム計算
            checksum = self._calculate_checksum(compressed_data)

            # 結果記録
            result = CompressionResult(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                algorithm=algorithm,
                compression_time_ms=(time.time() - start_time) * 1000,
                file_path=str(file_path),
                checksum=checksum,
            )

            self.compression_stats.append(result)

            # アーカイブカタログ更新
            self.archive_catalog[data_id] = ArchiveMetadata(
                data_id=data_id,
                original_size=original_size,
                compressed_size=compressed_size,
                compression_algorithm=algorithm,
                lifecycle_stage=lifecycle_stage,
                creation_date=datetime.now(),
                last_access_date=datetime.now(),
                access_count=0,
                checksum=checksum,
                retention_until=self._calculate_retention_date(lifecycle_stage),
            )

            logger.info(f"データ圧縮完了: {data_id}, 圧縮率: {compression_ratio:.3f}")
            return result

        except Exception as e:
            logger.error(f"データ圧縮エラー: {data_id}, {e}")
            raise

    async def _compress_with_algorithm(
        self, data: bytes, algorithm: CompressionAlgorithm
    ) -> Tuple[bytes, float]:
        """アルゴリズム別圧縮実行"""
        start_time = time.time()

        def compress_sync():
            if algorithm == CompressionAlgorithm.GZIP:
                return gzip.compress(data, compresslevel=self.config.compression_level)
            elif algorithm == CompressionAlgorithm.LZMA:
                return lzma.compress(data, preset=self.config.compression_level)
            elif algorithm == CompressionAlgorithm.ZLIB:
                return zlib.compress(data, level=self.config.compression_level)
            elif algorithm == CompressionAlgorithm.PICKLE:
                return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            elif algorithm == CompressionAlgorithm.CUSTOM_HYBRID:
                return self._custom_hybrid_compress(data)
            else:
                return gzip.compress(data)  # フォールバック

        # 非同期実行
        loop = asyncio.get_event_loop()
        compressed_data = await loop.run_in_executor(self.executor, compress_sync)
        compression_time = (time.time() - start_time) * 1000

        return compressed_data, compression_time

    def _custom_hybrid_compress(self, data: bytes) -> bytes:
        """カスタムハイブリッド圧縮"""
        # 複数アルゴリズム組み合わせ最適化

        # 1. データサイズに応じた前処理
        if len(data) > 1024 * 1024:  # 1MB以上
            # 大容量データ: LZMA（高圧縮率）
            return lzma.compress(data, preset=9)
        elif len(data) > 1024 * 10:  # 10KB以上
            # 中容量データ: GZIP（バランス）
            return gzip.compress(data, compresslevel=6)
        else:
            # 小容量データ: ZLIB（高速）
            return zlib.compress(data, level=3)

    def _serialize_dataframe(self, df: pd.DataFrame) -> bytes:
        """DataFrame最適シリアライズ"""
        # Parquet形式でシリアライズ（高効率）
        try:
            buffer = df.to_parquet(engine="pyarrow", compression=None)
            return buffer
        except ImportError:
            # フォールバック: pickle
            return pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)

    async def decompress_data(
        self, data_id: str, update_access_stats: bool = True
    ) -> Optional[Union[pd.DataFrame, Dict[str, Any], bytes]]:
        """データ復元"""
        if data_id not in self.archive_catalog:
            logger.warning(f"データが見つかりません: {data_id}")
            return None

        metadata = self.archive_catalog[data_id]
        start_time = time.time()

        try:
            # ファイル読み込み
            file_path = (
                Path(self._get_storage_path(metadata.lifecycle_stage))
                / f"{data_id}.{metadata.compression_algorithm.value}"
            )

            with open(file_path, "rb") as f:
                compressed_data = f.read()

            # チェックサム検証
            if self.config.enable_integrity_check:
                current_checksum = self._calculate_checksum(compressed_data)
                if current_checksum != metadata.checksum:
                    logger.error(f"データ整合性エラー: {data_id}")
                    return None

            # 復元実行
            decompressed_data = await self._decompress_with_algorithm(
                compressed_data, metadata.compression_algorithm
            )

            # アクセス統計更新
            if update_access_stats:
                metadata.last_access_date = datetime.now()
                metadata.access_count += 1

            decompression_time = (time.time() - start_time) * 1000
            logger.info(f"データ復元完了: {data_id}, {decompression_time:.2f}ms")

            return decompressed_data

        except Exception as e:
            logger.error(f"データ復元エラー: {data_id}, {e}")
            return None

    async def _decompress_with_algorithm(
        self, compressed_data: bytes, algorithm: CompressionAlgorithm
    ) -> Union[pd.DataFrame, Dict[str, Any], bytes]:
        """アルゴリズム別復元実行"""

        def decompress_sync():
            if algorithm == CompressionAlgorithm.GZIP:
                return gzip.decompress(compressed_data)
            elif algorithm == CompressionAlgorithm.LZMA:
                return lzma.decompress(compressed_data)
            elif algorithm == CompressionAlgorithm.ZLIB:
                return zlib.decompress(compressed_data)
            elif algorithm == CompressionAlgorithm.PICKLE:
                return pickle.loads(compressed_data)
            elif algorithm == CompressionAlgorithm.CUSTOM_HYBRID:
                return self._custom_hybrid_decompress(compressed_data)
            else:
                return gzip.decompress(compressed_data)  # フォールバック

        # 非同期実行
        loop = asyncio.get_event_loop()
        decompressed_data = await loop.run_in_executor(self.executor, decompress_sync)

        # データタイプ復元試行
        try:
            # JSON形式確認
            if isinstance(decompressed_data, bytes):
                json_str = decompressed_data.decode("utf-8")
                return json.loads(json_str)
        except:
            pass

        try:
            # DataFrame復元試行（Parquet）
            if isinstance(decompressed_data, bytes):
                import io

                return pd.read_parquet(io.BytesIO(decompressed_data))
        except:
            pass

        return decompressed_data

    def _custom_hybrid_decompress(self, compressed_data: bytes) -> bytes:
        """カスタムハイブリッド復元"""
        # 圧縮アルゴリズム自動検出・復元
        try:
            return lzma.decompress(compressed_data)
        except:
            try:
                return gzip.decompress(compressed_data)
            except:
                return zlib.decompress(compressed_data)

    async def lifecycle_management(self) -> Dict[str, Any]:
        """データライフサイクル管理実行"""
        management_stats = {
            "promoted_to_warm": 0,
            "promoted_to_cold": 0,
            "archived": 0,
            "deleted": 0,
            "errors": [],
        }

        current_time = datetime.now()

        for data_id, metadata in self.archive_catalog.items():
            try:
                days_since_creation = (current_time - metadata.creation_date).days
                days_since_access = (current_time - metadata.last_access_date).days

                # ライフサイクル段階判定
                new_stage = self._determine_lifecycle_stage(
                    days_since_creation, days_since_access, metadata.access_count
                )

                if new_stage != metadata.lifecycle_stage:
                    # データ移行実行
                    success = await self._migrate_data(data_id, metadata, new_stage)

                    if success:
                        if new_stage == DataLifecycleStage.WARM:
                            management_stats["promoted_to_warm"] += 1
                        elif new_stage == DataLifecycleStage.COLD:
                            management_stats["promoted_to_cold"] += 1
                        elif new_stage == DataLifecycleStage.ARCHIVED:
                            management_stats["archived"] += 1
                        elif new_stage == DataLifecycleStage.DELETED:
                            management_stats["deleted"] += 1

                # 保持期限チェック
                if metadata.retention_until and current_time > metadata.retention_until:
                    await self._delete_data(data_id)
                    management_stats["deleted"] += 1

            except Exception as e:
                error_msg = f"ライフサイクル管理エラー {data_id}: {e}"
                logger.error(error_msg)
                management_stats["errors"].append(error_msg)

        logger.info(f"ライフサイクル管理完了: {management_stats}")
        return management_stats

    def _determine_lifecycle_stage(
        self, days_since_creation: int, days_since_access: int, access_count: int
    ) -> DataLifecycleStage:
        """ライフサイクルステージ決定"""

        # アクセス頻度重視の判定
        if days_since_access <= 7 and access_count > 10:
            return DataLifecycleStage.HOT
        elif days_since_creation <= self.config.hot_retention_days and access_count > 3:
            return DataLifecycleStage.HOT
        elif days_since_creation <= self.config.warm_retention_days:
            return DataLifecycleStage.WARM
        elif days_since_creation <= self.config.cold_retention_days:
            return DataLifecycleStage.COLD
        else:
            return DataLifecycleStage.ARCHIVED

    async def _migrate_data(
        self, data_id: str, metadata: ArchiveMetadata, new_stage: DataLifecycleStage
    ) -> bool:
        """データ移行実行"""
        try:
            old_path = (
                self._get_storage_path(metadata.lifecycle_stage)
                / f"{data_id}.{metadata.compression_algorithm.value}"
            )
            new_path = (
                self._get_storage_path(new_stage)
                / f"{data_id}.{metadata.compression_algorithm.value}"
            )

            # ファイル移動
            old_path.rename(new_path)

            # メタデータ更新
            metadata.lifecycle_stage = new_stage

            logger.info(
                f"データ移行完了: {data_id}, {metadata.lifecycle_stage.value} -> {new_stage.value}"
            )
            return True

        except Exception as e:
            logger.error(f"データ移行エラー: {data_id}, {e}")
            return False

    async def _delete_data(self, data_id: str) -> bool:
        """データ削除"""
        try:
            metadata = self.archive_catalog[data_id]
            file_path = (
                self._get_storage_path(metadata.lifecycle_stage)
                / f"{data_id}.{metadata.compression_algorithm.value}"
            )

            # ファイル削除
            if file_path.exists():
                file_path.unlink()

            # カタログから削除
            del self.archive_catalog[data_id]

            logger.info(f"データ削除完了: {data_id}")
            return True

        except Exception as e:
            logger.error(f"データ削除エラー: {data_id}, {e}")
            return False

    def _get_storage_path(self, stage: DataLifecycleStage) -> Path:
        """ライフサイクル段階に応じたストレージパス取得"""
        if stage == DataLifecycleStage.HOT:
            return Path(self.config.hot_storage_path)
        elif stage == DataLifecycleStage.WARM:
            return Path(self.config.warm_storage_path)
        elif stage == DataLifecycleStage.COLD:
            return Path(self.config.cold_storage_path)
        elif stage == DataLifecycleStage.ARCHIVED:
            return Path(self.config.archive_storage_path)
        else:
            return Path(self.config.hot_storage_path)  # デフォルト

    def _calculate_retention_date(self, stage: DataLifecycleStage) -> datetime:
        """保持期限計算"""
        current_time = datetime.now()

        if stage == DataLifecycleStage.HOT:
            return current_time + timedelta(days=self.config.hot_retention_days)
        elif stage == DataLifecycleStage.WARM:
            return current_time + timedelta(days=self.config.warm_retention_days)
        elif stage == DataLifecycleStage.COLD:
            return current_time + timedelta(days=self.config.cold_retention_days)
        elif stage == DataLifecycleStage.ARCHIVED:
            return current_time + timedelta(
                days=self.config.archive_retention_years * 365
            )
        else:
            return current_time + timedelta(days=30)  # デフォルト

    def _calculate_checksum(self, data: bytes) -> str:
        """チェックサム計算"""
        import hashlib

        return hashlib.sha256(data).hexdigest()

    async def get_compression_statistics(self) -> Dict[str, Any]:
        """圧縮統計情報取得"""
        if not self.compression_stats:
            return {}

        compression_ratios = [stat.compression_ratio for stat in self.compression_stats]
        compression_times = [
            stat.compression_time_ms for stat in self.compression_stats
        ]
        original_sizes = [stat.original_size for stat in self.compression_stats]
        compressed_sizes = [stat.compressed_size for stat in self.compression_stats]

        return {
            "total_compressions": len(self.compression_stats),
            "avg_compression_ratio": np.mean(compression_ratios),
            "best_compression_ratio": np.min(compression_ratios),
            "worst_compression_ratio": np.max(compression_ratios),
            "avg_compression_time_ms": np.mean(compression_times),
            "total_original_size_mb": sum(original_sizes) / (1024 * 1024),
            "total_compressed_size_mb": sum(compressed_sizes) / (1024 * 1024),
            "total_space_saved_mb": (sum(original_sizes) - sum(compressed_sizes))
            / (1024 * 1024),
            "space_saving_percentage": (1 - sum(compressed_sizes) / sum(original_sizes))
            * 100
            if sum(original_sizes) > 0
            else 0,
        }

    async def get_archive_status(self) -> Dict[str, Any]:
        """アーカイブ状況取得"""
        stage_counts = {}
        total_size = 0

        for metadata in self.archive_catalog.values():
            stage = metadata.lifecycle_stage.value
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            total_size += metadata.compressed_size

        return {
            "total_archived_items": len(self.archive_catalog),
            "stage_distribution": stage_counts,
            "total_archived_size_mb": total_size / (1024 * 1024),
            "catalog_size": len(self.archive_catalog),
        }

    async def cleanup(self) -> None:
        """リソースクリーンアップ"""
        self.executor.shutdown(wait=True)
        logger.info("データ圧縮・アーカイブシステムクリーンアップ完了")


# 使用例・テスト用関数
async def test_compression_performance():
    """圧縮性能テスト"""
    system = DataCompressionArchiveSystem()

    # テストデータ生成
    test_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10000, freq="1min"),
            "price": np.random.normal(1000, 50, 10000),
            "volume": np.random.randint(1000, 100000, 10000),
        }
    )

    # 各アルゴリズムテスト
    algorithms = [
        CompressionAlgorithm.GZIP,
        CompressionAlgorithm.LZMA,
        CompressionAlgorithm.ZLIB,
        CompressionAlgorithm.CUSTOM_HYBRID,
    ]

    results = {}

    for algo in algorithms:
        result = await system.compress_data(test_df, f"test_{algo.value}", algo)
        results[algo.value] = {
            "compression_ratio": result.compression_ratio,
            "compression_time_ms": result.compression_time_ms,
            "original_size_mb": result.original_size / (1024 * 1024),
            "compressed_size_mb": result.compressed_size / (1024 * 1024),
        }

    # 復元テスト
    restored_data = await system.decompress_data(f"test_{algorithms[0].value}")

    await system.cleanup()

    return results, restored_data is not None


if __name__ == "__main__":

    async def main():
        results, restoration_success = await test_compression_performance()

        print("圧縮性能テスト結果:")
        for algo, stats in results.items():
            print(f"  {algo}:")
            for key, value in stats.items():
                print(f"    {key}: {value}")

        print(f"\nデータ復元成功: {restoration_success}")

    asyncio.run(main())
