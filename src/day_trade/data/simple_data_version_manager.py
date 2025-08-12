#!/usr/bin/env python3
"""
シンプルデータバージョン管理システム
Issue #420対応 - MLモデル再現性のためのデータ追跡
"""

import gzip
import hashlib
import json
import logging
import pickle
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataVersion:
    """データバージョン情報"""

    version_id: str
    dataset_name: str
    created_at: datetime
    data_hash: str
    metadata: Dict[str, Any]
    file_path: str
    size_bytes: int
    record_count: int
    columns: List[str]
    description: str = ""


class SimpleDataVersionManager:
    """シンプルデータバージョン管理システム"""

    def __init__(self, storage_path: Path = Path("data_versions")):
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True, parents=True)

        # データベース初期化
        self.db_path = self.storage_path / "data_versions.db"
        self._init_database()

    def _init_database(self) -> None:
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_versions (
                    version_id TEXT PRIMARY KEY,
                    dataset_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    record_count INTEGER NOT NULL,
                    columns TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    description TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dataset_name ON data_versions(dataset_name)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at ON data_versions(created_at)
            """
            )

            conn.commit()

    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """データハッシュ計算"""
        # データフレームの内容をバイト列に変換してハッシュ化
        data_string = df.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(data_string).hexdigest()

    def _generate_version_id(self, dataset_name: str, timestamp: datetime) -> str:
        """バージョンID生成"""
        time_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")  # マイクロ秒追加
        hash_part = hashlib.md5(f"{dataset_name}_{time_str}".encode()).hexdigest()[:8]
        return f"{dataset_name}_v{time_str}_{hash_part}"

    def save_data_version(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        description: str = "",
        metadata: Dict[str, Any] = None,
    ) -> DataVersion:
        """データバージョン保存"""
        logger.info(f"データバージョン保存開始: {dataset_name}")

        if metadata is None:
            metadata = {}

        timestamp = datetime.now(timezone.utc)
        version_id = self._generate_version_id(dataset_name, timestamp)
        data_hash = self._calculate_data_hash(df)

        # 既存の同じハッシュをチェック
        existing_version = self._find_by_hash(data_hash)
        if existing_version:
            logger.info(f"同じデータが既に存在: {existing_version.version_id}")
            return existing_version

        # データファイル保存（圧縮）
        file_name = f"{version_id}.pkl.gz"
        file_path = self.storage_path / file_name

        with gzip.open(file_path, "wb") as f:
            pickle.dump(df, f)

        size_bytes = file_path.stat().st_size

        # メタデータ追加
        enhanced_metadata = {
            **metadata,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_summary": {},
        }

        # 数値列の統計追加
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            enhanced_metadata["numeric_summary"][col] = {
                "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                "std": float(df[col].std()) if not df[col].isnull().all() else None,
                "min": float(df[col].min()) if not df[col].isnull().all() else None,
                "max": float(df[col].max()) if not df[col].isnull().all() else None,
            }

        # バージョン情報作成
        version = DataVersion(
            version_id=version_id,
            dataset_name=dataset_name,
            created_at=timestamp,
            data_hash=data_hash,
            metadata=enhanced_metadata,
            file_path=str(file_path),
            size_bytes=size_bytes,
            record_count=len(df),
            columns=list(df.columns),
            description=description,
        )

        # データベースに保存
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO data_versions
                (version_id, dataset_name, created_at, data_hash, file_path,
                 size_bytes, record_count, columns, metadata, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    version.version_id,
                    version.dataset_name,
                    version.created_at.isoformat(),
                    version.data_hash,
                    version.file_path,
                    version.size_bytes,
                    version.record_count,
                    json.dumps(version.columns),
                    json.dumps(version.metadata, ensure_ascii=False),
                    version.description,
                ),
            )
            conn.commit()

        logger.info(f"データバージョン保存完了: {version_id}")
        return version

    def load_data_version(self, version_id: str) -> Optional[pd.DataFrame]:
        """データバージョン読み込み"""
        version = self.get_version_info(version_id)
        if not version:
            logger.error(f"バージョンが見つかりません: {version_id}")
            return None

        try:
            with gzip.open(version.file_path, "rb") as f:
                df = pickle.load(f)
            logger.info(f"データ読み込み完了: {version_id}")
            return df
        except Exception as e:
            logger.error(f"データ読み込みエラー: {version_id}, {e}")
            return None

    def get_version_info(self, version_id: str) -> Optional[DataVersion]:
        """バージョン情報取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM data_versions WHERE version_id = ?
            """,
                (version_id,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            return DataVersion(
                version_id=row[0],
                dataset_name=row[1],
                created_at=datetime.fromisoformat(row[2]),
                data_hash=row[3],
                file_path=row[4],
                size_bytes=row[5],
                record_count=row[6],
                columns=json.loads(row[7]),
                metadata=json.loads(row[8]),
                description=row[9] or "",
            )

    def list_versions(self, dataset_name: str = None, limit: int = 10) -> List[DataVersion]:
        """バージョンリスト取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if dataset_name:
                cursor.execute(
                    """
                    SELECT * FROM data_versions
                    WHERE dataset_name = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (dataset_name, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM data_versions
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (limit,),
                )

            versions = []
            for row in cursor.fetchall():
                versions.append(
                    DataVersion(
                        version_id=row[0],
                        dataset_name=row[1],
                        created_at=datetime.fromisoformat(row[2]),
                        data_hash=row[3],
                        file_path=row[4],
                        size_bytes=row[5],
                        record_count=row[6],
                        columns=json.loads(row[7]),
                        metadata=json.loads(row[8]),
                        description=row[9] or "",
                    )
                )

            return versions

    def _find_by_hash(self, data_hash: str) -> Optional[DataVersion]:
        """ハッシュによるバージョン検索"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM data_versions WHERE data_hash = ?
            """,
                (data_hash,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            return DataVersion(
                version_id=row[0],
                dataset_name=row[1],
                created_at=datetime.fromisoformat(row[2]),
                data_hash=row[3],
                file_path=row[4],
                size_bytes=row[5],
                record_count=row[6],
                columns=json.loads(row[7]),
                metadata=json.loads(row[8]),
                description=row[9] or "",
            )

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """バージョン比較"""
        version1 = self.get_version_info(version_id1)
        version2 = self.get_version_info(version_id2)

        if not version1 or not version2:
            return {"error": "バージョンが見つかりません"}

        comparison = {
            "version1": {
                "id": version1.version_id,
                "created_at": version1.created_at.isoformat(),
                "records": version1.record_count,
                "columns": version1.columns,
                "size_mb": version1.size_bytes / 1024 / 1024,
            },
            "version2": {
                "id": version2.version_id,
                "created_at": version2.created_at.isoformat(),
                "records": version2.record_count,
                "columns": version2.columns,
                "size_mb": version2.size_bytes / 1024 / 1024,
            },
            "differences": {
                "record_count_diff": version2.record_count - version1.record_count,
                "size_diff_mb": (version2.size_bytes - version1.size_bytes) / 1024 / 1024,
                "columns_added": list(set(version2.columns) - set(version1.columns)),
                "columns_removed": list(set(version1.columns) - set(version2.columns)),
                "same_hash": version1.data_hash == version2.data_hash,
            },
        }

        return comparison

    def delete_version(self, version_id: str) -> bool:
        """バージョン削除"""
        version = self.get_version_info(version_id)
        if not version:
            return False

        try:
            # ファイル削除
            Path(version.file_path).unlink(missing_ok=True)

            # データベースから削除
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM data_versions WHERE version_id = ?", (version_id,))
                conn.commit()

            logger.info(f"バージョン削除完了: {version_id}")
            return True

        except Exception as e:
            logger.error(f"バージョン削除エラー: {version_id}, {e}")
            return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """ストレージ統計取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 総計取得
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_versions,
                    SUM(size_bytes) as total_size_bytes,
                    SUM(record_count) as total_records,
                    COUNT(DISTINCT dataset_name) as unique_datasets
                FROM data_versions
            """
            )
            row = cursor.fetchone()

            # データセット別統計
            cursor.execute(
                """
                SELECT
                    dataset_name,
                    COUNT(*) as version_count,
                    SUM(size_bytes) as size_bytes,
                    MAX(created_at) as latest_version
                FROM data_versions
                GROUP BY dataset_name
                ORDER BY latest_version DESC
            """
            )
            dataset_stats = cursor.fetchall()

            return {
                "total_versions": row[0] or 0,
                "total_size_mb": (row[1] or 0) / 1024 / 1024,
                "total_records": row[2] or 0,
                "unique_datasets": row[3] or 0,
                "dataset_breakdown": [
                    {
                        "name": ds[0],
                        "versions": ds[1],
                        "size_mb": ds[2] / 1024 / 1024,
                        "latest": ds[3],
                    }
                    for ds in dataset_stats
                ],
            }

    def generate_lineage_report(self, dataset_name: str) -> str:
        """データ系譜レポート生成"""
        versions = self.list_versions(dataset_name, limit=20)

        if not versions:
            return f"データセット '{dataset_name}' のバージョンが見つかりません。"

        report = [
            "=" * 60,
            f"データ系譜レポート: {dataset_name}",
            "=" * 60,
            f"総バージョン数: {len(versions)}",
            f"最新バージョン: {versions[0].version_id}",
            f"最新更新: {versions[0].created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "バージョン履歴:",
        ]

        for i, version in enumerate(versions[:10]):
            age_days = (datetime.now(timezone.utc) - version.created_at).days
            size_mb = version.size_bytes / 1024 / 1024

            report.append(f"  {i+1:2d}. {version.version_id}")
            report.append(
                f"      作成: {version.created_at.strftime('%Y-%m-%d %H:%M:%S')} ({age_days}日前)"
            )
            report.append(f"      レコード数: {version.record_count:,}行")
            report.append(f"      サイズ: {size_mb:.2f}MB")
            report.append(f"      説明: {version.description or '説明なし'}")
            report.append("")

        # ストレージ統計
        stats = self.get_storage_stats()
        dataset_stat = next(
            (ds for ds in stats["dataset_breakdown"] if ds["name"] == dataset_name),
            None,
        )

        if dataset_stat:
            report.append("ストレージ使用量:")
            report.append(f"  総サイズ: {dataset_stat['size_mb']:.2f}MB")
            report.append(
                f"  平均サイズ: {dataset_stat['size_mb'] / dataset_stat['versions']:.2f}MB/バージョン"
            )

        report.append("=" * 60)
        return "\n".join(report)


def demo_data_version_management():
    """データバージョン管理デモ"""
    print("=== データバージョン管理システム デモ ===")

    # テストデータ作成
    np.random.seed(42)

    # v1: 初期データ
    data_v1 = pd.DataFrame(
        {
            "symbol": ["AAPL", "GOOGL", "MSFT"] * 10,
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "price": np.random.uniform(100, 200, 30),
            "volume": np.random.randint(1000000, 50000000, 30),
        }
    )

    # v2: データ更新版
    data_v2 = data_v1.copy()
    data_v2["price"] = data_v2["price"] * 1.05  # 価格を5%上昇
    data_v2.loc[len(data_v2)] = [
        "TSLA",
        pd.Timestamp("2024-01-31"),
        150.0,
        25000000,
    ]  # 新しい行追加

    # バージョン管理システム初期化
    version_manager = SimpleDataVersionManager()

    # バージョン保存
    print("\n1. データバージョン保存...")

    version1 = version_manager.save_data_version(
        df=data_v1, dataset_name="stock_prices", description="初期株価データセット"
    )
    print(f"   v1保存: {version1.version_id}")

    version2 = version_manager.save_data_version(
        df=data_v2,
        dataset_name="stock_prices",
        description="更新後株価データセット（5%上昇 + TSLA追加）",
    )
    print(f"   v2保存: {version2.version_id}")

    # 同じデータ保存テスト（重複チェック）
    version1_dup = version_manager.save_data_version(
        df=data_v1, dataset_name="stock_prices_duplicate", description="重複テスト"
    )
    print(
        f"   重複チェック: {'重複検出' if version1_dup.version_id == version1.version_id else '新バージョン'}"
    )

    # バージョンリスト
    print("\n2. バージョンリスト:")
    versions = version_manager.list_versions("stock_prices")
    for v in versions:
        print(f"   {v.version_id}: {v.record_count}行, {v.size_bytes/1024:.1f}KB")

    # データ読み込み
    print("\n3. データ読み込みテスト:")
    loaded_data = version_manager.load_data_version(version1.version_id)
    print(f"   読み込み成功: {len(loaded_data)}行")
    print(f"   元データと一致: {loaded_data.equals(data_v1)}")

    # バージョン比較
    print("\n4. バージョン比較:")
    comparison = version_manager.compare_versions(version1.version_id, version2.version_id)
    print(f"   レコード数差: {comparison['differences']['record_count_diff']}")
    print(f"   サイズ差: {comparison['differences']['size_diff_mb']:.2f}MB")
    print(f"   同じデータ: {comparison['differences']['same_hash']}")

    # ストレージ統計
    print("\n5. ストレージ統計:")
    stats = version_manager.get_storage_stats()
    print(f"   総バージョン数: {stats['total_versions']}")
    print(f"   総サイズ: {stats['total_size_mb']:.2f}MB")
    print(f"   データセット数: {stats['unique_datasets']}")

    # 系譜レポート
    print("\n6. データ系譜レポート:")
    lineage = version_manager.generate_lineage_report("stock_prices")
    print(lineage)

    print("\n=== デモ完了 ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_data_version_management()
