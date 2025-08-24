#!/usr/bin/env python3
"""
Issue #800 Phase 5: バックアップシステム自動化
Day Trade ML System 包括的データ保護・復旧
"""

import os
import json
import logging
import boto3
import shutil
import gzip
import tarfile
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import hashlib
import psycopg2
import redis

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BackupType(Enum):
    """バックアップタイプ"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"

class BackupStatus(Enum):
    """バックアップステータス"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BackupDestination(Enum):
    """バックアップ保存先"""
    LOCAL = "local"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"
    NFS = "nfs"

@dataclass
class BackupJob:
    """バックアップジョブ"""
    id: str
    name: str
    backup_type: BackupType
    source_type: str  # database, file, volume, etc.
    source_path: str
    destination: BackupDestination
    destination_path: str
    schedule: str  # cron format
    retention_days: int
    compression: bool = True
    encryption: bool = True
    status: BackupStatus = BackupStatus.PENDING
    created_at: Optional[datetime] = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    size_bytes: int = 0
    checksum: str = ""
    error_message: str = ""

@dataclass
class BackupRecord:
    """バックアップ記録"""
    job_id: str
    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    size_bytes: int
    checksum: str
    file_path: str
    status: BackupStatus
    duration_seconds: int = 0
    error_message: str = ""

class BackupManager:
    """バックアップマネージャー"""

    def __init__(self):
        self.jobs: Dict[str, BackupJob] = {}
        self.records: List[BackupRecord] = []
        self.running_jobs: Dict[str, threading.Thread] = {}

        # AWS S3設定
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-1')
        )

        # バックアップ設定
        self.backup_root = os.getenv('BACKUP_ROOT_PATH', '/backup')
        self.encryption_key = os.getenv('BACKUP_ENCRYPTION_KEY', 'DayTrade2025BackupKey!')

        # デフォルトバックアップジョブ作成
        self._create_default_backup_jobs()

    def create_backup_job(self, job: BackupJob) -> bool:
        """バックアップジョブ作成"""
        try:
            job.created_at = datetime.utcnow()
            job.id = self._generate_job_id(job.name)

            # スケジュール計算
            job.next_run = self._calculate_next_run(job.schedule)

            self.jobs[job.id] = job

            logger.info(f"Backup job created: {job.name} ({job.id})")
            return True

        except Exception as e:
            logger.error(f"Failed to create backup job: {str(e)}")
            return False

    def run_backup_job(self, job_id: str, force: bool = False) -> bool:
        """バックアップジョブ実行"""
        if job_id not in self.jobs:
            logger.error(f"Backup job not found: {job_id}")
            return False

        job = self.jobs[job_id]

        # 実行中確認
        if job_id in self.running_jobs:
            logger.warning(f"Backup job already running: {job_id}")
            return False

        # 強制実行でない場合はスケジュール確認
        if not force and job.next_run and datetime.utcnow() < job.next_run:
            logger.info(f"Backup job not scheduled yet: {job_id}")
            return False

        # バックアップ実行スレッド開始
        backup_thread = threading.Thread(
            target=self._execute_backup,
            args=(job,),
            name=f"backup-{job_id}"
        )
        backup_thread.start()

        self.running_jobs[job_id] = backup_thread

        logger.info(f"Backup job started: {job.name} ({job_id})")
        return True

    def schedule_backup_jobs(self):
        """スケジュールされたバックアップジョブ実行"""
        current_time = datetime.utcnow()

        for job_id, job in self.jobs.items():
            # スケジュール確認
            if (job.next_run and current_time >= job.next_run and
                job_id not in self.running_jobs):

                logger.info(f"Triggering scheduled backup: {job.name}")
                self.run_backup_job(job_id, force=True)

    def get_backup_status(self, job_id: str = None) -> Union[Dict, List[Dict]]:
        """バックアップ状況取得"""
        if job_id:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                return {
                    'job': asdict(job),
                    'is_running': job_id in self.running_jobs,
                    'recent_records': [
                        asdict(r) for r in self.records
                        if r.job_id == job_id
                    ][-10:]  # 最新10件
                }
            return {}
        else:
            # 全ジョブ状況
            return [
                {
                    'job_id': job_id,
                    'name': job.name,
                    'status': job.status.value,
                    'last_run': job.last_run.isoformat() if job.last_run else None,
                    'next_run': job.next_run.isoformat() if job.next_run else None,
                    'is_running': job_id in self.running_jobs
                }
                for job_id, job in self.jobs.items()
            ]

    def restore_backup(self, backup_id: str, target_path: str = None) -> bool:
        """バックアップ復元"""
        try:
            # バックアップ記録検索
            backup_record = None
            for record in self.records:
                if record.backup_id == backup_id:
                    backup_record = record
                    break

            if not backup_record:
                logger.error(f"Backup record not found: {backup_id}")
                return False

            job = self.jobs.get(backup_record.job_id)
            if not job:
                logger.error(f"Backup job not found: {backup_record.job_id}")
                return False

            # 復元実行
            if job.source_type == 'database':
                return self._restore_database(backup_record, target_path)
            elif job.source_type == 'file':
                return self._restore_files(backup_record, target_path)
            elif job.source_type == 'volume':
                return self._restore_volume(backup_record, target_path)
            else:
                logger.error(f"Unsupported restore type: {job.source_type}")
                return False

        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            return False

    def cleanup_old_backups(self):
        """古いバックアップクリーンアップ"""
        current_time = datetime.utcnow()
        cleaned_count = 0

        for job in self.jobs.values():
            cutoff_date = current_time - timedelta(days=job.retention_days)

            # 期限切れバックアップ検索
            expired_records = [
                r for r in self.records
                if r.job_id == job.id and r.timestamp < cutoff_date
            ]

            for record in expired_records:
                try:
                    # ファイル削除
                    if job.destination == BackupDestination.LOCAL:
                        if os.path.exists(record.file_path):
                            os.remove(record.file_path)
                    elif job.destination == BackupDestination.S3:
                        self.s3_client.delete_object(
                            Bucket=job.destination_path.split('/')[0],
                            Key='/'.join(job.destination_path.split('/')[1:] + [os.path.basename(record.file_path)])
                        )

                    # 記録削除
                    self.records.remove(record)
                    cleaned_count += 1

                    logger.info(f"Old backup cleaned: {record.backup_id}")

                except Exception as e:
                    logger.error(f"Failed to clean backup {record.backup_id}: {str(e)}")

        logger.info(f"Cleanup completed: {cleaned_count} old backups removed")

    def _create_default_backup_jobs(self):
        """デフォルトバックアップジョブ作成"""
        default_jobs = [
            # PostgreSQLデータベース（日次）
            BackupJob(
                id="",
                name="postgresql_daily",
                backup_type=BackupType.FULL,
                source_type="database",
                source_path=f"postgresql://day_trade_user:{os.getenv('DB_PASSWORD', 'REPLACE_WITH_ACTUAL_PASSWORD')}@localhost:5432/day_trade_ml",
                destination=BackupDestination.S3,
                destination_path="day-trade-backups/database",
                schedule="0 2 * * *",  # 毎日午前2時
                retention_days=30,
                compression=True,
                encryption=True
            ),

            # MLモデルファイル（週次）
            BackupJob(
                id="",
                name="ml_models_weekly",
                backup_type=BackupType.FULL,
                source_type="file",
                source_path="/app/models",
                destination=BackupDestination.S3,
                destination_path="day-trade-backups/models",
                schedule="0 3 * * 0",  # 毎週日曜日午前3時
                retention_days=90,
                compression=True,
                encryption=True
            ),

            # 設定ファイル（日次）
            BackupJob(
                id="",
                name="config_daily",
                backup_type=BackupType.FULL,
                source_type="file",
                source_path="/app/config",
                destination=BackupDestination.LOCAL,
                destination_path="/backup/config",
                schedule="0 1 * * *",  # 毎日午前1時
                retention_days=14,
                compression=True,
                encryption=True
            ),

            # ログファイル（日次）
            BackupJob(
                id="",
                name="logs_daily",
                backup_type=BackupType.INCREMENTAL,
                source_type="file",
                source_path="/app/logs",
                destination=BackupDestination.S3,
                destination_path="day-trade-backups/logs",
                schedule="0 23 * * *",  # 毎日午後11時
                retention_days=60,
                compression=True,
                encryption=False  # ログは暗号化なし
            ),

            # Redisデータ（6時間毎）
            BackupJob(
                id="",
                name="redis_6hourly",
                backup_type=BackupType.SNAPSHOT,
                source_type="redis",
                source_path="redis://localhost:6379",
                destination=BackupDestination.LOCAL,
                destination_path="/backup/redis",
                schedule="0 */6 * * *",  # 6時間毎
                retention_days=7,
                compression=True,
                encryption=True
            )
        ]

        for job in default_jobs:
            self.create_backup_job(job)

    def _execute_backup(self, job: BackupJob):
        """バックアップ実行（スレッド内）"""
        start_time = datetime.utcnow()
        backup_id = f"{job.id}_{start_time.strftime('%Y%m%d_%H%M%S')}"

        try:
            job.status = BackupStatus.IN_PROGRESS
            job.last_run = start_time

            logger.info(f"Starting backup: {job.name} (ID: {backup_id})")

            # バックアップ実行
            if job.source_type == 'database':
                file_path, size_bytes = self._backup_database(job, backup_id)
            elif job.source_type == 'file':
                file_path, size_bytes = self._backup_files(job, backup_id)
            elif job.source_type == 'redis':
                file_path, size_bytes = self._backup_redis(job, backup_id)
            elif job.source_type == 'volume':
                file_path, size_bytes = self._backup_volume(job, backup_id)
            else:
                raise ValueError(f"Unsupported backup type: {job.source_type}")

            # チェックサム計算
            checksum = self._calculate_checksum(file_path)

            # バックアップ記録作成
            duration = (datetime.utcnow() - start_time).total_seconds()

            backup_record = BackupRecord(
                job_id=job.id,
                backup_id=backup_id,
                backup_type=job.backup_type,
                timestamp=start_time,
                size_bytes=size_bytes,
                checksum=checksum,
                file_path=file_path,
                status=BackupStatus.COMPLETED,
                duration_seconds=int(duration)
            )

            self.records.append(backup_record)

            # ジョブ更新
            job.status = BackupStatus.COMPLETED
            job.size_bytes = size_bytes
            job.checksum = checksum
            job.next_run = self._calculate_next_run(job.schedule)

            logger.info(f"Backup completed: {job.name} (Size: {size_bytes:,} bytes, Duration: {duration:.1f}s)")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Backup failed: {job.name} - {error_msg}")

            # 失敗記録
            backup_record = BackupRecord(
                job_id=job.id,
                backup_id=backup_id,
                backup_type=job.backup_type,
                timestamp=start_time,
                size_bytes=0,
                checksum="",
                file_path="",
                status=BackupStatus.FAILED,
                duration_seconds=int((datetime.utcnow() - start_time).total_seconds()),
                error_message=error_msg
            )

            self.records.append(backup_record)

            job.status = BackupStatus.FAILED
            job.error_message = error_msg

        finally:
            # 実行中ジョブから削除
            if job.id in self.running_jobs:
                del self.running_jobs[job.id]

    def _backup_database(self, job: BackupJob, backup_id: str) -> Tuple[str, int]:
        """データベースバックアップ"""
        # PostgreSQL接続情報解析
        db_url = job.source_path
        # postgresql://user:pass@host:port/dbname

        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"postgresql_{backup_id}_{timestamp}.sql"

        if job.destination == BackupDestination.LOCAL:
            backup_path = os.path.join(job.destination_path, filename)
        else:
            backup_path = os.path.join(self.backup_root, 'temp', filename)

        os.makedirs(os.path.dirname(backup_path), exist_ok=True)

        # pg_dump実行
        cmd = [
            'pg_dump',
            '--verbose',
            '--no-password',
            '--format=custom',
            '--compress=9',
            '--file', backup_path,
            db_url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"pg_dump failed: {result.stderr}")

        # 圧縮・暗号化
        if job.compression or job.encryption:
            backup_path = self._process_backup_file(backup_path, job.compression, job.encryption)

        # S3アップロード
        if job.destination == BackupDestination.S3:
            s3_key = f"{job.destination_path}/{filename}"
            self.s3_client.upload_file(backup_path, 'day-trade-backups', s3_key)

            # ローカル一時ファイル削除
            os.remove(backup_path)
            backup_path = f"s3://day-trade-backups/{s3_key}"

        file_size = os.path.getsize(backup_path) if backup_path.startswith('/') else 0
        return backup_path, file_size

    def _backup_files(self, job: BackupJob, backup_id: str) -> Tuple[str, int]:
        """ファイルバックアップ"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"files_{backup_id}_{timestamp}.tar"

        if job.destination == BackupDestination.LOCAL:
            backup_path = os.path.join(job.destination_path, filename)
        else:
            backup_path = os.path.join(self.backup_root, 'temp', filename)

        os.makedirs(os.path.dirname(backup_path), exist_ok=True)

        # tar アーカイブ作成
        with tarfile.open(backup_path, 'w') as tar:
            tar.add(job.source_path, arcname=os.path.basename(job.source_path))

        # 圧縮・暗号化処理
        if job.compression or job.encryption:
            backup_path = self._process_backup_file(backup_path, job.compression, job.encryption)

        # S3アップロード
        if job.destination == BackupDestination.S3:
            s3_key = f"{job.destination_path}/{filename}"
            self.s3_client.upload_file(backup_path, 'day-trade-backups', s3_key)

            os.remove(backup_path)
            backup_path = f"s3://day-trade-backups/{s3_key}"

        file_size = os.path.getsize(backup_path) if backup_path.startswith('/') else 0
        return backup_path, file_size

    def _backup_redis(self, job: BackupJob, backup_id: str) -> Tuple[str, int]:
        """Redisバックアップ"""
        # Redis接続
        redis_client = redis.Redis.from_url(job.source_path)

        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"redis_{backup_id}_{timestamp}.rdb"

        if job.destination == BackupDestination.LOCAL:
            backup_path = os.path.join(job.destination_path, filename)
        else:
            backup_path = os.path.join(self.backup_root, 'temp', filename)

        os.makedirs(os.path.dirname(backup_path), exist_ok=True)

        # Redis SAVE コマンド実行
        redis_client.save()

        # RDBファイルコピー（実際の実装では適切な方法で取得）
        # ここでは簡略化
        with open(backup_path, 'wb') as f:
            f.write(b"Redis backup placeholder")

        # 処理
        if job.compression or job.encryption:
            backup_path = self._process_backup_file(backup_path, job.compression, job.encryption)

        file_size = os.path.getsize(backup_path)
        return backup_path, file_size

    def _backup_volume(self, job: BackupJob, backup_id: str) -> Tuple[str, int]:
        """ボリュームバックアップ"""
        # 実装はストレージシステムに依存
        # ここでは基本的なディレクトリコピーとして実装
        return self._backup_files(job, backup_id)

    def _process_backup_file(self, file_path: str, compress: bool, encrypt: bool) -> str:
        """バックアップファイル後処理（圧縮・暗号化）"""
        processed_path = file_path

        # 圧縮
        if compress:
            compressed_path = f"{file_path}.gz"
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(file_path)
            processed_path = compressed_path

        # 暗号化（簡易実装）
        if encrypt:
            encrypted_path = f"{processed_path}.enc"
            # 実際の実装では適切な暗号化ライブラリを使用
            shutil.copy2(processed_path, encrypted_path)
            os.remove(processed_path)
            processed_path = encrypted_path

        return processed_path

    def _restore_database(self, record: BackupRecord, target_path: str = None) -> bool:
        """データベース復元"""
        try:
            # S3からダウンロード（必要に応じて）
            local_path = record.file_path
            if record.file_path.startswith('s3://'):
                # S3からダウンロード実装
                pass

            # 復号化・展開（必要に応じて）
            if local_path.endswith('.enc'):
                # 復号化実装
                pass

            if local_path.endswith('.gz'):
                # 展開実装
                pass

            # pg_restore実行
            target_db = target_path or "day_trade_ml_restored"

            cmd = [
                'pg_restore',
                '--verbose',
                '--clean',
                '--no-owner',
                '--dbname', target_db,
                local_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"pg_restore failed: {result.stderr}")
                return False

            logger.info(f"Database restored successfully to: {target_db}")
            return True

        except Exception as e:
            logger.error(f"Database restore failed: {str(e)}")
            return False

    def _restore_files(self, record: BackupRecord, target_path: str = None) -> bool:
        """ファイル復元"""
        try:
            # 復元先準備
            restore_path = target_path or f"/restore/{record.backup_id}"
            os.makedirs(restore_path, exist_ok=True)

            # バックアップファイル取得・処理
            local_path = record.file_path

            # 処理済みファイルの逆処理（復号化・展開）
            # 実装省略

            # tar 展開
            with tarfile.open(local_path, 'r') as tar:
                tar.extractall(path=restore_path)

            logger.info(f"Files restored successfully to: {restore_path}")
            return True

        except Exception as e:
            logger.error(f"File restore failed: {str(e)}")
            return False

    def _calculate_checksum(self, file_path: str) -> str:
        """ファイルチェックサム計算"""
        sha256_hash = hashlib.sha256()

        try:
            with open(file_path, 'rb') as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()

        except Exception:
            return ""

    def _calculate_next_run(self, schedule: str) -> datetime:
        """次回実行時刻計算（簡易cron実装）"""
        # 実際の実装ではcronlibなどを使用
        # ここでは簡略化
        from datetime import datetime, timedelta
        return datetime.utcnow() + timedelta(hours=24)  # 24時間後

    def _generate_job_id(self, name: str) -> str:
        """ジョブID生成"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"{name}_{timestamp}"

if __name__ == '__main__':
    # テスト用
    backup_manager = BackupManager()

    # バックアップ状況確認
    status = backup_manager.get_backup_status()
    print(f"Backup Jobs: {len(status)}")

    for job_status in status:
        print(f"- {job_status['name']}: {job_status['status']}")

    # スケジュール実行テスト
    backup_manager.schedule_backup_jobs()