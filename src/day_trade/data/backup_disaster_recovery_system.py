#!/usr/bin/env python3
"""
バックアップ・災害復旧システム
Issue #317: 高速データ管理システム - Phase 4

自動バックアップ・多重化・災害復旧システム
- 自動バックアップシステム
- 多重化・地理的分散
- 災害復旧手順自動化
- データ整合性検証
"""

import asyncio
import hashlib
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class BackupType(Enum):
    """バックアップ種類"""

    FULL = "full"  # フルバックアップ
    INCREMENTAL = "incremental"  # 増分バックアップ
    DIFFERENTIAL = "differential"  # 差分バックアップ
    SNAPSHOT = "snapshot"  # スナップショット


class BackupStatus(Enum):
    """バックアップステータス"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


class RecoveryStatus(Enum):
    """復旧ステータス"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class BackupConfig:
    """バックアップ設定"""

    # スケジュール設定
    full_backup_interval_days: int = 7
    incremental_backup_interval_hours: int = 6
    differential_backup_interval_days: int = 1

    # 保持期間設定
    full_backup_retention_weeks: int = 12
    incremental_backup_retention_days: int = 30
    differential_backup_retention_weeks: int = 4

    # ストレージ設定
    primary_backup_location: str = "backups/primary"
    secondary_backup_location: str = "backups/secondary"
    offsite_backup_location: str = "backups/offsite"

    # 圧縮・暗号化設定
    enable_compression: bool = True
    compression_level: int = 6
    enable_encryption: bool = True
    encryption_key: str = ""

    # 性能設定
    max_concurrent_backups: int = 3
    backup_timeout_hours: int = 24
    verification_enabled: bool = True

    # 災害復旧設定
    rto_minutes: int = 60  # Recovery Time Objective
    rpo_minutes: int = 15  # Recovery Point Objective


@dataclass
class BackupMetadata:
    """バックアップメタデータ"""

    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    start_time: datetime
    end_time: Optional[datetime]
    backup_size_bytes: int
    compressed_size_bytes: int
    file_count: int
    checksum: str
    location: str
    parent_backup_id: Optional[str] = None  # 増分/差分の場合
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryPlan:
    """復旧計画"""

    plan_id: str
    target_time: datetime
    required_backups: List[str]
    estimated_recovery_time_minutes: int
    recovery_steps: List[Dict[str, Any]]
    verification_steps: List[Dict[str, Any]]
    rollback_plan: Optional[Dict[str, Any]] = None


class BackupDisasterRecoverySystem:
    """バックアップ・災害復旧システム"""

    def __init__(self, config: Optional[BackupConfig] = None):
        self.config = config or BackupConfig()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_backups)

        # 状態管理
        self.backup_catalog: Dict[str, BackupMetadata] = {}
        self.active_backups: Dict[str, asyncio.Task] = {}
        self.recovery_history: List[Dict[str, Any]] = []

        # スケジュール管理
        self.scheduler_task: Optional[asyncio.Task] = None
        self._is_running = False

        self._initialize_backup_locations()

    def _initialize_backup_locations(self) -> None:
        """バックアップ場所初期化"""
        locations = [
            self.config.primary_backup_location,
            self.config.secondary_backup_location,
            self.config.offsite_backup_location,
        ]

        for location in locations:
            Path(location).mkdir(parents=True, exist_ok=True)

    async def start_system(self) -> None:
        """システム開始"""
        self._is_running = True

        # スケジューラ開始
        self.scheduler_task = asyncio.create_task(self._backup_scheduler())

        # カタログ読み込み
        await self._load_backup_catalog()

        logger.info("バックアップ・災害復旧システム開始")

    async def stop_system(self) -> None:
        """システム停止"""
        self._is_running = False

        # スケジューラ停止
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        # アクティブバックアップ停止
        for backup_id, task in self.active_backups.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # カタログ保存
        await self._save_backup_catalog()

        logger.info("バックアップ・災害復旧システム停止")

    async def create_backup(
        self,
        backup_type: BackupType,
        source_paths: List[str],
        backup_name: Optional[str] = None,
    ) -> str:
        """バックアップ作成"""
        backup_id = self._generate_backup_id(backup_type, backup_name)

        if backup_id in self.active_backups:
            raise ValueError(f"バックアップが既に実行中です: {backup_id}")

        # バックアップメタデータ初期化
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            status=BackupStatus.PENDING,
            start_time=datetime.now(),
            end_time=None,
            backup_size_bytes=0,
            compressed_size_bytes=0,
            file_count=0,
            checksum="",
            location=self.config.primary_backup_location,
        )

        self.backup_catalog[backup_id] = metadata

        # バックアップタスク開始
        backup_task = asyncio.create_task(self._execute_backup(backup_id, source_paths))
        self.active_backups[backup_id] = backup_task

        logger.info(f"バックアップ開始: {backup_id}, タイプ: {backup_type.value}")
        return backup_id

    async def _execute_backup(self, backup_id: str, source_paths: List[str]) -> None:
        """バックアップ実行"""
        metadata = self.backup_catalog[backup_id]

        try:
            metadata.status = BackupStatus.RUNNING
            metadata.start_time = datetime.now()

            # バックアップファイルパス
            backup_file = Path(metadata.location) / f"{backup_id}.backup.zip"

            # フル/増分/差分に応じた処理
            if metadata.backup_type == BackupType.FULL:
                await self._create_full_backup(backup_id, source_paths, backup_file)
            elif metadata.backup_type == BackupType.INCREMENTAL:
                await self._create_incremental_backup(backup_id, source_paths, backup_file)
            elif metadata.backup_type == BackupType.DIFFERENTIAL:
                await self._create_differential_backup(backup_id, source_paths, backup_file)
            elif metadata.backup_type == BackupType.SNAPSHOT:
                await self._create_snapshot_backup(backup_id, source_paths, backup_file)

            # バックアップ完了処理
            await self._finalize_backup(backup_id, backup_file)

            # 複製作成
            if metadata.backup_type == BackupType.FULL:
                await self._replicate_backup(backup_id, backup_file)

            metadata.status = BackupStatus.COMPLETED
            metadata.end_time = datetime.now()

            logger.info(f"バックアップ完了: {backup_id}")

        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.end_time = datetime.now()
            logger.error(f"バックアップエラー: {backup_id}, {e}")

        finally:
            # アクティブバックアップから除去
            if backup_id in self.active_backups:
                del self.active_backups[backup_id]

    async def _create_full_backup(
        self, backup_id: str, source_paths: List[str], backup_file: Path
    ) -> None:
        """フルバックアップ作成"""
        metadata = self.backup_catalog[backup_id]

        with zipfile.ZipFile(
            backup_file,
            "w",
            compression=(
                zipfile.ZIP_DEFLATED if self.config.enable_compression else zipfile.ZIP_STORED
            ),
            compresslevel=self.config.compression_level,
        ) as zf:
            total_size = 0
            file_count = 0

            for source_path in source_paths:
                source = Path(source_path)

                if source.is_file():
                    # ファイル追加
                    zf.write(source, source.name)
                    total_size += source.stat().st_size
                    file_count += 1

                elif source.is_dir():
                    # ディレクトリ追加（再帰）
                    for file_path in source.rglob("*"):
                        if file_path.is_file():
                            arcname = file_path.relative_to(source.parent)
                            zf.write(file_path, str(arcname))
                            total_size += file_path.stat().st_size
                            file_count += 1

        metadata.backup_size_bytes = total_size
        metadata.compressed_size_bytes = backup_file.stat().st_size
        metadata.file_count = file_count

    async def _create_incremental_backup(
        self, backup_id: str, source_paths: List[str], backup_file: Path
    ) -> None:
        """増分バックアップ作成"""
        # 最新のフル/増分バックアップを検索
        last_backup = self._find_last_backup([BackupType.FULL, BackupType.INCREMENTAL])

        if not last_backup:
            # 増分バックアップできない場合はフルバックアップ
            await self._create_full_backup(backup_id, source_paths, backup_file)
            return

        metadata = self.backup_catalog[backup_id]
        metadata.parent_backup_id = last_backup.backup_id

        last_backup_time = last_backup.start_time

        with zipfile.ZipFile(
            backup_file,
            "w",
            compression=(
                zipfile.ZIP_DEFLATED if self.config.enable_compression else zipfile.ZIP_STORED
            ),
            compresslevel=self.config.compression_level,
        ) as zf:
            total_size = 0
            file_count = 0

            for source_path in source_paths:
                source = Path(source_path)

                if source.is_file():
                    # 変更時刻チェック
                    if datetime.fromtimestamp(source.stat().st_mtime) > last_backup_time:
                        zf.write(source, source.name)
                        total_size += source.stat().st_size
                        file_count += 1

                elif source.is_dir():
                    # ディレクトリ内変更ファイル
                    for file_path in source.rglob("*"):
                        if (
                            file_path.is_file()
                            and datetime.fromtimestamp(file_path.stat().st_mtime) > last_backup_time
                        ):
                            arcname = file_path.relative_to(source.parent)
                            zf.write(file_path, str(arcname))
                            total_size += file_path.stat().st_size
                            file_count += 1

        metadata.backup_size_bytes = total_size
        metadata.compressed_size_bytes = backup_file.stat().st_size
        metadata.file_count = file_count

    async def _create_differential_backup(
        self, backup_id: str, source_paths: List[str], backup_file: Path
    ) -> None:
        """差分バックアップ作成"""
        # 最新のフルバックアップを検索
        last_full_backup = self._find_last_backup([BackupType.FULL])

        if not last_full_backup:
            # 差分バックアップできない場合はフルバックアップ
            await self._create_full_backup(backup_id, source_paths, backup_file)
            return

        metadata = self.backup_catalog[backup_id]
        metadata.parent_backup_id = last_full_backup.backup_id

        last_full_time = last_full_backup.start_time

        # 差分バックアップ実行（フルバックアップ以降の変更）
        with zipfile.ZipFile(
            backup_file,
            "w",
            compression=(
                zipfile.ZIP_DEFLATED if self.config.enable_compression else zipfile.ZIP_STORED
            ),
            compresslevel=self.config.compression_level,
        ) as zf:
            total_size = 0
            file_count = 0

            for source_path in source_paths:
                source = Path(source_path)

                if source.is_dir():
                    for file_path in source.rglob("*"):
                        if (
                            file_path.is_file()
                            and datetime.fromtimestamp(file_path.stat().st_mtime) > last_full_time
                        ):
                            arcname = file_path.relative_to(source.parent)
                            zf.write(file_path, str(arcname))
                            total_size += file_path.stat().st_size
                            file_count += 1

        metadata.backup_size_bytes = total_size
        metadata.compressed_size_bytes = backup_file.stat().st_size
        metadata.file_count = file_count

    async def _create_snapshot_backup(
        self, backup_id: str, source_paths: List[str], backup_file: Path
    ) -> None:
        """スナップショットバックアップ作成"""
        # スナップショット（時点コピー）
        await self._create_full_backup(backup_id, source_paths, backup_file)

    async def _finalize_backup(self, backup_id: str, backup_file: Path) -> None:
        """バックアップ完了処理"""
        metadata = self.backup_catalog[backup_id]

        # チェックサム計算
        metadata.checksum = await self._calculate_file_checksum(backup_file)

        # 検証実行
        if self.config.verification_enabled:
            await self._verify_backup(backup_id, backup_file)

    async def _verify_backup(self, backup_id: str, backup_file: Path) -> bool:
        """バックアップ検証"""
        try:
            # ファイル整合性チェック
            with zipfile.ZipFile(backup_file, "r") as zf:
                # ZIP整合性テスト
                bad_files = zf.testzip()
                if bad_files:
                    logger.error(f"バックアップ破損検出: {backup_id}, {bad_files}")
                    return False

            # チェックサム検証
            current_checksum = await self._calculate_file_checksum(backup_file)
            metadata = self.backup_catalog[backup_id]

            if current_checksum != metadata.checksum:
                logger.error(f"チェックサム不整合: {backup_id}")
                return False

            metadata.status = BackupStatus.VERIFIED
            logger.info(f"バックアップ検証成功: {backup_id}")
            return True

        except Exception as e:
            logger.error(f"バックアップ検証エラー: {backup_id}, {e}")
            return False

    async def _replicate_backup(self, backup_id: str, backup_file: Path) -> None:
        """バックアップ複製"""
        try:
            # セカンダリ場所へ複製
            secondary_file = Path(self.config.secondary_backup_location) / backup_file.name
            shutil.copy2(backup_file, secondary_file)

            # オフサイト場所へ複製（重要なバックアップのみ）
            if self.backup_catalog[backup_id].backup_type == BackupType.FULL:
                offsite_file = Path(self.config.offsite_backup_location) / backup_file.name
                shutil.copy2(backup_file, offsite_file)

            logger.info(f"バックアップ複製完了: {backup_id}")

        except Exception as e:
            logger.error(f"バックアップ複製エラー: {backup_id}, {e}")

    async def create_recovery_plan(
        self, target_time: datetime, recovery_point: Optional[datetime] = None
    ) -> RecoveryPlan:
        """復旧計画作成"""
        plan_id = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        recovery_point = recovery_point or target_time

        # 必要なバックアップ特定
        required_backups = self._identify_required_backups(recovery_point)

        # 復旧時間推定
        estimated_time = self._estimate_recovery_time(required_backups)

        # 復旧手順生成
        recovery_steps = self._generate_recovery_steps(required_backups)

        # 検証手順生成
        verification_steps = self._generate_verification_steps()

        plan = RecoveryPlan(
            plan_id=plan_id,
            target_time=target_time,
            required_backups=required_backups,
            estimated_recovery_time_minutes=estimated_time,
            recovery_steps=recovery_steps,
            verification_steps=verification_steps,
            rollback_plan=self._create_rollback_plan(required_backups),
        )

        logger.info(f"復旧計画作成完了: {plan_id}, 推定復旧時間: {estimated_time}分")
        return plan

    def _identify_required_backups(self, recovery_point: datetime) -> List[str]:
        """必要バックアップ特定"""
        required = []

        # 復旧ポイント以前の最新フルバックアップ
        full_backups = [
            (backup_id, metadata)
            for backup_id, metadata in self.backup_catalog.items()
            if (
                metadata.backup_type == BackupType.FULL
                and metadata.status == BackupStatus.VERIFIED
                and metadata.start_time <= recovery_point
            )
        ]

        if not full_backups:
            return []

        # 最新フルバックアップ
        latest_full = max(full_backups, key=lambda x: x[1].start_time)
        required.append(latest_full[0])
        full_backup_time = latest_full[1].start_time

        # フルバックアップ以降の増分バックアップ
        incremental_backups = [
            (backup_id, metadata)
            for backup_id, metadata in self.backup_catalog.items()
            if (
                metadata.backup_type == BackupType.INCREMENTAL
                and metadata.status == BackupStatus.VERIFIED
                and metadata.start_time > full_backup_time
                and metadata.start_time <= recovery_point
            )
        ]

        # 時系列順でソート
        incremental_backups.sort(key=lambda x: x[1].start_time)
        required.extend([backup_id for backup_id, _ in incremental_backups])

        return required

    def _estimate_recovery_time(self, required_backups: List[str]) -> int:
        """復旧時間推定"""
        total_size_gb = sum(
            self.backup_catalog[backup_id].compressed_size_bytes / (1024**3)
            for backup_id in required_backups
        )

        # 経験的推定（1GB=5分）
        base_time = max(int(total_size_gb * 5), 30)

        # バックアップ数による係数
        complexity_factor = 1 + (len(required_backups) - 1) * 0.2

        return int(base_time * complexity_factor)

    def _generate_recovery_steps(self, required_backups: List[str]) -> List[Dict[str, Any]]:
        """復旧手順生成"""
        steps = []

        # 事前準備
        steps.append(
            {
                "step_id": 1,
                "description": "システム停止・復旧環境準備",
                "action": "prepare_recovery_environment",
                "estimated_minutes": 5,
                "critical": True,
            }
        )

        # バックアップ復元（順次）
        for i, backup_id in enumerate(required_backups):
            metadata = self.backup_catalog[backup_id]
            steps.append(
                {
                    "step_id": i + 2,
                    "description": f"{metadata.backup_type.value}バックアップ復元: {backup_id}",
                    "action": "restore_backup",
                    "backup_id": backup_id,
                    "estimated_minutes": max(
                        int(metadata.compressed_size_bytes / (1024**3) * 3), 5
                    ),
                    "critical": True,
                }
            )

        # データ整合性検証
        steps.append(
            {
                "step_id": len(steps) + 1,
                "description": "データ整合性検証",
                "action": "verify_data_integrity",
                "estimated_minutes": 10,
                "critical": True,
            }
        )

        # システム再起動
        steps.append(
            {
                "step_id": len(steps) + 1,
                "description": "システム再起動・動作確認",
                "action": "restart_system_verification",
                "estimated_minutes": 15,
                "critical": True,
            }
        )

        return steps

    def _generate_verification_steps(self) -> List[Dict[str, Any]]:
        """検証手順生成"""
        return [
            {
                "step_id": 1,
                "description": "データベース接続確認",
                "action": "verify_database_connection",
                "timeout_minutes": 5,
            },
            {
                "step_id": 2,
                "description": "データ完整性確認",
                "action": "verify_data_completeness",
                "timeout_minutes": 15,
            },
            {
                "step_id": 3,
                "description": "システム機能確認",
                "action": "verify_system_functionality",
                "timeout_minutes": 20,
            },
        ]

    def _create_rollback_plan(self, required_backups: List[str]) -> Dict[str, Any]:
        """ロールバック計画作成"""
        return {
            "enabled": True,
            "trigger_conditions": [
                "verification_failure",
                "data_corruption_detected",
                "user_initiated",
            ],
            "rollback_steps": [
                {"action": "stop_system", "timeout_minutes": 5},
                {"action": "restore_previous_state", "timeout_minutes": 30},
                {"action": "verify_rollback", "timeout_minutes": 15},
            ],
            "estimated_rollback_time_minutes": 50,
        }

    async def execute_recovery(self, recovery_plan: RecoveryPlan) -> Dict[str, Any]:
        """災害復旧実行"""
        recovery_id = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        recovery_result = {
            "recovery_id": recovery_id,
            "plan_id": recovery_plan.plan_id,
            "status": RecoveryStatus.NOT_STARTED,
            "start_time": datetime.now(),
            "end_time": None,
            "steps_completed": 0,
            "total_steps": len(recovery_plan.recovery_steps),
            "errors": [],
        }

        try:
            recovery_result["status"] = RecoveryStatus.IN_PROGRESS

            # 復旧手順実行
            for step in recovery_plan.recovery_steps:
                try:
                    await self._execute_recovery_step(step, recovery_result)
                    recovery_result["steps_completed"] += 1

                except Exception as e:
                    error_msg = f"ステップ {step['step_id']} エラー: {e}"
                    recovery_result["errors"].append(error_msg)
                    logger.error(error_msg)

                    if step.get("critical", False):
                        recovery_result["status"] = RecoveryStatus.FAILED
                        break

            # 検証実行
            if recovery_result["status"] == RecoveryStatus.IN_PROGRESS:
                verification_success = await self._execute_verification_steps(
                    recovery_plan.verification_steps, recovery_result
                )

                if verification_success:
                    recovery_result["status"] = RecoveryStatus.COMPLETED
                else:
                    recovery_result["status"] = RecoveryStatus.FAILED

            recovery_result["end_time"] = datetime.now()

            # 復旧履歴記録
            self.recovery_history.append(recovery_result.copy())

            logger.info(
                f"災害復旧完了: {recovery_id}, ステータス: {recovery_result['status'].value}"
            )

        except Exception as e:
            recovery_result["status"] = RecoveryStatus.FAILED
            recovery_result["end_time"] = datetime.now()
            recovery_result["errors"].append(f"復旧実行エラー: {e}")
            logger.error(f"災害復旧エラー: {recovery_id}, {e}")

        return recovery_result

    async def _execute_recovery_step(
        self, step: Dict[str, Any], recovery_result: Dict[str, Any]
    ) -> None:
        """復旧ステップ実行"""
        action = step["action"]
        step_id = step["step_id"]

        logger.info(f"復旧ステップ実行開始: {step_id} - {step['description']}")

        if action == "prepare_recovery_environment":
            await self._prepare_recovery_environment()
        elif action == "restore_backup":
            await self._restore_backup_step(step["backup_id"])
        elif action == "verify_data_integrity":
            await self._verify_data_integrity()
        elif action == "restart_system_verification":
            await self._restart_system_verification()
        else:
            logger.warning(f"未知の復旧アクション: {action}")

        logger.info(f"復旧ステップ実行完了: {step_id}")

    async def _prepare_recovery_environment(self) -> None:
        """復旧環境準備"""
        # システム停止確認
        logger.info("システム停止確認")

        # 復旧作業ディレクトリ作成
        recovery_dir = Path("recovery_workspace")
        recovery_dir.mkdir(exist_ok=True)

        logger.info("復旧環境準備完了")

    async def _restore_backup_step(self, backup_id: str) -> None:
        """バックアップ復元ステップ"""
        metadata = self.backup_catalog[backup_id]
        backup_file = Path(metadata.location) / f"{backup_id}.backup.zip"

        # バックアップファイル存在確認
        if not backup_file.exists():
            # セカンダリ場所確認
            secondary_file = Path(self.config.secondary_backup_location) / backup_file.name
            if secondary_file.exists():
                backup_file = secondary_file
            else:
                raise FileNotFoundError(f"バックアップファイルが見つかりません: {backup_id}")

        # 復元実行
        with zipfile.ZipFile(backup_file, "r") as zf:
            zf.extractall("recovery_workspace")

        logger.info(f"バックアップ復元完了: {backup_id}")

    async def _verify_data_integrity(self) -> None:
        """データ整合性検証"""
        logger.info("データ整合性検証実行")
        # 実装：データベース整合性チェック、ファイル検証等

    async def _restart_system_verification(self) -> None:
        """システム再起動・検証"""
        logger.info("システム再起動・動作確認")
        # 実装：システム起動、基本動作確認

    async def _execute_verification_steps(
        self, verification_steps: List[Dict[str, Any]], recovery_result: Dict[str, Any]
    ) -> bool:
        """検証ステップ実行"""
        for step in verification_steps:
            try:
                # 検証ステップ実行（簡易実装）
                logger.info(f"検証実行: {step['description']}")
                await asyncio.sleep(0.1)  # 実際の検証処理に置換

            except Exception as e:
                error_msg = f"検証ステップエラー {step['step_id']}: {e}"
                recovery_result["errors"].append(error_msg)
                return False

        return True

    async def _backup_scheduler(self) -> None:
        """バックアップスケジューラ"""
        while self._is_running:
            try:
                current_time = datetime.now()

                # フルバックアップスケジュール確認
                if await self._should_run_full_backup(current_time):
                    await self._schedule_full_backup()

                # 増分バックアップスケジュール確認
                elif await self._should_run_incremental_backup(current_time):
                    await self._schedule_incremental_backup()

                # 差分バックアップスケジュール確認
                elif await self._should_run_differential_backup(current_time):
                    await self._schedule_differential_backup()

                # 古いバックアップクリーンアップ
                await self._cleanup_old_backups()

                # 1時間待機
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"バックアップスケジューラエラー: {e}")
                await asyncio.sleep(300)  # エラー時は5分待機

    async def _should_run_full_backup(self, current_time: datetime) -> bool:
        """フルバックアップ実行判定"""
        last_full = self._find_last_backup([BackupType.FULL])
        if not last_full:
            return True

        days_since_last = (current_time - last_full.start_time).days
        return days_since_last >= self.config.full_backup_interval_days

    async def _should_run_incremental_backup(self, current_time: datetime) -> bool:
        """増分バックアップ実行判定"""
        last_backup = self._find_last_backup([BackupType.FULL, BackupType.INCREMENTAL])
        if not last_backup:
            return False

        hours_since_last = (current_time - last_backup.start_time).total_seconds() / 3600
        return hours_since_last >= self.config.incremental_backup_interval_hours

    async def _should_run_differential_backup(self, current_time: datetime) -> bool:
        """差分バックアップ実行判定"""
        last_diff = self._find_last_backup([BackupType.DIFFERENTIAL])
        if not last_diff:
            return True

        days_since_last = (current_time - last_diff.start_time).days
        return days_since_last >= self.config.differential_backup_interval_days

    async def _schedule_full_backup(self) -> None:
        """フルバックアップスケジュール"""
        source_paths = ["data", "config", "logs"]  # 設定可能にする
        await self.create_backup(BackupType.FULL, source_paths, "scheduled_full")

    async def _schedule_incremental_backup(self) -> None:
        """増分バックアップスケジュール"""
        source_paths = ["data", "config"]  # 設定可能にする
        await self.create_backup(BackupType.INCREMENTAL, source_paths, "scheduled_incremental")

    async def _schedule_differential_backup(self) -> None:
        """差分バックアップスケジュール"""
        source_paths = ["data", "config"]  # 設定可能にする
        await self.create_backup(BackupType.DIFFERENTIAL, source_paths, "scheduled_differential")

    def _find_last_backup(self, backup_types: List[BackupType]) -> Optional[BackupMetadata]:
        """最新バックアップ検索"""
        candidates = [
            metadata
            for metadata in self.backup_catalog.values()
            if (
                metadata.backup_type in backup_types
                and metadata.status in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]
            )
        ]

        if not candidates:
            return None

        return max(candidates, key=lambda x: x.start_time)

    async def _cleanup_old_backups(self) -> None:
        """古いバックアップクリーンアップ"""
        current_time = datetime.now()
        to_delete = []

        for backup_id, metadata in self.backup_catalog.items():
            should_delete = False

            if metadata.backup_type == BackupType.FULL:
                weeks_old = (current_time - metadata.start_time).days / 7
                should_delete = weeks_old > self.config.full_backup_retention_weeks

            elif metadata.backup_type == BackupType.INCREMENTAL:
                days_old = (current_time - metadata.start_time).days
                should_delete = days_old > self.config.incremental_backup_retention_days

            elif metadata.backup_type == BackupType.DIFFERENTIAL:
                weeks_old = (current_time - metadata.start_time).days / 7
                should_delete = weeks_old > self.config.differential_backup_retention_weeks

            if should_delete:
                to_delete.append(backup_id)

        # 削除実行
        for backup_id in to_delete:
            await self._delete_backup(backup_id)

    async def _delete_backup(self, backup_id: str) -> None:
        """バックアップ削除"""
        if backup_id not in self.backup_catalog:
            return

        metadata = self.backup_catalog[backup_id]

        # ファイル削除
        backup_file = Path(metadata.location) / f"{backup_id}.backup.zip"
        if backup_file.exists():
            backup_file.unlink()

        # 複製ファイル削除
        for location in [
            self.config.secondary_backup_location,
            self.config.offsite_backup_location,
        ]:
            replica_file = Path(location) / backup_file.name
            if replica_file.exists():
                replica_file.unlink()

        # カタログから削除
        del self.backup_catalog[backup_id]

        logger.info(f"古いバックアップ削除: {backup_id}")

    def _generate_backup_id(self, backup_type: BackupType, backup_name: Optional[str]) -> str:
        """バックアップID生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if backup_name:
            return f"{backup_type.value}_{backup_name}_{timestamp}"
        else:
            return f"{backup_type.value}_{timestamp}"

    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """ファイルチェックサム計算"""
        hash_sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    async def _load_backup_catalog(self) -> None:
        """バックアップカタログ読み込み"""
        catalog_file = Path(self.config.primary_backup_location) / "backup_catalog.json"

        if catalog_file.exists():
            try:
                with open(catalog_file, encoding="utf-8") as f:
                    catalog_data = json.load(f)

                # メタデータ復元
                for backup_id, data in catalog_data.items():
                    self.backup_catalog[backup_id] = BackupMetadata(
                        backup_id=data["backup_id"],
                        backup_type=BackupType(data["backup_type"]),
                        status=BackupStatus(data["status"]),
                        start_time=datetime.fromisoformat(data["start_time"]),
                        end_time=(
                            datetime.fromisoformat(data["end_time"]) if data["end_time"] else None
                        ),
                        backup_size_bytes=data["backup_size_bytes"],
                        compressed_size_bytes=data["compressed_size_bytes"],
                        file_count=data["file_count"],
                        checksum=data["checksum"],
                        location=data["location"],
                        parent_backup_id=data.get("parent_backup_id"),
                        metadata=data.get("metadata", {}),
                    )

                logger.info(f"バックアップカタログ読み込み完了: {len(self.backup_catalog)}件")

            except Exception as e:
                logger.error(f"バックアップカタログ読み込みエラー: {e}")

    async def _save_backup_catalog(self) -> None:
        """バックアップカタログ保存"""
        catalog_file = Path(self.config.primary_backup_location) / "backup_catalog.json"

        try:
            catalog_data = {}

            for backup_id, metadata in self.backup_catalog.items():
                catalog_data[backup_id] = {
                    "backup_id": metadata.backup_id,
                    "backup_type": metadata.backup_type.value,
                    "status": metadata.status.value,
                    "start_time": metadata.start_time.isoformat(),
                    "end_time": metadata.end_time.isoformat() if metadata.end_time else None,
                    "backup_size_bytes": metadata.backup_size_bytes,
                    "compressed_size_bytes": metadata.compressed_size_bytes,
                    "file_count": metadata.file_count,
                    "checksum": metadata.checksum,
                    "location": metadata.location,
                    "parent_backup_id": metadata.parent_backup_id,
                    "metadata": metadata.metadata,
                }

            with open(catalog_file, "w", encoding="utf-8") as f:
                json.dump(catalog_data, f, indent=2, ensure_ascii=False)

            logger.info(f"バックアップカタログ保存完了: {len(catalog_data)}件")

        except Exception as e:
            logger.error(f"バックアップカタログ保存エラー: {e}")

    async def get_backup_status(self) -> Dict[str, Any]:
        """バックアップ状況取得"""
        total_backups = len(self.backup_catalog)
        status_counts = {}
        type_counts = {}
        total_size = 0

        for metadata in self.backup_catalog.values():
            # ステータス集計
            status = metadata.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

            # タイプ集計
            backup_type = metadata.backup_type.value
            type_counts[backup_type] = type_counts.get(backup_type, 0) + 1

            # サイズ集計
            total_size += metadata.compressed_size_bytes

        return {
            "total_backups": total_backups,
            "status_distribution": status_counts,
            "type_distribution": type_counts,
            "total_backup_size_gb": total_size / (1024**3),
            "active_backup_jobs": len(self.active_backups),
            "last_recovery_count": len(self.recovery_history),
            "system_running": self._is_running,
        }

    async def cleanup(self) -> None:
        """システムクリーンアップ"""
        await self.stop_system()
        self.executor.shutdown(wait=True)
        logger.info("バックアップ・災害復旧システムクリーンアップ完了")


# テスト用関数
async def test_backup_system():
    """バックアップシステムテスト"""
    config = BackupConfig(
        primary_backup_location="test_backups/primary",
        secondary_backup_location="test_backups/secondary",
        offsite_backup_location="test_backups/offsite",
    )

    system = BackupDisasterRecoverySystem(config)
    await system.start_system()

    # テスト用ディレクトリ・ファイル作成
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)

    test_file = test_dir / "test.txt"
    with open(test_file, "w") as f:
        f.write("テストデータ")

    # フルバックアップテスト
    backup_id = await system.create_backup(BackupType.FULL, ["test_data"], "test_full")

    # バックアップ完了待機
    await asyncio.sleep(2)

    # 復旧計画作成
    recovery_plan = await system.create_recovery_plan(datetime.now())

    # 状況取得
    status = await system.get_backup_status()

    await system.cleanup()

    # テストデータクリーンアップ
    import shutil

    shutil.rmtree("test_data", ignore_errors=True)
    shutil.rmtree("test_backups", ignore_errors=True)

    return backup_id, recovery_plan, status


if __name__ == "__main__":

    async def main():
        backup_id, recovery_plan, status = await test_backup_system()

        print(f"バックアップID: {backup_id}")
        print(f"復旧計画ID: {recovery_plan.plan_id}")
        print(f"推定復旧時間: {recovery_plan.estimated_recovery_time_minutes}分")
        print(f"バックアップ状況: {status}")

    asyncio.run(main())
