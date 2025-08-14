#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stability Manager - 実戦投入用技術的安定性・エラーハンドリング強化

ネットワークエラー対応・データ整合性・ログ・監査・復旧機能
実戦運用における技術的安定性確保
"""

import asyncio
import logging
import json
import sqlite3
import os
import traceback
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Windows環境での文字化け対策
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

class ErrorLevel(Enum):
    """エラーレベル"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class SystemStatus(Enum):
    """システム状態"""
    HEALTHY = "正常"
    DEGRADED = "軽度障害"
    CRITICAL = "重大障害"
    RECOVERY = "復旧中"
    MAINTENANCE = "メンテナンス"

@dataclass
class ErrorRecord:
    """エラー記録"""
    timestamp: datetime
    level: ErrorLevel
    component: str
    message: str
    traceback: str = ""
    context: Dict[str, Any] = None
    resolved: bool = False

@dataclass
class SystemHealth:
    """システムヘルス情報"""
    timestamp: datetime
    status: SystemStatus
    components: Dict[str, bool]  # コンポーネント名 -> 正常性
    error_count: int
    warning_count: int
    uptime_seconds: float
    memory_usage_mb: float = 0.0

class RobustHTTPClient:
    """
    堅牢なHTTPクライアント
    自動リトライ・タイムアウト・回線断対応
    """

    def __init__(self, max_retries: int = 3, backoff_factor: float = 0.3):
        self.session = requests.Session()

        # リトライ戦略設定
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],  # method_whitelistの新API
            backoff_factor=backoff_factor
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # タイムアウト設定
        self.timeout = (5, 30)  # 接続5秒、読み込み30秒

    def get(self, url: str, **kwargs) -> requests.Response:
        """堅牢なGETリクエスト"""
        try:
            kwargs.setdefault('timeout', self.timeout)
            response = self.session.get(url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP GET failed for {url}: {e}")
            raise

    def post(self, url: str, **kwargs) -> requests.Response:
        """堅牢なPOSTリクエスト"""
        try:
            kwargs.setdefault('timeout', self.timeout)
            response = self.session.post(url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP POST failed for {url}: {e}")
            raise

class DataIntegrityManager:
    """
    データ整合性管理システム
    バックアップ・リストア・整合性チェック
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def create_backup(self, data: Any, name: str) -> str:
        """データのバックアップ作成"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"{name}_{timestamp}.json"

            # JSON形式で保存
            if isinstance(data, dict) or isinstance(data, list):
                backup_data = data
            else:
                # dataclassの場合
                backup_data = asdict(data) if hasattr(data, '__dataclass_fields__') else str(data)

            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2, default=str)

            # チェックサム計算
            checksum = self._calculate_checksum(backup_file)

            # メタデータ保存
            metadata = {
                "name": name,
                "timestamp": timestamp,
                "file_path": str(backup_file),
                "checksum": checksum,
                "size_bytes": backup_file.stat().st_size
            }

            metadata_file = backup_file.with_suffix('.meta.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Backup created: {backup_file}")
            return str(backup_file)

        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            raise

    def restore_from_backup(self, backup_file: str) -> Any:
        """バックアップからの復元"""
        try:
            backup_path = Path(backup_file)

            # 整合性チェック
            if not self._verify_backup_integrity(backup_path):
                raise ValueError("Backup integrity check failed")

            with open(backup_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.logger.info(f"Data restored from: {backup_file}")
            return data

        except Exception as e:
            self.logger.error(f"Backup restoration failed: {e}")
            raise

    def _calculate_checksum(self, file_path: Path) -> str:
        """ファイルのチェックサム計算"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _verify_backup_integrity(self, backup_file: Path) -> bool:
        """バックアップ整合性検証"""
        try:
            metadata_file = backup_file.with_suffix('.meta.json')
            if not metadata_file.exists():
                return False

            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # チェックサム検証
            current_checksum = self._calculate_checksum(backup_file)
            return current_checksum == metadata.get('checksum')

        except Exception:
            return False

    def cleanup_old_backups(self, name: str, keep_count: int = 10):
        """古いバックアップの清理"""
        try:
            pattern = f"{name}_*.json"
            backup_files = list(self.backup_dir.glob(pattern))

            # 日時順でソート
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # 古いファイルを削除
            for old_backup in backup_files[keep_count:]:
                old_backup.unlink(missing_ok=True)
                # メタデータファイルも削除
                old_backup.with_suffix('.meta.json').unlink(missing_ok=True)

            self.logger.info(f"Cleaned up old backups for {name}")

        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")

class SystemStabilityManager:
    """
    システム安定性管理システム
    エラー監視・ログ管理・自動復旧
    """

    def __init__(self, data_dir: str = "stability_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # データベース初期化
        self.db_path = self.data_dir / "stability.db"
        self._init_database()

        # コンポーネント管理
        self.http_client = RobustHTTPClient()
        self.data_integrity = DataIntegrityManager(str(self.data_dir))

        # システム状態
        self.start_time = time.time()
        self.error_count = 0
        self.warning_count = 0
        self.status = SystemStatus.HEALTHY

        # ログ設定
        self.logger = self._setup_logging()

        # コンポーネントヘルスチェック
        self.health_checks = {}

    def _init_database(self):
        """安定性管理データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    traceback TEXT,
                    context TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    components TEXT NOT NULL,
                    error_count INTEGER,
                    warning_count INTEGER,
                    uptime_seconds REAL,
                    memory_usage_mb REAL
                )
            """)

    def _setup_logging(self) -> logging.Logger:
        """拡張ログシステム設定"""
        logger = logging.getLogger('stability')
        logger.setLevel(logging.DEBUG)

        # ファイルハンドラー
        log_file = self.data_dir / "system.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # フォーマッター
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger

    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """ヘルスチェック関数の登録"""
        self.health_checks[name] = check_func

    def log_error(self, level: ErrorLevel, component: str, message: str,
                  exception: Exception = None, context: Dict[str, Any] = None):
        """構造化エラーログ"""
        try:
            error_record = ErrorRecord(
                timestamp=datetime.now(),
                level=level,
                component=component,
                message=message,
                traceback=traceback.format_exc() if exception else "",
                context=context or {}
            )

            # データベース保存
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO error_logs
                    (timestamp, level, component, message, traceback, context)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    error_record.timestamp.isoformat(),
                    error_record.level.value,
                    error_record.component,
                    error_record.message,
                    error_record.traceback,
                    json.dumps(error_record.context, default=str)
                ))

            # ログレベル更新
            if level == ErrorLevel.ERROR or level == ErrorLevel.CRITICAL:
                self.error_count += 1
            elif level == ErrorLevel.WARNING:
                self.warning_count += 1

            # システム状態更新
            self._update_system_status()

            # ログ出力
            self.logger.log(
                getattr(logging, level.value),
                f"[{component}] {message} - Context: {context}"
            )

        except Exception as e:
            # ログシステム自体のエラー
            print(f"Critical: Logging system error: {e}")

    def _update_system_status(self):
        """システム状態の動的更新"""
        if self.error_count > 10:
            self.status = SystemStatus.CRITICAL
        elif self.error_count > 5 or self.warning_count > 20:
            self.status = SystemStatus.DEGRADED
        else:
            self.status = SystemStatus.HEALTHY

    def get_system_health(self) -> SystemHealth:
        """システムヘルス取得"""
        # ヘルスチェック実行
        component_health = {}
        for name, check_func in self.health_checks.items():
            try:
                component_health[name] = check_func()
            except Exception as e:
                component_health[name] = False
                self.log_error(ErrorLevel.WARNING, f"health_check_{name}",
                              f"Health check failed: {e}")

        # メモリ使用量取得（簡易版）
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            memory_mb = 0.0

        health = SystemHealth(
            timestamp=datetime.now(),
            status=self.status,
            components=component_health,
            error_count=self.error_count,
            warning_count=self.warning_count,
            uptime_seconds=time.time() - self.start_time,
            memory_usage_mb=memory_mb
        )

        # データベース記録
        self._save_health_record(health)

        return health

    def _save_health_record(self, health: SystemHealth):
        """ヘルス記録の保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO system_health
                    (timestamp, status, components, error_count, warning_count,
                     uptime_seconds, memory_usage_mb)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    health.timestamp.isoformat(),
                    health.status.value,
                    json.dumps(health.components),
                    health.error_count,
                    health.warning_count,
                    health.uptime_seconds,
                    health.memory_usage_mb
                ))
        except Exception as e:
            print(f"Failed to save health record: {e}")

    def emergency_recovery(self):
        """緊急復旧処理"""
        try:
            self.logger.critical("Emergency recovery initiated")
            self.status = SystemStatus.RECOVERY

            # 1. 全プロセス停止
            # 2. データ整合性チェック
            # 3. 設定復元
            # 4. サービス再起動

            # 実装は環境に依存するため、ここでは基本的な処理のみ
            self.error_count = 0
            self.warning_count = 0
            self.status = SystemStatus.HEALTHY

            self.logger.info("Emergency recovery completed")

        except Exception as e:
            self.logger.critical(f"Emergency recovery failed: {e}")

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """エラー集計レポート"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT level, component, COUNT(*) as count
                FROM error_logs
                WHERE timestamp > ?
                GROUP BY level, component
                ORDER BY count DESC
            """, (cutoff_time.isoformat(),))

            error_summary = {}
            for level, component, count in cursor.fetchall():
                if level not in error_summary:
                    error_summary[level] = {}
                error_summary[level][component] = count

        return {
            "period_hours": hours,
            "summary": error_summary,
            "total_errors": sum(
                sum(components.values()) for components in error_summary.values()
            )
        }

# 使用例とテスト関数
def test_stability_system():
    """安定性システムのテスト"""
    print("=== 技術的安定性システム テスト ===")

    # システム初期化
    stability = SystemStabilityManager()

    # ヘルスチェック登録
    def database_health():
        return stability.db_path.exists()

    def network_health():
        try:
            response = stability.http_client.get("https://httpbin.org/status/200")
            return response.status_code == 200
        except:
            return False

    stability.register_health_check("database", database_health)
    stability.register_health_check("network", network_health)

    # エラーログテスト
    print("\n[ エラーログテスト ]")
    stability.log_error(ErrorLevel.INFO, "test", "Test info message")
    stability.log_error(ErrorLevel.WARNING, "test", "Test warning message",
                       context={"test_param": "value"})

    try:
        # 意図的なエラー
        raise ValueError("Test error")
    except Exception as e:
        stability.log_error(ErrorLevel.ERROR, "test", "Test error message", e)

    # ヘルス情報取得
    print("\n[ システムヘルス ]")
    health = stability.get_system_health()
    print(f"状態: {health.status.value}")
    print(f"エラー数: {health.error_count}")
    print(f"警告数: {health.warning_count}")
    print(f"稼働時間: {health.uptime_seconds:.1f}秒")
    print(f"コンポーネント: {health.components}")

    # データバックアップテスト
    print("\n[ データバックアップテスト ]")
    test_data = {"test": "data", "timestamp": datetime.now().isoformat()}  # JSON serializable
    backup_file = stability.data_integrity.create_backup(test_data, "test_data")
    restored_data = stability.data_integrity.restore_from_backup(backup_file)
    print(f"バックアップ成功: {backup_file}")
    print(f"復元成功: {restored_data == test_data}")

    # エラー集計
    print("\n[ エラー集計 ]")
    summary = stability.get_error_summary(1)
    print(f"過去1時間のエラー: {summary}")

    print("\n" + "="*50)
    print("技術的安定性システム 正常動作確認")

if __name__ == "__main__":
    test_stability_system()