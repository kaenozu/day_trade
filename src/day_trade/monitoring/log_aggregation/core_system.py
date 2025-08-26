#!/usr/bin/env python3
"""
統合ログ集約・分析システムのメインクラス
"""

import asyncio
import logging
import threading
import time
import warnings
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .analytics import LogAnalytics
from .database import LogDatabase
from .enums import LogLevel, LogSource
from .export_utils import LogExporter
from .models import LogEntry, LogSearchQuery
from .parsers import LogParser, StandardLogParser, StructuredLogParser
from .pattern_detection import PatternDetectionEngine

try:
    from ...utils.logging_config import get_context_logger
    from ...utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    class UnifiedCacheManager:
        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True

    def generate_unified_cache_key(*args, **kwargs):
        return f"log_agg_key_{hash(str(args))}"


logger = get_context_logger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class LogAggregationSystem:
    """統合ログ集約・分析システム"""

    def __init__(
        self,
        storage_path: str = "data/logs",
        enable_cache: bool = True,
        max_memory_logs: int = 10000,
        retention_days: int = 30,
        enable_compression: bool = True,
    ):
        self.storage_path = Path(storage_path)
        self.enable_cache = enable_cache
        self.max_memory_logs = max_memory_logs
        self.retention_days = retention_days
        self.enable_compression = enable_compression

        # ディレクトリ初期化
        self._initialize_directories()

        # コンポーネント初期化
        self.database = LogDatabase(self.storage_path / "logs.db")
        self.pattern_engine = PatternDetectionEngine()
        self.analytics = LogAnalytics(self.storage_path / "logs.db")
        self.exporter = LogExporter(self.storage_path)

        # キャッシュマネージャー初期化
        self._initialize_cache()

        # ログパーサー初期化
        self.parsers: List[LogParser] = [StructuredLogParser(), StandardLogParser()]

        # ログストリーム処理
        self.log_stream = deque(maxlen=max_memory_logs)
        self.log_stream_lock = threading.Lock()

        # ストリーミング処理スレッド
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_processing_flag = threading.Event()

        logger.info("統合ログ集約システム初期化完了")
        logger.info(f"  - ストレージパス: {self.storage_path}")
        logger.info(f"  - メモリログ上限: {max_memory_logs}")
        logger.info(f"  - 保持期間: {retention_days}日")
        logger.info(f"  - 圧縮: {'有効' if enable_compression else '無効'}")

    def _initialize_directories(self):
        """ディレクトリ初期化"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "raw").mkdir(exist_ok=True)
        (self.storage_path / "processed").mkdir(exist_ok=True)
        (self.storage_path / "archives").mkdir(exist_ok=True)
        (self.storage_path / "indexes").mkdir(exist_ok=True)

    def _initialize_cache(self):
        """キャッシュ初期化"""
        if self.enable_cache:
            try:
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=128, l2_memory_mb=512, l3_disk_mb=2048
                )
                logger.info("ログ集約キャッシュシステム初期化完了")
            except Exception as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")
                self.cache_manager = None
        else:
            self.cache_manager = None

    async def start_processing(self):
        """ログストリーム処理開始"""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("ログ処理は既に開始されています")
            return

        logger.info("ログストリーム処理開始")

        self.stop_processing_flag.clear()
        self.processing_thread = threading.Thread(
            target=self._processing_loop, name="LogProcessingThread", daemon=True
        )
        self.processing_thread.start()

    async def stop_processing(self):
        """ログストリーム処理停止"""
        if not self.processing_thread or not self.processing_thread.is_alive():
            logger.warning("ログ処理は既に停止されています")
            return

        logger.info("ログストリーム処理停止中...")

        self.stop_processing_flag.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=10)

        logger.info("ログストリーム処理停止完了")

    def _processing_loop(self):
        """ログ処理メインループ"""
        logger.info("ログ処理ループ開始")

        while not self.stop_processing_flag.is_set():
            try:
                # メモリ内ログのバッチ処理
                logs_to_process = []
                with self.log_stream_lock:
                    if self.log_stream:
                        logs_to_process = list(self.log_stream)
                        self.log_stream.clear()

                if logs_to_process:
                    # パターン検出とアラート処理
                    for log_entry in logs_to_process:
                        self.pattern_engine.process_log_patterns(log_entry)

                    # データベースバッチ保存
                    asyncio.run_coroutine_threadsafe(
                        self.database.batch_save_logs(logs_to_process), 
                        asyncio.new_event_loop()
                    )

                # アラートチェック
                self.pattern_engine.check_pattern_alerts()

                # 定期クリーンアップ
                self._periodic_cleanup()

                # 短い間隔で再チェック
                self.stop_processing_flag.wait(5)  # 5秒間隔

            except Exception as e:
                logger.error(f"ログ処理ループエラー: {e}")
                self.stop_processing_flag.wait(30)  # エラー時は30秒待機

        logger.info("ログ処理ループ終了")

    def _periodic_cleanup(self):
        """定期クリーンアップ"""
        current_time = time.time()
        if (
            not hasattr(self, "_last_cleanup")
            or current_time - self._last_cleanup > 3600
        ):  # 1時間毎
            self._cleanup_old_data()
            self._last_cleanup = current_time

    async def ingest_log(
        self, raw_log: str, source_hint: Optional[LogSource] = None
    ) -> bool:
        """ログを取り込み"""
        try:
            # ログパース
            parsed_log = None
            for parser in self.parsers:
                if parser.can_parse(raw_log):
                    parsed_log = parser.parse(raw_log)
                    if parsed_log:
                        break

            if not parsed_log:
                # パースできない場合は基本的なエントリを作成
                parsed_log = LogEntry(
                    id=f"log_{int(time.time() * 1000000)}_{hash(raw_log[:100])}",
                    timestamp=datetime.utcnow(),
                    level=LogLevel.INFO,
                    source=source_hint or LogSource.APPLICATION,
                    component="unknown",
                    message=raw_log,
                    raw_log=raw_log,
                )

            # メモリストリームに追加
            with self.log_stream_lock:
                self.log_stream.append(parsed_log)

            # キャッシュに保存
            if self.cache_manager:
                cache_key = generate_unified_cache_key("log_entry", parsed_log.id)
                self.cache_manager.put(cache_key, parsed_log, priority=3.0)

            return True

        except Exception as e:
            logger.error(f"ログ取り込みエラー: {e}")
            return False

    async def search_logs(self, query: LogSearchQuery) -> List[LogEntry]:
        """ログ検索"""
        return await self.database.search_logs(query)

    async def get_log_analytics(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """ログ分析データ取得"""
        active_alerts_count = len(self.pattern_engine.active_alerts)
        patterns_enabled_count = sum(
            1 for p in self.pattern_engine.log_patterns.values() if p.enabled
        )
        
        return await self.analytics.get_log_analytics(
            start_time, end_time, active_alerts_count, patterns_enabled_count
        )

    async def export_logs(
        self,
        query: LogSearchQuery,
        export_format: str = "json",
        export_path: Optional[str] = None,
    ) -> str:
        """ログエクスポート"""
        # ログ検索
        logs = await self.search_logs(query)
        
        # エクスポート実行
        return await self.exporter.export_logs(logs, export_format, export_path)

    def _cleanup_old_data(self):
        """古いデータクリーンアップ"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

            # データベースクリーンアップ
            deleted_logs, deleted_alerts = self.database.cleanup_old_data(cutoff_date)

            if deleted_logs > 0 or deleted_alerts > 0:
                logger.info(
                    f"古いデータクリーンアップ: ログ{deleted_logs}件、アラート{deleted_alerts}件削除"
                )

            # 古いアーカイブファイルクリーンアップ
            archive_path = self.storage_path / "archives"
            for archive_file in archive_path.glob("*.gz"):
                if archive_file.stat().st_mtime < cutoff_date.timestamp():
                    archive_file.unlink()

        except Exception as e:
            logger.error(f"データクリーンアップエラー: {e}")

    async def cleanup(self):
        """リソースクリーンアップ"""
        logger.info("統合ログ集約システム クリーンアップ開始")

        # 処理停止
        await self.stop_processing()

        # メモリクリア
        with self.log_stream_lock:
            self.log_stream.clear()

        self.pattern_engine.active_alerts.clear()
        self.pattern_engine.pattern_counters.clear()

        logger.info("統合ログ集約システム クリーンアップ完了")

    # プロパティアクセス（後方互換性）
    @property
    def log_patterns(self):
        """ログパターンへのアクセス"""
        return self.pattern_engine.log_patterns

    @property
    def active_alerts(self):
        """アクティブアラートへのアクセス"""
        return self.pattern_engine.active_alerts