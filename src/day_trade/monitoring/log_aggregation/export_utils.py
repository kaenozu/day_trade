#!/usr/bin/env python3
"""
ログエクスポート機能
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .models import LogEntry, LogSearchQuery

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class LogExporter:
    """ログエクスポートクラス"""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.export_path = storage_path / "exports"
        self.export_path.mkdir(parents=True, exist_ok=True)

    async def export_logs(
        self,
        logs: List[LogEntry],
        export_format: str = "json",
        export_path: Optional[str] = None,
    ) -> str:
        """ログエクスポート"""
        logger.info(f"ログエクスポート開始: {export_format}")

        try:
            # エクスポートファイルパス
            if not export_path:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                export_path = str(
                    self.export_path / f"logs_{timestamp}.{export_format}"
                )

            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)

            if export_format == "json":
                await self._export_json(logs, export_file)
            elif export_format == "csv":
                await self._export_csv(logs, export_file)
            else:
                raise ValueError(f"サポートされていないエクスポート形式: {export_format}")

            logger.info(f"ログエクスポート完了: {export_file} ({len(logs)}件)")
            return str(export_file)

        except Exception as e:
            logger.error(f"ログエクスポートエラー: {e}")
            raise

    async def _export_json(self, logs: List[LogEntry], export_file: Path):
        """JSON形式でエクスポート"""
        log_data = []
        for log in logs:
            log_data.append(
                {
                    "id": log.id,
                    "timestamp": log.timestamp.isoformat(),
                    "level": log.level.value,
                    "source": log.source.value,
                    "component": log.component,
                    "message": log.message,
                    "structured_data": log.structured_data,
                    "tags": log.tags,
                    "trace_id": log.trace_id,
                    "user_id": log.user_id,
                    "session_id": log.session_id,
                }
            )

        with open(export_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    async def _export_csv(self, logs: List[LogEntry], export_file: Path):
        """CSV形式でエクスポート"""
        df_data = []
        for log in logs:
            df_data.append(
                {
                    "timestamp": log.timestamp.isoformat(),
                    "level": log.level.value,
                    "source": log.source.value,
                    "component": log.component,
                    "message": log.message,
                    "trace_id": log.trace_id or "",
                    "user_id": log.user_id or "",
                }
            )

        df = pd.DataFrame(df_data)
        df.to_csv(export_file, index=False, encoding="utf-8")

    async def export_analytics_report(
        self, analytics_data: dict, export_path: Optional[str] = None
    ) -> str:
        """分析レポートをエクスポート"""
        try:
            if not export_path:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                export_path = str(
                    self.export_path / f"analytics_report_{timestamp}.json"
                )

            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)

            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(analytics_data, f, indent=2, ensure_ascii=False)

            logger.info(f"分析レポートエクスポート完了: {export_file}")
            return str(export_file)

        except Exception as e:
            logger.error(f"分析レポートエクスポートエラー: {e}")
            raise

    def get_export_history(self, limit: int = 20) -> List[dict]:
        """エクスポート履歴を取得"""
        try:
            exports = []
            for export_file in self.export_path.glob("*"):
                if export_file.is_file():
                    stat = export_file.stat()
                    exports.append({
                        "filename": export_file.name,
                        "path": str(export_file),
                        "size_bytes": stat.st_size,
                        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })

            # 作成日時でソート（新しい順）
            exports.sort(key=lambda x: x["created_at"], reverse=True)
            return exports[:limit]

        except Exception as e:
            logger.error(f"エクスポート履歴取得エラー: {e}")
            return []

    def cleanup_old_exports(self, days_to_keep: int = 30) -> int:
        """古いエクスポートファイルをクリーンアップ"""
        try:
            cutoff_time = datetime.utcnow().timestamp() - (days_to_keep * 24 * 3600)
            deleted_count = 0

            for export_file in self.export_path.glob("*"):
                if export_file.is_file() and export_file.stat().st_ctime < cutoff_time:
                    export_file.unlink()
                    deleted_count += 1

            logger.info(f"古いエクスポートファイルクリーンアップ: {deleted_count}件削除")
            return deleted_count

        except Exception as e:
            logger.error(f"エクスポートファイルクリーンアップエラー: {e}")
            return 0