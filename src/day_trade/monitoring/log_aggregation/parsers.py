#!/usr/bin/env python3
"""
ログパーサークラス定義
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

from .enums import LogLevel, LogSource
from .models import LogEntry

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class LogParser(ABC):
    """抽象ログパーサー"""

    @abstractmethod
    def parse(self, raw_log: str) -> Optional[LogEntry]:
        """ログをパース"""
        pass

    @abstractmethod
    def can_parse(self, raw_log: str) -> bool:
        """このパーサーが対象ログをパース可能かチェック"""
        pass


class StructuredLogParser(LogParser):
    """構造化ログ（JSON）パーサー"""

    def parse(self, raw_log: str) -> Optional[LogEntry]:
        """JSON形式ログをパース"""
        try:
            data = json.loads(raw_log.strip())

            # 必須フィールドチェック
            if not all(key in data for key in ["timestamp", "level", "message"]):
                return None

            # タイムスタンプ解析
            timestamp = self._parse_timestamp(data["timestamp"])
            if not timestamp:
                return None

            # ログレベル解析
            try:
                level = LogLevel(data["level"].upper())
            except ValueError:
                level = LogLevel.INFO

            # ソース推定
            source = self._infer_source(data)

            # エントリ作成
            return LogEntry(
                id=f"log_{int(time.time() * 1000000)}_{hash(raw_log[:100])}",
                timestamp=timestamp,
                level=level,
                source=source,
                component=data.get("logger_name", data.get("component", "unknown")),
                message=data["message"],
                structured_data={
                    k: v
                    for k, v in data.items()
                    if k not in ["timestamp", "level", "message", "logger_name"]
                },
                tags=data.get("tags", []),
                trace_id=data.get("trace_id"),
                user_id=data.get("user_id"),
                session_id=data.get("session_id"),
                raw_log=raw_log,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"JSON ログパースエラー: {e}")
            return None

    def can_parse(self, raw_log: str) -> bool:
        """JSON形式かチェック"""
        stripped = raw_log.strip()
        return stripped.startswith("{") and stripped.endswith("}")

    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """タイムスタンプ文字列を解析"""
        try:
            # ISO形式
            if "T" in timestamp_str:
                return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            # その他の形式を試す
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y/%m/%d %H:%M:%S",
                "%d/%m/%Y %H:%M:%S",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue

            return None

        except Exception as e:
            logger.debug(f"タイムスタンプ解析エラー: {e}")
            return None

    def _infer_source(self, data: Dict[str, Any]) -> LogSource:
        """ログデータからソースを推定"""
        logger_name = data.get("logger_name", "").lower()
        component = data.get("component", "").lower()
        message = data.get("message", "").lower()

        # パターンマッチングでソース推定
        if any(
            keyword in logger_name + component + message
            for keyword in ["database", "db", "sql", "query"]
        ):
            return LogSource.DATABASE

        elif any(
            keyword in logger_name + component + message
            for keyword in ["api", "http", "rest", "endpoint"]
        ):
            return LogSource.API

        elif any(
            keyword in logger_name + component + message
            for keyword in ["trading", "trade", "order", "portfolio"]
        ):
            return LogSource.TRADING

        elif any(
            keyword in logger_name + component + message
            for keyword in ["ml", "model", "prediction", "training"]
        ):
            return LogSource.ML

        elif any(
            keyword in logger_name + component + message
            for keyword in ["security", "auth", "login", "permission"]
        ):
            return LogSource.SECURITY

        elif any(
            keyword in logger_name + component + message
            for keyword in ["performance", "metric", "cpu", "memory"]
        ):
            return LogSource.PERFORMANCE

        elif any(
            keyword in logger_name + component + message
            for keyword in ["system", "os", "disk", "network"]
        ):
            return LogSource.SYSTEM

        else:
            return LogSource.APPLICATION


class StandardLogParser(LogParser):
    """標準形式ログパーサー"""

    def __init__(self):
        # 標準ログ形式のパターン
        self.patterns = [
            # 2024-01-01 12:00:00 - component - LEVEL - message
            re.compile(
                r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*-\s*(?P<component>[^-]+)\s*-\s*(?P<level>\w+)\s*-\s*(?P<message>.*)"
            ),
            # [2024-01-01 12:00:00] LEVEL: message
            re.compile(
                r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*(?P<level>\w+):\s*(?P<message>.*)"
            ),
            # 2024-01-01 12:00:00 LEVEL message
            re.compile(
                r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(?P<level>\w+)\s+(?P<message>.*)"
            ),
        ]

    def parse(self, raw_log: str) -> Optional[LogEntry]:
        """標準形式ログをパース"""
        try:
            for pattern in self.patterns:
                match = pattern.match(raw_log.strip())
                if match:
                    groups = match.groupdict()

                    # タイムスタンプ解析
                    try:
                        timestamp = datetime.strptime(
                            groups["timestamp"], "%Y-%m-%d %H:%M:%S"
                        )
                    except ValueError:
                        continue

                    # ログレベル解析
                    try:
                        level = LogLevel(groups["level"].upper())
                    except ValueError:
                        level = LogLevel.INFO

                    return LogEntry(
                        id=f"log_{int(time.time() * 1000000)}_{hash(raw_log[:100])}",
                        timestamp=timestamp,
                        level=level,
                        source=LogSource.APPLICATION,
                        component=groups.get("component", "unknown"),
                        message=groups["message"],
                        raw_log=raw_log,
                    )

            return None

        except Exception as e:
            logger.debug(f"標準ログパースエラー: {e}")
            return None

    def can_parse(self, raw_log: str) -> bool:
        """標準形式かチェック"""
        return any(pattern.match(raw_log.strip()) for pattern in self.patterns)