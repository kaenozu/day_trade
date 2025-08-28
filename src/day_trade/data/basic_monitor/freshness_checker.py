#!/usr/bin/env python3
"""
Basic Monitor Freshness Checker
基本監視システムの鮮度チェック機能

データ鮮度監視の実装
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .base_checks import MonitorCheck
from .models import AlertSeverity, AlertType, MonitorAlert

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class FreshnessCheck(MonitorCheck):
    """データ鮮度チェック"""

    def __init__(self, threshold_minutes: int = 60):
        self.threshold_minutes = threshold_minutes

    async def execute_check(
        self, data_source: str, data: Any, context: Dict[str, Any]
    ) -> Tuple[bool, Optional[MonitorAlert]]:
        """鮮度チェック実行"""
        try:
            current_time = datetime.utcnow()
            data_timestamp = context.get("data_timestamp")

            if not data_timestamp:
                # データからタイムスタンプ推定
                data_timestamp = self._extract_timestamp(data)

            if not data_timestamp:
                # タイムスタンプが取得できない場合は警告
                return False, MonitorAlert(
                    alert_id=f"freshness_unknown_{int(time.time())}",
                    rule_id="freshness_check",
                    alert_type=AlertType.DATA_STALE,
                    severity=AlertSeverity.MEDIUM,
                    title="データタイムスタンプ不明",
                    message=f"データソース {data_source} のタイムスタンプを取得できません",
                    data_source=data_source,
                    triggered_at=current_time,
                )

            # 鮮度計算
            age_minutes = (current_time - data_timestamp).total_seconds() / 60

            if age_minutes > self.threshold_minutes:
                return False, MonitorAlert(
                    alert_id=f"freshness_violation_{int(time.time())}",
                    rule_id="freshness_check",
                    alert_type=AlertType.DATA_STALE,
                    severity=(
                        AlertSeverity.HIGH
                        if age_minutes > self.threshold_minutes * 2
                        else AlertSeverity.MEDIUM
                    ),
                    title="データ鮮度違反",
                    message=f"データソース {data_source} のデータが古すぎます ({age_minutes:.1f}分前)",
                    data_source=data_source,
                    triggered_at=current_time,
                    metadata={
                        "data_age_minutes": age_minutes,
                        "threshold_minutes": self.threshold_minutes,
                        "data_timestamp": data_timestamp.isoformat(),
                    },
                )

            return True, None

        except Exception as e:
            logger.error(f"鮮度チェックエラー {data_source}: {e}")
            return False, MonitorAlert(
                alert_id=f"freshness_error_{int(time.time())}",
                rule_id="freshness_check",
                alert_type=AlertType.SYSTEM_ERROR,
                severity=AlertSeverity.HIGH,
                title="鮮度チェックエラー",
                message=f"データソース {data_source} の鮮度チェックでエラー: {str(e)}",
                data_source=data_source,
                triggered_at=datetime.utcnow(),
            )

    def _extract_timestamp(self, data: Any) -> Optional[datetime]:
        """データからタイムスタンプ抽出"""
        try:
            if isinstance(data, pd.DataFrame):
                if hasattr(data.index, "max") and hasattr(data.index, "to_pydatetime"):
                    return pd.to_datetime(data.index.max()).to_pydatetime()
                elif "timestamp" in data.columns:
                    return pd.to_datetime(data["timestamp"].max()).to_pydatetime()
                elif "date" in data.columns:
                    return pd.to_datetime(data["date"].max()).to_pydatetime()

            elif isinstance(data, dict):
                if "timestamp" in data:
                    return pd.to_datetime(data["timestamp"]).to_pydatetime()
                elif "date" in data:
                    return pd.to_datetime(data["date"]).to_pydatetime()

            elif isinstance(data, list) and len(data) > 0:
                item = data[0]
                if isinstance(item, dict):
                    if "timestamp" in item:
                        return pd.to_datetime(item["timestamp"]).to_pydatetime()

            return None

        except Exception as e:
            logger.error(f"タイムスタンプ抽出エラー: {e}")
            return None

    def get_check_info(self) -> Dict[str, Any]:
        return {
            "check_type": "freshness",
            "threshold_minutes": self.threshold_minutes,
            "version": "1.0",
        }