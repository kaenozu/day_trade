#!/usr/bin/env python3
"""
Basic Monitor Consistency Checker
基本監視システムの整合性チェック機能

データ整合性監視の実装
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


class ConsistencyCheck(MonitorCheck):
    """データ整合性チェック"""

    def __init__(self, consistency_rules: Dict[str, Any] = None):
        self.consistency_rules = consistency_rules or {}

    async def execute_check(
        self, data_source: str, data: Any, context: Dict[str, Any]
    ) -> Tuple[bool, Optional[MonitorAlert]]:
        """整合性チェック実行"""
        try:
            violations = []

            if isinstance(data, pd.DataFrame):
                violations.extend(self._check_dataframe_consistency(data))
            elif isinstance(data, dict):
                violations.extend(self._check_dict_consistency(data))
            elif isinstance(data, list):
                violations.extend(self._check_list_consistency(data))

            if violations:
                return False, MonitorAlert(
                    alert_id=f"consistency_violation_{int(time.time())}",
                    rule_id="consistency_check",
                    alert_type=AlertType.INTEGRITY_VIOLATION,
                    severity=AlertSeverity.HIGH,
                    title="データ整合性違反",
                    message=f"データソース {data_source} で整合性違反: {', '.join(violations)}",
                    data_source=data_source,
                    triggered_at=datetime.utcnow(),
                    metadata={
                        "violations": violations,
                        "violation_count": len(violations),
                    },
                )

            return True, None

        except Exception as e:
            logger.error(f"整合性チェックエラー {data_source}: {e}")
            return False, MonitorAlert(
                alert_id=f"consistency_error_{int(time.time())}",
                rule_id="consistency_check",
                alert_type=AlertType.SYSTEM_ERROR,
                severity=AlertSeverity.HIGH,
                title="整合性チェックエラー",
                message=f"データソース {data_source} の整合性チェックでエラー: {str(e)}",
                data_source=data_source,
                triggered_at=datetime.utcnow(),
            )

    def _check_dataframe_consistency(self, df: pd.DataFrame) -> List[str]:
        """DataFrameの整合性チェック"""
        violations = []

        try:
            # 価格データの整合性チェック
            if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
                # 価格順序チェック
                invalid_prices = df[
                    (df["Low"] > df["High"])
                    | (df["Low"] > df["Open"])
                    | (df["Low"] > df["Close"])
                    | (df["High"] < df["Open"])
                    | (df["High"] < df["Close"])
                ]

                if len(invalid_prices) > 0:
                    violations.append(f"価格順序異常: {len(invalid_prices)}件")

            # 負の値チェック（Volume等）
            if "Volume" in df.columns:
                negative_volume = df[df["Volume"] < 0]
                if len(negative_volume) > 0:
                    violations.append(f"負の出来高: {len(negative_volume)}件")

            # 重複データチェック
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                violations.append(f"重複データ: {duplicates}件")

            # 異常な欠損率チェック
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if missing_ratio > 0.5:
                violations.append(f"高い欠損率: {missing_ratio:.2%}")

        except Exception as e:
            violations.append(f"整合性チェック処理エラー: {str(e)}")

        return violations

    def _check_dict_consistency(self, data: Dict[str, Any]) -> List[str]:
        """辞書データの整合性チェック"""
        violations = []

        try:
            # センチメント データの整合性
            if (
                "positive_ratio" in data
                and "negative_ratio" in data
                and "neutral_ratio" in data
            ):
                total_ratio = (
                    data["positive_ratio"]
                    + data["negative_ratio"]
                    + data["neutral_ratio"]
                )
                if abs(total_ratio - 1.0) > 0.01:  # 許容誤差1%
                    violations.append(f"センチメント比率合計異常: {total_ratio:.3f}")

            # 範囲チェック
            range_checks = {
                "overall_sentiment": (-1.0, 1.0),
                "positive_ratio": (0.0, 1.0),
                "negative_ratio": (0.0, 1.0),
                "interest_rate": (-10.0, 50.0),  # 金利範囲
                "inflation_rate": (-5.0, 30.0),  # インフレ率範囲
            }

            for field, (min_val, max_val) in range_checks.items():
                if field in data and isinstance(data[field], (int, float)):
                    value = data[field]
                    if not (min_val <= value <= max_val):
                        violations.append(
                            f"{field}範囲外: {value} (範囲: {min_val}-{max_val})"
                        )

        except Exception as e:
            violations.append(f"辞書整合性チェックエラー: {str(e)}")

        return violations

    def _check_list_consistency(self, data: List[Any]) -> List[str]:
        """リストデータの整合性チェック"""
        violations = []

        try:
            if not data:
                violations.append("空のリストデータ")
                return violations

            # 重複チェック（ニュースデータ想定）
            if isinstance(data[0], dict) and "title" in data[0]:
                titles = [item.get("title", "") for item in data]
                unique_titles = set(titles)
                if len(titles) != len(unique_titles):
                    duplicate_count = len(titles) - len(unique_titles)
                    violations.append(f"重複タイトル: {duplicate_count}件")

        except Exception as e:
            violations.append(f"リスト整合性チェックエラー: {str(e)}")

        return violations

    def get_check_info(self) -> Dict[str, Any]:
        return {
            "check_type": "consistency",
            "rules": self.consistency_rules,
            "version": "1.0",
        }