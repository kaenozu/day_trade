#!/usr/bin/env python3
"""
データ鮮度・整合性監視システム
Issue #420: データ管理とデータ品質保証メカニズムの強化

リアルタイムデータ品質監視とアラート機能:
- データ鮮度監視
- 整合性チェック
- リアルタイムアラート
- SLA追跡
- ヘルスメトリクス
- 自動回復機能
- 監視ダッシュボード
"""

import asyncio
import json
import logging
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from ..utils.data_quality_manager import DataQualityLevel, DataQualityMetrics
    from ..utils.logging_config import get_context_logger
    from ..utils.unified_cache_manager import (
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
        return f"monitor_key_{hash(str(args))}"

    class DataQualityLevel(Enum):
        EXCELLENT = "excellent"
        GOOD = "good"
        FAIR = "fair"
        POOR = "poor"
        CRITICAL = "critical"


logger = get_context_logger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class MonitorStatus(Enum):
    """監視状態"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AlertSeverity(Enum):
    """アラート重要度"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(Enum):
    """アラート種別"""

    DATA_STALE = "data_stale"
    DATA_MISSING = "data_missing"
    INTEGRITY_VIOLATION = "integrity_violation"
    THRESHOLD_BREACH = "threshold_breach"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SLA_VIOLATION = "sla_violation"


class RecoveryAction(Enum):
    """回復アクション"""

    RETRY_FETCH = "retry_fetch"
    USE_FALLBACK = "use_fallback"
    NOTIFY_ADMIN = "notify_admin"
    DISABLE_SOURCE = "disable_source"
    ESCALATE = "escalate"
    AUTO_FIX = "auto_fix"


@dataclass
class MonitorRule:
    """監視ルール定義"""

    rule_id: str
    name: str
    description: str
    data_source: str
    rule_type: str  # "freshness", "consistency", "completeness", "accuracy"
    threshold_value: float
    threshold_unit: str  # "minutes", "hours", "percentage", "count"
    severity: AlertSeverity
    enabled: bool = True
    check_interval_seconds: int = 300  # 5分
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorAlert:
    """監視アラート"""

    alert_id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    data_source: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    recovery_actions_taken: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSourceHealth:
    """データソースヘルス状態"""

    source_id: str
    source_type: str  # "api", "database", "file", "stream"
    last_update: datetime
    data_age_minutes: float
    quality_score: float
    availability: float  # 0-1
    error_rate: float  # 0-1
    response_time_ms: float
    health_status: str  # "healthy", "warning", "critical", "unknown"
    consecutive_failures: int = 0
    last_error: Optional[str] = None


@dataclass
class SLAMetrics:
    """SLA メトリクス"""

    sla_id: str
    name: str
    target_availability: float  # 目標可用性 (99.9%)
    target_freshness_minutes: int  # 目標データ鮮度（分）
    target_quality_score: float  # 目標品質スコア
    current_availability: float
    current_freshness_minutes: float
    current_quality_score: float
    violations_count: int
    measurement_period: str  # "daily", "weekly", "monthly"
    last_violation: Optional[datetime] = None


class MonitorCheck(ABC):
    """抽象監視チェッククラス"""

    @abstractmethod
    async def execute_check(
        self, data_source: str, data: Any, context: Dict[str, Any]
    ) -> Tuple[bool, Optional[MonitorAlert]]:
        """チェック実行"""
        pass

    @abstractmethod
    def get_check_info(self) -> Dict[str, Any]:
        """チェック情報取得"""
        pass


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
            if "positive_ratio" in data and "negative_ratio" in data and "neutral_ratio" in data:
                total_ratio = (
                    data["positive_ratio"] + data["negative_ratio"] + data["neutral_ratio"]
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
                        violations.append(f"{field}範囲外: {value} (範囲: {min_val}-{max_val})")

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


class DataFreshnessMonitor:
    """データ鮮度・整合性監視システム"""

    def __init__(
        self,
        storage_path: str = "data/monitoring",
        enable_cache: bool = True,
        alert_retention_days: int = 30,
        check_interval_seconds: int = 300,
    ):
        self.storage_path = Path(storage_path)
        self.enable_cache = enable_cache
        self.alert_retention_days = alert_retention_days
        self.check_interval_seconds = check_interval_seconds

        # ディレクトリ初期化
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # キャッシュマネージャー初期化
        if enable_cache:
            try:
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=32, l2_memory_mb=128, l3_disk_mb=256
                )
                logger.info("監視システムキャッシュ初期化完了")
            except Exception as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")
                self.cache_manager = None
        else:
            self.cache_manager = None

        # 監視状態管理
        self.monitor_status = MonitorStatus.INACTIVE
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # 監視ルール管理
        self.monitor_rules: Dict[str, MonitorRule] = {}

        # チェック実装
        self.checks: Dict[str, MonitorCheck] = {
            "freshness": FreshnessCheck(threshold_minutes=60),
            "consistency": ConsistencyCheck(),
        }

        # データソースヘルス管理
        self.data_source_health: Dict[str, DataSourceHealth] = {}

        # アラート管理
        self.active_alerts: Dict[str, MonitorAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)

        # SLA管理
        self.sla_metrics: Dict[str, SLAMetrics] = {}

        # メトリクス管理
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))

        # アラート通知コールバック
        self.alert_callbacks: List[Callable[[MonitorAlert], None]] = []

        # デフォルト設定
        self._setup_default_rules()
        self._setup_default_sla()

        logger.info("データ鮮度・整合性監視システム初期化完了")
        logger.info(f"  - ストレージパス: {self.storage_path}")
        logger.info(f"  - チェック間隔: {check_interval_seconds}秒")
        logger.info(f"  - アラート保持期間: {alert_retention_days}日")

    def _setup_default_rules(self):
        """デフォルト監視ルール設定"""
        # 価格データ鮮度ルール
        price_freshness_rule = MonitorRule(
            rule_id="price_freshness",
            name="価格データ鮮度監視",
            description="価格データが1時間以内に更新されているかチェック",
            data_source="price_data",
            rule_type="freshness",
            threshold_value=60,
            threshold_unit="minutes",
            severity=AlertSeverity.HIGH,
            check_interval_seconds=300,
            recovery_actions=[RecoveryAction.RETRY_FETCH, RecoveryAction.USE_FALLBACK],
        )
        self.monitor_rules[price_freshness_rule.rule_id] = price_freshness_rule

        # ニュースデータ鮮度ルール
        news_freshness_rule = MonitorRule(
            rule_id="news_freshness",
            name="ニュースデータ鮮度監視",
            description="ニュースデータが4時間以内に更新されているかチェック",
            data_source="news_data",
            rule_type="freshness",
            threshold_value=240,
            threshold_unit="minutes",
            severity=AlertSeverity.MEDIUM,
            check_interval_seconds=600,
        )
        self.monitor_rules[news_freshness_rule.rule_id] = news_freshness_rule

        # データ整合性ルール
        consistency_rule = MonitorRule(
            rule_id="data_consistency",
            name="データ整合性監視",
            description="データの論理的整合性をチェック",
            data_source="all",
            rule_type="consistency",
            threshold_value=0,
            threshold_unit="violations",
            severity=AlertSeverity.HIGH,
            check_interval_seconds=300,
            recovery_actions=[RecoveryAction.AUTO_FIX, RecoveryAction.NOTIFY_ADMIN],
        )
        self.monitor_rules[consistency_rule.rule_id] = consistency_rule

    def _setup_default_sla(self):
        """デフォルトSLA設定"""
        # 価格データSLA
        price_sla = SLAMetrics(
            sla_id="price_data_sla",
            name="価格データSLA",
            target_availability=0.999,  # 99.9%
            target_freshness_minutes=60,
            target_quality_score=0.95,
            current_availability=1.0,
            current_freshness_minutes=0.0,
            current_quality_score=1.0,
            violations_count=0,
            measurement_period="daily",
        )
        self.sla_metrics[price_sla.sla_id] = price_sla

    async def start_monitoring(self):
        """監視開始"""
        if self.monitor_status == MonitorStatus.ACTIVE:
            logger.warning("監視は既にアクティブです")
            return

        logger.info("データ鮮度・整合性監視開始")

        self.monitor_status = MonitorStatus.ACTIVE
        self.stop_event.clear()

        # 監視スレッド開始
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, name="DataFreshnessMonitor", daemon=True
        )
        self.monitor_thread.start()

        logger.info("監視システム開始完了")

    async def stop_monitoring(self):
        """監視停止"""
        if self.monitor_status != MonitorStatus.ACTIVE:
            logger.warning("監視は既に停止しています")
            return

        logger.info("データ鮮度・整合性監視停止中...")

        self.monitor_status = MonitorStatus.INACTIVE
        self.stop_event.set()

        # 監視スレッド終了待機
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)

        logger.info("監視システム停止完了")

    def _monitor_loop(self):
        """監視ループ（バックグラウンド実行）"""
        logger.info("監視ループ開始")

        while not self.stop_event.is_set():
            try:
                # 各監視ルールの実行時間をチェック
                current_time = time.time()

                for rule_id, rule in self.monitor_rules.items():
                    if not rule.enabled:
                        continue

                    # 前回チェックからの経過時間確認
                    last_check_key = f"last_check_{rule_id}"
                    last_check_time = getattr(self, last_check_key, 0)

                    if current_time - last_check_time >= rule.check_interval_seconds:
                        # 非同期でチェック実行
                        asyncio.run_coroutine_threadsafe(
                            self._execute_rule_check(rule), asyncio.new_event_loop()
                        )
                        setattr(self, last_check_key, current_time)

                # メトリクス更新
                self._update_system_metrics()

                # アラートクリーンアップ
                self._cleanup_expired_alerts()

                # 短い間隔で再チェック
                self.stop_event.wait(30)  # 30秒間隔

            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                self.stop_event.wait(60)  # エラー時は1分待機

        logger.info("監視ループ終了")

    async def _execute_rule_check(self, rule: MonitorRule):
        """監視ルールチェック実行"""
        try:
            logger.debug(f"監視ルールチェック実行: {rule.rule_id}")

            # データソース固有のチェック実行
            if rule.data_source != "all":
                await self._check_data_source(rule, rule.data_source)
            else:
                # 全データソースをチェック
                for source_id in self.data_source_health.keys():
                    await self._check_data_source(rule, source_id)

        except Exception as e:
            logger.error(f"ルールチェック実行エラー {rule.rule_id}: {e}")

    async def _check_data_source(self, rule: MonitorRule, data_source: str):
        """データソース個別チェック"""
        try:
            # データソースからデータ取得（模擬）
            data, context = await self._fetch_data_for_monitoring(data_source)

            if data is None:
                # データ取得失敗
                alert = MonitorAlert(
                    alert_id=f"data_missing_{data_source}_{int(time.time())}",
                    rule_id=rule.rule_id,
                    alert_type=AlertType.DATA_MISSING,
                    severity=AlertSeverity.HIGH,
                    title="データ取得失敗",
                    message=f"データソース {data_source} からデータを取得できません",
                    data_source=data_source,
                    triggered_at=datetime.utcnow(),
                )
                await self._handle_alert(alert)
                return

            # 適切なチェック実行
            check_passed = True
            generated_alert = None

            if rule.rule_type == "freshness" and "freshness" in self.checks:
                check_passed, generated_alert = await self.checks["freshness"].execute_check(
                    data_source, data, context
                )
            elif rule.rule_type == "consistency" and "consistency" in self.checks:
                check_passed, generated_alert = await self.checks["consistency"].execute_check(
                    data_source, data, context
                )

            # アラート処理
            if not check_passed and generated_alert:
                await self._handle_alert(generated_alert)

            # データソースヘルス更新
            await self._update_data_source_health(data_source, check_passed, context)

        except Exception as e:
            logger.error(f"データソースチェックエラー {data_source}: {e}")

    async def _fetch_data_for_monitoring(self, data_source: str) -> Tuple[Any, Dict[str, Any]]:
        """監視用データ取得（模擬実装）"""
        try:
            # 実際の実装では、各データソースから最新データを取得
            # ここでは模擬データを返す

            context = {
                "data_timestamp": datetime.utcnow() - timedelta(minutes=30),
                "source_type": "api",
                "response_time_ms": 150,
            }

            if data_source == "price_data":
                data = pd.DataFrame(
                    {
                        "Open": [2500, 2520],
                        "High": [2550, 2540],
                        "Low": [2480, 2500],
                        "Close": [2530, 2485],
                        "Volume": [1500000, 1200000],
                    },
                    index=pd.date_range(
                        datetime.utcnow() - timedelta(hours=2), periods=2, freq="H"
                    ),
                )

            elif data_source == "news_data":
                data = [
                    {
                        "title": "Test News 1",
                        "timestamp": datetime.utcnow() - timedelta(hours=1),
                        "summary": "Test summary 1",
                    },
                    {
                        "title": "Test News 2",
                        "timestamp": datetime.utcnow() - timedelta(minutes=30),
                        "summary": "Test summary 2",
                    },
                ]

            else:
                data = {"test_value": 123, "timestamp": datetime.utcnow()}

            return data, context

        except Exception as e:
            logger.error(f"監視用データ取得エラー {data_source}: {e}")
            return None, {}

    async def _handle_alert(self, alert: MonitorAlert):
        """アラート処理"""
        logger.warning(f"アラート発生: {alert.title} - {alert.message}")

        # アクティブアラート管理
        self.active_alerts[alert.alert_id] = alert

        # アラート履歴保存
        self.alert_history.append(alert)

        # コールバック実行
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"アラートコールバックエラー: {e}")

        # SLA更新
        await self._update_sla_violations(alert)

        # アラート通知（実装は環境に依存）
        await self._send_alert_notification(alert)

        # 自動回復アクション実行
        await self._execute_recovery_actions(alert)

        # アラート情報保存
        await self._save_alert_to_file(alert)

    async def _update_data_source_health(
        self, data_source: str, check_passed: bool, context: Dict[str, Any]
    ):
        """データソースヘルス状態更新"""
        try:
            current_time = datetime.utcnow()

            if data_source not in self.data_source_health:
                # 新規データソース
                self.data_source_health[data_source] = DataSourceHealth(
                    source_id=data_source,
                    source_type=context.get("source_type", "unknown"),
                    last_update=current_time,
                    data_age_minutes=0.0,
                    quality_score=1.0 if check_passed else 0.0,
                    availability=1.0,
                    error_rate=0.0,
                    response_time_ms=context.get("response_time_ms", 0),
                    health_status="healthy" if check_passed else "warning",
                    consecutive_failures=0 if check_passed else 1,
                )
            else:
                # 既存データソース更新
                health = self.data_source_health[data_source]
                health.last_update = current_time

                # データ年齢計算
                data_timestamp = context.get("data_timestamp")
                if data_timestamp:
                    health.data_age_minutes = (current_time - data_timestamp).total_seconds() / 60

                # 品質スコア更新（移動平均）
                if check_passed:
                    health.quality_score = health.quality_score * 0.9 + 0.1
                    health.consecutive_failures = 0
                else:
                    health.quality_score = health.quality_score * 0.9
                    health.consecutive_failures += 1

                # 可用性更新
                if check_passed:
                    health.availability = min(1.0, health.availability + 0.01)
                else:
                    health.availability = max(0.0, health.availability - 0.05)

                # レスポンス時間更新
                response_time = context.get("response_time_ms", 0)
                if response_time > 0:
                    health.response_time_ms = health.response_time_ms * 0.8 + response_time * 0.2

                # ヘルス状態判定
                if health.consecutive_failures >= 5:
                    health.health_status = "critical"
                elif health.consecutive_failures >= 2:
                    health.health_status = "warning"
                elif health.quality_score >= 0.9:
                    health.health_status = "healthy"
                else:
                    health.health_status = "warning"

        except Exception as e:
            logger.error(f"データソースヘルス更新エラー {data_source}: {e}")

    async def _update_sla_violations(self, alert: MonitorAlert):
        """SLA違反更新"""
        try:
            # データソースに対応するSLAを探す
            for sla_id, sla in self.sla_metrics.items():
                if alert.data_source in sla_id or "all" in sla_id:
                    sla.violations_count += 1
                    sla.last_violation = alert.triggered_at

                    # 可用性の更新
                    if alert.alert_type in [
                        AlertType.DATA_MISSING,
                        AlertType.SYSTEM_ERROR,
                    ]:
                        sla.current_availability = max(0.0, sla.current_availability - 0.001)

                    # 鮮度の更新
                    if (
                        alert.alert_type == AlertType.DATA_STALE
                        and "data_age_minutes" in alert.metadata
                    ):
                        sla.current_freshness_minutes = alert.metadata["data_age_minutes"]

                    break

        except Exception as e:
            logger.error(f"SLA違反更新エラー: {e}")

    async def _send_alert_notification(self, alert: MonitorAlert):
        """アラート通知送信（実装は環境依存）"""
        try:
            # ここでは通知の模擬実装
            notification_message = {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "data_source": alert.data_source,
                "timestamp": alert.triggered_at.isoformat(),
            }

            # 実際の実装では、メール、Slack、Teams等への通知
            logger.info(f"アラート通知: {json.dumps(notification_message, ensure_ascii=False)}")

        except Exception as e:
            logger.error(f"アラート通知送信エラー: {e}")

    async def _execute_recovery_actions(self, alert: MonitorAlert):
        """回復アクション実行"""
        try:
            rule = self.monitor_rules.get(alert.rule_id)
            if not rule or not rule.recovery_actions:
                return

            executed_actions = []

            for action in rule.recovery_actions:
                try:
                    if action == RecoveryAction.RETRY_FETCH:
                        # データ再取得試行
                        logger.info(f"回復アクション実行: データ再取得 ({alert.data_source})")
                        executed_actions.append("retry_fetch")

                    elif action == RecoveryAction.USE_FALLBACK:
                        # フォールバックデータ使用
                        logger.info(f"回復アクション実行: フォールバック使用 ({alert.data_source})")
                        executed_actions.append("use_fallback")

                    elif action == RecoveryAction.AUTO_FIX:
                        # 自動修正実行
                        logger.info(f"回復アクション実行: 自動修正 ({alert.data_source})")
                        executed_actions.append("auto_fix")

                    elif action == RecoveryAction.NOTIFY_ADMIN:
                        # 管理者通知
                        logger.info(f"回復アクション実行: 管理者通知 ({alert.data_source})")
                        executed_actions.append("notify_admin")

                except Exception as e:
                    logger.error(f"回復アクション実行エラー {action}: {e}")

            alert.recovery_actions_taken = executed_actions

        except Exception as e:
            logger.error(f"回復アクション実行エラー: {e}")

    async def _save_alert_to_file(self, alert: MonitorAlert):
        """アラート情報ファイル保存"""
        try:
            alert_file = self.storage_path / f"alert_{alert.alert_id}.json"

            alert_data = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "data_source": alert.data_source,
                "triggered_at": alert.triggered_at.isoformat(),
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "acknowledged_at": (
                    alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
                ),
                "acknowledged_by": alert.acknowledged_by,
                "recovery_actions_taken": alert.recovery_actions_taken,
                "metadata": alert.metadata,
            }

            with open(alert_file, "w", encoding="utf-8") as f:
                json.dump(alert_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"アラートファイル保存エラー: {e}")

    def _update_system_metrics(self):
        """システムメトリクス更新"""
        try:
            current_time = datetime.utcnow()

            # アクティブアラート数
            self.metrics_history["active_alerts_count"].append(
                {"timestamp": current_time, "value": len(self.active_alerts)}
            )

            # データソースヘルス統計
            healthy_sources = sum(
                1
                for health in self.data_source_health.values()
                if health.health_status == "healthy"
            )
            total_sources = len(self.data_source_health)

            if total_sources > 0:
                health_percentage = healthy_sources / total_sources * 100
                self.metrics_history["system_health_percentage"].append(
                    {"timestamp": current_time, "value": health_percentage}
                )

            # 平均品質スコア
            if self.data_source_health:
                avg_quality = np.mean(
                    [health.quality_score for health in self.data_source_health.values()]
                )
                self.metrics_history["avg_quality_score"].append(
                    {"timestamp": current_time, "value": avg_quality}
                )

        except Exception as e:
            logger.error(f"システムメトリクス更新エラー: {e}")

    def _cleanup_expired_alerts(self):
        """期限切れアラートクリーンアップ"""
        try:
            current_time = datetime.utcnow()
            retention_threshold = current_time - timedelta(days=self.alert_retention_days)

            # 期限切れアラート特定
            expired_alert_ids = []
            for alert_id, alert in self.active_alerts.items():
                if alert.triggered_at < retention_threshold:
                    expired_alert_ids.append(alert_id)

            # 期限切れアラート削除
            for alert_id in expired_alert_ids:
                del self.active_alerts[alert_id]

            if expired_alert_ids:
                logger.info(f"期限切れアラートクリーンアップ: {len(expired_alert_ids)}件")

        except Exception as e:
            logger.error(f"アラートクリーンアップエラー: {e}")

    def add_alert_callback(self, callback: Callable[[MonitorAlert], None]):
        """アラートコールバック追加"""
        self.alert_callbacks.append(callback)

    def add_monitor_rule(self, rule: MonitorRule):
        """監視ルール追加"""
        self.monitor_rules[rule.rule_id] = rule
        logger.info(f"監視ルール追加: {rule.rule_id}")

    def remove_monitor_rule(self, rule_id: str):
        """監視ルール削除"""
        if rule_id in self.monitor_rules:
            del self.monitor_rules[rule_id]
            logger.info(f"監視ルール削除: {rule_id}")

    def get_system_dashboard(self) -> Dict[str, Any]:
        """システムダッシュボード情報取得"""
        try:
            current_time = datetime.utcnow()

            # アラート統計
            alert_stats = {
                "total_active": len(self.active_alerts),
                "by_severity": defaultdict(int),
                "by_type": defaultdict(int),
            }

            for alert in self.active_alerts.values():
                alert_stats["by_severity"][alert.severity.value] += 1
                alert_stats["by_type"][alert.alert_type.value] += 1

            # データソースヘルス統計
            health_stats = {
                "total_sources": len(self.data_source_health),
                "by_status": defaultdict(int),
                "avg_quality_score": 0.0,
                "avg_availability": 0.0,
            }

            if self.data_source_health:
                for health in self.data_source_health.values():
                    health_stats["by_status"][health.health_status] += 1

                health_stats["avg_quality_score"] = np.mean(
                    [health.quality_score for health in self.data_source_health.values()]
                )
                health_stats["avg_availability"] = np.mean(
                    [health.availability for health in self.data_source_health.values()]
                )

            # SLA統計
            sla_stats = {}
            for sla_id, sla in self.sla_metrics.items():
                sla_stats[sla_id] = {
                    "name": sla.name,
                    "availability": sla.current_availability,
                    "freshness_minutes": sla.current_freshness_minutes,
                    "quality_score": sla.current_quality_score,
                    "violations_count": sla.violations_count,
                    "target_availability": sla.target_availability,
                    "target_freshness": sla.target_freshness_minutes,
                    "target_quality": sla.target_quality_score,
                }

            # 最近のメトリクス
            recent_metrics = {}
            for metric_name, metric_history in self.metrics_history.items():
                if metric_history:
                    latest_metric = metric_history[-1]
                    recent_metrics[metric_name] = latest_metric["value"]

            return {
                "generated_at": current_time.isoformat(),
                "monitor_status": self.monitor_status.value,
                "uptime_minutes": (
                    current_time - getattr(self, "_start_time", current_time)
                ).total_seconds()
                / 60,
                "alert_statistics": dict(alert_stats),
                "health_statistics": dict(health_stats),
                "sla_metrics": sla_stats,
                "recent_metrics": recent_metrics,
                "system_configuration": {
                    "check_interval_seconds": self.check_interval_seconds,
                    "alert_retention_days": self.alert_retention_days,
                    "total_rules": len(self.monitor_rules),
                    "active_rules": sum(1 for rule in self.monitor_rules.values() if rule.enabled),
                },
            }

        except Exception as e:
            logger.error(f"システムダッシュボード情報取得エラー: {e}")
            return {"error": str(e)}

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """アラート確認"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by

            logger.info(f"アラート確認: {alert_id} by {acknowledged_by}")
        else:
            logger.warning(f"アラートが見つかりません: {alert_id}")

    async def resolve_alert(self, alert_id: str, resolved_by: str = "system"):
        """アラート解決"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.utcnow()

            # アクティブアラートから削除
            del self.active_alerts[alert_id]

            logger.info(f"アラート解決: {alert_id} by {resolved_by}")
        else:
            logger.warning(f"アラートが見つかりません: {alert_id}")

    async def cleanup(self):
        """リソースクリーンアップ"""
        logger.info("データ鮮度・整合性監視システム クリーンアップ開始")

        # 監視停止
        await self.stop_monitoring()

        # メトリクス履歴クリア
        self.metrics_history.clear()

        # キャッシュクリア
        if self.cache_manager:
            # 具体的なクリア処理は実装に依存
            pass

        logger.info("データ鮮度・整合性監視システム クリーンアップ完了")


# Factory function
def create_data_freshness_monitor(
    storage_path: str = "data/monitoring",
    enable_cache: bool = True,
    alert_retention_days: int = 30,
    check_interval_seconds: int = 300,
) -> DataFreshnessMonitor:
    """データ鮮度・整合性監視システム作成"""
    return DataFreshnessMonitor(
        storage_path=storage_path,
        enable_cache=enable_cache,
        alert_retention_days=alert_retention_days,
        check_interval_seconds=check_interval_seconds,
    )


if __name__ == "__main__":
    # テスト実行
    async def test_data_freshness_monitor():
        print("=== Issue #420 データ鮮度・整合性監視システムテスト ===")

        try:
            # 監視システム初期化
            monitor = create_data_freshness_monitor(
                storage_path="test_monitoring",
                enable_cache=True,
                alert_retention_days=7,
                check_interval_seconds=60,
            )

            print("\n1. データ鮮度・整合性監視システム初期化完了")
            print(f"   ストレージパス: {monitor.storage_path}")
            print(f"   チェック間隔: {monitor.check_interval_seconds}秒")
            print(f"   監視ルール数: {len(monitor.monitor_rules)}")

            # アラートコールバック登録
            def alert_handler(alert: MonitorAlert):
                print(f"   📢 アラートコールバック: {alert.title}")

            monitor.add_alert_callback(alert_handler)

            # データソースヘルス初期化（模擬）
            print("\n2. データソースヘルス初期化...")
            monitor.data_source_health["price_data"] = DataSourceHealth(
                source_id="price_data",
                source_type="api",
                last_update=datetime.utcnow() - timedelta(minutes=45),
                data_age_minutes=45.0,
                quality_score=0.95,
                availability=0.99,
                error_rate=0.01,
                response_time_ms=120,
                health_status="healthy",
            )
            print("   価格データソース登録完了")

            # 監視開始
            print("\n3. 監視開始...")
            await monitor.start_monitoring()
            print(f"   監視状態: {monitor.monitor_status.value}")

            # 手動チェックテスト
            print("\n4. 手動チェック実行...")

            # 鮮度チェックテスト
            freshness_check = monitor.checks["freshness"]
            test_data = pd.DataFrame(
                {
                    "Open": [2500],
                    "High": [2550],
                    "Low": [2480],
                    "Close": [2530],
                    "Volume": [1000000],
                },
                index=[datetime.utcnow() - timedelta(hours=2)],
            )  # 2時間前のデータ

            test_context = {"data_timestamp": datetime.utcnow() - timedelta(hours=2)}

            check_passed, alert = await freshness_check.execute_check(
                "test_source", test_data, test_context
            )
            print(f"   鮮度チェック結果: {'合格' if check_passed else '失敗'}")
            if alert:
                print(f"   生成アラート: {alert.title}")

            # 整合性チェックテスト
            consistency_check = monitor.checks["consistency"]

            # 不正な価格データ
            invalid_data = pd.DataFrame(
                {
                    "Open": [2500],
                    "High": [2400],
                    "Low": [2600],
                    "Close": [2530],
                    "Volume": [-1000],  # High < Low, 負のVolume
                }
            )

            check_passed, alert = await consistency_check.execute_check(
                "test_source", invalid_data, {}
            )
            print(f"   整合性チェック結果: {'合格' if check_passed else '失敗'}")
            if alert:
                print(f"   生成アラート: {alert.title}")

            # ダッシュボード情報取得
            print("\n5. システムダッシュボード...")
            dashboard = monitor.get_system_dashboard()

            print(f"   監視状態: {dashboard['monitor_status']}")
            print(f"   アクティブアラート: {dashboard['alert_statistics']['total_active']}件")
            print(f"   データソース数: {dashboard['health_statistics']['total_sources']}")
            print(f"   平均品質スコア: {dashboard['health_statistics']['avg_quality_score']:.3f}")
            print(f"   平均可用性: {dashboard['health_statistics']['avg_availability']:.3f}")

            # SLA状況
            print("\n   SLA状況:")
            for sla_id, sla_info in dashboard["sla_metrics"].items():
                print(
                    f"     {sla_info['name']}: 可用性 {sla_info['availability']:.3f} "
                    f"(目標: {sla_info['target_availability']:.3f})"
                )

            # アラート管理テスト
            print("\n6. アラート管理テスト...")
            if monitor.active_alerts:
                first_alert_id = list(monitor.active_alerts.keys())[0]
                print(f"   アクティブアラート確認: {first_alert_id}")
                await monitor.acknowledge_alert(first_alert_id, "test_user")
                await monitor.resolve_alert(first_alert_id, "test_user")
                print("   アラート解決完了")

            # しばらく監視継続
            print("\n7. 監視継続テスト（10秒間）...")
            await asyncio.sleep(10)

            # 最終ダッシュボード確認
            final_dashboard = monitor.get_system_dashboard()
            print(
                f"   最終アクティブアラート数: {final_dashboard['alert_statistics']['total_active']}"
            )

            # 監視停止
            print("\n8. 監視停止...")
            await monitor.stop_monitoring()
            print(f"   監視状態: {monitor.monitor_status.value}")

            # クリーンアップ
            await monitor.cleanup()

            print("\n✅ Issue #420 データ鮮度・整合性監視システムテスト完了")

        except Exception as e:
            print(f"❌ テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_data_freshness_monitor())
