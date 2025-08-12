#!/usr/bin/env python3
"""
データ品質管理システム
Issue #322: ML Data Shortage Problem Resolution - Data Quality Component

データ品質向上・バックフィル機能・フォールバック戦略の実装
"""

import asyncio
import statistics
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from ..utils.logging_config import get_context_logger
    from ..utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    # キャッシュマネージャーのモック
    class UnifiedCacheManager:
        def __init__(self, **kwargs):
            pass

        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True

    def generate_unified_cache_key(*args, **kwargs):
        return f"quality_key_{hash(str(args) + str(kwargs))}"


logger = get_context_logger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class DataQualityLevel(Enum):
    """データ品質レベル"""

    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 75-89%
    FAIR = "fair"  # 50-74%
    POOR = "poor"  # 25-49%
    CRITICAL = "critical"  # 0-24%


class DataIssueType(Enum):
    """データ問題種別"""

    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    INCONSISTENCY = "inconsistency"
    STALENESS = "staleness"
    INVALID_FORMAT = "invalid_format"
    DUPLICATE_DATA = "duplicate_data"
    TIME_GAPS = "time_gaps"
    SOURCE_FAILURE = "source_failure"


@dataclass
class DataQualityIssue:
    """データ品質問題"""

    issue_type: DataIssueType
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    affected_fields: List[str]
    detection_time: datetime
    auto_fixable: bool = False
    fix_suggestion: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityMetrics:
    """データ品質メトリクス"""

    completeness: float  # 完全性 0-1
    accuracy: float  # 正確性 0-1
    consistency: float  # 一貫性 0-1
    timeliness: float  # 時間性 0-1
    validity: float  # 有効性 0-1
    uniqueness: float  # 一意性 0-1
    overall_score: float  # 総合スコア 0-1
    quality_level: DataQualityLevel
    total_records: int
    issues_found: int
    calculation_time: datetime = field(default_factory=datetime.now)


@dataclass
class BackfillRequest:
    """バックフィル要求"""

    symbol: str
    data_type: str
    start_date: datetime
    end_date: datetime
    priority: int = 5  # 1(最高) - 10(最低)
    reason: str = ""
    requested_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, processing, completed, failed
    progress: float = 0.0


class DataValidator:
    """データバリデーター"""

    def __init__(self):
        self.validation_rules = {
            "price": {
                "required_fields": ["Open", "High", "Low", "Close", "Volume"],
                "numeric_fields": ["Open", "High", "Low", "Close", "Volume"],
                "positive_fields": ["Volume"],
                "price_order_check": True,  # Low <= Open,Close <= High
                "volume_min": 0,
            },
            "news": {
                "required_fields": ["title", "timestamp"],
                "text_fields": ["title", "summary"],
                "timestamp_field": "timestamp",
                "max_age_hours": 48,
            },
            "sentiment": {
                "required_fields": ["overall_sentiment"],
                "numeric_fields": [
                    "overall_sentiment",
                    "positive_ratio",
                    "negative_ratio",
                ],
                "range_checks": {
                    "overall_sentiment": (-1.0, 1.0),
                    "positive_ratio": (0.0, 1.0),
                    "negative_ratio": (0.0, 1.0),
                },
            },
            "macro": {
                "required_fields": ["interest_rate", "inflation_rate"],
                "numeric_fields": ["interest_rate", "inflation_rate", "gdp_growth"],
                "percentage_fields": ["interest_rate", "inflation_rate"],
            },
        }

    def validate_data(self, data: Any, data_type: str) -> Tuple[bool, List[DataQualityIssue]]:
        """データ検証実行"""
        issues = []

        try:
            if data_type not in self.validation_rules:
                issues.append(
                    DataQualityIssue(
                        issue_type=DataIssueType.INVALID_FORMAT,
                        severity="medium",
                        description=f"不明なデータ型: {data_type}",
                        affected_fields=[],
                        detection_time=datetime.now(),
                    )
                )
                return False, issues

            rules = self.validation_rules[data_type]

            if data_type == "price":
                issues.extend(self._validate_price_data(data, rules))
            elif data_type == "news":
                issues.extend(self._validate_news_data(data, rules))
            elif data_type == "sentiment":
                issues.extend(self._validate_sentiment_data(data, rules))
            elif data_type == "macro":
                issues.extend(self._validate_macro_data(data, rules))

            # 重大問題があるかチェック
            has_critical_issues = any(issue.severity == "critical" for issue in issues)

            return not has_critical_issues, issues

        except Exception as e:
            logger.error(f"データ検証エラー {data_type}: {e}")
            issues.append(
                DataQualityIssue(
                    issue_type=DataIssueType.INVALID_FORMAT,
                    severity="critical",
                    description=f"検証処理エラー: {str(e)}",
                    affected_fields=[],
                    detection_time=datetime.now(),
                )
            )
            return False, issues

    def _validate_price_data(self, data: pd.DataFrame, rules: Dict) -> List[DataQualityIssue]:
        """価格データ検証"""
        issues = []

        # 必須フィールドチェック
        for field_name in rules["required_fields"]:
            if field_name not in data.columns:
                issues.append(
                    DataQualityIssue(
                        issue_type=DataIssueType.MISSING_VALUES,
                        severity="critical",
                        description=f"必須フィールド欠如: {field_name}",
                        affected_fields=[field_name],
                        detection_time=datetime.now(),
                    )
                )

        # 数値フィールドチェック
        for field_name in rules["numeric_fields"]:
            if field_name in data.columns:
                non_numeric_count = data[field_name].isna().sum()
                if non_numeric_count > 0:
                    issues.append(
                        DataQualityIssue(
                            issue_type=DataIssueType.MISSING_VALUES,
                            severity="medium",
                            description=f"数値欠損: {field_name} ({non_numeric_count}件)",
                            affected_fields=[field_name],
                            detection_time=datetime.now(),
                            auto_fixable=True,
                            fix_suggestion="前後値補間",
                        )
                    )

        # 価格順序チェック (Low <= Open,Close <= High)
        if rules.get("price_order_check") and all(
            col in data.columns for col in ["Low", "High", "Open", "Close"]
        ):
            invalid_rows = data[
                (data["Low"] > data["High"])
                | (data["Low"] > data["Open"])
                | (data["Low"] > data["Close"])
                | (data["High"] < data["Open"])
                | (data["High"] < data["Close"])
            ]

            if len(invalid_rows) > 0:
                issues.append(
                    DataQualityIssue(
                        issue_type=DataIssueType.INCONSISTENCY,
                        severity="high",
                        description=f"価格順序異常: {len(invalid_rows)}件",
                        affected_fields=["Low", "High", "Open", "Close"],
                        detection_time=datetime.now(),
                        auto_fixable=True,
                        fix_suggestion="価格順序自動修正",
                    )
                )

        # 異常値チェック
        for field_name in ["Open", "High", "Low", "Close"]:
            if field_name in data.columns:
                outliers = self._detect_outliers(data[field_name])
                if len(outliers) > 0:
                    issues.append(
                        DataQualityIssue(
                            issue_type=DataIssueType.OUTLIERS,
                            severity="medium",
                            description=f"{field_name}異常値: {len(outliers)}件",
                            affected_fields=[field_name],
                            detection_time=datetime.now(),
                            auto_fixable=True,
                            fix_suggestion="統計的異常値除去",
                        )
                    )

        return issues

    def _validate_news_data(self, data: List[Dict], rules: Dict) -> List[DataQualityIssue]:
        """ニュースデータ検証"""
        issues = []

        if not data:
            issues.append(
                DataQualityIssue(
                    issue_type=DataIssueType.MISSING_VALUES,
                    severity="high",
                    description="ニュースデータが空",
                    affected_fields=[],
                    detection_time=datetime.now(),
                )
            )
            return issues

        # 必須フィールドチェック
        for i, article in enumerate(data):
            for field_name in rules["required_fields"]:
                if field_name not in article or not article[field_name]:
                    issues.append(
                        DataQualityIssue(
                            issue_type=DataIssueType.MISSING_VALUES,
                            severity="medium",
                            description=f"記事{i}: {field_name}欠損",
                            affected_fields=[field_name],
                            detection_time=datetime.now(),
                        )
                    )

        # 重複チェック
        titles = [article.get("title", "") for article in data]
        unique_titles = set(titles)
        if len(titles) != len(unique_titles):
            duplicate_count = len(titles) - len(unique_titles)
            issues.append(
                DataQualityIssue(
                    issue_type=DataIssueType.DUPLICATE_DATA,
                    severity="low",
                    description=f"重複記事: {duplicate_count}件",
                    affected_fields=["title"],
                    detection_time=datetime.now(),
                    auto_fixable=True,
                    fix_suggestion="重複除去",
                )
            )

        return issues

    def _validate_sentiment_data(self, data: Dict, rules: Dict) -> List[DataQualityIssue]:
        """センチメントデータ検証"""
        issues = []

        # 必須フィールドチェック
        for field_name in rules["required_fields"]:
            if field_name not in data or data[field_name] is None:
                issues.append(
                    DataQualityIssue(
                        issue_type=DataIssueType.MISSING_VALUES,
                        severity="critical",
                        description=f"センチメント必須フィールド欠如: {field_name}",
                        affected_fields=[field_name],
                        detection_time=datetime.now(),
                    )
                )

        # 範囲チェック
        if "range_checks" in rules:
            for field_name, (min_val, max_val) in rules["range_checks"].items():
                if field_name in data and isinstance(data[field_name], (int, float)):
                    value = data[field_name]
                    if not (min_val <= value <= max_val):
                        issues.append(
                            DataQualityIssue(
                                issue_type=DataIssueType.OUTLIERS,
                                severity="high",
                                description=f"{field_name}範囲外: {value} (範囲: {min_val}-{max_val})",
                                affected_fields=[field_name],
                                detection_time=datetime.now(),
                                auto_fixable=True,
                                fix_suggestion=f"値を{min_val}-{max_val}に正規化",
                            )
                        )

        return issues

    def _validate_macro_data(self, data: Dict, rules: Dict) -> List[DataQualityIssue]:
        """マクロ経済データ検証"""
        issues = []

        # 必須フィールドチェック
        for field_name in rules["required_fields"]:
            if field_name not in data or data[field_name] is None:
                issues.append(
                    DataQualityIssue(
                        issue_type=DataIssueType.MISSING_VALUES,
                        severity="high",
                        description=f"マクロ指標欠損: {field_name}",
                        affected_fields=[field_name],
                        detection_time=datetime.now(),
                    )
                )

        # データ新鮮度チェック
        if "timestamp" in data:
            timestamp = data["timestamp"]
            if isinstance(timestamp, datetime):
                age_hours = (datetime.now() - timestamp).total_seconds() / 3600
                if age_hours > 24:  # 24時間以上古い
                    issues.append(
                        DataQualityIssue(
                            issue_type=DataIssueType.STALENESS,
                            severity="medium",
                            description=f"データが古い: {age_hours:.1f}時間前",
                            affected_fields=["timestamp"],
                            detection_time=datetime.now(),
                        )
                    )

        return issues

    def _detect_outliers(self, series: pd.Series, method: str = "iqr") -> pd.Series:
        """異常値検出"""
        try:
            if method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = series[(series < lower_bound) | (series > upper_bound)]
            elif method == "zscore":
                z_scores = np.abs(statistics.zscore(series.dropna()))
                outliers = series[z_scores > 3]
            else:
                outliers = pd.Series(dtype=float)

            return outliers

        except Exception as e:
            logger.error(f"異常値検出エラー: {e}")
            return pd.Series(dtype=float)


class DataQualityManager:
    """データ品質管理システム"""

    def __init__(
        self,
        enable_cache: bool = True,
        auto_fix_enabled: bool = True,
        quality_threshold: float = 0.7,
    ):
        """
        初期化

        Args:
            enable_cache: キャッシュ機能有効化
            auto_fix_enabled: 自動修正機能有効化
            quality_threshold: 品質閾値（0-1）
        """
        self.auto_fix_enabled = auto_fix_enabled
        self.quality_threshold = quality_threshold

        # Issue #324統合キャッシュ連携
        if enable_cache:
            try:
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=16,  # 品質データ用軽量キャッシュ
                    l2_memory_mb=64,  # 検証結果キャッシュ
                    l3_disk_mb=128,  # 品質履歴キャッシュ
                )
                self.cache_enabled = True
                logger.info("データ品質管理システム - 統合キャッシュ連携完了")
            except Exception as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")
                self.cache_manager = None
                self.cache_enabled = False
        else:
            self.cache_manager = None
            self.cache_enabled = False

        self.validator = DataValidator()

        # バックフィル管理
        self.backfill_queue = []
        self.backfill_history = {}

        # 品質履歴
        self.quality_history = {}

        # フォールバック戦略設定
        self.fallback_strategies = {
            "price": ["cache_stale", "interpolation", "previous_day"],
            "news": ["cache_stale", "alternative_source"],
            "sentiment": ["neutral_default", "cache_stale"],
            "macro": ["cache_stale", "previous_value", "default_values"],
        }

        logger.info("データ品質管理システム初期化完了")
        logger.info(f"  - 自動修正: {'有効' if auto_fix_enabled else '無効'}")
        logger.info(f"  - 品質閾値: {quality_threshold}")
        logger.info(f"  - キャッシュ: {'有効' if self.cache_enabled else '無効'}")

    def assess_data_quality(
        self, data: Any, data_type: str, symbol: str = ""
    ) -> DataQualityMetrics:
        """データ品質評価"""
        try:
            # キャッシュチェック
            if self.cache_enabled:
                cached_metrics = self._get_cached_quality_metrics(data, data_type, symbol)
                if cached_metrics:
                    logger.debug(f"品質評価キャッシュヒット: {data_type} {symbol}")
                    return cached_metrics

            # データ検証実行
            is_valid, issues = self.validator.validate_data(data, data_type)

            # 品質メトリクス計算
            metrics = self._calculate_quality_metrics(data, data_type, issues)

            # 品質レベル判定
            metrics.quality_level = self._determine_quality_level(metrics.overall_score)

            # 品質履歴更新
            self._update_quality_history(symbol, data_type, metrics)

            # キャッシュ保存
            if self.cache_enabled:
                self._cache_quality_metrics(data, data_type, symbol, metrics)

            logger.info(
                f"品質評価完了 {data_type} {symbol}: {metrics.overall_score:.3f} ({metrics.quality_level.value})"
            )

            return metrics

        except Exception as e:
            logger.error(f"品質評価エラー {data_type} {symbol}: {e}")
            return DataQualityMetrics(
                completeness=0.0,
                accuracy=0.0,
                consistency=0.0,
                timeliness=0.0,
                validity=0.0,
                uniqueness=0.0,
                overall_score=0.0,
                quality_level=DataQualityLevel.CRITICAL,
                total_records=0,
                issues_found=1,
            )

    def _calculate_quality_metrics(
        self, data: Any, data_type: str, issues: List[DataQualityIssue]
    ) -> DataQualityMetrics:
        """品質メトリクス計算"""

        # データサイズ取得
        if isinstance(data, (pd.DataFrame, list)):
            total_records = len(data)
        elif isinstance(data, dict):
            total_records = 1
        else:
            total_records = 1

        # 基本メトリクス初期化
        completeness = 1.0
        accuracy = 1.0
        consistency = 1.0
        timeliness = 1.0
        validity = 1.0
        uniqueness = 1.0

        # 問題タイプ別の影響度計算
        for issue in issues:
            severity_weight = {
                "critical": 0.5,
                "high": 0.3,
                "medium": 0.15,
                "low": 0.05,
            }.get(issue.severity, 0.1)

            if issue.issue_type == DataIssueType.MISSING_VALUES:
                completeness = max(0, completeness - severity_weight)
            elif issue.issue_type == DataIssueType.OUTLIERS:
                accuracy = max(0, accuracy - severity_weight)
            elif issue.issue_type == DataIssueType.INCONSISTENCY:
                consistency = max(0, consistency - severity_weight)
            elif issue.issue_type == DataIssueType.STALENESS:
                timeliness = max(0, timeliness - severity_weight)
            elif issue.issue_type in [
                DataIssueType.INVALID_FORMAT,
                DataIssueType.SOURCE_FAILURE,
            ]:
                validity = max(0, validity - severity_weight)
            elif issue.issue_type == DataIssueType.DUPLICATE_DATA:
                uniqueness = max(0, uniqueness - severity_weight)

        # 総合スコア計算（重み付け平均）
        weights = {
            "completeness": 0.25,
            "accuracy": 0.2,
            "consistency": 0.15,
            "timeliness": 0.15,
            "validity": 0.15,
            "uniqueness": 0.1,
        }

        overall_score = (
            completeness * weights["completeness"]
            + accuracy * weights["accuracy"]
            + consistency * weights["consistency"]
            + timeliness * weights["timeliness"]
            + validity * weights["validity"]
            + uniqueness * weights["uniqueness"]
        )

        return DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            validity=validity,
            uniqueness=uniqueness,
            overall_score=overall_score,
            quality_level=DataQualityLevel.GOOD,  # 後で更新
            total_records=total_records,
            issues_found=len(issues),
        )

    def _determine_quality_level(self, score: float) -> DataQualityLevel:
        """品質レベル判定"""
        if score >= 0.9:
            return DataQualityLevel.EXCELLENT
        elif score >= 0.75:
            return DataQualityLevel.GOOD
        elif score >= 0.5:
            return DataQualityLevel.FAIR
        elif score >= 0.25:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.CRITICAL

    def auto_fix_data_issues(
        self, data: Any, data_type: str, issues: List[DataQualityIssue]
    ) -> Tuple[Any, List[str]]:
        """データ問題自動修正"""
        if not self.auto_fix_enabled:
            return data, []

        fixed_issues = []

        try:
            for issue in issues:
                if issue.auto_fixable:
                    if data_type == "price" and isinstance(data, pd.DataFrame):
                        data = self._fix_price_data_issues(data, issue)
                        fixed_issues.append(issue.description)
                    elif data_type == "sentiment" and isinstance(data, dict):
                        data = self._fix_sentiment_data_issues(data, issue)
                        fixed_issues.append(issue.description)
                    elif data_type == "news" and isinstance(data, list):
                        data = self._fix_news_data_issues(data, issue)
                        fixed_issues.append(issue.description)

            if fixed_issues:
                logger.info(f"自動修正完了 {data_type}: {len(fixed_issues)}件")

            return data, fixed_issues

        except Exception as e:
            logger.error(f"自動修正エラー {data_type}: {e}")
            return data, []

    def _fix_price_data_issues(self, data: pd.DataFrame, issue: DataQualityIssue) -> pd.DataFrame:
        """価格データ問題修正"""
        if issue.issue_type == DataIssueType.MISSING_VALUES:
            # 前後値補間
            for field in issue.affected_fields:
                if field in data.columns:
                    data[field] = data[field].interpolate(method="linear")

        elif issue.issue_type == DataIssueType.OUTLIERS:
            # 統計的異常値除去（3σルール）
            for field in issue.affected_fields:
                if field in data.columns:
                    mean_val = data[field].mean()
                    std_val = data[field].std()
                    data[field] = np.where(
                        np.abs(data[field] - mean_val) > 3 * std_val,
                        mean_val,  # 異常値を平均値で置換
                        data[field],
                    )

        elif issue.issue_type == DataIssueType.INCONSISTENCY:
            # 価格順序修正
            if all(col in data.columns for col in ["Low", "High", "Open", "Close"]):
                # High = max(Open, High, Close)
                data["High"] = data[["Open", "High", "Close"]].max(axis=1)
                # Low = min(Open, Low, Close)
                data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        return data

    def _fix_sentiment_data_issues(self, data: Dict, issue: DataQualityIssue) -> Dict:
        """センチメントデータ問題修正"""
        if issue.issue_type == DataIssueType.OUTLIERS:
            # 範囲正規化
            for field in issue.affected_fields:
                if field in data and isinstance(data[field], (int, float)):
                    if field == "overall_sentiment":
                        data[field] = np.clip(data[field], -1.0, 1.0)
                    elif field in ["positive_ratio", "negative_ratio"]:
                        data[field] = np.clip(data[field], 0.0, 1.0)

        elif issue.issue_type == DataIssueType.MISSING_VALUES:
            # デフォルト値設定
            defaults = {
                "overall_sentiment": 0.0,
                "positive_ratio": 0.33,
                "negative_ratio": 0.33,
                "neutral_ratio": 0.34,
            }
            for field in issue.affected_fields:
                if field not in data or data[field] is None:
                    data[field] = defaults.get(field, 0.0)

        return data

    def _fix_news_data_issues(self, data: List[Dict], issue: DataQualityIssue) -> List[Dict]:
        """ニュースデータ問題修正"""
        if issue.issue_type == DataIssueType.DUPLICATE_DATA:
            # 重複除去
            seen_titles = set()
            unique_data = []
            for article in data:
                title = article.get("title", "")
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_data.append(article)
            data = unique_data

        elif issue.issue_type == DataIssueType.MISSING_VALUES:
            # 欠損フィールドの補完
            for article in data:
                if "title" not in article or not article["title"]:
                    article["title"] = "No title available"
                if "timestamp" not in article:
                    article["timestamp"] = datetime.now()

        return data

    async def request_backfill(
        self,
        symbol: str,
        data_type: str,
        start_date: datetime,
        end_date: datetime,
        reason: str = "",
    ) -> str:
        """バックフィル要求"""
        request_id = f"backfill_{symbol}_{data_type}_{int(time.time())}"

        backfill_request = BackfillRequest(
            symbol=symbol,
            data_type=data_type,
            start_date=start_date,
            end_date=end_date,
            reason=reason,
        )

        self.backfill_queue.append(backfill_request)
        logger.info(f"バックフィル要求登録: {request_id} ({symbol} {data_type})")

        return request_id

    async def execute_backfill_queue(self, max_concurrent: int = 3):
        """バックフィルキュー実行"""
        if not self.backfill_queue:
            logger.info("バックフィルキューが空です")
            return

        logger.info(f"バックフィル実行開始: {len(self.backfill_queue)}件")

        # 優先度順ソート
        self.backfill_queue.sort(key=lambda x: x.priority)

        # セマフォで同時実行数制限
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_backfill(request: BackfillRequest):
            async with semaphore:
                return await self._execute_single_backfill(request)

        # 並列実行
        tasks = [process_backfill(req) for req in self.backfill_queue[: max_concurrent * 2]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 処理済み要求をキューから削除
        completed_count = sum(1 for result in results if result is True)
        self.backfill_queue = self.backfill_queue[completed_count:]

        logger.info(f"バックフィル実行完了: {completed_count}件成功")

    async def _execute_single_backfill(self, request: BackfillRequest) -> bool:
        """単一バックフィル実行"""
        try:
            request.status = "processing"
            logger.info(f"バックフィル処理開始: {request.symbol} {request.data_type}")

            # 実際のデータ収集処理（モック）
            await asyncio.sleep(1)  # 処理時間シミュレーション

            # 成功として記録
            request.status = "completed"
            request.progress = 1.0

            # 履歴保存
            request_id = f"{request.symbol}_{request.data_type}_{request.start_date.date()}"
            self.backfill_history[request_id] = request

            logger.info(f"バックフィル完了: {request.symbol} {request.data_type}")
            return True

        except Exception as e:
            logger.error(f"バックフィル失敗 {request.symbol} {request.data_type}: {e}")
            request.status = "failed"
            return False

    def apply_fallback_strategy(self, data_type: str, symbol: str, error_context: str) -> Any:
        """フォールバック戦略適用"""
        try:
            strategies = self.fallback_strategies.get(data_type, ["default"])

            for strategy in strategies:
                fallback_data = self._execute_fallback_strategy(strategy, data_type, symbol)
                if fallback_data is not None:
                    logger.info(f"フォールバック成功 {data_type} {symbol}: {strategy}")
                    return fallback_data

            logger.warning(f"全フォールバック戦略失敗 {data_type} {symbol}")
            return None

        except Exception as e:
            logger.error(f"フォールバック戦略エラー {data_type} {symbol}: {e}")
            return None

    def _execute_fallback_strategy(self, strategy: str, data_type: str, symbol: str) -> Any:
        """フォールバック戦略実行"""

        if strategy == "cache_stale":
            # 期限切れキャッシュの利用
            if self.cache_enabled:
                cache_key = generate_unified_cache_key(
                    "fallback", data_type, symbol, time_bucket_minutes=1440
                )  # 1日
                stale_data = self.cache_manager.get(cache_key)
                if stale_data:
                    logger.info(f"期限切れキャッシュ利用: {data_type} {symbol}")
                    return stale_data

        elif strategy == "interpolation":
            # 補間データ生成
            if data_type == "price":
                return self._generate_interpolated_price_data(symbol)

        elif strategy == "neutral_default":
            # 中立デフォルト値
            if data_type == "sentiment":
                return {
                    "overall_sentiment": 0.0,
                    "positive_ratio": 0.33,
                    "negative_ratio": 0.33,
                    "neutral_ratio": 0.34,
                    "confidence": 0.1,
                    "fallback": True,
                }

        elif strategy == "default_values":
            # デフォルト値セット
            if data_type == "macro":
                return {
                    "interest_rate": 0.1,
                    "inflation_rate": 2.0,
                    "gdp_growth": 1.0,
                    "exchange_rate_usd": 150.0,
                    "fallback": True,
                }

        return None

    def _generate_interpolated_price_data(self, symbol: str) -> pd.DataFrame:
        """補間価格データ生成"""
        # 簡易的な価格データ生成（実際には履歴ベース補間を実装）
        dates = pd.date_range(start=datetime.now() - timedelta(days=5), periods=5)
        base_price = 2500  # 基準価格

        return pd.DataFrame(
            {
                "Open": [base_price * (1 + np.random.uniform(-0.02, 0.02)) for _ in range(5)],
                "High": [base_price * (1 + np.random.uniform(0.0, 0.03)) for _ in range(5)],
                "Low": [base_price * (1 + np.random.uniform(-0.03, 0.0)) for _ in range(5)],
                "Close": [base_price * (1 + np.random.uniform(-0.02, 0.02)) for _ in range(5)],
                "Volume": [np.random.randint(1000000, 5000000) for _ in range(5)],
                "interpolated": True,
            },
            index=dates,
        )

    def _get_cached_quality_metrics(
        self, data: Any, data_type: str, symbol: str
    ) -> Optional[DataQualityMetrics]:
        """キャッシュ品質メトリクス取得"""
        if not self.cache_enabled:
            return None

        # データ内容ベースのハッシュキー生成（簡易版）
        data_hash = hash(str(data)[:100]) if data else 0
        cache_key = generate_unified_cache_key(
            "quality", data_type, symbol, str(data_hash), time_bucket_minutes=30
        )

        return self.cache_manager.get(cache_key)

    def _cache_quality_metrics(
        self, data: Any, data_type: str, symbol: str, metrics: DataQualityMetrics
    ):
        """品質メトリクスキャッシュ"""
        if not self.cache_enabled:
            return

        data_hash = hash(str(data)[:100]) if data else 0
        cache_key = generate_unified_cache_key(
            "quality", data_type, symbol, str(data_hash), time_bucket_minutes=30
        )

        # 品質スコアに基づく優先度設定
        priority = 3.0 + (metrics.overall_score * 5.0)  # 3-8の範囲
        self.cache_manager.put(cache_key, metrics, priority=priority)

    def _update_quality_history(self, symbol: str, data_type: str, metrics: DataQualityMetrics):
        """品質履歴更新"""
        key = f"{symbol}_{data_type}"

        if key not in self.quality_history:
            self.quality_history[key] = []

        self.quality_history[key].append(metrics)

        # 履歴を最新100件に制限
        if len(self.quality_history[key]) > 100:
            self.quality_history[key] = self.quality_history[key][-100:]

    def get_quality_trend(self, symbol: str, data_type: str, days: int = 30) -> Dict[str, Any]:
        """品質トレンド取得"""
        key = f"{symbol}_{data_type}"

        if key not in self.quality_history:
            return {"trend": "no_data", "current_score": 0.0, "average_score": 0.0}

        recent_metrics = [
            m
            for m in self.quality_history[key]
            if (datetime.now() - m.calculation_time).days <= days
        ]

        if not recent_metrics:
            return {"trend": "no_data", "current_score": 0.0, "average_score": 0.0}

        scores = [m.overall_score for m in recent_metrics]
        current_score = scores[-1] if scores else 0.0
        average_score = np.mean(scores)

        # トレンド判定
        if len(scores) >= 2:
            recent_trend = np.polyfit(range(len(scores)), scores, 1)[0]  # 線形回帰の傾き
            if recent_trend > 0.01:
                trend = "improving"
            elif recent_trend < -0.01:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "current_score": current_score,
            "average_score": average_score,
            "sample_count": len(scores),
            "quality_level": self._determine_quality_level(current_score).value,
        }

    def get_comprehensive_quality_report(self) -> Dict[str, Any]:
        """包括品質レポート取得"""
        report = {
            "system_status": "active",
            "auto_fix_enabled": self.auto_fix_enabled,
            "quality_threshold": self.quality_threshold,
            "backfill_queue_size": len(self.backfill_queue),
            "quality_trends": {},
            "overall_statistics": {
                "total_assessments": 0,
                "average_quality_score": 0.0,
                "quality_distribution": {level.value: 0 for level in DataQualityLevel},
            },
        }

        # 品質トレンド集計
        all_scores = []
        for key in self.quality_history:
            if self.quality_history[key]:
                latest_metric = self.quality_history[key][-1]
                all_scores.append(latest_metric.overall_score)

                # 品質レベル分布更新
                level = latest_metric.quality_level.value
                report["overall_statistics"]["quality_distribution"][level] += 1

        # 全体統計
        if all_scores:
            report["overall_statistics"]["total_assessments"] = len(all_scores)
            report["overall_statistics"]["average_quality_score"] = np.mean(all_scores)

        # 個別トレンド（上位10件）
        for key in list(self.quality_history.keys())[:10]:
            symbol, data_type = key.split("_", 1)
            report["quality_trends"][key] = self.get_quality_trend(symbol, data_type)

        return report


if __name__ == "__main__":
    # テスト実行
    async def test_data_quality_manager():
        print("=== データ品質管理システムテスト ===")

        # 品質管理システム初期化
        quality_manager = DataQualityManager(
            enable_cache=True, auto_fix_enabled=True, quality_threshold=0.7
        )

        # テストデータ準備
        print("\n1. テストデータ準備...")

        # 価格データ（品質問題を含む）
        test_price_data = pd.DataFrame(
            {
                "Open": [2500, 2520, np.nan, 2480, 2510],  # 欠損値あり
                "High": [2550, 2540, 2490, 2500, 2530],
                "Low": [2480, 2500, 2460, 2470, 2495],
                "Close": [2530, 2485, 2475, 2495, 2525],
                "Volume": [1500000, 1200000, 1800000, 999999999, 1300000],  # 異常値あり
            }
        )

        # センチメントデータ（範囲外値を含む）
        test_sentiment_data = {
            "overall_sentiment": 1.5,  # 範囲外 (-1 to 1)
            "positive_ratio": 0.8,
            "negative_ratio": 0.3,  # 合計 > 1.0
            "confidence": 0.9,
        }

        # ニュースデータ（重複を含む）
        test_news_data = [
            {
                "title": "Test News 1",
                "summary": "Summary 1",
                "timestamp": datetime.now(),
            },
            {
                "title": "Test News 1",
                "summary": "Summary 1",
                "timestamp": datetime.now(),
            },  # 重複
            {
                "title": "Test News 2",
                "summary": "Summary 2",
                "timestamp": datetime.now(),
            },
        ]

        # 品質評価テスト
        print("\n2. データ品質評価テスト...")

        # 価格データ評価
        price_metrics = quality_manager.assess_data_quality(test_price_data, "price", "7203")
        print(
            f"   価格データ品質: {price_metrics.overall_score:.3f} ({price_metrics.quality_level.value})"
        )
        print(f"   - 完全性: {price_metrics.completeness:.3f}")
        print(f"   - 正確性: {price_metrics.accuracy:.3f}")
        print(f"   - 問題数: {price_metrics.issues_found}")

        # センチメントデータ評価
        sentiment_metrics = quality_manager.assess_data_quality(
            test_sentiment_data, "sentiment", "7203"
        )
        print(
            f"   センチメント品質: {sentiment_metrics.overall_score:.3f} ({sentiment_metrics.quality_level.value})"
        )

        # ニュースデータ評価
        news_metrics = quality_manager.assess_data_quality(test_news_data, "news", "7203")
        print(
            f"   ニュース品質: {news_metrics.overall_score:.3f} ({news_metrics.quality_level.value})"
        )

        # 自動修正テスト
        print("\n3. 自動修正テスト...")

        # 価格データ問題検証・修正
        is_valid, issues = quality_manager.validator.validate_data(test_price_data, "price")
        print(f"   価格データ検証: {'有効' if is_valid else '問題あり'} ({len(issues)}件の問題)")

        fixed_price_data, fixed_issues = quality_manager.auto_fix_data_issues(
            test_price_data, "price", issues
        )
        print(f"   自動修正: {len(fixed_issues)}件修正")

        # バックフィル要求テスト
        print("\n4. バックフィル要求テスト...")
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now() - timedelta(days=1)

        request_id = await quality_manager.request_backfill(
            "7203", "price", start_date, end_date, "データ欠損補填"
        )
        print(f"   バックフィル要求: {request_id}")
        print(f"   キューサイズ: {len(quality_manager.backfill_queue)}")

        # バックフィル実行
        await quality_manager.execute_backfill_queue()
        print(f"   バックフィル後キューサイズ: {len(quality_manager.backfill_queue)}")

        # フォールバック戦略テスト
        print("\n5. フォールバック戦略テスト...")
        fallback_data = quality_manager.apply_fallback_strategy("sentiment", "7203", "API障害")
        print(f"   フォールバック成功: {fallback_data is not None}")

        if fallback_data:
            print(f"   フォールバックデータ: {fallback_data}")

        # 包括品質レポート
        print("\n6. 包括品質レポート...")
        quality_report = quality_manager.get_comprehensive_quality_report()
        print(
            f"   評価済みデータセット: {quality_report['overall_statistics']['total_assessments']}"
        )
        print(
            f"   平均品質スコア: {quality_report['overall_statistics']['average_quality_score']:.3f}"
        )
        print(f"   バックフィルキュー: {quality_report['backfill_queue_size']}")

        print("\n✅ データ品質管理システムテスト完了")

    # 非同期テスト実行
    try:
        asyncio.run(test_data_quality_manager())
    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback

        traceback.print_exc()
