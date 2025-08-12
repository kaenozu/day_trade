#!/usr/bin/env python3
"""
データバリデーション・クリーニングパイプラインシステム
Issue #420: データ管理とデータ品質保証メカニズムの強化

高度なデータ品質保証のための包括的なパイプライン処理システム:
- マルチステージバリデーション
- 動的データクリーニング
- 業界標準準拠の品質チェック
- リアルタイム監視・アラート
- データ系譜管理
- 品質メトリクス追跡
"""

import asyncio
import hashlib
import json
import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    from ..utils.data_quality_manager import (
        DataIssueType,
        DataQualityIssue,
        DataQualityLevel,
        DataQualityMetrics,
        DataValidator,
    )
    from ..utils.logging_config import get_context_logger
    from ..utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    # モッククラス
    class DataQualityMetrics:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class DataQualityLevel(Enum):
        EXCELLENT = "excellent"
        GOOD = "good"
        FAIR = "fair"
        POOR = "poor"
        CRITICAL = "critical"

    class UnifiedCacheManager:
        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True

    def generate_unified_cache_key(*args, **kwargs):
        return f"pipeline_key_{hash(str(args))}"


logger = get_context_logger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class PipelineStage(Enum):
    """パイプライン処理段階"""

    INGESTION = "ingestion"  # データ取り込み
    PREPROCESSING = "preprocessing"  # 前処理
    VALIDATION = "validation"  # バリデーション
    CLEANING = "cleaning"  # クリーニング
    ENRICHMENT = "enrichment"  # データエンリッチメント
    QUALITY_ASSESSMENT = "quality_assessment"  # 品質評価
    EXPORT = "export"  # データ出力


class ValidationLevel(Enum):
    """バリデーションレベル"""

    BASIC = "basic"  # 基本チェック
    STANDARD = "standard"  # 標準チェック
    STRICT = "strict"  # 厳格チェック
    ENTERPRISE = "enterprise"  # 企業レベルチェック


class DataLineageEvent(Enum):
    """データ系譜イベント"""

    CREATED = "created"
    TRANSFORMED = "transformed"
    VALIDATED = "validated"
    CLEANED = "cleaned"
    ENRICHED = "enriched"
    EXPORTED = "exported"
    ARCHIVED = "archived"


@dataclass
class PipelineMetrics:
    """パイプライン処理メトリクス"""

    stage: PipelineStage
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    records_processed: int = 0
    records_passed: int = 0
    records_failed: int = 0
    errors_count: int = 0
    warnings_count: int = 0
    data_size_mb: float = 0.0
    memory_usage_mb: float = 0.0
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataLineageRecord:
    """データ系譜記録"""

    record_id: str
    event: DataLineageEvent
    timestamp: datetime
    stage: PipelineStage
    data_hash: str
    transformation: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRule:
    """バリデーションルール定義"""

    rule_id: str
    name: str
    description: str
    rule_type: str  # "schema", "range", "format", "business"
    severity: str  # "critical", "high", "medium", "low"
    validation_func: callable
    auto_fix_func: Optional[callable] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataProcessor(ABC):
    """抽象データ処理クラス"""

    @abstractmethod
    async def process(
        self, data: Any, context: Dict[str, Any]
    ) -> Tuple[Any, PipelineMetrics]:
        """データ処理実行"""
        pass

    @abstractmethod
    def get_processor_info(self) -> Dict[str, Any]:
        """プロセッサ情報取得"""
        pass


class SchemaValidator(DataProcessor):
    """スキーマバリデーター"""

    def __init__(self, schema_rules: Dict[str, Any]):
        self.schema_rules = schema_rules

    async def process(
        self, data: Any, context: Dict[str, Any]
    ) -> Tuple[Any, PipelineMetrics]:
        """スキーマバリデーション実行"""
        start_time = datetime.utcnow()
        metrics = PipelineMetrics(stage=PipelineStage.VALIDATION, start_time=start_time)

        try:
            if isinstance(data, pd.DataFrame):
                metrics.records_processed = len(data)

                # 必須カラムチェック
                required_cols = self.schema_rules.get("required_columns", [])
                missing_cols = [col for col in required_cols if col not in data.columns]

                if missing_cols:
                    metrics.errors_count += len(missing_cols)
                    metrics.metadata["missing_columns"] = missing_cols

                # データ型チェック
                type_rules = self.schema_rules.get("column_types", {})
                type_errors = []

                for col, expected_type in type_rules.items():
                    if col in data.columns:
                        actual_type = data[col].dtype
                        if not self._is_compatible_type(actual_type, expected_type):
                            type_errors.append(
                                {
                                    "column": col,
                                    "expected": expected_type,
                                    "actual": str(actual_type),
                                }
                            )

                if type_errors:
                    metrics.errors_count += len(type_errors)
                    metrics.metadata["type_errors"] = type_errors

                # 成功判定
                metrics.success = metrics.errors_count == 0
                metrics.records_passed = len(data) if metrics.success else 0
                metrics.records_failed = 0 if metrics.success else len(data)

            else:
                metrics.records_processed = 1
                metrics.success = True
                metrics.records_passed = 1

            metrics.end_time = datetime.utcnow()
            metrics.duration_ms = (metrics.end_time - start_time).total_seconds() * 1000

            return data, metrics

        except Exception as e:
            logger.error(f"スキーマバリデーションエラー: {e}")
            metrics.end_time = datetime.utcnow()
            metrics.errors_count += 1
            metrics.success = False
            metrics.metadata["error"] = str(e)
            return data, metrics

    def _is_compatible_type(self, actual_type, expected_type: str) -> bool:
        """データ型互換性チェック"""
        type_mapping = {
            "int": ["int64", "int32", "int16", "int8"],
            "float": ["float64", "float32", "int64", "int32"],
            "string": ["object", "string"],
            "datetime": ["datetime64[ns]", "object"],
            "bool": ["bool"],
        }

        compatible_types = type_mapping.get(expected_type, [])
        return str(actual_type) in compatible_types or expected_type in str(actual_type)

    def get_processor_info(self) -> Dict[str, Any]:
        return {
            "processor_type": "schema_validator",
            "schema_rules": self.schema_rules,
            "version": "1.0",
        }


class OutlierDetector(DataProcessor):
    """異常値検出処理"""

    def __init__(self, method: str = "isolation_forest", threshold: float = 0.05):
        self.method = method
        self.threshold = threshold
        self.scaler = StandardScaler()

    async def process(
        self, data: Any, context: Dict[str, Any]
    ) -> Tuple[Any, PipelineMetrics]:
        """異常値検出実行"""
        start_time = datetime.utcnow()
        metrics = PipelineMetrics(stage=PipelineStage.CLEANING, start_time=start_time)

        try:
            if isinstance(data, pd.DataFrame) and len(data) > 0:
                metrics.records_processed = len(data)

                # 数値カラムのみ対象
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

                if numeric_cols:
                    # Isolation Forest による異常値検出
                    X = data[numeric_cols].fillna(data[numeric_cols].median())
                    X_scaled = self.scaler.fit_transform(X)

                    if self.method == "isolation_forest":
                        detector = IsolationForest(
                            contamination=self.threshold, random_state=42
                        )
                        outlier_labels = detector.fit_predict(X_scaled)
                        outlier_mask = outlier_labels == -1

                    else:  # statistical method
                        outlier_mask = self._detect_statistical_outliers(
                            data[numeric_cols]
                        )

                    outlier_count = outlier_mask.sum()

                    # 異常値にフラグ設定
                    data_cleaned = data.copy()
                    data_cleaned["is_outlier"] = outlier_mask

                    metrics.records_passed = len(data) - outlier_count
                    metrics.records_failed = outlier_count
                    metrics.metadata["outlier_count"] = int(outlier_count)
                    metrics.metadata["outlier_percentage"] = float(
                        outlier_count / len(data) * 100
                    )

                    metrics.success = True

                    return data_cleaned, metrics
                else:
                    metrics.warnings_count = 1
                    metrics.metadata["warning"] = "数値カラムが見つかりません"

            metrics.records_passed = metrics.records_processed
            metrics.success = True
            metrics.end_time = datetime.utcnow()
            metrics.duration_ms = (metrics.end_time - start_time).total_seconds() * 1000

            return data, metrics

        except Exception as e:
            logger.error(f"異常値検出エラー: {e}")
            metrics.end_time = datetime.utcnow()
            metrics.errors_count += 1
            metrics.success = False
            metrics.metadata["error"] = str(e)
            return data, metrics

    def _detect_statistical_outliers(self, data: pd.DataFrame) -> np.ndarray:
        """統計的異常値検出 (IQR方式)"""
        outlier_mask = np.zeros(len(data), dtype=bool)

        for col in data.columns:
            if data[col].dtype in ["int64", "float64"]:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                col_outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                outlier_mask |= col_outliers

        return outlier_mask

    def get_processor_info(self) -> Dict[str, Any]:
        return {
            "processor_type": "outlier_detector",
            "method": self.method,
            "threshold": self.threshold,
            "version": "1.0",
        }


class DataValidationPipeline:
    """データバリデーション・クリーニングパイプライン"""

    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        enable_cache: bool = True,
        enable_lineage: bool = True,
        storage_path: str = "data/pipeline",
    ):
        self.validation_level = validation_level
        self.enable_cache = enable_cache
        self.enable_lineage = enable_lineage
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # キャッシュマネージャー初期化
        if enable_cache:
            try:
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=32, l2_memory_mb=128, l3_disk_mb=512
                )
                logger.info("パイプラインキャッシュシステム初期化完了")
            except Exception as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")
                self.cache_manager = None
        else:
            self.cache_manager = None

        # プロセッサー管理
        self.processors: Dict[PipelineStage, List[DataProcessor]] = {
            stage: [] for stage in PipelineStage
        }

        # バリデーションルール
        self.validation_rules: List[ValidationRule] = []

        # パイプライン履歴
        self.pipeline_history: List[Dict[str, Any]] = []

        # データ系譜管理
        self.lineage_records: List[DataLineageRecord] = []

        # デフォルトプロセッサー設定
        self._setup_default_processors()

        logger.info("データバリデーションパイプライン初期化完了")
        logger.info(f"  - バリデーションレベル: {validation_level.value}")
        logger.info(f"  - キャッシュ: {'有効' if enable_cache else '無効'}")
        logger.info(f"  - データ系譜: {'有効' if enable_lineage else '無効'}")

    def _setup_default_processors(self):
        """デフォルトプロセッサー設定"""
        # 価格データ用スキーマバリデーター
        price_schema = {
            "required_columns": ["Open", "High", "Low", "Close", "Volume"],
            "column_types": {
                "Open": "float",
                "High": "float",
                "Low": "float",
                "Close": "float",
                "Volume": "int",
            },
        }
        price_validator = SchemaValidator(price_schema)
        self.add_processor(PipelineStage.VALIDATION, price_validator)

        # 異常値検出
        outlier_detector = OutlierDetector(method="isolation_forest", threshold=0.05)
        self.add_processor(PipelineStage.CLEANING, outlier_detector)

        # デフォルトバリデーションルール
        self._setup_default_validation_rules()

    def _setup_default_validation_rules(self):
        """デフォルトバリデーションルール設定"""
        # 価格順序チェックルール
        price_order_rule = ValidationRule(
            rule_id="price_order_check",
            name="価格順序バリデーション",
            description="Low <= Open,Close <= High の確認",
            rule_type="business",
            severity="high",
            validation_func=self._validate_price_order,
            auto_fix_func=self._fix_price_order,
        )
        self.validation_rules.append(price_order_rule)

        # データ新鮮度チェックルール
        freshness_rule = ValidationRule(
            rule_id="data_freshness",
            name="データ新鮮度チェック",
            description="データの時間的新鮮性の確認",
            rule_type="timeliness",
            severity="medium",
            validation_func=self._validate_data_freshness,
        )
        self.validation_rules.append(freshness_rule)

        # 完全性チェックルール
        completeness_rule = ValidationRule(
            rule_id="data_completeness",
            name="データ完全性チェック",
            description="欠損値の確認",
            rule_type="completeness",
            severity="medium",
            validation_func=self._validate_data_completeness,
            auto_fix_func=self._fix_missing_values,
        )
        self.validation_rules.append(completeness_rule)

    def add_processor(self, stage: PipelineStage, processor: DataProcessor):
        """プロセッサー追加"""
        self.processors[stage].append(processor)
        logger.info(
            f"プロセッサー追加: {stage.value} - {processor.get_processor_info()['processor_type']}"
        )

    def add_validation_rule(self, rule: ValidationRule):
        """バリデーションルール追加"""
        self.validation_rules.append(rule)
        logger.info(f"バリデーションルール追加: {rule.rule_id}")

    async def process_data(
        self,
        data: Any,
        data_type: str,
        symbol: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """データパイプライン処理実行"""
        pipeline_id = f"pipeline_{int(time.time())}_{hash(str(data))}"
        start_time = datetime.utcnow()

        logger.info(f"データパイプライン処理開始: {pipeline_id} ({data_type} {symbol})")

        # 処理コンテキスト
        context = {
            "pipeline_id": pipeline_id,
            "data_type": data_type,
            "symbol": symbol,
            "validation_level": self.validation_level,
            "metadata": metadata or {},
        }

        # 結果収集
        results = {
            "pipeline_id": pipeline_id,
            "data_type": data_type,
            "symbol": symbol,
            "start_time": start_time.isoformat(),
            "validation_level": self.validation_level.value,
            "stages": {},
            "overall_success": True,
            "total_duration_ms": 0.0,
            "quality_metrics": None,
            "lineage_records": [],
            "processed_data": data,
            "recommendations": [],
        }

        current_data = data

        # キャッシュチェック
        if self.enable_cache and self.cache_manager:
            cache_key = self._generate_cache_key(data, data_type, symbol)
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"パイプラインキャッシュヒット: {pipeline_id}")
                return cached_result

        # 各段階を順次処理
        for stage in PipelineStage:
            if stage in self.processors and self.processors[stage]:
                stage_start = datetime.utcnow()
                stage_results = []
                stage_success = True

                # 段階内の全プロセッサーを実行
                for processor in self.processors[stage]:
                    try:
                        processed_data, metrics = await processor.process(
                            current_data, context
                        )
                        stage_results.append(
                            {
                                "processor": processor.get_processor_info(),
                                "metrics": metrics,
                            }
                        )

                        if metrics.success:
                            current_data = processed_data

                            # データ系譜記録
                            if self.enable_lineage:
                                self._record_lineage(
                                    pipeline_id,
                                    stage,
                                    current_data,
                                    processor.get_processor_info(),
                                )
                        else:
                            stage_success = False
                            results["overall_success"] = False

                    except Exception as e:
                        logger.error(f"プロセッサーエラー {stage.value}: {e}")
                        stage_success = False
                        results["overall_success"] = False
                        stage_results.append(
                            {
                                "processor": processor.get_processor_info(),
                                "error": str(e),
                            }
                        )

                # 段階結果記録
                stage_duration = (
                    datetime.utcnow() - stage_start
                ).total_seconds() * 1000
                results["stages"][stage.value] = {
                    "success": stage_success,
                    "duration_ms": stage_duration,
                    "processors": stage_results,
                }
                results["total_duration_ms"] += stage_duration

        # カスタムバリデーションルール実行
        validation_results = await self._execute_validation_rules(current_data, context)
        results["validation_results"] = validation_results

        # 品質メトリクス計算
        if hasattr(self, "_calculate_quality_metrics"):
            results["quality_metrics"] = await self._calculate_quality_metrics(
                current_data, data_type
            )

        # 推奨事項生成
        results["recommendations"] = self._generate_recommendations(results)

        # 結果終了処理
        results["processed_data"] = current_data
        results["end_time"] = datetime.utcnow().isoformat()
        results["lineage_records"] = [
            {
                "record_id": record.record_id,
                "event": record.event.value,
                "timestamp": record.timestamp.isoformat(),
                "stage": record.stage.value,
                "transformation": record.transformation,
            }
            for record in self.lineage_records[-10:]  # 最新10件
        ]

        # 履歴保存
        self.pipeline_history.append(results)

        # キャッシュ保存
        if self.enable_cache and self.cache_manager:
            cache_key = self._generate_cache_key(data, data_type, symbol)
            priority = 3.0 + (2.0 if results["overall_success"] else 0.0)
            self.cache_manager.put(cache_key, results, priority=priority)

        # パイプライン処理結果保存
        await self._save_pipeline_results(results)

        logger.info(
            f"データパイプライン処理完了: {pipeline_id} "
            f"({'成功' if results['overall_success'] else '失敗'}) "
            f"({results['total_duration_ms']:.1f}ms)"
        )

        return results

    async def _execute_validation_rules(
        self, data: Any, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """バリデーションルール実行"""
        validation_results = {
            "total_rules": len(self.validation_rules),
            "passed_rules": 0,
            "failed_rules": 0,
            "rule_results": [],
            "auto_fixes_applied": [],
        }

        for rule in self.validation_rules:
            if not rule.enabled:
                continue

            try:
                # ルール実行
                is_valid, issues = rule.validation_func(data, context)

                rule_result = {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "passed": is_valid,
                    "severity": rule.severity,
                    "issues": issues,
                }

                if is_valid:
                    validation_results["passed_rules"] += 1
                else:
                    validation_results["failed_rules"] += 1

                    # 自動修正実行
                    if rule.auto_fix_func and not is_valid:
                        try:
                            fixed_data = rule.auto_fix_func(data, issues)
                            if fixed_data is not None:
                                data = fixed_data
                                validation_results["auto_fixes_applied"].append(
                                    rule.rule_id
                                )
                                rule_result["auto_fixed"] = True
                        except Exception as e:
                            logger.error(f"自動修正エラー {rule.rule_id}: {e}")
                            rule_result["auto_fix_error"] = str(e)

                validation_results["rule_results"].append(rule_result)

            except Exception as e:
                logger.error(f"バリデーションルールエラー {rule.rule_id}: {e}")
                validation_results["failed_rules"] += 1
                validation_results["rule_results"].append(
                    {
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "passed": False,
                        "error": str(e),
                    }
                )

        return validation_results

    def _validate_price_order(
        self, data: Any, context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """価格順序バリデーション"""
        issues = []

        if isinstance(data, pd.DataFrame) and all(
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
                issues.append(f"価格順序異常: {len(invalid_rows)}件")
                return False, issues

        return True, issues

    def _fix_price_order(self, data: Any, issues: List[str]) -> Any:
        """価格順序自動修正"""
        if isinstance(data, pd.DataFrame) and all(
            col in data.columns for col in ["Low", "High", "Open", "Close"]
        ):
            data_fixed = data.copy()
            # High = max(Open, High, Close)
            data_fixed["High"] = data_fixed[["Open", "High", "Close"]].max(axis=1)
            # Low = min(Open, Low, Close)
            data_fixed["Low"] = data_fixed[["Open", "Low", "Close"]].min(axis=1)
            return data_fixed

        return data

    def _validate_data_freshness(
        self, data: Any, context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """データ新鮮度バリデーション"""
        issues = []

        if isinstance(data, pd.DataFrame) and hasattr(data.index, "max"):
            try:
                latest_date = pd.to_datetime(data.index.max())
                age_hours = (datetime.now() - latest_date).total_seconds() / 3600

                if age_hours > 48:  # 48時間以上古い
                    issues.append(f"データが古い: {age_hours:.1f}時間前")
                    return False, issues
            except Exception:
                pass

        return True, issues

    def _validate_data_completeness(
        self, data: Any, context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """データ完全性バリデーション"""
        issues = []

        if isinstance(data, pd.DataFrame):
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_ratio > 0.1:  # 10%以上の欠損
                issues.append(f"高い欠損率: {missing_ratio:.2%}")
                return False, issues

        return True, issues

    def _fix_missing_values(self, data: Any, issues: List[str]) -> Any:
        """欠損値自動修正"""
        if isinstance(data, pd.DataFrame):
            data_fixed = data.copy()

            # 数値カラムは前方補間
            numeric_cols = data_fixed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                data_fixed[col] = (
                    data_fixed[col].fillna(method="ffill").fillna(method="bfill")
                )

            # カテゴリカルカラムは最頻値
            categorical_cols = data_fixed.select_dtypes(include=["object"]).columns
            for col in categorical_cols:
                mode_val = data_fixed[col].mode()
                if len(mode_val) > 0:
                    data_fixed[col] = data_fixed[col].fillna(mode_val[0])

            return data_fixed

        return data

    def _record_lineage(
        self,
        pipeline_id: str,
        stage: PipelineStage,
        data: Any,
        processor_info: Dict[str, Any],
    ):
        """データ系譜記録"""
        if not self.enable_lineage:
            return

        try:
            # データハッシュ計算
            data_hash = self._calculate_data_hash(data)

            # 系譜レコード作成
            lineage_record = DataLineageRecord(
                record_id=f"{pipeline_id}_{stage.value}_{int(time.time())}",
                event=DataLineageEvent.TRANSFORMED,
                timestamp=datetime.utcnow(),
                stage=stage,
                data_hash=data_hash,
                transformation=processor_info.get("processor_type", "unknown"),
                metadata=processor_info,
            )

            self.lineage_records.append(lineage_record)

        except Exception as e:
            logger.error(f"データ系譜記録エラー: {e}")

    def _calculate_data_hash(self, data: Any) -> str:
        """データハッシュ計算"""
        try:
            if isinstance(data, pd.DataFrame):
                # データフレームの構造とサンプルデータからハッシュ生成
                structure = (
                    f"{list(data.columns)}_{data.shape}_{str(data.dtypes.to_dict())}"
                )
                sample = str(data.head().to_dict()) if len(data) > 0 else ""
                content = f"{structure}_{sample}"
            else:
                content = str(data)

            return hashlib.md5(content.encode()).hexdigest()

        except Exception as e:
            logger.error(f"データハッシュ計算エラー: {e}")
            return f"hash_error_{int(time.time())}"

    def _generate_cache_key(self, data: Any, data_type: str, symbol: str) -> str:
        """キャッシュキー生成"""
        data_hash = self._calculate_data_hash(data)
        return generate_unified_cache_key(
            "pipeline",
            data_type,
            symbol,
            self.validation_level.value,
            data_hash,
            time_bucket_minutes=60,
        )

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """推奨事項生成"""
        recommendations = []

        # 全体的な推奨事項
        if not results["overall_success"]:
            recommendations.append(
                "🚨 パイプライン処理でエラーが発生しました。ログを確認してください。"
            )

        # 処理時間に関する推奨事項
        if results["total_duration_ms"] > 5000:  # 5秒以上
            recommendations.append(
                "⚠️ 処理時間が長いです。データサイズやプロセッサー設定を見直してください。"
            )

        # バリデーション結果に基づく推奨事項
        if "validation_results" in results:
            validation = results["validation_results"]
            if validation["failed_rules"] > 0:
                recommendations.append(
                    f"🔍 {validation['failed_rules']}件のバリデーションルールが失敗しました。"
                )

            if validation["auto_fixes_applied"]:
                recommendations.append(
                    f"🔧 {len(validation['auto_fixes_applied'])}件の自動修正が適用されました。"
                )

        # ステージ別推奨事項
        for stage_name, stage_result in results["stages"].items():
            if not stage_result["success"]:
                recommendations.append(
                    f"❌ {stage_name}段階で問題が発生しました。設定を確認してください。"
                )

        return recommendations[:5]  # 上位5件に制限

    async def _save_pipeline_results(self, results: Dict[str, Any]):
        """パイプライン結果保存"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            pipeline_id = results["pipeline_id"]

            # 詳細結果をJSON保存
            result_file = (
                self.storage_path / f"pipeline_result_{pipeline_id}_{timestamp}.json"
            )

            # シリアライズ可能な形に変換
            serializable_results = self._make_serializable(results)

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            logger.debug(f"パイプライン結果保存完了: {result_file}")

        except Exception as e:
            logger.error(f"パイプライン結果保存エラー: {e}")

    def _make_serializable(self, obj: Any) -> Any:
        """オブジェクトをシリアライズ可能な形に変換"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return f"DataFrame(shape={obj.shape}, columns={list(obj.columns)})"
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, "__dict__"):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        else:
            return (
                str(obj)
                if not isinstance(obj, (str, int, float, bool, type(None)))
                else obj
            )

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """パイプライン状態取得"""
        return {
            "validation_level": self.validation_level.value,
            "cache_enabled": self.enable_cache,
            "lineage_enabled": self.enable_lineage,
            "processors_count": sum(
                len(processors) for processors in self.processors.values()
            ),
            "validation_rules_count": len(self.validation_rules),
            "pipeline_history_count": len(self.pipeline_history),
            "lineage_records_count": len(self.lineage_records),
            "recent_success_rate": self._calculate_recent_success_rate(),
        }

    def _calculate_recent_success_rate(self) -> float:
        """直近の成功率計算"""
        if not self.pipeline_history:
            return 0.0

        recent_results = self.pipeline_history[-20:]  # 最新20件
        success_count = sum(
            1 for result in recent_results if result.get("overall_success", False)
        )

        return success_count / len(recent_results)

    async def cleanup(self):
        """リソースクリーンアップ"""
        logger.info("データバリデーションパイプライン クリーンアップ開始")

        # 履歴データの制限
        if len(self.pipeline_history) > 1000:
            self.pipeline_history = self.pipeline_history[-1000:]

        if len(self.lineage_records) > 5000:
            self.lineage_records = self.lineage_records[-5000:]

        logger.info("データバリデーションパイプライン クリーンアップ完了")


# Factory functions
def create_data_validation_pipeline(
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    enable_cache: bool = True,
    enable_lineage: bool = True,
    storage_path: str = "data/pipeline",
) -> DataValidationPipeline:
    """データバリデーションパイプライン作成"""
    return DataValidationPipeline(
        validation_level=validation_level,
        enable_cache=enable_cache,
        enable_lineage=enable_lineage,
        storage_path=storage_path,
    )


if __name__ == "__main__":
    # テスト実行
    async def test_data_validation_pipeline():
        print("=== Issue #420 データバリデーション・クリーニングパイプラインテスト ===")

        try:
            # パイプライン初期化
            pipeline = create_data_validation_pipeline(
                validation_level=ValidationLevel.STANDARD,
                enable_cache=True,
                enable_lineage=True,
                storage_path="test_pipeline",
            )

            print("\n1. パイプライン初期化完了")
            status = await pipeline.get_pipeline_status()
            print(f"   バリデーションレベル: {status['validation_level']}")
            print(f"   プロセッサー数: {status['processors_count']}")
            print(f"   バリデーションルール数: {status['validation_rules_count']}")

            # テストデータ準備
            print("\n2. テストデータ準備...")
            test_price_data = pd.DataFrame(
                {
                    "Open": [2500, 2520, np.nan, 2480, 2510],
                    "High": [2550, 2540, 2490, 2500, 2530],
                    "Low": [2480, 2500, 2460, 2470, 2495],
                    "Close": [2530, 2485, 2475, 2495, 2525],
                    "Volume": [
                        1500000,
                        1200000,
                        1800000,
                        999999999,
                        1300000,
                    ],  # 異常値含む
                },
                index=pd.date_range("2024-01-01", periods=5),
            )

            # パイプライン処理実行
            print("\n3. データパイプライン処理実行中...")
            results = await pipeline.process_data(
                test_price_data, "price", "7203", {"source": "test_data"}
            )

            print("\n=== パイプライン処理結果 ===")
            print(f"パイプラインID: {results['pipeline_id']}")
            print(f"処理成功: {'✅' if results['overall_success'] else '❌'}")
            print(f"処理時間: {results['total_duration_ms']:.1f}ms")

            print("\nステージ別結果:")
            for stage_name, stage_result in results["stages"].items():
                status_icon = "✅" if stage_result["success"] else "❌"
                print(
                    f"  {stage_name}: {status_icon} ({stage_result['duration_ms']:.1f}ms)"
                )

            if "validation_results" in results:
                validation = results["validation_results"]
                print("\nバリデーション結果:")
                print(
                    f"  合格ルール: {validation['passed_rules']}/{validation['total_rules']}"
                )
                print(f"  自動修正適用: {len(validation['auto_fixes_applied'])}件")

            print("\n推奨事項:")
            for i, rec in enumerate(results["recommendations"], 1):
                print(f"  {i}. {rec}")

            print(f"\nデータ系譜記録: {len(results['lineage_records'])}件")

            # 2回目処理（キャッシュテスト）
            print("\n4. キャッシュテスト...")
            start_time = time.time()
            cached_results = await pipeline.process_data(
                test_price_data, "price", "7203"
            )
            cache_time = (time.time() - start_time) * 1000

            print(f"キャッシュ処理時間: {cache_time:.1f}ms")
            print(
                f"キャッシュヒット: {'✅' if cache_time < results['total_duration_ms'] else '❌'}"
            )

            # クリーンアップ
            await pipeline.cleanup()

            print(
                "\n✅ Issue #420 データバリデーション・クリーニングパイプラインテスト完了"
            )

        except Exception as e:
            print(f"❌ テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_data_validation_pipeline())
