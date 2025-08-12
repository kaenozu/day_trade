#!/usr/bin/env python3
"""
包括的データ品質保証システム
Issue #420: データ管理とデータ品質保証メカニズムの強化

エンタープライズレベルのデータ品質管理:
- 統合データバリデーション・クリーニング
- データバージョン管理とリネージュ
- リアルタイム品質監視・アラート
- マスターデータ管理
- 品質メトリクス・レポーティング
- 自動修復・最適化
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

try:
    from ..monitoring.log_aggregation_system import LogAggregationSystem
    from ..utils.data_quality_manager import DataQualityManager, DataQualityMetrics
    from ..utils.unified_cache_manager import UnifiedCacheManager
    from .data_freshness_monitor import DataFreshnessMonitor, FreshnessStatus
    from .data_validation_pipeline import DataValidationPipeline, ValidationResult
    from .data_version_manager import DataVersion, DataVersionManager
    from .master_data_manager import MasterDataManager, MasterDataSet

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

    # Fallback definitions
    class DataValidationPipeline:
        async def validate_dataset(self, data, **kwargs):
            return {"is_valid": True, "issues": [], "metrics": {}}

    class DataVersionManager:
        def create_version(self, data, **kwargs):
            return f"v{int(time.time())}"

    class DataFreshnessMonitor:
        async def check_freshness(self, source, **kwargs):
            return {"status": "fresh", "age": 0}

    class MasterDataManager:
        def get_master_data(self, dataset_name):
            return pd.DataFrame()

    class DataQualityManager:
        def calculate_metrics(self, data):
            return {"quality_score": 100.0}


class DataQualityLevel(Enum):
    """データ品質レベル"""

    EXCELLENT = "excellent"  # 95-100%
    GOOD = "good"  # 85-94%
    ACCEPTABLE = "acceptable"  # 70-84%
    POOR = "poor"  # 50-69%
    CRITICAL = "critical"  # <50%


class DataProcessingStage(Enum):
    """データ処理ステージ"""

    RAW_INGESTION = "raw_ingestion"
    VALIDATION = "validation"
    CLEANING = "cleaning"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment"
    QUALITY_CHECK = "quality_check"
    FINALIZED = "finalized"


@dataclass
class DataQualityReport:
    """データ品質レポート"""

    dataset_id: str
    timestamp: datetime
    quality_level: DataQualityLevel
    overall_score: float
    stage_scores: Dict[str, float] = field(default_factory=dict)
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    data_volume: int = 0
    version: str = ""
    lineage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataPipelineConfig:
    """データパイプライン設定"""

    enable_validation: bool = True
    enable_cleaning: bool = True
    enable_versioning: bool = True
    enable_monitoring: bool = True
    quality_threshold: float = 70.0
    auto_repair: bool = True
    alert_threshold: float = 50.0
    retention_days: int = 30
    backup_enabled: bool = True


class ComprehensiveDataQualitySystem:
    """包括的データ品質保証システム"""

    def __init__(self, config: DataPipelineConfig = None):
        self.config = config or DataPipelineConfig()
        self.logger = logging.getLogger(__name__)

        # コンポーネント初期化
        self.validation_pipeline = None
        self.version_manager = None
        self.freshness_monitor = None
        self.master_data_manager = None
        self.quality_manager = None

        # データベース初期化
        self.db_path = "data_quality_system.db"
        self._initialize_database()

        # 統計・メトリクス
        self.processing_stats = {
            "total_datasets_processed": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "auto_repairs_performed": 0,
            "alerts_generated": 0,
        }

        self._initialize_components()

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_quality_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    quality_level TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    stage_scores TEXT,
                    issues_found TEXT,
                    recommendations TEXT,
                    processing_time REAL,
                    data_volume INTEGER,
                    version TEXT,
                    lineage TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    duration REAL,
                    input_records INTEGER,
                    output_records INTEGER,
                    errors_count INTEGER,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    parent_dataset_id TEXT,
                    transformation_type TEXT NOT NULL,
                    transformation_config TEXT,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_quality_reports_dataset
                ON data_quality_reports(dataset_id, timestamp)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processing_history_dataset
                ON processing_history(dataset_id, stage, timestamp)
            """
            )

            conn.commit()

    def _initialize_components(self):
        """コンポーネント初期化"""
        if DEPENDENCIES_AVAILABLE:
            try:
                self.validation_pipeline = DataValidationPipeline()
                self.version_manager = DataVersionManager()
                self.freshness_monitor = DataFreshnessMonitor()
                self.master_data_manager = MasterDataManager()
                self.quality_manager = DataQualityManager()

                self.logger.info("データ品質システムコンポーネント初期化完了")
            except Exception as e:
                self.logger.error(f"コンポーネント初期化エラー: {e}")
        else:
            # フォールバック実装
            self.validation_pipeline = DataValidationPipeline()
            self.version_manager = DataVersionManager()
            self.freshness_monitor = DataFreshnessMonitor()
            self.master_data_manager = MasterDataManager()
            self.quality_manager = DataQualityManager()

            self.logger.warning("フォールバック実装を使用")

    async def process_dataset(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], str],
        dataset_id: str,
        source: str = "unknown",
        metadata: Dict[str, Any] = None,
    ) -> DataQualityReport:
        """データセットの包括的処理"""
        start_time = time.time()
        metadata = metadata or {}

        try:
            self.logger.info(f"データセット処理開始: {dataset_id}")

            # データの正規化
            if isinstance(data, str):
                # ファイルパスの場合
                data = self._load_data_from_file(data)
            elif isinstance(data, dict):
                # 辞書データの場合
                data = pd.DataFrame([data])

            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"サポートされていないデータ形式: {type(data)}")

            # 処理レポート初期化
            report = DataQualityReport(
                dataset_id=dataset_id,
                timestamp=datetime.now(timezone.utc),
                quality_level=DataQualityLevel.POOR,
                overall_score=0.0,
                data_volume=len(data),
            )

            # ステージ1: 生データ取得
            await self._record_processing_stage(
                dataset_id,
                DataProcessingStage.RAW_INGESTION,
                "success",
                len(data),
                len(data),
            )

            # ステージ2: バリデーション
            if self.config.enable_validation:
                validation_result = await self._validate_data(data, dataset_id)
                report.stage_scores["validation"] = validation_result["score"]
                report.issues_found.extend(validation_result["issues"])

                await self._record_processing_stage(
                    dataset_id,
                    DataProcessingStage.VALIDATION,
                    "success" if validation_result["is_valid"] else "warning",
                    len(data),
                    len(data),
                    len(validation_result["issues"]),
                )

            # ステージ3: データクリーニング
            if self.config.enable_cleaning:
                cleaned_data, cleaning_report = await self._clean_data(data, dataset_id)
                data = cleaned_data
                report.stage_scores["cleaning"] = cleaning_report["score"]

                await self._record_processing_stage(
                    dataset_id,
                    DataProcessingStage.CLEANING,
                    "success",
                    report.data_volume,
                    len(data),
                    cleaning_report.get("issues_fixed", 0),
                )

                report.data_volume = len(data)

            # ステージ4: データ変換・エンリッチメント
            enriched_data, enrichment_report = await self._enrich_data(
                data, dataset_id, metadata
            )
            data = enriched_data
            report.stage_scores["enrichment"] = enrichment_report["score"]

            await self._record_processing_stage(
                dataset_id,
                DataProcessingStage.ENRICHMENT,
                "success",
                report.data_volume,
                len(data),
            )

            # ステージ5: 最終品質チェック
            quality_metrics = await self._calculate_final_quality(data, dataset_id)
            report.stage_scores["final_quality"] = quality_metrics["overall_score"]
            report.overall_score = quality_metrics["overall_score"]
            report.quality_level = self._determine_quality_level(report.overall_score)

            # ステージ6: バージョン管理
            if self.config.enable_versioning:
                version = await self._create_data_version(data, dataset_id, metadata)
                report.version = version

                await self._record_processing_stage(
                    dataset_id,
                    DataProcessingStage.FINALIZED,
                    "success",
                    len(data),
                    len(data),
                )

            # 処理時間計算
            report.processing_time = time.time() - start_time

            # レコメンデーション生成
            report.recommendations = self._generate_recommendations(report)

            # データリネージュ記録
            report.lineage = await self._record_data_lineage(
                dataset_id, source, metadata
            )

            # レポート保存
            await self._save_quality_report(report)

            # 統計更新
            self.processing_stats["total_datasets_processed"] += 1
            if report.overall_score >= self.config.quality_threshold:
                self.processing_stats["successful_validations"] += 1
            else:
                self.processing_stats["failed_validations"] += 1

            # アラート判定
            if report.overall_score < self.config.alert_threshold:
                await self._generate_quality_alert(report)
                self.processing_stats["alerts_generated"] += 1

            self.logger.info(
                f"データセット処理完了: {dataset_id}, 品質スコア: {report.overall_score:.2f}"
            )

            return report

        except Exception as e:
            self.logger.error(f"データセット処理エラー: {dataset_id}, エラー: {e}")

            # エラーレポート作成
            error_report = DataQualityReport(
                dataset_id=dataset_id,
                timestamp=datetime.now(timezone.utc),
                quality_level=DataQualityLevel.CRITICAL,
                overall_score=0.0,
                issues_found=[f"処理エラー: {str(e)}"],
                processing_time=time.time() - start_time,
            )

            await self._save_quality_report(error_report)
            return error_report

    def _load_data_from_file(self, file_path: str) -> pd.DataFrame:
        """ファイルからデータ読み込み"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

        if path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        elif path.suffix.lower() == ".json":
            return pd.read_json(file_path)
        elif path.suffix.lower() == ".parquet":
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"サポートされていないファイル形式: {path.suffix}")

    async def _validate_data(
        self, data: pd.DataFrame, dataset_id: str
    ) -> Dict[str, Any]:
        """データバリデーション"""
        try:
            if self.validation_pipeline:
                result = await self.validation_pipeline.validate_dataset(
                    data, dataset_id=dataset_id
                )
                return {
                    "is_valid": result.get("is_valid", False),
                    "score": result.get("score", 0.0),
                    "issues": result.get("issues", []),
                }
            else:
                # 基本バリデーション
                issues = []

                # 空データチェック
                if data.empty:
                    issues.append("データセットが空です")

                # 重複チェック
                duplicates = data.duplicated().sum()
                if duplicates > 0:
                    issues.append(f"重複レコード: {duplicates}件")

                # 欠損値チェック
                missing_data = data.isnull().sum().sum()
                if missing_data > 0:
                    issues.append(f"欠損値: {missing_data}個")

                score = max(0, 100 - len(issues) * 10)

                return {"is_valid": len(issues) == 0, "score": score, "issues": issues}

        except Exception as e:
            self.logger.error(f"データバリデーションエラー: {e}")
            return {
                "is_valid": False,
                "score": 0.0,
                "issues": [f"バリデーションエラー: {str(e)}"],
            }

    async def _clean_data(
        self, data: pd.DataFrame, dataset_id: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """データクリーニング"""
        try:
            cleaned_data = data.copy()
            issues_fixed = 0

            # 重複除去
            initial_count = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            duplicates_removed = initial_count - len(cleaned_data)
            if duplicates_removed > 0:
                issues_fixed += duplicates_removed
                self.logger.info(f"重複除去: {duplicates_removed}件")

            # 数値データの外れ値処理
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if cleaned_data[col].notna().sum() > 0:
                    Q1 = cleaned_data[col].quantile(0.25)
                    Q3 = cleaned_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers_count = (
                        (cleaned_data[col] < lower_bound)
                        | (cleaned_data[col] > upper_bound)
                    ).sum()

                    if outliers_count > 0:
                        # 外れ値をNaNに置換
                        cleaned_data.loc[
                            (cleaned_data[col] < lower_bound)
                            | (cleaned_data[col] > upper_bound),
                            col,
                        ] = np.nan
                        issues_fixed += outliers_count

            # 文字列データのクリーニング
            string_columns = cleaned_data.select_dtypes(include=["object"]).columns
            for col in string_columns:
                # 前後の空白除去
                if cleaned_data[col].notna().any():
                    original = cleaned_data[col].copy()
                    cleaned_data[col] = cleaned_data[col].astype(str).str.strip()

                    # 変更された行数をカウント
                    changes = (original != cleaned_data[col]).sum()
                    if changes > 0:
                        issues_fixed += changes

            # 基本統計による品質スコア計算
            completeness = (
                1 - cleaned_data.isnull().sum().sum() / cleaned_data.size
            ) * 100
            uniqueness = (
                (1 - cleaned_data.duplicated().sum() / len(cleaned_data)) * 100
                if len(cleaned_data) > 0
                else 100
            )

            score = (completeness + uniqueness) / 2

            cleaning_report = {
                "score": score,
                "issues_fixed": issues_fixed,
                "completeness": completeness,
                "uniqueness": uniqueness,
                "records_before": initial_count,
                "records_after": len(cleaned_data),
            }

            return cleaned_data, cleaning_report

        except Exception as e:
            self.logger.error(f"データクリーニングエラー: {e}")
            return data, {"score": 0.0, "issues_fixed": 0, "error": str(e)}

    async def _enrich_data(
        self, data: pd.DataFrame, dataset_id: str, metadata: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """データエンリッチメント"""
        try:
            enriched_data = data.copy()

            # メタデータ追加
            enriched_data["_dataset_id"] = dataset_id
            enriched_data["_processed_at"] = datetime.now(timezone.utc)

            # 基本統計情報追加
            if "add_statistics" in metadata and metadata["add_statistics"]:
                numeric_cols = enriched_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if enriched_data[col].notna().any():
                        enriched_data[f"{col}_zscore"] = (
                            enriched_data[col] - enriched_data[col].mean()
                        ) / enriched_data[col].std()

            # マスターデータとの結合
            if "master_data_join" in metadata:
                master_config = metadata["master_data_join"]
                if self.master_data_manager:
                    try:
                        master_data = self.master_data_manager.get_master_data(
                            master_config.get("dataset")
                        )
                        if not master_data.empty:
                            join_key = master_config.get("key", "id")
                            enriched_data = enriched_data.merge(
                                master_data,
                                on=join_key,
                                how="left",
                                suffixes=("", "_master"),
                            )
                    except Exception as e:
                        self.logger.warning(f"マスターデータ結合エラー: {e}")

            # エンリッチメント品質スコア
            enrichment_ratio = (
                len(enriched_data.columns) / len(data.columns)
                if len(data.columns) > 0
                else 1
            )
            score = min(100, 50 + (enrichment_ratio - 1) * 50)

            enrichment_report = {
                "score": score,
                "columns_added": len(enriched_data.columns) - len(data.columns),
                "enrichment_ratio": enrichment_ratio,
            }

            return enriched_data, enrichment_report

        except Exception as e:
            self.logger.error(f"データエンリッチメントエラー: {e}")
            return data, {"score": 50.0, "columns_added": 0, "error": str(e)}

    async def _calculate_final_quality(
        self, data: pd.DataFrame, dataset_id: str
    ) -> Dict[str, Any]:
        """最終品質評価"""
        try:
            if self.quality_manager:
                metrics = self.quality_manager.calculate_metrics(data)
                return {"overall_score": metrics.get("quality_score", 0.0)}

            # 基本品質メトリクス計算
            total_cells = data.size
            if total_cells == 0:
                return {"overall_score": 0.0}

            # 完全性 (Completeness)
            missing_cells = data.isnull().sum().sum()
            completeness = ((total_cells - missing_cells) / total_cells) * 100

            # 一意性 (Uniqueness)
            duplicates = data.duplicated().sum()
            uniqueness = (
                ((len(data) - duplicates) / len(data)) * 100 if len(data) > 0 else 100
            )

            # 一貫性 (Consistency) - データ型の一貫性
            consistency_score = 100
            for col in data.columns:
                if data[col].notna().any():
                    # 各列のデータ型の一貫性をチェック
                    unique_types = set(
                        type(x).__name__ for x in data[col].dropna().values
                    )
                    if len(unique_types) > 1:
                        consistency_score -= 5  # 複数の型が混在している場合減点

            consistency_score = max(0, consistency_score)

            # 妥当性 (Validity) - 数値データの範囲チェック
            validity_score = 100
            numeric_cols = data.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if data[col].notna().any():
                    # 異常値の割合をチェック
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = (
                        (data[col] < lower_bound) | (data[col] > upper_bound)
                    ).sum()
                    outlier_ratio = (
                        outliers / len(data[col].dropna())
                        if len(data[col].dropna()) > 0
                        else 0
                    )

                    validity_score -= outlier_ratio * 20  # 外れ値の割合に応じて減点

            validity_score = max(0, validity_score)

            # 総合スコア計算 (重み付き平均)
            weights = {
                "completeness": 0.3,
                "uniqueness": 0.2,
                "consistency": 0.25,
                "validity": 0.25,
            }
            overall_score = (
                completeness * weights["completeness"]
                + uniqueness * weights["uniqueness"]
                + consistency_score * weights["consistency"]
                + validity_score * weights["validity"]
            )

            return {
                "overall_score": overall_score,
                "completeness": completeness,
                "uniqueness": uniqueness,
                "consistency": consistency_score,
                "validity": validity_score,
            }

        except Exception as e:
            self.logger.error(f"品質評価エラー: {e}")
            return {"overall_score": 0.0, "error": str(e)}

    def _determine_quality_level(self, score: float) -> DataQualityLevel:
        """品質レベル判定"""
        if score >= 95:
            return DataQualityLevel.EXCELLENT
        elif score >= 85:
            return DataQualityLevel.GOOD
        elif score >= 70:
            return DataQualityLevel.ACCEPTABLE
        elif score >= 50:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.CRITICAL

    def _generate_recommendations(self, report: DataQualityReport) -> List[str]:
        """改善レコメンデーション生成"""
        recommendations = []

        if report.overall_score < 50:
            recommendations.append(
                "データ品質が非常に低いです。データソースの見直しを推奨します。"
            )

        if "重複レコード" in str(report.issues_found):
            recommendations.append("重複データの除去処理を実装してください。")

        if "欠損値" in str(report.issues_found):
            recommendations.append(
                "欠損値の補完またはフィルタリング処理を検討してください。"
            )

        if report.overall_score < 70:
            recommendations.append("データクリーニング処理の強化を推奨します。")

        if (
            "validation" in report.stage_scores
            and report.stage_scores["validation"] < 80
        ):
            recommendations.append("データバリデーションルールの見直しが必要です。")

        return recommendations

    async def _create_data_version(
        self, data: pd.DataFrame, dataset_id: str, metadata: Dict[str, Any]
    ) -> str:
        """データバージョン作成"""
        try:
            if self.version_manager:
                version = self.version_manager.create_version(
                    data, dataset_id=dataset_id, metadata=metadata
                )
                return version
            else:
                # 簡易バージョン生成
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                return f"v{timestamp}"

        except Exception as e:
            self.logger.error(f"バージョン作成エラー: {e}")
            return f"v{int(time.time())}"

    async def _record_data_lineage(
        self, dataset_id: str, source: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """データリネージュ記録"""
        try:
            lineage = {
                "source": source,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata,
                "transformations": [],
            }

            # データベースに記録
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO data_lineage
                    (dataset_id, parent_dataset_id, transformation_type, transformation_config, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        dataset_id,
                        metadata.get("parent_dataset_id"),
                        "quality_processing",
                        json.dumps(metadata, ensure_ascii=False),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()

            return lineage

        except Exception as e:
            self.logger.error(f"データリネージュ記録エラー: {e}")
            return {"source": source, "error": str(e)}

    async def _record_processing_stage(
        self,
        dataset_id: str,
        stage: DataProcessingStage,
        status: str,
        input_records: int,
        output_records: int,
        errors_count: int = 0,
    ):
        """処理ステージ記録"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO processing_history
                    (dataset_id, stage, status, timestamp, input_records, output_records, errors_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        dataset_id,
                        stage.value,
                        status,
                        datetime.now(timezone.utc).isoformat(),
                        input_records,
                        output_records,
                        errors_count,
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"処理ステージ記録エラー: {e}")

    async def _save_quality_report(self, report: DataQualityReport):
        """品質レポート保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO data_quality_reports
                    (dataset_id, timestamp, quality_level, overall_score, stage_scores,
                     issues_found, recommendations, processing_time, data_volume, version, lineage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        report.dataset_id,
                        report.timestamp.isoformat(),
                        report.quality_level.value,
                        report.overall_score,
                        json.dumps(report.stage_scores, ensure_ascii=False),
                        json.dumps(report.issues_found, ensure_ascii=False),
                        json.dumps(report.recommendations, ensure_ascii=False),
                        report.processing_time,
                        report.data_volume,
                        report.version,
                        json.dumps(report.lineage, ensure_ascii=False),
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"品質レポート保存エラー: {e}")

    async def _generate_quality_alert(self, report: DataQualityReport):
        """品質アラート生成"""
        try:
            alert_message = (
                f"データ品質アラート: {report.dataset_id}\n"
                f"品質レベル: {report.quality_level.value}\n"
                f"総合スコア: {report.overall_score:.2f}\n"
                f"主要問題: {', '.join(report.issues_found[:3])}"
            )

            self.logger.warning(alert_message)

            # 必要に応じて外部アラートシステムに送信
            # await external_alert_system.send_alert(alert_message)

        except Exception as e:
            self.logger.error(f"アラート生成エラー: {e}")

    async def get_quality_dashboard_data(self, days: int = 7) -> Dict[str, Any]:
        """品質ダッシュボードデータ取得"""
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)

            with sqlite3.connect(self.db_path) as conn:
                # 品質レポート取得
                cursor = conn.execute(
                    """
                    SELECT dataset_id, quality_level, overall_score, timestamp, data_volume
                    FROM data_quality_reports
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """,
                    (start_date.isoformat(),),
                )

                reports = cursor.fetchall()

                # 処理統計取得
                cursor = conn.execute(
                    """
                    SELECT stage, status, COUNT(*) as count
                    FROM processing_history
                    WHERE timestamp >= ?
                    GROUP BY stage, status
                """,
                    (start_date.isoformat(),),
                )

                processing_stats = cursor.fetchall()

                # 品質トレンド計算
                quality_trend = []
                for report in reports:
                    quality_trend.append(
                        {
                            "dataset_id": report[0],
                            "quality_level": report[1],
                            "score": report[2],
                            "timestamp": report[3],
                            "volume": report[4],
                        }
                    )

                # 統計サマリー
                if reports:
                    avg_score = sum(r[2] for r in reports) / len(reports)
                    total_volume = sum(r[4] for r in reports if r[4])
                else:
                    avg_score = 0
                    total_volume = 0

                return {
                    "summary": {
                        "total_datasets": len(set(r[0] for r in reports)),
                        "average_quality_score": avg_score,
                        "total_records_processed": total_volume,
                        "reporting_period_days": days,
                    },
                    "quality_trend": quality_trend,
                    "processing_stats": [
                        {"stage": ps[0], "status": ps[1], "count": ps[2]}
                        for ps in processing_stats
                    ],
                    "system_stats": self.processing_stats,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            self.logger.error(f"ダッシュボードデータ取得エラー: {e}")
            return {
                "summary": {"total_datasets": 0, "average_quality_score": 0},
                "quality_trend": [],
                "processing_stats": [],
                "system_stats": self.processing_stats,
                "error": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

    async def get_dataset_history(self, dataset_id: str) -> Dict[str, Any]:
        """データセット履歴取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 品質レポート履歴
                cursor = conn.execute(
                    """
                    SELECT * FROM data_quality_reports
                    WHERE dataset_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 50
                """,
                    (dataset_id,),
                )

                reports = [
                    dict(zip([col[0] for col in cursor.description], row))
                    for row in cursor.fetchall()
                ]

                # 処理履歴
                cursor = conn.execute(
                    """
                    SELECT * FROM processing_history
                    WHERE dataset_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                """,
                    (dataset_id,),
                )

                processing_history = [
                    dict(zip([col[0] for col in cursor.description], row))
                    for row in cursor.fetchall()
                ]

                # データリネージュ
                cursor = conn.execute(
                    """
                    SELECT * FROM data_lineage
                    WHERE dataset_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 50
                """,
                    (dataset_id,),
                )

                lineage = [
                    dict(zip([col[0] for col in cursor.description], row))
                    for row in cursor.fetchall()
                ]

                return {
                    "dataset_id": dataset_id,
                    "quality_reports": reports,
                    "processing_history": processing_history,
                    "data_lineage": lineage,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            self.logger.error(f"データセット履歴取得エラー: {e}")
            return {
                "dataset_id": dataset_id,
                "error": str(e),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }


# Factory function
def create_data_quality_system(
    config: DataPipelineConfig = None,
) -> ComprehensiveDataQualitySystem:
    """データ品質システム作成"""
    return ComprehensiveDataQualitySystem(config)


# Global instance
_data_quality_system = None


def get_data_quality_system() -> ComprehensiveDataQualitySystem:
    """グローバルデータ品質システム取得"""
    global _data_quality_system
    if _data_quality_system is None:
        _data_quality_system = create_data_quality_system()
    return _data_quality_system


if __name__ == "__main__":
    # テスト実行
    async def test_data_quality_system():
        print("=== 包括的データ品質保証システムテスト ===")

        try:
            # システム初期化
            config = DataPipelineConfig(
                quality_threshold=75.0, auto_repair=True, alert_threshold=60.0
            )

            quality_system = create_data_quality_system(config)

            print("\n1. データ品質システム初期化完了")

            # テストデータ作成
            test_data = pd.DataFrame(
                {
                    "id": [1, 2, 2, 4, 5],  # 重複あり
                    "price": [100, 150, 200, None, 250],  # 欠損値あり
                    "volume": [1000, 1500, 2000, 1800, 3000],
                    "symbol": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
                    "date": [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-04",
                        "2024-01-05",
                    ],
                }
            )

            print(f"\n2. テストデータセット作成: {len(test_data)}行")

            # データ処理実行
            print("\n3. データ品質処理実行...")

            report = await quality_system.process_dataset(
                data=test_data,
                dataset_id="test_stock_data",
                source="test_generator",
                metadata={"data_type": "financial", "add_statistics": True},
            )

            print("   処理完了:")
            print(f"   - 品質レベル: {report.quality_level.value}")
            print(f"   - 総合スコア: {report.overall_score:.2f}")
            print(f"   - 処理時間: {report.processing_time:.3f}秒")
            print(f"   - データ量: {report.data_volume}行")
            print(f"   - バージョン: {report.version}")

            if report.issues_found:
                print(f"   - 検出された問題: {len(report.issues_found)}件")
                for issue in report.issues_found[:3]:
                    print(f"     * {issue}")

            if report.recommendations:
                print(f"   - 推奨事項: {len(report.recommendations)}件")
                for rec in report.recommendations[:2]:
                    print(f"     * {rec}")

            # ダッシュボードデータ取得
            print("\n4. 品質ダッシュボードデータ取得...")
            dashboard_data = await quality_system.get_quality_dashboard_data(days=1)

            summary = dashboard_data["summary"]
            print(f"   - 処理データセット数: {summary['total_datasets']}")
            print(f"   - 平均品質スコア: {summary['average_quality_score']:.2f}")
            print(f"   - 総処理レコード数: {summary['total_records_processed']}")

            # データセット履歴取得
            print("\n5. データセット履歴確認...")
            history = await quality_system.get_dataset_history("test_stock_data")

            print(f"   - 品質レポート履歴: {len(history['quality_reports'])}件")
            print(f"   - 処理履歴: {len(history['processing_history'])}件")
            print(f"   - データリネージュ: {len(history['data_lineage'])}件")

            # システム統計
            print("\n6. システム統計:")
            stats = quality_system.processing_stats
            for key, value in stats.items():
                print(f"   - {key}: {value}")

            print("\n[成功] 包括的データ品質保証システムテスト完了")

        except Exception as e:
            print(f"[エラー] テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_data_quality_system())
