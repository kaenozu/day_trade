#!/usr/bin/env python3
"""
実用的データ品質管理システム
Issue #420対応 - シンプルかつ実用的なデータ品質保証
"""

import json
import logging
import sqlite3
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityLevel(Enum):
    """データ品質レベル"""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class IssueType(Enum):
    """問題タイプ"""

    MISSING_DATA = "missing_data"
    DUPLICATE_DATA = "duplicate_data"
    INVALID_VALUE = "invalid_value"
    INCONSISTENCY = "inconsistency"
    OUTLIER = "outlier"
    FORMAT_ERROR = "format_error"


@dataclass
class DataQualityIssue:
    """データ品質問題"""

    issue_type: IssueType
    column: str
    description: str
    severity: DataQualityLevel
    count: int
    examples: List[Any]
    suggested_fix: str


@dataclass
class DataQualityReport:
    """データ品質レポート"""

    dataset_name: str
    timestamp: datetime
    total_records: int
    quality_score: float
    quality_level: DataQualityLevel
    issues: List[DataQualityIssue]
    summary: Dict[str, Any]


class PracticalDataQualityManager:
    """実用的データ品質管理システム"""

    def __init__(self, storage_path: Path = Path("data_quality")):
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True, parents=True)

        # SQLiteデータベース初期化
        self.db_path = self.storage_path / "data_quality.db"
        self._init_database()

        # 金融データ用バリデーションルール
        self.financial_validation_rules = {
            "price_columns": ["open", "high", "low", "close", "adj_close"],
            "volume_columns": ["volume"],
            "required_columns": ["symbol", "date"],
            "min_price": 0.01,  # 最小価格
            "max_price": 10000000,  # 最大価格（1000万円）
            "min_volume": 0,
            "max_volume": 1e12,  # 最大出来高
        }

    def _init_database(self) -> None:
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    quality_level TEXT NOT NULL,
                    total_records INTEGER NOT NULL,
                    issues_count INTEGER NOT NULL,
                    report_data TEXT NOT NULL
                )
            """
            )
            conn.commit()

    def validate_financial_data(
        self, df: pd.DataFrame, dataset_name: str = "unnamed_dataset"
    ) -> DataQualityReport:
        """金融データの品質検証"""
        logger.info(f"データ品質検証開始: {dataset_name}")

        issues = []
        total_records = len(df)

        # 1. 必須列の存在確認
        for col in self.financial_validation_rules["required_columns"]:
            if col not in df.columns:
                issues.append(
                    DataQualityIssue(
                        issue_type=IssueType.MISSING_DATA,
                        column=col,
                        description=f"必須列 '{col}' が存在しません",
                        severity=DataQualityLevel.CRITICAL,
                        count=1,
                        examples=[],
                        suggested_fix=f"列 '{col}' を追加してください",
                    )
                )

        # 2. 欠損値チェック
        missing_data = df.isnull().sum()
        for col, count in missing_data[missing_data > 0].items():
            severity = (
                DataQualityLevel.CRITICAL
                if col in self.financial_validation_rules["required_columns"]
                else DataQualityLevel.POOR
            )
            issues.append(
                DataQualityIssue(
                    issue_type=IssueType.MISSING_DATA,
                    column=col,
                    description=f"欠損値が {count}/{total_records} 件存在",
                    severity=severity,
                    count=int(count),
                    examples=[],
                    suggested_fix="前方補完またはデータソースでの補完を検討",
                )
            )

        # 3. 重複データチェック
        duplicates = df.duplicated()
        if duplicates.any():
            duplicate_count = duplicates.sum()
            issues.append(
                DataQualityIssue(
                    issue_type=IssueType.DUPLICATE_DATA,
                    column="全列",
                    description=f"重複行が {duplicate_count} 件存在",
                    severity=DataQualityLevel.POOR,
                    count=int(duplicate_count),
                    examples=df[duplicates].head(3).to_dict("records"),
                    suggested_fix="drop_duplicates()で重複を削除",
                )
            )

        # 4. 価格データの妥当性チェック
        for col in self.financial_validation_rules["price_columns"]:
            if col in df.columns:
                # 負の価格
                negative_prices = df[col] < 0
                if negative_prices.any():
                    count = negative_prices.sum()
                    examples = df.loc[negative_prices, col].head(3).tolist()
                    issues.append(
                        DataQualityIssue(
                            issue_type=IssueType.INVALID_VALUE,
                            column=col,
                            description=f"負の価格が {count} 件存在",
                            severity=DataQualityLevel.CRITICAL,
                            count=int(count),
                            examples=examples,
                            suggested_fix="負の価格を NaN に変換または削除",
                        )
                    )

                # 極端な価格
                min_price = self.financial_validation_rules["min_price"]
                max_price = self.financial_validation_rules["max_price"]
                extreme_prices = (df[col] < min_price) | (df[col] > max_price)
                if extreme_prices.any():
                    count = extreme_prices.sum()
                    examples = df.loc[extreme_prices, col].head(3).tolist()
                    issues.append(
                        DataQualityIssue(
                            issue_type=IssueType.OUTLIER,
                            column=col,
                            description=f"極端な価格が {count} 件存在 (< {min_price} or > {max_price})",
                            severity=DataQualityLevel.ACCEPTABLE,
                            count=int(count),
                            examples=examples,
                            suggested_fix="価格の妥当性を確認し、必要に応じて修正",
                        )
                    )

        # 5. 出来高データの妥当性チェック
        for col in self.financial_validation_rules["volume_columns"]:
            if col in df.columns:
                # 負の出来高
                negative_volume = df[col] < 0
                if negative_volume.any():
                    count = negative_volume.sum()
                    examples = df.loc[negative_volume, col].head(3).tolist()
                    issues.append(
                        DataQualityIssue(
                            issue_type=IssueType.INVALID_VALUE,
                            column=col,
                            description=f"負の出来高が {count} 件存在",
                            severity=DataQualityLevel.CRITICAL,
                            count=int(count),
                            examples=examples,
                            suggested_fix="負の出来高を 0 または NaN に変換",
                        )
                    )

        # 6. OHLC整合性チェック
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            ohlc_issues = (
                (df["high"] < df["low"])
                | (df["high"] < df["open"])
                | (df["high"] < df["close"])
                | (df["low"] > df["open"])
                | (df["low"] > df["close"])
            )
            if ohlc_issues.any():
                count = ohlc_issues.sum()
                examples = (
                    df.loc[ohlc_issues, ["open", "high", "low", "close"]]
                    .head(3)
                    .to_dict("records")
                )
                issues.append(
                    DataQualityIssue(
                        issue_type=IssueType.INCONSISTENCY,
                        column="OHLC",
                        description=f"OHLC整合性エラーが {count} 件存在",
                        severity=DataQualityLevel.CRITICAL,
                        count=int(count),
                        examples=examples,
                        suggested_fix="High >= Low >= Open, Close の関係を修正",
                    )
                )

        # 7. 日付フォーマットチェック
        if "date" in df.columns:
            try:
                pd.to_datetime(df["date"])
            except Exception as e:
                issues.append(
                    DataQualityIssue(
                        issue_type=IssueType.FORMAT_ERROR,
                        column="date",
                        description=f"日付フォーマットエラー: {str(e)}",
                        severity=DataQualityLevel.CRITICAL,
                        count=1,
                        examples=[],
                        suggested_fix="日付を標準フォーマット（YYYY-MM-DD）に変換",
                    )
                )

        # 品質スコア計算
        quality_score = self._calculate_quality_score(issues, total_records)
        quality_level = self._determine_quality_level(quality_score)

        # サマリー作成
        summary = self._create_summary(df, issues, quality_score)

        # レポート作成
        report = DataQualityReport(
            dataset_name=dataset_name,
            timestamp=datetime.now(timezone.utc),
            total_records=total_records,
            quality_score=quality_score,
            quality_level=quality_level,
            issues=issues,
            summary=summary,
        )

        # レポートを保存
        self._save_report(report)

        logger.info(f"データ品質検証完了: {dataset_name}, スコア: {quality_score:.2f}")
        return report

    def _calculate_quality_score(
        self, issues: List[DataQualityIssue], total_records: int
    ) -> float:
        """品質スコア計算（0-100）"""
        if not issues:
            return 100.0

        penalty = 0.0
        severity_weights = {
            DataQualityLevel.CRITICAL: 20.0,
            DataQualityLevel.POOR: 10.0,
            DataQualityLevel.ACCEPTABLE: 3.0,
            DataQualityLevel.GOOD: 1.0,
            DataQualityLevel.EXCELLENT: 0.0,
        }

        for issue in issues:
            weight = severity_weights.get(issue.severity, 5.0)
            ratio = min(issue.count / total_records, 1.0) if total_records > 0 else 1.0
            penalty += weight * ratio

        score = max(0.0, 100.0 - penalty)
        return round(score, 2)

    def _determine_quality_level(self, score: float) -> DataQualityLevel:
        """スコアから品質レベル決定"""
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

    def _create_summary(
        self, df: pd.DataFrame, issues: List[DataQualityIssue], score: float
    ) -> Dict[str, Any]:
        """データサマリー作成"""
        return {
            "columns": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "issue_counts_by_type": {
                issue_type.value: len([i for i in issues if i.issue_type == issue_type])
                for issue_type in IssueType
            },
            "issue_counts_by_severity": {
                severity.value: len([i for i in issues if i.severity == severity])
                for severity in DataQualityLevel
            },
            "quality_score": score,
        }

    def _save_report(self, report: DataQualityReport) -> None:
        """レポート保存"""
        report_data = {
            "dataset_name": report.dataset_name,
            "timestamp": report.timestamp.isoformat(),
            "total_records": report.total_records,
            "quality_score": report.quality_score,
            "quality_level": report.quality_level.value,
            "issues": [
                {
                    "issue_type": issue.issue_type.value,
                    "column": issue.column,
                    "description": issue.description,
                    "severity": issue.severity.value,
                    "count": issue.count,
                    "examples": issue.examples,
                    "suggested_fix": issue.suggested_fix,
                }
                for issue in report.issues
            ],
            "summary": report.summary,
        }

        # SQLiteに保存
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO quality_reports
                (dataset_name, timestamp, quality_score, quality_level, total_records, issues_count, report_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    report.dataset_name,
                    report.timestamp.isoformat(),
                    report.quality_score,
                    report.quality_level.value,
                    report.total_records,
                    len(report.issues),
                    json.dumps(report_data, ensure_ascii=False),
                ),
            )
            conn.commit()

        # JSONファイルに保存
        report_file = (
            self.storage_path
            / f"report_{report.dataset_name}_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

    def clean_financial_data(
        self, df: pd.DataFrame, report: DataQualityReport = None
    ) -> pd.DataFrame:
        """金融データクリーニング"""
        logger.info("データクリーニング開始")

        cleaned_df = df.copy()

        if report is None:
            report = self.validate_financial_data(df, "temp_for_cleaning")

        # 1. 重複削除
        if any(issue.issue_type == IssueType.DUPLICATE_DATA for issue in report.issues):
            cleaned_df = cleaned_df.drop_duplicates()
            logger.info(f"重複削除: {len(df)} → {len(cleaned_df)} 行")

        # 2. 負の価格・出来高修正
        for col in self.financial_validation_rules["price_columns"]:
            if col in cleaned_df.columns:
                negative_mask = cleaned_df[col] < 0
                if negative_mask.any():
                    cleaned_df.loc[negative_mask, col] = np.nan
                    logger.info(f"負の価格修正: {col}列 {negative_mask.sum()}件")

        for col in self.financial_validation_rules["volume_columns"]:
            if col in cleaned_df.columns:
                negative_mask = cleaned_df[col] < 0
                if negative_mask.any():
                    cleaned_df.loc[negative_mask, col] = 0
                    logger.info(f"負の出来高修正: {col}列 {negative_mask.sum()}件")

        # 3. 前方補完で欠損値修正
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in cleaned_df.columns and cleaned_df[col].isnull().any():
                before_count = cleaned_df[col].isnull().sum()
                cleaned_df[col] = cleaned_df[col].fillna(method="ffill")
                after_count = cleaned_df[col].isnull().sum()
                if before_count > after_count:
                    logger.info(f"欠損値補完: {col}列 {before_count} → {after_count}件")

        # 4. 日付列の標準化
        if "date" in cleaned_df.columns:
            try:
                cleaned_df["date"] = pd.to_datetime(cleaned_df["date"])
                logger.info("日付列を標準化しました")
            except Exception as e:
                logger.warning(f"日付変換エラー: {e}")

        logger.info("データクリーニング完了")
        return cleaned_df

    def get_quality_history(
        self, dataset_name: str = None, days: int = 30
    ) -> List[Dict[str, Any]]:
        """品質履歴取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if dataset_name:
                cursor.execute(
                    """
                    SELECT * FROM quality_reports
                    WHERE dataset_name = ? AND timestamp > datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                """.format(days),
                    (dataset_name,),
                )
            else:
                cursor.execute(
                    f"""
                    SELECT * FROM quality_reports
                    WHERE timestamp > datetime('now', '-{days} days')
                    ORDER BY timestamp DESC
                """
                )

            return [
                dict(zip([col[0] for col in cursor.description], row))
                for row in cursor.fetchall()
            ]

    def generate_quality_summary(self, dataset_name: str = None) -> str:
        """品質サマリー生成"""
        history = self.get_quality_history(dataset_name, days=7)

        if not history:
            return "データ品質履歴が見つかりません。"

        latest = history[0]
        report_data = json.loads(latest["report_data"])

        summary = [
            "=" * 60,
            "データ品質サマリー",
            "=" * 60,
            f"データセット: {latest['dataset_name']}",
            f"最新評価: {latest['timestamp'][:19]}",
            f"品質スコア: {latest['quality_score']:.2f}/100",
            f"品質レベル: {latest['quality_level'].upper()}",
            f"総レコード数: {latest['total_records']:,}行",
            f"検出問題数: {latest['issues_count']}件",
            "",
        ]

        # 問題概要
        if report_data.get("issues"):
            summary.append("【主要な品質問題】")
            for issue in report_data["issues"][:5]:  # 上位5件
                summary.append(f"  - {issue['column']}: {issue['description']}")
                summary.append(f"    修正案: {issue['suggested_fix']}")
            summary.append("")

        # 推奨アクション
        score = latest["quality_score"]
        summary.append("【推奨アクション】")
        if score >= 95:
            summary.append("  - 品質は優秀です。現在の管理プロセスを継続してください。")
        elif score >= 85:
            summary.append("  - 良好な品質です。軽微な問題の修正を検討してください。")
        elif score >= 70:
            summary.append("  - 許容範囲の品質です。重要な問題の優先的修正が必要です。")
        elif score >= 50:
            summary.append(
                "  - 品質改善が必要です。データクリーニング処理の実行を推奨します。"
            )
        else:
            summary.append(
                "  - 緊急対応が必要です。即座にデータ品質の改善に取り組んでください。"
            )

        summary.append("=" * 60)

        return "\n".join(summary)


def demo_practical_data_quality():
    """実用的データ品質管理デモ"""
    print("=== 実用的データ品質管理システム デモ ===")

    # テストデータ作成
    np.random.seed(42)
    test_data = pd.DataFrame(
        {
            "symbol": ["AAPL", "GOOGL", "MSFT"] * 20,
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "open": np.random.uniform(100, 200, 60),
            "high": np.random.uniform(110, 220, 60),
            "low": np.random.uniform(90, 180, 60),
            "close": np.random.uniform(100, 200, 60),
            "volume": np.random.randint(1000000, 100000000, 60),
        }
    )

    # 意図的に問題を作成
    test_data.loc[5, "close"] = -10  # 負の価格
    test_data.loc[10:15, "volume"] = np.nan  # 欠損値
    test_data.loc[20, "low"] = test_data.loc[20, "high"] + 10  # OHLC整合性エラー

    # 品質管理システム初期化
    dq_manager = PracticalDataQualityManager()

    # 品質検証
    print("\n1. データ品質検証実行...")
    report = dq_manager.validate_financial_data(test_data, "demo_financial_data")

    print(f"   品質スコア: {report.quality_score}/100")
    print(f"   品質レベル: {report.quality_level.value}")
    print(f"   検出問題数: {len(report.issues)}件")

    # データクリーニング
    print("\n2. データクリーニング実行...")
    cleaned_data = dq_manager.clean_financial_data(test_data, report)

    print(f"   クリーニング前: {len(test_data)}行")
    print(f"   クリーニング後: {len(cleaned_data)}行")
    print(
        f"   欠損値: {test_data.isnull().sum().sum()} → {cleaned_data.isnull().sum().sum()}"
    )

    # 再検証
    print("\n3. クリーニング後検証...")
    final_report = dq_manager.validate_financial_data(cleaned_data, "demo_cleaned_data")

    print(f"   改善後スコア: {final_report.quality_score}/100")
    print(f"   改善後レベル: {final_report.quality_level.value}")

    # サマリー表示
    print("\n4. 品質サマリー:")
    summary = dq_manager.generate_quality_summary("demo_cleaned_data")
    print(summary)

    print("\n=== デモ完了 ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        demo_practical_data_quality()
