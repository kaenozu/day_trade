#!/usr/bin/env python3
"""
品質ゲートシステム - データベースコア

データベースの基本操作（初期化、保存）を担当するモジュール。
SQLiteデータベースの作成とデータ保存機能を提供する。
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from .types import FileQualityReport, QualityResult


class QualityDatabaseCore:
    """品質データベースコア管理
    
    データベースの初期化と基本的な保存操作を担当する。
    テーブル作成とデータ永続化機能を提供する。
    """

    def __init__(self, project_root: str = "."):
        """初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        self.project_root = Path(project_root)
        self.db_path = self.project_root / "quality_metrics.db"
        self._initialize_database()

    def _initialize_database(self):
        """データベースを初期化
        
        必要なテーブルを作成し、データベースの基本構造を設定する。
        品質レポート、メトリクス、ファイル品質のテーブルを作成。
        """
        with sqlite3.connect(self.db_path) as conn:
            # 品質レポートメインテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_reports (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    overall_score REAL NOT NULL,
                    overall_level TEXT NOT NULL,
                    gates_passed INTEGER NOT NULL,
                    gates_total INTEGER NOT NULL,
                    details TEXT NOT NULL,
                    recommendations TEXT NOT NULL
                )
            """
            )

            # 品質メトリクス詳細テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT NOT NULL,
                    gate_id TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    level TEXT NOT NULL,
                    passed BOOLEAN NOT NULL,
                    timestamp DATETIME NOT NULL,
                    FOREIGN KEY (report_id) REFERENCES quality_reports (id)
                )
            """
            )

            # ファイル品質テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    lines_of_code INTEGER NOT NULL,
                    complexity_score REAL NOT NULL,
                    maintainability_index REAL NOT NULL,
                    quality_level TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    FOREIGN KEY (report_id) REFERENCES quality_reports (id)
                )
            """
            )

            # インデックス作成（パフォーマンス向上）
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_quality_reports_timestamp "
                "ON quality_reports(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_quality_metrics_report_id "
                "ON quality_metrics(report_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_file_quality_report_id "
                "ON file_quality(report_id)"
            )

            conn.commit()

    async def save_quality_report(self, report: Dict[str, Any]):
        """品質レポートをデータベースに保存
        
        品質評価の全結果をデータベースに永続化する。
        メインレポート、メトリクス詳細、ファイル品質データを
        関連テーブルに保存する。
        
        Args:
            report: 保存する品質レポート
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # メインレポート保存
                conn.execute(
                    """
                    INSERT OR REPLACE INTO quality_reports
                    (id, timestamp, overall_score, overall_level, gates_passed, gates_total, details, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        report["report_id"],
                        report["timestamp"],
                        report["overall_score"],
                        report["overall_level"],
                        report["gates_passed"],
                        report["gates_total"],
                        json.dumps(report, default=str, ensure_ascii=False),
                        json.dumps(report["recommendations"], ensure_ascii=False),
                    ),
                )

                # メトリクス詳細保存
                for result in report["gate_results"]:
                    conn.execute(
                        """
                        INSERT INTO quality_metrics
                        (report_id, gate_id, metric_type, value, level, passed, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            report["report_id"],
                            result.gate_id,
                            result.metric_type.value,
                            result.value,
                            result.level.value,
                            result.passed,
                            report["timestamp"],
                        ),
                    )

                # ファイル品質データ保存
                for file_report in report.get("file_reports", []):
                    conn.execute(
                        """
                        INSERT INTO file_quality
                        (report_id, file_path, lines_of_code, complexity_score,
                         maintainability_index, quality_level, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            report["report_id"],
                            file_report.file_path,
                            file_report.lines_of_code,
                            file_report.complexity_score,
                            file_report.maintainability_index,
                            file_report.quality_level.value,
                            report["timestamp"],
                        ),
                    )

                conn.commit()
                
        except Exception as e:
            print(f"レポート保存エラー: {e}")
            raise

    def get_connection(self) -> sqlite3.Connection:
        """データベース接続を取得
        
        Returns:
            SQLite接続オブジェクト
        """
        return sqlite3.connect(self.db_path)

    def execute_query(self, query: str, params=None) -> List[Dict[str, Any]]:
        """クエリを実行して結果を取得
        
        Args:
            query: 実行するSQL
            params: クエリパラメータ
            
        Returns:
            クエリ結果のリスト
        """
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params or ())
            return [dict(row) for row in cursor.fetchall()]

    def execute_update(self, query: str, params=None) -> int:
        """更新系クエリを実行
        
        Args:
            query: 実行するSQL
            params: クエリパラメータ
            
        Returns:
            影響を受けた行数
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, params or ())
            conn.commit()
            return cursor.rowcount