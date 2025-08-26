#!/usr/bin/env python3
"""
品質ゲートシステム - データベース管理

品質評価結果の永続化とクエリ機能を提供するモジュール。
SQLiteを使用して品質レポート、メトリクス詳細、
ファイル品質データの保存・取得を行う。
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from .types import FileQualityReport, QualityResult


class QualityDatabase:
    """品質データベース管理システム
    
    品質評価結果をSQLiteデータベースに保存し、
    履歴管理や分析のためのクエリ機能を提供する。
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

    def get_latest_report(self) -> Dict[str, Any]:
        """最新の品質レポートを取得
        
        Returns:
            最新の品質レポート、またはNone
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT details FROM quality_reports
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                )
                row = cursor.fetchone()
                
                if row:
                    return json.loads(row[0])
                return None

        except Exception:
            return None

    def get_quality_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """品質履歴を取得
        
        指定された日数分の品質レポート履歴を取得する。
        
        Args:
            days: 取得する日数
            
        Returns:
            品質レポートの履歴リスト
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, timestamp, overall_score, overall_level, 
                           gates_passed, gates_total
                    FROM quality_reports
                    WHERE timestamp >= datetime('now', '-' || ? || ' days')
                    ORDER BY timestamp DESC
                """,
                    (days,)
                )
                
                history = []
                for row in cursor.fetchall():
                    history.append({
                        "report_id": row[0],
                        "timestamp": row[1],
                        "overall_score": row[2],
                        "overall_level": row[3],
                        "gates_passed": row[4],
                        "gates_total": row[5],
                    })
                
                return history

        except Exception:
            return []

    def get_metric_trend(self, metric_type: str, days: int = 90) -> List[Dict[str, Any]]:
        """特定メトリクスの推移を取得
        
        指定されたメトリクスの時系列データを取得する。
        
        Args:
            metric_type: メトリクス種類
            days: 取得する日数
            
        Returns:
            メトリクス推移データ
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT timestamp, value, level, passed
                    FROM quality_metrics
                    WHERE metric_type = ? 
                    AND timestamp >= datetime('now', '-' || ? || ' days')
                    ORDER BY timestamp ASC
                """,
                    (metric_type, days)
                )
                
                trend = []
                for row in cursor.fetchall():
                    trend.append({
                        "timestamp": row[0],
                        "value": row[1],
                        "level": row[2],
                        "passed": bool(row[3]),
                    })
                
                return trend

        except Exception:
            return []

    def get_file_quality_trends(self, file_path: str, days: int = 90) -> List[Dict[str, Any]]:
        """ファイル品質の推移を取得
        
        指定されたファイルの品質指標の推移を取得する。
        
        Args:
            file_path: ファイルパス
            days: 取得する日数
            
        Returns:
            ファイル品質推移データ
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT timestamp, lines_of_code, complexity_score, 
                           maintainability_index, quality_level
                    FROM file_quality
                    WHERE file_path = ? 
                    AND timestamp >= datetime('now', '-' || ? || ' days')
                    ORDER BY timestamp ASC
                """,
                    (file_path, days)
                )
                
                trends = []
                for row in cursor.fetchall():
                    trends.append({
                        "timestamp": row[0],
                        "lines_of_code": row[1],
                        "complexity_score": row[2],
                        "maintainability_index": row[3],
                        "quality_level": row[4],
                    })
                
                return trends

        except Exception:
            return []

    def get_worst_files(self, limit: int = 10) -> List[Dict[str, Any]]:
        """品質の悪いファイルを取得
        
        最新レポートから品質の低いファイルを取得する。
        
        Args:
            limit: 取得する最大件数
            
        Returns:
            低品質ファイルのリスト
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 最新レポートIDを取得
                cursor = conn.execute(
                    "SELECT id FROM quality_reports ORDER BY timestamp DESC LIMIT 1"
                )
                latest_report_id = cursor.fetchone()
                
                if not latest_report_id:
                    return []
                
                # 低品質ファイルを取得
                cursor = conn.execute(
                    """
                    SELECT file_path, complexity_score, maintainability_index, quality_level
                    FROM file_quality
                    WHERE report_id = ?
                    ORDER BY maintainability_index ASC, complexity_score DESC
                    LIMIT ?
                """,
                    (latest_report_id[0], limit)
                )
                
                worst_files = []
                for row in cursor.fetchall():
                    worst_files.append({
                        "file_path": row[0],
                        "complexity_score": row[1],
                        "maintainability_index": row[2],
                        "quality_level": row[3],
                    })
                
                return worst_files

        except Exception:
            return []

    def cleanup_old_reports(self, keep_days: int = 90):
        """古いレポートをクリーンアップ
        
        指定された日数より古いレポートを削除する。
        
        Args:
            keep_days: 保持する日数
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 古いレポートのIDを取得
                cursor = conn.execute(
                    """
                    SELECT id FROM quality_reports
                    WHERE timestamp < datetime('now', '-' || ? || ' days')
                """,
                    (keep_days,)
                )
                old_report_ids = [row[0] for row in cursor.fetchall()]
                
                if not old_report_ids:
                    return
                
                # 関連データを削除
                placeholders = ','.join(['?' for _ in old_report_ids])
                
                conn.execute(
                    f"DELETE FROM file_quality WHERE report_id IN ({placeholders})",
                    old_report_ids
                )
                
                conn.execute(
                    f"DELETE FROM quality_metrics WHERE report_id IN ({placeholders})",
                    old_report_ids
                )
                
                conn.execute(
                    f"DELETE FROM quality_reports WHERE id IN ({placeholders})",
                    old_report_ids
                )
                
                conn.commit()
                print(f"古いレポート {len(old_report_ids)} 件を削除しました")

        except Exception as e:
            print(f"クリーンアップエラー: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """データベース統計情報を取得
        
        Returns:
            データベースの統計情報
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # レポート数
                cursor = conn.execute("SELECT COUNT(*) FROM quality_reports")
                stats["total_reports"] = cursor.fetchone()[0]
                
                # 最新レポート日時
                cursor = conn.execute("SELECT MAX(timestamp) FROM quality_reports")
                stats["latest_report"] = cursor.fetchone()[0]
                
                # 最古レポート日時
                cursor = conn.execute("SELECT MIN(timestamp) FROM quality_reports")
                stats["oldest_report"] = cursor.fetchone()[0]
                
                # 平均スコア
                cursor = conn.execute("SELECT AVG(overall_score) FROM quality_reports")
                stats["average_score"] = cursor.fetchone()[0] or 0
                
                return stats

        except Exception:
            return {}