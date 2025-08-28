#!/usr/bin/env python3
"""
品質ゲートシステム - データベースメンテナンス

データベースのクリーンアップ、統計情報生成、
最適化機能を提供するモジュール。
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict

from .database_core import QualityDatabaseCore


class QualityDatabaseMaintenance:
    """品質データベースメンテナンス管理
    
    データベースの保守・最適化機能を提供する。
    古いデータの削除、統計情報の生成、パフォーマンス向上を行う。
    """

    def __init__(self, db_core: QualityDatabaseCore):
        """初期化
        
        Args:
            db_core: データベースコア
        """
        self.db_core = db_core

    def cleanup_old_reports(self, keep_days: int = 90):
        """古いレポートをクリーンアップ
        
        指定された保持期間を超える古い品質レポートと
        関連データを削除する。
        
        Args:
            keep_days: 保持する日数
        """
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        try:
            with self.db_core.get_connection() as conn:
                # 古いレポートIDを取得
                cursor = conn.execute(
                    """
                    SELECT id FROM quality_reports
                    WHERE datetime(timestamp) < datetime(?)
                    """,
                    (cutoff_date.isoformat(),)
                )
                
                old_report_ids = [row[0] for row in cursor.fetchall()]
                
                if old_report_ids:
                    # プレースホルダーを作成
                    placeholders = ','.join('?' * len(old_report_ids))
                    
                    # 関連データを削除
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
                    
                    print(f"クリーンアップ完了: {len(old_report_ids)}件のレポートを削除しました")
                else:
                    print("削除対象のレポートはありません")
                    
        except Exception as e:
            print(f"クリーンアップエラー: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """データベース統計情報を取得
        
        テーブルサイズ、レコード数、期間別統計などの
        データベース統計情報を生成する。
        
        Returns:
            統計情報の辞書
        """
        try:
            with self.db_core.get_connection() as conn:
                stats = {}
                
                # テーブル別レコード数
                tables = ['quality_reports', 'quality_metrics', 'file_quality']
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f'{table}_count'] = cursor.fetchone()[0]
                
                # 最新・最古のレポート日時
                cursor = conn.execute(
                    "SELECT MIN(timestamp), MAX(timestamp) FROM quality_reports"
                )
                min_date, max_date = cursor.fetchone()
                
                stats['oldest_report'] = min_date
                stats['newest_report'] = max_date
                
                # 期間別レポート数
                periods = [
                    ('last_7_days', 7),
                    ('last_30_days', 30),
                    ('last_90_days', 90)
                ]
                
                for period_name, days in periods:
                    cutoff = datetime.now() - timedelta(days=days)
                    cursor = conn.execute(
                        """
                        SELECT COUNT(*) FROM quality_reports
                        WHERE datetime(timestamp) >= datetime(?)
                        """,
                        (cutoff.isoformat(),)
                    )
                    stats[f'reports_{period_name}'] = cursor.fetchone()[0]
                
                # 品質レベル別カウント
                cursor = conn.execute(
                    """
                    SELECT overall_level, COUNT(*) 
                    FROM quality_reports 
                    GROUP BY overall_level
                    """
                )
                
                quality_levels = {}
                for level, count in cursor.fetchall():
                    quality_levels[level] = count
                    
                stats['quality_level_distribution'] = quality_levels
                
                # メトリクス種類別統計
                cursor = conn.execute(
                    """
                    SELECT 
                        metric_type,
                        COUNT(*) as count,
                        AVG(value) as avg_value,
                        MIN(value) as min_value,
                        MAX(value) as max_value
                    FROM quality_metrics
                    GROUP BY metric_type
                    """
                )
                
                metric_stats = {}
                for row in cursor.fetchall():
                    metric_stats[row[0]] = {
                        'count': row[1],
                        'average': round(row[2], 2) if row[2] else 0,
                        'minimum': row[3],
                        'maximum': row[4]
                    }
                    
                stats['metric_statistics'] = metric_stats
                
                # データベースサイズ
                cursor = conn.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                
                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                
                stats['database_size_bytes'] = page_size * page_count
                stats['database_size_mb'] = round(stats['database_size_bytes'] / 1024 / 1024, 2)
                
                return stats
                
        except Exception as e:
            print(f"統計取得エラー: {e}")
            return {}

    def optimize_database(self):
        """データベースを最適化
        
        インデックスの再構築、VACUUM実行、
        統計情報の更新などを行い、
        データベースのパフォーマンスを向上させる。
        """
        try:
            with self.db_core.get_connection() as conn:
                print("データベース最適化を開始...")
                
                # 統計情報を更新
                conn.execute("ANALYZE")
                print("  統計情報を更新しました")
                
                # インデックス再構築
                conn.execute("REINDEX")
                print("  インデックスを再構築しました")
                
                # VACUUM実行（データベースを再編成）
                conn.execute("VACUUM")
                print("  データベースファイルを最適化しました")
                
                print("データベース最適化が完了しました")
                
        except Exception as e:
            print(f"最適化エラー: {e}")

    def backup_database(self, backup_path: str):
        """データベースをバックアップ
        
        現在のデータベースを指定されたパスにバックアップする。
        
        Args:
            backup_path: バックアップファイルのパス
        """
        try:
            with self.db_core.get_connection() as source:
                with sqlite3.connect(backup_path) as backup:
                    source.backup(backup)
                    
            print(f"データベースバックアップ完了: {backup_path}")
            
        except Exception as e:
            print(f"バックアップエラー: {e}")

    def validate_database_integrity(self) -> bool:
        """データベースの整合性をチェック
        
        データベースファイルの破損チェックと
        外部キー制約の検証を行う。
        
        Returns:
            整合性チェックの結果（True=正常）
        """
        try:
            with self.db_core.get_connection() as conn:
                # データベース整合性チェック
                cursor = conn.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                
                if integrity_result != "ok":
                    print(f"データベース整合性エラー: {integrity_result}")
                    return False
                
                # 外部キー制約チェック
                conn.execute("PRAGMA foreign_keys = ON")
                cursor = conn.execute("PRAGMA foreign_key_check")
                
                fk_violations = cursor.fetchall()
                if fk_violations:
                    print(f"外部キー制約違反: {len(fk_violations)}件")
                    for violation in fk_violations:
                        print(f"  {violation}")
                    return False
                
                print("データベース整合性チェック: 正常")
                return True
                
        except Exception as e:
            print(f"整合性チェックエラー: {e}")
            return False

    def get_disk_usage_report(self) -> Dict[str, Any]:
        """ディスク使用量レポートを生成
        
        テーブル別のディスク使用量と
        データ効率性の分析結果を返す。
        
        Returns:
            ディスク使用量レポート
        """
        try:
            with self.db_core.get_connection() as conn:
                report = {}
                
                # テーブル別ページ数
                tables = ['quality_reports', 'quality_metrics', 'file_quality']
                
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    record_count = cursor.fetchone()[0]
                    
                    # 概算ページ数（正確ではないが目安として）
                    cursor = conn.execute("PRAGMA page_size")
                    page_size = cursor.fetchone()[0]
                    
                    estimated_pages = max(1, record_count // 10)  # 簡易計算
                    estimated_size = estimated_pages * page_size
                    
                    report[table] = {
                        'record_count': record_count,
                        'estimated_size_bytes': estimated_size,
                        'estimated_size_kb': round(estimated_size / 1024, 2)
                    }
                
                return report
                
        except Exception as e:
            print(f"ディスク使用量レポートエラー: {e}")
            return {}