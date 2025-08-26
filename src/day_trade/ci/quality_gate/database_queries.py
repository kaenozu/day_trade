#!/usr/bin/env python3
"""
品質ゲートシステム - データベースクエリ

品質データの検索・集計機能を提供するモジュール。
履歴取得、トレンド分析、統計情報の生成を行う。
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List

from .database_core import QualityDatabaseCore


class QualityDatabaseQueries:
    """品質データベースクエリ管理
    
    品質データの検索・分析機能を提供する。
    履歴データ、トレンド情報、統計データを生成する。
    """

    def __init__(self, db_core: QualityDatabaseCore):
        """初期化
        
        Args:
            db_core: データベースコア
        """
        self.db_core = db_core

    def get_latest_report(self) -> Dict[str, Any]:
        """最新の品質レポートを取得
        
        最も新しい品質評価レポートを返す。
        
        Returns:
            最新の品質レポート、存在しない場合は空辞書
        """
        query = """
            SELECT * FROM quality_reports
            ORDER BY timestamp DESC
            LIMIT 1
        """
        
        results = self.db_core.execute_query(query)
        
        if results:
            report = results[0]
            # JSON文字列をパース
            if report.get("details"):
                try:
                    report["details"] = json.loads(report["details"])
                except json.JSONDecodeError:
                    report["details"] = {}
            
            if report.get("recommendations"):
                try:
                    report["recommendations"] = json.loads(report["recommendations"])
                except json.JSONDecodeError:
                    report["recommendations"] = []
                    
            return report
            
        return {}

    def get_quality_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """品質履歴を取得
        
        指定された期間の品質スコア履歴を取得する。
        
        Args:
            days: 取得する日数
            
        Returns:
            品質履歴のリスト
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
            SELECT 
                timestamp,
                overall_score,
                overall_level,
                gates_passed,
                gates_total
            FROM quality_reports
            WHERE datetime(timestamp) >= datetime(?)
            ORDER BY timestamp ASC
        """
        
        return self.db_core.execute_query(query, (cutoff_date.isoformat(),))

    def get_metric_trend(self, metric_type: str, days: int = 90) -> List[Dict[str, Any]]:
        """メトリクス推移を取得
        
        特定のメトリクス種類の時系列データを取得する。
        
        Args:
            metric_type: メトリクス種類
            days: 取得する日数
            
        Returns:
            メトリクス推移データ
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
            SELECT 
                timestamp,
                value,
                level,
                passed
            FROM quality_metrics
            WHERE metric_type = ?
              AND datetime(timestamp) >= datetime(?)
            ORDER BY timestamp ASC
        """
        
        return self.db_core.execute_query(query, (metric_type, cutoff_date.isoformat()))

    def get_file_quality_trends(self, file_path: str, days: int = 90) -> List[Dict[str, Any]]:
        """ファイル品質推移を取得
        
        特定ファイルの品質指標の推移を取得する。
        
        Args:
            file_path: 対象ファイルパス
            days: 取得する日数
            
        Returns:
            ファイル品質推移データ
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
            SELECT 
                timestamp,
                lines_of_code,
                complexity_score,
                maintainability_index,
                quality_level
            FROM file_quality
            WHERE file_path = ?
              AND datetime(timestamp) >= datetime(?)
            ORDER BY timestamp ASC
        """
        
        return self.db_core.execute_query(query, (file_path, cutoff_date.isoformat()))

    def get_worst_files(self, limit: int = 10) -> List[Dict[str, Any]]:
        """最も品質の低いファイルを取得
        
        複雑度と保守性指標に基づいて、最も問題のあるファイルを特定する。
        
        Args:
            limit: 取得する件数
            
        Returns:
            品質の低いファイルのリスト
        """
        query = """
            SELECT 
                file_path,
                AVG(complexity_score) as avg_complexity,
                AVG(maintainability_index) as avg_maintainability,
                COUNT(*) as report_count,
                MAX(timestamp) as latest_timestamp
            FROM file_quality
            WHERE datetime(timestamp) >= datetime('now', '-30 days')
            GROUP BY file_path
            HAVING COUNT(*) >= 3  -- 最低3回のレポートがあるファイルのみ
            ORDER BY 
                avg_complexity DESC,
                avg_maintainability ASC
            LIMIT ?
        """
        
        return self.db_core.execute_query(query, (limit,))

    def get_metric_summary(self, days: int = 30) -> Dict[str, Any]:
        """メトリクス概要を取得
        
        指定期間の各メトリクスの統計情報を取得する。
        
        Args:
            days: 集計対象日数
            
        Returns:
            メトリクス概要データ
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
            SELECT 
                metric_type,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                COUNT(*) as measurement_count,
                SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as pass_count
            FROM quality_metrics
            WHERE datetime(timestamp) >= datetime(?)
            GROUP BY metric_type
        """
        
        results = self.db_core.execute_query(query, (cutoff_date.isoformat(),))
        
        summary = {}
        for row in results:
            metric_type = row['metric_type']
            summary[metric_type] = {
                'average': row['avg_value'],
                'minimum': row['min_value'],
                'maximum': row['max_value'],
                'measurement_count': row['measurement_count'],
                'pass_count': row['pass_count'],
                'pass_rate': (row['pass_count'] / row['measurement_count'] * 100) 
                           if row['measurement_count'] > 0 else 0
            }
            
        return summary

    def get_quality_distribution(self, days: int = 30) -> Dict[str, int]:
        """品質レベル分布を取得
        
        指定期間の品質レベルの分布を集計する。
        
        Args:
            days: 集計対象日数
            
        Returns:
            品質レベル別カウント
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
            SELECT 
                overall_level,
                COUNT(*) as count
            FROM quality_reports
            WHERE datetime(timestamp) >= datetime(?)
            GROUP BY overall_level
        """
        
        results = self.db_core.execute_query(query, (cutoff_date.isoformat(),))
        
        distribution = {
            'excellent': 0,
            'good': 0, 
            'acceptable': 0,
            'needs_improvement': 0,
            'critical': 0
        }
        
        for row in results:
            level = row['overall_level']
            count = row['count']
            if level in distribution:
                distribution[level] = count
                
        return distribution

    def search_reports_by_score(self, min_score: float, max_score: float = 100.0) -> List[Dict[str, Any]]:
        """スコア範囲でレポートを検索
        
        指定されたスコア範囲の品質レポートを検索する。
        
        Args:
            min_score: 最小スコア
            max_score: 最大スコア
            
        Returns:
            該当する品質レポートのリスト
        """
        query = """
            SELECT 
                id,
                timestamp,
                overall_score,
                overall_level,
                gates_passed,
                gates_total
            FROM quality_reports
            WHERE overall_score >= ? AND overall_score <= ?
            ORDER BY timestamp DESC
        """
        
        return self.db_core.execute_query(query, (min_score, max_score))