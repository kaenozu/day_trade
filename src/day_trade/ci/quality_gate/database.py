#!/usr/bin/env python3
"""
品質ゲートシステム - データベース管理（統合版）

分割されたデータベース機能を統合し、
既存のAPIとの互換性を維持するファサードクラス。
"""

from typing import Any, Dict, List

from .database_core import QualityDatabaseCore
from .database_maintenance import QualityDatabaseMaintenance
from .database_queries import QualityDatabaseQueries


class QualityDatabase:
    """品質データベース管理システム（統合版）
    
    分割されたデータベース機能を統合し、
    単一のインターフェースとして提供する。
    既存のAPIとの完全互換を保つ。
    """

    def __init__(self, project_root: str = "."):
        """初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        self.core = QualityDatabaseCore(project_root)
        self.queries = QualityDatabaseQueries(self.core)
        self.maintenance = QualityDatabaseMaintenance(self.core)

    async def save_quality_report(self, report: Dict[str, Any]):
        """品質レポートをデータベースに保存
        
        Args:
            report: 保存する品質レポート
        """
        return await self.core.save_quality_report(report)

    def get_latest_report(self) -> Dict[str, Any]:
        """最新の品質レポートを取得
        
        Returns:
            最新の品質レポート、存在しない場合は空辞書
        """
        return self.queries.get_latest_report()

    def get_quality_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """品質履歴を取得
        
        Args:
            days: 取得する日数
            
        Returns:
            品質履歴のリスト
        """
        return self.queries.get_quality_history(days)

    def get_metric_trend(self, metric_type: str, days: int = 90) -> List[Dict[str, Any]]:
        """メトリクス推移を取得
        
        Args:
            metric_type: メトリクス種類
            days: 取得する日数
            
        Returns:
            メトリクス推移データ
        """
        return self.queries.get_metric_trend(metric_type, days)

    def get_file_quality_trends(self, file_path: str, days: int = 90) -> List[Dict[str, Any]]:
        """ファイル品質推移を取得
        
        Args:
            file_path: 対象ファイルパス
            days: 取得する日数
            
        Returns:
            ファイル品質推移データ
        """
        return self.queries.get_file_quality_trends(file_path, days)

    def get_worst_files(self, limit: int = 10) -> List[Dict[str, Any]]:
        """最も品質の低いファイルを取得
        
        Args:
            limit: 取得する件数
            
        Returns:
            品質の低いファイルのリスト
        """
        return self.queries.get_worst_files(limit)

    def cleanup_old_reports(self, keep_days: int = 90):
        """古いレポートをクリーンアップ
        
        Args:
            keep_days: 保持する日数
        """
        return self.maintenance.cleanup_old_reports(keep_days)

    def get_statistics(self) -> Dict[str, Any]:
        """データベース統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        return self.maintenance.get_statistics()

    # 新機能（既存にない機能）
    def get_metric_summary(self, days: int = 30) -> Dict[str, Any]:
        """メトリクス概要を取得
        
        Args:
            days: 集計対象日数
            
        Returns:
            メトリクス概要データ
        """
        return self.queries.get_metric_summary(days)

    def get_quality_distribution(self, days: int = 30) -> Dict[str, int]:
        """品質レベル分布を取得
        
        Args:
            days: 集計対象日数
            
        Returns:
            品質レベル別カウント
        """
        return self.queries.get_quality_distribution(days)

    def optimize_database(self):
        """データベースを最適化"""
        return self.maintenance.optimize_database()

    def backup_database(self, backup_path: str):
        """データベースをバックアップ
        
        Args:
            backup_path: バックアップファイルのパス
        """
        return self.maintenance.backup_database(backup_path)

    def validate_integrity(self) -> bool:
        """データベースの整合性をチェック
        
        Returns:
            整合性チェックの結果（True=正常）
        """
        return self.maintenance.validate_database_integrity()