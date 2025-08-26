#!/usr/bin/env python3
"""
統合ログ集約・分析システム

ELK Stack風のログ集約・検索・分析機能を提供:
- ログ収集・パース・インデックス化
- リアルタイムログストリーム処理
- 高速ログ検索・フィルタリング
- ログ分析・パターン検出
- アラート生成・通知
- ダッシュボード統合
- ログローテーション・アーカイブ

Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード
"""

# 列挙型とデータクラス
from .enums import AlertSeverity, LogLevel, LogSource
from .models import LogAlert, LogEntry, LogPattern, LogSearchQuery

# パーサークラス
from .parsers import LogParser, StandardLogParser, StructuredLogParser

# コアシステム
from .core_system import LogAggregationSystem

# ファクトリ関数
from .factory import create_log_aggregation_system, get_log_aggregation_system

# その他のコンポーネント（必要に応じて）
from .analytics import LogAnalytics
from .database import LogDatabase
from .export_utils import LogExporter
from .pattern_detection import PatternDetectionEngine

# 後方互換性のためのエクスポート
__all__ = [
    # 列挙型
    "LogLevel",
    "LogSource", 
    "AlertSeverity",
    
    # データモデル
    "LogEntry",
    "LogPattern",
    "LogAlert",
    "LogSearchQuery",
    
    # パーサー
    "LogParser",
    "StructuredLogParser",
    "StandardLogParser",
    
    # メインシステム
    "LogAggregationSystem",
    
    # ファクトリ関数
    "create_log_aggregation_system",
    "get_log_aggregation_system",
    
    # コンポーネント
    "LogAnalytics",
    "LogDatabase", 
    "LogExporter",
    "PatternDetectionEngine",
]

# バージョン情報
__version__ = "1.0.0"
__author__ = "Day Trade System"
__description__ = "統合ログ集約・分析システム"