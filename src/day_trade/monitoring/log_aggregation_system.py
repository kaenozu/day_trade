#!/usr/bin/env python3
"""
統合ログ集約・分析システム（後方互換性モジュール）

このファイルは後方互換性のために残されています。
実際の実装は log_aggregation パッケージに移動されました。

元のインポートが引き続き機能するようにするためのエイリアスを提供します。
"""

# 元のモジュールからの全てのインポートを再エクスポート
from .log_aggregation import *

# 明示的に重要なクラスと関数をエクスポート
from .log_aggregation import (
    AlertSeverity,
    LogAggregationSystem,
    LogAlert,
    LogEntry,
    LogLevel,
    LogParser,
    LogPattern,
    LogSearchQuery,
    LogSource,
    StandardLogParser,
    StructuredLogParser,
    create_log_aggregation_system,
    get_log_aggregation_system,
)

# 後方互換性のための __all__ 定義
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
]

# メイン実行部分（テスト）
if __name__ == "__main__":
    from .log_aggregation.factory import test_log_aggregation_system
    import asyncio
    asyncio.run(test_log_aggregation_system())