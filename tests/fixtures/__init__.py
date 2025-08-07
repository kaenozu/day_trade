"""
テスト用共通フィクスチャとファクトリ関数

Phase 5: テストコード共通化の一環として、
重複するテストコードを統合し、保守性を向上させます。

使用可能なフィクスチャ:
- mock_factories: Mock作成ファクトリ
- sample_data: サンプルデータ生成
- test_fixtures: 共通フィクスチャ
- assertions: カスタムアサート関数
- database_fixtures: データベース関連フィクスチャ
- performance_fixtures: パフォーマンステスト用フィクスチャ
"""

# 主要なファクトリ関数をエクスポート
from .assertions import (
    assert_dataframe_equal,
    assert_error_handling,
    assert_portfolio_state,
    assert_trade_equal,
)
from .mock_factories import (
    create_mock_alert_manager,
    create_mock_database_manager,
    create_mock_stock_fetcher,
    create_mock_trade_manager,
)
from .sample_data import (
    create_sample_market_data,
    create_sample_portfolio_data,
    create_sample_stock_data,
    create_sample_trade_data,
)

__all__ = [
    # Mock ファクトリ
    "create_mock_stock_fetcher",
    "create_mock_database_manager",
    "create_mock_trade_manager",
    "create_mock_alert_manager",
    # サンプルデータ
    "create_sample_stock_data",
    "create_sample_trade_data",
    "create_sample_market_data",
    "create_sample_portfolio_data",
    # アサート関数
    "assert_dataframe_equal",
    "assert_trade_equal",
    "assert_portfolio_state",
    "assert_error_handling",
]
