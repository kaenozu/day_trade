"""
フォーマッターモジュール
後方互換性のために、すべてのフォーマット機能を単一の場所からインポート可能にします
"""

# 基本フォーマット機能
from .basic_formatters import (
    format_currency,
    format_large_number,
    format_percentage,
    format_volume,
    get_change_color,
)

# テーブル作成機能
from .table_formatters import (
    create_comparison_table,
    create_company_info_table,
    create_historical_data_table,
    create_stock_info_table,
    create_watchlist_table,
    format_cli_table_data,
)

# パネル作成機能
from .panel_formatters import (
    create_cli_confirmation_panel,
    create_cli_header,
    create_cli_section,
    create_error_panel,
    create_info_panel,
    create_progress_bar_panel,
    create_success_panel,
    create_warning_panel,
)

# チャート・グラフィック機能
from .chart_formatters import (
    create_ascii_chart,
    create_distribution_chart,
    create_heatmap,
    create_sparkline,
)

# 高度な表示機能
from .advanced_formatters import (
    create_metric_cards,
    create_status_indicator,
    create_trend_indicator,
)

# CLI統一フォーマット機能
from .cli_formatters import (
    create_cli_command_help,
    create_cli_list_item,
    create_cli_loading_indicator,
    create_cli_status_bar,
)

# 後方互換性のためのすべての公開関数のリスト
__all__ = [
    # 基本フォーマット機能
    "format_currency",
    "format_percentage",
    "format_volume",
    "get_change_color",
    "format_large_number",
    # テーブル作成機能
    "create_stock_info_table",
    "create_company_info_table",
    "create_historical_data_table",
    "create_watchlist_table",
    "create_comparison_table",
    "format_cli_table_data",
    # パネル作成機能
    "create_error_panel",
    "create_success_panel",
    "create_warning_panel",
    "create_info_panel",
    "create_progress_bar_panel",
    "create_cli_header",
    "create_cli_section",
    "create_cli_confirmation_panel",
    # チャート・グラフィック機能
    "create_ascii_chart",
    "create_sparkline",
    "create_heatmap",
    "create_distribution_chart",
    # 高度な表示機能
    "create_metric_cards",
    "create_trend_indicator",
    "create_status_indicator",
    # CLI統一フォーマット機能
    "create_cli_command_help",
    "create_cli_status_bar",
    "create_cli_list_item",
    "create_cli_loading_indicator",
]