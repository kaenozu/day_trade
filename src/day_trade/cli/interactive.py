"""
対話型CLIのメインスクリプト（互換性レイヤー）
大規模ファイルを複数のモジュールに分割し、ここで統合
"""

# 後方互換性のため、すべての主要機能をここで再エクスポート
from .interactive.cli import cli
from .interactive.interactive_mode import InteractiveMode
from .interactive.screening import run_interactive_backtest
from .interactive.helpers import (
    get_watchlist_manager as _get_watchlist_manager,
    display_stock_details as _display_stock_details,
    display_historical_data as _display_historical_data,
)

# メイン実行
if __name__ == "__main__":
    cli()