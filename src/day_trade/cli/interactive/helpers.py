"""
対話型CLIのヘルパー関数
共通的に使用される関数群を提供
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from rich.console import Console

from ...core.config import config_manager
from ...core.watchlist import WatchlistManager
from ...data.stock_fetcher import StockFetcher
from ...utils.formatters import (
    create_company_info_table,
    create_error_panel,
    create_historical_data_table,
    create_stock_info_table,
)

logger = logging.getLogger(__name__)
console = Console()


def get_watchlist_manager(config_path: Optional[Path] = None) -> WatchlistManager:
    """
    ウォッチリストマネージャーのインスタンスを取得
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        WatchlistManager: ウォッチリストマネージャーインスタンス
    """
    _config_manager = config_manager.__class__(config_path)
    return WatchlistManager()


def display_stock_details(code: str, stock_data: Dict[str, Any], show_details: bool):
    """
    銘柄詳細を表示
    
    Args:
        code: 銘柄コード
        stock_data: 株価データ
        show_details: 詳細情報を表示するかどうか
    """
    if not stock_data:
        console.print(
            create_error_panel(
                f"銘柄コード '{code}' の現在価格または詳細情報を取得できませんでした。"
                "コードが正しいか、または市場が開いているかご確認ください。",
                title="情報取得エラー",
            )
        )
        return

    table = create_stock_info_table(stock_data)
    console.print(table)

    if show_details:
        fetcher = StockFetcher()
        with console.status("企業情報を取得中..."):
            info = fetcher.get_company_info(code)
        if info:
            detail_table = create_company_info_table(info)
            console.print("\n")
            console.print(detail_table)
        else:
            console.print("\n")
            console.print(
                create_error_panel(
                    f"銘柄コード '{code}' の企業詳細情報を取得できませんでした。"
                    "データプロバイダーの問題か、情報が利用できない可能性があります。",
                    title="企業情報エラー",
                )
            )


def display_historical_data(
    code: str, df: pd.DataFrame, period: str, interval: str, rows: int
):
    """
    ヒストリカルデータを表示
    
    Args:
        code: 銘柄コード
        df: ヒストリカルデータ
        period: 期間
        interval: 間隔
        rows: 表示行数
    """
    if df is None or df.empty:
        console.print(
            create_error_panel(
                "ヒストリカルデータを取得できませんでした。"
                "指定された銘柄コード、期間、または間隔が正しいかご確認ください。"
            )
        )
        return

    table = create_historical_data_table(df, code, period, interval, max_rows=rows)
    console.print(table)
    console.print("\n[bold]サマリー:[/bold]")
    console.print(f"期間高値: ¥{df['High'].max():,.0f}")
    console.print(f"期間安値: ¥{df['Low'].min():,.0f}")
    console.print(f"平均出来高: {int(df['Volume'].mean()):,}")