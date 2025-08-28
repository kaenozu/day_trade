"""
テーブル作成機能
株価情報、企業情報、履歴データなどの様々なテーブル作成処理
"""

from typing import Any, Dict, List, Set

import pandas as pd
from rich import box
from rich.table import Table

from .basic_formatters import (
    format_currency,
    format_large_number,
    format_percentage,
    format_volume,
    get_change_color,
)


def create_stock_info_table(stock_data: dict) -> Table:
    """
    株価情報テーブルを作成

    Args:
        stock_data: 株価データ

    Returns:
        Richテーブル
    """
    table = Table(title=f"銘柄情報: {stock_data.get('symbol', 'N/A')}")
    table.add_column("項目", style="cyan", no_wrap=True)
    table.add_column("値", style="white")

    # 現在価格
    current_price = stock_data.get("current_price")
    if current_price:
        table.add_row("現在値", format_currency(current_price))

    # 前日終値
    previous_close = stock_data.get("previous_close")
    if previous_close:
        table.add_row("前日終値", format_currency(previous_close))

    # 前日比
    change = stock_data.get("change")
    change_percent = stock_data.get("change_percent")
    if change is not None and change_percent is not None:
        change_color = get_change_color(change)
        change_text = f"[{change_color}]{format_currency(change, currency='¥', decimal_places=0)} ({format_percentage(change_percent)})[/{change_color}]"
        table.add_row("前日比", change_text)

    # 出来高
    volume = stock_data.get("volume")
    if volume:
        table.add_row("出来高", format_volume(volume))

    return table


def create_company_info_table(company_data: dict) -> Table:
    """
    企業情報テーブルを作成

    Args:
        company_data: 企業データ

    Returns:
        Richテーブル
    """
    table = Table(title="企業情報")
    table.add_column("項目", style="cyan", no_wrap=True)
    table.add_column("値", style="white")

    # 企業名
    name = company_data.get("name")
    if name:
        table.add_row("企業名", name)

    # セクター
    sector = company_data.get("sector")
    if sector:
        table.add_row("セクター", sector)

    # 業種
    industry = company_data.get("industry")
    if industry:
        table.add_row("業種", industry)

    # 時価総額
    market_cap = company_data.get("market_cap")
    if market_cap:
        table.add_row("時価総額", format_currency(market_cap))

    # 従業員数
    employees = company_data.get("employees")
    if employees:
        table.add_row("従業員数", f"{employees:,}人")

    return table


def create_historical_data_table(
    df: pd.DataFrame, symbol: str, period: str, interval: str, max_rows: int = 10
) -> Table:
    """
    ヒストリカルデータテーブルを作成

    Args:
        df: ヒストリカルデータ
        symbol: 銘柄コード
        period: 期間
        interval: 間隔
        max_rows: 最大表示行数

    Returns:
        Richテーブル
    """
    table = Table(title=f"{symbol} - 過去 {period} ({interval})")
    table.add_column("日時", style="cyan")
    table.add_column("始値", justify="right")
    table.add_column("高値", justify="right")
    table.add_column("安値", justify="right")
    table.add_column("終値", justify="right")
    table.add_column("出来高", justify="right")

    # 最新のmax_rows件を表示
    display_df = df.tail(max_rows)

    for idx, row in display_df.iterrows():
        # 前日比で色分け
        color = "white"
        if len(df) > 1 and idx > df.index[0]:
            try:
                prev_idx = df.index[df.index.get_loc(idx) - 1]
                prev_close = df.loc[prev_idx, "Close"]
                color = get_change_color(row["Close"] - prev_close)
            except (KeyError, IndexError):
                pass

        table.add_row(
            str(idx.strftime("%Y-%m-%d %H:%M")),
            format_currency(row["Open"]),
            format_currency(row["High"]),
            format_currency(row["Low"]),
            f"[{color}]{format_currency(row['Close'])}[/{color}]",
            format_volume(row["Volume"]),
        )

    return table


def create_watchlist_table(watchlist_data: dict) -> Table:
    """
    ウォッチリストテーブルを作成

    Args:
        watchlist_data: ウォッチリストデータ

    Returns:
        Richテーブル
    """
    table = Table(title="ウォッチリスト")
    table.add_column("コード", style="cyan")
    table.add_column("現在値", justify="right")
    table.add_column("前日比", justify="right")
    table.add_column("前日比率", justify="right")
    table.add_column("出来高", justify="right")

    for code, data in watchlist_data.items():
        if data:
            change_color = get_change_color(data.get("change", 0))
            table.add_row(
                code,
                format_currency(data.get("current_price")),
                f"[{change_color}]{format_currency(data.get('change', 0))}[/{change_color}]",
                f"[{change_color}]{format_percentage(data.get('change_percent', 0))}[/{change_color}]",
                format_volume(data.get("volume")),
            )

    return table


def create_comparison_table(
    data: Dict[str, Dict[str, Any]], title: str = "比較表"
) -> Table:
    """
    比較テーブルを作成

    Args:
        data: 比較データ（キー: 項目名, 値: データ辞書）
        title: テーブルタイトル

    Returns:
        比較テーブル
    """
    table = Table(title=title, box=box.ROUNDED)

    if not data:
        table.add_column("No Data", style="dim")
        table.add_row("データがありません")
        return table

    # カラムを追加
    table.add_column("項目", style="cyan", no_wrap=True)
    for key in data:
        table.add_column(key, justify="right")

    # 共通キーを取得
    all_keys: Set[str] = set()
    for item_data in data.values():
        all_keys.update(item_data.keys())

    # 行を追加
    for metric in sorted(all_keys):
        row = [metric]
        for _item_key, item_data in data.items():
            value = item_data.get(metric, "N/A")

            # 数値の場合は適切にフォーマット
            if isinstance(value, (int, float)) and value != "N/A":
                if abs(value) >= 1000:
                    formatted_value = format_large_number(value)
                elif metric.lower().endswith(("rate", "ratio", "percent")):
                    formatted_value = format_percentage(value)
                elif metric.lower().endswith(("price", "cost", "value")):
                    formatted_value = format_currency(value)
                else:
                    formatted_value = f"{value:,.2f}"
            else:
                formatted_value = str(value)

            row.append(formatted_value)

        table.add_row(*row)

    return table


def format_cli_table_data(
    headers: List[str], rows: List[List[str]], title: str = None
) -> Table:
    """
    CLI用テーブルデータをフォーマット

    Args:
        headers: ヘッダー行
        rows: データ行のリスト
        title: テーブルタイトル（オプション）

    Returns:
        フォーマット済みテーブル
    """
    table = Table(title=title, box=box.ROUNDED)

    # ヘッダーを追加
    for i, header in enumerate(headers):
        style = "cyan" if i == 0 else "white"
        table.add_column(header, style=style)

    # データ行を追加
    for row in rows:
        # 行の長さをヘッダーに合わせる
        padded_row = row[: len(headers)] + [""] * (len(headers) - len(row))
        table.add_row(*padded_row)

    return table