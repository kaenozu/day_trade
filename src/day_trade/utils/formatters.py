"""
表示フォーマット用ユーティリティ
"""
from typing import Union, Optional
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
import pandas as pd


def format_currency(
    amount: Union[int, float], 
    currency: str = "¥", 
    decimal_places: int = 0
) -> str:
    """
    通貨フォーマット
    
    Args:
        amount: 金額
        currency: 通貨記号
        decimal_places: 小数点以下の桁数
        
    Returns:
        フォーマット済み金額文字列
    """
    if amount is None:
        return "N/A"
    
    if decimal_places == 0:
        return f"{currency}{amount:,.0f}"
    else:
        return f"{currency}{amount:,.{decimal_places}f}"


def format_percentage(
    value: Union[int, float], 
    decimal_places: int = 2,
    show_sign: bool = True
) -> str:
    """
    パーセンテージフォーマット
    
    Args:
        value: 値
        decimal_places: 小数点以下の桁数
        show_sign: 符号を表示するか
        
    Returns:
        フォーマット済みパーセンテージ文字列
    """
    if value is None:
        return "N/A"
    
    sign = "+" if show_sign and value > 0 else ""
    return f"{sign}{value:.{decimal_places}f}%"


def format_volume(volume: Union[int, float]) -> str:
    """
    出来高フォーマット
    
    Args:
        volume: 出来高
        
    Returns:
        フォーマット済み出来高文字列
    """
    if volume is None:
        return "N/A"
    
    volume = int(volume)
    if volume >= 1_000_000:
        return f"{volume / 1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"{volume / 1_000:.1f}K"
    else:
        return f"{volume:,}"


def get_change_color(value: Union[int, float]) -> str:
    """
    変化値に基づく色を取得
    
    Args:
        value: 変化値
        
    Returns:
        色名
    """
    if value > 0:
        return "green"
    elif value < 0:
        return "red"
    else:
        return "white"


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
    current_price = stock_data.get('current_price')
    if current_price:
        table.add_row("現在値", format_currency(current_price))
    
    # 前日終値
    previous_close = stock_data.get('previous_close')
    if previous_close:
        table.add_row("前日終値", format_currency(previous_close))
    
    # 前日比
    change = stock_data.get('change')
    change_percent = stock_data.get('change_percent')
    if change is not None and change_percent is not None:
        change_color = get_change_color(change)
        change_text = f"[{change_color}]{format_currency(change, currency='¥', decimal_places=0)} ({format_percentage(change_percent)})[/{change_color}]"
        table.add_row("前日比", change_text)
    
    # 出来高
    volume = stock_data.get('volume')
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
    name = company_data.get('name')
    if name:
        table.add_row("企業名", name)
    
    # セクター
    sector = company_data.get('sector')
    if sector:
        table.add_row("セクター", sector)
    
    # 業種
    industry = company_data.get('industry')
    if industry:
        table.add_row("業種", industry)
    
    # 時価総額
    market_cap = company_data.get('market_cap')
    if market_cap:
        table.add_row("時価総額", format_currency(market_cap))
    
    # 従業員数
    employees = company_data.get('employees')
    if employees:
        table.add_row("従業員数", f"{employees:,}人")
    
    return table


def create_historical_data_table(
    df: pd.DataFrame, 
    symbol: str, 
    period: str, 
    interval: str,
    max_rows: int = 10
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
                prev_close = df.loc[prev_idx, 'Close']
                color = get_change_color(row['Close'] - prev_close)
            except (KeyError, IndexError):
                pass
        
        table.add_row(
            str(idx.strftime('%Y-%m-%d %H:%M')),
            format_currency(row['Open']),
            format_currency(row['High']),
            format_currency(row['Low']),
            f"[{color}]{format_currency(row['Close'])}[/{color}]",
            format_volume(row['Volume'])
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
            change_color = get_change_color(data.get('change', 0))
            table.add_row(
                code,
                format_currency(data.get('current_price')),
                f"[{change_color}]{format_currency(data.get('change', 0))}[/{change_color}]",
                f"[{change_color}]{format_percentage(data.get('change_percent', 0))}[/{change_color}]",
                format_volume(data.get('volume'))
            )
    
    return table


def create_error_panel(message: str, title: str = "エラー") -> Panel:
    """
    エラーパネルを作成
    
    Args:
        message: エラーメッセージ
        title: パネルタイトル
        
    Returns:
        Richパネル
    """
    return Panel(
        Text(message, style="red"),
        title=title,
        border_style="red"
    )


def create_success_panel(message: str, title: str = "成功") -> Panel:
    """
    成功パネルを作成
    
    Args:
        message: 成功メッセージ
        title: パネルタイトル
        
    Returns:
        Richパネル
    """
    return Panel(
        Text(message, style="green"),
        title=title,
        border_style="green"
    )