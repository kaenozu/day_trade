"""
統合フォーマットユーティリティ

分散していたフォーマット機能を統合し、
共通のフォーマット関数を提供します。

統合される機能:
- formatters.py からの基本フォーマット機能
- 各モジュールで重複していたフォーマット機能
- 新しい統一されたフォーマットAPI
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from rich import box
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..config.unified_config import get_unified_config_manager


def format_currency(
    amount: Union[int, float, Decimal],
    currency: str = "JPY",
    decimal_places: Optional[int] = None,
) -> str:
    """
    通貨フォーマット（設定考慮版）

    Args:
        amount: 金額
        currency: 通貨記号
        decimal_places: 小数点以下の桁数（Noneの場合は設定から取得）

    Returns:
        フォーマット済み金額文字列
    """
    if amount is None:
        return "N/A"

    if decimal_places is None:
        config = get_unified_config_manager()
        decimal_places = config.get_screening_config().formatting.get("currency_precision", 0)

    if isinstance(amount, Decimal):
        amount = float(amount)

    if decimal_places == 0:
        return f"{currency}{amount:,.0f}"
    else:
        return f"{currency}{amount:,.{decimal_places}f}"


def format_percentage(
    value: Union[int, float, Decimal],
    decimal_places: Optional[int] = None,
    show_sign: bool = True,
) -> str:
    """
    パーセンテージフォーマット（設定考慮版）

    Args:
        value: 値
        decimal_places: 小数点以下の桁数（Noneの場合は設定から取得）
        show_sign: 符号を表示するか

    Returns:
        フォーマット済みパーセンテージ文字列
    """
    if value is None:
        return "N/A"

    if decimal_places is None:
        config = get_unified_config_manager()
        decimal_places = config.get_screening_config().formatting.get("percentage_precision", 2)

    if isinstance(value, Decimal):
        value = float(value)

    sign = "+" if show_sign and value > 0 else ""
    return f"{sign}{value:.{decimal_places}f}%"


def format_volume(volume: Union[int, float], compact: Optional[bool] = None) -> str:
    """
    出来高フォーマット（設定考慮版）

    Args:
        volume: 出来高
        compact: コンパクト表示するか（Noneの場合は設定から取得）

    Returns:
        フォーマット済み出来高文字列
    """
    if volume is None:
        return "N/A"

    if compact is None:
        config = get_unified_config_manager()
        compact = config.get_screening_config().formatting.get("volume_compact", True)

    volume = int(volume)

    if not compact:
        return f"{volume:,}"

    if volume >= 1_000_000_000:
        return f"{volume / 1_000_000_000:.1f}B"
    elif volume >= 1_000_000:
        return f"{volume / 1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"{volume / 1_000:.1f}K"
    else:
        return f"{volume:,}"


def format_large_number(number: Union[int, float, Decimal], precision: int = 1) -> str:
    """
    大きな数値を適切にフォーマット

    Args:
        number: 数値
        precision: 小数点以下桁数

    Returns:
        フォーマット済み数値文字列
    """
    if number is None:
        return "N/A"

    if isinstance(number, Decimal):
        number = float(number)

    abs_number = abs(number)
    sign = "-" if number < 0 else ""

    if abs_number >= 1_000_000_000_000:  # 1兆以上
        return f"{sign}{abs_number / 1_000_000_000_000:.{precision}f}T"
    elif abs_number >= 1_000_000_000:  # 10億以上
        return f"{sign}{abs_number / 1_000_000_000:.{precision}f}B"
    elif abs_number >= 1_000_000:  # 100万以上
        return f"{sign}{abs_number / 1_000_000:.{precision}f}M"
    elif abs_number >= 1_000:  # 1000以上
        return f"{sign}{abs_number / 1_000:.{precision}f}K"
    else:
        return f"{sign}{abs_number:.{precision}f}"


def format_datetime(dt: Union[datetime, date, str], format_type: str = "default") -> str:
    """
    日時フォーマット

    Args:
        dt: 日時オブジェクト
        format_type: フォーマットタイプ ("default", "short", "long", "iso")

    Returns:
        フォーマット済み日時文字列
    """
    if dt is None:
        return "N/A"

    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt

    format_map = {
        "default": "%Y-%m-%d %H:%M:%S",
        "short": "%m/%d %H:%M",
        "long": "%Y年%m月%d日 %H時%M分%S秒",
        "iso": "%Y-%m-%dT%H:%M:%S",
        "date_only": "%Y-%m-%d",
        "time_only": "%H:%M:%S",
    }

    format_string = format_map.get(format_type, format_map["default"])

    try:
        return dt.strftime(format_string)
    except (ValueError, AttributeError):
        return str(dt)


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


def format_decimal_safe(value: Any, decimal_places: int = 2) -> str:
    """
    Decimal型を含む数値の安全なフォーマット

    Args:
        value: フォーマットする値
        decimal_places: 小数点以下桁数

    Returns:
        フォーマット済み文字列
    """
    if value is None:
        return "N/A"

    try:
        if isinstance(value, Decimal):
            if value.is_nan():
                return "NaN"
            elif value.is_infinite():
                return "∞" if value > 0 else "-∞"
            else:
                return f"{value:.{decimal_places}f}"
        elif isinstance(value, (int, float)):
            if value != value:  # NaN check
                return "NaN"
            elif value == float("inf"):
                return "∞"
            elif value == float("-inf"):
                return "-∞"
            else:
                return f"{value:.{decimal_places}f}"
        else:
            return str(value)
    except (ValueError, TypeError, OverflowError):
        return str(value)


def create_unified_table(
    data: List[Dict[str, Any]],
    headers: Optional[List[str]] = None,
    title: Optional[str] = None,
    formatters: Optional[Dict[str, callable]] = None,
) -> Table:
    """
    統合テーブル作成（設定考慮版）

    Args:
        data: テーブルデータ
        headers: ヘッダー（Noneの場合は最初の行のキーを使用）
        title: テーブルタイトル
        formatters: カラム固有のフォーマッター辞書

    Returns:
        Rich Table
    """
    if not data:
        table = Table(title=title or "データなし")
        table.add_column("メッセージ", style="dim")
        table.add_row("データがありません")
        return table

    # ヘッダーを決定
    if headers is None:
        headers = list(data[0].keys())

    # テーブル作成
    table = Table(title=title, box=box.ROUNDED)

    # カラム追加
    for i, header in enumerate(headers):
        style = "cyan" if i == 0 else "white"
        justify = "right" if _is_numeric_column(data, header) else "left"
        table.add_column(header, style=style, justify=justify)

    # データ追加
    for row_data in data:
        row_values = []
        for header in headers:
            value = row_data.get(header, "")

            # フォーマッター適用
            if formatters and header in formatters:
                try:
                    value = formatters[header](value)
                except Exception:
                    value = str(value)
            else:
                # デフォルトフォーマット適用
                value = _apply_default_format(header, value)

            row_values.append(str(value))

        table.add_row(*row_values)

    return table


def _is_numeric_column(data: List[Dict[str, Any]], column: str) -> bool:
    """カラムが数値型かどうかを判定"""
    if not data:
        return False

    # 最初の非None値をチェック
    for row in data[:10]:  # 最初の10行をチェック
        value = row.get(column)
        if value is not None:
            return isinstance(value, (int, float, Decimal))

    return False


def _apply_default_format(column_name: str, value: Any) -> str:
    """カラム名に基づいてデフォルトフォーマットを適用"""
    if value is None:
        return "N/A"

    column_lower = column_name.lower()

    # 価格系
    if any(keyword in column_lower for keyword in ["price", "cost", "value", "金額", "価格"]):
        if isinstance(value, (int, float, Decimal)):
            return format_currency(value)

    # パーセンテージ系
    elif any(keyword in column_lower for keyword in ["rate", "ratio", "percent", "率", "比率"]):
        if isinstance(value, (int, float, Decimal)):
            return format_percentage(value)

    # 出来高系
    elif any(keyword in column_lower for keyword in ["volume", "出来高"]):
        if isinstance(value, (int, float)):
            return format_volume(value)

    # 日時系
    elif any(keyword in column_lower for keyword in ["date", "time", "日時", "時刻"]):
        if isinstance(value, (datetime, date)) or isinstance(value, str) and "T" in value:
            return format_datetime(value)

    # 数値系（大きな数値）
    elif isinstance(value, (int, float, Decimal)) and abs(value) >= 1000:
        return format_large_number(value)

    # デフォルト
    return format_decimal_safe(value)


def create_summary_panel(
    title: str, metrics: Dict[str, Any], columns: int = 2, panel_style: str = "blue"
) -> Panel:
    """
    サマリーパネル作成

    Args:
        title: パネルタイトル
        metrics: メトリクス辞書
        columns: 列数
        panel_style: パネルスタイル

    Returns:
        Rich Panel
    """
    if not metrics:
        return Panel(
            Text("データがありません", style="dim"),
            title=title,
            border_style=panel_style,
        )

    # メトリクスを文字列に変換
    metric_lines = []
    for key, value in metrics.items():
        formatted_value = _apply_default_format(key, value)
        metric_lines.append(f"{key}: {formatted_value}")

    # 複数列に分割
    if len(metric_lines) > 4 and columns > 1:
        mid = len(metric_lines) // 2
        left_column = "\n".join(metric_lines[:mid])
        right_column = "\n".join(metric_lines[mid:])

        columns_obj = Columns(
            [
                Panel(Text(left_column, style="white"), border_style="dim"),
                Panel(Text(right_column, style="white"), border_style="dim"),
            ],
            equal=True,
        )

        return Panel(columns_obj, title=title, border_style=panel_style)
    else:
        content = "\n".join(metric_lines)
        return Panel(Text(content, style="white"), title=title, border_style=panel_style)


def create_status_table(status_data: Dict[str, Dict[str, Any]], title: str = "ステータス") -> Table:
    """
    ステータステーブル作成

    Args:
        status_data: ステータスデータ
        title: テーブルタイトル

    Returns:
        Rich Table
    """
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("項目", style="cyan", no_wrap=True)
    table.add_column("ステータス", justify="center")
    table.add_column("値", justify="right")
    table.add_column("詳細", style="dim")

    for item, data in status_data.items():
        status = data.get("status", "unknown")
        value = data.get("value", "N/A")
        details = data.get("details", "")

        # ステータスに基づく色付け
        if status == "ok":
            status_text = f"[green]✓ {status.upper()}[/green]"
        elif status == "error":
            status_text = f"[red]✗ {status.upper()}[/red]"
        elif status == "warning":
            status_text = f"[yellow]⚠ {status.upper()}[/yellow]"
        else:
            status_text = f"[dim]• {status.upper()}[/dim]"

        formatted_value = _apply_default_format(item, value)

        table.add_row(item, status_text, formatted_value, details)

    return table


# 後方互換性のためのエイリアス
def create_stock_info_table(stock_data: dict) -> Table:
    """株価情報テーブル作成（後方互換性）"""
    return create_unified_table([stock_data], title=f"銘柄情報: {stock_data.get('symbol', 'N/A')}")


def create_error_panel(
    message: str,
    title: str = "エラー",
    solutions: Optional[List[str]] = None,
) -> Panel:
    """エラーパネル作成（統合版）"""
    content_lines = [f"[bold red]❌ {message}[/bold red]"]

    if solutions:
        content_lines.extend(["", "[bold yellow]💡 解決方法:[/bold yellow]"])
        for i, solution in enumerate(solutions, 1):
            content_lines.append(f"  {i}. {solution}")

    content = "\n".join(content_lines)

    return Panel(
        content,
        title=f"[bold red]{title}[/bold red]",
        border_style="red",
        padding=(1, 2),
    )
