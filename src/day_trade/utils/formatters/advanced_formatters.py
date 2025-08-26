"""
高度な表示機能
メトリクスカード、トレンド指標、ステータス表示などの高度な表示処理
"""

from typing import Any, Dict

from rich.align import Align
from rich.columns import Columns
from rich.panel import Panel
from rich.text import Text

from .basic_formatters import format_currency, format_large_number, format_percentage


def create_metric_cards(metrics: Dict[str, Any], columns: int = 3) -> Columns:
    """
    メトリクスカードを作成

    Args:
        metrics: メトリクスデータ
        columns: 列数

    Returns:
        カラム表示
    """
    cards = []

    for key, value in metrics.items():
        # 値をフォーマット
        if isinstance(value, (int, float)):
            if key.lower().endswith(("rate", "ratio", "percent")):
                formatted_value = format_percentage(value)
            elif key.lower().endswith(("price", "cost", "value", "amount")):
                formatted_value = format_currency(value)
            elif abs(value) >= 1000:
                formatted_value = format_large_number(value)
            else:
                formatted_value = f"{value:,.2f}"
        else:
            formatted_value = str(value)

        # カードを作成
        card_content = Text()
        card_content.append(formatted_value, style="bold white")

        card = Panel(
            Align.center(card_content), title=key, border_style="blue", width=20
        )

        cards.append(card)

    return Columns(cards, equal=True, expand=True)


def create_trend_indicator(
    current: float, previous: float, label: str = "Trend"
) -> Text:
    """
    トレンド指標を作成

    Args:
        current: 現在値
        previous: 前回値
        label: ラベル

    Returns:
        トレンド表示テキスト
    """
    if previous == 0:
        return Text(f"{label}: N/A", style="dim")

    change = current - previous
    change_percent = (change / previous) * 100

    # トレンド記号
    if change > 0:
        trend_symbol = "↗"
        color = "green"
    elif change < 0:
        trend_symbol = "↘"
        color = "red"
    else:
        trend_symbol = "→"
        color = "yellow"

    text = Text()
    text.append(f"{label}: ", style="white")
    text.append(f"{trend_symbol} ", style=color)
    text.append(f"{format_percentage(change_percent)}", style=color)

    return text


def create_status_indicator(status: str, label: str = "Status") -> Text:
    """
    ステータス指標を作成

    Args:
        status: ステータス文字列
        label: ラベル

    Returns:
        ステータス表示テキスト
    """
    status_colors = {
        "success": "green",
        "error": "red",
        "warning": "yellow",
        "info": "blue",
        "active": "green",
        "inactive": "dim",
        "pending": "yellow",
        "completed": "green",
        "failed": "red",
    }

    status_symbols = {
        "success": "✓",
        "error": "✗",
        "warning": "⚠",
        "info": "ℹ",
        "active": "●",
        "inactive": "○",
        "pending": "◐",
        "completed": "✓",
        "failed": "✗",
    }

    color = status_colors.get(status.lower(), "white")
    symbol = status_symbols.get(status.lower(), "•")

    text = Text()
    text.append(f"{label}: ", style="white")
    text.append(f"{symbol} ", style=color)
    text.append(status.title(), style=color)

    return text