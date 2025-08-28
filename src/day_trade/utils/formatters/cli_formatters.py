"""
CLI統一フォーマット機能
CLI専用の表示フォーマット、コマンドヘルプ、ステータスバーなどの処理
"""

from typing import Dict, List

from rich import box
from rich.table import Table
from rich.text import Text


def create_cli_command_help(commands: Dict[str, str]) -> Table:
    """
    CLI用コマンドヘルプテーブルを作成

    Args:
        commands: コマンド辞書（コマンド: 説明）

    Returns:
        コマンドヘルプテーブル
    """
    table = Table(title="利用可能なコマンド", box=box.ROUNDED)
    table.add_column("コマンド", style="cyan", no_wrap=True)
    table.add_column("説明", style="white")

    for command, description in commands.items():
        table.add_row(command, description)

    return table


def create_cli_status_bar(status_items: Dict[str, str]) -> Text:
    """
    CLI用ステータスバーを作成

    Args:
        status_items: ステータス項目（ラベル: 値）

    Returns:
        ステータスバーテキスト
    """
    text = Text()

    for i, (label, value) in enumerate(status_items.items()):
        if i > 0:
            text.append(" | ", style="dim")

        text.append(f"{label}: ", style="yellow")
        text.append(value, style="white")

    return text


def create_cli_list_item(
    index: int, title: str, description: str = None, status: str = None
) -> Text:
    """
    CLI用リストアイテムを作成

    Args:
        index: インデックス番号
        title: アイテムタイトル
        description: アイテム説明（オプション）
        status: ステータス（オプション）

    Returns:
        リストアイテムテキスト
    """
    text = Text()

    # インデックス
    text.append(f"{index:2d}. ", style="dim")

    # タイトル
    text.append(title, style="bold white")

    # ステータス
    if status:
        text.append(f" [{status}]", style="green" if status == "OK" else "red")

    # 説明
    if description:
        text.append(f" - {description}", style="dim")

    return text


def create_cli_loading_indicator(message: str = "処理中...") -> Text:
    """
    CLI用ローディング指標を作成

    Args:
        message: ローディングメッセージ

    Returns:
        ローディング指標テキスト
    """
    text = Text()
    text.append("⏳ ", style="yellow")
    text.append(message, style="white")

    return text