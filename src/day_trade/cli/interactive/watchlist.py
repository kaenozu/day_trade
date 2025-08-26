"""
ウォッチリスト管理のCLIコマンド群
ウォッチリストの追加、削除、表示、編集機能を提供
"""

import logging
from typing import List, Optional

import click
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from ...data.stock_fetcher import InvalidSymbolError
from ...utils.formatters import (
    create_error_panel,
    create_info_panel,
    create_success_panel,
    create_warning_panel,
    format_currency,
    format_percentage,
)
from ...utils.validators import normalize_stock_codes
from .helpers import get_watchlist_manager

logger = logging.getLogger(__name__)
console = Console()


@click.group()
def watchlist():
    """ウォッチリスト管理"""
    pass


@watchlist.command()
@click.argument("codes", nargs=-1, required=True)
@click.option("--group", "-g", default="default", help="グループ名")
@click.option("--priority", "-p", default="medium", help="優先度 (low, medium, high)")
def add(codes: List[str], group: str, priority: str):
    """ウォッチリストに銘柄を追加"""
    manager = get_watchlist_manager()
    normalized_codes = normalize_stock_codes(codes)
    if not normalized_codes:
        console.print(
            create_error_panel(
                "ウォッチリストに追加するための有効な銘柄コードが指定されていません。",
                title="入力エラー",
            )
        )
        return

    added_count = 0
    for code in normalized_codes:
        try:
            success = manager.add_stock(code, group, priority)
            if success:
                console.print(
                    create_success_panel(f"{code} をウォッチリストに追加しました。")
                )
                added_count += 1
            else:
                console.print(create_warning_panel(f"{code} は既に追加されています。"))
        except InvalidSymbolError as e:
            console.print(
                create_error_panel(
                    f"銘柄コード '{code}' は無効です。詳細: {e}\n"
                    "正しい銘柄コードを入力してください。",
                    title="無効な銘柄コード",
                )
            )
        except Exception as e:
            console.print(
                create_error_panel(
                    f"銘柄コード '{code}' をウォッチリストに追加中に"
                    f"予期せぬエラーが発生しました。詳細: {e}",
                    title="追加エラー",
                )
            )

    if added_count > 0:
        console.print(create_success_panel(f"{added_count} 件の銘柄を追加しました。"))


@watchlist.command()
@click.argument("codes", nargs=-1, required=True)
def remove(codes: List[str]):
    """ウォッチリストから銘柄を削除"""
    manager = get_watchlist_manager()
    normalized_codes = normalize_stock_codes(codes)
    if not normalized_codes:
        console.print(
            create_error_panel(
                "ウォッチリストから削除するための有効な銘柄コードが指定されていません。",
                title="入力エラー",
            )
        )
        return

    removed_count = 0
    for code in normalized_codes:
        try:
            success = manager.remove_stock(code)
            if success:
                console.print(
                    create_success_panel(f"{code} をウォッチリストから削除しました。")
                )
                removed_count += 1
            else:
                console.print(
                    create_warning_panel(f"{code} はウォッチリストにありません。")
                )
        except Exception as e:
            console.print(
                create_error_panel(
                    f"銘柄コード '{code}' をウォッチリストから削除中に"
                    f"予期せぬエラーが発生しました。詳細: {e}",
                    title="削除エラー",
                )
            )

    if removed_count > 0:
        console.print(create_success_panel(f"{removed_count} 件の銘柄を削除しました。"))


@watchlist.command("list")
def list_watchlist():
    """ウォッチリストの内容を表示"""
    manager = get_watchlist_manager()
    with console.status("[bold green]ウォッチリストを取得中..."):
        items = manager.get_watchlist()

    if not items:
        console.print(
            create_info_panel(
                "ウォッチリストは空です。`add` コマンドで銘柄を追加してください。"
            )
        )
        return

    table = Table(title="ウォッチリスト")
    table.add_column("銘柄コード", style="cyan", justify="left")
    table.add_column("銘柄名", style="white", justify="left")
    table.add_column("グループ", style="magenta", justify="left")
    table.add_column("優先度", style="yellow", justify="left")
    table.add_column("価格", style="green", justify="right")
    table.add_column("変化率", style="white", justify="right")
    table.add_column("メモ", style="dim", justify="left")

    for item in items:
        change_color = "red" if item.get("change_percent", 0) < 0 else "green"
        memo_text = item.get("memo", "")
        if len(memo_text) > 20:
            memo_text = memo_text[:20] + "..."

        table.add_row(
            item.get("stock_code", "N/A"),
            item.get("stock_name", "N/A"),
            item.get("group", "N/A"),
            item.get("priority", "N/A"),
            format_currency(item.get("current_price")),
            f"[{change_color}]{format_percentage(item.get('change_percent', 0))}[/{change_color}]",
            memo_text,
        )

    console.print(table)


@watchlist.command()
@click.argument("code")
@click.option("--memo", "-m", help="メモの内容")
def memo(code: str, memo: Optional[str]):
    """ウォッチリスト銘柄にメモを追加・更新"""
    manager = get_watchlist_manager()
    normalized_codes = normalize_stock_codes([code])
    if not normalized_codes:
        console.print(
            create_error_panel(
                "メモを追加・更新するための銘柄コードが指定されていません。",
                title="入力エラー",
            )
        )
        return
    code = normalized_codes[0]

    if memo is None:
        watchlist_items = manager.get_watchlist(codes=[code])
        current_memo = (
            watchlist_items[0].get("memo", "") if watchlist_items else ""
        )
        memo = Prompt.ask(
            f"[cyan]メモを入力してください (現在のメモ: '{current_memo}')[/cyan]"
        )

    try:
        success = manager.update_memo(code, memo)
        if success:
            console.print(create_success_panel(f"{code} のメモを更新しました。"))
        else:
            console.print(create_error_panel(f"{code} はウォッチリストにありません。"))
    except Exception as e:
        console.print(
            create_error_panel(
                f"銘柄コード '{code}' のメモ更新中に予期せぬエラーが発生しました。"
                f"詳細: {e}",
                title="メモ更新エラー",
            )
        )


@watchlist.command()
@click.argument("code")
@click.argument("group")
def move(code: str, group: str):
    """ウォッチリスト銘柄を別のグループに移動"""
    manager = get_watchlist_manager()
    normalized_codes = normalize_stock_codes([code])
    if not normalized_codes:
        console.print(
            create_error_panel(
                "銘柄を移動するための銘柄コードが指定されていません。",
                title="入力エラー",
            )
        )
        return
    code = normalized_codes[0]

    try:
        success = manager.move_to_group(code, group)
        if success:
            console.print(
                create_success_panel(f"{code} を {group} グループに移動しました。")
            )
        else:
            console.print(create_error_panel(f"{code} はウォッチリストにありません。"))
    except Exception as e:
        console.print(
            create_error_panel(
                f"銘柄コード '{code}' のグループ移動中に予期せぬエラーが"
                f"発生しました。詳細: {e}",
                title="グループ移動エラー",
            )
        )


@watchlist.command()
@click.confirmation_option(prompt="本当にウォッチリストをクリアしますか？")
def clear():
    """ウォッチリストの内容を全てクリア"""
    manager = get_watchlist_manager()
    try:
        manager.clear_watchlist()
        console.print(create_success_panel("ウォッチリストを全てクリアしました。"))
    except Exception as e:
        console.print(
            create_error_panel(
                f"ウォッチリストのクリア中に予期せぬエラーが発生しました。"
                f"詳細: {e}",
                title="クリアエラー",
            )
        )