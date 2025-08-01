"""
ウォッチリスト管理用CLIコマンド
"""

import click
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from ..core.watchlist import WatchlistManager
from ..utils.formatters import (
    create_error_panel,
    create_success_panel,
    format_currency,
    format_percentage,
    format_volume,
    get_change_color,
)
from ..utils.validators import normalize_stock_codes

console = Console()


@click.group()
def watchlist():
    """ウォッチリスト管理"""
    pass


@watchlist.command("add")
@click.argument("codes", nargs=-1, required=True)
@click.option("--group", "-g", default="default", help="グループ名")
@click.option("--memo", "-m", default="", help="メモ")
def watchlist_add(codes, group: str, memo: str):
    """銘柄をウォッチリストに追加"""
    manager = WatchlistManager()

    # 銘柄コードを正規化
    normalized_codes = normalize_stock_codes(list(codes))
    if not normalized_codes:
        console.print(
            create_error_panel(
                "ウォッチリストに追加するための有効な銘柄コードが指定されていません。",
                title="入力エラー",
            )
        )
        return

    success_count = 0
    failed_codes = []

    for code in normalized_codes:
        if manager.add_stock(code, group, memo):
            success_count += 1
        else:
            failed_codes.append(code)

    # 結果表示
    if success_count > 0:
        console.print(
            create_success_panel(
                f"{success_count}件の銘柄をウォッチリスト（{group}）に追加しました。"
            )
        )

    if failed_codes:
        console.print(
            create_error_panel(
                f"以下の銘柄のウォッチリストへの追加に失敗しました。銘柄コードが正しいか確認し、再度お試しください: {', '.join(failed_codes)}",
                title="追加エラー",
            )
        )


@watchlist.command("remove")
@click.argument("codes", nargs=-1, required=True)
@click.option("--group", "-g", default="default", help="グループ名")
def watchlist_remove(codes, group: str):
    """銘柄をウォッチリストから削除"""
    manager = WatchlistManager()

    # 銘柄コードを正規化
    normalized_codes = normalize_stock_codes(list(codes))
    if not normalized_codes:
        console.print(
            create_error_panel(
                "ウォッチリストから削除するための有効な銘柄コードが指定されていません。",
                title="入力エラー",
            )
        )
        return

    success_count = 0
    failed_codes = []

    for code in normalized_codes:
        if manager.remove_stock(code, group):
            success_count += 1
        else:
            failed_codes.append(code)

    # 結果表示
    if success_count > 0:
        console.print(
            create_success_panel(
                f"{success_count}件の銘柄をウォッチリスト（{group}）から削除しました。"
            )
        )

    if failed_codes:
        console.print(
            create_error_panel(
                f"以下の銘柄のウォッチリストからの削除に失敗しました。銘柄コードが正しいか確認し、再度お試しください: {', '.join(failed_codes)}",
                title="削除エラー",
            )
        )


@watchlist.command("list")
@click.option("--group", "-g", help="グループ名（指定しない場合は全て）")
@click.option("--prices", "-p", is_flag=True, help="価格情報も表示")
def watchlist_list(group: str, prices: bool):
    """ウォッチリストを表示"""
    manager = WatchlistManager()

    if prices:
        # 価格情報付きで取得
        with console.status("[bold green]価格情報を取得中..."):
            data = manager.get_watchlist_with_prices(group)

        if not data:
            console.print(
                create_error_panel(
                    "ウォッチリストは空です。`watchlist add` コマンドで銘柄を追加してください。",
                    title="ウォッチリスト情報",
                )
            )
            return

        # 価格情報付きテーブル
        title = (
            f"ウォッチリスト（{group}）" if group else "ウォッチリスト（全グループ）"
        )
        table = Table(title=title)
        table.add_column("コード", style="cyan")
        table.add_column("銘柄名", style="white")
        table.add_column("現在値", justify="right")
        table.add_column("前日比", justify="right")
        table.add_column("前日比率", justify="right")
        table.add_column("出来高", justify="right")
        table.add_column("グループ", style="yellow")
        table.add_column("メモ", style="dim")

        for code, item in data.items():
            change_color = get_change_color(item.get("change", 0))

            table.add_row(
                code,
                item.get("stock_name", "N/A"),
                format_currency(item.get("current_price")),
                f"[{change_color}]{format_currency(item.get('change', 0))}[/{change_color}]",
                f"[{change_color}]{format_percentage(item.get('change_percent', 0))}[/{change_color}]",
                format_volume(item.get("volume")),
                item.get("group_name", "N/A"),
                (
                    item.get("memo", "")[:20] + "..."
                    if len(item.get("memo", "")) > 20
                    else item.get("memo", "")
                ),
            )

    else:
        # 基本情報のみ
        data = manager.get_watchlist(group)

        if not data:
            console.print(
                create_error_panel(
                    "ウォッチリストは空です。`watchlist add` コマンドで銘柄を追加してください。",
                    title="ウォッチリスト情報",
                )
            )
            return

        title = (
            f"ウォッチリスト（{group}）" if group else "ウォッチリスト（全グループ）"
        )
        table = Table(title=title)
        table.add_column("コード", style="cyan")
        table.add_column("銘柄名", style="white")
        table.add_column("グループ", style="yellow")
        table.add_column("メモ", style="dim")
        table.add_column("追加日", style="dim")

        for item in data:
            table.add_row(
                item["stock_code"],
                item["stock_name"],
                item["group_name"],
                item["memo"][:30] + "..." if len(item["memo"]) > 30 else item["memo"],
                (
                    item["added_date"].strftime("%Y-%m-%d")
                    if item["added_date"]
                    else "N/A"
                ),
            )

    console.print(table)


@watchlist.command("groups")
def watchlist_groups():
    """ウォッチリストのグループ一覧を表示"""
    manager = WatchlistManager()
    groups = manager.get_groups()

    if not groups:
        console.print(
            create_error_panel(
                "ウォッチリストグループがまだ作成されていません。`watchlist add --group <グループ名>` コマンドで新しいグループに銘柄を追加してください。",
                title="グループ情報",
            )
        )
        return

    table = Table(title="ウォッチリストグループ")
    table.add_column("グループ名", style="cyan")
    table.add_column("銘柄数", justify="right")

    for group in groups:
        items = manager.get_watchlist(group)
        table.add_row(group, str(len(items)))

    console.print(table)


@watchlist.command("memo")
@click.argument("code")
@click.option("--group", "-g", default="default", help="グループ名")
@click.option("--text", "-t", help="メモテキスト（指定しない場合は対話入力）")
def watchlist_memo(code: str, group: str, text: str):
    """銘柄のメモを更新"""
    # 銘柄コードを正規化
    normalized_codes = normalize_stock_codes([code])
    if not normalized_codes:
        console.print(
            create_error_panel(
                "メモを更新するための銘柄コードが指定されていません。",
                title="入力エラー",
            )
        )
        return

    code = normalized_codes[0]

    # メモテキストの取得
    if text is None:
        text = Prompt.ask("メモを入力してください", default="")

    manager = WatchlistManager()

    if manager.update_memo(code, group, text):
        console.print(create_success_panel(f"銘柄 {code} のメモを更新しました。"))
    else:
        console.print(
            create_error_panel(
                f"銘柄コード '{code}' のメモを更新できませんでした。銘柄がウォッチリストに存在しないか、予期せぬエラーが発生しました。",
                title="メモ更新エラー",
            )
        )


@watchlist.command("move")
@click.argument("code")
@click.argument("to_group")
@click.option("--from-group", "-f", default="default", help="移動元グループ名")
def watchlist_move(code: str, to_group: str, from_group: str):
    """銘柄を別のグループに移動"""
    # 銘柄コードを正規化
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

    manager = WatchlistManager()

    if manager.move_to_group(code, from_group, to_group):
        console.print(
            create_success_panel(
                f"銘柄 {code} を {from_group} から {to_group} に移動しました。"
            )
        )
    else:
        console.print(
            create_error_panel(
                f"銘柄コード '{code}' のグループ移動に失敗しました。指定された銘柄がウォッチリストに見つからないか、移動元グループに存在しない可能性があります。",
                title="移動エラー",
            )
        )


@watchlist.command("clear")
@click.option("--group", "-g", help="削除するグループ名（指定しない場合は全て）")
def watchlist_clear(group: str):
    """ウォッチリストをクリア"""
    manager = WatchlistManager()

    # 確認
    if group:
        if not Confirm.ask(f"グループ '{group}' のウォッチリストを全て削除しますか？"):
            console.print("キャンセルしました。")
            return
        items = manager.get_watchlist(group)
    else:
        if not Confirm.ask("全てのウォッチリストを削除しますか？"):
            console.print("キャンセルしました。")
            return
        items = manager.get_watchlist()

    # 削除実行
    success_count = 0
    for item in items:
        if manager.remove_stock(item["stock_code"], item["group_name"]):
            success_count += 1

    console.print(create_success_panel(f"{success_count}件の銘柄を削除しました。"))


# メインCLIに統合するための関数
def add_watchlist_commands(main_cli):
    """メインCLIにウォッチリストコマンドを追加"""
    main_cli.add_command(watchlist)
