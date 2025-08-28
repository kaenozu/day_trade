"""
基本的なCLIコマンドの定義
stock, history, validate, init等の基本機能を提供
"""

import logging
from typing import List

import click
from rich.console import Console
from rich.table import Table

from ...data.stock_fetcher import (
    DataNotFoundError,
    InvalidSymbolError,
    StockFetcher,
)
from ...models.database import init_db
from ...utils.formatters import (
    create_error_panel,
    create_success_panel,
    create_watchlist_table,
)
from ...utils.validators import (
    normalize_stock_codes,
    suggest_stock_code_correction,
    validate_stock_code,
)
from .helpers import display_historical_data, display_stock_details

logger = logging.getLogger(__name__)
console = Console()


@click.command()
def init():
    """データベースを初期化"""
    try:
        init_db()
        console.print(create_success_panel("データベースを初期化しました。"))
    except Exception as e:
        console.print(
            create_error_panel(
                f"データベースの初期化中にエラーが発生しました。詳細: {e}\n"
                "システム管理者にお問い合わせいただくか、再度お試しください。",
                title="データベースエラー",
            )
        )


@click.command()
@click.argument("code")
@click.option("--details", "-d", is_flag=True, help="詳細情報を表示")
def stock(code: str, details: bool):
    """個別銘柄の情報を表示"""
    if not validate_stock_code(code):
        suggestion = suggest_stock_code_correction(code)
        if suggestion:
            console.print(
                create_error_panel(
                    f"無効な銘柄コードが入力されました: '{code}'。"
                    f"修正候補: {suggestion}",
                    title="入力エラー",
                )
            )
        else:
            console.print(
                create_error_panel(
                    f"無効な銘柄コードが入力されました: '{code}'。"
                    "正しい銘柄コードを入力してください。",
                    title="入力エラー",
                )
            )
        return

    fetcher = StockFetcher()
    normalized_codes = normalize_stock_codes([code])
    if not normalized_codes:
        console.print(
            create_error_panel(
                f"銘柄コード '{code}' をシステム内部で処理できる形式に変換"
                "できませんでした。入力を見直すか、サポートされている"
                "銘柄コード形式をご確認ください。",
                title="正規化エラー",
            )
        )
        return

    code = normalized_codes[0]

    with console.status(f"[bold green]{code}の情報を取得中..."):
        current = fetcher.get_current_price(code)

    display_stock_details(code, current, details)


@click.command()
@click.argument("code")
@click.option("--period", "-p", default="5d", help="期間 (1d,5d,1mo,3mo,6mo,1y)")
@click.option("--interval", "-i", default="1d", help="間隔 (1m,5m,15m,30m,60m,1d)")
@click.option("--rows", "-r", default=10, help="表示行数")
def history(code: str, period: str, interval: str, rows: int):
    """ヒストリカルデータを表示"""
    if not validate_stock_code(code):
        console.print(create_error_panel(f"無効な銘柄コード: {code}"))
        return

    fetcher = StockFetcher()
    normalized_codes = normalize_stock_codes([code])
    if not normalized_codes:
        console.print(
            create_error_panel(
                f"銘柄コード '{code}' をシステム内部で処理できる形式に変換"
                "できませんでした。入力を見直すか、サポートされている"
                "銘柄コード形式をご確認ください。",
                title="正規化エラー",
            )
        )
        return

    code = normalized_codes[0]

    with console.status(f"[bold green]{code}のヒストリカルデータを取得中..."):
        try:
            df = fetcher.get_historical_data(code, period=period, interval=interval)
            display_historical_data(code, df, period, interval, rows)
        except (DataNotFoundError, InvalidSymbolError) as e:
            console.print(
                create_error_panel(
                    f"銘柄コード '{code}' のヒストリカルデータの取得中に"
                    "エラーが発生しました。インターネット接続を確認するか、"
                    f"銘柄コードが正しいことを再確認してください。詳細: {e}",
                    title="データ取得エラー",
                )
            )
        except Exception as e:
            console.print(
                create_error_panel(f"予期しないエラー: {e}", title="予期せぬエラー")
            )


@click.command()
@click.argument("codes", nargs=-1, required=True)
def watch(codes):
    """複数銘柄の現在価格を一覧表示"""
    normalized_codes = normalize_stock_codes(list(codes))
    if not normalized_codes:
        console.print(
            create_error_panel(
                "有効な銘柄コードが一つも指定されていません。"
                "少なくとも一つ正しい銘柄コードを入力してください。",
                title="入力エラー",
            )
        )
        return

    fetcher = StockFetcher()
    with console.status("[bold green]価格情報を取得中..."):
        results = fetcher.get_realtime_data(normalized_codes)

    if not results:
        console.print(
            create_error_panel(
                "指定された銘柄コードの現在価格情報を取得できませんでした。"
                "市場が開いているか、インターネット接続をご確認ください。",
                title="情報取得エラー",
            )
        )
        return

    table = create_watchlist_table(results)
    console.print(table)


@click.command("validate")
@click.argument("codes", nargs=-1, required=True)
def validate_codes(codes: List[str]):
    """銘柄コードの妥当性を検証"""
    table = Table(title="銘柄コード検証結果")
    table.add_column("コード", style="cyan")
    table.add_column("有効性", style="white")
    table.add_column("正規化後", style="yellow")
    table.add_column("提案", style="green")

    for code in codes:
        is_valid = validate_stock_code(code)
        normalized = normalize_stock_codes([code])
        suggestion = suggest_stock_code_correction(code)

        validity = "[green]有効[/green]" if is_valid else "[red]無効[/red]"
        normalized_str = normalized[0] if normalized else "N/A"
        suggestion_str = suggestion or "なし"

        table.add_row(code, validity, normalized_str, suggestion_str)

    console.print(table)