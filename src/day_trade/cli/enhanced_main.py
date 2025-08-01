"""
拡張されたメインCLIエントリーポイント
"""

import click
from rich.console import Console
from rich.table import Table
from pathlib import Path

from ..data.stock_fetcher import StockFetcher
from ..models import init_db
from ..core.config import config_manager
from ..utils.validators import (
    validate_stock_code,
    validate_period,
    validate_interval,
    normalize_stock_codes,
    suggest_stock_code_correction,
)
from ..utils.formatters import (
    create_stock_info_table,
    create_company_info_table,
    create_historical_data_table,
    create_watchlist_table,
    create_error_panel,
    create_success_panel,
)

console = Console()


@click.group()
@click.version_option(version="0.1.0")
@click.option("--config", "-c", type=click.Path(), help="設定ファイルのパス")
@click.pass_context
def cli(ctx, config):
    """デイトレード支援ツール"""
    # コンテキストに設定を保存
    ctx.ensure_object(dict)
    if config:
        ctx.obj["config_path"] = Path(config)
    else:
        ctx.obj["config_path"] = None


@cli.command()
def init():
    """データベースを初期化"""
    try:
        init_db()
        console.print(create_success_panel("データベースを初期化しました。"))
    except Exception as e:
        console.print(create_error_panel(f"データベース初期化エラー: {e}"))


@cli.command()
@click.argument("code")
@click.option("--details", "-d", is_flag=True, help="詳細情報を表示")
def stock(code: str, details: bool):
    """個別銘柄の情報を表示"""
    # 入力検証
    if not validate_stock_code(code):
        suggestion = suggest_stock_code_correction(code)
        if suggestion:
            console.print(
                create_error_panel(f"無効な銘柄コード: {code}\n修正候補: {suggestion}")
            )
        else:
            console.print(create_error_panel(f"無効な銘柄コード: {code}"))
        return

    fetcher = StockFetcher()
    normalized_codes = normalize_stock_codes([code])
    if not normalized_codes:
        console.print(
            create_error_panel(f"銘柄コード {code} を正規化できませんでした。")
        )
        return

    code = normalized_codes[0]

    # 現在価格を取得
    with console.status(f"[bold green]{code}の情報を取得中..."):
        current = fetcher.get_current_price(code)

    if not current:
        console.print(
            create_error_panel(f"銘柄コード {code} の情報を取得できませんでした。")
        )
        return

    # 基本情報テーブル
    table = create_stock_info_table(current)
    console.print(table)

    # 詳細情報
    if details:
        with console.status("企業情報を取得中..."):
            info = fetcher.get_company_info(code)

        if info:
            detail_table = create_company_info_table(info)
            console.print("\n")
            console.print(detail_table)
        else:
            console.print("\n")
            console.print(create_error_panel("企業情報を取得できませんでした。"))


@cli.command()
@click.argument("code")
@click.option("--period", "-p", default="5d", help="期間 (1d,5d,1mo,3mo,6mo,1y)")
@click.option("--interval", "-i", default="1d", help="間隔 (1m,5m,15m,30m,60m,1d)")
@click.option("--rows", "-r", default=10, help="表示行数")
def history(code: str, period: str, interval: str, rows: int):
    """ヒストリカルデータを表示"""
    # 入力検証
    if not validate_stock_code(code):
        console.print(create_error_panel(f"無効な銘柄コード: {code}"))
        return

    if not validate_period(period):
        console.print(create_error_panel(f"無効な期間: {period}"))
        return

    if not validate_interval(interval):
        console.print(create_error_panel(f"無効な間隔: {interval}"))
        return

    fetcher = StockFetcher()
    normalized_codes = normalize_stock_codes([code])
    if not normalized_codes:
        console.print(
            create_error_panel(f"銘柄コード {code} を正規化できませんでした。")
        )
        return

    code = normalized_codes[0]

    with console.status(f"[bold green]{code}のヒストリカルデータを取得中..."):
        df = fetcher.get_historical_data(code, period=period, interval=interval)

    if df is None or df.empty:
        console.print(create_error_panel("データを取得できませんでした。"))
        return

    # テーブル作成
    table = create_historical_data_table(df, code, period, interval, max_rows=rows)
    console.print(table)

    # サマリー
    console.print("\n[bold]サマリー:[/bold]")
    console.print(f"期間高値: ¥{df['High'].max():,.0f}")
    console.print(f"期間安値: ¥{df['Low'].min():,.0f}")
    console.print(f"平均出来高: {int(df['Volume'].mean()):,}")


@cli.command()
@click.argument("codes", nargs=-1, required=True)
def watch(codes):
    """複数銘柄の現在価格を一覧表示"""
    # 入力検証と正規化
    normalized_codes = normalize_stock_codes(list(codes))
    if not normalized_codes:
        console.print(create_error_panel("有効な銘柄コードがありません。"))
        return

    fetcher = StockFetcher()

    with console.status("[bold green]価格情報を取得中..."):
        results = fetcher.get_realtime_data(normalized_codes)

    if not results:
        console.print(create_error_panel("価格情報を取得できませんでした。"))
        return

    table = create_watchlist_table(results)
    console.print(table)


@cli.group()
def config():
    """設定管理"""
    pass


@config.command("show")
def config_show():
    """現在の設定を表示"""
    config_dict = config_manager.config.model_dump()

    table = Table(title="設定情報")
    table.add_column("設定項目", style="cyan")
    table.add_column("値", style="white")

    def add_config_rows(data, prefix=""):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                add_config_rows(value, full_key)
            else:
                table.add_row(full_key, str(value))

    add_config_rows(config_dict)
    console.print(table)


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """設定値を変更"""
    try:
        # 型推定（簡易版）
        if value.lower() in ("true", "false"):
            typed_value = value.lower() == "true"
        elif value.isdigit():
            typed_value = int(value)
        elif value.replace(".", "").isdigit():
            typed_value = float(value)
        else:
            typed_value = value

        config_manager.set(key, typed_value)
        console.print(
            create_success_panel(f"設定を更新しました: {key} = {typed_value}")
        )
    except Exception as e:
        console.print(create_error_panel(f"設定更新エラー: {e}"))


@config.command("reset")
@click.confirmation_option(prompt="設定をリセットしますか？")
def config_reset():
    """設定をデフォルトに戻す"""
    try:
        config_manager.reset()
        console.print(create_success_panel("設定をデフォルトにリセットしました。"))
    except Exception as e:
        console.print(create_error_panel(f"設定リセットエラー: {e}"))


@config.command("export")
@click.argument("file_path", type=click.Path())
def config_export(file_path: str):
    """設定をファイルにエクスポート"""
    try:
        config_manager.export_config(Path(file_path))
        console.print(create_success_panel(f"設定をエクスポートしました: {file_path}"))
    except Exception as e:
        console.print(create_error_panel(f"設定エクスポートエラー: {e}"))


@config.command("import")
@click.argument("file_path", type=click.Path(exists=True))
def config_import(file_path: str):
    """設定をファイルからインポート"""
    try:
        config_manager.import_config(Path(file_path))
        console.print(create_success_panel(f"設定をインポートしました: {file_path}"))
    except Exception as e:
        console.print(create_error_panel(f"設定インポートエラー: {e}"))


@cli.command("validate")
@click.argument("codes", nargs=-1, required=True)
def validate(codes):
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


def main():
    """メインエントリーポイント"""
    cli()


if __name__ == "__main__":
    main()
