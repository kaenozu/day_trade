"""
メインCLIエントリーポイント
"""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..core.config import config_manager
from ..data.stock_fetcher import StockFetcher
from ..models import init_db
from ..utils.formatters import (
    create_company_info_table,
    create_error_panel,
    create_historical_data_table,
    create_stock_info_table,
    create_success_panel,
    create_watchlist_table,
)
from ..utils.logging_config import setup_logging
from ..utils.validators import (
    normalize_stock_codes,
    suggest_stock_code_correction,
    validate_interval,
    validate_period,
    validate_stock_code,
)
from .watchlist_commands import watchlist  # Moved to top

console = Console()


@click.group()
@click.version_option(version="0.1.0")
@click.option("--config", "-c", type=click.Path(), help="設定ファイルのパス")
@click.pass_context
def cli(ctx, config):
    """デイトレード支援ツール"""
    # 構造化ロギングを初期化
    setup_logging()

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
        console.print(
            create_error_panel(
                f"データベースの初期化中にエラーが発生しました。詳細: {e}\nシステム管理者にお問い合わせいただくか、再度お試しください。",
                title="データベースエラー",
            )
        )


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
                create_error_panel(
                    f"無効な銘柄コードが入力されました: '{code}'。修正候補: {suggestion}",
                    title="入力エラー",
                )
            )
        else:
            console.print(
                create_error_panel(
                    f"無効な銘柄コードが入力されました: '{code}'。正しい銘柄コードを入力してください。",
                    title="入力エラー",
                )
            )
        return

    fetcher = StockFetcher()
    normalized_codes = normalize_stock_codes([code])
    if not normalized_codes:
        console.print(
            create_error_panel(
                f"銘柄コード '{code}' をシステム内部で処理できる形式に変換できませんでした。入力を見直すか、サポートされている銘柄コード形式をご確認ください。",
                title="正規化エラー",
            )
        )
        return

    code = normalized_codes[0]

    # 現在価格を取得
    with console.status(f"[bold green]{code}の情報を取得中..."):
        current = fetcher.get_current_price(code)

    if not current:
        console.print(
            create_error_panel(
                f"銘柄コード '{code}' の現在価格または詳細情報を取得できませんでした。コードが正しいか、または市場が開いているかご確認ください。",
                title="情報取得エラー",
            )
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
            console.print(
                create_error_panel(
                    f"銘柄コード '{code}' の企業詳細情報を取得できませんでした。データプロバイダーの問題か、情報が利用できない可能性があります。",
                    title="企業情報エラー",
                )
            )


@cli.command()
@click.argument("code")
@click.option("--period", "-p", default="5d", help="期間 (1d,5d,1mo,3mo,6mo,1y)")
@click.option("--interval", "-i", default="1d", help="間隔 (1m,5m,15m,30m,60m,1d)")
@click.option("--rows", "-r", default=10, help="表示行数")
def history(code: str, period: str, interval: str, rows: int):
    """ヒストリカルデータを表示"""
    # 入力検証
    if not validate_stock_code(code):
        console.print(
            create_error_panel(
                f"無効な銘柄コードが入力されました: '{code}'。正しい銘柄コードを入力してください。",
                title="入力エラー",
            )
        )
        return

    if not validate_period(period):
        console.print(
            create_error_panel(
                f"指定された期間 '{period}' が無効です。サポートされている期間形式 (例: 1d, 5d, 1mo, 3mo, 6mo, 1y) を使用してください。",
                title="入力エラー",
            )
        )
        return

    if not validate_interval(interval):
        console.print(
            create_error_panel(
                f"指定された間隔 '{interval}' が無効です。サポートされている間隔形式 (例: 1m, 5m, 15m, 30m, 60m, 1d) を使用してください。",
                title="入力エラー",
            )
        )
        return

    fetcher = StockFetcher()
    normalized_codes = normalize_stock_codes([code])
    if not normalized_codes:
        console.print(
            create_error_panel(
                f"銘柄コード '{code}' をシステム内部で処理できる形式に変換できませんでした。入力を見直すか、サポートされている銘柄コード形式をご確認ください。",
                title="正規化エラー",
            )
        )
        return

    code = normalized_codes[0]

    with console.status(f"[bold green]{code}のヒストリカルデータを取得中..."):
        df = fetcher.get_historical_data(code, period=period, interval=interval)

    if df is None or df.empty:
        console.print(
            create_error_panel(
                f"銘柄コード '{code}' のヒストリカルデータを取得できませんでした。インターネット接続を確認するか、銘柄コード、期間、間隔が正しいことを再確認してください。",
                title="データ取得エラー",
            )
        )
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
        console.print(
            create_error_panel(
                "有効な銘柄コードが一つも指定されていません。少なくとも一つ正しい銘柄コードを入力してください。",
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
                "指定された銘柄コードの現在価格情報を取得できませんでした。市場が開いているか、インターネット接続をご確認ください。",
                title="情報取得エラー",
            )
        )
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
        console.print(create_success_panel(f"設定を更新しました: {key} = {typed_value}"))
    except Exception as e:
        console.print(
            create_error_panel(
                f"設定項目 '{key}' の更新中にエラーが発生しました。入力値が正しいかご確認ください。詳細: {e}",
                title="設定更新エラー",
            )
        )


@config.command("reset")
@click.confirmation_option(prompt="設定をリセットしますか？")
def config_reset():
    """設定をデフォルトに戻す"""
    try:
        config_manager.reset()
        console.print(create_success_panel("設定をデフォルトにリセットしました。"))
    except Exception as e:
        console.print(
            create_error_panel(
                f"設定のリセット中に予期せぬエラーが発生しました。詳細: {e}",
                title="設定リセットエラー",
            )
        )


@cli.command("validate")
@click.argument("codes", nargs=-1, required=True)
def validate_codes(codes):
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


@cli.command()
@click.option("--symbols", "-n", default=5, help="最大選択銘柄数", type=int)
@click.option(
    "--depth",
    "-d",
    default="balanced",
    type=click.Choice(["fast", "balanced", "comprehensive"]),
    help="最適化の深さ",
)
@click.option("--no-progress", is_flag=True, help="進捗表示を無効化")
def auto(symbols, depth, no_progress):
    """全自動最善選択を実行"""
    try:
        from ..automation.auto_optimizer import AutoOptimizer

        console.print("[bold green]🚀 全自動最善選択を開始します...[/bold green]")

        # 全自動最適化実行
        optimizer = AutoOptimizer()
        result = optimizer.run_auto_optimization(
            max_symbols=symbols, optimization_depth=depth, show_progress=not no_progress
        )

        # 成功メッセージ
        console.print("\n[bold green]✅ 最適化完了![/bold green]")
        console.print(
            f"選択銘柄: {', '.join(result.best_symbols[:3])}{'...' if len(result.best_symbols) > 3 else ''}"
        )
        console.print(f"期待リターン: {result.expected_return:.2%}")
        console.print(f"信頼度: {result.confidence:.1%}")

        # 次のアクション提案
        console.print("\n[bold cyan]🎯 推奨アクション:[/bold cyan]")
        console.print("1. 選択された銘柄の詳細を確認: daytrade stock <銘柄コード>")
        console.print("2. ウォッチリストに追加: daytrade watchlist add <銘柄コード>")
        console.print("3. 履歴データを確認: daytrade history <銘柄コード>")

    except ImportError as e:
        console.print(
            create_error_panel(
                f"全自動最適化機能の初期化に失敗しました。必要なモジュールが見つかりません: {e}",
                title="機能エラー",
            )
        )
    except Exception as e:
        console.print(
            create_error_panel(
                f"全自動最適化中にエラーが発生しました: {e}\n詳細なログを確認してください。",
                title="実行エラー",
            )
        )


@cli.command()
@click.argument("symbols", nargs=-1)
@click.option("--config", "-c", type=click.Path(), help="設定ファイルのパス")
@click.option("--report-only", is_flag=True, help="レポート生成のみ")
def run(symbols, config, report_only):
    """フル自動化処理を実行（従来機能）"""
    try:
        from ..automation.orchestrator import DayTradeOrchestrator

        if config:
            orchestrator = DayTradeOrchestrator(config)
        else:
            orchestrator = DayTradeOrchestrator()

        symbol_list = list(symbols) if symbols else None

        console.print("[bold green]🔄 デイトレード自動化処理を開始...[/bold green]")

        with console.status("[bold green]処理中..."):
            report = orchestrator.run_full_automation(
                symbols=symbol_list, report_only=report_only, show_progress=True
            )

        # 結果表示
        console.print("\n[bold green]✅ 処理完了![/bold green]")
        console.print(f"対象銘柄: {report.total_symbols}")
        console.print(f"成功: {report.successful_symbols}")
        console.print(f"失敗: {report.failed_symbols}")
        console.print(f"生成シグナル: {len(report.generated_signals)}")
        console.print(f"トリガーアラート: {len(report.triggered_alerts)}")

        if report.errors:
            console.print(f"\n[red]エラー ({len(report.errors)}件):[/red]")
            for error in report.errors[:3]:  # 最初の3件のみ表示
                console.print(f"  • {error}")
            if len(report.errors) > 3:
                console.print(f"  ... 他{len(report.errors) - 3}件")

    except ImportError as e:
        console.print(
            create_error_panel(
                f"自動化機能の初期化に失敗しました: {e}",
                title="機能エラー",
            )
        )
    except Exception as e:
        console.print(
            create_error_panel(
                f"自動化処理中にエラーが発生しました: {e}",
                title="実行エラー",
            )
        )


cli.add_command(watchlist)


def main():
    """メインエントリーポイント"""
    cli()


if __name__ == "__main__":
    main()
