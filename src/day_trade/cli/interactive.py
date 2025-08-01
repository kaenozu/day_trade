"""
対話型CLIのメインスクリプト
拡張されたインタラクティブ機能とオートコンプリート対応
"""

import logging
import random
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

# 拡張インタラクティブ機能のインポート
try:
    from .enhanced_interactive import run_enhanced_interactive

    ENHANCED_MODE_AVAILABLE = True
except ImportError as e:
    ENHANCED_MODE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        f"拡張インタラクティブモードは利用できません（prompt_toolkitが必要）: {e}"
    )

import click
import pandas as pd
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

from ..analysis.backtest import BacktestConfig, BacktestEngine
from ..core.config import config_manager
from ..core.portfolio import PortfolioManager
from ..core.watchlist import WatchlistManager
from ..data.stock_fetcher import DataNotFoundError, InvalidSymbolError, StockFetcher
from ..models.database import db_manager, init_db
from ..utils.formatters import (
    create_ascii_chart,
    create_company_info_table,
    create_error_panel,
    create_historical_data_table,
    create_info_panel,
    create_stock_info_table,
    create_success_panel,
    create_warning_panel,
    create_watchlist_table,
    format_currency,
    format_percentage,
)
from ..utils.validators import (
    normalize_stock_codes,
    suggest_stock_code_correction,
    validate_stock_code,
)

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
@click.option("--config", "-c", type=click.Path(), help="設定ファイルのパス")
@click.pass_context
def cli(ctx, config):
    """対話型デイトレード支援ツール"""
    ctx.ensure_object(dict)
    if config:
        ctx.obj["config_path"] = Path(config)
    else:
        ctx.obj["config_path"] = None


# ==============================================================
#                           ヘルパー関数
# ==============================================================


def _get_watchlist_manager(config_path: Optional[Path] = None) -> WatchlistManager:
    """ウォッチリストマネージャーのインスタンスを取得"""
    # CLIコンテキストからconfig_pathを取得（もしあれば）
    # if click.get_current_context():
    #     config_path = click.get_current_context().obj.get("config_path")

    # config_managerはシングルトンなので、init時にconfig_pathを渡す
    _config_manager = config_manager.__class__(config_path)

    # データベースマネージャーはConfigManager内で管理されるべき
    # しかし、現在の実装ではグローバルなdb_managerを使用しているため、ここでの処理は不要
    # db_manager.initialize(config_manager.get_database_url())
    # db_manager.create_tables()

    return WatchlistManager(
        _config_manager,
        db_manager,
        stock_fetcher=StockFetcher(),
        portfolio_manager=PortfolioManager(),
    )


def _display_stock_details(code: str, stock_data: Dict[str, Any], show_details: bool):
    """銘柄詳細を表示"""
    if not stock_data:
        console.print(
            create_error_panel(
                f"銘柄コード '{code}' の現在価格または詳細情報を取得できませんでした。コードが正しいか、または市場が開いているかご確認ください。",
                title="情報取得エラー",
            )
        )
        return

    table = create_stock_info_table(stock_data)
    console.print(table)

    if show_details:
        fetcher = StockFetcher()
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


def _display_historical_data(
    code: str, df: pd.DataFrame, period: str, interval: str, rows: int
):
    """ヒストリカルデータを表示"""
    if df is None or df.empty:
        console.print(
            create_error_panel(
                "ヒストリカルデータを取得できませんでした。指定された銘柄コード、期間、または間隔が正しいかご確認ください。"
            )
        )
        return

    table = create_historical_data_table(df, code, period, interval, max_rows=rows)
    console.print(table)
    console.print("\n[bold]サマリー:[/bold]")
    console.print(f"期間高値: ¥{df['High'].max():,.0f}")
    console.print(f"期間安値: ¥{df['Low'].min():,.0f}")
    console.print(f"平均出来高: {int(df['Volume'].mean()):,}")


def run_interactive_backtest():
    """インタラクティブバックテストを実行"""
    console.print(
        Rule("[bold green]インタラクティブバックテスト[/bold green]", style="green")
    )
    console.print(
        "[yellow]リアルタイムでバックテストの進行状況を表示します...[/yellow]"
    )
    console.print("[dim]Ctrl+C で終了[/dim]\n")

    # モックデータフェッチャーを使用
    mock_fetcher = StockFetcher()
    _engine = BacktestEngine(stock_fetcher=mock_fetcher)

    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 3, 31),  # 短期間
        initial_capital=Decimal("1000000"),
    )

    _symbols = ["7203", "9984", "8306"]

    def create_progress_layout(current_date, portfolio_value, trades_count):
        layout = Layout()
        progress_info = Panel(
            f"[cyan]現在日付:[/cyan] {current_date.strftime('%Y-%m-%d')}\n"
            f"[green]ポートフォリオ価値:[/green] {format_currency(int(portfolio_value))}\n"
            f"[yellow]取引回数:[/yellow] {trades_count}",
            title="📊 バックテスト進捗",
            border_style="blue",
        )
        chart_data = [float(portfolio_value)] * 20  # プレースホルダー
        mini_chart = create_ascii_chart(
            chart_data, width=40, height=6, title="ポートフォリオ推移"
        )
        layout.split_column(
            Layout(progress_info, size=6),
            Layout(Panel(mini_chart, border_style="green"), size=10),
        )
        return layout

    try:
        with Live(
            create_progress_layout(config.start_date, config.initial_capital, 0),
            refresh_per_second=4,
            screen=False,
        ) as live:  # noqa: F841
            # 短いデモバックテスト
            for day in range(30):
                current_date = config.start_date + timedelta(days=day)
                current_value = int(
                    config.initial_capital * (1 + random.gauss(0.1, 0.2))
                )
                trades_count = random.randint(0, day + 1)

                live.update(
                    create_progress_layout(current_date, current_value, trades_count)
                )
                time.sleep(0.3)

        console.print("\n[green]インタラクティブデモが完了しました！[/green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]デモを中断しました。[/yellow]")


# ==============================================================
#                           CLI コマンド
# ==============================================================


@cli.command()
def init():
    """データベースを初期化"""
    try:
        init_db()
        console.print(create_success_panel("データベースを初期化しました。"))
    except Exception as e:  # noqa: E722
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

    _display_stock_details(code, current, details)


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
        try:
            df = fetcher.get_historical_data(code, period=period, interval=interval)
            _display_historical_data(code, df, period, interval, rows)
        except (DataNotFoundError, InvalidSymbolError) as e:
            console.print(
                create_error_panel(
                    f"銘柄コード '{code}' のヒストリカルデータの取得中にエラーが発生しました。インターネット接続を確認するか、銘柄コードが正しいことを再確認してください。詳細: {e}",
                    title="データ取得エラー",
                )
            )
        except Exception as e:  # noqa: E722
            console.print(
                create_error_panel(f"予期しないエラー: {e}", title="予期せぬエラー")
            )


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
def watchlist():
    """ウォッチリスト管理"""
    pass


@watchlist.command()
@click.argument("codes", nargs=-1, required=True)
@click.option("--group", "-g", default="default", help="グループ名")
@click.option("--priority", "-p", default="medium", help="優先度 (low, medium, high)")
def add(codes: List[str], group: str, priority: str):
    """ウォッチリストに銘柄を追加"""
    manager = _get_watchlist_manager()
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
                    f"銘柄コード '{code}' は無効です。詳細: {e}\n正しい銘柄コードを入力してください。",
                    title="無効な銘柄コード",
                )
            )
        except Exception as e:  # noqa: E722
            console.print(
                create_error_panel(
                    f"銘柄コード '{code}' をウォッチリストに追加中に予期せぬエラーが発生しました。詳細: {e}",
                    title="追加エラー",
                )
            )

    if added_count > 0:
        console.print(create_success_panel(f"{added_count} 件の銘柄を追加しました。"))


@watchlist.command()
@click.argument("codes", nargs=-1, required=True)
def remove(codes: List[str]):
    """ウォッチリストから銘柄を削除"""
    manager = _get_watchlist_manager()
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
        except Exception as e:  # noqa: E722
            console.print(
                create_error_panel(
                    f"銘柄コード '{code}' をウォッチリストから削除中に予期せぬエラーが発生しました。詳細: {e}",
                    title="削除エラー",
                )
            )

    if removed_count > 0:
        console.print(create_success_panel(f"{removed_count} 件の銘柄を削除しました。"))


@watchlist.command()
def list():
    """ウォッチリストの内容を表示"""
    manager = _get_watchlist_manager()
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
        table.add_row(
            item.get("stock_code", "N/A"),
            item.get("stock_name", "N/A"),
            item.get("group", "N/A"),
            item.get("priority", "N/A"),
            format_currency(item.get("current_price")),
            f"[{change_color}]{format_percentage(item.get('change_percent', 0))}[/{change_color}]",
            (
                item.get("memo", "")[:20] + "..."
                if len(item.get("memo", "")) > 20
                else item.get("memo", "")
            ),
        )
    console.print(table)


@watchlist.command()
@click.argument("code")
@click.option("--memo", "-m", help="メモの内容")
def memo(code: str, memo: Optional[str]):
    """ウォッチリスト銘柄にメモを追加・更新"""
    manager = _get_watchlist_manager()
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
        # メモが指定されなければ対話的に入力
        current_memo = (
            manager.get_watchlist(codes=[code])[0].get("memo", "")
            if manager.get_watchlist(codes=[code])
            else ""
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
    except Exception as e:  # noqa: E722
        console.print(
            create_error_panel(
                f"銘柄コード '{code}' のメモ更新中に予期せぬエラーが発生しました。詳細: {e}",
                title="メモ更新エラー",
            )
        )


@watchlist.command()
@click.argument("code")
@click.argument("group")
def move(code: str, group: str):
    """ウォッチリスト銘柄を別のグループに移動"""
    manager = _get_watchlist_manager()
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
    except Exception as e:  # noqa: E722
        console.print(
            create_error_panel(
                f"銘柄コード '{code}' のグループ移動中に予期せぬエラーが発生しました。詳細: {e}",
                title="グループ移動エラー",
            )
        )


@watchlist.command()
@click.confirmation_option(prompt="本当にウォッチリストをクリアしますか？")
def clear():
    """ウォッチリストの内容を全てクリア"""
    manager = _get_watchlist_manager()
    try:
        manager.clear_watchlist()
        console.print(create_success_panel("ウォッチリストを全てクリアしました。"))
    except Exception as e:  # noqa: E722
        console.print(
            create_error_panel(
                f"ウォッチリストのクリア中に予期せぬエラーが発生しました。詳細: {e}",
                title="クリアエラー",
            )
        )


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
    except Exception as e:  # noqa: E722
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
    except Exception as e:  # noqa: E722
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


@cli.command("backtest")
def backtest_command():
    """インタラクティブバックテストの実行"""
    run_interactive_backtest()


@cli.command("enhanced")
@click.pass_context
def enhanced_mode(ctx):
    """拡張インタラクティブモードを開始（オートコンプリート、履歴機能など）"""
    if not ENHANCED_MODE_AVAILABLE:
        console.print(
            create_error_panel(
                "拡張インタラクティブモードは利用できません。\n"
                "prompt_toolkit>=3.0.0 をインストールしてください。\n"
                "コマンド: pip install prompt_toolkit>=3.0.0",
                title="拡張機能エラー",
            )
        )
        return

    config_path = ctx.obj.get("config_path") if ctx.obj else None
    console.print(
        create_info_panel(
            "拡張インタラクティブモードを開始します...\n"
            "• オートコンプリート機能\n"
            "• コマンド履歴\n"
            "• 色分け表示\n"
            "• カスタムキーバインディング"
        )
    )

    try:
        run_enhanced_interactive(config_path)
    except Exception as e:
        console.print(
            create_error_panel(f"拡張モードの実行中にエラーが発生しました: {e}")
        )
        logger.error(f"Enhanced interactive mode error: {e}")


@cli.command("interactive")
@click.option("--enhanced", "-e", is_flag=True, help="拡張インタラクティブモードを使用")
@click.pass_context
def interactive_mode(ctx, enhanced: bool):
    """インタラクティブモードを開始"""
    if enhanced:
        # 拡張モードを呼び出し
        ctx.invoke(enhanced_mode)
    else:
        # 既存の基本モード
        console.print(
            create_info_panel(
                "基本インタラクティブモード\n"
                "拡張機能を使用するには --enhanced オプションを指定してください。"
            )
        )
        console.print("[dim]対話的なコマンド実行機能は開発中です...[/dim]")


@cli.command("screen")
@click.option(
    "--type",
    "-t",
    default="default",
    type=click.Choice(["default", "growth", "value", "momentum"]),
    help="スクリーナータイプを指定",
)
@click.option(
    "--min-score",
    "-s",
    default=0.1,
    type=float,
    help="最小スコア閾値 (デフォルト: 0.1)",
)
@click.option(
    "--max-results", "-n", default=20, type=int, help="最大結果数 (デフォルト: 20)"
)
@click.option("--symbols", help="対象銘柄をカンマ区切りで指定")
@click.pass_context
def screen_stocks(
    ctx, type: str, min_score: float, max_results: int, symbols: Optional[str]
):
    """銘柄スクリーニングを実行"""
    try:
        from ..automation.orchestrator import DayTradeOrchestrator

        config_path = ctx.obj.get("config_path") if ctx.obj else None
        orchestrator = DayTradeOrchestrator(config_path)

        # 銘柄リストの処理
        symbol_list = None
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(",")]
            console.print(f"[cyan]対象銘柄: {len(symbol_list)}銘柄[/cyan]")

        # スクリーニング実行
        with console.status(
            f"[bold green]{type}スクリーナーで銘柄をスクリーニング中..."
        ):
            results = orchestrator.run_stock_screening(
                symbols=symbol_list,
                screener_type=type,
                min_score=min_score,
                max_results=max_results,
            )

        if not results:
            console.print(
                create_warning_panel(
                    "条件を満たす銘柄が見つかりませんでした。\n"
                    "スコア閾値を下げるか、別のスクリーナータイプを試してください。"
                )
            )
            return

        # 結果をテーブル形式で表示
        table = Table(title=f"🔍 {type.title()}スクリーニング結果")
        table.add_column("順位", style="dim", width=4)
        table.add_column("銘柄コード", style="cyan", justify="center")
        table.add_column("スコア", style="green", justify="right")
        table.add_column("現在価格", style="white", justify="right")
        table.add_column("1日変化率", style="white", justify="right")
        table.add_column("RSI", style="yellow", justify="right")
        table.add_column("マッチ条件", style="magenta", justify="left")

        for i, result in enumerate(results, 1):
            # 価格変化率の色分け
            change_1d = result.get("technical_data", {}).get("price_change_1d", 0)
            change_color = "red" if change_1d < 0 else "green"
            change_text = f"[{change_color}]{change_1d:+.2f}%[/{change_color}]"

            # RSI値
            rsi = result.get("technical_data", {}).get("rsi")
            rsi_text = f"{rsi:.1f}" if rsi else "N/A"

            # マッチした条件（最初の3個まで表示）
            conditions = result.get("matched_conditions", [])
            conditions_text = ", ".join(conditions[:3])
            if len(conditions) > 3:
                conditions_text += f" (+{len(conditions) - 3})"

            table.add_row(
                str(i),
                result["symbol"],
                f"{result['score']:.2f}",
                f"¥{result['last_price']:,.0f}" if result["last_price"] else "N/A",
                change_text,
                rsi_text,
                (
                    conditions_text[:40] + "..."
                    if len(conditions_text) > 40
                    else conditions_text
                ),
            )

        console.print(table)

        # サマリー情報
        console.print(
            f"\n[bold green]✅ {len(results)}銘柄がスクリーニング条件を満たしました[/bold green]"
        )

        # 上位3銘柄の詳細表示
        if len(results) >= 3:
            console.print("\n[bold]🏆 トップ3銘柄の詳細:[/bold]")
            for i, result in enumerate(results[:3], 1):
                tech_data = result.get("technical_data", {})
                console.print(
                    f"{i}. {result['symbol']} (スコア: {result['score']:.2f})"
                )
                if "price_position" in tech_data:
                    console.print(
                        f"   52週レンジでの位置: {tech_data['price_position']:.1f}%"
                    )
                if "volume_avg_20d" in tech_data:
                    console.print(f"   20日平均出来高: {tech_data['volume_avg_20d']:,}")

    except ImportError:
        console.print(
            create_error_panel(
                "スクリーニング機能が利用できません。\n"
                "必要なモジュールがインストールされているか確認してください。"
            )
        )
    except Exception as e:
        console.print(create_error_panel(f"スクリーニング実行エラー: {e}"))
        logger.error(f"Screening command error: {e}")


if __name__ == "__main__":
    cli()
