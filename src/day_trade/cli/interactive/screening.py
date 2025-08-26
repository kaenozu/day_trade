"""
スクリーニングとバックテスト機能のCLIコマンド群
高度な分析機能と拡張インタラクティブモードを提供
"""

import logging
import random
from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import Optional

import click
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from ...analysis.backtest import BacktestConfig, BacktestEngine
from ...data.stock_fetcher import StockFetcher
from ...utils.formatters import (
    create_ascii_chart,
    create_error_panel,
    create_info_panel,
    create_warning_panel,
    format_currency,
)

logger = logging.getLogger(__name__)
console = Console()

# 拡張インタラクティブ機能のインポート
try:
    from ..enhanced_interactive import run_enhanced_interactive
    ENHANCED_MODE_AVAILABLE = True
except ImportError as e:
    ENHANCED_MODE_AVAILABLE = False
    logger.warning(
        f"拡張インタラクティブモードは利用できません"
        f"（prompt_toolkitが必要）: {e}"
    )


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
        """バックテストの進捗レイアウトを作成"""
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
        ) as live:
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


@click.command("backtest")
def backtest_command():
    """インタラクティブバックテストの実行"""
    run_interactive_backtest()


@click.command("enhanced")
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


@click.command("interactive")
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


@click.command("screen")
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
    "--max-results",
    "-n",
    default=20,
    type=int,
    help="最大結果数 (デフォルト: 20)",
)
@click.option("--symbols", help="対象銘柄をカンマ区切りで指定")
@click.pass_context
def screen_stocks(
    ctx,
    type: str,
    min_score: float,
    max_results: int,
    symbols: Optional[str],
):
    """銘柄スクリーニングを実行"""
    try:
        from ...automation.orchestrator import DayTradeOrchestrator

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
                (
                    f"¥{result['last_price']:,.0f}"
                    if result["last_price"]
                    else "N/A"
                ),
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
            f"\n[bold green]✅ {len(results)}銘柄が"
            "スクリーニング条件を満たしました[/bold green]"
        )

        # 上位3銘柄の詳細表示
        if len(results) >= 3:
            console.print("\n[bold]🏆 トップ3銘柄の詳細:[/bold]")
            for i, result in enumerate(results[:3], 1):
                tech_data = result.get("technical_data", {})
                console.print(
                    f"{i}. {result['symbol']} "
                    f"(スコア: {result['score']:.2f})"
                )
                if "price_position" in tech_data:
                    console.print(
                        f"   52週レンジでの位置: "
                        f"{tech_data['price_position']:.1f}%"
                    )
                if "volume_avg_20d" in tech_data:
                    console.print(
                        f"   20日平均出来高: "
                        f"{tech_data['volume_avg_20d']:,}"
                    )

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