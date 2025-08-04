"""
インタラクティブモードのデモンストレーション
Rich TUIダッシュボードの機能紹介
"""

import time

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Windows環境対応
try:
    from src.day_trade.utils.windows_console_fix import create_safe_live_context
    safe_live_context = create_safe_live_context()
except ImportError:
    from contextlib import contextmanager

    from rich.live import Live

    @contextmanager
    def safe_live_context(*args, **kwargs):
        with Live(*args, **kwargs) as live:
            yield live


def demo_rich_interface():
    """Rich TUIデモ"""
    console = Console()

    console.print("[bold green]Day Trade Interactive Mode Demo[/bold green]\n")

    # レイアウトデモ
    console.print("=== レイアウトデモ ===")

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )

    layout["main"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )

    # ヘッダー
    header_text = Text()
    header_text.append("Day Trade Dashboard", style="bold blue")
    header_text.append(" | 最終更新: 2025-08-01 15:30:45", style="dim")
    header_text.append(" | 表示: dashboard", style="yellow")

    layout["header"].update(
        Panel(Align.center(header_text), box=box.ROUNDED, style="blue")
    )

    # メイン左側 - 市場サマリー
    market_table = Table(title="📈 市場サマリー", box=box.ROUNDED)
    market_table.add_column("項目", style="cyan")
    market_table.add_column("値", justify="right")
    market_table.add_column("変化", justify="right")

    market_data = [
        ("日経平均", "33,500", "[green]+250 (+0.75%)[/green]"),
        ("TOPIX", "2,350", "[green]+15 (+0.64%)[/green]"),
        ("JPY/USD", "150.25", "[red]-0.35 (-0.23%)[/red]"),
        ("VIX", "18.5", "[red]+1.2 (+6.95%)[/red]"),
    ]

    for item, value, change in market_data:
        market_table.add_row(item, value, change)

    layout["left"].update(Panel(market_table, border_style="blue"))

    # 右側 - ポートフォリオ
    portfolio_table = Table(title="💼 ポートフォリオ", box=box.ROUNDED)
    portfolio_table.add_column("項目", style="cyan")
    portfolio_table.add_column("値", justify="right")

    portfolio_data = [
        ("保有銘柄数", "5銘柄"),
        ("総取引数", "15件"),
        ("総コスト", "1,250,000円"),
        ("時価総額", "1,328,750円"),
        ("評価損益", "[green]+78,750円 (+6.30%)[/green]"),
    ]

    for item, value in portfolio_data:
        portfolio_table.add_row(item, value)

    layout["right"].update(Panel(portfolio_table, border_style="magenta"))

    # フッター
    footer_text = Text()
    footer_text.append("操作: ", style="bold")
    footer_text.append(
        "[1]Dashboard [2]Watchlist [3]Portfolio [4]Alerts ", style="cyan"
    )
    footer_text.append("[R]更新 [H]ヘルプ [Q]終了", style="yellow")

    layout["footer"].update(
        Panel(Align.center(footer_text), box=box.ROUNDED, style="green")
    )

    # 静的表示
    console.print(layout)
    console.print("\n" + "=" * 60)

    # 動的更新デモ
    console.print("\n[bold yellow]動的更新デモ（5秒間）[/bold yellow]")
    console.print("実際のインタラクティブモードではリアルタイムで更新されます")

    # カウンターデモ
    with safe_live_context(console=console, refresh_per_second=2) as live:
        for i in range(10):
            # 時刻更新デモ
            current_time = time.strftime("%H:%M:%S")

            demo_text = Text()
            demo_text.append(
                f"リアルタイム更新デモ: {current_time}\n", style="bold green"
            )
            demo_text.append(f"更新回数: {i + 1}/10\n", style="cyan")
            demo_text.append("データ取得中", style="yellow")

            if i % 3 == 0:
                demo_text.append(" 📊", style="blue")
            elif i % 3 == 1:
                demo_text.append(" 📈", style="green")
            else:
                demo_text.append(" 💹", style="red")

            if live:  # Liveが有効な場合のみ更新
                live.update(
                    Panel(
                        Align.center(demo_text),
                    title="Live Update Demo",
                    border_style="yellow",
                )
            )

            time.sleep(0.5)

    console.print("\n[bold green]デモ完了！[/bold green]")


def demo_interactive_features():
    """インタラクティブ機能の紹介"""
    console = Console()

    console.print("\n=== インタラクティブモード機能紹介 ===\n")

    features = [
        {
            "title": "🎯 ダッシュボード表示",
            "description": "市場サマリー、ウォッチリスト概要、ポートフォリオ情報を一画面で表示",
            "key": "[1]キー",
        },
        {
            "title": "📋 ウォッチリスト表示",
            "description": "詳細なウォッチリスト、グループ情報、選択銘柄の詳細を表示",
            "key": "[2]キー",
        },
        {
            "title": "💼 ポートフォリオ表示",
            "description": "ポートフォリオメトリクス、保有銘柄、パフォーマンス情報を表示",
            "key": "[3]キー",
        },
        {
            "title": "🚨 アラート表示",
            "description": "アクティブアラート、履歴、統計情報を表示",
            "key": "[4]キー",
        },
        {
            "title": "🔄 リアルタイム更新",
            "description": "30秒間隔での自動データ更新（手動更新も可能）",
            "key": "[R]キー",
        },
        {
            "title": "❓ ヘルプ表示",
            "description": "操作方法とキーバインディングの確認",
            "key": "[H]キー",
        },
    ]

    for feature in features:
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Info", width=50)

        table.add_row(f"[bold cyan]{feature['title']}[/bold cyan]")
        table.add_row(f"操作: [yellow]{feature['key']}[/yellow]")
        table.add_row(f"機能: {feature['description']}")

        console.print(Panel(table, border_style="blue"))
        time.sleep(0.3)


def demo_layout_examples():
    """レイアウト例の紹介"""
    console = Console()

    console.print("\n=== 各表示モードのレイアウト例 ===\n")

    # ウォッチリスト表示例
    console.print("[bold]2. ウォッチリスト表示モード[/bold]")

    watchlist_table = Table(title="📋 詳細ウォッチリスト", box=box.ROUNDED, width=60)
    watchlist_table.add_column("コード", width=8)
    watchlist_table.add_column("銘柄名", width=20)
    watchlist_table.add_column("グループ", width=12)
    watchlist_table.add_column("メモ", width=15)

    watchlist_data = [
        ("7203", "トヨタ自動車", "自動車", "主力株"),
        ("8306", "三菱UFJ銀行", "金融", "配当重視"),
        ("9984", "ソフトバンクG", "テック", "成長株"),
        ("6758", "ソニーグループ", "テック", "エンタメ"),
        ("4689", "Zホールディングス", "テック", "IT関連"),
    ]

    for code, name, group, memo in watchlist_data:
        watchlist_table.add_row(code, name, group, memo)

    console.print(Panel(watchlist_table, border_style="cyan"))

    time.sleep(1)

    # アラート表示例
    console.print("\n[bold]4. アラート表示モード[/bold]")

    alert_table = Table(title="🚨 アクティブアラート", box=box.ROUNDED, width=50)
    alert_table.add_column("銘柄", width=8)
    alert_table.add_column("タイプ", width=15)
    alert_table.add_column("閾値", justify="right", width=10)
    alert_table.add_column("状態", width=8)

    alert_data = [
        ("7203", "price above", "3000", "[green]ON[/green]"),
        ("8306", "price below", "700", "[green]ON[/green]"),
        ("9984", "change percent up", "5.0", "[dim]OFF[/dim]"),
        ("6758", "volume spike", "5000000", "[green]ON[/green]"),
    ]

    for code, type_name, threshold, status in alert_data:
        alert_table.add_row(code, type_name, threshold, status)

    console.print(Panel(alert_table, border_style="red"))


def main():
    """メイン関数"""
    console = Console()

    try:
        console.print(
            Panel(
                "[bold blue]Day Trade Interactive Mode Demo[/bold blue]\n"
                "Rich TUIフレームワークを使用したダッシュボードのデモンストレーション",
                title="デモ開始",
                border_style="green",
            )
        )

        time.sleep(1)

        # 機能紹介
        demo_interactive_features()

        time.sleep(1)

        # レイアウトデモ
        demo_rich_interface()

        time.sleep(1)

        # レイアウト例
        demo_layout_examples()

        console.print(
            Panel(
                "[bold green]デモ完了！[/bold green]\n\n"
                "[cyan]実際のインタラクティブモードの起動方法:[/cyan]\n"
                "[yellow]```python[/yellow]\n"
                "[white]from src.day_trade.cli.interactive import InteractiveMode[/white]\n"
                "[white]interactive = InteractiveMode()[/white]\n"
                "[white]interactive.run()[/white]\n"
                "[yellow]```[/yellow]\n\n"
                "[dim]注意: 実際の使用には適切なデータベース設定が必要です[/dim]",
                title="使用方法",
                border_style="blue",
            )
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]デモを中断しました[/yellow]")
    except Exception as e:
        console.print(f"\n[red]エラー: {e}[/red]")


if __name__ == "__main__":
    main()
