#!/usr/bin/env python3
"""
高度な表示フォーマッタのデモンストレーション
Issue #68の実装内容を示すための実行可能なデモ
"""

import random
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from src.day_trade.utils.formatters import (
    # 高度なASCII表示機能
    create_ascii_chart,
    create_comparison_table,
    create_distribution_chart,
    create_heatmap,
    create_metric_cards,
    create_progress_bar_panel,
    create_sparkline,
    create_status_indicator,
    create_stock_info_table,
    create_trend_indicator,
    create_watchlist_table,
    # 基本フォーマッタ
    format_currency,
    format_large_number,
    format_percentage,
    format_volume,
)

console = Console()


def demo_basic_formatters():
    """基本フォーマッタのデモ"""
    console.print(Rule("[bold blue]基本フォーマッタ機能", style="blue"))

    # 通貨フォーマット例
    console.print(
        Panel(
            f"[green]大型株価格:[/green] {format_currency(2_847_500)}\n"
            f"[red]小型株価格:[/red] {format_currency(125.75, decimal_places=2)}\n"
            f"[blue]時価総額:[/blue] {format_currency(25_000_000_000_000)}\n"
            f"[yellow]ドル建て:[/yellow] {format_currency(150.25, currency='$', decimal_places=2)}",
            title="💰 通貨フォーマット例",
            border_style="green",
        )
    )

    # パーセンテージフォーマット例
    console.print(
        Panel(
            f"[green]上昇率:[/green] {format_percentage(12.5)}\n"
            f"[red]下落率:[/red] {format_percentage(-8.2)}\n"
            f"[white]変化なし:[/white] {format_percentage(0)}\n"
            f"[cyan]高精度:[/cyan] {format_percentage(3.14159, decimal_places=4)}",
            title="📊 パーセンテージフォーマット例",
            border_style="cyan",
        )
    )

    # 出来高フォーマット例
    console.print(
        Panel(
            f"[bright_blue]大型株:[/bright_blue] {format_volume(15_234_567_890)}\n"
            f"[blue]中型株:[/blue] {format_volume(2_345_678)}\n"
            f"[dim_blue]小型株:[/dim_blue] {format_volume(12_345)}\n"
            f"[white]少量:[/white] {format_volume(567)}",
            title="📈 出来高フォーマット例",
            border_style="blue",
        )
    )


def demo_advanced_numbers():
    """大きな数値フォーマットのデモ"""
    console.print(Rule("[bold cyan]大きな数値フォーマット機能", style="cyan"))

    numbers = [
        ("国内総生産", 540_000_000_000_000),
        ("大企業時価総額", 12_500_000_000_000),
        ("企業売上", 850_000_000_000),
        ("個人資産", 125_000_000),
        ("年収", 8_500_000),
        ("株価", 2_847),
    ]

    content = "\n".join(
        [
            f"[white]{label}:[/white] [green]{format_large_number(value)}[/green]"
            for label, value in numbers
        ]
    )

    console.print(
        Panel(content, title="💹 大きな数値の自動フォーマット", border_style="cyan")
    )


def demo_ascii_charts():
    """ASCIIチャート機能のデモ"""
    console.print(Rule("[bold yellow]ASCIIチャート描画機能", style="yellow"))

    # サンプル株価データ生成
    base_price = 2500
    price_data = [base_price]
    for _ in range(30):
        change = random.randint(-50, 80)
        new_price = max(price_data[-1] + change, 1000)  # 最低1000円
        price_data.append(new_price)

    # ASCIIチャート作成
    chart = create_ascii_chart(
        price_data, width=50, height=12, title="🏪 30日間の株価推移チャート"
    )

    console.print(
        Panel(chart, title="📊 フルサイズASCIIチャート", border_style="yellow")
    )

    # スパークライン例
    sparklines_content = []
    stocks = ["トヨタ(7203)", "ソフトバンク(9984)", "三菱UFJ(8306)", "ソニー(6758)"]

    for stock in stocks:
        # ランダムな価格推移データ
        trend_data = [random.randint(80, 120) for _ in range(20)]
        sparkline = create_sparkline(trend_data, width=25)

        # トレンド計算
        trend_pct = ((trend_data[-1] - trend_data[0]) / trend_data[0]) * 100
        color = "green" if trend_pct > 0 else "red"

        sparklines_content.append(
            f"[white]{stock:12}[/white] [{color}]{sparkline}[/{color}] "
            f"[{color}]{trend_pct:+5.1f}%[/{color}]"
        )

    console.print(
        Panel(
            "\n".join(sparklines_content),
            title="⚡ スパークライン表示",
            border_style="magenta",
        )
    )


def demo_heatmap():
    """ヒートマップのデモ"""
    console.print(Rule("[bold magenta]ヒートマップ表示機能", style="magenta"))

    # セクター別パフォーマンスデータ
    sectors = ["Tech", "Auto", "Bank", "Retail", "Pharma"]
    periods = ["1D", "1W", "1M", "3M", "1Y"]

    # ランダムなパフォーマンスデータ生成（-10%から+15%）
    performance_data = []
    for _ in sectors:
        row = [random.uniform(-10, 15) for _ in periods]
        performance_data.append(row)

    heatmap = create_heatmap(
        performance_data,
        periods,
        sectors,
        title="📊 セクター別パフォーマンス ヒートマップ",
    )

    console.print(
        Panel(heatmap, title="🔥 セクター別パフォーマンス分析", border_style="red")
    )


def demo_progress_and_status():
    """プログレスバーとステータス表示のデモ"""
    console.print(Rule("[bold green]プログレス＆ステータス表示", style="green"))

    # 複数のプログレスバー
    progress_panels = []
    tasks = [
        ("データ取得", 85, 100),
        ("テクニカル分析", 67, 100),
        ("シグナル生成", 23, 50),
        ("レポート作成", 0, 25),
    ]

    for task_name, current, total in tasks:
        panel = create_progress_bar_panel(
            current, total, title=f"📊 {task_name}", show_percentage=True
        )
        progress_panels.append(panel)

    for panel in progress_panels:
        console.print(panel)

    # ステータス指標例
    statuses = [
        ("データ接続", "active"),
        ("取引システム", "success"),
        ("アラート", "warning"),
        ("バックアップ", "error"),
        ("レポート生成", "pending"),
    ]

    status_content = []
    for label, status in statuses:
        status_text = create_status_indicator(status, label)
        status_content.append(status_text)

    console.print(
        Panel(
            Text.assemble(
                *[Text.assemble(status, "\n") for status in status_content[:-1]],
                status_content[-1],
            ),
            title="🚦 システムステータス",
            border_style="blue",
        )
    )


def demo_comparison_table():
    """比較テーブルのデモ"""
    console.print(Rule("[bold blue]比較テーブル機能", style="blue"))

    # 銘柄比較データ
    stock_comparison = {
        "トヨタ(7203)": {
            "price": 2847,
            "market_cap": 25000000000000,
            "pe_ratio": 12.5,
            "dividend_yield": 2.8,
            "volume": 15234567,
        },
        "ソフトバンク(9984)": {
            "price": 1456,
            "market_cap": 7500000000000,
            "pe_ratio": 15.2,
            "dividend_yield": 5.1,
            "volume": 8765432,
        },
        "三菱UFJ(8306)": {
            "price": 845,
            "market_cap": 11200000000000,
            "pe_ratio": 8.7,
            "dividend_yield": 4.2,
            "volume": 12345678,
        },
    }

    comparison_table = create_comparison_table(
        stock_comparison, title="📊 主要銘柄比較分析"
    )

    console.print(comparison_table)


def demo_metric_cards():
    """メトリクスカード表示のデモ"""
    console.print(Rule("[bold red]メトリクスカード表示", style="red"))

    # ポートフォリオメトリクス
    portfolio_metrics = {
        "総資産": 12500000,
        "実現損益": 234567,
        "含み損益": -45678,
        "勝率": 68.5,
        "シャープレシオ": 1.25,
        "最大DD": -8.2,
    }

    metric_cards = create_metric_cards(portfolio_metrics, columns=3)
    console.print(
        Panel(metric_cards, title="💼 ポートフォリオ メトリクス", border_style="red")
    )


def demo_trend_indicators():
    """トレンド指標のデモ"""
    console.print(Rule("[bold cyan]トレンド指標表示", style="cyan"))

    # 様々なトレンド例
    trends = [
        ("日経平均", 28500, 27800),
        ("TOPIX", 1950, 2020),
        ("ドル円", 148.5, 149.2),
        ("ビットコイン", 4200000, 4500000),
        ("金価格", 9500, 9500),  # 変化なし
    ]

    trend_content = []
    for label, current, previous in trends:
        trend_indicator = create_trend_indicator(current, previous, label)
        trend_content.append(trend_indicator)

    console.print(
        Panel(
            Text.assemble(
                *[Text.assemble(trend, "\n") for trend in trend_content[:-1]],
                trend_content[-1],
            ),
            title="📈 市場トレンド指標",
            border_style="cyan",
        )
    )


def demo_distribution_chart():
    """分布チャートのデモ"""
    console.print(Rule("[bold yellow]分布チャート機能", style="yellow"))

    # 株価リターンの分布データ生成（正規分布近似）
    returns = []
    for _ in range(1000):
        # 正規分布に近いリターンデータ
        ret = random.gauss(0.05, 0.15)  # 平均5%、標準偏差15%
        returns.append(ret * 100)  # パーセント表示

    distribution = create_distribution_chart(
        returns, bins=15, title="📊 日次リターン分布 (過去1000日)"
    )

    console.print(
        Panel(distribution, title="📈 リターン分布分析", border_style="yellow")
    )


def demo_stock_tables():
    """株価情報テーブルのデモ"""
    console.print(Rule("[bold white]株価情報テーブル", style="white"))

    # サンプル株価データ
    stock_data = {
        "symbol": "7203",
        "current_price": 2847,
        "previous_close": 2795,
        "change": 52,
        "change_percent": 1.86,
        "volume": 15234567,
        "high": 2865,
        "low": 2820,
    }

    stock_table = create_stock_info_table(stock_data)
    console.print(stock_table)

    # ウォッチリストデータ
    watchlist_data = {
        "7203": {
            "current_price": 2847,
            "change": 52,
            "change_percent": 1.86,
            "volume": 15234567,
        },
        "9984": {
            "current_price": 1456,
            "change": -23,
            "change_percent": -1.56,
            "volume": 8765432,
        },
        "8306": {
            "current_price": 845,
            "change": 8,
            "change_percent": 0.96,
            "volume": 12345678,
        },
    }

    watchlist_table = create_watchlist_table(watchlist_data)
    console.print(watchlist_table)


def interactive_demo():
    """インタラクティブなリアルタイムデモ"""
    console.print(Rule("[bold green]リアルタイム表示デモ", style="green"))
    console.print(
        "[yellow]リアルタイムで更新されるダッシュボードのデモを開始します...[/yellow]"
    )
    console.print("[dim]Ctrl+C で終了[/dim]\n")

    def create_dashboard():
        """ダッシュボードレイアウト作成"""
        layout = Layout()

        # 上部：メトリクス
        metrics = {
            "日経平均": random.randint(28000, 29000),
            "TOPIX": random.randint(1900, 2000),
            "取引高": random.randint(1000000, 2000000),
            "上昇銘柄": random.randint(800, 1200),
        }
        metric_cards = create_metric_cards(metrics, columns=4)

        # 中央：チャート
        price_data = [random.randint(2800, 2900) for _ in range(30)]
        chart = create_ascii_chart(
            price_data, width=60, height=8, title="📊 リアルタイム価格チャート"
        )

        # 下部：スパークライン
        sparkline_content = []
        for i, stock in enumerate(["NIKKEI", "TOPIX", "USDJPY", "GOLD"]):
            data = [random.randint(80, 120) for _ in range(20)]
            sparkline = create_sparkline(data, width=20)
            color = ["green", "blue", "yellow", "magenta"][i]
            sparkline_content.append(
                f"[white]{stock:6}[/white] [{color}]{sparkline}[/{color}]"
            )

        layout.split_column(
            Layout(
                Panel(metric_cards, title="📊 市場メトリクス", border_style="blue"),
                size=8,
            ),
            Layout(
                Panel(chart, title="📈 チャート表示", border_style="green"), size=12
            ),
            Layout(
                Panel(
                    "\n".join(sparkline_content),
                    title="⚡ 主要指標",
                    border_style="yellow",
                ),
                size=6,
            ),
        )

        return layout

    try:
        with Live(create_dashboard(), refresh_per_second=2, screen=True) as live:
            while True:
                time.sleep(0.5)
                live.update(create_dashboard())
    except KeyboardInterrupt:
        console.print("\n[green]デモを終了しました。[/green]")


def main():
    """メインデモ実行"""
    console.print(
        Panel(
            "[bold cyan]🎯 高度な表示フォーマッタ デモンストレーション[/bold cyan]\n"
            "[white]Issue #68で実装された全ての機能を順次紹介します[/white]",
            title="🚀 デイトレーディングシステム",
            border_style="bright_blue",
        )
    )

    console.print(
        "\n[yellow]各セクションを順番に表示します。Enterキーで次に進んでください...[/yellow]"
    )

    demos = [
        ("基本フォーマッタ", demo_basic_formatters),
        ("大きな数値フォーマット", demo_advanced_numbers),
        ("ASCIIチャート機能", demo_ascii_charts),
        ("ヒートマップ表示", demo_heatmap),
        ("プログレス＆ステータス", demo_progress_and_status),
        ("比較テーブル", demo_comparison_table),
        ("メトリクスカード", demo_metric_cards),
        ("トレンド指標", demo_trend_indicators),
        ("分布チャート", demo_distribution_chart),
        ("株価情報テーブル", demo_stock_tables),
    ]

    try:
        for name, demo_func in demos:
            input(f"\n[dim]Press Enter to show {name}...[/dim]")
            console.clear()
            demo_func()
            console.print(f"\n[green]✅ {name} デモ完了[/green]")

        # 最後にインタラクティブデモを提供
        response = input(
            "\n[yellow]リアルタイムダッシュボードデモを実行しますか？ (y/N): [/yellow]"
        )
        if response.lower() in ["y", "yes"]:
            interactive_demo()

        console.print(
            Panel(
                "[bold green]🎉 全ての高度な表示機能のデモが完了しました！[/bold green]\n"
                "[white]これらの機能により、デイトレーディングシステムで\n"
                "豊富で視覚的な情報表示が可能になりました。[/white]",
                title="🏆 デモ完了",
                border_style="green",
            )
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]デモを中断しました。[/yellow]")
    except Exception as e:
        console.print(f"\n[red]エラーが発生しました: {e}[/red]")


if __name__ == "__main__":
    main()
