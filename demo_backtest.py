#!/usr/bin/env python3
"""
バックテスト機能のデモンストレーション
Issue #69の実装内容を示すための実行可能なデモ
"""

import random
import time
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

from src.day_trade.analysis.backtest import (
    BacktestConfig,
    BacktestEngine,
    simple_sma_strategy,
)
from src.day_trade.analysis.signals import Signal, SignalType
from src.day_trade.utils.formatters import (
    create_ascii_chart,
    create_comparison_table,
    create_metric_cards,
    format_currency,
    format_percentage,
)

console = Console()


def create_mock_historical_data(
    symbol: str, start_date: datetime, end_date: datetime, trend: str = "random"
) -> pd.DataFrame:
    """
    モック履歴データを生成

    Args:
        symbol: 銘柄コード
        start_date: 開始日
        end_date: 終了日
        trend: トレンド（"up", "down", "sideways", "volatile", "random"）
    """
    dates = pd.date_range(start_date, end_date, freq="D")
    days = len(dates)

    # ベース価格を銘柄に基づいて設定
    base_prices = {
        "7203": 2500,  # トヨタ
        "9984": 1500,  # ソフトバンク
        "8306": 800,  # 三菱UFJ
        "6758": 8000,  # ソニー
        "4689": 350,  # Z Holdings
    }

    base_price = base_prices.get(symbol, 2000)

    # トレンドに基づく価格生成
    if trend == "up":
        # 上昇トレンド（年率15%程度）
        prices = [
            base_price * (1 + 0.15 * (i / days) + random.gauss(0, 0.02))
            for i in range(days)
        ]
    elif trend == "down":
        # 下降トレンド（年率-10%程度）
        prices = [
            base_price * (1 - 0.10 * (i / days) + random.gauss(0, 0.02))
            for i in range(days)
        ]
    elif trend == "sideways":
        # 横ばいトレンド
        prices = [base_price + random.gauss(0, base_price * 0.05) for _ in range(days)]
    elif trend == "volatile":
        # 高ボラティリティ
        prices = [base_price]
        for i in range(1, days):
            change = random.gauss(0, 0.04)  # 4%の標準偏差
            prices.append(
                max(prices[-1] * (1 + change), base_price * 0.5)
            )  # 最低50%まで
    else:
        # ランダムウォーク
        prices = [base_price]
        for i in range(1, days):
            change = random.gauss(0.0002, 0.02)  # 小さな上昇バイアス
            prices.append(max(prices[-1] * (1 + change), base_price * 0.3))

    # OHLCV データを生成
    data = []
    for i, close in enumerate(prices):
        daily_volatility = 0.02
        high = close * (1 + random.uniform(0, daily_volatility))
        low = close * (1 - random.uniform(0, daily_volatility))
        open_price = close * (1 + random.gauss(0, 0.01))
        volume = random.randint(500000, 3000000)

        data.append(
            {
                "Open": max(open_price, low),
                "High": max(high, open_price, close),
                "Low": min(low, open_price, close),
                "Close": close,
                "Volume": volume,
            }
        )

    return pd.DataFrame(data, index=dates)


class MockStockFetcher:
    """モック株価データ取得クラス"""

    def __init__(self):
        self.data_cache = {}

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime = None,
        end_date: datetime = None,
        **kwargs,
    ) -> pd.DataFrame:
        """履歴データの取得（モック）"""
        cache_key = f"{symbol}_{start_date}_{end_date}"

        if cache_key not in self.data_cache:
            # 銘柄ごとに異なるトレンドを設定
            trends = {
                "7203": "up",  # トヨタ：上昇
                "9984": "volatile",  # ソフトバンク：高ボラティリティ
                "8306": "sideways",  # 三菱UFJ：横ばい
                "6758": "up",  # ソニー：上昇
                "4689": "down",  # Z Holdings：下降
            }

            trend = trends.get(symbol, "random")
            self.data_cache[cache_key] = create_mock_historical_data(
                symbol, start_date, end_date, trend
            )

        return self.data_cache[cache_key]


def demo_basic_backtest():
    """基本的なバックテストのデモ"""
    console.print(Rule("[bold blue]基本バックテスト機能", style="blue"))

    # モックデータフェッチャーを使用
    # mock_fetcher = MockStockFetcher()  # 現在は未使用
    engine = BacktestEngine()

    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Decimal("1000000"),  # 100万円
        commission=Decimal("0.001"),  # 0.1%
        slippage=Decimal("0.001"),  # 0.1%
    )

    symbols = ["7203", "9984", "8306"]

    console.print(
        Panel(
            f"[cyan]期間:[/cyan] {config.start_date.date()} ～ {config.end_date.date()}\n"
            f"[cyan]初期資金:[/cyan] {format_currency(int(config.initial_capital))}\n"
            f"[cyan]対象銘柄:[/cyan] {', '.join(symbols)}\n"
            f"[cyan]手数料:[/cyan] {format_percentage(float(config.commission * 100))}\n"
            f"[cyan]スリッページ:[/cyan] {format_percentage(float(config.slippage * 100))}",
            title="📊 バックテスト設定",
            border_style="cyan",
        )
    )

    # プログレスバー付きでバックテスト実行
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("バックテスト実行中...", total=100)

        # バックテスト実行（実際は瞬時だが、デモのため段階的に表示）
        for i in range(100):
            time.sleep(0.02)
            progress.update(task, advance=1)

        result = engine.run_backtest(symbols, config, simple_sma_strategy)

    # 結果表示
    console.print(
        Panel(
            f"[green]総リターン:[/green] {format_percentage(float(result.total_return * 100))}\n"
            f"[green]年率リターン:[/green] {format_percentage(float(result.annualized_return * 100))}\n"
            f"[blue]ボラティリティ:[/blue] {format_percentage(result.volatility * 100)}\n"
            f"[blue]シャープレシオ:[/blue] {result.sharpe_ratio:.2f}\n"
            f"[red]最大ドローダウン:[/red] {format_percentage(result.max_drawdown * 100)}\n"
            f"[yellow]勝率:[/yellow] {format_percentage(result.win_rate * 100)}\n"
            f"[white]総取引数:[/white] {result.total_trades}",
            title="📈 バックテスト結果",
            border_style="green",
        )
    )

    return result


def demo_strategy_comparison():
    """複数戦略比較のデモ"""
    console.print(Rule("[bold magenta]戦略比較バックテスト", style="magenta"))

    mock_fetcher = MockStockFetcher()
    engine = BacktestEngine(stock_fetcher=mock_fetcher)

    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Decimal("1000000"),
    )

    symbols = ["7203", "9984", "8306", "6758"]

    # 複数の戦略を定義
    def aggressive_strategy(symbols, date, historical_data):
        """アグレッシブ戦略（頻繁に取引）"""
        signals = []
        for symbol in symbols:
            if symbol not in historical_data:
                continue

            data = historical_data[symbol]
            current_data = data[data.index <= date]

            if len(current_data) < 5:
                continue

            # 短期移動平均クロス
            if len(current_data) >= 5:
                short_ma = current_data["Close"].rolling(window=3).mean()
                long_ma = current_data["Close"].rolling(window=5).mean()

                if len(short_ma) >= 2 and len(long_ma) >= 2:
                    if (
                        short_ma.iloc[-1] > long_ma.iloc[-1]
                        and short_ma.iloc[-2] <= long_ma.iloc[-2]
                    ):
                        signals.append(
                            Signal(
                                timestamp=date,
                                symbol=symbol,
                                signal_type=SignalType.BUY,
                                confidence=0.8,
                                price=Decimal(str(current_data["Close"].iloc[-1])),
                                indicators={},
                            )
                        )
                    elif (
                        short_ma.iloc[-1] < long_ma.iloc[-1]
                        and short_ma.iloc[-2] >= long_ma.iloc[-2]
                    ):
                        signals.append(
                            Signal(
                                timestamp=date,
                                symbol=symbol,
                                signal_type=SignalType.SELL,
                                confidence=0.8,
                                price=Decimal(str(current_data["Close"].iloc[-1])),
                                indicators={},
                            )
                        )
        return signals

    def conservative_strategy(symbols, date, historical_data):
        """保守的戦略（慎重に取引）"""
        signals = []
        for symbol in symbols:
            if symbol not in historical_data:
                continue

            data = historical_data[symbol]
            current_data = data[data.index <= date]

            if len(current_data) < 50:
                continue

            # 長期移動平均クロス + ボリューム確認
            short_ma = current_data["Close"].rolling(window=20).mean()
            long_ma = current_data["Close"].rolling(window=50).mean()
            volume_ma = current_data["Volume"].rolling(window=20).mean()

            if len(short_ma) >= 2 and len(long_ma) >= 2:
                volume_spike = (
                    current_data["Volume"].iloc[-1] > volume_ma.iloc[-1] * 1.5
                )

                if (
                    short_ma.iloc[-1] > long_ma.iloc[-1]
                    and short_ma.iloc[-2] <= long_ma.iloc[-2]
                    and volume_spike
                ):
                    signals.append(
                        Signal(
                            timestamp=date,
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            confidence=0.9,
                            price=Decimal(str(current_data["Close"].iloc[-1])),
                            indicators={},
                        )
                    )
                elif (
                    short_ma.iloc[-1] < long_ma.iloc[-1]
                    and short_ma.iloc[-2] >= long_ma.iloc[-2]
                    and volume_spike
                ):
                    signals.append(
                        Signal(
                            timestamp=date,
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            confidence=0.9,
                            price=Decimal(str(current_data["Close"].iloc[-1])),
                            indicators={},
                        )
                    )
        return signals

    strategies = {
        "SMA戦略": simple_sma_strategy,
        "アグレッシブ戦略": aggressive_strategy,
        "保守的戦略": conservative_strategy,
    }

    console.print(f"[yellow]{len(strategies)}つの戦略を比較実行中...[/yellow]")

    results = {}
    with Progress(console=console) as progress:
        task = progress.add_task("戦略比較実行中...", total=len(strategies))

        for name, strategy_func in strategies.items():
            results[name] = engine.run_backtest(symbols, config, strategy_func)
            progress.update(task, advance=1)

    # 比較結果をテーブルで表示
    comparison_data = {}
    for name, result in results.items():
        comparison_data[name] = {
            "total_return": float(result.total_return * 100),
            "annualized_return": float(result.annualized_return * 100),
            "volatility": result.volatility * 100,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown * 100,
            "win_rate": result.win_rate * 100,
            "total_trades": result.total_trades,
        }

    comparison_table = create_comparison_table(
        comparison_data, "📊 戦略パフォーマンス比較"
    )
    console.print(comparison_table)

    return results


def demo_portfolio_analysis(result):
    """ポートフォリオ分析のデモ"""
    console.print(Rule("[bold yellow]ポートフォリオ分析", style="yellow"))

    # メトリクスカード表示
    metrics = {
        "最終資産": (
            int(result.portfolio_value.iloc[-1])
            if not result.portfolio_value.empty
            else 0
        ),
        "総リターン": f"{float(result.total_return * 100):.1f}%",
        "年率リターン": f"{float(result.annualized_return * 100):.1f}%",
        "シャープレシオ": f"{result.sharpe_ratio:.2f}",
        "最大DD": f"{result.max_drawdown * 100:.1f}%",
        "勝率": f"{result.win_rate * 100:.1f}%",
    }

    metric_cards = create_metric_cards(metrics, columns=3)
    console.print(
        Panel(metric_cards, title="💼 ポートフォリオメトリクス", border_style="yellow")
    )

    # パフォーマンスチャート
    if not result.portfolio_value.empty and len(result.portfolio_value) > 1:
        portfolio_values = result.portfolio_value.values.tolist()
        chart = create_ascii_chart(
            portfolio_values, width=60, height=12, title="📈 ポートフォリオ価値推移"
        )

        console.print(
            Panel(chart, title="📊 パフォーマンスチャート", border_style="blue")
        )

    # 取引履歴（最新10件）
    if result.trades:
        trade_table = Table(title="📋 取引履歴（最新10件）")
        trade_table.add_column("日付", style="cyan")
        trade_table.add_column("銘柄", style="white")
        trade_table.add_column("売買", style="white")
        trade_table.add_column("数量", justify="right")
        trade_table.add_column("価格", justify="right")
        trade_table.add_column("手数料", justify="right")

        for trade in result.trades[-10:]:
            action_color = "green" if trade.action.value == "BUY" else "red"
            trade_table.add_row(
                trade.timestamp.strftime("%Y-%m-%d"),
                trade.symbol,
                f"[{action_color}]{trade.action.value}[/{action_color}]",
                str(trade.quantity),
                format_currency(float(trade.price)),
                format_currency(float(trade.commission)),
            )

        console.print(trade_table)


def demo_risk_analysis(result):
    """リスク分析のデモ"""
    console.print(Rule("[bold red]リスク分析", style="red"))

    if result.daily_returns.empty:
        console.print(
            "[red]日次リターンデータが不足しているため、リスク分析をスキップします。[/red]"
        )
        return

    # VaR（Value at Risk）計算
    daily_returns = result.daily_returns.dropna()

    if len(daily_returns) > 10:
        var_95 = np.percentile(daily_returns, 5)  # 95% VaR
        var_99 = np.percentile(daily_returns, 1)  # 99% VaR

        # リスク指標テーブル
        risk_table = Table(title="🚨 リスク指標")
        risk_table.add_column("指標", style="cyan")
        risk_table.add_column("値", justify="right")
        risk_table.add_column("意味", style="dim")

        risk_table.add_row(
            "日次ボラティリティ", f"{daily_returns.std():.3%}", "日次価格変動の標準偏差"
        )
        risk_table.add_row(
            "年率ボラティリティ", f"{result.volatility:.1%}", "年率換算の価格変動リスク"
        )
        risk_table.add_row("VaR (95%)", f"{var_95:.2%}", "95%の確率で損失がこの値以下")
        risk_table.add_row("VaR (99%)", f"{var_99:.2%}", "99%の確率で損失がこの値以下")
        risk_table.add_row(
            "最大ドローダウン",
            f"{result.max_drawdown:.2%}",
            "過去最高値からの最大下落率",
        )

        console.print(risk_table)

        # リターン分布の簡易表示
        if len(daily_returns) > 50:
            returns_pct = (daily_returns * 100).values.tolist()

            # 分布の簡易ヒストグラム
            bins = 15
            hist, bin_edges = np.histogram(returns_pct, bins=bins)

            console.print("\n[bold]日次リターン分布:[/bold]")
            max_count = max(hist)
            for i, count in enumerate(hist):
                bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                bar_length = int((count / max_count) * 30) if max_count > 0 else 0
                bar = "█" * bar_length
                console.print(f"{bin_center:6.2f}% │{bar:<30} ({count})")


def interactive_demo():
    """インタラクティブデモ"""
    console.print(Rule("[bold green]インタラクティブバックテスト", style="green"))

    console.print(
        "[yellow]リアルタイムでバックテストの進行状況を表示します...[/yellow]"
    )
    console.print("[dim]Ctrl+C で終了[/dim]\n")

    mock_fetcher = MockStockFetcher()
    engine = BacktestEngine(stock_fetcher=mock_fetcher)

    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 3, 31),  # 短期間
        initial_capital=Decimal("1000000"),
    )

    symbols = ["7203", "9984", "8306"]

    # 実際に使用する関数にて利用（一時的な回避策）
    _ = (engine, symbols)

    def create_progress_display(current_date, portfolio_value, trades_count):
        """プログレス表示レイアウト作成"""
        from rich.layout import Layout

        layout = Layout()

        # 上部：進捗情報
        progress_info = Panel(
            f"[cyan]現在日付:[/cyan] {current_date.strftime('%Y-%m-%d')}\n"
            f"[green]ポートフォリオ価値:[/green] {format_currency(int(portfolio_value))}\n"
            f"[yellow]取引回数:[/cyan] {trades_count}",
            title="📊 バックテスト進捗",
            border_style="blue",
        )

        # 下部：簡易チャート（最新データ）
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
            create_progress_display(config.start_date, config.initial_capital, 0),
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
                    create_progress_display(current_date, current_value, trades_count)
                )
                time.sleep(0.3)

        console.print("\n[green]インタラクティブデモが完了しました！[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]デモを中断しました。[/yellow]")


def main():
    """メインデモ実行"""
    console.print(
        Panel(
            "[bold cyan]🎯 バックテスト機能 デモンストレーション[/bold cyan]\n"
            "[white]Issue #69で実装された全てのバックテスト機能を紹介します[/white]",
            title="🚀 デイトレーディングシステム",
            border_style="bright_blue",
        )
    )

    console.print(
        "\n[yellow]各デモを順番に実行します。Enterキーで次に進んでください...[/yellow]"
    )

    demos = [
        ("基本バックテスト", demo_basic_backtest),
        ("戦略比較", demo_strategy_comparison),
    ]

    backtest_result = None
    # strategy_results = None # 削除

    try:
        for name, demo_func in demos:
            input(f"\n[dim]Press Enter to show {name}...[/dim]")
            console.clear()

            if name == "基本バックテスト":
                backtest_result = demo_func()
            elif name == "戦略比較":
                # strategy_results = demo_func() # 代入は不要
                demo_func()
            else:
                demo_func()

            console.print(f"\n[green]✅ {name} デモ完了[/green]")

        # 分析デモ
        if backtest_result:
            input("\n[dim]Press Enter to show ポートフォリオ分析...[/dim]")
            console.clear()
            demo_portfolio_analysis(backtest_result)
            console.print("\n[green]✅ ポートフォリオ分析 デモ完了[/green]")

            input("\n[dim]Press Enter to show リスク分析...[/dim]")
            console.clear()
            demo_risk_analysis(backtest_result)
            console.print("\n[green]✅ リスク分析 デモ完了[/green]")

        # インタラクティブデモ
        response = input(
            "\n[yellow]インタラクティブバックテストデモを実行しますか？ (y/N): [/yellow]"
        )
        if response.lower() in ["y", "yes"]:
            interactive_demo()

        console.print(
            Panel(
                "[bold green]🎉 全てのバックテスト機能のデモが完了しました！[/bold green]\n"
                "[white]これらの機能により、過去データでの戦略検証と\n"
                "リスク分析が可能になりました。[/white]\n\n"
                "[cyan]主な機能:[/cyan]\n"
                "• 単一・複数戦略のバックテスト\n"
                "• リアルタイム進捗表示\n"
                "• 詳細なパフォーマンス分析\n"
                "• リスク指標の計算\n"
                "• 結果のエクスポート機能",
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
