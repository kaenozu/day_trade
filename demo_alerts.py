#!/usr/bin/env python3
"""
アラート機能のデモンストレーション
Issue #70の実装内容を示すための実行可能なデモ
"""

import random
import time
from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

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

from src.day_trade.core.alerts import (
    AlertCondition,
    AlertManager,
    AlertPriority,
    AlertTrigger,
    AlertType,
    NotificationHandler,
    NotificationMethod,
    create_change_alert,
    create_price_alert,
)
from src.day_trade.utils.formatters import (
    create_comparison_table,
    create_metric_cards,
    create_status_indicator,
    format_currency,
    format_percentage,
)

console = Console()


class MockStockFetcher:
    """デモ用モック株価データフェッチャー"""

    def __init__(self):
        # 初期価格設定
        self.base_prices = {
            "7203": 2500,  # トヨタ
            "9984": 1500,  # ソフトバンク
            "8306": 800,  # 三菱UFJ
            "6758": 8000,  # ソニー
            "4689": 350,  # Z Holdings
        }

        # 価格変動のシミュレーション用
        self.current_prices = self.base_prices.copy()
        self.volume_multipliers = {symbol: 1.0 for symbol in self.base_prices}
        self.trends = {symbol: 0.0 for symbol in self.base_prices}  # トレンド方向

        # 変動パラメータ
        self.volatility = 0.02  # 2%の標準偏差
        self.trend_change_probability = 0.05  # 5%の確率でトレンド変更

    def get_current_price(self, symbol: str) -> dict:
        """現在価格を取得（シミュレーション）"""
        if symbol not in self.current_prices:
            return None

        # 価格変動シミュレーション
        base_price = self.current_prices[symbol]
        trend = self.trends[symbol]

        # トレンド変更
        if random.random() < self.trend_change_probability:
            self.trends[symbol] = random.gauss(0, 0.01)  # -1%から+1%のトレンド

        # 価格変動
        change = random.gauss(trend, self.volatility)
        new_price = base_price * (1 + change)

        # 価格制限（元の価格の50%-200%）
        min_price = self.base_prices[symbol] * 0.5
        max_price = self.base_prices[symbol] * 2.0
        new_price = max(min_price, min(max_price, new_price))

        self.current_prices[symbol] = new_price

        # 前日比計算
        change_percent = (
            (new_price - self.base_prices[symbol]) / self.base_prices[symbol]
        ) * 100

        # 出来高シミュレーション
        base_volume = random.randint(800000, 2000000)
        volume_multiplier = self.volume_multipliers[symbol]

        # 価格変動が大きいと出来高も増加
        volume_spike = abs(change) * 10  # 変動率に比例
        volume_multiplier = max(
            0.5, volume_multiplier + volume_spike - 0.1
        )  # 徐々に減衰
        self.volume_multipliers[symbol] = volume_multiplier

        volume = int(base_volume * volume_multiplier)

        return {
            "current_price": new_price,
            "previous_close": self.base_prices[symbol],
            "change": new_price - self.base_prices[symbol],
            "change_percent": change_percent,
            "volume": volume,
            "high": new_price * random.uniform(1.0, 1.02),
            "low": new_price * random.uniform(0.98, 1.0),
        }

    def get_historical_data(self, symbol: str, **kwargs) -> pd.DataFrame:
        """履歴データを取得（簡易版）"""
        if symbol not in self.base_prices:
            return pd.DataFrame()

        # 30日分のサンプルデータ生成
        dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
        base_price = self.base_prices[symbol]

        prices = []
        volumes = []
        current_price = base_price

        for _ in dates:
            # 価格変動
            change = random.gauss(0, self.volatility)
            current_price = max(current_price * (1 + change), base_price * 0.7)
            prices.append(current_price)

            # 出来高
            volumes.append(random.randint(500000, 2500000))

        return pd.DataFrame(
            {
                "Open": [p * random.uniform(0.99, 1.01) for p in prices],
                "High": [p * random.uniform(1.0, 1.03) for p in prices],
                "Low": [p * random.uniform(0.97, 1.0) for p in prices],
                "Close": prices,
                "Volume": volumes,
            },
            index=dates,
        )


def demo_basic_alerts():
    """基本的なアラート機能のデモ"""
    console.print(Rule("[bold blue]基本アラート機能", style="blue"))

    # モックフェッチャーとアラートマネージャーを作成
    mock_fetcher = MockStockFetcher()
    alert_manager = AlertManager(stock_fetcher=mock_fetcher)

    # 価格アラートの追加
    price_alerts = [
        create_price_alert(
            "toyota_3000",
            "7203",
            Decimal("3000"),
            above=True,
            priority=AlertPriority.HIGH,
        ),
        create_price_alert(
            "sony_7500",
            "6758",
            Decimal("7500"),
            above=False,
            priority=AlertPriority.MEDIUM,
        ),
        create_change_alert(
            "softbank_up_5", "9984", 5.0, up=True, priority=AlertPriority.MEDIUM
        ),
        create_change_alert(
            "ufj_down_3", "8306", -3.0, up=False, priority=AlertPriority.LOW
        ),
    ]

    for alert in price_alerts:
        alert_manager.add_alert(alert)

    # 追加されたアラートを表示
    alerts_table = Table(title="🚨 設定されたアラート条件")
    alerts_table.add_column("ID", style="cyan")
    alerts_table.add_column("銘柄", style="white")
    alerts_table.add_column("タイプ", style="yellow")
    alerts_table.add_column("条件", style="green")
    alerts_table.add_column("優先度", style="red")
    alerts_table.add_column("説明", style="dim")

    for alert in alert_manager.get_alerts():
        priority_colors = {
            AlertPriority.LOW: "dim",
            AlertPriority.MEDIUM: "yellow",
            AlertPriority.HIGH: "red",
            AlertPriority.CRITICAL: "bright_red",
        }

        priority_color = priority_colors.get(alert.priority, "white")

        alerts_table.add_row(
            alert.alert_id,
            alert.symbol,
            alert.alert_type.value.replace("_", " ").title(),
            str(alert.condition_value),
            f"[{priority_color}]{alert.priority.value.upper()}[/{priority_color}]",
            alert.description,
        )

    console.print(alerts_table)

    return alert_manager


def demo_custom_alerts():
    """カスタムアラートのデモ"""
    console.print(Rule("[bold magenta]カスタムアラート機能", style="magenta"))

    mock_fetcher = MockStockFetcher()
    alert_manager = AlertManager(stock_fetcher=mock_fetcher)

    # カスタムアラート関数の定義
    def volume_price_breakout(
        symbol, price, volume, change_pct, historical_data, params
    ):
        """出来高・価格ブレイクアウト検出"""
        min_price = params.get("min_price", 0)
        min_volume_ratio = params.get("min_volume_ratio", 1.5)
        min_change_pct = params.get("min_change_pct", 2.0)

        # 履歴データがある場合の出来高比較
        volume_ratio = 1.0
        if (
            historical_data is not None
            and not historical_data.empty
            and len(historical_data) > 10
        ):
            avg_volume = historical_data["Volume"].rolling(window=10).mean().iloc[-1]
            if avg_volume > 0:
                volume_ratio = volume / avg_volume

        # 条件チェック
        price_ok = float(price) >= min_price
        volume_ok = volume_ratio >= min_volume_ratio
        change_ok = abs(change_pct) >= min_change_pct

        return price_ok and volume_ok and change_ok

    def rsi_divergence(symbol, price, volume, change_pct, historical_data, params):
        """RSIダイバージェンス検出（簡易版）"""
        if (
            historical_data is None
            or historical_data.empty
            or len(historical_data) < 14
        ):
            return False

        # 簡易RSI計算
        closes = historical_data["Close"]
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        if len(rsi) < 2:
            return False

        current_rsi = rsi.iloc[-1]
        threshold = params.get("rsi_threshold", 70)

        return current_rsi >= threshold

    # カスタムアラートを追加
    custom_alerts = [
        AlertCondition(
            alert_id="breakout_7203",
            symbol="7203",
            alert_type=AlertType.CUSTOM_CONDITION,
            condition_value="custom",
            priority=AlertPriority.HIGH,
            custom_function=volume_price_breakout,
            custom_parameters={
                "min_price": 2800,
                "min_volume_ratio": 1.8,
                "min_change_pct": 3.0,
            },
            description="価格2800円以上 + 出来高1.8倍 + 変動3%以上",
        ),
        AlertCondition(
            alert_id="rsi_overbought_9984",
            symbol="9984",
            alert_type=AlertType.CUSTOM_CONDITION,
            condition_value="custom",
            priority=AlertPriority.MEDIUM,
            custom_function=rsi_divergence,
            custom_parameters={"rsi_threshold": 75},
            description="RSI 75以上（買われすぎ）",
        ),
    ]

    for alert in custom_alerts:
        alert_manager.add_alert(alert)

    # カスタムアラート表示
    console.print(
        Panel(
            "[cyan]カスタムアラート条件:[/cyan]\n\n"
            "[white]1. ブレイクアウト検出 (7203):[/white]\n"
            "   • 価格: 2800円以上\n"
            "   • 出来高: 平均の1.8倍以上\n"
            "   • 変動率: 3%以上\n\n"
            "[white]2. RSI買われすぎ (9984):[/white]\n"
            "   • RSI: 75以上\n"
            "   • 14日間RSI指標を使用",
            title="🔧 カスタムアラート設定",
            border_style="magenta",
        )
    )

    return alert_manager


def demo_notification_methods():
    """通知方法のデモ"""
    console.print(Rule("[bold yellow]通知機能", style="yellow"))

    # 通知ハンドラーの作成
    notification_handler = NotificationHandler()

    # サンプルアラート発火記録
    sample_trigger = AlertTrigger(
        alert_id="demo_trigger",
        symbol="7203",
        trigger_time=datetime.now(),
        alert_type=AlertType.PRICE_ABOVE,
        current_value=Decimal("3100"),
        condition_value=Decimal("3000"),
        message="デモアラート: 価格が3000円を突破しました",
        priority=AlertPriority.HIGH,
        current_price=Decimal("3100"),
        volume=2500000,
        change_percent=4.2,
    )

    console.print(
        Panel(
            "[cyan]利用可能な通知方法:[/cyan]\n\n"
            "[green]✓ コンソール出力[/green] - リアルタイム表示\n"
            "[green]✓ ファイルログ[/green] - alerts_YYYYMMDD.log\n"
            "[yellow]○ メール通知[/yellow] - SMTP設定が必要\n"
            "[yellow]○ Webhook[/yellow] - カスタムエンドポイント\n"
            "[yellow]○ コールバック関数[/yellow] - カスタム処理",
            title="📧 通知システム",
            border_style="yellow",
        )
    )

    console.print("\n[white]コンソール通知のデモ:[/white]")
    notification_handler._send_console_notification(sample_trigger)

    console.print("\n[dim]ファイルログとメール通知は実際の運用時に使用されます。[/dim]")

    return notification_handler


def demo_real_time_monitoring():
    """リアルタイム監視のデモ"""
    console.print(Rule("[bold green]リアルタイム監視", style="green"))

    mock_fetcher = MockStockFetcher()
    alert_manager = AlertManager(stock_fetcher=mock_fetcher)

    # アラート条件を設定（発火しやすい設定）
    monitoring_alerts = [
        AlertCondition(
            alert_id="monitor_toyota",
            symbol="7203",
            alert_type=AlertType.CHANGE_PERCENT_UP,
            condition_value=2.0,  # 2%上昇で発火
            priority=AlertPriority.MEDIUM,
            cooldown_minutes=1,  # 1分クールダウン
            description="トヨタ 2%上昇アラート",
        ),
        AlertCondition(
            alert_id="monitor_volume",
            symbol="9984",
            alert_type=AlertType.VOLUME_SPIKE,
            condition_value=1.3,  # 平均の1.3倍で発火
            priority=AlertPriority.HIGH,
            cooldown_minutes=2,
            description="ソフトバンク 出来高急増",
        ),
    ]

    for alert in monitoring_alerts:
        alert_manager.add_alert(alert)

    # カスタム通知ハンドラー（デモ用）
    triggered_alerts = []

    def demo_notification_handler(trigger):
        triggered_alerts.append(
            {
                "time": trigger.trigger_time,
                "symbol": trigger.symbol,
                "type": trigger.alert_type.value,
                "message": trigger.message,
                "priority": trigger.priority.value,
            }
        )

    alert_manager.notification_handler.add_custom_handler(
        NotificationMethod.CALLBACK, demo_notification_handler
    )
    alert_manager.configure_notifications(
        [NotificationMethod.CONSOLE, NotificationMethod.CALLBACK]
    )

    def create_monitoring_display():
        """監視画面レイアウト作成"""
        layout = Layout()

        # 上部：アクティブアラート
        active_alerts_table = Table(title="🔍 監視中のアラート")
        active_alerts_table.add_column("銘柄", style="cyan")
        active_alerts_table.add_column("タイプ", style="yellow")
        active_alerts_table.add_column("条件", style="green")
        active_alerts_table.add_column("状態", style="white")

        for alert in alert_manager.get_alerts():
            status = create_status_indicator("active", "")
            active_alerts_table.add_row(
                alert.symbol,
                alert.alert_type.value.replace("_", " ").title(),
                str(alert.condition_value),
                str(status).replace("Status: ", ""),
            )

        # 中央：現在価格
        current_prices = {}
        symbols = ["7203", "9984", "8306"]
        for symbol in symbols:
            price_data = mock_fetcher.get_current_price(symbol)
            if price_data:
                current_prices[symbol] = price_data

        prices_table = Table(title="💹 現在価格")
        prices_table.add_column("銘柄", style="cyan")
        prices_table.add_column("価格", justify="right")
        prices_table.add_column("変化率", justify="right")
        prices_table.add_column("出来高", justify="right")

        for symbol, data in current_prices.items():
            change_color = (
                "green"
                if data["change_percent"] > 0
                else "red"
                if data["change_percent"] < 0
                else "white"
            )
            prices_table.add_row(
                symbol,
                format_currency(data["current_price"]),
                f"[{change_color}]{format_percentage(data['change_percent'])}[/{change_color}]",
                f"{data['volume']:,}",
            )

        # 下部：アラート履歴
        history_table = Table(title="📋 最近のアラート")
        history_table.add_column("時刻", style="dim")
        history_table.add_column("銘柄", style="cyan")
        history_table.add_column("メッセージ", style="white")
        history_table.add_column("優先度", style="yellow")

        for alert_record in triggered_alerts[-5:]:  # 最新5件
            priority_color = {
                "low": "dim",
                "medium": "yellow",
                "high": "red",
                "critical": "bright_red",
            }.get(alert_record["priority"], "white")

            history_table.add_row(
                alert_record["time"].strftime("%H:%M:%S"),
                alert_record["symbol"],
                (
                    alert_record["message"][:50] + "..."
                    if len(alert_record["message"]) > 50
                    else alert_record["message"]
                ),
                f"[{priority_color}]{alert_record['priority'].upper()}[/{priority_color}]",
            )

        if not triggered_alerts:
            history_table.add_row("--", "--", "アラートはまだ発火していません", "--")

        layout.split_column(
            Layout(Panel(active_alerts_table, border_style="blue"), size=8),
            Layout(Panel(prices_table, border_style="green"), size=8),
            Layout(Panel(history_table, border_style="yellow"), size=10),
        )

        return layout

    console.print("[yellow]リアルタイム監視を開始します（30秒間）...[/yellow]")
    console.print("[dim]価格変動によってアラートが発火します[/dim]\n")

    try:
        with safe_live_context(
            create_monitoring_display(), refresh_per_second=2, screen=False
        ) as live:
            # 監視開始
            alert_manager.start_monitoring(interval_seconds=3)

            # 30秒間監視
            for _ in range(30):
                time.sleep(1)
                if live:  # Liveが有効な場合のみ更新
                    live.update(create_monitoring_display())

        alert_manager.stop_monitoring()

    except KeyboardInterrupt:
        alert_manager.stop_monitoring()
        console.print("\n[yellow]監視を中断しました。[/yellow]")

    console.print(
        f"\n[green]監視完了！ {len(triggered_alerts)} 件のアラートが発火しました。[/green]"
    )

    return triggered_alerts


def demo_alert_analysis():
    """アラート分析のデモ"""
    console.print(Rule("[bold cyan]アラート分析", style="cyan"))

    # サンプルアラート履歴を生成
    sample_history = []
    symbols = ["7203", "9984", "8306", "6758"]
    alert_types = [
        AlertType.PRICE_ABOVE,
        AlertType.PRICE_BELOW,
        AlertType.CHANGE_PERCENT_UP,
        AlertType.VOLUME_SPIKE,
    ]
    priorities = [AlertPriority.LOW, AlertPriority.MEDIUM, AlertPriority.HIGH]

    base_time = datetime.now() - timedelta(hours=12)

    for i in range(20):
        trigger_time = base_time + timedelta(
            minutes=random.randint(0, 720)
        )  # 12時間以内

        sample_history.append(
            AlertTrigger(
                alert_id=f"sample_{i}",
                symbol=random.choice(symbols),
                trigger_time=trigger_time,
                alert_type=random.choice(alert_types),
                current_value=random.uniform(800, 3000),
                condition_value=random.uniform(800, 3000),
                message=f"サンプルアラート {i + 1}",
                priority=random.choice(priorities),
                current_price=Decimal(str(random.uniform(800, 3000))),
                volume=random.randint(500000, 3000000),
                change_percent=random.uniform(-5, 5),
            )
        )

    # 統計分析
    total_alerts = len(sample_history)
    symbol_counts = {}
    type_counts = {}
    priority_counts = {}

    for trigger in sample_history:
        # 銘柄別カウント
        symbol_counts[trigger.symbol] = symbol_counts.get(trigger.symbol, 0) + 1

        # タイプ別カウント
        type_name = trigger.alert_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # 優先度別カウント
        priority_name = trigger.priority.value
        priority_counts[priority_name] = priority_counts.get(priority_name, 0) + 1

    # 統計表示
    stats_metrics = {
        "総アラート数": total_alerts,
        "対象銘柄数": len(symbol_counts),
        "平均/時間": f"{total_alerts / 12:.1f}",
        "最多銘柄": (
            max(symbol_counts, key=symbol_counts.get) if symbol_counts else "N/A"
        ),
        "高優先度": priority_counts.get("high", 0),
        "中優先度": priority_counts.get("medium", 0),
    }

    metric_cards = create_metric_cards(stats_metrics, columns=3)
    console.print(
        Panel(metric_cards, title="📊 アラート統計（過去12時間）", border_style="cyan")
    )

    # 銘柄別分析テーブル
    symbol_analysis_data = {}
    for symbol, count in symbol_counts.items():
        symbol_triggers = [t for t in sample_history if t.symbol == symbol]
        avg_priority_score = sum(
            [
                {"low": 1, "medium": 2, "high": 3, "critical": 4}[t.priority.value]
                for t in symbol_triggers
            ]
        ) / len(symbol_triggers)

        symbol_analysis_data[symbol] = {
            "alert_count": count,
            "avg_priority": avg_priority_score,
            "frequency_per_hour": count / 12,
        }

    symbol_table = create_comparison_table(
        symbol_analysis_data, "📈 銘柄別アラート分析"
    )
    console.print(symbol_table)

    # タイムライン表示（最新10件）
    recent_alerts = sorted(sample_history, key=lambda x: x.trigger_time, reverse=True)[
        :10
    ]

    timeline_table = Table(title="⏰ 最近のアラート タイムライン")
    timeline_table.add_column("時刻", style="dim")
    timeline_table.add_column("銘柄", style="cyan")
    timeline_table.add_column("タイプ", style="yellow")
    timeline_table.add_column("優先度", style="red")
    timeline_table.add_column("価格", justify="right")
    timeline_table.add_column("変化率", justify="right")

    for trigger in recent_alerts:
        priority_colors = {
            AlertPriority.LOW: "dim",
            AlertPriority.MEDIUM: "yellow",
            AlertPriority.HIGH: "red",
            AlertPriority.CRITICAL: "bright_red",
        }

        priority_color = priority_colors.get(trigger.priority, "white")
        change_color = (
            "green"
            if trigger.change_percent > 0
            else "red"
            if trigger.change_percent < 0
            else "white"
        )

        timeline_table.add_row(
            trigger.trigger_time.strftime("%m/%d %H:%M"),
            trigger.symbol,
            trigger.alert_type.value.replace("_", " ").title(),
            f"[{priority_color}]{trigger.priority.value.upper()}[/{priority_color}]",
            (
                format_currency(float(trigger.current_price))
                if trigger.current_price
                else "N/A"
            ),
            (
                f"[{change_color}]{format_percentage(trigger.change_percent)}[/{change_color}]"
                if trigger.change_percent is not None
                else "N/A"
            ),
        )

    console.print(timeline_table)


def main():
    """メインデモ実行"""
    console.print(
        Panel(
            "[bold cyan]🚨 アラート機能 デモンストレーション[/bold cyan]\n"
            "[white]Issue #70で実装された全てのアラート機能を紹介します[/white]",
            title="🚀 デイトレーディングシステム",
            border_style="bright_blue",
        )
    )

    console.print(
        "\n[yellow]各デモを順番に実行します。Enterキーで次に進んでください...[/yellow]"
    )

    demos = [
        ("基本アラート機能", demo_basic_alerts),
        ("カスタムアラート", demo_custom_alerts),
        ("通知機能", demo_notification_methods),
        ("リアルタイム監視", demo_real_time_monitoring),
        ("アラート分析", demo_alert_analysis),
    ]

    results = {}

    try:
        for name, demo_func in demos:
            input(f"\n[dim]Press Enter to show {name}...[/dim]")
            console.clear()

            result = demo_func()
            results[name] = result

            console.print(f"\n[green]✅ {name} デモ完了[/green]")

        console.print(
            Panel(
                "[bold green]🎉 全てのアラート機能のデモが完了しました！[/bold green]\n"
                "[white]これらの機能により、株価の動きを24時間監視し、\n"
                "重要な変化を即座に通知することが可能になりました。[/white]\n\n"
                "[cyan]主な機能:[/cyan]\n"
                "• 価格・変化率・出来高・テクニカル指標アラート\n"
                "• カスタムアラート条件\n"
                "• 複数通知方法（コンソール・メール・ログ・Webhook）\n"
                "• リアルタイム監視とクールダウン管理\n"
                "• アラート履歴と統計分析\n"
                "• 優先度管理と有効期限設定",
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
