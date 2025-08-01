"""
インタラクティブモード - RichベースのTUIダッシュボード
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from threading import Thread, Event
import time

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.tree import Tree

from ..core.watchlist import WatchlistManager, AlertType
from ..core.trade_manager import TradeManager
from ..core.portfolio import PortfolioAnalyzer
from ..data.stock_fetcher import StockFetcher
from ..analysis.indicators import TechnicalIndicators
from ..analysis.signals import TradingSignalGenerator

logger = logging.getLogger(__name__)


class InteractiveMode:
    """インタラクティブTUIダッシュボード"""
    
    def __init__(self):
        """初期化"""
        self.console = Console()
        self.layout = Layout()
        self.running = Event()
        self.update_thread = None
        
        # データ管理オブジェクト
        self.watchlist_manager = WatchlistManager()
        self.trade_manager = TradeManager()
        self.stock_fetcher = StockFetcher()
        self.signal_generator = TradingSignalGenerator()
        
        # 状態管理
        self.current_data = {}
        self.selected_stock = None
        self.update_interval = 30  # 30秒間隔
        self.last_update = None
        
        # キーバインディング
        self.keybindings = {
            'q': self._quit,
            'r': self._refresh,
            'h': self._show_help,
            '1': lambda: self._switch_view('dashboard'),
            '2': lambda: self._switch_view('watchlist'),
            '3': lambda: self._switch_view('portfolio'),
            '4': lambda: self._switch_view('alerts'),
        }
        
        self.current_view = 'dashboard'
        
        # レイアウト初期化
        self._setup_layout()
    
    def _setup_layout(self):
        """レイアウトセットアップ"""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        
        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )
        
        self.layout["left"].split_column(
            Layout(name="primary", ratio=2),
            Layout(name="secondary", ratio=1),
        )
    
    def run(self):
        """インタラクティブモードを開始"""
        try:
            self.console.print("[bold green]Day Trade Interactive Mode[/bold green]")
            self.console.print("Loading data...")
            
            # 初期データ読み込み
            self._load_initial_data()
            
            # バックグラウンド更新スレッド開始
            self.running.set()
            self.update_thread = Thread(target=self._background_update, daemon=True)
            self.update_thread.start()
            
            # リアルタイム表示開始
            with Live(self.layout, console=self.console, refresh_per_second=2) as live:
                self._update_display()
                
                while self.running.is_set():
                    try:
                        # キーボード入力をシミュレート（実際の実装では適切な入力処理が必要）
                        time.sleep(1)
                        self._update_display()
                        
                        # デモ用の自動終了（実際の実装では削除）
                        if hasattr(self, '_demo_counter'):
                            self._demo_counter -= 1
                            if self._demo_counter <= 0:
                                break
                        else:
                            self._demo_counter = 30  # 30秒でデモ終了
                            
                    except KeyboardInterrupt:
                        break
            
        except Exception as e:
            logger.error(f"インタラクティブモード実行エラー: {e}")
            self.console.print(f"[red]エラー: {e}[/red]")
        finally:
            self._cleanup()
    
    def _load_initial_data(self):
        """初期データ読み込み"""
        try:
            # ウォッチリストデータ取得
            watchlist = self.watchlist_manager.get_watchlist()
            
            # サンプルデータがない場合は作成
            if not watchlist:
                self._create_sample_data()
                watchlist = self.watchlist_manager.get_watchlist()
            
            # 価格データ取得（制限あり）
            self.current_data = {
                'watchlist': watchlist,
                'alerts': self.watchlist_manager.get_alerts(),
                'portfolio_summary': self.trade_manager.get_portfolio_summary(),
                'last_update': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"初期データ読み込みエラー: {e}")
            self.current_data = {
                'watchlist': [],
                'alerts': [],
                'portfolio_summary': {},
                'last_update': datetime.now()
            }
    
    def _create_sample_data(self):
        """サンプルデータ作成（デモ用）"""
        try:
            # サンプル銘柄を追加
            sample_stocks = [
                ("7203", "主力株", "トヨタ自動車"),
                ("8306", "銀行株", "三菱UFJ銀行"),
                ("9984", "テック株", "ソフトバンクグループ")
            ]
            
            for code, group, memo in sample_stocks:
                self.watchlist_manager.add_stock(code, group, memo)
            
        except Exception as e:
            logger.error(f"サンプルデータ作成エラー: {e}")
    
    def _background_update(self):
        """バックグラウンドデータ更新"""
        while self.running.is_set():
            try:
                # データ更新
                self._update_data()
                
                # 指定間隔で待機
                for _ in range(self.update_interval):
                    if not self.running.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"バックグラウンド更新エラー: {e}")
                time.sleep(5)
    
    def _update_data(self):
        """データ更新"""
        try:
            # ウォッチリストを更新
            watchlist = self.watchlist_manager.get_watchlist()
            alerts = self.watchlist_manager.get_alerts()
            
            # ポートフォリオ情報を更新
            portfolio_summary = self.trade_manager.get_portfolio_summary()
            
            self.current_data.update({
                'watchlist': watchlist,
                'alerts': alerts,
                'portfolio_summary': portfolio_summary,
                'last_update': datetime.now()
            })
            
        except Exception as e:
            logger.error(f"データ更新エラー: {e}")
    
    def _update_display(self):
        """表示更新"""
        try:
            # ヘッダー更新
            self.layout["header"].update(self._create_header())
            
            # メイン表示更新
            if self.current_view == 'dashboard':
                self._update_dashboard_view()
            elif self.current_view == 'watchlist':
                self._update_watchlist_view()
            elif self.current_view == 'portfolio':
                self._update_portfolio_view()
            elif self.current_view == 'alerts':
                self._update_alerts_view()
            
            # フッター更新
            self.layout["footer"].update(self._create_footer())
            
        except Exception as e:
            logger.error(f"表示更新エラー: {e}")
    
    def _create_header(self) -> Panel:
        """ヘッダー作成"""
        last_update = self.current_data.get('last_update', datetime.now())
        update_text = last_update.strftime("%Y-%m-%d %H:%M:%S")
        
        header_text = Text()
        header_text.append("Day Trade Dashboard", style="bold blue")
        header_text.append(f" | 最終更新: {update_text}", style="dim")
        header_text.append(f" | 表示: {self.current_view}", style="yellow")
        
        return Panel(
            Align.center(header_text),
            box=box.ROUNDED,
            style="blue"
        )
    
    def _create_footer(self) -> Panel:
        """フッター作成"""
        help_text = Text()
        help_text.append("操作: ", style="bold")
        help_text.append("[1]Dashboard [2]Watchlist [3]Portfolio [4]Alerts ", style="cyan")
        help_text.append("[R]更新 [H]ヘルプ [Q]終了", style="yellow")
        
        return Panel(
            Align.center(help_text),
            box=box.ROUNDED,
            style="green"
        )
    
    def _update_dashboard_view(self):
        """ダッシュボード表示更新"""
        # 左上: 市場サマリー
        self.layout["primary"].update(self._create_market_summary())
        
        # 左下: ウォッチリスト（簡易）
        self.layout["secondary"].update(self._create_watchlist_summary())
        
        # 右: ポートフォリオ情報
        self.layout["right"].update(self._create_portfolio_panel())
    
    def _update_watchlist_view(self):
        """ウォッチリスト表示更新"""
        # 左上: 詳細ウォッチリスト
        self.layout["primary"].update(self._create_detailed_watchlist())
        
        # 左下: グループ情報
        self.layout["secondary"].update(self._create_group_info())
        
        # 右: 選択銘柄詳細
        self.layout["right"].update(self._create_stock_detail())
    
    def _update_portfolio_view(self):
        """ポートフォリオ表示更新"""
        # 左上: ポートフォリオメトリクス
        self.layout["primary"].update(self._create_portfolio_metrics())
        
        # 左下: 保有銘柄一覧
        self.layout["secondary"].update(self._create_holdings_list())
        
        # 右: パフォーマンス情報
        self.layout["right"].update(self._create_performance_info())
    
    def _update_alerts_view(self):
        """アラート表示更新"""
        # 左上: アクティブアラート
        self.layout["primary"].update(self._create_active_alerts())
        
        # 左下: アラート履歴
        self.layout["secondary"].update(self._create_alert_history())
        
        # 右: アラート統計
        self.layout["right"].update(self._create_alert_stats())
    
    def _create_market_summary(self) -> Panel:
        """市場サマリーパネル作成"""
        table = Table(title="📈 市場サマリー", box=box.ROUNDED)
        table.add_column("項目", style="cyan")
        table.add_column("値", justify="right")
        table.add_column("変化", justify="right")
        
        # サンプルデータ（実際の実装では市場データを取得）
        market_data = [
            ("日経平均", "33,500", "+250 (+0.75%)"),
            ("TOPIX", "2,350", "+15 (+0.64%)"),
            ("JPY/USD", "150.25", "-0.35 (-0.23%)"),
            ("VIX", "18.5", "+1.2 (+6.95%)"),
        ]
        
        for item, value, change in market_data:
            change_color = "green" if change.startswith("+") else "red" if change.startswith("-") else "white"
            table.add_row(item, value, f"[{change_color}]{change}[/{change_color}]")
        
        return Panel(table, border_style="blue")
    
    def _create_watchlist_summary(self) -> Panel:
        """ウォッチリストサマリー作成"""
        watchlist = self.current_data.get('watchlist', [])
        
        if not watchlist:
            return Panel(
                Align.center("[yellow]ウォッチリストが空です[/yellow]"),
                title="📋 ウォッチリスト",
                border_style="yellow"
            )
        
        table = Table(title="📋 ウォッチリスト (上位5銘柄)", box=box.SIMPLE)
        table.add_column("コード", width=8)
        table.add_column("銘柄名", width=20)
        table.add_column("グループ", width=10)
        
        for item in watchlist[:5]:  # 上位5銘柄のみ表示
            table.add_row(
                item['stock_code'],
                item.get('stock_name', 'N/A')[:18],  # 名前を18文字に制限
                item['group_name']
            )
        
        if len(watchlist) > 5:
            table.add_row("...", f"他{len(watchlist)-5}銘柄", "...")
        
        return Panel(table, border_style="cyan")
    
    def _create_portfolio_panel(self) -> Panel:
        """ポートフォリオパネル作成"""
        summary = self.current_data.get('portfolio_summary', {})
        
        if not summary:
            return Panel(
                Align.center("[yellow]ポートフォリオデータなし[/yellow]"),
                title="💼 ポートフォリオ",
                border_style="yellow"
            )
        
        # ポートフォリオ情報をテーブル形式で表示
        table = Table(title="💼 ポートフォリオ", box=box.ROUNDED)
        table.add_column("項目", style="cyan")
        table.add_column("値", justify="right")
        
        total_cost = summary.get('total_cost', '0')
        total_value = summary.get('total_market_value', '0')
        total_pnl = summary.get('total_unrealized_pnl', '0')
        
        table.add_row("保有銘柄数", str(summary.get('total_positions', 0)))
        table.add_row("総取引数", str(summary.get('total_trades', 0)))
        table.add_row("総コスト", f"{total_cost}円")
        table.add_row("時価総額", f"{total_value}円")
        
        # 損益の色分け
        pnl_color = "green" if total_pnl.replace('-', '').isdigit() and int(total_pnl) >= 0 else "red"
        table.add_row("評価損益", f"[{pnl_color}]{total_pnl}円[/{pnl_color}]")
        
        return Panel(table, border_style="magenta")
    
    def _create_detailed_watchlist(self) -> Panel:
        """詳細ウォッチリスト作成"""
        watchlist = self.current_data.get('watchlist', [])
        
        table = Table(title="📋 詳細ウォッチリスト", box=box.ROUNDED)
        table.add_column("コード", width=8)
        table.add_column("銘柄名", width=25)
        table.add_column("グループ", width=12)
        table.add_column("メモ", width=20)
        table.add_column("追加日", width=12)
        
        for item in watchlist:
            added_date = ""
            if item.get('added_date'):
                try:
                    if hasattr(item['added_date'], 'strftime'):
                        added_date = item['added_date'].strftime('%m-%d')
                    else:
                        added_date = str(item['added_date'])[:10]
                except:
                    added_date = ""
            
            table.add_row(
                item['stock_code'],
                item.get('stock_name', 'N/A')[:23],
                item['group_name'][:10],
                item.get('memo', '')[:18],
                added_date
            )
        
        if not watchlist:
            return Panel(
                Align.center("[yellow]ウォッチリストが空です[/yellow]"),
                title="📋 詳細ウォッチリスト",
                border_style="yellow"
            )
        
        return Panel(table, border_style="cyan")
    
    def _create_group_info(self) -> Panel:
        """グループ情報作成"""
        try:
            groups = self.watchlist_manager.get_groups()
            watchlist = self.current_data.get('watchlist', [])
            
            # グループ別銘柄数を計算
            group_counts = {}
            for item in watchlist:
                group = item['group_name']
                group_counts[group] = group_counts.get(group, 0) + 1
            
            table = Table(title="📁 グループ情報", box=box.SIMPLE)
            table.add_column("グループ名", width=15)
            table.add_column("銘柄数", justify="right", width=8)
            
            for group in groups:
                count = group_counts.get(group, 0)
                table.add_row(group, str(count))
            
            if not groups:
                return Panel(
                    Align.center("[yellow]グループなし[/yellow]"),
                    title="📁 グループ情報",
                    border_style="yellow"
                )
            
            return Panel(table, border_style="blue")
            
        except Exception as e:
            return Panel(
                Align.center(f"[red]エラー: {e}[/red]"),
                title="📁 グループ情報",
                border_style="red"
            )
    
    def _create_stock_detail(self) -> Panel:
        """選択銘柄詳細作成"""
        if not self.selected_stock:
            return Panel(
                Align.center("[dim]銘柄を選択してください[/dim]"),
                title="🔍 銘柄詳細",
                border_style="dim"
            )
        
        # 実際の実装では選択された銘柄の詳細情報を表示
        detail_text = Text()
        detail_text.append(f"銘柄コード: {self.selected_stock}\n", style="bold")
        detail_text.append("現在価格: 取得中...\n", style="cyan")
        detail_text.append("前日比: 取得中...\n", style="green")
        detail_text.append("出来高: 取得中...\n", style="yellow")
        
        return Panel(
            detail_text,
            title="🔍 銘柄詳細",
            border_style="green"
        )
    
    def _create_portfolio_metrics(self) -> Panel:
        """ポートフォリオメトリクス作成"""
        try:
            analyzer = PortfolioAnalyzer(self.trade_manager)
            metrics = analyzer.get_portfolio_metrics()
            
            table = Table(title="📊 ポートフォリオメトリクス", box=box.ROUNDED)
            table.add_column("指標", style="cyan")
            table.add_column("値", justify="right")
            
            table.add_row("総資産額", f"{metrics.total_value:,}円")
            table.add_row("総投資額", f"{metrics.total_cost:,}円")
            
            # 損益の色分け
            pnl_color = "green" if metrics.total_pnl >= 0 else "red"
            table.add_row("評価損益", f"[{pnl_color}]{metrics.total_pnl:+,}円[/{pnl_color}]")
            table.add_row("損益率", f"[{pnl_color}]{metrics.total_pnl_percent:+.2f}%[/{pnl_color}]")
            
            if metrics.volatility:
                table.add_row("ボラティリティ", f"{metrics.volatility:.1%}")
            if metrics.sharpe_ratio:
                table.add_row("シャープレシオ", f"{metrics.sharpe_ratio:.2f}")
            
            return Panel(table, border_style="magenta")
            
        except Exception as e:
            return Panel(
                Align.center(f"[red]メトリクス取得エラー: {e}[/red]"),
                title="📊 ポートフォリオメトリクス",
                border_style="red"
            )
    
    def _create_holdings_list(self) -> Panel:
        """保有銘柄一覧作成"""
        try:
            positions = self.trade_manager.get_all_positions()
            
            if not positions:
                return Panel(
                    Align.center("[yellow]保有銘柄なし[/yellow]"),
                    title="🏢 保有銘柄",
                    border_style="yellow"
                )
            
            table = Table(title="🏢 保有銘柄", box=box.SIMPLE)
            table.add_column("コード", width=8)
            table.add_column("数量", justify="right", width=8)
            table.add_column("平均単価", justify="right", width=10)
            table.add_column("評価損益", justify="right", width=12)
            
            for symbol, position in positions.items():
                pnl_color = "green" if position.unrealized_pnl >= 0 else "red"
                table.add_row(
                    symbol,
                    str(position.quantity),
                    f"{position.average_price:,.0f}円",
                    f"[{pnl_color}]{position.unrealized_pnl:+,.0f}円[/{pnl_color}]"
                )
            
            return Panel(table, border_style="blue")
            
        except Exception as e:
            return Panel(
                Align.center(f"[red]保有銘柄取得エラー: {e}[/red]"),
                title="🏢 保有銘柄",
                border_style="red"
            )
    
    def _create_performance_info(self) -> Panel:
        """パフォーマンス情報作成"""
        try:
            analyzer = PortfolioAnalyzer(self.trade_manager)
            top_performers, worst_performers = analyzer.get_performance_rankings(3)
            
            content = []
            
            if top_performers:
                content.append(Text("🏆 上位銘柄:", style="bold green"))
                for symbol, pnl_pct in top_performers:
                    content.append(Text(f"  {symbol}: +{pnl_pct:.2f}%", style="green"))
            
            content.append(Text())  # 空行
            
            if worst_performers:
                content.append(Text("📉 下位銘柄:", style="bold red"))
                for symbol, pnl_pct in worst_performers:
                    content.append(Text(f"  {symbol}: {pnl_pct:.2f}%", style="red"))
            
            if not content:
                content = [Text("データなし", style="dim")]
            
            return Panel(
                Text.assemble(*content),
                title="📈 パフォーマンス",
                border_style="yellow"
            )
            
        except Exception as e:
            return Panel(
                Align.center(f"[red]パフォーマンス取得エラー: {e}[/red]"),
                title="📈 パフォーマンス",
                border_style="red"
            )
    
    def _create_active_alerts(self) -> Panel:
        """アクティブアラート作成"""
        alerts = self.current_data.get('alerts', [])
        
        if not alerts:
            return Panel(
                Align.center("[yellow]アラートなし[/yellow]"),
                title="🚨 アクティブアラート",
                border_style="yellow"
            )
        
        table = Table(title="🚨 アクティブアラート", box=box.ROUNDED)
        table.add_column("銘柄", width=8)
        table.add_column("タイプ", width=15)
        table.add_column("閾値", justify="right", width=10)
        table.add_column("状態", width=8)
        
        for alert in alerts:
            status_color = "green" if alert['is_active'] else "dim"
            status_text = "ON" if alert['is_active'] else "OFF"
            
            table.add_row(
                alert['stock_code'],
                alert['alert_type'].replace('_', ' '),
                str(alert['threshold']),
                f"[{status_color}]{status_text}[/{status_color}]"
            )
        
        return Panel(table, border_style="red")
    
    def _create_alert_history(self) -> Panel:
        """アラート履歴作成"""
        alerts = self.current_data.get('alerts', [])
        
        # 最近トリガーされたアラートを表示
        triggered_alerts = [
            alert for alert in alerts 
            if alert.get('last_triggered')
        ]
        
        if not triggered_alerts:
            return Panel(
                Align.center("[dim]履歴なし[/dim]"),
                title="📋 アラート履歴",
                border_style="dim"
            )
        
        table = Table(title="📋 アラート履歴", box=box.SIMPLE)
        table.add_column("銘柄", width=8)
        table.add_column("タイプ", width=15)
        table.add_column("発動時刻", width=15)
        
        for alert in triggered_alerts[-5:]:  # 最新5件
            triggered_time = ""
            if alert.get('last_triggered'):
                try:
                    if hasattr(alert['last_triggered'], 'strftime'):
                        triggered_time = alert['last_triggered'].strftime('%m-%d %H:%M')
                    else:
                        triggered_time = str(alert['last_triggered'])[:16]
                except:
                    triggered_time = "不明"
            
            table.add_row(
                alert['stock_code'],
                alert['alert_type'].replace('_', ' '),
                triggered_time
            )
        
        return Panel(table, border_style="cyan")
    
    def _create_alert_stats(self) -> Panel:
        """アラート統計作成"""
        alerts = self.current_data.get('alerts', [])
        
        if not alerts:
            return Panel(
                Align.center("[yellow]統計なし[/yellow]"),
                title="📊 アラート統計",
                border_style="yellow"
            )
        
        # 統計計算
        total_alerts = len(alerts)
        active_alerts = len([a for a in alerts if a['is_active']])
        triggered_alerts = len([a for a in alerts if a.get('last_triggered')])
        
        # タイプ別集計
        type_counts = {}
        for alert in alerts:
            alert_type = alert['alert_type']
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        table = Table(title="📊 アラート統計", box=box.ROUNDED)
        table.add_column("項目", style="cyan")
        table.add_column("数", justify="right")
        
        table.add_row("総アラート数", str(total_alerts))
        table.add_row("アクティブ", str(active_alerts))
        table.add_row("発動済み", str(triggered_alerts))
        
        if type_counts:
            table.add_row("", "")  # 区切り線
            for alert_type, count in type_counts.items():
                display_type = alert_type.replace('_', ' ').title()
                table.add_row(display_type, str(count))
        
        return Panel(table, border_style="blue")
    
    def _switch_view(self, view: str):
        """表示切り替え"""
        self.current_view = view
        self._update_display()
    
    def _refresh(self):
        """手動更新"""
        self._update_data()
        self._update_display()
    
    def _show_help(self):
        """ヘルプ表示"""
        help_text = """
[bold]Day Trade Interactive Mode - ヘルプ[/bold]

[cyan]キーバインディング:[/cyan]
  1 - ダッシュボード表示
  2 - ウォッチリスト表示
  3 - ポートフォリオ表示
  4 - アラート表示
  R - データ手動更新
  H - このヘルプを表示
  Q - 終了

[cyan]機能:[/cyan]
  • リアルタイムデータ更新（30秒間隔）
  • ウォッチリスト管理
  • ポートフォリオ分析
  • アラート監視
  • 複数表示モード

[yellow]注意:[/yellow]
価格データの取得には制限があります。
デモモードでは一部の機能が制限されます。
        """
        
        self.console.print(Panel(help_text, title="ヘルプ", border_style="blue"))
        input("\nEnterキーで戻る...")
    
    def _quit(self):
        """終了処理"""
        self.running.clear()
    
    def _cleanup(self):
        """クリーンアップ"""
        self.running.clear()
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1)
        
        self.console.print("[green]インタラクティブモードを終了しました[/green]")


def main():
    """メイン関数"""
    try:
        interactive = InteractiveMode()
        interactive.run()
    except KeyboardInterrupt:
        print("\n終了しました")
    except Exception as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    main()