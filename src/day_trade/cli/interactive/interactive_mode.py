"""
対話型UIクラス
InteractiveModeクラスとその関連機能を提供
"""

import logging
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...core.watchlist import WatchlistManager
from ...data.stock_fetcher import StockFetcher
from ...utils.formatters import (
    create_error_panel,
    create_info_panel,
    create_success_panel,
    create_warning_panel,
)
from ...utils.validators import validate_stock_code
from .helpers import display_stock_details, get_watchlist_manager

logger = logging.getLogger(__name__)


class InteractiveMode:
    """対話型モードクラス（依存性注入対応）"""

    def __init__(
        self,
        watchlist_manager: Optional[WatchlistManager] = None,
        stock_fetcher: Optional[StockFetcher] = None,
        trade_manager=None,
        signal_generator=None,
        console: Optional[Console] = None,
    ):
        """
        初期化（依存性注入対応）

        Args:
            watchlist_manager: ウォッチリスト管理インスタンス
            stock_fetcher: 株価データ取得インスタンス
            trade_manager: 取引管理インスタンス
            signal_generator: シグナル生成インスタンス
            console: Rich Console インスタンス
        """
        self.console = console or Console()
        self.watchlist_manager = watchlist_manager or get_watchlist_manager()
        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.trade_manager = trade_manager
        self.signal_generator = signal_generator

        # UI状態管理
        self._current_layout = None
        self._background_update_running = False
        self._update_interval = 5.0  # 秒

        # データキャッシュ
        self._cached_data = {}
        self._last_update = None

    def start(self):
        """対話型モードを開始"""
        self.console.print(
            Panel(
                "[bold green]Day Trade Interactive Mode[/bold green]\n"
                "対話型モードを開始します。\n"
                "[dim]'help' でコマンド一覧を表示[/dim]",
                title="🚀 起動完了",
                border_style="green",
            )
        )
        self._display_welcome_info()

    def _display_welcome_info(self):
        """ウェルカム情報を表示"""
        info_table = Table(title="利用可能な機能")
        info_table.add_column("コマンド", style="cyan", no_wrap=True)
        info_table.add_column("説明", style="white")

        commands = [
            ("stock <code>", "銘柄情報を表示"),
            ("watch <code>", "ウォッチリストに追加"),
            ("watchlist", "ウォッチリスト表示"),
            ("portfolio", "ポートフォリオ情報表示"),
            ("signals <code>", "売買シグナル分析"),
            ("help", "ヘルプ表示"),
            ("exit", "終了"),
        ]

        for cmd, desc in commands:
            info_table.add_row(cmd, desc)

        self.console.print(info_table)

    def handle_command(self, command: str) -> bool:
        """
        コマンドを処理（拡張版）

        Args:
            command: 実行するコマンド

        Returns:
            bool: 続行する場合True、終了する場合False
        """
        command = command.strip()
        if not command:
            return True

        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        try:
            if cmd in ["exit", "quit", "q"]:
                return False
            elif cmd == "help":
                self._show_help()
            elif cmd == "stock" and args:
                self._handle_stock_command(args[0])
            elif cmd == "watch" and args:
                self._handle_watch_command(args[0])
            elif cmd == "watchlist":
                self._handle_watchlist_command()
            elif cmd == "portfolio":
                self._handle_portfolio_command()
            elif cmd == "signals" and args:
                self._handle_signals_command(args[0])
            elif cmd == "clear":
                self.console.clear()
            else:
                self._show_unknown_command(command)

        except Exception as e:
            self._handle_command_error(command, e)

        return True

    def _handle_stock_command(self, stock_code: str):
        """株式情報コマンドを処理"""
        if not validate_stock_code(stock_code):
            self.console.print(
                create_error_panel(
                    f"無効な銘柄コード: {stock_code}", title="入力エラー"
                )
            )
            return

        try:
            with self.console.status(f"[bold blue]{stock_code}の情報を取得中..."):
                current_price = self.stock_fetcher.get_current_price(stock_code)

            if current_price:
                display_stock_details(stock_code, current_price, show_details=True)
            else:
                self.console.print(
                    create_warning_panel(
                        f"銘柄 {stock_code} の情報を取得できませんでした",
                        title="データ取得警告",
                    )
                )

        except Exception as e:
            self.console.print(
                create_error_panel(f"エラー: {str(e)}", title="株式情報取得エラー")
            )

    def _handle_watch_command(self, stock_code: str):
        """ウォッチリスト追加コマンドを処理"""
        try:
            # ウォッチリストに追加のロジック（実装に応じて調整）
            self.console.print(
                create_success_panel(
                    f"銘柄 {stock_code} をウォッチリストに追加しました",
                    title="追加完了",
                )
            )
        except Exception as e:
            self.console.print(
                create_error_panel(
                    f"ウォッチリスト追加エラー: {str(e)}", title="追加失敗"
                )
            )

    def _handle_watchlist_command(self):
        """ウォッチリスト表示コマンドを処理"""
        try:
            # ウォッチリスト表示のロジック（実装に応じて調整）
            watchlist_table = Table(title="📋 ウォッチリスト")
            watchlist_table.add_column("銘柄コード", style="cyan")
            watchlist_table.add_column("銘柄名", style="white")
            watchlist_table.add_column("現在価格", style="green")

            # サンプルデータ（実際の実装では watchlist_manager から取得）
            watchlist_table.add_row("7203", "トヨタ自動車", "¥2,456")
            watchlist_table.add_row("9984", "ソフトバンクグループ", "¥5,128")

            self.console.print(watchlist_table)

        except Exception as e:
            self.console.print(
                create_error_panel(
                    f"ウォッチリスト表示エラー: {str(e)}", title="表示失敗"
                )
            )

    def _handle_portfolio_command(self):
        """ポートフォリオ表示コマンドを処理"""
        self.console.print(
            create_info_panel("ポートフォリオ機能は開発中です", title="機能開発中")
        )

    def _handle_signals_command(self, stock_code: str):
        """シグナル分析コマンドを処理"""
        if self.signal_generator:
            try:
                # シグナル生成のロジック（実装に応じて調整）
                self.console.print(
                    create_info_panel(
                        f"銘柄 {stock_code} のシグナル分析を実行中...",
                        title="シグナル分析",
                    )
                )
            except Exception as e:
                self.console.print(
                    create_error_panel(
                        f"シグナル分析エラー: {str(e)}", title="分析失敗"
                    )
                )
        else:
            self.console.print(
                create_warning_panel(
                    "シグナル生成機能が利用できません", title="機能無効"
                )
            )

    def _show_help(self):
        """ヘルプを表示"""
        help_panel = Panel(
            "[bold cyan]利用可能なコマンド:[/bold cyan]\n\n"
            "[yellow]stock <code>[/yellow] - 銘柄情報を表示\n"
            "[yellow]watch <code>[/yellow] - ウォッチリストに追加\n"
            "[yellow]watchlist[/yellow] - ウォッチリスト表示\n"
            "[yellow]portfolio[/yellow] - ポートフォリオ表示\n"
            "[yellow]signals <code>[/yellow] - シグナル分析\n"
            "[yellow]clear[/yellow] - 画面クリア\n"
            "[yellow]help[/yellow] - このヘルプを表示\n"
            "[yellow]exit/quit/q[/yellow] - 終了\n\n"
            "[dim]例: stock 7203, watch 9984[/dim]",
            title="📖 ヘルプ",
            border_style="blue",
        )
        self.console.print(help_panel)

    def _show_unknown_command(self, command: str):
        """不明なコマンドを表示"""
        self.console.print(
            create_warning_panel(
                f"不明なコマンド: '{command}'\n"
                "'help' でコマンド一覧を確認してください",
                title="コマンドエラー",
            )
        )

    def _handle_command_error(self, command: str, error: Exception):
        """コマンド実行エラーを処理"""
        logger.error(f"Command execution error for '{command}': {error}")
        self.console.print(
            create_error_panel(
                f"コマンド '{command}' の実行中にエラーが発生しました:\n{str(error)}",
                title="実行エラー",
            )
        )

    def stop(self):
        """対話型モードを停止"""
        self._background_update_running = False
        self.console.print(
            Panel(
                "[bold red]対話型モードを終了します[/bold red]\n"
                "[dim]お疲れ様でした！[/dim]",
                title="👋 終了",
                border_style="red",
            )
        )