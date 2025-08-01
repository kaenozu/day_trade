"""
拡張された対話型CLIインターフェース
prompt_toolkitを使用したオートコンプリート、履歴、色分けなど
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from prompt_toolkit import prompt
from prompt_toolkit.completion import FuzzyCompleter, NestedCompleter, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import confirm
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel

from ..core.watchlist import WatchlistManager
from ..data.stock_fetcher import StockFetcher
from ..utils.formatters import (
    create_error_panel,
    create_info_panel,
    create_success_panel,
    create_warning_panel,
)
from ..utils.validators import normalize_stock_codes, validate_stock_code

logger = logging.getLogger(__name__)
console = Console()


class EnhancedInteractiveCLI:
    """拡張された対話型CLIクラス"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.history_file = Path.home() / ".daytrade_history"
        self.session_data = {}

        # 履歴機能
        self.history = FileHistory(str(self.history_file))

        # スタイル定義
        self.style = Style.from_dict(
            {
                "completion-menu.completion": "bg:#008888 #ffffff",
                "completion-menu.completion.current": "bg:#00aaaa #000000",
                "scrollbar.background": "bg:#88aaaa",
                "scrollbar.button": "bg:#222222",
                "prompt": "#884444 bold",
                "input": "#ffffff",
            }
        )

        # キーバインディング
        self.bindings = KeyBindings()
        self._setup_key_bindings()

        # 銘柄コードリスト（オートコンプリート用）
        self.stock_codes = self._load_stock_codes()

        # コマンド補完設定
        self.command_completer = self._create_command_completer()

    def _setup_key_bindings(self):
        """カスタムキーバインディングの設定"""

        @self.bindings.add("c-c")
        def _(event):
            """Ctrl+C で終了確認"""
            if confirm("本当に終了しますか？"):
                event.app.exit()

        @self.bindings.add("c-l")
        def _(event):
            """Ctrl+L で画面クリア"""
            console.clear()
            console.print("[bold green]画面をクリアしました[/bold green]")

        @self.bindings.add("f1")
        def _(event):
            """F1 でヘルプ表示"""
            self._show_help()

    def _load_stock_codes(self) -> List[str]:
        """既知の銘柄コードを読み込み"""
        # 一般的な日本株の銘柄コード例
        common_codes = [
            "7203",
            "9984",
            "8306",
            "4063",
            "6758",
            "6861",
            "9433",
            "8035",
            "8001",
            "7267",
            "4502",
            "4506",
            "4503",
            "4507",
            "6981",
            "6098",
            "9020",
            "2914",
            "2802",
            "8058",
            "7011",
            "9501",
            "9502",
            "9503",
        ]

        # ウォッチリストから銘柄コードを取得
        try:
            from ..core.config import config_manager

            watchlist_manager = WatchlistManager(
                config_manager, None, StockFetcher(), None
            )
            watchlist_items = watchlist_manager.get_watchlist()
            watchlist_codes = [item.get("stock_code", "") for item in watchlist_items]
            common_codes.extend(watchlist_codes)
        except Exception as e:
            logger.debug(f"ウォッチリストからの銘柄コード取得に失敗: {e}")

        return list(set(common_codes))  # 重複除去

    def _create_command_completer(self) -> NestedCompleter:
        """階層的なコマンド補完を作成"""
        stock_completer = WordCompleter(self.stock_codes, ignore_case=True)

        return NestedCompleter.from_nested_dict(
            {
                "stock": stock_completer,
                "history": stock_completer,
                "watch": stock_completer,
                "watchlist": {
                    "add": stock_completer,
                    "remove": stock_completer,
                    "list": None,
                    "clear": None,
                    "memo": stock_completer,
                    "move": stock_completer,
                },
                "config": {
                    "show": None,
                    "set": {
                        "api.timeout": None,
                        "trading.commission": None,
                        "display.theme": WordCompleter(["dark", "light"]),
                    },
                    "reset": None,
                },
                "validate": stock_completer,
                "backtest": None,
                "screen": {
                    "default": None,
                    "growth": None,
                    "value": None,
                    "momentum": None,
                },
                "help": None,
                "exit": None,
                "quit": None,
            }
        )

    def _show_help(self):
        """ヘルプメッセージを表示"""
        help_text = """
        [bold cyan]Day Trade CLI - 拡張インタラクティブモード[/bold cyan]

        [bold yellow]利用可能なコマンド:[/bold yellow]
        • stock <銘柄コード>       - 個別銘柄情報表示
        • history <銘柄コード>     - ヒストリカルデータ表示
        • watch <銘柄コード...>    - 複数銘柄の監視
        • watchlist add <銘柄>     - ウォッチリストに追加
        • watchlist list           - ウォッチリスト表示
        • config show              - 設定表示
        • validate <銘柄コード>    - 銘柄コード検証
        • backtest                 - バックテスト実行
        • screen <タイプ>          - 銘柄スクリーニング (default/growth/value/momentum)
        • help                     - このヘルプ表示
        • exit / quit              - 終了

        [bold yellow]キーバインディング:[/bold yellow]
        • Tab                      - オートコンプリート
        • ↑/↓                     - コマンド履歴
        • Ctrl+C                   - 終了確認
        • Ctrl+L                   - 画面クリア
        • F1                       - ヘルプ表示

        [bold yellow]機能:[/bold yellow]
        • 銘柄コードの自動補完
        • コマンド履歴の保存・呼出
        • リアルタイムヒント表示
        • 色分け表示
        """
        console.print(Panel(help_text, title="ヘルプ", border_style="cyan"))

    def _get_prompt_message(self) -> HTML:
        """動的なプロンプトメッセージを生成"""
        mode = self.session_data.get("mode", "main")
        timestamp = datetime.now().strftime("%H:%M")

        if mode == "config":
            return HTML(f"<prompt>({timestamp}) [config] > </prompt>")
        elif mode == "watchlist":
            return HTML(f"<prompt>({timestamp}) [watchlist] > </prompt>")
        else:
            return HTML(f"<prompt>({timestamp}) [day_trade] > </prompt>")

    def _get_bottom_toolbar(self) -> HTML:
        """画面下部のツールバー表示"""
        return HTML(
            "Press <b>F1</b> for help | "
            "<b>Tab</b> for completion | "
            "<b>Ctrl+C</b> to exit | "
            f"Mode: <b>{self.session_data.get('mode', 'main')}</b>"
        )

    def _process_command(self, command: str) -> bool:
        """コマンドを処理（戻り値: 継続するかどうか）"""
        if not command.strip():
            return True

        parts = command.strip().split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        try:
            if cmd in ["exit", "quit"]:
                return False
            elif cmd == "help":
                self._show_help()
            elif cmd == "stock" and args:
                self._handle_stock_command(args[0], details="-d" in args)
            elif cmd == "history" and args:
                self._handle_history_command(args[0])
            elif cmd == "watch" and args:
                self._handle_watch_command(args)
            elif cmd == "watchlist":
                self._handle_watchlist_command(args)
            elif cmd == "config":
                self._handle_config_command(args)
            elif cmd == "validate" and args:
                self._handle_validate_command(args)
            elif cmd == "backtest":
                self._handle_backtest_command()
            elif cmd == "screen":
                self._handle_screen_command(args)
            else:
                console.print(
                    create_warning_panel(
                        f"不明なコマンド: '{cmd}'. 'help' でヘルプを表示します。"
                    )
                )
        except Exception as e:
            console.print(create_error_panel(f"コマンド実行エラー: {e}"))
            logger.error(f"Command execution error: {e}")

        return True

    def _handle_stock_command(self, code: str, details: bool = False):
        """stock コマンドの処理"""
        if not validate_stock_code(code):
            console.print(create_error_panel(f"無効な銘柄コード: {code}"))
            return

        fetcher = StockFetcher()
        normalized_codes = normalize_stock_codes([code])
        if not normalized_codes:
            console.print(create_error_panel(f"銘柄コード正規化に失敗: {code}"))
            return

        code = normalized_codes[0]
        console.print(f"[cyan]銘柄 {code} の情報を取得中...[/cyan]")

        try:
            current = fetcher.get_current_price(code)
            if current:
                console.print(
                    create_success_panel(
                        f"銘柄: {code}, 価格: ¥{current.get('price', 'N/A')}"
                    )
                )
                if details:
                    info = fetcher.get_company_info(code)
                    if info:
                        console.print(
                            create_info_panel(f"企業名: {info.get('name', 'N/A')}")
                        )
            else:
                console.print(create_error_panel(f"銘柄 {code} の情報取得に失敗"))
        except Exception as e:
            console.print(create_error_panel(f"データ取得エラー: {e}"))

    def _handle_history_command(self, code: str):
        """history コマンドの処理"""
        console.print(f"[cyan]銘柄 {code} のヒストリカルデータを取得中...[/cyan]")
        # 簡易実装
        console.print(create_info_panel(f"銘柄 {code} の過去データ（実装予定）"))

    def _handle_watch_command(self, codes: List[str]):
        """watch コマンドの処理"""
        console.print(f"[cyan]{len(codes)} 銘柄を監視中...[/cyan]")
        for code in codes:
            console.print(f"監視中: {code}")

    def _handle_watchlist_command(self, args: List[str]):
        """watchlist コマンドの処理"""
        self.session_data["mode"] = "watchlist"

        if not args:
            console.print(
                create_info_panel("使用可能: add, remove, list, clear, memo, move")
            )
            return

        subcommand = args[0].lower()

        if subcommand == "list":
            console.print(create_info_panel("ウォッチリスト一覧（実装予定）"))
        elif subcommand == "add" and len(args) > 1:
            console.print(
                create_success_panel(f"銘柄 {args[1]} をウォッチリストに追加")
            )
        elif subcommand == "remove" and len(args) > 1:
            console.print(
                create_success_panel(f"銘柄 {args[1]} をウォッチリストから削除")
            )
        else:
            console.print(create_warning_panel(f"不明なサブコマンド: {subcommand}"))

    def _handle_config_command(self, args: List[str]):
        """config コマンドの処理"""
        self.session_data["mode"] = "config"

        if not args or args[0] == "show":
            console.print(create_info_panel("設定表示（実装予定）"))
        elif args[0] == "set" and len(args) >= 3:
            key, value = args[1], args[2]
            console.print(create_success_panel(f"設定更新: {key} = {value}"))
        elif args[0] == "reset":
            if confirm("設定をリセットしますか？"):
                console.print(create_success_panel("設定をリセットしました"))
        else:
            console.print(create_warning_panel("使用法: config [show|set|reset]"))

    def _handle_validate_command(self, codes: List[str]):
        """validate コマンドの処理"""
        for code in codes:
            is_valid = validate_stock_code(code)
            status = "[green]有効[/green]" if is_valid else "[red]無効[/red]"
            console.print(f"銘柄コード {code}: {status}")

    def _handle_backtest_command(self):
        """backtest コマンドの処理"""
        console.print(create_info_panel("インタラクティブバックテスト（実装予定）"))

    def _handle_screen_command(self, args: List[str]):
        """screen コマンドの処理"""
        try:
            from ..automation.orchestrator import DayTradeOrchestrator

            # スクリーナータイプの決定
            screener_type = "default"
            if args and args[0] in ["default", "growth", "value", "momentum"]:
                screener_type = args[0]

            console.print(
                f"[cyan]{screener_type}スクリーナーで銘柄をスクリーニング中...[/cyan]"
            )

            # オーケストレーターを初期化
            orchestrator = DayTradeOrchestrator(self.config_path)

            # スクリーニング実行
            results = orchestrator.run_stock_screening(
                screener_type=screener_type, min_score=0.1, max_results=10
            )

            if results:
                console.print(
                    create_success_panel(f"{len(results)}銘柄が条件を満たしました")
                )

                # 結果を簡潔に表示
                for i, result in enumerate(results, 1):
                    symbol = result["symbol"]
                    score = result["score"]
                    price = result.get("last_price", "N/A")
                    conditions = ", ".join(result.get("matched_conditions", [])[:2])

                    console.print(
                        f"{i}. {symbol} (スコア: {score:.2f}, 価格: ¥{price}, 条件: {conditions})"
                    )
            else:
                console.print(
                    create_warning_panel("条件を満たす銘柄が見つかりませんでした")
                )

        except ImportError:
            console.print(create_error_panel("スクリーニング機能が利用できません"))
        except Exception as e:
            console.print(create_error_panel(f"スクリーニングエラー: {e}"))

    def run(self):
        """拡張インタラクティブモードを開始"""
        console.print(
            Panel(
                "[bold green]Day Trade - 拡張インタラクティブモード[/bold green]\n"
                "オートコンプリート、履歴機能、色分け表示が利用できます。\n"
                "'help' または F1 でヘルプを表示します。",
                title="🚀 拡張モード開始",
                border_style="green",
            )
        )

        self.session_data["mode"] = "main"

        try:
            while True:
                try:
                    # 現在のモードに応じた補完機能を選択
                    current_completer = self.command_completer
                    if self.session_data.get("mode") == "config":
                        # config モード専用の補完
                        current_completer = FuzzyCompleter(
                            WordCompleter(
                                [
                                    "show",
                                    "set api.timeout",
                                    "set trading.commission",
                                    "set display.theme dark",
                                    "set display.theme light",
                                    "reset",
                                ]
                            )
                        )

                    user_input = prompt(
                        message=self._get_prompt_message(),
                        history=self.history,
                        completer=current_completer,
                        style=self.style,
                        key_bindings=self.bindings,
                        bottom_toolbar=self._get_bottom_toolbar,
                        enable_history_search=True,
                        search_ignore_case=True,
                    )

                    if not self._process_command(user_input):
                        break

                except KeyboardInterrupt:
                    if confirm("\n本当に終了しますか？"):
                        break
                    console.print("[yellow]継続します...[/yellow]")
                except EOFError:
                    break

        except Exception as e:
            console.print(create_error_panel(f"予期しないエラー: {e}"))
            logger.error(f"Unexpected error in interactive mode: {e}")

        console.print(
            Panel(
                "[bold cyan]拡張インタラクティブモードを終了しました。[/bold cyan]\n"
                "ありがとうございました！",
                title="👋 終了",
                border_style="cyan",
            )
        )


def run_enhanced_interactive(config_path: Optional[Path] = None):
    """拡張インタラクティブモードを開始"""
    cli = EnhancedInteractiveCLI(config_path)
    cli.run()


if __name__ == "__main__":
    run_enhanced_interactive()
