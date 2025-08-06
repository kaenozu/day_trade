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

from ..automation.orchestrator import DayTradeOrchestrator
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


class CommandHandler:
    """コマンドハンドラーの基底クラス"""

    def __init__(self, cli_instance):
        self.cli = cli_instance

    def can_handle(self, command: str) -> bool:
        """このハンドラーがコマンドを処理できるかどうか"""
        raise NotImplementedError

    def handle(self, command: str, args: List[str]) -> bool:
        """コマンドを処理（戻り値: 継続するかどうか）"""
        raise NotImplementedError


class ExitCommandHandler(CommandHandler):
    """終了コマンドハンドラー"""

    def can_handle(self, command: str) -> bool:
        return command in ["exit", "quit"]

    def handle(self, command: str, args: List[str]) -> bool:
        return False


class HelpCommandHandler(CommandHandler):
    """ヘルプコマンドハンドラー"""

    def can_handle(self, command: str) -> bool:
        return command == "help"

    def handle(self, command: str, args: List[str]) -> bool:
        self.cli._show_help()
        return True


class StockCommandHandler(CommandHandler):
    """株式関連コマンドハンドラー"""

    def can_handle(self, command: str) -> bool:
        return command in ["stock", "history", "watch", "validate"]

    def handle(self, command: str, args: List[str]) -> bool:
        if command == "stock" and args:
            self.cli._handle_stock_command(args[0], details="-d" in args)
        elif command == "history" and args:
            self.cli._handle_history_command(args[0])
        elif command == "watch" and args:
            self.cli._handle_watch_command(args)
        elif command == "validate" and args:
            self.cli._handle_validate_command(args)
        else:
            console.print(
                create_warning_panel(f"'{command}' コマンドには引数が必要です")
            )
        return True


class WatchlistCommandHandler(CommandHandler):
    """ウォッチリストコマンドハンドラー"""

    def can_handle(self, command: str) -> bool:
        return command == "watchlist"

    def handle(self, command: str, args: List[str]) -> bool:
        self.cli._handle_watchlist_command(args)
        return True


class ConfigCommandHandler(CommandHandler):
    """設定コマンドハンドラー"""

    def can_handle(self, command: str) -> bool:
        return command == "config"

    def handle(self, command: str, args: List[str]) -> bool:
        self.cli._handle_config_command(args)
        return True


class AnalysisCommandHandler(CommandHandler):
    """分析関連コマンドハンドラー"""

    def can_handle(self, command: str) -> bool:
        return command in ["backtest", "screen"]

    def handle(self, command: str, args: List[str]) -> bool:
        if command == "backtest":
            self.cli._handle_backtest_command()
        elif command == "screen":
            self.cli._handle_screen_command(args)
        return True


class EnhancedInteractiveCLI:
    """拡張された対話型CLIクラス"""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        stock_fetcher: Optional["StockFetcher"] = None,
        watchlist_manager: Optional["WatchlistManager"] = None,
        orchestrator: Optional["DayTradeOrchestrator"] = None,
    ):
        self.config_path = config_path
        self.history_file = Path.home() / ".daytrade_history"
        self.session_data = {}

        # 依存関係の注入（遅延初期化対応）
        self._stock_fetcher = stock_fetcher
        self._watchlist_manager = watchlist_manager
        self._orchestrator = orchestrator

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

        # コマンドハンドラーの初期化
        self.command_handlers = [
            ExitCommandHandler(self),
            HelpCommandHandler(self),
            StockCommandHandler(self),
            WatchlistCommandHandler(self),
            ConfigCommandHandler(self),
            AnalysisCommandHandler(self),
        ]

    @property
    def stock_fetcher(self) -> "StockFetcher":
        """StockFetcherインスタンスを取得（遅延初期化）"""
        if self._stock_fetcher is None:
            self._stock_fetcher = StockFetcher()
        return self._stock_fetcher

    @property
    def watchlist_manager(self) -> "WatchlistManager":
        """WatchlistManagerインスタンスを取得（遅延初期化）"""
        if self._watchlist_manager is None:
            from ..core.config import config_manager

            self._watchlist_manager = WatchlistManager(
                config_manager, None, self.stock_fetcher, None
            )
        return self._watchlist_manager

    @property
    def orchestrator(self) -> "DayTradeOrchestrator":
        """DayTradeOrchestratorインスタンスを取得（遅延初期化）"""
        if self._orchestrator is None:
            from ..automation.orchestrator import DayTradeOrchestrator

            self._orchestrator = DayTradeOrchestrator(self.config_path)
        return self._orchestrator

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
            watchlist_items = self.watchlist_manager.get_watchlist()
            watchlist_codes = [item.get("stock_code", "") for item in watchlist_items]
            common_codes.extend(watchlist_codes)
        except Exception as e:
            logger.warning(f"ウォッチリストからの銘柄コード取得に失敗: {e}")

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
            # 適切なハンドラーを検索
            for handler in self.command_handlers:
                if handler.can_handle(cmd):
                    return handler.handle(cmd, args)

            # 不明なコマンド
            console.print(
                create_warning_panel(
                    f"不明なコマンド: '{cmd}'. 'help' でヘルプを表示します。"
                )
            )

        except Exception as e:
            console.print(create_error_panel(f"コマンド実行エラー: {e}"))
            logger.error(f"Command execution error: {e}")

        return True

    def _handle_error(self, error: Exception, context: str = "操作") -> None:
        """
        統一的なエラーハンドリング

        Args:
            error: 例外オブジェクト
            context: エラーが発生した文脈
        """
        error_type = type(error).__name__

        # 一般的なエラーの解決策を提供
        solutions = []

        if "ConnectionError" in error_type or "timeout" in str(error).lower():
            solutions = [
                "インターネット接続を確認してください",
                "VPNまたはプロキシの設定を確認してください",
                "少し時間をおいて再試行してください",
            ]
        elif "FileNotFoundError" in error_type:
            solutions = [
                "ファイルパスが正しいか確認してください",
                "ファイルが存在するか確認してください",
                "権限があるか確認してください",
            ]
        elif "ImportError" in error_type or "ModuleNotFoundError" in error_type:
            solutions = [
                "必要なライブラリがインストールされているか確認してください",
                "pip install -e .[dev] を実行してください",
                "仮想環境が正しく有効化されているか確認してください",
            ]
        elif "ValueError" in error_type or "TypeError" in error_type:
            solutions = [
                "入力データの形式を確認してください",
                "必要な引数がすべて提供されているか確認してください",
            ]
        elif "KeyError" in error_type:
            solutions = [
                "設定ファイルに必要なキーが存在するか確認してください",
                "データが正しく初期化されているか確認してください",
            ]

        if solutions:
            console.print(
                create_error_panel(
                    f"{context}中にエラーが発生しました: {error}", solutions=solutions
                )
            )
        else:
            console.print(
                create_error_panel(f"{context}中にエラーが発生しました: {error}")
            )

        logger.error(f"Error in {context}: {error_type}: {error}")

    def _validate_input(self, value: str, validation_type: str) -> bool:
        """
        入力値の検証

        Args:
            value: 検証する値
            validation_type: 検証タイプ

        Returns:
            検証結果
        """
        if not value or not value.strip():
            console.print(create_warning_panel("値が入力されていません"))
            return False

        if validation_type == "stock_code":
            if not validate_stock_code(value):
                console.print(create_warning_panel(f"無効な銘柄コード: {value}"))
                return False
        elif validation_type == "numeric":
            try:
                float(value)
            except ValueError:
                console.print(create_warning_panel(f"数値ではありません: {value}"))
                return False
        elif validation_type == "integer":
            try:
                int(value)
            except ValueError:
                console.print(create_warning_panel(f"整数ではありません: {value}"))
                return False

        return True

    def _handle_stock_command(self, code: str, details: bool = False):
        """stock コマンドの処理"""
        if not validate_stock_code(code):
            console.print(create_error_panel(f"無効な銘柄コード: {code}"))
            return

        normalized_codes = normalize_stock_codes([code])
        if not normalized_codes:
            console.print(create_error_panel(f"銘柄コード正規化に失敗: {code}"))
            return

        code = normalized_codes[0]
        console.print(f"[cyan]銘柄 {code} の情報を取得中...[/cyan]")

        try:
            current = self.stock_fetcher.get_current_price(code)
            if current:
                console.print(
                    create_success_panel(
                        f"銘柄: {code}, 価格: ¥{current.get('price', 'N/A')}"
                    )
                )
                if details:
                    info = self.stock_fetcher.get_company_info(code)
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

        try:
            if subcommand == "list":
                self._show_watchlist()
            elif subcommand == "add" and len(args) > 1:
                self._add_to_watchlist(
                    args[1], memo=" ".join(args[2:]) if len(args) > 2 else None
                )
            elif subcommand == "remove" and len(args) > 1:
                self._remove_from_watchlist(args[1])
            elif subcommand == "clear":
                self._clear_watchlist()
            elif subcommand == "memo" and len(args) > 2:
                self._update_watchlist_memo(args[1], " ".join(args[2:]))
            else:
                console.print(
                    create_warning_panel(
                        f"不明なサブコマンドまたは引数不足: {subcommand}"
                    )
                )
        except Exception as e:
            console.print(create_error_panel(f"ウォッチリスト操作エラー: {e}"))
            logger.error(f"Watchlist operation error: {e}")

    def _show_watchlist(self):
        """ウォッチリスト一覧表示"""
        items = self.watchlist_manager.get_watchlist()
        if not items:
            console.print(create_info_panel("ウォッチリストは空です"))
            return

        console.print("\n[bold cyan]📋 ウォッチリスト[/bold cyan]")
        for i, item in enumerate(items, 1):
            stock_code = item.get("stock_code", "N/A")
            memo = item.get("memo", "")
            memo_text = f" - {memo}" if memo else ""
            console.print(f"{i}. [yellow]{stock_code}[/yellow]{memo_text}")

        console.print(f"\n合計: {len(items)}銘柄")

    def _add_to_watchlist(self, stock_code: str, memo: Optional[str] = None):
        """ウォッチリストに銘柄を追加"""
        if not validate_stock_code(stock_code):
            console.print(create_error_panel(f"無効な銘柄コード: {stock_code}"))
            return

        normalized_codes = normalize_stock_codes([stock_code])
        if not normalized_codes:
            console.print(create_error_panel(f"銘柄コード正規化に失敗: {stock_code}"))
            return

        stock_code = normalized_codes[0]

        # 既存チェック
        existing_items = self.watchlist_manager.get_watchlist()
        if any(item.get("stock_code") == stock_code for item in existing_items):
            console.print(
                create_warning_panel(
                    f"銘柄 {stock_code} は既にウォッチリストに存在します"
                )
            )
            return

        # 追加実行
        success = self.watchlist_manager.add_stock(stock_code, memo=memo)
        if success:
            memo_text = f" (メモ: {memo})" if memo else ""
            console.print(
                create_success_panel(
                    f"銘柄 {stock_code} をウォッチリストに追加しました{memo_text}"
                )
            )
            # 補完用銘柄コードリストを更新
            self.stock_codes = self._load_stock_codes()
            self.command_completer = self._create_command_completer()
        else:
            console.print(create_error_panel(f"銘柄 {stock_code} の追加に失敗しました"))

    def _remove_from_watchlist(self, stock_code: str):
        """ウォッチリストから銘柄を削除"""
        normalized_codes = normalize_stock_codes([stock_code])
        if normalized_codes:
            stock_code = normalized_codes[0]

        success = self.watchlist_manager.remove_stock(stock_code)
        if success:
            console.print(
                create_success_panel(
                    f"銘柄 {stock_code} をウォッチリストから削除しました"
                )
            )
            # 補完用銘柄コードリストを更新
            self.stock_codes = self._load_stock_codes()
            self.command_completer = self._create_command_completer()
        else:
            console.print(
                create_warning_panel(
                    f"銘柄 {stock_code} はウォッチリストに存在しません"
                )
            )

    def _clear_watchlist(self):
        """ウォッチリストをクリア"""
        from prompt_toolkit.shortcuts import confirm

        if confirm("ウォッチリストをすべてクリアしますか？"):
            self.watchlist_manager.clear_watchlist()
            console.print(create_success_panel("ウォッチリストをクリアしました"))
            # 補完用銘柄コードリストを更新
            self.stock_codes = self._load_stock_codes()
            self.command_completer = self._create_command_completer()
        else:
            console.print(create_info_panel("キャンセルしました"))

    def _update_watchlist_memo(self, stock_code: str, memo: str):
        """ウォッチリストのメモを更新"""
        normalized_codes = normalize_stock_codes([stock_code])
        if normalized_codes:
            stock_code = normalized_codes[0]

        # 既存チェック
        existing_items = self.watchlist_manager.get_watchlist()
        if not any(item.get("stock_code") == stock_code for item in existing_items):
            console.print(
                create_warning_panel(
                    f"銘柄 {stock_code} はウォッチリストに存在しません"
                )
            )
            return

        # メモ更新（add_stockを使って既存を上書き）
        success = self.watchlist_manager.add_stock(stock_code, memo=memo)
        if success:
            console.print(
                create_success_panel(f"銘柄 {stock_code} のメモを更新しました: {memo}")
            )
        else:
            console.print(
                create_error_panel(f"銘柄 {stock_code} のメモ更新に失敗しました")
            )

    def _handle_config_command(self, args: List[str]):
        """config コマンドの処理"""
        self.session_data["mode"] = "config"

        try:
            if not args or args[0] == "show":
                self._show_config()
            elif args[0] == "set" and len(args) >= 3:
                key, value = args[1], args[2]
                self._set_config(key, value)
            elif args[0] == "reset":
                self._reset_config()
            else:
                console.print(create_warning_panel("使用法: config [show|set|reset]"))
        except Exception as e:
            console.print(create_error_panel(f"設定操作エラー: {e}"))
            logger.error(f"Config operation error: {e}")

    def _show_config(self):
        """設定表示"""
        try:
            from ..core.config import config_manager

            console.print("\n[bold cyan]⚙️ 現在の設定[/bold cyan]")

            # API設定
            api_timeout = getattr(config_manager, "api_timeout", 30)
            console.print("[yellow]API設定[/yellow]")
            console.print(f"  timeout: {api_timeout}秒")

            # 取引設定
            commission = getattr(config_manager, "commission", 0.0)
            console.print("[yellow]取引設定[/yellow]")
            console.print(f"  commission: {commission}%")

            # 表示設定
            theme = getattr(config_manager, "theme", "dark")
            console.print("[yellow]表示設定[/yellow]")
            console.print(f"  theme: {theme}")

            # データベース設定
            db_path = getattr(config_manager, "database_path", "day_trade.db")
            console.print("[yellow]データベース設定[/yellow]")
            console.print(f"  database_path: {db_path}")

        except Exception as e:
            console.print(create_error_panel(f"設定読み込みエラー: {e}"))

    def _set_config(self, key: str, value: str):
        """設定更新"""
        try:
            from ..core.config import config_manager

            # 設定キーの検証と変換
            valid_keys = {
                "api.timeout": ("api_timeout", int),
                "trading.commission": ("commission", float),
                "display.theme": ("theme", str),
                "database.path": ("database_path", str),
            }

            if key not in valid_keys:
                console.print(create_error_panel(f"無効な設定キー: {key}"))
                console.print(
                    create_info_panel(f"有効なキー: {', '.join(valid_keys.keys())}")
                )
                return

            attr_name, value_type = valid_keys[key]

            # 値の型変換
            try:
                if value_type is int:
                    converted_value = int(value)
                elif value_type is float:
                    converted_value = float(value)
                else:
                    converted_value = value

                # テーマの検証
                if key == "display.theme" and converted_value not in ["dark", "light"]:
                    console.print(
                        create_error_panel(
                            "テーマは 'dark' または 'light' を指定してください"
                        )
                    )
                    return

            except ValueError:
                console.print(
                    create_error_panel(
                        f"無効な値の型: {value} (期待する型: {value_type.__name__})"
                    )
                )
                return

            # 設定更新
            setattr(config_manager, attr_name, converted_value)
            console.print(
                create_success_panel(f"設定を更新しました: {key} = {converted_value}")
            )

            # 設定保存の試行
            if hasattr(config_manager, "save"):
                config_manager.save()
                console.print(create_info_panel("設定をファイルに保存しました"))

        except Exception as e:
            console.print(create_error_panel(f"設定更新エラー: {e}"))

    def _reset_config(self):
        """設定リセット"""
        from prompt_toolkit.shortcuts import confirm

        if confirm("設定をデフォルト値にリセットしますか？"):
            try:
                from ..core.config import config_manager

                # デフォルト値に戻す
                config_manager.api_timeout = 30
                config_manager.commission = 0.0
                config_manager.theme = "dark"
                config_manager.database_path = "day_trade.db"

                if hasattr(config_manager, "save"):
                    config_manager.save()

                console.print(
                    create_success_panel("設定をデフォルト値にリセットしました")
                )

            except Exception as e:
                console.print(create_error_panel(f"設定リセットエラー: {e}"))
        else:
            console.print(create_info_panel("キャンセルしました"))

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
            # スクリーナータイプの決定
            screener_type = "default"
            if args and args[0] in ["default", "growth", "value", "momentum"]:
                screener_type = args[0]

            console.print(
                f"[cyan]{screener_type}スクリーナーで銘柄をスクリーニング中...[/cyan]"
            )

            # スクリーニング実行
            results = self.orchestrator.run_stock_screening(
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
