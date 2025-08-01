"""
インタラクティブモードのテスト
"""

import threading
import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.day_trade.cli.interactive import InteractiveMode


class TestInteractiveMode:
    """InteractiveModeクラスのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        # 各種マネージャーをモック化
        self.mock_watchlist_manager = Mock()
        self.mock_trade_manager = Mock()
        self.mock_stock_fetcher = Mock()
        self.mock_signal_generator = Mock()

        with patch(
            "src.day_trade.cli.interactive.WatchlistManager",
            return_value=self.mock_watchlist_manager,
        ):
            with patch(
                "src.day_trade.cli.interactive.TradeManager",
                return_value=self.mock_trade_manager,
            ):
                with patch(
                    "src.day_trade.cli.interactive.StockFetcher",
                    return_value=self.mock_stock_fetcher,
                ):
                    with patch(
                        "src.day_trade.cli.interactive.TradingSignalGenerator",
                        return_value=self.mock_signal_generator,
                    ):
                        self.interactive = InteractiveMode()

    def test_initialization(self):
        """初期化テスト"""
        assert self.interactive.console is not None
        assert self.interactive.layout is not None
        assert self.interactive.current_view == "dashboard"
        assert self.interactive.update_interval == 30
        assert not self.interactive.running.is_set()
        assert "q" in self.interactive.keybindings
        assert "1" in self.interactive.keybindings

    def test_setup_layout(self):
        """レイアウトセットアップテスト"""
        # レイアウト構造の確認
        assert "header" in self.interactive.layout
        assert "main" in self.interactive.layout
        assert "footer" in self.interactive.layout
        assert "left" in self.interactive.layout["main"]
        assert "right" in self.interactive.layout["main"]
        assert "primary" in self.interactive.layout["left"]
        assert "secondary" in self.interactive.layout["left"]

    def test_load_initial_data_with_existing_watchlist(self):
        """既存ウォッチリストありの初期データ読み込みテスト"""
        # モックデータを設定
        mock_watchlist = [
            {"stock_code": "7203", "stock_name": "トヨタ自動車", "group_name": "main"},
            {"stock_code": "8306", "stock_name": "三菱UFJ銀行", "group_name": "bank"},
        ]
        mock_alerts = [
            {
                "id": 1,
                "stock_code": "7203",
                "alert_type": "price_above",
                "threshold": 3000,
            }
        ]
        mock_portfolio = {"total_positions": 2, "total_trades": 5}

        self.mock_watchlist_manager.get_watchlist.return_value = mock_watchlist
        self.mock_watchlist_manager.get_alerts.return_value = mock_alerts
        self.mock_trade_manager.get_portfolio_summary.return_value = mock_portfolio

        self.interactive._load_initial_data()

        assert len(self.interactive.current_data["watchlist"]) == 2
        assert len(self.interactive.current_data["alerts"]) == 1
        assert (
            self.interactive.current_data["portfolio_summary"]["total_positions"] == 2
        )
        assert "last_update" in self.interactive.current_data

    def test_load_initial_data_empty_watchlist(self):
        """空ウォッチリストの初期データ読み込みテスト"""
        # 最初は空のウォッチリスト、サンプル作成後にデータありに設定
        mock_empty_watchlist = []
        mock_sample_watchlist = [
            {"stock_code": "7203", "stock_name": "トヨタ自動車", "group_name": "主力株"}
        ]

        self.mock_watchlist_manager.get_watchlist.side_effect = [
            mock_empty_watchlist,  # 最初の呼び出し
            mock_sample_watchlist,  # サンプル作成後の呼び出し
        ]
        self.mock_watchlist_manager.get_alerts.return_value = []
        self.mock_trade_manager.get_portfolio_summary.return_value = {}

        self.interactive._load_initial_data()

        # サンプルデータ作成のためのadd_stockが呼ばれることを確認
        assert (
            self.mock_watchlist_manager.add_stock.call_count == 3
        )  # 3つのサンプル銘柄
        assert len(self.interactive.current_data["watchlist"]) == 1

    def test_create_sample_data(self):
        """サンプルデータ作成テスト"""
        self.interactive._create_sample_data()

        # 3つのサンプル銘柄が追加されることを確認
        assert self.mock_watchlist_manager.add_stock.call_count == 3

        # 呼び出された引数を確認
        calls = self.mock_watchlist_manager.add_stock.call_args_list
        stock_codes = [call[0][0] for call in calls]  # 第1引数（stock_code）を取得

        assert "7203" in stock_codes
        assert "8306" in stock_codes
        assert "9984" in stock_codes

    def test_update_data(self):
        """データ更新テスト"""
        # モックデータを設定
        mock_watchlist = [{"stock_code": "7203"}]
        mock_alerts = [{"id": 1, "stock_code": "7203"}]
        mock_portfolio = {"total_positions": 1}

        self.mock_watchlist_manager.get_watchlist.return_value = mock_watchlist
        self.mock_watchlist_manager.get_alerts.return_value = mock_alerts
        self.mock_trade_manager.get_portfolio_summary.return_value = mock_portfolio

        # データ更新実行
        self.interactive._update_data()

        # データが更新されていることを確認
        assert self.interactive.current_data["watchlist"] == mock_watchlist
        assert self.interactive.current_data["alerts"] == mock_alerts
        assert self.interactive.current_data["portfolio_summary"] == mock_portfolio
        assert "last_update" in self.interactive.current_data
        assert isinstance(self.interactive.current_data["last_update"], datetime)

    def test_switch_view(self):
        """表示切り替えテスト"""
        # 初期状態はdashboard
        assert self.interactive.current_view == "dashboard"

        # watchlistに切り替え
        self.interactive._switch_view("watchlist")
        assert self.interactive.current_view == "watchlist"

        # portfolioに切り替え
        self.interactive._switch_view("portfolio")
        assert self.interactive.current_view == "portfolio"

        # alertsに切り替え
        self.interactive._switch_view("alerts")
        assert self.interactive.current_view == "alerts"

    def test_keybindings(self):
        """キーバインディングテスト"""
        # 表示切り替えキー
        self.interactive.keybindings["1"]()  # dashboard
        assert self.interactive.current_view == "dashboard"

        self.interactive.keybindings["2"]()  # watchlist
        assert self.interactive.current_view == "watchlist"

        self.interactive.keybindings["3"]()  # portfolio
        assert self.interactive.current_view == "portfolio"

        self.interactive.keybindings["4"]()  # alerts
        assert self.interactive.current_view == "alerts"

        # 終了キー
        assert not self.interactive.running.is_set()
        self.interactive.keybindings["q"]()
        assert not self.interactive.running.is_set()  # まだセットされていない状態

    def test_refresh_functionality(self):
        """更新機能テスト"""
        # モックデータを設定
        self.mock_watchlist_manager.get_watchlist.return_value = []
        self.mock_watchlist_manager.get_alerts.return_value = []
        self.mock_trade_manager.get_portfolio_summary.return_value = {}

        # 手動更新実行
        self.interactive._refresh()

        # データ取得メソッドが呼ばれることを確認
        self.mock_watchlist_manager.get_watchlist.assert_called()
        self.mock_watchlist_manager.get_alerts.assert_called()
        self.mock_trade_manager.get_portfolio_summary.assert_called()

    def test_create_header(self):
        """ヘッダー作成テスト"""
        self.interactive.current_data["last_update"] = datetime(2024, 1, 1, 12, 0, 0)
        self.interactive.current_view = "watchlist"

        header = self.interactive._create_header()

        # パネルが作成されることを確認
        assert header is not None
        # より詳細なアサーションは実際のRichオブジェクトの構造に依存

    def test_create_footer(self):
        """フッター作成テスト"""
        footer = self.interactive._create_footer()

        # パネルが作成されることを確認
        assert footer is not None

    @patch("src.day_trade.cli.interactive.PortfolioAnalyzer")
    def test_create_portfolio_metrics(self, mock_portfolio_analyzer):
        """ポートフォリオメトリクス作成テスト"""
        # モックメトリクスを設定
        mock_metrics = Mock()
        mock_metrics.total_value = 1000000
        mock_metrics.total_cost = 950000
        mock_metrics.total_pnl = 50000
        mock_metrics.total_pnl_percent = 5.26
        mock_metrics.volatility = 0.15
        mock_metrics.sharpe_ratio = 0.8

        mock_analyzer_instance = Mock()
        mock_analyzer_instance.get_portfolio_metrics.return_value = mock_metrics
        mock_portfolio_analyzer.return_value = mock_analyzer_instance

        panel = self.interactive._create_portfolio_metrics()

        # パネルが作成されることを確認
        assert panel is not None
        mock_portfolio_analyzer.assert_called_once_with(self.mock_trade_manager)
        mock_analyzer_instance.get_portfolio_metrics.assert_called_once()

    def test_create_watchlist_summary_empty(self):
        """空ウォッチリストサマリー作成テスト"""
        self.interactive.current_data = {"watchlist": []}

        panel = self.interactive._create_watchlist_summary()

        # 空のメッセージが表示されることを確認
        assert panel is not None

    def test_create_watchlist_summary_with_data(self):
        """データありウォッチリストサマリー作成テスト"""
        mock_watchlist = [
            {
                "stock_code": "7203",
                "stock_name": "トヨタ自動車",
                "group_name": "自動車",
            },
            {"stock_code": "8306", "stock_name": "三菱UFJ銀行", "group_name": "銀行"},
            {
                "stock_code": "9984",
                "stock_name": "ソフトバンクグループ",
                "group_name": "テック",
            },
        ]
        self.interactive.current_data = {"watchlist": mock_watchlist}

        panel = self.interactive._create_watchlist_summary()

        # パネルが作成されることを確認
        assert panel is not None

    def test_create_portfolio_panel_empty(self):
        """空ポートフォリオパネル作成テスト"""
        self.interactive.current_data = {"portfolio_summary": {}}

        panel = self.interactive._create_portfolio_panel()

        # 空のメッセージが表示されることを確認
        assert panel is not None

    def test_create_portfolio_panel_with_data(self):
        """データありポートフォリオパネル作成テスト"""
        mock_summary = {
            "total_positions": 3,
            "total_trades": 10,
            "total_cost": "950000",
            "total_market_value": "1000000",
            "total_unrealized_pnl": "50000",
        }
        self.interactive.current_data = {"portfolio_summary": mock_summary}

        panel = self.interactive._create_portfolio_panel()

        # パネルが作成されることを確認
        assert panel is not None

    def test_create_active_alerts_empty(self):
        """空アクティブアラート作成テスト"""
        self.interactive.current_data = {"alerts": []}

        panel = self.interactive._create_active_alerts()

        # 空のメッセージが表示されることを確認
        assert panel is not None

    def test_create_active_alerts_with_data(self):
        """データありアクティブアラート作成テスト"""
        mock_alerts = [
            {
                "stock_code": "7203",
                "alert_type": "price_above",
                "threshold": 3000.0,
                "is_active": True,
            },
            {
                "stock_code": "8306",
                "alert_type": "price_below",
                "threshold": 700.0,
                "is_active": False,
            },
        ]
        self.interactive.current_data = {"alerts": mock_alerts}

        panel = self.interactive._create_active_alerts()

        # パネルが作成されることを確認
        assert panel is not None

    def test_background_update_thread_functionality(self):
        """バックグラウンド更新スレッド機能テスト"""
        # モックデータを設定
        self.mock_watchlist_manager.get_watchlist.return_value = []
        self.mock_watchlist_manager.get_alerts.return_value = []
        self.mock_trade_manager.get_portfolio_summary.return_value = {}

        # runningフラグをセット
        self.interactive.running.set()

        # バックグラウンド更新を短時間実行
        self.interactive.update_interval = 1  # 1秒間隔に短縮
        thread = threading.Thread(
            target=self.interactive._background_update, daemon=True
        )
        thread.start()

        # 少し待機
        time.sleep(0.1)

        # 停止
        self.interactive.running.clear()
        thread.join(timeout=2)

        # データ取得メソッドが呼ばれることを確認
        assert self.mock_watchlist_manager.get_watchlist.called

    def test_cleanup(self):
        """クリーンアップテスト"""
        # runningフラグをセット
        self.interactive.running.set()

        # クリーンアップ実行
        self.interactive._cleanup()

        # runningフラグがクリアされることを確認
        assert not self.interactive.running.is_set()

    def test_view_updates(self):
        """各ビュー更新テスト"""
        # 各ビューの更新メソッドが例外を発生させないことを確認
        self.interactive.current_data = {
            "watchlist": [],
            "alerts": [],
            "portfolio_summary": {},
            "last_update": datetime.now(),
        }

        # ダッシュボードビュー
        self.interactive.current_view = "dashboard"
        self.interactive._update_dashboard_view()

        # ウォッチリストビュー
        self.interactive.current_view = "watchlist"
        self.interactive._update_watchlist_view()

        # ポートフォリオビュー
        self.interactive.current_view = "portfolio"
        self.interactive._update_portfolio_view()

        # アラートビュー
        self.interactive.current_view = "alerts"
        self.interactive._update_alerts_view()

    @patch("builtins.input", return_value="")
    @patch("src.day_trade.cli.interactive.Console.print")
    def test_show_help(self, mock_print, mock_input):
        """ヘルプ表示テスト"""
        self.interactive._show_help()

        # ヘルプが表示されることを確認
        mock_print.assert_called()
        mock_input.assert_called_once()


class TestInteractiveModeIntegration:
    """インタラクティブモード統合テスト"""

    @patch("src.day_trade.cli.interactive.WatchlistManager")
    @patch("src.day_trade.cli.interactive.TradeManager")
    @patch("src.day_trade.cli.interactive.StockFetcher")
    @patch("src.day_trade.cli.interactive.TradingSignalGenerator")
    def test_full_initialization_flow(
        self, mock_signal_gen, mock_fetcher, mock_trade_mgr, mock_watchlist_mgr
    ):
        """完全初期化フローテスト"""
        # モックインスタンスを設定
        mock_watchlist_instance = Mock()
        mock_trade_instance = Mock()

        mock_watchlist_mgr.return_value = mock_watchlist_instance
        mock_trade_mgr.return_value = mock_trade_instance

        # 空のデータを返すように設定
        mock_watchlist_instance.get_watchlist.return_value = []
        mock_watchlist_instance.get_alerts.return_value = []
        mock_trade_instance.get_portfolio_summary.return_value = {}

        # InteractiveModeインスタンス作成
        interactive = InteractiveMode()

        # 初期化が正常に完了することを確認
        assert interactive.console is not None
        assert interactive.layout is not None
        assert interactive.watchlist_manager == mock_watchlist_instance
        assert interactive.trade_manager == mock_trade_instance

    def test_error_handling_in_data_update(self):
        """データ更新時のエラーハンドリングテスト"""
        with patch(
            "src.day_trade.cli.interactive.WatchlistManager"
        ) as mock_watchlist_mgr:
            with patch("src.day_trade.cli.interactive.TradeManager") as mock_trade_mgr:
                # エラーを発生させるモックを設定
                mock_watchlist_instance = Mock()
                mock_trade_instance = Mock()

                mock_watchlist_mgr.return_value = mock_watchlist_instance
                mock_trade_mgr.return_value = mock_trade_instance

                mock_watchlist_instance.get_watchlist.side_effect = Exception(
                    "Test error"
                )
                mock_watchlist_instance.get_alerts.return_value = []
                mock_trade_instance.get_portfolio_summary.return_value = {}

                interactive = InteractiveMode()

                # エラーが発生してもクラッシュしないことを確認
                try:
                    interactive._update_data()
                    # 例外を制御して処理が続行されることを確認
                except Exception as e:
                    pytest.fail(f"Unexpected exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
