"""
インタラクティブモードのテスト
"""

from unittest.mock import Mock, patch

import pytest

from src.day_trade.cli.interactive import InteractiveMode


class TestInteractiveMode:
    """InteractiveModeクラスのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        # 各種マネージャーをモック化
        self.mock_watchlist_manager = Mock()
        self.mock_stock_fetcher = Mock()

        with patch(
            "src.day_trade.cli.interactive._get_watchlist_manager",
            return_value=self.mock_watchlist_manager,
        ):
            with patch(
                "src.day_trade.cli.interactive.StockFetcher",
                return_value=self.mock_stock_fetcher,
            ):
                self.interactive = InteractiveMode()

    def test_initialization(self):
        """初期化テスト"""
        assert self.interactive.console is not None
        assert self.interactive.watchlist_manager is not None
        assert self.interactive.stock_fetcher is not None

    def test_handle_command(self):
        """コマンド処理テスト"""
        # 終了コマンドのテスト
        assert self.interactive.handle_command("exit") is False
        assert self.interactive.handle_command("quit") is False
        assert self.interactive.handle_command("q") is False

        # 通常コマンドのテスト
        assert self.interactive.handle_command("test") is True

    def test_start_method(self):
        """startメソッドのテスト"""
        # startメソッドが例外を発生させないことを確認
        try:
            self.interactive.start()
        except Exception as e:
            pytest.fail(f"start method raised an exception: {e}")

    def test_stop_method(self):
        """stopメソッドのテスト"""
        # stopメソッドが例外を発生させないことを確認
        try:
            self.interactive.stop()
        except Exception as e:
            pytest.fail(f"stop method raised an exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
