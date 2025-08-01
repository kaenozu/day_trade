"""
表示フォーマッタのテスト
高度なCLI表示機能とASCIIチャート描画機能のテスト
"""
import pytest
from decimal import Decimal
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns

from src.day_trade.utils.formatters import (
    format_currency, format_percentage, format_volume, get_change_color,
    create_stock_info_table, create_company_info_table, create_historical_data_table,
    create_watchlist_table, create_error_panel, create_success_panel,
    
    # 高度なフォーマッタ機能
    format_large_number, create_ascii_chart, create_sparkline,
    create_progress_bar_panel, create_comparison_table, create_heatmap,
    create_metric_cards, create_trend_indicator, create_distribution_chart,
    create_status_indicator
)


class TestBasicFormatters:
    """基本フォーマッタのテスト"""
    
    def test_format_currency(self):
        """通貨フォーマットテスト"""
        assert format_currency(1234567) == "¥1,234,567"
        assert format_currency(1234.56, decimal_places=2) == "¥1,234.56"
        assert format_currency(0) == "¥0"
        assert format_currency(None) == "N/A"
        assert format_currency(1000, currency="$") == "$1,000"
    
    def test_format_percentage(self):
        """パーセンテージフォーマットテスト"""
        assert format_percentage(12.34) == "+12.34%"
        assert format_percentage(-5.67) == "-5.67%"
        assert format_percentage(0) == "0.00%"
        assert format_percentage(None) == "N/A"
        assert format_percentage(3.14159, decimal_places=3) == "+3.142%"
        assert format_percentage(10.5, show_sign=False) == "10.50%"
    
    def test_format_volume(self):
        """出来高フォーマットテスト"""
        assert format_volume(1234567890) == "1234.6M"
        assert format_volume(1234567) == "1.2M"
        assert format_volume(1234) == "1.2K"
        assert format_volume(123) == "123"
        assert format_volume(None) == "N/A"
    
    def test_get_change_color(self):
        """変化色取得テスト"""
        assert get_change_color(10) == "green"
        assert get_change_color(-5) == "red"
        assert get_change_color(0) == "white"


class TestAdvancedFormatters:
    """高度なフォーマッタのテスト"""
    
    def test_format_large_number(self):
        """大きな数値フォーマットテスト"""
        assert format_large_number(1500000000000) == "1.5T"
        assert format_large_number(2500000000) == "2.5B"
        assert format_large_number(1500000) == "1.5M"
        assert format_large_number(2500) == "2.5K"
        assert format_large_number(250) == "250.0"
        assert format_large_number(-1500000) == "-1.5M"
        assert format_large_number(None) == "N/A"
    
    def test_create_ascii_chart(self):
        """ASCIIチャート作成テスト"""
        data = [10, 20, 15, 30, 25, 35, 40]
        chart = create_ascii_chart(data, width=20, height=5, title="Test Chart")
        
        assert "Test Chart" in chart
        assert "=" in chart
        assert "│" in chart
        assert "█" in chart or "▀" in chart or "▄" in chart
    
    def test_create_ascii_chart_edge_cases(self):
        """ASCIIチャート作成エッジケーステスト"""
        # 空データ
        empty_chart = create_ascii_chart([], title="Empty")
        assert "Empty" in empty_chart
        assert "[No data to display]" in empty_chart
        
        # 単一データ
        single_chart = create_ascii_chart([10], title="Single")
        assert "Single" in single_chart
        assert "[No data to display]" in single_chart
        
        # 同じ値のデータ
        same_chart = create_ascii_chart([10, 10, 10], title="Same")
        assert "Same" in same_chart
        assert "[Data has no variation]" in same_chart
    
    def test_create_sparkline(self):
        """スパークライン作成テスト"""
        data = [10, 15, 12, 20, 18, 25]
        sparkline = create_sparkline(data, width=6)
        
        assert len(sparkline) == 6
        assert all(c in "▁▂▃▄▅▆▇█" for c in sparkline)
    
    def test_create_sparkline_edge_cases(self):
        """スパークラインエッジケーステスト"""
        # 空データ
        assert create_sparkline([]) == "No data"
        
        # 単一データ
        single_line = create_sparkline([10], width=5)
        assert single_line == "▄" * 5
        
        # 同じ値
        same_line = create_sparkline([10, 10, 10], width=3)
        assert same_line == "▄" * 3
    
    def test_create_progress_bar_panel(self):
        """プログレスバーパネル作成テスト"""
        panel = create_progress_bar_panel(75, 100, title="Test Progress")
        
        assert isinstance(panel, Panel)
        assert "75/100" in str(panel.renderable)
        assert "75.0%" in str(panel.renderable)
    
    def test_create_progress_bar_panel_edge_cases(self):
        """プログレスバーパネルエッジケーステスト"""
        # ゼロ除算
        panel = create_progress_bar_panel(0, 0, title="Zero")
        assert "0/0" in str(panel.renderable)
        assert "0.0%" in str(panel.renderable)
    
    def test_create_comparison_table(self):
        """比較テーブル作成テスト"""
        data = {
            "Stock A": {"price": 1000, "volume": 50000, "change_rate": 2.5},
            "Stock B": {"price": 1500, "volume": 30000, "change_rate": -1.2}
        }
        
        table = create_comparison_table(data, title="Stock Comparison")
        assert isinstance(table, Table)
    
    def test_create_comparison_table_empty(self):
        """比較テーブル空データテスト"""
        table = create_comparison_table({}, title="Empty Table")
        assert isinstance(table, Table)
    
    def test_create_heatmap(self):
        """ヒートマップ作成テスト"""
        data = [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0]
        ]
        labels_x = ["A", "B", "C"]
        labels_y = ["X", "Y", "Z"]
        
        heatmap = create_heatmap(data, labels_x, labels_y, title="Test Heatmap")
        
        assert "Test Heatmap" in heatmap
        assert "X" in heatmap
        assert "Y" in heatmap
        assert "Z" in heatmap
    
    def test_create_heatmap_edge_cases(self):
        """ヒートマップエッジケーステスト"""
        # 空データ
        empty_heatmap = create_heatmap([], [], [], title="Empty")
        assert "Empty" in empty_heatmap
        assert "[No data to display]" in empty_heatmap
        
        # Noneデータ
        none_data = [[None, None], [None, None]]
        none_heatmap = create_heatmap(none_data, ["A", "B"], ["X", "Y"], title="None")
        assert "None" in none_heatmap
        assert "[No valid data]" in none_heatmap
        
        # 同じ値
        same_data = [[1.0, 1.0], [1.0, 1.0]]
        same_heatmap = create_heatmap(same_data, ["A", "B"], ["X", "Y"], title="Same")
        assert "Same" in same_heatmap
        assert "[Data has no variation]" in same_heatmap
    
    def test_create_metric_cards(self):
        """メトリクスカード作成テスト"""
        metrics = {
            "Revenue": 1000000,
            "Profit Rate": 15.5,
            "Market Share": 23.2,
            "Growth": 8.7
        }
        
        cards = create_metric_cards(metrics, columns=2)
        assert isinstance(cards, Columns)
    
    def test_create_trend_indicator(self):
        """トレンド指標作成テスト"""
        # 上昇トレンド
        up_trend = create_trend_indicator(110, 100, "Price")
        assert isinstance(up_trend, Text)
        assert "↗" in str(up_trend)
        
        # 下降トレンド
        down_trend = create_trend_indicator(90, 100, "Price")
        assert isinstance(down_trend, Text)
        assert "↘" in str(down_trend)
        
        # 横ばい
        flat_trend = create_trend_indicator(100, 100, "Price")
        assert isinstance(flat_trend, Text)
        assert "→" in str(flat_trend)
        
        # ゼロ除算ケース
        zero_trend = create_trend_indicator(100, 0, "Price")
        assert isinstance(zero_trend, Text)
        assert "N/A" in str(zero_trend)
    
    def test_create_distribution_chart(self):
        """分布チャート作成テスト"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10  # 100個のデータ点
        chart = create_distribution_chart(data, bins=5, title="Test Distribution")
        
        assert "Test Distribution" in chart
        assert "│" in chart
        assert "█" in chart
    
    def test_create_distribution_chart_edge_cases(self):
        """分布チャートエッジケーステスト"""
        # 空データ
        empty_chart = create_distribution_chart([], title="Empty")
        assert "Empty" in empty_chart
        assert "[No data to display]" in empty_chart
        
        # 同じ値
        same_chart = create_distribution_chart([5, 5, 5, 5], title="Same")
        assert "Same" in same_chart
        assert "[All values are the same: 5]" in same_chart
    
    def test_create_status_indicator(self):
        """ステータス指標作成テスト"""
        # 成功ステータス
        success_status = create_status_indicator("success", "Status")
        assert isinstance(success_status, Text)
        assert "✓" in str(success_status)
        
        # エラーステータス
        error_status = create_status_indicator("error", "Status")
        assert isinstance(error_status, Text)
        assert "✗" in str(error_status)
        
        # 警告ステータス
        warning_status = create_status_indicator("warning", "Status")
        assert isinstance(warning_status, Text)
        assert "⚠" in str(warning_status)
        
        # 未知のステータス
        unknown_status = create_status_indicator("unknown", "Status")
        assert isinstance(unknown_status, Text)
        assert "•" in str(unknown_status)


class TestTableCreation:
    """テーブル作成のテスト"""
    
    def test_create_stock_info_table(self):
        """株価情報テーブル作成テスト"""
        stock_data = {
            'symbol': '7203',
            'current_price': 2500,
            'previous_close': 2450,
            'change': 50,
            'change_percent': 2.04,
            'volume': 1500000
        }
        
        table = create_stock_info_table(stock_data)
        assert isinstance(table, Table)
    
    def test_create_company_info_table(self):
        """企業情報テーブル作成テスト"""
        company_data = {
            'name': 'トヨタ自動車',
            'sector': '自動車',
            'industry': '自動車製造',
            'market_cap': 25000000000000,
            'employees': 370000
        }
        
        table = create_company_info_table(company_data)
        assert isinstance(table, Table)
    
    def test_create_watchlist_table(self):
        """ウォッチリストテーブル作成テスト"""
        watchlist_data = {
            '7203': {
                'current_price': 2500,
                'change': 50,
                'change_percent': 2.04,
                'volume': 1500000
            },
            '8306': {
                'current_price': 850,
                'change': -10,
                'change_percent': -1.16,
                'volume': 800000
            }
        }
        
        table = create_watchlist_table(watchlist_data)
        assert isinstance(table, Table)


class TestPanelCreation:
    """パネル作成のテスト"""
    
    def test_create_error_panel(self):
        """エラーパネル作成テスト"""
        panel = create_error_panel("テストエラー", "エラー発生")
        assert isinstance(panel, Panel)
    
    def test_create_success_panel(self):
        """成功パネル作成テスト"""
        panel = create_success_panel("処理成功", "完了")
        assert isinstance(panel, Panel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])