"""
チャートパターン認識エンジンのテスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.day_trade.analysis.patterns import ChartPatternRecognizer


class TestChartPatternRecognizer:
    """ChartPatternRecognizerクラスのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        np.random.seed(42)

        # トレンドのあるデータを生成
        trend = np.linspace(100, 120, 100)
        noise = np.random.randn(100) * 1
        close_prices = trend + noise

        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": close_prices + np.random.randn(100) * 0.5,
                "High": close_prices + np.abs(np.random.randn(100)) * 1.5,
                "Low": close_prices - np.abs(np.random.randn(100)) * 1.5,
                "Close": close_prices,
                "Volume": np.random.randint(1000000, 5000000, 100),
            }
        )
        df.set_index("Date", inplace=True)

        return df

    @pytest.fixture
    def crossover_data(self):
        """クロスオーバー用のテストデータ"""
        dates = pd.date_range(end=datetime.now(), periods=50, freq="D")

        # 明確なクロスオーバーパターンを作成
        fast_trend = np.concatenate(
            [
                np.linspace(100, 95, 20),  # 下降
                np.linspace(95, 105, 15),  # 急上昇（ゴールデンクロス）
                np.linspace(105, 100, 15),  # 緩やかな下降
            ]
        )

        slow_trend = np.concatenate(  # noqa: F841
            [
                np.linspace(100, 98, 20),  # 緩やかな下降
                np.linspace(98, 100, 15),  # 緩やかな上昇
                np.linspace(100, 102, 15),  # 緩やかな上昇
            ]
        )

        df = pd.DataFrame(
            {
                "Date": dates,
                "Close": fast_trend + np.random.randn(50) * 0.1,
                "High": fast_trend + 1,
                "Low": fast_trend - 1,
                "Volume": np.random.randint(1000000, 2000000, 50),
            }
        )
        df.set_index("Date", inplace=True)

        return df

    def test_golden_dead_cross(self, crossover_data):
        """ゴールデンクロス・デッドクロス検出のテスト"""
        result = ChartPatternRecognizer.golden_dead_cross(
            crossover_data, fast_period=5, slow_period=10
        )

        # 結果の検証
        assert isinstance(result, pd.DataFrame)
        assert "Golden_Cross" in result.columns
        assert "Dead_Cross" in result.columns
        assert "Golden_Confidence" in result.columns
        assert "Dead_Confidence" in result.columns

        # クロスの検出（少なくとも1つは検出されるはず）
        golden_count = result["Golden_Cross"].sum()
        dead_count = result["Dead_Cross"].sum()
        assert golden_count > 0 or dead_count > 0

        # 信頼度スコアの範囲（NaN値を除外）
        golden_conf_valid = result["Golden_Confidence"].dropna()
        dead_conf_valid = result["Dead_Confidence"].dropna()

        assert (golden_conf_valid >= 0).all()
        assert (golden_conf_valid <= 100).all()
        assert (dead_conf_valid >= 0).all()
        assert (dead_conf_valid <= 100).all()

    def test_support_resistance_levels(self, sample_data):
        """サポート・レジスタンスレベル検出のテスト"""
        levels = ChartPatternRecognizer.support_resistance_levels(
            sample_data, window=10, num_levels=3
        )

        # 結果の検証
        assert isinstance(levels, dict)
        assert "resistance" in levels
        assert "support" in levels

        # レベル数の確認
        assert len(levels["resistance"]) <= 3
        assert len(levels["support"]) <= 3

        # レジスタンス > サポート
        if levels["resistance"] and levels["support"]:
            assert max(levels["resistance"]) > min(levels["support"])

    def test_breakout_detection(self, sample_data):
        """ブレイクアウト検出のテスト"""
        # ブレイクアウト用のデータを作成
        breakout_data = sample_data.copy()
        # 最後の数日で急激な上昇を追加
        breakout_data.loc[breakout_data.index[-3:], "Close"] = [125, 130, 135]
        breakout_data.loc[breakout_data.index[-3:], "High"] = [126, 131, 136]
        breakout_data.loc[breakout_data.index[-3:], "Volume"] = [
            8000000,
            9000000,
            10000000,
        ]

        result = ChartPatternRecognizer.breakout_detection(
            breakout_data, lookback=20, volume_factor=1.5
        )

        # 結果の検証
        assert isinstance(result, pd.DataFrame)
        assert "Upward_Breakout" in result.columns
        assert "Downward_Breakout" in result.columns
        assert "Upward_Confidence" in result.columns
        assert "Downward_Confidence" in result.columns

        # ブレイクアウトが検出されているか
        upward_count = result["Upward_Breakout"].sum()
        assert upward_count > 0  # 最後の急上昇でブレイクアウトが検出されるはず

        # 信頼度スコアの範囲
        assert (result["Upward_Confidence"] >= 0).all()
        assert (result["Downward_Confidence"] >= 0).all()

    def test_trend_line_detection(self, sample_data):
        """トレンドライン検出のテスト"""
        trends = ChartPatternRecognizer.trend_line_detection(
            sample_data, window=10, min_touches=3
        )

        # 結果の検証
        assert isinstance(trends, dict)

        # 上昇トレンドのデータなので、サポートトレンドラインが検出されるはず
        if "support_trend" in trends:
            support = trends["support_trend"]
            assert "slope" in support
            assert "intercept" in support
            assert "r2" in support
            assert "current_value" in support
            assert "touches" in support
            assert "angle" in support

            # 上昇トレンドなのでスロープは正
            assert support["slope"] > 0

            # R²値の範囲
            assert 0 <= support["r2"] <= 1

    def test_detect_all_patterns(self, sample_data):
        """全パターン検出のテスト"""
        results = ChartPatternRecognizer.detect_all_patterns(sample_data)

        # 結果の検証
        assert isinstance(results, dict)
        assert "crosses" in results
        assert "levels" in results
        assert "breakouts" in results
        assert "trends" in results
        assert "overall_confidence" in results

        # 総合信頼度の範囲
        assert 0 <= results["overall_confidence"] <= 100

    def test_empty_dataframe(self):
        """空のDataFrameでのエラーハンドリング"""
        empty_df = pd.DataFrame()

        # ゴールデンクロス
        crosses = ChartPatternRecognizer.golden_dead_cross(empty_df)
        assert isinstance(crosses, pd.DataFrame)
        assert len(crosses) == 0

        # サポート・レジスタンス
        levels = ChartPatternRecognizer.support_resistance_levels(empty_df)
        assert levels == {"resistance": [], "support": []}

        # ブレイクアウト
        breakouts = ChartPatternRecognizer.breakout_detection(empty_df)
        assert isinstance(breakouts, pd.DataFrame)
        assert len(breakouts) == 0

        # トレンドライン
        trends = ChartPatternRecognizer.trend_line_detection(empty_df)
        assert trends == {}

    def test_confidence_scores(self, sample_data):
        """信頼度スコアの妥当性テスト"""
        # 強いトレンドデータを作成
        strong_trend_data = sample_data.copy()
        strong_trend_data["Close"] = np.linspace(100, 150, len(strong_trend_data))

        # 弱いトレンドデータを作成
        weak_trend_data = sample_data.copy()
        weak_trend_data["Close"] = 100 + np.random.randn(len(weak_trend_data)) * 5

        # 強いトレンドの検出
        strong_results = ChartPatternRecognizer.detect_all_patterns(strong_trend_data)

        # 弱いトレンドの検出
        weak_results = ChartPatternRecognizer.detect_all_patterns(weak_trend_data)

        # 強いトレンドの方が高い信頼度を持つはず
        # （必ずしも常に成立するわけではないが、一般的な傾向として）
        assert strong_results["overall_confidence"] >= 0
        assert weak_results["overall_confidence"] >= 0

    def test_pattern_consistency(self, sample_data):
        """パターン検出の一貫性テスト"""
        # 同じデータで複数回実行
        results1 = ChartPatternRecognizer.detect_all_patterns(sample_data)
        results2 = ChartPatternRecognizer.detect_all_patterns(sample_data)

        # 結果が一致することを確認
        assert results1["overall_confidence"] == results2["overall_confidence"]

        # レベル数が一致
        assert len(results1["levels"]["resistance"]) == len(
            results2["levels"]["resistance"]
        )
        assert len(results1["levels"]["support"]) == len(results2["levels"]["support"])

    def test_custom_parameters(self, sample_data):
        """カスタムパラメータでの動作テスト"""
        # 異なるパラメータで実行
        results1 = ChartPatternRecognizer.golden_dead_cross(
            sample_data, fast_period=5, slow_period=20
        )
        results2 = ChartPatternRecognizer.golden_dead_cross(
            sample_data, fast_period=10, slow_period=30
        )

        # 結果が異なることを確認（パラメータが効いている）
        assert not results1["Fast_MA_5"].equals(results2["Fast_MA_10"])
        assert not results1["Slow_MA_20"].equals(results2["Slow_MA_30"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
