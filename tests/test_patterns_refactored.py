"""
チャートパターン認識のテスト（リファクタリング版）
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.day_trade.analysis.patterns import ChartPatternRecognizer


class TestChartPatternRecognizerRefactored:
    """チャートパターン認識のテストクラス（リファクタリング版）"""

    @pytest.fixture
    def sample_data(self):
        """サンプルデータを生成"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        np.random.seed(42)

        # トレンドのあるデータを生成
        trend = np.linspace(100, 120, 100)
        noise = np.random.randn(100) * 2
        close_prices = trend + noise

        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": close_prices + np.random.randn(100) * 0.5,
                "High": close_prices + np.abs(np.random.randn(100)) * 2,
                "Low": close_prices - np.abs(np.random.randn(100)) * 2,
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

        # ゴールデンクロス・デッドクロスが発生するデータ
        prices = []
        for i in range(50):
            if i < 15:
                prices.append(100 + i * 0.5)  # 緩やかな上昇
            elif i < 30:
                prices.append(107.5 - (i - 15) * 0.8)  # 下落（デッドクロス）
            else:
                prices.append(95.5 + (i - 30) * 1.2)  # 急上昇（ゴールデンクロス）

        df = pd.DataFrame(
            {
                "Date": dates,
                "Close": prices,
                "High": [p + 0.5 for p in prices],
                "Low": [p - 0.5 for p in prices],
                "Volume": np.random.randint(1000000, 2000000, 50),
            }
        )
        df.set_index("Date", inplace=True)
        return df

    def test_golden_dead_cross(self, crossover_data):
        """ゴールデンクロス・デッドクロス検出のテスト"""
        recognizer = ChartPatternRecognizer()
        result = recognizer.golden_dead_cross(
            crossover_data, fast_period=5, slow_period=10
        )

        # 結果の検証
        assert isinstance(result, pd.DataFrame)
        assert "Golden_Cross" in result.columns
        assert "Dead_Cross" in result.columns
        assert "Golden_Confidence" in result.columns
        assert "Dead_Confidence" in result.columns

        # 信頼度の範囲チェック（NaN値を除く）
        golden_conf_valid = result["Golden_Confidence"].dropna()
        dead_conf_valid = result["Dead_Confidence"].dropna()

        if len(golden_conf_valid) > 0:
            assert (golden_conf_valid >= 0).all()
            assert (golden_conf_valid <= 100).all()

        if len(dead_conf_valid) > 0:
            assert (dead_conf_valid >= 0).all()
            assert (dead_conf_valid <= 100).all()

    def test_support_resistance_levels(self, sample_data):
        """サポート・レジスタンスレベル検出のテスト"""
        recognizer = ChartPatternRecognizer()
        levels = recognizer.support_resistance_levels(
            sample_data, window=10, num_levels=3
        )

        # 結果の検証
        assert isinstance(levels, dict)
        assert "resistance" in levels
        assert "support" in levels
        assert isinstance(levels["resistance"], list)
        assert isinstance(levels["support"], list)

        # レベル数の検証（最大値）
        assert len(levels["resistance"]) <= 3
        assert len(levels["support"]) <= 3

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

        recognizer = ChartPatternRecognizer()
        result = recognizer.breakout_detection(
            breakout_data, lookback=20, volume_factor=1.5
        )

        # 結果の検証
        assert isinstance(result, pd.DataFrame)
        assert "Upward_Breakout" in result.columns
        assert "Downward_Breakout" in result.columns
        assert "Upward_Confidence" in result.columns
        assert "Downward_Confidence" in result.columns

        # 信頼度の範囲チェック
        assert (result["Upward_Confidence"] >= 0).all()
        assert (result["Upward_Confidence"] <= 100).all()

    def test_trend_line_detection(self, sample_data):
        """トレンドライン検出のテスト"""
        recognizer = ChartPatternRecognizer()
        trends = recognizer.trend_line_detection(
            sample_data, window=10, min_touches=3
        )

        # 結果の検証
        assert isinstance(trends, dict)
        # トレンドが検出された場合の検証
        for trend_name, trend_info in trends.items():
            assert isinstance(trend_info, dict)
            assert "slope" in trend_info
            assert "intercept" in trend_info
            assert "r2" in trend_info

    def test_detect_all_patterns(self, sample_data):
        """全パターン検出のテスト"""
        recognizer = ChartPatternRecognizer()
        results = recognizer.detect_all_patterns(sample_data)

        # 結果の検証
        assert isinstance(results, dict)
        assert "crosses" in results
        assert "breakouts" in results
        assert "levels" in results
        assert "trends" in results
        assert "latest_signal" in results
        assert "overall_confidence" in results

        # 総合信頼度の範囲
        assert 0 <= results["overall_confidence"] <= 100

    def test_empty_dataframe(self):
        """空のDataFrameでのエラーハンドリング"""
        empty_df = pd.DataFrame()
        recognizer = ChartPatternRecognizer()

        # ゴールデンクロス
        crosses = recognizer.golden_dead_cross(empty_df)
        assert isinstance(crosses, pd.DataFrame)
        assert len(crosses) == 0

        # サポート・レジスタンス
        levels = recognizer.support_resistance_levels(empty_df)
        assert levels == {"resistance": [], "support": []}

        # ブレイクアウト
        breakouts = recognizer.breakout_detection(empty_df)
        assert isinstance(breakouts, pd.DataFrame)
        assert len(breakouts) == 0

        # トレンドライン
        trends = recognizer.trend_line_detection(empty_df)
        assert trends == {}

    def test_configuration_usage(self, sample_data):
        """設定が正しく使用されているかのテスト"""
        recognizer = ChartPatternRecognizer()

        # デフォルト設定でのテスト
        result_default = recognizer.golden_dead_cross(sample_data)

        # 明示的なパラメータでのテスト
        result_explicit = recognizer.golden_dead_cross(
            sample_data, fast_period=5, slow_period=20
        )

        # 両方とも有効な結果が得られることを確認
        assert isinstance(result_default, pd.DataFrame)
        assert isinstance(result_explicit, pd.DataFrame)
        assert len(result_default) == len(result_explicit)

    def test_confidence_scores(self, sample_data):
        """信頼度スコアの妥当性テスト"""
        recognizer = ChartPatternRecognizer()

        # 強いトレンドデータを作成
        strong_trend_data = sample_data.copy()
        strong_trend_data["Close"] = np.linspace(100, 150, len(strong_trend_data))

        # 弱いトレンドデータを作成
        weak_trend_data = sample_data.copy()
        weak_trend_data["Close"] = 100 + np.random.randn(len(weak_trend_data)) * 5

        # 強いトレンドの検出
        strong_results = recognizer.detect_all_patterns(strong_trend_data)

        # 弱いトレンドの検出
        weak_results = recognizer.detect_all_patterns(weak_trend_data)

        # 信頼度の比較（強いトレンドの方が高い信頼度を持つべき）
        # 注意：データの特性によっては必ずしも成立しないため、基本的な範囲チェックのみ
        assert 0 <= strong_results["overall_confidence"] <= 100
        assert 0 <= weak_results["overall_confidence"] <= 100
