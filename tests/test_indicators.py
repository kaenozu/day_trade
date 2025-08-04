"""
テクニカル指標計算エンジンのテスト
"""

import numpy as np
import pandas as pd
import pytest

from src.day_trade.analysis.indicators import IndicatorsConfig, TechnicalIndicators


class TestTechnicalIndicators:
    """TechnicalIndicatorsクラスのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        # テストの再現性のため固定日付を使用
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)

        # より現実的な価格データを生成
        base_price = 100
        price_changes = np.random.randn(50) * 2
        close_prices = base_price + np.cumsum(price_changes)

        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": close_prices + np.random.randn(50) * 0.5,
                "High": close_prices + np.abs(np.random.randn(50)) * 1.5,
                "Low": close_prices - np.abs(np.random.randn(50)) * 1.5,
                "Close": close_prices,
                "Volume": np.random.randint(1000000, 5000000, 50),
            }
        )
        df.set_index("Date", inplace=True)

        return df

    def test_sma(self, sample_data):
        """SMA計算のテスト"""
        # 20日SMAを計算
        sma = TechnicalIndicators.sma(sample_data, period=20)

        # 結果の検証
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(sample_data)

        # 最初の19日はNaN
        assert sma.iloc[:19].isna().all()

        # 20日目以降は値が存在
        assert sma.iloc[19:].notna().all()

        # 手動計算と比較（20日目）
        manual_sma = sample_data["Close"].iloc[:20].mean()
        assert abs(sma.iloc[19] - manual_sma) < 0.0001

    def test_ema(self, sample_data):
        """EMA計算のテスト"""
        # 12日EMAを計算
        ema = TechnicalIndicators.ema(sample_data, period=12)

        # 結果の検証
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(sample_data)

        # 最初から値が存在（EMAの特性）
        assert ema.notna().all()

        # EMAの値が妥当な範囲内にあることを確認
        price_range = sample_data["Close"].max() - sample_data["Close"].min()
        assert ema.min() >= sample_data["Close"].min() - price_range * 0.1
        assert ema.max() <= sample_data["Close"].max() + price_range * 0.1

        # 具体的な値のアサーション（固定シードによる予期される値）
        # 最初の値は終値と同じ
        assert abs(ema.iloc[0] - sample_data["Close"].iloc[0]) < 0.0001

        # 特定の日付での具体的な値を検証（np.random.seed(42)の結果）
        # 10日目のEMA値の検証
        expected_ema_10 = 105.3165  # 実際の計算結果
        assert abs(ema.iloc[9] - expected_ema_10) < 0.01

        # 20日目のEMA値の検証
        expected_ema_20 = 100.0069  # 実際の計算結果
        assert abs(ema.iloc[19] - expected_ema_20) < 0.01

    def test_bollinger_bands(self, sample_data):
        """ボリンジャーバンド計算のテスト"""
        bb = TechnicalIndicators.bollinger_bands(sample_data, period=20, num_std=2)

        # 結果の検証
        assert isinstance(bb, pd.DataFrame)
        assert "BB_Upper" in bb.columns
        assert "BB_Middle" in bb.columns
        assert "BB_Lower" in bb.columns

        # 上部バンド > 中間バンド > 下部バンド
        valid_idx = bb["BB_Middle"].notna()
        assert (bb.loc[valid_idx, "BB_Upper"] > bb.loc[valid_idx, "BB_Middle"]).all()
        assert (bb.loc[valid_idx, "BB_Middle"] > bb.loc[valid_idx, "BB_Lower"]).all()

        # 具体的な値のアサーション（20日目、最初の有効な値）
        # 中間バンドは20日移動平均と同じ
        expected_middle = sample_data["Close"].iloc[:20].mean()
        assert abs(bb["BB_Middle"].iloc[19] - expected_middle) < 0.0001

        # バンド幅の計算検証（2倍標準偏差）
        expected_std = sample_data["Close"].iloc[:20].std()
        expected_upper = expected_middle + (expected_std * 2)
        expected_lower = expected_middle - (expected_std * 2)
        assert abs(bb["BB_Upper"].iloc[19] - expected_upper) < 0.0001
        assert abs(bb["BB_Lower"].iloc[19] - expected_lower) < 0.0001

    def test_macd(self, sample_data):
        """MACD計算のテスト"""
        macd = TechnicalIndicators.macd(sample_data)

        # 結果の検証
        assert isinstance(macd, pd.DataFrame)
        assert "MACD" in macd.columns
        assert "MACD_Signal" in macd.columns
        assert "MACD_Histogram" in macd.columns

        # ヒストグラム = MACD - シグナル
        histogram_check = macd["MACD"] - macd["MACD_Signal"]
        diff = abs(macd["MACD_Histogram"] - histogram_check)
        assert diff.max() < 0.0001

        # 具体的な値のアサーション（31日目、十分なデータがある地点）
        expected_macd_31 = -3.6576  # 実際の計算結果
        expected_signal_31 = -2.8824  # 実際の計算結果
        assert abs(macd["MACD"].iloc[30] - expected_macd_31) < 0.01
        assert abs(macd["MACD_Signal"].iloc[30] - expected_signal_31) < 0.01

        # ヒストグラムの検証
        expected_histogram = expected_macd_31 - expected_signal_31
        assert abs(macd["MACD_Histogram"].iloc[30] - expected_histogram) < 0.01

    def test_rsi(self, sample_data):
        """RSI計算のテスト"""
        rsi = TechnicalIndicators.rsi(sample_data, period=14)

        # 結果の検証
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_data)

        # RSIは0〜100の範囲
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

        # 具体的な値のアサーション（20日目、最初の有効な値）
        expected_rsi_20 = 24.7806  # 実際の計算結果
        assert abs(rsi.iloc[19] - expected_rsi_20) < 0.01

        # 31日目のRSI値の検証
        expected_rsi_31 = 24.2663  # 実際の計算結果
        assert abs(rsi.iloc[30] - expected_rsi_31) < 0.01

    def test_stochastic(self, sample_data):
        """ストキャスティクス計算のテスト"""
        stoch = TechnicalIndicators.stochastic(sample_data)

        # 結果の検証
        assert isinstance(stoch, pd.DataFrame)
        assert "Stoch_K" in stoch.columns
        assert "Stoch_D" in stoch.columns

        # ストキャスティクスは0〜100の範囲
        valid_k = stoch["Stoch_K"].dropna()
        valid_d = stoch["Stoch_D"].dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_volume_analysis(self, sample_data):
        """出来高分析のテスト"""
        volume = TechnicalIndicators.volume_analysis(sample_data, period=20)

        # 結果の検証
        assert isinstance(volume, pd.DataFrame)
        assert "Volume_MA" in volume.columns
        assert "Volume_Ratio" in volume.columns

        # 出来高比率は正の値
        valid_ratio = volume["Volume_Ratio"].dropna()
        assert (valid_ratio > 0).all()

    def test_atr(self, sample_data):
        """ATR計算のテスト"""
        atr = TechnicalIndicators.atr(sample_data, period=14)

        # 結果の検証
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_data)

        # ATRは正の値
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()

    def test_calculate_all(self, sample_data):
        """全指標計算のテスト"""
        result = TechnicalIndicators.calculate_all(sample_data)

        # 結果の検証
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)

        # 元のデータ列が保持されている
        for col in sample_data.columns:
            assert col in result.columns

        # 各指標が追加されている
        expected_indicators = [
            "SMA_5",
            "SMA_20",
            "SMA_60",
            "EMA_12",
            "EMA_26",
            "BB_Upper",
            "BB_Middle",
            "BB_Lower",
            "MACD",
            "MACD_Signal",
            "MACD_Histogram",
            "RSI",
            "Stoch_K",
            "Stoch_D",
            "Volume_MA",
            "Volume_Ratio",
            "ATR",
        ]

        for indicator in expected_indicators:
            assert indicator in result.columns

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 空のDataFrame
        empty_df = pd.DataFrame()

        # SMAは空のSeriesを返す
        sma = TechnicalIndicators.sma(empty_df)
        assert isinstance(sma, pd.Series)
        assert len(sma) == 0

        # MACDは空のDataFrameを返す
        macd = TechnicalIndicators.macd(empty_df)
        assert isinstance(macd, pd.DataFrame)
        assert len(macd) == 0

    def test_custom_column(self, sample_data):
        """カスタム列での計算テスト"""
        # High列でSMAを計算
        sma_high = TechnicalIndicators.sma(sample_data, period=10, column="High")

        # 結果の検証
        assert isinstance(sma_high, pd.Series)
        assert len(sma_high) == len(sample_data)

        # 手動計算と比較
        manual_sma = sample_data["High"].iloc[:10].mean()
        assert abs(sma_high.iloc[9] - manual_sma) < 0.0001

    def test_edge_cases(self, sample_data):
        """エッジケースのテスト"""
        # 期間がデータ長より長い場合
        sma_long = TechnicalIndicators.sma(sample_data, period=100)
        assert sma_long.isna().all()  # 全てNaN

        # 期間が1の場合
        sma_1 = TechnicalIndicators.sma(sample_data, period=1)
        assert (sma_1 == sample_data["Close"]).all()

    def test_missing_columns(self):
        """必須カラム欠如時のエラーハンドリングテスト"""
        # Close列がないDataFrame
        df_no_close = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [102, 103, 104],
                "Low": [99, 100, 101],
                "Volume": [1000000, 1100000, 1200000],
            }
        )

        # SMAはCloseカラムがないとエラーまたは空のSeriesを返す
        result = TechnicalIndicators.sma(df_no_close)
        assert isinstance(result, pd.Series)
        assert len(result) == 0 or result.isna().all()

        # High、Low、Closeカラムがない場合のストキャスティクス
        result_stoch = TechnicalIndicators.stochastic(df_no_close)
        assert isinstance(result_stoch, pd.DataFrame)
        assert len(result_stoch) == 0

    def test_insufficient_data(self):
        """データ不足時のテスト"""
        # 極少データ（5行のみ）
        small_data = pd.DataFrame(
            {
                "Close": [100, 101, 99, 102, 98],
                "High": [101, 102, 100, 103, 99],
                "Low": [99, 100, 98, 101, 97],
                "Volume": [1000000] * 5,
            }
        )

        # 14日RSIを計算（データ不足でも値は出るが、最初の値はNaN）
        rsi = TechnicalIndicators.rsi(small_data, period=14)
        assert rsi.iloc[0] is np.nan or pd.isna(
            rsi.iloc[0]
        )  # 最初の値はNaN（差分がないため）
        assert len(rsi) == 5  # データと同じ長さ

        # 20日SMAを計算（データ不足）
        sma = TechnicalIndicators.sma(small_data, period=20)
        assert sma.isna().all()  # 全てNaN（期間がデータ長を超えるため）

    def test_rsi_zero_division_handling(self):
        """RSI計算でのゼロ除算ハンドリングテスト"""
        # 全て同じ価格（変動なし）でゼロ除算が発生する可能性
        constant_data = pd.DataFrame(
            {
                "Close": [100.0] * 20,
                "High": [100.5] * 20,
                "Low": [99.5] * 20,
                "Volume": [1000000] * 20,
            }
        )

        rsi = TechnicalIndicators.rsi(constant_data, period=14)
        # 変動がない場合、RSIは50またはNaNになるべき
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            # 変動がない場合のRSI処理を確認
            assert all(v == 50.0 or pd.isna(v) for v in valid_rsi)

    def test_extreme_values(self):
        """極端な値でのテスト"""
        # 非常に大きな値
        extreme_data = pd.DataFrame(
            {
                "Close": [1e6 + i for i in range(20)],
                "High": [1e6 + i + 0.5 for i in range(20)],
                "Low": [1e6 + i - 0.5 for i in range(20)],
                "Volume": [1000000] * 20,
            }
        )

        # SMA計算が正常に動作するか
        sma = TechnicalIndicators.sma(extreme_data, period=10)
        assert sma.iloc[9:].notna().all()

        # 非常に小さな値
        tiny_data = pd.DataFrame(
            {
                "Close": [0.00001 + i * 0.000001 for i in range(20)],
                "High": [0.000011 + i * 0.000001 for i in range(20)],
                "Low": [0.000009 + i * 0.000001 for i in range(20)],
                "Volume": [1000] * 20,
            }
        )

        # EMA計算が正常に動作するか
        ema = TechnicalIndicators.ema(tiny_data, period=5)
        assert ema.notna().all()

    def test_nan_values_in_data(self):
        """データにNaN値が含まれる場合のテスト"""
        data_with_nan = pd.DataFrame(
            {
                "Close": [100, 101, np.nan, 102, 103, 104, np.nan, 105],
                "High": [101, 102, np.nan, 103, 104, 105, np.nan, 106],
                "Low": [99, 100, np.nan, 101, 102, 103, np.nan, 104],
                "Volume": [1000000] * 8,
            }
        )

        # SMA計算でNaNの影響を確認
        sma = TechnicalIndicators.sma(data_with_nan, period=3)
        # NaNを含む期間の計算結果を確認
        assert isinstance(sma, pd.Series)

        # RSI計算でNaNの影響を確認
        rsi = TechnicalIndicators.rsi(data_with_nan, period=5)
        assert isinstance(rsi, pd.Series)

    def test_calculate_all_performance(self):
        """calculate_allのパフォーマンステスト"""
        import time

        # 大規模データ（1年分の日次データ）
        large_dates = pd.date_range(start="2022-01-01", periods=365, freq="D")
        np.random.seed(123)

        large_price_changes = np.random.randn(365) * 2
        large_close_prices = 100 + np.cumsum(large_price_changes)

        large_data = pd.DataFrame(
            {
                "Date": large_dates,
                "Open": large_close_prices + np.random.randn(365) * 0.5,
                "High": large_close_prices + np.abs(np.random.randn(365)) * 1.5,
                "Low": large_close_prices - np.abs(np.random.randn(365)) * 1.5,
                "Close": large_close_prices,
                "Volume": np.random.randint(1000000, 5000000, 365),
            }
        )
        large_data.set_index("Date", inplace=True)

        # パフォーマンス測定
        start_time = time.time()
        result = TechnicalIndicators.calculate_all(large_data)
        end_time = time.time()

        execution_time = end_time - start_time

        # 結果の検証
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 365

        # パフォーマンス要件（1年分のデータを5秒以内で処理）
        assert execution_time < 5.0, f"Performance issue: took {execution_time:.2f}s"

        # メモリ使用量の簡易チェック（結果が元データより過度に大きくないか）
        memory_ratio = (
            result.memory_usage(deep=True).sum()
            / large_data.memory_usage(deep=True).sum()
        )
        assert memory_ratio < 10.0, f"Memory usage issue: ratio {memory_ratio:.2f}"

    def test_precision_consistency(self, sample_data):
        """計算精度の一貫性テスト"""
        # 同じデータで複数回計算した場合の一貫性
        sma1 = TechnicalIndicators.sma(sample_data, period=20)
        sma2 = TechnicalIndicators.sma(sample_data, period=20)

        # 結果が完全に同じであることを確認
        pd.testing.assert_series_equal(sma1, sma2)

        # EMAの一貫性
        ema1 = TechnicalIndicators.ema(sample_data, period=12)
        ema2 = TechnicalIndicators.ema(sample_data, period=12)

        pd.testing.assert_series_equal(ema1, ema2)


class TestIndicatorsConfig:
    """テクニカル指標設定管理のテスト"""

    def test_config_initialization(self):
        """設定初期化テスト"""
        config = IndicatorsConfig()
        assert config.config is not None
        assert "default_parameters" in config.config
        assert "calculate_all_defaults" in config.config

    def test_parameter_retrieval(self):
        """パラメータ取得テスト"""
        config = IndicatorsConfig()

        # デフォルト値の取得
        sma_default = config.get_parameter("sma", "default_period")
        assert sma_default == 20

        rsi_period = config.get_parameter("rsi", "period")
        assert rsi_period == 14

        # 存在しないパラメータ
        unknown = config.get_parameter("unknown", "param", "default_value")
        assert unknown == "default_value"

    def test_calculate_all_defaults(self):
        """calculate_all設定取得テスト"""
        config = IndicatorsConfig()
        defaults = config.get_calculate_all_defaults()

        assert "sma_periods" in defaults
        assert "bb_period" in defaults
        assert "rsi_period" in defaults
        assert isinstance(defaults["sma_periods"], list)

    def test_custom_config_file(self):
        """カスタム設定ファイルテスト"""
        import json
        import os
        import tempfile

        # テスト用設定
        test_config = {
            "default_parameters": {
                "sma": {"default_period": 25},
                "rsi": {"period": 21},
            },
            "calculate_all_defaults": {"sma_periods": [10, 30], "rsi_period": 21},
            "performance_settings": {"progress_threshold": 50},
        }

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_config, f)
            temp_path = f.name

        try:
            config = IndicatorsConfig(temp_path)
            assert config.get_parameter("sma", "default_period") == 25
            assert config.get_parameter("rsi", "period") == 21

            defaults = config.get_calculate_all_defaults()
            assert defaults["sma_periods"] == [10, 30]
            assert defaults["rsi_period"] == 21

            perf = config.get_performance_settings()
            assert perf["progress_threshold"] == 50

        finally:
            os.unlink(temp_path)


class TestImprovedIndicators:
    """改善されたテクニカル指標機能のテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用データ"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

        return pd.DataFrame(
            {
                "Open": close_prices + np.random.randn(100) * 0.1,
                "High": close_prices + np.abs(np.random.randn(100)) * 0.2,
                "Low": close_prices - np.abs(np.random.randn(100)) * 0.2,
                "Close": close_prices,
                "Volume": np.random.randint(1000000, 5000000, 100),
            },
            index=dates,
        )

    def test_sma_with_config_defaults(self, sample_data):
        """設定デフォルト値を使用したSMAテスト"""
        # デフォルト期間で計算
        sma_default = TechnicalIndicators.sma(sample_data)

        # 明示的に期間を指定して計算
        sma_explicit = TechnicalIndicators.sma(sample_data, period=20)

        # 結果が同じであることを確認
        pd.testing.assert_series_equal(sma_default, sma_explicit)

    def test_ema_with_config_defaults(self, sample_data):
        """設定デフォルト値を使用したEMAテスト"""
        # デフォルト期間で計算
        ema_default = TechnicalIndicators.ema(sample_data)

        # 明示的に期間を指定して計算
        ema_explicit = TechnicalIndicators.ema(sample_data, period=20)

        # 結果が同じであることを確認
        pd.testing.assert_series_equal(ema_default, ema_explicit)

    def test_rsi_with_config_defaults(self, sample_data):
        """設定デフォルト値を使用したRSIテスト"""
        # デフォルト期間で計算
        rsi_default = TechnicalIndicators.rsi(sample_data)

        # 明示的に期間を指定して計算
        rsi_explicit = TechnicalIndicators.rsi(sample_data, period=14)

        # 結果が同じであることを確認
        pd.testing.assert_series_equal(rsi_default, rsi_explicit)

    def test_rsi_zero_division_handling(self):
        """RSIのゼロ除算対策テスト"""
        # 同じ価格（損失がゼロ）
        constant_data = pd.DataFrame(
            {
                "Close": [100.0] * 20,
                "High": [100.1] * 20,
                "Low": [99.9] * 20,
                "Volume": [1000000] * 20,
            }
        )
        constant_data.index = pd.date_range(start="2023-01-01", periods=20, freq="D")

        rsi = TechnicalIndicators.rsi(constant_data)

        # 変動がない場合、適切に処理されることを確認
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == 20

        # 最初の値以外はNaNでないことを確認（値が存在する）
        valid_values = rsi.dropna()
        if len(valid_values) > 0:
            # 値が0-100の範囲内であることを確認
            assert (valid_values >= 0).all()
            assert (valid_values <= 100).all()

    def test_calculate_all_optimized(self, sample_data):
        """最適化されたcalculate_allテスト"""
        # 新しい実装
        result = TechnicalIndicators.calculate_all(sample_data)

        # 基本的な検証
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)

        # 元のデータ列が保持されている
        for col in sample_data.columns:
            assert col in result.columns

        # 期待される指標が含まれている
        expected_indicators = [
            "SMA_5",
            "SMA_20",
            "SMA_60",
            "EMA_12",
            "EMA_26",
            "BB_Upper",
            "BB_Middle",
            "BB_Lower",
            "MACD",
            "MACD_Signal",
            "MACD_Histogram",
            "RSI",
            "Stoch_K",
            "Stoch_D",
            "Volume_MA",
            "Volume_Ratio",
            "ATR",
        ]

        for indicator in expected_indicators:
            assert indicator in result.columns, f"指標 {indicator} が見つかりません"

    def test_calculate_all_with_custom_parameters(self, sample_data):
        """カスタムパラメータでのcalculate_allテスト"""
        result = TechnicalIndicators.calculate_all(
            sample_data,
            sma_periods=[10, 50],
            ema_periods=[9, 21],
            bb_period=25,
            rsi_period=21,
        )

        # カスタム設定の指標が含まれている
        assert "SMA_10" in result.columns
        assert "SMA_50" in result.columns
        assert "EMA_9" in result.columns
        assert "EMA_21" in result.columns

        # デフォルトの指標は含まれていない
        assert "SMA_5" not in result.columns
        assert "SMA_20" not in result.columns

    def test_error_handling_empty_dataframe(self):
        """空DataFrameのエラーハンドリングテスト"""
        empty_df = pd.DataFrame()

        # SMA
        sma = TechnicalIndicators.sma(empty_df)
        assert isinstance(sma, pd.Series)
        assert len(sma) == 0

        # calculate_all
        result = TechnicalIndicators.calculate_all(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_error_handling_missing_columns(self):
        """必須列欠如のエラーハンドリングテスト"""
        df_no_close = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [102, 103, 104],
                "Low": [99, 100, 101],
            }
        )

        # Close列がない場合のSMA
        sma = TechnicalIndicators.sma(df_no_close)
        assert isinstance(sma, pd.Series)
        assert len(sma) == 0

    def test_performance_settings(self, sample_data):
        """パフォーマンス設定テスト"""
        # 進捗表示なしで実行
        result1 = TechnicalIndicators.calculate_all(sample_data, show_progress=False)

        # 進捗表示ありで実行（データサイズが閾値以下なので実際は表示されない）
        result2 = TechnicalIndicators.calculate_all(sample_data, show_progress=True)

        # 結果が同じであることを確認
        pd.testing.assert_frame_equal(result1, result2)

    def test_detailed_logging_validation(self, sample_data):
        """詳細ログ機能のテスト"""
        # 正常なケースでのログ出力を確認
        result = TechnicalIndicators.calculate_all(sample_data)
        assert isinstance(result, pd.DataFrame)

        # 不正なデータでのエラーログを確認
        bad_data = pd.DataFrame({"Close": [np.nan, np.inf, -np.inf]})
        bad_data.index = pd.date_range(start="2023-01-01", periods=3, freq="D")

        # エラーが発生しても例外は発生せず、適切に処理される
        rsi = TechnicalIndicators.rsi(bad_data)
        assert isinstance(rsi, pd.Series)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
