"""
テクニカル指標計算エンジンのテスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.day_trade.analysis.indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """TechnicalIndicatorsクラスのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        dates = pd.date_range(end=datetime.now(), periods=50, freq="D")
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
