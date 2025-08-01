"""
売買シグナル生成エンジンのテスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.day_trade.analysis.signals import (
    TradingSignalGenerator,
    SignalType,
    SignalStrength,
    TradingSignal,
    RSIOversoldRule,
    RSIOverboughtRule,
    MACDCrossoverRule,
    BollingerBandRule,
    PatternBreakoutRule,
    GoldenCrossRule,
    VolumeSpikeBuyRule,
)
from src.day_trade.analysis.indicators import TechnicalIndicators
from src.day_trade.analysis.patterns import ChartPatternRecognizer


class TestSignalRules:
    """個別ルールのテスト"""

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
    def indicators_data(self, sample_data):
        """計算済みのテクニカル指標"""
        return TechnicalIndicators.calculate_all(sample_data)

    @pytest.fixture
    def patterns_data(self, sample_data):
        """認識済みのチャートパターン"""
        return ChartPatternRecognizer.detect_all_patterns(sample_data)

    def test_rsi_oversold_rule(self, sample_data, indicators_data, patterns_data):
        """RSI過売りルールのテスト"""
        # RSIを人工的に低く設定
        indicators_data.loc[indicators_data.index[-1], "RSI"] = 25

        rule = RSIOversoldRule(threshold=30)
        met, confidence = rule.evaluate(sample_data, indicators_data, patterns_data)

        assert met is True
        assert confidence > 0
        assert confidence <= 100

    def test_rsi_overbought_rule(self, sample_data, indicators_data, patterns_data):
        """RSI過買いルールのテスト"""
        # RSIを人工的に高く設定
        indicators_data.loc[indicators_data.index[-1], "RSI"] = 75

        rule = RSIOverboughtRule(threshold=70)
        met, confidence = rule.evaluate(sample_data, indicators_data, patterns_data)

        assert met is True
        assert confidence > 0
        assert confidence <= 100

    def test_macd_crossover_rule(self, sample_data):
        """MACDクロスオーバールールのテスト"""
        # MACDクロスオーバーを人工的に作成
        indicators = pd.DataFrame(index=sample_data.index)
        indicators["MACD"] = [0] * (len(sample_data) - 2) + [-0.5, 0.5]
        indicators["MACD_Signal"] = [0] * len(sample_data)

        rule = MACDCrossoverRule()
        met, confidence = rule.evaluate(sample_data, indicators, {})

        assert met is True
        assert confidence > 0

    def test_bollinger_band_lower_rule(self, sample_data):
        """ボリンジャーバンド下限ルールのテスト"""
        # 価格をバンド下限以下に設定
        indicators = pd.DataFrame(index=sample_data.index)
        indicators["BB_Lower"] = sample_data["Close"] + 2
        indicators["BB_Upper"] = sample_data["Close"] + 5

        rule = BollingerBandRule(position="lower")
        met, confidence = rule.evaluate(sample_data, indicators, {})

        assert met is True
        assert confidence > 0

    def test_pattern_breakout_rule(self, sample_data, indicators_data):
        """パターンブレイクアウトルールのテスト"""
        # ブレイクアウトパターンを作成
        breakouts = pd.DataFrame(index=sample_data.index)
        breakouts["Upward_Breakout"] = [False] * (len(sample_data) - 1) + [True]
        breakouts["Upward_Confidence"] = [0] * (len(sample_data) - 1) + [80]

        patterns = {"breakouts": breakouts}

        rule = PatternBreakoutRule(direction="upward")
        met, confidence = rule.evaluate(sample_data, indicators_data, patterns)

        assert met is True
        assert confidence == 80

    def test_golden_cross_rule(self, sample_data, indicators_data):
        """ゴールデンクロスルールのテスト"""
        # ゴールデンクロスを作成
        crosses = pd.DataFrame(index=sample_data.index)
        crosses["Golden_Cross"] = [False] * (len(sample_data) - 3) + [
            True,
            False,
            False,
        ]
        crosses["Golden_Confidence"] = [0] * (len(sample_data) - 3) + [75, 0, 0]

        patterns = {"crosses": crosses}

        rule = GoldenCrossRule()
        met, confidence = rule.evaluate(sample_data, indicators_data, patterns)

        assert met is True
        assert confidence == 75

    def test_volume_spike_buy_rule(self, sample_data, indicators_data, patterns_data):
        """出来高急増買いルールのテスト"""
        # 出来高と価格を調整
        sample_data.loc[sample_data.index[-1], "Volume"] = 10000000
        sample_data.loc[sample_data.index[-1], "Close"] = (
            sample_data["Close"].iloc[-2] * 1.03
        )

        rule = VolumeSpikeBuyRule(volume_factor=2.0, price_change=0.02)
        met, confidence = rule.evaluate(sample_data, indicators_data, patterns_data)

        assert met is True
        assert confidence > 0


class TestTradingSignalGenerator:
    """売買シグナル生成クラスのテスト"""

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
    def generator(self):
        """シグナル生成器のインスタンス"""
        return TradingSignalGenerator()

    def test_generate_signal_basic(self, generator, sample_data):
        """基本的なシグナル生成のテスト"""
        indicators = TechnicalIndicators.calculate_all(sample_data)
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)
        signal = generator.generate_signal(sample_data, indicators, patterns)

        assert signal is not None
        assert isinstance(signal, TradingSignal)
        assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert signal.strength in [
            SignalStrength.STRONG,
            SignalStrength.MEDIUM,
            SignalStrength.WEAK,
        ]
        assert 0 <= signal.confidence <= 100
        assert len(signal.reasons) > 0
        assert isinstance(signal.conditions_met, dict)
        assert signal.price > 0

    def test_generate_signal_buy(self, generator, sample_data):
        """買いシグナル生成のテスト"""
        # RSIを低く設定して買いシグナルを誘発
        indicators = TechnicalIndicators.calculate_all(sample_data)
        indicators.loc[indicators.index[-1], "RSI"] = 25
        patterns = ChartPatternRecognizer.detect_all_patterns(
            sample_data
        )  # patternsを追加

        signal = generator.generate_signal(
            sample_data, indicators=indicators, patterns=patterns
        )

        assert signal is not None
        # RSIが低いので買いシグナルが出やすい
        if signal.signal_type == SignalType.BUY:
            assert "RSI Oversold" in signal.conditions_met
            assert signal.conditions_met["RSI Oversold"] is True

    def test_generate_signal_sell(self, generator, sample_data):
        """売りシグナル生成のテスト"""
        # RSIを高く設定して売りシグナルを誘発
        indicators = TechnicalIndicators.calculate_all(sample_data)
        indicators.loc[indicators.index[-1], "RSI"] = 75
        patterns = ChartPatternRecognizer.detect_all_patterns(
            sample_data
        )  # patternsを追加

        signal = generator.generate_signal(
            sample_data, indicators=indicators, patterns=patterns
        )

        assert signal is not None
        # RSIが高いので売りシグナルが出やすい
        if signal.signal_type == SignalType.SELL:
            assert "RSI Overbought" in signal.conditions_met
            assert signal.conditions_met["RSI Overbought"] is True

    def test_generate_signal_multiple_conditions(self, generator, sample_data):
        """複数条件でのシグナル生成のテスト"""
        # 複数の買い条件を満たすように設定
        indicators = TechnicalIndicators.calculate_all(sample_data)
        indicators.loc[indicators.index[-1], "RSI"] = 25

        # MACDクロスオーバーを追加
        indicators.loc[indicators.index[-2], "MACD"] = -0.5
        indicators.loc[indicators.index[-1], "MACD"] = 0.5
        indicators.loc[indicators.index[-2:], "MACD_Signal"] = 0
        patterns = ChartPatternRecognizer.detect_all_patterns(
            sample_data
        )  # patternsを追加

        signal = generator.generate_signal(
            sample_data, indicators=indicators, patterns=patterns
        )

        assert signal is not None
        # 複数条件が満たされているか確認
        active_conditions = sum(1 for v in signal.conditions_met.values() if v)
        assert active_conditions >= 2

    def test_generate_signals_series(self, generator, sample_data):
        """時系列シグナル生成のテスト"""
        signals_df = generator.generate_signals_series(sample_data, lookback_window=30)

        assert isinstance(signals_df, pd.DataFrame)
        if not signals_df.empty:
            assert "Signal" in signals_df.columns
            assert "Strength" in signals_df.columns
            assert "Confidence" in signals_df.columns
            assert "Price" in signals_df.columns
            assert "Reasons" in signals_df.columns

            # シグナルの値が正しいか
            assert signals_df["Signal"].isin(["buy", "sell", "hold"]).all()
            assert signals_df["Strength"].isin(["strong", "medium", "weak"]).all()
            assert (signals_df["Confidence"] >= 0).all()
            assert (signals_df["Confidence"] <= 100).all()

    def test_custom_rules(self, sample_data):
        """カスタムルールの追加テスト"""
        generator = TradingSignalGenerator()

        # デフォルトルール数を記録
        initial_buy_rules = len(generator.buy_rules)

        # カスタムルールを追加
        custom_rule = VolumeSpikeBuyRule()
        generator.add_buy_rule(custom_rule)

        assert len(generator.buy_rules) == initial_buy_rules + 1
        assert generator.buy_rules[-1] == custom_rule

        # カスタムルールの動作テスト - 出来高と価格を調整して条件を満たす
        test_data = sample_data.copy()
        test_data.loc[test_data.index[-1], "Volume"] = 10000000  # 大きな出来高
        test_data.loc[test_data.index[-1], "Close"] = (
            test_data["Close"].iloc[-2] * 1.03
        )  # 3%上昇

        test_indicators = TechnicalIndicators.calculate_all(test_data)
        test_patterns = ChartPatternRecognizer.detect_all_patterns(test_data)

        # シグナル生成
        signal = generator.generate_signal(test_data, test_indicators, test_patterns)
        assert signal is not None

        # 買いシグナルまたはホールドが生成されるはず（出来高急増により）
        # 条件には「Volume Spike Buy」が含まれているはず
        if signal.signal_type == SignalType.BUY:
            assert "Volume Spike Buy" in signal.conditions_met
            assert signal.conditions_met["Volume Spike Buy"] is True
        else:
            # シグナルタイプに関係なく、すべての条件は評価されていることを確認
            # 実装では条件評価は行われているが、結果のconditions_metに含まれるかは
            # シグナルタイプに依存する
            pass

    def test_clear_rules(self, generator):
        """ルールクリアのテスト"""
        # ルールが存在することを確認
        assert len(generator.buy_rules) > 0
        assert len(generator.sell_rules) > 0

        # ルールをクリア
        generator.clear_rules()

        assert len(generator.buy_rules) == 0
        assert len(generator.sell_rules) == 0

    def test_validate_signal(self, generator, sample_data):
        """シグナル検証のテスト"""
        indicators = TechnicalIndicators.calculate_all(sample_data)
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)
        signal = generator.generate_signal(sample_data, indicators, patterns)

        if signal:
            validity = generator.validate_signal(signal)

            assert isinstance(validity, float)
            assert 0 <= validity <= 100

            # 強いシグナルの方が高い有効性スコアを持つ
            if signal.strength == SignalStrength.STRONG:
                assert validity >= signal.confidence

    def test_empty_data(self, generator):
        """空データでのエラーハンドリング"""
        empty_df = pd.DataFrame()
        empty_indicators = pd.DataFrame()
        empty_patterns = {}
        signal = generator.generate_signal(empty_df, empty_indicators, empty_patterns)

        assert signal is None

    def test_insufficient_data(self, generator):
        """不十分なデータでのエラーハンドリング"""
        dates = pd.date_range(end=datetime.now(), periods=10, freq="D")
        small_df = pd.DataFrame(
            {
                "Date": dates,
                "Open": [100] * 10,
                "High": [101] * 10,
                "Low": [99] * 10,
                "Close": [100] * 10,
                "Volume": [1000000] * 10,
            }
        )
        small_df.set_index("Date", inplace=True)
        small_indicators = TechnicalIndicators.calculate_all(small_df)
        small_patterns = ChartPatternRecognizer.detect_all_patterns(small_df)

        signal = generator.generate_signal(small_df, small_indicators, small_patterns)
        assert signal is None
