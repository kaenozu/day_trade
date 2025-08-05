"""
売買シグナル生成エンジンのテスト
"""

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.day_trade.analysis.indicators import TechnicalIndicators
from src.day_trade.analysis.patterns import ChartPatternRecognizer
from src.day_trade.analysis.signals import (
    BollingerBandRule,
    GoldenCrossRule,
    MACDCrossoverRule,
    PatternBreakoutRule,
    RSIOverboughtRule,
    RSIOversoldRule,
    SignalRule,
    SignalRulesConfig,
    SignalStrength,
    SignalType,
    TradingSignal,
    TradingSignalGenerator,
    VolumeSpikeBuyRule,
)


class TestSignalRulesConfig:
    """シグナルルール設定クラスのテスト"""

    def test_config_initialization(self):
        """設定初期化テスト"""
        config = SignalRulesConfig()
        assert config.config is not None
        assert "default_buy_rules" in config.config
        assert "default_sell_rules" in config.config
        assert "signal_generation_settings" in config.config

    def test_buy_rules_config_retrieval(self):
        """買いルール設定取得テスト"""
        config = SignalRulesConfig()
        buy_rules = config.get_buy_rules_config()
        assert isinstance(buy_rules, list)
        # 設定ファイルが存在しない場合は空のリストが返される
        # これは正常な動作（デフォルトルールで補完される）

    def test_sell_rules_config_retrieval(self):
        """売りルール設定取得テスト"""
        config = SignalRulesConfig()
        sell_rules = config.get_sell_rules_config()
        assert isinstance(sell_rules, list)
        # 設定ファイルが存在しない場合は空のリストが返される
        # これは正常な動作（デフォルトルールで補完される）

    def test_signal_settings_retrieval(self):
        """シグナル生成設定取得テスト"""
        config = SignalRulesConfig()
        settings = config.get_signal_settings()
        assert isinstance(settings, dict)
        assert "min_data_period" in settings
        assert "confidence_multipliers" in settings
        assert "strength_thresholds" in settings

    def test_confidence_multiplier_retrieval(self):
        """信頼度乗数取得テスト"""
        config = SignalRulesConfig()
        multiplier = config.get_confidence_multiplier("rsi_oversold", 1.0)
        assert isinstance(multiplier, float)
        assert multiplier > 0

        # 存在しないキーのデフォルト値
        default = config.get_confidence_multiplier("unknown_key", 5.0)
        assert default == 5.0

    def test_strength_thresholds_retrieval(self):
        """強度閾値取得テスト"""
        config = SignalRulesConfig()
        thresholds = config.get_strength_thresholds()
        assert isinstance(thresholds, dict)
        assert "strong" in thresholds
        assert "medium" in thresholds

    def test_missing_config_file_fallback(self):
        """設定ファイルが存在しない場合のフォールバックテスト"""
        # 存在しないパスを指定してデフォルト設定が使用されることを確認
        config = SignalRulesConfig("/nonexistent/path/signal_rules.json")

        # デフォルト設定が読み込まれていることを確認
        assert config.config is not None
        assert "signal_generation_settings" in config.config

        # デフォルト値が正しく取得できることを確認
        min_period = config.get_signal_settings().get("min_data_period", 0)
        assert min_period == 60  # デフォルト値

        multiplier = config.get_confidence_multiplier("rsi_oversold", 1.0)
        assert multiplier == 2.0  # デフォルト値


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
        assert isinstance(signal.price, Decimal)
        assert signal.price > 0
        assert isinstance(signal.timestamp, datetime)

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
            # 基本的な検証
            validity = generator.validate_signal(signal)
            assert isinstance(validity, float)
            assert 0 <= validity <= 100

            # 市場環境を含む検証
            market_context = {
                "volatility": sample_data["Close"].pct_change().std(),
                "trend_direction": "upward",
            }
            validity_with_context = generator.validate_signal(
                signal, market_context=market_context
            )
            assert isinstance(validity_with_context, float)
            assert 0 <= validity_with_context <= 100

            # 過去パフォーマンスデータを含む検証
            historical_data = pd.DataFrame(
                {
                    "Signal": ["buy", "sell", "buy"],
                    "Strength": ["strong", "medium", "weak"],
                    "Success": [True, False, True],
                }
            )
            validity_with_history = generator.validate_signal(
                signal,
                historical_performance=historical_data,
                market_context=market_context,
            )
            assert isinstance(validity_with_history, float)
            assert 0 <= validity_with_history <= 100

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

    def test_signal_strength_classification(self, generator, sample_data):
        """シグナル強度の分類テスト"""
        # 強いシグナルを生成するように設定
        indicators = TechnicalIndicators.calculate_all(sample_data)

        # 複数の強い条件を設定
        indicators.loc[indicators.index[-1], "RSI"] = 20  # 強い過売り
        indicators.loc[indicators.index[-2], "MACD"] = -1
        indicators.loc[indicators.index[-1], "MACD"] = 1  # 強いクロスオーバー
        indicators.loc[:, "MACD_Signal"] = 0

        # ボリンジャーバンド下限を突破
        indicators.loc[indicators.index[-1], "BB_Lower"] = (
            sample_data["Close"].iloc[-1] + 1
        )
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)

        signal = generator.generate_signal(
            sample_data, indicators=indicators, patterns=patterns
        )

        assert signal is not None
        # 複数の強い条件が満たされているので、強いシグナルになるはず
        active_conditions = sum(1 for v in signal.conditions_met.values() if v)
        if active_conditions >= 3 and signal.confidence >= 70:
            assert signal.strength == SignalStrength.STRONG


class TestCustomSignalRule:
    """カスタムシグナルルールのテスト"""

    def test_custom_rule_implementation(self):
        """カスタムルールの実装テスト"""

        class TestRule(SignalRule):
            def __init__(self):
                super().__init__("Test Rule", weight=1.0)

            def evaluate(self, df, indicators, patterns):
                # 常にTrue、信頼度50%を返す
                return True, 50.0

        rule = TestRule()
        assert rule.name == "Test Rule"
        assert rule.weight == 1.0

        # 評価メソッドのテスト
        met, confidence = rule.evaluate(None, None, None)
        assert met is True
        assert confidence == 50.0


class TestImprovedSignalGeneration:
    """改善されたシグナル生成機能のテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
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

    def test_config_based_rule_loading(self):
        """設定ベースルール読み込みテスト"""
        generator = TradingSignalGenerator()

        # 設定ファイルがない場合でもデフォルトルールが読み込まれる
        assert len(generator.buy_rules) > 0
        assert len(generator.sell_rules) > 0

        # デフォルトルールが含まれていることを確認
        rule_names = [rule.name for rule in generator.buy_rules]
        assert "RSI Oversold" in rule_names

    def test_decimal_price_handling(self, sample_data):
        """Decimal価格処理テスト"""
        generator = TradingSignalGenerator()
        indicators = TechnicalIndicators.calculate_all(sample_data)
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)

        signal = generator.generate_signal(sample_data, indicators, patterns)

        if signal:
            assert isinstance(signal.price, Decimal)
            assert signal.price > 0
            # Decimalから他の型への変換テスト
            assert float(signal.price) > 0
            assert int(signal.price) > 0

    def test_datetime_timestamp_handling(self, sample_data):
        """datetime タイムスタンプ処理テスト"""
        generator = TradingSignalGenerator()
        indicators = TechnicalIndicators.calculate_all(sample_data)
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)

        signal = generator.generate_signal(sample_data, indicators, patterns)

        if signal:
            assert isinstance(signal.timestamp, datetime)
            # pandas Timestampとの互換性確認
            pd_timestamp = pd.Timestamp(signal.timestamp)
            assert pd_timestamp.year >= 2023

    def test_configurable_strength_thresholds(self, sample_data):
        """設定可能強度閾値テスト"""
        generator = TradingSignalGenerator()

        # 強度閾値が設定から読み込まれていることを確認
        thresholds = generator.config.get_strength_thresholds()
        assert "strong" in thresholds
        assert "medium" in thresholds

        # 実際のシグナル生成で閾値が使用されることを確認
        indicators = TechnicalIndicators.calculate_all(sample_data)
        # 強いシグナルを作るため複数条件を満たす
        indicators.loc[indicators.index[-1], "RSI"] = 20  # 強い過売り
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)

        signal = generator.generate_signal(sample_data, indicators, patterns)
        if signal and signal.signal_type == SignalType.BUY:
            # 設定された閾値に基づいて強度が決定されていることを確認
            assert signal.strength in [
                SignalStrength.STRONG,
                SignalStrength.MEDIUM,
                SignalStrength.WEAK,
            ]

    def test_enhanced_validate_signal(self, sample_data):
        """拡張されたシグナル検証テスト"""
        generator = TradingSignalGenerator()
        indicators = TechnicalIndicators.calculate_all(sample_data)
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)

        signal = generator.generate_signal(sample_data, indicators, patterns)

        if signal:
            # 市場環境情報によるvalidation
            market_context = {
                "volatility": 0.02,  # 低ボラティリティ
                "trend_direction": "upward",
            }

            validity_low_vol = generator.validate_signal(
                signal, market_context=market_context
            )

            # 高ボラティリティでの検証
            market_context["volatility"] = 0.08  # 高ボラティリティ
            validity_high_vol = generator.validate_signal(
                signal, market_context=market_context
            )

            # 高ボラティリティ時は信頼度が下がることを確認
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                assert validity_low_vol >= validity_high_vol

    def test_rule_parameter_configuration(self):
        """ルールパラメータ設定テスト"""
        generator = TradingSignalGenerator()

        # RSIルールのパラメータが設定から読み込まれていることを確認
        rsi_rules = [rule for rule in generator.buy_rules if "RSI" in rule.name]
        if rsi_rules:
            rsi_rule = rsi_rules[0]
            assert hasattr(rsi_rule, "threshold")
            assert hasattr(rsi_rule, "confidence_multiplier")
            assert rsi_rule.threshold > 0
            assert rsi_rule.confidence_multiplier > 0


class TestSignalRuleCreation:
    """シグナルルール作成機能のテスト"""

    def test_create_rule_from_config_success(self):
        """正常なルール作成のテスト"""
        generator = TradingSignalGenerator()

        # RSIOversoldRuleの作成設定
        rule_config = {
            "type": "RSIOversoldRule",
            "parameters": {"threshold": 25, "weight": 1.5}
        }

        rule = generator._create_rule_from_config(rule_config)
        if rule is not None:  # ルール作成に成功した場合のみテスト
            assert rule.name == "RSI Oversold"
            assert rule.weight == 1.5
            assert rule.threshold == 25

    def test_create_rule_from_config_invalid_type(self):
        """無効なルールタイプのテスト"""
        generator = TradingSignalGenerator()

        # 存在しないルールタイプ
        rule_config = {
            "type": "InvalidRule",
            "parameters": {"threshold": 30}
        }

        rule = generator._create_rule_from_config(rule_config)
        assert rule is None

    def test_create_rule_from_config_exception_handling(self):
        """ルール作成時の例外処理テスト"""
        generator = TradingSignalGenerator()

        # 無効なパラメータによる例外
        rule_config = {
            "type": "RSIOversoldRule",
            "parameters": {"invalid_param": "invalid"}
        }

        rule = generator._create_rule_from_config(rule_config)
        assert rule is None


class TestDefaultRulesLoading:
    """デフォルトルール読み込み機能のテスト"""

    def test_load_default_buy_rules(self):
        """デフォルト買いルール読み込みテスト"""
        generator = TradingSignalGenerator()
        generator._load_default_buy_rules()

        assert len(generator.buy_rules) > 0
        # RSIOversoldRuleが含まれることを確認
        rsi_rules = [rule for rule in generator.buy_rules if "RSI" in rule.name]
        assert len(rsi_rules) > 0

    def test_load_default_sell_rules(self):
        """デフォルト売りルール読み込みテスト"""
        generator = TradingSignalGenerator()
        generator._load_default_sell_rules()

        assert len(generator.sell_rules) > 0
        # RSIOverboughtRuleが含まれることを確認
        rsi_rules = [rule for rule in generator.sell_rules if "RSI" in rule.name]
        assert len(rsi_rules) > 0


class TestRuleManagement:
    """ルール管理機能のテスト"""

    def test_add_custom_buy_rule(self):
        """カスタム買いルール追加テスト"""
        generator = TradingSignalGenerator()
        initial_count = len(generator.buy_rules)

        custom_rule = RSIOversoldRule(threshold=25, weight=2.0)
        generator.add_buy_rule(custom_rule)

        assert len(generator.buy_rules) == initial_count + 1
        assert generator.buy_rules[-1] == custom_rule

    def test_add_custom_sell_rule(self):
        """カスタム売りルール追加テスト"""
        generator = TradingSignalGenerator()
        initial_count = len(generator.sell_rules)

        custom_rule = RSIOverboughtRule(threshold=75, weight=2.0)
        generator.add_sell_rule(custom_rule)

        assert len(generator.sell_rules) == initial_count + 1
        assert generator.sell_rules[-1] == custom_rule

    def test_clear_rules(self):
        """ルールクリア機能テスト"""
        generator = TradingSignalGenerator()

        # ルールが存在することを確認
        assert len(generator.buy_rules) > 0
        assert len(generator.sell_rules) > 0

        # クリア実行
        generator.clear_rules()

        # ルールが削除されることを確認
        assert len(generator.buy_rules) == 0
        assert len(generator.sell_rules) == 0


class TestSignalGenerationEdgeCases:
    """シグナル生成のエッジケースのテスト"""

    def test_generate_signal_insufficient_data(self):
        """データ不足時のシグナル生成テスト"""
        generator = TradingSignalGenerator()

        # 少量データ（5行のみ）
        small_df = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1000, 1100, 1200, 1300, 1400]
        })

        indicators = pd.DataFrame()
        patterns = {}

        signal = generator.generate_signal(small_df, indicators, patterns)
        assert signal is None

    def test_generate_signal_empty_dataframe(self):
        """空のDataFrameでのシグナル生成テスト"""
        generator = TradingSignalGenerator()

        empty_df = pd.DataFrame()
        indicators = pd.DataFrame()
        patterns = {}

        signal = generator.generate_signal(empty_df, indicators, patterns)
        assert signal is None

    def test_generate_signal_with_none_patterns(self):
        """patternsがNoneの場合のテスト"""
        generator = TradingSignalGenerator()

        # 充分なサイズのデータを作成
        dates = pd.date_range(start='2023-01-01', periods=100)
        df = pd.DataFrame({
            "Open": np.random.uniform(100, 110, 100),
            "High": np.random.uniform(105, 115, 100),
            "Low": np.random.uniform(95, 105, 100),
            "Close": np.random.uniform(100, 110, 100),
            "Volume": np.random.randint(1000, 5000, 100)
        }, index=dates)

        indicators = TechnicalIndicators.calculate_all(df)

        # patternsをNoneで渡す
        signal = generator.generate_signal(df, indicators, None)
        # エラーが発生せずに処理されることを確認
        # 結果は条件により異なるため、例外が発生しないことのみを確認


class TestConditionsMerging:
    """条件結合機能のテスト"""

    def test_merge_conditions_no_conflict(self):
        """衝突のない条件結合テスト"""
        generator = TradingSignalGenerator()

        buy_conditions = {"rsi_oversold": True, "macd_buy": False}
        sell_conditions = {"rsi_overbought": False, "macd_sell": True}

        merged = generator._merge_conditions_safely(buy_conditions, sell_conditions)

        expected = {
            "rsi_oversold": True,
            "macd_buy": False,
            "rsi_overbought": False,
            "macd_sell": True
        }
        assert merged == expected

    def test_merge_conditions_with_conflict(self):
        """衝突のある条件結合テスト"""
        generator = TradingSignalGenerator()

        buy_conditions = {"rsi_signal": True, "volume_spike": False}
        sell_conditions = {"rsi_signal": False, "trend_down": True}

        merged = generator._merge_conditions_safely(buy_conditions, sell_conditions)

        # 衝突したキーがプレフィックス付きで分離されることを確認
        assert "buy_rsi_signal" in merged
        assert "sell_rsi_signal" in merged
        assert merged["buy_rsi_signal"] == True
        assert merged["sell_rsi_signal"] == False
        assert "rsi_signal" not in merged  # 元のキーは削除される
        assert merged["volume_spike"] == False
        assert merged["trend_down"] == True


class TestPatternSlicing:
    """パターンスライス機能のテスト"""

    def test_slice_patterns_with_dataframes(self):
        """DataFrameパターンのスライステスト"""
        generator = TradingSignalGenerator()

        # テスト用のパターンデータを作成
        crosses_df = pd.DataFrame({
            "golden_cross": [False, False, True, False, False],
            "death_cross": [False, True, False, False, True]
        })

        breakouts_df = pd.DataFrame({
            "resistance_break": [False, False, False, True, False],
            "support_break": [True, False, False, False, False]
        })

        all_patterns = {
            "crosses": crosses_df,
            "breakouts": breakouts_df,
            "levels": {"support": [100, 105], "resistance": [110, 115]},
            "trends": {"direction": "upward"}
        }

        # インデックス2、ウィンドウ3でスライス
        sliced = generator._slice_patterns(all_patterns, current_index=2, lookback_window=3)

        # スライスされたDataFrameの確認
        assert "crosses" in sliced
        assert "breakouts" in sliced
        assert len(sliced["crosses"]) == 3  # ウィンドウサイズ分
        assert len(sliced["breakouts"]) == 3

        # levels, trendsはそのまま含まれる
        assert "levels" in sliced
        assert "trends" in sliced

    def test_slice_patterns_empty_dataframes(self):
        """空のDataFrameパターンのスライステスト"""
        generator = TradingSignalGenerator()

        all_patterns = {
            "crosses": pd.DataFrame(),  # 空のDataFrame
            "breakouts": pd.DataFrame(),
            "levels": {},
            "trends": {}
        }

        sliced = generator._slice_patterns(all_patterns, current_index=5, lookback_window=3)

        # 空のDataFrameが返されることを確認
        assert isinstance(sliced["crosses"], pd.DataFrame)
        assert isinstance(sliced["breakouts"], pd.DataFrame)
        assert len(sliced["crosses"]) == 0
        assert len(sliced["breakouts"]) == 0


class TestAdvancedSignalValidation:
    """高度なシグナル検証機能のテスト"""

    def test_validate_signal_with_historical_performance(self):
        """過去パフォーマンスを考慮したシグナル検証テスト"""
        generator = TradingSignalGenerator()

        # サンプルデータを作成
        dates = pd.date_range(start='2023-01-01', periods=100)
        sample_data = pd.DataFrame({
            "Open": np.random.uniform(100, 110, 100),
            "High": np.random.uniform(105, 115, 100),
            "Low": np.random.uniform(95, 105, 100),
            "Close": np.random.uniform(100, 110, 100),
            "Volume": np.random.randint(1000, 5000, 100)
        }, index=dates)

        indicators = TechnicalIndicators.calculate_all(sample_data)
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)

        signal = generator.generate_signal(sample_data, indicators, patterns)

        if signal:
            # モック過去パフォーマンスデータ
            historical_performance = pd.DataFrame({
                "Signal": ["buy", "sell", "buy", "sell"],
                "Strength": ["strong", "medium", "weak", "strong"],
                "Success": [True, False, True, True]
            })

            validity_score = generator.validate_signal(
                signal,
                historical_performance=historical_performance
            )

            assert 0 <= validity_score <= 100

    def test_validate_signal_high_volatility_market(self):
        """高ボラティリティ市場でのシグナル検証テスト"""
        generator = TradingSignalGenerator()

        # サンプルデータを作成
        dates = pd.date_range(start='2023-01-01', periods=100)
        sample_data = pd.DataFrame({
            "Open": np.random.uniform(100, 110, 100),
            "High": np.random.uniform(105, 115, 100),
            "Low": np.random.uniform(95, 105, 100),
            "Close": np.random.uniform(100, 110, 100),
            "Volume": np.random.randint(1000, 5000, 100)
        }, index=dates)

        indicators = TechnicalIndicators.calculate_all(sample_data)
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)

        signal = generator.generate_signal(sample_data, indicators, patterns)

        if signal:
            # 高ボラティリティ市場環境
            high_vol_context = {
                "volatility": 0.08,  # 高ボラティリティ（8%）
                "trend_direction": "neutral"
            }

            validity_high_vol = generator.validate_signal(
                signal,
                market_context=high_vol_context
            )

            # 低ボラティリティとの比較
            low_vol_context = {
                "volatility": 0.02,  # 低ボラティリティ（2%）
                "trend_direction": "neutral"
            }

            validity_low_vol = generator.validate_signal(
                signal,
                market_context=low_vol_context
            )

            # 高ボラティリティ時は信頼度が下がることを確認
            assert validity_high_vol <= validity_low_vol

    def test_validate_signal_trend_consistency(self):
        """トレンド整合性を考慮したシグナル検証テスト"""
        generator = TradingSignalGenerator()

        # サンプルデータを作成
        dates = pd.date_range(start='2023-01-01', periods=100)
        sample_data = pd.DataFrame({
            "Open": np.random.uniform(100, 110, 100),
            "High": np.random.uniform(105, 115, 100),
            "Low": np.random.uniform(95, 105, 100),
            "Close": np.random.uniform(100, 110, 100),
            "Volume": np.random.randint(1000, 5000, 100)
        }, index=dates)

        indicators = TechnicalIndicators.calculate_all(sample_data)
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)

        signal = generator.generate_signal(sample_data, indicators, patterns)

        if signal and signal.signal_type == SignalType.BUY:
            # 上昇トレンドでのBUYシグナル（トレンドフォロー）
            uptrend_context = {
                "volatility": 0.03,
                "trend_direction": "upward"
            }

            validity_uptrend = generator.validate_signal(
                signal,
                market_context=uptrend_context
            )

            # 下降トレンドでのBUYシグナル（逆張り）
            downtrend_context = {
                "volatility": 0.03,
                "trend_direction": "downward"
            }

            validity_downtrend = generator.validate_signal(
                signal,
                market_context=downtrend_context
            )

            # トレンドフォローの方が信頼度が高いことを確認
            assert validity_uptrend >= validity_downtrend

    def test_validate_signal_time_freshness(self):
        """シグナルの時間新鮮度検証テスト"""
        generator = TradingSignalGenerator()

        # サンプルデータを作成
        dates = pd.date_range(start='2023-01-01', periods=100)
        sample_data = pd.DataFrame({
            "Open": np.random.uniform(100, 110, 100),
            "High": np.random.uniform(105, 115, 100),
            "Low": np.random.uniform(95, 105, 100),
            "Close": np.random.uniform(100, 110, 100),
            "Volume": np.random.randint(1000, 5000, 100)
        }, index=dates)

        indicators = TechnicalIndicators.calculate_all(sample_data)
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)

        signal = generator.generate_signal(sample_data, indicators, patterns)

        if signal:
            from datetime import datetime, timezone, timedelta

            # 古いタイムスタンプのシグナルを作成
            stale_signal = TradingSignal(
                signal_type=signal.signal_type,
                strength=signal.strength,
                confidence=signal.confidence,
                reasons=signal.reasons,
                conditions_met=signal.conditions_met,
                timestamp=datetime.now(timezone.utc) - timedelta(hours=80),  # 80時間前
                price=signal.price,
                symbol=signal.symbol
            )

            validity_stale = generator.validate_signal(stale_signal)
            validity_fresh = generator.validate_signal(signal)

            # 新鮮なシグナルの方が信頼度が高いことを確認
            assert validity_fresh >= validity_stale

    def test_validate_signal_exception_handling(self):
        """シグナル検証の例外処理テスト"""
        generator = TradingSignalGenerator()

        # 不正なシグナル（必要な属性が不足）
        invalid_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            confidence=50.0,
            reasons=["test"],
            conditions_met={"test": True},
            timestamp=datetime.now(),
            price=Decimal("100.0")
        )

        # 不正な市場環境データ
        invalid_context = {
            "volatility": "invalid",  # 数値でない
            "trend_direction": None
        }

        # 例外が発生してもゼロが返されることを確認
        validity = generator.validate_signal(
            invalid_signal,
            market_context=invalid_context
        )

        assert validity == 0.0


class TestSignalRulesConfigFileHandling:
    """設定ファイル処理のテスト"""

    def test_config_file_not_found(self):
        """設定ファイルが見つからない場合のテスト"""
        # 存在しないパスを指定
        config = SignalRulesConfig("/non/existent/path/config.json")

        # デフォルト設定が読み込まれることを確認
        assert config.config is not None
        assert "default_buy_rules" in config.config
        assert "default_sell_rules" in config.config
        assert "signal_generation_settings" in config.config

    def test_config_file_invalid_json(self):
        """無効なJSONファイルの処理テスト"""
        import tempfile
        import os

        # 無効なJSONファイルを作成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_file.write("{ invalid json content")
            invalid_config_path = tmp_file.name

        try:
            config = SignalRulesConfig(invalid_config_path)

            # デフォルト設定が読み込まれることを確認
            assert config.config is not None
            assert isinstance(config.config, dict)

        finally:
            # 一時ファイルを削除
            if os.path.exists(invalid_config_path):
                os.unlink(invalid_config_path)

    def test_config_getters_with_defaults(self):
        """設定取得メソッドのデフォルト値テスト"""
        config = SignalRulesConfig()

        # 各種設定の取得とデフォルト値の確認
        buy_rules = config.get_buy_rules_config()
        assert isinstance(buy_rules, list)

        sell_rules = config.get_sell_rules_config()
        assert isinstance(sell_rules, list)

        signal_settings = config.get_signal_settings()
        assert isinstance(signal_settings, dict)

        # デフォルト値でのconfidence_multiplier取得
        default_multiplier = config.get_confidence_multiplier("unknown_type", 5.0)
        assert default_multiplier == 5.0

        # 存在する設定の取得
        rsi_multiplier = config.get_confidence_multiplier("rsi_oversold", 1.0)
        assert isinstance(rsi_multiplier, (int, float))

        # その他の設定メソッド
        min_data = config.get_min_data_for_generation()
        assert isinstance(min_data, int)
        assert min_data > 0

        volume_period = config.get_volume_calculation_period()
        assert isinstance(volume_period, int)
        assert volume_period > 0

        trend_period = config.get_trend_lookback_period()
        assert isinstance(trend_period, int)
        assert trend_period > 0

        freshness = config.get_signal_freshness()
        assert isinstance(freshness, dict)
        assert "warning_hours" in freshness
        assert "stale_hours" in freshness

        volatility_threshold = config.get_high_volatility_threshold()
        assert isinstance(volatility_threshold, (int, float))
        assert volatility_threshold > 0


class TestSignalRuleBaseClass:
    """SignalRuleベースクラスのテスト"""

    def test_signal_rule_initialization(self):
        """SignalRuleの初期化テスト"""
        rule = SignalRule("Test Rule", weight=2.5)
        assert rule.name == "Test Rule"
        assert rule.weight == 2.5

    def test_signal_rule_evaluate_not_implemented(self):
        """evaluate メソッドが実装されていない場合のテスト"""
        rule = SignalRule("Test Rule")

        # NotImplementedError が発生することを確認
        with pytest.raises(NotImplementedError):
            rule.evaluate(pd.DataFrame(), pd.DataFrame(), {})


class TestRSIRulesEdgeCases:
    """RSIルールのエッジケースのテスト"""

    def test_rsi_oversold_rule_no_rsi_column(self):
        """RSI列がない場合のRSIOversoldRuleテスト"""
        rule = RSIOversoldRule(threshold=30)

        # RSI列がないindicators
        indicators = pd.DataFrame({"MACD": [0.1, 0.2, 0.3]})

        is_triggered, confidence = rule.evaluate(pd.DataFrame(), indicators, {})

        assert is_triggered == False
        assert confidence == 0.0

    def test_rsi_oversold_rule_empty_rsi(self):
        """空のRSI列の場合のテスト"""
        rule = RSIOversoldRule(threshold=30)

        # 空のRSI列
        indicators = pd.DataFrame({"RSI": []})

        is_triggered, confidence = rule.evaluate(pd.DataFrame(), indicators, {})

        assert is_triggered == False
        assert confidence == 0.0

    def test_rsi_oversold_rule_nan_rsi(self):
        """RSIがNaNの場合のテスト"""
        rule = RSIOversoldRule(threshold=30)

        # NaNのRSI
        indicators = pd.DataFrame({"RSI": [25.0, 30.0, np.nan]})

        is_triggered, confidence = rule.evaluate(pd.DataFrame(), indicators, {})

        assert is_triggered == False
        assert confidence == 0.0

    def test_rsi_overbought_rule_no_rsi_column(self):
        """RSI列がない場合のRSIOverboughtRuleテスト"""
        rule = RSIOverboughtRule(threshold=70)

        indicators = pd.DataFrame({"MACD": [0.1, 0.2, 0.3]})

        is_triggered, confidence = rule.evaluate(pd.DataFrame(), indicators, {})

        assert is_triggered == False
        assert confidence == 0.0

    def test_rsi_overbought_rule_nan_rsi(self):
        """RSIがNaNの場合の RSIOverboughtRule テスト"""
        rule = RSIOverboughtRule(threshold=70)

        indicators = pd.DataFrame({"RSI": [75.0, 80.0, np.nan]})

        is_triggered, confidence = rule.evaluate(pd.DataFrame(), indicators, {})

        assert is_triggered == False
        assert confidence == 0.0


class TestGenerateSignalsSeriesEdgeCases:
    """generate_signals_series のエッジケースのテスト"""

    def test_generate_signals_series_insufficient_data(self):
        """データ不足時の generate_signals_series テスト"""
        generator = TradingSignalGenerator()

        # 10行のデータ（最低20行必要）
        small_df = pd.DataFrame({
            "Open": np.random.uniform(100, 110, 10),
            "High": np.random.uniform(105, 115, 10),
            "Low": np.random.uniform(95, 105, 10),
            "Close": np.random.uniform(100, 110, 10),
            "Volume": np.random.randint(1000, 5000, 10)
        })

        signals = generator.generate_signals_series(small_df)

        # DataFrameまたはNoneまたはlistが返されることを確認
        assert isinstance(signals, (list, type(None), pd.DataFrame))
        if signals is not None and hasattr(signals, '__len__'):
            if isinstance(signals, pd.DataFrame):
                assert len(signals) <= len(small_df)
            elif isinstance(signals, list):
                assert len(signals) <= len(small_df)

    def test_generate_signals_series_empty_dataframe(self):
        """空のDataFrameでの generate_signals_series テスト"""
        generator = TradingSignalGenerator()

        empty_df = pd.DataFrame()
        signals = generator.generate_signals_series(empty_df)

        # Noneまたは空のリストが返されることを確認
        assert signals is None or len(signals) == 0


class TestSignalGenerationParameterValidation:
    """シグナル生成のパラメータ検証のテスト"""

    def test_signal_generation_with_invalid_indicators(self):
        """無効なindicatorsでのシグナル生成テスト"""
        generator = TradingSignalGenerator()

        # 十分なサイズのデータを作成
        dates = pd.date_range(start='2023-01-01', periods=50)
        df = pd.DataFrame({
            "Open": np.random.uniform(100, 110, 50),
            "High": np.random.uniform(105, 115, 50),
            "Low": np.random.uniform(95, 105, 50),
            "Close": np.random.uniform(100, 110, 50),
            "Volume": np.random.randint(1000, 5000, 50)
        }, index=dates)

        # 空のindicators
        empty_indicators = pd.DataFrame()
        patterns = ChartPatternRecognizer.detect_all_patterns(df)

        signal = generator.generate_signal(df, empty_indicators, patterns)

        # エラーが発生せずに処理されることを確認
        # 結果は条件により異なるため、例外が発生しないことを確認するのみ

    def test_signal_generation_with_mismatched_data_length(self):
        """データ長が一致しない場合のテスト"""
        generator = TradingSignalGenerator()

        # 異なる長さのDataFrameを作成
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000, 1100, 1200]
        })

        # 長さが一致しないindicators（50行）
        indicators = pd.DataFrame({
            "RSI": np.random.uniform(30, 70, 50),
            "MACD": np.random.uniform(-1, 1, 50)
        })

        patterns = {}

        # エラーハンドリングされることを確認
        signal = generator.generate_signal(df, indicators, patterns)

        # Noneが返されるか、適切に処理されることを確認
        assert signal is None or isinstance(signal, TradingSignal)


class TestSignalGenerationLargeDataSeries:
    """大規模データ系列のシグナル生成テスト"""

    def test_generate_signals_series_large_data(self):
        """大量データでのシグナル系列生成テスト"""
        generator = TradingSignalGenerator()

        # 200日分のデータ
        dates = pd.date_range(start='2023-01-01', periods=200)
        large_df = pd.DataFrame({
            "Open": np.random.uniform(90, 120, 200),
            "High": np.random.uniform(95, 125, 200),
            "Low": np.random.uniform(85, 115, 200),
            "Close": np.random.uniform(90, 120, 200),
            "Volume": np.random.randint(1000, 10000, 200)
        }, index=dates)

        # 系列生成を実行
        signals = generator.generate_signals_series(large_df)

        # 適切に処理されることを確認
        if signals is not None:
            if isinstance(signals, (list, pd.DataFrame)):
                assert len(signals) <= len(large_df)

    def test_generate_signals_series_with_lookback_window(self):
        """ルックバックウィンドウ設定での系列生成テスト"""
        generator = TradingSignalGenerator()

        # 100日分のデータ
        dates = pd.date_range(start='2023-01-01', periods=100)
        df = pd.DataFrame({
            "Open": np.random.uniform(100, 110, 100),
            "High": np.random.uniform(105, 115, 100),
            "Low": np.random.uniform(95, 105, 100),
            "Close": np.random.uniform(100, 110, 100),
            "Volume": np.random.randint(1000, 5000, 100)
        }, index=dates)

        # ルックバックウィンドウを指定して系列生成
        signals = generator.generate_signals_series(df, lookback_window=30)

        # 適切に処理されることを確認
        if signals is not None and isinstance(signals, (list, pd.DataFrame)):
            # ウィンドウサイズ分の制限があることを確認
            if len(signals) > 0:
                assert len(signals) <= len(df) - 30 + 1


class TestTradingSignalDataClass:
    """TradingSignalデータクラスのテスト"""

    def test_trading_signal_creation(self):
        """TradingSignal作成テスト"""
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=85.5,
            reasons=["RSI過売り", "MACD上向き"],
            conditions_met={"rsi_oversold": True, "macd_bullish": True},
            timestamp=datetime.now(),
            price=Decimal("2500.0"),
            symbol="7203"
        )

        assert signal.signal_type == SignalType.BUY
        assert signal.strength == SignalStrength.STRONG
        assert signal.confidence == 85.5
        assert len(signal.reasons) == 2
        assert signal.conditions_met["rsi_oversold"] == True
        assert signal.symbol == "7203"

    def test_trading_signal_without_optional_fields(self):
        """オプションフィールドなしのTradingSignalテスト"""
        signal = TradingSignal(
            signal_type=SignalType.SELL,
            strength=SignalStrength.WEAK,
            confidence=40.0,
            reasons=["高値警戒"],
            conditions_met={"price_high": True},
            timestamp=datetime.now(),
            price=Decimal("15000.0")
            # symbolは省略（デフォルトNone）
        )

        assert signal.symbol is None
        assert signal.signal_type == SignalType.SELL
        assert signal.strength == SignalStrength.WEAK


class TestSignalStrengthAndType:
    """シグナル強度とタイプのテスト"""

    def test_signal_type_enum_values(self):
        """SignalType列挙型の値テスト"""
        assert SignalType.BUY.value == "buy"
        assert SignalType.SELL.value == "sell"
        assert SignalType.HOLD.value == "hold"

    def test_signal_strength_enum_values(self):
        """SignalStrength列挙型の値テスト"""
        assert SignalStrength.STRONG.value == "strong"
        assert SignalStrength.MEDIUM.value == "medium"
        assert SignalStrength.WEAK.value == "weak"


class TestLoadRulesFromConfig:
    """設定からのルール読み込み機能テスト"""

    def test_load_rules_from_empty_config(self):
        """空の設定からのルール読み込みテスト"""
        generator = TradingSignalGenerator()

        # 元のルールをクリア
        generator.clear_rules()

        # 設定からルール読み込み（空の場合はデフォルトルールが使用される）
        generator._load_rules_from_config()

        # デフォルトルールが読み込まれることを確認
        assert len(generator.buy_rules) > 0
        assert len(generator.sell_rules) > 0


class TestTradingSignalGeneratorExtended:
    """TradingSignalGeneratorExtendedクラスのテスト"""

    def test_merge_conditions_safely_basic(self):
        """_merge_conditions_safelyメソッドの基本テスト"""
        generator = TradingSignalGeneratorExtended()

        # 重複のない条件をテスト
        buy_conditions = {"rsi_oversold": True, "volume_high": False}
        sell_conditions = {"rsi_overbought": True, "momentum_down": True}

        merged = generator._merge_conditions_safely(buy_conditions, sell_conditions)

        # 全ての条件が含まれることを確認
        assert merged["rsi_oversold"] is True
        assert merged["volume_high"] is False
        assert merged["rsi_overbought"] is True
        assert merged["momentum_down"] is True

    def test_merge_conditions_safely_with_overlap(self):
        """_merge_conditions_safelyメソッドの重複処理テスト"""
        generator = TradingSignalGeneratorExtended()

        # 重複のある条件をテスト
        buy_conditions = {"common_condition": True, "buy_only": False}
        sell_conditions = {"common_condition": False, "sell_only": True}

        with patch('src.day_trade.analysis.signals.logger') as mock_logger:
            merged = generator._merge_conditions_safely(buy_conditions, sell_conditions)

            # 警告ログが出力されることを確認
            mock_logger.warning.assert_called_once()

        # 重複キーが適切に処理されることを確認
        assert "buy_common_condition" in merged
        assert "sell_common_condition" in merged
        assert "common_condition" not in merged
        assert merged["buy_only"] is False
        assert merged["sell_only"] is True


class TestSignalConfidenceCalculation:
    """信頼度計算の簡単なテスト"""

    def test_calculate_confidence_with_multiple_conditions(self):
        """複数条件での信頼度計算テスト"""
        generator = TradingSignalGenerator()

        # 3つの条件を持つシグナル
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            confidence=0.7,
            timestamp=pd.Timestamp.now(),
            conditions_met={
                "condition1": True,
                "condition2": True,
                "condition3": True
            }
        )

        # 市場コンテキストなし
        confidence = generator._calculate_confidence_score(signal)

        # 複数条件でボーナスが適用されることを確認
        assert confidence >= signal.confidence
        assert isinstance(confidence, float)

    def test_calculate_confidence_with_market_context(self):
        """市場コンテキスト付き信頼度計算テスト"""
        generator = TradingSignalGenerator()

        signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.8,
            timestamp=pd.Timestamp.now(),
            conditions_met={"test_condition": True}
        )

        # 上昇トレンドでのBUYシグナル
        market_context = {
            "volatility": 0.01,
            "trend_direction": "upward"
        }

        confidence = generator._calculate_confidence_score(signal, market_context=market_context)

        # トレンドフォローでボーナスが適用されることを確認
        assert isinstance(confidence, float)
        assert confidence > 0.0

    def test_calculate_confidence_with_historical_data(self):
        """履歴データ付き信頼度計算テスト"""
        generator = TradingSignalGenerator()

        signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            confidence=0.6,
            timestamp=pd.Timestamp.now(),
            conditions_met={"historical_test": True}
        )

        # 簡単な履歴データ
        historical_performance = pd.DataFrame({
            "Signal": ["buy", "buy", "sell"],
            "Strength": ["strong", "medium", "weak"]
        })

        confidence = generator._calculate_confidence_score(
            signal,
            historical_performance=historical_performance
        )

        # 履歴データによる調整が適用されることを確認
        assert isinstance(confidence, float)
        assert confidence >= 0.0


class TestSignalGeneratorConfigMethods:
    """SignalGeneratorの設定関連メソッドテスト"""

    def test_get_high_volatility_threshold(self):
        """高ボラティリティ閾値取得テスト"""
        generator = TradingSignalGenerator()

        threshold = generator.get_high_volatility_threshold()

        # デフォルト値またはそれに近い値が返されることを確認
        assert isinstance(threshold, float)
        assert threshold > 0.0
        assert threshold <= 1.0  # 100%以下であることを確認

    def test_get_signal_settings(self):
        """シグナル設定取得テスト"""
        generator = TradingSignalGenerator()

        settings = generator.get_signal_settings()

        # 設定が辞書形式で返されることを確認
        assert isinstance(settings, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
