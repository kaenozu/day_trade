"""
Test for Trading Signal Generation Engine
"""

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

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
    """Test class for signal rule configuration"""

    def test_config_initialization(self):
        """Test config initialization"""
        config = SignalRulesConfig()
        assert config.config is not None
        assert "default_buy_rules" in config.config
        assert "default_sell_rules" in config.config
        assert "signal_generation_settings" in config.config

    def test_buy_rules_config_retrieval(self):
        """Test retrieval of buy rule configuration"""
        config = SignalRulesConfig()
        buy_rules = config.get_buy_rules_config()
        assert isinstance(buy_rules, list)
        # If config file is missing, an empty list is returned (default rules are used)

    def test_sell_rules_config_retrieval(self):
        """Test retrieval of sell rule configuration"""
        config = SignalRulesConfig()
        sell_rules = config.get_sell_rules_config()
        assert isinstance(sell_rules, list)
        # If config file is missing, an empty list is returned (default rules are used)

    def test_signal_settings_retrieval(self):
        """Test retrieval of signal generation settings"""
        config = SignalRulesConfig()
        settings = config.get_signal_settings()
        assert isinstance(settings, dict)
        assert "min_data_period" in settings
        assert "confidence_multipliers" in settings
        assert "strength_thresholds" in settings

    def test_confidence_multiplier_retrieval(self):
        """Test retrieval of confidence multiplier"""
        config = SignalRulesConfig()
        multiplier = config.get_confidence_multiplier("rsi_oversold", 1.0)
        assert isinstance(multiplier, float)
        assert multiplier > 0

        # Default value for non-existent key
        default = config.get_confidence_multiplier("unknown_key", 5.0)
        assert default == 5.0

    def test_strength_thresholds_retrieval(self):
        """Test retrieval of strength thresholds"""
        config = SignalRulesConfig()
        thresholds = config.get_strength_thresholds()
        assert isinstance(thresholds, dict)
        assert "strong" in thresholds
        assert "medium" in thresholds

    def test_missing_config_file_fallback(self):
        """Test fallback when config file is missing"""
        # Specify a non-existent path to ensure default settings are used
        config = SignalRulesConfig("/nonexistent/path/signal_rules.json")

        # Verify that default settings are loaded
        assert config.config is not None
        assert "signal_generation_settings" in config.config

        # Verify that default values are correctly retrieved
        min_period = config.get_signal_settings().get("min_data_period", 0)
        assert min_period == 60  # Default value

        multiplier = config.get_confidence_multiplier("rsi_oversold", 1.0)
        assert multiplier == 2.0  # Default value


class TestSignalRules:
    """Test for individual rules"""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        np.random.seed(42)

        # Generate data with a trend
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
        """Calculated technical indicators"""
        return TechnicalIndicators.calculate_all(sample_data)

    @pytest.fixture
    def patterns_data(self, sample_data):
        """Recognized chart patterns"""
        return ChartPatternRecognizer.detect_all_patterns(sample_data)

    def test_rsi_oversold_rule(self, sample_data, indicators_data, patterns_data):
        """Test RSI oversold rule"""
        # Artificially set RSI low
        indicators_data.loc[indicators_data.index[-1], "RSI"] = 25

        rule = RSIOversoldRule(threshold=30)
        met, confidence = rule.evaluate(sample_data, indicators_data, patterns_data)

        assert met is True
        assert confidence > 0
        assert confidence <= 100

    def test_rsi_overbought_rule(self, sample_data, indicators_data, patterns_data):
        """Test RSI overbought rule"""
        # Artificially set RSI high
        indicators_data.loc[indicators_data.index[-1], "RSI"] = 75

        rule = RSIOverboughtRule(threshold=70)
        met, confidence = rule.evaluate(sample_data, indicators_data, patterns_data)

        assert met is True
        assert confidence > 0
        assert confidence <= 100

    def test_macd_crossover_rule(self, sample_data):
        """Test MACD crossover rule"""
        # Artificially create MACD crossover
        indicators = pd.DataFrame(index=sample_data.index)
        indicators["MACD"] = [0] * (len(sample_data) - 2) + [-0.5, 0.5]
        indicators["MACD_Signal"] = [0] * len(sample_data)

        rule = MACDCrossoverRule()
        met, confidence = rule.evaluate(sample_data, indicators, {})

        assert met is True
        assert confidence > 0

    def test_bollinger_band_lower_rule(self, sample_data):
        """Test Bollinger Band lower rule"""
        # Set price below lower band
        indicators = pd.DataFrame(index=sample_data.index)
        indicators["BB_Lower"] = sample_data["Close"] + 2
        indicators["BB_Upper"] = sample_data["Close"] + 5

        rule = BollingerBandRule(position="lower")
        met, confidence = rule.evaluate(sample_data, indicators, {})

        assert met is True
        assert confidence > 0

    def test_pattern_breakout_rule(self, sample_data, indicators_data):
        """Test pattern breakout rule"""
        # Create breakout pattern
        breakouts = pd.DataFrame(index=sample_data.index)
        breakouts["Upward_Breakout"] = [False] * (len(sample_data) - 1) + [True]
        breakouts["Upward_Confidence"] = [0] * (len(sample_data) - 1) + [80]

        patterns = {"breakouts": breakouts}

        rule = PatternBreakoutRule(direction="upward")
        met, confidence = rule.evaluate(sample_data, indicators_data, patterns)

        assert met is True
        assert confidence == 80

    def test_golden_cross_rule(self, sample_data, indicators_data):
        """Test golden cross rule"""
        # Create golden cross
        crosses = pd.DataFrame(index=sample_data.index)
        crosses["Golden_Cross"] = [False] * (len(sample_data) - 3) + [
            True,
            False,
            False,
        ]
        crosses["Golden_Confidence"] = [0] * (len(sample_data) - 3) + [75, 75, 75]

        patterns = {"crosses": crosses}

        rule = GoldenCrossRule()
        met, confidence = rule.evaluate(sample_data, indicators_data, patterns)

        assert met is True
        assert confidence == 75

    def test_volume_spike_buy_rule(self, sample_data, indicators_data, patterns_data):
        """Test volume spike buy rule"""
        # Adjust volume and price
        sample_data.loc[sample_data.index[-1], "Volume"] = 10000000
        sample_data.loc[sample_data.index[-1], "Close"] = (
            sample_data["Close"].iloc[-2] * 1.03
        )

        rule = VolumeSpikeBuyRule(threshold=2.0)
        met, confidence = rule.evaluate(sample_data, indicators_data, patterns_data)

        assert met is True
        assert confidence > 0


class TestTradingSignalGenerator:
    """Test class for trading signal generator"""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        np.random.seed(42)

        # Generate data with a trend
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
        """Instance of signal generator"""
        return TradingSignalGenerator()

    def test_generate_signal_basic(self, generator, sample_data):
        """Test basic signal generation"""
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
        """Test buy signal generation"""
        # Set RSI low to induce buy signal
        indicators = TechnicalIndicators.calculate_all(sample_data)
        indicators.loc[indicators.index[-1], "RSI"] = 25
        patterns = ChartPatternRecognizer.detect_all_patterns(
            sample_data
        )  # Add patterns

        signal = generator.generate_signal(
            sample_data, indicators=indicators, patterns=patterns
        )

        assert signal is not None
        # If RSI is low, buy signal is likely
        if signal.signal_type == SignalType.BUY:
            assert "RSI Oversold" in signal.conditions_met
            assert signal.conditions_met["RSI Oversold"] is True

    def test_generate_signal_sell(self, generator, sample_data):
        """Test sell signal generation"""
        # Set RSI high to induce sell signal
        indicators = TechnicalIndicators.calculate_all(sample_data)
        indicators.loc[indicators.index[-1], "RSI"] = 75
        patterns = ChartPatternRecognizer.detect_all_patterns(
            sample_data
        )  # Add patterns

        signal = generator.generate_signal(
            sample_data, indicators=indicators, patterns=patterns
        )

        assert signal is not None
        # If RSI is high, sell signal is likely
        if signal.signal_type == SignalType.SELL:
            assert "RSI Overbought" in signal.conditions_met
            assert signal.conditions_met["RSI Overbought"] is True

    def test_generate_signal_multiple_conditions(self, generator, sample_data):
        """Test signal generation with multiple conditions"""
        # Set multiple buy conditions
        indicators = TechnicalIndicators.calculate_all(sample_data)
        indicators.loc[indicators.index[-1], "RSI"] = 25

        # Add MACD crossover
        indicators.loc[indicators.index[-2], "MACD"] = -0.5
        indicators.loc[indicators.index[-1], "MACD"] = 0.5
        indicators.loc[indicators.index[-2:], "MACD_Signal"] = 0
        patterns = ChartPatternRecognizer.detect_all_patterns(
            sample_data
        )  # Add patterns

        signal = generator.generate_signal(
            sample_data, indicators=indicators, patterns=patterns
        )

        assert signal is not None
        # Verify multiple conditions are met
        active_conditions = sum(1 for v in signal.conditions_met.values() if v)
        assert active_conditions >= 2

    def test_generate_signals_series(self, generator, sample_data):
        """Test time-series signal generation"""
        signals_df = generator.generate_signals_series(sample_data, lookback_window=30)

        assert isinstance(signals_df, pd.DataFrame)
        if not signals_df.empty:
            assert "Signal" in signals_df.columns
            assert "Strength" in signals_df.columns
            assert "Confidence" in signals_df.columns
            assert "Price" in signals_df.columns
            assert "Reasons" in signals_df.columns

            # Verify signal values
            assert signals_df["Signal"].isin(["buy", "sell", "hold"]).all()
            assert signals_df["Strength"].isin(["strong", "medium", "weak"]).all()
            assert (signals_df["Confidence"] >= 0).all()
            assert (signals_df["Confidence"] <= 100).all()

    def test_custom_rules(self, sample_data):
        """Test adding custom rules"""
        generator = TradingSignalGenerator()

        # Record initial number of rules
        initial_buy_rules = len(generator.buy_rules)

        # Add custom rule
        custom_rule = VolumeSpikeBuyRule()
        generator.add_buy_rule(custom_rule)

        assert len(generator.buy_rules) == initial_buy_rules + 1
        assert generator.buy_rules[-1] == custom_rule

        # Test custom rule behavior - adjust volume and price to meet conditions
        test_data = sample_data.copy()
        test_data.loc[test_data.index[-1], "Volume"] = 10000000  # Large volume
        test_data.loc[test_data.index[-1], "Close"] = (
            test_data["Close"].iloc[-2] * 1.03
        )  # 3% increase

        test_indicators = TechnicalIndicators.calculate_all(test_data)
        test_patterns = ChartPatternRecognizer.detect_all_patterns(test_data)

        # Generate signal
        signal = generator.generate_signal(test_data, test_indicators, test_patterns)
        assert signal is not None

        # Buy signal or hold should be generated (due to volume spike)
        # "Volume Spike Buy" should be included in conditions
        if signal.signal_type == SignalType.BUY:
            assert "Volume Spike Buy" in signal.conditions_met
            assert signal.conditions_met["Volume Spike Buy"] is True
        else:
            # Regardless of signal type, all conditions should be evaluated
            # Whether included in conditions_met depends on signal type in implementation
            pass

    def test_clear_rules(self, generator):
        """Test clearing rules"""
        # Verify rules exist
        assert len(generator.buy_rules) > 0
        assert len(generator.sell_rules) > 0

        # Clear rules
        generator.clear_rules()

        assert len(generator.buy_rules) == 0
        assert len(generator.sell_rules) == 0

    def test_validate_signal(self, generator, sample_data):
        """Test signal validation"""
        indicators = TechnicalIndicators.calculate_all(sample_data)
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)
        signal = generator.generate_signal(sample_data, indicators, patterns)

        if signal:
            # Basic validation
            validity = generator.validate_signal(signal)
            assert isinstance(validity, float)
            assert 0 <= validity <= 100

            # Validation with market context
            market_context = {
                "volatility": sample_data["Close"].pct_change().std(),
                "trend_direction": "upward",
            }
            validity_with_context = generator.validate_signal(
                signal, market_context=market_context
            )
            assert isinstance(validity_with_context, float)
            assert 0 <= validity_with_context <= 100

            # Validation with historical performance data
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
        """Error handling with empty data"""
        empty_df = pd.DataFrame()
        empty_indicators = pd.DataFrame()
        empty_patterns = {}
        signal = generator.generate_signal(empty_df, empty_indicators, empty_patterns)

        assert signal is None

    def test_insufficient_data(self, generator):
        """Error handling with insufficient data"""
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
        """Test signal strength classification"""
        # Configure to generate strong signals
        indicators = TechnicalIndicators.calculate_all(sample_data)

        # Set multiple strong conditions
        indicators.loc[indicators.index[-1], "RSI"] = 20  # Strong oversold
        indicators.loc[indicators.index[-2], "MACD"] = -1
        indicators.loc[indicators.index[-1], "MACD"] = 1  # Strong crossover
        indicators.loc[:, "MACD_Signal"] = 0

        # Break through Bollinger Band lower
        indicators.loc[indicators.index[-1], "BB_Lower"] = (
            sample_data["Close"].iloc[-1] + 1
        )
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)

        signal = generator.generate_signal(
            sample_data, indicators=indicators, patterns=patterns
        )

        assert signal is not None
        # Should be a strong signal as multiple strong conditions are met
        active_conditions = sum(1 for v in signal.conditions_met.values() if v)
        if active_conditions >= 3 and signal.confidence >= 70:
            assert signal.strength == SignalStrength.STRONG


class TestCustomSignalRule:
    """Test custom signal rule"""

    def test_custom_rule_implementation(self):
        """Test custom rule implementation"""

        class TestRule(SignalRule):
            def __init__(self):
                super().__init__("Test Rule", weight=1.0)

            def evaluate(self, df, indicators, patterns):
                # Always return True, 50% confidence
                return True, 50.0

        rule = TestRule()
        assert rule.name == "Test Rule"
        assert rule.weight == 1.0

        # Test evaluation method
        met, confidence = rule.evaluate(None, None, None)
        assert met is True
        assert confidence == 50.0


class TestImprovedSignalGeneration:
    """Test for improved signal generation functionality"""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
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
        """Test config-based rule loading"""
        generator = TradingSignalGenerator()

        # Default rules are loaded even if config file is missing
        assert len(generator.buy_rules) > 0
        assert len(generator.sell_rules) > 0

        # Verify default rules are included
        rule_names = [rule.name for rule in generator.buy_rules]
        assert "RSI Oversold" in rule_names

    def test_decimal_price_handling(self, sample_data):
        """Test Decimal price handling"""
        generator = TradingSignalGenerator()
        indicators = TechnicalIndicators.calculate_all(sample_data)
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)

        signal = generator.generate_signal(sample_data, indicators, patterns)

        if signal:
            assert isinstance(signal.price, Decimal)
            assert signal.price > 0
            # Test conversion from Decimal to other types
            assert float(signal.price) > 0
            assert int(signal.price) > 0

    def test_datetime_timestamp_handling(self, sample_data):
        """Test datetime timestamp handling"""
        generator = TradingSignalGenerator()
        indicators = TechnicalIndicators.calculate_all(sample_data)
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)

        signal = generator.generate_signal(sample_data, indicators, patterns)

        if signal:
            assert isinstance(signal.timestamp, datetime)
            # Verify compatibility with pandas Timestamp
            pd_timestamp = pd.Timestamp(signal.timestamp)
            assert pd_timestamp.year >= 2023

    def test_configurable_strength_thresholds(self, sample_data):
        """Test configurable strength thresholds"""
        generator = TradingSignalGenerator()

        # Verify strength thresholds are loaded from config
        thresholds = generator.config.get_strength_thresholds()
        assert "strong" in thresholds
        assert "medium" in thresholds

        # Verify thresholds are used in actual signal generation
        indicators = TechnicalIndicators.calculate_all(sample_data)
        # Meet multiple conditions to create a strong signal
        indicators.loc[indicators.index[-1], "RSI"] = 20  # Strong oversold
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)

        signal = generator.generate_signal(sample_data, indicators, patterns)
        if signal and signal.signal_type == SignalType.BUY:
            # Verify strength is determined based on configured thresholds
            assert signal.strength in [
                SignalStrength.STRONG,
                SignalStrength.MEDIUM,
                SignalStrength.WEAK,
            ]

    def test_enhanced_validate_signal(self, sample_data):
        """Test enhanced signal validation"""
        generator = TradingSignalGenerator()
        indicators = TechnicalIndicators.calculate_all(sample_data)
        patterns = ChartPatternRecognizer.detect_all_patterns(sample_data)

        signal = generator.generate_signal(sample_data, indicators, patterns)

        if signal:
            # Validation with market context
            market_context = {
                "volatility": 0.02,  # Low volatility
                "trend_direction": "upward",
            }

            validity_low_vol = generator.validate_signal(
                signal, market_context=market_context
            )

            # Validation with high volatility
            market_context["volatility"] = 0.08  # High volatility
            validity_high_vol = generator.validate_signal(
                signal, market_context=market_context
            )

            # Verify confidence decreases with high volatility
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                assert validity_low_vol >= validity_high_vol

    def test_rule_parameter_configuration(self):
        """Test rule parameter configuration"""
        generator = TradingSignalGenerator()

        # Verify RSI rule parameters are loaded from config
        rsi_rules = [rule for rule in generator.buy_rules if "RSI" in rule.name]
        if rsi_rules:
            rsi_rule = rsi_rules[0]
            assert hasattr(rsi_rule, "threshold")
            assert hasattr(rsi_rule, "confidence_multiplier")
            assert rsi_rule.threshold > 0
            assert rsi_rule.confidence_multiplier > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
