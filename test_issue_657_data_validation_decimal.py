"""
Issue #657: データ検証とDecimal型処理の改善のテスト

TradingSignalGenerator.generate_signalメソッドにおいて:
- indicators や patterns の完全性（NaN値など）チェックの強化
- Decimal型の扱いの一貫性改善
- 堅牢なデータ検証機能の追加
"""

import unittest
from unittest.mock import Mock, patch
from decimal import Decimal, InvalidOperation
from datetime import datetime
import pandas as pd
import numpy as np

from src.day_trade.analysis.signals import (
    TradingSignalGenerator,
    SignalRulesConfig,
    TradingSignal,
    SignalType,
    SignalStrength
)


class TestDataValidationImprovements(unittest.TestCase):
    """データ検証改善のテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.generator = TradingSignalGenerator()
        self.valid_df = self._create_valid_price_data()
        self.valid_indicators = self._create_valid_indicators()
        self.valid_patterns = self._create_valid_patterns()

    def _create_valid_price_data(self) -> pd.DataFrame:
        """有効な価格データを作成"""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        np.random.seed(42)

        close_prices = 100 + np.cumsum(np.random.randn(30) * 0.5)

        df = pd.DataFrame({
            'Open': close_prices + np.random.randn(30) * 0.2,
            'High': close_prices + np.abs(np.random.randn(30)) * 0.5,
            'Low': close_prices - np.abs(np.random.randn(30)) * 0.5,
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 5000000, 30)
        }, index=dates)

        # 高値・安値の調整
        df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
        df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))

        return df

    def _create_valid_indicators(self) -> pd.DataFrame:
        """有効な指標データを作成"""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')

        return pd.DataFrame({
            'RSI': np.random.uniform(20, 80, 30),
            'MACD': np.random.randn(30) * 0.5,
            'MACD_Signal': np.random.randn(30) * 0.4,
            'BB_Upper': np.random.uniform(105, 110, 30),
            'BB_Lower': np.random.uniform(95, 100, 30),
        }, index=dates)

    def _create_valid_patterns(self) -> dict:
        """有効なパターンデータを作成"""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')

        return {
            'crosses': pd.DataFrame({
                'Golden_Cross': np.random.choice([True, False], 30),
                'Golden_Confidence': np.random.uniform(0, 100, 30),
                'Dead_Cross': np.random.choice([True, False], 30),
                'Dead_Confidence': np.random.uniform(0, 100, 30),
            }, index=dates),
            'breakouts': pd.DataFrame({
                'Upward_Breakout': np.random.choice([True, False], 30),
                'Upward_Confidence': np.random.uniform(0, 100, 30),
                'Downward_Breakout': np.random.choice([True, False], 30),
                'Downward_Confidence': np.random.uniform(0, 100, 30),
            }, index=dates),
            'levels': {},
            'trends': {}
        }

    def test_validate_input_data_success(self):
        """正常データでの検証成功テスト"""
        result = self.generator._validate_input_data(
            self.valid_df, self.valid_indicators, self.valid_patterns
        )
        self.assertTrue(result)

    def test_validate_input_data_empty_df(self):
        """空の価格データでの検証失敗テスト"""
        empty_df = pd.DataFrame()
        result = self.generator._validate_input_data(
            empty_df, self.valid_indicators, self.valid_patterns
        )
        self.assertFalse(result)

    def test_validate_input_data_missing_columns(self):
        """必須カラム不足での検証失敗テスト"""
        invalid_df = self.valid_df.drop('Close', axis=1)
        result = self.generator._validate_input_data(
            invalid_df, self.valid_indicators, self.valid_patterns
        )
        self.assertFalse(result)

    def test_validate_input_data_nan_values(self):
        """NaN値含む価格データでの検証失敗テスト"""
        invalid_df = self.valid_df.copy()
        invalid_df.iloc[-1, invalid_df.columns.get_loc('Close')] = np.nan
        result = self.generator._validate_input_data(
            invalid_df, self.valid_indicators, self.valid_patterns
        )
        self.assertFalse(result)

    def test_validate_price_data_negative_values(self):
        """負の価格値での検証失敗テスト"""
        invalid_row = self.valid_df.iloc[-1].copy()
        invalid_row['Close'] = -10.0
        result = self.generator._validate_price_data(invalid_row)
        self.assertFalse(result)

    def test_validate_price_data_logical_inconsistency(self):
        """価格の論理的整合性違反での検証失敗テスト"""
        invalid_row = self.valid_df.iloc[-1].copy()
        invalid_row['High'] = 100.0
        invalid_row['Low'] = 110.0  # High < Low
        result = self.generator._validate_price_data(invalid_row)
        self.assertFalse(result)

    def test_validate_price_data_open_out_of_range(self):
        """始値が高安値範囲外での検証失敗テスト"""
        invalid_row = self.valid_df.iloc[-1].copy()
        invalid_row['High'] = 100.0
        invalid_row['Low'] = 95.0
        invalid_row['Open'] = 105.0  # 範囲外
        result = self.generator._validate_price_data(invalid_row)
        self.assertFalse(result)

    def test_validate_indicators_data_empty(self):
        """空の指標データでの検証成功テスト（空でも継続可能）"""
        empty_indicators = pd.DataFrame()
        result = self.generator._validate_indicators_data(empty_indicators)
        self.assertTrue(result)

    def test_validate_indicators_data_rsi_out_of_range(self):
        """RSI範囲外値での警告テスト"""
        invalid_indicators = self.valid_indicators.copy()
        invalid_indicators.iloc[-1, invalid_indicators.columns.get_loc('RSI')] = 150.0

        with patch('src.day_trade.analysis.signals.logger') as mock_logger:
            result = self.generator._validate_indicators_data(invalid_indicators)
            self.assertTrue(result)  # 警告は出すが処理は継続
            mock_logger.warning.assert_called()

    def test_validate_patterns_data_wrong_type(self):
        """パターンデータの型違いでの警告テスト"""
        invalid_patterns = {
            'crosses': "not_a_dataframe",  # DataFrameではない
            'breakouts': self.valid_patterns['breakouts']
        }

        with patch('src.day_trade.analysis.signals.logger') as mock_logger:
            result = self.generator._validate_patterns_data(invalid_patterns)
            self.assertTrue(result)  # 警告は出すが処理は継続
            mock_logger.warning.assert_called()


class TestDecimalHandlingImprovements(unittest.TestCase):
    """Decimal型処理改善のテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.generator = TradingSignalGenerator()

    def test_safe_decimal_conversion_success(self):
        """正常なDecimal変換テスト"""
        test_values = [100.5, 50, "75.25"]

        for value in test_values:
            result = self.generator._safe_decimal_conversion(value)
            self.assertIsInstance(result, Decimal)
            self.assertGreater(result, 0)

    def test_safe_decimal_conversion_nan_value(self):
        """NaN値のDecimal変換失敗テスト"""
        result = self.generator._safe_decimal_conversion(np.nan)
        self.assertIsNone(result)

    def test_safe_decimal_conversion_negative_value(self):
        """負の値のDecimal変換失敗テスト"""
        result = self.generator._safe_decimal_conversion(-10.5)
        self.assertIsNone(result)

    def test_safe_decimal_conversion_zero_value(self):
        """ゼロ値のDecimal変換失敗テスト"""
        result = self.generator._safe_decimal_conversion(0)
        self.assertIsNone(result)

    def test_safe_decimal_conversion_infinite_value(self):
        """無限大値のDecimal変換失敗テスト"""
        result = self.generator._safe_decimal_conversion(float('inf'))
        self.assertIsNone(result)

    def test_safe_decimal_conversion_invalid_string(self):
        """無効な文字列のDecimal変換失敗テスト"""
        result = self.generator._safe_decimal_conversion("invalid_number")
        self.assertIsNone(result)

    def test_safe_decimal_to_float_success(self):
        """正常なDecimal->float変換テスト"""
        decimal_value = Decimal("123.45")
        result = self.generator._safe_decimal_to_float(decimal_value)
        self.assertEqual(result, 123.45)
        self.assertIsInstance(result, float)

    def test_safe_decimal_to_float_none_input(self):
        """None入力でのDecimal->float変換テスト"""
        result = self.generator._safe_decimal_to_float(None)
        self.assertIsNone(result)

    def test_safe_decimal_to_float_extreme_value(self):
        """極端な値でのDecimal->float変換テスト"""
        # 非常に大きな値
        extreme_decimal = Decimal("1" + "0" * 400)  # 非常に大きな数
        result = self.generator._safe_decimal_to_float(extreme_decimal)
        # オーバーフローでNoneになることを期待
        self.assertIsNone(result)


class TestIntegratedSignalGeneration(unittest.TestCase):
    """統合的なシグナル生成テスト"""

    def setUp(self):
        """テスト前の準備"""
        self.generator = TradingSignalGenerator()
        self.valid_df = self._create_valid_price_data()
        self.valid_indicators = self._create_valid_indicators()
        self.valid_patterns = self._create_valid_patterns()

    def _create_valid_price_data(self) -> pd.DataFrame:
        """有効な価格データを作成"""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        np.random.seed(42)

        close_prices = 100 + np.cumsum(np.random.randn(30) * 0.5)

        df = pd.DataFrame({
            'Open': close_prices + np.random.randn(30) * 0.2,
            'High': close_prices + np.abs(np.random.randn(30)) * 0.5,
            'Low': close_prices - np.abs(np.random.randn(30)) * 0.5,
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 5000000, 30)
        }, index=dates)

        # 高値・安値の調整
        df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
        df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))

        return df

    def _create_valid_indicators(self) -> pd.DataFrame:
        """有効な指標データを作成"""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')

        return pd.DataFrame({
            'RSI': np.random.uniform(20, 80, 30),
            'MACD': np.random.randn(30) * 0.5,
            'MACD_Signal': np.random.randn(30) * 0.4,
            'BB_Upper': np.random.uniform(105, 110, 30),
            'BB_Lower': np.random.uniform(95, 100, 30),
        }, index=dates)

    def _create_valid_patterns(self) -> dict:
        """有効なパターンデータを作成"""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')

        return {
            'crosses': pd.DataFrame({
                'Golden_Cross': np.random.choice([True, False], 30),
                'Golden_Confidence': np.random.uniform(0, 100, 30),
                'Dead_Cross': np.random.choice([True, False], 30),
                'Dead_Confidence': np.random.uniform(0, 100, 30),
            }, index=dates),
            'breakouts': pd.DataFrame({
                'Upward_Breakout': np.random.choice([True, False], 30),
                'Upward_Confidence': np.random.uniform(0, 100, 30),
                'Downward_Breakout': np.random.choice([True, False], 30),
                'Downward_Confidence': np.random.uniform(0, 100, 30),
            }, index=dates),
            'levels': {},
            'trends': {}
        }

    def test_generate_signal_with_valid_data(self):
        """正常データでのシグナル生成テスト"""
        signal = self.generator.generate_signal(
            self.valid_df, self.valid_indicators, self.valid_patterns
        )

        # シグナルが生成されることを確認
        if signal is not None:
            self.assertIsInstance(signal, TradingSignal)
            self.assertIsInstance(signal.price, Decimal)
            self.assertGreater(signal.price, 0)
            self.assertIsInstance(signal.confidence, (int, float))
            self.assertGreaterEqual(signal.confidence, 0)
            self.assertLessEqual(signal.confidence, 100)

    def test_generate_signal_with_invalid_data(self):
        """無効データでのシグナル生成失敗テスト"""
        # NaN値を含む価格データ
        invalid_df = self.valid_df.copy()
        invalid_df.iloc[-1, invalid_df.columns.get_loc('Close')] = np.nan

        signal = self.generator.generate_signal(
            invalid_df, self.valid_indicators, self.valid_patterns
        )

        # 無効データではシグナルが生成されないことを確認
        self.assertIsNone(signal)

    def test_generate_signal_with_insufficient_data(self):
        """データ不足でのシグナル生成失敗テスト"""
        # 最小データ数未満のデータ
        insufficient_df = self.valid_df.head(5)
        insufficient_indicators = self.valid_indicators.head(5)

        signal = self.generator.generate_signal(
            insufficient_df, insufficient_indicators, self.valid_patterns
        )

        # データ不足ではシグナルが生成されないことを確認
        self.assertIsNone(signal)

    def test_generate_signals_series_decimal_conversion(self):
        """時系列シグナル生成でのDecimal変換テスト"""
        # 少ないルックバックウィンドウでテスト
        signals_df = self.generator.generate_signals_series(self.valid_df, lookback_window=25)

        if not signals_df.empty:
            # Price列がfloatに正しく変換されていることを確認
            self.assertTrue(all(isinstance(price, (int, float)) for price in signals_df['Price']))
            self.assertTrue(all(price > 0 for price in signals_df['Price']))

    def test_generate_signal_none_patterns(self):
        """Noneパターンでのシグナル生成テスト"""
        signal = self.generator.generate_signal(
            self.valid_df, self.valid_indicators, None
        )

        # Noneパターンでも処理が継続されることを確認
        # シグナルが生成されるかは条件次第だが、エラーにならないことが重要
        self.assertIsInstance(signal, (TradingSignal, type(None)))


class TestErrorHandling(unittest.TestCase):
    """エラーハンドリングのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.generator = TradingSignalGenerator()

    def test_safe_decimal_conversion_with_mock_exception(self):
        """Decimal変換での例外処理テスト"""
        # float()が例外を投げる状況をシミュレート
        with patch('builtins.float', side_effect=ValueError("変換エラー")):
            result = self.generator._safe_decimal_conversion("123.45")
            self.assertIsNone(result)

    def test_safe_decimal_to_float_with_mock_exception(self):
        """Decimal->float変換での例外処理テスト"""
        # float()が例外を投げる状況をシミュレート
        decimal_value = Decimal("123.45")
        with patch('builtins.float', side_effect=OverflowError("オーバーフロー")):
            result = self.generator._safe_decimal_to_float(decimal_value)
            self.assertIsNone(result)

    def test_validate_input_data_with_mock_exception(self):
        """入力データ検証での例外処理テスト"""
        # DataFrameアクセスで例外が発生する状況をシミュレート
        mock_df = Mock()
        mock_df.empty = False
        mock_df.__len__ = Mock(side_effect=Exception("アクセスエラー"))

        result = self.generator._validate_input_data(mock_df, pd.DataFrame(), {})
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)