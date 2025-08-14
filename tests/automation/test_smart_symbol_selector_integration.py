#!/usr/bin/env python3
"""
Issue #755 Phase 3: SmartSymbolSelector統合テスト

SmartSymbolSelectorとメインシステムの統合検証
- DataFetcher統合テスト
- EnsembleSystem統合テスト
- 自動化ワークフロー統合テスト
- エンドツーエンド統合テスト
"""

import unittest
import pytest
import asyncio
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# テスト対象システムのインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.day_trade.automation.smart_symbol_selector import (
        SmartSymbolSelector,
        SymbolMetrics,
        SelectionCriteria,
        MarketSegment,
        get_smart_selected_symbols
    )
    from src.day_trade.data_fetcher import DataFetcher
    from src.day_trade.ml.ensemble_system import EnsembleSystem, EnsembleConfig
    from src.day_trade.analysis.trend_analyzer import TrendAnalyzer

except ImportError as e:
    print(f"統合テスト用インポートエラー: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")


class TestSmartSymbolSelectorDataIntegration(unittest.TestCase):
    """SmartSymbolSelector - DataFetcher統合テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.selector = SmartSymbolSelector()
        self.data_fetcher = DataFetcher()

        # モックデータ準備
        self.mock_market_data = self._create_mock_market_data()

    def _create_mock_market_data(self) -> pd.DataFrame:
        """モック市場データ作成"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        symbols = ['7203.T', '6758.T', '9984.T', '4519.T']

        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'Symbol': symbol,
                    'Date': date,
                    'Open': np.random.uniform(1000, 1100),
                    'High': np.random.uniform(1100, 1200),
                    'Low': np.random.uniform(900, 1000),
                    'Close': np.random.uniform(1000, 1100),
                    'Volume': np.random.randint(1000000, 5000000),
                    'Adj Close': np.random.uniform(1000, 1100)
                })

        return pd.DataFrame(data)

    @patch('src.day_trade.data_fetcher.DataFetcher.fetch_data')
    @patch('yfinance.Ticker')
    def test_data_fetcher_symbol_selection_integration(self, mock_ticker, mock_fetch_data):
        """DataFetcherと銘柄選択の統合テスト"""
        # SmartSymbolSelectorのモック設定
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {'marketCap': 5000000000000}
        mock_ticker_instance.history.return_value = self._create_valid_price_data()
        mock_ticker.return_value = mock_ticker_instance

        # DataFetcherのモック設定
        mock_fetch_data.return_value = self.mock_market_data

        async def run_test():
            # Step 1: スマート銘柄選択
            selected_symbols = await self.selector.select_optimal_symbols(
                SelectionCriteria(target_symbols=3)
            )

            # Step 2: 選択された銘柄でデータ取得
            if selected_symbols:
                market_data = self.data_fetcher.fetch_data(selected_symbols)

                # 統合結果検証
                self.assertIsInstance(selected_symbols, list)
                self.assertGreater(len(selected_symbols), 0)
                self.assertIsInstance(market_data, pd.DataFrame)

                # 選択された銘柄のデータが含まれることを確認
                if not market_data.empty:
                    fetched_symbols = market_data['Symbol'].unique()
                    for symbol in selected_symbols:
                        if symbol in self.mock_market_data['Symbol'].unique():
                            self.assertIn(symbol, fetched_symbols)

        asyncio.run(run_test())

    @patch('yfinance.Ticker')
    def test_realtime_data_validation(self, mock_ticker):
        """リアルタイムデータ検証統合テスト"""
        # リアルタイムデータをシミュレート
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {'marketCap': 3000000000000}
        mock_ticker_instance.history.return_value = self._create_realtime_data()
        mock_ticker.return_value = mock_ticker_instance

        async def run_test():
            # リアルタイム銘柄選択
            selected = await self.selector.select_optimal_symbols(
                SelectionCriteria(target_symbols=2)
            )

            # データ品質検証
            self.assertIsInstance(selected, list)

            # 各選択銘柄について流動性・品質確認
            for symbol in selected:
                symbol_info = self.selector.get_symbol_info(symbol)
                self.assertIsNotNone(symbol_info)

        asyncio.run(run_test())

    def _create_valid_price_data(self) -> pd.DataFrame:
        """有効な価格データ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        data = {
            'Open': np.random.uniform(1000, 1100, 60),
            'High': np.random.uniform(1100, 1200, 60),
            'Low': np.random.uniform(900, 1000, 60),
            'Close': np.random.uniform(1000, 1100, 60),
            'Volume': np.random.randint(2000000, 4000000, 60)
        }

        return pd.DataFrame(data, index=dates)

    def _create_realtime_data(self) -> pd.DataFrame:
        """リアルタイムデータ作成"""
        dates = pd.date_range(datetime.now() - timedelta(days=60), periods=60, freq='D')

        data = {
            'Open': np.random.uniform(1050, 1150, 60),
            'High': np.random.uniform(1150, 1250, 60),
            'Low': np.random.uniform(950, 1050, 60),
            'Close': np.random.uniform(1050, 1150, 60),
            'Volume': np.random.randint(1500000, 3500000, 60)
        }

        return pd.DataFrame(data, index=dates)


class TestSmartSymbolSelectorMLIntegration(unittest.TestCase):
    """SmartSymbolSelector - EnsembleSystem統合テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.selector = SmartSymbolSelector()
        self.ensemble_config = EnsembleConfig(verbose=False)
        self.ensemble = EnsembleSystem(self.ensemble_config)

    @patch('yfinance.Ticker')
    def test_ensemble_prediction_integration(self, mock_ticker):
        """EnsembleSystem予測統合テスト"""
        # yfinanceモック設定
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {'marketCap': 4000000000000}
        mock_ticker_instance.history.return_value = self._create_ml_training_data()
        mock_ticker.return_value = mock_ticker_instance

        async def run_test():
            # Step 1: スマート銘柄選択
            selected_symbols = await self.selector.select_optimal_symbols(
                SelectionCriteria(target_symbols=3)
            )

            if selected_symbols:
                # Step 2: 選択銘柄でML訓練データ準備
                training_features, training_targets = self._prepare_ml_features(selected_symbols)

                if len(training_features) > 50:
                    # Step 3: EnsembleSystem訓練
                    X_train = training_features[:40]
                    y_train = training_targets[:40]
                    X_test = training_features[40:50]

                    self.ensemble.fit(X_train, y_train)

                    # Step 4: 予測実行
                    predictions = self.ensemble.predict(X_test)

                    # 統合結果検証
                    self.assertIsNotNone(predictions)
                    self.assertGreater(len(predictions.final_predictions), 0)
                    self.assertEqual(len(predictions.final_predictions), len(X_test))

        asyncio.run(run_test())

    def _prepare_ml_features(self, symbols: List[str]) -> tuple:
        """ML特徴量準備"""
        # シミュレーション用の特徴量・ターゲット生成
        n_samples = 60
        n_features = 10

        features = np.random.randn(n_samples, n_features)
        targets = np.random.randn(n_samples)

        return features, targets

    @patch('yfinance.Ticker')
    def test_feature_engineering_integration(self, mock_ticker):
        """特徴量エンジニアリング統合テスト"""
        # モック設定
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {'marketCap': 6000000000000}
        mock_ticker_instance.history.return_value = self._create_feature_data()
        mock_ticker.return_value = mock_ticker_instance

        async def run_test():
            # SmartSymbolSelectorで高品質銘柄選択
            selected_symbols = await self.selector.select_optimal_symbols(
                SelectionCriteria(
                    target_symbols=2,
                    min_liquidity_score=70.0,
                    max_volatility=6.0
                )
            )

            # 選択された銘柄の特徴量抽出
            features = self._extract_technical_features(selected_symbols)

            # 特徴量品質検証
            self.assertIsInstance(features, np.ndarray)
            self.assertGreater(features.shape[0], 0)
            self.assertGreater(features.shape[1], 0)

        asyncio.run(run_test())

    def _extract_technical_features(self, symbols: List[str]) -> np.ndarray:
        """テクニカル特徴量抽出"""
        # シミュレーション: 実際のテクニカル指標計算
        features_list = []

        for symbol in symbols:
            # RSI, MACD, ボリンジャーバンドなどをシミュレート
            symbol_features = np.random.randn(20, 15)  # 20日分、15特徴量
            features_list.append(symbol_features)

        if features_list:
            return np.vstack(features_list)
        else:
            return np.array([]).reshape(0, 15)

    def _create_ml_training_data(self) -> pd.DataFrame:
        """ML訓練用データ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        data = {
            'Open': np.random.uniform(1000, 1100, 60),
            'High': np.random.uniform(1100, 1200, 60),
            'Low': np.random.uniform(900, 1000, 60),
            'Close': np.random.uniform(1000, 1100, 60),
            'Volume': np.random.randint(2500000, 4500000, 60)
        }

        return pd.DataFrame(data, index=dates)

    def _create_feature_data(self) -> pd.DataFrame:
        """特徴量抽出用データ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        # より現実的な価格動向を持つデータ
        base_price = 1000
        prices = []

        for i in range(60):
            # トレンド + ランダム変動
            trend = 0.001 * i  # 上昇トレンド
            noise = np.random.normal(0, 0.02)
            price_change = trend + noise
            base_price *= (1 + price_change)
            prices.append(base_price)

        data = {
            'Open': [p * 0.999 for p in prices],
            'High': [p * 1.015 for p in prices],
            'Low': [p * 0.985 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(2000000, 4000000, 60)
        }

        return pd.DataFrame(data, index=dates)


class TestSmartSymbolSelectorAutomationIntegration(unittest.TestCase):
    """自動化ワークフロー統合テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.selector = SmartSymbolSelector()

    @patch('yfinance.Ticker')
    def test_automated_workflow_integration(self, mock_ticker):
        """自動化ワークフロー統合テスト"""
        # モック設定
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {'marketCap': 5000000000000}
        mock_ticker_instance.history.return_value = self._create_workflow_data()
        mock_ticker.return_value = mock_ticker_instance

        async def run_test():
            # Step 1: 自動銘柄選択
            start_time = time.time()

            selected_symbols = await self.selector.select_optimal_symbols(
                SelectionCriteria(target_symbols=4)
            )

            selection_time = time.time() - start_time

            # Step 2: 選択結果の妥当性検証
            self.assertIsInstance(selected_symbols, list)
            self.assertLessEqual(len(selected_symbols), 4)

            # Step 3: パフォーマンス検証
            self.assertLess(selection_time, 30.0,
                           f"自動選択時間 {selection_time:.2f}秒 が目標 30秒 を上回りました")

            print(f"自動化ワークフロー完了時間: {selection_time:.2f}秒")

        asyncio.run(run_test())

    @patch('yfinance.Ticker')
    @patch('src.day_trade.analysis.trend_analyzer.TrendAnalyzer.analyze')
    def test_trend_analysis_integration(self, mock_trend_analyze, mock_ticker):
        """トレンド分析統合テスト"""
        # TrendAnalyzerのモック設定
        mock_trend_analyze.return_value = {
            'trend': 'upward',
            'strength': 0.75,
            'support_levels': [1000, 1050],
            'resistance_levels': [1150, 1200]
        }

        # yfinanceモック設定
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {'marketCap': 7000000000000}
        mock_ticker_instance.history.return_value = self._create_trend_data()
        mock_ticker.return_value = mock_ticker_instance

        async def run_test():
            # スマート銘柄選択
            selected_symbols = await self.selector.select_optimal_symbols()

            if selected_symbols:
                # トレンド分析との統合
                analyzer = TrendAnalyzer()

                for symbol in selected_symbols[:2]:  # 最初の2銘柄をテスト
                    trend_result = analyzer.analyze(self._create_sample_data())

                    # トレンド情報と銘柄選択の整合性確認
                    self.assertIsInstance(trend_result, dict)
                    self.assertIn('trend', trend_result)
                    self.assertIn('strength', trend_result)

        asyncio.run(run_test())

    def _create_workflow_data(self) -> pd.DataFrame:
        """ワークフロー用データ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        data = {
            'Open': np.random.uniform(1020, 1120, 60),
            'High': np.random.uniform(1120, 1220, 60),
            'Low': np.random.uniform(920, 1020, 60),
            'Close': np.random.uniform(1020, 1120, 60),
            'Volume': np.random.randint(1800000, 3800000, 60)
        }

        return pd.DataFrame(data, index=dates)

    def _create_trend_data(self) -> pd.DataFrame:
        """トレンド分析用データ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        # 上昇トレンドを持つデータ
        base_price = 1000
        trend_prices = []

        for i in range(60):
            # 明確な上昇トレンド
            trend_component = 0.002 * i
            noise = np.random.normal(0, 0.01)
            price = base_price * (1 + trend_component + noise)
            trend_prices.append(price)

        data = {
            'Open': [p * 0.998 for p in trend_prices],
            'High': [p * 1.012 for p in trend_prices],
            'Low': [p * 0.988 for p in trend_prices],
            'Close': trend_prices,
            'Volume': np.random.randint(2200000, 4200000, 60)
        }

        return pd.DataFrame(data, index=dates)

    def _create_sample_data(self) -> pd.DataFrame:
        """サンプルデータ作成"""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')

        data = {
            'Symbol': ['TEST.T'] * 30,
            'Date': dates,
            'Open': np.random.uniform(1000, 1100, 30),
            'High': np.random.uniform(1100, 1200, 30),
            'Low': np.random.uniform(900, 1000, 30),
            'Close': np.random.uniform(1000, 1100, 30),
            'Volume': np.random.randint(2000000, 4000000, 30)
        }

        return pd.DataFrame(data)


class TestSmartSymbolSelectorEndToEnd(unittest.TestCase):
    """エンドツーエンド統合テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.selector = SmartSymbolSelector()

    @patch('yfinance.Ticker')
    @patch('src.day_trade.data_fetcher.DataFetcher.fetch_data')
    def test_complete_pipeline_integration(self, mock_fetch_data, mock_ticker):
        """完全パイプライン統合テスト"""
        # データフェッチャーのモック設定
        mock_fetch_data.return_value = self._create_pipeline_market_data()

        # yfinanceモック設定
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {'marketCap': 8000000000000}
        mock_ticker_instance.history.return_value = self._create_pipeline_data()
        mock_ticker.return_value = mock_ticker_instance

        async def run_test():
            # Step 1: スマート銘柄選択
            selected_symbols = await self.selector.select_optimal_symbols(
                SelectionCriteria(target_symbols=3)
            )

            if selected_symbols:
                # Step 2: 選択銘柄でデータ取得
                data_fetcher = DataFetcher()
                market_data = data_fetcher.fetch_data(selected_symbols)

                # Step 3: 特徴量エンジニアリング
                features = self._comprehensive_feature_engineering(market_data)

                # Step 4: ML予測システム統合
                if len(features) > 20:
                    ensemble = EnsembleSystem(EnsembleConfig(verbose=False))

                    X = features[:, :-1]  # 特徴量
                    y = features[:, -1]   # ターゲット

                    if len(X) > 15:
                        # 訓練・予測
                        split_point = int(len(X) * 0.8)
                        X_train, X_test = X[:split_point], X[split_point:]
                        y_train, y_test = y[:split_point], y[split_point:]

                        ensemble.fit(X_train, y_train)
                        predictions = ensemble.predict(X_test)

                        # エンドツーエンド検証
                        self.assertIsNotNone(predictions)
                        self.assertEqual(len(predictions.final_predictions), len(y_test))

                        # 予測品質確認
                        from sklearn.metrics import r2_score
                        r2 = r2_score(y_test, predictions.final_predictions)
                        self.assertGreater(r2, 0.3, f"エンドツーエンドR² {r2:.3f} が低すぎます")

                        print(f"エンドツーエンド統合テスト完了 - R²: {r2:.3f}")

        asyncio.run(run_test())

    def _create_pipeline_market_data(self) -> pd.DataFrame:
        """パイプライン用市場データ作成"""
        dates = pd.date_range('2023-01-01', periods=80, freq='D')
        symbols = ['7203.T', '6758.T', '9984.T']

        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'Symbol': symbol,
                    'Date': date,
                    'Open': np.random.uniform(1000, 1100),
                    'High': np.random.uniform(1100, 1200),
                    'Low': np.random.uniform(900, 1000),
                    'Close': np.random.uniform(1000, 1100),
                    'Volume': np.random.randint(2000000, 4000000),
                    'Adj Close': np.random.uniform(1000, 1100)
                })

        return pd.DataFrame(data)

    def _create_pipeline_data(self) -> pd.DataFrame:
        """パイプライン用価格データ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        data = {
            'Open': np.random.uniform(1050, 1150, 60),
            'High': np.random.uniform(1150, 1250, 60),
            'Low': np.random.uniform(950, 1050, 60),
            'Close': np.random.uniform(1050, 1150, 60),
            'Volume': np.random.randint(2500000, 4500000, 60)
        }

        return pd.DataFrame(data, index=dates)

    def _comprehensive_feature_engineering(self, market_data: pd.DataFrame) -> np.ndarray:
        """包括的特徴量エンジニアリング"""
        features_list = []

        for symbol in market_data['Symbol'].unique():
            symbol_data = market_data[market_data['Symbol'] == symbol].copy()

            if len(symbol_data) > 20:
                # 基本指標
                symbol_data['Returns'] = symbol_data['Close'].pct_change()
                symbol_data['Volume_MA'] = symbol_data['Volume'].rolling(10).mean()
                symbol_data['Price_MA'] = symbol_data['Close'].rolling(10).mean()
                symbol_data['Volatility'] = symbol_data['Returns'].rolling(10).std()

                # ターゲット（翌日リターン）
                symbol_data['Target'] = symbol_data['Returns'].shift(-1)

                # NaN除去
                clean_data = symbol_data[['Returns', 'Volume_MA', 'Price_MA', 'Volatility', 'Target']].dropna()

                if len(clean_data) > 10:
                    # 正規化
                    normalized_data = (clean_data - clean_data.mean()) / clean_data.std()
                    features_list.append(normalized_data.values)

        if features_list:
            return np.vstack(features_list)
        else:
            return np.array([]).reshape(0, 5)

    @patch('yfinance.Ticker')
    def test_performance_end_to_end(self, mock_ticker):
        """エンドツーエンドパフォーマンステスト"""
        # モック設定
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {'marketCap': 6000000000000}
        mock_ticker_instance.history.return_value = self._create_performance_pipeline_data()
        mock_ticker.return_value = mock_ticker_instance

        async def run_test():
            start_time = time.time()

            # 完全自動化パイプライン実行
            selected_symbols = await self.selector.select_optimal_symbols(
                SelectionCriteria(target_symbols=5)
            )

            total_time = time.time() - start_time

            # パフォーマンス要件確認
            self.assertLess(total_time, 60.0,
                           f"エンドツーエンド実行時間 {total_time:.2f}秒 が目標 60秒 を上回りました")

            # 結果品質確認
            self.assertIsInstance(selected_symbols, list)
            self.assertGreater(len(selected_symbols), 0)

            print(f"エンドツーエンドパフォーマンス: {total_time:.2f}秒")

        asyncio.run(run_test())

    def _create_performance_pipeline_data(self) -> pd.DataFrame:
        """パフォーマンス測定用データ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        data = {
            'Open': np.random.uniform(1000, 1100, 60),
            'High': np.random.uniform(1100, 1200, 60),
            'Low': np.random.uniform(900, 1000, 60),
            'Close': np.random.uniform(1000, 1100, 60),
            'Volume': np.random.randint(3000000, 5000000, 60)
        }

        return pd.DataFrame(data, index=dates)


if __name__ == '__main__':
    # テストスイート設定
    test_suite = unittest.TestSuite()

    # データ統合テスト
    test_suite.addTest(unittest.makeSuite(TestSmartSymbolSelectorDataIntegration))

    # ML統合テスト
    test_suite.addTest(unittest.makeSuite(TestSmartSymbolSelectorMLIntegration))

    # 自動化統合テスト
    test_suite.addTest(unittest.makeSuite(TestSmartSymbolSelectorAutomationIntegration))

    # エンドツーエンドテスト
    test_suite.addTest(unittest.makeSuite(TestSmartSymbolSelectorEndToEnd))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 結果サマリー
    print(f"\n{'='*75}")
    print(f"SmartSymbolSelector統合テスト完了")
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗数: {len(result.failures)}")
    print(f"エラー数: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*75}")