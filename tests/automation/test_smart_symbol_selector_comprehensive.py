#!/usr/bin/env python3
"""
Issue #755 Phase 3: SmartSymbolSelector包括的テストスイート

スマート銘柄自動選択システムの詳細検証
- 銘柄選択アルゴリズムテスト
- 流動性・ボラティリティ分析テスト
- 選定基準・フィルタリングテスト
- パフォーマンス・堅牢性テスト
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
except ImportError as e:
    print(f"インポートエラー: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")


class TestSmartSymbolSelectorCore(unittest.TestCase):
    """SmartSymbolSelectorコア機能テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.selector = SmartSymbolSelector()
        self.test_symbols = ['7203.T', '6758.T', '9984.T', '4519.T', '6098.T']

        # モックデータ準備
        self.mock_historical_data = self._create_mock_historical_data()
        self.mock_ticker_info = self._create_mock_ticker_info()

    def _create_mock_historical_data(self) -> pd.DataFrame:
        """モック履歴データ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        data = {
            'Open': np.random.uniform(1000, 1100, 60),
            'High': np.random.uniform(1100, 1200, 60),
            'Low': np.random.uniform(900, 1000, 60),
            'Close': np.random.uniform(1000, 1100, 60),
            'Volume': np.random.randint(1000000, 5000000, 60)
        }

        df = pd.DataFrame(data, index=dates)

        # より現実的な価格関係を確保
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('Low')] = min(
                df.iloc[i, df.columns.get_loc('Low')],
                df.iloc[i, df.columns.get_loc('Open')],
                df.iloc[i, df.columns.get_loc('Close')]
            )
            df.iloc[i, df.columns.get_loc('High')] = max(
                df.iloc[i, df.columns.get_loc('High')],
                df.iloc[i, df.columns.get_loc('Open')],
                df.iloc[i, df.columns.get_loc('Close')]
            )

        return df

    def _create_mock_ticker_info(self) -> Dict:
        """モックティッカー情報作成"""
        return {
            'marketCap': 30000000000000,  # 30兆円
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'recommendationKey': 'buy'
        }

    @patch('yfinance.Ticker')
    def test_symbol_metrics_calculation(self, mock_ticker):
        """銘柄メトリクス計算テスト"""
        # yfinanceのモック設定
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = self.mock_ticker_info
        mock_ticker_instance.history.return_value = self.mock_historical_data
        mock_ticker.return_value = mock_ticker_instance

        # 非同期テスト実行
        async def run_test():
            semaphore = asyncio.Semaphore(1)
            metrics = await self.selector._calculate_single_metric('7203.T', semaphore)

            # 基本属性確認
            self.assertEqual(metrics.symbol, '7203.T')
            self.assertGreater(metrics.market_cap, 0)
            self.assertGreater(metrics.avg_volume, 0)
            self.assertGreater(metrics.price, 0)

            # スコア範囲確認
            self.assertGreaterEqual(metrics.liquidity_score, 0)
            self.assertLessEqual(metrics.liquidity_score, 100)
            self.assertGreaterEqual(metrics.volume_consistency, 0)
            self.assertLessEqual(metrics.volume_consistency, 100)
            self.assertGreaterEqual(metrics.selection_score, 0)
            self.assertLessEqual(metrics.selection_score, 100)

            # 市場セグメント確認
            self.assertIsInstance(metrics.market_segment, MarketSegment)

            # 価格トレンド確認
            self.assertIn(metrics.price_trend, ['up', 'down', 'sideways'])

        asyncio.run(run_test())

    def test_liquidity_score_calculation(self):
        """流動性スコア計算テスト"""
        # 高流動性データ
        high_liquidity_data = self.mock_historical_data.copy()
        high_liquidity_data['Volume'] = high_liquidity_data['Volume'] * 10

        score_high = self.selector._calculate_liquidity_score(
            high_liquidity_data, 50000000000000  # 50兆円
        )

        # 低流動性データ
        low_liquidity_data = self.mock_historical_data.copy()
        low_liquidity_data['Volume'] = high_liquidity_data['Volume'] * 0.1

        score_low = self.selector._calculate_liquidity_score(
            low_liquidity_data, 1000000000000  # 1兆円
        )

        # 高流動性の方が高スコアであることを確認
        self.assertGreater(score_high, score_low)
        self.assertGreaterEqual(score_high, 0)
        self.assertLessEqual(score_high, 100)

    def test_volume_consistency_calculation(self):
        """出来高安定性計算テスト"""
        # 安定した出来高データ
        stable_data = self.mock_historical_data.copy()
        stable_data['Volume'] = 2000000  # 一定出来高

        score_stable = self.selector._calculate_volume_consistency(stable_data)

        # 不安定な出来高データ
        volatile_data = self.mock_historical_data.copy()
        volatile_volumes = []
        for i in range(60):
            if i % 10 == 0:
                volatile_volumes.append(10000000)  # 異常高出来高
            else:
                volatile_volumes.append(1000000)   # 通常出来高
        volatile_data['Volume'] = volatile_volumes

        score_volatile = self.selector._calculate_volume_consistency(volatile_data)

        # 安定した出来高の方が高スコアであることを確認
        self.assertGreater(score_stable, score_volatile)

    def test_price_trend_determination(self):
        """価格トレンド判定テスト"""
        # 上昇トレンドデータ
        upward_data = self.mock_historical_data.copy()
        upward_data['Close'] = np.linspace(1000, 1200, 60)  # 継続上昇

        trend_up = self.selector._determine_price_trend(upward_data)

        # 下降トレンドデータ
        downward_data = self.mock_historical_data.copy()
        downward_data['Close'] = np.linspace(1200, 1000, 60)  # 継続下降

        trend_down = self.selector._determine_price_trend(downward_data)

        # 横ばいトレンドデータ
        sideways_data = self.mock_historical_data.copy()
        sideways_data['Close'] = 1100 + np.random.normal(0, 10, 60)  # 横ばい

        trend_sideways = self.selector._determine_price_trend(sideways_data)

        # トレンド判定結果確認
        self.assertEqual(trend_up, 'up')
        self.assertEqual(trend_down, 'down')
        self.assertIn(trend_sideways, ['sideways', 'up', 'down'])

    def test_market_segment_classification(self):
        """市場セグメント分類テスト"""
        # 各セグメントのテスト
        mega_cap = self.selector._classify_market_segment(15e12)  # 15兆円
        large_cap = self.selector._classify_market_segment(5e12)   # 5兆円
        mid_cap = self.selector._classify_market_segment(2e12)     # 2兆円
        small_cap = self.selector._classify_market_segment(0.5e12) # 0.5兆円

        self.assertEqual(mega_cap, MarketSegment.MEGA_CAP)
        self.assertEqual(large_cap, MarketSegment.LARGE_CAP)
        self.assertEqual(mid_cap, MarketSegment.MID_CAP)
        self.assertEqual(small_cap, MarketSegment.SMALL_CAP)

    def test_selection_score_calculation(self):
        """選定スコア計算テスト"""
        # 理想的な指標値
        ideal_score = self.selector._calculate_selection_score(
            liquidity=90.0,
            volatility=4.0,  # 理想的ボラティリティ
            volume_consistency=85.0,
            market_cap=5e12
        )

        # 非理想的な指標値
        poor_score = self.selector._calculate_selection_score(
            liquidity=30.0,
            volatility=15.0,  # 高すぎるボラティリティ
            volume_consistency=20.0,
            market_cap=0.1e12
        )

        # 理想的指標の方が高スコアであることを確認
        self.assertGreater(ideal_score, poor_score)
        self.assertGreaterEqual(ideal_score, 0)
        self.assertLessEqual(ideal_score, 100)


class TestSmartSymbolSelectorSelection(unittest.TestCase):
    """銘柄選択プロセステスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.selector = SmartSymbolSelector()
        self.criteria = SelectionCriteria(target_symbols=5)

    def test_symbol_filtering(self):
        """銘柄フィルタリングテスト"""
        # テスト用メトリクス作成
        test_metrics = [
            SymbolMetrics(
                symbol='GOOD.T',
                market_cap=5e12,      # 基準を満たす
                avg_volume=2e6,       # 基準を満たす
                price=1000,
                volatility=6.0,       # 基準を満たす
                liquidity_score=70.0, # 基準を満たす
                volume_consistency=80.0,
                price_trend='up',
                market_segment=MarketSegment.LARGE_CAP,
                selection_score=85.0
            ),
            SymbolMetrics(
                symbol='BAD.T',
                market_cap=0.5e12,    # 基準未満
                avg_volume=0.5e6,     # 基準未満
                price=500,
                volatility=12.0,      # 基準超過
                liquidity_score=40.0, # 基準未満
                volume_consistency=30.0,
                price_trend='down',
                market_segment=MarketSegment.SMALL_CAP,
                selection_score=30.0
            )
        ]

        filtered = self.selector._filter_symbols(test_metrics, self.criteria)

        # 基準を満たす銘柄のみフィルタされることを確認
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].symbol, 'GOOD.T')

    def test_ranking_and_selection(self):
        """ランキング・選定テスト"""
        # 異なるスコアのメトリクス作成
        test_metrics = []
        scores = [95, 85, 75, 65, 55, 45, 35, 25]
        segments = [MarketSegment.MEGA_CAP, MarketSegment.LARGE_CAP,
                   MarketSegment.MID_CAP, MarketSegment.SMALL_CAP] * 2

        for i, (score, segment) in enumerate(zip(scores, segments)):
            test_metrics.append(
                SymbolMetrics(
                    symbol=f'TEST{i}.T',
                    market_cap=5e12,
                    avg_volume=2e6,
                    price=1000,
                    volatility=5.0,
                    liquidity_score=70.0,
                    volume_consistency=70.0,
                    price_trend='up',
                    market_segment=segment,
                    selection_score=score
                )
            )

        selected = self.selector._rank_and_select(test_metrics, 4)

        # 選定数確認
        self.assertEqual(len(selected), 4)

        # 高スコア銘柄が選定されることを確認
        self.assertIn('TEST0.T', selected)  # 最高スコア
        self.assertIn('TEST1.T', selected)  # 2番目スコア

    @patch('yfinance.Ticker')
    def test_optimal_selection_integration(self, mock_ticker):
        """最適選択統合テスト"""
        # yfinanceモック設定
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {'marketCap': 5000000000000}
        mock_ticker_instance.history.return_value = self._create_valid_historical_data()
        mock_ticker.return_value = mock_ticker_instance

        async def run_test():
            # 少数銘柄で実行
            test_criteria = SelectionCriteria(
                target_symbols=3,
                min_market_cap=1e12,
                min_avg_volume=1e6
            )

            # モックシンボルプールを制限
            original_pool = self.selector.symbol_pool
            self.selector.symbol_pool = {
                '7203.T': 'トヨタ自動車',
                '6758.T': 'ソニーグループ',
                '9984.T': 'ソフトバンクグループ'
            }

            try:
                selected = await self.selector.select_optimal_symbols(test_criteria)

                # 結果検証
                self.assertIsInstance(selected, list)
                self.assertLessEqual(len(selected), 3)
                self.assertTrue(all(symbol.endswith('.T') for symbol in selected))

            finally:
                self.selector.symbol_pool = original_pool

        asyncio.run(run_test())

    def _create_valid_historical_data(self) -> pd.DataFrame:
        """有効な履歴データ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        # より現実的なデータパターン
        base_price = 1000
        prices = []

        for i in range(60):
            # ランダムウォーク + 小さなトレンド
            change = np.random.normal(0, 0.02) + 0.001
            base_price *= (1 + change)
            prices.append(base_price)

        data = {
            'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'High': [p * np.random.uniform(1.01, 1.03) for p in prices],
            'Low': [p * np.random.uniform(0.97, 0.99) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1500000, 3000000, 60)
        }

        return pd.DataFrame(data, index=dates)


class TestSmartSymbolSelectorRobustness(unittest.TestCase):
    """堅牢性・エラーハンドリングテスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.selector = SmartSymbolSelector()

    @patch('yfinance.Ticker')
    def test_invalid_data_handling(self, mock_ticker):
        """無効データ処理テスト"""
        # 空データの場合
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {}
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance

        async def run_test():
            semaphore = asyncio.Semaphore(1)

            # エラーが適切に発生することを確認
            with self.assertRaises(Exception):
                await self.selector._calculate_single_metric('INVALID.T', semaphore)

        asyncio.run(run_test())

    @patch('yfinance.Ticker')
    def test_network_error_handling(self, mock_ticker):
        """ネットワークエラー処理テスト"""
        # ネットワークエラーをシミュレート
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = Exception("Network error")
        mock_ticker.return_value = mock_ticker_instance

        async def run_test():
            semaphore = asyncio.Semaphore(1)

            with self.assertRaises(Exception):
                await self.selector._calculate_single_metric('ERROR.T', semaphore)

        asyncio.run(run_test())

    def test_empty_symbol_pool_handling(self):
        """空銘柄プール処理テスト"""
        # 銘柄プールを空にする
        original_pool = self.selector.symbol_pool
        self.selector.symbol_pool = {}

        async def run_test():
            try:
                selected = await self.selector.select_optimal_symbols()

                # 空リストが返されることを確認
                self.assertEqual(len(selected), 0)

            finally:
                self.selector.symbol_pool = original_pool

        asyncio.run(run_test())

    def test_extreme_criteria_handling(self):
        """極端な選定基準処理テスト"""
        # 非常に厳しい基準
        strict_criteria = SelectionCriteria(
            min_market_cap=100e12,      # 100兆円（非現実的）
            min_avg_volume=100e6,       # 1億株（非現実的）
            max_volatility=0.1,         # 0.1%（非現実的）
            min_liquidity_score=99.0,   # 99点（非現実的）
            target_symbols=20
        )

        # テスト用メトリクス（基準を満たさない）
        test_metrics = [
            SymbolMetrics(
                symbol='TEST.T',
                market_cap=1e12,
                avg_volume=1e6,
                price=1000,
                volatility=5.0,
                liquidity_score=70.0,
                volume_consistency=70.0,
                price_trend='up',
                market_segment=MarketSegment.LARGE_CAP,
                selection_score=70.0
            )
        ]

        filtered = self.selector._filter_symbols(test_metrics, strict_criteria)

        # 基準を満たさないため空リストが返されることを確認
        self.assertEqual(len(filtered), 0)

    def test_concurrent_access_safety(self):
        """並行アクセス安全性テスト"""
        async def run_test():
            # 複数の並行選択実行
            tasks = []
            for i in range(3):
                task = asyncio.create_task(
                    self.selector.select_optimal_symbols(
                        SelectionCriteria(target_symbols=2)
                    )
                )
                tasks.append(task)

            # すべてのタスクが正常完了することを確認
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    self.fail(f"並行実行でエラー: {result}")
                else:
                    self.assertIsInstance(result, list)

        # yfinanceをモック化して実行
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {'marketCap': 5000000000000}
            mock_ticker_instance.history.return_value = self._create_stable_data()
            mock_ticker.return_value = mock_ticker_instance

            asyncio.run(run_test())

    def _create_stable_data(self) -> pd.DataFrame:
        """安定したテストデータ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        data = {
            'Open': [1000] * 60,
            'High': [1020] * 60,
            'Low': [980] * 60,
            'Close': [1010] * 60,
            'Volume': [2000000] * 60
        }

        return pd.DataFrame(data, index=dates)


class TestSmartSymbolSelectorPerformance(unittest.TestCase):
    """パフォーマンステスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.selector = SmartSymbolSelector()

    @patch('yfinance.Ticker')
    def test_selection_performance(self, mock_ticker):
        """選択処理パフォーマンステスト"""
        # yfinanceモック設定
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {'marketCap': 5000000000000}
        mock_ticker_instance.history.return_value = self._create_performance_data()
        mock_ticker.return_value = mock_ticker_instance

        async def run_test():
            import time

            start_time = time.time()

            # 選択実行
            selected = await self.selector.select_optimal_symbols(
                SelectionCriteria(target_symbols=5)
            )

            execution_time = time.time() - start_time

            # パフォーマンス要件確認（30秒以内）
            self.assertLess(execution_time, 30.0,
                           f"選択処理時間 {execution_time:.2f}秒 が目標 30秒 を上回りました")

            # 結果が返されることを確認
            self.assertIsInstance(selected, list)

            print(f"選択処理時間: {execution_time:.2f}秒")

        asyncio.run(run_test())

    def test_memory_efficiency(self):
        """メモリ効率性テスト"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 大量の銘柄メトリクス処理をシミュレート
        large_metrics_list = []
        for i in range(1000):
            metric = SymbolMetrics(
                symbol=f'TEST{i}.T',
                market_cap=np.random.uniform(1e12, 10e12),
                avg_volume=np.random.uniform(1e6, 10e6),
                price=np.random.uniform(500, 2000),
                volatility=np.random.uniform(2, 10),
                liquidity_score=np.random.uniform(0, 100),
                volume_consistency=np.random.uniform(0, 100),
                price_trend=np.random.choice(['up', 'down', 'sideways']),
                market_segment=np.random.choice(list(MarketSegment)),
                selection_score=np.random.uniform(0, 100)
            )
            large_metrics_list.append(metric)

        # フィルタリング・ランキング処理
        criteria = SelectionCriteria(target_symbols=50)
        filtered = self.selector._filter_symbols(large_metrics_list, criteria)
        selected = self.selector._rank_and_select(filtered, 50)

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # メモリ使用量が合理的範囲内であることを確認
        self.assertLess(memory_increase, 100,
                       f"メモリ使用量増加 {memory_increase:.1f}MB が過大です")

        print(f"メモリ使用量増加: {memory_increase:.1f}MB")

    def _create_performance_data(self) -> pd.DataFrame:
        """パフォーマンステスト用データ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        data = {
            'Open': np.random.uniform(1000, 1100, 60),
            'High': np.random.uniform(1100, 1200, 60),
            'Low': np.random.uniform(900, 1000, 60),
            'Close': np.random.uniform(1000, 1100, 60),
            'Volume': np.random.randint(1000000, 5000000, 60)
        }

        return pd.DataFrame(data, index=dates)


class TestSmartSymbolSelectorIntegration(unittest.TestCase):
    """統合・外部インターフェーステスト"""

    @patch('yfinance.Ticker')
    def test_external_interface(self, mock_ticker):
        """外部インターフェーステスト"""
        # yfinanceモック設定
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {'marketCap': 5000000000000}
        mock_ticker_instance.history.return_value = self._create_interface_data()
        mock_ticker.return_value = mock_ticker_instance

        async def run_test():
            # 外部インターフェース関数テスト
            symbols = await get_smart_selected_symbols(target_count=3)

            # 結果検証
            self.assertIsInstance(symbols, list)
            self.assertLessEqual(len(symbols), 3)
            self.assertTrue(all(isinstance(s, str) for s in symbols))

        asyncio.run(run_test())

    def test_symbol_info_retrieval(self):
        """銘柄情報取得テスト"""
        # 既知の銘柄
        info = self.selector.get_symbol_info('7203.T')
        self.assertEqual(info, 'トヨタ自動車')

        # 未知の銘柄
        info_unknown = self.selector.get_symbol_info('UNKNOWN.T')
        self.assertIsNone(info_unknown)

    def test_configuration_customization(self):
        """設定カスタマイズテスト"""
        # カスタム設定
        custom_criteria = SelectionCriteria(
            min_market_cap=2e12,
            min_avg_volume=2e6,
            max_volatility=6.0,
            min_liquidity_score=70.0,
            target_symbols=8
        )

        # 設定値が正しく設定されることを確認
        self.assertEqual(custom_criteria.min_market_cap, 2e12)
        self.assertEqual(custom_criteria.target_symbols, 8)

    def _create_interface_data(self) -> pd.DataFrame:
        """インターフェーステスト用データ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        data = {
            'Open': [1050] * 60,
            'High': [1070] * 60,
            'Low': [1030] * 60,
            'Close': [1060] * 60,
            'Volume': [2500000] * 60
        }

        return pd.DataFrame(data, index=dates)


if __name__ == '__main__':
    # テストスイート設定
    test_suite = unittest.TestSuite()

    # コア機能テスト
    test_suite.addTest(unittest.makeSuite(TestSmartSymbolSelectorCore))

    # 選択プロセステスト
    test_suite.addTest(unittest.makeSuite(TestSmartSymbolSelectorSelection))

    # 堅牢性テスト
    test_suite.addTest(unittest.makeSuite(TestSmartSymbolSelectorRobustness))

    # パフォーマンステスト
    test_suite.addTest(unittest.makeSuite(TestSmartSymbolSelectorPerformance))

    # 統合テスト
    test_suite.addTest(unittest.makeSuite(TestSmartSymbolSelectorIntegration))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 結果サマリー
    print(f"\n{'='*70}")
    print(f"SmartSymbolSelector包括的テスト完了")
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗数: {len(result.failures)}")
    print(f"エラー数: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*70}")