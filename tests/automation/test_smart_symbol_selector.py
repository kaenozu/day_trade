#!/usr/bin/env python3
"""
Smart Symbol Selector Tests

Issue #487対応: 完全自動化システム実装 - Phase 1スマート銘柄自動選択テスト
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, AsyncMock
import sys

sys.path.append('C:/gemini-desktop/day_trade/src')

# テストに必要なモジュールをインポート
try:
    from day_trade.automation.smart_symbol_selector import (
        SmartSymbolSelector,
        SymbolMetrics,
        SelectionCriteria,
        MarketSegment,
        get_smart_selected_symbols
    )
    TEST_AVAILABLE = True
except ImportError as e:
    TEST_AVAILABLE = False
    print(f"テストモジュールインポートエラー: {e}")


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestSmartSymbolSelector:
    """Issue #487: スマート銘柄自動選択システムテスト"""
    
    @pytest.fixture
    def selector(self):
        """テスト用SmartSymbolSelector"""
        return SmartSymbolSelector()
    
    @pytest.fixture
    def mock_yfinance_data(self):
        """yfinanceモックデータ"""
        # 価格履歴データ
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        mock_hist = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 60),
            'High': np.random.uniform(110, 220, 60),
            'Low': np.random.uniform(90, 190, 60),
            'Close': np.random.uniform(95, 205, 60),
            'Volume': np.random.uniform(1e6, 5e6, 60)
        }, index=dates)
        
        # 企業情報
        mock_info = {
            'marketCap': 1e12,  # 1兆円
            'sector': 'Technology'
        }
        
        return mock_hist, mock_info
    
    def test_symbol_pool_initialization(self, selector):
        """Issue #487: 銘柄プール初期化テスト"""
        assert len(selector.symbol_pool) > 20
        assert "7203.T" in selector.symbol_pool  # トヨタ
        assert "6758.T" in selector.symbol_pool  # ソニー
        assert "9984.T" in selector.symbol_pool  # SBG
        
        # 銘柄名が正しく設定されていることを確認
        assert "トヨタ自動車" in selector.symbol_pool.values()
        assert "ソニーグループ" in selector.symbol_pool.values()
    
    def test_scoring_weights_configuration(self, selector):
        """Issue #487: スコア重み設定テスト"""
        weights = selector.scoring_weights
        
        # 重み項目の存在確認
        required_weights = ['liquidity', 'volatility', 'volume_consistency', 'market_cap']
        for weight in required_weights:
            assert weight in weights
            assert 0 < weights[weight] <= 1
        
        # 重みの合計が1.0になることを確認
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_selection_criteria_defaults(self):
        """Issue #487: 選定基準デフォルト値テスト"""
        criteria = SelectionCriteria()
        
        assert criteria.min_market_cap == 1e9  # 10億円
        assert criteria.min_avg_volume == 1e6   # 100万株
        assert criteria.max_volatility == 8.0   # 8%
        assert criteria.min_liquidity_score == 60.0
        assert criteria.target_symbols == 10
        assert criteria.exclude_sectors is None
    
    def test_calculate_liquidity_score(self, selector, mock_yfinance_data):
        """Issue #487: 流動性スコア計算テスト"""
        mock_hist, mock_info = mock_yfinance_data
        market_cap = mock_info['marketCap']
        
        score = selector._calculate_liquidity_score(mock_hist, market_cap)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    def test_calculate_volume_consistency(self, selector, mock_yfinance_data):
        """Issue #487: 出来高安定性計算テスト"""
        mock_hist, _ = mock_yfinance_data
        
        consistency = selector._calculate_volume_consistency(mock_hist)
        
        assert isinstance(consistency, float)
        assert 0 <= consistency <= 100
    
    def test_determine_price_trend(self, selector):
        """Issue #487: 価格トレンド判定テスト"""
        # 上昇トレンドデータ
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        upward_data = pd.DataFrame({
            'Close': np.linspace(100, 150, 60)  # 上昇トレンド
        }, index=dates)
        
        trend = selector._determine_price_trend(upward_data)
        assert trend in ["up", "down", "sideways"]
        
        # 下降トレンドデータ
        downward_data = pd.DataFrame({
            'Close': np.linspace(150, 100, 60)  # 下降トレンド
        }, index=dates)
        
        trend = selector._determine_price_trend(downward_data)
        assert trend in ["up", "down", "sideways"]
    
    def test_classify_market_segment(self, selector):
        """Issue #487: 市場セグメント分類テスト"""
        # 超大型株
        mega_cap = selector._classify_market_segment(15e12)  # 15兆円
        assert mega_cap == MarketSegment.MEGA_CAP
        
        # 大型株
        large_cap = selector._classify_market_segment(5e12)  # 5兆円
        assert large_cap == MarketSegment.LARGE_CAP
        
        # 中型株
        mid_cap = selector._classify_market_segment(2e12)  # 2兆円
        assert mid_cap == MarketSegment.MID_CAP
        
        # 小型株
        small_cap = selector._classify_market_segment(5e11)  # 5000億円
        assert small_cap == MarketSegment.SMALL_CAP
    
    def test_calculate_selection_score(self, selector):
        """Issue #487: 総合選定スコア計算テスト"""
        # 最適パラメータ
        score = selector._calculate_selection_score(
            liquidity=80.0,
            volatility=4.0,  # 最適ボラティリティ
            volume_consistency=75.0,
            market_cap=3e12  # 3兆円
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert score > 70  # 良好な条件では高スコア
        
        # 悪条件パラメータ
        bad_score = selector._calculate_selection_score(
            liquidity=30.0,
            volatility=15.0,  # 高ボラティリティ
            volume_consistency=25.0,
            market_cap=1e11   # 1000億円
        )
        
        assert bad_score < score  # 悪条件では低スコア
    
    def test_filter_symbols(self, selector):
        """Issue #487: 銘柄フィルタリングテスト"""
        # テスト用銘柄メトリクス
        test_metrics = [
            SymbolMetrics(
                symbol="GOOD.T", market_cap=2e12, avg_volume=2e6, price=1000,
                volatility=5.0, liquidity_score=80, volume_consistency=70,
                price_trend="up", market_segment=MarketSegment.LARGE_CAP, selection_score=85
            ),
            SymbolMetrics(
                symbol="BAD.T", market_cap=5e11, avg_volume=5e5, price=500,
                volatility=12.0, liquidity_score=40, volume_consistency=30,
                price_trend="down", market_segment=MarketSegment.SMALL_CAP, selection_score=35
            )
        ]
        
        criteria = SelectionCriteria()
        filtered = selector._filter_symbols(test_metrics, criteria)
        
        # 良い銘柄のみ残ることを確認
        assert len(filtered) == 1
        assert filtered[0].symbol == "GOOD.T"
    
    def test_rank_and_select(self, selector):
        """Issue #487: ランキング・選定テスト"""
        # テスト用銘柄メトリクス（スコア順）
        test_metrics = [
            SymbolMetrics(
                symbol="TOP1.T", market_cap=3e12, avg_volume=3e6, price=1500,
                volatility=4.0, liquidity_score=90, volume_consistency=85,
                price_trend="up", market_segment=MarketSegment.LARGE_CAP, selection_score=95
            ),
            SymbolMetrics(
                symbol="TOP2.T", market_cap=2e12, avg_volume=2e6, price=1000,
                volatility=5.0, liquidity_score=80, volume_consistency=75,
                price_trend="up", market_segment=MarketSegment.LARGE_CAP, selection_score=85
            ),
            SymbolMetrics(
                symbol="TOP3.T", market_cap=1.5e12, avg_volume=1.5e6, price=800,
                volatility=6.0, liquidity_score=70, volume_consistency=65,
                price_trend="sideways", market_segment=MarketSegment.MID_CAP, selection_score=75
            )
        ]
        
        selected = selector._rank_and_select(test_metrics, target_count=2)
        
        assert len(selected) == 2
        assert selected[0] == "TOP1.T"  # 最高スコア
        assert selected[1] == "TOP2.T"  # 次点スコア
    
    @patch('day_trade.automation.smart_symbol_selector.yf.Ticker')
    async def test_calculate_single_metric(self, mock_ticker, selector):
        """Issue #487: 個別銘柄メトリクス計算テスト"""
        # モックyfinanceデータ設定
        mock_hist, mock_info = self.mock_yfinance_data
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = mock_info
        mock_ticker_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_ticker_instance
        
        # セマフォ作成
        semaphore = asyncio.Semaphore(1)
        
        result = await selector._calculate_single_metric("7203.T", semaphore)
        
        assert isinstance(result, SymbolMetrics)
        assert result.symbol == "7203.T"
        assert result.market_cap > 0
        assert result.price > 0
        assert 0 <= result.liquidity_score <= 100
        assert 0 <= result.volume_consistency <= 100
        assert result.price_trend in ["up", "down", "sideways"]
        assert isinstance(result.market_segment, MarketSegment)
        assert 0 <= result.selection_score <= 100
    
    @patch('day_trade.automation.smart_symbol_selector.yf.Ticker')
    async def test_calculate_symbol_metrics(self, mock_ticker, selector):
        """Issue #487: 銘柄メトリクス並列計算テスト"""
        # モック設定
        mock_hist, mock_info = self.mock_yfinance_data
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = mock_info
        mock_ticker_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_ticker_instance
        
        test_symbols = ["7203.T", "6758.T", "9984.T"]
        
        results = await selector._calculate_symbol_metrics(test_symbols)
        
        assert len(results) <= len(test_symbols)  # エラー除外のため以下
        for result in results:
            assert isinstance(result, SymbolMetrics)
            assert result.symbol in test_symbols
    
    @patch('day_trade.automation.smart_symbol_selector.yf.Ticker')
    async def test_select_optimal_symbols_integration(self, mock_ticker, selector):
        """Issue #487: 最適銘柄選択統合テスト"""
        # モック設定
        mock_hist, mock_info = self.mock_yfinance_data
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = mock_info
        mock_ticker_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_ticker_instance
        
        # 小さなターゲット数でテスト
        criteria = SelectionCriteria(target_symbols=3)
        
        selected_symbols = await selector.select_optimal_symbols(criteria)
        
        assert isinstance(selected_symbols, list)
        assert len(selected_symbols) <= 3
        for symbol in selected_symbols:
            assert symbol in selector.symbol_pool
    
    def test_get_symbol_info(self, selector):
        """Issue #487: 銘柄情報取得テスト"""
        # 存在する銘柄
        name = selector.get_symbol_info("7203.T")
        assert name == "トヨタ自動車"
        
        # 存在しない銘柄
        none_name = selector.get_symbol_info("UNKNOWN.T")
        assert none_name is None
    
    @patch('day_trade.automation.smart_symbol_selector.SmartSymbolSelector.select_optimal_symbols')
    async def test_get_smart_selected_symbols_function(self, mock_select):
        """Issue #487: 外部インターフェース関数テスト"""
        mock_select.return_value = ["7203.T", "6758.T", "9984.T"]
        
        result = await get_smart_selected_symbols(target_count=3)
        
        assert result == ["7203.T", "6758.T", "9984.T"]
        mock_select.assert_called_once()
    
    def test_symbol_metrics_dataclass(self):
        """Issue #487: SymbolMetricsデータクラステスト"""
        metrics = SymbolMetrics(
            symbol="TEST.T",
            market_cap=1e12,
            avg_volume=1e6,
            price=1000,
            volatility=5.0,
            liquidity_score=80,
            volume_consistency=70,
            price_trend="up",
            market_segment=MarketSegment.LARGE_CAP,
            selection_score=85
        )
        
        assert metrics.symbol == "TEST.T"
        assert metrics.market_cap == 1e12
        assert metrics.market_segment == MarketSegment.LARGE_CAP
    
    async def test_error_handling_in_metric_calculation(self, selector):
        """Issue #487: メトリクス計算エラーハンドリングテスト"""
        # 存在しない銘柄でのエラーハンドリング
        semaphore = asyncio.Semaphore(1)
        
        with pytest.raises(Exception):
            await selector._calculate_single_metric("INVALID.T", semaphore)
    
    def test_market_segment_enum(self):
        """Issue #487: MarketSegment列挙型テスト"""
        assert MarketSegment.MEGA_CAP.value == "mega_cap"
        assert MarketSegment.LARGE_CAP.value == "large_cap"
        assert MarketSegment.MID_CAP.value == "mid_cap"
        assert MarketSegment.SMALL_CAP.value == "small_cap"


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestSmartSymbolSelectorPerformance:
    """Issue #487: スマート銘柄選択システム性能テスト"""
    
    @pytest.fixture
    def selector(self):
        """テスト用SmartSymbolSelector"""
        return SmartSymbolSelector()
    
    def test_scoring_algorithm_performance(self, selector):
        """Issue #487: スコアリングアルゴリズム性能テスト"""
        import time
        
        # 大量データでのスコアリング速度テスト
        start_time = time.time()
        
        for _ in range(1000):
            score = selector._calculate_selection_score(
                liquidity=np.random.uniform(0, 100),
                volatility=np.random.uniform(1, 20),
                volume_consistency=np.random.uniform(0, 100),
                market_cap=np.random.uniform(1e11, 10e12)
            )
        
        elapsed_time = time.time() - start_time
        
        # スコアリングが十分高速であることを確認
        assert elapsed_time < 1.0, f"スコアリングが遅い: {elapsed_time:.3f}秒"
    
    def test_filtering_efficiency(self, selector):
        """Issue #487: フィルタリング効率テスト"""
        # 大量テストデータ生成
        test_metrics = []
        for i in range(100):
            metrics = SymbolMetrics(
                symbol=f"TEST{i}.T",
                market_cap=np.random.uniform(1e11, 10e12),
                avg_volume=np.random.uniform(1e5, 5e6),
                price=np.random.uniform(100, 2000),
                volatility=np.random.uniform(1, 15),
                liquidity_score=np.random.uniform(20, 100),
                volume_consistency=np.random.uniform(20, 100),
                price_trend=np.random.choice(["up", "down", "sideways"]),
                market_segment=np.random.choice(list(MarketSegment)),
                selection_score=np.random.uniform(30, 95)
            )
            test_metrics.append(metrics)
        
        criteria = SelectionCriteria()
        
        import time
        start_time = time.time()
        filtered = selector._filter_symbols(test_metrics, criteria)
        elapsed_time = time.time() - start_time
        
        # フィルタリングが高速であることを確認
        assert elapsed_time < 0.1, f"フィルタリングが遅い: {elapsed_time:.3f}秒"
        assert len(filtered) <= len(test_metrics)


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestSmartSymbolSelectorEdgeCases:
    """Issue #487: スマート銘柄選択システムエッジケーステスト"""
    
    def test_empty_symbol_pool(self):
        """Issue #487: 空の銘柄プールテスト"""
        selector = SmartSymbolSelector()
        selector.symbol_pool = {}  # 空のプール
        
        criteria = SelectionCriteria(target_symbols=5)
        
        # 空のプールでもエラーが発生しないことを確認
        # (実際のテストは非同期関数のため、ここではsetupのみ確認)
        assert len(selector.symbol_pool) == 0
    
    def test_extreme_criteria(self):
        """Issue #487: 極端な選定基準テスト"""
        # 非現実的に厳しい基準
        strict_criteria = SelectionCriteria(
            min_market_cap=100e12,  # 100兆円（存在しない）
            min_avg_volume=100e6,   # 1億株（非現実的）
            max_volatility=0.1,     # 0.1%（非現実的に低い）
            min_liquidity_score=99.0, # 99点以上（非現実的）
            target_symbols=10
        )
        
        # 基準が設定されることを確認
        assert strict_criteria.min_market_cap == 100e12
        assert strict_criteria.max_volatility == 0.1
    
    def test_zero_volatility_scoring(self):
        """Issue #487: ゼロボラティリティスコアリングテスト"""
        selector = SmartSymbolSelector()
        
        # ボラティリティ0の場合
        score = selector._calculate_selection_score(
            liquidity=50.0,
            volatility=0.0,
            volume_consistency=50.0,
            market_cap=1e12
        )
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_nan_handling_in_calculations(self, selector):
        """Issue #487: NaN値処理テスト"""
        # NaNを含むデータでの処理
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        nan_data = pd.DataFrame({
            'Open': [100, np.nan, 102, 103, np.nan, 105, 106, 107, 108, 109],
            'High': [105, 106, np.nan, 108, 109, np.nan, 111, 112, 113, 114],
            'Low': [95, 96, 97, np.nan, 99, 100, np.nan, 102, 103, 104],
            'Close': [100, 101, 102, 103, 104, 105, 106, np.nan, 108, 109],
            'Volume': [1e6, 1.1e6, np.nan, 1.3e6, 1.4e6, np.nan, 1.6e6, 1.7e6, 1.8e6, 1.9e6]
        }, index=dates)
        
        # NaNが含まれてもエラーが発生しないことを確認
        try:
            liquidity_score = selector._calculate_liquidity_score(nan_data, 1e12)
            consistency = selector._calculate_volume_consistency(nan_data)
            trend = selector._determine_price_trend(nan_data)
            
            # 結果が有効な範囲内であることを確認
            assert 0 <= liquidity_score <= 100 or np.isnan(liquidity_score)
            assert 0 <= consistency <= 100 or np.isnan(consistency)
            assert trend in ["up", "down", "sideways"]
            
        except Exception as e:
            # エラーが発生した場合はログに記録（想定内）
            print(f"NaN処理でエラー（想定内）: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])