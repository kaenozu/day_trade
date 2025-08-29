#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Technical Indicators Analysis Tests
テクニカル指標分析テスト
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# モッククラスを定義
class MockTechnicalIndicators:
    """テクニカル指標計算モック"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """単純移動平均"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, span: int) -> pd.Series:
        """指数移動平均"""
        return data.ewm(span=span).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """RSI（相対力指数）"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD"""
        ema_fast = MockTechnicalIndicators.ema(data, fast)
        ema_slow = MockTechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = MockTechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """ボリンジャーバンド"""
        sma = MockTechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'bandwidth': (upper - lower) / sma * 100
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """ストキャスティクス"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """平均真の値幅（ATR）"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, 
                   window: int = 14) -> pd.Series:
        """ウィリアムズ%R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        wr = ((highest_high - close) / (highest_high - lowest_low)) * -100
        return wr
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, 
            window: int = 20) -> pd.Series:
        """商品チャネル指数（CCI）"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci


class MockPatternDetector:
    """パターン検出モック"""
    
    @staticmethod
    def detect_double_top(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        """ダブルトップ検出"""
        # 簡単な実装
        peaks = MockPatternDetector._find_peaks(high, window)
        double_tops = pd.Series(False, index=high.index)
        
        for i in range(1, len(peaks)):
            if peaks.iloc[i] and peaks.iloc[i-window:i-1].any():
                double_tops.iloc[i] = True
                
        return double_tops
    
    @staticmethod
    def detect_double_bottom(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        """ダブルボトム検出"""
        # 簡単な実装
        troughs = MockPatternDetector._find_troughs(low, window)
        double_bottoms = pd.Series(False, index=low.index)
        
        for i in range(1, len(troughs)):
            if troughs.iloc[i] and troughs.iloc[i-window:i-1].any():
                double_bottoms.iloc[i] = True
                
        return double_bottoms
    
    @staticmethod
    def detect_head_and_shoulders(high: pd.Series, low: pd.Series, 
                                 window: int = 20) -> pd.Series:
        """ヘッドアンドショルダー検出"""
        peaks = MockPatternDetector._find_peaks(high, window)
        patterns = pd.Series(False, index=high.index)
        
        # 簡単な3つのピークパターン検出
        for i in range(2*window, len(peaks)-window):
            if (peaks.iloc[i-2*window] and peaks.iloc[i] and 
                peaks.iloc[i+window] and 
                high.iloc[i] > high.iloc[i-2*window] and 
                high.iloc[i] > high.iloc[i+window]):
                patterns.iloc[i] = True
                
        return patterns
    
    @staticmethod
    def _find_peaks(data: pd.Series, window: int) -> pd.Series:
        """ピーク検出"""
        peaks = pd.Series(False, index=data.index)
        
        for i in range(window, len(data) - window):
            if (data.iloc[i] == data.iloc[i-window:i+window+1].max() and
                data.iloc[i] > data.iloc[i-1] and
                data.iloc[i] > data.iloc[i+1]):
                peaks.iloc[i] = True
                
        return peaks
    
    @staticmethod
    def _find_troughs(data: pd.Series, window: int) -> pd.Series:
        """トラフ検出"""
        troughs = pd.Series(False, index=data.index)
        
        for i in range(window, len(data) - window):
            if (data.iloc[i] == data.iloc[i-window:i+window+1].min() and
                data.iloc[i] < data.iloc[i-1] and
                data.iloc[i] < data.iloc[i+1]):
                troughs.iloc[i] = True
                
        return troughs


class MockTechnicalAnalyzer:
    """テクニカル分析器モック"""
    
    def __init__(self):
        self.indicators = MockTechnicalIndicators()
        self.patterns = MockPatternDetector()
    
    def analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """トレンド分析"""
        close = data['close']
        
        # 移動平均トレンド
        sma_short = self.indicators.sma(close, 20)
        sma_long = self.indicators.sma(close, 50)
        
        # トレンド判定
        current_trend = "上昇" if sma_short.iloc[-1] > sma_long.iloc[-1] else "下降"
        
        # トレンド強度
        trend_strength = abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / close.iloc[-1] * 100
        
        return {
            'trend_direction': current_trend,
            'trend_strength': trend_strength,
            'sma_20': sma_short.iloc[-1],
            'sma_50': sma_long.iloc[-1],
            'price_above_sma20': close.iloc[-1] > sma_short.iloc[-1],
            'price_above_sma50': close.iloc[-1] > sma_long.iloc[-1]
        }
    
    def analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """モメンタム分析"""
        close = data['close']
        high = data['high']
        low = data['low']
        
        # RSI
        rsi = self.indicators.rsi(close)
        
        # MACD
        macd_data = self.indicators.macd(close)
        
        # ストキャスティクス
        stoch_data = self.indicators.stochastic(high, low, close)
        
        return {
            'rsi': rsi.iloc[-1],
            'rsi_signal': self._get_rsi_signal(rsi.iloc[-1]),
            'macd_line': macd_data['macd'].iloc[-1],
            'macd_signal': macd_data['signal'].iloc[-1],
            'macd_histogram': macd_data['histogram'].iloc[-1],
            'macd_cross': self._get_macd_signal(macd_data),
            'stoch_k': stoch_data['k'].iloc[-1],
            'stoch_d': stoch_data['d'].iloc[-1],
            'stoch_signal': self._get_stoch_signal(stoch_data)
        }
    
    def analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ボラティリティ分析"""
        close = data['close']
        high = data['high']
        low = data['low']
        
        # ボリンジャーバンド
        bb = self.indicators.bollinger_bands(close)
        
        # ATR
        atr = self.indicators.atr(high, low, close)
        
        return {
            'bb_upper': bb['upper'].iloc[-1],
            'bb_middle': bb['middle'].iloc[-1],
            'bb_lower': bb['lower'].iloc[-1],
            'bb_position': self._get_bb_position(close.iloc[-1], bb),
            'bb_squeeze': bb['bandwidth'].iloc[-1] < 10,
            'atr': atr.iloc[-1],
            'volatility_level': self._get_volatility_level(atr.iloc[-5:])
        }
    
    def detect_signals(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """シグナル検出"""
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'neutral_signals': []
        }
        
        # 各分析結果からシグナル生成
        trend = self.analyze_trends(data)
        momentum = self.analyze_momentum(data)
        volatility = self.analyze_volatility(data)
        
        # トレンドシグナル
        if trend['trend_direction'] == '上昇' and trend['trend_strength'] > 2:
            signals['buy_signals'].append('強い上昇トレンド')
        elif trend['trend_direction'] == '下降' and trend['trend_strength'] > 2:
            signals['sell_signals'].append('強い下降トレンド')
        
        # モメンタムシグナル
        if momentum['rsi_signal'] == '買われすぎ':
            signals['sell_signals'].append('RSI買われすぎ')
        elif momentum['rsi_signal'] == '売られすぎ':
            signals['buy_signals'].append('RSI売られすぎ')
        
        if momentum['macd_cross'] == 'ゴールデンクロス':
            signals['buy_signals'].append('MACDゴールデンクロス')
        elif momentum['macd_cross'] == 'デッドクロス':
            signals['sell_signals'].append('MACDデッドクロス')
        
        return signals
    
    def _get_rsi_signal(self, rsi_value: float) -> str:
        """RSIシグナル判定"""
        if rsi_value > 70:
            return '買われすぎ'
        elif rsi_value < 30:
            return '売られすぎ'
        else:
            return '中性'
    
    def _get_macd_signal(self, macd_data: Dict[str, pd.Series]) -> str:
        """MACDシグナル判定"""
        current_histogram = macd_data['histogram'].iloc[-1]
        previous_histogram = macd_data['histogram'].iloc[-2]
        
        if current_histogram > 0 and previous_histogram <= 0:
            return 'ゴールデンクロス'
        elif current_histogram < 0 and previous_histogram >= 0:
            return 'デッドクロス'
        else:
            return '継続'
    
    def _get_stoch_signal(self, stoch_data: Dict[str, pd.Series]) -> str:
        """ストキャスティクスシグナル判定"""
        k_value = stoch_data['k'].iloc[-1]
        d_value = stoch_data['d'].iloc[-1]
        
        if k_value > 80 and d_value > 80:
            return '買われすぎ'
        elif k_value < 20 and d_value < 20:
            return '売られすぎ'
        else:
            return '中性'
    
    def _get_bb_position(self, price: float, bb: Dict[str, pd.Series]) -> str:
        """ボリンジャーバンド内での価格位置"""
        upper = bb['upper'].iloc[-1]
        lower = bb['lower'].iloc[-1]
        
        if price > upper:
            return 'バンド上抜け'
        elif price < lower:
            return 'バンド下抜け'
        else:
            return 'バンド内'
    
    def _get_volatility_level(self, atr_series: pd.Series) -> str:
        """ボラティリティレベル判定"""
        current_atr = atr_series.iloc[-1]
        avg_atr = atr_series.mean()
        
        if current_atr > avg_atr * 1.5:
            return '高'
        elif current_atr < avg_atr * 0.5:
            return '低'
        else:
            return '中'


def create_sample_data(days: int = 100) -> pd.DataFrame:
    """サンプルデータ作成"""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # ランダムウォークベースの価格生成
    np.random.seed(42)
    price_changes = np.random.normal(0, 2, days)
    prices = 100 + np.cumsum(price_changes)
    
    # OHLC生成
    opens = prices + np.random.normal(0, 0.5, days)
    highs = np.maximum(opens, prices) + np.abs(np.random.normal(0, 1, days))
    lows = np.minimum(opens, prices) - np.abs(np.random.normal(0, 1, days))
    volumes = np.random.randint(100000, 1000000, days)
    
    return pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })


class TestMockTechnicalIndicators:
    """テクニカル指標テストクラス"""
    
    @pytest.fixture
    def sample_data(self):
        """サンプルデータフィクスチャ"""
        return create_sample_data()
    
    def test_sma_calculation(self, sample_data):
        """単純移動平均計算テスト"""
        close = sample_data['close']
        sma_20 = MockTechnicalIndicators.sma(close, 20)
        
        assert len(sma_20) == len(close)
        assert sma_20.iloc[:19].isna().all()  # 最初の19個はNaN
        assert not sma_20.iloc[19:].isna().any()  # 20個目以降は値あり
        
        # 手動計算との比較
        manual_sma = close.iloc[0:20].mean()
        assert abs(sma_20.iloc[19] - manual_sma) < 0.001
    
    def test_ema_calculation(self, sample_data):
        """指数移動平均計算テスト"""
        close = sample_data['close']
        ema_12 = MockTechnicalIndicators.ema(close, 12)
        
        assert len(ema_12) == len(close)
        assert not ema_12.isna().any()  # EMAは初回から値あり
        
        # EMAは最新の値により重みを置く
        assert ema_12.iloc[-1] != close.iloc[-12:].mean()
    
    def test_rsi_calculation(self, sample_data):
        """RSI計算テスト"""
        close = sample_data['close']
        rsi = MockTechnicalIndicators.rsi(close)
        
        assert len(rsi) == len(close)
        
        # RSIは0-100の範囲
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd_calculation(self, sample_data):
        """MACD計算テスト"""
        close = sample_data['close']
        macd = MockTechnicalIndicators.macd(close)
        
        assert 'macd' in macd
        assert 'signal' in macd
        assert 'histogram' in macd
        
        # 全て同じ長さ
        assert len(macd['macd']) == len(close)
        assert len(macd['signal']) == len(close)
        assert len(macd['histogram']) == len(close)
        
        # ヒストグラム = MACD - シグナル
        histogram_calc = macd['macd'] - macd['signal']
        assert (abs(macd['histogram'] - histogram_calc) < 0.001).all()
    
    def test_bollinger_bands(self, sample_data):
        """ボリンジャーバンド計算テスト"""
        close = sample_data['close']
        bb = MockTechnicalIndicators.bollinger_bands(close)
        
        assert 'upper' in bb
        assert 'middle' in bb
        assert 'lower' in bb
        assert 'bandwidth' in bb
        
        # 上バンド >= 中間バンド >= 下バンド
        valid_data = bb['upper'].dropna()
        for i in valid_data.index:
            assert bb['upper'].loc[i] >= bb['middle'].loc[i]
            assert bb['middle'].loc[i] >= bb['lower'].loc[i]
    
    def test_stochastic(self, sample_data):
        """ストキャスティクス計算テスト"""
        high = sample_data['high']
        low = sample_data['low']
        close = sample_data['close']
        
        stoch = MockTechnicalIndicators.stochastic(high, low, close)
        
        assert 'k' in stoch
        assert 'd' in stoch
        
        # 0-100の範囲
        valid_k = stoch['k'].dropna()
        valid_d = stoch['d'].dropna()
        
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()
        assert (valid_d >= 0).all()
        assert (valid_d <= 100).all()
    
    def test_atr_calculation(self, sample_data):
        """ATR計算テスト"""
        high = sample_data['high']
        low = sample_data['low']
        close = sample_data['close']
        
        atr = MockTechnicalIndicators.atr(high, low, close)
        
        assert len(atr) == len(close)
        
        # ATRは正の値
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()


class TestMockPatternDetector:
    """パターン検出テストクラス"""
    
    @pytest.fixture
    def sample_data(self):
        """サンプルデータフィクスチャ"""
        return create_sample_data()
    
    def test_double_top_detection(self, sample_data):
        """ダブルトップ検出テスト"""
        high = sample_data['high']
        low = sample_data['low']
        
        double_tops = MockPatternDetector.detect_double_top(high, low)
        
        assert len(double_tops) == len(high)
        assert double_tops.dtype == bool
        
        # パターン検出数の妥当性チェック
        pattern_count = double_tops.sum()
        assert pattern_count >= 0
        assert pattern_count < len(high) * 0.1  # 全体の10%未満
    
    def test_double_bottom_detection(self, sample_data):
        """ダブルボトム検出テスト"""
        high = sample_data['high']
        low = sample_data['low']
        
        double_bottoms = MockPatternDetector.detect_double_bottom(high, low)
        
        assert len(double_bottoms) == len(low)
        assert double_bottoms.dtype == bool
    
    def test_head_and_shoulders_detection(self, sample_data):
        """ヘッドアンドショルダー検出テスト"""
        high = sample_data['high']
        low = sample_data['low']
        
        patterns = MockPatternDetector.detect_head_and_shoulders(high, low)
        
        assert len(patterns) == len(high)
        assert patterns.dtype == bool


class TestMockTechnicalAnalyzer:
    """テクニカル分析器テストクラス"""
    
    @pytest.fixture
    def analyzer(self):
        """分析器フィクスチャ"""
        return MockTechnicalAnalyzer()
    
    @pytest.fixture
    def sample_data(self):
        """サンプルデータフィクスチャ"""
        return create_sample_data()
    
    def test_trend_analysis(self, analyzer, sample_data):
        """トレンド分析テスト"""
        result = analyzer.analyze_trends(sample_data)
        
        assert 'trend_direction' in result
        assert 'trend_strength' in result
        assert 'sma_20' in result
        assert 'sma_50' in result
        assert 'price_above_sma20' in result
        assert 'price_above_sma50' in result
        
        assert result['trend_direction'] in ['上昇', '下降']
        assert isinstance(result['trend_strength'], float)
        assert isinstance(result['price_above_sma20'], bool)
        assert isinstance(result['price_above_sma50'], bool)
    
    def test_momentum_analysis(self, analyzer, sample_data):
        """モメンタム分析テスト"""
        result = analyzer.analyze_momentum(sample_data)
        
        assert 'rsi' in result
        assert 'rsi_signal' in result
        assert 'macd_line' in result
        assert 'macd_signal' in result
        assert 'macd_histogram' in result
        assert 'macd_cross' in result
        assert 'stoch_k' in result
        assert 'stoch_d' in result
        assert 'stoch_signal' in result
        
        assert 0 <= result['rsi'] <= 100
        assert result['rsi_signal'] in ['買われすぎ', '売られすぎ', '中性']
        assert result['stoch_signal'] in ['買われすぎ', '売られすぎ', '中性']
    
    def test_volatility_analysis(self, analyzer, sample_data):
        """ボラティリティ分析テスト"""
        result = analyzer.analyze_volatility(sample_data)
        
        assert 'bb_upper' in result
        assert 'bb_middle' in result
        assert 'bb_lower' in result
        assert 'bb_position' in result
        assert 'bb_squeeze' in result
        assert 'atr' in result
        assert 'volatility_level' in result
        
        assert result['bb_upper'] >= result['bb_middle'] >= result['bb_lower']
        assert result['bb_position'] in ['バンド上抜け', 'バンド下抜け', 'バンド内']
        assert isinstance(result['bb_squeeze'], bool)
        assert result['volatility_level'] in ['高', '中', '低']
    
    def test_signal_detection(self, analyzer, sample_data):
        """シグナル検出テスト"""
        signals = analyzer.detect_signals(sample_data)
        
        assert 'buy_signals' in signals
        assert 'sell_signals' in signals
        assert 'neutral_signals' in signals
        
        assert isinstance(signals['buy_signals'], list)
        assert isinstance(signals['sell_signals'], list)
        assert isinstance(signals['neutral_signals'], list)
    
    def test_comprehensive_analysis(self, analyzer, sample_data):
        """包括的分析テスト"""
        # 全ての分析を実行
        trend_result = analyzer.analyze_trends(sample_data)
        momentum_result = analyzer.analyze_momentum(sample_data)
        volatility_result = analyzer.analyze_volatility(sample_data)
        signals = analyzer.detect_signals(sample_data)
        
        # 結果の整合性チェック
        assert trend_result is not None
        assert momentum_result is not None
        assert volatility_result is not None
        assert signals is not None
        
        # シグナル数の妥当性
        total_signals = (len(signals['buy_signals']) + 
                        len(signals['sell_signals']) + 
                        len(signals['neutral_signals']))
        assert total_signals >= 0


class TestTechnicalAnalysisIntegration:
    """テクニカル分析統合テスト"""
    
    def test_full_analysis_workflow(self):
        """完全分析ワークフローテスト"""
        # データ準備
        data = create_sample_data(200)  # より多くのデータ
        analyzer = MockTechnicalAnalyzer()
        
        # 分析実行
        analysis_results = {
            'trend': analyzer.analyze_trends(data),
            'momentum': analyzer.analyze_momentum(data),
            'volatility': analyzer.analyze_volatility(data),
            'signals': analyzer.detect_signals(data)
        }
        
        # 結果の検証
        assert all(key in analysis_results for key in 
                  ['trend', 'momentum', 'volatility', 'signals'])
        
        # 各分析結果が適切な構造を持つ
        trend = analysis_results['trend']
        assert 'trend_direction' in trend
        
        momentum = analysis_results['momentum']
        assert 'rsi' in momentum
        
        volatility = analysis_results['volatility']
        assert 'atr' in volatility
        
        signals = analysis_results['signals']
        assert 'buy_signals' in signals
    
    def test_different_timeframes(self):
        """異なる時間軸テスト"""
        analyzer = MockTechnicalAnalyzer()
        
        # 短期データ
        short_data = create_sample_data(30)
        short_analysis = analyzer.analyze_trends(short_data)
        
        # 長期データ
        long_data = create_sample_data(300)
        long_analysis = analyzer.analyze_trends(long_data)
        
        # 両方とも有効な結果
        assert short_analysis['trend_direction'] in ['上昇', '下降']
        assert long_analysis['trend_direction'] in ['上昇', '下降']
    
    def test_edge_cases(self):
        """エッジケーステスト"""
        analyzer = MockTechnicalAnalyzer()
        
        # 最小限のデータ
        minimal_data = create_sample_data(30)  # 30日分
        
        try:
            result = analyzer.analyze_trends(minimal_data)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Minimal data analysis failed: {e}")
        
        # 平坦なデータ（価格変動なし）
        flat_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'open': [100] * 50,
            'high': [100] * 50,
            'low': [100] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50
        })
        
        try:
            result = analyzer.analyze_trends(flat_data)
            assert result is not None
            assert result['trend_strength'] == 0.0
        except Exception as e:
            pytest.fail(f"Flat data analysis failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])