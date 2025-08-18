#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Technical Analysis System - 高度技術分析システム

Issue #789実装：高度技術指標・分析手法拡張
最新の技術分析手法とカスタム指標による高精度分析
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 数値計算・統計
from scipy import stats
from scipy.signal import argrelextrema
import ta

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

class AnalysisType(Enum):
    """分析タイプ"""
    TREND_ANALYSIS = "trend_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    VOLUME_ANALYSIS = "volume_analysis"
    VOLATILITY_ANALYSIS = "volatility_analysis"
    MOMENTUM_ANALYSIS = "momentum_analysis"
    CYCLE_ANALYSIS = "cycle_analysis"
    FRACTAL_ANALYSIS = "fractal_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

@dataclass
class TechnicalSignal:
    """技術シグナル"""
    indicator_name: str
    signal_type: str  # "BUY", "SELL", "HOLD"
    strength: float  # 0-100
    confidence: float  # 0-1
    timeframe: str
    description: str
    timestamp: datetime

@dataclass
class PatternMatch:
    """パターンマッチ結果"""
    pattern_name: str
    match_score: float
    start_index: int
    end_index: int
    pattern_type: str  # "continuation", "reversal"
    reliability: float
    target_price: Optional[float] = None

@dataclass
class TechnicalAnalysisResult:
    """技術分析結果"""
    symbol: str
    analysis_type: AnalysisType
    signals: List[TechnicalSignal]
    patterns: List[PatternMatch]
    indicators: Dict[str, float]
    overall_sentiment: str
    confidence_score: float
    risk_level: str
    recommendations: List[str]
    timestamp: datetime

class AdvancedTechnicalIndicators:
    """高度技術指標群"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_advanced_momentum(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """高度モメンタム指標"""

        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']

        indicators = {}

        try:
            # 1. Awesome Oscillator
            median_price = (high + low) / 2
            ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
            indicators['awesome_oscillator'] = ao

            # 2. Accelerator Oscillator
            ac = ao - ao.rolling(5).mean()
            indicators['accelerator_oscillator'] = ac

            # 3. Balance of Power
            bop = (close - low) - (high - close) / (high - low + 1e-10)
            indicators['balance_of_power'] = bop

            # 4. Chande Momentum Oscillator
            n = 14
            mom = close.diff()
            up = mom.where(mom > 0, 0).rolling(n).sum()
            down = mom.where(mom < 0, 0).abs().rolling(n).sum()
            cmo = 100 * (up - down) / (up + down + 1e-10)
            indicators['chande_momentum'] = cmo

            # 5. Connors RSI
            rsi = ta.momentum.rsi(close, window=3)
            up_down_length = self._calculate_updown_length(close)
            percent_rank = self._percent_rank(close.pct_change(), 100)
            crsi = (rsi + up_down_length + percent_rank) / 3
            indicators['connors_rsi'] = crsi

            # 6. Klinger Volume Oscillator
            kvo = self._klinger_volume_oscillator(high, low, close, volume)
            indicators['klinger_volume'] = kvo

            # 7. Money Flow Index
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
            indicators['money_flow_index'] = mfi

            # 8. Relative Vigor Index
            rvi = self._relative_vigor_index(data)
            indicators['relative_vigor_index'] = rvi

        except Exception as e:
            self.logger.error(f"高度モメンタム計算エラー: {e}")

        return indicators

    def calculate_advanced_trend(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """高度トレンド指標"""

        high = data['High']
        low = data['Low']
        close = data['Close']

        indicators = {}

        try:
            # 1. Parabolic SAR
            psar = ta.trend.psar(high, low, close)
            indicators['parabolic_sar'] = psar

            # 2. Directional Movement Index (DMI)
            dmi_pos = ta.trend.adx_pos(high, low, close, window=14)
            dmi_neg = ta.trend.adx_neg(high, low, close, window=14)
            adx = ta.trend.adx(high, low, close, window=14)
            indicators['dmi_positive'] = dmi_pos
            indicators['dmi_negative'] = dmi_neg
            indicators['adx'] = adx

            # 3. Vortex Indicator
            vm_pos, vm_neg = self._vortex_indicator(high, low, close)
            indicators['vortex_positive'] = vm_pos
            indicators['vortex_negative'] = vm_neg

            # 4. Mass Index
            mass_index = self._mass_index(high, low)
            indicators['mass_index'] = mass_index

            # 5. Commodity Channel Index
            cci = ta.trend.cci(high, low, close, window=20)
            indicators['cci'] = cci

            # 6. Detrended Price Oscillator
            dpo = self._detrended_price_oscillator(close)
            indicators['detrended_price'] = dpo

            # 7. Aroon
            aroon_up = ta.trend.aroon_up(close, window=14)
            aroon_down = ta.trend.aroon_down(close, window=14)
            indicators['aroon_up'] = aroon_up
            indicators['aroon_down'] = aroon_down

            # 8. TRIX
            trix = ta.trend.trix(close, window=14)
            indicators['trix'] = trix

        except Exception as e:
            self.logger.error(f"高度トレンド計算エラー: {e}")

        return indicators

    def calculate_advanced_volatility(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """高度ボラティリティ指標"""

        high = data['High']
        low = data['Low']
        close = data['Close']

        indicators = {}

        try:
            # 1. Average True Range variations
            atr = ta.volatility.average_true_range(high, low, close, window=14)
            indicators['atr'] = atr

            # 2. Bollinger Bands
            bb_upper = ta.volatility.bollinger_hband(close, window=20, window_dev=2)
            bb_lower = ta.volatility.bollinger_lband(close, window=20, window_dev=2)
            bb_width = (bb_upper - bb_lower) / close
            bb_percent = (close - bb_lower) / (bb_upper - bb_lower)
            indicators['bollinger_width'] = bb_width
            indicators['bollinger_percent'] = bb_percent

            # 3. Keltner Channels
            kc_upper = ta.volatility.keltner_channel_hband(high, low, close)
            kc_lower = ta.volatility.keltner_channel_lband(high, low, close)
            indicators['keltner_upper'] = kc_upper
            indicators['keltner_lower'] = kc_lower

            # 4. Donchian Channels
            dc_upper = ta.volatility.donchian_channel_hband(high, low, close)
            dc_lower = ta.volatility.donchian_channel_lband(high, low, close)
            indicators['donchian_upper'] = dc_upper
            indicators['donchian_lower'] = dc_lower

            # 5. Ulcer Index
            ulcer = self._ulcer_index(close)
            indicators['ulcer_index'] = ulcer

            # 6. Historical Volatility
            returns = close.pct_change()
            hv_10 = returns.rolling(10).std() * np.sqrt(252)
            hv_30 = returns.rolling(30).std() * np.sqrt(252)
            indicators['historical_vol_10'] = hv_10
            indicators['historical_vol_30'] = hv_30

            # 7. Volatility Ratio
            vol_ratio = hv_10 / (hv_30 + 1e-10)
            indicators['volatility_ratio'] = vol_ratio

        except Exception as e:
            self.logger.error(f"高度ボラティリティ計算エラー: {e}")

        return indicators

    def calculate_advanced_volume(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """高度ボリューム指標"""

        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']

        indicators = {}

        try:
            # 1. Accumulation/Distribution Line
            ad_line = ta.volume.acc_dist_index(high, low, close, volume)
            indicators['ad_line'] = ad_line

            # 2. Chaikin Money Flow
            cmf = ta.volume.chaikin_money_flow(high, low, close, volume)
            indicators['chaikin_money_flow'] = cmf

            # 3. Force Index
            fi = ta.volume.force_index(close, volume)
            indicators['force_index'] = fi

            # 4. Negative Volume Index
            nvi = ta.volume.negative_volume_index(close, volume)
            indicators['negative_volume_index'] = nvi

            # 5. Volume Price Trend
            vpt = ta.volume.volume_price_trend(close, volume)
            indicators['volume_price_trend'] = vpt

            # 6. Volume Weighted Average Price (VWAP)
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            indicators['vwap'] = vwap

            # 7. Price Volume Trend
            pvt = ((close - close.shift(1)) / close.shift(1) * volume).cumsum()
            indicators['price_volume_trend'] = pvt

            # 8. Ease of Movement
            eom = self._ease_of_movement(high, low, volume)
            indicators['ease_of_movement'] = eom

            # 9. Volume Oscillator
            vol_osc = (volume.rolling(5).mean() - volume.rolling(10).mean()) / volume.rolling(10).mean() * 100
            indicators['volume_oscillator'] = vol_osc

        except Exception as e:
            self.logger.error(f"高度ボリューム計算エラー: {e}")

        return indicators

    def _calculate_updown_length(self, close: pd.Series) -> pd.Series:
        """上昇下降連続日数"""
        direction = np.sign(close.diff())
        groups = (direction != direction.shift()).cumsum()
        return direction.groupby(groups).cumcount() + 1

    def _percent_rank(self, series: pd.Series, window: int) -> pd.Series:
        """パーセントランク"""
        return series.rolling(window).rank(pct=True) * 100

    def _klinger_volume_oscillator(self, high: pd.Series, low: pd.Series,
                                 close: pd.Series, volume: pd.Series) -> pd.Series:
        """Klinger Volume Oscillator"""
        hlc = (high + low + close) / 3
        dm = high - low
        trend = np.where(hlc > hlc.shift(1), 1, -1)
        kvo = (trend * volume * dm).rolling(34).mean() - (trend * volume * dm).rolling(55).mean()
        return kvo

    def _relative_vigor_index(self, data: pd.DataFrame) -> pd.Series:
        """Relative Vigor Index"""
        close = data['Close']
        open_price = data['Open']
        high = data['High']
        low = data['Low']

        numerator = (close - open_price).rolling(4).mean()
        denominator = (high - low).rolling(4).mean()
        rvi = numerator / (denominator + 1e-10)
        return rvi

    def _vortex_indicator(self, high: pd.Series, low: pd.Series,
                        close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Vortex Indicator"""
        period = 14
        vm_pos = abs(high - low.shift(1)).rolling(period).sum()
        vm_neg = abs(low - high.shift(1)).rolling(period).sum()
        true_range = np.maximum(high - low,
                               np.maximum(abs(high - close.shift(1)),
                                        abs(low - close.shift(1))))
        vi_pos = vm_pos / true_range.rolling(period).sum()
        vi_neg = vm_neg / true_range.rolling(period).sum()
        return vi_pos, vi_neg

    def _mass_index(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """Mass Index"""
        hl_ratio = (high - low) / ((high + low) / 2 + 1e-10)
        ema_9 = hl_ratio.ewm(span=9).mean()
        ema_9_of_ema_9 = ema_9.ewm(span=9).mean()
        mass_index = (ema_9 / ema_9_of_ema_9).rolling(25).sum()
        return mass_index

    def _detrended_price_oscillator(self, close: pd.Series) -> pd.Series:
        """Detrended Price Oscillator"""
        period = 20
        sma = close.rolling(period).mean()
        dpo = close - sma.shift(period // 2 + 1)
        return dpo

    def _ulcer_index(self, close: pd.Series) -> pd.Series:
        """Ulcer Index"""
        period = 14
        max_close = close.rolling(period).max()
        drawdown = 100 * (close - max_close) / max_close
        ulcer = np.sqrt((drawdown ** 2).rolling(period).mean())
        return ulcer

    def _ease_of_movement(self, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """Ease of Movement"""
        distance = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        box_height = volume / (high - low + 1e-10)
        eom = distance / box_height
        return eom.rolling(14).mean()

class PatternRecognition:
    """パターン認識システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_candlestick_patterns(self, data: pd.DataFrame) -> List[PatternMatch]:
        """ローソク足パターン検出"""

        patterns = []

        try:
            open_price = data['Open']
            high = data['High']
            low = data['Low']
            close = data['Close']

            # 1. Doji
            doji_mask = abs(close - open_price) <= (high - low) * 0.1
            patterns.extend(self._create_pattern_matches("Doji", doji_mask, "reversal", 0.7))

            # 2. Hammer
            body = abs(close - open_price)
            upper_shadow = high - np.maximum(close, open_price)
            lower_shadow = np.minimum(close, open_price) - low
            hammer_mask = (lower_shadow > 2 * body) & (upper_shadow < body)
            patterns.extend(self._create_pattern_matches("Hammer", hammer_mask, "reversal", 0.8))

            # 3. Engulfing
            bullish_engulf = (close > open_price) & (close.shift(1) < open_price.shift(1)) & \
                           (close > open_price.shift(1)) & (open_price < close.shift(1))
            bearish_engulf = (close < open_price) & (close.shift(1) > open_price.shift(1)) & \
                           (close < open_price.shift(1)) & (open_price > close.shift(1))

            patterns.extend(self._create_pattern_matches("Bullish Engulfing", bullish_engulf, "reversal", 0.85))
            patterns.extend(self._create_pattern_matches("Bearish Engulfing", bearish_engulf, "reversal", 0.85))

            # 4. Morning Star / Evening Star
            morning_star = self._detect_morning_star(open_price, high, low, close)
            evening_star = self._detect_evening_star(open_price, high, low, close)

            patterns.extend(self._create_pattern_matches("Morning Star", morning_star, "reversal", 0.9))
            patterns.extend(self._create_pattern_matches("Evening Star", evening_star, "reversal", 0.9))

        except Exception as e:
            self.logger.error(f"ローソク足パターン検出エラー: {e}")

        return patterns

    def detect_price_patterns(self, data: pd.DataFrame) -> List[PatternMatch]:
        """価格パターン検出"""

        patterns = []

        try:
            close = data['Close']
            high = data['High']
            low = data['Low']

            # 1. Head and Shoulders
            patterns.extend(self._detect_head_shoulders(high, low, close))

            # 2. Double Top/Bottom
            patterns.extend(self._detect_double_top_bottom(high, low, close))

            # 3. Triangle Patterns
            patterns.extend(self._detect_triangles(high, low, close))

            # 4. Flag and Pennant
            patterns.extend(self._detect_flag_pennant(high, low, close))

            # 5. Cup and Handle
            patterns.extend(self._detect_cup_handle(close))

        except Exception as e:
            self.logger.error(f"価格パターン検出エラー: {e}")

        return patterns

    def _create_pattern_matches(self, pattern_name: str, mask: pd.Series,
                              pattern_type: str, reliability: float) -> List[PatternMatch]:
        """パターンマッチ作成"""

        matches = []
        indices = mask[mask].index

        for idx in indices:
            match = PatternMatch(
                pattern_name=pattern_name,
                match_score=reliability * 100,
                start_index=idx,
                end_index=idx,
                pattern_type=pattern_type,
                reliability=reliability
            )
            matches.append(match)

        return matches

    def _detect_morning_star(self, open_price: pd.Series, high: pd.Series,
                           low: pd.Series, close: pd.Series) -> pd.Series:
        """明けの明星パターン"""

        # 3日間のパターン
        day1_bear = close.shift(2) < open_price.shift(2)  # 1日目：陰線
        day2_small = abs(close.shift(1) - open_price.shift(1)) < abs(close.shift(2) - open_price.shift(2)) * 0.3  # 2日目：小さい実体
        day3_bull = close > open_price  # 3日目：陽線
        day3_recovery = close > (open_price.shift(2) + close.shift(2)) / 2  # 3日目：1日目の半分以上回復

        return day1_bear & day2_small & day3_bull & day3_recovery

    def _detect_evening_star(self, open_price: pd.Series, high: pd.Series,
                           low: pd.Series, close: pd.Series) -> pd.Series:
        """宵の明星パターン"""

        # 3日間のパターン
        day1_bull = close.shift(2) > open_price.shift(2)  # 1日目：陽線
        day2_small = abs(close.shift(1) - open_price.shift(1)) < abs(close.shift(2) - open_price.shift(2)) * 0.3  # 2日目：小さい実体
        day3_bear = close < open_price  # 3日目：陰線
        day3_decline = close < (open_price.shift(2) + close.shift(2)) / 2  # 3日目：1日目の半分以下に下落

        return day1_bull & day2_small & day3_bear & day3_decline

    def _detect_head_shoulders(self, high: pd.Series, low: pd.Series, close: pd.Series) -> List[PatternMatch]:
        """ヘッド・アンド・ショルダーズパターン"""
        # 簡単な実装（実際はより複雑な判定が必要）
        return []

    def _detect_double_top_bottom(self, high: pd.Series, low: pd.Series, close: pd.Series) -> List[PatternMatch]:
        """ダブルトップ・ボトムパターン"""
        # 簡単な実装
        return []

    def _detect_triangles(self, high: pd.Series, low: pd.Series, close: pd.Series) -> List[PatternMatch]:
        """三角形パターン"""
        # 簡単な実装
        return []

    def _detect_flag_pennant(self, high: pd.Series, low: pd.Series, close: pd.Series) -> List[PatternMatch]:
        """フラッグ・ペナントパターン"""
        # 簡単な実装
        return []

    def _detect_cup_handle(self, close: pd.Series) -> List[PatternMatch]:
        """カップ・アンド・ハンドルパターン"""
        # 簡単な実装
        return []

class AdvancedTechnicalAnalysis:
    """高度技術分析システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # コンポーネント初期化
        self.indicators = AdvancedTechnicalIndicators()
        self.pattern_recognition = PatternRecognition()

        # データベース設定
        self.db_path = Path("technical_analysis_data/advanced_analysis.db")
        self.db_path.parent.mkdir(exist_ok=True)

        self._init_database()
        self.logger.info("Advanced technical analysis system initialized")

    def _init_database(self):
        """データベース初期化"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 技術分析結果テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS technical_analysis_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        analysis_type TEXT NOT NULL,
                        signals TEXT,
                        patterns TEXT,
                        indicators TEXT,
                        overall_sentiment TEXT,
                        confidence_score REAL,
                        risk_level TEXT,
                        recommendations TEXT,
                        timestamp TEXT NOT NULL
                    )
                ''')

                # シグナル履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signal_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        indicator_name TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        strength REAL,
                        confidence REAL,
                        timeframe TEXT,
                        description TEXT,
                        timestamp TEXT NOT NULL
                    )
                ''')

                conn.commit()

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")

    async def perform_comprehensive_analysis(self, symbol: str, period: str = "3mo") -> TechnicalAnalysisResult:
        """包括的技術分析実行"""

        self.logger.info(f"包括的技術分析開始: {symbol}")

        try:
            # データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, period)

            if data is None or len(data) < 50:
                raise ValueError("分析に十分なデータがありません")

            # 各種分析実行
            all_signals = []
            all_patterns = []
            all_indicators = {}

            # 1. 高度モメンタム分析
            momentum_indicators = self.indicators.calculate_advanced_momentum(data)
            momentum_signals = self._analyze_momentum_signals(momentum_indicators, data)
            all_signals.extend(momentum_signals)
            all_indicators.update({k: v.iloc[-1] if len(v) > 0 else 0 for k, v in momentum_indicators.items()})

            # 2. 高度トレンド分析
            trend_indicators = self.indicators.calculate_advanced_trend(data)
            trend_signals = self._analyze_trend_signals(trend_indicators, data)
            all_signals.extend(trend_signals)
            all_indicators.update({k: v.iloc[-1] if len(v) > 0 else 0 for k, v in trend_indicators.items()})

            # 3. 高度ボラティリティ分析
            volatility_indicators = self.indicators.calculate_advanced_volatility(data)
            volatility_signals = self._analyze_volatility_signals(volatility_indicators, data)
            all_signals.extend(volatility_signals)
            all_indicators.update({k: v.iloc[-1] if len(v) > 0 else 0 for k, v in volatility_indicators.items()})

            # 4. 高度ボリューム分析
            volume_indicators = self.indicators.calculate_advanced_volume(data)
            volume_signals = self._analyze_volume_signals(volume_indicators, data)
            all_signals.extend(volume_signals)
            all_indicators.update({k: v.iloc[-1] if len(v) > 0 else 0 for k, v in volume_indicators.items()})

            # 5. パターン認識
            candlestick_patterns = self.pattern_recognition.detect_candlestick_patterns(data)
            price_patterns = self.pattern_recognition.detect_price_patterns(data)
            all_patterns.extend(candlestick_patterns)
            all_patterns.extend(price_patterns)

            # 6. 総合判定
            overall_sentiment, confidence_score = self._calculate_overall_sentiment(all_signals)
            risk_level = self._assess_risk_level(all_indicators, volatility_indicators)
            recommendations = self._generate_recommendations(all_signals, all_patterns, overall_sentiment)

            # 結果作成
            result = TechnicalAnalysisResult(
                symbol=symbol,
                analysis_type=AnalysisType.TREND_ANALYSIS,  # メインタイプ
                signals=all_signals,
                patterns=all_patterns,
                indicators=all_indicators,
                overall_sentiment=overall_sentiment,
                confidence_score=confidence_score,
                risk_level=risk_level,
                recommendations=recommendations,
                timestamp=datetime.now()
            )

            # 結果保存
            await self._save_analysis_result(result)

            self.logger.info(f"包括的技術分析完了: {len(all_signals)}シグナル, {len(all_patterns)}パターン")

            return result

        except Exception as e:
            self.logger.error(f"包括的技術分析エラー: {e}")
            raise

    def _analyze_momentum_signals(self, indicators: Dict[str, pd.Series], data: pd.DataFrame) -> List[TechnicalSignal]:
        """モメンタムシグナル分析"""

        signals = []
        current_price = data['Close'].iloc[-1]

        try:
            # RSI系シグナル
            if 'connors_rsi' in indicators:
                crsi = indicators['connors_rsi'].iloc[-1]
                if not np.isnan(crsi):
                    if crsi < 20:
                        signals.append(TechnicalSignal(
                            indicator_name="Connors RSI",
                            signal_type="BUY",
                            strength=80,
                            confidence=0.8,
                            timeframe="short",
                            description=f"Connors RSI過売り水準 ({crsi:.1f})",
                            timestamp=datetime.now()
                        ))
                    elif crsi > 80:
                        signals.append(TechnicalSignal(
                            indicator_name="Connors RSI",
                            signal_type="SELL",
                            strength=80,
                            confidence=0.8,
                            timeframe="short",
                            description=f"Connors RSI過買い水準 ({crsi:.1f})",
                            timestamp=datetime.now()
                        ))

            # Money Flow Index
            if 'money_flow_index' in indicators:
                mfi = indicators['money_flow_index'].iloc[-1]
                if not np.isnan(mfi):
                    if mfi < 20:
                        signals.append(TechnicalSignal(
                            indicator_name="Money Flow Index",
                            signal_type="BUY",
                            strength=70,
                            confidence=0.7,
                            timeframe="medium",
                            description=f"MFI過売り水準 ({mfi:.1f})",
                            timestamp=datetime.now()
                        ))
                    elif mfi > 80:
                        signals.append(TechnicalSignal(
                            indicator_name="Money Flow Index",
                            signal_type="SELL",
                            strength=70,
                            confidence=0.7,
                            timeframe="medium",
                            description=f"MFI過買い水準 ({mfi:.1f})",
                            timestamp=datetime.now()
                        ))

            # Awesome Oscillator
            if 'awesome_oscillator' in indicators and len(indicators['awesome_oscillator']) > 1:
                ao_current = indicators['awesome_oscillator'].iloc[-1]
                ao_prev = indicators['awesome_oscillator'].iloc[-2]
                if not np.isnan(ao_current) and not np.isnan(ao_prev):
                    if ao_current > 0 and ao_prev <= 0:
                        signals.append(TechnicalSignal(
                            indicator_name="Awesome Oscillator",
                            signal_type="BUY",
                            strength=60,
                            confidence=0.6,
                            timeframe="medium",
                            description="AO ゼロライン上抜け",
                            timestamp=datetime.now()
                        ))
                    elif ao_current < 0 and ao_prev >= 0:
                        signals.append(TechnicalSignal(
                            indicator_name="Awesome Oscillator",
                            signal_type="SELL",
                            strength=60,
                            confidence=0.6,
                            timeframe="medium",
                            description="AO ゼロライン下抜け",
                            timestamp=datetime.now()
                        ))

        except Exception as e:
            self.logger.error(f"モメンタムシグナル分析エラー: {e}")

        return signals

    def _analyze_trend_signals(self, indicators: Dict[str, pd.Series], data: pd.DataFrame) -> List[TechnicalSignal]:
        """トレンドシグナル分析"""

        signals = []
        current_price = data['Close'].iloc[-1]

        try:
            # Parabolic SAR
            if 'parabolic_sar' in indicators:
                psar = indicators['parabolic_sar'].iloc[-1]
                if not np.isnan(psar):
                    if current_price > psar:
                        signals.append(TechnicalSignal(
                            indicator_name="Parabolic SAR",
                            signal_type="BUY",
                            strength=70,
                            confidence=0.7,
                            timeframe="medium",
                            description="価格がSAR上位",
                            timestamp=datetime.now()
                        ))
                    else:
                        signals.append(TechnicalSignal(
                            indicator_name="Parabolic SAR",
                            signal_type="SELL",
                            strength=70,
                            confidence=0.7,
                            timeframe="medium",
                            description="価格がSAR下位",
                            timestamp=datetime.now()
                        ))

            # ADX
            if all(k in indicators for k in ['adx', 'dmi_positive', 'dmi_negative']):
                adx = indicators['adx'].iloc[-1]
                dmi_pos = indicators['dmi_positive'].iloc[-1]
                dmi_neg = indicators['dmi_negative'].iloc[-1]

                if not any(np.isnan([adx, dmi_pos, dmi_neg])):
                    if adx > 25:  # 強いトレンド
                        if dmi_pos > dmi_neg:
                            signals.append(TechnicalSignal(
                                indicator_name="ADX/DMI",
                                signal_type="BUY",
                                strength=75,
                                confidence=0.75,
                                timeframe="medium",
                                description=f"強い上昇トレンド (ADX:{adx:.1f})",
                                timestamp=datetime.now()
                            ))
                        else:
                            signals.append(TechnicalSignal(
                                indicator_name="ADX/DMI",
                                signal_type="SELL",
                                strength=75,
                                confidence=0.75,
                                timeframe="medium",
                                description=f"強い下降トレンド (ADX:{adx:.1f})",
                                timestamp=datetime.now()
                            ))

            # CCI
            if 'cci' in indicators:
                cci = indicators['cci'].iloc[-1]
                if not np.isnan(cci):
                    if cci > 100:
                        signals.append(TechnicalSignal(
                            indicator_name="CCI",
                            signal_type="SELL",
                            strength=65,
                            confidence=0.65,
                            timeframe="short",
                            description=f"CCI過買い水準 ({cci:.1f})",
                            timestamp=datetime.now()
                        ))
                    elif cci < -100:
                        signals.append(TechnicalSignal(
                            indicator_name="CCI",
                            signal_type="BUY",
                            strength=65,
                            confidence=0.65,
                            timeframe="short",
                            description=f"CCI過売り水準 ({cci:.1f})",
                            timestamp=datetime.now()
                        ))

        except Exception as e:
            self.logger.error(f"トレンドシグナル分析エラー: {e}")

        return signals

    def _analyze_volatility_signals(self, indicators: Dict[str, pd.Series], data: pd.DataFrame) -> List[TechnicalSignal]:
        """ボラティリティシグナル分析"""

        signals = []

        try:
            # Bollinger Bands
            if 'bollinger_percent' in indicators:
                bb_percent = indicators['bollinger_percent'].iloc[-1]
                if not np.isnan(bb_percent):
                    if bb_percent > 0.95:
                        signals.append(TechnicalSignal(
                            indicator_name="Bollinger Bands",
                            signal_type="SELL",
                            strength=60,
                            confidence=0.6,
                            timeframe="short",
                            description="ボリンジャーバンド上限近く",
                            timestamp=datetime.now()
                        ))
                    elif bb_percent < 0.05:
                        signals.append(TechnicalSignal(
                            indicator_name="Bollinger Bands",
                            signal_type="BUY",
                            strength=60,
                            confidence=0.6,
                            timeframe="short",
                            description="ボリンジャーバンド下限近く",
                            timestamp=datetime.now()
                        ))

            # Volatility Ratio
            if 'volatility_ratio' in indicators:
                vol_ratio = indicators['volatility_ratio'].iloc[-1]
                if not np.isnan(vol_ratio):
                    if vol_ratio > 1.5:
                        signals.append(TechnicalSignal(
                            indicator_name="Volatility Ratio",
                            signal_type="HOLD",
                            strength=40,
                            confidence=0.5,
                            timeframe="short",
                            description="高ボラティリティ環境",
                            timestamp=datetime.now()
                        ))

        except Exception as e:
            self.logger.error(f"ボラティリティシグナル分析エラー: {e}")

        return signals

    def _analyze_volume_signals(self, indicators: Dict[str, pd.Series], data: pd.DataFrame) -> List[TechnicalSignal]:
        """ボリュームシグナル分析"""

        signals = []

        try:
            # Chaikin Money Flow
            if 'chaikin_money_flow' in indicators:
                cmf = indicators['chaikin_money_flow'].iloc[-1]
                if not np.isnan(cmf):
                    if cmf > 0.2:
                        signals.append(TechnicalSignal(
                            indicator_name="Chaikin Money Flow",
                            signal_type="BUY",
                            strength=65,
                            confidence=0.65,
                            timeframe="medium",
                            description=f"強い買い圧力 (CMF:{cmf:.2f})",
                            timestamp=datetime.now()
                        ))
                    elif cmf < -0.2:
                        signals.append(TechnicalSignal(
                            indicator_name="Chaikin Money Flow",
                            signal_type="SELL",
                            strength=65,
                            confidence=0.65,
                            timeframe="medium",
                            description=f"強い売り圧力 (CMF:{cmf:.2f})",
                            timestamp=datetime.now()
                        ))

            # Volume Oscillator
            if 'volume_oscillator' in indicators:
                vol_osc = indicators['volume_oscillator'].iloc[-1]
                if not np.isnan(vol_osc):
                    if vol_osc > 10:
                        signals.append(TechnicalSignal(
                            indicator_name="Volume Oscillator",
                            signal_type="BUY",
                            strength=50,
                            confidence=0.5,
                            timeframe="short",
                            description="ボリューム増加トレンド",
                            timestamp=datetime.now()
                        ))

        except Exception as e:
            self.logger.error(f"ボリュームシグナル分析エラー: {e}")

        return signals

    def _calculate_overall_sentiment(self, signals: List[TechnicalSignal]) -> Tuple[str, float]:
        """総合センチメント計算"""

        if not signals:
            return "NEUTRAL", 0.5

        buy_score = sum(s.strength * s.confidence for s in signals if s.signal_type == "BUY")
        sell_score = sum(s.strength * s.confidence for s in signals if s.signal_type == "SELL")
        total_signals = len([s for s in signals if s.signal_type in ["BUY", "SELL"]])

        if total_signals == 0:
            return "NEUTRAL", 0.5

        net_score = (buy_score - sell_score) / (buy_score + sell_score + 1e-10)
        confidence = min(1.0, (buy_score + sell_score) / (total_signals * 100))

        if net_score > 0.3:
            sentiment = "BULLISH"
        elif net_score < -0.3:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        return sentiment, confidence

    def _assess_risk_level(self, indicators: Dict[str, float], volatility_indicators: Dict[str, pd.Series]) -> str:
        """リスクレベル評価"""

        risk_factors = []

        # ボラティリティリスク
        if 'historical_vol_10' in indicators:
            vol = indicators['historical_vol_10']
            if vol > 0.3:  # 30%以上の年率ボラティリティ
                risk_factors.append("高ボラティリティ")

        # トレンド強度
        if 'adx' in indicators:
            adx = indicators['adx']
            if adx < 20:
                risk_factors.append("弱いトレンド")

        risk_score = len(risk_factors)

        if risk_score >= 2:
            return "HIGH"
        elif risk_score == 1:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_recommendations(self, signals: List[TechnicalSignal],
                                patterns: List[PatternMatch], sentiment: str) -> List[str]:
        """推奨事項生成"""

        recommendations = []

        # センチメントベース推奨
        if sentiment == "BULLISH":
            recommendations.append("技術指標は上昇トレンドを示唆")
            recommendations.append("押し目での買い機会を検討")
        elif sentiment == "BEARISH":
            recommendations.append("技術指標は下降トレンドを示唆")
            recommendations.append("戻り売り機会を検討")
        else:
            recommendations.append("中立的な市場環境")
            recommendations.append("明確なトレンド確認まで様子見推奨")

        # 強いシグナルから推奨
        strong_signals = [s for s in signals if s.strength > 70]
        if strong_signals:
            for signal in strong_signals[:3]:  # 上位3つ
                recommendations.append(f"{signal.indicator_name}: {signal.description}")

        # パターンから推奨
        reliable_patterns = [p for p in patterns if p.reliability > 0.8]
        if reliable_patterns:
            for pattern in reliable_patterns[:2]:  # 上位2つ
                recommendations.append(f"{pattern.pattern_name}パターン検出")

        return recommendations

    async def _save_analysis_result(self, result: TechnicalAnalysisResult):
        """分析結果保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO technical_analysis_results
                    (symbol, analysis_type, signals, patterns, indicators,
                     overall_sentiment, confidence_score, risk_level, recommendations, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.symbol,
                    result.analysis_type.value,
                    json.dumps([{
                        'indicator_name': s.indicator_name,
                        'signal_type': s.signal_type,
                        'strength': s.strength,
                        'confidence': s.confidence,
                        'timeframe': s.timeframe,
                        'description': s.description
                    } for s in result.signals]),
                    json.dumps([{
                        'pattern_name': p.pattern_name,
                        'match_score': p.match_score,
                        'pattern_type': p.pattern_type,
                        'reliability': p.reliability
                    } for p in result.patterns]),
                    json.dumps(result.indicators),
                    result.overall_sentiment,
                    result.confidence_score,
                    result.risk_level,
                    json.dumps(result.recommendations),
                    result.timestamp.isoformat()
                ))

                # 個別シグナル保存
                for signal in result.signals:
                    cursor.execute('''
                        INSERT INTO signal_history
                        (symbol, indicator_name, signal_type, strength, confidence,
                         timeframe, description, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        result.symbol,
                        signal.indicator_name,
                        signal.signal_type,
                        signal.strength,
                        signal.confidence,
                        signal.timeframe,
                        signal.description,
                        signal.timestamp.isoformat()
                    ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"分析結果保存エラー: {e}")

# グローバルインスタンス
advanced_technical_analysis = AdvancedTechnicalAnalysis()

# テスト実行
async def run_advanced_technical_analysis_test():
    """高度技術分析テスト実行"""

    print("=== 📈 高度技術分析システムテスト ===")

    test_symbols = ["7203", "8306"]

    for symbol in test_symbols:
        print(f"\n🔬 {symbol} 高度技術分析")

        try:
            # 包括的技術分析実行
            result = await advanced_technical_analysis.perform_comprehensive_analysis(symbol)

            print(f"  ✅ 分析完了:")
            print(f"    総合センチメント: {result.overall_sentiment}")
            print(f"    信頼度: {result.confidence_score:.1%}")
            print(f"    リスクレベル: {result.risk_level}")
            print(f"    検出シグナル数: {len(result.signals)}")
            print(f"    検出パターン数: {len(result.patterns)}")
            print(f"    計算指標数: {len(result.indicators)}")

            # 主要シグナル表示
            strong_signals = [s for s in result.signals if s.strength > 60]
            if strong_signals:
                print(f"    主要シグナル:")
                for signal in strong_signals[:5]:
                    print(f"      - {signal.indicator_name}: {signal.signal_type} ({signal.strength:.0f})")

            # パターン表示
            if result.patterns:
                print(f"    検出パターン:")
                for pattern in result.patterns[:3]:
                    print(f"      - {pattern.pattern_name} (信頼度: {pattern.reliability:.1%})")

            # 推奨事項表示
            if result.recommendations:
                print(f"    推奨事項:")
                for rec in result.recommendations[:3]:
                    print(f"      - {rec}")

        except Exception as e:
            print(f"  ❌ {symbol} エラー: {e}")

    print(f"\n✅ 高度技術分析システムテスト完了")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(run_advanced_technical_analysis_test())