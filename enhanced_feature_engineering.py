#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Feature Engineering - 強化特徴量エンジニアリング

RSI、MACD、ボリンジャーバンド等の高度技術指標による特徴量拡張
Issue #796-1実装：特徴量エンジニアリング強化
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json

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

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from real_data_provider_v2 import real_data_provider
    REAL_DATA_PROVIDER_AVAILABLE = True
except ImportError:
    REAL_DATA_PROVIDER_AVAILABLE = False

@dataclass
class FeatureSet:
    """特徴量セット"""
    symbol: str
    timestamp: datetime

    # 基本価格特徴量
    price_features: Dict[str, float]

    # 技術指標特徴量
    technical_features: Dict[str, float]

    # 出来高特徴量
    volume_features: Dict[str, float]

    # モメンタム特徴量
    momentum_features: Dict[str, float]

    # ボラティリティ特徴量
    volatility_features: Dict[str, float]

    # パターン特徴量
    pattern_features: Dict[str, float]

    # 市場環境特徴量
    market_features: Dict[str, float]

    # 統計特徴量
    statistical_features: Dict[str, float]

class EnhancedFeatureEngineering:
    """強化特徴量エンジニアリング"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 特徴量計算設定
        self.feature_config = {
            # 移動平均期間
            'sma_periods': [5, 10, 20, 50],
            'ema_periods': [12, 26],

            # RSI設定
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,

            # MACD設定
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,

            # ボリンジャーバンド設定
            'bb_period': 20,
            'bb_std': 2,

            # ATR設定
            'atr_period': 14,

            # ストキャスティクス設定
            'stoch_k_period': 14,
            'stoch_d_period': 3,

            # CCI設定
            'cci_period': 20,

            # Williams %R設定
            'williams_period': 14,

            # ROC設定
            'roc_period': 12,

            # 出来高分析設定
            'volume_sma_period': 20,
            'volume_ratio_periods': [5, 10, 20],

            # ボラティリティ分析設定
            'volatility_periods': [5, 10, 20],

            # フラクタル分析設定
            'fractal_periods': [5, 8, 13]
        }

        self.logger.info("Enhanced feature engineering initialized")

    async def extract_comprehensive_features(self, symbol: str, data: pd.DataFrame) -> FeatureSet:
        """包括的特徴量抽出"""

        self.logger.debug(f"Extracting comprehensive features for {symbol}")

        if data.empty or len(data) < 50:
            return self._create_empty_feature_set(symbol)

        # データクリーニング
        data = self._clean_data(data)

        # 各カテゴリの特徴量抽出
        price_features = self._extract_price_features(data)
        technical_features = self._extract_technical_features(data)
        volume_features = self._extract_volume_features(data)
        momentum_features = self._extract_momentum_features(data)
        volatility_features = self._extract_volatility_features(data)
        pattern_features = self._extract_pattern_features(data)
        market_features = self._extract_market_features(data)
        statistical_features = self._extract_statistical_features(data)

        return FeatureSet(
            symbol=symbol,
            timestamp=datetime.now(),
            price_features=price_features,
            technical_features=technical_features,
            volume_features=volume_features,
            momentum_features=momentum_features,
            volatility_features=volatility_features,
            pattern_features=pattern_features,
            market_features=market_features,
            statistical_features=statistical_features
        )

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """データクリーニング"""

        # 必要な列の存在確認
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in data.columns:
                self.logger.error(f"Required column {col} not found")
                return data

        # Volume列がない場合は作成
        if 'Volume' not in data.columns:
            data['Volume'] = 1000000  # デフォルト値

        # NaN値の処理
        data = data.fillna(method='ffill').fillna(method='bfill')

        # 異常値の処理
        for col in ['Open', 'High', 'Low', 'Close']:
            # ゼロ以下の値を前の値で置換
            data[col] = data[col].mask(data[col] <= 0).fillna(method='ffill')

        return data

    def _extract_price_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """価格関連特徴量"""

        features = {}

        try:
            # 現在価格
            current_price = float(data['Close'].iloc[-1])
            features['current_price'] = current_price

            # 価格変化率
            if len(data) >= 2:
                prev_price = float(data['Close'].iloc[-2])
                features['price_change_pct'] = (current_price - prev_price) / prev_price

            # OHLC関係
            features['hl_ratio'] = float(data['High'].iloc[-1] / data['Low'].iloc[-1])
            features['oc_ratio'] = float(data['Open'].iloc[-1] / data['Close'].iloc[-1])

            # 価格位置（High-Lowレンジ内での終値位置）
            high = float(data['High'].iloc[-1])
            low = float(data['Low'].iloc[-1])
            close = float(data['Close'].iloc[-1])
            if high != low:
                features['price_position'] = (close - low) / (high - low)
            else:
                features['price_position'] = 0.5

            # 各移動平均との乖離
            for period in self.feature_config['sma_periods']:
                if len(data) >= period:
                    sma = data['Close'].rolling(period).mean().iloc[-1]
                    features[f'sma_{period}_ratio'] = float(current_price / sma)

            # EMA乖離
            for period in self.feature_config['ema_periods']:
                if len(data) >= period:
                    ema = data['Close'].ewm(span=period).mean().iloc[-1]
                    features[f'ema_{period}_ratio'] = float(current_price / ema)

        except Exception as e:
            self.logger.error(f"Price feature extraction failed: {e}")

        return features

    def _extract_technical_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """技術指標特徴量"""

        features = {}

        try:
            # RSI
            rsi = self._calculate_rsi(data['Close'], self.feature_config['rsi_period'])
            if not rsi.empty:
                features['rsi'] = float(rsi.iloc[-1])
                features['rsi_overbought'] = 1 if rsi.iloc[-1] > self.feature_config['rsi_overbought'] else 0
                features['rsi_oversold'] = 1 if rsi.iloc[-1] < self.feature_config['rsi_oversold'] else 0

            # MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(
                data['Close'],
                self.feature_config['macd_fast'],
                self.feature_config['macd_slow'],
                self.feature_config['macd_signal']
            )

            if not macd_line.empty:
                features['macd_line'] = float(macd_line.iloc[-1])
                features['macd_signal'] = float(macd_signal.iloc[-1])
                features['macd_histogram'] = float(macd_histogram.iloc[-1])
                features['macd_bullish'] = 1 if macd_line.iloc[-1] > macd_signal.iloc[-1] else 0

            # ボリンジャーバンド
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
                data['Close'],
                self.feature_config['bb_period'],
                self.feature_config['bb_std']
            )

            if not bb_upper.empty:
                current_price = data['Close'].iloc[-1]
                features['bb_upper'] = float(bb_upper.iloc[-1])
                features['bb_middle'] = float(bb_middle.iloc[-1])
                features['bb_lower'] = float(bb_lower.iloc[-1])
                features['bb_position'] = float((current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]))
                features['bb_squeeze'] = float((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1])

            # ATR (Average True Range)
            atr = self._calculate_atr(data, self.feature_config['atr_period'])
            if not atr.empty:
                features['atr'] = float(atr.iloc[-1])
                features['atr_ratio'] = float(atr.iloc[-1] / data['Close'].iloc[-1])

            # ストキャスティクス
            stoch_k, stoch_d = self._calculate_stochastic(
                data,
                self.feature_config['stoch_k_period'],
                self.feature_config['stoch_d_period']
            )

            if not stoch_k.empty:
                features['stoch_k'] = float(stoch_k.iloc[-1])
                features['stoch_d'] = float(stoch_d.iloc[-1])
                features['stoch_overbought'] = 1 if stoch_k.iloc[-1] > 80 else 0
                features['stoch_oversold'] = 1 if stoch_k.iloc[-1] < 20 else 0

            # CCI (Commodity Channel Index)
            cci = self._calculate_cci(data, self.feature_config['cci_period'])
            if not cci.empty:
                features['cci'] = float(cci.iloc[-1])
                features['cci_overbought'] = 1 if cci.iloc[-1] > 100 else 0
                features['cci_oversold'] = 1 if cci.iloc[-1] < -100 else 0

            # Williams %R
            williams_r = self._calculate_williams_r(data, self.feature_config['williams_period'])
            if not williams_r.empty:
                features['williams_r'] = float(williams_r.iloc[-1])
                features['williams_overbought'] = 1 if williams_r.iloc[-1] > -20 else 0
                features['williams_oversold'] = 1 if williams_r.iloc[-1] < -80 else 0

        except Exception as e:
            self.logger.error(f"Technical feature extraction failed: {e}")

        return features

    def _extract_volume_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """出来高特徴量"""

        features = {}

        try:
            current_volume = float(data['Volume'].iloc[-1])
            features['current_volume'] = current_volume

            # 出来高移動平均比
            volume_sma = data['Volume'].rolling(self.feature_config['volume_sma_period']).mean()
            if not volume_sma.empty and volume_sma.iloc[-1] > 0:
                features['volume_sma_ratio'] = float(current_volume / volume_sma.iloc[-1])

            # 各期間での出来高比較
            for period in self.feature_config['volume_ratio_periods']:
                if len(data) >= period:
                    avg_volume = data['Volume'].rolling(period).mean().iloc[-1]
                    if avg_volume > 0:
                        features[f'volume_ratio_{period}'] = float(current_volume / avg_volume)

            # 出来高価格トレンド（OBV風）
            if len(data) >= 10:
                obv = self._calculate_obv(data)
                if not obv.empty:
                    features['obv_trend'] = float(obv.iloc[-1] - obv.iloc[-10])

            # 出来高加重平均価格（VWAP）
            if len(data) >= 20:
                vwap = self._calculate_vwap(data)
                if not vwap.empty and vwap.iloc[-1] > 0:
                    current_price = data['Close'].iloc[-1]
                    features['vwap_ratio'] = float(current_price / vwap.iloc[-1])

        except Exception as e:
            self.logger.error(f"Volume feature extraction failed: {e}")

        return features

    def _extract_momentum_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """モメンタム特徴量"""

        features = {}

        try:
            # Rate of Change (ROC)
            roc = self._calculate_roc(data['Close'], self.feature_config['roc_period'])
            if not roc.empty:
                features['roc'] = float(roc.iloc[-1])

            # 各期間のリターン
            returns_periods = [1, 3, 5, 10, 20]
            for period in returns_periods:
                if len(data) > period:
                    ret = (data['Close'].iloc[-1] / data['Close'].iloc[-1-period]) - 1
                    features[f'return_{period}d'] = float(ret)

            # モメンタム指標
            momentum_periods = [10, 20]
            for period in momentum_periods:
                if len(data) >= period:
                    momentum = data['Close'].iloc[-1] / data['Close'].iloc[-period]
                    features[f'momentum_{period}'] = float(momentum)

            # 価格加速度（2次微分）
            if len(data) >= 3:
                returns = data['Close'].pct_change()
                acceleration = returns.diff().iloc[-1]
                features['price_acceleration'] = float(acceleration) if not np.isnan(acceleration) else 0

        except Exception as e:
            self.logger.error(f"Momentum feature extraction failed: {e}")

        return features

    def _extract_volatility_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """ボラティリティ特徴量"""

        features = {}

        try:
            # 各期間のボラティリティ
            for period in self.feature_config['volatility_periods']:
                if len(data) >= period:
                    returns = data['Close'].pct_change()
                    volatility = returns.rolling(period).std().iloc[-1]
                    features[f'volatility_{period}d'] = float(volatility) if not np.isnan(volatility) else 0

            # True Range
            if len(data) >= 2:
                tr = self._calculate_true_range(data)
                if not tr.empty:
                    features['true_range'] = float(tr.iloc[-1])
                    features['true_range_ratio'] = float(tr.iloc[-1] / data['Close'].iloc[-1])

            # ボラティリティ比率（短期/長期）
            if len(data) >= 20:
                vol_short = data['Close'].pct_change().rolling(5).std().iloc[-1]
                vol_long = data['Close'].pct_change().rolling(20).std().iloc[-1]
                if vol_long > 0 and not np.isnan(vol_short):
                    features['volatility_ratio'] = float(vol_short / vol_long)

            # GARCH風ボラティリティ
            if len(data) >= 30:
                garch_vol = self._estimate_garch_volatility(data['Close'])
                features['garch_volatility'] = float(garch_vol)

        except Exception as e:
            self.logger.error(f"Volatility feature extraction failed: {e}")

        return features

    def _extract_pattern_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """パターン特徴量"""

        features = {}

        try:
            # トレンド検出
            if len(data) >= 20:
                trend_strength = self._calculate_trend_strength(data['Close'])
                features['trend_strength'] = float(trend_strength)

            # サポート・レジスタンス
            if len(data) >= 50:
                support, resistance = self._identify_support_resistance(data)
                current_price = data['Close'].iloc[-1]

                if support > 0:
                    features['support_distance'] = float((current_price - support) / support)
                if resistance > 0:
                    features['resistance_distance'] = float((resistance - current_price) / resistance)

            # フラクタル次元
            for period in self.feature_config['fractal_periods']:
                if len(data) >= period:
                    fractal_dim = self._calculate_fractal_dimension(data['Close'].tail(period))
                    features[f'fractal_dim_{period}'] = float(fractal_dim)

            # チャートパターン検出
            pattern_score = self._detect_chart_patterns(data)
            features.update(pattern_score)

        except Exception as e:
            self.logger.error(f"Pattern feature extraction failed: {e}")

        return features

    def _extract_market_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """市場環境特徴量"""

        features = {}

        try:
            # 市場時間特徴量
            current_time = datetime.now()
            features['hour'] = current_time.hour
            features['day_of_week'] = current_time.weekday()
            features['is_market_open'] = 1 if 9 <= current_time.hour <= 15 else 0

            # 月間効果
            features['month'] = current_time.month
            features['is_year_end'] = 1 if current_time.month == 12 else 0

            # 四半期効果
            quarter = (current_time.month - 1) // 3 + 1
            features['quarter'] = quarter

        except Exception as e:
            self.logger.error(f"Market feature extraction failed: {e}")

        return features

    def _extract_statistical_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """統計特徴量"""

        features = {}

        try:
            if SCIPY_AVAILABLE and len(data) >= 20:
                returns = data['Close'].pct_change().dropna()

                # 統計モーメント
                features['skewness'] = float(stats.skew(returns))
                features['kurtosis'] = float(stats.kurtosis(returns))

                # Jarque-Bera正規性テスト
                jb_stat, jb_pvalue = stats.jarque_bera(returns)
                features['jarque_bera_stat'] = float(jb_stat)
                features['is_normal_distribution'] = 1 if jb_pvalue > 0.05 else 0

                # Hurst指数（トレンド持続性）
                hurst = self._calculate_hurst_exponent(data['Close'])
                features['hurst_exponent'] = float(hurst)

                # エントロピー（ランダム性）
                entropy = self._calculate_entropy(returns)
                features['entropy'] = float(entropy)

        except Exception as e:
            self.logger.error(f"Statistical feature extraction failed: {e}")

        return features

    # 技術指標計算メソッド
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI計算"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series()

    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD計算"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            macd_histogram = macd_line - macd_signal
            return macd_line, macd_signal, macd_histogram
        except:
            return pd.Series(), pd.Series(), pd.Series()

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ボリンジャーバンド計算"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, sma, lower_band
        except:
            return pd.Series(), pd.Series(), pd.Series()

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """ATR計算"""
        try:
            tr = self._calculate_true_range(data)
            atr = tr.rolling(window=period).mean()
            return atr
        except:
            return pd.Series()

    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """True Range計算"""
        try:
            high_low = data['High'] - data['Low']
            high_close_prev = np.abs(data['High'] - data['Close'].shift(1))
            low_close_prev = np.abs(data['Low'] - data['Close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            return pd.Series(true_range, index=data.index)
        except:
            return pd.Series()

    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
        """ストキャスティクス計算"""
        try:
            low_min = data['Low'].rolling(window=k_period).min()
            high_max = data['High'].rolling(window=k_period).max()
            k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(window=d_period).mean()
            return k_percent, d_percent
        except:
            return pd.Series(), pd.Series()

    def _calculate_cci(self, data: pd.DataFrame, period: int) -> pd.Series:
        """CCI計算"""
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            sma = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma) / (0.015 * mad)
            return cci
        except:
            return pd.Series()

    def _calculate_williams_r(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Williams %R計算"""
        try:
            high_max = data['High'].rolling(window=period).max()
            low_min = data['Low'].rolling(window=period).min()
            williams_r = -100 * ((high_max - data['Close']) / (high_max - low_min))
            return williams_r
        except:
            return pd.Series()

    def _calculate_roc(self, prices: pd.Series, period: int) -> pd.Series:
        """ROC計算"""
        try:
            roc = 100 * (prices / prices.shift(period) - 1)
            return roc
        except:
            return pd.Series()

    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """OBV計算"""
        try:
            obv = pd.Series(index=data.index, dtype=float)
            obv.iloc[0] = data['Volume'].iloc[0]

            for i in range(1, len(data)):
                if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
                elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]

            return obv
        except:
            return pd.Series()

    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """VWAP計算"""
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
            return vwap
        except:
            return pd.Series()

    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """トレンド強度計算"""
        try:
            if len(prices) < 10:
                return 0.0

            x = np.arange(len(prices))
            y = prices.values

            # 線形回帰
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # R二乗値をトレンド強度とする
            return r_value ** 2
        except:
            return 0.0

    def _identify_support_resistance(self, data: pd.DataFrame) -> Tuple[float, float]:
        """サポート・レジスタンス識別"""
        try:
            # 簡易的な実装：最近のlow/highの平均
            recent_data = data.tail(20)
            support = recent_data['Low'].min()
            resistance = recent_data['High'].max()
            return support, resistance
        except:
            return 0.0, 0.0

    def _calculate_fractal_dimension(self, prices: pd.Series) -> float:
        """フラクタル次元計算"""
        try:
            if len(prices) < 4:
                return 1.5

            # Higuchi法による簡易計算
            N = len(prices)
            k_max = int(N / 4)

            if k_max < 2:
                return 1.5

            lengths = []
            k_values = range(1, k_max + 1)

            for k in k_values:
                length = 0
                for m in range(k):
                    indices = np.arange(m, N, k)
                    if len(indices) > 1:
                        sub_series = prices.iloc[indices]
                        length += np.sum(np.abs(np.diff(sub_series)))

                if length > 0:
                    lengths.append(length / k)

            if len(lengths) < 2:
                return 1.5

            # 対数回帰
            log_k = np.log(k_values[:len(lengths)])
            log_l = np.log(lengths)

            slope, _, _, _, _ = stats.linregress(log_k, log_l)
            fractal_dimension = -slope

            return max(1.0, min(2.0, fractal_dimension))
        except:
            return 1.5

    def _detect_chart_patterns(self, data: pd.DataFrame) -> Dict[str, float]:
        """チャートパターン検出"""

        patterns = {}

        try:
            if len(data) < 20:
                return patterns

            prices = data['Close']

            # ダブルトップ/ボトム検出（簡易版）
            recent_highs = data['High'].rolling(5).max()
            recent_lows = data['Low'].rolling(5).min()

            # 上昇/下降トレンド
            trend_period = min(10, len(prices))
            start_price = prices.iloc[-trend_period]
            end_price = prices.iloc[-1]

            patterns['trend_direction'] = 1 if end_price > start_price else -1
            patterns['trend_magnitude'] = abs(end_price - start_price) / start_price

            # ブレイクアウト可能性
            if len(data) >= 20:
                volatility = prices.pct_change().rolling(20).std().iloc[-1]
                recent_volatility = prices.pct_change().rolling(5).std().iloc[-1]

                if volatility > 0:
                    patterns['breakout_potential'] = recent_volatility / volatility
                else:
                    patterns['breakout_potential'] = 1.0

        except Exception as e:
            self.logger.error(f"Pattern detection failed: {e}")

        return patterns

    def _estimate_garch_volatility(self, prices: pd.Series) -> float:
        """GARCH風ボラティリティ推定"""
        try:
            returns = prices.pct_change().dropna()
            if len(returns) < 10:
                return returns.std()

            # 簡易EWMA
            lambda_param = 0.94
            var_estimate = returns.iloc[0] ** 2

            for ret in returns[1:]:
                var_estimate = lambda_param * var_estimate + (1 - lambda_param) * (ret ** 2)

            return np.sqrt(var_estimate)
        except:
            return 0.0

    def _calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """Hurst指数計算"""
        try:
            if len(prices) < 20:
                return 0.5

            returns = prices.pct_change().dropna()

            # R/S分析
            n = len(returns)
            rs_values = []

            for lag in [5, 10, 15, 20]:
                if n > lag:
                    sub_returns = returns.iloc[:lag]
                    mean_return = sub_returns.mean()
                    deviations = sub_returns - mean_return
                    cumulative_deviations = deviations.cumsum()

                    R = cumulative_deviations.max() - cumulative_deviations.min()
                    S = sub_returns.std()

                    if S > 0:
                        rs_values.append(R / S)

            if len(rs_values) < 2:
                return 0.5

            # H指数推定
            hurst = np.log(np.mean(rs_values)) / np.log(2)
            return max(0.0, min(1.0, hurst))
        except:
            return 0.5

    def _calculate_entropy(self, returns: pd.Series) -> float:
        """エントロピー計算"""
        try:
            if len(returns) < 10:
                return 0.0

            # リターンを離散化
            bins = 10
            counts, _ = np.histogram(returns, bins=bins)
            probabilities = counts / np.sum(counts)

            # シャノンエントロピー
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return entropy
        except:
            return 0.0

    def _create_empty_feature_set(self, symbol: str) -> FeatureSet:
        """空の特徴量セット作成"""

        return FeatureSet(
            symbol=symbol,
            timestamp=datetime.now(),
            price_features={},
            technical_features={},
            volume_features={},
            momentum_features={},
            volatility_features={},
            pattern_features={},
            market_features={},
            statistical_features={}
        )

    def feature_set_to_vector(self, feature_set: FeatureSet) -> np.ndarray:
        """特徴量セットをベクトルに変換"""

        all_features = {}
        all_features.update(feature_set.price_features)
        all_features.update(feature_set.technical_features)
        all_features.update(feature_set.volume_features)
        all_features.update(feature_set.momentum_features)
        all_features.update(feature_set.volatility_features)
        all_features.update(feature_set.pattern_features)
        all_features.update(feature_set.market_features)
        all_features.update(feature_set.statistical_features)

        # NaN値を0で置換
        vector = []
        for key in sorted(all_features.keys()):
            value = all_features[key]
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            vector.append(value)

        return np.array(vector)

# グローバルインスタンス
enhanced_feature_engineer = EnhancedFeatureEngineering()

# テスト関数
async def test_enhanced_feature_engineering():
    """強化特徴量エンジニアリングのテスト"""

    print("=== 強化特徴量エンジニアリング テスト ===")

    engineer = EnhancedFeatureEngineering()

    # テスト銘柄
    test_symbol = "7203"

    print(f"\n[ {test_symbol} 特徴量抽出テスト ]")

    try:
        # データ取得
        if REAL_DATA_PROVIDER_AVAILABLE:
            data = await real_data_provider.get_stock_data(test_symbol, "3mo")
        else:
            # ダミーデータ
            dates = pd.date_range(start='2025-05-01', end='2025-08-14', freq='D')
            np.random.seed(42)
            data = pd.DataFrame({
                'Open': np.random.uniform(2500, 3000, len(dates)),
                'High': np.random.uniform(2600, 3100, len(dates)),
                'Low': np.random.uniform(2400, 2900, len(dates)),
                'Close': np.random.uniform(2500, 3000, len(dates)),
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)

        if data is not None and not data.empty:
            print(f"データ期間: {len(data)}日分")

            # 特徴量抽出
            feature_set = await engineer.extract_comprehensive_features(test_symbol, data)

            # 結果表示
            categories = [
                ('価格特徴量', feature_set.price_features),
                ('技術指標', feature_set.technical_features),
                ('出来高特徴量', feature_set.volume_features),
                ('モメンタム', feature_set.momentum_features),
                ('ボラティリティ', feature_set.volatility_features),
                ('パターン', feature_set.pattern_features),
                ('市場環境', feature_set.market_features),
                ('統計特徴量', feature_set.statistical_features)
            ]

            total_features = 0
            for category_name, features in categories:
                count = len(features)
                total_features += count
                print(f"\n{category_name}: {count}件")

                # 主要特徴量を表示
                for i, (key, value) in enumerate(features.items()):
                    if i < 3:  # 最初の3つのみ表示
                        print(f"  {key}: {value:.4f}")
                    elif i == 3 and count > 3:
                        print(f"  ... 他{count-3}件")
                        break

            print(f"\n総特徴量数: {total_features}")

            # ベクトル変換テスト
            feature_vector = engineer.feature_set_to_vector(feature_set)
            print(f"特徴量ベクトル次元: {len(feature_vector)}")
            print(f"ベクトル統計: 平均={np.mean(feature_vector):.4f}, 標準偏差={np.std(feature_vector):.4f}")

        else:
            print("❌ データ取得に失敗しました")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n=== 強化特徴量エンジニアリング テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_enhanced_feature_engineering())