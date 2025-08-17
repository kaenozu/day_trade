#!/usr/bin/env python3
"""
advanced_technical_analysis.py - AdvancedTechnicalIndicators

リファクタリングにより分割されたモジュール
"""

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
