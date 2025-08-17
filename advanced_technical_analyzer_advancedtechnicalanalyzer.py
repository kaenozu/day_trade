#!/usr/bin/env python3
"""
advanced_technical_analyzer.py - AdvancedTechnicalAnalyzer

リファクタリングにより分割されたモジュール
"""

class AdvancedTechnicalAnalyzer:
    """
    高度技術指標・分析手法拡張システム
    先進的技術分析の包括的実装
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データディレクトリ
        self.data_dir = Path("advanced_technical_data")
        self.data_dir.mkdir(exist_ok=True)

        # 拡張システム統合
        if ENHANCED_SYMBOLS_AVAILABLE:
            self.symbol_manager = EnhancedSymbolManager()

        # 技術指標計算用キャッシュ
        self.indicator_cache = {}
        self.price_data_cache = {}

        # 機械学習モデル
        self.ml_models = {}
        self.scalers = {}

        # 統計分析用パラメータ
        self.statistical_lookback = 252  # 1年間
        self.volatility_regimes = {
            "超低ボラ": (0, 10),
            "低ボラ": (10, 20),
            "通常ボラ": (20, 35),
            "高ボラ": (35, 50),
            "超高ボラ": (50, 100)
        }

        # パターン認識設定
        self.pattern_configs = {
            "support_resistance_strength": 3,
            "pattern_min_length": 10,
            "breakout_threshold": 0.02
        }

        self.logger.info("Advanced technical analyzer initialized with ML capabilities")

    async def analyze_symbol(self, symbol: str, period: str = "1mo") -> AdvancedAnalysis:
        """
        銘柄の高度技術分析

        Args:
            symbol: 分析対象銘柄
            period: 分析期間

        Returns:
            高度分析結果
        """
        try:
            # 価格データ取得
            df = await self._get_price_data(symbol, period)

            if df is None or len(df) < 20:
                self.logger.warning(f"Insufficient data for {symbol}")
                return None

            current_price = float(df['Close'].iloc[-1])
            price_change = float((df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100)
            volume_ratio = float(df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1])

            # 各種技術指標計算
            trend_indicators = await self._calculate_trend_indicators(df)
            momentum_indicators = await self._calculate_momentum_indicators(df)
            volatility_indicators = await self._calculate_volatility_indicators(df)
            volume_indicators = await self._calculate_volume_indicators(df)

            # 複合分析
            composite_score = self._calculate_composite_score(
                trend_indicators, momentum_indicators, volatility_indicators
            )

            trend_strength = self._calculate_trend_strength(trend_indicators)
            momentum_score = self._calculate_momentum_score(momentum_indicators)
            volatility_regime = self._determine_volatility_regime(volatility_indicators)

            # シグナル生成
            primary_signals, secondary_signals = await self._generate_signals(
                df, trend_indicators, momentum_indicators, volatility_indicators, volume_indicators
            )

            # 統計分析
            statistical_profile = self._perform_statistical_analysis(df)
            anomaly_score = self._calculate_anomaly_score(df)

            # 機械学習予測
            ml_prediction = await self._generate_ml_prediction(symbol, df)

            # パターン認識
            pattern_recognition = self._perform_pattern_recognition(df)

            return AdvancedAnalysis(
                symbol=symbol,
                analysis_time=datetime.now(),
                current_price=current_price,
                price_change=price_change,
                volume_ratio=volume_ratio,
                trend_indicators=trend_indicators,
                momentum_indicators=momentum_indicators,
                volatility_indicators=volatility_indicators,
                volume_indicators=volume_indicators,
                composite_score=composite_score,
                trend_strength=trend_strength,
                momentum_score=momentum_score,
                volatility_regime=volatility_regime,
                primary_signals=primary_signals,
                secondary_signals=secondary_signals,
                statistical_profile=statistical_profile,
                anomaly_score=anomaly_score,
                ml_prediction=ml_prediction,
                pattern_recognition=pattern_recognition
            )

        except Exception as e:
            self.logger.error(f"Advanced analysis failed for {symbol}: {e}")
            return None

    async def _get_price_data(self, symbol: str, period: str) -> pd.DataFrame:
        """価格データ取得（実データプロバイダー対応）"""

        cache_key = f"{symbol}_{period}"

        if cache_key in self.price_data_cache:
            cached_data, cache_time = self.price_data_cache[cache_key]
            if datetime.now() - cache_time < timedelta(minutes=15):
                return cached_data

        try:
            # 実データプロバイダーV2使用
            if REAL_DATA_PROVIDER_AVAILABLE:
                df = await real_data_provider.get_stock_data(symbol, period)

                if df is not None and not df.empty:
                    # キャッシュに保存
                    self.price_data_cache[cache_key] = (df, datetime.now())
                    self.logger.info(f"Real data fetched for {symbol}: {len(df)} days")
                    return df

            # フォールバック: 従来のYahoo Finance
            if YFINANCE_AVAILABLE:
                ticker_symbol = f"{symbol}.T" if not symbol.endswith('.T') else symbol
                ticker = yf.Ticker(ticker_symbol)
                df = ticker.history(period=period)

                if not df.empty:
                    self.price_data_cache[cache_key] = (df, datetime.now())
                    self.logger.info(f"Yahoo Finance data fetched for {symbol}: {len(df)} days")
                    return df

            # 最後のフォールバック: サンプルデータ
            self.logger.warning(f"Using sample data for {symbol}")
            return self._generate_sample_data(symbol, period)

        except Exception as e:
            self.logger.warning(f"Failed to fetch real data for {symbol}: {e}")
            return self._generate_sample_data(symbol, period)

    def _generate_sample_data(self, symbol: str, period: str) -> pd.DataFrame:
        """サンプル価格データ生成"""

        days = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}.get(period, 30)

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # ランダムウォーク + トレンド
        np.random.seed(hash(symbol) % 2**32)
        base_price = 1000 + (hash(symbol) % 5000)

        returns = np.random.normal(0.001, 0.02, days)  # 日次リターン
        trend = np.linspace(0, np.random.normal(0, 0.1), days)
        cumulative_returns = np.cumsum(returns + trend)
        prices = base_price * np.exp(cumulative_returns)

        # OHLC計算
        opens = prices * (1 + np.random.normal(0, 0.005, days))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.01, days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.01, days)))

        volumes = np.random.lognormal(14, 0.5, days).astype(int) * 1000

        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=dates)

        return df

    async def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """トレンド系指標計算"""

        indicators = {}

        try:
            closes = df['Close']
            highs = df['High']
            lows = df['Low']

            # 移動平均群
            indicators['SMA_5'] = float(closes.rolling(5).mean().iloc[-1])
            indicators['SMA_20'] = float(closes.rolling(20).mean().iloc[-1])
            indicators['SMA_50'] = float(closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else closes.mean())

            indicators['EMA_12'] = float(closes.ewm(span=12).mean().iloc[-1])
            indicators['EMA_26'] = float(closes.ewm(span=26).mean().iloc[-1])

            # MACD
            macd_line = closes.ewm(span=12).mean() - closes.ewm(span=26).mean()
            signal_line = macd_line.ewm(span=9).mean()
            indicators['MACD'] = float(macd_line.iloc[-1])
            indicators['MACD_Signal'] = float(signal_line.iloc[-1])
            indicators['MACD_Histogram'] = float(macd_line.iloc[-1] - signal_line.iloc[-1])

            # ADX（方向性指数）
            indicators['ADX'] = self._calculate_adx(highs, lows, closes)

            # パラボリックSAR
            indicators['PSAR'] = self._calculate_psar(highs, lows, closes)

            # 一目均衡表
            ichimoku = self._calculate_ichimoku(highs, lows, closes)
            indicators.update(ichimoku)

            # SuperTrend
            indicators['SuperTrend'] = self._calculate_supertrend(highs, lows, closes)

        except Exception as e:
            self.logger.error(f"Trend indicators calculation error: {e}")

        return indicators

    async def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """モメンタム系指標計算"""

        indicators = {}

        try:
            closes = df['Close']
            highs = df['High']
            lows = df['Low']
            volumes = df['Volume']

            # RSI群
            indicators['RSI_14'] = self._calculate_rsi(closes, 14)
            indicators['RSI_21'] = self._calculate_rsi(closes, 21)

            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(highs, lows, closes)
            indicators['Stoch_K'] = stoch_k
            indicators['Stoch_D'] = stoch_d

            # Williams %R
            indicators['Williams_R'] = self._calculate_williams_r(highs, lows, closes)

            # CCI（商品チャンネル指数）
            indicators['CCI'] = self._calculate_cci(highs, lows, closes)

            # ROC（変化率）
            indicators['ROC_12'] = float((closes.iloc[-1] / closes.iloc[-13] - 1) * 100 if len(closes) >= 13 else 0)

            # Money Flow Index
            indicators['MFI'] = self._calculate_mfi(highs, lows, closes, volumes)

            # Ultimate Oscillator
            indicators['UO'] = self._calculate_ultimate_oscillator(highs, lows, closes)

            # Awesome Oscillator
            indicators['AO'] = self._calculate_awesome_oscillator(highs, lows)

        except Exception as e:
            self.logger.error(f"Momentum indicators calculation error: {e}")

        return indicators

    async def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """ボラティリティ系指標計算"""

        indicators = {}

        try:
            closes = df['Close']
            highs = df['High']
            lows = df['Low']

            # ボリンジャーバンド
            bb_middle, bb_upper, bb_lower = self._calculate_bollinger_bands(closes)
            indicators['BB_Middle'] = bb_middle
            indicators['BB_Upper'] = bb_upper
            indicators['BB_Lower'] = bb_lower
            try:
                indicators['BB_Width'] = float((bb_upper - bb_lower) / bb_middle * 100)
                indicators['BB_Position'] = float((closes.iloc[-1] - bb_lower) / (bb_upper - bb_lower) * 100)
            except (TypeError, ValueError):
                indicators['BB_Width'] = 2.0  # デフォルト値
                indicators['BB_Position'] = 50.0

            # ATR（真の値幅）
            indicators['ATR_14'] = self._calculate_atr(highs, lows, closes, 14)
            indicators['ATR_21'] = self._calculate_atr(highs, lows, closes, 21)

            # 標準偏差
            indicators['StdDev_20'] = float(closes.rolling(20).std())

            # Keltner Channel
            kc_middle, kc_upper, kc_lower = self._calculate_keltner_channel(highs, lows, closes)
            indicators['KC_Middle'] = kc_middle
            indicators['KC_Upper'] = kc_upper
            indicators['KC_Lower'] = kc_lower

            # 歴史的ボラティリティ
            returns = closes.pct_change().dropna()
            indicators['Historical_Vol'] = float(returns.std() * np.sqrt(252) * 100)

            # VIX風指標（ATRベース）
            indicators['VIX_Proxy'] = float(indicators['ATR_14'] / closes.iloc[-1] * 100)

        except Exception as e:
            self.logger.error(f"Volatility indicators calculation error: {e}")

        return indicators

    async def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """出来高系指標計算"""

        indicators = {}

        try:
            closes = df['Close']
            volumes = df['Volume']

            # 出来高移動平均
            indicators['Volume_SMA_20'] = float(volumes.rolling(20).mean().iloc[-1])
            indicators['Volume_Ratio'] = float(volumes.iloc[-1] / volumes.rolling(20).mean().iloc[-1])

            # OBV（オンバランスボリューム）
            indicators['OBV'] = self._calculate_obv(closes, volumes)

            # A/D Line（集積/配布ライン）
            indicators['AD_Line'] = self._calculate_ad_line(df)

            # Chaikin Money Flow
            indicators['CMF'] = self._calculate_cmf(df)

            # Volume Oscillator
            vol_short = volumes.rolling(5).mean()
            vol_long = volumes.rolling(20).mean()
            indicators['Volume_Osc'] = float((vol_short.iloc[-1] - vol_long.iloc[-1]) / vol_long.iloc[-1] * 100)

            # Price Volume Trend
            indicators['PVT'] = self._calculate_pvt(closes, volumes)

            # VWAP（出来高加重平均価格）
            indicators['VWAP'] = self._calculate_vwap(df)

        except Exception as e:
            self.logger.error(f"Volume indicators calculation error: {e}")

        return indicators

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI計算"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50.0

    def _calculate_stochastic(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """ストキャスティクス計算"""
        try:
            lowest_low = lows.rolling(window=k_period).min()
            highest_high = highs.rolling(window=k_period).max()
            k_percent = 100 * ((closes - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            return float(k_percent.iloc[-1]), float(d_percent.iloc[-1])
        except:
            return 50.0, 50.0

    def _calculate_williams_r(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> float:
        """Williams %R計算"""
        try:
            highest_high = highs.rolling(window=period).max()
            lowest_low = lows.rolling(window=period).min()
            wr = -100 * ((highest_high - closes) / (highest_high - lowest_low))
            return float(wr.iloc[-1])
        except:
            return -50.0

    def _calculate_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> float:
        """ATR計算"""
        try:
            high_low = highs - lows
            high_close = np.abs(highs - closes.shift())
            low_close = np.abs(lows - closes.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return float(atr.iloc[-1])
        except:
            return 0.0

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """ボリンジャーバンド計算"""
        try:
            middle = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return float(middle.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1])
        except:
            last_price = float(prices.iloc[-1])
            return last_price, last_price * 1.02, last_price * 0.98

    def _calculate_obv(self, closes: pd.Series, volumes: pd.Series) -> float:
        """OBV計算"""
        try:
            price_change = closes.diff()
            obv = volumes.copy()
            obv[price_change < 0] = -obv[price_change < 0]
            obv[price_change == 0] = 0
            return float(obv.cumsum().iloc[-1])
        except:
            return 0.0

    def _calculate_composite_score(self, trend: Dict, momentum: Dict, volatility: Dict) -> float:
        """複合スコア計算"""
        try:
            score = 50.0  # 中立からスタート

            # トレンド評価
            if 'SMA_5' in trend and 'SMA_20' in trend:
                if trend['SMA_5'] > trend['SMA_20']:
                    score += 10
                else:
                    score -= 10

            # MACD評価
            if 'MACD' in trend and 'MACD_Signal' in trend:
                if trend['MACD'] > trend['MACD_Signal']:
                    score += 8
                else:
                    score -= 8

            # RSI評価
            if 'RSI_14' in momentum:
                rsi = momentum['RSI_14']
                if 30 <= rsi <= 70:
                    score += 5  # 適正レンジ
                elif rsi > 80:
                    score -= 10  # 買われすぎ
                elif rsi < 20:
                    score += 10  # 売られすぎ（反転期待）

            # ボラティリティ評価
            if 'BB_Position' in volatility:
                bb_pos = volatility['BB_Position']
                if 20 <= bb_pos <= 80:
                    score += 3  # 適正範囲
                else:
                    score -= 3  # 極端な位置

            return max(0, min(100, score))

        except Exception as e:
            self.logger.error(f"Composite score calculation error: {e}")
            return 50.0

    def _calculate_trend_strength(self, indicators: Dict[str, float]) -> float:
        """トレンド強度計算"""
        try:
            strength = 0.0

            # MACD強度
            if 'MACD' in indicators and 'MACD_Signal' in indicators:
                macd_strength = (indicators['MACD'] - indicators['MACD_Signal']) * 10
                strength += max(-30, min(30, macd_strength))

            # ADX強度
            if 'ADX' in indicators:
                adx = indicators['ADX']
                if adx > 25:
                    strength += min(25, adx)
                else:
                    strength += adx / 2

            # 移動平均の位置関係
            if all(key in indicators for key in ['SMA_5', 'SMA_20', 'SMA_50']):
                if indicators['SMA_5'] > indicators['SMA_20'] > indicators['SMA_50']:
                    strength += 20  # 上昇トレンド
                elif indicators['SMA_5'] < indicators['SMA_20'] < indicators['SMA_50']:
                    strength -= 20  # 下降トレンド

            return max(-100, min(100, strength))

        except Exception as e:
            self.logger.error(f"Trend strength calculation error: {e}")
            return 0.0

    def _calculate_momentum_score(self, indicators: Dict[str, float]) -> float:
        """モメンタムスコア計算"""
        try:
            score = 0.0
            count = 0

            # RSI評価
            if 'RSI_14' in indicators:
                rsi = indicators['RSI_14']
                rsi_score = (rsi - 50) * 2  # -100~100に正規化
                score += rsi_score
                count += 1

            # Stochastic評価
            if 'Stoch_K' in indicators:
                stoch = indicators['Stoch_K']
                stoch_score = (stoch - 50) * 2
                score += stoch_score
                count += 1

            # Williams %R評価
            if 'Williams_R' in indicators:
                wr = indicators['Williams_R']
                wr_score = (wr + 50) * 2  # -100~0を-100~100に変換
                score += wr_score
                count += 1

            # ROC評価
            if 'ROC_12' in indicators:
                roc = indicators['ROC_12']
                roc_score = max(-50, min(50, roc * 2))
                score += roc_score
                count += 1

            return (score / count) if count > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Momentum score calculation error: {e}")
            return 0.0

    def _determine_volatility_regime(self, indicators: Dict[str, float]) -> str:
        """ボラティリティ局面判定"""
        try:
            if 'Historical_Vol' in indicators:
                vol = indicators['Historical_Vol']
                for regime, (low, high) in self.volatility_regimes.items():
                    if low <= vol < high:
                        return regime

            return "通常ボラ"

        except Exception as e:
            self.logger.error(f"Volatility regime determination error: {e}")
            return "通常ボラ"

    async def _generate_signals(self, df: pd.DataFrame, trend: Dict, momentum: Dict,
                              volatility: Dict, volume: Dict) -> Tuple[List[TechnicalSignal], List[TechnicalSignal]]:
        """シグナル生成"""

        primary_signals = []
        secondary_signals = []

        try:
            current_price = float(df['Close'].iloc[-1])
            timestamp = datetime.now()

            # MACD買いシグナル
            if 'MACD' in trend and 'MACD_Signal' in trend:
                if trend['MACD'] > trend['MACD_Signal'] and trend['MACD_Histogram'] > 0:
                    signal = TechnicalSignal(
                        indicator_name="MACD",
                        signal_type="BUY",
                        strength=SignalStrength.MODERATE,
                        confidence=75.0,
                        price_level=current_price,
                        timestamp=timestamp,
                        indicator_value=trend['MACD'],
                        trend_direction="UP"
                    )
                    primary_signals.append(signal)

            # RSI逆張りシグナル
            if 'RSI_14' in momentum:
                rsi = momentum['RSI_14']
                if rsi < 30:
                    signal = TechnicalSignal(
                        indicator_name="RSI_Oversold",
                        signal_type="BUY",
                        strength=SignalStrength.STRONG,
                        confidence=80.0,
                        price_level=current_price,
                        timestamp=timestamp,
                        indicator_value=rsi,
                        threshold_lower=30.0
                    )
                    primary_signals.append(signal)
                elif rsi > 70:
                    signal = TechnicalSignal(
                        indicator_name="RSI_Overbought",
                        signal_type="SELL",
                        strength=SignalStrength.MODERATE,
                        confidence=70.0,
                        price_level=current_price,
                        timestamp=timestamp,
                        indicator_value=rsi,
                        threshold_upper=70.0
                    )
                    secondary_signals.append(signal)

            # ボリンジャーバンドブレイクアウト
            if all(key in volatility for key in ['BB_Upper', 'BB_Lower', 'BB_Position']):
                bb_pos = volatility['BB_Position']
                if bb_pos > 95:  # 上限突破
                    signal = TechnicalSignal(
                        indicator_name="BB_Breakout_Up",
                        signal_type="BUY",
                        strength=SignalStrength.STRONG,
                        confidence=85.0,
                        price_level=current_price,
                        timestamp=timestamp,
                        indicator_value=bb_pos,
                        target_price=current_price * 1.03
                    )
                    primary_signals.append(signal)
                elif bb_pos < 5:  # 下限反発
                    signal = TechnicalSignal(
                        indicator_name="BB_Bounce_Up",
                        signal_type="BUY",
                        strength=SignalStrength.MODERATE,
                        confidence=75.0,
                        price_level=current_price,
                        timestamp=timestamp,
                        indicator_value=bb_pos,
                        target_price=current_price * 1.02
                    )
                    secondary_signals.append(signal)

            # 出来高急増シグナル
            if 'Volume_Ratio' in volume and volume['Volume_Ratio'] > 2.0:
                signal = TechnicalSignal(
                    indicator_name="Volume_Surge",
                    signal_type="HOLD",
                    strength=SignalStrength.MODERATE,
                    confidence=70.0,
                    price_level=current_price,
                    timestamp=timestamp,
                    indicator_value=volume['Volume_Ratio']
                )
                secondary_signals.append(signal)

        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")

        return primary_signals, secondary_signals

    def _perform_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, float]:
        """統計分析実行"""

        try:
            closes = df['Close']
            returns = closes.pct_change().dropna()

            return {
                'mean_return': float(returns.mean() * 252),  # 年率化
                'volatility': float(returns.std() * np.sqrt(252)),
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns)),
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)),
                'max_drawdown': self._calculate_max_drawdown(closes),
                'var_95': float(np.percentile(returns, 5) * 100),
                'autocorrelation': float(returns.autocorr(lag=1))
            }

        except Exception as e:
            self.logger.error(f"Statistical analysis error: {e}")
            return {}

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """最大ドローダウン計算"""
        try:
            cumulative = (1 + prices.pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return float(drawdown.min() * 100)
        except:
            return 0.0

    def _calculate_anomaly_score(self, df: pd.DataFrame) -> float:
        """異常度スコア計算"""
        try:
            # 価格変動率の異常度
            returns = df['Close'].pct_change().dropna()
            if len(returns) < 10:
                return 0.0

            # Z-score計算
            z_score = abs((returns.iloc[-1] - returns.mean()) / returns.std())

            # 0-100スケールに変換
            anomaly_score = min(100, z_score * 20)

            return float(anomaly_score)

        except Exception as e:
            self.logger.error(f"Anomaly score calculation error: {e}")
            return 0.0

    async def _generate_ml_prediction(self, symbol: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """機械学習予測生成"""
        try:
            if len(df) < 50:
                return None

            # 特徴量作成
            features = self._create_ml_features(df)

            if features is None or len(features) < 20:
                return None

            # 簡易予測（実装例）
            returns = df['Close'].pct_change().dropna()
            recent_trend = returns.rolling(10).mean().iloc[-1]
            trend_strength = abs(recent_trend)

            # 方向性予測
            direction = "上昇" if recent_trend > 0.001 else "下落" if recent_trend < -0.001 else "横ばい"
            confidence = min(90, trend_strength * 1000 + 50)

            return {
                'direction': direction,
                'confidence': confidence,
                'expected_return': recent_trend * 100,
                'risk_level': "高" if trend_strength > 0.02 else "中" if trend_strength > 0.01 else "低",
                'model_type': 'trend_based',
                'features_used': len(features)
            }

        except Exception as e:
            self.logger.error(f"ML prediction error: {e}")
            return None

    def _create_ml_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """機械学習用特徴量作成"""
        try:
            features = pd.DataFrame(index=df.index)

            # 価格特徴量
            features['return_1d'] = df['Close'].pct_change()
            features['return_5d'] = df['Close'].pct_change(5)
            features['return_20d'] = df['Close'].pct_change(20)

            # 移動平均特徴量
            features['ma_5_ratio'] = df['Close'] / df['Close'].rolling(5).mean()
            features['ma_20_ratio'] = df['Close'] / df['Close'].rolling(20).mean()

            # ボラティリティ特徴量
            features['volatility_5d'] = df['Close'].pct_change().rolling(5).std()
            features['volatility_20d'] = df['Close'].pct_change().rolling(20).std()

            # 出来高特徴量
            features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

            return features.dropna()

        except Exception as e:
            self.logger.error(f"Feature creation error: {e}")
            return None

    def _perform_pattern_recognition(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """パターン認識実行"""
        try:
            closes = df['Close']

            # サポート・レジスタンスレベル検出
            support_levels, resistance_levels = self._find_support_resistance(closes)

            # チャートパターン検出（簡易版）
            pattern = self._detect_chart_patterns(closes)

            return {
                'support_levels': support_levels[-3:] if support_levels else [],  # 直近3レベル
                'resistance_levels': resistance_levels[-3:] if resistance_levels else [],
                'detected_pattern': pattern,
                'current_position': 'above_support' if support_levels and closes.iloc[-1] > max(support_levels) else 'neutral'
            }

        except Exception as e:
            self.logger.error(f"Pattern recognition error: {e}")
            return None

    def _find_support_resistance(self, prices: pd.Series) -> Tuple[List[float], List[float]]:
        """サポート・レジスタンスレベル検出"""
        try:
            # ピーク・ボトム検出
            peaks, _ = find_peaks(prices, distance=5)
            troughs, _ = find_peaks(-prices, distance=5)

            # レジスタンス（ピーク）
            resistance_levels = [float(prices.iloc[i]) for i in peaks[-5:]]

            # サポート（ボトム）
            support_levels = [float(prices.iloc[i]) for i in troughs[-5:]]

            return support_levels, resistance_levels

        except Exception as e:
            self.logger.error(f"Support/resistance detection error: {e}")
            return [], []

    def _detect_chart_patterns(self, prices: pd.Series) -> str:
        """チャートパターン検出（簡易版）"""
        try:
            if len(prices) < 20:
                return "insufficient_data"

            recent_prices = prices.iloc[-20:]

            # 上昇トレンド
            if recent_prices.iloc[-1] > recent_prices.iloc[0] * 1.05:
                return "uptrend"

            # 下降トレンド
            elif recent_prices.iloc[-1] < recent_prices.iloc[0] * 0.95:
                return "downtrend"

            # レンジ相場
            elif recent_prices.std() / recent_prices.mean() < 0.02:
                return "sideways"

            else:
                return "unclear"

        except Exception as e:
            self.logger.error(f"Pattern detection error: {e}")
            return "error"

    # 追加の技術指標計算メソッド（簡易実装）
    def _calculate_adx(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> float:
        """ADX計算（簡易版）"""
        try:
            # True Rangeベースの簡易ADX
            atr = self._calculate_atr(highs, lows, closes, period)
            return min(100, max(0, atr / closes.iloc[-1] * 100 * 5))
        except:
            return 25.0

    def _calculate_psar(self, highs: pd.Series, lows: pd.Series, closes: pd.Series) -> float:
        """パラボリックSAR計算（簡易版）"""
        try:
            # 簡易PSAR（実際の実装はより複雑）
            recent_low = lows.iloc[-10:].min()
            recent_high = highs.iloc[-10:].max()

            if closes.iloc[-1] > closes.iloc[-5]:  # 上昇トレンド
                return float(recent_low * 0.98)
            else:  # 下降トレンド
                return float(recent_high * 1.02)
        except:
            return float(closes.iloc[-1])

    def _calculate_ichimoku(self, highs: pd.Series, lows: pd.Series, closes: pd.Series) -> Dict[str, float]:
        """一目均衡表計算"""
        try:
            # 転換線 (9日間の高値と安値の平均)
            tenkan = (highs.rolling(9).max() + lows.rolling(9).min()) / 2

            # 基準線 (26日間の高値と安値の平均)
            kijun = (highs.rolling(26).max() + lows.rolling(26).min()) / 2

            return {
                'Ichimoku_Tenkan': float(tenkan.iloc[-1] if not tenkan.empty else closes.iloc[-1]),
                'Ichimoku_Kijun': float(kijun.iloc[-1] if not kijun.empty else closes.iloc[-1])
            }
        except:
            return {'Ichimoku_Tenkan': float(closes.iloc[-1]), 'Ichimoku_Kijun': float(closes.iloc[-1])}

    def _calculate_supertrend(self, highs: pd.Series, lows: pd.Series, closes: pd.Series) -> float:
        """SuperTrend計算（簡易版）"""
        try:
            atr = self._calculate_atr(highs, lows, closes)
            hl2 = (highs + lows) / 2

            upper_band = hl2.iloc[-1] + 2 * atr
            lower_band = hl2.iloc[-1] - 2 * atr

            if closes.iloc[-1] > closes.iloc[-5]:  # 上昇
                return float(lower_band)
            else:  # 下降
                return float(upper_band)
        except:
            return float(closes.iloc[-1])

    def _calculate_cci(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 20) -> float:
        """CCI計算"""
        try:
            tp = (highs + lows + closes) / 3
            sma = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: pd.Series(x).mad())
            cci = (tp - sma) / (0.015 * mad)
            return float(cci.iloc[-1])
        except:
            return 0.0

    def _calculate_mfi(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, volumes: pd.Series, period: int = 14) -> float:
        """MFI計算"""
        try:
            tp = (highs + lows + closes) / 3
            mf = tp * volumes

            pos_mf = mf.where(tp > tp.shift(), 0).rolling(period).sum()
            neg_mf = mf.where(tp < tp.shift(), 0).rolling(period).sum()

            mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
            return float(mfi.iloc[-1])
        except:
            return 50.0

    def _calculate_ultimate_oscillator(self, highs: pd.Series, lows: pd.Series, closes: pd.Series) -> float:
        """Ultimate Oscillator計算（簡易版）"""
        try:
            # 3つの期間のモメンタム平均
            periods = [7, 14, 28]
            values = []

            for period in periods:
                if len(closes) >= period:
                    roc = (closes.iloc[-1] / closes.iloc[-period] - 1) * 100
                    values.append(roc)

            if values:
                uo = sum(values) / len(values)
                return float(max(-100, min(100, uo + 50)))  # -100~100を0~100に変換

            return 50.0
        except:
            return 50.0

    def _calculate_awesome_oscillator(self, highs: pd.Series, lows: pd.Series) -> float:
        """Awesome Oscillator計算"""
        try:
            median_price = (highs + lows) / 2
            ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
            return float(ao.iloc[-1])
        except:
            return 0.0

    def _calculate_keltner_channel(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 20) -> Tuple[float, float, float]:
        """Keltner Channel計算"""
        try:
            middle = closes.rolling(period).mean()
            atr = self._calculate_atr(highs, lows, closes, period)

            upper = middle + 2 * atr
            lower = middle - 2 * atr

            return float(middle.iloc[-1]), float(upper), float(lower)
        except:
            price = float(closes.iloc[-1])
            return price, price * 1.02, price * 0.98

    def _calculate_ad_line(self, df: pd.DataFrame) -> float:
        """A/D Line計算"""
        try:
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            ad_line = (clv * df['Volume']).cumsum()
            return float(ad_line.iloc[-1])
        except:
            return 0.0

    def _calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> float:
        """Chaikin Money Flow計算"""
        try:
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            cmf = (clv * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()
            return float(cmf.iloc[-1])
        except:
            return 0.0

    def _calculate_pvt(self, closes: pd.Series, volumes: pd.Series) -> float:
        """Price Volume Trend計算"""
        try:
            pvt = ((closes.pct_change()) * volumes).cumsum()
            return float(pvt.iloc[-1])
        except:
            return 0.0

    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """VWAP計算"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
            return float(vwap.iloc[-1])
        except:
            return float(df['Close'].iloc[-1])
