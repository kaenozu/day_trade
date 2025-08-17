#!/usr/bin/env python3
"""
advanced_technical_analysis.py - AdvancedTechnicalAnalysis

リファクタリングにより分割されたモジュール
"""

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
