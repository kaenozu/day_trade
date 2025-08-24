#!/usr/bin/env python3
"""
機械学習結果可視化システム - データ処理機能

Issue #315: 高度テクニカル指標・ML機能拡張
分析結果データの前処理、変換、検証機能
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from .types import (
    LSTMResults,
    VolatilityResults,
    MultiFrameResults,
    TradingAction,
    RiskLevel,
    validate_lstm_results,
    validate_volatility_results,
    validate_multiframe_results,
    ERROR_MESSAGES
)

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class DataProcessor:
    """
    機械学習結果データ処理クラス
    
    分析結果の前処理、変換、検証を担当
    """
    
    def __init__(self):
        """初期化"""
        logger.info("データ処理システム初期化")
    
    def process_lstm_results(
        self, 
        raw_results: Dict,
        data: Optional[pd.DataFrame] = None
    ) -> Optional[LSTMResults]:
        """
        LSTM結果の処理・検証
        
        Args:
            raw_results: 生のLSTM結果
            data: 参照用価格データ
            
        Returns:
            処理済みLSTM結果、エラー時はNone
        """
        try:
            if not validate_lstm_results(raw_results):
                logger.error(f"LSTM結果検証失敗: {raw_results.keys()}")
                return None
            
            # 予測価格の処理
            predicted_prices = raw_results.get('predicted_prices', [])
            if not predicted_prices:
                logger.warning("予測価格データが空です")
                return None
            
            # 予測リターンの処理
            predicted_returns = raw_results.get('predicted_returns', [])
            if not predicted_returns:
                # 価格データから計算
                if len(predicted_prices) > 1:
                    predicted_returns = self._calculate_returns_from_prices(
                        predicted_prices
                    )
                else:
                    predicted_returns = [0.0]
            
            # 予測日付の処理
            prediction_dates = raw_results.get('prediction_dates', [])
            if not prediction_dates:
                # データから推定
                prediction_dates = self._generate_prediction_dates(
                    len(predicted_prices), data
                )
            
            # 日付の正規化
            prediction_dates = [
                self._normalize_date_string(date) for date in prediction_dates
            ]
            
            # 信頼度の処理
            confidence_score = max(0, min(100, 
                raw_results.get('confidence_score', 0)
            ))
            
            # モデルメトリクスの処理
            model_metrics = raw_results.get('model_metrics', {})
            
            return LSTMResults(
                predicted_prices=predicted_prices,
                predicted_returns=predicted_returns,
                prediction_dates=prediction_dates,
                confidence_score=confidence_score,
                model_metrics=model_metrics
            )
            
        except Exception as e:
            logger.error(f"LSTM結果処理エラー: {e}")
            return None
    
    def process_volatility_results(
        self, 
        raw_results: Dict
    ) -> Optional[VolatilityResults]:
        """
        ボラティリティ結果の処理・検証
        
        Args:
            raw_results: 生のボラティリティ結果
            
        Returns:
            処理済みボラティリティ結果、エラー時はNone
        """
        try:
            if not validate_volatility_results(raw_results):
                logger.error("ボラティリティ結果検証失敗")
                return None
            
            # 現在メトリクスの処理
            current_metrics_raw = raw_results.get('current_metrics', {})
            current_metrics = self._process_current_metrics(current_metrics_raw)
            
            # アンサンブル予測の処理
            ensemble_raw = raw_results.get('ensemble_forecast', {})
            ensemble = self._process_ensemble_forecast(ensemble_raw)
            
            # リスク評価の処理
            risk_raw = raw_results.get('risk_assessment', {})
            risk_assessment = self._process_risk_assessment(risk_raw)
            
            # 投資への示唆の処理
            implications = raw_results.get('investment_implications', {})
            processed_implications = self._process_investment_implications(
                implications
            )
            
            return VolatilityResults(
                current_metrics=current_metrics,
                ensemble_forecast=ensemble,
                risk_assessment=risk_assessment,
                investment_implications=processed_implications
            )
            
        except Exception as e:
            logger.error(f"ボラティリティ結果処理エラー: {e}")
            return None
    
    def process_multiframe_results(
        self, 
        raw_results: Dict
    ) -> Optional[MultiFrameResults]:
        """
        マルチタイムフレーム結果の処理・検証
        
        Args:
            raw_results: 生のマルチタイムフレーム結果
            
        Returns:
            処理済みマルチタイムフレーム結果、エラー時はNone
        """
        try:
            if not validate_multiframe_results(raw_results):
                logger.error("マルチタイムフレーム結果検証失敗")
                return None
            
            # 時間軸データの処理
            timeframes_raw = raw_results.get('timeframes', {})
            processed_timeframes = {}
            
            for tf_key, tf_data in timeframes_raw.items():
                processed_tf = self._process_timeframe_data(tf_key, tf_data)
                if processed_tf:
                    processed_timeframes[tf_key] = processed_tf
            
            # 統合分析の処理
            integrated_raw = raw_results.get('integrated_analysis', {})
            integrated_analysis = self._process_integrated_analysis(
                integrated_raw
            )
            
            if not integrated_analysis:
                logger.error("統合分析データが不正です")
                return None
            
            return MultiFrameResults(
                timeframes=processed_timeframes,
                integrated_analysis=integrated_analysis
            )
            
        except Exception as e:
            logger.error(f"マルチタイムフレーム結果処理エラー: {e}")
            return None
    
    def merge_analysis_results(
        self,
        lstm_results: Optional[LSTMResults] = None,
        volatility_results: Optional[VolatilityResults] = None,
        multiframe_results: Optional[MultiFrameResults] = None,
        technical_results: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        複数の分析結果をマージ
        
        Args:
            lstm_results: LSTM結果
            volatility_results: ボラティリティ結果
            multiframe_results: マルチタイムフレーム結果
            technical_results: テクニカル分析結果
            
        Returns:
            統合された分析結果辞書
        """
        merged_results = {
            'timestamp': datetime.now(),
            'available_analyses': [],
            'integrated_signals': [],
            'risk_factors': [],
            'recommendations': []
        }
        
        # LSTM結果の統合
        if lstm_results:
            merged_results['lstm'] = lstm_results
            merged_results['available_analyses'].append('LSTM')
            
            # LSTM からシグナル抽出
            if lstm_results.predicted_returns:
                avg_return = np.mean(lstm_results.predicted_returns)
                lstm_signal = self._extract_lstm_signal(avg_return)
                merged_results['integrated_signals'].append({
                    'source': 'LSTM',
                    'signal': lstm_signal,
                    'confidence': lstm_results.confidence_score
                })
        
        # ボラティリティ結果の統合
        if volatility_results:
            merged_results['volatility'] = volatility_results
            merged_results['available_analyses'].append('Volatility')
            
            # リスク要因追加
            risk_factors = volatility_results.risk_assessment.risk_factors
            merged_results['risk_factors'].extend(risk_factors)
        
        # マルチタイムフレーム結果の統合
        if multiframe_results:
            merged_results['multiframe'] = multiframe_results
            merged_results['available_analyses'].append('MultiTimeframe')
            
            # 統合シグナル追加
            integrated_signal = multiframe_results.integrated_analysis.integrated_signal
            merged_results['integrated_signals'].append({
                'source': 'MultiTimeframe',
                'signal': integrated_signal.action,
                'confidence': multiframe_results.integrated_analysis.trend_confidence
            })
            
            # 推奨追加
            recommendation = multiframe_results.integrated_analysis.investment_recommendation
            merged_results['recommendations'].append({
                'source': 'MultiTimeframe',
                'action': recommendation.recommendation,
                'position_size': recommendation.position_size,
                'confidence': recommendation.confidence
            })
        
        # テクニカル結果の統合
        if technical_results:
            merged_results['technical'] = technical_results
            merged_results['available_analyses'].append('Technical')
        
        # 統合判定の実行
        merged_results['final_assessment'] = self._create_final_assessment(
            merged_results
        )
        
        return merged_results
    
    def prepare_chart_data(
        self, 
        data: pd.DataFrame, 
        analysis_results: Dict,
        lookback_days: int = 100
    ) -> Dict[str, Any]:
        """
        チャート作成用データの準備
        
        Args:
            data: 価格データ
            analysis_results: 統合分析結果
            lookback_days: 表示期間（日数）
            
        Returns:
            チャート用データ辞書
        """
        try:
            # 基本価格データの準備
            recent_data = data.tail(lookback_days) if len(data) > lookback_days else data
            
            chart_data = {
                'price_data': recent_data,
                'dates': recent_data.index,
                'prices': recent_data['Close'].values,
                'volumes': recent_data.get('Volume', pd.Series()).values,
                'high_prices': recent_data.get('High', recent_data['Close']).values,
                'low_prices': recent_data.get('Low', recent_data['Close']).values
            }
            
            # LSTM予測データ
            if 'lstm' in analysis_results:
                lstm_data = analysis_results['lstm']
                chart_data['lstm'] = {
                    'predicted_prices': lstm_data.predicted_prices,
                    'predicted_returns': lstm_data.predicted_returns,
                    'prediction_dates': pd.to_datetime(lstm_data.prediction_dates),
                    'confidence': lstm_data.confidence_score
                }
            
            # ボラティリティデータ
            if 'volatility' in analysis_results:
                vol_data = analysis_results['volatility']
                chart_data['volatility'] = {
                    'current_volatility': vol_data.current_metrics.realized_volatility * 100,
                    'vix_indicator': vol_data.current_metrics.vix_like_indicator,
                    'risk_level': vol_data.risk_assessment.risk_level,
                    'risk_score': vol_data.risk_assessment.risk_score
                }
            
            # マルチタイムフレームデータ
            if 'multiframe' in analysis_results:
                mf_data = analysis_results['multiframe']
                chart_data['multiframe'] = {
                    'timeframes': mf_data.timeframes,
                    'overall_trend': mf_data.integrated_analysis.overall_trend,
                    'consistency': mf_data.integrated_analysis.consistency_score,
                    'signal': mf_data.integrated_analysis.integrated_signal
                }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"チャートデータ準備エラー: {e}")
            return {}
    
    # プライベートメソッド群
    
    def _calculate_returns_from_prices(self, prices: List[float]) -> List[float]:
        """価格からリターンを計算"""
        if len(prices) <= 1:
            return [0.0]
        
        returns = []
        for i in range(1, len(prices)):
            ret = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
            returns.append(ret)
        
        return returns
    
    def _generate_prediction_dates(
        self, 
        count: int, 
        reference_data: Optional[pd.DataFrame]
    ) -> List[str]:
        """予測日付の生成"""
        if reference_data is not None and len(reference_data) > 0:
            last_date = reference_data.index[-1]
            start_date = last_date + timedelta(days=1)
        else:
            start_date = datetime.now() + timedelta(days=1)
        
        dates = []
        for i in range(count):
            date = start_date + timedelta(days=i)
            dates.append(date.strftime('%Y-%m-%d'))
        
        return dates
    
    def _normalize_date_string(self, date_str: str) -> str:
        """日付文字列の正規化"""
        try:
            # 様々な形式に対応
            if isinstance(date_str, str):
                # ISO形式への変換を試行
                date_obj = pd.to_datetime(date_str)
                return date_obj.strftime('%Y-%m-%d')
            else:
                return str(date_str)
        except:
            return str(date_str)
    
    def _process_current_metrics(self, raw_metrics: Dict) -> Any:
        """現在メトリクスの処理"""
        from .types import CurrentMetrics
        
        return CurrentMetrics(
            realized_volatility=max(0, raw_metrics.get('realized_volatility', 0)),
            vix_like_indicator=max(0, raw_metrics.get('vix_like_indicator', 20)),
            volatility_regime=raw_metrics.get('volatility_regime', 'medium_vol')
        )
    
    def _process_ensemble_forecast(self, raw_ensemble: Dict) -> Any:
        """アンサンブル予測の処理"""
        from .types import EnsembleForecast
        
        individual_forecasts = raw_ensemble.get('individual_forecasts', {})
        # 数値型に変換
        processed_forecasts = {}
        for model, value in individual_forecasts.items():
            try:
                processed_forecasts[model] = float(value)
            except (ValueError, TypeError):
                processed_forecasts[model] = 0.0
        
        return EnsembleForecast(
            ensemble_volatility=max(0, raw_ensemble.get('ensemble_volatility', 20)),
            ensemble_confidence=max(0, min(1, 
                raw_ensemble.get('ensemble_confidence', 0.5)
            )),
            individual_forecasts=processed_forecasts
        )
    
    def _process_risk_assessment(self, raw_risk: Dict) -> Any:
        """リスク評価の処理"""
        from .types import RiskAssessment
        
        risk_level_str = raw_risk.get('risk_level', 'MEDIUM')
        try:
            risk_level = RiskLevel(risk_level_str)
        except ValueError:
            risk_level = RiskLevel.MEDIUM
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_score=max(0, min(100, raw_risk.get('risk_score', 50))),
            risk_factors=raw_risk.get('risk_factors', [])
        )
    
    def _process_investment_implications(self, raw_implications: Dict) -> Dict[str, List[str]]:
        """投資示唆の処理"""
        processed = {}
        for category, suggestions in raw_implications.items():
            if isinstance(suggestions, list):
                processed[category] = [str(s) for s in suggestions]
            else:
                processed[category] = [str(suggestions)] if suggestions else []
        
        return processed
    
    def _process_timeframe_data(self, tf_key: str, tf_data: Dict) -> Optional[Any]:
        """時間軸データの処理"""
        try:
            from .types import TimeframeData, TechnicalIndicators, TrendDirection
            
            # テクニカル指標の処理
            indicators_raw = tf_data.get('technical_indicators', {})
            technical_indicators = TechnicalIndicators(
                rsi=indicators_raw.get('rsi'),
                macd=indicators_raw.get('macd'), 
                bb_position=indicators_raw.get('bb_position'),
                sma_20=indicators_raw.get('sma_20'),
                sma_50=indicators_raw.get('sma_50'),
                bollinger_upper=indicators_raw.get('bollinger_upper'),
                bollinger_lower=indicators_raw.get('bollinger_lower')
            )
            
            # トレンド方向の処理
            trend_str = tf_data.get('trend_direction', 'unknown')
            try:
                trend_direction = TrendDirection(trend_str)
            except ValueError:
                trend_direction = TrendDirection.UNKNOWN
            
            return TimeframeData(
                timeframe=tf_data.get('timeframe', tf_key),
                trend_direction=trend_direction,
                trend_strength=max(0, min(100, tf_data.get('trend_strength', 50))),
                technical_indicators=technical_indicators,
                support_level=tf_data.get('support_level'),
                resistance_level=tf_data.get('resistance_level')
            )
            
        except Exception as e:
            logger.error(f"時間軸データ処理エラー ({tf_key}): {e}")
            return None
    
    def _process_integrated_analysis(self, raw_integrated: Dict) -> Optional[Any]:
        """統合分析の処理"""
        try:
            from .types import (
                IntegratedAnalysis, IntegratedSignal, InvestmentRecommendation,
                TrendDirection, TradingAction, SignalStrength, PositionSize
            )
            
            # 統合シグナルの処理
            signal_raw = raw_integrated.get('integrated_signal', {})
            integrated_signal = IntegratedSignal(
                action=TradingAction(signal_raw.get('action', 'HOLD')),
                strength=SignalStrength(signal_raw.get('strength', 'WEAK')),
                signal_score=signal_raw.get('signal_score', 0)
            )
            
            # 投資推奨の処理
            rec_raw = raw_integrated.get('investment_recommendation', {})
            investment_recommendation = InvestmentRecommendation(
                recommendation=TradingAction(rec_raw.get('recommendation', 'HOLD')),
                position_size=PositionSize(rec_raw.get('position_size', 'NEUTRAL')),
                confidence=rec_raw.get('confidence', 0),
                reasons=rec_raw.get('reasons', []),
                stop_loss_suggestion=rec_raw.get('stop_loss_suggestion'),
                take_profit_suggestion=rec_raw.get('take_profit_suggestion')
            )
            
            return IntegratedAnalysis(
                overall_trend=TrendDirection(raw_integrated.get('overall_trend', 'unknown')),
                trend_confidence=raw_integrated.get('trend_confidence', 0),
                consistency_score=raw_integrated.get('consistency_score', 0),
                integrated_signal=integrated_signal,
                investment_recommendation=investment_recommendation
            )
            
        except Exception as e:
            logger.error(f"統合分析処理エラー: {e}")
            return None
    
    def _extract_lstm_signal(self, avg_return: float) -> TradingAction:
        """LSTM平均リターンからシグナル抽出"""
        if avg_return > 1.0:
            return TradingAction.BUY
        elif avg_return < -1.0:
            return TradingAction.SELL
        else:
            return TradingAction.HOLD
    
    def _create_final_assessment(self, merged_results: Dict) -> Dict[str, Any]:
        """最終評価の作成"""
        signals = merged_results.get('integrated_signals', [])
        recommendations = merged_results.get('recommendations', [])
        
        # シグナル集計
        buy_count = sum(1 for s in signals if s['signal'] == TradingAction.BUY)
        sell_count = sum(1 for s in signals if s['signal'] == TradingAction.SELL)
        hold_count = len(signals) - buy_count - sell_count
        
        # 最終判定
        if buy_count > sell_count and buy_count > hold_count:
            final_action = TradingAction.BUY
        elif sell_count > buy_count and sell_count > hold_count:
            final_action = TradingAction.SELL
        else:
            final_action = TradingAction.HOLD
        
        # 平均信頼度計算
        confidences = [s.get('confidence', 0) for s in signals if s.get('confidence')]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'final_action': final_action,
            'confidence': avg_confidence,
            'signal_distribution': {
                'buy': buy_count,
                'sell': sell_count, 
                'hold': hold_count
            },
            'total_analyses': len(merged_results.get('available_analyses', [])),
            'risk_factor_count': len(merged_results.get('risk_factors', []))
        }