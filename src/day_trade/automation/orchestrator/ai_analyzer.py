"""
AI分析エンジン
機械学習による単一銘柄分析と高度分析機能
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger
from ..analysis_only_engine import AnalysisOnlyEngine
from .types import AIAnalysisResult

# データレスポンス型定義（CI環境対応）
try:
    from ...data.batch_data_fetcher import DataResponse
except ImportError:
    DataResponse = None

logger = get_context_logger(__name__)


class AIAnalyzer:
    """AI分析エンジンクラス"""

    def __init__(self, core, config, stock_helper):
        """
        初期化

        Args:
            core: オーケストレーターコア
            config: 設定オブジェクト
            stock_helper: 銘柄ヘルパー
        """
        self.core = core
        self.config = config
        self.stock_helper = stock_helper

    def analyze_single_symbol(
        self,
        symbol: str,
        data_response: Optional[DataResponse],
        analysis_type: str,
        include_predictions: bool,
    ) -> Dict:
        """
        単一銘柄AI分析

        Args:
            symbol: 銘柄コード
            data_response: データレスポンス
            analysis_type: 分析タイプ
            include_predictions: 予測を含むか

        Returns:
            分析結果辞書
        """
        start_time = time.time()

        try:
            # データ品質チェック
            if not data_response or not data_response.success:
                stock_display = self.stock_helper.format_stock_display(symbol)
                return {
                    "success": False,
                    "errors": [f"{stock_display}: データ取得失敗"],
                    "analysis": None,
                    "signals": [],
                    "alerts": [],
                }

            market_data = data_response.data
            data_quality = data_response.data_quality_score

            # データ品質閾値チェック
            stock_display = self.stock_helper.format_stock_display(symbol)
            if data_quality < self.config.data_quality_threshold:
                logger.warning(f"{stock_display}: データ品質不足 ({data_quality:.1f})")

            # 基本分析エンジン実行
            if symbol not in self.core.analysis_engines:
                self.core.analysis_engines[symbol] = AnalysisOnlyEngine([symbol])

            engine = self.core.analysis_engines[symbol]
            basic_status = engine.get_status()
            market_summary = engine.get_market_summary()

            # 高度AI分析実行
            ai_predictions = {}
            confidence_scores = {}

            if include_predictions and self.core.ml_engine and len(market_data) > 100:
                try:
                    # ML予測実行
                    X_sequences, y_sequences = self.core.ml_engine.prepare_data(market_data)

                    if len(X_sequences) > 0:
                        # 最新データで予測
                        latest_sequence = X_sequences[-1:]
                        prediction_result = self.core.ml_engine.predict(latest_sequence)

                        ai_predictions = {
                            "price_direction": (
                                "up" if prediction_result.predictions[0] > 0 else "down"
                            ),
                            "predicted_change": float(prediction_result.predictions[0]),
                            "confidence": (
                                float(prediction_result.confidence[0])
                                if prediction_result.confidence is not None
                                else 0.5
                            ),
                        }

                        confidence_scores = {
                            "ml_model": (
                                float(prediction_result.confidence[0])
                                if prediction_result.confidence is not None
                                else 0.5
                            ),
                            "data_quality": data_quality / 100.0,
                            "overall": (
                                float(prediction_result.confidence[0])
                                if prediction_result.confidence is not None
                                else 0.5
                            )
                            * (data_quality / 100.0),
                        }

                except Exception as e:
                    logger.warning(f"{symbol}: ML予測エラー - {e}")
                    ai_predictions = {"error": str(e)}
                    confidence_scores = {"overall": 0.3}

            # テクニカル分析シグナル
            technical_signals = self._generate_technical_signals(market_data, symbol)

            # ML特徴量サマリー
            ml_features = self._extract_ml_features_summary(market_data)

            # パフォーマンス指標
            performance_metrics = {
                "analysis_time": time.time() - start_time,
                "data_points": len(market_data),
                "feature_count": len(market_data.columns),
                "memory_usage": self._estimate_memory_usage(market_data),
            }

            # リスク評価
            risk_assessment = self._calculate_risk_assessment(
                market_data, ai_predictions, confidence_scores
            )

            # 推奨アクション生成
            recommendation = self._generate_recommendation(
                ai_predictions, confidence_scores, technical_signals, risk_assessment
            )

            # AI分析結果作成
            ai_analysis = AIAnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                predictions=ai_predictions,
                confidence_scores=confidence_scores,
                technical_signals=technical_signals,
                ml_features=ml_features,
                performance_metrics=performance_metrics,
                data_quality=data_quality,
                recommendation=recommendation,
                risk_assessment=risk_assessment,
            )

            return {
                "success": True,
                "errors": [],
                "analysis": ai_analysis,
                "signals": [],
                "alerts": [],
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [f"{symbol}: 分析エラー - {str(e)}"],
                "analysis": None,
                "signals": [],
                "alerts": [],
            }

    def _generate_technical_signals(
        self, data: pd.DataFrame, symbol: str
    ) -> Dict[str, Any]:
        """テクニカル分析シグナル生成"""
        signals = {}

        try:
            if "終値" in data.columns and len(data) >= 50:
                current_price = data["終値"].iloc[-1]

                # 移動平均シグナル
                sma_20 = data["終値"].rolling(20).mean().iloc[-1]
                sma_50 = data["終値"].rolling(50).mean().iloc[-1]

                signals["moving_average"] = {
                    "sma_20_signal": "bullish" if current_price > sma_20 else "bearish",
                    "sma_50_signal": "bullish" if current_price > sma_50 else "bearish",
                    "golden_cross": sma_20 > sma_50,
                    "death_cross": sma_20 < sma_50,
                }

                # RSIシグナル
                if "RSI_14" in data.columns:
                    rsi = data["RSI_14"].iloc[-1]
                    signals["rsi"] = {
                        "value": rsi,
                        "signal": (
                            "oversold"
                            if rsi < 30
                            else "overbought" if rsi > 70 else "neutral"
                        ),
                    }

                # ボラティリティシグナル
                if "volatility_20d" in data.columns:
                    volatility = data["volatility_20d"].iloc[-1]
                    vol_percentile = data["volatility_20d"].rank(pct=True).iloc[-1]

                    signals["volatility"] = {
                        "current": volatility,
                        "percentile": vol_percentile,
                        "regime": (
                            "high"
                            if vol_percentile > 0.8
                            else "low" if vol_percentile < 0.2 else "normal"
                        ),
                    }

        except Exception as e:
            logger.error(f"テクニカルシグナル生成エラー {symbol}: {e}")
            signals = {"error": str(e)}

        return signals

    def _extract_ml_features_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ML特徴量サマリー抽出"""
        features = {}

        try:
            # 基本統計
            numeric_columns = data.select_dtypes(include=[np.number]).columns

            if len(numeric_columns) > 0:
                features["basic_stats"] = {
                    "feature_count": len(numeric_columns),
                    "data_completeness": 1.0
                    - data[numeric_columns].isnull().sum().sum()
                    / (len(data) * len(numeric_columns)),
                    "value_ranges": {
                        col: {
                            "min": float(data[col].min()),
                            "max": float(data[col].max()),
                        }
                        for col in numeric_columns[:5]  # 最初の5列のみ
                    },
                }

            # 時系列特性
            if "終値" in data.columns:
                returns = data["終値"].pct_change()

                features["time_series"] = {
                    "trend": (
                        "upward"
                        if data["終値"].iloc[-1] > data["終値"].iloc[0]
                        else "downward"
                    ),
                    "volatility": float(returns.std()),
                    "sharpe_estimate": (
                        float(returns.mean() / returns.std())
                        if returns.std() > 0
                        else 0
                    ),
                    "max_drawdown": float(
                        (data["終値"] / data["終値"].expanding().max() - 1).min()
                    ),
                }

        except Exception as e:
            logger.error(f"ML特徴量サマリーエラー: {e}")
            features = {"error": str(e)}

        return features

    def _calculate_risk_assessment(
        self,
        data: pd.DataFrame,
        predictions: Dict[str, Any],
        confidence_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """リスク評価計算"""
        risk_assessment = {}

        try:
            # データ品質リスク
            data_completeness = 1.0 - data.isnull().sum().sum() / (
                len(data) * len(data.columns)
            )

            # 予測不確実性リスク
            prediction_risk = 1.0 - confidence_scores.get("overall", 0.5)

            # ボラティリティリスク
            if "終値" in data.columns:
                returns = data["終値"].pct_change()
                volatility_risk = min(returns.std() * 10, 1.0)  # 正規化
            else:
                volatility_risk = 0.5

            # 流動性リスク（出来高ベース）
            if "出来高" in data.columns:
                volume_trend = data["出来高"].rolling(20).mean().pct_change().iloc[-1]
                liquidity_risk = max(0, -volume_trend)  # 出来高減少時にリスク増
            else:
                liquidity_risk = 0.3

            # 総合リスクスコア
            overall_risk = np.mean(
                [
                    data_completeness * 0.2,
                    prediction_risk * 0.3,
                    volatility_risk * 0.3,
                    liquidity_risk * 0.2,
                ]
            )

            risk_assessment = {
                "data_quality_risk": 1.0 - data_completeness,
                "prediction_uncertainty": prediction_risk,
                "volatility_risk": volatility_risk,
                "liquidity_risk": liquidity_risk,
                "overall_risk_score": overall_risk,
                "risk_level": (
                    "high"
                    if overall_risk > 0.7
                    else "medium" if overall_risk > 0.4 else "low"
                ),
            }

        except Exception as e:
            logger.error(f"リスク評価エラー: {e}")
            risk_assessment = {
                "overall_risk_score": 0.5,
                "risk_level": "unknown",
                "error": str(e),
            }

        return risk_assessment

    def _generate_recommendation(
        self,
        predictions: Dict[str, Any],
        confidence_scores: Dict[str, float],
        technical_signals: Dict[str, Any],
        risk_assessment: Dict[str, Any],
    ) -> str:
        """推奨アクション生成"""
        try:
            overall_confidence = confidence_scores.get("overall", 0.5)
            overall_risk = risk_assessment.get("overall_risk_score", 0.5)

            # 信頼度とリスクに基づく推奨
            if (
                overall_confidence > self.config.confidence_threshold
                and overall_risk < 0.4
            ):
                if predictions.get("predicted_change", 0) > 0.02:  # 2%以上の上昇予測
                    return "STRONG_BUY_SIGNAL"
                elif predictions.get("predicted_change", 0) < -0.02:  # 2%以上の下落予測
                    return "STRONG_SELL_SIGNAL"
                else:
                    return "HOLD"
            elif overall_confidence > 0.5 and overall_risk < 0.6:
                if predictions.get("predicted_change", 0) > 0:
                    return "WEAK_BUY_SIGNAL"
                else:
                    return "WEAK_SELL_SIGNAL"
            else:
                return "INSUFFICIENT_CONFIDENCE"

        except Exception as e:
            logger.error(f"推奨生成エラー: {e}")
            return "ANALYSIS_ERROR"

    def _estimate_memory_usage(self, data: pd.DataFrame) -> float:
        """メモリ使用量推定"""
        try:
            return data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        except Exception:
            return 0.0