"""
分析実行エンジンモジュール
AI分析、テクニカル分析、ML予測を統合的に実行
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...automation.analysis_only_engine import AnalysisOnlyEngine
from ...config.trading_mode_config import is_safe_mode
from ...utils.logging_config import get_context_logger
from ...utils.stock_name_helper import get_stock_helper
from .config import AIAnalysisResult, CI_MODE, OrchestrationConfig

logger = get_context_logger(__name__)

# 条件付きMLエンジンインポート
if not CI_MODE:
    try:
        from ...data.advanced_ml_engine import AdvancedMLEngine
        from ...data.batch_data_fetcher import DataResponse
    except ImportError:
        AdvancedMLEngine = None
        DataResponse = None
        logger.warning("MLエンジンまたはバッチデータフェッチャーのインポートに失敗")
else:
    AdvancedMLEngine = None
    DataResponse = None


class AnalysisEngine:
    """
    分析実行エンジン
    
    AI分析、テクニカル分析、ML予測を統合して、
    包括的な市場分析を提供します。
    """

    def __init__(self, config: OrchestrationConfig):
        """
        初期化
        
        Args:
            config: オーケストレーション設定
        """
        self.config = config
        self.stock_helper = get_stock_helper()
        self.analysis_engines: Dict[str, AnalysisOnlyEngine] = {}
        self.ml_engine: Optional[AdvancedMLEngine] = None

    def set_ml_engine(self, ml_engine: Optional[AdvancedMLEngine]) -> None:
        """
        MLエンジンを設定
        
        Args:
            ml_engine: MLエンジンインスタンス
        """
        self.ml_engine = ml_engine

    def analyze_single_symbol(
        self,
        symbol: str,
        data_response: Optional['DataResponse'],
        analysis_type: str,
        include_predictions: bool,
    ) -> Dict:
        """
        単一銘柄AI分析
        
        Args:
            symbol: 銘柄コード
            data_response: データレスポンス
            analysis_type: 分析タイプ
            include_predictions: 予測分析を含むか
            
        Returns:
            Dict: 分析結果辞書
        """
        start_time = time.time()

        try:
            # セーフモードチェック
            if not is_safe_mode():
                return {
                    "success": False,
                    "errors": ["セーフモードでない場合は分析を実行できません"],
                    "analysis": None,
                    "signals": [],
                    "alerts": [],
                }

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
            if symbol not in self.analysis_engines:
                self.analysis_engines[symbol] = AnalysisOnlyEngine([symbol])

            engine = self.analysis_engines[symbol]
            basic_status = engine.get_status()
            market_summary = engine.get_market_summary()

            # 高度AI分析実行
            ai_predictions = {}
            confidence_scores = {}

            if include_predictions and self.ml_engine and len(market_data) > 100:
                try:
                    # ML予測実行
                    X_sequences, y_sequences = self.ml_engine.prepare_data(market_data)

                    if len(X_sequences) > 0:
                        # 最新データで予測
                        latest_sequence = X_sequences[-1:]
                        prediction_result = self.ml_engine.predict(latest_sequence)

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
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [f"{symbol}: 分析エラー - {str(e)}"],
                "analysis": None,
            }

    def _generate_technical_signals(
        self, data: pd.DataFrame, symbol: str
    ) -> Dict[str, Any]:
        """
        テクニカル分析シグナル生成
        
        Args:
            data: 市場データ
            symbol: 銘柄コード
            
        Returns:
            Dict[str, Any]: テクニカルシグナル辞書
        """
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
        """
        ML特徴量サマリー抽出
        
        Args:
            data: 市場データ
            
        Returns:
            Dict[str, Any]: ML特徴量サマリー
        """
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
        """
        リスク評価計算
        
        Args:
            data: 市場データ
            predictions: 予測結果
            confidence_scores: 信頼度スコア
            
        Returns:
            Dict[str, Any]: リスク評価結果
        """
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
        """
        推奨アクション生成
        
        Args:
            predictions: 予測結果
            confidence_scores: 信頼度スコア
            technical_signals: テクニカルシグナル
            risk_assessment: リスク評価
            
        Returns:
            str: 推奨アクション
        """
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
        """
        メモリ使用量推定
        
        Args:
            data: データフレーム
            
        Returns:
            float: メモリ使用量（MB）
        """
        try:
            return data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        except Exception:
            return 0.0

    def cleanup(self) -> Dict[str, Any]:
        """
        分析エンジンのクリーンアップ
        
        Returns:
            Dict[str, Any]: クリーンアップ結果サマリー
        """
        cleanup_summary = {
            "analysis_engines": 0,
            "ml_engine": False,
            "errors": []
        }

        try:
            # 分析エンジンクリーンアップ
            for symbol, engine in self.analysis_engines.items():
                try:
                    if hasattr(engine, "stop"):
                        engine.stop()
                    if hasattr(engine, "close"):
                        engine.close()
                    if hasattr(engine, "cleanup"):
                        engine.cleanup()
                    cleanup_summary["analysis_engines"] += 1
                    logger.debug(f"エンジン {symbol} クリーンアップ完了")
                except Exception as e:
                    error_msg = f"エンジン {symbol} クリーンアップエラー: {e}"
                    logger.warning(error_msg)
                    cleanup_summary["errors"].append(error_msg)

            self.analysis_engines.clear()

            # MLエンジンクリーンアップは外部で管理
            if self.ml_engine:
                self.ml_engine = None
                cleanup_summary["ml_engine"] = True

        except Exception as e:
            error_msg = f"AnalysisEngine クリーンアップ致命的エラー: {e}"
            logger.error(error_msg)
            cleanup_summary["errors"].append(error_msg)

        return cleanup_summary