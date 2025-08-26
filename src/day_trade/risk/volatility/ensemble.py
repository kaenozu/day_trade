#!/usr/bin/env python3
"""
アンサンブル予測エンジン

GARCH、機械学習、VIX風指標を統合した総合ボラティリティ予測:
- 複数モデルの統合
- 重み付きアンサンブル
- 総合予測レポート生成
- 前向き投影機能
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any

from .base import VolatilityEngineBase
from .garch_models import GARCHModelEngine
from .ml_models import MLVolatilityPredictor
from .vix_indicator import VIXIndicatorCalculator
from .realized_volatility import RealizedVolatilityCalculator
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class VolatilityEnsembleEngine(VolatilityEngineBase):
    """
    ボラティリティアンサンブル予測エンジン

    複数の予測手法を統合した高精度ボラティリティ予測を提供します。
    """

    def __init__(self, model_cache_dir: str = "data/volatility_models"):
        """
        初期化

        Args:
            model_cache_dir: モデルキャッシュディレクトリ
        """
        super().__init__(model_cache_dir)

        # 各予測エンジンの初期化
        self.garch_engine = GARCHModelEngine(model_cache_dir)
        self.ml_predictor = MLVolatilityPredictor(model_cache_dir)
        self.vix_calculator = VIXIndicatorCalculator(model_cache_dir)
        self.rv_calculator = RealizedVolatilityCalculator(model_cache_dir)

        # デフォルト重み設定
        self.default_weights = {
            "garch": 0.35,
            "machine_learning": 0.40,
            "vix_like": 0.25,
        }

        logger.info("ボラティリティアンサンブルエンジン初期化完了")

    def generate_comprehensive_volatility_forecast(
        self, 
        data: pd.DataFrame, 
        symbol: str = "UNKNOWN", 
        forecast_horizon: int = 10,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        総合ボラティリティ予測

        GARCH・機械学習・VIX風指標を統合した総合予測

        Args:
            data: 価格データ
            symbol: 銘柄コード
            forecast_horizon: 予測期間
            weights: モデル重み辞書

        Returns:
            総合予測結果辞書
        """
        try:
            if weights is None:
                weights = self.default_weights

            comprehensive_forecast = {
                "symbol": symbol,
                "forecast_horizon": forecast_horizon,
                "forecast_timestamp": datetime.now().isoformat(),
                "models": {},
                "ensemble_forecast": {},
                "current_metrics": {},
                "model_weights": weights,
            }

            # 個別モデル予測の収集
            garch_result = self._get_garch_forecast(data, symbol, forecast_horizon)
            ml_result = self._get_ml_forecast(data, symbol)
            vix_result = self._get_vix_forecast(data, forecast_horizon)

            # 予測結果を統合データに追加
            if garch_result:
                comprehensive_forecast["models"]["garch"] = garch_result

            if ml_result:
                comprehensive_forecast["models"]["machine_learning"] = ml_result

            if vix_result:
                comprehensive_forecast["models"]["vix_like"] = vix_result

            # 現在のメトリクス計算
            comprehensive_forecast["current_metrics"] = self._calculate_current_metrics(data)

            # アンサンブル予測作成
            ensemble = self._create_volatility_ensemble(
                garch_result, ml_result, vix_result, 
                comprehensive_forecast["current_metrics"]["realized_volatility"],
                weights
            )
            comprehensive_forecast["ensemble_forecast"] = ensemble

            logger.info(f"総合ボラティリティ予測完了: {symbol}")
            return comprehensive_forecast

        except Exception as e:
            logger.error(f"総合予測エラー ({symbol}): {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "forecast_timestamp": datetime.now().isoformat(),
            }

    def _get_garch_forecast(
        self, data: pd.DataFrame, symbol: str, horizon: int
    ) -> Optional[Dict]:
        """
        GARCH予測を取得

        Args:
            data: 価格データ
            symbol: 銘柄コード
            horizon: 予測期間

        Returns:
            GARCH予測結果
        """
        try:
            if len(data) >= 100:
                garch_fit = self.garch_engine.fit_garch_model(data, symbol=symbol)
                if garch_fit:
                    garch_result = self.garch_engine.predict_garch_volatility(
                        symbol, horizon=horizon
                    )
                    if garch_result:
                        logger.info("GARCH予測を統合予測に追加")
                        return garch_result
            
            logger.warning("GARCH予測をスキップ（データ不足またはモデル適合失敗）")
            return None

        except Exception as e:
            logger.warning(f"GARCH予測エラー: {e}")
            return None

    def _get_ml_forecast(self, data: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        機械学習予測を取得

        Args:
            data: 価格データ
            symbol: 銘柄コード

        Returns:
            ML予測結果
        """
        try:
            if len(data) >= 150:
                ml_train = self.ml_predictor.train_volatility_ml_model(
                    data, symbol=symbol, target_horizon=5
                )
                if ml_train:
                    ml_result = self.ml_predictor.predict_volatility_ml(
                        data, symbol, horizon=5
                    )
                    if ml_result:
                        logger.info("ML予測を統合予測に追加")
                        return ml_result

            logger.warning("ML予測をスキップ（データ不足またはモデル訓練失敗）")
            return None

        except Exception as e:
            logger.warning(f"ML予測エラー: {e}")
            return None

    def _get_vix_forecast(self, data: pd.DataFrame, horizon: int) -> Dict:
        """
        VIX風予測を取得

        Args:
            data: 価格データ
            horizon: 予測期間

        Returns:
            VIX予測結果
        """
        try:
            vix_like = self.vix_calculator.calculate_vix_like_indicator(data)
            current_vix = vix_like.iloc[-1] if len(vix_like) > 0 else 20
            vix_forecast = self.vix_calculator.project_vix_forward(vix_like, horizon)

            return {
                "current_vix": float(current_vix),
                "vix_forecast": vix_forecast,
                "vix_regime": self.vix_calculator.classify_vix_regime(current_vix),
            }

        except Exception as e:
            logger.warning(f"VIX予測エラー: {e}")
            return {
                "current_vix": 20.0,
                "vix_forecast": [20.0] * horizon,
                "vix_regime": "normal",
            }

    def _calculate_current_metrics(self, data: pd.DataFrame) -> Dict:
        """
        現在のボラティリティメトリクス計算

        Args:
            data: 価格データ

        Returns:
            現在のメトリクス辞書
        """
        try:
            # 実現ボラティリティ
            realized_vol = self.rv_calculator.calculate_realized_volatility(
                data, window=20, annualize=True
            )
            current_realized = realized_vol.iloc[-1] if len(realized_vol) > 0 else 0.2

            # VIX風指標
            vix_like = self.vix_calculator.calculate_vix_like_indicator(data)
            current_vix = vix_like.iloc[-1] if len(vix_like) > 0 else 20

            # ボラティリティレジーム
            regimes = self.rv_calculator.create_volatility_regime_classifier(data)
            current_regime = regimes.iloc[-1] if len(regimes) > 0 else "unknown"

            return {
                "realized_volatility": float(current_realized),
                "vix_like_indicator": float(current_vix),
                "volatility_regime": current_regime,
            }

        except Exception as e:
            logger.error(f"現在メトリクス計算エラー: {e}")
            return {
                "realized_volatility": 0.2,
                "vix_like_indicator": 20.0,
                "volatility_regime": "unknown",
            }

    def _create_volatility_ensemble(
        self,
        garch_result: Optional[Dict],
        ml_result: Optional[Dict],
        vix_result: Dict,
        current_realized: float,
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        ボラティリティアンサンブル予測作成

        Args:
            garch_result: GARCH予測結果
            ml_result: ML予測結果
            vix_result: VIX予測結果
            current_realized: 現在の実現ボラティリティ
            weights: モデル重み

        Returns:
            アンサンブル予測辞書
        """
        try:
            forecasts = []
            model_weights = []
            models = []

            # GARCH予測
            if garch_result and "volatility_forecast" in garch_result:
                garch_vol = np.mean(
                    garch_result["volatility_forecast"][:5]
                )  # 最初の5日平均
                forecasts.append(garch_vol * 100)  # パーセンテージ変換
                model_weights.append(weights.get("garch", 0.35))
                models.append("GARCH")

            # ML予測
            if ml_result and "predicted_volatility" in ml_result:
                forecasts.append(ml_result["predicted_volatility"])
                model_weights.append(weights.get("machine_learning", 0.40))
                models.append("Machine Learning")

            # VIX風予測
            if vix_result and "vix_forecast" in vix_result:
                vix_avg = np.mean(vix_result["vix_forecast"][:5])  # 最初の5日平均
                forecasts.append(vix_avg)
                model_weights.append(weights.get("vix_like", 0.25))
                models.append("VIX-like")

            if not forecasts:
                # フォールバック: 現在の実現ボラティリティ
                ensemble_vol = current_realized * 100
                confidence = 0.3
                model_weights = [1.0]
            else:
                # 重み付き平均
                model_weights = np.array(model_weights)
                if model_weights.sum() > 0:
                    model_weights = model_weights / model_weights.sum()  # 正規化
                else:
                    model_weights = np.ones(len(forecasts)) / len(forecasts)

                ensemble_vol = np.average(forecasts, weights=model_weights)

                # 信頼度計算（予測の一致度）
                forecast_std = np.std(forecasts) if len(forecasts) > 1 else 0
                confidence = (
                    max(0.2, 1 - (forecast_std / ensemble_vol))
                    if ensemble_vol > 0
                    else 0.5
                )

            return {
                "ensemble_volatility": float(ensemble_vol),
                "individual_forecasts": dict(zip(models, forecasts)),
                "model_weights": dict(zip(models, model_weights)) if models else {},
                "ensemble_confidence": float(confidence),
                "forecast_range": {
                    "min": float(min(forecasts)) if forecasts else ensemble_vol,
                    "max": float(max(forecasts)) if forecasts else ensemble_vol,
                },
                "forecast_std": float(np.std(forecasts)) if len(forecasts) > 1 else 0.0,
            }

        except Exception as e:
            logger.error(f"アンサンブル作成エラー: {e}")
            return {
                "ensemble_volatility": current_realized * 100,
                "ensemble_confidence": 0.3,
                "error": str(e),
            }

    def optimize_ensemble_weights(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        validation_window: int = 60
    ) -> Dict[str, float]:
        """
        アンサンブル重みの最適化

        Args:
            data: 価格データ
            symbol: 銘柄コード  
            validation_window: 検証ウィンドウ

        Returns:
            最適化された重み辞書
        """
        try:
            if len(data) < validation_window + 100:
                logger.warning("重み最適化には十分なデータが必要")
                return self.default_weights

            # 検証期間の分割
            train_data = data.iloc[:-validation_window]
            validation_data = data.iloc[-validation_window:]

            # 各モデルの予測精度を評価
            model_performances = self._evaluate_model_performances(
                train_data, validation_data, symbol
            )

            # 逆誤差重み付け
            optimized_weights = self._calculate_optimal_weights(model_performances)
            
            logger.info(f"アンサンブル重み最適化完了: {optimized_weights}")
            return optimized_weights

        except Exception as e:
            logger.error(f"重み最適化エラー: {e}")
            return self.default_weights

    def _evaluate_model_performances(
        self, train_data: pd.DataFrame, validation_data: pd.DataFrame, symbol: str
    ) -> Dict[str, float]:
        """
        各モデルのパフォーマンス評価

        Args:
            train_data: 訓練データ
            validation_data: 検証データ
            symbol: 銘柄コード

        Returns:
            モデル別パフォーマンス辞書
        """
        performances = {}

        # 実際のボラティリティ（検証用）
        actual_vol = self.rv_calculator.calculate_realized_volatility(
            validation_data, window=5, annualize=True
        ).mean()

        # GARCH評価
        try:
            garch_result = self._get_garch_forecast(train_data, f"{symbol}_opt", 5)
            if garch_result:
                garch_pred = np.mean(garch_result["volatility_forecast"]) * 100
                performances["garch"] = abs(garch_pred - actual_vol * 100)
        except Exception:
            performances["garch"] = float("inf")

        # ML評価  
        try:
            ml_result = self._get_ml_forecast(train_data, f"{symbol}_opt")
            if ml_result:
                ml_pred = ml_result["predicted_volatility"]
                performances["machine_learning"] = abs(ml_pred - actual_vol * 100)
        except Exception:
            performances["machine_learning"] = float("inf")

        # VIX評価
        try:
            vix_result = self._get_vix_forecast(train_data, 5)
            if vix_result:
                vix_pred = np.mean(vix_result["vix_forecast"])
                performances["vix_like"] = abs(vix_pred - actual_vol * 100)
        except Exception:
            performances["vix_like"] = float("inf")

        return performances

    def _calculate_optimal_weights(self, performances: Dict[str, float]) -> Dict[str, float]:
        """
        パフォーマンスに基づく最適重み計算

        Args:
            performances: モデル別パフォーマンス辞書

        Returns:
            最適化された重み辞書
        """
        # 逆誤差重み付け（誤差が小さいほど重みが大きい）
        weights = {}
        total_inverse_error = 0

        for model, error in performances.items():
            if error != float("inf") and error > 0:
                inverse_error = 1.0 / error
                weights[model] = inverse_error
                total_inverse_error += inverse_error
            else:
                weights[model] = 0.01  # 最小重み

        # 正規化
        if total_inverse_error > 0:
            for model in weights:
                weights[model] = weights[model] / total_inverse_error
        else:
            # フォールバック: 等重み
            n_models = len(weights)
            for model in weights:
                weights[model] = 1.0 / n_models

        return weights

    def get_ensemble_summary(
        self, comprehensive_forecast: Dict
    ) -> Dict[str, Any]:
        """
        アンサンブル予測のサマリー取得

        Args:
            comprehensive_forecast: 総合予測結果

        Returns:
            サマリー辞書
        """
        try:
            ensemble = comprehensive_forecast.get("ensemble_forecast", {})
            current = comprehensive_forecast.get("current_metrics", {})

            summary = {
                "symbol": comprehensive_forecast.get("symbol", "UNKNOWN"),
                "forecast_timestamp": comprehensive_forecast.get("forecast_timestamp"),
                "current_volatility": current.get("realized_volatility", 0) * 100,
                "predicted_volatility": ensemble.get("ensemble_volatility", 0),
                "volatility_change": (
                    ensemble.get("ensemble_volatility", 0) 
                    - current.get("realized_volatility", 0) * 100
                ),
                "confidence_level": ensemble.get("ensemble_confidence", 0),
                "participating_models": list(ensemble.get("individual_forecasts", {}).keys()),
                "forecast_range": ensemble.get("forecast_range", {}),
                "regime": current.get("volatility_regime", "unknown"),
                "vix_level": current.get("vix_like_indicator", 20),
            }

            # 予測の特徴
            vol_change = summary["volatility_change"]
            if vol_change > 5:
                summary["outlook"] = "上昇"
                summary["trend_strength"] = "強い"
            elif vol_change > 2:
                summary["outlook"] = "上昇"
                summary["trend_strength"] = "弱い"  
            elif vol_change < -5:
                summary["outlook"] = "下降"
                summary["trend_strength"] = "強い"
            elif vol_change < -2:
                summary["outlook"] = "下降" 
                summary["trend_strength"] = "弱い"
            else:
                summary["outlook"] = "横ばい"
                summary["trend_strength"] = "安定"

            return summary

        except Exception as e:
            logger.error(f"サマリー生成エラー: {e}")
            return {"error": str(e)}