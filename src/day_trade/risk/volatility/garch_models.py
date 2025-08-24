#!/usr/bin/env python3
"""
GARCHモデルモジュール

GARCH、GJR-GARCH、EGARCHモデルの適合・予測機能を提供します。
ボラティリティクラスタリングと時変ボラティリティの高精度モデリング。
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional

from .base import VolatilityEngineBase, ARCH_AVAILABLE
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

if ARCH_AVAILABLE:
    from arch import arch_model
    from arch.univariate import EGARCH, GARCH, GJR


class GARCHModelEngine(VolatilityEngineBase):
    """
    GARCHモデル適合・予測エンジン

    複数のGARCH系モデル（GARCH、GJR-GARCH、EGARCH）による
    ボラティリティモデリングと予測を提供します。
    """

    def __init__(self, model_cache_dir: str = "data/volatility_models"):
        """
        初期化

        Args:
            model_cache_dir: モデルキャッシュディレクトリ
        """
        super().__init__(model_cache_dir)
        self.supported_models = ["GARCH", "GJR-GARCH", "EGARCH"]
        logger.info("GARCHモデルエンジン初期化完了")

    def fit_garch_model(
        self,
        data: pd.DataFrame,
        model_type: str = "GARCH",
        p: int = 1,
        q: int = 1,
        symbol: str = "UNKNOWN",
    ) -> Optional[Dict]:
        """
        GARCHモデル適合

        Args:
            data: 価格データ
            model_type: モデルタイプ ('GARCH', 'GJR-GARCH', 'EGARCH')
            p: ARCH次数
            q: GARCH次数
            symbol: 銘柄コード

        Returns:
            適合結果辞書
        """
        if not self._validate_dependencies(["arch"]):
            return None

        if model_type not in self.supported_models:
            logger.error(f"サポートされていないモデルタイプ: {model_type}")
            return None

        try:
            # リターン計算と前処理
            returns = self._prepare_returns(data)
            if returns is None:
                return None

            # モデル定義
            model = self._create_garch_model(returns, model_type, p, q)
            if model is None:
                return None

            # モデル適合
            logger.info(f"GARCHモデル適合開始: {model_type}({p},{q}) - {symbol}")
            fitted_model = model.fit(disp="off", show_warning=False)

            # 結果処理
            result = self._process_garch_results(fitted_model, symbol, model_type, p, q)
            
            # モデル保存
            self.garch_models[symbol] = fitted_model
            
            logger.info(f"GARCHモデル適合完了: {model_type} AIC={result['aic']:.2f}")
            return result

        except Exception as e:
            logger.error(f"GARCHモデル適合エラー ({symbol}): {e}")
            return None

    def _prepare_returns(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """
        GARCHモデル用のリターン前処理

        Args:
            data: 価格データ

        Returns:
            前処理済みリターン系列
        """
        try:
            # リターン計算
            returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()

            if len(returns) < 100:
                logger.warning(
                    f"GARCHモデルには最低100データポイントが必要: {len(returns)}"
                )
                return None

            # リターンをパーセンテージに変換（数値安定性のため）
            returns = returns * 100
            
            return returns

        except Exception as e:
            logger.error(f"リターン前処理エラー: {e}")
            return None

    def _create_garch_model(
        self, returns: pd.Series, model_type: str, p: int, q: int
    ):
        """
        GARCHモデル定義作成

        Args:
            returns: リターン系列
            model_type: モデルタイプ
            p: ARCH次数
            q: GARCH次数

        Returns:
            GARCHモデルオブジェクト
        """
        try:
            if model_type == "GARCH":
                return arch_model(returns, vol="Garch", p=p, q=q, rescale=False)
            elif model_type == "GJR-GARCH":
                return arch_model(returns, vol="GARCH", p=p, o=1, q=q, rescale=False)
            elif model_type == "EGARCH":
                return arch_model(returns, vol="EGARCH", p=p, q=q, rescale=False)
            else:
                raise ValueError(f"サポートされていないモデルタイプ: {model_type}")

        except Exception as e:
            logger.error(f"GARCHモデル定義エラー: {e}")
            return None

    def _process_garch_results(
        self, fitted_model, symbol: str, model_type: str, p: int, q: int
    ) -> Dict:
        """
        GARCH適合結果の処理

        Args:
            fitted_model: 適合済みモデル
            symbol: 銘柄コード
            model_type: モデルタイプ
            p: ARCH次数
            q: GARCH次数

        Returns:
            処理済み結果辞書
        """
        # 適合統計
        aic = fitted_model.aic
        bic = fitted_model.bic
        log_likelihood = fitted_model.loglikelihood

        # パラメータ抽出
        params = fitted_model.params

        # 1期先予測
        forecast = fitted_model.forecast(horizon=1, reindex=False)
        next_period_vol = (
            np.sqrt(forecast.variance.iloc[-1, 0]) / 100
        )  # パーセンテージから戻す

        return {
            "symbol": symbol,
            "model_type": model_type,
            "p": p,
            "q": q,
            "aic": float(aic),
            "bic": float(bic),
            "log_likelihood": float(log_likelihood),
            "parameters": {k: float(v) for k, v in params.items()},
            "next_period_volatility": float(next_period_vol),
            "fitted_volatility": fitted_model.conditional_volatility / 100,  # 時系列
            "data_points": len(fitted_model.resid),
            "fitting_timestamp": datetime.now().isoformat(),
        }

    def predict_garch_volatility(
        self, symbol: str, horizon: int = 5, confidence_level: float = 0.95
    ) -> Optional[Dict]:
        """
        GARCHモデルによるボラティリティ予測

        Args:
            symbol: 銘柄コード
            horizon: 予測期間
            confidence_level: 信頼水準

        Returns:
            予測結果辞書
        """
        if not self._validate_dependencies(["arch"]) or symbol not in self.garch_models:
            logger.error(f"GARCHモデルが利用できません: {symbol}")
            return None

        try:
            fitted_model = self.garch_models[symbol]

            # 複数期間予測
            forecast = fitted_model.forecast(horizon=horizon, reindex=False)

            # 分散予測を標準偏差（ボラティリティ）に変換
            vol_forecast = np.sqrt(
                forecast.variance.iloc[-1] / 10000
            )  # パーセンテージ調整

            # 信頼区間計算
            confidence_intervals = self._calculate_confidence_intervals(
                vol_forecast, confidence_level
            )

            result = {
                "symbol": symbol,
                "forecast_horizon": horizon,
                "confidence_level": confidence_level,
                "volatility_forecast": [float(x) for x in vol_forecast],
                "volatility_upper": confidence_intervals["upper"],
                "volatility_lower": confidence_intervals["lower"],
                "forecast_dates": [
                    (datetime.now() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                    for i in range(horizon)
                ],
                "prediction_timestamp": datetime.now().isoformat(),
            }

            logger.info(f"GARCH予測完了: {symbol} {horizon}期間先")
            return result

        except Exception as e:
            logger.error(f"GARCH予測エラー ({symbol}): {e}")
            return None

    def _calculate_confidence_intervals(
        self, vol_forecast: pd.Series, confidence_level: float
    ) -> Dict:
        """
        予測信頼区間計算

        Args:
            vol_forecast: ボラティリティ予測
            confidence_level: 信頼水準

        Returns:
            上限・下限を含む信頼区間辞書
        """
        from scipy import stats

        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)

        # 予測の不確実性（簡易版）
        vol_std = (
            vol_forecast.std()
            if len(vol_forecast) > 1
            else vol_forecast.iloc[0] * 0.1
        )

        vol_upper = vol_forecast + z_score * vol_std
        vol_lower = np.maximum(
            vol_forecast - z_score * vol_std, 0.001
        )  # 負の値を避ける

        return {
            "upper": [float(x) for x in vol_upper],
            "lower": [float(x) for x in vol_lower],
        }

    def compare_garch_models(
        self, data: pd.DataFrame, symbol: str = "UNKNOWN"
    ) -> Dict:
        """
        複数GARCHモデルの比較

        Args:
            data: 価格データ
            symbol: 銘柄コード

        Returns:
            モデル比較結果
        """
        comparison_results = {
            "symbol": symbol,
            "models": {},
            "best_model": None,
            "comparison_timestamp": datetime.now().isoformat(),
        }

        # 各モデルを適合
        for model_type in self.supported_models:
            try:
                result = self.fit_garch_model(data, model_type=model_type, symbol=f"{symbol}_{model_type}")
                if result:
                    comparison_results["models"][model_type] = result
            except Exception as e:
                logger.warning(f"{model_type}モデル適合失敗: {e}")

        # 最良モデル選択（AIC基準）
        if comparison_results["models"]:
            best_model = min(
                comparison_results["models"].keys(),
                key=lambda x: comparison_results["models"][x]["aic"]
            )
            comparison_results["best_model"] = best_model
            
            logger.info(f"最良GARCHモデル: {best_model} (AIC={comparison_results['models'][best_model]['aic']:.2f})")

        return comparison_results

    def get_model_diagnostics(self, symbol: str) -> Optional[Dict]:
        """
        モデル診断情報取得

        Args:
            symbol: 銘柄コード

        Returns:
            診断情報辞書
        """
        if symbol not in self.garch_models:
            logger.error(f"モデルが存在しません: {symbol}")
            return None

        try:
            fitted_model = self.garch_models[symbol]
            
            diagnostics = {
                "symbol": symbol,
                "model_summary": str(fitted_model.summary()),
                "residuals_stats": {
                    "mean": float(fitted_model.resid.mean()),
                    "std": float(fitted_model.resid.std()),
                    "skewness": float(fitted_model.resid.skew()),
                    "kurtosis": float(fitted_model.resid.kurt()),
                },
                "volatility_stats": {
                    "mean": float(fitted_model.conditional_volatility.mean()),
                    "std": float(fitted_model.conditional_volatility.std()),
                    "min": float(fitted_model.conditional_volatility.min()),
                    "max": float(fitted_model.conditional_volatility.max()),
                },
                "model_info": {
                    "loglikelihood": float(fitted_model.loglikelihood),
                    "aic": float(fitted_model.aic),
                    "bic": float(fitted_model.bic),
                    "num_params": len(fitted_model.params),
                },
            }

            return diagnostics

        except Exception as e:
            logger.error(f"診断情報取得エラー ({symbol}): {e}")
            return {"symbol": symbol, "error": str(e)}