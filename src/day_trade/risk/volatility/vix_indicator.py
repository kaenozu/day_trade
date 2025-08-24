#!/usr/bin/env python3
"""
VIX風指標計算モジュール

VIX（恐怖指数）に類似した指標の計算とボラティリティレジーム分類を提供します。
簡易GARCH(1,1)による条件付きボラティリティと前向き投影機能。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .base import VolatilityEngineBase
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class VIXIndicatorCalculator(VolatilityEngineBase):
    """
    VIX風指標計算クラス

    VIX風指標の計算、レジーム分類、前向き投影を提供します。
    """

    def __init__(self, model_cache_dir: str = "data/volatility_models"):
        """
        初期化

        Args:
            model_cache_dir: モデルキャッシュディレクトリ
        """
        super().__init__(model_cache_dir)
        
        # VIXレジーム分類閾値
        self.vix_regime_thresholds = {
            "complacency": 15,  # 楽観・自己満足
            "normal": 25,       # 通常
            "concern": 35,      # 懸念
            "fear": 50,         # 恐怖
            "panic": float("inf")  # パニック
        }
        
        logger.info("VIX風指標計算器初期化完了")

    def calculate_vix_like_indicator(
        self, 
        data: pd.DataFrame, 
        window: int = 30, 
        garch_params: Optional[Dict] = None
    ) -> pd.Series:
        """
        VIX風指標計算

        Args:
            data: OHLCV価格データ
            window: 計算ウィンドウ
            garch_params: GARCHパラメータ

        Returns:
            VIX風指標系列（0-100スケール）
        """
        try:
            if garch_params is None:
                garch_params = self.get_default_vix_params()

            returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()

            if len(returns) < window:
                logger.warning(
                    f"VIX計算にはwindow以上のデータが必要: {len(returns)} < {window}"
                )
                return pd.Series([20] * len(data), index=data.index)

            # 簡易GARCH(1,1)による条件付きボラティリティ
            conditional_vol = self._calculate_conditional_volatility(
                returns, garch_params
            )

            # 年率化してVIXスケール（0-100）に変換
            vix_like = self._convert_to_vix_scale(conditional_vol)

            # データフレーム全体のインデックスに拡張
            result = self._extend_to_full_index(data, returns, vix_like, window)

            logger.info("VIX風指標計算完了")
            return result

        except Exception as e:
            logger.error(f"VIX風指標計算エラー: {e}")
            # フォールバック: 単純な移動ボラティリティ
            returns = data["Close"].pct_change()
            simple_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
            return simple_vol.fillna(20)

    def _calculate_conditional_volatility(
        self, returns: pd.Series, garch_params: Dict
    ) -> pd.Series:
        """
        簡易GARCH(1,1)による条件付きボラティリティ計算

        Args:
            returns: リターン系列
            garch_params: GARCHパラメータ

        Returns:
            条件付きボラティリティ系列
        """
        alpha = garch_params["garch_alpha"]
        beta = garch_params["garch_beta"]
        omega = garch_params["garch_omega"]

        conditional_vol = pd.Series(index=returns.index, dtype=float)

        # 初期値設定
        unconditional_vol = returns.std()
        conditional_vol.iloc[0] = unconditional_vol

        for i in range(1, len(returns)):
            if i < len(conditional_vol):
                prev_return_sq = returns.iloc[i - 1] ** 2
                prev_vol_sq = (
                    conditional_vol.iloc[i - 1] ** 2
                    if conditional_vol.iloc[i - 1] > 0
                    else unconditional_vol**2
                )

                conditional_variance = (
                    omega + alpha * prev_return_sq + beta * prev_vol_sq
                )
                conditional_vol.iloc[i] = np.sqrt(max(conditional_variance, 1e-6))

        return conditional_vol

    def _convert_to_vix_scale(self, conditional_vol: pd.Series) -> pd.Series:
        """
        ボラティリティをVIXスケールに変換

        Args:
            conditional_vol: 条件付きボラティリティ

        Returns:
            VIXスケール系列
        """
        # 年率化してVIXスケール（0-100）に変換
        annual_vol = conditional_vol * np.sqrt(252)
        vix_like = annual_vol * 100

        # 極端な値をクリッピング
        vix_like = np.clip(vix_like, 5, 150)
        
        return vix_like

    def _extend_to_full_index(
        self, 
        data: pd.DataFrame, 
        returns: pd.Series, 
        vix_like: pd.Series, 
        window: int
    ) -> pd.Series:
        """
        VIX風指標をデータフレーム全体のインデックスに拡張

        Args:
            data: 元の価格データ
            returns: リターン系列
            vix_like: VIX風指標
            window: 計算ウィンドウ

        Returns:
            拡張されたVIX風指標系列
        """
        result = pd.Series(index=data.index, dtype=float)

        # 初期期間は移動平均で埋める
        if len(returns) >= window:
            initial_vol = returns.iloc[:window].std() * np.sqrt(252) * 100
            result.iloc[: len(result) - len(vix_like)] = initial_vol

        # GARCH結果を適用
        result.iloc[len(result) - len(vix_like) :] = vix_like.values

        # NaN値を前方埋め
        result = result.fillna(method="ffill").fillna(20)

        return result

    def classify_vix_regime(self, vix_value: float) -> str:
        """
        VIX値によるレジーム分類

        Args:
            vix_value: VIX値

        Returns:
            レジーム分類文字列
        """
        if vix_value < self.vix_regime_thresholds["complacency"]:
            return "complacency"
        elif vix_value < self.vix_regime_thresholds["normal"]:
            return "normal"
        elif vix_value < self.vix_regime_thresholds["concern"]:
            return "concern"
        elif vix_value < self.vix_regime_thresholds["fear"]:
            return "fear"
        else:
            return "panic"

    def project_vix_forward(
        self, vix_series: pd.Series, horizon: int
    ) -> List[float]:
        """
        VIX風指標の前方投影

        Args:
            vix_series: VIX風指標系列
            horizon: 予測期間

        Returns:
            前方投影されたVIX値のリスト
        """
        try:
            if len(vix_series) < 20:
                return (
                    [vix_series.iloc[-1]] * horizon
                    if len(vix_series) > 0
                    else [20] * horizon
                )

            # 短期移動平均トレンド
            recent_trend = vix_series.rolling(5).mean().diff().iloc[-1]
            current_vix = vix_series.iloc[-1]

            # 平均回帰要素
            long_term_mean = (
                vix_series.rolling(60).mean().iloc[-1] 
                if len(vix_series) >= 60 
                else 20
            )
            mean_reversion_rate = 0.1  # 平均回帰速度

            forecast = []
            for i in range(horizon):
                # トレンド継続 + 平均回帰
                trend_component = recent_trend * (0.9**i)  # トレンド減衰
                mean_reversion = (
                    (long_term_mean - current_vix) * mean_reversion_rate * (i + 1)
                )

                projected_vix = current_vix + trend_component + mean_reversion
                projected_vix = max(5, min(projected_vix, 100))  # 範囲制限

                forecast.append(float(projected_vix))
                current_vix = projected_vix  # 次の期間の基準値更新

            return forecast

        except Exception as e:
            logger.error(f"VIX前方投影エラー: {e}")
            return [20] * horizon

    def calculate_vix_statistics(
        self, data: pd.DataFrame, window: int = 30
    ) -> Dict:
        """
        VIX風指標の統計情報計算

        Args:
            data: 価格データ
            window: 計算ウィンドウ

        Returns:
            VIX統計情報辞書
        """
        try:
            vix_like = self.calculate_vix_like_indicator(data, window=window)
            
            current_vix = vix_like.iloc[-1] if len(vix_like) > 0 else 20
            current_regime = self.classify_vix_regime(current_vix)
            
            # 前方投影
            forecast = self.project_vix_forward(vix_like, horizon=10)
            
            # レジーム別統計
            regime_stats = {}
            for regime in ["complacency", "normal", "concern", "fear", "panic"]:
                regime_mask = vix_like.apply(lambda x: self.classify_vix_regime(x) == regime)
                regime_count = regime_mask.sum()
                regime_stats[regime] = {
                    "count": int(regime_count),
                    "percentage": float(regime_count / len(vix_like) * 100) if len(vix_like) > 0 else 0
                }

            statistics = {
                "current_vix": float(current_vix),
                "current_regime": current_regime,
                "vix_statistics": {
                    "mean": float(vix_like.mean()),
                    "std": float(vix_like.std()),
                    "min": float(vix_like.min()),
                    "max": float(vix_like.max()),
                    "percentiles": {
                        "25th": float(vix_like.quantile(0.25)),
                        "50th": float(vix_like.quantile(0.50)),
                        "75th": float(vix_like.quantile(0.75)),
                        "95th": float(vix_like.quantile(0.95)),
                    }
                },
                "regime_distribution": regime_stats,
                "vix_forecast": forecast,
                "forecast_regime": self.classify_vix_regime(forecast[0]) if forecast else "unknown",
            }

            return statistics

        except Exception as e:
            logger.error(f"VIX統計計算エラー: {e}")
            return {"error": str(e)}

    def get_vix_signals(self, data: pd.DataFrame) -> Dict:
        """
        VIX風指標に基づくトレーディングシグナル生成

        Args:
            data: 価格データ

        Returns:
            シグナル情報辞書
        """
        try:
            vix_like = self.calculate_vix_like_indicator(data)
            current_vix = vix_like.iloc[-1] if len(vix_like) > 0 else 20
            current_regime = self.classify_vix_regime(current_vix)
            
            # 短期・長期移動平均
            vix_ma_short = vix_like.rolling(5).mean().iloc[-1]
            vix_ma_long = vix_like.rolling(20).mean().iloc[-1]
            
            signals = {
                "current_vix": float(current_vix),
                "current_regime": current_regime,
                "signals": [],
                "risk_level": "MEDIUM",
            }

            # シグナル生成ロジック
            if current_vix > 40:
                signals["signals"].append("極端な恐怖水準 - 逆張り機会")
                signals["risk_level"] = "HIGH"
            elif current_vix > 30:
                signals["signals"].append("高い恐怖水準 - 慎重な逆張り検討")
                signals["risk_level"] = "HIGH"
            elif current_vix < 15:
                signals["signals"].append("楽観的水準 - リスクオフ準備")
                signals["risk_level"] = "MEDIUM"

            # トレンドシグナル
            if vix_ma_short > vix_ma_long * 1.1:
                signals["signals"].append("VIX上昇トレンド - 市場ストレス増加")
            elif vix_ma_short < vix_ma_long * 0.9:
                signals["signals"].append("VIX下降トレンド - 市場安定化")

            # ボラティリティスパイク検出
            vix_change = (current_vix - vix_like.iloc[-2]) if len(vix_like) > 1 else 0
            if vix_change > 5:
                signals["signals"].append("急激なVIX上昇 - 市場ショック発生")

            return signals

        except Exception as e:
            logger.error(f"VIXシグナル生成エラー: {e}")
            return {"error": str(e)}

    def update_vix_regime_thresholds(self, **kwargs) -> None:
        """
        VIXレジーム分類閾値の更新

        Args:
            **kwargs: 更新する閾値
        """
        for regime, threshold in kwargs.items():
            if regime in self.vix_regime_thresholds:
                self.vix_regime_thresholds[regime] = threshold
                logger.info(f"VIXレジーム閾値更新: {regime}={threshold}")
            else:
                logger.warning(f"未知のレジーム: {regime}")