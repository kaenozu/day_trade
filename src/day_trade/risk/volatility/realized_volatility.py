#!/usr/bin/env python3
"""
実現ボラティリティ計算モジュール

複数の高精度ボラティリティ推定手法を提供します:
- Close-to-close
- Parkinson推定量（高値・安値利用）
- Garman-Klass推定量（OHLC全利用）
- Yang-Zhang推定量（ギャップ考慮、最高精度）
"""

import numpy as np
import pandas as pd
from typing import Optional

from .base import VolatilityEngineBase
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class RealizedVolatilityCalculator(VolatilityEngineBase):
    """
    実現ボラティリティ計算クラス

    多様な推定手法による高精度ボラティリティ計算を提供します。
    """

    def __init__(self, model_cache_dir: str = "data/volatility_models"):
        """
        初期化

        Args:
            model_cache_dir: モデルキャッシュディレクトリ
        """
        super().__init__(model_cache_dir)
        logger.info("実現ボラティリティ計算器初期化完了")

    def calculate_realized_volatility(
        self,
        data: pd.DataFrame,
        method: str = "close_to_close",
        window: int = 30,
        annualize: bool = True,
    ) -> pd.Series:
        """
        実現ボラティリティ計算

        Args:
            data: OHLCV価格データ
            method: 計算方法 ('close_to_close', 'parkinson', 'garman_klass', 'yang_zhang')
            window: 計算ウィンドウ
            annualize: 年率化フラグ

        Returns:
            実現ボラティリティ系列
        """
        try:
            if method == "close_to_close":
                realized_vol = self._close_to_close_volatility(data, window)

            elif method == "parkinson":
                realized_vol = self._parkinson_volatility(data, window)

            elif method == "garman_klass":
                realized_vol = self._garman_klass_volatility(data, window)

            elif method == "yang_zhang":
                realized_vol = self._yang_zhang_volatility(data, window)

            else:
                raise ValueError(f"サポートされていない方法: {method}")

            # 年率化（252営業日）
            if annualize:
                realized_vol = realized_vol * np.sqrt(252)

            logger.info(f"実現ボラティリティ計算完了: {method}法")
            return realized_vol.fillna(0)

        except Exception as e:
            logger.error(f"実現ボラティリティ計算エラー ({method}): {e}")
            # フォールバック: 単純な移動標準偏差
            returns = data["Close"].pct_change()
            fallback_vol = returns.rolling(window=window).std()
            if annualize:
                fallback_vol = fallback_vol * np.sqrt(252)
            return fallback_vol.fillna(0)

    def _close_to_close_volatility(
        self, data: pd.DataFrame, window: int
    ) -> pd.Series:
        """
        Close-to-close ボラティリティ計算

        最も基本的な手法。終値の対数収益率の標準偏差を計算。

        Args:
            data: 価格データ
            window: 計算ウィンドウ

        Returns:
            ボラティリティ系列
        """
        returns = np.log(data["Close"] / data["Close"].shift(1))
        return returns.rolling(window=window).std()

    def _parkinson_volatility(self, data: pd.DataFrame, window: int) -> pd.Series:
        """
        Parkinson推定量によるボラティリティ計算

        高値と安値の情報を利用することで、より効率的な推定を実現。

        Args:
            data: 価格データ
            window: 計算ウィンドウ

        Returns:
            ボラティリティ系列
        """
        hl_ratio = np.log(data["High"] / data["Low"])
        return hl_ratio.rolling(window=window).apply(
            lambda x: np.sqrt(np.sum(x**2) / (4 * np.log(2) * len(x)))
        )

    def _garman_klass_volatility(self, data: pd.DataFrame, window: int) -> pd.Series:
        """
        Garman-Klass推定量によるボラティリティ計算

        OHLC全ての情報を活用した高精度推定。

        Args:
            data: 価格データ
            window: 計算ウィンドウ

        Returns:
            ボラティリティ系列
        """
        ln_ho = np.log(data["High"] / data["Open"])
        ln_lo = np.log(data["Low"] / data["Open"])
        ln_co = np.log(data["Close"] / data["Open"])

        gk_component = (ln_ho * ln_lo) + (0.5 * ln_co**2)
        return gk_component.rolling(window=window).apply(
            lambda x: np.sqrt(np.sum(x) / len(x))
        )

    def _yang_zhang_volatility(self, data: pd.DataFrame, window: int) -> pd.Series:
        """
        Yang-Zhang推定量によるボラティリティ計算

        オーバーナイトギャップを考慮した最も正確な推定量。

        Args:
            data: 価格データ
            window: 計算ウィンドウ

        Returns:
            ボラティリティ系列
        """
        ln_co = np.log(data["Close"] / data["Open"])
        ln_ho = np.log(data["High"] / data["Open"])
        ln_lo = np.log(data["Low"] / data["Open"])
        ln_cc = np.log(data["Close"] / data["Close"].shift(1))
        ln_oc = np.log(data["Open"] / data["Close"].shift(1))

        # オーバーナイトボラティリティ
        overnight_vol = ln_oc.rolling(window=window).var()

        # 開場中ボラティリティ（Rogers-Satchell推定量）
        rs_vol = (
            (ln_ho * (ln_ho - ln_co) + ln_lo * (ln_lo - ln_co))
            .rolling(window=window)
            .mean()
        )

        # クローズ-トゥ-クローズボラティリティ
        cc_vol = ln_cc.rolling(window=window).var()

        # Yang-Zhang組み合わせ
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        return np.sqrt(overnight_vol + k * cc_vol + (1 - k) * rs_vol)

    def calculate_multiple_volatilities(
        self, data: pd.DataFrame, window: int = 30, annualize: bool = True
    ) -> pd.DataFrame:
        """
        複数手法による実現ボラティリティを同時計算

        Args:
            data: 価格データ
            window: 計算ウィンドウ
            annualize: 年率化フラグ

        Returns:
            複数手法の結果を含むDataFrame
        """
        methods = ["close_to_close", "parkinson", "garman_klass", "yang_zhang"]
        results = pd.DataFrame(index=data.index)

        for method in methods:
            try:
                vol = self.calculate_realized_volatility(
                    data, method=method, window=window, annualize=annualize
                )
                results[f"vol_{method}"] = vol
            except Exception as e:
                logger.warning(f"手法 {method} でエラー: {e}")
                results[f"vol_{method}"] = np.nan

        # 平均値とアンサンブル
        valid_cols = [col for col in results.columns if not results[col].isnull().all()]
        if valid_cols:
            results["vol_ensemble"] = results[valid_cols].mean(axis=1)

        logger.info(f"複数手法ボラティリティ計算完了: {len(valid_cols)}手法")
        return results

    def create_volatility_regime_classifier(
        self,
        data: pd.DataFrame,
        thresholds: Optional[dict] = None,
        window: int = 20,
    ) -> pd.Series:
        """
        ボラティリティレジーム分類

        Args:
            data: 価格データ
            thresholds: 分類閾値
            window: ボラティリティ計算ウィンドウ

        Returns:
            レジーム分類結果
        """
        try:
            if thresholds is None:
                thresholds = self.get_default_thresholds()

            # 実現ボラティリティ計算
            realized_vol = self.calculate_realized_volatility(
                data, window=window, annualize=True
            )

            # レジーム分類
            regimes = pd.Series(index=data.index, dtype="object")

            for i in range(len(realized_vol)):
                vol = realized_vol.iloc[i]

                if pd.isna(vol):
                    regimes.iloc[i] = "unknown"
                elif vol < thresholds["low"]:
                    regimes.iloc[i] = "low_vol"
                elif vol < thresholds["medium"]:
                    regimes.iloc[i] = "medium_vol"
                elif vol < thresholds["high"]:
                    regimes.iloc[i] = "high_vol"
                else:
                    regimes.iloc[i] = "extreme_vol"

            logger.info("ボラティリティレジーム分類完了")
            return regimes

        except Exception as e:
            logger.error(f"レジーム分類エラー: {e}")
            return pd.Series(["unknown"] * len(data), index=data.index)

    def get_volatility_statistics(
        self, data: pd.DataFrame, window: int = 30
    ) -> dict:
        """
        ボラティリティ統計情報を取得

        Args:
            data: 価格データ
            window: 計算ウィンドウ

        Returns:
            統計情報辞書
        """
        try:
            multi_vol = self.calculate_multiple_volatilities(data, window=window)
            regimes = self.create_volatility_regime_classifier(data, window=window)

            stats = {
                "current_volatilities": {},
                "average_volatilities": {},
                "regime_distribution": regimes.value_counts().to_dict(),
                "current_regime": regimes.iloc[-1] if len(regimes) > 0 else "unknown",
            }

            # 各手法の現在値と平均値
            for col in multi_vol.columns:
                if not multi_vol[col].isnull().all():
                    stats["current_volatilities"][col] = (
                        float(multi_vol[col].iloc[-1])
                        if pd.notna(multi_vol[col].iloc[-1])
                        else None
                    )
                    stats["average_volatilities"][col] = float(
                        multi_vol[col].mean()
                    )

            return stats

        except Exception as e:
            logger.error(f"ボラティリティ統計計算エラー: {e}")
            return {"error": str(e)}