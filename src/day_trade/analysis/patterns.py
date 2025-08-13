"""
チャートパターン認識エンジン
価格データからチャートパターンを認識する
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression, RANSACRegressor

from ..utils.logging_config import get_context_logger
from .patterns_config import get_patterns_config_class

logger = get_context_logger(__name__, component="chart_patterns")


class ChartPatternRecognizer:
    """チャートパターン認識クラス"""

    def __init__(self):
        """初期化"""
        self.config = get_patterns_config_class()

    def golden_dead_cross(
        self,
        df: pd.DataFrame,
        fast_period: Optional[int] = None,
        slow_period: Optional[int] = None,
        column: str = "Close",
    ) -> pd.DataFrame:
        """
        ゴールデンクロス・デッドクロスの検出

        Args:
            df: 価格データのDataFrame
            fast_period: 短期移動平均の期間
            slow_period: 長期移動平均の期間
            column: 計算対象の列名

        Returns:
            クロスポイントを含むDataFrame
        """
        if fast_period is None:
            fast_period = self.config.get_golden_cross_fast_period()
        if slow_period is None:
            slow_period = self.config.get_golden_cross_slow_period()
        try:
            # 移動平均を計算
            fast_ma = df[column].rolling(window=fast_period).mean()
            slow_ma = df[column].rolling(window=slow_period).mean()

            # クロスポイントを検出
            fast_above = fast_ma > slow_ma
            # pandasの将来警告を回避するため、明示的にbool型で初期化
            with pd.option_context("future.no_silent_downcasting", True):
                fast_above_shifted = fast_above.shift(1).fillna(False)
            golden_cross = fast_above & (~fast_above_shifted)
            dead_cross = (~fast_above) & fast_above_shifted

            # 信頼度スコアを計算（クロス角度に基づく）
            ma_diff = fast_ma - slow_ma
            ma_diff_change = ma_diff.diff()

            confidence_multiplier = self.config.get_golden_cross_confidence_multiplier()
            confidence_clip_max = self.config.get_golden_cross_confidence_clip_max()

            golden_confidence = golden_cross * (
                ma_diff_change.abs() / df[column] * confidence_multiplier
            )
            dead_confidence = dead_cross * (
                ma_diff_change.abs() / df[column] * confidence_multiplier
            )

            return pd.DataFrame(
                {
                    f"Fast_MA_{fast_period}": fast_ma,
                    f"Slow_MA_{slow_period}": slow_ma,
                    "Golden_Cross": golden_cross,
                    "Dead_Cross": dead_cross,
                    "Golden_Confidence": golden_confidence.clip(0, confidence_clip_max),
                    "Dead_Confidence": dead_confidence.clip(0, confidence_clip_max),
                },
                index=df.index,  # indexを追加
            )

        except Exception as e:
            should_log_detailed = self.config.should_log_detailed_errors()
            should_raise = self.config.should_raise_exceptions()

            if should_log_detailed:
                logger.error(f"ゴールデン・デッドクロス検出エラー: {e}", exc_info=True)
            else:
                logger.warning("ゴールデン・デッドクロス検出でエラーが発生しました")

            if should_raise:
                raise

            return (
                pd.DataFrame() if self.config.should_return_empty_on_error() else None
            )

    def support_resistance_levels(
        self,
        df: pd.DataFrame,
        window: Optional[int] = None,
        num_levels: Optional[int] = None,
        column: str = "Close",
    ) -> Dict[str, List[float]]:
        """
        サポート・レジスタンスラインの検出

        Args:
            df: 価格データのDataFrame
            window: 極値検出のウィンドウサイズ
            num_levels: 検出するレベル数
            column: 計算対象の列名

        Returns:
            サポート・レジスタンスレベルの辞書
        """
        if window is None:
            window = self.config.get_support_resistance_window()
        if num_levels is None:
            num_levels = self.config.get_support_resistance_num_levels()
        try:
            prices = df[column].values

            # 極大値と極小値を検出
            max_idx = argrelextrema(prices, np.greater, order=window)[0]
            min_idx = argrelextrema(prices, np.less, order=window)[0]

            # 極値の価格を取得
            resistance_candidates = prices[max_idx] if len(max_idx) > 0 else []
            support_candidates = prices[min_idx] if len(min_idx) > 0 else []

            # 改良されたクラスタリングで主要なレベルを特定
            def cluster_levels(levels, num_clusters):
                if len(levels) == 0:
                    return []

                levels = np.array(levels)
                if len(levels) <= num_clusters:
                    return sorted(levels.tolist(), reverse=True)

                # 統計的手法でクラスタ数を調整（外れ値を除外）
                q1, q3 = np.percentile(levels, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # 外れ値を除外
                filtered_levels = levels[
                    (levels >= lower_bound) & (levels <= upper_bound)
                ]

                if len(filtered_levels) == 0:
                    filtered_levels = levels  # フォールバック

                sorted_levels = np.sort(filtered_levels)
                if len(sorted_levels) <= num_clusters:
                    return sorted(sorted_levels.tolist(), reverse=True)

                # 改良されたK-meansクラスタリング
                # 平均値ではなく中央値を使用して外れ値に強くする
                indices = np.linspace(
                    0, len(sorted_levels) - 1, num_clusters, dtype=int
                )
                centers = sorted_levels[indices]

                clustering_iterations = (
                    self.config.get_support_resistance_clustering_iterations()
                )
                for iteration in range(clustering_iterations):
                    clusters = [[] for _ in range(num_clusters)]

                    # 各レベルを最も近いクラスタに割り当て
                    for level in sorted_levels:
                        distances = np.abs(centers - level)
                        nearest = np.argmin(distances)
                        clusters[nearest].append(level)

                    # クラスタ中心を更新（中央値を使用）
                    new_centers = []
                    for cluster in clusters:
                        if len(cluster) > 0:
                            # 中央値を使用して外れ値の影響を減らす
                            new_centers.append(np.median(cluster))

                    # 収束判定
                    if len(new_centers) == num_clusters:
                        if iteration > 0 and np.allclose(
                            centers, new_centers, rtol=1e-3
                        ):
                            break
                        centers = np.array(new_centers)
                    else:
                        break

                # 改善されたクラスタサイズに基づくフィルタリング
                min_cluster_size = max(1, len(sorted_levels) // (num_clusters * 3))
                final_centers = []
                
                for center in centers:
                    # より効率的な距離計算
                    distances_to_center = np.abs(sorted_levels - center)
                    min_distance_to_other_centers = np.min([np.abs(center - other_center) 
                                                          for other_center in centers if other_center != center] + [float('inf')])
                    
                    cluster_size = np.sum(distances_to_center <= min_distance_to_other_centers / 2)
                    
                    if cluster_size >= min_cluster_size:
                        final_centers.append(center)

                return sorted(final_centers, reverse=True)[:num_clusters]

            resistance_levels = cluster_levels(resistance_candidates, num_levels)
            support_levels = cluster_levels(support_candidates, num_levels)

            return {"resistance": resistance_levels, "support": sorted(support_levels)}

        except Exception as e:
            should_log_detailed = self.config.should_log_detailed_errors()
            should_raise = self.config.should_raise_exceptions()

            if should_log_detailed:
                logger.error(
                    f"サポート・レジスタンスレベル検出エラー: {e}", exc_info=True
                )
            else:
                logger.warning("サポート・レジスタンスレベル検出でエラーが発生しました")

            if should_raise:
                raise

            return (
                {"resistance": [], "support": []}
                if self.config.should_return_empty_on_error()
                else None
            )

    def breakout_detection(
        self,
        df: pd.DataFrame,
        lookback: Optional[int] = None,
        threshold: Optional[float] = None,
        volume_factor: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        ブレイクアウトパターンの検出

        Args:
            df: 価格データのDataFrame（Close, Volume列が必要）
            lookback: 過去を見る期間
            threshold: ブレイクアウト閾値（%）
            volume_factor: ボリューム増加係数

        Returns:
            ブレイクアウトシグナルを含むDataFrame
        """
        if lookback is None:
            lookback = self.config.get_breakout_lookback()
        if threshold is None:
            threshold = self.config.get_breakout_threshold()
        if volume_factor is None:
            volume_factor = self.config.get_breakout_volume_factor()
        try:
            # ローリング最高値・最安値
            rolling_high = df["High"].rolling(window=lookback).max()
            rolling_low = df["Low"].rolling(window=lookback).min()

            # ボリューム移動平均
            volume_ma = df["Volume"].rolling(window=lookback).mean()

            # 上方ブレイクアウト
            upward_breakout = (df["Close"] > rolling_high.shift(1)) & (
                df["Volume"] > volume_ma * volume_factor
            )

            # 下方ブレイクアウト
            downward_breakout = (df["Close"] < rolling_low.shift(1)) & (
                df["Volume"] > volume_ma * volume_factor
            )

            # ブレイクアウトの強度を計算
            upward_strength = np.where(
                upward_breakout,
                ((df["Close"] - rolling_high.shift(1)) / rolling_high.shift(1)) * 100,
                0,
            )

            downward_strength = np.where(
                downward_breakout,
                ((rolling_low.shift(1) - df["Close"]) / rolling_low.shift(1)) * 100,
                0,
            )

            # 信頼度スコア（ブレイクアウトの強度とボリューム増加率に基づく）
            volume_increase = df["Volume"] / volume_ma - 1
            strength_multiplier = self.config.get_breakout_strength_multiplier()
            volume_clip_max = self.config.get_breakout_volume_clip_max()
            confidence_cap = self.config.get_breakout_confidence_cap()

            upward_confidence = np.where(
                upward_breakout,
                np.minimum(
                    (upward_strength * strength_multiplier)
                    * (1 + volume_increase.clip(0, volume_clip_max)),
                    confidence_cap,
                ),
                0,
            )

            downward_confidence = np.where(
                downward_breakout,
                np.minimum(
                    (downward_strength * strength_multiplier)
                    * (1 + volume_increase.clip(0, volume_clip_max)),
                    confidence_cap,
                ),
                0,
            )

            return pd.DataFrame(
                {
                    "Rolling_High": rolling_high,
                    "Rolling_Low": rolling_low,
                    "Upward_Breakout": upward_breakout,
                    "Downward_Breakout": downward_breakout,
                    "Upward_Strength": upward_strength,
                    "Downward_Strength": downward_strength,
                    "Upward_Confidence": upward_confidence,
                    "Downward_Confidence": downward_confidence,
                },
                index=df.index,  # indexを追加
            )

        except Exception as e:
            should_log_detailed = self.config.should_log_detailed_errors()
            should_raise = self.config.should_raise_exceptions()

            if should_log_detailed:
                logger.error(f"ブレイクアウトパターン検出エラー: {e}", exc_info=True)
            else:
                logger.warning("ブレイクアウトパターン検出でエラーが発生しました")

            if should_raise:
                raise

            return (
                pd.DataFrame() if self.config.should_return_empty_on_error() else None
            )

    def trend_line_detection(
        self,
        df: pd.DataFrame,
        window: Optional[int] = None,
        min_touches: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        トレンドラインの検出

        Args:
            df: 価格データのDataFrame
            window: 極値検出のウィンドウサイズ
            min_touches: 最小接触回数

        Returns:
            トレンドライン情報の辞書
        """
        if window is None:
            window = self.config.get_trend_line_window()
        if min_touches is None:
            min_touches = self.config.get_trend_line_min_touches()
        try:
            prices_high = df["High"].values
            prices_low = df["Low"].values

            # 極大値と極小値を検出
            max_idx = argrelextrema(prices_high, np.greater, order=window)[0]
            min_idx = argrelextrema(prices_low, np.less, order=window)[0]

            result = {}

            # 上昇トレンドライン（極小値を結ぶ）
            if len(min_idx) >= min_touches:
                X = min_idx.reshape(-1, 1)
                y = prices_low[min_idx]

                # RANSACで外れ値に頑健なトレンドラインを検出
                try:
                    ransac_residual_threshold = (
                        self.config.get_trend_line_ransac_residual_threshold()
                    )
                    ransac_max_trials = self.config.get_trend_line_ransac_max_trials()
                    ransac_min_samples = self.config.get_trend_line_ransac_min_samples()

                    # min_samplesを整数に変換（比率の場合）
                    min_samples_count = self._convert_ransac_min_samples(ransac_min_samples, len(min_idx))

                    ransac = RANSACRegressor(
                        estimator=LinearRegression(),
                        residual_threshold=ransac_residual_threshold,
                        max_trials=ransac_max_trials,
                        min_samples=min_samples_count,
                        random_state=42,
                    )
                    ransac.fit(X, y)

                    slope = ransac.estimator_.coef_[0]
                    intercept = ransac.estimator_.intercept_
                    r2 = ransac.score(X, y)
                    inliers = np.sum(ransac.inlier_mask_)

                except Exception as e:
                    logger.debug(
                        f"RANSACトレンドライン検出エラー: {e}, 線形回帰にフォールバック"
                    )
                    # フォールバック: 線形回帰
                    model = LinearRegression()
                    model.fit(X, y)
                    slope = model.coef_[0]
                    intercept = model.intercept_
                    r2 = model.score(X, y)
                    inliers = len(min_idx)

                # 最新のトレンドライン値
                current_value = slope * len(df) + intercept

                result["support_trend"] = {
                    "slope": slope,
                    "intercept": intercept,
                    "r2": r2,
                    "current_value": current_value,
                    "touches": len(min_idx),
                    "inliers": inliers,
                    "angle": np.degrees(np.arctan(slope)) if np.mean(prices_low) != 0 else 0.0,
                }

            # 下降トレンドライン（極大値を結ぶ）
            if len(max_idx) >= min_touches:
                X = max_idx.reshape(-1, 1)
                y = prices_high[max_idx]

                # RANSACで外れ値に頑健なトレンドラインを検出
                try:
                    ransac_residual_threshold = (
                        self.config.get_trend_line_ransac_residual_threshold()
                    )
                    ransac_max_trials = self.config.get_trend_line_ransac_max_trials()
                    ransac_min_samples = self.config.get_trend_line_ransac_min_samples()

                    # min_samplesを整数に変換（比率の場合）
                    min_samples_count = self._convert_ransac_min_samples(ransac_min_samples, len(max_idx))

                    ransac = RANSACRegressor(
                        estimator=LinearRegression(),
                        residual_threshold=ransac_residual_threshold,
                        max_trials=ransac_max_trials,
                        min_samples=min_samples_count,
                        random_state=42,
                    )
                    ransac.fit(X, y)

                    slope = ransac.estimator_.coef_[0]
                    intercept = ransac.estimator_.intercept_
                    r2 = ransac.score(X, y)
                    inliers = np.sum(ransac.inlier_mask_)

                except Exception as e:
                    logger.debug(
                        f"RANSACトレンドライン検出エラー: {e}, 線形回帰にフォールバック"
                    )
                    # フォールバック: 線形回帰
                    model = LinearRegression()
                    model.fit(X, y)
                    slope = model.coef_[0]
                    intercept = model.intercept_
                    r2 = model.score(X, y)
                    inliers = len(max_idx)

                # 最新のトレンドライン値
                current_value = slope * len(df) + intercept

                result["resistance_trend"] = {
                    "slope": slope,
                    "intercept": intercept,
                    "r2": r2,
                    "current_value": current_value,
                    "touches": len(max_idx),
                    "inliers": inliers,
                    "angle": np.degrees(np.arctan(slope)) if np.mean(prices_high) != 0 else 0.0,
                }

            return result

        except Exception as e:
            should_log_detailed = self.config.should_log_detailed_errors()
            should_raise = self.config.should_raise_exceptions()

            if should_log_detailed:
                logger.error(f"トレンドライン検出エラー: {e}", exc_info=True)
            else:
                logger.warning("トレンドライン検出でエラーが発生しました")

            if should_raise:
                raise

            return {} if self.config.should_return_empty_on_error() else None

    def detect_all_patterns(
        self,
        df: pd.DataFrame,
        golden_cross_fast: Optional[int] = None,
        golden_cross_slow: Optional[int] = None,
        support_resistance_window: Optional[int] = None,
        breakout_lookback: Optional[int] = None,
        trend_window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        全パターンを検出（簡素化版 - メンテナンス性向上）

        Args:
            df: 価格データのDataFrame
            各種パラメータ

        Returns:
            パターン検出結果の辞書
        """
        try:
            # 設定値の初期化
            golden_cross_fast = golden_cross_fast or self.config.get_all_patterns_golden_cross_fast()
            golden_cross_slow = golden_cross_slow or self.config.get_all_patterns_golden_cross_slow()
            support_resistance_window = support_resistance_window or self.config.get_all_patterns_support_resistance_window()
            breakout_lookback = breakout_lookback or self.config.get_all_patterns_breakout_lookback()
            trend_window = trend_window or self.config.get_all_patterns_trend_window()

            # 各パターンの検出を並列実行
            pattern_results = {
                "crosses": self.golden_dead_cross(df, golden_cross_fast, golden_cross_slow),
                "breakouts": self.breakout_detection(df, breakout_lookback),
                "levels": self.support_resistance_levels(df, support_resistance_window),
                "trends": self.trend_line_detection(df, trend_window),
            }

            # 分析結果の追加
            pattern_results.update({
                "latest_signal": self._get_latest_signal(pattern_results["crosses"], pattern_results["breakouts"]),
                "overall_confidence": self._calculate_overall_confidence(
                    pattern_results["crosses"], pattern_results["breakouts"], pattern_results["trends"]
                ),
                "pattern_summary": self._generate_pattern_summary(pattern_results),
            })

            return pattern_results

        except Exception as e:
            should_log_detailed = self.config.should_log_detailed_errors()
            should_raise = self.config.should_raise_exceptions()

            if should_log_detailed:
                logger.error(f"チャートパターン検出エラー: {e}", exc_info=True)
            else:
                logger.warning("チャートパターン検出でエラーが発生しました")

            if should_raise:
                raise

            return (
                self._get_empty_results()
                if self.config.should_return_empty_on_error()
                else None
            )

    def _initialize_detection_params(
        self,
        golden_cross_fast: Optional[int],
        golden_cross_slow: Optional[int],
        support_resistance_window: Optional[int],
        breakout_lookback: Optional[int],
        trend_window: Optional[int],
    ) -> Dict[str, int]:
        """パターン検出パラメータの初期化"""
        return {
            "golden_cross_fast": golden_cross_fast
            or self.config.get_all_patterns_golden_cross_fast(),
            "golden_cross_slow": golden_cross_slow
            or self.config.get_all_patterns_golden_cross_slow(),
            "support_resistance_window": support_resistance_window
            or self.config.get_all_patterns_support_resistance_window(),
            "breakout_lookback": breakout_lookback
            or self.config.get_all_patterns_breakout_lookback(),
            "trend_window": trend_window or self.config.get_all_patterns_trend_window(),
        }

    def _execute_pattern_detection(
        self, df: pd.DataFrame, params: Dict[str, int]
    ) -> Dict[str, Any]:
        """パターン検出の実行"""
        return {
            "crosses": self.golden_dead_cross(
                df, params["golden_cross_fast"], params["golden_cross_slow"]
            ),
            "breakouts": self.breakout_detection(df, params["breakout_lookback"]),
            "levels": self.support_resistance_levels(
                df, params["support_resistance_window"]
            ),
            "trends": self.trend_line_detection(df, params["trend_window"]),
        }

    def _analyze_pattern_results(
        self, pattern_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """パターン結果の分析"""
        return {
            "latest_signal": self._get_latest_signal(
                pattern_results["crosses"], pattern_results["breakouts"]
            ),
            "overall_confidence": self._calculate_overall_confidence(
                pattern_results["crosses"],
                pattern_results["breakouts"],
                pattern_results["trends"],
            ),
            "pattern_summary": self._generate_pattern_summary(pattern_results),
        }

    def _build_final_results(
        self, pattern_results: Dict[str, Any], analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """最終結果の組み立て"""
        final_results = pattern_results.copy()
        final_results.update(analysis_results)
        return final_results

    def _generate_pattern_summary(
        self, pattern_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """パターンの簡易サマリを生成"""
        summary = {
            "total_patterns_detected": 0,
            "strong_signals": [],
            "weak_signals": [],
        }

        # クロスパターンのサマリ
        if not pattern_results["crosses"].empty:
            golden_crosses = pattern_results["crosses"]["Golden_Cross"].sum()
            dead_crosses = pattern_results["crosses"]["Dead_Cross"].sum()
            summary["total_patterns_detected"] += golden_crosses + dead_crosses

            if golden_crosses > 0:
                max_golden_conf = pattern_results["crosses"]["Golden_Confidence"].max()
                signal_threshold = self.config.get_pattern_summary_signal_threshold()
                if max_golden_conf > signal_threshold:
                    summary["strong_signals"].append(
                        f"Golden Cross ({max_golden_conf:.1f})"
                    )
                else:
                    summary["weak_signals"].append(
                        f"Golden Cross ({max_golden_conf:.1f})"
                    )

        # ブレイクアウトパターンのサマリ
        if not pattern_results["breakouts"].empty:
            upward_breakouts = pattern_results["breakouts"]["Upward_Breakout"].sum()
            downward_breakouts = pattern_results["breakouts"]["Downward_Breakout"].sum()
            summary["total_patterns_detected"] += upward_breakouts + downward_breakouts

        # サポートレジスタンスレベルのサマリ
        summary["support_levels_count"] = len(
            pattern_results["levels"].get("support", [])
        )
        summary["resistance_levels_count"] = len(
            pattern_results["levels"].get("resistance", [])
        )

        # トレンドラインのサマリ
        summary["trend_lines_count"] = len(pattern_results["trends"])

        return summary

    def _get_empty_results(self) -> Dict[str, Any]:
        """空の結果を返す"""
        return {
            "crosses": pd.DataFrame(),
            "breakouts": pd.DataFrame(),
            "levels": {"resistance": [], "support": []},
            "trends": {},
            "latest_signal": None,
            "overall_confidence": 0,
            "pattern_summary": {
                "total_patterns_detected": 0,
                "strong_signals": [],
                "weak_signals": [],
                "support_levels_count": 0,
                "resistance_levels_count": 0,
                "trend_lines_count": 0,
            },
        }

    def _get_latest_signal(
        self, cross_data: pd.DataFrame, breakout_data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """最新のシグナル情報を取得"""
        signals = []

        # 最新のクロスシグナルを収集
        if not cross_data.empty:
            if cross_data["Golden_Cross"].iloc[-1]:
                signals.append(
                    {
                        "type": "Golden Cross",
                        "confidence": cross_data["Golden_Confidence"].iloc[-1],
                        "timestamp": (
                            cross_data.index[-1]
                            if hasattr(cross_data.index, "__getitem__")
                            else None
                        ),
                    }
                )
            if cross_data["Dead_Cross"].iloc[-1]:
                signals.append(
                    {
                        "type": "Dead Cross",
                        "confidence": cross_data["Dead_Confidence"].iloc[-1],
                        "timestamp": (
                            cross_data.index[-1]
                            if hasattr(cross_data.index, "__getitem__")
                            else None
                        ),
                    }
                )

        # 最新のブレイクアウトシグナルを収集
        if not breakout_data.empty:
            if breakout_data["Upward_Breakout"].iloc[-1]:
                signals.append(
                    {
                        "type": "Upward Breakout",
                        "confidence": breakout_data["Upward_Confidence"].iloc[-1],
                        "timestamp": (
                            breakout_data.index[-1]
                            if hasattr(breakout_data.index, "__getitem__")
                            else None
                        ),
                    }
                )
            if breakout_data["Downward_Breakout"].iloc[-1]:
                signals.append(
                    {
                        "type": "Downward Breakout",
                        "confidence": breakout_data["Downward_Confidence"].iloc[-1],
                        "timestamp": (
                            breakout_data.index[-1]
                            if hasattr(breakout_data.index, "__getitem__")
                            else None
                        ),
                    }
                )

        if not signals:
            return None

        # 最も信頼度の高いシグナルを返す
        return max(signals, key=lambda x: x.get("confidence", 0.0))

    def _calculate_overall_confidence(
        self,
        cross_data: pd.DataFrame,
        breakout_data: pd.DataFrame,
        trends: Dict[str, Any],
    ) -> float:
        """
        改善された総合信頼度スコアを計算
        正規化とバランス調整を実装
        """
        weights = self.config.get_confidence_weights()
        normalization = self.config.get_confidence_normalization()

        # 各カテゴリの信頼度を計算
        confidence_components = {
            "cross": self._calculate_cross_confidence(cross_data, weights),
            "breakout": self._calculate_breakout_confidence(breakout_data, weights),
            "trend": self._calculate_trend_confidence(trends, weights),
        }

        # 有効なコンポーネントの重みつき平均
        total_weight = 0
        weighted_sum = 0

        for component, confidence in confidence_components.items():
            if confidence > 0:
                component_weight = self._get_component_weight(component, weights)
                weighted_sum += confidence * component_weight
                total_weight += component_weight

        if total_weight == 0:
            return 0.0

        # 正規化された総合信頼度
        overall_confidence = weighted_sum / total_weight

        # 範囲制限と正規化
        if isinstance(normalization, dict):
            min_conf = normalization.get("min_confidence", 0.0)
            max_conf = normalization.get("max_confidence", 100.0)
        else:
            min_conf = 0.0
            max_conf = float(normalization) if normalization else 100.0

        return np.clip(overall_confidence, min_conf, max_conf)

    def _calculate_cross_confidence(
        self, cross_data: pd.DataFrame, weights: Dict[str, float]
    ) -> float:
        """クロスパターンの信頼度を計算"""
        if cross_data.empty:
            return 0.0

        # 最新のシグナルを取得
        golden_conf = (
            cross_data["Golden_Confidence"].iloc[-1] if len(cross_data) > 0 else 0
        )
        dead_conf = cross_data["Dead_Confidence"].iloc[-1] if len(cross_data) > 0 else 0

        # 最近のシグナルを優先し、過去のシグナルも考慮
        recent_golden = (
            np.nanmax(cross_data["Golden_Confidence"].iloc[-5:])
            if len(cross_data) >= 5
            else golden_conf
        )
        recent_dead = (
            np.nanmax(cross_data["Dead_Confidence"].iloc[-5:])
            if len(cross_data) >= 5
            else dead_conf
        )

        return max(recent_golden, recent_dead)

    def _calculate_breakout_confidence(
        self, breakout_data: pd.DataFrame, weights: Dict[str, float]
    ) -> float:
        """ブレイクアウトパターンの信頼度を計算"""
        if breakout_data.empty:
            return 0.0

        # 最新のシグナルを取得
        up_conf = (
            breakout_data["Upward_Confidence"].iloc[-1] if len(breakout_data) > 0 else 0
        )
        down_conf = (
            breakout_data["Downward_Confidence"].iloc[-1]
            if len(breakout_data) > 0
            else 0
        )

        # 最近のシグナルを優先
        recent_up = (
            np.nanmax(breakout_data["Upward_Confidence"].iloc[-3:])
            if len(breakout_data) >= 3
            else up_conf
        )
        recent_down = (
            np.nanmax(breakout_data["Downward_Confidence"].iloc[-3:])
            if len(breakout_data) >= 3
            else down_conf
        )

        return max(recent_up, recent_down)

    def _calculate_trend_confidence(
        self, trends: Dict[str, Any], weights: Dict[str, float]
    ) -> float:
        """トレンドラインの信頼度を計算"""
        if not trends:
            return 0.0

        trend_confidences = []
        for _trend_name, trend_info in trends.items():
            if isinstance(trend_info, dict) and "r2" in trend_info:
                # R²値をパーセントに変換し、inliers数で重みづけ
                r2_confidence = trend_info["r2"] * 100
                inliers = trend_info.get("inliers", trend_info.get("touches", 1))

                # inliers数に基づく補正係数
                inlier_bonus = (
                    min(1.0 + (inliers - 3) * 0.1, 1.5) if inliers >= 3 else 0.8
                )

                trend_confidences.append(r2_confidence * inlier_bonus)

        return np.mean(trend_confidences) if trend_confidences else 0.0

    def _get_component_weight(self, component: str, weights: Dict[str, float]) -> float:
        """コンポーネントの重みを取得"""
        weight_mapping = {
            "cross": (weights.get("golden_cross", 0.3) + weights.get("dead_cross", 0.3))
            / 2,
            "breakout": (
                weights.get("upward_breakout", 0.25)
                + weights.get("downward_breakout", 0.25)
            )
            / 2,
            "trend": weights.get("trend_r2", 0.2),
        }
        return weight_mapping.get(component, 0.1)

    def _convert_ransac_min_samples(self, ransac_min_samples: Union[int, float], data_size: int) -> int:
        """
        RANSAC min_samplesを適切な整数値に変換
        
        Args:
            ransac_min_samples: 設定値（整数または比率）
            data_size: データサイズ
            
        Returns:
            適切な最小サンプル数
        """
        if isinstance(ransac_min_samples, float) and ransac_min_samples < 1.0:
            return max(2, int(data_size * ransac_min_samples))
        else:
            return int(ransac_min_samples)


# 使用例
if __name__ == "__main__":
    from datetime import datetime

    import numpy as np

    # サンプルデータ作成
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    np.random.seed(42)

    # トレンドのあるデータを生成
    trend = np.linspace(100, 120, 100)
    noise = np.random.randn(100) * 2
    close_prices = trend + noise

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close_prices + np.random.randn(100) * 0.5,
            "High": close_prices + np.abs(np.random.randn(100)) * 2,
            "Low": close_prices - np.abs(np.random.randn(100)) * 2,
            "Close": close_prices,
            "Volume": np.random.randint(1000000, 5000000, 100),
        }
    )
    df.set_index("Date", inplace=True)

    # パターン認識
    recognizer = ChartPatternRecognizer()

    logger.info("ゴールデン・デッドクロス分析開始")
    crosses = recognizer.golden_dead_cross(df)
    golden_dates = df.index[crosses["Golden_Cross"]]
    dead_dates = df.index[crosses["Dead_Cross"]]

    golden_cross_data = []
    for date in golden_dates:
        confidence = crosses.loc[date, "Golden_Confidence"]
        golden_cross_data.append(
            {"date": date.date().isoformat(), "confidence": round(confidence, 1)}
        )

    dead_cross_data = []
    for date in dead_dates:
        confidence = crosses.loc[date, "Dead_Confidence"]
        dead_cross_data.append(
            {"date": date.date().isoformat(), "confidence": round(confidence, 1)}
        )

    logger.info(
        "ゴールデンクロス・デッドクロス検出結果",
        golden_cross_count=len(golden_dates),
        dead_cross_count=len(dead_dates),
        golden_crosses=golden_cross_data,
        dead_crosses=dead_cross_data,
    )

    logger.info("サポート・レジスタンス分析")
    levels = recognizer.support_resistance_levels(df)
    logger.info(
        "サポート・レジスタンス検出結果",
        resistance_levels=[round(level, 2) for level in levels["resistance"]],
        support_levels=[round(level, 2) for level in levels["support"]],
        resistance_count=len(levels["resistance"]),
        support_count=len(levels["support"]),
    )

    logger.info("全パターン検出分析")
    all_patterns = recognizer.detect_all_patterns(df)

    pattern_summary = {
        "overall_confidence": round(all_patterns["overall_confidence"], 1),
        "pattern_analysis_complete": True,
    }

    if "latest_signal" in all_patterns:
        signal = all_patterns["latest_signal"]
        pattern_summary["latest_signal"] = {
            "type": signal["type"],
            "confidence": round(signal["confidence"], 1),
        }

    logger.info("全パターン検出完了", **pattern_summary)
