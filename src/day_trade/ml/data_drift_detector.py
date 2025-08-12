import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    データドリフト検出器

    ベースラインデータと新しいデータの統計的特性を比較し、ドリフトを検出します。
    """

    def __init__(self, threshold: float = 0.05):
        """
        初期化

        Args:
            threshold (float): ドリフト検出の閾値 (例: KS検定のp値閾値)
        """
        self.baseline_stats = {}
        self.threshold = threshold

    def fit(self, baseline_data: pd.DataFrame):
        """
        ベースラインデータを学習し、統計情報を保存します。

        Args:
            baseline_data (pd.DataFrame): ベースラインデータ
        """
        logger.info("データドリフト検出器: ベースラインデータを学習中...")
        self.baseline_stats = self._calculate_statistics(baseline_data)
        logger.info("データドリフト検出器: ベースライン統計情報を保存しました。")

    def detect_drift(self, new_data: pd.DataFrame) -> dict:
        """
        新しいデータとベースラインデータを比較し、ドリフトを検出します。

        Args:
            new_data (pd.DataFrame): 新しいデータ

        Returns:
            dict: ドリフト検出結果。各特徴量ごとのドリフト情報を含みます。
        """
        if not self.baseline_stats:
            logger.warning(
                "ベースライン統計情報が設定されていません。fit()を先に実行してください。"
            )
            return {"drift_detected": False, "features": {}}

        logger.info("データドリフト検出器: ドリフトを検出中...")
        drift_results = {}
        overall_drift_detected = False

        for _feature, baseline_stat in self.baseline_stats.items():
            if _feature not in new_data.columns:
                logger.warning(
                    f"新しいデータに特徴量 '{_feature}' が見つかりません。スキップします。"
                )
                continue

            new_data_series = new_data[_feature].dropna()
            if new_data_series.empty:
                logger.warning(
                    f"新しいデータの特徴量 '{_feature}' が空です。スキップします。"
                )
                continue

            # 数値データのみを対象
            if pd.api.types.is_numeric_dtype(
                new_data_series
            ) and pd.api.types.is_numeric_dtype(pd.Series(baseline_stat["values"])):
                # Kolmogorov-Smirnov (KS) 検定で分布の差を検出
                try:
                    statistic, p_value = ks_2samp(
                        baseline_stat["values"], new_data_series.values
                    )
                    drift_detected = p_value < self.threshold
                    if drift_detected:
                        overall_drift_detected = True

                    drift_results[_feature] = {
                        "drift_detected": drift_detected,
                        "p_value": p_value,
                        "statistic": statistic,
                        "baseline_mean": baseline_stat.get("mean"),
                        "new_mean": new_data_series.mean(),
                        "baseline_std": baseline_stat.get("std"),
                        "new_std": new_data_series.std(),
                    }
                except ValueError as e:
                    logger.error(f"KS検定エラー for feature '{_feature}': {e}")
                    drift_results[_feature] = {"drift_detected": False, "error": str(e)}
            else:
                logger.info(
                    f"特徴量 '{_feature}' は数値データではないため、KS検定をスキップします。"
                )
                drift_results[_feature] = {
                    "drift_detected": False,
                    "reason": "非数値データ",
                }

        logger.info(
            f"データドリフト検出完了。全体ドリフト検出: {overall_drift_detected}"
        )
        return {"drift_detected": overall_drift_detected, "features": drift_results}

    def _calculate_statistics(self, data: pd.DataFrame) -> dict:
        """
        データフレームの各数値特徴量の統計情報を計算します。

        Args:
            data (pd.DataFrame): 統計情報を計算するデータフレーム

        Returns:
            dict: 各特徴量の統計情報 (平均、標準偏差、値のリストなど)
        """
        stats = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                series = data[col].dropna()
                if not series.empty:
                    stats[col] = {
                        "mean": series.mean(),
                        "std": series.std(),
                        "min": series.min(),
                        "max": series.max(),
                        "median": series.median(),
                        "values": series.tolist(),  # 分布比較のために値を保存
                    }
        return stats

    def save_baseline(self, file_path: str):
        """
        ベースライン統計情報をファイルに保存します。

        Args:
            file_path (str): 保存先のファイルパス (例: 'baseline_stats.json')
        """
        try:
            # NumPy配列をPythonのリストに変換
            serializable_stats = {}
            for _feature, stat in self.baseline_stats.items():
                serializable_stat = stat.copy()
                if "values" in serializable_stat:
                    serializable_stat["values"] = [
                        float(v) for v in serializable_stat["values"]
                    ]
                serializable_stats[_feature] = serializable_stat

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(serializable_stats, f, ensure_ascii=False, indent=4)
            logger.info(f"ベースライン統計情報を '{file_path}' に保存しました。")
        except Exception as e:
            logger.error(f"ベースライン統計情報の保存中にエラーが発生しました: {e}")

    def load_baseline(self, file_path: str):
        """
        ファイルからベースライン統計情報を読み込みます。

        Args:
            file_path (str): 読み込み元のファイルパス
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                loaded_stats = json.load(f)

            # リストをNumPy配列に戻す（必要であれば）
            for _feature, stat in loaded_stats.items():
                if "values" in stat:
                    stat["values"] = np.array(stat["values"])

            self.baseline_stats = loaded_stats
            logger.info(f"ベースライン統計情報を '{file_path}' から読み込みました。")
        except FileNotFoundError:
            logger.warning(f"ベースラインファイル '{file_path}' が見つかりません。")
        except json.JSONDecodeError as e:
            logger.error(
                f"ベースラインファイルの読み込み中にJSONデコードエラーが発生しました: {e}"
            )
        except Exception as e:
            logger.error(f"ベースライン統計情報の読み込み中にエラーが発生しました: {e}")
