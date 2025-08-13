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
                        "values": series.values,  # Issue #712対応: NumPy配列で直接保存
                    }
        return stats

    def save_baseline(self, file_path: str, format: str = 'auto'):
        """
        ベースライン統計情報をファイルに保存します。

        Args:
            file_path (str): 保存先のファイルパス
            format (str): 保存形式 ('json', 'pickle', 'joblib', 'auto')
                        'auto'の場合、データサイズに応じて最適な形式を自動選択
        """
        try:
            # Issue #713対応: データサイズに応じた保存形式の自動選択
            total_data_size = self._estimate_data_size()

            if format == 'auto':
                # 100MB以上の場合はバイナリ形式を使用
                if total_data_size > 100 * 1024 * 1024:  # 100MB
                    format = 'joblib'
                    logger.info(f"大規模データ検出 ({total_data_size / 1024 / 1024:.1f}MB), joblib形式で保存します")
                else:
                    format = 'json'

            if format == 'json':
                self._save_as_json(file_path)
            elif format == 'pickle':
                # pickle形式の場合は.pkl拡張子を確保
                if not file_path.endswith('.pkl') and not file_path.endswith('.pickle'):
                    file_path = f"{file_path}.pkl"
                self._save_as_pickle(file_path)
            elif format == 'joblib':
                # joblib形式の場合は.joblib拡張子を確保
                if not file_path.endswith('.joblib') and not file_path.endswith('.jl'):
                    file_path = f"{file_path}.joblib"
                self._save_as_joblib(file_path)
            else:
                raise ValueError(f"サポートされていない保存形式: {format}")

            logger.info(f"ベースライン統計情報を '{file_path}' ({format}形式) に保存しました。")

        except Exception as e:
            logger.error(f"ベースライン統計情報の保存中にエラーが発生しました: {e}")

    def _estimate_data_size(self) -> int:
        """データサイズを推定"""
        total_size = 0
        for stat in self.baseline_stats.values():
            if "values" in stat and hasattr(stat["values"], 'nbytes'):
                total_size += stat["values"].nbytes
            # 他の統計値も含む（概算）
            total_size += 1024  # 統計値のオーバーヘッド
        return total_size

    def _save_as_json(self, file_path: str):
        """JSON形式で保存"""
        # Issue #712対応: NumPy配列をPythonのリストに変換
        serializable_stats = {}
        for _feature, stat in self.baseline_stats.items():
            serializable_stat = stat.copy()
            if "values" in serializable_stat:
                # NumPy配列をtolist()で直接変換
                serializable_stat["values"] = serializable_stat["values"].tolist()
            serializable_stats[_feature] = serializable_stat

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_stats, f, ensure_ascii=False, indent=4)

    def _save_as_pickle(self, file_path: str):
        """Pickle形式で保存"""
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self.baseline_stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_as_joblib(self, file_path: str):
        """Joblib形式で保存（NumPy配列に最適化）"""
        try:
            import joblib
            joblib.dump(self.baseline_stats, file_path, compress=3)  # 圧縮レベル3
        except ImportError:
            logger.warning("joblib未インストール、pickle形式で代替保存")
            self._save_as_pickle(file_path)

    def load_baseline(self, file_path: str, format: str = 'auto'):
        """
        ファイルからベースライン統計情報を読み込みます。

        Args:
            file_path (str): 読み込み元のファイルパス
            format (str): 読み込み形式 ('json', 'pickle', 'joblib', 'auto')
                        'auto'の場合、ファイル拡張子から自動判定
        """
        try:
            # Issue #713対応: ファイル形式の自動判定
            if format == 'auto':
                format = self._detect_file_format(file_path)

            if format == 'json':
                loaded_stats = self._load_from_json(file_path)
            elif format == 'pickle':
                # pickle形式の場合は.pkl拡張子を確保
                if not file_path.endswith('.pkl') and not file_path.endswith('.pickle'):
                    file_path = f"{file_path}.pkl"
                loaded_stats = self._load_from_pickle(file_path)
            elif format == 'joblib':
                # joblib形式の場合は.joblib拡張子を確保
                if not file_path.endswith('.joblib') and not file_path.endswith('.jl'):
                    file_path = f"{file_path}.joblib"
                loaded_stats = self._load_from_joblib(file_path)
            else:
                raise ValueError(f"サポートされていない読み込み形式: {format}")

            self.baseline_stats = loaded_stats
            logger.info(f"ベースライン統計情報を '{file_path}' ({format}形式) から読み込みました。")

        except FileNotFoundError:
            logger.warning(f"ベースラインファイル '{file_path}' が見つかりません。")
        except Exception as e:
            logger.error(f"ベースライン統計情報の読み込み中にエラーが発生しました: {e}")

    def _detect_file_format(self, file_path: str) -> str:
        """ファイル拡張子から形式を自動判定"""
        import os
        _, ext = os.path.splitext(file_path.lower())

        if ext == '.json':
            return 'json'
        elif ext in ['.pkl', '.pickle']:
            return 'pickle'
        elif ext in ['.joblib', '.jl']:
            return 'joblib'
        else:
            # デフォルトはJSON（従来との互換性）
            return 'json'

    def _load_from_json(self, file_path: str) -> dict:
        """JSON形式から読み込み"""
        with open(file_path, encoding="utf-8") as f:
            loaded_stats = json.load(f)

        # リストをNumPy配列に戻す
        for _feature, stat in loaded_stats.items():
            if "values" in stat:
                stat["values"] = np.array(stat["values"])

        return loaded_stats

    def _load_from_pickle(self, file_path: str) -> dict:
        """Pickle形式から読み込み"""
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def _load_from_joblib(self, file_path: str) -> dict:
        """Joblib形式から読み込み"""
        try:
            import joblib
            return joblib.load(file_path)
        except ImportError:
            logger.warning("joblib未インストール、pickle形式として読み込み試行")
            return self._load_from_pickle(file_path)

    def get_baseline_info(self) -> dict:
        """
        Issue #713対応: ベースライン統計情報のサマリーを取得

        Returns:
            dict: ベースライン情報のサマリー
        """
        if not self.baseline_stats:
            return {"status": "no_baseline_data"}

        total_size = self._estimate_data_size()
        feature_count = len(self.baseline_stats)

        # 各特徴量のデータサイズ
        feature_sizes = {}
        for feature, stats in self.baseline_stats.items():
            if "values" in stats and hasattr(stats["values"], 'nbytes'):
                feature_sizes[feature] = stats["values"].nbytes

        return {
            "feature_count": feature_count,
            "total_data_size_mb": total_size / (1024 * 1024),
            "feature_sizes": feature_sizes,
            "recommended_format": "joblib" if total_size > 100 * 1024 * 1024 else "json"
        }

    def save_baseline_optimized(self, base_path: str) -> dict:
        """
        Issue #713対応: 最適化された保存（大規模データ対応）

        Args:
            base_path (str): ベースパス（拡張子は自動設定）

        Returns:
            dict: 保存結果の情報
        """
        if not self.baseline_stats:
            raise ValueError("保存するベースラインデータがありません")

        info = self.get_baseline_info()
        recommended_format = info["recommended_format"]

        # 拡張子を自動設定
        if recommended_format == "json":
            file_path = f"{base_path}.json"
        elif recommended_format == "joblib":
            file_path = f"{base_path}.joblib"
        else:
            file_path = f"{base_path}.pkl"

        import time
        start_time = time.time()

        self.save_baseline(file_path, format=recommended_format)

        save_time = time.time() - start_time

        # ファイルサイズを取得
        import os
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        return {
            "file_path": file_path,
            "format_used": recommended_format,
            "save_time_seconds": save_time,
            "file_size_mb": file_size / (1024 * 1024),
            "data_size_mb": info["total_data_size_mb"],
            "compression_ratio": info["total_data_size_mb"] / (file_size / (1024 * 1024)) if file_size > 0 else 1.0
        }
