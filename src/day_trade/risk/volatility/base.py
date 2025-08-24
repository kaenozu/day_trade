#!/usr/bin/env python3
"""
ボラティリティ予測エンジン - 基底クラス

基底クラス、共通設定、依存関係チェック、ユーティリティ機能を提供します。
"""

import warnings
from pathlib import Path
from typing import Dict

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 依存パッケージチェック
try:
    from arch import arch_model
    from arch.univariate import EGARCH, GARCH, GJR

    ARCH_AVAILABLE = True
    logger.info("arch利用可能")
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch未インストール - pip install archでインストールしてください")

try:
    import joblib
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn利用可能")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn未インストール")

# 警告フィルタ設定
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class VolatilityEngineBase:
    """
    ボラティリティ予測エンジンの基底クラス

    共通の初期化、設定、ユーティリティ機能を提供します。
    """

    def __init__(self, model_cache_dir: str = "data/volatility_models"):
        """
        初期化

        Args:
            model_cache_dir: モデルキャッシュディレクトリ
        """
        self.model_cache_dir = Path(model_cache_dir)

        # モデルキャッシュディレクトリ作成
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # 各種モデルの保存用辞書
        self.garch_models = {}
        self.ml_models = {}
        self.scalers = {}

        # VIX計算パラメータ
        self.vix_params = {
            "window": 30,
            "garch_alpha": 0.1,
            "garch_beta": 0.85,
            "garch_omega": 0.05,
        }

        logger.info("ボラティリティ予測エンジン基底クラス初期化完了")

    @property
    def arch_available(self) -> bool:
        """archパッケージの利用可能性"""
        return ARCH_AVAILABLE

    @property
    def sklearn_available(self) -> bool:
        """scikit-learnパッケージの利用可能性"""
        return SKLEARN_AVAILABLE

    def _validate_dependencies(self, required_libs: list) -> bool:
        """
        必要な依存ライブラリの確認

        Args:
            required_libs: 必要なライブラリのリスト ['arch', 'sklearn']

        Returns:
            すべての依存関係が満たされているかどうか
        """
        for lib in required_libs:
            if lib == "arch" and not self.arch_available:
                logger.error(f"必要な依存関係が不足: {lib}")
                return False
            elif lib == "sklearn" and not self.sklearn_available:
                logger.error(f"必要な依存関係が不足: {lib}")
                return False

        return True

    def get_default_thresholds(self) -> Dict[str, float]:
        """
        デフォルトのボラティリティ閾値を取得

        Returns:
            ボラティリティレジーム分類用の閾値辞書
        """
        return {
            "low": 0.15,  # 15%未満
            "medium": 0.30,  # 15-30%
            "high": 0.50,  # 30-50%
            "extreme": float("inf"),  # 50%以上
        }

    def get_default_vix_params(self) -> Dict[str, float]:
        """
        デフォルトのVIX計算パラメータを取得

        Returns:
            VIX計算用のパラメータ辞書
        """
        return self.vix_params.copy()

    def update_vix_params(self, **kwargs) -> None:
        """
        VIX計算パラメータを更新

        Args:
            **kwargs: 更新するパラメータ
        """
        for key, value in kwargs.items():
            if key in self.vix_params:
                self.vix_params[key] = value
                logger.info(f"VIXパラメータ更新: {key}={value}")
            else:
                logger.warning(f"未知のVIXパラメータ: {key}")