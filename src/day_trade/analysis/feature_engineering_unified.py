"""
特徴量エンジニアリング統合システム（Strategy Pattern実装）

標準特徴量エンジニアリングと最適化版を統一し、設定ベースで選択可能なアーキテクチャ
"""

import gc
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats

from ..core.optimization_strategy import (
    OptimizationStrategy,
    OptimizationLevel,
    OptimizationConfig,
    optimization_strategy,
    get_optimized_implementation
)
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# オプショナル依存パッケージ
try:
    from numba import njit
    import numba
    NUMBA_AVAILABLE = True
    logger.info("Numba利用可能 - 高速計算が有効")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.info("Numba未利用 - 標準計算を使用")

try:
    from sklearn.preprocessing import RobustScaler, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn未利用 - 基本正規化のみ")

try:
    from ..utils.optimized_pandas import (
        chunked_processing,
        get_optimized_processor,
        optimize_dataframe_dtypes,
        vectorized_technical_indicators,
    )
    from ..utils.performance_analyzer import profile_performance
    OPTIMIZATION_UTILS_AVAILABLE = True
except ImportError:
    OPTIMIZATION_UTILS_AVAILABLE = False
    logger.info("最適化ユーティリティ未利用")

warnings.filterwarnings("ignore")


@dataclass
class FeatureConfig:
    """統合特徴量生成設定"""
    # 基本設定
    lookback_periods: List[int]
    volatility_windows: List[int]
    momentum_periods: List[int]

    # 複合特徴量設定
    enable_cross_features: bool = True
    enable_statistical_features: bool = True
    enable_regime_features: bool = True

    # 正規化設定
    scaling_method: str = "robust"  # standard, robust, minmax
    outlier_threshold: float = 3.0

    # 最適化固有設定
    enable_parallel: bool = True
    enable_numba: bool = NUMBA_AVAILABLE
    enable_vectorization: bool = True
    chunk_size: int = 10000
    max_workers: int = 4
    memory_limit_mb: int = 1000

    @classmethod
    def default(cls) -> "FeatureConfig":
        """デフォルト設定"""
        return cls(
            lookback_periods=[5, 10, 20, 50],
            volatility_windows=[10, 20, 50],
            momentum_periods=[5, 10, 20],
        )


@dataclass
class FeatureResult:
    """特徴量生成結果"""
    features: pd.DataFrame
    feature_names: List[str]
    metadata: Dict[str, Any]
    generation_time: float
    strategy_used: str


# Numba最適化関数群（条件付き定義）
if NUMBA_AVAILABLE:
    @njit
    def fast_rsi_calculation(prices: np.ndarray, window: int = 14) -> np.ndarray:
        """高速RSI計算（Numba最適化）"""
        n = len(prices)
        if n < window + 1:
            return np.full(n, 50.0)

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        rsi = np.full(n, 50.0)

        # 初期平均計算
        avg_gain = np.mean(gains[:window])
        avg_loss = np.mean(losses[:window])

        for i in range(window, n - 1):
            avg_gain = (avg_gain * (window - 1) + gains[i]) / window
            avg_loss = (avg_loss * (window - 1) + losses[i]) / window

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    @njit
    def fast_bollinger_bands(prices: np.ndarray, window: int = 20, std_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """高速ボリンジャーバンド計算"""
        n = len(prices)
        sma = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)

        for i in range(window - 1, n):
            window_data = prices[i - window + 1:i + 1]
            mean_val = np.mean(window_data)
            std_val = np.std(window_data)

            sma[i] = mean_val
            upper[i] = mean_val + std_factor * std_val
            lower[i] = mean_val - std_factor * std_val

        return sma, upper, lower

    @njit
    def fast_momentum_features(prices: np.ndarray, periods: np.ndarray) -> np.ndarray:
        """高速モメンタム特徴量計算"""
        n = len(prices)
        max_period = np.max(periods)
        features = np.full((n, len(periods)), np.nan)

        for i in range(max_period, n):
            for j, period in enumerate(periods):
                if i >= period:
                    features[i, j] = (prices[i] - prices[i - period]) / prices[i - period] * 100

        return features
else:
    # フォールバック実装
    def fast_rsi_calculation(prices: np.ndarray, window: int = 14) -> np.ndarray:
        """標準RSI計算"""
        df = pd.DataFrame({'price': prices})
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0).values

    def fast_bollinger_bands(prices: np.ndarray, window: int = 20, std_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """標準ボリンジャーバンド計算"""
        df = pd.DataFrame({'price': prices})
        sma = df['price'].rolling(window=window).mean()
        std = df['price'].rolling(window=window).std()
        upper = sma + (std * std_factor)
        lower = sma - (std * std_factor)
        return sma.values, upper.values, lower.values

    def fast_momentum_features(prices: np.ndarray, periods: np.ndarray) -> np.ndarray:
        """標準モメンタム特徴量計算"""
        df = pd.DataFrame({'price': prices})
        features = []
        for period in periods:
            momentum = df['price'].pct_change(periods=period) * 100
            features.append(momentum.values)
        return np.column_stack(features)


class FeatureEngineeringBase(OptimizationStrategy):
    """特徴量エンジニアリングの基底戦略クラス"""

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.feature_config = FeatureConfig.default()

    def execute(self, data: pd.DataFrame, feature_config: Optional[FeatureConfig] = None, **kwargs) -> FeatureResult:
        """特徴量生成の実行"""
        start_time = time.time()

        if feature_config:
            self.feature_config = feature_config

        try:
            features_df = self._generate_features(data, **kwargs)
            execution_time = time.time() - start_time
            self.record_execution(execution_time, True)

            return FeatureResult(
                features=features_df,
                feature_names=list(features_df.columns),
                metadata={
                    "input_shape": data.shape,
                    "output_shape": features_df.shape,
                    "config": self.feature_config,
                },
                generation_time=execution_time,
                strategy_used=self.get_strategy_name()
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.record_execution(execution_time, False)
            logger.error(f"特徴量生成エラー: {e}")
            raise

    def _generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """特徴量生成の実装（サブクラスで拡張）"""
        close_col = self._get_price_column(data)
        features = pd.DataFrame(index=data.index)

        # 基本テクニカル指標
        features.update(self._calculate_basic_indicators(data[close_col]))

        # 統計的特徴量
        if self.feature_config.enable_statistical_features:
            features.update(self._calculate_statistical_features(data))

        # 複合特徴量
        if self.feature_config.enable_cross_features:
            features.update(self._calculate_cross_features(data))

        return features

    def _get_price_column(self, data: pd.DataFrame) -> str:
        """価格カラムの特定"""
        if '終値' in data.columns:
            return '終値'
        elif 'Close' in data.columns:
            return 'Close'
        else:
            return data.columns[0]  # フォールバック

    def _calculate_basic_indicators(self, prices: pd.Series) -> pd.DataFrame:
        """基本指標の計算"""
        features = pd.DataFrame(index=prices.index)

        # 移動平均
        for period in self.feature_config.lookback_periods:
            features[f'sma_{period}'] = prices.rolling(window=period).mean()
            features[f'ema_{period}'] = prices.ewm(span=period).mean()

        # ボラティリティ
        for window in self.feature_config.volatility_windows:
            features[f'volatility_{window}'] = prices.pct_change().rolling(window=window).std()

        # モメンタム
        for period in self.feature_config.momentum_periods:
            features[f'momentum_{period}'] = prices.pct_change(periods=period)
            features[f'roc_{period}'] = (prices / prices.shift(period) - 1) * 100

        return features

    def _calculate_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """統計的特徴量の計算"""
        close_col = self._get_price_column(data)
        prices = data[close_col]
        features = pd.DataFrame(index=data.index)

        # 価格統計
        for window in [10, 20, 50]:
            features[f'price_percentile_{window}'] = prices.rolling(window=window).rank(pct=True)
            features[f'price_zscore_{window}'] = (prices - prices.rolling(window=window).mean()) / prices.rolling(window=window).std()

        # リターン統計
        returns = prices.pct_change()
        for window in [10, 20]:
            features[f'returns_skew_{window}'] = returns.rolling(window=window).skew()
            features[f'returns_kurt_{window}'] = returns.rolling(window=window).kurt()

        return features

    def _calculate_cross_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """複合特徴量の計算"""
        features = pd.DataFrame(index=data.index)
        close_col = self._get_price_column(data)
        prices = data[close_col]

        # 価格と移動平均の関係
        sma_20 = prices.rolling(window=20).mean()
        features['price_vs_sma20'] = (prices - sma_20) / sma_20

        # ボリンジャーバンド位置
        bb_middle = prices.rolling(window=20).mean()
        bb_std = prices.rolling(window=20).std()
        features['bb_position'] = (prices - bb_middle) / (2 * bb_std)

        return features


@optimization_strategy("feature_engineering", OptimizationLevel.STANDARD)
class StandardFeatureEngineering(FeatureEngineeringBase):
    """標準特徴量エンジニアリング実装"""

    def get_strategy_name(self) -> str:
        return "標準特徴量エンジニアリング"


@optimization_strategy("feature_engineering", OptimizationLevel.OPTIMIZED)
class OptimizedFeatureEngineering(FeatureEngineeringBase):
    """最適化特徴量エンジニアリング実装"""

    def get_strategy_name(self) -> str:
        return "最適化特徴量エンジニアリング"

    def _generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """最適化された特徴量生成"""
        if not OPTIMIZATION_UTILS_AVAILABLE:
            # フォールバック
            return super()._generate_features(data, **kwargs)

        # チャンク処理による大容量データ対応
        if len(data) > self.feature_config.chunk_size:
            return self._generate_features_chunked(data, **kwargs)

        # メモリ最適化
        if self.config.cache_enabled:
            data = optimize_dataframe_dtypes(data)

        close_col = self._get_price_column(data)
        features = pd.DataFrame(index=data.index)

        # 並列特徴量計算
        if self.feature_config.enable_parallel and self.feature_config.max_workers > 1:
            features.update(self._calculate_features_parallel(data))
        else:
            # Numba高速化
            if self.feature_config.enable_numba and NUMBA_AVAILABLE:
                features.update(self._calculate_features_numba(data))
            else:
                features.update(self._calculate_basic_indicators(data[close_col]))

        # 統計・複合特徴量
        if self.feature_config.enable_statistical_features:
            features.update(self._calculate_statistical_features(data))

        if self.feature_config.enable_cross_features:
            features.update(self._calculate_cross_features(data))

        # メモリクリーンアップ
        gc.collect()

        return features

    def _generate_features_chunked(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """チャンク処理による特徴量生成"""
        logger.info(f"チャンク処理開始: {len(data)}行 -> {self.feature_config.chunk_size}行ずつ")

        chunk_results = []
        for i in range(0, len(data), self.feature_config.chunk_size):
            chunk = data.iloc[i:i + self.feature_config.chunk_size]
            chunk_features = super()._generate_features(chunk, **kwargs)
            chunk_results.append(chunk_features)

        return pd.concat(chunk_results, axis=0)

    def _calculate_features_parallel(self, data: pd.DataFrame) -> pd.DataFrame:
        """並列特徴量計算"""
        close_col = self._get_price_column(data)
        prices = data[close_col].values

        with ThreadPoolExecutor(max_workers=self.feature_config.max_workers) as executor:
            # 並列タスクを投入
            future_sma = executor.submit(self._calculate_sma_parallel, prices)
            future_ema = executor.submit(self._calculate_ema_parallel, prices)
            future_volatility = executor.submit(self._calculate_volatility_parallel, prices)

            # 結果を収集
            features = pd.DataFrame(index=data.index)
            features.update(future_sma.result())
            features.update(future_ema.result())
            features.update(future_volatility.result())

        return features

    def _calculate_sma_parallel(self, prices: np.ndarray) -> pd.DataFrame:
        """並列SMA計算"""
        features = {}
        for period in self.feature_config.lookback_periods:
            df_temp = pd.DataFrame({'price': prices})
            features[f'sma_{period}'] = df_temp['price'].rolling(window=period).mean()
        return pd.DataFrame(features)

    def _calculate_ema_parallel(self, prices: np.ndarray) -> pd.DataFrame:
        """並列EMA計算"""
        features = {}
        for period in self.feature_config.lookback_periods:
            df_temp = pd.DataFrame({'price': prices})
            features[f'ema_{period}'] = df_temp['price'].ewm(span=period).mean()
        return pd.DataFrame(features)

    def _calculate_volatility_parallel(self, prices: np.ndarray) -> pd.DataFrame:
        """並列ボラティリティ計算"""
        features = {}
        df_temp = pd.DataFrame({'price': prices})
        returns = df_temp['price'].pct_change()

        for window in self.feature_config.volatility_windows:
            features[f'volatility_{window}'] = returns.rolling(window=window).std()

        return pd.DataFrame(features)

    def _calculate_features_numba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Numba高速化特徴量計算"""
        close_col = self._get_price_column(data)
        prices = data[close_col].values
        features = pd.DataFrame(index=data.index)

        # RSI
        if 14 in self.feature_config.lookback_periods:
            features['rsi_14'] = fast_rsi_calculation(prices, 14)

        # ボリンジャーバンド
        bb_sma, bb_upper, bb_lower = fast_bollinger_bands(prices, 20, 2.0)
        features['bb_sma'] = bb_sma
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_width'] = (bb_upper - bb_lower) / bb_sma

        # モメンタム特徴量
        momentum_periods = np.array(self.feature_config.momentum_periods)
        momentum_features = fast_momentum_features(prices, momentum_periods)
        for i, period in enumerate(momentum_periods):
            features[f'momentum_{period}'] = momentum_features[:, i]

        return features


# 統合インターフェース
class FeatureEngineeringManager:
    """特徴量エンジニアリング統合マネージャー"""

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig.from_env()
        self._strategy = None

    def get_strategy(self) -> OptimizationStrategy:
        """現在の戦略を取得"""
        if self._strategy is None:
            self._strategy = get_optimized_implementation("feature_engineering", self.config)
        return self._strategy

    def generate_features(self, data: pd.DataFrame, feature_config: Optional[FeatureConfig] = None, **kwargs) -> FeatureResult:
        """特徴量生成の実行"""
        strategy = self.get_strategy()
        return strategy.execute(data, feature_config, **kwargs)

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要"""
        if self._strategy:
            return self._strategy.get_performance_metrics()
        return {}

    def reset_performance_metrics(self) -> None:
        """パフォーマンス指標のリセット"""
        if self._strategy:
            self._strategy.reset_metrics()


# 便利関数
def generate_features(
    data: pd.DataFrame,
    feature_config: Optional[FeatureConfig] = None,
    optimization_config: Optional[OptimizationConfig] = None,
    **kwargs
) -> FeatureResult:
    """特徴量生成のヘルパー関数"""
    manager = FeatureEngineeringManager(optimization_config)
    return manager.generate_features(data, feature_config, **kwargs)
