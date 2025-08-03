"""
パフォーマンス最適化済み特徴量エンジニアリング

大容量データ処理、並列計算、メモリ効率化に特化した
特徴量エンジニアリングモジュール。
"""

import gc
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import jit, njit
from scipy import stats

from ..utils.logging_config import get_context_logger, log_performance_metric
from ..utils.performance_analyzer import profile_performance

warnings.filterwarnings('ignore')
logger = get_context_logger(__name__)


@dataclass
class OptimizationConfig:
    """最適化設定"""

    enable_parallel: bool = True
    enable_numba: bool = True
    enable_vectorization: bool = True
    chunk_size: int = 10000
    max_workers: int = 4
    memory_limit_mb: int = 1000


# Numba最適化関数群
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

    if avg_loss != 0:
        rs = avg_gain / avg_loss
        rsi[window] = 100.0 - (100.0 / (1.0 + rs))

    # 指数移動平均で更新
    alpha = 1.0 / window
    for i in range(window + 1, n):
        avg_gain = (1 - alpha) * avg_gain + alpha * gains[i - 1]
        avg_loss = (1 - alpha) * avg_loss + alpha * losses[i - 1]

        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@njit
def fast_bollinger_bands(prices: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """高速ボリンジャーバンド計算（Numba最適化）"""
    n = len(prices)
    sma = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = prices[i - window + 1:i + 1]
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)

        sma[i] = mean_val
        upper[i] = mean_val + num_std * std_val
        lower[i] = mean_val - num_std * std_val

    return sma, upper, lower


@njit
def fast_momentum_indicators(prices: np.ndarray, volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """高速モメンタム指標計算"""
    n = len(prices)
    price_momentum = np.full(n, 0.0)
    volume_momentum = np.full(n, 0.0)

    # 価格モメンタム（5日）
    for i in range(5, n):
        price_momentum[i] = (prices[i] - prices[i - 5]) / prices[i - 5] * 100.0

    # 出来高モメンタム（5日）
    for i in range(5, n):
        if volumes[i - 5] != 0:
            volume_momentum[i] = (volumes[i] - volumes[i - 5]) / volumes[i - 5] * 100.0

    return price_momentum, volume_momentum


class OptimizedDataQualityEnhancer:
    """最適化済みデータ品質向上器"""

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers) if self.config.enable_parallel else None

        logger.info(
            "最適化済みデータ品質向上器初期化",
            section="data_quality_init",
            parallel_enabled=self.config.enable_parallel,
            numba_enabled=self.config.enable_numba
        )

    @profile_performance
    def clean_ohlcv_data(
        self,
        df: pd.DataFrame,
        remove_outliers: bool = True,
        outlier_method: str = 'iqr',
        smooth_data: bool = True,
        fill_missing: bool = True
    ) -> pd.DataFrame:
        """OHLCV データクリーニング（最適化済み）"""

        logger.info(
            "データクリーニング開始",
            section="data_cleaning",
            data_shape=df.shape,
            method=outlier_method
        )

        # データコピー（メモリ効率化）
        cleaned_df = df.copy()

        if fill_missing:
            cleaned_df = self._fill_missing_values_optimized(cleaned_df)

        if remove_outliers:
            cleaned_df = self._remove_outliers_optimized(cleaned_df, method=outlier_method)

        if smooth_data:
            cleaned_df = self._smooth_data_optimized(cleaned_df)

        # データ整合性チェック
        cleaned_df = self._ensure_data_consistency(cleaned_df)

        logger.info(
            "データクリーニング完了",
            section="data_cleaning",
            original_size=len(df),
            cleaned_size=len(cleaned_df)
        )

        return cleaned_df

    def _fill_missing_values_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値補完（最適化済み）"""
        # 数値列のみを効率的に処理
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].isnull().any():
                if col in ['Volume']:
                    # 出来高は前日値で補完
                    df[col] = df[col].fillna(method='ffill')
                else:
                    # 価格系は線形補間
                    df[col] = df[col].interpolate(method='linear')

        # 残りの欠損値は前方補完
        df = df.fillna(method='ffill').fillna(method='bfill')

        return df

    def _remove_outliers_optimized(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """外れ値除去（最適化済み）"""
        if method == 'iqr':
            return self._remove_outliers_iqr_vectorized(df)
        elif method == 'zscore':
            return self._remove_outliers_zscore_vectorized(df)
        else:
            return df

    def _remove_outliers_iqr_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """IQR法による外れ値除去（ベクトル化）"""
        price_cols = ['Open', 'High', 'Low', 'Close']

        for col in price_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # ベクトル化された外れ値処理
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

        return df

    def _remove_outliers_zscore_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score法による外れ値除去（ベクトル化）"""
        price_cols = ['Open', 'High', 'Low', 'Close']
        threshold = 3.0

        for col in price_cols:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
                median_val = df[col].median()

                # 外れ値を中央値で置換
                df[col] = np.where(z_scores > threshold, median_val, df[col])

        return df

    def _smooth_data_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ平滑化（最適化済み）"""
        smoothed_df = df.copy()

        # 指数移動平均による平滑化（軽量）
        alpha = 0.1  # 平滑化パラメータ
        price_cols = ['Open', 'High', 'Low', 'Close']

        for col in price_cols:
            if col in smoothed_df.columns:
                # pandas の ewm を使用（高速）
                smoothed_df[col] = smoothed_df[col].ewm(alpha=alpha).mean()

        return smoothed_df

    def _ensure_data_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ整合性確保"""
        # OHLC の整合性チェック（ベクトル化）
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High は Open, Close より大きくなければならない
            df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))

            # Low は Open, Close より小さくなければならない
            df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))

        return df


class OptimizedAdvancedFeatureEngineer:
    """最適化済み高度特徴量エンジニア"""

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.feature_cache = {}
        self.executor = ProcessPoolExecutor(max_workers=self.config.max_workers) if self.config.enable_parallel else None

        logger.info(
            "最適化済み特徴量エンジニア初期化",
            section="feature_engineering_init",
            parallel_enabled=self.config.enable_parallel
        )

    @profile_performance
    def generate_composite_features(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        enable_advanced: bool = True
    ) -> pd.DataFrame:
        """複合特徴量生成（最適化済み）"""

        logger.info(
            "複合特徴量生成開始",
            section="feature_generation",
            data_shape=data.shape,
            indicators_count=len(indicators)
        )

        # ベースデータコピー
        feature_data = data.copy()

        # 並列処理用のタスク分割
        if self.config.enable_parallel and len(data) > self.config.chunk_size:
            feature_data = self._generate_features_parallel(feature_data, indicators, enable_advanced)
        else:
            feature_data = self._generate_features_sequential(feature_data, indicators, enable_advanced)

        # メモリクリーンアップ
        self._cleanup_memory()

        logger.info(
            "複合特徴量生成完了",
            section="feature_generation",
            final_features=feature_data.shape[1],
            features_added=feature_data.shape[1] - data.shape[1]
        )

        return feature_data

    def _generate_features_parallel(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        enable_advanced: bool
    ) -> pd.DataFrame:
        """並列特徴量生成"""

        # データをチャンクに分割
        chunks = [data.iloc[i:i + self.config.chunk_size]
                 for i in range(0, len(data), self.config.chunk_size)]

        # 並列処理実行
        def process_chunk(chunk):
            return self._generate_features_for_chunk(chunk, indicators, enable_advanced)

        if self.executor:
            futures = [self.executor.submit(process_chunk, chunk) for chunk in chunks]
            processed_chunks = [future.result() for future in futures]
        else:
            processed_chunks = [process_chunk(chunk) for chunk in chunks]

        # チャンクを結合
        return pd.concat(processed_chunks, ignore_index=False)

    def _generate_features_sequential(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        enable_advanced: bool
    ) -> pd.DataFrame:
        """逐次特徴量生成"""
        return self._generate_features_for_chunk(data, indicators, enable_advanced)

    def _generate_features_for_chunk(
        self,
        chunk: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        enable_advanced: bool
    ) -> pd.DataFrame:
        """チャンク単位での特徴量生成"""

        # 基本価格特徴量（Numba最適化）
        chunk = self._add_price_features_optimized(chunk)

        # ボラティリティ特徴量
        chunk = self._add_volatility_features_optimized(chunk)

        # テクニカル指標特徴量
        chunk = self._add_technical_features_optimized(chunk, indicators)

        if enable_advanced:
            # 高度な特徴量
            chunk = self._add_advanced_features_optimized(chunk)

        return chunk

    def _add_price_features_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """価格特徴量追加（最適化済み）"""

        if self.config.enable_numba:
            # Numba最適化版
            prices = data['Close'].values

            # リターン計算（複数期間）
            for period in [1, 3, 5, 10]:
                returns = np.full(len(prices), 0.0)
                for i in range(period, len(prices)):
                    if prices[i - period] != 0:
                        returns[i] = (prices[i] - prices[i - period]) / prices[i - period]
                data[f'returns_{period}d'] = returns

            # 相対価格位置
            for window in [10, 20, 50]:
                rel_position = np.full(len(prices), 0.5)
                for i in range(window, len(prices)):
                    window_data = prices[i - window:i]
                    min_price = np.min(window_data)
                    max_price = np.max(window_data)
                    if max_price != min_price:
                        rel_position[i] = (prices[i] - min_price) / (max_price - min_price)
                data[f'rel_position_{window}d'] = rel_position
        else:
            # pandas版（fallback）
            for period in [1, 3, 5, 10]:
                data[f'returns_{period}d'] = data['Close'].pct_change(period)

            for window in [10, 20, 50]:
                rolling_min = data['Close'].rolling(window).min()
                rolling_max = data['Close'].rolling(window).max()
                data[f'rel_position_{window}d'] = (data['Close'] - rolling_min) / (rolling_max - rolling_min)

        return data

    def _add_volatility_features_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ特徴量追加（最適化済み）"""

        # リアライズドボラティリティ（効率的計算）
        returns = data['Close'].pct_change().fillna(0)

        for window in [5, 10, 20]:
            # 標準偏差ベース
            data[f'realized_vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)

            # Parkinson推定量（High-Low ベース）
            if all(col in data.columns for col in ['High', 'Low']):
                hl_ratio = np.log(data['High'] / data['Low']) ** 2
                data[f'parkinson_vol_{window}d'] = hl_ratio.rolling(window).mean() * np.sqrt(252 / (4 * np.log(2)))

        return data

    def _add_technical_features_optimized(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> pd.DataFrame:
        """テクニカル指標特徴量追加（最適化済み）"""

        # 高速RSI計算
        if self.config.enable_numba:
            prices = data['Close'].values
            rsi_values = fast_rsi_calculation(prices, 14)
            data['rsi_optimized'] = rsi_values

            # 高速ボリンジャーバンド
            bb_sma, bb_upper, bb_lower = fast_bollinger_bands(prices, 20, 2.0)
            data['bb_position'] = (prices - bb_sma) / (bb_upper - bb_sma)
            data['bb_width'] = (bb_upper - bb_lower) / bb_sma

        # 既存指標の統合
        for name, series in indicators.items():
            if len(series) == len(data):
                data[f'indicator_{name}'] = series.values

        # テクニカル強度複合指標
        if 'rsi' in indicators:
            rsi_values = indicators['rsi'].reindex(data.index, fill_value=50)
            data['technical_strength'] = (
                (rsi_values - 50) / 50 * 0.4 +  # RSIの寄与
                data.get('bb_position', 0) * 0.3 +  # ボリンジャーバンドの寄与
                data['returns_5d'].fillna(0) * 0.3  # 短期リターンの寄与
            )

        return data

    def _add_advanced_features_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """高度特徴量追加（最適化済み）"""

        # モメンタム指標（Numba最適化）
        if self.config.enable_numba and all(col in data.columns for col in ['Close', 'Volume']):
            prices = data['Close'].values
            volumes = data['Volume'].values

            price_momentum, volume_momentum = fast_momentum_indicators(prices, volumes)
            data['price_momentum_5d'] = price_momentum
            data['volume_momentum_5d'] = volume_momentum

        # 時系列特徴量（軽量版）
        data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24) if hasattr(data.index, 'hour') else 0
        data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24) if hasattr(data.index, 'hour') else 0
        data['day_of_week_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7) if hasattr(data.index, 'dayofweek') else 0
        data['day_of_week_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7) if hasattr(data.index, 'dayofweek') else 0

        # 複合統計指標
        returns = data['Close'].pct_change().fillna(0)

        # 歪度・尖度（効率的計算）
        for window in [10, 20]:
            data[f'skewness_{window}d'] = returns.rolling(window).skew()
            data[f'kurtosis_{window}d'] = returns.rolling(window).kurt()

        return data

    def generate_market_features(
        self,
        feature_data: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """市場特徴量生成（軽量版）"""

        for market_name, market_df in market_data.items():
            if 'Close' in market_df.columns:
                # 市場相関（効率的計算）
                market_returns = market_df['Close'].pct_change().fillna(0)
                stock_returns = feature_data['Close'].pct_change().fillna(0)

                # ローリング相関（軽量版）
                correlation = stock_returns.rolling(20).corr(market_returns.reindex(stock_returns.index, fill_value=0))
                feature_data[f'correlation_{market_name}'] = correlation.fillna(0)

                # 相対パフォーマンス
                relative_perf = stock_returns - market_returns.reindex(stock_returns.index, fill_value=0)
                feature_data[f'relative_performance_{market_name}'] = relative_perf.rolling(10).mean()

        return feature_data

    def _cleanup_memory(self):
        """メモリクリーンアップ"""
        # キャッシュクリア
        if len(self.feature_cache) > 100:
            self.feature_cache.clear()

        # ガベージコレクション
        gc.collect()

    def __del__(self):
        """リソースクリーンアップ"""
        if self.executor:
            self.executor.shutdown(wait=False)


# ユーティリティ関数
@profile_performance
def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame メモリ使用量最適化"""
    optimized_df = df.copy()

    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype

        if col_type == 'float64':
            # float32 に変換可能かチェック
            if optimized_df[col].min() >= np.finfo(np.float32).min and optimized_df[col].max() <= np.finfo(np.float32).max:
                optimized_df[col] = optimized_df[col].astype(np.float32)

        elif col_type == 'int64':
            # より小さい整数型に変換可能かチェック
            if optimized_df[col].min() >= np.iinfo(np.int32).min and optimized_df[col].max() <= np.iinfo(np.int32).max:
                optimized_df[col] = optimized_df[col].astype(np.int32)

    logger.debug(
        "DataFrameメモリ最適化完了",
        section="memory_optimization",
        original_memory=df.memory_usage(deep=True).sum() / 1024**2,
        optimized_memory=optimized_df.memory_usage(deep=True).sum() / 1024**2
    )

    return optimized_df


# 使用例とデモ
if __name__ == "__main__":
    logger.info("最適化済み特徴量エンジニアリングデモ開始", section="demo")

    try:
        # テストデータ生成
        dates = pd.date_range(end=pd.Timestamp.now(), periods=5000, freq="1min")  # 軽量化
        np.random.seed(42)

        base_price = 1000
        prices = base_price + np.cumsum(np.random.randn(5000) * 1)

        test_data = pd.DataFrame({
            'Open': prices + np.random.randn(5000) * 0.5,
            'High': prices + np.abs(np.random.randn(5000)) * 1,
            'Low': prices - np.abs(np.random.randn(5000)) * 1,
            'Close': prices,
            'Volume': np.random.randint(100000, 500000, 5000)
        }, index=dates)

        # 基本指標
        indicators = {
            'rsi': pd.Series(50 + np.random.randn(5000) * 15, index=dates),
            'macd': pd.Series(np.random.randn(5000), index=dates)
        }

        # 最適化済みコンポーネント
        config = OptimizationConfig(
            enable_parallel=True,
            enable_numba=True,
            chunk_size=1000
        )

        data_enhancer = OptimizedDataQualityEnhancer(config)
        feature_engineer = OptimizedAdvancedFeatureEngineer(config)

        # データクリーニング
        clean_data = data_enhancer.clean_ohlcv_data(test_data)

        # 特徴量生成
        feature_data = feature_engineer.generate_composite_features(clean_data, indicators)

        # メモリ最適化
        optimized_data = optimize_dataframe_memory(feature_data)

        logger.info(
            "最適化済み特徴量エンジニアリングデモ完了",
            section="demo",
            original_shape=test_data.shape,
            final_shape=optimized_data.shape,
            features_added=optimized_data.shape[1] - test_data.shape[1]
        )

    except Exception as e:
        logger.error(f"デモ実行エラー: {e}", section="demo")

    finally:
        # メモリクリーンアップ
        gc.collect()
