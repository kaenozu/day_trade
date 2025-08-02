"""
高度な特徴量エンジニアリングモジュール

複数のテクニカル指標を組み合わせた複合特徴量の生成、
市場全体の特徴量抽出、データ品質向上処理を提供する。
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ..utils.logging_config import get_context_logger, log_performance_metric

logger = get_context_logger(__name__)


class DataQualityEnhancer:
    """データ品質向上処理クラス"""

    def __init__(self):
        self.outlier_methods = {
            'iqr': self._remove_outliers_iqr,
            'zscore': self._remove_outliers_zscore,
            'isolation_forest': self._remove_outliers_isolation_forest
        }

    def clean_ohlcv_data(
        self,
        df: pd.DataFrame,
        remove_outliers: bool = True,
        outlier_method: str = 'iqr',
        smooth_data: bool = True,
        fill_missing: bool = True
    ) -> pd.DataFrame:
        """
        OHLCV データの品質向上処理

        Args:
            df: OHLCV データフレーム
            remove_outliers: 外れ値除去フラグ
            outlier_method: 外れ値除去手法 ('iqr', 'zscore', 'isolation_forest')
            smooth_data: データ平滑化フラグ
            fill_missing: 欠損値補完フラグ

        Returns:
            クリーニング済みデータフレーム
        """
        logger.info(
            "データ品質向上処理開始",
            section="data_cleaning",
            rows=len(df),
            remove_outliers=remove_outliers,
            outlier_method=outlier_method,
            smooth_data=smooth_data
        )

        df_cleaned = df.copy()

        # 基本的な妥当性チェック
        df_cleaned = self._validate_ohlcv_consistency(df_cleaned)

        # 欠損値処理
        if fill_missing:
            df_cleaned = self._fill_missing_values(df_cleaned)

        # 外れ値除去
        if remove_outliers and outlier_method in self.outlier_methods:
            df_cleaned = self.outlier_methods[outlier_method](df_cleaned)

        # データ平滑化
        if smooth_data:
            df_cleaned = self._smooth_data(df_cleaned)

        logger.info(
            "データ品質向上処理完了",
            section="data_cleaning",
            original_rows=len(df),
            cleaned_rows=len(df_cleaned),
            removed_rows=len(df) - len(df_cleaned)
        )

        return df_cleaned

    def _validate_ohlcv_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """OHLCV データの整合性チェックと修正"""
        df = df.copy()

        # High >= Low の確認
        invalid_hl = df['High'] < df['Low']
        if invalid_hl.any():
            logger.warning(
                "High < Low の異常データを検出",
                section="data_validation",
                invalid_count=invalid_hl.sum()
            )
            # High と Low を入れ替え
            df.loc[invalid_hl, ['High', 'Low']] = df.loc[invalid_hl, ['Low', 'High']].values

        # Open, Close が High-Low 範囲内にあることを確認
        for col in ['Open', 'Close']:
            below_low = df[col] < df['Low']
            above_high = df[col] > df['High']

            if below_low.any():
                df.loc[below_low, col] = df.loc[below_low, 'Low']
            if above_high.any():
                df.loc[above_high, col] = df.loc[above_high, 'High']

        # Volume の負値チェック
        negative_volume = df['Volume'] < 0
        if negative_volume.any():
            logger.warning(
                "負の出来高データを検出",
                section="data_validation",
                negative_count=negative_volume.sum()
            )
            df.loc[negative_volume, 'Volume'] = 0

        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値補完"""
        df = df.copy()

        # 価格データは前方補完 + 後方補完
        price_cols = ['Open', 'High', 'Low', 'Close']
        df[price_cols] = df[price_cols].fillna(method='ffill').fillna(method='bfill')

        # 出来高は0で補完
        df['Volume'] = df['Volume'].fillna(0)

        return df

    def _remove_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """IQR法による外れ値除去"""
        df_clean = df.copy()

        for col in ['Open', 'High', 'Low', 'Close']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            df_clean = df_clean[~outliers]

        return df_clean

    def _remove_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Z-score法による外れ値除去"""
        df_clean = df.copy()

        for col in ['Open', 'High', 'Low', 'Close']:
            z_scores = np.abs(stats.zscore(df_clean[col]))
            outliers = z_scores > threshold
            df_clean = df_clean[~outliers]

        return df_clean

    def _remove_outliers_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Isolation Forest による外れ値除去"""
        try:
            from sklearn.ensemble import IsolationForest

            price_data = df[['Open', 'High', 'Low', 'Close']].values
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(price_data) == -1

            return df[~outliers]
        except ImportError:
            logger.warning(
                "scikit-learn が利用できないため、IQR法を使用",
                section="outlier_removal"
            )
            return self._remove_outliers_iqr(df)

    def _smooth_data(self, df: pd.DataFrame, window_length: int = 5) -> pd.DataFrame:
        """Savitzky-Golay フィルターによるデータ平滑化"""
        df_smooth = df.copy()

        if len(df_smooth) >= window_length:
            for col in ['Open', 'High', 'Low', 'Close']:
                try:
                    df_smooth[col] = savgol_filter(
                        df_smooth[col],
                        window_length=window_length,
                        polyorder=2
                    )
                except Exception as e:
                    logger.warning(
                        f"データ平滑化エラー: {col}",
                        section="data_smoothing",
                        error=str(e)
                    )

        return df_smooth


class AdvancedFeatureEngineer:
    """高度な特徴量エンジニアリングクラス"""

    def __init__(self):
        self.scaler = None

    def generate_composite_features(self, df: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
        """
        複合特徴量の生成

        Args:
            df: OHLCV データフレーム
            indicators: テクニカル指標辞書

        Returns:
            複合特徴量を含むデータフレーム
        """
        logger.info("複合特徴量生成開始", section="feature_engineering")

        features_df = df.copy()

        # 基本的な価格変化特徴量
        features_df = self._add_price_change_features(features_df)

        # ボラティリティ特徴量
        features_df = self._add_volatility_features(features_df)

        # 複合テクニカル指標
        features_df = self._add_composite_technical_features(features_df, indicators)

        # 統計的特徴量
        features_df = self._add_statistical_features(features_df)

        # 時間的特徴量
        features_df = self._add_temporal_features(features_df)

        logger.info(
            "複合特徴量生成完了",
            section="feature_engineering",
            total_features=len(features_df.columns),
            original_features=len(df.columns)
        )

        return features_df

    def _add_price_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """価格変化関連の特徴量"""
        df = df.copy()

        # 基本的なリターン
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_3d'] = df['Close'].pct_change(3)
        df['returns_7d'] = df['Close'].pct_change(7)
        df['returns_14d'] = df['Close'].pct_change(14)

        # 対数リターン
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # 価格レンジ
        df['daily_range'] = (df['High'] - df['Low']) / df['Open']
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

        # ボディとヒゲの比率
        df['body_ratio'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])
        df['upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / (df['High'] - df['Low'])
        df['lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / (df['High'] - df['Low'])

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ関連の特徴量"""
        df = df.copy()

        # 実現ボラティリティ（複数期間）
        for window in [5, 10, 20, 50]:
            df[f'realized_vol_{window}d'] = df['returns_1d'].rolling(window).std() * np.sqrt(252)

        # Parkinson ボラティリティ推定
        df['parkinson_vol'] = np.sqrt(
            np.log(df['High'] / df['Low']) ** 2 / (4 * np.log(2))
        ).rolling(20).mean() * np.sqrt(252)

        # Garman-Klass ボラティリティ推定
        df['gk_vol'] = (
            0.5 * np.log(df['High'] / df['Low']) ** 2 -
            (2 * np.log(2) - 1) * np.log(df['Close'] / df['Open']) ** 2
        ).rolling(20).mean() * np.sqrt(252)

        return df

    def _add_composite_technical_features(self, df: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
        """複合テクニカル指標特徴量"""
        df = df.copy()

        if 'rsi' in indicators:
            rsi = indicators['rsi']
            # RSI の勢い
            df['rsi_momentum'] = rsi.diff(5)
            # RSI の相対位置
            df['rsi_relative'] = (rsi - rsi.rolling(20).mean()) / rsi.rolling(20).std()

        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            # MACD の収束・発散
            df['macd_convergence'] = (macd - macd_signal).rolling(10).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1
            )

        if 'bb_upper' in indicators and 'bb_lower' in indicators:
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            # ボリンジャーバンド位置
            df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
            # バンド幅の変化
            df['bb_width'] = (bb_upper - bb_lower) / df['Close']
            df['bb_width_change'] = df['bb_width'].pct_change(5)

        # 複数指標の合成スコア
        if all(key in indicators for key in ['rsi', 'macd']):
            # テクニカル強度スコア
            rsi_norm = (indicators['rsi'] - 50) / 50  # -1 to 1
            macd_norm = np.tanh(indicators['macd'] / df['Close'].rolling(20).std())  # -1 to 1
            df['technical_strength'] = (rsi_norm + macd_norm) / 2

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """統計的特徴量"""
        df = df.copy()

        # 移動平均からの乖離
        for window in [5, 20, 50]:
            ma = df['Close'].rolling(window).mean()
            df[f'ma_deviation_{window}'] = (df['Close'] - ma) / ma

        # 価格分布の特徴
        for window in [20, 50]:
            rolling = df['Close'].rolling(window)
            df[f'skewness_{window}'] = rolling.skew()
            df[f'kurtosis_{window}'] = rolling.kurt()

        # 出来高関連統計
        df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_volatility'] = df['Volume'].rolling(20).std() / df['Volume'].rolling(20).mean()

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間的特徴量"""
        df = df.copy()

        # 日付インデックスから時間特徴量を抽出
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter

            # 周期性のエンコーディング
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def generate_market_features(self, symbol_data: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        市場全体の特徴量を生成

        Args:
            symbol_data: 対象銘柄データ
            market_data: 市場データ辞書 (e.g., {'NIKKEI': df, 'USDJPY': df})

        Returns:
            市場特徴量を追加したデータフレーム
        """
        logger.info("市場特徴量生成開始", section="market_features")

        features_df = symbol_data.copy()

        for market_name, market_df in market_data.items():
            if 'Close' not in market_df.columns:
                continue

            # 市場との相関
            correlation = self._calculate_rolling_correlation(
                symbol_data['Close'], market_df['Close'], window=50
            )
            features_df[f'{market_name}_correlation'] = correlation

            # 市場のリターンとの関係
            market_returns = market_df['Close'].pct_change()
            symbol_returns = symbol_data['Close'].pct_change()

            beta = self._calculate_rolling_beta(symbol_returns, market_returns, window=50)
            features_df[f'{market_name}_beta'] = beta

            # 市場のボラティリティ
            market_vol = market_returns.rolling(20).std() * np.sqrt(252)
            features_df[f'{market_name}_volatility'] = market_vol.reindex(features_df.index, method='ffill')

        logger.info("市場特徴量生成完了", section="market_features")
        return features_df

    def _calculate_rolling_correlation(self, series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
        """ローリング相関計算"""
        return series1.rolling(window).corr(series2.reindex(series1.index, method='ffill'))

    def _calculate_rolling_beta(self, asset_returns: pd.Series, market_returns: pd.Series, window: int) -> pd.Series:
        """ローリングベータ計算"""
        def beta_calc(y, x):
            if len(y) < window or len(x) < window:
                return np.nan
            covariance = np.cov(y, x)[0, 1]
            market_variance = np.var(x)
            return covariance / market_variance if market_variance != 0 else np.nan

        aligned_market = market_returns.reindex(asset_returns.index, method='ffill')

        beta_series = []
        for i in range(len(asset_returns)):
            if i >= window - 1:
                y_window = asset_returns.iloc[i-window+1:i+1]
                x_window = aligned_market.iloc[i-window+1:i+1]
                beta_series.append(beta_calc(y_window.values, x_window.values))
            else:
                beta_series.append(np.nan)

        return pd.Series(beta_series, index=asset_returns.index)

    def normalize_features(self, df: pd.DataFrame, method: str = 'robust') -> Tuple[pd.DataFrame, object]:
        """
        特徴量の正規化

        Args:
            df: 特徴量データフレーム
            method: 正規化手法 ('standard', 'minmax', 'robust')

        Returns:
            正規化済みデータフレームとscalerオブジェクト
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_normalized = df.copy()

        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

        if method not in scaler_map:
            raise ValueError(f"サポートされていない正規化手法: {method}")

        scaler = scaler_map[method]
        df_normalized[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        self.scaler = scaler
        return df_normalized, scaler

    def select_important_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'correlation',
        top_k: int = 50
    ) -> List[str]:
        """
        重要な特徴量の選択

        Args:
            X: 特徴量データフレーム
            y: ターゲット変数
            method: 特徴量選択手法 ('correlation', 'mutual_info', 'chi2')
            top_k: 選択する特徴量数

        Returns:
            選択された特徴量名のリスト
        """
        if method == 'correlation':
            # 相関による特徴量選択
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            return correlations.head(top_k).index.tolist()

        elif method == 'mutual_info':
            try:
                from sklearn.feature_selection import mutual_info_regression

                # 欠損値を含む行を除去
                mask = ~(X.isna().any(axis=1) | y.isna())
                X_clean = X[mask]
                y_clean = y[mask]

                mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
                feature_scores = pd.Series(mi_scores, index=X_clean.columns).sort_values(ascending=False)
                return feature_scores.head(top_k).index.tolist()

            except ImportError:
                logger.warning("scikit-learn が利用できないため、相関による選択を使用")
                return self.select_important_features(X, y, 'correlation', top_k)

        else:
            logger.warning(f"未対応の特徴量選択手法: {method}、相関による選択を使用")
            return self.select_important_features(X, y, 'correlation', top_k)


# ユーティリティ関数

def calculate_feature_importance_scores(
    X: pd.DataFrame,
    y: pd.Series,
    feature_engineer: AdvancedFeatureEngineer
) -> Dict[str, float]:
    """特徴量重要度スコアの計算"""
    importance_scores = {}

    # 相関ベース重要度
    correlations = X.corrwith(y).abs()
    importance_scores['correlation'] = correlations.to_dict()

    # 分散ベース重要度（低分散特徴量の除外用）
    variances = X.var()
    normalized_variances = (variances - variances.min()) / (variances.max() - variances.min())
    importance_scores['variance'] = normalized_variances.to_dict()

    return importance_scores


def generate_feature_report(df: pd.DataFrame, target_col: str = None) -> Dict:
    """特徴量分析レポート生成"""
    report = {
        'basic_stats': df.describe(),
        'missing_values': df.isnull().sum(),
        'data_types': df.dtypes,
        'feature_count': len(df.columns),
        'sample_count': len(df)
    }

    if target_col and target_col in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        report['target_correlations'] = correlations

    return report


# 使用例とデモ
if __name__ == "__main__":
    # サンプルデータ生成
    import yfinance as yf

    # データ取得
    ticker = "7203.T"  # トヨタ
    data = yf.download(ticker, period="1y")

    # データ品質向上
    quality_enhancer = DataQualityEnhancer()
    clean_data = quality_enhancer.clean_ohlcv_data(data)

    # 基本的なテクニカル指標（ダミー）
    indicators = {
        'rsi': clean_data['Close'].rolling(14).apply(lambda x: 50),  # ダミーRSI
        'macd': clean_data['Close'].ewm(12).mean() - clean_data['Close'].ewm(26).mean(),
        'macd_signal': clean_data['Close'].rolling(9).mean(),
        'bb_upper': clean_data['Close'].rolling(20).mean() + clean_data['Close'].rolling(20).std() * 2,
        'bb_lower': clean_data['Close'].rolling(20).mean() - clean_data['Close'].rolling(20).std() * 2
    }

    # 特徴量エンジニアリング
    feature_engineer = AdvancedFeatureEngineer()
    feature_data = feature_engineer.generate_composite_features(clean_data, indicators)

    # 特徴量レポート生成
    report = generate_feature_report(feature_data, 'returns_1d')

    logger.info(
        "特徴量エンジニアリングデモ完了",
        section="demo",
        features_generated=len(feature_data.columns),
        top_correlations=report.get('target_correlations', {}).head(10).to_dict() if 'target_correlations' in report else {}
    )
