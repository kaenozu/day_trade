"""
高度な特徴量エンジニアリング
テクニカル指標の複合化、市場全体特徴量、時系列特有の特徴量を生成
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="feature_engineering")


@dataclass
class FeatureConfig:
    """特徴量生成設定"""

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


class AdvancedFeatureEngineer:
    """高度な特徴量エンジニアリング"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Args:
            config: 特徴量生成設定
        """
        self.config = config or FeatureConfig(
            lookback_periods=[5, 10, 20, 50],
            volatility_windows=[5, 10, 20],
            momentum_periods=[1, 3, 5, 10, 20],
        )

        # スケーラーの初期化
        if self.config.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None

    def generate_all_features(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.Series] = None,
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        全ての高度な特徴量を生成

        Args:
            price_data: 価格データ（OHLCV）
            volume_data: 出来高データ
            market_data: 市場全体データ（インデックス、セクターなど）

        Returns:
            生成された特徴量DataFrame
        """
        logger.info("高度な特徴量生成を開始")

        features = pd.DataFrame(index=price_data.index)

        try:
            # 1. 基本価格特徴量
            basic_features = self._generate_basic_features(price_data)
            features = pd.concat([features, basic_features], axis=1)

            # 2. 複合テクニカル特徴量
            if self.config.enable_cross_features:
                cross_features = self._generate_cross_features(price_data)
                features = pd.concat([features, cross_features], axis=1)

            # 3. 統計的特徴量
            if self.config.enable_statistical_features:
                stat_features = self._generate_statistical_features(price_data)
                features = pd.concat([features, stat_features], axis=1)

            # 4. 市場レジーム特徴量
            if self.config.enable_regime_features:
                regime_features = self._generate_regime_features(price_data)
                features = pd.concat([features, regime_features], axis=1)

            # 5. 出来高特徴量
            if volume_data is not None:
                volume_features = self._generate_volume_features(
                    price_data, volume_data
                )
                features = pd.concat([features, volume_features], axis=1)

            # 6. 市場全体特徴量
            if market_data:
                market_features = self._generate_market_features(
                    price_data, market_data
                )
                features = pd.concat([features, market_features], axis=1)

            # 7. 時系列ラグ特徴量
            lag_features = self._generate_lag_features(price_data)
            features = pd.concat([features, lag_features], axis=1)

            # 8. 外れ値除去と正規化
            features = self._preprocess_features(features)

            logger.info(f"特徴量生成完了: {len(features.columns)}個の特徴量")
            return features

        except Exception as e:
            logger.error(f"特徴量生成エラー: {e}")
            return pd.DataFrame(index=price_data.index)

    def _generate_basic_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """基本価格特徴量の生成"""
        features = pd.DataFrame(index=price_data.index)

        # リターン系特徴量
        features["returns_1d"] = price_data["Close"].pct_change()
        features["returns_log"] = np.log(
            price_data["Close"] / price_data["Close"].shift(1)
        )

        # 価格レンジ特徴量
        features["true_range"] = np.maximum(
            price_data["High"] - price_data["Low"],
            np.maximum(
                abs(price_data["High"] - price_data["Close"].shift(1)),
                abs(price_data["Low"] - price_data["Close"].shift(1)),
            ),
        )

        features["price_range_pct"] = (
            price_data["High"] - price_data["Low"]
        ) / price_data["Close"]
        features["body_to_range"] = abs(price_data["Close"] - price_data["Open"]) / (
            price_data["High"] - price_data["Low"] + 1e-8
        )

        # 複数期間のボラティリティ
        for window in self.config.volatility_windows:
            features[f"volatility_{window}d"] = (
                features["returns_1d"].rolling(window).std()
            )
            features[f"volatility_log_{window}d"] = (
                features["returns_log"].rolling(window).std()
            )

        # 複数期間のモメンタム
        for period in self.config.momentum_periods:
            features[f"momentum_{period}d"] = (
                price_data["Close"] / price_data["Close"].shift(period) - 1
            )
            features[f"momentum_log_{period}d"] = np.log(
                price_data["Close"] / price_data["Close"].shift(period)
            )

        return features

    def _generate_cross_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """複合テクニカル特徴量の生成"""
        features = pd.DataFrame(index=price_data.index)

        # 移動平均クロス特徴量
        for short_period in [5, 10, 20]:
            for long_period in [20, 50, 100]:
                if short_period < long_period:
                    short_ma = price_data["Close"].rolling(short_period).mean()
                    long_ma = price_data["Close"].rolling(long_period).mean()

                    features[f"ma_cross_{short_period}_{long_period}"] = (
                        short_ma / long_ma - 1
                    )
                    features[f"ma_cross_momentum_{short_period}_{long_period}"] = (
                        short_ma / long_ma
                    ).pct_change()

        # RSI組み合わせ
        for period in [14, 21]:
            # 簡易RSI計算
            delta = price_data["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta).where(delta < 0, 0).rolling(period).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))

            features[f"rsi_{period}"] = rsi
            features[f"rsi_{period}_momentum"] = rsi.diff()
            features[f"rsi_{period}_overbought"] = (rsi > 70).astype(int)
            features[f"rsi_{period}_oversold"] = (rsi < 30).astype(int)

        # ボリンジャーバンド複合特徴量
        for window in [20, 50]:
            sma = price_data["Close"].rolling(window).mean()
            std = price_data["Close"].rolling(window).std()

            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)

            features[f"bb_position_{window}"] = (price_data["Close"] - lower_band) / (
                upper_band - lower_band
            )
            features[f"bb_squeeze_{window}"] = (upper_band - lower_band) / sma
            features[f"bb_breakout_upper_{window}"] = (
                price_data["Close"] > upper_band
            ).astype(int)
            features[f"bb_breakout_lower_{window}"] = (
                price_data["Close"] < lower_band
            ).astype(int)

        return features

    def _generate_statistical_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """統計的特徴量の生成"""
        features = pd.DataFrame(index=price_data.index)

        returns = price_data["Close"].pct_change()

        for window in self.config.lookback_periods:
            # 歪度と尖度
            features[f"skewness_{window}d"] = returns.rolling(window).skew()
            features[f"kurtosis_{window}d"] = returns.rolling(window).kurt()

            # パーセンタイル特徴量
            features[f"percentile_rank_{window}d"] = (
                price_data["Close"]
                .rolling(window)
                .apply(
                    lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) / 100
                    if len(x) > 1
                    else 0.5
                )
            )

            # 最高値・最安値からの位置
            high_max = price_data["High"].rolling(window).max()
            low_min = price_data["Low"].rolling(window).min()
            features[f"high_low_position_{window}d"] = (
                price_data["Close"] - low_min
            ) / (high_max - low_min + 1e-8)

            # トレンド強度（線形回帰の傾き）
            features[f"trend_strength_{window}d"] = (
                price_data["Close"]
                .rolling(window)
                .apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
            )

        return features

    def _generate_regime_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """市場レジーム特徴量の生成"""
        features = pd.DataFrame(index=price_data.index)

        returns = price_data["Close"].pct_change()

        for window in [20, 50, 100]:
            # トレンド判定
            sma_short = price_data["Close"].rolling(window // 2).mean()
            sma_long = price_data["Close"].rolling(window).mean()

            features[f"trend_regime_{window}d"] = (sma_short > sma_long).astype(int)

            # ボラティリティレジーム
            vol = returns.rolling(window).std()
            vol_percentile = vol.rolling(window * 2).apply(
                lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) / 100
                if len(x) > 1
                else 0.5
            )
            features[f"vol_regime_high_{window}d"] = (vol_percentile > 0.75).astype(int)
            features[f"vol_regime_low_{window}d"] = (vol_percentile < 0.25).astype(int)

            # 平均回帰 vs トレンド継続
            autocorr = returns.rolling(window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 2 else 0
            )
            features[f"mean_reversion_{window}d"] = (autocorr < -0.1).astype(int)
            features[f"trend_continuation_{window}d"] = (autocorr > 0.1).astype(int)

        return features

    def _generate_volume_features(
        self, price_data: pd.DataFrame, volume_data: pd.Series
    ) -> pd.DataFrame:
        """出来高特徴量の生成"""
        features = pd.DataFrame(index=price_data.index)

        # 基本出来高特徴量
        features["volume_ma_ratio_20d"] = volume_data / volume_data.rolling(20).mean()
        features["volume_std_ratio_20d"] = volume_data / volume_data.rolling(20).std()

        # 価格-出来高関係
        returns = price_data["Close"].pct_change()
        features["price_volume_correlation_20d"] = returns.rolling(20).corr(
            volume_data.pct_change()
        )

        # On-Balance Volume (簡易版)
        obv_direction = np.where(returns > 0, 1, np.where(returns < 0, -1, 0))
        features["obv_normalized"] = (obv_direction * volume_data).rolling(
            20
        ).sum() / volume_data.rolling(20).sum()

        # Volume Rate of Change
        for period in [5, 10, 20]:
            features[f"volume_roc_{period}d"] = volume_data.pct_change(period)

        return features

    def _generate_market_features(
        self, price_data: pd.DataFrame, market_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """市場全体特徴量の生成"""
        features = pd.DataFrame(index=price_data.index)

        returns = price_data["Close"].pct_change()

        for market_name, market_df in market_data.items():
            if "Close" in market_df.columns:
                market_returns = market_df["Close"].pct_change()

                # 市場との相関
                for window in [20, 50]:
                    correlation = returns.rolling(window).corr(market_returns)
                    features[f"market_correlation_{market_name}_{window}d"] = (
                        correlation
                    )

                # 市場に対するベータ
                for window in [20, 50]:
                    beta = (
                        returns.rolling(window).cov(market_returns)
                        / market_returns.rolling(window).var()
                    )
                    features[f"market_beta_{market_name}_{window}d"] = beta

                # 相対パフォーマンス
                relative_performance = returns - market_returns
                features[f"relative_performance_{market_name}"] = relative_performance
                features[f"relative_performance_{market_name}_ma_20d"] = (
                    relative_performance.rolling(20).mean()
                )

        return features

    def _generate_lag_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """時系列ラグ特徴量の生成"""
        features = pd.DataFrame(index=price_data.index)

        returns = price_data["Close"].pct_change()

        # リターンのラグ特徴量
        for lag in [1, 2, 3, 5, 10]:
            features[f"returns_lag_{lag}d"] = returns.shift(lag)

        # 価格レベルのラグ特徴量（正規化）
        close_normalized = price_data["Close"] / price_data["Close"].rolling(20).mean()
        for lag in [1, 2, 3, 5]:
            features[f"price_normalized_lag_{lag}d"] = close_normalized.shift(lag)

        # ボラティリティのラグ特徴量
        volatility = returns.rolling(5).std()
        for lag in [1, 2, 3]:
            features[f"volatility_lag_{lag}d"] = volatility.shift(lag)

        return features

    def _preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """特徴量の前処理（外れ値除去・正規化）"""
        # 無限値とNaNの処理
        features = features.replace([np.inf, -np.inf], np.nan)

        # 外れ値のキャッピング（Z-score基準）
        for col in features.select_dtypes(include=[np.number]).columns:
            series = features[col]
            if series.std() > 0:
                z_scores = np.abs(stats.zscore(series.dropna()))
                outlier_mask = z_scores > self.config.outlier_threshold
                if outlier_mask.any():
                    # 外れ値を99%パーセンタイルでキャップ
                    percentile_99 = series.quantile(0.99)
                    percentile_1 = series.quantile(0.01)
                    features[col] = series.clip(lower=percentile_1, upper=percentile_99)

        # スケーリング
        if self.scaler is not None:
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                # 最初の100行でスケーラーをフィット（十分なデータがある場合）
                fit_data = features[numeric_columns].dropna()
                if len(fit_data) > 100:
                    self.scaler.fit(fit_data.iloc[:100])
                    scaled_data = self.scaler.transform(
                        features[numeric_columns].fillna(0)
                    )
                    features[numeric_columns] = scaled_data

        logger.info(
            f"特徴量前処理完了: {len(features.columns)}列, {features.shape[0]}行"
        )
        return features

    def select_important_features(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        method: str = "correlation",
        top_k: int = 50,
    ) -> List[str]:
        """
        重要な特徴量を選択

        Args:
            features: 特徴量DataFrame
            target: ターゲット変数
            method: 選択手法（correlation, mutual_info）
            top_k: 選択する特徴量数

        Returns:
            選択された特徴量名のリスト
        """
        valid_data = pd.concat([features, target], axis=1).dropna()

        if len(valid_data) < 100:
            logger.warning("特徴量選択に十分なデータがありません")
            return features.columns.tolist()[:top_k]

        X = valid_data.iloc[:, :-1]
        y = valid_data.iloc[:, -1]

        if method == "correlation":
            # 相関ベースの選択
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = correlations.head(top_k).index.tolist()

        elif method == "mutual_info":
            try:
                from sklearn.feature_selection import mutual_info_regression

                mi_scores = mutual_info_regression(X.fillna(0), y, random_state=42)
                feature_scores = pd.Series(mi_scores, index=X.columns).sort_values(
                    ascending=False
                )
                selected_features = feature_scores.head(top_k).index.tolist()
            except ImportError:
                logger.warning(
                    "scikit-learnが利用できません。相関ベースの選択を使用します"
                )
                correlations = X.corrwith(y).abs().sort_values(ascending=False)
                selected_features = correlations.head(top_k).index.tolist()

        logger.info(f"特徴量選択完了: {len(selected_features)}個の特徴量を選択")
        return selected_features


def create_target_variables(
    price_data: pd.DataFrame, prediction_horizon: int = 5
) -> Dict[str, pd.Series]:
    """
    予測用ターゲット変数を生成

    Args:
        price_data: 価格データ
        prediction_horizon: 予測期間

    Returns:
        ターゲット変数の辞書
    """
    targets = {}

    # 将来リターン
    future_returns = (
        price_data["Close"].pct_change(prediction_horizon).shift(-prediction_horizon)
    )
    targets["future_returns"] = future_returns

    # 将来の方向性（上昇=1, 下降=0）
    targets["future_direction"] = (future_returns > 0).astype(int)

    # 将来の大きな変動（閾値以上の変動=1）
    volatility_threshold = future_returns.rolling(100).std().median()
    targets["future_high_volatility"] = (
        abs(future_returns) > volatility_threshold
    ).astype(int)

    # 将来の最大ドローダウン
    future_prices = pd.concat(
        [price_data["Close"].shift(-i) for i in range(1, prediction_horizon + 1)],
        axis=1,
    )
    future_max_dd = future_prices.min(axis=1) / price_data["Close"] - 1
    targets["future_max_drawdown"] = future_max_dd

    return targets


if __name__ == "__main__":
    # テスト用のサンプルデータ

    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)

    # トレンドのある価格データを生成
    trend = np.linspace(100, 120, len(dates))
    noise = np.random.randn(len(dates)) * 2
    close_prices = trend + noise

    sample_data = pd.DataFrame(
        {
            "Date": dates,
            "Open": close_prices + np.random.randn(len(dates)) * 0.5,
            "High": close_prices + np.abs(np.random.randn(len(dates))) * 2,
            "Low": close_prices - np.abs(np.random.randn(len(dates))) * 2,
            "Close": close_prices,
            "Volume": np.random.randint(1000000, 5000000, len(dates)),
        }
    )
    sample_data.set_index("Date", inplace=True)

    # 特徴量エンジニアリングのテスト
    feature_engineer = AdvancedFeatureEngineer()

    features = feature_engineer.generate_all_features(
        price_data=sample_data, volume_data=sample_data["Volume"]
    )

    print(f"生成された特徴量数: {len(features.columns)}")
    print(f"データ期間: {features.index[0]} - {features.index[-1]}")
    print(f"有効データ数: {len(features.dropna())}")

    # ターゲット変数の生成
    targets = create_target_variables(sample_data)
    print(f"ターゲット変数: {list(targets.keys())}")

    # 特徴量選択のテスト
    if len(features.dropna()) > 0:
        selected_features = feature_engineer.select_important_features(
            features, targets["future_returns"], top_k=20
        )
        print(f"選択された特徴量: {selected_features[:10]}...")
