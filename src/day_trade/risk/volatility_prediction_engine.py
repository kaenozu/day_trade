#!/usr/bin/env python3
"""
ボラティリティ予測エンジン
Issue #315: 高度テクニカル指標・ML機能拡張

GARCH・VIX風指標・機械学習を組み合わせたボラティリティ予測システム
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

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

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class VolatilityPredictionEngine:
    """
    ボラティリティ予測エンジン

    GARCH系モデル・VIX風指標・機械学習を統合した高精度ボラティリティ予測
    """

    def __init__(self, model_cache_dir: str = "data/volatility_models"):
        """
        初期化

        Args:
            model_cache_dir: モデルキャッシュディレクトリ
        """
        self.model_cache_dir = (
            Path(model_cache_dir) if hasattr(Path, "__name__") else None
        )

        # モデルキャッシュディレクトリ作成
        if self.model_cache_dir:
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

        logger.info("ボラティリティ予測エンジン初期化完了")

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
                returns = np.log(data["Close"] / data["Close"].shift(1))
                realized_vol = returns.rolling(window=window).std()

            elif method == "parkinson":
                # Parkinson推定量: 高値・安値を使用
                hl_ratio = np.log(data["High"] / data["Low"])
                realized_vol = hl_ratio.rolling(window=window).apply(
                    lambda x: np.sqrt(np.sum(x**2) / (4 * np.log(2) * len(x)))
                )

            elif method == "garman_klass":
                # Garman-Klass推定量: OHLC全てを使用
                ln_ho = np.log(data["High"] / data["Open"])
                ln_lo = np.log(data["Low"] / data["Open"])
                ln_co = np.log(data["Close"] / data["Open"])

                gk_component = (ln_ho * ln_lo) + (0.5 * ln_co**2)
                realized_vol = gk_component.rolling(window=window).apply(
                    lambda x: np.sqrt(np.sum(x) / len(x))
                )

            elif method == "yang_zhang":
                # Yang-Zhang推定量: 隙間を考慮した最も正確な推定量
                ln_co = np.log(data["Close"] / data["Open"])
                ln_ho = np.log(data["High"] / data["Open"])
                ln_lo = np.log(data["Low"] / data["Open"])
                ln_cc = np.log(data["Close"] / data["Close"].shift(1))
                ln_oc = np.log(data["Open"] / data["Close"].shift(1))

                # オーバーナイトボラティリティ
                overnight_vol = ln_oc.rolling(window=window).var()

                # 開場中ボラティリティ
                rs_vol = (
                    (ln_ho * (ln_ho - ln_co) + ln_lo * (ln_lo - ln_co))
                    .rolling(window=window)
                    .mean()
                )

                # クローズ-トゥ-クローズボラティリティ
                cc_vol = ln_cc.rolling(window=window).var()

                # Yang-Zhang組み合わせ
                k = 0.34 / (1.34 + (window + 1) / (window - 1))
                realized_vol = np.sqrt(overnight_vol + k * cc_vol + (1 - k) * rs_vol)

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

    def fit_garch_model(
        self,
        data: pd.DataFrame,
        model_type: str = "GARCH",
        p: int = 1,
        q: int = 1,
        symbol: str = "UNKNOWN",
    ) -> Optional[Dict]:
        """
        GARCHモデル適合

        Args:
            data: 価格データ
            model_type: モデルタイプ ('GARCH', 'GJR-GARCH', 'EGARCH')
            p: ARCH次数
            q: GARCH次数
            symbol: 銘柄コード

        Returns:
            適合結果辞書
        """
        if not ARCH_AVAILABLE:
            logger.error("arch パッケージが利用できません")
            return None

        try:
            # リターン計算
            returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()

            if len(returns) < 100:
                logger.warning(
                    f"GARCHモデルには最低100データポイントが必要: {len(returns)}"
                )
                return None

            # リターンをパーセンテージに変換（数値安定性のため）
            returns = returns * 100

            # モデル定義
            if model_type == "GARCH":
                model = arch_model(returns, vol="Garch", p=p, q=q, rescale=False)
            elif model_type == "GJR-GARCH":
                model = arch_model(returns, vol="GARCH", p=p, o=1, q=q, rescale=False)
            elif model_type == "EGARCH":
                model = arch_model(returns, vol="EGARCH", p=p, q=q, rescale=False)
            else:
                raise ValueError(f"サポートされていないモデルタイプ: {model_type}")

            # モデル適合
            logger.info(f"GARCHモデル適合開始: {model_type}({p},{q}) - {symbol}")
            fitted_model = model.fit(disp="off", show_warning=False)

            # 適合統計
            aic = fitted_model.aic
            bic = fitted_model.bic
            log_likelihood = fitted_model.loglikelihood

            # パラメータ抽出
            params = fitted_model.params

            # 1期先予測
            forecast = fitted_model.forecast(horizon=1, reindex=False)
            next_period_vol = (
                np.sqrt(forecast.variance.iloc[-1, 0]) / 100
            )  # パーセンテージから戻す

            # モデル保存
            self.garch_models[symbol] = fitted_model

            result = {
                "symbol": symbol,
                "model_type": model_type,
                "p": p,
                "q": q,
                "aic": float(aic),
                "bic": float(bic),
                "log_likelihood": float(log_likelihood),
                "parameters": {k: float(v) for k, v in params.items()},
                "next_period_volatility": float(next_period_vol),
                "fitted_volatility": fitted_model.conditional_volatility
                / 100,  # 時系列
                "data_points": len(returns),
                "fitting_timestamp": datetime.now().isoformat(),
            }

            logger.info(f"GARCHモデル適合完了: {model_type} AIC={aic:.2f}")
            return result

        except Exception as e:
            logger.error(f"GARCHモデル適合エラー ({symbol}): {e}")
            return None

    def predict_garch_volatility(
        self, symbol: str, horizon: int = 5, confidence_level: float = 0.95
    ) -> Optional[Dict]:
        """
        GARCHモデルによるボラティリティ予測

        Args:
            symbol: 銘柄コード
            horizon: 予測期間
            confidence_level: 信頼水準

        Returns:
            予測結果辞書
        """
        if not ARCH_AVAILABLE or symbol not in self.garch_models:
            logger.error(f"GARCHモデルが利用できません: {symbol}")
            return None

        try:
            fitted_model = self.garch_models[symbol]

            # 複数期間予測
            forecast = fitted_model.forecast(horizon=horizon, reindex=False)

            # 分散予測を標準偏差（ボラティリティ）に変換
            vol_forecast = np.sqrt(
                forecast.variance.iloc[-1] / 10000
            )  # パーセンテージ調整

            # 信頼区間計算
            alpha = 1 - confidence_level
            from scipy import stats

            z_score = stats.norm.ppf(1 - alpha / 2)

            # 予測の不確実性（簡易版）
            vol_std = (
                vol_forecast.std()
                if len(vol_forecast) > 1
                else vol_forecast.iloc[0] * 0.1
            )

            vol_upper = vol_forecast + z_score * vol_std
            vol_lower = np.maximum(
                vol_forecast - z_score * vol_std, 0.001
            )  # 負の値を避ける

            result = {
                "symbol": symbol,
                "forecast_horizon": horizon,
                "confidence_level": confidence_level,
                "volatility_forecast": [float(x) for x in vol_forecast],
                "volatility_upper": [float(x) for x in vol_upper],
                "volatility_lower": [float(x) for x in vol_lower],
                "forecast_dates": [
                    (datetime.now() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                    for i in range(horizon)
                ],
                "prediction_timestamp": datetime.now().isoformat(),
            }

            logger.info(f"GARCH予測完了: {symbol} {horizon}期間先")
            return result

        except Exception as e:
            logger.error(f"GARCH予測エラー ({symbol}): {e}")
            return None

    def calculate_vix_like_indicator(
        self, data: pd.DataFrame, window: int = 30, garch_params: Dict = None
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
                garch_params = self.vix_params

            returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()

            if len(returns) < window:
                logger.warning(
                    f"VIX計算にはwindow以上のデータが必要: {len(returns)} < {window}"
                )
                return pd.Series([20] * len(data), index=data.index)

            # 簡易GARCH(1,1)による条件付きボラティリティ
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

            # 前向き予測を追加（1期先）
            if len(conditional_vol) > 0:
                last_return_sq = returns.iloc[-1] ** 2
                last_vol_sq = conditional_vol.iloc[-1] ** 2
                next_variance = omega + alpha * last_return_sq + beta * last_vol_sq
                next_vol = np.sqrt(max(next_variance, 1e-6))
            else:
                next_vol = unconditional_vol

            # 年率化してVIXスケール（0-100）に変換
            annual_vol = conditional_vol * np.sqrt(252)
            vix_like = annual_vol * 100

            # 極端な値をクリッピング
            vix_like = np.clip(vix_like, 5, 150)

            # データフレーム全体のインデックスに拡張
            result = pd.Series(index=data.index, dtype=float)

            # 初期期間は移動平均で埋める
            if len(returns) >= window:
                initial_vol = returns.iloc[:window].std() * np.sqrt(252) * 100
                result.iloc[: len(result) - len(vix_like)] = initial_vol

            # GARCH結果を適用
            result.iloc[len(result) - len(vix_like) :] = vix_like.values

            # NaN値を前方埋め
            result = result.fillna(method="ffill").fillna(20)

            logger.info("VIX風指標計算完了")
            return result

        except Exception as e:
            logger.error(f"VIX風指標計算エラー: {e}")
            # フォールバック: 単純な移動ボラティリティ
            returns = data["Close"].pct_change()
            simple_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
            return simple_vol.fillna(20)

    def prepare_ml_features_for_volatility(
        self, data: pd.DataFrame, lookback_days: int = 60
    ) -> pd.DataFrame:
        """
        ボラティリティ予測用機械学習特徴量準備

        Args:
            data: OHLCV価格データ
            lookback_days: 特徴量作成に使用する過去データ日数

        Returns:
            特徴量DataFrame
        """
        try:
            features = pd.DataFrame(index=data.index)

            # 基本価格特徴量
            returns = np.log(data["Close"] / data["Close"].shift(1))
            features["returns"] = returns
            features["abs_returns"] = np.abs(returns)
            features["squared_returns"] = returns**2

            # 複数期間の実現ボラティリティ
            for period in [5, 10, 20, 30]:
                features[f"realized_vol_{period}"] = self.calculate_realized_volatility(
                    data, window=period, annualize=True
                )

            # 価格レンジ指標
            features["high_low_ratio"] = (data["High"] - data["Low"]) / data["Close"]
            features["open_close_ratio"] = (
                np.abs(data["Open"] - data["Close"]) / data["Close"]
            )

            # 出来高関連特徴量
            features["volume"] = data["Volume"]
            features["volume_ma"] = data["Volume"].rolling(20).mean()
            features["volume_ratio"] = data["Volume"] / features["volume_ma"]
            features["price_volume"] = data["Close"] * data["Volume"]

            # ボラティリティクラスタリング指標
            for lag in [1, 2, 3, 5]:
                features[f"vol_lag_{lag}"] = features["realized_vol_20"].shift(lag)
                features[f"return_lag_{lag}"] = returns.shift(lag)

            # 移動統計量
            for window in [10, 20, 50]:
                features[f"return_mean_{window}"] = returns.rolling(window).mean()
                features[f"return_std_{window}"] = returns.rolling(window).std()
                features[f"return_skew_{window}"] = returns.rolling(window).skew()
                features[f"return_kurt_{window}"] = returns.rolling(window).kurt()

            # エクストリームリターン指標
            features["extreme_return_5d"] = (
                np.abs(returns.rolling(5).sum()) > returns.rolling(60).std() * 2
            ).astype(int)

            # VIX風指標
            features["vix_like"] = self.calculate_vix_like_indicator(data)
            features["vix_change"] = features["vix_like"].diff()
            features["vix_relative"] = (
                features["vix_like"] / features["vix_like"].rolling(30).mean()
            )

            # 技術的指標
            # RSI
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features["rsi"] = 100 - (100 / (1 + rs))

            # ボリンジャーバンド
            sma_20 = data["Close"].rolling(20).mean()
            std_20 = data["Close"].rolling(20).std()
            features["bb_position"] = (data["Close"] - (sma_20 - 2 * std_20)) / (
                4 * std_20
            )
            features["bb_width"] = (4 * std_20) / sma_20

            # 時間系特徴量（曜日効果・月効果）
            features["day_of_week"] = data.index.dayofweek
            features["day_of_month"] = data.index.day
            features["month"] = data.index.month

            # 季節調整項
            features["seasonal_factor"] = np.sin(
                2 * np.pi * data.index.dayofyear / 365.25
            )

            # ラグ特徴量の追加（時系列の記憶効果）
            for col in ["realized_vol_20", "vix_like", "high_low_ratio"]:
                if col in features.columns:
                    for lag in range(1, 6):
                        features[f"{col}_lag_{lag}"] = features[col].shift(lag)

            # NaN値処理
            features = features.fillna(method="ffill").fillna(method="bfill").fillna(0)

            logger.info(f"ML特徴量準備完了: {len(features.columns)}特徴量")
            return features

        except Exception as e:
            logger.error(f"ML特徴量準備エラー: {e}")
            return pd.DataFrame(index=data.index)

    def train_volatility_ml_model(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        target_horizon: int = 5,
        model_types: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """
        ボラティリティ予測機械学習モデル訓練

        Args:
            data: 価格データ
            symbol: 銘柄コード
            target_horizon: 予測ホライゾン（日数）
            model_types: 使用するモデルタイプリスト

        Returns:
            訓練結果辞書
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn が利用できません")
            return None

        if model_types is None:
            model_types = ["random_forest", "gradient_boosting"]

        try:
            # 特徴量準備
            features = self.prepare_ml_features_for_volatility(data)

            if features.empty:
                logger.error("特徴量準備に失敗")
                return None

            # ターゲット変数: target_horizon日先の実現ボラティリティ
            target_vol = self.calculate_realized_volatility(
                data, window=5, annualize=True
            )
            target = target_vol.shift(-target_horizon)

            # 有効データのみ抽出
            valid_mask = ~(features.isnull().any(axis=1) | target.isnull())
            X = features[valid_mask]
            y = target[valid_mask]

            if len(X) < 100:
                logger.warning(f"訓練データ不足: {len(X)}行 (最低100行必要)")
                return None

            # 訓練・テストデータ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, shuffle=False
            )

            # 特徴量正規化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {}
            results = {}

            # モデル定義と訓練
            if "random_forest" in model_types:
                rf_model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                )
                rf_model.fit(X_train, y_train)
                models["random_forest"] = rf_model

                # 評価
                rf_pred = rf_model.predict(X_test)
                rf_mse = mean_squared_error(y_test, rf_pred)
                rf_r2 = r2_score(y_test, rf_pred)

                results["random_forest"] = {
                    "mse": float(rf_mse),
                    "rmse": float(np.sqrt(rf_mse)),
                    "r2": float(rf_r2),
                    "feature_importance": rf_model.feature_importances_.tolist(),
                }

            if "gradient_boosting" in model_types:
                gb_model = GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    subsample=0.8,
                    random_state=42,
                )
                gb_model.fit(X_train, y_train)
                models["gradient_boosting"] = gb_model

                # 評価
                gb_pred = gb_model.predict(X_test)
                gb_mse = mean_squared_error(y_test, gb_pred)
                gb_r2 = r2_score(y_test, gb_pred)

                results["gradient_boosting"] = {
                    "mse": float(gb_mse),
                    "rmse": float(np.sqrt(gb_mse)),
                    "r2": float(gb_r2),
                    "feature_importance": gb_model.feature_importances_.tolist(),
                }

            # 最良モデル選択
            best_model_name = min(results.keys(), key=lambda x: results[x]["mse"])
            best_model = models[best_model_name]

            # モデルとスケーラーを保存
            self.ml_models[symbol] = {
                "models": models,
                "scaler": scaler,
                "feature_names": X.columns.tolist(),
                "best_model_name": best_model_name,
                "target_horizon": target_horizon,
            }

            training_result = {
                "symbol": symbol,
                "target_horizon": target_horizon,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(X.columns),
                "models": results,
                "best_model": best_model_name,
                "best_performance": results[best_model_name],
                "feature_names": X.columns.tolist(),
                "training_timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"ML訓練完了: {symbol} - 最良モデル: {best_model_name} (R²={results[best_model_name]['r2']:.3f})"
            )
            return training_result

        except Exception as e:
            logger.error(f"ML訓練エラー ({symbol}): {e}")
            return None

    def predict_volatility_ml(
        self, data: pd.DataFrame, symbol: str, horizon: int = 5
    ) -> Optional[Dict]:
        """
        機械学習によるボラティリティ予測

        Args:
            data: 価格データ
            symbol: 銘柄コード
            horizon: 予測ホライゾン

        Returns:
            予測結果辞書
        """
        if not SKLEARN_AVAILABLE or symbol not in self.ml_models:
            logger.error(f"MLモデルが利用できません: {symbol}")
            return None

        try:
            model_info = self.ml_models[symbol]
            models = model_info["models"]
            scaler = model_info["scaler"]
            feature_names = model_info["feature_names"]
            best_model_name = model_info["best_model_name"]

            # 特徴量準備
            features = self.prepare_ml_features_for_volatility(data)

            if features.empty or len(features) == 0:
                logger.error("予測用特徴量準備に失敗")
                return None

            # 最新データポイントを使用
            latest_features = features.iloc[-1:][feature_names]

            if latest_features.isnull().any().any():
                logger.warning("最新特徴量にNaN値を含む")
                latest_features = latest_features.fillna(0)

            # 特徴量正規化
            latest_scaled = scaler.transform(latest_features)

            # 各モデルで予測
            predictions = {}

            for model_name, model in models.items():
                pred = model.predict(latest_scaled)[0]
                predictions[model_name] = float(max(pred, 0.001))  # 負の値を避ける

            # アンサンブル予測（重み付き平均）
            best_pred = predictions[best_model_name]

            # 信頼区間の簡易計算（予測の不確実性）
            prediction_std = np.std(list(predictions.values()))
            confidence_interval = {
                "lower": max(best_pred - 1.96 * prediction_std, 0.001),
                "upper": best_pred + 1.96 * prediction_std,
            }

            # 現在の実現ボラティリティとの比較
            current_vol = self.calculate_realized_volatility(
                data, window=20, annualize=True
            ).iloc[-1]

            result = {
                "symbol": symbol,
                "prediction_horizon": horizon,
                "predicted_volatility": float(best_pred),
                "current_volatility": (
                    float(current_vol) if pd.notna(current_vol) else None
                ),
                "volatility_change": (
                    float(best_pred - current_vol) if pd.notna(current_vol) else None
                ),
                "confidence_interval": confidence_interval,
                "model_predictions": predictions,
                "best_model": best_model_name,
                "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                "prediction_timestamp": datetime.now().isoformat(),
            }

            logger.info(f"ML予測完了: {symbol} - 予測ボラティリティ: {best_pred:.1f}%")
            return result

        except Exception as e:
            logger.error(f"ML予測エラー ({symbol}): {e}")
            return None

    def create_volatility_regime_classifier(
        self, data: pd.DataFrame, thresholds: Optional[Dict[str, float]] = None
    ):
        """
        ボラティリティレジーム分類

        Args:
            data: 価格データ
            thresholds: 分類閾値

        Returns:
            レジーム分類結果
        """
        try:
            if thresholds is None:
                thresholds = {
                    "low": 0.15,  # 15%未満
                    "medium": 0.30,  # 15-30%
                    "high": 0.50,  # 30-50%
                    "extreme": float("inf"),  # 50%以上
                }

            # 実現ボラティリティ計算
            realized_vol = self.calculate_realized_volatility(
                data, window=20, annualize=True
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

    def generate_comprehensive_volatility_forecast(
        self, data: pd.DataFrame, symbol: str = "UNKNOWN", forecast_horizon: int = 10
    ) -> Dict[str, any]:
        """
        総合ボラティリティ予測

        GARCH・機械学習・VIX風指標を統合した総合予測

        Args:
            data: 価格データ
            symbol: 銘柄コード
            forecast_horizon: 予測期間

        Returns:
            総合予測結果辞書
        """
        try:
            comprehensive_forecast = {
                "symbol": symbol,
                "forecast_horizon": forecast_horizon,
                "forecast_timestamp": datetime.now().isoformat(),
                "models": {},
                "ensemble_forecast": {},
                "risk_assessment": {},
            }

            # GARCH予測
            garch_result = None
            if len(data) >= 100:  # GARCH用最小データ
                garch_fit = self.fit_garch_model(data, symbol=symbol)
                if garch_fit:
                    garch_result = self.predict_garch_volatility(
                        symbol, horizon=forecast_horizon
                    )
                    if garch_result:
                        comprehensive_forecast["models"]["garch"] = garch_result
                        logger.info("GARCH予測を統合予測に追加")

            # ML予測
            ml_result = None
            if len(data) >= 150:  # ML用最小データ
                ml_train = self.train_volatility_ml_model(
                    data, symbol=symbol, target_horizon=5
                )
                if ml_train:
                    ml_result = self.predict_volatility_ml(data, symbol, horizon=5)
                    if ml_result:
                        comprehensive_forecast["models"]["machine_learning"] = ml_result
                        logger.info("ML予測を統合予測に追加")

            # VIX風指標
            vix_like = self.calculate_vix_like_indicator(data)
            current_vix = vix_like.iloc[-1] if len(vix_like) > 0 else 20
            vix_forecast = self._project_vix_forward(vix_like, forecast_horizon)

            comprehensive_forecast["models"]["vix_like"] = {
                "current_vix": float(current_vix),
                "vix_forecast": vix_forecast,
                "vix_regime": self._classify_vix_regime(current_vix),
            }

            # 実現ボラティリティ
            realized_vol = self.calculate_realized_volatility(
                data, window=20, annualize=True
            )
            current_realized = realized_vol.iloc[-1] if len(realized_vol) > 0 else 0.2

            comprehensive_forecast["current_metrics"] = {
                "realized_volatility": float(current_realized),
                "vix_like_indicator": float(current_vix),
                "volatility_regime": self.create_volatility_regime_classifier(
                    data
                ).iloc[-1],
            }

            # アンサンブル予測
            ensemble = self._create_volatility_ensemble(
                garch_result, ml_result, vix_forecast, current_realized
            )
            comprehensive_forecast["ensemble_forecast"] = ensemble

            # リスク評価
            risk_assessment = self._assess_volatility_risk(
                current_realized, current_vix, ensemble
            )
            comprehensive_forecast["risk_assessment"] = risk_assessment

            # 投資への示唆
            implications = self._generate_volatility_implications(
                ensemble, risk_assessment, comprehensive_forecast["current_metrics"]
            )
            comprehensive_forecast["investment_implications"] = implications

            logger.info(f"総合ボラティリティ予測完了: {symbol}")
            return comprehensive_forecast

        except Exception as e:
            logger.error(f"総合予測エラー ({symbol}): {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "forecast_timestamp": datetime.now().isoformat(),
            }

    def _project_vix_forward(self, vix_series: pd.Series, horizon: int) -> List[float]:
        """VIX風指標の前方投影"""
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
                vix_series.rolling(60).mean().iloc[-1] if len(vix_series) >= 60 else 20
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

    def _classify_vix_regime(self, vix_value: float) -> str:
        """VIX値によるレジーム分類"""
        if vix_value < 15:
            return "complacency"
        elif vix_value < 25:
            return "normal"
        elif vix_value < 35:
            return "concern"
        elif vix_value < 50:
            return "fear"
        else:
            return "panic"

    def _create_volatility_ensemble(
        self,
        garch_result: Optional[Dict],
        ml_result: Optional[Dict],
        vix_forecast: List[float],
        current_realized: float,
    ) -> Dict[str, any]:
        """ボラティリティアンサンブル予測作成"""
        try:
            forecasts = []
            weights = []
            models = []

            # GARCH予測
            if garch_result and "volatility_forecast" in garch_result:
                garch_vol = np.mean(
                    garch_result["volatility_forecast"][:5]
                )  # 最初の5日平均
                forecasts.append(garch_vol * 100)  # パーセンテージ変換
                weights.append(0.35)
                models.append("GARCH")

            # ML予測
            if ml_result and "predicted_volatility" in ml_result:
                forecasts.append(ml_result["predicted_volatility"])
                weights.append(0.40)
                models.append("Machine Learning")

            # VIX風予測
            if vix_forecast:
                vix_avg = np.mean(vix_forecast[:5])  # 最初の5日平均
                forecasts.append(vix_avg)
                weights.append(0.25)
                models.append("VIX-like")

            if not forecasts:
                # フォールバック: 現在の実現ボラティリティ
                ensemble_vol = current_realized * 100
                confidence = 0.3
            else:
                # 重み付き平均
                weights = np.array(weights) / np.sum(weights)  # 正規化
                ensemble_vol = np.average(forecasts, weights=weights)

                # 信頼度計算（予測の一致度）
                forecast_std = np.std(forecasts) if len(forecasts) > 1 else 0
                confidence = (
                    max(0.2, 1 - (forecast_std / ensemble_vol))
                    if ensemble_vol > 0
                    else 0.5
                )

            return {
                "ensemble_volatility": float(ensemble_vol),
                "individual_forecasts": dict(zip(models, forecasts)),
                "model_weights": dict(zip(models, weights)) if weights else {},
                "ensemble_confidence": float(confidence),
                "forecast_range": {
                    "min": float(min(forecasts)) if forecasts else ensemble_vol,
                    "max": float(max(forecasts)) if forecasts else ensemble_vol,
                },
            }

        except Exception as e:
            logger.error(f"アンサンブル作成エラー: {e}")
            return {
                "ensemble_volatility": current_realized * 100,
                "ensemble_confidence": 0.3,
                "error": str(e),
            }

    def _assess_volatility_risk(
        self, current_vol: float, current_vix: float, ensemble: Dict
    ) -> Dict[str, any]:
        """ボラティリティリスク評価"""
        try:
            risk_score = 0  # 0-100, 高いほど高リスク
            risk_factors = []

            # 現在のボラティリティレベル
            annual_vol_pct = current_vol * 100
            if annual_vol_pct > 40:
                risk_score += 30
                risk_factors.append("極端に高いボラティリティ")
            elif annual_vol_pct > 25:
                risk_score += 15
                risk_factors.append("高ボラティリティ環境")
            elif annual_vol_pct < 10:
                risk_score += 10
                risk_factors.append("異常に低いボラティリティ(反発リスク)")

            # VIXレベル評価
            if current_vix > 40:
                risk_score += 25
                risk_factors.append("パニック水準のVIX")
            elif current_vix > 25:
                risk_score += 10
                risk_factors.append("不安水準のVIX")
            elif current_vix < 15:
                risk_score += 15
                risk_factors.append("自己満足水準のVIX(反発リスク)")

            # アンサンブル予測の不確実性
            ensemble_vol = ensemble.get("ensemble_volatility", annual_vol_pct)
            forecast_range = ensemble.get("forecast_range", {})

            if (
                forecast_range.get("max", ensemble_vol)
                - forecast_range.get("min", ensemble_vol)
                > 10
            ):
                risk_score += 20
                risk_factors.append("モデル間の予測分散が大きい")

            # ボラティリティの急上昇リスク
            vol_change = ensemble_vol - annual_vol_pct
            if vol_change > 10:
                risk_score += 20
                risk_factors.append("ボラティリティ急上昇予測")

            # リスクレベル分類
            if risk_score >= 60:
                risk_level = "HIGH"
            elif risk_score >= 35:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            return {
                "risk_level": risk_level,
                "risk_score": int(risk_score),
                "risk_factors": risk_factors,
                "volatility_outlook": (
                    "increasing"
                    if vol_change > 2
                    else "decreasing" if vol_change < -2 else "stable"
                ),
            }

        except Exception as e:
            logger.error(f"ボラティリティリスク評価エラー: {e}")
            return {"risk_level": "UNKNOWN", "risk_score": 50, "error": str(e)}

    def _generate_volatility_implications(
        self, ensemble: Dict, risk_assessment: Dict, current_metrics: Dict
    ) -> Dict[str, any]:
        """ボラティリティの投資への示唆"""
        try:
            implications = {
                "portfolio_adjustments": [],
                "trading_strategies": [],
                "risk_management": [],
                "market_timing": [],
            }

            ensemble_vol = ensemble.get("ensemble_volatility", 20)
            risk_level = risk_assessment.get("risk_level", "MEDIUM")
            outlook = risk_assessment.get("volatility_outlook", "stable")

            # ポートフォリオ調整
            if ensemble_vol > 30:
                implications["portfolio_adjustments"].extend(
                    [
                        "ポジションサイズの縮小を検討",
                        "防御的資産の比重増加",
                        "分散投資の強化",
                    ]
                )
            elif ensemble_vol < 15:
                implications["portfolio_adjustments"].extend(
                    [
                        "リスク資産の比重増加を検討",
                        "レバレッジ活用の機会",
                        "集中投資の機会",
                    ]
                )

            # トレーディング戦略
            if outlook == "increasing":
                implications["trading_strategies"].extend(
                    [
                        "ボラティリティ・ブレイクアウト戦略",
                        "ストラドル・ストラングル戦略(オプション)",
                        "短期トレーディングの機会増加",
                    ]
                )
            elif outlook == "decreasing":
                implications["trading_strategies"].extend(
                    [
                        "平均回帰戦略",
                        "レンジトレーディング",
                        "バイ・アンド・ホールド戦略",
                    ]
                )

            # リスク管理
            if risk_level == "HIGH":
                implications["risk_management"].extend(
                    [
                        "ストップロス幅の拡大",
                        "より頻繁なポジション見直し",
                        "ヘッジ比率の増加",
                    ]
                )
            elif risk_level == "LOW":
                implications["risk_management"].extend(
                    [
                        "ストップロス幅の縮小可能",
                        "ポジション保有期間の延長",
                        "コスト効率の重視",
                    ]
                )

            # マーケットタイミング
            current_vix = current_metrics.get("vix_like_indicator", 20)
            if current_vix > 30 and outlook == "decreasing":
                implications["market_timing"].append("逆張り投資の機会")
            elif current_vix < 20 and outlook == "increasing":
                implications["market_timing"].append("リスクオフの準備")

            return implications

        except Exception as e:
            logger.error(f"示唆生成エラー: {e}")
            return {
                "portfolio_adjustments": ["ボラティリティ情報を基に慎重な判断を"],
                "error": str(e),
            }


# パスインポート修正
from pathlib import Path

if __name__ == "__main__":
    # テスト実行
    print("=== ボラティリティ予測エンジン テスト ===")

    # サンプルデータ生成（より現実的なボラティリティクラスタリング）
    dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")
    np.random.seed(42)

    # ボラティリティレジームのシミュレーション
    base_price = 3000
    prices = [base_price]

    # ボラティリティレジーム（0: 低ボラ, 1: 中ボラ, 2: 高ボラ）
    vol_regimes = []
    regime_length = 50  # 各レジームの平均継続期間

    current_regime = 0
    regime_countdown = regime_length

    vol_levels = [0.01, 0.02, 0.04]  # 低・中・高ボラティリティ

    for i in range(len(dates)):
        # レジーム変更判定
        if regime_countdown <= 0:
            # 次のレジームをランダム選択（平均回帰的に）
            transition_probs = [
                [0.7, 0.25, 0.05],  # 低ボラ -> [低, 中, 高]
                [0.2, 0.6, 0.2],  # 中ボラ -> [低, 中, 高]
                [0.1, 0.3, 0.6],
            ]  # 高ボラ -> [低, 中, 高]

            current_regime = np.random.choice(3, p=transition_probs[current_regime])
            regime_countdown = np.random.poisson(regime_length)

        vol_regimes.append(current_regime)
        regime_countdown -= 1

        # 価格変動生成
        vol = vol_levels[current_regime]
        # ボラティリティクラスタリング効果
        if i > 0 and vol_regimes[i - 1] == current_regime:
            vol *= 1.1  # 同じレジーム継続時は少し高く

        price_change = np.random.normal(0.0005, vol)
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(new_price, 1000))  # 価格下限

    # OHLCV生成
    sample_data = pd.DataFrame(index=dates)
    sample_data["Close"] = prices[1:]  # 長さ調整
    sample_data["Open"] = [
        p * np.random.uniform(0.998, 1.002) for p in sample_data["Close"]
    ]
    sample_data["High"] = [
        max(o, c) * np.random.uniform(1.000, 1.020)
        for o, c in zip(sample_data["Open"], sample_data["Close"])
    ]
    sample_data["Low"] = [
        min(o, c) * np.random.uniform(0.980, 1.000)
        for o, c in zip(sample_data["Open"], sample_data["Close"])
    ]
    sample_data["Volume"] = np.random.randint(500000, 15000000, len(dates))

    try:
        engine = VolatilityPredictionEngine()

        print(f"サンプルデータ: {len(sample_data)}日分")
        print(
            f"価格範囲: {sample_data['Close'].min():.2f} - {sample_data['Close'].max():.2f}"
        )

        # 1. 実現ボラティリティ計算テスト
        print("\n1. 実現ボラティリティ計算テスト")
        methods = ["close_to_close", "parkinson", "garman_klass"]

        for method in methods:
            real_vol = engine.calculate_realized_volatility(sample_data, method=method)
            current_vol = real_vol.iloc[-1] if len(real_vol) > 0 else 0
            avg_vol = real_vol.mean() if len(real_vol) > 0 else 0
            print(f"✅ {method}: 現在={current_vol:.1f}%, 平均={avg_vol:.1f}%")

        # 2. VIX風指標テスト
        print("\n2. VIX風指標計算テスト")
        vix_like = engine.calculate_vix_like_indicator(sample_data)
        current_vix = vix_like.iloc[-1]
        max_vix = vix_like.max()
        min_vix = vix_like.min()
        print("✅ VIX風指標計算完了")
        print(f"   現在値: {current_vix:.1f}")
        print(f"   範囲: {min_vix:.1f} - {max_vix:.1f}")
        print(f"   レジーム: {engine._classify_vix_regime(current_vix)}")

        # 3. ボラティリティレジーム分類テスト
        print("\n3. ボラティリティレジーム分類テスト")
        regimes = engine.create_volatility_regime_classifier(sample_data)
        regime_counts = regimes.value_counts()
        print("✅ レジーム分類完了:")
        for regime, count in regime_counts.items():
            print(f"   {regime}: {count}日 ({count / len(regimes) * 100:.1f}%)")

        # 4. 機械学習特徴量準備テスト
        print("\n4. ML特徴量準備テスト")
        ml_features = engine.prepare_ml_features_for_volatility(sample_data)
        print(f"✅ ML特徴量準備完了: {len(ml_features.columns)}特徴量")
        print(f"   データポイント: {len(ml_features)}")

        # 主要特徴量の統計
        key_features = ["realized_vol_20", "vix_like", "high_low_ratio", "volume_ratio"]
        for feature in key_features:
            if feature in ml_features.columns:
                value = ml_features[feature].iloc[-1]
                mean_val = ml_features[feature].mean()
                print(f"   {feature}: 現在={value:.3f}, 平均={mean_val:.3f}")

        # 5. GARCH モデルテスト（データが十分な場合）
        if ARCH_AVAILABLE and len(sample_data) >= 100:
            print("\n5. GARCHモデルテスト")
            garch_result = engine.fit_garch_model(sample_data, symbol="TEST_STOCK")

            if garch_result:
                print("✅ GARCHモデル適合完了")
                print(f"   AIC: {garch_result['aic']:.2f}")
                print(
                    f"   次期予測ボラティリティ: {garch_result['next_period_volatility']:.3f}"
                )

                # GARCH予測テスト
                garch_forecast = engine.predict_garch_volatility(
                    "TEST_STOCK", horizon=5
                )
                if garch_forecast:
                    print("✅ GARCH予測完了")
                    print(f"   5日先予測: {garch_forecast['volatility_forecast']}")
            else:
                print("❌ GARCHモデル適合失敗")
        else:
            print("\n5. GARCHモデル - スキップ（arch未インストールまたはデータ不足）")

        # 6. 機械学習モデルテスト（データが十分な場合）
        if SKLEARN_AVAILABLE and len(sample_data) >= 150:
            print("\n6. 機械学習モデルテスト")
            ml_train_result = engine.train_volatility_ml_model(
                sample_data, symbol="TEST_STOCK"
            )

            if ml_train_result:
                print("✅ ML訓練完了")
                print(f"   最良モデル: {ml_train_result['best_model']}")
                best_perf = ml_train_result["best_performance"]
                print(f"   R²スコア: {best_perf['r2']:.3f}")
                print(f"   RMSE: {best_perf['rmse']:.3f}")

                # ML予測テスト
                ml_forecast = engine.predict_volatility_ml(sample_data, "TEST_STOCK")
                if ml_forecast:
                    print("✅ ML予測完了")
                    print(
                        f"   予測ボラティリティ: {ml_forecast['predicted_volatility']:.1f}%"
                    )
                    print(f"   信頼区間: {ml_forecast['confidence_interval']}")
            else:
                print("❌ ML訓練失敗")
        else:
            print(
                "\n6. 機械学習モデル - スキップ（sklearn未インストールまたはデータ不足）"
            )

        # 7. 総合予測テスト
        print("\n7. 総合ボラティリティ予測テスト")
        comprehensive = engine.generate_comprehensive_volatility_forecast(
            sample_data, symbol="TEST_STOCK", forecast_horizon=10
        )

        if "error" not in comprehensive:
            print("✅ 総合予測完了")

            # 現在のメトリクス
            current = comprehensive.get("current_metrics", {})
            print("\n📊 現在の状況:")
            print(
                f"   実現ボラティリティ: {current.get('realized_volatility', 0):.1f}%"
            )
            print(f"   VIX風指標: {current.get('vix_like_indicator', 0):.1f}")
            print(
                f"   ボラティリティレジーム: {current.get('volatility_regime', 'unknown')}"
            )

            # アンサンブル予測
            ensemble = comprehensive.get("ensemble_forecast", {})
            if ensemble:
                print("\n🔮 アンサンブル予測:")
                print(
                    f"   予測ボラティリティ: {ensemble.get('ensemble_volatility', 0):.1f}%"
                )
                print(f"   予測信頼度: {ensemble.get('ensemble_confidence', 0):.1f}")

                individual = ensemble.get("individual_forecasts", {})
                if individual:
                    print("   個別モデル予測:")
                    for model, pred in individual.items():
                        print(f"     {model}: {pred:.1f}%")

            # リスク評価
            risk = comprehensive.get("risk_assessment", {})
            if risk:
                print("\n⚠️  リスク評価:")
                print(f"   リスクレベル: {risk.get('risk_level', 'UNKNOWN')}")
                print(f"   リスクスコア: {risk.get('risk_score', 0)}")
                print(
                    f"   ボラティリティ見通し: {risk.get('volatility_outlook', 'unknown')}"
                )

                risk_factors = risk.get("risk_factors", [])
                if risk_factors:
                    print("   リスク要因:")
                    for factor in risk_factors[:3]:
                        print(f"     - {factor}")

            # 投資への示唆
            implications = comprehensive.get("investment_implications", {})
            if implications:
                print("\n💡 投資への示唆:")
                for category, suggestions in implications.items():
                    if suggestions and category != "error":
                        print(f"   {category}:")
                        for suggestion in suggestions[:2]:  # 上位2個表示
                            print(f"     - {suggestion}")
        else:
            print(f"❌ 総合予測エラー: {comprehensive['error']}")

        print("\n✅ ボラティリティ予測エンジン テスト完了！")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
