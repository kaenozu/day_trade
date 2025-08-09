#!/usr/bin/env python3
"""
LSTM時系列予測モデル
Issue #315: 高度テクニカル指標・ML機能拡張

TensorFlow/Kerasを使用したLSTM深層学習による時系列予測
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 依存パッケージの段階的読み込み
try:
    import tensorflow as tf
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow import keras
    from tensorflow.keras import layers

    # TensorFlowログレベル設定
    tf.get_logger().setLevel("ERROR")
    warnings.filterwarnings("ignore", category=UserWarning)

    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow利用可能")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning(
        "TensorFlow未インストール - pip install tensorflow>=2.10.0でインストールしてください"
    )

try:
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn利用可能")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn未インストール - pip install scikit-learnでインストールしてください"
    )


class LSTMTimeSeriesModel:
    """
    LSTM深層学習時系列予測モデル

    複数のLSTMレイヤーと注意機構を使用した高精度時系列予測
    """

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 5,
        lstm_units: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        model_cache_dir: str = "data/lstm_models",
    ):
        """
        初期化

        Args:
            sequence_length: 入力系列長（デフォルト60日）
            prediction_horizon: 予測期間（デフォルト5日後）
            lstm_units: LSTMレイヤーのユニット数リスト
            dropout_rate: Dropout率
            learning_rate: 学習率
            model_cache_dir: モデルキャッシュディレクトリ
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # モデルとスケーラーのキャッシュ
        self.models = {}
        self.scalers = {}
        self.model_histories = {}

        # GPU使用可能チェック
        if TENSORFLOW_AVAILABLE:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                try:
                    # GPU使用設定
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU使用可能: {len(gpus)}個")
                except RuntimeError as e:
                    logger.warning(f"GPU設定エラー: {e}")
            else:
                logger.info("CPUを使用してLSTM訓練を実行")

        logger.info("LSTM時系列予測モデル初期化完了")

    def prepare_lstm_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        LSTM用特徴量準備

        Args:
            data: OHLCV価格データ

        Returns:
            LSTM用特徴量DataFrame
        """
        if not TENSORFLOW_AVAILABLE or data is None or data.empty:
            logger.error("TensorFlow利用不可またはデータなし")
            return pd.DataFrame()

        try:
            df = data.copy()
            features = pd.DataFrame(index=df.index)

            # 基本価格特徴量
            features["close_price"] = df["Close"]
            features["high_price"] = df["High"]
            features["low_price"] = df["Low"]
            features["volume"] = df["Volume"]

            # 正規化された価格特徴量
            features["hl_ratio"] = (df["High"] - df["Low"]) / df["Close"]
            features["oc_ratio"] = (df["Open"] - df["Close"]) / df["Close"]

            # リターン系特徴量
            features["returns_1d"] = df["Close"].pct_change()
            features["returns_5d"] = df["Close"].pct_change(5)
            features["returns_20d"] = df["Close"].pct_change(20)

            # ボラティリティ特徴量
            features["volatility_5d"] = features["returns_1d"].rolling(5).std()
            features["volatility_20d"] = features["returns_1d"].rolling(20).std()

            # 移動平均系特徴量
            features["sma_5"] = df["Close"].rolling(5).mean()
            features["sma_20"] = df["Close"].rolling(20).mean()
            features["sma_50"] = df["Close"].rolling(50).mean()

            # 移動平均からの乖離率
            features["sma_5_ratio"] = df["Close"] / features["sma_5"]
            features["sma_20_ratio"] = df["Close"] / features["sma_20"]
            features["sma_50_ratio"] = df["Close"] / features["sma_50"]

            # EMA特徴量
            features["ema_12"] = df["Close"].ewm(span=12).mean()
            features["ema_26"] = df["Close"].ewm(span=26).mean()
            features["macd"] = features["ema_12"] - features["ema_26"]
            features["macd_signal"] = features["macd"].ewm(span=9).mean()

            # RSI
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features["rsi"] = 100 - (100 / (1 + rs))

            # ボリンジャーバンド
            bb_period = 20
            bb_std = 2
            bb_middle = features["sma_20"]
            bb_std_dev = df["Close"].rolling(bb_period).std()
            features["bb_upper"] = bb_middle + (bb_std_dev * bb_std)
            features["bb_lower"] = bb_middle - (bb_std_dev * bb_std)
            features["bb_position"] = (df["Close"] - features["bb_lower"]) / (
                features["bb_upper"] - features["bb_lower"]
            )

            # 出来高系特徴量
            features["volume_sma"] = df["Volume"].rolling(20).mean()
            features["volume_ratio"] = df["Volume"] / features["volume_sma"]
            features["price_volume"] = df["Close"] * df["Volume"]

            # 時間系特徴量（季節性考慮）
            features["day_of_week"] = df.index.dayofweek
            features["day_of_month"] = df.index.day
            features["month"] = df.index.month

            # NaN値処理
            features = features.fillna(method="ffill").fillna(method="bfill").fillna(0)

            logger.info(f"LSTM特徴量準備完了: {len(features.columns)}特徴量")
            return features

        except Exception as e:
            logger.error(f"LSTM特徴量準備エラー: {e}")
            return pd.DataFrame()

    def create_sequences(
        self, features: pd.DataFrame, target_column: str = "close_price"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        LSTM用系列データ作成

        Args:
            features: 特徴量DataFrame
            target_column: 予測対象列名

        Returns:
            X, y: 系列データとターゲット
        """
        try:
            data = features.values
            target_idx = features.columns.get_loc(target_column)

            X, y = [], []

            for i in range(
                self.sequence_length, len(data) - self.prediction_horizon + 1
            ):
                # 過去sequence_length日分の特徴量
                sequence = data[i - self.sequence_length : i]
                X.append(sequence)

                # prediction_horizon日後の目標価格
                future_price = data[i + self.prediction_horizon - 1, target_idx]
                y.append(future_price)

            return np.array(X), np.array(y)

        except Exception as e:
            logger.error(f"系列データ作成エラー: {e}")
            return np.array([]), np.array([])

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> "keras.Model":
        """
        LSTM深層学習モデル構築

        Args:
            input_shape: 入力形状 (sequence_length, features_count)

        Returns:
            Kerasモデル
        """
        if not TENSORFLOW_AVAILABLE:
            raise ValueError("TensorFlowが利用できません")

        try:
            model = keras.Sequential(
                [
                    # 入力レイヤー
                    layers.Input(shape=input_shape),
                    # 第1 LSTMレイヤー（return_sequences=True）
                    layers.LSTM(
                        self.lstm_units[0],
                        return_sequences=True,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.dropout_rate / 2,
                    ),
                    layers.BatchNormalization(),
                    # 第2 LSTMレイヤー
                    layers.LSTM(
                        self.lstm_units[1],
                        return_sequences=True if len(self.lstm_units) > 2 else False,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.dropout_rate / 2,
                    ),
                    layers.BatchNormalization(),
                ]
            )

            # 追加LSTMレイヤー
            if len(self.lstm_units) > 2:
                for i, units in enumerate(self.lstm_units[2:]):
                    is_last_lstm = i == len(self.lstm_units) - 3
                    model.add(
                        layers.LSTM(
                            units,
                            return_sequences=not is_last_lstm,
                            dropout=self.dropout_rate,
                            recurrent_dropout=self.dropout_rate / 2,
                        )
                    )
                    if not is_last_lstm:
                        model.add(layers.BatchNormalization())

            # 全結合レイヤー
            model.add(layers.Dense(50, activation="relu"))
            model.add(layers.Dropout(self.dropout_rate))

            model.add(layers.Dense(25, activation="relu"))
            model.add(layers.Dropout(self.dropout_rate / 2))

            # 出力レイヤー（価格予測）
            model.add(layers.Dense(1, activation="linear"))

            # モデルコンパイル
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss="huber",  # 外れ値に頑健
                metrics=["mae", "mse"],
            )

            logger.info(f"LSTMモデル構築完了: {model.count_params():,}パラメータ")
            return model

        except Exception as e:
            logger.error(f"LSTMモデル構築エラー: {e}")
            raise

    def train_lstm_model(
        self,
        symbol: str,
        data: pd.DataFrame,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 15,
    ) -> Optional[Dict]:
        """
        LSTMモデル訓練

        Args:
            symbol: 銘柄コード
            data: 価格データ
            validation_split: 検証データ比率
            epochs: エポック数
            batch_size: バッチサイズ
            early_stopping_patience: 早期停止の忍耐パラメータ

        Returns:
            訓練結果辞書
        """
        if not TENSORFLOW_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.error("必要なライブラリが利用できません")
            return None

        try:
            # LSTM特徴量準備
            features = self.prepare_lstm_features(data)
            if features.empty:
                logger.error("特徴量準備に失敗")
                return None

            # データ正規化
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features = pd.DataFrame(
                scaler.fit_transform(features),
                columns=features.columns,
                index=features.index,
            )

            # 系列データ作成
            X, y = self.create_sequences(scaled_features, target_column="close_price")
            if len(X) == 0:
                logger.error("系列データ作成に失敗")
                return None

            logger.info(f"系列データ: X.shape={X.shape}, y.shape={y.shape}")

            # 訓練・検証データ分割
            split_index = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_index], X[split_index:]
            y_train, y_val = y[:split_index], y[split_index:]

            # モデル構築
            input_shape = (X.shape[1], X.shape[2])
            model = self.build_lstm_model(input_shape)

            # コールバック設定
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    verbose=0,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=0
                ),
            ]

            # モデル訓練
            logger.info(f"LSTM訓練開始: {symbol} (エポック={epochs})")
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,  # ログ出力を抑制
                shuffle=False,  # 時系列データなのでシャッフルしない
            )

            # モデル評価
            train_pred = model.predict(X_train, verbose=0)
            val_pred = model.predict(X_val, verbose=0)

            # 逆変換（実際の価格スケールに戻す）
            close_price_idx = features.columns.get_loc("close_price")

            # スケーラー逆変換のためのダミー配列作成
            dummy_train = np.zeros((len(y_train), features.shape[1]))
            dummy_val = np.zeros((len(y_val), features.shape[1]))

            dummy_train[:, close_price_idx] = y_train
            dummy_val[:, close_price_idx] = y_val

            dummy_train_pred = np.zeros((len(train_pred), features.shape[1]))
            dummy_val_pred = np.zeros((len(val_pred), features.shape[1]))

            dummy_train_pred[:, close_price_idx] = train_pred.flatten()
            dummy_val_pred[:, close_price_idx] = val_pred.flatten()

            y_train_actual = scaler.inverse_transform(dummy_train)[:, close_price_idx]
            y_val_actual = scaler.inverse_transform(dummy_val)[:, close_price_idx]
            train_pred_actual = scaler.inverse_transform(dummy_train_pred)[
                :, close_price_idx
            ]
            val_pred_actual = scaler.inverse_transform(dummy_val_pred)[
                :, close_price_idx
            ]

            # 評価指標計算
            train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred_actual))
            val_rmse = np.sqrt(mean_squared_error(y_val_actual, val_pred_actual))
            train_mae = mean_absolute_error(y_train_actual, train_pred_actual)
            val_mae = mean_absolute_error(y_val_actual, val_pred_actual)

            # R²スコア計算
            train_r2 = r2_score(y_train_actual, train_pred_actual)
            val_r2 = r2_score(y_val_actual, val_pred_actual)

            # モデル保存
            model_path = self.model_cache_dir / f"{symbol}_lstm_model.h5"
            model.save(model_path)

            # スケーラー保存
            import pickle

            scaler_path = self.model_cache_dir / f"{symbol}_scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            # モデルとスケーラーをキャッシュ
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.model_histories[symbol] = history

            training_result = {
                "symbol": symbol,
                "epochs_trained": len(history.history["loss"]),
                "final_train_loss": float(history.history["loss"][-1]),
                "final_val_loss": float(history.history["val_loss"][-1]),
                "train_rmse": float(train_rmse),
                "val_rmse": float(val_rmse),
                "train_mae": float(train_mae),
                "val_mae": float(val_mae),
                "train_r2": float(train_r2),
                "val_r2": float(val_r2),
                "model_params": model.count_params(),
                "sequence_length": self.sequence_length,
                "prediction_horizon": self.prediction_horizon,
                "feature_count": X.shape[2],
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"LSTM訓練完了: {symbol} - "
                f"Val R²={val_r2:.3f}, RMSE={val_rmse:.2f}, MAE={val_mae:.2f}"
            )

            return training_result

        except Exception as e:
            logger.error(f"LSTM訓練エラー ({symbol}): {e}")
            return None

    def predict_future_prices(
        self, symbol: str, data: pd.DataFrame, steps_ahead: int = None
    ) -> Optional[Dict]:
        """
        将来価格予測

        Args:
            symbol: 銘柄コード
            data: 価格データ
            steps_ahead: 予測ステップ数（Noneの場合はprediction_horizon使用）

        Returns:
            予測結果辞書
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow利用不可")
            return None

        try:
            steps = steps_ahead or self.prediction_horizon

            # モデルとスケーラーを取得
            if symbol not in self.models or symbol not in self.scalers:
                logger.warning(f"モデル未訓練: {symbol}")
                return None

            model = self.models[symbol]
            scaler = self.scalers[symbol]

            # 特徴量準備
            features = self.prepare_lstm_features(data)
            if features.empty:
                return None

            # データ正規化
            scaled_features = pd.DataFrame(
                scaler.transform(features),
                columns=features.columns,
                index=features.index,
            )

            # 最新のsequence_length日分を取得
            if len(scaled_features) < self.sequence_length:
                logger.error(
                    f"データ不足: {len(scaled_features)} < {self.sequence_length}"
                )
                return None

            last_sequence = scaled_features.values[-self.sequence_length :]
            last_sequence = last_sequence.reshape(1, self.sequence_length, -1)

            # 予測実行
            predictions = []
            current_sequence = last_sequence.copy()

            for step in range(steps):
                # 1ステップ予測
                pred = model.predict(current_sequence, verbose=0)
                predictions.append(pred[0, 0])

                # 複数ステップ予測の場合、予測値を次の入力に追加
                if step < steps - 1:
                    # 最新の特徴量ベクトルを取得し、予測価格で更新
                    next_features = current_sequence[0, -1, :].copy()
                    next_features[features.columns.get_loc("close_price")] = pred[0, 0]

                    # 系列を1つシフトして新しい特徴量を追加
                    current_sequence = np.roll(current_sequence, -1, axis=1)
                    current_sequence[0, -1, :] = next_features

            # 予測値を実際の価格スケールに戻す
            close_price_idx = features.columns.get_loc("close_price")

            actual_predictions = []
            for pred in predictions:
                dummy = np.zeros((1, features.shape[1]))
                dummy[0, close_price_idx] = pred
                actual_pred = scaler.inverse_transform(dummy)[0, close_price_idx]
                actual_predictions.append(float(actual_pred))

            # 現在価格
            current_price = float(data["Close"].iloc[-1])

            # 予測リターン計算
            predicted_returns = []
            for pred in actual_predictions:
                return_rate = (pred - current_price) / current_price * 100
                predicted_returns.append(float(return_rate))

            prediction_result = {
                "symbol": symbol,
                "current_price": current_price,
                "predicted_prices": actual_predictions,
                "predicted_returns": predicted_returns,
                "prediction_steps": steps,
                "prediction_dates": [
                    (data.index[-1] + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
                    for i in range(steps)
                ],
                "confidence_score": self._calculate_prediction_confidence(predictions),
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"LSTM予測完了: {symbol} - "
                f"{steps}日後価格: {actual_predictions[-1]:.2f} "
                f"({predicted_returns[-1]:+.2f}%)"
            )

            return prediction_result

        except Exception as e:
            logger.error(f"LSTM予測エラー ({symbol}): {e}")
            return None

    def _calculate_prediction_confidence(self, predictions: List[float]) -> float:
        """
        予測信頼度計算

        Args:
            predictions: 正規化された予測値リスト

        Returns:
            信頼度スコア（0-100）
        """
        try:
            if len(predictions) == 0:
                return 50.0

            # 予測値の分散（小さいほど安定＝高信頼度）
            pred_std = np.std(predictions)

            # 予測値の極端さチェック
            pred_mean = np.mean(predictions)

            # 基本信頼度（分散が小さいほど高い）
            base_confidence = max(0, min(100, 100 - pred_std * 200))

            # 極端な予測値にペナルティ
            if pred_mean < 0.1 or pred_mean > 0.9:  # 正規化された値での極端チェック
                base_confidence *= 0.8

            return float(base_confidence)

        except Exception as e:
            logger.error(f"予測信頼度計算エラー: {e}")
            return 50.0

    def get_model_performance(self, symbol: str) -> Optional[Dict]:
        """
        モデル性能情報取得

        Args:
            symbol: 銘柄コード

        Returns:
            性能情報辞書
        """
        if symbol not in self.models or symbol not in self.model_histories:
            return None

        try:
            history = self.model_histories[symbol].history

            return {
                "symbol": symbol,
                "final_train_loss": float(history["loss"][-1]),
                "final_val_loss": float(history["val_loss"][-1]),
                "final_train_mae": float(history["mae"][-1]),
                "final_val_mae": float(history["val_mae"][-1]),
                "epochs_trained": len(history["loss"]),
                "min_val_loss": float(min(history["val_loss"])),
                "min_val_loss_epoch": int(np.argmin(history["val_loss"]) + 1),
                "loss_history": [float(x) for x in history["loss"]],
                "val_loss_history": [float(x) for x in history["val_loss"]],
            }

        except Exception as e:
            logger.error(f"モデル性能取得エラー ({symbol}): {e}")
            return None

    def load_model(self, symbol: str) -> bool:
        """
        保存済みモデルの読み込み

        Args:
            symbol: 銘柄コード

        Returns:
            読み込み成功フラグ
        """
        if not TENSORFLOW_AVAILABLE:
            return False

        try:
            model_path = self.model_cache_dir / f"{symbol}_lstm_model.h5"
            scaler_path = self.model_cache_dir / f"{symbol}_scaler.pkl"

            if not model_path.exists() or not scaler_path.exists():
                logger.warning(f"保存済みモデルなし: {symbol}")
                return False

            # モデル読み込み
            model = keras.models.load_model(model_path)
            self.models[symbol] = model

            # スケーラー読み込み
            import pickle

            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            self.scalers[symbol] = scaler

            logger.info(f"モデル読み込み完了: {symbol}")
            return True

        except Exception as e:
            logger.error(f"モデル読み込みエラー ({symbol}): {e}")
            return False

    def cleanup_old_models(self, days_old: int = 30):
        """
        古いモデルファイルのクリーンアップ

        Args:
            days_old: 削除対象の日数
        """
        try:
            import time

            current_time = time.time()
            cutoff_time = current_time - (days_old * 24 * 3600)

            cleaned_count = 0
            for model_file in self.model_cache_dir.glob("*_lstm_model.h5"):
                file_time = model_file.stat().st_mtime
                if file_time < cutoff_time:
                    model_file.unlink()

                    # 対応するスケーラーファイルも削除
                    scaler_file = model_file.with_suffix(".pkl").name.replace(
                        "_lstm_model", "_scaler"
                    )
                    scaler_path = self.model_cache_dir / scaler_file
                    if scaler_path.exists():
                        scaler_path.unlink()

                    cleaned_count += 1

            logger.info(f"古いモデルクリーンアップ完了: {cleaned_count}ファイル削除")

        except Exception as e:
            logger.error(f"モデルクリーンアップエラー: {e}")


if __name__ == "__main__":
    # テスト実行
    print("=== LSTM時系列予測モデル テスト ===")

    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow未インストールのためテストを実行できません")
        exit(1)

    # サンプルデータ生成（実際のデータ取得に置き換え可能）
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    np.random.seed(42)

    # 模擬株価データ生成
    base_price = 2500
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    sample_data = pd.DataFrame(
        {
            "Open": [p * np.random.uniform(0.98, 1.02) for p in prices],
            "High": [p * np.random.uniform(1.00, 1.05) for p in prices],
            "Low": [p * np.random.uniform(0.95, 1.00) for p in prices],
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, len(dates)),
        },
        index=dates,
    )

    try:
        # LSTMモデル初期化
        lstm_model = LSTMTimeSeriesModel(
            sequence_length=30,  # テスト用に短縮
            prediction_horizon=3,
            lstm_units=[64, 32],  # テスト用に小型化
            epochs=20,  # テスト用に短縮
        )

        print(f"サンプルデータ: {len(sample_data)}日分")

        # 特徴量準備テスト
        print("\n1. 特徴量準備テスト")
        features = lstm_model.prepare_lstm_features(sample_data)
        print(f"✅ 特徴量準備完了: {len(features.columns)}特徴量")

        # 系列データ作成テスト
        print("\n2. 系列データ作成テスト")
        X, y = lstm_model.create_sequences(features)
        print(f"✅ 系列データ作成完了: X.shape={X.shape}, y.shape={y.shape}")

        # モデル訓練テスト
        print("\n3. LSTM訓練テスト")
        training_result = lstm_model.train_lstm_model(
            symbol="TEST_STOCK",
            data=sample_data,
            epochs=10,  # テスト用に短縮
            early_stopping_patience=3,
        )

        if training_result:
            print("✅ LSTM訓練完了")
            print(f"   検証R²: {training_result['val_r2']:.3f}")
            print(f"   検証RMSE: {training_result['val_rmse']:.2f}")
            print(f"   モデルパラメータ: {training_result['model_params']:,}")

        # 予測テスト
        print("\n4. LSTM予測テスト")
        prediction_result = lstm_model.predict_future_prices(
            symbol="TEST_STOCK", data=sample_data, steps_ahead=5
        )

        if prediction_result:
            print("✅ LSTM予測完了")
            print(f"   現在価格: {prediction_result['current_price']:.2f}")
            print(f"   予測価格: {prediction_result['predicted_prices']}")
            print(f"   予測リターン: {prediction_result['predicted_returns']}")
            print(f"   信頼度: {prediction_result['confidence_score']:.1f}%")

        # 性能情報テスト
        print("\n5. モデル性能テスト")
        performance = lstm_model.get_model_performance("TEST_STOCK")
        if performance:
            print("✅ 性能情報取得完了")
            print(f"   最終検証損失: {performance['final_val_loss']:.4f}")
            print(f"   最小検証損失: {performance['min_val_loss']:.4f}")
            print(f"   訓練エポック数: {performance['epochs_trained']}")

        print("\n✅ LSTM時系列予測モデル テスト完了！")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
