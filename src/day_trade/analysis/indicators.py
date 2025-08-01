"""
テクニカル指標計算エンジン
各種テクニカル指標を計算するクラス
"""

import logging

import numpy as np
import pandas as pd

from ..utils.progress import ProgressType, progress_context

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """テクニカル指標計算クラス"""

    @staticmethod
    def sma(df: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.Series:
        """
        単純移動平均線（Simple Moving Average）

        Args:
            df: 価格データのDataFrame
            period: 期間
            column: 計算対象の列名

        Returns:
            SMAのSeries
        """
        try:
            return df[column].rolling(window=period).mean()
        except Exception as e:
            logger.error(
                f"SMA (単純移動平均線) の計算中にエラーが発生しました。入力データを確認してください。詳細: {e}"
            )
            return pd.Series(dtype=float)

    @staticmethod
    def ema(df: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.Series:
        """
        指数移動平均線（Exponential Moving Average）

        Args:
            df: 価格データのDataFrame
            period: 期間
            column: 計算対象の列名

        Returns:
            EMAのSeries
        """
        try:
            return df[column].ewm(span=period, adjust=False).mean()
        except Exception as e:
            logger.error(
                f"EMA (指数移動平均線) の計算中にエラーが発生しました。入力データを確認してください。詳細: {e}"
            )
            return pd.Series(dtype=float)

    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame, period: int = 20, num_std: float = 2, column: str = "Close"
    ) -> pd.DataFrame:
        """
        ボリンジャーバンド

        Args:
            df: 価格データのDataFrame
            period: 期間
            num_std: 標準偏差の倍数
            column: 計算対象の列名

        Returns:
            上部バンド、中間バンド、下部バンドを含むDataFrame
        """
        try:
            middle_band = df[column].rolling(window=period).mean()
            std = df[column].rolling(window=period).std()
            upper_band = middle_band + (std * num_std)
            lower_band = middle_band - (std * num_std)

            return pd.DataFrame(
                {
                    "BB_Upper": upper_band,
                    "BB_Middle": middle_band,
                    "BB_Lower": lower_band,
                }
            )
        except Exception as e:
            logger.error(
                f"ボリンジャーバンドの計算中にエラーが発生しました。入力データを確認してください。詳細: {e}"
            )
            return pd.DataFrame()

    @staticmethod
    def macd(
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "Close",
    ) -> pd.DataFrame:
        """
        MACD（Moving Average Convergence Divergence）

        Args:
            df: 価格データのDataFrame
            fast_period: 短期EMA期間
            slow_period: 長期EMA期間
            signal_period: シグナル線の期間
            column: 計算対象の列名

        Returns:
            MACD、シグナル、ヒストグラムを含むDataFrame
        """
        try:
            exp1 = df[column].ewm(span=fast_period, adjust=False).mean()
            exp2 = df[column].ewm(span=slow_period, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line

            return pd.DataFrame(
                {
                    "MACD": macd_line,
                    "MACD_Signal": signal_line,
                    "MACD_Histogram": histogram,
                }
            )
        except Exception as e:
            logger.error(
                f"MACD (移動平均収束拡散) の計算中にエラーが発生しました。入力データを確認してください。詳細: {e}"
            )
            return pd.DataFrame()

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
        """
        RSI（Relative Strength Index）

        Args:
            df: 価格データのDataFrame
            period: 期間
            column: 計算対象の列名

        Returns:
            RSIのSeries
        """
        try:
            delta = df[column].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
            avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi
        except Exception as e:
            logger.error(f"RSI計算エラー: {e}")
            return pd.Series(dtype=float)

    @staticmethod
    def stochastic(
        df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k: int = 3
    ) -> pd.DataFrame:
        """
        ストキャスティクス

        Args:
            df: 価格データのDataFrame（High, Low, Close列が必要）
            k_period: %K期間
            d_period: %D期間
            smooth_k: %Kの平滑化期間

        Returns:
            %K、%Dを含むDataFrame
        """
        try:
            # 期間内の最高値と最安値
            low_min = df["Low"].rolling(window=k_period).min()
            high_max = df["High"].rolling(window=k_period).max()

            # Fast %K
            fast_k = 100 * ((df["Close"] - low_min) / (high_max - low_min))

            # Slow %K（平滑化）
            slow_k = fast_k.rolling(window=smooth_k).mean()

            # %D
            slow_d = slow_k.rolling(window=d_period).mean()

            return pd.DataFrame({"Stoch_K": slow_k, "Stoch_D": slow_d})
        except Exception as e:
            logger.error(f"ストキャスティクス計算エラー: {e}")
            return pd.DataFrame()

    @staticmethod
    def volume_analysis(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        出来高分析

        Args:
            df: 価格データのDataFrame（Volume列が必要）
            period: 移動平均期間

        Returns:
            出来高移動平均と出来高比率を含むDataFrame
        """
        try:
            volume_ma = df["Volume"].rolling(window=period).mean()
            volume_ratio = df["Volume"] / volume_ma

            return pd.DataFrame({"Volume_MA": volume_ma, "Volume_Ratio": volume_ratio})
        except Exception as e:
            logger.error(f"出来高分析エラー: {e}")
            return pd.DataFrame()

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        ATR（Average True Range）

        Args:
            df: 価格データのDataFrame（High, Low, Close列が必要）
            period: 期間

        Returns:
            ATRのSeries
        """
        try:
            high_low = df["High"] - df["Low"]
            high_close = np.abs(df["High"] - df["Close"].shift())
            low_close = np.abs(df["Low"] - df["Close"].shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
                axis=1
            )
            atr = true_range.ewm(com=period - 1, adjust=False).mean()

            return atr
        except Exception as e:
            logger.error(f"ATR計算エラー: {e}")
            return pd.Series(dtype=float)

    @staticmethod
    def calculate_all(
        df: pd.DataFrame,
        sma_periods: list = [5, 20, 60],
        ema_periods: list = [12, 26],
        bb_period: int = 20,
        bb_std: float = 2,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
        stoch_k: int = 14,
        stoch_d: int = 3,
        stoch_smooth: int = 3,
        volume_period: int = 20,
        atr_period: int = 14,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """
        全てのテクニカル指標を計算

        Args:
            df: 価格データのDataFrame
            各種パラメータ
            show_progress: 進捗表示フラグ

        Returns:
            全指標を含むDataFrame
        """
        try:
            result = df.copy()

            # 計算対象指標の総数を計算
            total_indicators = (
                len(sma_periods)
                + len(ema_periods)
                + 1  # ボリンジャーバンド
                + 1  # MACD
                + 1  # RSI
                + 1  # ストキャスティクス
                + 1  # 出来高分析
                + 1  # ATR
            )

            if show_progress and len(df) > 100:  # 100行以上のデータで進捗表示
                with progress_context(
                    f"テクニカル指標計算 ({total_indicators}指標)",
                    total=total_indicators,
                    progress_type=ProgressType.DETERMINATE,
                ) as progress:
                    # SMA
                    for period in sma_periods:
                        progress.set_description(f"SMA_{period} 計算中")
                        result[f"SMA_{period}"] = TechnicalIndicators.sma(df, period)
                        progress.update(1)

                    # EMA
                    for period in ema_periods:
                        progress.set_description(f"EMA_{period} 計算中")
                        result[f"EMA_{period}"] = TechnicalIndicators.ema(df, period)
                        progress.update(1)

                    # ボリンジャーバンド
                    progress.set_description("ボリンジャーバンド計算中")
                    bb = TechnicalIndicators.bollinger_bands(df, bb_period, bb_std)
                    result = pd.concat([result, bb], axis=1)
                    progress.update(1)

                    # MACD
                    progress.set_description("MACD計算中")
                    macd = TechnicalIndicators.macd(
                        df, macd_fast, macd_slow, macd_signal
                    )
                    result = pd.concat([result, macd], axis=1)
                    progress.update(1)

                    # RSI
                    progress.set_description("RSI計算中")
                    result["RSI"] = TechnicalIndicators.rsi(df, rsi_period)
                    progress.update(1)

                    # ストキャスティクス
                    progress.set_description("ストキャスティクス計算中")
                    stoch = TechnicalIndicators.stochastic(
                        df, stoch_k, stoch_d, stoch_smooth
                    )
                    result = pd.concat([result, stoch], axis=1)
                    progress.update(1)

                    # 出来高分析
                    progress.set_description("出来高分析計算中")
                    volume = TechnicalIndicators.volume_analysis(df, volume_period)
                    result = pd.concat([result, volume], axis=1)
                    progress.update(1)

                    # ATR
                    progress.set_description("ATR計算中")
                    result["ATR"] = TechnicalIndicators.atr(df, atr_period)
                    progress.update(1)
            else:
                # 進捗表示なしで実行（従来通り）
                # SMA
                for period in sma_periods:
                    result[f"SMA_{period}"] = TechnicalIndicators.sma(df, period)

                # EMA
                for period in ema_periods:
                    result[f"EMA_{period}"] = TechnicalIndicators.ema(df, period)

                # ボリンジャーバンド
                bb = TechnicalIndicators.bollinger_bands(df, bb_period, bb_std)
                result = pd.concat([result, bb], axis=1)

                # MACD
                macd = TechnicalIndicators.macd(df, macd_fast, macd_slow, macd_signal)
                result = pd.concat([result, macd], axis=1)

                # RSI
                result["RSI"] = TechnicalIndicators.rsi(df, rsi_period)

                # ストキャスティクス
                stoch = TechnicalIndicators.stochastic(
                    df, stoch_k, stoch_d, stoch_smooth
                )
                result = pd.concat([result, stoch], axis=1)

                # 出来高分析
                volume = TechnicalIndicators.volume_analysis(df, volume_period)
                result = pd.concat([result, volume], axis=1)

                # ATR
                result["ATR"] = TechnicalIndicators.atr(df, atr_period)

            return result

        except Exception as e:
            logger.error(f"全指標計算エラー: {e}")
            return df


# 使用例
if __name__ == "__main__":
    # サンプルデータ作成
    from datetime import datetime

    import numpy as np

    # ダミーデータ生成
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    np.random.seed(42)

    # ランダムウォークで価格データを生成
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)

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

    # テクニカル指標計算
    ti = TechnicalIndicators()

    print("=== SMA（20日） ===")
    sma20 = ti.sma(df, period=20)
    print(sma20.tail())

    print("\n=== RSI（14日） ===")
    rsi = ti.rsi(df, period=14)
    print(rsi.tail())

    print("\n=== MACD ===")
    macd = ti.macd(df)
    print(macd.tail())

    print("\n=== 全指標計算 ===")
    all_indicators = ti.calculate_all(df)
    print(all_indicators.columns.tolist())
    print(f"計算完了: {len(all_indicators.columns)}列")
