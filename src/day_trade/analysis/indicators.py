"""
テクニカル指標計算エンジン
各種テクニカル指標を計算するクラス
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger
from ..utils.progress import ProgressType, progress_context

logger = get_context_logger(__name__, component="technical_indicators")


class IndicatorsConfig:
    """テクニカル指標設定管理クラス"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "indicators_config.json"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(self.config_path, encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            "default_parameters": {
                "sma": {"periods": [5, 20, 60], "default_period": 20},
                "ema": {"periods": [12, 26], "default_period": 20},
                "bollinger_bands": {"period": 20, "num_std": 2.0},
                "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "rsi": {"period": 14},
                "stochastic": {"k_period": 14, "d_period": 3, "smooth_k": 3},
                "volume_analysis": {"period": 20},
                "atr": {"period": 14}
            },
            "calculate_all_defaults": {
                "sma_periods": [5, 20, 60],
                "ema_periods": [12, 26],
                "bb_period": 20,
                "bb_std": 2.0,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "rsi_period": 14,
                "stoch_k": 14,
                "stoch_d": 3,
                "stoch_smooth": 3,
                "volume_period": 20,
                "atr_period": 14
            },
            "performance_settings": {
                "progress_threshold": 100,
                "batch_concat": True
            },
            "error_handling": {
                "return_none_on_error": False,
                "detailed_logging": True
            }
        }

    def get_parameter(self, indicator: str, param: str, default=None):
        """指標パラメータを取得"""
        return self.config.get("default_parameters", {}).get(indicator, {}).get(param, default)

    def get_calculate_all_defaults(self) -> Dict[str, Any]:
        """calculate_allのデフォルト設定を取得"""
        return self.config.get("calculate_all_defaults", {})

    def get_performance_settings(self) -> Dict[str, Any]:
        """パフォーマンス設定を取得"""
        return self.config.get("performance_settings", {})

    def get_error_handling_settings(self) -> Dict[str, Any]:
        """エラーハンドリング設定を取得"""
        return self.config.get("error_handling", {})


class TechnicalIndicators:
    """テクニカル指標計算クラス"""

    _config = None

    @classmethod
    def _get_config(cls) -> IndicatorsConfig:
        """設定インスタンスを取得（シングルトンパターン）"""
        if cls._config is None:
            cls._config = IndicatorsConfig()
        return cls._config

    @classmethod
    def sma(cls, df: pd.DataFrame, period: Optional[int] = None, column: str = "Close") -> pd.Series:
        """
        単純移動平均線（Simple Moving Average）

        Args:
            df: 価格データのDataFrame
            period: 期間（Noneの場合は設定から取得）
            column: 計算対象の列名

        Returns:
            SMAのSeries
        """
        if period is None:
            config = cls._get_config()
            period = config.get_parameter("sma", "default_period", 20)

        try:
            if df.empty or column not in df.columns:
                logger.warning(f"SMA計算: データが空または列'{column}'が存在しません")
                return pd.Series(dtype=float)

            result = df[column].rolling(window=period).mean()
            logger.debug(f"SMA({period})計算完了: {len(result)}行処理")
            return result
        except Exception as e:
            config = cls._get_config()
            error_settings = config.get_error_handling_settings()

            if error_settings.get("detailed_logging", True):
                logger.error(
                    f"SMA (単純移動平均線) の計算中にエラーが発生しました。"
                    f"期間: {period}, 列: {column}, データ形状: {df.shape}, 詳細: {e}"
                )
            else:
                logger.error(f"SMA計算エラー: {e}")

            if error_settings.get("return_none_on_error", False):
                return None
            return pd.Series(dtype=float)

    @classmethod
    def ema(cls, df: pd.DataFrame, period: Optional[int] = None, column: str = "Close") -> pd.Series:
        """
        指数移動平均線（Exponential Moving Average）

        Args:
            df: 価格データのDataFrame
            period: 期間（Noneの場合は設定から取得）
            column: 計算対象の列名

        Returns:
            EMAのSeries
        """
        if period is None:
            config = cls._get_config()
            period = config.get_parameter("ema", "default_period", 20)

        try:
            if df.empty or column not in df.columns:
                logger.warning(f"EMA計算: データが空または列'{column}'が存在しません")
                return pd.Series(dtype=float)

            result = df[column].ewm(span=period, adjust=False).mean()
            logger.debug(f"EMA({period})計算完了: {len(result)}行処理")
            return result
        except Exception as e:
            config = cls._get_config()
            error_settings = config.get_error_handling_settings()

            if error_settings.get("detailed_logging", True):
                logger.error(
                    f"EMA (指数移動平均線) の計算中にエラーが発生しました。"
                    f"期間: {period}, 列: {column}, データ形状: {df.shape}, 詳細: {e}"
                )
            else:
                logger.error(f"EMA計算エラー: {e}")

            if error_settings.get("return_none_on_error", False):
                return None
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

    @classmethod
    def rsi(cls, df: pd.DataFrame, period: Optional[int] = None, column: str = "Close") -> pd.Series:
        """
        RSI（Relative Strength Index）

        Args:
            df: 価格データのDataFrame
            period: 期間（Noneの場合は設定から取得）
            column: 計算対象の列名

        Returns:
            RSIのSeries
        """
        if period is None:
            config = cls._get_config()
            period = config.get_parameter("rsi", "period", 14)

        try:
            if df.empty or column not in df.columns:
                logger.warning(f"RSI計算: データが空または列'{column}'が存在しません")
                return pd.Series(dtype=float)

            if len(df) < period + 1:
                logger.warning(f"RSI計算: データ不足（必要: {period + 1}行、実際: {len(df)}行）")
                return pd.Series(index=df.index, dtype=float)

            delta = df[column].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
            avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

            # ゼロ除算対策: avg_lossが0の場合の処理
            rs = np.where(avg_loss == 0,
                         np.where(avg_gain == 0, 1, np.inf),  # gain=0,loss=0なら50, gainのみならRSI=100
                         avg_gain / avg_loss)

            # RSI計算（無限大の場合は100になる）
            rsi = np.where(np.isinf(rs), 100, 100 - (100 / (1 + rs)))

            result = pd.Series(rsi, index=df.index)
            logger.debug(f"RSI({period})計算完了: {len(result)}行処理")
            return result

        except Exception as e:
            config = cls._get_config()
            error_settings = config.get_error_handling_settings()

            if error_settings.get("detailed_logging", True):
                logger.error(
                    f"RSI計算中にエラーが発生しました。"
                    f"期間: {period}, 列: {column}, データ形状: {df.shape}, 詳細: {e}"
                )
            else:
                logger.error(f"RSI計算エラー: {e}")

            if error_settings.get("return_none_on_error", False):
                return None
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

    @classmethod
    def calculate_all(
        cls,
        df: pd.DataFrame,
        sma_periods: Optional[List[int]] = None,
        ema_periods: Optional[List[int]] = None,
        bb_period: Optional[int] = None,
        bb_std: Optional[float] = None,
        macd_fast: Optional[int] = None,
        macd_slow: Optional[int] = None,
        macd_signal: Optional[int] = None,
        rsi_period: Optional[int] = None,
        stoch_k: Optional[int] = None,
        stoch_d: Optional[int] = None,
        stoch_smooth: Optional[int] = None,
        volume_period: Optional[int] = None,
        atr_period: Optional[int] = None,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """
        全てのテクニカル指標を計算

        Args:
            df: 価格データのDataFrame
            各種パラメータ（Noneの場合は設定から取得）
            show_progress: 進捗表示フラグ

        Returns:
            全指標を含むDataFrame
        """
        # 設定から各パラメータを取得
        config = cls._get_config()
        defaults = config.get_calculate_all_defaults()
        perf_settings = config.get_performance_settings()

        # パラメータのデフォルト値設定
        if sma_periods is None:
            sma_periods = defaults.get("sma_periods", [5, 20, 60])
        if ema_periods is None:
            ema_periods = defaults.get("ema_periods", [12, 26])
        if bb_period is None:
            bb_period = defaults.get("bb_period", 20)
        if bb_std is None:
            bb_std = defaults.get("bb_std", 2.0)
        if macd_fast is None:
            macd_fast = defaults.get("macd_fast", 12)
        if macd_slow is None:
            macd_slow = defaults.get("macd_slow", 26)
        if macd_signal is None:
            macd_signal = defaults.get("macd_signal", 9)
        if rsi_period is None:
            rsi_period = defaults.get("rsi_period", 14)
        if stoch_k is None:
            stoch_k = defaults.get("stoch_k", 14)
        if stoch_d is None:
            stoch_d = defaults.get("stoch_d", 3)
        if stoch_smooth is None:
            stoch_smooth = defaults.get("stoch_smooth", 3)
        if volume_period is None:
            volume_period = defaults.get("volume_period", 20)
        if atr_period is None:
            atr_period = defaults.get("atr_period", 14)

        try:
            if df.empty:
                logger.warning("空のDataFrameが渡されました")
                return pd.DataFrame()

            # パフォーマンス最適化: 全ての指標を辞書に計算してから一度に結合
            indicators = {}

            # 元のデータを含める
            for col in df.columns:
                indicators[col] = df[col]

            # 計算対象指標の総数を計算
            total_indicators = (
                len(sma_periods)
                + len(ema_periods)
                + 4  # BB, MACD系, RSI, ATR
                + 2  # ストキャスティクス, 出来高分析
            )

            progress_threshold = perf_settings.get("progress_threshold", 100)
            use_progress = show_progress and len(df) > progress_threshold

            def _calculate_with_progress(task_name, calculation_func):
                if use_progress:
                    progress.set_description(task_name)
                result = calculation_func()
                if use_progress:
                    progress.update(1)
                return result

            if use_progress:
                with progress_context(
                    f"テクニカル指標計算 ({total_indicators}指標)",
                    total=total_indicators,
                    progress_type=ProgressType.DETERMINATE,
                ) as progress:
                    cls._calculate_all_indicators(
                        df, indicators, sma_periods, ema_periods, bb_period, bb_std,
                        macd_fast, macd_slow, macd_signal, rsi_period,
                        stoch_k, stoch_d, stoch_smooth, volume_period, atr_period,
                        progress=progress
                    )
            else:
                # 進捗表示なしで実行
                cls._calculate_all_indicators(
                    df, indicators, sma_periods, ema_periods, bb_period, bb_std,
                    macd_fast, macd_slow, macd_signal, rsi_period,
                    stoch_k, stoch_d, stoch_smooth, volume_period, atr_period
                )

            # 一度にDataFrameを構築（パフォーマンス最適化）
            result = pd.DataFrame(indicators, index=df.index)
            logger.debug(f"テクニカル指標計算完了: 元データ{len(df.columns)}列 → 結果{len(result.columns)}列")
            return result

        except Exception as e:
            config = cls._get_config()
            error_settings = config.get_error_handling_settings()

            if error_settings.get("detailed_logging", True):
                logger.error(
                    f"全指標計算中にエラーが発生しました。"
                    f"データ形状: {df.shape}, 詳細: {e}"
                )
            else:
                logger.error(f"全指標計算エラー: {e}")

            if error_settings.get("return_none_on_error", False):
                return None
            return df

    @classmethod
    def _calculate_all_indicators(
        cls, df: pd.DataFrame, indicators: Dict[str, pd.Series],
        sma_periods: List[int], ema_periods: List[int],
        bb_period: int, bb_std: float,
        macd_fast: int, macd_slow: int, macd_signal: int, rsi_period: int,
        stoch_k: int, stoch_d: int, stoch_smooth: int,
        volume_period: int, atr_period: int,
        progress=None
    ):
        """指標計算の内部ロジック（進捗管理含む）"""
        def update_progress(description: str):
            if progress:
                progress.set_description(description)
                progress.update(1)

        # SMA計算
        for period in sma_periods:
            indicators[f"SMA_{period}"] = cls.sma(df, period)
            update_progress(f"SMA_{period} 計算中")

        # EMA計算
        for period in ema_periods:
            indicators[f"EMA_{period}"] = cls.ema(df, period)
            update_progress(f"EMA_{period} 計算中")

        # ボリンジャーバンド
        bb = cls.bollinger_bands(df, bb_period, bb_std)
        for col in bb.columns:
            indicators[col] = bb[col]
        update_progress("ボリンジャーバンド計算中")

        # MACD
        macd = cls.macd(df, macd_fast, macd_slow, macd_signal)
        for col in macd.columns:
            indicators[col] = macd[col]
        update_progress("MACD計算中")

        # RSI
        indicators["RSI"] = cls.rsi(df, rsi_period)
        update_progress("RSI計算中")

        # ストキャスティクス
        stoch = cls.stochastic(df, stoch_k, stoch_d, stoch_smooth)
        for col in stoch.columns:
            indicators[col] = stoch[col]
        update_progress("ストキャスティクス計算中")

        # 出来高分析
        volume = cls.volume_analysis(df, volume_period)
        for col in volume.columns:
            indicators[col] = volume[col]
        update_progress("出来高分析計算中")

        # ATR
        indicators["ATR"] = cls.atr(df, atr_period)
        update_progress("ATR計算中")


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

    logger.info("SMA（20日）計算", period=20)
    sma20 = ti.sma(df, period=20)
    logger.info("SMA計算結果",
                last_values=sma20.tail().to_dict(),
                calculation_period=20)

    logger.info("RSI（14日）計算", period=14)
    rsi = ti.rsi(df, period=14)
    logger.info("RSI計算結果",
                last_values=rsi.tail().to_dict(),
                calculation_period=14)

    logger.info("MACD計算")
    macd = ti.macd(df)
    logger.info("MACD計算結果",
                columns=macd.columns.tolist(),
                sample_data=macd.tail().to_dict())

    logger.info("全指標計算実行")
    all_indicators = ti.calculate_all(df)
    logger.info("全指標計算完了",
                total_columns=len(all_indicators.columns),
                available_indicators=all_indicators.columns.tolist(),
                data_rows=len(all_indicators),
                operation="calculate_all_indicators")
