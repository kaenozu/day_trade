#!/usr/bin/env python3
"""
拡張可能な特徴量エンジンシステム
Issue #575: カスタム特徴量の拡張性・エラーハンドリング改善対応

プラグイン式アーキテクチャでカスタム特徴量を動的に追加可能
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class FeatureEngineError(Exception):
    """特徴量エンジン固有エラー"""
    pass


class FeatureEngine(ABC):
    """特徴量エンジン抽象基底クラス"""

    name: str = ""
    required_columns: List[str] = []
    description: str = ""

    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """特徴量計算実行"""
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """データ検証"""
        if data is None or data.empty:
            return False

        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if missing_cols:
            logger.warning(f"{self.name}: 必要列が不足 - {missing_cols}")
            return False

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """メタデータ取得"""
        return {
            "name": self.name,
            "required_columns": self.required_columns,
            "description": self.description,
            "class": self.__class__.__name__
        }


class TrendStrengthEngine(FeatureEngine):
    """トレンド強度特徴量エンジン"""

    name = "trend_strength"
    required_columns = ["終値"]
    description = "短期・長期移動平均によるトレンド強度計算"

    def calculate(
        self,
        data: pd.DataFrame,
        short_period: int = 10,
        long_period: int = 30,
        **kwargs
    ) -> pd.DataFrame:
        """トレンド強度計算"""
        result = data.copy()

        try:
            short_ma = data["終値"].rolling(short_period).mean()
            long_ma = data["終値"].rolling(long_period).mean()

            # ゼロ除算対策
            with np.errstate(divide='ignore', invalid='ignore'):
                result["trend_strength"] = (short_ma - long_ma) / long_ma * 100
                result["trend_strength"] = result["trend_strength"].replace([np.inf, -np.inf], np.nan)

            result["trend_direction"] = np.where(
                result["trend_strength"] > 5, 1,  # 上昇
                np.where(result["trend_strength"] < -5, -1, 0)  # 下落/横這い
            )

            logger.debug(f"トレンド強度計算完了: {short_period}日/{long_period}日MA")

        except Exception as e:
            logger.error(f"トレンド強度計算エラー: {e}")
            raise FeatureEngineError(f"トレンド強度計算失敗: {str(e)}")

        return result


class MomentumEngine(FeatureEngine):
    """モメンタム特徴量エンジン"""

    name = "momentum"
    required_columns = ["終値"]
    description = "価格モメンタム指標計算"

    def calculate(
        self,
        data: pd.DataFrame,
        periods: List[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """モメンタム計算"""
        result = data.copy()

        if periods is None:
            periods = [5, 10, 20]

        try:
            for period in periods:
                col_name = f"momentum_{period}"
                result[col_name] = data["終値"].pct_change(period)

                # 異常値フィルタリング（±50%以上の変化を制限）
                result[col_name] = result[col_name].clip(-0.5, 0.5)

            # モメンタム平均
            momentum_cols = [f"momentum_{p}" for p in periods]
            result["momentum_avg"] = result[momentum_cols].mean(axis=1)

            # モメンタム強度（絶対値平均）
            result["momentum_strength"] = result[momentum_cols].abs().mean(axis=1)

            logger.debug(f"モメンタム計算完了: {periods}日間")

        except Exception as e:
            logger.error(f"モメンタム計算エラー: {e}")
            raise FeatureEngineError(f"モメンタム計算失敗: {str(e)}")

        return result


class PriceChannelEngine(FeatureEngine):
    """価格チャネル特徴量エンジン"""

    name = "price_channel"
    required_columns = ["高値", "安値", "終値"]
    description = "価格チャネル内でのポジション分析"

    def calculate(
        self,
        data: pd.DataFrame,
        period: int = 20,
        **kwargs
    ) -> pd.DataFrame:
        """価格チャネル計算"""
        result = data.copy()

        try:
            highest_high = data["高値"].rolling(period).max()
            lowest_low = data["安値"].rolling(period).min()

            # チャネル幅
            channel_width = highest_high - lowest_low

            # ゼロ除算対策
            with np.errstate(divide='ignore', invalid='ignore'):
                result["price_channel_position"] = (
                    (data["終値"] - lowest_low) / channel_width
                ).replace([np.inf, -np.inf], np.nan)

            # チャネル幅（ボラティリティ指標）
            result["price_channel_width"] = channel_width / data["終値"] * 100

            # チャネル境界近接度
            result["near_upper_bound"] = (result["price_channel_position"] > 0.8).astype(int)
            result["near_lower_bound"] = (result["price_channel_position"] < 0.2).astype(int)

            logger.debug(f"価格チャネル計算完了: {period}日間")

        except Exception as e:
            logger.error(f"価格チャネル計算エラー: {e}")
            raise FeatureEngineError(f"価格チャネル計算失敗: {str(e)}")

        return result


class GapAnalysisEngine(FeatureEngine):
    """ギャップ分析特徴量エンジン"""

    name = "gap_analysis"
    required_columns = ["始値", "終値"]
    description = "価格ギャップの検出と分析"

    def calculate(
        self,
        data: pd.DataFrame,
        gap_threshold: float = 0.02,
        **kwargs
    ) -> pd.DataFrame:
        """ギャップ分析計算"""
        result = data.copy()

        try:
            prev_close = data["終値"].shift(1)

            # ギャップサイズ計算
            gap_size = (data["始値"] - prev_close) / prev_close
            result["gap_size"] = gap_size

            # ギャップアップ・ダウン判定
            result["gap_up"] = (gap_size > gap_threshold).astype(int)
            result["gap_down"] = (gap_size < -gap_threshold).astype(int)

            # ギャップフィル分析（当日中にギャップが埋められたか）
            result["gap_filled"] = np.where(
                result["gap_up"] == 1,
                (data["安値"] <= prev_close).astype(int),
                np.where(
                    result["gap_down"] == 1,
                    (data["高値"] >= prev_close).astype(int),
                    0
                )
            )

            # ギャップ強度（絶対値）
            result["gap_strength"] = np.abs(gap_size)

            logger.debug(f"ギャップ分析完了: 閾値±{gap_threshold*100:.1f}%")

        except Exception as e:
            logger.error(f"ギャップ分析エラー: {e}")
            raise FeatureEngineError(f"ギャップ分析失敗: {str(e)}")

        return result


class VolumeAnalysisEngine(FeatureEngine):
    """出来高分析特徴量エンジン"""

    name = "volume_analysis"
    required_columns = ["出来高"]
    description = "出来高パターンとトレンド分析"

    def calculate(
        self,
        data: pd.DataFrame,
        short_period: int = 5,
        long_period: int = 20,
        **kwargs
    ) -> pd.DataFrame:
        """出来高分析計算"""
        result = data.copy()

        try:
            # 出来高移動平均
            vol_ma_short = data["出来高"].rolling(short_period).mean()
            vol_ma_long = data["出来高"].rolling(long_period).mean()

            result[f"volume_ma_{short_period}"] = vol_ma_short
            result[f"volume_ma_{long_period}"] = vol_ma_long

            # 出来高比率
            with np.errstate(divide='ignore', invalid='ignore'):
                result["volume_ratio_short"] = (
                    data["出来高"] / vol_ma_short
                ).replace([np.inf, -np.inf], np.nan)

                result["volume_ratio_long"] = (
                    data["出来高"] / vol_ma_long
                ).replace([np.inf, -np.inf], np.nan)

            # 出来高トレンド
            result["volume_trend"] = (vol_ma_short / vol_ma_long).replace([np.inf, -np.inf], np.nan)

            # 異常出来高判定（平均の2倍以上）
            result["volume_spike"] = (result["volume_ratio_long"] > 2.0).astype(int)

            # 出来高変化率
            result["volume_change"] = data["出来高"].pct_change()

            logger.debug(f"出来高分析完了: {short_period}日/{long_period}日MA")

        except Exception as e:
            logger.error(f"出来高分析エラー: {e}")
            raise FeatureEngineError(f"出来高分析失敗: {str(e)}")

        return result


class FeatureEngineRegistry:
    """特徴量エンジン登録システム"""

    def __init__(self):
        self._engines: Dict[str, Type[FeatureEngine]] = {}
        self._register_default_engines()

    def _register_default_engines(self):
        """デフォルトエンジンの登録"""
        default_engines = [
            TrendStrengthEngine,
            MomentumEngine,
            PriceChannelEngine,
            GapAnalysisEngine,
            VolumeAnalysisEngine,
        ]

        for engine_class in default_engines:
            self.register(engine_class)

    def register(self, engine_class: Type[FeatureEngine]):
        """特徴量エンジン登録"""
        if not issubclass(engine_class, FeatureEngine):
            raise ValueError(f"{engine_class} はFeatureEngineのサブクラスではありません")

        # 一時インスタンス作成でname取得
        temp_instance = engine_class()
        if not temp_instance.name:
            raise ValueError(f"{engine_class} にnameが設定されていません")

        self._engines[temp_instance.name] = engine_class
        logger.info(f"特徴量エンジン登録: {temp_instance.name} ({engine_class.__name__})")

    def get_engine(self, name: str) -> Optional[FeatureEngine]:
        """特徴量エンジン取得"""
        if name not in self._engines:
            logger.warning(f"未知の特徴量エンジン: {name}")
            return None

        try:
            return self._engines[name]()
        except Exception as e:
            logger.error(f"特徴量エンジン初期化エラー {name}: {e}")
            return None

    def list_engines(self) -> Dict[str, Dict[str, Any]]:
        """登録済みエンジン一覧"""
        engines_info = {}

        for name, engine_class in self._engines.items():
            try:
                temp_instance = engine_class()
                engines_info[name] = temp_instance.get_metadata()
            except Exception as e:
                engines_info[name] = {"error": str(e)}

        return engines_info

    def calculate_feature(
        self,
        data: pd.DataFrame,
        feature_name: str,
        **kwargs
    ) -> pd.DataFrame:
        """特徴量計算実行"""
        engine = self.get_engine(feature_name)
        if engine is None:
            logger.error(f"特徴量エンジンが見つかりません: {feature_name}")
            return data

        # データ検証
        if not engine.validate_data(data):
            logger.error(f"データ検証失敗: {feature_name}")
            return data

        try:
            # エンジン固有パラメータ抽出
            engine_params = self._extract_engine_params(engine, kwargs)
            result = engine.calculate(data, **engine_params)

            logger.info(f"特徴量計算成功: {feature_name}")
            return result

        except FeatureEngineError:
            # エンジン固有エラーは再発生
            raise
        except Exception as e:
            logger.error(f"特徴量計算で予期しないエラー {feature_name}: {e}")
            raise FeatureEngineError(f"特徴量計算失敗: {str(e)}")

    def _extract_engine_params(self, engine: FeatureEngine, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """エンジン固有パラメータ抽出"""
        try:
            # calculateメソッドのシグネチャ取得
            sig = inspect.signature(engine.calculate)
            valid_params = {}

            for param_name, param in sig.parameters.items():
                if param_name in ['self', 'data']:
                    continue

                if param_name in kwargs:
                    valid_params[param_name] = kwargs[param_name]
                elif param.default != inspect.Parameter.empty:
                    # デフォルト値があればそれを使用
                    valid_params[param_name] = param.default

            return valid_params

        except Exception as e:
            logger.warning(f"パラメータ抽出エラー: {e}")
            return {}


# グローバル登録システム
_global_registry = FeatureEngineRegistry()

def get_feature_registry() -> FeatureEngineRegistry:
    """グローバル特徴量エンジン登録システム取得"""
    return _global_registry

def register_custom_engine(engine_class: Type[FeatureEngine]):
    """カスタム特徴量エンジン登録（便利関数）"""
    _global_registry.register(engine_class)

def calculate_custom_feature(data: pd.DataFrame, feature_name: str, **kwargs) -> pd.DataFrame:
    """カスタム特徴量計算（便利関数）"""
    return _global_registry.calculate_feature(data, feature_name, **kwargs)

def list_available_features() -> Dict[str, Dict[str, Any]]:
    """利用可能特徴量一覧（便利関数）"""
    return _global_registry.list_engines()


if __name__ == "__main__":
    # テスト実行
    print("=== 拡張可能特徴量エンジンシステム テスト ===")

    # サンプルデータ作成
    import datetime

    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    test_data = pd.DataFrame({
        '始値': 100 + np.random.randn(100).cumsum() * 0.5,
        '高値': 100 + np.random.randn(100).cumsum() * 0.5 + np.random.rand(100) * 2,
        '安値': 100 + np.random.randn(100).cumsum() * 0.5 - np.random.rand(100) * 2,
        '終値': 100 + np.random.randn(100).cumsum() * 0.5,
        '出来高': 1000000 + np.random.randint(-100000, 100000, 100),
    }, index=dates)

    # 高値・安値を適切に調整
    test_data['高値'] = np.maximum(test_data[['始値', '終値']].max(axis=1), test_data['高値'])
    test_data['安値'] = np.minimum(test_data[['始値', '終値']].min(axis=1), test_data['安値'])

    print(f"テストデータ作成: {len(test_data)}日間")

    # 利用可能な特徴量一覧
    features = list_available_features()
    print(f"\n利用可能特徴量: {len(features)}種類")
    for name, info in features.items():
        print(f"  - {name}: {info.get('description', 'N/A')}")

    # 特徴量計算テスト
    print("\n特徴量計算テスト:")

    for feature_name in ["trend_strength", "momentum", "price_channel", "volume_analysis"]:
        try:
            result = calculate_custom_feature(test_data, feature_name)
            new_cols = set(result.columns) - set(test_data.columns)
            print(f"✓ {feature_name}: {len(new_cols)}個の新特徴量追加")
            print(f"    追加列: {list(new_cols)[:3]}{'...' if len(new_cols) > 3 else ''}")

            test_data = result  # 結果を累積

        except Exception as e:
            print(f"✗ {feature_name}: エラー - {e}")

    print(f"\n最終データセット: {test_data.shape[1]}列")
    print("✅ 拡張可能特徴量エンジンシステム テスト完了")