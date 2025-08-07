"""
レガシー設定管理（後方互換性のため）

このモジュールは既存のコードとの互換性を保つために、
古い設定クラスへのブリッジを提供します。

⚠️ 非推奨: 新しいコードでは unified_config を使用してください
"""

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.logging_config import get_context_logger
from .unified_config import get_unified_config_manager

logger = get_context_logger(__name__, component="legacy_config")


class ScreeningConfig:
    """
    レガシー ScreeningConfig クラス（後方互換性のため）

    ⚠️ 非推奨: 新しいコードでは unified_config.ScreeningConfig を使用してください
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        初期化

        Args:
            config_path: 設定ファイルのパス（無視されます - 統合設定を使用）
        """
        warnings.warn(
            "ScreeningConfig は非推奨です。unified_config.get_screening_config() を使用してください。",
            DeprecationWarning,
            stacklevel=2,
        )

        self._unified_manager = get_unified_config_manager()
        self._config = self._unified_manager.get_screening_config()

    def get_threshold(self, condition: str, default: float = None) -> float:
        """条件の閾値を取得"""
        return self._config.default_thresholds.get(condition, default)

    def get_lookback_days(self, condition: str, default: int = 20) -> int:
        """条件の参照期間を取得"""
        return self._config.default_lookback_days.get(condition, default)

    def get_performance_setting(self, key: str, default: Any = None) -> Any:
        """パフォーマンス設定を取得"""
        return self._config.performance_settings.get(key, default)

    def get_predefined_screener(self, name: str) -> Optional[Dict[str, Any]]:
        """事前定義スクリーナーを取得"""
        return self._config.predefined_screeners.get(name)

    def get_all_predefined_screeners(self) -> Dict[str, Dict[str, Any]]:
        """すべての事前定義スクリーナーを取得"""
        return self._config.predefined_screeners

    def should_use_actual_52_weeks(self) -> bool:
        """実際の52週間で計算するかどうか"""
        return self._config.week_calculation.get("use_actual_52_weeks", True)

    def should_fallback_to_available_data(self) -> bool:
        """利用可能データにフォールバックするかどうか"""
        return self._config.week_calculation.get("fallback_to_available_data", True)

    def should_use_formatters(self) -> bool:
        """フォーマッターを使用するかどうか"""
        return self._config.formatting.get("use_formatters", True)

    def get_currency_precision(self) -> int:
        """通貨の精度を取得"""
        return self._config.formatting.get("currency_precision", 0)

    def get_percentage_precision(self) -> int:
        """パーセンテージの精度を取得"""
        return self._config.formatting.get("percentage_precision", 2)

    def should_use_compact_volume(self) -> bool:
        """出来高をコンパクト表示するかどうか"""
        return self._config.formatting.get("volume_compact", True)

    def get_max_workers(self) -> int:
        """最大ワーカー数を取得（CPUコア数に基づく動的調整）"""
        import os

        configured = self.get_performance_setting("max_workers", 5)
        cpu_count = os.cpu_count() or 4
        # CPUコア数の75%を上限とする
        max_recommended = max(1, int(cpu_count * 0.75))
        return min(configured, max_recommended)

    def get_cache_size(self) -> int:
        """キャッシュサイズを取得"""
        return self.get_performance_setting("cache_size", 100)

    def get_data_period(self) -> str:
        """データ取得期間を取得"""
        return self.get_performance_setting("data_period", "3mo")

    def get_min_data_points(self) -> int:
        """最小データポイント数を取得"""
        return self.get_performance_setting("min_data_points", 30)

    def update_config(self, new_config: Dict[str, Any]):
        """設定を更新"""
        # 統合設定を更新
        self._unified_manager.update_config("screening", new_config)
        self._config = self._unified_manager.get_screening_config()

    def save_config(self):
        """設定をファイルに保存"""
        self._unified_manager.save_config()

    def get_config_dict(self) -> Dict[str, Any]:
        """設定辞書を取得（デバッグ用）"""
        return self._config.model_dump()


# グローバル設定インスタンス（後方互換性）
_global_screening_config = None


def get_screening_config() -> ScreeningConfig:
    """グローバル設定インスタンスを取得（後方互換性）"""
    global _global_screening_config
    if _global_screening_config is None:
        _global_screening_config = ScreeningConfig()
    return _global_screening_config


def set_screening_config(config: ScreeningConfig):
    """グローバル設定インスタンスを設定（後方互換性）"""
    global _global_screening_config
    _global_screening_config = config
