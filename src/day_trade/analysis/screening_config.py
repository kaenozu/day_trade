"""
スクリーニング機能の設定管理
設定値の外部化とデフォルト値の管理
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class ScreeningConfig:
    """スクリーニング設定管理クラス"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        初期化

        Args:
            config_path: 設定ファイルのパス
        """
        if config_path is None:
            # デフォルトの設定ファイルパスを設定
            current_dir = Path(__file__).parent.parent.parent.parent
            config_path = current_dir / "config" / "screening_config.json"

        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"スクリーニング設定を読み込み: {self.config_path}")
                return config
            else:
                logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            "default_thresholds": {
                "rsi_oversold": 30.0,
                "rsi_overbought": 70.0,
                "volume_spike": 2.0,
                "strong_momentum": 0.05,
                "bollinger_squeeze": 0.02,
                "price_near_support": 0.03,
                "price_near_resistance": 0.03
            },
            "default_lookback_days": {
                "rsi_oversold": 20,
                "rsi_overbought": 20,
                "macd_bullish": 20,
                "macd_bearish": 20,
                "golden_cross": 20,
                "dead_cross": 20,
                "volume_spike": 20,
                "strong_momentum": 20,
                "bollinger_breakout": 20,
                "bollinger_squeeze": 20,
                "price_near_support": 20,
                "price_near_resistance": 20,
                "reversal_pattern": 20
            },
            "performance_settings": {
                "max_workers": 5,
                "cache_size": 100,
                "data_period": "3mo",
                "min_data_points": 30
            },
            "predefined_screeners": {},
            "52_week_calculation": {
                "use_actual_52_weeks": True,
                "fallback_to_available_data": True
            },
            "formatting": {
                "use_formatters": True,
                "currency_precision": 0,
                "percentage_precision": 2,
                "volume_compact": True
            }
        }

    def get_threshold(self, condition: str, default: float = None) -> float:
        """条件の閾値を取得"""
        return self._config.get("default_thresholds", {}).get(condition, default)

    def get_lookback_days(self, condition: str, default: int = 20) -> int:
        """条件の参照期間を取得"""
        return self._config.get("default_lookback_days", {}).get(condition, default)

    def get_performance_setting(self, key: str, default: Any = None) -> Any:
        """パフォーマンス設定を取得"""
        return self._config.get("performance_settings", {}).get(key, default)

    def get_predefined_screener(self, name: str) -> Optional[Dict[str, Any]]:
        """事前定義スクリーナーを取得"""
        return self._config.get("predefined_screeners", {}).get(name)

    def get_all_predefined_screeners(self) -> Dict[str, Dict[str, Any]]:
        """すべての事前定義スクリーナーを取得"""
        return self._config.get("predefined_screeners", {})

    def should_use_actual_52_weeks(self) -> bool:
        """実際の52週間で計算するかどうか"""
        return self._config.get("52_week_calculation", {}).get("use_actual_52_weeks", True)

    def should_fallback_to_available_data(self) -> bool:
        """利用可能データにフォールバックするかどうか"""
        return self._config.get("52_week_calculation", {}).get("fallback_to_available_data", True)

    def should_use_formatters(self) -> bool:
        """フォーマッターを使用するかどうか"""
        return self._config.get("formatting", {}).get("use_formatters", True)

    def get_currency_precision(self) -> int:
        """通貨の精度を取得"""
        return self._config.get("formatting", {}).get("currency_precision", 0)

    def get_percentage_precision(self) -> int:
        """パーセンテージの精度を取得"""
        return self._config.get("formatting", {}).get("percentage_precision", 2)

    def should_use_compact_volume(self) -> bool:
        """出来高をコンパクト表示するかどうか"""
        return self._config.get("formatting", {}).get("volume_compact", True)

    def update_config(self, new_config: Dict[str, Any]):
        """設定を更新"""
        self._config.update(new_config)

    def save_config(self):
        """設定をファイルに保存"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
            logger.info(f"設定を保存: {self.config_path}")
        except Exception as e:
            logger.error(f"設定保存エラー: {e}")

    def get_max_workers(self) -> int:
        """最大ワーカー数を取得（CPUコア数に基づく動的調整）"""
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

    def get_config_dict(self) -> Dict[str, Any]:
        """設定辞書を取得（デバッグ用）"""
        return self._config.copy()


# グローバル設定インスタンス
_global_config = None


def get_screening_config() -> ScreeningConfig:
    """グローバル設定インスタンスを取得"""
    global _global_config
    if _global_config is None:
        _global_config = ScreeningConfig()
    return _global_config


def set_screening_config(config: ScreeningConfig):
    """グローバル設定インスタンスを設定"""
    global _global_config
    _global_config = config
