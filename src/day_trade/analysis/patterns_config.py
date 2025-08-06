"""
パターン分析関連の設定

テクニカル分析パターンの設定値を定義する。
チャートパターン認識の設定管理
パラメータの外部化とマジックナンバーの排除
"""

import json
from pathlib import Path
from typing import Any, Dict


def get_patterns_config() -> Dict[str, Any]:
    """
    パターン分析設定を取得

    Returns:
        設定辞書
    """
    # デフォルト設定
    default_config = {
        "golden_dead_cross": {
            "fast_period": 5,
            "slow_period": 25,
            "confidence_threshold": 70,
        },
        "support_resistance": {"window_size": 20, "strength_threshold": 3},
        "breakout_detection": {"volume_factor": 1.5, "price_change_threshold": 2.0},
        "rsi": {"period": 14, "oversold": 30, "overbought": 70},
        "bollinger_bands": {"period": 20, "std_dev": 2},
    }

    # 設定ファイルからの読み込みを試行
    config_path = (
        Path(__file__).parent.parent.parent.parent / "config" / "patterns_config.json"
    )

    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                file_config = json.load(f)
                # デフォルト設定にファイル設定をマージ
                _merge_config(default_config, file_config)
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"設定ファイル読み込みエラー: {e}、デフォルト設定を使用")

    return default_config


def _merge_config(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    """
    設定辞書をマージ

    Args:
        base: ベース設定（更新される）
        overlay: オーバーレイ設定
    """
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge_config(base[key], value)
        else:
            base[key] = value


def save_patterns_config(config: Dict[str, Any]) -> None:
    """
    パターン分析設定を保存

    Args:
        config: 設定辞書
    """
    config_dir = Path(__file__).parent.parent.parent.parent / "config"
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / "patterns_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# Issue #122互換性のために、関数ベースのインターフェースも保持
# mainブランチのクラスベース実装も利用可能にする
from typing import Any, Optional  # noqa: E402

from ..utils.logging_config import get_context_logger  # noqa: E402

logger = get_context_logger(__name__)


class PatternsConfig:
    """チャートパターン認識設定管理クラス"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        初期化

        Args:
            config_path: 設定ファイルのパス
        """
        if config_path is None:
            # デフォルトの設定ファイルパスを設定
            current_dir = Path(__file__).parent.parent.parent.parent
            config_path = current_dir / "config" / "patterns_config.json"

        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, encoding="utf-8") as f:
                    config = json.load(f)
                logger.info(f"チャートパターン設定を読み込み: {self.config_path}")
                return config
            else:
                logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す（Issue #122互換デフォルト含む）"""
        return {
            "golden_dead_cross": {
                "fast_period": 5,
                "slow_period": 25,  # Issue #122デフォルト
                "confidence_threshold": 70,
                "default_fast_period": 5,
                "default_slow_period": 20,
                "confidence_multiplier": 100,
                "confidence_clip_max": 100,
            },
            "support_resistance": {
                "window_size": 20,
                "strength_threshold": 3,
                "default_window": 20,
                "default_num_levels": 3,
                "clustering_iterations": 10,
                "min_candidates_threshold": 0,
            },
            "breakout_detection": {
                "volume_factor": 1.5,
                "price_change_threshold": 2.0,
                "default_lookback": 20,
                "default_threshold": 0.02,
                "default_volume_factor": 1.5,
                "strength_multiplier": 10,
                "volume_clip_max": 2,
                "confidence_cap": 100,
            },
            "rsi": {"period": 14, "oversold": 30, "overbought": 70},
            "bollinger_bands": {"period": 20, "std_dev": 2},
            "trend_line_detection": {"default_window": 20, "default_min_touches": 3},
            "detect_all_patterns": {
                "default_golden_cross_fast": 5,
                "default_golden_cross_slow": 20,
                "default_support_resistance_window": 20,
                "default_breakout_lookback": 20,
                "default_trend_window": 20,
            },
            "confidence_calculation": {
                "weights": {
                    "golden_cross": 0.3,
                    "dead_cross": 0.3,
                    "upward_breakout": 0.25,
                    "downward_breakout": 0.25,
                    "trend_r2": 0.2,
                }
            },
            "error_handling": {
                "return_empty_on_error": True,
                "log_detailed_errors": True,
                "raise_exceptions": False,
            },
        }

    def get_config_dict(self) -> Dict[str, Any]:
        """設定辞書を取得（Issue #122互換）"""
        return self._config.copy()


# グローバル設定インスタンス
_global_config_class = None


def get_patterns_config_class() -> PatternsConfig:
    """グローバル設定クラスインスタンスを取得"""
    global _global_config_class
    if _global_config_class is None:
        _global_config_class = PatternsConfig()
    return _global_config_class
