"""
チャートパターン認識の設定管理
パラメータの外部化とマジックナンバーの排除
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.logging_config import get_context_logger

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
                with open(self.config_path, encoding='utf-8') as f:
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
        """デフォルト設定を返す"""
        return {
            "golden_dead_cross": {
                "default_fast_period": 5,
                "default_slow_period": 20,
                "confidence_multiplier": 100,
                "confidence_clip_max": 100
            },
            "support_resistance": {
                "default_window": 20,
                "default_num_levels": 3,
                "clustering_iterations": 10,
                "min_candidates_threshold": 0
            },
            "breakout_detection": {
                "default_lookback": 20,
                "default_threshold": 0.02,
                "default_volume_factor": 1.5,
                "strength_multiplier": 10,
                "volume_clip_max": 2,
                "confidence_cap": 100
            },
            "trend_line_detection": {
                "default_window": 20,
                "default_min_touches": 3
            },
            "detect_all_patterns": {
                "default_golden_cross_fast": 5,
                "default_golden_cross_slow": 20,
                "default_support_resistance_window": 20,
                "default_breakout_lookback": 20,
                "default_trend_window": 20
            },
            "confidence_calculation": {
                "weights": {
                    "golden_cross": 0.3,
                    "dead_cross": 0.3,
                    "upward_breakout": 0.25,
                    "downward_breakout": 0.25,
                    "trend_r2": 0.2
                }
            },
            "error_handling": {
                "return_empty_on_error": True,
                "log_detailed_errors": True,
                "raise_exceptions": False
            }
        }

    # ゴールデン・デッドクロス設定
    def get_golden_cross_fast_period(self) -> int:
        return self._config.get("golden_dead_cross", {}).get("default_fast_period", 5)

    def get_golden_cross_slow_period(self) -> int:
        return self._config.get("golden_dead_cross", {}).get("default_slow_period", 20)

    def get_golden_cross_confidence_multiplier(self) -> int:
        return self._config.get("golden_dead_cross", {}).get("confidence_multiplier", 100)

    def get_golden_cross_confidence_clip_max(self) -> int:
        return self._config.get("golden_dead_cross", {}).get("confidence_clip_max", 100)

    # サポート・レジスタンス設定
    def get_support_resistance_window(self) -> int:
        return self._config.get("support_resistance", {}).get("default_window", 20)

    def get_support_resistance_num_levels(self) -> int:
        return self._config.get("support_resistance", {}).get("default_num_levels", 3)

    def get_support_resistance_clustering_iterations(self) -> int:
        return self._config.get("support_resistance", {}).get("clustering_iterations", 10)

    def get_support_resistance_min_candidates_threshold(self) -> int:
        return self._config.get("support_resistance", {}).get("min_candidates_threshold", 0)

    # ブレイクアウト検出設定
    def get_breakout_lookback(self) -> int:
        return self._config.get("breakout_detection", {}).get("default_lookback", 20)

    def get_breakout_threshold(self) -> float:
        return self._config.get("breakout_detection", {}).get("default_threshold", 0.02)

    def get_breakout_volume_factor(self) -> float:
        return self._config.get("breakout_detection", {}).get("default_volume_factor", 1.5)

    def get_breakout_strength_multiplier(self) -> int:
        return self._config.get("breakout_detection", {}).get("strength_multiplier", 10)

    def get_breakout_volume_clip_max(self) -> int:
        return self._config.get("breakout_detection", {}).get("volume_clip_max", 2)

    def get_breakout_confidence_cap(self) -> int:
        return self._config.get("breakout_detection", {}).get("confidence_cap", 100)

    # トレンドライン検出設定
    def get_trend_line_window(self) -> int:
        return self._config.get("trend_line_detection", {}).get("default_window", 20)

    def get_trend_line_min_touches(self) -> int:
        return self._config.get("trend_line_detection", {}).get("default_min_touches", 3)

    def get_trend_line_ransac_residual_threshold(self) -> float:
        return self._config.get("trend_line_detection", {}).get("ransac_residual_threshold", 0.1)

    def get_trend_line_ransac_max_trials(self) -> int:
        return self._config.get("trend_line_detection", {}).get("ransac_max_trials", 100)

    def get_trend_line_ransac_min_samples(self) -> float:
        return self._config.get("trend_line_detection", {}).get("ransac_min_samples", 0.5)

    # detect_all_patterns設定
    def get_all_patterns_golden_cross_fast(self) -> int:
        return self._config.get("detect_all_patterns", {}).get("default_golden_cross_fast", 5)

    def get_all_patterns_golden_cross_slow(self) -> int:
        return self._config.get("detect_all_patterns", {}).get("default_golden_cross_slow", 20)

    def get_all_patterns_support_resistance_window(self) -> int:
        return self._config.get("detect_all_patterns", {}).get("default_support_resistance_window", 20)

    def get_all_patterns_breakout_lookback(self) -> int:
        return self._config.get("detect_all_patterns", {}).get("default_breakout_lookback", 20)

    def get_all_patterns_trend_window(self) -> int:
        return self._config.get("detect_all_patterns", {}).get("default_trend_window", 20)

    # 信頼度計算設定
    def get_confidence_weights(self) -> Dict[str, float]:
        return self._config.get("confidence_calculation", {}).get("weights", {
            "golden_cross": 0.3,
            "dead_cross": 0.3,
            "upward_breakout": 0.25,
            "downward_breakout": 0.25,
            "trend_r2": 0.2
        })

    def get_confidence_normalization(self) -> Dict[str, float]:
        return self._config.get("confidence_calculation", {}).get("normalization", {
            "min_confidence": 0.0,
            "max_confidence": 100.0
        })

    # エラー処理設定
    def should_return_empty_on_error(self) -> bool:
        return self._config.get("error_handling", {}).get("return_empty_on_error", True)

    def should_log_detailed_errors(self) -> bool:
        return self._config.get("error_handling", {}).get("log_detailed_errors", True)

    def should_raise_exceptions(self) -> bool:
        return self._config.get("error_handling", {}).get("raise_exceptions", False)

    # 設定更新・保存
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

    def get_config_dict(self) -> Dict[str, Any]:
        """設定辞書を取得（デバッグ用）"""
        return self._config.copy()


# グローバル設定インスタンス
_global_config = None


def get_patterns_config() -> PatternsConfig:
    """グローバル設定インスタンスを取得"""
    global _global_config
    if _global_config is None:
        _global_config = PatternsConfig()
    return _global_config


def set_patterns_config(config: PatternsConfig):
    """グローバル設定インスタンスを設定"""
    global _global_config
    _global_config = config
