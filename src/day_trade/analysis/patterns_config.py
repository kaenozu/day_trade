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
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "patterns_config.json"

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

    # patterns.pyで使用されるメソッド群
    def get_golden_cross_fast_period(self) -> int:
        """ゴールデンクロス短期移動平均期間を取得"""
        return self._config.get("golden_dead_cross", {}).get("fast_period", 5)

    def get_golden_cross_slow_period(self) -> int:
        """ゴールデンクロス長期移動平均期間を取得"""
        return self._config.get("golden_dead_cross", {}).get("slow_period", 25)

    def get_all_patterns_golden_cross_fast(self) -> int:
        """全パターン検出用ゴールデンクロス短期期間を取得"""
        return self._config.get("detect_all_patterns", {}).get("default_golden_cross_fast", 5)

    def get_all_patterns_golden_cross_slow(self) -> int:
        """全パターン検出用ゴールデンクロス長期期間を取得"""
        return self._config.get("detect_all_patterns", {}).get("default_golden_cross_slow", 20)

    def get_all_patterns_support_resistance_window(self) -> int:
        """全パターン検出用サポートレジスタンス期間を取得"""
        return self._config.get("detect_all_patterns", {}).get(
            "default_support_resistance_window", 20
        )

    def get_all_patterns_breakout_lookback(self) -> int:
        """全パターン検出用ブレイクアウト期間を取得"""
        return self._config.get("detect_all_patterns", {}).get("default_breakout_lookback", 20)

    def get_all_patterns_trend_window(self) -> int:
        """全パターン検出用トレンド期間を取得"""
        return self._config.get("detect_all_patterns", {}).get("default_trend_window", 20)

    # ゴールデンクロス関連メソッド
    def get_golden_cross_confidence_multiplier(self) -> float:
        """ゴールデンクロス信頼度倍率を取得"""
        return float(self._config.get("golden_dead_cross", {}).get("confidence_multiplier", 100))

    def get_golden_cross_confidence_clip_max(self) -> int:
        """ゴールデンクロス信頼度上限を取得"""
        return self._config.get("golden_dead_cross", {}).get("confidence_clip_max", 100)

    # サポートレジスタンス関連メソッド
    def get_support_resistance_window(self) -> int:
        """サポートレジスタンスウィンドウサイズを取得"""
        return self._config.get("support_resistance", {}).get("window_size", 20)

    def get_support_resistance_num_levels(self) -> int:
        """サポートレジスタンスレベル数を取得"""
        return self._config.get("support_resistance", {}).get("default_num_levels", 3)

    def get_support_resistance_clustering_iterations(self) -> int:
        """サポートレジスタンスクラスタリング反復回数を取得"""
        return self._config.get("support_resistance", {}).get("clustering_iterations", 10)

    # ブレイクアウト関連メソッド
    def get_breakout_lookback(self) -> int:
        """ブレイクアウト期間を取得"""
        return self._config.get("breakout_detection", {}).get("default_lookback", 20)

    def get_breakout_threshold(self) -> float:
        """ブレイクアウト閾値を取得"""
        return float(self._config.get("breakout_detection", {}).get("default_threshold", 0.02))

    def get_breakout_volume_factor(self) -> float:
        """ブレイクアウトボリューム係数を取得"""
        return float(self._config.get("breakout_detection", {}).get("default_volume_factor", 1.5))

    def get_breakout_strength_multiplier(self) -> float:
        """ブレイクアウト強度倍率を取得"""
        return float(self._config.get("breakout_detection", {}).get("strength_multiplier", 10))

    def get_breakout_volume_clip_max(self) -> int:
        """ブレイクアウトボリューム上限を取得"""
        return self._config.get("breakout_detection", {}).get("volume_clip_max", 2)

    def get_breakout_confidence_cap(self) -> int:
        """ブレイクアウト信頼度上限を取得"""
        return self._config.get("breakout_detection", {}).get("confidence_cap", 100)

    # トレンドライン関連メソッド
    def get_trend_line_window(self) -> int:
        """トレンドラインウィンドウサイズを取得"""
        return self._config.get("trend_line_detection", {}).get("default_window", 20)

    def get_trend_line_min_touches(self) -> int:
        """トレンドライン最小接触数を取得"""
        return self._config.get("trend_line_detection", {}).get("default_min_touches", 3)

    def get_trend_line_ransac_residual_threshold(self) -> float:
        """トレンドラインRANSAC残差閾値を取得"""
        return 0.01

    def get_trend_line_ransac_max_trials(self) -> int:
        """トレンドラインRANSAC最大試行回数を取得"""
        return 100

    def get_trend_line_ransac_min_samples(self) -> int:
        """トレンドラインRANSAC最小サンプル数を取得"""
        return 2

    # エラーハンドリング関連メソッド
    def should_log_detailed_errors(self) -> bool:
        """詳細エラーログ出力フラグを取得"""
        return self._config.get("error_handling", {}).get("log_detailed_errors", True)

    def should_raise_exceptions(self) -> bool:
        """例外発生フラグを取得"""
        return self._config.get("error_handling", {}).get("raise_exceptions", False)

    def should_return_empty_on_error(self) -> bool:
        """エラー時空を返すフラグを取得"""
        return self._config.get("error_handling", {}).get("return_empty_on_error", True)

    # 信頼度計算関連メソッド
    def get_confidence_weights(self) -> Dict[str, float]:
        """信頼度重みを取得"""
        return self._config.get("confidence_calculation", {}).get(
            "weights",
            {
                "golden_cross": 0.3,
                "dead_cross": 0.3,
                "upward_breakout": 0.25,
                "downward_breakout": 0.25,
                "trend_r2": 0.2,
            },
        )

    def get_confidence_normalization(self) -> float:
        """信頼度正規化係数を取得"""
        return 100.0


# グローバル設定インスタンス
_global_config_class = None


def get_patterns_config_class() -> PatternsConfig:
    """グローバル設定クラスインスタンスを取得"""
    global _global_config_class
    if _global_config_class is None:
        _global_config_class = PatternsConfig()
    return _global_config_class
