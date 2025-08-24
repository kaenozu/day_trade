"""
設定管理クラス

シグナル生成ルールの設定ファイルの読み込みと管理
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class SignalRulesConfig:
    """シグナルルール設定管理クラス"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = self._resolve_config_path(config_path)
        self.config = self._load_config()

    def _resolve_config_path(self, config_path: Optional[str]) -> Path:
        """
        Issue #647対応: 設定ファイルパスの堅牢性改善
        環境変数、相対パス、複数の候補パスを考慮
        """
        if config_path is not None:
            resolved_path = Path(config_path)
            if resolved_path.is_absolute():
                return resolved_path
            # 相対パスの場合は現在の作業ディレクトリから解決
            return Path.cwd() / resolved_path

        # 環境変数から設定パスを取得
        env_config_path = os.getenv("SIGNAL_RULES_CONFIG_PATH")
        if env_config_path:
            return Path(env_config_path)

        # 複数の候補パスを試行
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent

        candidate_paths = [
            project_root / "config" / "signal_rules.json",
            Path.cwd() / "config" / "signal_rules.json",
            current_file.parent / "config" / "signal_rules.json",
            Path.home() / ".day_trade" / "signal_rules.json"
        ]

        # 存在する最初のパスを返す、存在しない場合は最初の候補
        for path in candidate_paths:
            if path.exists():
                return path

        return candidate_paths[0]  # デフォルトとして最初の候補を使用

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(self.config_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"シグナル設定ファイルが見つかりません: {self.config_path}")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"シグナル設定ファイル読み込みエラー: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Issue #648対応: デフォルト設定値の統合
        全ての設定項目を一箇所に集約してメンテナンス性を向上
        """
        return self._create_consolidated_default_config()

    def _create_consolidated_default_config(self) -> Dict[str, Any]:
        """統合されたデフォルト設定を作成"""
        # 基本設定
        base_config = {
            "default_buy_rules": [],
            "default_sell_rules": [],
        }

        # シグナル生成設定
        signal_settings = self._get_default_signal_settings()

        # ルール別設定
        rule_settings = {
            "rsi_default_thresholds": {"oversold": 30, "overbought": 70},
            "macd_default_settings": {"lookback_period": 2, "default_weight": 1.5},
            "pattern_breakout_settings": {"default_weight": 2.0},
            "golden_dead_cross_settings": {"default_weight": 2.0},
            "bollinger_band_settings": {"deviation_multiplier": 10.0},
            "volume_spike_settings": {
                "volume_factor": 2.0,
                "price_change_threshold": 0.02,
                "confidence_base": 50.0,
                "confidence_multiplier": 20.0,
            },
        }

        # 統合して返す
        return {
            **base_config,
            "signal_generation_settings": signal_settings,
            **rule_settings
        }

    def _get_default_signal_settings(self) -> Dict[str, Any]:
        """デフォルトシグナル設定を取得"""
        return {
            "min_data_period": 60,
            "min_data_for_generation": 20,
            "volume_calculation_period": 20,
            "trend_lookback_period": 20,
            "confidence_multipliers": {
                "rsi_oversold": 2.0,
                "rsi_overbought": 2.0,
                "macd_angle": 20.0,
                "bollinger_deviation": 10.0,
            },
            "strength_thresholds": {
                "strong": {"confidence": 70, "min_active_rules": 3},
                "medium": {"confidence": 40, "min_active_rules": 2},
            },
            "signal_freshness": {"warning_hours": 24, "stale_hours": 72},
            "validation_adjustments": {
                "high_volatility_threshold": 0.05,
            },
        }

    # 設定取得メソッド群
    def get_buy_rules_config(self) -> List[Dict[str, Any]]:
        """買いルール設定を取得"""
        return self.config.get("default_buy_rules", [])

    def get_sell_rules_config(self) -> List[Dict[str, Any]]:
        """売りルール設定を取得"""
        return self.config.get("default_sell_rules", [])

    def get_signal_settings(self) -> Dict[str, Any]:
        """シグナル生成設定を取得"""
        return self.config.get("signal_generation_settings", {})

    def get_confidence_multiplier(self, rule_type: str, default: float = 1.0) -> float:
        """信頼度乗数を取得"""
        multipliers = self.get_signal_settings().get("confidence_multipliers", {})
        return multipliers.get(rule_type, default)

    def get_strength_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """強度閾値を取得"""
        return self.get_signal_settings().get("strength_thresholds", {})

    def get_min_data_for_generation(self) -> int:
        """シグナル生成用最小データ数を取得"""
        return self.get_signal_settings().get("min_data_for_generation", 20)

    def get_volume_calculation_period(self) -> int:
        """出来高計算期間を取得"""
        return self.get_signal_settings().get("volume_calculation_period", 20)

    def get_trend_lookback_period(self) -> int:
        """トレンド判定期間を取得"""
        return self.get_signal_settings().get("trend_lookback_period", 20)

    def get_signal_freshness(self) -> Dict[str, int]:
        """シグナル新鮮度設定を取得"""
        return self.get_signal_settings().get(
            "signal_freshness", {"warning_hours": 24, "stale_hours": 72}
        )

    def get_high_volatility_threshold(self) -> float:
        """高ボラティリティ判定閾値を取得"""
        return (
            self.get_signal_settings()
            .get("validation_adjustments", {})
            .get("high_volatility_threshold", 0.05)
        )

    def get_volume_spike_settings(self) -> Dict[str, float]:
        """出来高急増設定を取得"""
        return self.config.get(
            "volume_spike_settings",
            {
                "volume_factor": 2.0,
                "price_change_threshold": 0.02,
                "confidence_base": 50.0,
                "confidence_multiplier": 20.0,
            },
        )

    def get_rsi_thresholds(self) -> Dict[str, float]:
        """RSI閾値を取得"""
        return self.config.get(
            "rsi_default_thresholds", {"oversold": 30, "overbought": 70}
        )

    def get_macd_settings(self) -> Dict[str, float]:
        """MACD設定を取得"""
        return self.config.get(
            "macd_default_settings", {"lookback_period": 2, "default_weight": 1.5}
        )

    def get_pattern_breakout_settings(self) -> Dict[str, float]:
        """パターンブレイクアウト設定を取得"""
        return self.config.get("pattern_breakout_settings", {"default_weight": 2.0})

    def get_cross_settings(self) -> Dict[str, float]:
        """ゴールデン/デッドクロス設定を取得"""
        return self.config.get("golden_dead_cross_settings", {"default_weight": 2.0})


# グローバル設定インスタンス（Issue #649対応）
_shared_config_instance: Optional[SignalRulesConfig] = None


def get_shared_config() -> SignalRulesConfig:
    """共有設定インスタンスを取得"""
    global _shared_config_instance
    if _shared_config_instance is None:
        _shared_config_instance = SignalRulesConfig()
    return _shared_config_instance