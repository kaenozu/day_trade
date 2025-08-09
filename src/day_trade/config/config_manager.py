"""
設定管理システム
全自動化機能の設定ファイル読み込みと管理
"""

import json
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="config_manager")


class WatchlistSymbol(BaseModel):
    """監視銘柄情報"""

    code: str
    name: str
    group: str
    priority: str


class MarketHours(BaseModel):
    """市場営業時間"""

    start: time
    end: time
    lunch_start: time
    lunch_end: time


class TechnicalIndicatorSettings(BaseModel):
    """テクニカル指標設定"""

    enabled: bool
    sma_periods: List[int]
    ema_periods: List[int]
    rsi_period: int
    macd_params: Dict[str, int]
    bollinger_params: Dict[str, Any]


class PatternRecognitionSettings(BaseModel):
    """パターン認識設定"""

    enabled: bool
    patterns: List[str]


class SignalGenerationSettings(BaseModel):
    """シグナル生成設定"""

    enabled: bool
    strategies: List[str]
    confidence_threshold: float


class EnsembleSettings(BaseModel):
    """アンサンブル戦略設定"""

    enabled: bool = Field(default=True, description="アンサンブル戦略の有効/無効")
    strategy_type: str = Field(
        default="balanced",
        description="戦略タイプ（conservative, aggressive, balanced, adaptive）",
    )
    voting_type: str = Field(
        default="soft", description="投票タイプ（soft, hard, weighted）"
    )
    performance_file_path: str = Field(
        default="data/ensemble_performance.json",
        description="パフォーマンス履歴ファイルのパス",
    )
    strategy_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "conservative_rsi": 0.2,
            "aggressive_momentum": 0.25,
            "trend_following": 0.25,
            "mean_reversion": 0.2,
            "default_integrated": 0.1,
        },
        description="各戦略の重み",
    )
    confidence_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "conservative": 60.0,
            "aggressive": 30.0,
            "balanced": 45.0,
            "adaptive": 70.0,  # ensemble.pyのデフォルト値（ADAPTIVE時）に合わせる
        },
        description="各戦略タイプの信頼度閾値",
    )
    meta_learning_enabled: bool = Field(default=True, description="メタ学習の有効/無効")
    adaptive_weights_enabled: bool = Field(
        default=True, description="適応型重み調整の有効/無効"
    )

    @field_validator("strategy_type")
    @classmethod
    def validate_strategy_type(cls, v):
        """戦略タイプのバリデーション"""
        valid_types = ["conservative", "aggressive", "balanced", "adaptive"]
        if v not in valid_types:
            raise ValueError(f"strategy_type must be one of: {valid_types}")
        return v

    @field_validator("voting_type")
    @classmethod
    def validate_voting_type(cls, v):
        """投票タイプのバリデーション"""
        valid_types = ["soft", "hard", "weighted"]
        if v not in valid_types:
            raise ValueError(f"voting_type must be one of: {valid_types}")
        return v

    @field_validator("strategy_weights")
    @classmethod
    def validate_strategy_weights(cls, v):
        """戦略重みのバリデーション"""
        # 重みの合計チェック
        total_weight = sum(v.values())
        if not (0.95 <= total_weight <= 1.05):  # 誤差許容範囲
            raise ValueError(
                f"Sum of strategy weights must be close to 1.0, got {total_weight}"
            )

        # 個別重みの範囲チェック
        for strategy, weight in v.items():
            if not (0.0 <= weight <= 1.0):
                raise ValueError(
                    f"Weight for {strategy} must be between 0.0 and 1.0, got {weight}"
                )

        return v

    @field_validator("confidence_thresholds")
    @classmethod
    def validate_confidence_thresholds(cls, v):
        """信頼度閾値のバリデーション"""
        for threshold_type, threshold_value in v.items():
            if not (0.0 <= threshold_value <= 100.0):
                raise ValueError(
                    f"Confidence threshold for {threshold_type} must be between 0.0 and 100.0, "
                    f"got {threshold_value}"
                )
        return v


class AlertSettings(BaseModel):
    """アラート設定"""

    enabled: bool
    price_alerts: Dict[str, Any]
    volume_alerts: Dict[str, Any]
    technical_alerts: Dict[str, Any]
    notification_methods: List[str]


class BacktestSettings(BaseModel):
    """バックテスト設定"""

    enabled: bool
    period_days: int
    initial_capital: int
    position_size_percent: int
    max_positions: int
    stop_loss_percent: float
    take_profit_percent: float


class ReportSettings(BaseModel):
    """レポート設定"""

    enabled: bool
    output_directory: str
    formats: List[str]
    daily_report: Dict[str, Any]
    weekly_summary: Dict[str, Any]


class ExecutionSettings(BaseModel):
    """実行設定"""

    max_concurrent_requests: int
    timeout_seconds: int
    retry_attempts: int
    error_tolerance: str
    log_level: str


class DatabaseSettings(BaseModel):
    """データベース設定"""

    url: str
    backup_enabled: bool
    backup_interval_hours: int


class AutoOptimizerSettings(BaseModel):
    """全自動最適化設定"""

    enabled: bool = Field(True)
    default_max_symbols: int = Field(5)
    default_optimization_depth: str = Field("balanced")
    data_quality_threshold: float = Field(0.7)
    performance_threshold: float = Field(0.05)
    risk_tolerance: float = Field(0.7)
    fallback_symbols: List[str] = Field(default_factory=lambda: ["7203", "8306", "9984"])
    screening_strategies: List[str] = Field(default_factory=lambda: ["default", "momentum"])
    backtest_period_months: int = Field(6)
    ml_training_enabled: bool = Field(False)


class ConfigManager:
    """設定管理クラス"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent.parent.parent / "config" / "settings.json"
            )

        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """設定ファイルを読み込み"""
        try:
            if not self.config_path.exists():
                logger.error(
                    f"設定ファイルが見つかりません。パス: '{self.config_path}'。指定されたパスが正しいか、ファイルが存在するか確認してください。"
                )
                raise FileNotFoundError(f"Config file not found: {self.config_path}")

            with open(self.config_path, encoding="utf-8") as f:
                self.config = json.load(f)

            logger.info(f"設定ファイルを読み込みました: {self.config_path}")
            self._validate_config()

        except Exception as e:
            logger.error(
                f"設定ファイルの読み込み中にエラーが発生しました。ファイル形式が正しいか、破損していないか確認してください。詳細: {e}"
            )
            raise

    def _validate_config(self):
        """設定の妥当性チェック"""
        required_sections = ["watchlist", "analysis", "alerts", "reports", "execution"]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(
                    f"必須設定セクション '{section}' が設定ファイルに見つかりません。設定ファイルが完全であるか確認してください。"
                )

        # 監視銘柄の妥当性チェック
        if not self.config["watchlist"]["symbols"]:
            raise ValueError(
                "ウォッチリストに監視銘柄が一つも設定されていません。少なくとも一つ銘柄を追加してください。"
            )

        logger.info("設定の妥当性チェックが完了しました")

    def get_watchlist_symbols(self) -> List[WatchlistSymbol]:
        """監視銘柄リストを取得"""
        # Pydanticモデルのバリデーションを利用
        return [WatchlistSymbol(**symbol_data) for symbol_data in self.config["watchlist"]["symbols"]]

    def get_symbol_codes(self) -> List[str]:
        """銘柄コードのリストを取得"""
        return [symbol["code"] for symbol in self.config["watchlist"]["symbols"]]

    def get_market_hours(self) -> MarketHours:
        """市場営業時間を取得"""
        # Pydanticモデルのバリデーションを利用
        return MarketHours(**self.config["watchlist"]["market_hours"])

    def get_technical_indicator_settings(self) -> TechnicalIndicatorSettings:
        """テクニカル指標設定を取得"""
        # Pydanticモデルのバリデーションを利用
        return TechnicalIndicatorSettings(**self.config["analysis"]["technical_indicators"])

    def get_pattern_recognition_settings(self) -> PatternRecognitionSettings:
        """パターン認識設定を取得"""
        # Pydanticモデルのバリデーションを利用
        return PatternRecognitionSettings(**self.config["analysis"]["pattern_recognition"])

    def get_signal_generation_settings(self) -> SignalGenerationSettings:
        """シグナル生成設定を取得"""
        # Pydanticモデルのバリデーションを利用
        return SignalGenerationSettings(**self.config["analysis"]["signal_generation"])

    def get_ensemble_settings(self) -> EnsembleSettings:
        """アンサンブル戦略設定を取得"""
        config_data = self.config["analysis"].get("ensemble", {})
        # Pydanticモデルを使用してバリデーション付きで作成
        return EnsembleSettings(**config_data)

    def get_alert_settings(self) -> AlertSettings:
        """アラート設定を取得"""
        # Pydanticモデルのバリデーションを利用
        return AlertSettings(**self.config["alerts"])

    def get_backtest_settings(self) -> BacktestSettings:
        """バックテスト設定を取得"""
        # Pydanticモデルのバリデーションを利用
        return BacktestSettings(**self.config["backtest"])

    def get_report_settings(self) -> ReportSettings:
        """レポート設定を取得"""
        # Pydanticモデルのバリデーションを利用
        return ReportSettings(**self.config["reports"])

    def get_execution_settings(self) -> ExecutionSettings:
        """実行設定を取得"""
        # Pydanticモデルのバリデーションを利用
        return ExecutionSettings(**self.config["execution"])

    def get_database_settings(self) -> DatabaseSettings:
        """データベース設定を取得"""
        # Pydanticモデルのバリデーションを利用
        return DatabaseSettings(**self.config["database"])

    def get_auto_optimizer_settings(self) -> AutoOptimizerSettings:
        """全自動最適化設定を取得"""
        config_data = self.config.get("auto_optimizer", {})
        # Pydanticモデルのバリデーションを利用
        return AutoOptimizerSettings(**config_data)

    def is_market_open(self, current_time: Optional[datetime] = None) -> bool:
        """市場営業時間かどうかを判定"""
        if current_time is None:
            current_time = datetime.now()

        current_time_only = current_time.time()
        market_hours = self.get_market_hours()

        # 営業時間内かチェック
        if (
            current_time_only < market_hours.start
            or current_time_only > market_hours.end
        ):
            return False

        # 昼休み時間かチェック
        return (
            not market_hours.lunch_start <= current_time_only <= market_hours.lunch_end
        )

    def get_high_priority_symbols(self) -> List[str]:
        """高優先度銘柄のコードリストを取得"""
        symbols = self.get_watchlist_symbols()
        return [symbol.code for symbol in symbols if symbol.priority == "high"]

    def save_config(self):
        """設定ファイルを保存"""
        try:
            # 各Pydanticモデルの現在の状態をself.configに反映
            self.config["watchlist"]["symbols"] = [
                s.model_dump() for s in self.get_watchlist_symbols()
            ]
            self.config["watchlist"]["market_hours"] = self.get_market_hours().model_dump()
            self.config["analysis"]["technical_indicators"] = self.get_technical_indicator_settings().model_dump()
            self.config["analysis"]["pattern_recognition"] = self.get_pattern_recognition_settings().model_dump()
            self.config["analysis"]["signal_generation"] = self.get_signal_generation_settings().model_dump()
            self.config["analysis"]["ensemble"] = self.get_ensemble_settings().model_dump()
            self.config["alerts"] = self.get_alert_settings().model_dump()
            self.config["backtest"] = self.get_backtest_settings().model_dump()
            self.config["reports"] = self.get_report_settings().model_dump()
            self.config["execution"] = self.get_execution_settings().model_dump()
            self.config["database"] = self.get_database_settings().model_dump()
            self.config["auto_optimizer"] = self.get_auto_optimizer_settings().model_dump()

            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            logger.info(f"設定ファイルを保存しました: {self.config_path}")
        except Exception as e:
            logger.error(
                f"設定ファイルの保存中にエラーが発生しました。ファイルへの書き込み権限を確認してください。詳細: {e}"
            )
            raise

    def update_symbol_priority(self, symbol_code: str, priority: str):
        """銘柄の優先度を更新"""
        watchlist_symbols = self.get_watchlist_symbols()  # Pydanticモデルのリストを取得
        found = False
        for symbol in watchlist_symbols:
            if symbol.code == symbol_code:
                symbol.priority = priority
                found = True
                break
        if not found:
            raise ValueError(f"ウォッチリストに銘柄コード '{symbol_code}' が見つかりません。")
        # 更新したリストをConfigManagerの内部辞書に反映（save_configでまとめて行うためここでは不要）
        # self.config["watchlist"]["symbols"] = [s.model_dump() for s in watchlist_symbols]

    def add_symbol(self, code: str, name: str, group: str, priority: str = "medium"):
        """新しい銘柄を追加"""
        new_symbol = WatchlistSymbol(code=code, name=name, group=group, priority=priority)
        # Pydanticモデルのリストに追加（save_configでまとめて反映）
        watchlist_symbols = self.get_watchlist_symbols()
        watchlist_symbols.append(new_symbol)
        # self.config["watchlist"]["symbols"] = [s.model_dump() for s in watchlist_symbols]
        logger.info(f"新しい銘柄を追加しました: {code} ({name})")

    def remove_symbol(self, symbol_code: str):
        """銘柄を削除"""
        watchlist_symbols = self.get_watchlist_symbols()  # Pydanticモデルのリストを取得
        original_count = len(watchlist_symbols)
        # フィルターして新しいリストを作成
        updated_symbols = [s for s in watchlist_symbols if s.code != symbol_code]

        if len(updated_symbols) == original_count:
            raise ValueError(
                f"ウォッチリストに銘柄コード '{symbol_code}' が見つからなかったため、削除できませんでした。"
            )
        # 更新したリストをConfigManagerの内部辞書に反映（save_configでまとめて行うためここでは不要）
        # self.config["watchlist"]["symbols"] = [s.model_dump() for s in updated_symbols]
        logger.info(f"銘柄を削除しました: {symbol_code}")


# 使用例
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    try:
        # 設定管理クラスのテスト
        config_manager = ConfigManager()

        # 各種設定の取得テスト
        tech_settings = config_manager.get_technical_indicator_settings()
        alert_settings = config_manager.get_alert_settings()
        report_settings = config_manager.get_report_settings()

        config_info = {
            "watchlist_symbols_count": len(config_manager.get_symbol_codes()),
            "symbol_codes": config_manager.get_symbol_codes(),
            "high_priority_symbols": config_manager.get_high_priority_symbols(),
            "market_open": config_manager.is_market_open(),
            "technical_indicators_enabled": tech_settings.enabled,
            "alerts_enabled": alert_settings.enabled,
            "report_formats": report_settings.formats,
        }

        logger.info("設定管理システムテスト完了", **config_info)

        # 銘柄の追加・更新・削除のテスト
        logger.info("
--- 銘柄操作テスト ---")
        initial_symbols = config_manager.get_symbol_codes()
        logger.info(f"初期銘柄数: {len(initial_symbols)}")

        test_code = "TEST"
        test_name = "テスト銘柄"
        test_group = "TestGroup"

        try:
            config_manager.add_symbol(test_code, test_name, test_group, "high")
            config_manager.save_config()
            updated_symbols = config_manager.get_symbol_codes()
            logger.info(f"追加後の銘柄数: {len(updated_symbols)}")
            logger.info(f"追加された銘柄: {test_code}")

            config_manager.update_symbol_priority(test_code, "low")
            config_manager.save_config()
            updated_symbol = next(s for s in config_manager.get_watchlist_symbols() if s.code == test_code)
            logger.info(f"{test_code}の優先度を更新: {updated_symbol.priority}")

            config_manager.remove_symbol(test_code)
            config_manager.save_config()
            final_symbols = config_manager.get_symbol_codes()
            logger.info(f"削除後の銘柄数: {len(final_symbols)}")
            assert test_code not in final_symbols

            logger.info("銘柄操作テスト成功")

        except Exception as e:
            logger.error(f"銘柄操作テストエラー: {e}")


    except Exception as e:
        logger.error(
            "設定管理システムテストエラー", error=str(e), error_type=type(e).__name__
        )
