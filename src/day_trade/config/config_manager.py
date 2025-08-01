"""
設定管理システム
全自動化機能の設定ファイル読み込みと管理
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, time

logger = logging.getLogger(__name__)


@dataclass
class WatchlistSymbol:
    """監視銘柄情報"""

    code: str
    name: str
    group: str
    priority: str


@dataclass
class MarketHours:
    """市場営業時間"""

    start: time
    end: time
    lunch_start: time
    lunch_end: time


@dataclass
class TechnicalIndicatorSettings:
    """テクニカル指標設定"""

    enabled: bool
    sma_periods: List[int]
    ema_periods: List[int]
    rsi_period: int
    macd_params: Dict[str, int]
    bollinger_params: Dict[str, Any]


@dataclass
class PatternRecognitionSettings:
    """パターン認識設定"""

    enabled: bool
    patterns: List[str]


@dataclass
class SignalGenerationSettings:
    """シグナル生成設定"""

    enabled: bool
    strategies: List[str]
    confidence_threshold: float


@dataclass
class EnsembleSettings:
    """アンサンブル戦略設定"""

    enabled: bool
    strategy_type: str
    voting_type: str
    performance_file_path: str
    strategy_weights: Dict[str, float]
    confidence_thresholds: Dict[str, float]
    meta_learning_enabled: bool
    adaptive_weights_enabled: bool


@dataclass
class AlertSettings:
    """アラート設定"""

    enabled: bool
    price_alerts: Dict[str, Any]
    volume_alerts: Dict[str, Any]
    technical_alerts: Dict[str, Any]
    notification_methods: List[str]


@dataclass
class BacktestSettings:
    """バックテスト設定"""

    enabled: bool
    period_days: int
    initial_capital: int
    position_size_percent: int
    max_positions: int
    stop_loss_percent: float
    take_profit_percent: float


@dataclass
class ReportSettings:
    """レポート設定"""

    enabled: bool
    output_directory: str
    formats: List[str]
    daily_report: Dict[str, Any]
    weekly_summary: Dict[str, Any]


@dataclass
class ExecutionSettings:
    """実行設定"""

    max_concurrent_requests: int
    timeout_seconds: int
    retry_attempts: int
    error_tolerance: str
    log_level: str


@dataclass
class DatabaseSettings:
    """データベース設定"""

    url: str
    backup_enabled: bool
    backup_interval_hours: int


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

            with open(self.config_path, "r", encoding="utf-8") as f:
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
        symbols = []
        for symbol_data in self.config["watchlist"]["symbols"]:
            symbols.append(
                WatchlistSymbol(
                    code=symbol_data["code"],
                    name=symbol_data["name"],
                    group=symbol_data["group"],
                    priority=symbol_data["priority"],
                )
            )
        return symbols

    def get_symbol_codes(self) -> List[str]:
        """銘柄コードのリストを取得"""
        return [symbol["code"] for symbol in self.config["watchlist"]["symbols"]]

    def get_market_hours(self) -> MarketHours:
        """市場営業時間を取得"""
        hours_config = self.config["watchlist"]["market_hours"]
        return MarketHours(
            start=time.fromisoformat(hours_config["start"]),
            end=time.fromisoformat(hours_config["end"]),
            lunch_start=time.fromisoformat(hours_config["lunch_start"]),
            lunch_end=time.fromisoformat(hours_config["lunch_end"]),
        )

    def get_technical_indicator_settings(self) -> TechnicalIndicatorSettings:
        """テクニカル指標設定を取得"""
        config = self.config["analysis"]["technical_indicators"]
        return TechnicalIndicatorSettings(
            enabled=config["enabled"],
            sma_periods=config["sma_periods"],
            ema_periods=config["ema_periods"],
            rsi_period=config["rsi_period"],
            macd_params=config["macd_params"],
            bollinger_params=config["bollinger_params"],
        )

    def get_pattern_recognition_settings(self) -> PatternRecognitionSettings:
        """パターン認識設定を取得"""
        config = self.config["analysis"]["pattern_recognition"]
        return PatternRecognitionSettings(
            enabled=config["enabled"], patterns=config["patterns"]
        )

    def get_signal_generation_settings(self) -> SignalGenerationSettings:
        """シグナル生成設定を取得"""
        config = self.config["analysis"]["signal_generation"]
        return SignalGenerationSettings(
            enabled=config["enabled"],
            strategies=config["strategies"],
            confidence_threshold=config["confidence_threshold"],
        )

    def get_ensemble_settings(self) -> EnsembleSettings:
        """アンサンブル戦略設定を取得"""
        config = self.config["analysis"].get("ensemble", {})

        # デフォルト値を設定
        default_weights = {
            "conservative_rsi": 0.2,
            "aggressive_momentum": 0.25,
            "trend_following": 0.25,
            "mean_reversion": 0.2,
            "default_integrated": 0.1,
        }

        default_thresholds = {
            "conservative": 60.0,
            "aggressive": 30.0,
            "balanced": 45.0,
            "adaptive": 40.0,
        }

        return EnsembleSettings(
            enabled=config.get("enabled", True),
            strategy_type=config.get("strategy_type", "balanced"),
            voting_type=config.get("voting_type", "soft"),
            performance_file_path=config.get(
                "performance_file_path", "data/ensemble_performance.json"
            ),
            strategy_weights=config.get("strategy_weights", default_weights),
            confidence_thresholds=config.get(
                "confidence_thresholds", default_thresholds
            ),
            meta_learning_enabled=config.get("meta_learning_enabled", True),
            adaptive_weights_enabled=config.get("adaptive_weights_enabled", True),
        )

    def get_alert_settings(self) -> AlertSettings:
        """アラート設定を取得"""
        config = self.config["alerts"]
        return AlertSettings(
            enabled=config["enabled"],
            price_alerts=config["price_alerts"],
            volume_alerts=config["volume_alerts"],
            technical_alerts=config["technical_alerts"],
            notification_methods=config["notification_methods"],
        )

    def get_backtest_settings(self) -> BacktestSettings:
        """バックテスト設定を取得"""
        config = self.config["backtest"]
        return BacktestSettings(
            enabled=config["enabled"],
            period_days=config["period_days"],
            initial_capital=config["initial_capital"],
            position_size_percent=config["position_size_percent"],
            max_positions=config["max_positions"],
            stop_loss_percent=config["stop_loss_percent"],
            take_profit_percent=config["take_profit_percent"],
        )

    def get_report_settings(self) -> ReportSettings:
        """レポート設定を取得"""
        config = self.config["reports"]
        return ReportSettings(
            enabled=config["enabled"],
            output_directory=config["output_directory"],
            formats=config["formats"],
            daily_report=config["daily_report"],
            weekly_summary=config["weekly_summary"],
        )

    def get_execution_settings(self) -> ExecutionSettings:
        """実行設定を取得"""
        config = self.config["execution"]
        return ExecutionSettings(
            max_concurrent_requests=config["max_concurrent_requests"],
            timeout_seconds=config["timeout_seconds"],
            retry_attempts=config["retry_attempts"],
            error_tolerance=config["error_tolerance"],
            log_level=config["log_level"],
        )

    def get_database_settings(self) -> DatabaseSettings:
        """データベース設定を取得"""
        config = self.config["database"]
        return DatabaseSettings(
            url=config["url"],
            backup_enabled=config["backup_enabled"],
            backup_interval_hours=config["backup_interval_hours"],
        )

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
        if market_hours.lunch_start <= current_time_only <= market_hours.lunch_end:
            return False

        return True

    def get_high_priority_symbols(self) -> List[str]:
        """高優先度銘柄のコードリストを取得"""
        symbols = self.get_watchlist_symbols()
        return [symbol.code for symbol in symbols if symbol.priority == "high"]

    def save_config(self):
        """設定ファイルを保存"""
        try:
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
        for symbol in self.config["watchlist"]["symbols"]:
            if symbol["code"] == symbol_code:
                symbol["priority"] = priority
                break
        else:
            raise ValueError(
                f"ウォッチリストに銘柄コード '{symbol_code}' が見つかりません。"
            )

    def add_symbol(self, code: str, name: str, group: str, priority: str = "medium"):
        """新しい銘柄を追加"""
        new_symbol = {"code": code, "name": name, "group": group, "priority": priority}
        self.config["watchlist"]["symbols"].append(new_symbol)
        logger.info(f"新しい銘柄を追加しました: {code} ({name})")

    def remove_symbol(self, symbol_code: str):
        """銘柄を削除"""
        original_count = len(self.config["watchlist"]["symbols"])
        self.config["watchlist"]["symbols"] = [
            symbol
            for symbol in self.config["watchlist"]["symbols"]
            if symbol["code"] != symbol_code
        ]

        if len(self.config["watchlist"]["symbols"]) == original_count:
            raise ValueError(
                f"ウォッチリストに銘柄コード '{symbol_code}' が見つからなかったため、削除できませんでした。"
            )

        logger.info(f"銘柄を削除しました: {symbol_code}")


# 使用例
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    try:
        # 設定管理クラスのテスト
        config_manager = ConfigManager()

        print("=== 設定情報 ===")
        print(f"監視銘柄数: {len(config_manager.get_symbol_codes())}")
        print(f"銘柄コード: {config_manager.get_symbol_codes()}")
        print(f"高優先度銘柄: {config_manager.get_high_priority_symbols()}")

        # 市場営業時間チェック
        print(f"現在市場オープン中: {config_manager.is_market_open()}")

        # 各種設定の取得テスト
        tech_settings = config_manager.get_technical_indicator_settings()
        print(f"テクニカル指標有効: {tech_settings.enabled}")

        alert_settings = config_manager.get_alert_settings()
        print(f"アラート有効: {alert_settings.enabled}")

        report_settings = config_manager.get_report_settings()
        print(f"レポート出力形式: {report_settings.formats}")

        print("設定管理システムのテストが完了しました")

    except Exception as e:
        print(f"エラー: {e}")
