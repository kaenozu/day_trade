"""
取引モード設定

自動売買機能を無効化し、システムを分析・情報提供・手動取引支援に特化する設定
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class TradingMode(Enum):
    """取引モード"""

    ANALYSIS_ONLY = "analysis_only"  # 分析のみ
    INFORMATION = "information"  # 情報提供
    MANUAL_SUPPORT = "manual_support"  # 手動取引支援
    SIMULATION = "simulation"  # シミュレーション（自動売買なし）
    DISABLED = "disabled"  # 自動取引完全無効


@dataclass
class TradingModeConfig:
    """取引モード設定"""

    # 基本モード
    current_mode: TradingMode = TradingMode.ANALYSIS_ONLY

    # 機能別有効/無効設定
    enable_market_data: bool = True  # 市場データ取得
    enable_analysis: bool = True  # 分析機能
    enable_signals: bool = True  # シグナル生成
    enable_backtesting: bool = True  # バックテスト
    enable_portfolio_tracking: bool = True  # ポートフォリオ追跡
    enable_alerts: bool = True  # アラート機能

    # 自動取引関連（全て無効）
    enable_automatic_trading: bool = False  # 自動取引
    enable_order_execution: bool = False  # 注文実行
    enable_position_management: bool = False  # ポジション管理
    enable_risk_management: bool = False  # リスク管理（自動）

    # 手動支援機能
    enable_trade_suggestions: bool = True  # 取引提案
    enable_risk_analysis: bool = True  # リスク分析
    enable_performance_tracking: bool = True  # パフォーマンス追跡
    enable_manual_logging: bool = True  # 手動ログ記録

    # 情報提供機能
    enable_news_integration: bool = True  # ニュース統合
    enable_earnings_calendar: bool = True  # 決算カレンダー
    enable_technical_indicators: bool = True  # テクニカル指標
    enable_fundamental_analysis: bool = True  # ファンダメンタル分析

    # セキュリティ設定
    require_manual_confirmation: bool = True  # 手動確認必須
    log_all_activities: bool = True  # 全活動ログ
    disable_order_api: bool = True  # 注文API無効化

    def __post_init__(self):
        """初期化後の設定確認"""
        if self.current_mode == TradingMode.ANALYSIS_ONLY:
            self._set_analysis_only_mode()
        elif self.current_mode == TradingMode.INFORMATION:
            self._set_information_mode()
        elif self.current_mode == TradingMode.MANUAL_SUPPORT:
            self._set_manual_support_mode()
        elif self.current_mode == TradingMode.DISABLED:
            self._set_disabled_mode()

        # 安全確認：自動取引関連は必ず無効
        self._enforce_safety_settings()

        logger.info(f"取引モード設定完了: {self.current_mode.value}")

    def _set_analysis_only_mode(self):
        """分析専用モード設定"""
        self.enable_market_data = True
        self.enable_analysis = True
        self.enable_signals = True
        self.enable_backtesting = True
        self.enable_alerts = True

        # 取引関連は全て無効
        self._disable_all_trading()

        logger.info("分析専用モードに設定されました")

    def _set_information_mode(self):
        """情報提供モード設定"""
        self.enable_market_data = True
        self.enable_analysis = True
        self.enable_news_integration = True
        self.enable_earnings_calendar = True
        self.enable_technical_indicators = True
        self.enable_fundamental_analysis = True

        # 取引関連は全て無効
        self._disable_all_trading()

        logger.info("情報提供モードに設定されました")

    def _set_manual_support_mode(self):
        """手動取引支援モード設定"""
        self.enable_market_data = True
        self.enable_analysis = True
        self.enable_signals = True
        self.enable_trade_suggestions = True
        self.enable_risk_analysis = True
        self.enable_performance_tracking = True
        self.enable_manual_logging = True

        # 自動取引は無効、手動支援は有効
        self._disable_all_trading()

        logger.info("手動取引支援モードに設定されました")

    def _set_disabled_mode(self):
        """完全無効モード設定"""
        # 基本的な分析機能のみ
        self.enable_market_data = True
        self.enable_analysis = False
        self.enable_signals = False
        self.enable_backtesting = False

        # その他全て無効
        self._disable_all_trading()
        self._disable_all_support()

        logger.warning("システム機能が大幅に制限されました")

    def _disable_all_trading(self):
        """全ての自動取引機能を無効化"""
        self.enable_automatic_trading = False
        self.enable_order_execution = False
        self.enable_position_management = False
        self.enable_risk_management = False
        self.disable_order_api = True
        self.require_manual_confirmation = True

    def _disable_all_support(self):
        """全ての支援機能を無効化"""
        self.enable_trade_suggestions = False
        self.enable_risk_analysis = False
        self.enable_performance_tracking = False
        self.enable_manual_logging = False
        self.enable_news_integration = False
        self.enable_earnings_calendar = False
        self.enable_technical_indicators = False
        self.enable_fundamental_analysis = False

    def _enforce_safety_settings(self):
        """安全設定の強制適用"""
        # 絶対に自動取引させない設定
        self.enable_automatic_trading = False
        self.enable_order_execution = False
        self.disable_order_api = True
        self.require_manual_confirmation = True
        self.log_all_activities = True

        logger.info("安全設定が強制適用されました")

    def is_trading_enabled(self) -> bool:
        """取引機能が有効かチェック"""
        return False  # 常にFalseを返す（自動取引完全無効）

    def is_analysis_enabled(self) -> bool:
        """分析機能が有効かチェック"""
        return self.enable_analysis

    def is_manual_support_enabled(self) -> bool:
        """手動支援機能が有効かチェック"""
        return self.enable_trade_suggestions or self.enable_risk_analysis

    def get_enabled_features(self) -> List[str]:
        """有効な機能リストを取得"""
        enabled = []

        if self.enable_market_data:
            enabled.append("市場データ取得")
        if self.enable_analysis:
            enabled.append("分析機能")
        if self.enable_signals:
            enabled.append("シグナル生成")
        if self.enable_backtesting:
            enabled.append("バックテスト")
        if self.enable_portfolio_tracking:
            enabled.append("ポートフォリオ追跡")
        if self.enable_trade_suggestions:
            enabled.append("取引提案")
        if self.enable_risk_analysis:
            enabled.append("リスク分析")
        if self.enable_news_integration:
            enabled.append("ニュース統合")
        if self.enable_technical_indicators:
            enabled.append("テクニカル指標")
        if self.enable_fundamental_analysis:
            enabled.append("ファンダメンタル分析")

        return enabled

    def get_disabled_features(self) -> List[str]:
        """無効な機能リストを取得"""
        return [
            "自動取引",
            "注文実行",
            "ポジション管理（自動）",
            "リスク管理（自動）",
            "注文API",
        ]

    def validate_configuration(self) -> Dict[str, bool]:
        """設定の妥当性検証"""
        validation_result = {
            "safety_enforced": not self.enable_automatic_trading,
            "api_disabled": self.disable_order_api,
            "manual_confirmation": self.require_manual_confirmation,
            "logging_enabled": self.log_all_activities,
            "basic_functions_available": self.enable_market_data and self.enable_analysis,
        }

        all_valid = all(validation_result.values())

        if all_valid:
            logger.info("設定の妥当性検証: 合格")
        else:
            logger.error(f"設定の妥当性検証: 不合格 - {validation_result}")

        return validation_result


# デフォルト設定（分析専用モード）
DEFAULT_CONFIG = TradingModeConfig(
    current_mode=TradingMode.ANALYSIS_ONLY,
    enable_market_data=True,
    enable_analysis=True,
    enable_signals=True,
    enable_backtesting=True,
    enable_portfolio_tracking=True,
    enable_alerts=True,
    enable_trade_suggestions=True,
    enable_risk_analysis=True,
    enable_news_integration=True,
    enable_technical_indicators=True,
    enable_fundamental_analysis=True,
    # 自動取引関連は全て無効
    enable_automatic_trading=False,
    enable_order_execution=False,
    enable_position_management=False,
    enable_risk_management=False,
    disable_order_api=True,
    require_manual_confirmation=True,
    log_all_activities=True,
)


def get_current_trading_config() -> TradingModeConfig:
    """現在の取引設定を取得"""
    return DEFAULT_CONFIG


def is_safe_mode() -> bool:
    """セーフモード（自動取引無効）かチェック"""
    config = get_current_trading_config()
    return (
        not config.enable_automatic_trading
        and not config.enable_order_execution
        and config.disable_order_api
        and config.require_manual_confirmation
    )


def log_current_configuration():
    """現在の設定をログ出力"""
    config = get_current_trading_config()

    logger.info("=" * 60)
    logger.info("取引システム設定状況")
    logger.info("=" * 60)
    logger.info(f"現在のモード: {config.current_mode.value}")
    logger.info("有効な機能:")
    for feature in config.get_enabled_features():
        logger.info(f"  + {feature}")

    logger.info("無効な機能:")
    for feature in config.get_disabled_features():
        logger.info(f"  - {feature}")

    logger.info("=" * 60)
    logger.info(f"セーフモード: {'有効' if is_safe_mode() else '無効'}")
    logger.info("=" * 60)


# 起動時に設定を確認
if __name__ == "__main__":
    log_current_configuration()
