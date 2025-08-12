"""
コンプライアンスチェック

法的規制・取引制限・リスク管理ルールの遵守確認機能
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ...utils.logging_config import get_context_logger
from ..core.types import Trade, TradeType

logger = get_context_logger(__name__)


class ComplianceLevel(Enum):
    """コンプライアンス違反レベル"""

    CRITICAL = "重大"
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"
    INFO = "情報"


class ComplianceRule:
    """コンプライアンスルール"""

    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str,
        level: ComplianceLevel,
        enabled: bool = True,
    ):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.level = level
        self.enabled = enabled


class ComplianceChecker:
    """
    コンプライアンスチェッククラス

    取引に関する法的規制・社内規定・リスク管理の遵守確認機能を提供
    """

    def __init__(
        self,
        max_single_trade_value: Decimal = Decimal("10000000"),  # 1千万円
        max_daily_trade_value: Decimal = Decimal("50000000"),  # 5千万円
        max_position_concentration: Decimal = Decimal("20.0"),  # 20%
        insider_trading_blackout_days: int = 7,
    ):
        """
        初期化

        Args:
            max_single_trade_value: 単一取引最大額
            max_daily_trade_value: 1日最大取引額
            max_position_concentration: 最大ポジション集中度(%)
            insider_trading_blackout_days: インサイダー取引ブラックアウト期間
        """
        self.max_single_trade_value = max_single_trade_value
        self.max_daily_trade_value = max_daily_trade_value
        self.max_position_concentration = max_position_concentration
        self.insider_trading_blackout_days = insider_trading_blackout_days

        self.compliance_rules = self._initialize_compliance_rules()
        self.blacklisted_symbols: Set[str] = set()
        self.restricted_periods: List[Dict] = []

        logger.info("コンプライアンスチェッカー初期化完了")

    def _initialize_compliance_rules(self) -> List[ComplianceRule]:
        """コンプライアンスルール初期化"""
        return [
            ComplianceRule(
                "single_trade_limit",
                "単一取引限度額",
                "単一取引の上限金額チェック",
                ComplianceLevel.HIGH,
            ),
            ComplianceRule(
                "daily_trade_limit",
                "日次取引限度額",
                "1日の取引上限金額チェック",
                ComplianceLevel.HIGH,
            ),
            ComplianceRule(
                "position_concentration",
                "ポジション集中度",
                "特定銘柄への集中投資制限",
                ComplianceLevel.MEDIUM,
            ),
            ComplianceRule(
                "blacklist_check",
                "ブラックリスト銘柄",
                "取引禁止銘柄チェック",
                ComplianceLevel.CRITICAL,
            ),
            ComplianceRule(
                "trading_hours",
                "取引時間",
                "市場取引時間内チェック",
                ComplianceLevel.MEDIUM,
            ),
            ComplianceRule(
                "wash_sale",
                "ウォッシュセール",
                "短期売買往復取引チェック",
                ComplianceLevel.LOW,
            ),
            ComplianceRule(
                "insider_trading",
                "インサイダー取引",
                "決算発表前後の取引制限",
                ComplianceLevel.CRITICAL,
            ),
            ComplianceRule(
                "settlement_period",
                "決済期間",
                "決済可能期間内チェック",
                ComplianceLevel.HIGH,
            ),
        ]

    def check_trade_compliance(
        self,
        trade: Trade,
        existing_trades: List[Trade] = None,
        portfolio_value: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """
        取引コンプライアンスチェック

        Args:
            trade: チェック対象取引
            existing_trades: 既存取引履歴
            portfolio_value: 現在のポートフォリオ価値

        Returns:
            コンプライアンスチェック結果
        """
        if existing_trades is None:
            existing_trades = []

        compliance_result = {
            "trade_id": trade.id,
            "overall_compliant": True,
            "violations": [],
            "warnings": [],
            "info": [],
            "rule_results": {},
        }

        try:
            # 各コンプライアンスルールをチェック
            checks = {
                "single_trade_limit": self._check_single_trade_limit(trade),
                "daily_trade_limit": self._check_daily_trade_limit(trade, existing_trades),
                "position_concentration": self._check_position_concentration(
                    trade, existing_trades, portfolio_value
                ),
                "blacklist_check": self._check_blacklist(trade),
                "trading_hours": self._check_trading_hours(trade),
                "wash_sale": self._check_wash_sale(trade, existing_trades),
                "insider_trading": self._check_insider_trading(trade),
                "settlement_period": self._check_settlement_period(trade),
            }

            # 結果分類
            for rule_id, (is_compliant, message, level) in checks.items():
                rule = next((r for r in self.compliance_rules if r.rule_id == rule_id), None)

                if not rule or not rule.enabled:
                    continue

                compliance_result["rule_results"][rule_id] = {
                    "compliant": is_compliant,
                    "message": message,
                    "level": level.value,
                }

                if not is_compliant:
                    violation_info = {
                        "rule_id": rule_id,
                        "rule_name": rule.name,
                        "message": message,
                        "level": level.value,
                    }

                    if level == ComplianceLevel.CRITICAL:
                        compliance_result["violations"].append(violation_info)
                        compliance_result["overall_compliant"] = False
                    elif level == ComplianceLevel.HIGH:
                        compliance_result["violations"].append(violation_info)
                        compliance_result["overall_compliant"] = False
                    elif level == ComplianceLevel.MEDIUM:
                        compliance_result["warnings"].append(violation_info)
                    else:
                        compliance_result["info"].append(violation_info)

            # ログ出力
            if not compliance_result["overall_compliant"]:
                logger.warning(
                    f"コンプライアンス違反: {trade.id} - "
                    f"{len(compliance_result['violations'])}件違反"
                )
            elif compliance_result["warnings"]:
                logger.info(
                    f"コンプライアンス注意: {trade.id} - "
                    f"{len(compliance_result['warnings'])}件警告"
                )

            return compliance_result

        except Exception as e:
            logger.error(f"コンプライアンスチェックエラー: {trade.id} - {e}")
            compliance_result["overall_compliant"] = False
            compliance_result["violations"].append(
                {
                    "rule_id": "system_error",
                    "message": f"システムエラー: {str(e)}",
                    "level": ComplianceLevel.CRITICAL.value,
                }
            )
            return compliance_result

    def _check_single_trade_limit(self, trade: Trade) -> tuple:
        """単一取引限度額チェック"""
        trade_value = trade.price * Decimal(trade.quantity)

        if trade_value > self.max_single_trade_value:
            return (
                False,
                f"単一取引限度額超過: {trade_value:,.0f}円 > {self.max_single_trade_value:,.0f}円",
                ComplianceLevel.HIGH,
            )

        return (True, "単一取引限度額OK", ComplianceLevel.HIGH)

    def _check_daily_trade_limit(self, trade: Trade, existing_trades: List[Trade]) -> tuple:
        """日次取引限度額チェック"""
        trade_date = trade.timestamp.date()

        # 同日の取引を集計
        daily_trades = [t for t in existing_trades if t.timestamp.date() == trade_date]

        daily_total = sum(t.price * Decimal(t.quantity) for t in daily_trades)

        # 今回の取引を追加
        trade_value = trade.price * Decimal(trade.quantity)
        daily_total += trade_value

        if daily_total > self.max_daily_trade_value:
            return (
                False,
                f"日次取引限度額超過: {daily_total:,.0f}円 > {self.max_daily_trade_value:,.0f}円",
                ComplianceLevel.HIGH,
            )

        return (True, "日次取引限度額OK", ComplianceLevel.HIGH)

    def _check_position_concentration(
        self,
        trade: Trade,
        existing_trades: List[Trade],
        portfolio_value: Optional[Decimal],
    ) -> tuple:
        """ポジション集中度チェック"""
        if not portfolio_value or portfolio_value <= 0:
            return (
                True,
                "ポートフォリオ価値未設定のため集中度チェックスキップ",
                ComplianceLevel.MEDIUM,
            )

        # 銘柄別ポジション計算
        symbol_positions = {}

        for t in existing_trades:
            symbol = t.symbol
            if symbol not in symbol_positions:
                symbol_positions[symbol] = Decimal("0")

            if t.trade_type == TradeType.BUY:
                symbol_positions[symbol] += t.price * Decimal(t.quantity)
            elif t.trade_type == TradeType.SELL:
                symbol_positions[symbol] -= t.price * Decimal(t.quantity)

        # 今回の取引を追加
        symbol = trade.symbol
        if symbol not in symbol_positions:
            symbol_positions[symbol] = Decimal("0")

        if trade.trade_type == TradeType.BUY:
            symbol_positions[symbol] += trade.price * Decimal(trade.quantity)
        elif trade.trade_type == TradeType.SELL:
            symbol_positions[symbol] -= trade.price * Decimal(trade.quantity)

        # 集中度計算
        position_value = symbol_positions.get(symbol, Decimal("0"))
        concentration_percentage = (position_value / portfolio_value) * 100

        if concentration_percentage > self.max_position_concentration:
            return (
                False,
                f"ポジション集中度超過: {symbol} {concentration_percentage:.1f}% > {self.max_position_concentration}%",
                ComplianceLevel.MEDIUM,
            )

        return (True, "ポジション集中度OK", ComplianceLevel.MEDIUM)

    def _check_blacklist(self, trade: Trade) -> tuple:
        """ブラックリスト銘柄チェック"""
        if trade.symbol in self.blacklisted_symbols:
            return (
                False,
                f"ブラックリスト銘柄: {trade.symbol}",
                ComplianceLevel.CRITICAL,
            )

        return (True, "ブラックリストチェックOK", ComplianceLevel.CRITICAL)

    def _check_trading_hours(self, trade: Trade) -> tuple:
        """取引時間チェック"""
        trade_time = trade.timestamp.time()

        # 日本株式市場の取引時間（9:00-11:30, 12:30-15:00）
        morning_start = datetime.strptime("09:00", "%H:%M").time()
        morning_end = datetime.strptime("11:30", "%H:%M").time()
        afternoon_start = datetime.strptime("12:30", "%H:%M").time()
        afternoon_end = datetime.strptime("15:00", "%H:%M").time()

        is_trading_hours = (morning_start <= trade_time <= morning_end) or (
            afternoon_start <= trade_time <= afternoon_end
        )

        if not is_trading_hours:
            return (
                False,
                f"取引時間外: {trade_time.strftime('%H:%M')}",
                ComplianceLevel.MEDIUM,
            )

        return (True, "取引時間OK", ComplianceLevel.MEDIUM)

    def _check_wash_sale(self, trade: Trade, existing_trades: List[Trade]) -> tuple:
        """ウォッシュセール（短期売買往復）チェック"""
        if trade.trade_type == TradeType.SELL:
            # 売却から30日以内の同銘柄買い戻しチェック
            cutoff_date = trade.timestamp - timedelta(days=30)

            recent_buys = [
                t
                for t in existing_trades
                if (
                    t.symbol == trade.symbol
                    and t.trade_type == TradeType.BUY
                    and t.timestamp >= cutoff_date
                    and t.timestamp < trade.timestamp
                )
            ]

            if recent_buys:
                return (
                    False,
                    f"ウォッシュセール懸念: {trade.symbol} 30日以内に{len(recent_buys)}回購入",
                    ComplianceLevel.LOW,
                )

        elif trade.trade_type == TradeType.BUY:
            # 購入から30日以内の同銘柄売却後の買い戻しチェック
            cutoff_date = trade.timestamp - timedelta(days=30)

            recent_sells = [
                t
                for t in existing_trades
                if (
                    t.symbol == trade.symbol
                    and t.trade_type == TradeType.SELL
                    and t.timestamp >= cutoff_date
                    and t.timestamp < trade.timestamp
                )
            ]

            if recent_sells:
                return (
                    False,
                    f"ウォッシュセール懸念: {trade.symbol} 30日以内に{len(recent_sells)}回売却後の買い戻し",
                    ComplianceLevel.LOW,
                )

        return (True, "ウォッシュセールチェックOK", ComplianceLevel.LOW)

    def _check_insider_trading(self, trade: Trade) -> tuple:
        """インサイダー取引チェック"""
        # 決算発表期間の制限（簡易版）
        # 実際の実装では決算カレンダーAPIを使用

        trade_date = trade.timestamp.date()

        # 制限期間チェック
        for restriction in self.restricted_periods:
            if (
                restriction["symbol"] == trade.symbol
                and restriction["start_date"] <= trade_date <= restriction["end_date"]
            ):
                return (
                    False,
                    f"制限期間内取引: {trade.symbol} ({restriction['reason']})",
                    ComplianceLevel.CRITICAL,
                )

        return (True, "インサイダー取引チェックOK", ComplianceLevel.CRITICAL)

    def _check_settlement_period(self, trade: Trade) -> tuple:
        """決済期間チェック"""
        # T+2決済ルール（取引日から2営業日以内）
        current_date = datetime.now().date()
        trade_date = trade.timestamp.date()

        # 決済期限計算（簡易版：土日を除く）
        settlement_date = trade_date
        business_days = 0

        while business_days < 2:
            settlement_date += timedelta(days=1)
            if settlement_date.weekday() < 5:  # 月-金
                business_days += 1

        if current_date > settlement_date:
            return (
                False,
                f"決済期限超過: 取引日{trade_date}, 期限{settlement_date}",
                ComplianceLevel.HIGH,
            )

        return (True, "決済期間OK", ComplianceLevel.HIGH)

    def add_blacklisted_symbol(self, symbol: str, reason: str = "") -> None:
        """ブラックリスト銘柄追加"""
        self.blacklisted_symbols.add(symbol)
        logger.info(f"ブラックリスト追加: {symbol} - {reason}")

    def remove_blacklisted_symbol(self, symbol: str) -> None:
        """ブラックリスト銘柄削除"""
        self.blacklisted_symbols.discard(symbol)
        logger.info(f"ブラックリスト削除: {symbol}")

    def add_restricted_period(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
        reason: str = "決算発表期間",
    ) -> None:
        """制限期間追加"""
        restriction = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "reason": reason,
        }

        self.restricted_periods.append(restriction)
        logger.info(f"制限期間追加: {symbol} {start_date} - {end_date} ({reason})")

    def get_compliance_summary(self, compliance_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        コンプライアンスサマリー取得

        Args:
            compliance_results: コンプライアンスチェック結果リスト

        Returns:
            サマリー情報
        """
        try:
            total_checks = len(compliance_results)
            compliant_trades = sum(1 for r in compliance_results if r["overall_compliant"])

            violation_count = 0
            warning_count = 0
            violation_types = {}

            for result in compliance_results:
                violation_count += len(result.get("violations", []))
                warning_count += len(result.get("warnings", []))

                # 違反種別集計
                for violation in result.get("violations", []):
                    rule_id = violation.get("rule_id", "unknown")
                    violation_types[rule_id] = violation_types.get(rule_id, 0) + 1

            summary = {
                "total_trades_checked": total_checks,
                "compliant_trades": compliant_trades,
                "non_compliant_trades": total_checks - compliant_trades,
                "compliance_rate": (
                    (compliant_trades / total_checks * 100) if total_checks > 0 else 0
                ),
                "total_violations": violation_count,
                "total_warnings": warning_count,
                "violation_types": violation_types,
                "most_common_violations": sorted(
                    violation_types.items(), key=lambda x: x[1], reverse=True
                )[:5],
                "risk_level": self._assess_overall_risk_level(
                    compliant_trades, total_checks, violation_count
                ),
            }

            return summary

        except Exception as e:
            logger.error(f"コンプライアンスサマリー取得エラー: {e}")
            return {}

    def _assess_overall_risk_level(
        self, compliant_trades: int, total_trades: int, violation_count: int
    ) -> str:
        """全体リスクレベル評価"""
        if total_trades == 0:
            return "不明"

        compliance_rate = compliant_trades / total_trades
        violation_rate = violation_count / total_trades

        if compliance_rate >= 0.95 and violation_rate < 0.05:
            return "低"
        elif compliance_rate >= 0.90 and violation_rate < 0.10:
            return "中"
        elif compliance_rate >= 0.80 and violation_rate < 0.20:
            return "高"
        else:
            return "重大"

    def generate_compliance_report(self, compliance_results: List[Dict[str, Any]]) -> str:
        """
        コンプライアンスレポート生成

        Args:
            compliance_results: コンプライアンスチェック結果

        Returns:
            レポートテキスト
        """
        try:
            summary = self.get_compliance_summary(compliance_results)

            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("コンプライアンスチェックレポート")
            report_lines.append("=" * 60)
            report_lines.append(f"生成日時: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
            report_lines.append("")

            # サマリー
            report_lines.append("【チェックサマリー】")
            report_lines.append(f"総チェック数: {summary['total_trades_checked']:,}件")
            report_lines.append(f"適合取引: {summary['compliant_trades']:,}件")
            report_lines.append(f"非適合取引: {summary['non_compliant_trades']:,}件")
            report_lines.append(f"適合率: {summary['compliance_rate']:.1f}%")
            report_lines.append(f"総違反数: {summary['total_violations']:,}件")
            report_lines.append(f"総警告数: {summary['total_warnings']:,}件")
            report_lines.append(f"リスクレベル: {summary['risk_level']}")
            report_lines.append("")

            # 頻出違反
            if summary.get("most_common_violations"):
                report_lines.append("【頻出違反】")
                for rule_id, count in summary["most_common_violations"]:
                    rule = next((r for r in self.compliance_rules if r.rule_id == rule_id), None)
                    rule_name = rule.name if rule else rule_id
                    report_lines.append(f"  {rule_name}: {count}件")
                report_lines.append("")

            # 推奨アクション
            report_lines.append("【推奨アクション】")
            risk_level = summary.get("risk_level", "不明")

            if risk_level in ["重大", "高"]:
                report_lines.append("- 緊急対応：違反取引の詳細調査と修正措置")
                report_lines.append("- コンプライアンス体制の見直し")
            elif risk_level == "中":
                report_lines.append("- 警告事項の詳細確認と改善策検討")
                report_lines.append("- 定期的なコンプライアンス教育の実施")
            else:
                report_lines.append("- 現在の適合水準を維持")
                report_lines.append("- 継続的なモニタリング")

            # ブラックリスト情報
            if self.blacklisted_symbols:
                report_lines.append("")
                report_lines.append("【現在のブラックリスト銘柄】")
                for symbol in sorted(self.blacklisted_symbols):
                    report_lines.append(f"  {symbol}")

            # 制限期間情報
            if self.restricted_periods:
                report_lines.append("")
                report_lines.append("【現在の制限期間】")
                for restriction in self.restricted_periods:
                    report_lines.append(
                        f"  {restriction['symbol']}: "
                        f"{restriction['start_date']} - {restriction['end_date']} "
                        f"({restriction['reason']})"
                    )

            report_lines.append("")
            report_lines.append("=" * 60)

            return "\n".join(report_lines)

        except Exception as e:
            logger.error(f"コンプライアンスレポート生成エラー: {e}")
            return f"コンプライアンスレポート生成エラー: {str(e)}"
