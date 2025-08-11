"""
取引データ検証

取引データの整合性・有効性・ビジネスルール検証機能
"""

import re
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Tuple

from ...utils.logging_config import get_context_logger
from ..core.types import Trade, TradeType

logger = get_context_logger(__name__)


class ValidationRule:
    """検証ルールクラス"""

    def __init__(self, name: str, description: str, severity: str = "error"):
        self.name = name
        self.description = description
        self.severity = severity  # error, warning, info


class TradeValidator:
    """
    取引データ検証クラス

    取引データの包括的な検証機能を提供
    """

    def __init__(self):
        """初期化"""
        self.validation_rules = self._initialize_validation_rules()
        logger.info("取引データ検証機初期化完了")

    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """検証ルール初期化"""
        return [
            ValidationRule("trade_id_format", "取引ID形式検証", "error"),
            ValidationRule("symbol_format", "銘柄コード形式検証", "error"),
            ValidationRule("quantity_positive", "数量正数検証", "error"),
            ValidationRule("price_positive", "価格正数検証", "error"),
            ValidationRule("commission_non_negative", "手数料非負検証", "error"),
            ValidationRule("timestamp_valid", "タイムスタンプ有効性検証", "error"),
            ValidationRule("trade_type_valid", "取引タイプ有効性検証", "error"),
            ValidationRule("unit_share_compliance", "単元株制準拠検証", "warning"),
            ValidationRule("price_range_check", "価格レンジチェック", "warning"),
            ValidationRule("weekend_trading_check", "週末取引チェック", "warning"),
            ValidationRule("commission_reasonableness", "手数料妥当性チェック", "info"),
        ]

    def validate_single_trade(self, trade: Trade) -> Dict[str, Any]:
        """
        単一取引データ検証

        Args:
            trade: 取引データ

        Returns:
            検証結果辞書
        """
        validation_result = {
            "trade_id": trade.id,
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "info": [],
            "validation_details": {},
        }

        try:
            # 各検証ルールを実行
            validations = {
                "trade_id_format": self._validate_trade_id_format(trade),
                "symbol_format": self._validate_symbol_format(trade),
                "quantity_positive": self._validate_quantity_positive(trade),
                "price_positive": self._validate_price_positive(trade),
                "commission_non_negative": self._validate_commission_non_negative(
                    trade
                ),
                "timestamp_valid": self._validate_timestamp_valid(trade),
                "trade_type_valid": self._validate_trade_type_valid(trade),
                "unit_share_compliance": self._validate_unit_share_compliance(trade),
                "price_range_check": self._validate_price_range(trade),
                "weekend_trading_check": self._validate_weekend_trading(trade),
                "commission_reasonableness": self._validate_commission_reasonableness(
                    trade
                ),
            }

            # 結果分類
            for rule_name, (is_valid, message) in validations.items():
                validation_result["validation_details"][rule_name] = {
                    "is_valid": is_valid,
                    "message": message,
                }

                if not is_valid:
                    # ルールの重要度に基づいて分類
                    rule = next(
                        (r for r in self.validation_rules if r.name == rule_name), None
                    )
                    severity = rule.severity if rule else "error"

                    if severity == "error":
                        validation_result["errors"].append(f"{rule_name}: {message}")
                        validation_result["is_valid"] = False
                    elif severity == "warning":
                        validation_result["warnings"].append(f"{rule_name}: {message}")
                    elif severity == "info":
                        validation_result["info"].append(f"{rule_name}: {message}")

            # サマリーログ
            if not validation_result["is_valid"]:
                logger.warning(
                    f"取引検証失敗: {trade.id} - {len(validation_result['errors'])}エラー"
                )
            elif validation_result["warnings"]:
                logger.info(
                    f"取引検証注意: {trade.id} - {len(validation_result['warnings'])}警告"
                )

            return validation_result

        except Exception as e:
            logger.error(f"取引検証エラー: {trade.id} - {e}")
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"検証処理エラー: {str(e)}")
            return validation_result

    def _validate_trade_id_format(self, trade: Trade) -> Tuple[bool, str]:
        """取引ID形式検証"""
        if not trade.id:
            return False, "取引IDが空です"

        if len(trade.id) < 5:
            return False, "取引IDが短すぎます"

        # ID形式パターン（例: TRD_20240101120000_12345678）
        pattern = r"^TRD_\d{14}_[a-zA-Z0-9]{8}$"
        if not re.match(pattern, trade.id):
            return False, "取引ID形式が正しくありません"

        return True, "取引ID形式OK"

    def _validate_symbol_format(self, trade: Trade) -> Tuple[bool, str]:
        """銘柄コード形式検証"""
        if not trade.symbol:
            return False, "銘柄コードが空です"

        if not trade.symbol.strip():
            return False, "銘柄コードが空白のみです"

        # 日本株銘柄コードパターン（4桁数字）
        pattern = r"^\d{4}$"
        if not re.match(pattern, trade.symbol):
            return (
                False,
                "銘柄コード形式が正しくありません（4桁数字である必要があります）",
            )

        return True, "銘柄コード形式OK"

    def _validate_quantity_positive(self, trade: Trade) -> Tuple[bool, str]:
        """数量正数検証"""
        if trade.quantity <= 0:
            return False, f"数量が0以下です: {trade.quantity}"

        return True, "数量OK"

    def _validate_price_positive(self, trade: Trade) -> Tuple[bool, str]:
        """価格正数検証"""
        if trade.price <= Decimal("0"):
            return False, f"価格が0以下です: {trade.price}"

        return True, "価格OK"

    def _validate_commission_non_negative(self, trade: Trade) -> Tuple[bool, str]:
        """手数料非負検証"""
        if trade.commission < Decimal("0"):
            return False, f"手数料が負の値です: {trade.commission}"

        return True, "手数料OK"

    def _validate_timestamp_valid(self, trade: Trade) -> Tuple[bool, str]:
        """タイムスタンプ有効性検証"""
        current_time = datetime.now()
        min_date = datetime(2000, 1, 1)
        max_future_date = current_time + timedelta(days=1)

        if trade.timestamp < min_date:
            return False, f"取引日時が古すぎます: {trade.timestamp}"

        if trade.timestamp > max_future_date:
            return False, f"取引日時が未来すぎます: {trade.timestamp}"

        return True, "タイムスタンプOK"

    def _validate_trade_type_valid(self, trade: Trade) -> Tuple[bool, str]:
        """取引タイプ有効性検証"""
        valid_types = [TradeType.BUY, TradeType.SELL]

        if trade.trade_type not in valid_types:
            return False, f"無効な取引タイプ: {trade.trade_type}"

        return True, "取引タイプOK"

    def _validate_unit_share_compliance(self, trade: Trade) -> Tuple[bool, str]:
        """単元株制準拠検証"""
        if trade.quantity % 100 != 0:
            return False, f"単元株数に準拠していません: {trade.quantity}株"

        return True, "単元株制準拠OK"

    def _validate_price_range(self, trade: Trade) -> Tuple[bool, str]:
        """価格レンジチェック"""
        min_price = Decimal("1")
        max_price = Decimal("100000")

        if trade.price < min_price:
            return False, f"価格が低すぎます: {trade.price}円"

        if trade.price > max_price:
            return False, f"価格が高すぎます: {trade.price}円"

        return True, "価格レンジOK"

    def _validate_weekend_trading(self, trade: Trade) -> Tuple[bool, str]:
        """週末取引チェック"""
        weekday = trade.timestamp.weekday()

        # 土曜日(5)、日曜日(6)
        if weekday in [5, 6]:
            return False, f"週末の取引です: {trade.timestamp.strftime('%Y/%m/%d (%A)')}"

        return True, "取引日OK"

    def _validate_commission_reasonableness(self, trade: Trade) -> Tuple[bool, str]:
        """手数料妥当性チェック"""
        trade_value = trade.price * Decimal(trade.quantity)
        commission_rate = trade.commission / trade_value * 100

        # 手数料率が異常に高い場合
        if commission_rate > Decimal("1.0"):  # 1%超
            return False, f"手数料率が高すぎます: {commission_rate:.3f}%"

        # 手数料が異常に低い場合
        if commission_rate < Decimal("0.01"):  # 0.01%未満
            return False, f"手数料率が低すぎます: {commission_rate:.3f}%"

        return True, "手数料妥当性OK"

    def validate_trade_batch(self, trades: List[Trade]) -> Dict[str, Any]:
        """
        取引データ一括検証

        Args:
            trades: 取引データリスト

        Returns:
            一括検証結果
        """
        batch_result = {
            "total_trades": len(trades),
            "valid_trades": 0,
            "invalid_trades": 0,
            "trades_with_warnings": 0,
            "validation_details": [],
            "summary": {
                "error_types": {},
                "warning_types": {},
                "most_common_errors": [],
                "most_common_warnings": [],
            },
        }

        try:
            for trade in trades:
                trade_result = self.validate_single_trade(trade)
                batch_result["validation_details"].append(trade_result)

                if trade_result["is_valid"]:
                    batch_result["valid_trades"] += 1
                else:
                    batch_result["invalid_trades"] += 1

                if trade_result["warnings"]:
                    batch_result["trades_with_warnings"] += 1

                # エラー・警告の統計
                for error in trade_result["errors"]:
                    error_type = error.split(":")[0]
                    batch_result["summary"]["error_types"][error_type] = (
                        batch_result["summary"]["error_types"].get(error_type, 0) + 1
                    )

                for warning in trade_result["warnings"]:
                    warning_type = warning.split(":")[0]
                    batch_result["summary"]["warning_types"][warning_type] = (
                        batch_result["summary"]["warning_types"].get(warning_type, 0)
                        + 1
                    )

            # 最頻出エラー・警告
            batch_result["summary"]["most_common_errors"] = sorted(
                batch_result["summary"]["error_types"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            batch_result["summary"]["most_common_warnings"] = sorted(
                batch_result["summary"]["warning_types"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            logger.info(
                f"一括検証完了: {batch_result['total_trades']}取引中 "
                f"{batch_result['valid_trades']}有効, "
                f"{batch_result['invalid_trades']}無効, "
                f"{batch_result['trades_with_warnings']}警告あり"
            )

            return batch_result

        except Exception as e:
            logger.error(f"一括検証エラー: {e}")
            batch_result["error"] = str(e)
            return batch_result

    def validate_trade_sequence(self, trades: List[Trade]) -> Dict[str, Any]:
        """
        取引シーケンス検証

        Args:
            trades: 時系列取引データ

        Returns:
            シーケンス検証結果
        """
        sequence_result = {
            "total_trades": len(trades),
            "sequence_valid": True,
            "issues": [],
            "position_tracking": {},
            "timeline_issues": [],
        }

        try:
            # 時系列ソート
            sorted_trades = sorted(trades, key=lambda t: t.timestamp)
            position_balances = {}  # 銘柄別ポジション追跡

            prev_timestamp = None

            for i, trade in enumerate(sorted_trades):
                symbol = trade.symbol

                # タイムスタンプ順序チェック
                if prev_timestamp and trade.timestamp < prev_timestamp:
                    sequence_result["timeline_issues"].append(
                        {
                            "trade_id": trade.id,
                            "issue": "時刻順序不正",
                            "details": f"前の取引より古い時刻: {trade.timestamp}",
                        }
                    )

                # ポジション追跡初期化
                if symbol not in position_balances:
                    position_balances[symbol] = 0

                # 売却時のポジション不足チェック
                if trade.trade_type == TradeType.SELL:
                    if position_balances[symbol] < trade.quantity:
                        sequence_result["issues"].append(
                            {
                                "trade_id": trade.id,
                                "issue": "売却数量過多",
                                "details": f"保有{position_balances[symbol]}株に対し{trade.quantity}株売却",
                                "severity": "error",
                            }
                        )
                        sequence_result["sequence_valid"] = False

                # ポジション更新
                if trade.trade_type == TradeType.BUY:
                    position_balances[symbol] += trade.quantity
                elif trade.trade_type == TradeType.SELL:
                    position_balances[symbol] -= trade.quantity

                prev_timestamp = trade.timestamp

            # 最終ポジション記録
            sequence_result["position_tracking"] = position_balances

            # 結果サマリー
            if sequence_result["sequence_valid"]:
                logger.info("取引シーケンス検証成功")
            else:
                logger.warning(
                    f"取引シーケンス検証失敗: {len(sequence_result['issues'])}問題"
                )

            return sequence_result

        except Exception as e:
            logger.error(f"シーケンス検証エラー: {e}")
            sequence_result["sequence_valid"] = False
            sequence_result["error"] = str(e)
            return sequence_result

    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        検証レポート生成

        Args:
            validation_results: 検証結果データ

        Returns:
            レポートテキスト
        """
        try:
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("取引データ検証レポート")
            report_lines.append("=" * 60)
            report_lines.append(
                f"生成日時: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}"
            )
            report_lines.append("")

            # サマリー
            report_lines.append("【検証サマリー】")
            report_lines.append(
                f"総取引数: {validation_results.get('total_trades', 0):,}件"
            )
            report_lines.append(
                f"有効取引: {validation_results.get('valid_trades', 0):,}件"
            )
            report_lines.append(
                f"無効取引: {validation_results.get('invalid_trades', 0):,}件"
            )
            report_lines.append(
                f"警告あり: {validation_results.get('trades_with_warnings', 0):,}件"
            )
            report_lines.append("")

            # エラー統計
            summary = validation_results.get("summary", {})
            if summary.get("most_common_errors"):
                report_lines.append("【頻出エラー】")
                for error_type, count in summary["most_common_errors"]:
                    report_lines.append(f"  {error_type}: {count}件")
                report_lines.append("")

            # 警告統計
            if summary.get("most_common_warnings"):
                report_lines.append("【頻出警告】")
                for warning_type, count in summary["most_common_warnings"]:
                    report_lines.append(f"  {warning_type}: {count}件")
                report_lines.append("")

            # 推奨アクション
            report_lines.append("【推奨アクション】")
            if validation_results.get("invalid_trades", 0) > 0:
                report_lines.append("- 無効な取引データの修正または除外を検討")

            if validation_results.get("trades_with_warnings", 0) > 0:
                report_lines.append("- 警告のある取引データの詳細確認を推奨")

            error_types = summary.get("error_types", {})
            if "unit_share_compliance" in error_types:
                report_lines.append("- 単元株制に準拠しない取引の確認")

            if "weekend_trading_check" in error_types:
                report_lines.append("- 週末取引の取引日確認")

            report_lines.append("")
            report_lines.append("=" * 60)

            return "\n".join(report_lines)

        except Exception as e:
            logger.error(f"検証レポート生成エラー: {e}")
            return f"検証レポート生成エラー: {str(e)}"

    def get_validation_statistics(
        self, validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        検証統計情報取得

        Args:
            validation_results: 検証結果

        Returns:
            統計情報
        """
        try:
            total_trades = validation_results.get("total_trades", 0)
            valid_trades = validation_results.get("valid_trades", 0)
            invalid_trades = validation_results.get("invalid_trades", 0)

            statistics = {
                "validation_success_rate": (valid_trades / total_trades * 100)
                if total_trades > 0
                else 0,
                "error_rate": (invalid_trades / total_trades * 100)
                if total_trades > 0
                else 0,
                "warning_rate": (
                    validation_results.get("trades_with_warnings", 0)
                    / total_trades
                    * 100
                )
                if total_trades > 0
                else 0,
                "data_quality_score": self._calculate_data_quality_score(
                    validation_results
                ),
                "most_critical_issue": self._identify_most_critical_issue(
                    validation_results
                ),
            }

            return statistics

        except Exception as e:
            logger.error(f"検証統計取得エラー: {e}")
            return {}

    def _calculate_data_quality_score(
        self, validation_results: Dict[str, Any]
    ) -> float:
        """データ品質スコア計算"""
        total_trades = validation_results.get("total_trades", 0)
        if total_trades == 0:
            return 0.0

        valid_trades = validation_results.get("valid_trades", 0)
        trades_with_warnings = validation_results.get("trades_with_warnings", 0)

        # スコア計算（有効率 - 警告ペナルティ）
        success_rate = valid_trades / total_trades
        warning_penalty = (trades_with_warnings / total_trades) * 0.1  # 10%ペナルティ

        quality_score = max(0, (success_rate - warning_penalty) * 100)

        return round(quality_score, 2)

    def _identify_most_critical_issue(self, validation_results: Dict[str, Any]) -> str:
        """最重要問題特定"""
        summary = validation_results.get("summary", {})
        most_common_errors = summary.get("most_common_errors", [])

        if most_common_errors:
            most_common_error, count = most_common_errors[0]
            return f"{most_common_error} ({count}件)"

        return "問題なし"
