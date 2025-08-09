"""
税務計算

株式取引の税務処理・損益計算・申告サポート機能
"""

from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple
from enum import Enum

from ...utils.logging_config import get_context_logger
from ..core.types import Trade, TradeType, RealizedPnL

logger = get_context_logger(__name__)


class TaxType(Enum):
    """税区分"""
    GENERAL = "一般"
    NISA = "NISA"
    SPECIFIED = "特定"


class TaxCalculator:
    """
    税務計算クラス

    株式取引の税務処理・確定申告データ生成機能を提供
    """

    def __init__(
        self,
        tax_rate: Decimal = Decimal("0.20315"),
        local_tax_rate: Decimal = Decimal("0.05"),
    ):
        """
        初期化

        Args:
            tax_rate: 所得税・住民税・復興特別所得税の合計税率
            local_tax_rate: 地方税率
        """
        self.tax_rate = tax_rate
        self.local_tax_rate = local_tax_rate
        logger.info(f"税務計算機初期化完了 - 税率: {tax_rate:.5%}")

    def calculate_annual_tax_summary(
        self,
        trades: List[Trade],
        target_year: int,
        tax_type: TaxType = TaxType.GENERAL,
    ) -> Dict[str, Decimal]:
        """
        年間税務サマリー計算

        Args:
            trades: 取引履歴
            target_year: 対象年度
            tax_type: 税区分

        Returns:
            年間税務サマリー
        """
        try:
            # 対象年度の取引抽出
            year_trades = self._filter_trades_by_year(trades, target_year)
            
            if not year_trades:
                logger.info(f"{target_year}年度の取引データなし")
                return self._empty_tax_summary()
            
            # 実現損益計算
            realized_gains = self._calculate_realized_gains(year_trades)
            
            # 配当所得（簡易版）
            dividend_income = self._calculate_dividend_income(year_trades)
            
            # 手数料合計
            total_fees = sum(trade.commission for trade in year_trades)
            
            # 課税対象所得
            taxable_income = max(realized_gains - total_fees, Decimal("0"))
            
            # 税額計算
            income_tax, local_tax, special_reconstruction_tax = self._calculate_tax_amounts(
                taxable_income, tax_type
            )
            
            # 損失繰越
            loss_carryover = max(Decimal("0") - (realized_gains - total_fees), Decimal("0"))
            
            tax_summary = {
                "target_year": Decimal(str(target_year)),
                "total_trades": Decimal(len(year_trades)),
                "total_buy_amount": sum(
                    trade.price * Decimal(trade.quantity) + trade.commission
                    for trade in year_trades if trade.trade_type == TradeType.BUY
                ),
                "total_sell_amount": sum(
                    trade.price * Decimal(trade.quantity) - trade.commission
                    for trade in year_trades if trade.trade_type == TradeType.SELL
                ),
                "realized_gains": realized_gains,
                "dividend_income": dividend_income,
                "total_fees": total_fees,
                "taxable_income": taxable_income,
                "income_tax": income_tax,
                "local_tax": local_tax,
                "special_reconstruction_tax": special_reconstruction_tax,
                "total_tax": income_tax + local_tax + special_reconstruction_tax,
                "loss_carryover": loss_carryover,
                "effective_tax_rate": (
                    (income_tax + local_tax + special_reconstruction_tax) / taxable_income * 100
                    if taxable_income > 0 else Decimal("0")
                ),
            }
            
            logger.info(
                f"{target_year}年度税務計算完了: "
                f"課税所得{taxable_income:,.0f}円, 税額{tax_summary['total_tax']:,.0f}円"
            )
            
            return tax_summary
            
        except Exception as e:
            logger.error(f"年間税務サマリー計算エラー: {e}")
            return self._empty_tax_summary()

    def _filter_trades_by_year(self, trades: List[Trade], year: int) -> List[Trade]:
        """年度別取引フィルタリング"""
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        
        return [
            trade for trade in trades
            if start_date <= trade.timestamp <= end_date
        ]

    def _empty_tax_summary(self) -> Dict[str, Decimal]:
        """空の税務サマリー"""
        return {
            "target_year": Decimal("0"),
            "total_trades": Decimal("0"),
            "total_buy_amount": Decimal("0"),
            "total_sell_amount": Decimal("0"),
            "realized_gains": Decimal("0"),
            "dividend_income": Decimal("0"),
            "total_fees": Decimal("0"),
            "taxable_income": Decimal("0"),
            "income_tax": Decimal("0"),
            "local_tax": Decimal("0"),
            "special_reconstruction_tax": Decimal("0"),
            "total_tax": Decimal("0"),
            "loss_carryover": Decimal("0"),
            "effective_tax_rate": Decimal("0"),
        }

    def _calculate_realized_gains(self, trades: List[Trade]) -> Decimal:
        """実現損益計算（税務用）"""
        realized_gains = Decimal("0")
        positions = {}  # 銘柄別ポジション管理
        
        # 時刻順でソート
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        
        for trade in sorted_trades:
            symbol = trade.symbol
            
            if symbol not in positions:
                positions[symbol] = {
                    "quantity": 0,
                    "total_cost": Decimal("0"),
                    "purchase_dates": [],
                }
            
            if trade.trade_type == TradeType.BUY:
                # 購入：ポジション追加
                positions[symbol]["quantity"] += trade.quantity
                cost = trade.price * Decimal(trade.quantity) + trade.commission
                positions[symbol]["total_cost"] += cost
                positions[symbol]["purchase_dates"].extend([trade.timestamp] * trade.quantity)
                
            elif trade.trade_type == TradeType.SELL:
                # 売却：FIFO法で実現損益計算
                if positions[symbol]["quantity"] >= trade.quantity:
                    # 平均取得価格
                    avg_cost = positions[symbol]["total_cost"] / Decimal(positions[symbol]["quantity"])
                    
                    # 売却損益
                    sell_proceeds = trade.price * Decimal(trade.quantity) - trade.commission
                    cost_basis = avg_cost * Decimal(trade.quantity)
                    gain_loss = sell_proceeds - cost_basis
                    
                    realized_gains += gain_loss
                    
                    # ポジション更新
                    positions[symbol]["quantity"] -= trade.quantity
                    positions[symbol]["total_cost"] -= cost_basis
                    positions[symbol]["purchase_dates"] = positions[symbol]["purchase_dates"][trade.quantity:]
                    
                    # ポジション完全売却
                    if positions[symbol]["quantity"] == 0:
                        del positions[symbol]
        
        return realized_gains

    def _calculate_dividend_income(self, trades: List[Trade]) -> Decimal:
        """配当所得計算（簡易版）"""
        # 実際の実装では配当記録を別途管理
        # ここでは簡易的に0を返す
        return Decimal("0")

    def _calculate_tax_amounts(
        self,
        taxable_income: Decimal,
        tax_type: TaxType,
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """税額計算"""
        if taxable_income <= 0:
            return Decimal("0"), Decimal("0"), Decimal("0")
        
        if tax_type == TaxType.NISA:
            # NISA口座は非課税
            return Decimal("0"), Decimal("0"), Decimal("0")
        
        # 一般口座・特定口座の税額計算
        
        # 所得税（15.315%）
        income_tax_rate = Decimal("0.15315")
        income_tax = (taxable_income * income_tax_rate).quantize(Decimal("1"), rounding=ROUND_DOWN)
        
        # 住民税（5%）
        local_tax = (taxable_income * self.local_tax_rate).quantize(Decimal("1"), rounding=ROUND_DOWN)
        
        # 復興特別所得税（0.315%）
        special_reconstruction_rate = Decimal("0.00315")
        special_reconstruction_tax = (taxable_income * special_reconstruction_rate).quantize(
            Decimal("1"), rounding=ROUND_DOWN
        )
        
        return income_tax, local_tax, special_reconstruction_tax

    def calculate_loss_carryforward(
        self,
        trades: List[Trade],
        previous_carryover: Decimal = Decimal("0"),
        target_year: int = None,
    ) -> Dict[str, Decimal]:
        """
        損失繰越控除計算

        Args:
            trades: 取引履歴
            previous_carryover: 前年繰越損失
            target_year: 対象年度

        Returns:
            損失繰越計算結果
        """
        try:
            if target_year is None:
                target_year = datetime.now().year
            
            year_trades = self._filter_trades_by_year(trades, target_year)
            
            # 当年度実現損益
            current_year_gains = self._calculate_realized_gains(year_trades)
            
            # 繰越控除適用
            if previous_carryover > 0 and current_year_gains > 0:
                # 利益から損失を控除
                carryover_used = min(previous_carryover, current_year_gains)
                adjusted_gains = current_year_gains - carryover_used
                remaining_carryover = previous_carryover - carryover_used
            else:
                carryover_used = Decimal("0")
                adjusted_gains = current_year_gains
                remaining_carryover = previous_carryover
            
            # 当年度新規損失
            if adjusted_gains < 0:
                new_loss = abs(adjusted_gains)
                remaining_carryover += new_loss
                adjusted_gains = Decimal("0")
            else:
                new_loss = Decimal("0")
            
            result = {
                "target_year": Decimal(str(target_year)),
                "previous_carryover": previous_carryover,
                "current_year_gains": current_year_gains,
                "carryover_used": carryover_used,
                "adjusted_gains": adjusted_gains,
                "new_loss": new_loss,
                "remaining_carryover": remaining_carryover,
                "carryover_expires_year": Decimal(str(target_year + 3)),  # 3年間繰越可能
            }
            
            logger.info(f"損失繰越計算完了: 使用{carryover_used:,.0f}円, 残{remaining_carryover:,.0f}円")
            return result
            
        except Exception as e:
            logger.error(f"損失繰越計算エラー: {e}")
            return {}

    def generate_tax_report_data(
        self,
        trades: List[Trade],
        target_year: int,
        previous_carryover: Decimal = Decimal("0"),
    ) -> Dict[str, any]:
        """
        確定申告用レポートデータ生成

        Args:
            trades: 取引履歴
            target_year: 対象年度
            previous_carryover: 前年繰越損失

        Returns:
            確定申告用データ
        """
        try:
            # 年間サマリー
            tax_summary = self.calculate_annual_tax_summary(trades, target_year)
            
            # 損失繰越
            loss_carryforward = self.calculate_loss_carryforward(trades, previous_carryover, target_year)
            
            # 銘柄別損益明細
            stock_details = self._generate_stock_details(trades, target_year)
            
            # 月別サマリー
            monthly_summary = self._generate_monthly_summary(trades, target_year)
            
            # 取引明細（確定申告添付用）
            transaction_details = self._generate_transaction_details(trades, target_year)
            
            report_data = {
                "report_info": {
                    "target_year": target_year,
                    "generation_date": datetime.now().isoformat(),
                    "total_transactions": len(self._filter_trades_by_year(trades, target_year)),
                },
                "tax_summary": tax_summary,
                "loss_carryforward": loss_carryforward,
                "stock_details": stock_details,
                "monthly_summary": monthly_summary,
                "transaction_details": transaction_details,
                "required_forms": self._get_required_tax_forms(tax_summary),
            }
            
            logger.info(f"{target_year}年度確定申告データ生成完了")
            return report_data
            
        except Exception as e:
            logger.error(f"確定申告データ生成エラー: {e}")
            return {
                "error": str(e),
                "generation_date": datetime.now().isoformat(),
            }

    def _generate_stock_details(self, trades: List[Trade], year: int) -> List[Dict]:
        """銘柄別損益明細生成"""
        year_trades = self._filter_trades_by_year(trades, year)
        
        # 銘柄別グループ化
        stock_groups = {}
        for trade in year_trades:
            if trade.symbol not in stock_groups:
                stock_groups[trade.symbol] = []
            stock_groups[trade.symbol].append(trade)
        
        stock_details = []
        for symbol, symbol_trades in stock_groups.items():
            # 銘柄別実現損益計算
            realized_pnl = self._calculate_realized_gains(symbol_trades)
            
            # 取引統計
            buy_trades = [t for t in symbol_trades if t.trade_type == TradeType.BUY]
            sell_trades = [t for t in symbol_trades if t.trade_type == TradeType.SELL]
            
            total_buy_amount = sum(
                t.price * Decimal(t.quantity) + t.commission for t in buy_trades
            )
            total_sell_amount = sum(
                t.price * Decimal(t.quantity) - t.commission for t in sell_trades
            )
            
            stock_detail = {
                "symbol": symbol,
                "buy_transactions": len(buy_trades),
                "sell_transactions": len(sell_trades),
                "total_buy_amount": total_buy_amount,
                "total_sell_amount": total_sell_amount,
                "realized_pnl": realized_pnl,
                "total_fees": sum(t.commission for t in symbol_trades),
            }
            
            stock_details.append(stock_detail)
        
        # 損益順でソート
        stock_details.sort(key=lambda x: x["realized_pnl"], reverse=True)
        
        return stock_details

    def _generate_monthly_summary(self, trades: List[Trade], year: int) -> List[Dict]:
        """月別サマリー生成"""
        year_trades = self._filter_trades_by_year(trades, year)
        
        monthly_data = {}
        for month in range(1, 13):
            month_start = datetime(year, month, 1)
            if month == 12:
                month_end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
            else:
                month_end = datetime(year, month + 1, 1) - timedelta(seconds=1)
            
            month_trades = [
                t for t in year_trades
                if month_start <= t.timestamp <= month_end
            ]
            
            if month_trades:
                monthly_data[month] = {
                    "month": month,
                    "trades_count": len(month_trades),
                    "realized_pnl": self._calculate_realized_gains(month_trades),
                    "total_fees": sum(t.commission for t in month_trades),
                    "buy_amount": sum(
                        t.price * Decimal(t.quantity) + t.commission
                        for t in month_trades if t.trade_type == TradeType.BUY
                    ),
                    "sell_amount": sum(
                        t.price * Decimal(t.quantity) - t.commission
                        for t in month_trades if t.trade_type == TradeType.SELL
                    ),
                }
        
        return list(monthly_data.values())

    def _generate_transaction_details(self, trades: List[Trade], year: int) -> List[Dict]:
        """取引明細生成（確定申告添付用）"""
        year_trades = self._filter_trades_by_year(trades, year)
        
        transaction_details = []
        for trade in year_trades:
            detail = {
                "date": trade.timestamp.strftime("%Y/%m/%d"),
                "symbol": trade.symbol,
                "transaction_type": trade.trade_type.value,
                "quantity": trade.quantity,
                "price": trade.price,
                "amount": trade.price * Decimal(trade.quantity),
                "commission": trade.commission,
                "net_amount": (
                    trade.price * Decimal(trade.quantity) +
                    (trade.commission if trade.trade_type == TradeType.BUY else -trade.commission)
                ),
            }
            transaction_details.append(detail)
        
        # 日付順でソート
        transaction_details.sort(key=lambda x: x["date"])
        
        return transaction_details

    def _get_required_tax_forms(self, tax_summary: Dict[str, Decimal]) -> List[str]:
        """必要な税務書類リスト"""
        forms = ["確定申告書B"]
        
        if tax_summary["realized_gains"] != 0:
            forms.append("株式等に係る譲渡所得等の金額の計算明細書")
        
        if tax_summary["loss_carryover"] > 0:
            forms.append("損失申告用の計算明細書")
        
        return forms

    def calculate_estimated_quarterly_tax(
        self,
        trades: List[Trade],
        current_year: int,
        quarter: int,
    ) -> Dict[str, Decimal]:
        """
        予定納税額計算

        Args:
            trades: 取引履歴
            current_year: 現在年度
            quarter: 四半期（1-4）

        Returns:
            予定納税計算結果
        """
        try:
            # 四半期末日計算
            quarter_end_month = quarter * 3
            quarter_end = datetime(current_year, quarter_end_month, 1)
            if quarter_end_month == 12:
                # 12月は31日まで
                quarter_end = datetime(current_year, 12, 31)
            else:
                # 翌月1日-1秒
                quarter_end = datetime(current_year, quarter_end_month + 1, 1) - timedelta(seconds=1)
            
            # 四半期までの取引
            qtd_trades = [
                trade for trade in trades
                if datetime(current_year, 1, 1) <= trade.timestamp <= quarter_end
            ]
            
            # 四半期実現損益
            qtd_gains = self._calculate_realized_gains(qtd_trades)
            qtd_fees = sum(t.commission for t in qtd_trades)
            qtd_taxable = max(qtd_gains - qtd_fees, Decimal("0"))
            
            # 年間予想（線形外挿）
            annual_estimate = qtd_taxable * Decimal("4") / Decimal(str(quarter))
            
            # 予定納税額
            estimated_tax = annual_estimate * self.tax_rate
            quarterly_payment = estimated_tax / Decimal("4")
            
            result = {
                "quarter": Decimal(str(quarter)),
                "qtd_realized_gains": qtd_gains,
                "qtd_taxable_income": qtd_taxable,
                "annual_estimate": annual_estimate,
                "estimated_annual_tax": estimated_tax,
                "quarterly_payment": quarterly_payment,
                "payment_due_date": self._get_quarterly_due_date(current_year, quarter),
            }
            
            logger.info(f"Q{quarter}予定納税計算完了: {quarterly_payment:,.0f}円")
            return result
            
        except Exception as e:
            logger.error(f"予定納税計算エラー: {e}")
            return {}

    def _get_quarterly_due_date(self, year: int, quarter: int) -> str:
        """予定納税期限日取得"""
        due_dates = {
            1: f"{year}/05/31",  # 第1期
            2: f"{year}/08/31",  # 第2期
            3: f"{year}/11/30",  # 第3期
            4: f"{year + 1}/03/15",  # 確定申告期限
        }
        return due_dates.get(quarter, "")