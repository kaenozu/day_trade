"""
ポートフォリオ分析

パフォーマンス測定・リスク分析・投資効率評価機能
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import statistics
import math

from ...utils.logging_config import get_context_logger
from ..core.types import Trade, Position, TradeType, RealizedPnL

logger = get_context_logger(__name__)


class PortfolioAnalyzer:
    """
    ポートフォリオ分析クラス

    取引履歴からパフォーマンス・リスク・投資効率を分析
    """

    def __init__(self):
        """初期化"""
        logger.info("ポートフォリオアナライザー初期化完了")

    def calculate_portfolio_performance(
        self,
        trades: List[Trade],
        positions: Dict[str, Position],
        benchmark_return: Optional[Decimal] = None,
    ) -> Dict[str, Decimal]:
        """
        ポートフォリオパフォーマンス計算

        Args:
            trades: 取引履歴
            positions: 現在のポジション
            benchmark_return: ベンチマークリターン（%）

        Returns:
            パフォーマンス指標辞書
        """
        try:
            # 実現損益計算
            realized_pnl = self._calculate_realized_pnl(trades)
            
            # 未実現損益計算
            unrealized_pnl = sum(pos.unrealized_pnl for pos in positions.values())
            
            # 総投資額
            total_investment = sum(
                trade.price * Decimal(trade.quantity) + trade.commission
                for trade in trades if trade.trade_type == TradeType.BUY
            )
            
            # 総リターン
            total_return = realized_pnl + unrealized_pnl
            
            # リターン率
            return_percentage = (
                (total_return / total_investment * 100) if total_investment > 0 else Decimal("0")
            )
            
            # 年率換算リターン
            annualized_return = self._calculate_annualized_return(trades, total_return, total_investment)
            
            # シャープレシオ
            sharpe_ratio = self._calculate_sharpe_ratio(trades, annualized_return)
            
            # 最大ドローダウン
            max_drawdown = self._calculate_max_drawdown(trades)
            
            # ベンチマーク比較
            alpha = Decimal("0")
            if benchmark_return:
                alpha = return_percentage - benchmark_return
            
            performance = {
                "total_investment": total_investment,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_return": total_return,
                "return_percentage": return_percentage,
                "annualized_return": annualized_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown_percentage": max_drawdown,
                "alpha": alpha,
                "total_trades": Decimal(len(trades)),
            }
            
            logger.info(f"パフォーマンス計算完了: リターン{return_percentage:.2f}%")
            return performance
            
        except Exception as e:
            logger.error(f"パフォーマンス計算エラー: {e}")
            return {}

    def _calculate_realized_pnl(self, trades: List[Trade]) -> Decimal:
        """実現損益計算"""
        realized_pnl = Decimal("0")
        positions_cost = {}  # 銘柄別取得コスト追跡
        
        # 取引を時刻順でソート
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        
        for trade in sorted_trades:
            symbol = trade.symbol
            trade_value = trade.price * Decimal(trade.quantity)
            
            if trade.trade_type == TradeType.BUY:
                # 買い：取得コストに追加
                if symbol not in positions_cost:
                    positions_cost[symbol] = {"quantity": 0, "total_cost": Decimal("0")}
                
                positions_cost[symbol]["quantity"] += trade.quantity
                positions_cost[symbol]["total_cost"] += trade_value + trade.commission
                
            elif trade.trade_type == TradeType.SELL:
                # 売り：実現損益計算
                if symbol in positions_cost and positions_cost[symbol]["quantity"] >= trade.quantity:
                    # 平均取得価格計算
                    avg_cost_per_share = (
                        positions_cost[symbol]["total_cost"] / Decimal(positions_cost[symbol]["quantity"])
                    )
                    
                    # 売却分のコスト
                    sell_cost = avg_cost_per_share * Decimal(trade.quantity)
                    
                    # 実現損益
                    pnl = trade_value - trade.commission - sell_cost
                    realized_pnl += pnl
                    
                    # 残りポジション更新
                    positions_cost[symbol]["quantity"] -= trade.quantity
                    positions_cost[symbol]["total_cost"] -= sell_cost
                    
                    if positions_cost[symbol]["quantity"] == 0:
                        del positions_cost[symbol]
        
        return realized_pnl

    def _calculate_annualized_return(
        self,
        trades: List[Trade],
        total_return: Decimal,
        total_investment: Decimal,
    ) -> Decimal:
        """年率換算リターン計算"""
        if not trades or total_investment == 0:
            return Decimal("0")
        
        # 取引期間計算
        earliest_trade = min(trades, key=lambda t: t.timestamp)
        latest_trade = max(trades, key=lambda t: t.timestamp)
        
        days_diff = (latest_trade.timestamp - earliest_trade.timestamp).days
        years_diff = max(days_diff / 365.25, 0.1)  # 最小0.1年
        
        # 年率換算
        total_return_rate = total_return / total_investment
        annualized_return = (
            (1 + total_return_rate) ** Decimal(str(1 / years_diff)) - 1
        ) * 100
        
        return annualized_return

    def _calculate_sharpe_ratio(self, trades: List[Trade], annualized_return: Decimal) -> Decimal:
        """シャープレシオ計算"""
        if len(trades) < 2:
            return Decimal("0")
        
        try:
            # 日次リターン計算
            daily_returns = self._calculate_daily_returns(trades)
            if not daily_returns:
                return Decimal("0")
            
            # リターンの標準偏差（年率換算）
            return_std = Decimal(str(statistics.stdev(daily_returns))) * Decimal("15.81")  # √252
            
            # リスクフリーレート（仮定：1%）
            risk_free_rate = Decimal("1.0")
            
            # シャープレシオ
            if return_std > 0:
                sharpe_ratio = (annualized_return - risk_free_rate) / return_std
                return sharpe_ratio
            
            return Decimal("0")
            
        except Exception as e:
            logger.warning(f"シャープレシオ計算エラー: {e}")
            return Decimal("0")

    def _calculate_daily_returns(self, trades: List[Trade]) -> List[float]:
        """日次リターン計算"""
        try:
            # 日別損益の簡易計算
            daily_pnl = {}
            
            for trade in trades:
                date_key = trade.timestamp.date()
                
                if date_key not in daily_pnl:
                    daily_pnl[date_key] = Decimal("0")
                
                # 簡易的な日次損益（売買差額）
                if trade.trade_type == TradeType.SELL:
                    daily_pnl[date_key] += trade.price * Decimal(trade.quantity) - trade.commission
                else:
                    daily_pnl[date_key] -= trade.price * Decimal(trade.quantity) + trade.commission
            
            # Decimal → float変換（統計計算のため）
            daily_returns = [float(pnl) for pnl in daily_pnl.values() if pnl != 0]
            
            return daily_returns
            
        except Exception as e:
            logger.warning(f"日次リターン計算エラー: {e}")
            return []

    def _calculate_max_drawdown(self, trades: List[Trade]) -> Decimal:
        """最大ドローダウン計算"""
        if not trades:
            return Decimal("0")
        
        try:
            # 累積損益履歴
            cumulative_pnl = []
            running_pnl = Decimal("0")
            
            sorted_trades = sorted(trades, key=lambda t: t.timestamp)
            
            for trade in sorted_trades:
                if trade.trade_type == TradeType.SELL:
                    # 売却時に簡易的な損益を加算
                    running_pnl += trade.price * Decimal(trade.quantity) - trade.commission
                else:
                    # 購入時はコストとして減算
                    running_pnl -= trade.price * Decimal(trade.quantity) + trade.commission
                
                cumulative_pnl.append(running_pnl)
            
            if not cumulative_pnl:
                return Decimal("0")
            
            # 最大ドローダウン計算
            max_drawdown = Decimal("0")
            peak = cumulative_pnl[0]
            
            for pnl in cumulative_pnl[1:]:
                if pnl > peak:
                    peak = pnl
                
                drawdown = (peak - pnl) / abs(peak) * 100 if peak != 0 else Decimal("0")
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            logger.warning(f"ドローダウン計算エラー: {e}")
            return Decimal("0")

    def analyze_trade_efficiency(self, trades: List[Trade]) -> Dict[str, any]:
        """
        取引効率分析

        Args:
            trades: 取引履歴

        Returns:
            取引効率指標辞書
        """
        try:
            if not trades:
                return {}
            
            # 基本統計
            total_trades = len(trades)
            buy_trades = sum(1 for t in trades if t.trade_type == TradeType.BUY)
            sell_trades = sum(1 for t in trades if t.trade_type == TradeType.SELL)
            
            # 手数料分析
            total_commission = sum(t.commission for t in trades)
            avg_commission = total_commission / Decimal(total_trades) if total_trades > 0 else Decimal("0")
            
            # 取引サイズ分析
            trade_values = [
                t.price * Decimal(t.quantity) for t in trades
            ]
            avg_trade_value = sum(trade_values) / len(trade_values) if trade_values else Decimal("0")
            
            # 保有期間分析（売買ペアから推定）
            holding_periods = self._calculate_holding_periods(trades)
            avg_holding_days = (
                sum(holding_periods) / len(holding_periods) if holding_periods else 0
            )
            
            # 取引頻度分析
            trading_frequency = self._calculate_trading_frequency(trades)
            
            efficiency_metrics = {
                "total_trades": total_trades,
                "buy_trades": buy_trades,
                "sell_trades": sell_trades,
                "total_commission": total_commission,
                "average_commission": avg_commission,
                "commission_percentage": (
                    total_commission / sum(trade_values) * 100 if sum(trade_values) > 0 else Decimal("0")
                ),
                "average_trade_value": avg_trade_value,
                "average_holding_days": Decimal(str(avg_holding_days)),
                "trading_frequency": trading_frequency,
                "trade_balance_ratio": Decimal(buy_trades) / Decimal(sell_trades) if sell_trades > 0 else Decimal("0"),
            }
            
            logger.info(f"取引効率分析完了: {total_trades}取引, 平均手数料{avg_commission:.0f}円")
            return efficiency_metrics
            
        except Exception as e:
            logger.error(f"取引効率分析エラー: {e}")
            return {}

    def _calculate_holding_periods(self, trades: List[Trade]) -> List[int]:
        """保有期間計算"""
        holding_periods = []
        
        # 銘柄別に取引をグループ化
        symbol_trades = {}
        for trade in trades:
            if trade.symbol not in symbol_trades:
                symbol_trades[trade.symbol] = []
            symbol_trades[trade.symbol].append(trade)
        
        # 銘柄別に売買ペアを分析
        for symbol, symbol_trade_list in symbol_trades.items():
            symbol_trade_list.sort(key=lambda t: t.timestamp)
            
            buy_queue = []  # FIFO買いキュー
            
            for trade in symbol_trade_list:
                if trade.trade_type == TradeType.BUY:
                    buy_queue.append(trade)
                elif trade.trade_type == TradeType.SELL:
                    # 売却分を買いキューから消費
                    remaining_quantity = trade.quantity
                    
                    while remaining_quantity > 0 and buy_queue:
                        buy_trade = buy_queue[0]
                        
                        if buy_trade.quantity <= remaining_quantity:
                            # 完全消費
                            days_held = (trade.timestamp - buy_trade.timestamp).days
                            holding_periods.append(days_held)
                            
                            remaining_quantity -= buy_trade.quantity
                            buy_queue.pop(0)
                        else:
                            # 部分消費
                            days_held = (trade.timestamp - buy_trade.timestamp).days
                            holding_periods.append(days_held)
                            
                            buy_trade.quantity -= remaining_quantity
                            remaining_quantity = 0
        
        return holding_periods

    def _calculate_trading_frequency(self, trades: List[Trade]) -> Dict[str, Decimal]:
        """取引頻度分析"""
        if not trades:
            return {}
        
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        start_date = sorted_trades[0].timestamp
        end_date = sorted_trades[-1].timestamp
        
        total_days = (end_date - start_date).days + 1
        total_weeks = total_days / 7
        total_months = total_days / 30.44
        
        frequency = {
            "trades_per_day": Decimal(len(trades)) / Decimal(total_days) if total_days > 0 else Decimal("0"),
            "trades_per_week": Decimal(len(trades)) / Decimal(str(total_weeks)) if total_weeks > 0 else Decimal("0"),
            "trades_per_month": Decimal(len(trades)) / Decimal(str(total_months)) if total_months > 0 else Decimal("0"),
            "trading_days": Decimal(total_days),
        }
        
        return frequency

    def calculate_risk_metrics(
        self,
        trades: List[Trade],
        positions: Dict[str, Position],
    ) -> Dict[str, Decimal]:
        """
        リスク指標計算

        Args:
            trades: 取引履歴
            positions: 現在のポジション

        Returns:
            リスク指標辞書
        """
        try:
            # ポートフォリオ価値
            total_value = sum(pos.market_value for pos in positions.values())
            
            # VaR（Value at Risk）簡易計算
            var_95 = self._calculate_var(trades, 0.95)
            
            # 集中リスク
            concentration_risk = self._calculate_concentration_risk(positions, total_value)
            
            # ベータ（市場との相関、簡易版）
            portfolio_beta = self._calculate_portfolio_beta(trades)
            
            # ボラティリティ
            volatility = self._calculate_portfolio_volatility(trades)
            
            risk_metrics = {
                "portfolio_value": total_value,
                "value_at_risk_95": var_95,
                "concentration_risk_percentage": concentration_risk,
                "portfolio_beta": portfolio_beta,
                "annualized_volatility": volatility,
                "risk_adjusted_return": self._calculate_risk_adjusted_return(trades, volatility),
            }
            
            logger.info(f"リスク指標計算完了: VaR {var_95:.0f}円, 集中リスク {concentration_risk:.1f}%")
            return risk_metrics
            
        except Exception as e:
            logger.error(f"リスク指標計算エラー: {e}")
            return {}

    def _calculate_var(self, trades: List[Trade], confidence_level: float) -> Decimal:
        """VaR計算（簡易版）"""
        daily_returns = self._calculate_daily_returns(trades)
        
        if len(daily_returns) < 10:
            return Decimal("0")
        
        # 指定信頼水準のパーセンタイル
        var_percentile = (1 - confidence_level) * 100
        var_value = Decimal(str(statistics.quantiles(daily_returns, n=100)[int(var_percentile)]))
        
        return abs(var_value)

    def _calculate_concentration_risk(
        self,
        positions: Dict[str, Position],
        total_value: Decimal,
    ) -> Decimal:
        """集中リスク計算"""
        if total_value == 0:
            return Decimal("0")
        
        # 最大ポジション比率
        max_position_value = max((pos.market_value for pos in positions.values()), default=Decimal("0"))
        concentration_percentage = max_position_value / total_value * 100
        
        return concentration_percentage

    def _calculate_portfolio_beta(self, trades: List[Trade]) -> Decimal:
        """ポートフォリオベータ計算（簡易版）"""
        # 市場指数データが必要なため、簡易的に1.0を返す
        # 実際の実装では、日経平均等との相関を計算
        return Decimal("1.0")

    def _calculate_portfolio_volatility(self, trades: List[Trade]) -> Decimal:
        """ポートフォリオボラティリティ計算"""
        daily_returns = self._calculate_daily_returns(trades)
        
        if len(daily_returns) < 2:
            return Decimal("0")
        
        # 標準偏差を年率換算
        std_dev = statistics.stdev(daily_returns)
        annualized_volatility = Decimal(str(std_dev)) * Decimal("15.81")  # √252
        
        return annualized_volatility

    def _calculate_risk_adjusted_return(self, trades: List[Trade], volatility: Decimal) -> Decimal:
        """リスク調整後リターン計算"""
        if volatility == 0:
            return Decimal("0")
        
        # 簡易的なリターン計算
        total_return = self._calculate_realized_pnl(trades)
        risk_adjusted_return = total_return / volatility if volatility > 0 else Decimal("0")
        
        return risk_adjusted_return

    def generate_performance_report(
        self,
        trades: List[Trade],
        positions: Dict[str, Position],
        period_days: int = 30,
    ) -> Dict[str, any]:
        """
        総合パフォーマンスレポート生成

        Args:
            trades: 取引履歴
            positions: 現在のポジション
            period_days: 分析期間（日数）

        Returns:
            総合レポート辞書
        """
        try:
            # 期間フィルタリング
            cutoff_date = datetime.now() - timedelta(days=period_days)
            period_trades = [t for t in trades if t.timestamp >= cutoff_date]
            
            # 各種分析実行
            performance = self.calculate_portfolio_performance(period_trades, positions)
            efficiency = self.analyze_trade_efficiency(period_trades)
            risk_metrics = self.calculate_risk_metrics(period_trades, positions)
            
            # レポート統合
            report = {
                "report_period": {
                    "days": period_days,
                    "start_date": cutoff_date.isoformat(),
                    "end_date": datetime.now().isoformat(),
                },
                "performance_metrics": performance,
                "efficiency_metrics": efficiency,
                "risk_metrics": risk_metrics,
                "summary": {
                    "total_trades_analyzed": len(period_trades),
                    "current_positions": len(positions),
                    "overall_score": self._calculate_overall_score(performance, risk_metrics),
                },
            }
            
            logger.info(f"パフォーマンスレポート生成完了: {period_days}日間, {len(period_trades)}取引")
            return report
            
        except Exception as e:
            logger.error(f"パフォーマンスレポート生成エラー: {e}")
            return {
                "error": str(e),
                "report_timestamp": datetime.now().isoformat(),
            }

    def _calculate_overall_score(
        self,
        performance: Dict[str, Decimal],
        risk_metrics: Dict[str, Decimal],
    ) -> Decimal:
        """総合スコア計算"""
        try:
            # 各指標を0-100でスコア化
            return_score = min(max(performance.get("return_percentage", Decimal("0")) + 50, 0), 100)
            
            sharpe_score = min(max(performance.get("sharpe_ratio", Decimal("0")) * 20 + 50, 0), 100)
            
            risk_score = max(100 - risk_metrics.get("concentration_risk_percentage", Decimal("0")), 0)
            
            # 加重平均スコア
            overall_score = (return_score * 0.4 + sharpe_score * 0.4 + risk_score * 0.2)
            
            return overall_score
            
        except Exception:
            return Decimal("50")  # 中性スコア