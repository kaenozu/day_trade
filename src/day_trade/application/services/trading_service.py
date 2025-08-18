"""
取引アプリケーションサービス

ユースケースを実装し、ドメインとインフラストラクチャを協調
"""

from typing import List, Optional, Dict, Any, Tuple
from decimal import Decimal
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass

from ...domain.trading.entities import Trade, Position, Portfolio, TradeId, PositionId
from ...domain.trading.repositories import UnitOfWork
from ...domain.trading.services import (
    TradingDomainService, RiskManagementService,
    PerformanceAnalysisService, MarketAnalysisService,
    PortfolioOptimizationService, TradingOpportunity, RiskAssessmentResult
)
from ...domain.common.value_objects import Symbol, Price, Quantity, Money, Percentage
from ...domain.common.domain_events import domain_event_dispatcher
from ...core.error_handling.enhanced_error_system import (
    ValidationError, BusinessRuleViolationError, InsufficientDataError
)


@dataclass
class TradeRequest:
    """取引リクエスト"""
    symbol: str
    direction: str  # "buy" or "sell"
    quantity: int
    price: Optional[Decimal] = None  # None の場合は成行
    order_type: str = "market"  # "market" or "limit"
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None


@dataclass
class TradeResult:
    """取引結果"""
    success: bool
    trade_id: Optional[UUID] = None
    error_message: Optional[str] = None
    position_id: Optional[UUID] = None
    executed_price: Optional[Decimal] = None
    commission: Optional[Decimal] = None


@dataclass
class PortfolioSummary:
    """ポートフォリオサマリー"""
    total_value: Decimal
    total_pnl: Decimal
    open_positions_count: int
    cash_balance: Decimal
    day_change: Decimal
    day_change_percent: Decimal


class TradingApplicationService:
    """取引アプリケーションサービス"""

    def __init__(self, unit_of_work: UnitOfWork):
        self.uow = unit_of_work
        self.trading_service = TradingDomainService(unit_of_work)
        self.risk_service = RiskManagementService(unit_of_work)
        self.performance_service = PerformanceAnalysisService(unit_of_work)
        self.market_service = MarketAnalysisService(unit_of_work.market_data)
        self.optimization_service = PortfolioOptimizationService(unit_of_work)

    def execute_trade(
        self,
        portfolio_id: UUID,
        trade_request: TradeRequest
    ) -> TradeResult:
        """取引実行"""
        try:
            with self.uow:
                # ポートフォリオ取得
                portfolio = self.uow.portfolios.find_by_id(portfolio_id)
                if not portfolio:
                    return TradeResult(
                        success=False,
                        error_message="ポートフォリオが見つかりません"
                    )

                # バリデーション
                validation_error = self._validate_trade_request(trade_request)
                if validation_error:
                    return TradeResult(success=False, error_message=validation_error)

                # 価格取得
                symbol = Symbol(trade_request.symbol)
                current_price = self._get_execution_price(symbol, trade_request)

                # リスク評価
                risk_result = self._assess_trade_risk(
                    portfolio, symbol, trade_request, current_price
                )
                if not risk_result.is_acceptable:
                    return TradeResult(
                        success=False,
                        error_message=f"リスク制限に違反: {', '.join(risk_result.violated_rules)}"
                    )

                # 手数料計算
                commission = self._calculate_commission(
                    current_price.to_money(Quantity(trade_request.quantity))
                )

                # 取引実行
                trade, events = self.trading_service.execute_trade(
                    portfolio=portfolio,
                    symbol=symbol,
                    direction=trade_request.direction,
                    quantity=Quantity(trade_request.quantity),
                    price=current_price,
                    commission=commission
                )

                # 永続化
                self.uow.trades.save(trade)
                self.uow.portfolios.save(portfolio)

                # イベント発行
                domain_event_dispatcher.dispatch_all(events)

                # 監査ログ
                self.uow.audit_log.log_action(
                    action="trade_executed",
                    details={
                        "trade_id": str(trade.id.value),
                        "symbol": trade_request.symbol,
                        "direction": trade_request.direction,
                        "quantity": trade_request.quantity,
                        "price": str(current_price.value),
                        "commission": str(commission.amount)
                    }
                )

                self.uow.commit()

                return TradeResult(
                    success=True,
                    trade_id=trade.id.value,
                    position_id=portfolio.get_position(symbol).id.value,
                    executed_price=current_price.value,
                    commission=commission.amount
                )

        except Exception as e:
            self.uow.rollback()
            return TradeResult(
                success=False,
                error_message=f"取引実行エラー: {str(e)}"
            )

    def get_portfolio_summary(
        self,
        portfolio_id: UUID
    ) -> Optional[PortfolioSummary]:
        """ポートフォリオサマリー取得"""
        try:
            portfolio = self.uow.portfolios.find_by_id(portfolio_id)
            if not portfolio:
                return None

            # 現在価格取得
            positions = portfolio.get_all_positions()
            current_prices = {}

            for position in positions:
                price_data = self.uow.market_data.get_current_price(position.symbol)
                if price_data:
                    current_prices[position.symbol] = Price(
                        Decimal(str(price_data['price']))
                    )

            # サマリー計算
            total_value = portfolio.calculate_total_value(current_prices)
            total_pnl = portfolio.calculate_total_pnl(current_prices)

            # 日次変動計算（簡易実装）
            day_change = Money(Decimal('0'))  # 実際には前日終値との差を計算
            day_change_percent = Decimal('0')

            return PortfolioSummary(
                total_value=total_value.amount,
                total_pnl=total_pnl.amount,
                open_positions_count=len([p for p in positions if not p.is_closed]),
                cash_balance=Decimal('0'),  # 実装必要
                day_change=day_change.amount,
                day_change_percent=day_change_percent
            )

        except Exception as e:
            raise InsufficientDataError(f"ポートフォリオサマリー取得エラー: {e}")

    def get_trading_opportunities(
        self,
        max_results: int = 10
    ) -> List[TradingOpportunity]:
        """取引機会取得"""
        try:
            # ウォッチリスト取得
            watchlist = self.uow.configuration.get_watchlist()

            # 分析基準取得
            config = self.uow.configuration.get_trading_config()
            analysis_criteria = {
                'min_volume': config.get('min_volume', 1000),
                'volatility_threshold': config.get('volatility_threshold', 0.02),
                'trend_strength': config.get('trend_strength', 0.6)
            }

            # 機会分析
            opportunities = self.market_service.identify_trading_opportunities(
                watchlist, analysis_criteria
            )

            # 結果をソート（信頼度降順）
            opportunities.sort(
                key=lambda x: x.confidence.value,
                reverse=True
            )

            return opportunities[:max_results]

        except Exception as e:
            raise InsufficientDataError(f"取引機会分析エラー: {e}")

    def assess_portfolio_risk(
        self,
        portfolio_id: UUID
    ) -> RiskAssessmentResult:
        """ポートフォリオリスク評価"""
        try:
            portfolio = self.uow.portfolios.find_by_id(portfolio_id)
            if not portfolio:
                raise ValidationError("ポートフォリオが見つかりません")

            # 現在価格取得
            positions = portfolio.get_all_positions()
            current_prices = {}

            for position in positions:
                price_data = self.uow.market_data.get_current_price(position.symbol)
                if price_data:
                    current_prices[position.symbol] = Price(
                        Decimal(str(price_data['price']))
                    )

            # リスク評価
            return self.risk_service.assess_portfolio_risk(portfolio, current_prices)

        except Exception as e:
            raise InsufficientDataError(f"リスク評価エラー: {e}")

    def optimize_portfolio(
        self,
        portfolio_id: UUID,
        optimization_method: str = "equal_weight"
    ) -> List[Dict[str, Any]]:
        """ポートフォリオ最適化"""
        try:
            portfolio = self.uow.portfolios.find_by_id(portfolio_id)
            if not portfolio:
                raise ValidationError("ポートフォリオが見つかりません")

            # 現在のポジション
            positions = portfolio.get_all_positions()
            symbols = [pos.symbol for pos in positions]

            if not symbols:
                return []

            # 期待リターン計算（簡易実装）
            expected_returns = {}
            for symbol in symbols:
                expected_returns[symbol] = Decimal('0.08')  # 8%固定

            # 最適化実行
            config = self.uow.configuration.get_risk_config()
            risk_tolerance = Decimal(str(config.get('risk_tolerance', 0.1)))

            optimal_weights = self.optimization_service.optimize_portfolio_weights(
                symbols, expected_returns, risk_tolerance
            )

            # 現在価格取得
            current_prices = {}
            for symbol in symbols:
                price_data = self.uow.market_data.get_current_price(symbol)
                if price_data:
                    current_prices[symbol] = Price(Decimal(str(price_data['price'])))

            # リバランス推奨
            recommendations = self.optimization_service.rebalance_recommendations(
                portfolio, optimal_weights, current_prices
            )

            return recommendations

        except Exception as e:
            raise InsufficientDataError(f"ポートフォリオ最適化エラー: {e}")

    def get_performance_metrics(
        self,
        portfolio_id: UUID,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """パフォーマンスメトリクス取得"""
        try:
            # パフォーマンスデータ取得
            performance_data = self.uow.performance.get_performance_summary(
                start_date, end_date
            )

            # 取引履歴取得
            trades = self.uow.trades.find_by_date_range(start_date, end_date)

            # メトリクス計算
            returns = []  # 実際には日次リターンを計算
            portfolio_values = []  # 実際にはポートフォリオ価値履歴を取得

            metrics = {
                'total_return': performance_data.get('total_return', 0),
                'sharpe_ratio': self.performance_service.calculate_sharpe_ratio(returns),
                'max_drawdown': self.performance_service.calculate_max_drawdown(portfolio_values)[0],
                'win_rate': self.performance_service.calculate_win_rate(trades).value,
                'total_trades': len(trades),
                'avg_trade_size': sum(t.total_value.amount for t in trades) / len(trades) if trades else 0
            }

            return metrics

        except Exception as e:
            raise InsufficientDataError(f"パフォーマンス分析エラー: {e}")

    def _validate_trade_request(self, request: TradeRequest) -> Optional[str]:
        """取引リクエストバリデーション"""
        if not request.symbol or not request.symbol.strip():
            return "銘柄コードは必須です"

        if request.direction not in ['buy', 'sell']:
            return "取引方向は'buy'または'sell'である必要があります"

        if request.quantity <= 0:
            return "数量は正の値である必要があります"

        if request.price is not None and request.price <= 0:
            return "価格は正の値である必要があります"

        return None

    def _get_execution_price(self, symbol: Symbol, request: TradeRequest) -> Price:
        """実行価格取得"""
        if request.order_type == "limit" and request.price:
            return Price(request.price)

        # 成行注文の場合、現在価格を取得
        market_data = self.uow.market_data.get_current_price(symbol)
        if not market_data:
            raise InsufficientDataError(f"価格データが取得できません: {symbol}")

        return Price(Decimal(str(market_data['price'])))

    def _assess_trade_risk(
        self,
        portfolio: Portfolio,
        symbol: Symbol,
        request: TradeRequest,
        price: Price
    ) -> RiskAssessmentResult:
        """取引リスク評価"""
        # 制約チェック
        if not self.trading_service.validate_trade_constraints(portfolio, Trade(
            trade_id=TradeId(),
            symbol=symbol,
            quantity=Quantity(request.quantity),
            price=price,
            direction=request.direction,
            executed_at=datetime.now(),
            commission=Money(Decimal('0'))
        )):
            return RiskAssessmentResult(
                is_acceptable=False,
                risk_score=Decimal('1.0'),
                violated_rules=["取引制約に違反しています"],
                recommendations=["ポジションサイズを調整してください"]
            )

        return RiskAssessmentResult(
            is_acceptable=True,
            risk_score=Decimal('0.1'),
            violated_rules=[],
            recommendations=[]
        )

    def _calculate_commission(self, trade_amount: Money) -> Money:
        """手数料計算"""
        # 簡易手数料計算（0.1%）
        commission_rate = Decimal('0.001')
        commission_amount = trade_amount.amount * commission_rate
        return Money(commission_amount)