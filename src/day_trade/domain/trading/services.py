"""
取引ドメインサービス

複数のエンティティにまたがるビジネスロジックを実装
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass

from .entities import Trade, Position, Portfolio, TradeId, PositionId
from .repositories import UnitOfWork, MarketDataRepository
from ..common.value_objects import Symbol, Price, Quantity, Money, Percentage
from ..common.domain_events import (
    DomainEvent, TradeExecutedEvent, PositionOpenedEvent, 
    PositionClosedEvent, RiskThresholdExceededEvent
)


@dataclass
class RiskAssessmentResult:
    """リスク評価結果"""
    is_acceptable: bool
    risk_score: Decimal
    violated_rules: List[str]
    recommendations: List[str]
    max_position_size: Optional[Quantity] = None


@dataclass
class TradingOpportunity:
    """取引機会"""
    symbol: Symbol
    direction: str  # "buy" or "sell"
    confidence: Percentage
    target_price: Price
    stop_loss: Price
    reasoning: str
    risk_reward_ratio: Decimal


class TradingDomainService:
    """取引ドメインサービス"""
    
    def __init__(self, unit_of_work: UnitOfWork):
        self.uow = unit_of_work
    
    def execute_trade(
        self,
        portfolio: Portfolio,
        symbol: Symbol,
        direction: str,
        quantity: Quantity,
        price: Price,
        commission: Money
    ) -> Tuple[Trade, List[DomainEvent]]:
        """取引実行"""
        # 取引作成
        trade_id = TradeId()
        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            quantity=quantity,
            price=price,
            direction=direction,
            executed_at=datetime.now(),
            commission=commission
        )
        
        # ポートフォリオで取引実行
        position = portfolio.execute_trade(trade)
        
        # イベント収集
        events = []
        events.extend(portfolio.get_domain_events())
        portfolio.clear_domain_events()
        
        return trade, events
    
    def calculate_position_size(
        self,
        portfolio: Portfolio,
        symbol: Symbol,
        risk_percentage: Percentage,
        stop_loss_price: Price,
        current_price: Price
    ) -> Quantity:
        """ポジションサイズ計算"""
        # ポートフォリオ総額
        current_prices = {symbol: current_price}
        total_value = portfolio.calculate_total_value(current_prices)
        
        # リスク許容額
        risk_amount = risk_percentage.apply_to_money(total_value)
        
        # 1株あたりのリスク
        price_diff = abs(current_price.value - stop_loss_price.value)
        if price_diff == 0:
            return Quantity(100)  # デフォルト
        
        # 最大ポジションサイズ
        max_shares = risk_amount.amount / price_diff
        
        return Quantity(int(max_shares))
    
    def validate_trade_constraints(
        self,
        portfolio: Portfolio,
        trade: Trade
    ) -> bool:
        """取引制約検証"""
        # 最大ポジション数チェック
        if len(portfolio.get_all_positions()) >= 10:  # 設定可能にする
            return False
        
        # 銘柄別最大ポジションサイズチェック
        existing_position = portfolio.get_position(trade.symbol)
        if existing_position:
            total_quantity = existing_position.calculate_current_quantity()
            if trade.is_buy and total_quantity.value + trade.quantity.value > 1000:  # 設定可能
                return False
        
        return True


class RiskManagementService:
    """リスク管理サービス"""
    
    def __init__(self, unit_of_work: UnitOfWork):
        self.uow = unit_of_work
    
    def assess_portfolio_risk(
        self,
        portfolio: Portfolio,
        current_prices: Dict[Symbol, Price]
    ) -> RiskAssessmentResult:
        """ポートフォリオリスク評価"""
        violated_rules = []
        recommendations = []
        
        # 総損益計算
        total_pnl = portfolio.calculate_total_pnl(current_prices)
        total_value = portfolio.calculate_total_value(current_prices)
        
        risk_score = Decimal('0')
        
        # 集中リスクチェック
        positions = portfolio.get_all_positions()
        if len(positions) > 0 and total_value.amount > Decimal('0'):
            max_position_ratio = self._calculate_max_position_ratio(positions, current_prices, total_value)
            if max_position_ratio > Decimal('0.3'):  # 30%
                violated_rules.append("単一銘柄の集中度が30%を超えています")
                risk_score += Decimal('0.3')
        
        # 損失限度チェック
        if total_pnl.amount < Money(Decimal('-100000')).amount:  # -10万円
            violated_rules.append("総損失が限度を超えています")
            recommendations.append("損失拡大を防ぐため、一部ポジションの決済を検討してください")
            risk_score += Decimal('0.5')
        
        # ドローダウンチェック
        # 実装は省略（履歴データが必要）
        
        return RiskAssessmentResult(
            is_acceptable=len(violated_rules) == 0,
            risk_score=risk_score,
            violated_rules=violated_rules,
            recommendations=recommendations
        )
    
    def _calculate_max_position_ratio(
        self,
        positions: List[Position],
        current_prices: Dict[Symbol, Price],
        total_value: Money
    ) -> Decimal:
        """最大ポジション比率計算"""
        max_ratio = Decimal('0')
        
        for position in positions:
            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]
                qty = position.calculate_current_quantity()
                position_value = current_price.value * Decimal(str(qty.value))
                
                ratio = position_value / total_value.amount if total_value.amount > 0 else Decimal('0')
                max_ratio = max(max_ratio, ratio)
        
        return max_ratio
    
    def calculate_var(
        self,
        portfolio: Portfolio,
        confidence_level: Percentage = Percentage(Decimal('95')),
        time_horizon_days: int = 1
    ) -> Money:
        """バリュー・アット・リスク計算"""
        # 簡易実装（実際にはMonte Carloシミュレーション等を使用）
        current_prices = {}  # 実際の価格データが必要
        total_value = portfolio.calculate_total_value(current_prices)
        
        # 仮の日次変動率（2%）
        daily_volatility = Percentage(Decimal('2'))
        var_amount = daily_volatility.apply_to_money(total_value)
        
        return var_amount


class PerformanceAnalysisService:
    """パフォーマンス分析サービス"""
    
    def __init__(self, unit_of_work: UnitOfWork):
        self.uow = unit_of_work
    
    def calculate_sharpe_ratio(
        self,
        returns: List[Decimal],
        risk_free_rate: Decimal = Decimal('0.001')  # 0.1%
    ) -> Decimal:
        """シャープレシオ計算"""
        if not returns:
            return Decimal('0')
        
        avg_return = sum(returns) / len(returns)
        excess_return = avg_return - risk_free_rate
        
        if len(returns) < 2:
            return Decimal('0')
        
        # 標準偏差計算
        variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = variance.sqrt()
        
        if std_dev == 0:
            return Decimal('0')
        
        return excess_return / std_dev
    
    def calculate_max_drawdown(self, portfolio_values: List[Decimal]) -> Tuple[Decimal, int, int]:
        """最大ドローダウン計算"""
        if not portfolio_values:
            return Decimal('0'), 0, 0
        
        peak = portfolio_values[0]
        max_drawdown = Decimal('0')
        peak_index = 0
        trough_index = 0
        
        for i, value in enumerate(portfolio_values):
            if value > peak:
                peak = value
                peak_index = i
            
            drawdown = (peak - value) / peak if peak > 0 else Decimal('0')
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                trough_index = i
        
        return max_drawdown, peak_index, trough_index
    
    def calculate_win_rate(self, trades: List[Trade]) -> Percentage:
        """勝率計算"""
        if not trades:
            return Percentage(Decimal('0'))
        
        profitable_trades = 0
        total_trades = len(trades)
        
        # 簡易実装（実際には損益計算が必要）
        for trade in trades:
            # ここでは仮の判定
            if trade.is_buy:  # 買い取引が利益と仮定
                profitable_trades += 1
        
        win_rate = (Decimal(str(profitable_trades)) / Decimal(str(total_trades))) * Decimal('100')
        return Percentage(win_rate)


class MarketAnalysisService:
    """市場分析サービス"""
    
    def __init__(self, market_data_repo: MarketDataRepository):
        self.market_data_repo = market_data_repo
    
    def identify_trading_opportunities(
        self,
        symbols: List[Symbol],
        analysis_criteria: Dict[str, Any]
    ) -> List[TradingOpportunity]:
        """取引機会特定"""
        opportunities = []
        
        for symbol in symbols:
            market_data = self.market_data_repo.get_market_data(symbol)
            if not market_data:
                continue
            
            # 簡易分析（実際にはテクニカル分析等を実装）
            opportunity = self._analyze_symbol(symbol, market_data, analysis_criteria)
            if opportunity:
                opportunities.append(opportunity)
        
        return opportunities
    
    def _analyze_symbol(
        self,
        symbol: Symbol,
        market_data: Dict[str, Any],
        criteria: Dict[str, Any]
    ) -> Optional[TradingOpportunity]:
        """銘柄分析"""
        current_price = Price(Decimal(str(market_data.get('price', 0))))
        volume = market_data.get('volume', 0)
        
        # 簡易判定（実際にはより複雑な分析）
        if volume > criteria.get('min_volume', 1000):
            return TradingOpportunity(
                symbol=symbol,
                direction="buy",
                confidence=Percentage(Decimal('60')),
                target_price=Price(current_price.value * Decimal('1.05')),
                stop_loss=Price(current_price.value * Decimal('0.97')),
                reasoning="Volume breakout detected",
                risk_reward_ratio=Decimal('1.67')
            )
        
        return None
    
    def calculate_correlation_matrix(
        self,
        symbols: List[Symbol],
        period_days: int = 30
    ) -> Dict[Tuple[Symbol, Symbol], Decimal]:
        """相関マトリックス計算"""
        # 実装は省略（価格履歴データから相関係数を計算）
        correlation_matrix = {}
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i:], i):
                if symbol1 == symbol2:
                    correlation = Decimal('1.0')
                else:
                    # 仮の相関値
                    correlation = Decimal('0.3')
                
                correlation_matrix[(symbol1, symbol2)] = correlation
        
        return correlation_matrix


class PortfolioOptimizationService:
    """ポートフォリオ最適化サービス"""
    
    def __init__(self, unit_of_work: UnitOfWork):
        self.uow = unit_of_work
    
    def optimize_portfolio_weights(
        self,
        symbols: List[Symbol],
        expected_returns: Dict[Symbol, Decimal],
        risk_tolerance: Decimal
    ) -> Dict[Symbol, Percentage]:
        """ポートフォリオ重み最適化"""
        # 簡易均等分割（実際には平均分散最適化等を実装）
        equal_weight = Percentage(Decimal('100') / Decimal(str(len(symbols))))
        
        optimized_weights = {}
        for symbol in symbols:
            optimized_weights[symbol] = equal_weight
        
        return optimized_weights
    
    def rebalance_recommendations(
        self,
        current_portfolio: Portfolio,
        target_weights: Dict[Symbol, Percentage],
        current_prices: Dict[Symbol, Price]
    ) -> List[Dict[str, Any]]:
        """リバランス推奨計算"""
        recommendations = []
        
        total_value = current_portfolio.calculate_total_value(current_prices)
        
        for symbol, target_weight in target_weights.items():
            target_value = target_weight.apply_to_money(total_value)
            
            position = current_portfolio.get_position(symbol)
            current_value = Money(Decimal('0'))
            
            if position and symbol in current_prices:
                current_price = current_prices[symbol]
                qty = position.calculate_current_quantity()
                current_value = current_price.to_money(qty)
            
            diff = target_value.amount - current_value.amount
            
            if abs(diff) > total_value.amount * Decimal('0.05'):  # 5%以上の差
                action = "buy" if diff > 0 else "sell"
                amount = abs(diff)
                
                recommendations.append({
                    'symbol': symbol,
                    'action': action,
                    'amount': amount,
                    'current_weight': (current_value.amount / total_value.amount) * 100,
                    'target_weight': target_weight.value,
                    'reason': f'Rebalance to target allocation'
                })
        
        return recommendations