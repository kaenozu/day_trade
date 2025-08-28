"""
舟券管理システム

舟券の購入、結果確認、収支管理を行う
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

from ..core.data_models import PredictionResult
from ..data.database import Database, get_database, BettingTicket, Race, RaceResult

logger = logging.getLogger(__name__)


class BetType(Enum):
    """舟券種別"""
    WIN = "win"                    # 単勝
    PLACE_SHOW = "place_show"      # 複勝
    EXACTA = "exacta"              # 2連単
    QUINELLA = "quinella"          # 2連複
    TRIFECTA = "trifecta"          # 3連単
    TRIO = "trio"                  # 3連複


@dataclass
class BettingResult:
    """投票結果"""
    ticket_id: int
    race_id: str
    bet_type: str
    numbers: str
    amount: Decimal
    is_hit: bool
    payout: Decimal
    profit: Decimal
    return_rate: float


class TicketManager:
    """舟券管理クラス"""
    
    def __init__(self, database: Optional[Database] = None):
        """
        初期化
        
        Args:
            database: データベース
        """
        self.database = database or get_database()
    
    def purchase_ticket(self,
                       race_id: str,
                       bet_type: BetType,
                       numbers: str,
                       amount: Decimal,
                       strategy_name: Optional[str] = None,
                       confidence: Optional[Decimal] = None) -> int:
        """
        舟券を購入（記録）
        
        Args:
            race_id: レースID
            bet_type: 舟券種別
            numbers: 買い目（例: "1-2-3"）
            amount: 購入金額
            strategy_name: 戦略名
            confidence: 信頼度
            
        Returns:
            チケットID
        """
        with self.database.get_session() as session:
            ticket = BettingTicket(
                race_id=race_id,
                bet_type=bet_type.value,
                numbers=numbers,
                amount=amount,
                strategy_name=strategy_name,
                confidence=confidence,
                purchased_at=datetime.now()
            )
            
            session.add(ticket)
            session.flush()  # IDを取得するためにflush
            ticket_id = ticket.id
            
            logger.info(f"舟券購入記録: {race_id} {bet_type.value} {numbers} {amount}円")
            return ticket_id
    
    def purchase_from_prediction(self,
                                prediction: PredictionResult,
                                budget: Decimal,
                                strategy_name: str = "AI予想") -> List[int]:
        """
        予想結果から舟券を購入
        
        Args:
            prediction: 予想結果
            budget: 予算
            strategy_name: 戦略名
            
        Returns:
            購入したチケットIDリスト
        """
        ticket_ids = []
        
        if not prediction.recommended_bets:
            logger.warning(f"推奨買い目がありません: {prediction.race_id}")
            return ticket_ids
        
        # 買い目数で予算を分配
        bet_count = len(prediction.recommended_bets)
        amount_per_bet = budget / bet_count
        
        for bet_info in prediction.recommended_bets:
            try:
                bet_type = BetType(bet_info['bet_type'])
                ticket_id = self.purchase_ticket(
                    race_id=prediction.race_id,
                    bet_type=bet_type,
                    numbers=bet_info['numbers'],
                    amount=amount_per_bet,
                    strategy_name=strategy_name,
                    confidence=Decimal(str(bet_info.get('confidence', 0.5)))
                )
                ticket_ids.append(ticket_id)
                
            except Exception as e:
                logger.error(f"舟券購入エラー: {e}")
                continue
        
        return ticket_ids
    
    def update_results(self, race_id: str) -> List[BettingResult]:
        """
        レース結果で舟券の当否を更新
        
        Args:
            race_id: レースID
            
        Returns:
            更新された投票結果
        """
        with self.database.get_session() as session:
            # 該当レースの舟券取得
            tickets = session.query(BettingTicket).filter(
                BettingTicket.race_id == race_id,
                BettingTicket.is_hit.is_(None)  # 未確定のもの
            ).all()
            
            if not tickets:
                return []
            
            # レース結果取得
            race_result = session.query(RaceResult).filter(
                RaceResult.race_id == race_id
            ).first()
            
            if not race_result:
                logger.warning(f"レース結果が見つかりません: {race_id}")
                return []
            
            results = []
            
            for ticket in tickets:
                # 当否判定
                is_hit = self._check_hit(ticket, race_result)
                payout = self._calculate_payout(ticket, race_result) if is_hit else Decimal('0')
                profit = payout - ticket.amount
                
                # 結果更新
                ticket.is_hit = is_hit
                ticket.payout = payout
                ticket.profit = profit
                
                # 結果オブジェクト作成
                return_rate = float(payout / ticket.amount) if ticket.amount > 0 else 0.0
                
                result = BettingResult(
                    ticket_id=ticket.id,
                    race_id=ticket.race_id,
                    bet_type=ticket.bet_type,
                    numbers=ticket.numbers,
                    amount=ticket.amount,
                    is_hit=is_hit,
                    payout=payout,
                    profit=profit,
                    return_rate=return_rate
                )
                results.append(result)
            
            logger.info(f"結果更新完了: {race_id} {len(results)} tickets")
            return results
    
    def get_portfolio_summary(self, 
                            days_back: int = 30) -> Dict[str, Any]:
        """
        投票成績サマリーを取得
        
        Args:
            days_back: 集計対象日数
            
        Returns:
            成績サマリー
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        with self.database.get_session() as session:
            tickets = session.query(BettingTicket).filter(
                BettingTicket.purchased_at >= cutoff_date
            ).all()
            
            if not tickets:
                return {'error': '投票データがありません'}
            
            # 基本統計
            total_tickets = len(tickets)
            total_investment = sum(ticket.amount for ticket in tickets)
            confirmed_tickets = [t for t in tickets if t.is_hit is not None]
            hit_tickets = [t for t in confirmed_tickets if t.is_hit]
            
            total_payout = sum(ticket.payout for ticket in confirmed_tickets)
            total_profit = sum(ticket.profit for ticket in confirmed_tickets)
            
            # 勝率・回収率
            hit_rate = len(hit_tickets) / len(confirmed_tickets) * 100 if confirmed_tickets else 0
            return_rate = float(total_payout / total_investment) * 100 if total_investment > 0 else 0
            
            # 舟券種別別統計
            bet_type_stats = defaultdict(lambda: {'count': 0, 'investment': Decimal('0'), 'payout': Decimal('0')})
            
            for ticket in confirmed_tickets:
                bet_type_stats[ticket.bet_type]['count'] += 1
                bet_type_stats[ticket.bet_type]['investment'] += ticket.amount
                bet_type_stats[ticket.bet_type]['payout'] += ticket.payout or Decimal('0')
            
            # 戦略別統計
            strategy_stats = defaultdict(lambda: {'count': 0, 'investment': Decimal('0'), 'profit': Decimal('0')})
            
            for ticket in confirmed_tickets:
                strategy = ticket.strategy_name or '不明'
                strategy_stats[strategy]['count'] += 1
                strategy_stats[strategy]['investment'] += ticket.amount
                strategy_stats[strategy]['profit'] += ticket.profit or Decimal('0')
            
            return {
                'period': f'{days_back}日間',
                'summary': {
                    'total_tickets': total_tickets,
                    'confirmed_tickets': len(confirmed_tickets),
                    'hit_tickets': len(hit_tickets),
                    'hit_rate': round(hit_rate, 2),
                    'total_investment': float(total_investment),
                    'total_payout': float(total_payout),
                    'total_profit': float(total_profit),
                    'return_rate': round(return_rate, 2)
                },
                'bet_type_performance': {
                    bet_type: {
                        'count': stats['count'],
                        'investment': float(stats['investment']),
                        'payout': float(stats['payout']),
                        'return_rate': round(float(stats['payout'] / stats['investment']) * 100, 2) if stats['investment'] > 0 else 0
                    }
                    for bet_type, stats in bet_type_stats.items()
                },
                'strategy_performance': {
                    strategy: {
                        'count': stats['count'],
                        'investment': float(stats['investment']),
                        'profit': float(stats['profit']),
                        'roi': round(float(stats['profit'] / stats['investment']) * 100, 2) if stats['investment'] > 0 else 0
                    }
                    for strategy, stats in strategy_stats.items()
                }
            }
    
    def get_daily_results(self, target_date: date) -> List[Dict[str, Any]]:
        """
        指定日の投票結果を取得
        
        Args:
            target_date: 対象日
            
        Returns:
            日別投票結果
        """
        start_datetime = datetime.combine(target_date, datetime.min.time())
        end_datetime = datetime.combine(target_date, datetime.max.time())
        
        with self.database.get_session() as session:
            tickets = session.query(BettingTicket).join(Race).filter(
                Race.date == target_date,
                BettingTicket.purchased_at >= start_datetime,
                BettingTicket.purchased_at <= end_datetime
            ).all()
            
            results = []
            
            for ticket in tickets:
                race = session.query(Race).filter(Race.id == ticket.race_id).first()
                
                result = {
                    'ticket_id': ticket.id,
                    'race_info': {
                        'race_id': ticket.race_id,
                        'stadium_number': race.stadium_number if race else None,
                        'race_number': race.race_number if race else None,
                        'title': race.title if race else None
                    },
                    'bet_type': ticket.bet_type,
                    'numbers': ticket.numbers,
                    'amount': float(ticket.amount),
                    'strategy': ticket.strategy_name,
                    'is_hit': ticket.is_hit,
                    'payout': float(ticket.payout) if ticket.payout else 0,
                    'profit': float(ticket.profit) if ticket.profit else None,
                    'purchased_at': ticket.purchased_at.isoformat()
                }
                results.append(result)
            
            return results
    
    def get_best_performing_strategies(self, 
                                     days_back: int = 90,
                                     min_bets: int = 10) -> List[Dict[str, Any]]:
        """
        成績の良い戦略を取得
        
        Args:
            days_back: 分析対象日数
            min_bets: 最小投票数
            
        Returns:
            戦略成績ランキング
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        with self.database.get_session() as session:
            tickets = session.query(BettingTicket).filter(
                BettingTicket.purchased_at >= cutoff_date,
                BettingTicket.is_hit.is_not(None)  # 確定済みのみ
            ).all()
            
            strategy_performance = defaultdict(lambda: {
                'count': 0,
                'investment': Decimal('0'),
                'payout': Decimal('0'),
                'profit': Decimal('0'),
                'hits': 0
            })
            
            for ticket in tickets:
                strategy = ticket.strategy_name or '戦略不明'
                stats = strategy_performance[strategy]
                
                stats['count'] += 1
                stats['investment'] += ticket.amount
                stats['payout'] += ticket.payout or Decimal('0')
                stats['profit'] += ticket.profit or Decimal('0')
                
                if ticket.is_hit:
                    stats['hits'] += 1
            
            # 最小投票数フィルタ
            qualified_strategies = {
                strategy: stats for strategy, stats in strategy_performance.items()
                if stats['count'] >= min_bets
            }
            
            # ROIでソート
            performance_list = []
            for strategy, stats in qualified_strategies.items():
                roi = float(stats['profit'] / stats['investment']) * 100 if stats['investment'] > 0 else 0
                hit_rate = stats['hits'] / stats['count'] * 100
                return_rate = float(stats['payout'] / stats['investment']) * 100 if stats['investment'] > 0 else 0
                
                performance_list.append({
                    'strategy': strategy,
                    'bets': stats['count'],
                    'investment': float(stats['investment']),
                    'payout': float(stats['payout']),
                    'profit': float(stats['profit']),
                    'roi': round(roi, 2),
                    'hit_rate': round(hit_rate, 2),
                    'return_rate': round(return_rate, 2)
                })
            
            # ROI降順でソート
            performance_list.sort(key=lambda x: x['roi'], reverse=True)
            
            return performance_list
    
    def _check_hit(self, ticket: BettingTicket, race_result: RaceResult) -> bool:
        """舟券の当否をチェック"""
        bet_type = ticket.bet_type
        numbers = ticket.numbers
        
        try:
            if bet_type == "win":
                # 単勝
                predicted_winner = int(numbers)
                return predicted_winner == race_result.first_place
                
            elif bet_type == "place_show":
                # 複勝（3着以内）
                predicted_place = int(numbers)
                return predicted_place in [race_result.first_place, race_result.second_place, race_result.third_place]
                
            elif bet_type == "exacta":
                # 2連単
                first, second = map(int, numbers.split('-'))
                return (first == race_result.first_place and 
                       second == race_result.second_place)
                
            elif bet_type == "quinella":
                # 2連複
                boat1, boat2 = map(int, numbers.split('-'))
                top_two = {race_result.first_place, race_result.second_place}
                return {boat1, boat2} == top_two
                
            elif bet_type == "trifecta":
                # 3連単
                first, second, third = map(int, numbers.split('-'))
                return (first == race_result.first_place and
                       second == race_result.second_place and
                       third == race_result.third_place)
                
            elif bet_type == "trio":
                # 3連複
                boats = set(map(int, numbers.split('-')))
                top_three = {race_result.first_place, race_result.second_place, race_result.third_place}
                return boats == top_three
                
        except (ValueError, AttributeError) as e:
            logger.error(f"当否チェックエラー: {e}")
            return False
        
        return False
    
    def _calculate_payout(self, ticket: BettingTicket, race_result: RaceResult) -> Decimal:
        """配当を計算"""
        bet_type = ticket.bet_type
        
        # 実際の配当データから取得
        payout_map = {
            "win": race_result.win_payout,
            "place_show": race_result.place_show_payout,
            "exacta": race_result.exacta_payout,
            "quinella": race_result.quinella_payout,
            "trifecta": race_result.trifecta_payout,
            "trio": race_result.trio_payout
        }
        
        base_payout = payout_map.get(bet_type, Decimal('0'))
        if not base_payout:
            return Decimal('0')
        
        # 100円あたりの配当 × 購入金額 / 100
        return base_payout * ticket.amount / Decimal('100')