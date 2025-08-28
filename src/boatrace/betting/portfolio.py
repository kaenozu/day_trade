"""
投資ポートフォリオ管理システム

舟券投資の収支管理、リスク分析、パフォーマンス追跡
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from collections import defaultdict
from statistics import mean, median, stdev
import math

from .ticket_manager import TicketManager, BettingResult
from ..data.database import Database, get_database

logger = logging.getLogger(__name__)


class Portfolio:
    """ポートフォリオ管理クラス"""
    
    def __init__(self, 
                 initial_capital: Decimal = Decimal('100000'),
                 database: Optional[Database] = None):
        """
        初期化
        
        Args:
            initial_capital: 初期資金
            database: データベース
        """
        self.initial_capital = initial_capital
        self.database = database or get_database()
        self.ticket_manager = TicketManager(database)
    
    def get_current_balance(self) -> Dict[str, Any]:
        """
        現在の収支状況を取得
        
        Returns:
            収支情報
        """
        with self.database.get_session() as session:
            # 全投票履歴を取得
            from ..data.database import BettingTicket
            
            tickets = session.query(BettingTicket).all()
            
            if not tickets:
                return {
                    'initial_capital': float(self.initial_capital),
                    'current_balance': float(self.initial_capital),
                    'total_invested': 0,
                    'total_return': 0,
                    'net_profit': 0,
                    'roi': 0.0
                }
            
            # 基本計算
            total_invested = sum(ticket.amount for ticket in tickets)
            confirmed_tickets = [t for t in tickets if t.is_hit is not None]
            total_return = sum(ticket.payout or Decimal('0') for ticket in confirmed_tickets)
            net_profit = total_return - total_invested
            current_balance = self.initial_capital + net_profit
            
            # ROI計算
            roi = float(net_profit / total_invested) * 100 if total_invested > 0 else 0
            
            return {
                'initial_capital': float(self.initial_capital),
                'current_balance': float(current_balance),
                'total_invested': float(total_invested),
                'total_return': float(total_return),
                'net_profit': float(net_profit),
                'roi': round(roi, 2),
                'confirmed_bets': len(confirmed_tickets),
                'pending_bets': len(tickets) - len(confirmed_tickets)
            }
    
    def get_performance_metrics(self, days_back: int = 30) -> Dict[str, Any]:
        """
        パフォーマンス指標を取得
        
        Args:
            days_back: 分析対象日数
            
        Returns:
            パフォーマンス指標
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        with self.database.get_session() as session:
            from ..data.database import BettingTicket
            
            tickets = session.query(BettingTicket).filter(
                BettingTicket.purchased_at >= cutoff_date,
                BettingTicket.is_hit.is_not(None)  # 確定済み
            ).all()
            
            if not tickets:
                return {'error': 'データが不足しています'}
            
            # 日別収支計算
            daily_results = defaultdict(lambda: {'investment': Decimal('0'), 'return': Decimal('0')})
            
            for ticket in tickets:
                day = ticket.purchased_at.date()
                daily_results[day]['investment'] += ticket.amount
                daily_results[day]['return'] += ticket.payout or Decimal('0')
            
            # 日別利益率
            daily_returns = []
            for day_data in daily_results.values():
                if day_data['investment'] > 0:
                    daily_return = float((day_data['return'] - day_data['investment']) / day_data['investment'])
                    daily_returns.append(daily_return)
            
            if not daily_returns:
                return {'error': 'リターンデータがありません'}
            
            # 統計指標計算
            avg_return = mean(daily_returns)
            return_std = stdev(daily_returns) if len(daily_returns) > 1 else 0
            
            # シャープレシオ（リスクフリーレート0と仮定）
            sharpe_ratio = avg_return / return_std if return_std > 0 else 0
            
            # 最大ドローダウン
            max_drawdown = self._calculate_max_drawdown(daily_results)
            
            # 勝率
            winning_days = len([r for r in daily_returns if r > 0])
            win_rate = winning_days / len(daily_returns) * 100
            
            # VaR (5%信頼区間)
            sorted_returns = sorted(daily_returns)
            var_5 = sorted_returns[int(len(sorted_returns) * 0.05)] if sorted_returns else 0
            
            return {
                'period': f'{days_back}日間',
                'trading_days': len(daily_returns),
                'average_daily_return': round(avg_return * 100, 3),
                'volatility': round(return_std * 100, 3),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'max_drawdown': round(max_drawdown * 100, 2),
                'win_rate': round(win_rate, 2),
                'var_5_percent': round(var_5 * 100, 3),
                'best_day': round(max(daily_returns) * 100, 2),
                'worst_day': round(min(daily_returns) * 100, 2)
            }
    
    def get_risk_analysis(self) -> Dict[str, Any]:
        """
        リスク分析を実行
        
        Returns:
            リスク分析結果
        """
        summary = self.ticket_manager.get_portfolio_summary(days_back=90)
        
        if 'error' in summary:
            return summary
        
        current_balance = self.get_current_balance()
        
        # リスク指標計算
        total_investment = current_balance['total_invested']
        roi = current_balance['roi']
        
        # 資金管理指標
        if total_investment > 0:
            capital_utilization = total_investment / float(self.initial_capital) * 100
        else:
            capital_utilization = 0
        
        # リスクレベル判定
        risk_level = "低"
        if abs(roi) > 50:
            risk_level = "高"
        elif abs(roi) > 20:
            risk_level = "中"
        
        # 投資集中度（戦略別）
        strategy_concentration = self._analyze_strategy_concentration(summary)
        
        # 舟券種別リスク
        bet_type_risk = self._analyze_bet_type_risk(summary)
        
        return {
            'overall_risk_level': risk_level,
            'capital_utilization': round(capital_utilization, 2),
            'roi_volatility': abs(roi),
            'strategy_concentration': strategy_concentration,
            'bet_type_risk': bet_type_risk,
            'recommendations': self._generate_risk_recommendations(
                risk_level, capital_utilization, roi, strategy_concentration
            )
        }
    
    def optimize_bet_sizing(self, 
                          prediction_confidence: float,
                          expected_odds: float,
                          max_risk_per_bet: float = 0.05) -> Decimal:
        """
        最適な賭け金を計算
        
        Args:
            prediction_confidence: 予想信頼度 (0.0-1.0)
            expected_odds: 期待オッズ
            max_risk_per_bet: 1回あたりの最大リスク率
            
        Returns:
            推奨賭け金
        """
        current_balance = self.get_current_balance()['current_balance']
        
        # Kelly基準の簡易版
        win_prob = prediction_confidence
        lose_prob = 1 - win_prob
        
        if expected_odds > 1 and win_prob > (1 / expected_odds):
            # Kelly比率 = (勝率 * オッズ - 1) / (オッズ - 1)
            kelly_fraction = (win_prob * expected_odds - 1) / (expected_odds - 1)
            
            # 保守的に調整（Kelly比率の25%）
            adjusted_fraction = kelly_fraction * 0.25
            
            # 最大リスク制限
            final_fraction = min(adjusted_fraction, max_risk_per_bet)
            
            bet_size = Decimal(str(current_balance * final_fraction))
            
            # 最小・最大制限
            min_bet = Decimal('100')
            max_bet = Decimal(str(current_balance * max_risk_per_bet))
            
            return max(min_bet, min(bet_size, max_bet))
        
        # リスクが高すぎる場合は最小額
        return Decimal('100')
    
    def get_monthly_report(self, year: int, month: int) -> Dict[str, Any]:
        """
        月次レポートを生成
        
        Args:
            year: 年
            month: 月
            
        Returns:
            月次レポート
        """
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)
        
        with self.database.get_session() as session:
            from ..data.database import BettingTicket
            
            tickets = session.query(BettingTicket).filter(
                BettingTicket.purchased_at >= datetime.combine(start_date, datetime.min.time()),
                BettingTicket.purchased_at <= datetime.combine(end_date, datetime.max.time())
            ).all()
            
            if not tickets:
                return {
                    'period': f'{year}年{month}月',
                    'summary': {'total_bets': 0}
                }
            
            # 基本統計
            total_bets = len(tickets)
            confirmed_bets = [t for t in tickets if t.is_hit is not None]
            hit_bets = [t for t in confirmed_bets if t.is_hit]
            
            total_investment = sum(t.amount for t in tickets)
            total_return = sum(t.payout or Decimal('0') for t in confirmed_bets)
            net_profit = total_return - total_investment
            
            hit_rate = len(hit_bets) / len(confirmed_bets) * 100 if confirmed_bets else 0
            roi = float(net_profit / total_investment) * 100 if total_investment > 0 else 0
            
            # 日別分析
            daily_analysis = self._analyze_daily_performance(tickets)
            
            # ベストパフォーマンス
            best_day = max(daily_analysis.values(), key=lambda x: x['profit']) if daily_analysis else None
            worst_day = min(daily_analysis.values(), key=lambda x: x['profit']) if daily_analysis else None
            
            return {
                'period': f'{year}年{month}月',
                'summary': {
                    'total_bets': total_bets,
                    'confirmed_bets': len(confirmed_bets),
                    'hit_bets': len(hit_bets),
                    'hit_rate': round(hit_rate, 2),
                    'total_investment': float(total_investment),
                    'total_return': float(total_return),
                    'net_profit': float(net_profit),
                    'roi': round(roi, 2),
                    'trading_days': len(daily_analysis)
                },
                'daily_analysis': {
                    'best_day': {
                        'date': best_day['date'] if best_day else None,
                        'profit': float(best_day['profit']) if best_day else 0
                    },
                    'worst_day': {
                        'date': worst_day['date'] if worst_day else None,
                        'profit': float(worst_day['profit']) if worst_day else 0
                    },
                    'profitable_days': len([d for d in daily_analysis.values() if d['profit'] > 0])
                }
            }
    
    def _calculate_max_drawdown(self, daily_results: Dict[date, Dict[str, Decimal]]) -> float:
        """最大ドローダウンを計算"""
        if not daily_results:
            return 0.0
        
        # 累積収益計算
        sorted_days = sorted(daily_results.keys())
        cumulative_returns = []
        cumulative = 0.0
        
        for day in sorted_days:
            day_data = daily_results[day]
            if day_data['investment'] > 0:
                day_return = float((day_data['return'] - day_data['investment']) / day_data['investment'])
                cumulative += day_return
                cumulative_returns.append(cumulative)
        
        if not cumulative_returns:
            return 0.0
        
        # 最大ドローダウン計算
        peak = cumulative_returns[0]
        max_dd = 0.0
        
        for ret in cumulative_returns:
            if ret > peak:
                peak = ret
            
            drawdown = peak - ret
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _analyze_strategy_concentration(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """戦略集中度を分析"""
        strategy_perf = summary.get('strategy_performance', {})
        
        if not strategy_perf:
            return {'concentration_risk': '不明'}
        
        total_investment = sum(s['investment'] for s in strategy_perf.values())
        
        if total_investment == 0:
            return {'concentration_risk': '不明'}
        
        # 最大戦略の比率
        max_strategy_investment = max(s['investment'] for s in strategy_perf.values())
        concentration_ratio = max_strategy_investment / total_investment
        
        if concentration_ratio > 0.7:
            risk_level = "高"
        elif concentration_ratio > 0.5:
            risk_level = "中"
        else:
            risk_level = "低"
        
        return {
            'concentration_risk': risk_level,
            'max_strategy_ratio': round(concentration_ratio * 100, 1),
            'strategy_count': len(strategy_perf)
        }
    
    def _analyze_bet_type_risk(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """舟券種別リスクを分析"""
        bet_type_perf = summary.get('bet_type_performance', {})
        
        if not bet_type_perf:
            return {'risk_level': '不明'}
        
        # 高リスク舟券（3連単等）の比率
        high_risk_types = ['trifecta']
        high_risk_investment = sum(
            bt['investment'] for bt_name, bt in bet_type_perf.items() 
            if bt_name in high_risk_types
        )
        
        total_investment = sum(bt['investment'] for bt in bet_type_perf.values())
        
        if total_investment == 0:
            return {'risk_level': '不明'}
        
        high_risk_ratio = high_risk_investment / total_investment
        
        if high_risk_ratio > 0.5:
            risk_level = "高"
        elif high_risk_ratio > 0.2:
            risk_level = "中"
        else:
            risk_level = "低"
        
        return {
            'risk_level': risk_level,
            'high_risk_ratio': round(high_risk_ratio * 100, 1),
            'bet_type_diversity': len(bet_type_perf)
        }
    
    def _generate_risk_recommendations(self,
                                     risk_level: str,
                                     capital_utilization: float,
                                     roi: float,
                                     strategy_concentration: Dict[str, Any]) -> List[str]:
        """リスク推奨事項を生成"""
        recommendations = []
        
        if capital_utilization > 80:
            recommendations.append("資金使用率が高すぎます。投資額を抑制してください")
        
        if abs(roi) > 50:
            recommendations.append("ROIの変動が大きいです。リスク管理を見直してください")
        
        if strategy_concentration.get('concentration_risk') == '高':
            recommendations.append("特定戦略に集中しています。戦略の分散を検討してください")
        
        if roi < -20:
            recommendations.append("損失が拡大しています。投資戦略の見直しを推奨します")
        
        if not recommendations:
            recommendations.append("現在のリスク水準は適切です")
        
        return recommendations
    
    def _analyze_daily_performance(self, tickets: List) -> Dict[date, Dict[str, Any]]:
        """日別パフォーマンスを分析"""
        daily_data = defaultdict(lambda: {
            'investment': Decimal('0'),
            'return': Decimal('0'),
            'bets': 0
        })
        
        for ticket in tickets:
            day = ticket.purchased_at.date()
            daily_data[day]['investment'] += ticket.amount
            daily_data[day]['return'] += ticket.payout or Decimal('0')
            daily_data[day]['bets'] += 1
        
        # 利益計算
        result = {}
        for day, data in daily_data.items():
            profit = data['return'] - data['investment']
            result[day] = {
                'date': day.isoformat(),
                'investment': data['investment'],
                'return': data['return'],
                'profit': profit,
                'bets': data['bets']
            }
        
        return result