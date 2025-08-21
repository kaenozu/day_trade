"""
投票戦略システム

様々な投票戦略・アルゴリズムを実装
"""

import logging
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from ..core.data_models import PredictionResult
from .ticket_manager import BetType

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """リスクレベル"""
    CONSERVATIVE = "conservative"  # 保守的
    MODERATE = "moderate"         # 適度
    AGGRESSIVE = "aggressive"     # 積極的


@dataclass
class BettingRecommendation:
    """投票推奨"""
    bet_type: BetType
    numbers: str
    amount: Decimal
    confidence: float
    expected_return: float
    risk_level: RiskLevel
    reason: str


class BettingStrategy(ABC):
    """投票戦略ベースクラス"""
    
    def __init__(self, name: str, risk_level: RiskLevel):
        self.name = name
        self.risk_level = risk_level
    
    @abstractmethod
    def generate_bets(self,
                     prediction: PredictionResult,
                     budget: Decimal,
                     race_info: Dict[str, Any]) -> List[BettingRecommendation]:
        """
        投票推奨を生成
        
        Args:
            prediction: 予想結果
            budget: 予算
            race_info: レース情報
            
        Returns:
            投票推奨リスト
        """
        pass


class ConservativeStrategy(BettingStrategy):
    """保守的戦略 - 手堅い本命狙い"""
    
    def __init__(self):
        super().__init__("保守的戦略", RiskLevel.CONSERVATIVE)
    
    def generate_bets(self,
                     prediction: PredictionResult,
                     budget: Decimal,
                     race_info: Dict[str, Any]) -> List[BettingRecommendation]:
        
        recommendations = []
        
        # 1着確率上位2艇を取得
        sorted_win = sorted(prediction.win_probabilities.items(), 
                          key=lambda x: x[1], reverse=True)
        
        if len(sorted_win) < 2:
            return recommendations
        
        top_boat = sorted_win[0][0]
        second_boat = sorted_win[1][0]
        top_prob = float(sorted_win[0][1])
        
        # 本命の勝率が高い場合のみベット
        if top_prob >= 0.25:  # 25%以上
            # 単勝メイン (予算の60%)
            recommendations.append(BettingRecommendation(
                bet_type=BetType.WIN,
                numbers=str(top_boat),
                amount=budget * Decimal('0.6'),
                confidence=top_prob,
                expected_return=top_prob * 2.0,  # 仮の期待値
                risk_level=self.risk_level,
                reason=f"{top_boat}号艇 単勝 (勝率{top_prob:.1%})"
            ))
            
            # 2連複サポート (予算の40%)
            recommendations.append(BettingRecommendation(
                bet_type=BetType.QUINELLA,
                numbers=f"{top_boat}-{second_boat}",
                amount=budget * Decimal('0.4'),
                confidence=top_prob * 0.8,
                expected_return=top_prob * 1.5,
                risk_level=self.risk_level,
                reason=f"{top_boat}-{second_boat} 2連複サポート"
            ))
        
        return recommendations


class BalancedStrategy(BettingStrategy):
    """バランス戦略 - 本命と穴のバランス"""
    
    def __init__(self):
        super().__init__("バランス戦略", RiskLevel.MODERATE)
    
    def generate_bets(self,
                     prediction: PredictionResult,
                     budget: Decimal,
                     race_info: Dict[str, Any]) -> List[BettingRecommendation]:
        
        recommendations = []
        
        sorted_win = sorted(prediction.win_probabilities.items(),
                          key=lambda x: x[1], reverse=True)
        
        if len(sorted_win) < 3:
            return recommendations
        
        # 上位3艇を取得
        top_boats = [boat for boat, prob in sorted_win[:3]]
        top_prob = float(sorted_win[0][1])
        
        # 本命軸 2連単 (予算の50%)
        main_amount = budget * Decimal('0.5') / Decimal('2')
        for second in top_boats[1:3]:
            recommendations.append(BettingRecommendation(
                bet_type=BetType.EXACTA,
                numbers=f"{top_boats[0]}-{second}",
                amount=main_amount,
                confidence=top_prob * 0.6,
                expected_return=top_prob * 3.0,
                risk_level=self.risk_level,
                reason=f"{top_boats[0]}軸 2連単"
            ))
        
        # 3連複ワイド (予算の30%)
        recommendations.append(BettingRecommendation(
            bet_type=BetType.TRIO,
            numbers=f"{top_boats[0]}-{top_boats[1]}-{top_boats[2]}",
            amount=budget * Decimal('0.3'),
            confidence=top_prob * 0.4,
            expected_return=top_prob * 4.0,
            risk_level=self.risk_level,
            reason="上位3艇の3連複"
        ))
        
        # 穴狙い (予算の20%) - 4-6号艇から選択
        for boat, prob in sorted_win[3:]:
            if float(prob) > 0.05:  # 5%以上の確率
                recommendations.append(BettingRecommendation(
                    bet_type=BetType.WIN,
                    numbers=str(boat),
                    amount=budget * Decimal('0.2'),
                    confidence=float(prob),
                    expected_return=float(prob) * 8.0,  # 高配当期待
                    risk_level=self.risk_level,
                    reason=f"{boat}号艇 穴狙い単勝"
                ))
                break
        
        return recommendations


class AggressiveStrategy(BettingStrategy):
    """積極戦略 - 高配当狙い"""
    
    def __init__(self):
        super().__init__("積極戦略", RiskLevel.AGGRESSIVE)
    
    def generate_bets(self,
                     prediction: PredictionResult,
                     budget: Decimal,
                     race_info: Dict[str, Any]) -> List[BettingRecommendation]:
        
        recommendations = []
        
        sorted_win = sorted(prediction.win_probabilities.items(),
                          key=lambda x: x[1], reverse=True)
        
        # 高配当を狙うため、下位艇にも注目
        all_boats = list(range(1, 7))
        mid_boats = [boat for boat, prob in sorted_win[2:] if float(prob) > 0.08]
        
        if len(mid_boats) >= 2:
            # 3連単多点勝負 (予算の70%)
            bet_amount = budget * Decimal('0.7') / Decimal(str(len(mid_boats)))
            
            for boat in mid_boats:
                # 穴軸の3連単
                other_boats = [b for b in all_boats if b != boat][:2]
                numbers = f"{boat}-{other_boats[0]}-{other_boats[1]}"
                
                recommendations.append(BettingRecommendation(
                    bet_type=BetType.TRIFECTA,
                    numbers=numbers,
                    amount=bet_amount,
                    confidence=0.3,
                    expected_return=0.3 * 50.0,  # 高配当期待
                    risk_level=self.risk_level,
                    reason=f"{boat}軸 高配当3連単"
                ))
        
        # 荒れる条件なら大穴狙い (予算の30%)
        competitiveness = race_info.get('competitiveness', 'normal')
        if competitiveness in ['激戦', '混戦']:
            # 最下位艇の単勝
            if sorted_win:
                last_boat = sorted_win[-1][0]
                recommendations.append(BettingRecommendation(
                    bet_type=BetType.WIN,
                    numbers=str(last_boat),
                    amount=budget * Decimal('0.3'),
                    confidence=0.1,
                    expected_return=0.1 * 100.0,  # 超高配当期待
                    risk_level=self.risk_level,
                    reason=f"{last_boat}号艇 大穴単勝"
                ))
        
        return recommendations


class MLBasedStrategy(BettingStrategy):
    """機械学習ベース戦略 - 予想確率を信頼"""
    
    def __init__(self):
        super().__init__("ML予想戦略", RiskLevel.MODERATE)
    
    def generate_bets(self,
                     prediction: PredictionResult,
                     budget: Decimal,
                     race_info: Dict[str, Any]) -> List[BettingRecommendation]:
        
        recommendations = []
        confidence_threshold = float(prediction.confidence)
        
        if confidence_threshold < 0.5:
            # 信頼度が低い場合は少額ベット
            budget = budget * Decimal('0.3')
        
        # 勝率確率に基づく投資配分
        total_prob = sum(float(prob) for prob in prediction.win_probabilities.values())
        
        for boat, prob in prediction.win_probabilities.items():
            prob_float = float(prob)
            if prob_float > 0.15:  # 15%以上の確率
                # Kelly基準を簡易適用
                allocation = (prob_float - 0.167) * 2  # 期待値を仮定
                if allocation > 0:
                    bet_amount = budget * Decimal(str(min(allocation, 0.3)))
                    
                    recommendations.append(BettingRecommendation(
                        bet_type=BetType.WIN,
                        numbers=str(boat),
                        amount=bet_amount,
                        confidence=prob_float,
                        expected_return=prob_float * 4.0,
                        risk_level=self.risk_level,
                        reason=f"ML確率{prob_float:.1%}に基づく配分"
                    ))
        
        # 複勝ヘッジ - 上位2艇
        sorted_win = sorted(prediction.win_probabilities.items(),
                          key=lambda x: x[1], reverse=True)
        
        if len(sorted_win) >= 2:
            top_two = sorted_win[:2]
            hedge_amount = budget * Decimal('0.2')
            
            for boat, prob in top_two:
                recommendations.append(BettingRecommendation(
                    bet_type=BetType.PLACE_SHOW,
                    numbers=str(boat),
                    amount=hedge_amount / Decimal('2'),
                    confidence=float(prob) * 1.5,  # 複勝は当たりやすい
                    expected_return=float(prob) * 1.8,
                    risk_level=RiskLevel.CONSERVATIVE,
                    reason=f"{boat}号艇 複勝ヘッジ"
                ))
        
        return recommendations


class StrategyManager:
    """戦略管理クラス"""
    
    def __init__(self):
        self.strategies = {
            "conservative": ConservativeStrategy(),
            "balanced": BalancedStrategy(),
            "aggressive": AggressiveStrategy(),
            "ml_based": MLBasedStrategy()
        }
    
    def get_strategy(self, strategy_name: str) -> Optional[BettingStrategy]:
        """戦略を取得"""
        return self.strategies.get(strategy_name)
    
    def get_all_strategies(self) -> Dict[str, BettingStrategy]:
        """全戦略を取得"""
        return self.strategies.copy()
    
    def compare_strategies(self,
                         prediction: PredictionResult,
                         budget: Decimal,
                         race_info: Dict[str, Any]) -> Dict[str, List[BettingRecommendation]]:
        """
        全戦略の推奨を比較
        
        Args:
            prediction: 予想結果
            budget: 予算
            race_info: レース情報
            
        Returns:
            戦略別推奨
        """
        comparison = {}
        
        for name, strategy in self.strategies.items():
            try:
                recommendations = strategy.generate_bets(prediction, budget, race_info)
                comparison[name] = recommendations
            except Exception as e:
                logger.error(f"戦略 {name} でエラー: {e}")
                comparison[name] = []
        
        return comparison
    
    def select_best_strategy(self,
                           prediction: PredictionResult,
                           race_info: Dict[str, Any],
                           user_preference: Optional[str] = None) -> str:
        """
        最適戦略を選択
        
        Args:
            prediction: 予想結果
            race_info: レース情報
            user_preference: ユーザー選好
            
        Returns:
            推奨戦略名
        """
        if user_preference and user_preference in self.strategies:
            return user_preference
        
        # レース特性に基づく戦略選択
        competitiveness = race_info.get('competitiveness', 'normal')
        confidence = float(prediction.confidence)
        
        if confidence >= 0.8:
            return "conservative"  # 高信頼度なら手堅く
        elif competitiveness in ['激戦', '混戦']:
            return "aggressive"    # 荒れそうなら積極的
        else:
            return "balanced"      # 標準はバランス
    
    def create_custom_strategy(self,
                             name: str,
                             risk_level: RiskLevel,
                             bet_types: List[BetType],
                             allocation: Dict[str, float]) -> BettingStrategy:
        """
        カスタム戦略を作成
        
        Args:
            name: 戦略名
            risk_level: リスクレベル
            bet_types: 使用する舟券種別
            allocation: 配分比率
            
        Returns:
            カスタム戦略
        """
        # 簡易実装 - 実際には設定に基づいてクラスを生成
        return BalancedStrategy()  # プレースホルダー