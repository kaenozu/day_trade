#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Management Service - リスク管理サービス
ポジションサイジング、リスク計算、リスク制限機能
"""

import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path

class RiskLevel(Enum):
    """リスクレベル"""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"

class PositionType(Enum):
    """ポジションタイプ"""
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class RiskParameters:
    """リスクパラメータ"""
    max_portfolio_risk_pct: float = 2.0  # ポートフォリオ最大リスク率
    max_position_risk_pct: float = 1.0   # 単一ポジション最大リスク率
    max_sector_concentration_pct: float = 20.0  # セクター集中度上限
    max_single_position_pct: float = 10.0       # 単一銘柄上限
    risk_free_rate: float = 0.02                # リスクフリーレート
    confidence_level: float = 0.95              # VaR信頼度
    lookback_days: int = 252                    # 過去データ参照期間

@dataclass
class PositionRisk:
    """ポジションリスク"""
    symbol: str
    position_size: int
    current_price: float
    position_value: float
    stop_loss_price: Optional[float]
    potential_loss: float
    potential_loss_pct: float
    risk_amount: float
    beta: Optional[float] = None
    volatility: Optional[float] = None

@dataclass
class PortfolioRisk:
    """ポートフォリオリスク"""
    total_value: float
    total_risk: float
    total_risk_pct: float
    var_95: float
    var_99: float
    expected_shortfall: float
    sharpe_ratio: float
    max_drawdown_pct: float
    sector_concentrations: Dict[str, float]
    correlation_risk: float

@dataclass
class PositionSizing:
    """ポジションサイジング結果"""
    symbol: str
    recommended_size: int
    recommended_value: float
    risk_per_share: float
    stop_loss_price: float
    position_risk_pct: float
    reasoning: str

class RiskCalculator:
    """リスク計算エンジン"""
    
    def __init__(self, risk_params: RiskParameters = None):
        self.risk_params = risk_params or RiskParameters()
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, symbol: str, current_price: float, 
                              stop_loss_price: float, account_balance: float,
                              volatility: float = None) -> PositionSizing:
        """ポジションサイズ計算"""
        try:
            # リスク金額計算
            risk_per_share = abs(current_price - stop_loss_price)
            max_risk_amount = account_balance * (self.risk_params.max_position_risk_pct / 100)
            
            # 基本ポジションサイズ
            basic_size = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
            
            # ボラティリティ調整
            if volatility:
                volatility_adjustment = min(1.0, 0.20 / volatility)  # 20%基準で調整
                adjusted_size = int(basic_size * volatility_adjustment)
            else:
                adjusted_size = basic_size
            
            # 最小/最大制約
            min_size = 1
            max_value = account_balance * (self.risk_params.max_single_position_pct / 100)
            max_size = int(max_value / current_price)
            
            recommended_size = max(min_size, min(adjusted_size, max_size))
            recommended_value = recommended_size * current_price
            position_risk_pct = (risk_per_share * recommended_size) / account_balance * 100
            
            # 理由説明
            reasoning_parts = []
            if adjusted_size != basic_size:
                reasoning_parts.append(f"ボラティリティ調整: {volatility:.2%}")
            if recommended_size == max_size:
                reasoning_parts.append("単一銘柄上限適用")
            if recommended_size == min_size:
                reasoning_parts.append("最小サイズ適用")
            
            reasoning = "; ".join(reasoning_parts) if reasoning_parts else "標準計算"
            
            return PositionSizing(
                symbol=symbol,
                recommended_size=recommended_size,
                recommended_value=recommended_value,
                risk_per_share=risk_per_share,
                stop_loss_price=stop_loss_price,
                position_risk_pct=position_risk_pct,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"ポジションサイズ計算エラー: {e}")
            raise
    
    def calculate_optimal_stop_loss(self, symbol: str, current_price: float,
                                  entry_price: float, position_type: PositionType,
                                  volatility: float = None, 
                                  support_resistance: float = None) -> float:
        """最適ストップロス計算"""
        try:
            if position_type == PositionType.LONG:
                # ロングポジション
                base_stop = entry_price * 0.95  # 5%下
                
                if volatility:
                    # ボラティリティベース
                    vol_stop = entry_price - (2 * volatility * entry_price)
                    base_stop = max(base_stop, vol_stop)
                
                if support_resistance:
                    # サポートレベル考慮
                    base_stop = max(base_stop, support_resistance * 0.99)
                
                return round(min(base_stop, current_price * 0.98), 2)
            
            else:
                # ショートポジション
                base_stop = entry_price * 1.05  # 5%上
                
                if volatility:
                    vol_stop = entry_price + (2 * volatility * entry_price)
                    base_stop = min(base_stop, vol_stop)
                
                if support_resistance:
                    base_stop = min(base_stop, support_resistance * 1.01)
                
                return round(max(base_stop, current_price * 1.02), 2)
                
        except Exception as e:
            self.logger.error(f"ストップロス計算エラー: {e}")
            return current_price * (0.95 if position_type == PositionType.LONG else 1.05)
    
    def calculate_position_risk(self, symbol: str, quantity: int, current_price: float,
                              stop_loss_price: float = None, volatility: float = None) -> PositionRisk:
        """ポジションリスク計算"""
        try:
            position_value = quantity * current_price
            
            if stop_loss_price:
                potential_loss = abs(current_price - stop_loss_price) * quantity
                potential_loss_pct = abs(current_price - stop_loss_price) / current_price * 100
                risk_amount = potential_loss
            else:
                # ストップロスが設定されていない場合、ボラティリティベース
                if volatility:
                    risk_amount = position_value * volatility * 2  # 2σ
                    potential_loss = risk_amount
                    potential_loss_pct = volatility * 200
                else:
                    # デフォルト5%リスク
                    risk_amount = position_value * 0.05
                    potential_loss = risk_amount
                    potential_loss_pct = 5.0
            
            return PositionRisk(
                symbol=symbol,
                position_size=quantity,
                current_price=current_price,
                position_value=position_value,
                stop_loss_price=stop_loss_price,
                potential_loss=potential_loss,
                potential_loss_pct=potential_loss_pct,
                risk_amount=risk_amount,
                volatility=volatility
            )
            
        except Exception as e:
            self.logger.error(f"ポジションリスク計算エラー: {e}")
            raise
    
    def calculate_portfolio_var(self, positions: List[PositionRisk], 
                               correlation_matrix: np.ndarray = None) -> Tuple[float, float]:
        """ポートフォリオVaR計算"""
        try:
            if not positions:
                return 0.0, 0.0
            
            # 個別リスク集計
            individual_vars = [pos.risk_amount for pos in positions]
            
            if correlation_matrix is not None and len(positions) > 1:
                # 相関を考慮したVaR
                weights = np.array([pos.position_value for pos in positions])
                total_value = sum(weights)
                weights = weights / total_value
                
                volatilities = np.array([pos.volatility or 0.05 for pos in positions])
                
                portfolio_variance = np.dot(weights, np.dot(correlation_matrix * np.outer(volatilities, volatilities), weights))
                portfolio_volatility = math.sqrt(portfolio_variance)
                
                # 95% VaR
                var_95 = total_value * portfolio_volatility * 1.645
                # 99% VaR  
                var_99 = total_value * portfolio_volatility * 2.326
                
            else:
                # 単純合計（最悪ケース）
                var_95 = sum(individual_vars) * 0.95
                var_99 = sum(individual_vars) * 0.99
            
            return var_95, var_99
            
        except Exception as e:
            self.logger.error(f"VaR計算エラー: {e}")
            return 0.0, 0.0

class RiskMonitor:
    """リスク監視"""
    
    def __init__(self, risk_params: RiskParameters = None):
        self.risk_params = risk_params or RiskParameters()
        self.risk_calculator = RiskCalculator(risk_params)
        self.alerts: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def check_portfolio_limits(self, portfolio_value: float, portfolio_risk: float,
                             positions: List[PositionRisk], 
                             sector_allocations: Dict[str, float]) -> List[Dict[str, Any]]:
        """ポートフォリオ制限チェック"""
        violations = []
        
        try:
            # 総リスク制限
            risk_pct = (portfolio_risk / portfolio_value) * 100
            if risk_pct > self.risk_params.max_portfolio_risk_pct:
                violations.append({
                    'type': 'portfolio_risk_limit',
                    'severity': 'HIGH',
                    'current': risk_pct,
                    'limit': self.risk_params.max_portfolio_risk_pct,
                    'message': f'ポートフォリオリスク上限超過: {risk_pct:.2f}% > {self.risk_params.max_portfolio_risk_pct}%'
                })
            
            # 個別ポジション制限
            for pos in positions:
                pos_pct = (pos.position_value / portfolio_value) * 100
                if pos_pct > self.risk_params.max_single_position_pct:
                    violations.append({
                        'type': 'position_concentration',
                        'severity': 'MEDIUM',
                        'symbol': pos.symbol,
                        'current': pos_pct,
                        'limit': self.risk_params.max_single_position_pct,
                        'message': f'{pos.symbol} ポジション集中度過多: {pos_pct:.2f}% > {self.risk_params.max_single_position_pct}%'
                    })
            
            # セクター集中度制限
            for sector, allocation in sector_allocations.items():
                if allocation > self.risk_params.max_sector_concentration_pct:
                    violations.append({
                        'type': 'sector_concentration',
                        'severity': 'MEDIUM',
                        'sector': sector,
                        'current': allocation,
                        'limit': self.risk_params.max_sector_concentration_pct,
                        'message': f'{sector} セクター集中度過多: {allocation:.2f}% > {self.risk_params.max_sector_concentration_pct}%'
                    })
            
            return violations
            
        except Exception as e:
            self.logger.error(f"制限チェックエラー: {e}")
            return []
    
    def generate_risk_report(self, positions: List[PositionRisk], 
                           portfolio_value: float) -> Dict[str, Any]:
        """リスクレポート生成"""
        try:
            total_risk = sum(pos.risk_amount for pos in positions)
            risk_pct = (total_risk / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            # VaR計算
            var_95, var_99 = self.risk_calculator.calculate_portfolio_var(positions)
            
            # セクター分散（サンプル）
            sector_map = {
                '7203': 'Automotive', '9984': 'Technology', '8306': 'Financial',
                '6758': 'Technology', '4751': 'Technology'
            }
            
            sector_allocations = {}
            for pos in positions:
                sector = sector_map.get(pos.symbol, 'Other')
                if sector not in sector_allocations:
                    sector_allocations[sector] = 0
                sector_allocations[sector] += (pos.position_value / portfolio_value) * 100
            
            # リスクアラート
            violations = self.check_portfolio_limits(portfolio_value, total_risk, positions, sector_allocations)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': portfolio_value,
                'total_risk_amount': total_risk,
                'total_risk_pct': risk_pct,
                'var_95': var_95,
                'var_99': var_99,
                'position_count': len(positions),
                'largest_position_pct': max([pos.position_value / portfolio_value * 100 for pos in positions]) if positions else 0,
                'sector_allocations': sector_allocations,
                'risk_violations': violations,
                'risk_metrics': {
                    'concentration_risk': self._calculate_concentration_risk(positions, portfolio_value),
                    'correlation_risk': 'Low',  # 簡易評価
                    'liquidity_risk': 'Medium'  # 簡易評価
                },
                'recommendations': self._generate_risk_recommendations(violations, sector_allocations)
            }
            
        except Exception as e:
            self.logger.error(f"リスクレポート生成エラー: {e}")
            return {'error': str(e)}
    
    def _calculate_concentration_risk(self, positions: List[PositionRisk], portfolio_value: float) -> str:
        """集中リスク評価"""
        if not positions:
            return "N/A"
        
        position_weights = [pos.position_value / portfolio_value for pos in positions]
        hhi = sum(w**2 for w in position_weights)  # ハーフィンダール指数
        
        if hhi > 0.25:
            return "High"
        elif hhi > 0.15:
            return "Medium"
        else:
            return "Low"
    
    def _generate_risk_recommendations(self, violations: List[Dict[str, Any]], 
                                     sector_allocations: Dict[str, float]) -> List[str]:
        """リスク推奨事項生成"""
        recommendations = []
        
        if violations:
            high_severity = [v for v in violations if v['severity'] == 'HIGH']
            if high_severity:
                recommendations.append("緊急: 高リスクポジションの縮小を検討してください")
        
        if len(sector_allocations) < 3:
            recommendations.append("セクター分散を改善してリスクを軽減してください")
        
        max_sector_allocation = max(sector_allocations.values()) if sector_allocations else 0
        if max_sector_allocation > 50:
            recommendations.append("特定セクターへの過度な集中を解消してください")
        
        if not recommendations:
            recommendations.append("現在のリスク水準は適切な範囲内です")
        
        return recommendations

class RiskManagementService:
    """リスク管理サービス統合"""
    
    def __init__(self, risk_params: RiskParameters = None):
        self.risk_params = risk_params or RiskParameters()
        self.risk_calculator = RiskCalculator(risk_params)
        self.risk_monitor = RiskMonitor(risk_params)
        self.logger = logging.getLogger(__name__)
    
    def analyze_trade_risk(self, symbol: str, current_price: float, 
                          quantity: int, stop_loss_price: float = None,
                          account_balance: float = 1000000) -> Dict[str, Any]:
        """取引リスク分析"""
        try:
            # ポジションリスク計算
            position_risk = self.risk_calculator.calculate_position_risk(
                symbol, quantity, current_price, stop_loss_price
            )
            
            # 推奨ストップロス（設定されていない場合）
            if not stop_loss_price:
                optimal_stop = self.risk_calculator.calculate_optimal_stop_loss(
                    symbol, current_price, current_price, PositionType.LONG
                )
            else:
                optimal_stop = stop_loss_price
            
            # リスク評価
            risk_rating = self._assess_risk_rating(position_risk, account_balance)
            
            return {
                'symbol': symbol,
                'position_risk': asdict(position_risk),
                'optimal_stop_loss': optimal_stop,
                'risk_rating': risk_rating,
                'account_risk_pct': (position_risk.risk_amount / account_balance) * 100,
                'recommendations': self._generate_trade_recommendations(position_risk, account_balance),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"取引リスク分析エラー: {e}")
            return {'error': str(e)}
    
    def calculate_position_sizing(self, symbol: str, current_price: float,
                                account_balance: float, risk_level: RiskLevel = RiskLevel.MODERATE,
                                support_level: float = None) -> Dict[str, Any]:
        """ポジションサイジング計算"""
        try:
            # リスクレベル別パラメータ調整
            risk_multipliers = {
                RiskLevel.CONSERVATIVE: 0.5,
                RiskLevel.MODERATE: 1.0,
                RiskLevel.AGGRESSIVE: 1.5
            }
            
            adjusted_params = RiskParameters(
                max_position_risk_pct=self.risk_params.max_position_risk_pct * risk_multipliers[risk_level]
            )
            
            calculator = RiskCalculator(adjusted_params)
            
            # ストップロス価格計算
            stop_loss = support_level or (current_price * 0.95)
            
            # ポジションサイジング
            sizing = calculator.calculate_position_size(
                symbol, current_price, stop_loss, account_balance
            )
            
            # 追加分析
            risk_reward_ratio = self._calculate_risk_reward_ratio(current_price, stop_loss)
            
            return {
                'sizing': asdict(sizing),
                'risk_level': risk_level.value,
                'risk_reward_ratio': risk_reward_ratio,
                'max_loss_amount': sizing.risk_per_share * sizing.recommended_size,
                'position_allocation_pct': (sizing.recommended_value / account_balance) * 100,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"ポジションサイジング計算エラー: {e}")
            return {'error': str(e)}
    
    def _assess_risk_rating(self, position_risk: PositionRisk, account_balance: float) -> str:
        """リスク評価"""
        risk_pct = (position_risk.risk_amount / account_balance) * 100
        
        if risk_pct > 3.0:
            return "HIGH"
        elif risk_pct > 1.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_trade_recommendations(self, position_risk: PositionRisk, 
                                      account_balance: float) -> List[str]:
        """取引推奨事項"""
        recommendations = []
        
        risk_pct = (position_risk.risk_amount / account_balance) * 100
        
        if risk_pct > 2.0:
            recommendations.append("リスクが高すぎます。ポジションサイズを縮小してください")
        
        if not position_risk.stop_loss_price:
            recommendations.append("ストップロスを設定してリスクを制限してください")
        
        if position_risk.potential_loss_pct > 10:
            recommendations.append("潜在損失が大きすぎます。ストップロスを近くに設定してください")
        
        return recommendations or ["リスク水準は適切です"]
    
    def _calculate_risk_reward_ratio(self, current_price: float, stop_loss: float, 
                                   target_price: float = None) -> float:
        """リスクリワード比計算"""
        risk = abs(current_price - stop_loss)
        
        if target_price:
            reward = abs(target_price - current_price)
        else:
            # デフォルト10%利益目標
            reward = current_price * 0.10
        
        return round(reward / risk, 2) if risk > 0 else 0