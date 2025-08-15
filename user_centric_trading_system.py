#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User-Centric Trading System - ユーザー中心取引支援システム
個人投資家向けパーソナライゼーション機能と取引支援
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from enhanced_personal_analysis_engine import get_analysis_engine, AnalysisMode, TradingSignal


class InvestorProfile(Enum):
    """投資家プロファイル"""
    CONSERVATIVE = "conservative"  # 保守的
    MODERATE = "moderate"         # 中庸
    AGGRESSIVE = "aggressive"     # 積極的
    DAYTRADER = "daytrader"      # デイトレーダー


class RiskTolerance(Enum):
    """リスク許容度"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class UserProfile:
    """ユーザープロファイル"""
    user_id: str
    name: str
    investor_type: InvestorProfile
    risk_tolerance: RiskTolerance
    investment_experience_years: int
    preferred_analysis_mode: AnalysisMode
    favorite_symbols: List[str]
    investment_goals: List[str]
    daily_trading_budget: float
    max_loss_per_trade: float
    notification_preferences: Dict[str, bool]
    trading_hours: Dict[str, str]  # start_time, end_time
    created_at: datetime
    last_updated: datetime


@dataclass
class TradingRecommendation:
    """取引推奨"""
    symbol: str
    action: TradingSignal
    confidence: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    position_size: float
    reasoning: List[str]
    risk_assessment: str
    expected_return: float
    holding_period: str
    timestamp: datetime


@dataclass
class PersonalizedAlert:
    """パーソナライズドアラート"""
    alert_id: str
    user_id: str
    alert_type: str
    symbol: str
    condition: str
    threshold: float
    message: str
    is_active: bool
    created_at: datetime
    triggered_at: Optional[datetime]


class UserCentricTradingSystem:
    """ユーザー中心取引支援システム"""
    
    def __init__(self):
        self.analysis_engine = get_analysis_engine()
        
        # ユーザーデータ管理
        self.users_file = Path("data/user_profiles.json")
        self.recommendations_file = Path("data/trading_recommendations.json")
        self.alerts_file = Path("data/personalized_alerts.json")
        
        # データ保存ディレクトリ作成
        for file_path in [self.users_file, self.recommendations_file, self.alerts_file]:
            file_path.parent.mkdir(exist_ok=True)
        
        # インメモリキャッシュ
        self.user_profiles = {}
        self.active_recommendations = {}
        self.active_alerts = {}
        
        # 設定
        self.max_recommendations_per_user = 10
        self.recommendation_validity_hours = 24
        
        from daytrade_logging import get_logger
        self.logger = get_logger("user_centric_trading")
        
        # データ読み込み
        self._load_user_data()
        
        self.logger.info("User-Centric Trading System initialized")
    
    def _load_user_data(self):
        """ユーザーデータを読み込み"""
        try:
            # ユーザープロファイル
            if self.users_file.exists():
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    users_data = json.load(f)
                    for user_id, data in users_data.items():
                        # Enumと日付の復元
                        data['investor_type'] = InvestorProfile(data['investor_type'])
                        data['risk_tolerance'] = RiskTolerance(data['risk_tolerance'])
                        data['preferred_analysis_mode'] = AnalysisMode(data['preferred_analysis_mode'])
                        data['created_at'] = datetime.fromisoformat(data['created_at'])
                        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
                        
                        self.user_profiles[user_id] = UserProfile(**data)
            
            # 取引推奨
            if self.recommendations_file.exists():
                with open(self.recommendations_file, 'r', encoding='utf-8') as f:
                    recommendations_data = json.load(f)
                    for user_id, recs in recommendations_data.items():
                        self.active_recommendations[user_id] = []
                        for rec_data in recs:
                            # Enumと日付の復元
                            rec_data['action'] = TradingSignal(rec_data['action'])
                            rec_data['timestamp'] = datetime.fromisoformat(rec_data['timestamp'])
                            
                            recommendation = TradingRecommendation(**rec_data)
                            self.active_recommendations[user_id].append(recommendation)
            
            # パーソナライズドアラート
            if self.alerts_file.exists():
                with open(self.alerts_file, 'r', encoding='utf-8') as f:
                    alerts_data = json.load(f)
                    for alert_data in alerts_data:
                        # 日付の復元
                        alert_data['created_at'] = datetime.fromisoformat(alert_data['created_at'])
                        if alert_data['triggered_at']:
                            alert_data['triggered_at'] = datetime.fromisoformat(alert_data['triggered_at'])
                        
                        alert = PersonalizedAlert(**alert_data)
                        if alert.user_id not in self.active_alerts:
                            self.active_alerts[alert.user_id] = []
                        self.active_alerts[alert.user_id].append(alert)
            
            self.logger.info(f"Loaded {len(self.user_profiles)} user profiles")
            
        except Exception as e:
            self.logger.error(f"Failed to load user data: {e}")
    
    def _save_user_data(self):
        """ユーザーデータを保存"""
        try:
            # ユーザープロファイル
            users_data = {}
            for user_id, profile in self.user_profiles.items():
                data = asdict(profile)
                # Enumと日付の文字列化
                data['investor_type'] = profile.investor_type.value
                data['risk_tolerance'] = profile.risk_tolerance.value
                data['preferred_analysis_mode'] = profile.preferred_analysis_mode.value
                data['created_at'] = profile.created_at.isoformat()
                data['last_updated'] = profile.last_updated.isoformat()
                users_data[user_id] = data
            
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(users_data, f, indent=2, ensure_ascii=False)
            
            # 取引推奨
            recommendations_data = {}
            for user_id, recommendations in self.active_recommendations.items():
                recs = []
                for rec in recommendations:
                    data = asdict(rec)
                    # Enumと日付の文字列化
                    data['action'] = rec.action.value
                    data['timestamp'] = rec.timestamp.isoformat()
                    recs.append(data)
                recommendations_data[user_id] = recs
            
            with open(self.recommendations_file, 'w', encoding='utf-8') as f:
                json.dump(recommendations_data, f, indent=2, ensure_ascii=False)
            
            # パーソナライズドアラート
            alerts_data = []
            for user_alerts in self.active_alerts.values():
                for alert in user_alerts:
                    data = asdict(alert)
                    # 日付の文字列化
                    data['created_at'] = alert.created_at.isoformat()
                    if alert.triggered_at:
                        data['triggered_at'] = alert.triggered_at.isoformat()
                    alerts_data.append(data)
            
            with open(self.alerts_file, 'w', encoding='utf-8') as f:
                json.dump(alerts_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"Failed to save user data: {e}")
    
    def create_user_profile(self, user_data: Dict[str, Any]) -> UserProfile:
        """ユーザープロファイルを作成"""
        user_id = user_data.get('user_id', f"user_{int(time.time())}")
        
        profile = UserProfile(
            user_id=user_id,
            name=user_data.get('name', 'Unknown User'),
            investor_type=InvestorProfile(user_data.get('investor_type', 'moderate')),
            risk_tolerance=RiskTolerance(user_data.get('risk_tolerance', 'medium')),
            investment_experience_years=user_data.get('investment_experience_years', 1),
            preferred_analysis_mode=AnalysisMode(user_data.get('preferred_analysis_mode', 'enhanced')),
            favorite_symbols=user_data.get('favorite_symbols', ['7203', '8306', '9984']),
            investment_goals=user_data.get('investment_goals', ['capital_growth']),
            daily_trading_budget=user_data.get('daily_trading_budget', 100000.0),
            max_loss_per_trade=user_data.get('max_loss_per_trade', 10000.0),
            notification_preferences=user_data.get('notification_preferences', {
                'price_alerts': True,
                'recommendation_updates': True,
                'market_news': False
            }),
            trading_hours=user_data.get('trading_hours', {
                'start_time': '09:00',
                'end_time': '15:00'
            }),
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.user_profiles[user_id] = profile
        self._save_user_data()
        
        self.logger.info(f"Created user profile: {user_id}")
        return profile
    
    def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """ユーザープロファイルを更新"""
        if user_id not in self.user_profiles:
            return False
        
        profile = self.user_profiles[user_id]
        
        # 更新可能フィールド
        updatable_fields = [
            'name', 'investor_type', 'risk_tolerance', 'investment_experience_years',
            'preferred_analysis_mode', 'favorite_symbols', 'investment_goals',
            'daily_trading_budget', 'max_loss_per_trade', 'notification_preferences',
            'trading_hours'
        ]
        
        for field, value in updates.items():
            if field in updatable_fields and hasattr(profile, field):
                # Enumフィールドの処理
                if field == 'investor_type':
                    value = InvestorProfile(value)
                elif field == 'risk_tolerance':
                    value = RiskTolerance(value)
                elif field == 'preferred_analysis_mode':
                    value = AnalysisMode(value)
                
                setattr(profile, field, value)
        
        profile.last_updated = datetime.now()
        self._save_user_data()
        
        self.logger.info(f"Updated user profile: {user_id}")
        return True
    
    async def generate_personalized_recommendations(self, user_id: str) -> List[TradingRecommendation]:
        """パーソナライズド取引推奨を生成"""
        if user_id not in self.user_profiles:
            return []
        
        profile = self.user_profiles[user_id]
        recommendations = []
        
        try:
            # ユーザーの好みの銘柄を分析
            analysis_results = await self.analysis_engine.analyze_portfolio(
                profile.favorite_symbols, 
                profile.preferred_analysis_mode
            )
            
            for result in analysis_results:
                # リスク許容度に基づく推奨生成
                if self._should_recommend(result, profile):
                    recommendation = self._create_recommendation(result, profile)
                    recommendations.append(recommendation)
            
            # 既存の推奨をクリーンアップ
            self._cleanup_old_recommendations(user_id)
            
            # 新しい推奨を保存
            if user_id not in self.active_recommendations:
                self.active_recommendations[user_id] = []
            
            # 最大数制限
            current_recs = self.active_recommendations[user_id]
            total_recs = current_recs + recommendations
            
            if len(total_recs) > self.max_recommendations_per_user:
                # 古い推奨を削除
                total_recs.sort(key=lambda x: x.timestamp, reverse=True)
                self.active_recommendations[user_id] = total_recs[:self.max_recommendations_per_user]
            else:
                self.active_recommendations[user_id] = total_recs
            
            self._save_user_data()
            
            self.logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations for {user_id}: {e}")
            return []
    
    def _should_recommend(self, analysis_result, profile: UserProfile) -> bool:
        """推奨すべきかどうかの判定"""
        # リスク許容度チェック
        if profile.risk_tolerance == RiskTolerance.VERY_LOW:
            return analysis_result.confidence > 0.8 and analysis_result.signal in [TradingSignal.HOLD, TradingSignal.BUY]
        elif profile.risk_tolerance == RiskTolerance.LOW:
            return analysis_result.confidence > 0.7 and analysis_result.signal != TradingSignal.STRONG_SELL
        elif profile.risk_tolerance == RiskTolerance.MEDIUM:
            return analysis_result.confidence > 0.6
        elif profile.risk_tolerance == RiskTolerance.HIGH:
            return analysis_result.confidence > 0.5
        else:  # VERY_HIGH
            return True
    
    def _create_recommendation(self, analysis_result, profile: UserProfile) -> TradingRecommendation:
        """分析結果から推奨を作成"""
        # ポジションサイズの計算
        position_size = self._calculate_position_size(analysis_result, profile)
        
        # ターゲット価格とストップロスの計算
        target_price, stop_loss = self._calculate_price_targets(analysis_result, profile)
        
        # 期待リターンの計算
        expected_return = self._calculate_expected_return(analysis_result, target_price)
        
        # 保有期間の推定
        holding_period = self._estimate_holding_period(analysis_result, profile)
        
        return TradingRecommendation(
            symbol=analysis_result.symbol,
            action=analysis_result.signal,
            confidence=analysis_result.confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            position_size=position_size,
            reasoning=analysis_result.reasons,
            risk_assessment=analysis_result.risk_level,
            expected_return=expected_return,
            holding_period=holding_period,
            timestamp=datetime.now()
        )
    
    def _calculate_position_size(self, analysis_result, profile: UserProfile) -> float:
        """ポジションサイズの計算"""
        # リスク管理に基づくポジションサイズ
        max_risk = profile.max_loss_per_trade
        confidence_factor = analysis_result.confidence
        
        # 基本ポジションサイズ
        base_size = profile.daily_trading_budget * 0.1  # 予算の10%を基準
        
        # 信頼度による調整
        adjusted_size = base_size * confidence_factor
        
        # リスク制限
        risk_adjusted_size = min(adjusted_size, max_risk / 0.05)  # 5%損失想定
        
        return max(10000, risk_adjusted_size)  # 最低1万円
    
    def _calculate_price_targets(self, analysis_result, profile: UserProfile) -> Tuple[Optional[float], Optional[float]]:
        """目標価格とストップロス価格の計算"""
        current_price = analysis_result.price
        
        if analysis_result.signal in [TradingSignal.BUY, TradingSignal.STRONG_BUY]:
            # 買い推奨の場合
            target_multiplier = 1.05 if profile.risk_tolerance in [RiskTolerance.LOW, RiskTolerance.VERY_LOW] else 1.10
            stop_multiplier = 0.95 if profile.risk_tolerance in [RiskTolerance.HIGH, RiskTolerance.VERY_HIGH] else 0.97
            
            target_price = current_price * target_multiplier
            stop_loss = current_price * stop_multiplier
            
        elif analysis_result.signal in [TradingSignal.SELL, TradingSignal.STRONG_SELL]:
            # 売り推奨の場合
            target_multiplier = 0.95 if profile.risk_tolerance in [RiskTolerance.LOW, RiskTolerance.VERY_LOW] else 0.90
            stop_multiplier = 1.05 if profile.risk_tolerance in [RiskTolerance.HIGH, RiskTolerance.VERY_HIGH] else 1.03
            
            target_price = current_price * target_multiplier
            stop_loss = current_price * stop_multiplier
            
        else:  # HOLD
            target_price = None
            stop_loss = None
        
        return target_price, stop_loss
    
    def _calculate_expected_return(self, analysis_result, target_price: Optional[float]) -> float:
        """期待リターンの計算"""
        if not target_price:
            return 0.0
        
        current_price = analysis_result.price
        confidence = analysis_result.confidence
        
        # 基本リターン
        if analysis_result.signal in [TradingSignal.BUY, TradingSignal.STRONG_BUY]:
            base_return = (target_price - current_price) / current_price
        elif analysis_result.signal in [TradingSignal.SELL, TradingSignal.STRONG_SELL]:
            base_return = (current_price - target_price) / current_price
        else:
            base_return = 0.0
        
        # 信頼度で調整
        return base_return * confidence
    
    def _estimate_holding_period(self, analysis_result, profile: UserProfile) -> str:
        """保有期間の推定"""
        if profile.investor_type == InvestorProfile.DAYTRADER:
            return "短期（1日以内）"
        elif profile.investor_type == InvestorProfile.AGGRESSIVE:
            return "短期（1週間以内）"
        elif profile.investor_type == InvestorProfile.MODERATE:
            return "中期（1ヶ月以内）"
        else:  # CONSERVATIVE
            return "長期（3ヶ月以上）"
    
    def _cleanup_old_recommendations(self, user_id: str):
        """古い推奨をクリーンアップ"""
        if user_id not in self.active_recommendations:
            return
        
        cutoff_time = datetime.now() - timedelta(hours=self.recommendation_validity_hours)
        
        self.active_recommendations[user_id] = [
            rec for rec in self.active_recommendations[user_id]
            if rec.timestamp > cutoff_time
        ]
    
    def create_personalized_alert(self, user_id: str, alert_data: Dict[str, Any]) -> PersonalizedAlert:
        """パーソナライズドアラートを作成"""
        alert = PersonalizedAlert(
            alert_id=f"alert_{user_id}_{int(time.time())}",
            user_id=user_id,
            alert_type=alert_data.get('alert_type', 'price'),
            symbol=alert_data['symbol'],
            condition=alert_data.get('condition', 'above'),
            threshold=float(alert_data['threshold']),
            message=alert_data.get('message', f"{alert_data['symbol']}の価格アラート"),
            is_active=True,
            created_at=datetime.now(),
            triggered_at=None
        )
        
        if user_id not in self.active_alerts:
            self.active_alerts[user_id] = []
        
        self.active_alerts[user_id].append(alert)
        self._save_user_data()
        
        self.logger.info(f"Created alert for user {user_id}: {alert.symbol}")
        return alert
    
    def get_user_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """ユーザーダッシュボードデータを取得"""
        if user_id not in self.user_profiles:
            return {'error': 'User not found'}
        
        profile = self.user_profiles[user_id]
        recommendations = self.active_recommendations.get(user_id, [])
        alerts = self.active_alerts.get(user_id, [])
        
        # 推奨の統計
        buy_recommendations = [r for r in recommendations if r.action in [TradingSignal.BUY, TradingSignal.STRONG_BUY]]
        sell_recommendations = [r for r in recommendations if r.action in [TradingSignal.SELL, TradingSignal.STRONG_SELL]]
        
        # アクティブアラート数
        active_alerts_count = len([a for a in alerts if a.is_active])
        
        return {
            'user_profile': {
                'name': profile.name,
                'investor_type': profile.investor_type.value,
                'risk_tolerance': profile.risk_tolerance.value,
                'favorite_symbols': profile.favorite_symbols,
                'daily_budget': profile.daily_trading_budget
            },
            'recommendations_summary': {
                'total_recommendations': len(recommendations),
                'buy_recommendations': len(buy_recommendations),
                'sell_recommendations': len(sell_recommendations),
                'latest_recommendations': [
                    {
                        'symbol': r.symbol,
                        'action': r.action.value,
                        'confidence': r.confidence,
                        'expected_return': r.expected_return,
                        'timestamp': r.timestamp.isoformat()
                    }
                    for r in recommendations[:5]  # 最新5件
                ]
            },
            'alerts_summary': {
                'total_alerts': len(alerts),
                'active_alerts': active_alerts_count,
                'recent_alerts': [
                    {
                        'symbol': a.symbol,
                        'alert_type': a.alert_type,
                        'condition': a.condition,
                        'threshold': a.threshold,
                        'is_active': a.is_active
                    }
                    for a in alerts[-5:]  # 最新5件
                ]
            },
            'timestamp': datetime.now().isoformat()
        }


# グローバルインスタンス
_trading_system = None


def get_trading_system() -> UserCentricTradingSystem:
    """グローバル取引システムを取得"""
    global _trading_system
    if _trading_system is None:
        _trading_system = UserCentricTradingSystem()
    return _trading_system


async def generate_user_recommendations(user_id: str) -> List[TradingRecommendation]:
    """ユーザー推奨生成（便利関数）"""
    return await get_trading_system().generate_personalized_recommendations(user_id)


def get_user_dashboard(user_id: str) -> Dict[str, Any]:
    """ユーザーダッシュボード取得（便利関数）"""
    return get_trading_system().get_user_dashboard_data(user_id)


if __name__ == "__main__":
    import asyncio
    
    print("👤 ユーザー中心取引支援システムテスト")
    print("=" * 50)
    
    system = UserCentricTradingSystem()
    
    # テストユーザー作成
    test_user_data = {
        'user_id': 'test_user_001',
        'name': 'テスト太郎',
        'investor_type': 'moderate',
        'risk_tolerance': 'medium',
        'investment_experience_years': 3,
        'preferred_analysis_mode': 'enhanced',
        'favorite_symbols': ['7203', '8306', '9984', '6758'],
        'daily_trading_budget': 500000.0,
        'max_loss_per_trade': 50000.0
    }
    
    profile = system.create_user_profile(test_user_data)
    print(f"テストユーザー作成: {profile.name}")
    
    # パーソナライズド推奨生成
    async def test_recommendations():
        recommendations = await system.generate_personalized_recommendations('test_user_001')
        print(f"\n推奨生成: {len(recommendations)}件")
        
        for rec in recommendations[:3]:  # 最初の3件を表示
            print(f"  {rec.symbol}: {rec.action.value} (信頼度: {rec.confidence:.1%})")
            print(f"    期待リターン: {rec.expected_return:.1%}")
            print(f"    ポジションサイズ: {rec.position_size:,.0f}円")
    
    asyncio.run(test_recommendations())
    
    # アラート作成
    alert_data = {
        'symbol': '7203',
        'alert_type': 'price',
        'condition': 'above',
        'threshold': 3000.0,
        'message': 'トヨタ株価が3000円を超えました'
    }
    
    alert = system.create_personalized_alert('test_user_001', alert_data)
    print(f"\nアラート作成: {alert.symbol} ({alert.condition} {alert.threshold})")
    
    # ダッシュボードデータ
    dashboard = system.get_user_dashboard_data('test_user_001')
    print(f"\nダッシュボードデータ:")
    print(f"  総推奨数: {dashboard['recommendations_summary']['total_recommendations']}")
    print(f"  アクティブアラート: {dashboard['alerts_summary']['active_alerts']}")
    
    print("\nテスト完了")