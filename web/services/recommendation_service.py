#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recommendation Service - Issue #939対応
"""

import random
import time
import sys
import os
import json
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from src.day_trade.data.providers.real_data_provider import ImprovedMultiSourceDataProvider
    from src.day_trade.analysis.technical_indicators import create_trading_recommendation_pl
    REAL_DATA_AVAILABLE = True
except ImportError as e:
    print(f"リアルデータモジュール読み込みエラー: {e}")
    REAL_DATA_AVAILABLE = False

class RecommendationService:
    """株式推奨サービス (モデル選択対応)"""
    
    CACHE_EXPIRATION_SECONDS = 3600

    def __init__(self, redis_client: Optional[Any] = None):
        self.symbols_data = self._initialize_symbols_data()
        self.redis_client = redis_client
        
    def _initialize_symbols_data(self) -> List[Dict[str, Any]]:
        # ... (same as before) ...
        return [
            {'code': '7203', 'name': 'トヨタ自動車', 'sector': '自動車', 'category': '大型株', 'stability': '高安定'},
            {'code': '8306', 'name': '三菱UFJ銀行', 'sector': '金融', 'category': '大型株', 'stability': '高安定'},
            {'code': '9984', 'name': 'ソフトバンクグループ', 'sector': 'テクノロジー', 'category': '大型株', 'stability': '中安定'},
            {'code': '6758', 'name': 'ソニー', 'sector': 'テクノロジー', 'category': '大型株', 'stability': '高安定'},
            {'code': '4689', 'name': 'Z Holdings', 'sector': 'テクノロジー', 'category': '大型株', 'stability': '中安定'},
            {'code': '9434', 'name': 'ソフトバンク', 'sector': '通信', 'category': '大型株', 'stability': '高安定'},
            {'code': '8001', 'name': '伊藤忠商事', 'sector': '商社', 'category': '大型株', 'stability': '高安定'},
            {'code': '7267', 'name': 'ホンダ', 'sector': '自動車', 'category': '大型株', 'stability': '高安定'},
            
            # 中型株（成長期待） - 9銘柄
            {'code': '6861', 'name': 'キーエンス', 'sector': '精密機器', 'category': '中型株', 'stability': '中安定'},
            {'code': '4755', 'name': '楽天グループ', 'sector': 'テクノロジー', 'category': '中型株', 'stability': '低安定'},
            {'code': '4502', 'name': '武田薬品工業', 'sector': '製薬', 'category': '中型株', 'stability': '中安定'},
            {'code': '9983', 'name': 'ファーストリテイリング', 'sector': 'アパレル', 'category': '中型株', 'stability': '中安定'},
            {'code': '7974', 'name': '任天堂', 'sector': 'ゲーム', 'category': '中型株', 'stability': '中安定'},
            {'code': '6954', 'name': 'ファナック', 'sector': '工作機械', 'category': '中型株', 'stability': '中安定'},
            {'code': '8316', 'name': '三井住友FG', 'sector': '金融', 'category': '中型株', 'stability': '高安定'},
            {'code': '4578', 'name': '大塚ホールディングス', 'sector': '製薬', 'category': '中型株', 'stability': '中安定'},
            {'code': '8058', 'name': '三菱商事', 'sector': '商社', 'category': '中型株', 'stability': '高安定'},
            
            # 高配当株（収益重視） - 9銘柄
            {'code': '8031', 'name': '三井物産', 'sector': '商社', 'category': '高配当株', 'stability': '高安定'},
            {'code': '1605', 'name': 'INPEX', 'sector': 'エネルギー', 'category': '高配当株', 'stability': '中安定'},
            {'code': '8766', 'name': '東京海上HD', 'sector': '保険', 'category': '高配当株', 'stability': '高安定'},
            {'code': '2914', 'name': '日本たばこ産業', 'sector': 'タバコ', 'category': '高配当株', 'stability': '高安定'},
            {'code': '8411', 'name': 'みずほFG', 'sector': '金融', 'category': '高配当株', 'stability': '中安定'},
            {'code': '5401', 'name': '日本製鉄', 'sector': '鉄鋼', 'category': '高配当株', 'stability': '中安定'},
            {'code': '9433', 'name': 'KDDI', 'sector': '通信', 'category': '高配当株', 'stability': '高安定'},
            {'code': '2802', 'name': '味の素', 'sector': '食品', 'category': '高配当株', 'stability': '高安定'},
            {'code': '3382', 'name': '7&i HD', 'sector': '小売', 'category': '高配当株', 'stability': '高安定'},
            
            # 成長株（将来性重視） - 9銘柄
            {'code': '4503', 'name': 'アステラス製薬', 'sector': '製薬', 'category': '成長株', 'stability': '中安定'},
            {'code': '6981', 'name': '村田製作所', 'sector': '電子部品', 'category': '成長株', 'stability': '中安定'},
            {'code': '8035', 'name': '東京エレクトロン', 'sector': '半導体', 'category': '成長株', 'stability': '低安定'},
            {'code': '4751', 'name': 'サイバーエージェント', 'sector': 'テクノロジー', 'category': '成長株', 'stability': '低安定'},
            {'code': '3659', 'name': 'ネクソン', 'sector': 'ゲーム', 'category': '成長株', 'stability': '低安定'},
            {'code': '4385', 'name': 'メルカリ', 'sector': 'テクノロジー', 'category': '成長株', 'stability': '低安定'},
            {'code': '4704', 'name': 'トレンドマイクロ', 'sector': 'セキュリティ', 'category': '成長株', 'stability': '中安定'},
            {'code': '2491', 'name': 'バリューコマース', 'sector': '広告', 'category': '成長株', 'stability': '低安定'},
            {'code': '3900', 'name': 'クラウドワークス', 'sector': 'テクノロジー', 'category': '成長株', 'stability': '低安定'}
        ]

    def get_recommendations(self) -> List[Dict[str, Any]]:
        # ... (same as before) ...
        recommendations = []
        for symbol_data in self.symbols_data:
            analysis_result = self.analyze_single_symbol(symbol_data['code'])
            recommendations.append(analysis_result)
        return recommendations

    def analyze_single_symbol(self, symbol: str, model_type: str = 'lightgbm') -> Dict[str, Any]:
        """個別銘柄分析 (モデル選択対応)"""
        if self.redis_client:
            cache_key = f"recommendation:{symbol}:{model_type}"
            try:
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
            except Exception as e:
                print(f"Redis GET error: {e}")

        symbol_data = next((s for s in self.symbols_data if s['code'] == symbol), None)
        if not symbol_data:
            symbol_data = {'code': symbol, 'name': f'銘柄{symbol}', 'sector': '不明', 'category': '一般株', 'stability': '中安定'}
        
        analysis_result = self._perform_actual_analysis(symbol_data, model_type)

        if self.redis_client:
            cache_key = f"recommendation:{symbol}:{model_type}"
            try:
                self.redis_client.setex(cache_key, self.CACHE_EXPIRATION_SECONDS, json.dumps(analysis_result))
            except Exception as e:
                print(f"Redis SETEX error: {e}")

        return analysis_result

    def _perform_actual_analysis(self, symbol_data: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        try:
            analysis_result = self._get_real_analysis_pl(symbol_data['code'], model_type)
            if analysis_result:
                return self._enrich_real_analysis(analysis_result, symbol_data)
        except Exception as e:
            print(f"リアル分析エラー ({symbol_data['code']}): {e}")
        
        return self._create_simulated_analysis(symbol_data)

    def _get_real_analysis_pl(self, symbol: str, model_type: str) -> Optional[Dict[str, Any]]:
        if not REAL_DATA_AVAILABLE:
            return None
        
        try:
            provider = ImprovedMultiSourceDataProvider()
            jp_symbol = f"{symbol}.T"
            result = provider.get_stock_data_sync(jp_symbol, period="1mo")
            
            if result.success and result.data is not None and len(result.data) > 20:
                pl_data = pl.from_pandas(result.data.reset_index())
                
                recommendation = create_trading_recommendation_pl(
                    symbol=symbol,
                    data=pl_data,
                    account_balance=1000000,
                    model_type=model_type
                )
                recommendation['real_data'] = True
                return recommendation
            
        except Exception as e:
            print(f"Polarsリアル分析エラー ({symbol}): {e}")
        
        return None

    def _enrich_real_analysis(self, analysis: Dict[str, Any], symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        # ... (same as before, but add model_used)
        recommendation = analysis.get('signal', 'HOLD')
        confidence = analysis.get('confidence', 0.7)
        price = analysis.get('current_price', 0)
        category = symbol_data.get('category', '一般株')

        enriched_result = analysis.copy()
        enriched_result.update({
            'name': symbol_data['name'],
            'sector': symbol_data['sector'],
            'category': category,
            'recommendation': recommendation,
            'recommendation_friendly': recommendation,
            'confidence_friendly': self._get_confidence_friendly(confidence),
            'star_rating': self._get_star_rating(confidence),
            'change': round(price - analysis.get('previous_close', price), 2), # 仮
            'risk_level': self._get_risk_level(category),
            'risk_friendly': self._get_risk_level(category),
            'stability': symbol_data.get('stability', '中安定'),
            'who_suitable': self._get_who_suitable(symbol_data),
            'friendly_reason': f"AIのリアルタイム分析により、{analysis.get('reason', '総合的に判断')}",
            'action': self._get_action_advice(recommendation, confidence),
            'timing': self._get_timing_advice(recommendation),
            'amount_suggestion': self._get_amount_suggestion(price, category),
            'timestamp': time.time(),
            'model_used': analysis.get('model_used', 'rule_based') # モデル情報を追加
        })
        return enriched_result

    def _create_simulated_analysis(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        # ... (same as before) ...
        recommendation = random.choice(['BUY', 'SELL', 'HOLD'])
        confidence = round(random.uniform(0.6, 0.95), 2)
        price = 1000 + abs(hash(symbol_data['code'])) % 2000
        category = symbol_data.get('category', '一般株')

        return {
            'action': self._get_action_advice(recommendation, confidence),
            'timing': self._get_timing_advice(recommendation),
            'amount_suggestion': self._get_amount_suggestion(price, category),
            'symbol': symbol_data['code'],
            'name': symbol_data['name'],
            'sector': symbol_data['sector'],
            'category': category,
            'recommendation': recommendation,
            'recommendation_friendly': recommendation,
            'confidence': confidence,
            'confidence_friendly': self._get_confidence_friendly(confidence),
            'star_rating': self._get_star_rating(confidence),
            'price': price,
            'change': round(random.uniform(-5.0, 5.0), 2),
            'risk_level': self._get_risk_level(category),
            'risk_friendly': self._get_risk_level(category),
            'stability': symbol_data.get('stability', '中安定'),
            'who_suitable': self._get_who_suitable(symbol_data),
            'reason': f"{symbol_data['sector']}セクターの代表的な{category}",
            'friendly_reason': f"AI分析により{symbol_data['sector']}セクターで{self._get_confidence_friendly(confidence)}と判定",
            'timestamp': time.time(),
            'real_data': False,
            'model_used': 'simulation'
        }

    # --- Helper methods (no change) ---
    def _get_confidence_friendly(self, confidence: float) -> str:
        if confidence > 0.85: return "超おすすめ！"
        if confidence > 0.75: return "かなりおすすめ"
        if confidence > 0.65: return "まあまあ"
        return "様子見"

    def _get_star_rating(self, confidence: float) -> str:
        if confidence > 0.85: return "★★★★★"
        if confidence > 0.75: return "★★★★☆"
        if confidence > 0.65: return "★★★☆☆"
        return "★★☆☆☆"

    def _get_risk_level(self, category: str) -> str:
        risk_mapping = {'大型株': '低リスク', '中型株': '中リスク', '高配当株': '低リスク', '成長株': '高リスク'}
        return risk_mapping.get(category, '中リスク')

    def _get_who_suitable(self, symbol_data: Dict[str, Any]) -> str:
        stability = symbol_data.get('stability', '中安定')
        category = symbol_data.get('category', '一般株')
        if stability == '高安定' and category in ['大型株', '高配当株']: return "安定重視の初心者におすすめ"
        if category == '成長株': return "成長重視の積極投資家向け"
        if category == '高配当株': return "配当収入を重視する投資家向け"
        return "バランス重視の投資家向け"

    def _get_action_advice(self, recommendation: str, confidence: float) -> str:
        if recommendation == 'BUY':
            return "今すぐ購入を検討してください" if confidence > 0.8 else "少量から購入を始めてみてください"
        if recommendation == 'SELL':
            return "早めの売却を検討してください" if confidence > 0.8 else "様子を見ながら部分的な売却を検討してください"
        return "しばらく様子を見てください"

    def _get_timing_advice(self, recommendation: str) -> str:
        if recommendation == 'BUY': return "次の押し目で購入タイミング"
        if recommendation == 'SELL': return "次の戻りで売却タイミング"
        return "明確なシグナルを待ちましょう"

    def _get_amount_suggestion(self, price: float, category: str) -> str:
        if category == '大型株': return "資金の5-10%を目安に"
        if category == '成長株': return "資金の2-5%を目安に（リスク高）"
        if category == '高配当株': return "資金の10-15%を目安に（安定重視）"
        return "資金の3-8%を目安に"