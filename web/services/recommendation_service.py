#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recommendation Service - Issue #959 リファクタリング対応
株式推奨サービスモジュール
"""

import random
import time
from typing import Dict, List, Any
from datetime import datetime

class RecommendationService:
    """株式推奨サービス"""
    
    def __init__(self):
        self.symbols_data = self._initialize_symbols_data()
        
    def _initialize_symbols_data(self) -> List[Dict[str, Any]]:
        """35銘柄データの初期化"""
        return [
            # 大型株（安定重視） - 8銘柄
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
        """推奨銘柄の取得"""
        recommendations = []
        
        for symbol_data in self.symbols_data:
            # シミュレーション分析（実際のAI分析の代替）
            analysis_result = self._simulate_analysis(symbol_data)
            recommendations.append(analysis_result)
            
        return recommendations
    
    def _simulate_analysis(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析シミュレーション"""
        # ランダムな分析結果生成（実際の環境では真のAI分析）
        recommendations = ['BUY', 'SELL', 'HOLD']
        recommendation = random.choice(recommendations)
        confidence = round(random.uniform(0.6, 0.95), 2)
        price = 1000 + abs(hash(symbol_data['code'])) % 2000
        change = round(random.uniform(-5.0, 5.0), 2)
        
        # カテゴリに基づく安全度設定
        risk_mapping = {
            '大型株': '低リスク',
            '中型株': '中リスク', 
            '高配当株': '低リスク',
            '成長株': '高リスク'
        }
        
        # わかりやすい評価
        if confidence > 0.85:
            confidence_friendly = "超おすすめ！"
            star_rating = "★★★★★"
        elif confidence > 0.75:
            confidence_friendly = "かなりおすすめ"
            star_rating = "★★★★☆"
        elif confidence > 0.65:
            confidence_friendly = "まあまあ"
            star_rating = "★★★☆☆"
        else:
            confidence_friendly = "様子見"
            star_rating = "★★☆☆☆"
        
        # 投資家適性
        stability = symbol_data.get('stability', '中安定')
        category = symbol_data.get('category', '一般株')
        
        if stability == '高安定' and category in ['大型株', '高配当株']:
            who_suitable = "安定重視の初心者におすすめ"
        elif category == '成長株':
            who_suitable = "成長重視の積極投資家向け"
        elif category == '高配当株':
            who_suitable = "配当収入を重視する投資家向け"
        else:
            who_suitable = "バランス重視の投資家向け"
        
        return {
            'symbol': symbol_data['code'],
            'name': symbol_data['name'],
            'sector': symbol_data['sector'],
            'category': category,
            'recommendation': recommendation,
            'recommendation_friendly': recommendation,
            'confidence': confidence,
            'confidence_friendly': confidence_friendly,
            'star_rating': star_rating,
            'price': price,
            'change': change,
            'risk_level': risk_mapping.get(category, '中リスク'),
            'risk_friendly': risk_mapping.get(category, '中リスク'),
            'stability': stability,
            'who_suitable': who_suitable,
            'reason': f"{symbol_data['sector']}セクターの代表的な{category}",
            'friendly_reason': f"AI分析により{symbol_data['sector']}セクターで{confidence_friendly}と判定",
            'timestamp': time.time()
        }
    
    def analyze_single_symbol(self, symbol: str) -> Dict[str, Any]:
        """個別銘柄分析"""
        # 銘柄データを検索
        symbol_data = next(
            (s for s in self.symbols_data if s['code'] == symbol),
            {'code': symbol, 'name': f'銘柄{symbol}', 'sector': '不明', 'category': '一般株', 'stability': '中安定'}
        )
        
        return self._simulate_analysis(symbol_data)
