#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recommendation Service - Issue #939対応
"""

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
    print("Yahoo Finance APIが利用可能です")
except ImportError as e:
    print(f"Yahoo Finance読み込みエラー: {e}")
    try:
        from src.day_trade.data.providers.real_data_provider import ImprovedMultiSourceDataProvider
        REAL_DATA_AVAILABLE = True
        print("リアルデータプロバイダーが利用可能です")
    except ImportError:
        REAL_DATA_AVAILABLE = False
        print("リアルデータモジュールが利用できません")

class RecommendationService:
    """株式推奨サービス (センチメント特徴量対応)"""

    CACHE_EXPIRATION_SECONDS = 3600

    def __init__(self, redis_client: Optional[Any] = None):
        self.symbols_data = self._initialize_symbols_data()
        self.redis_client = redis_client

    def _initialize_symbols_data(self) -> List[Dict[str, Any]]:
        """Issues #953-956対応: 35銘柄に拡張（多様化・リスク分散強化）"""
        return [
            # 大型株（8銘柄） - 安定性重視
            {'code': '7203', 'name': 'トヨタ自動車', 'sector': '自動車', 'category': '大型株', 'stability': '高安定'},
            {'code': '9984', 'name': 'ソフトバンクグループ', 'sector': 'テクノロジー', 'category': '大型株', 'stability': '中安定'},
            {'code': '6758', 'name': 'ソニー', 'sector': 'テクノロジー', 'category': '大型株', 'stability': '高安定'},
            {'code': '7974', 'name': '任天堂', 'sector': 'ゲーム', 'category': '大型株', 'stability': '中安定'},
            {'code': '8306', 'name': '三菱UFJ銀行', 'sector': '金融', 'category': '大型株', 'stability': '高安定'},
            {'code': '8001', 'name': '伊藤忠商事', 'sector': '商社', 'category': '大型株', 'stability': '高安定'},
            {'code': '4502', 'name': '武田薬品工業', 'sector': '医薬品', 'category': '大型株', 'stability': '中安定'},
            {'code': '6501', 'name': '日立製作所', 'sector': 'テクノロジー', 'category': '大型株', 'stability': '高安定'},
            
            # 中型株（8銘柄） - 成長期待
            {'code': '4755', 'name': '楽天グループ', 'sector': 'テクノロジー', 'category': '中型株', 'stability': '低安定'},
            {'code': '4385', 'name': 'メルカリ', 'sector': 'テクノロジー', 'category': '中型株', 'stability': '低安定'},
            {'code': '4689', 'name': 'Z Holdings', 'sector': 'テクノロジー', 'category': '中型株', 'stability': '中安定'},
            {'code': '9437', 'name': 'NTTドコモ', 'sector': '通信', 'category': '中型株', 'stability': '高安定'},
            {'code': '2914', 'name': '日本たばこ産業', 'sector': '食品・タバコ', 'category': '中型株', 'stability': '高安定'},
            {'code': '4704', 'name': 'トレンドマイクロ', 'sector': 'セキュリティ', 'category': '中型株', 'stability': '中安定'},
            {'code': '4751', 'name': 'サイバーエージェント', 'sector': 'テクノロジー', 'category': '中型株', 'stability': '低安定'},
            {'code': '3659', 'name': 'ネクソン', 'sector': 'ゲーム', 'category': '中型株', 'stability': '低安定'},
            
            # 高配当株（8銘柄） - 収益重視
            {'code': '8593', 'name': '三菱HCキャピタル', 'sector': '金融', 'category': '高配当株', 'stability': '中安定'},
            {'code': '8316', 'name': 'みずほフィナンシャルグループ', 'sector': '金融', 'category': '高配当株', 'stability': '中安定'},
            {'code': '5020', 'name': 'JXTGホールディングス', 'sector': 'エネルギー', 'category': '高配当株', 'stability': '中安定'},
            {'code': '8411', 'name': 'みずほ銀行', 'sector': '金融', 'category': '高配当株', 'stability': '中安定'},
            {'code': '9201', 'name': '日本航空', 'sector': '航空', 'category': '高配当株', 'stability': '中安定'},
            {'code': '9432', 'name': '日本電信電話', 'sector': '通信', 'category': '高配当株', 'stability': '高安定'},
            {'code': '8604', 'name': '野村ホールディングス', 'sector': '金融', 'category': '高配当株', 'stability': '中安定'},
            {'code': '7751', 'name': 'キヤノン', 'sector': '精密機器', 'category': '高配当株', 'stability': '中安定'},
            
            # 成長株（6銘柄） - 将来性重視
            {'code': '4503', 'name': 'アステラス製薬', 'sector': '医薬品', 'category': '成長株', 'stability': '中安定'},
            {'code': '6981', 'name': '村田製作所', 'sector': '電子部品', 'category': '成長株', 'stability': '中安定'},
            {'code': '8035', 'name': '東京エレクトロン', 'sector': '半導体', 'category': '成長株', 'stability': '低安定'},
            {'code': '4704', 'name': 'トレンドマイクロ', 'sector': 'セキュリティ', 'category': '成長株', 'stability': '中安定'},
            {'code': '2491', 'name': 'バリューコマース', 'sector': 'テクノロジー', 'category': '成長株', 'stability': '低安定'},
            {'code': '3900', 'name': 'クラウドワークス', 'sector': 'テクノロジー', 'category': '成長株', 'stability': '低安定'},
            
            # 小型株（5銘柄） - 新規カテゴリ（上級者・積極投資家向け）
            {'code': '4478', 'name': 'フリー', 'sector': 'テクノロジー', 'category': '小型株', 'stability': '低安定'},
            {'code': '3900', 'name': 'クラウドワークス', 'sector': 'テクノロジー', 'category': '小型株', 'stability': '低安定'},
            {'code': '4375', 'name': 'セーフィー', 'sector': 'テクノロジー', 'category': '小型株', 'stability': '低安定'},
            {'code': '4420', 'name': 'イーソル', 'sector': 'テクノロジー', 'category': '小型株', 'stability': '低安定'},
            {'code': '4475', 'name': 'HENNGE', 'sector': 'テクノロジー', 'category': '小型株', 'stability': '低安定'}
        ]

    def get_recommendations(self) -> List[Dict[str, Any]]:
        # ... (same as before) ...
        recommendations = []
        for symbol_data in self.symbols_data:
            analysis_result = self.analyze_single_symbol(symbol_data['code'])
            recommendations.append(analysis_result)
        
        # BUY推奨銘柄を一番上に表示するようにソート
        def sort_by_recommendation(rec):
            # BUY系推奨を最優先、その後confidence順
            if rec['recommendation'] in ['BUY', 'STRONG_BUY']:
                return (0, -rec['confidence'])  # BUY系は0で最優先、confidenceで降順
            elif rec['recommendation'] == 'HOLD':
                return (1, -rec['confidence'])  # HOLDは2番目
            else:  # SELL, STRONG_SELL
                return (2, -rec['confidence'])  # SELLは最後
        
        recommendations.sort(key=sort_by_recommendation)
        
        return recommendations

    def analyze_single_symbol(self, symbol: str, model_type: str = 'lightgbm') -> Dict[str, Any]:
        # ... (same as before) ...
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
            analysis_result = self._get_real_analysis_pl(symbol_data, model_type)
            if analysis_result:
                return self._enrich_real_analysis(analysis_result, symbol_data)
        except Exception as e:
            print(f"リアル分析エラー ({symbol_data['code']}): {e}")

        # フォールバック分析（実データベース）
        fallback_result = self._create_simulated_analysis(symbol_data)
        if fallback_result:
            return fallback_result
            
        # 最後の手段として基本情報のみ返す
        print(f"警告: {symbol_data['code']} のデータを取得できません。基本情報のみ表示")
        return {
            'symbol': symbol_data['code'],
            'name': symbol_data['name'],
            'sector': symbol_data['sector'],
            'category': symbol_data.get('category', '一般株'),
            'recommendation': 'データ取得不可',
            'confidence': 0.0,
            'price': 0,
            'change': 0,
            'error': 'リアルタイムデータを取得できませんでした',
            'real_data': False,
            'model_used': 'error'
        }

    def _get_real_analysis_pl(self, symbol_data: Dict[str, Any], model_type: str) -> Optional[Dict[str, Any]]:
        if not REAL_DATA_AVAILABLE:
            return None

        try:
            symbol = symbol_data['code']
            company_name = symbol_data['name']

            provider = ImprovedMultiSourceDataProvider()
            jp_symbol = f"{symbol}.T"
            result = provider.get_stock_data_sync(jp_symbol, period="1mo")

            if result.success and result.data is not None and len(result.data) > 20:
                pl_data = pl.from_pandas(result.data.reset_index())

                recommendation = create_trading_recommendation_pl(
                    symbol=symbol,
                    company_name=company_name,
                    data=pl_data,
                    account_balance=1000000,
                    model_type=model_type
                )
                recommendation['real_data'] = True
                return recommendation

        except Exception as e:
            print(f"Polarsリアル分析エラー ({symbol}): {e}")

        return None

    def _get_real_price(self, symbol: str) -> Optional[float]:
        """実際の株価データを取得（簡易版）"""
        if not REAL_DATA_AVAILABLE:
            return None
            
        try:
            provider = ImprovedMultiSourceDataProvider()
            jp_symbol = f"{symbol}.T"
            result = provider.get_stock_data_sync(jp_symbol, period="1d")
            
            if result.success and result.data is not None and len(result.data) > 0:
                # 最新の終値を取得
                latest_price = float(result.data['Close'].iloc[-1])
                print(f"実際の株価取得成功 {symbol}: ¥{latest_price:,.0f}")
                return latest_price
        except Exception as e:
            print(f"株価取得エラー ({symbol}): {e}")
        
        return None

    def _get_real_price_with_history(self, symbol: str) -> Optional[Dict[str, float]]:
        """実際の株価データ（現在価格＋前日終値）を取得"""
        if not REAL_DATA_AVAILABLE:
            return None
            
        try:
            provider = ImprovedMultiSourceDataProvider()
            jp_symbol = f"{symbol}.T"
            result = provider.get_stock_data_sync(jp_symbol, period="5d")  # 5日分取得
            
            if result.success and result.data is not None and len(result.data) >= 2:
                current_price = float(result.data['Close'].iloc[-1])
                previous_close = float(result.data['Close'].iloc[-2])
                
                print(f"実際の株価取得成功 {symbol}: 現在¥{current_price:,.0f} (前日¥{previous_close:,.0f})")
                
                return {
                    'current_price': current_price,
                    'previous_close': previous_close,
                    'high': float(result.data['High'].iloc[-1]),
                    'low': float(result.data['Low'].iloc[-1]),
                    'volume': float(result.data['Volume'].iloc[-1])
                }
        except Exception as e:
            print(f"株価履歴取得エラー ({symbol}): {e}")
        
        return None

    def _enrich_real_analysis(self, analysis: Dict[str, Any], symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        # ... (same as before) ...
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
            'model_used': analysis.get('model_used', 'rule_based')
        })
        return enriched_result

    def _create_simulated_analysis(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """リアルデータフォールバック分析（模擬データ完全排除版）"""
        symbol = symbol_data['code']
        
        # 実際の株価データを必ず取得
        real_price_data = self._get_real_price_with_history(symbol)
        if not real_price_data:
            print(f"警告: {symbol} の実際の価格データを取得できません")
            return None
        
        price = real_price_data['current_price']
        previous_close = real_price_data.get('previous_close', price)
        change = round(price - previous_close, 2)
        change_pct = round((change / previous_close) * 100, 2) if previous_close > 0 else 0
        
        category = symbol_data.get('category', '一般株')
        
        # 価格変動とセクターに基づく推奨決定（実際の技術分析風）
        if change_pct > 2.0:
            recommendation = 'BUY'
            confidence = min(0.9, 0.7 + (change_pct * 0.05))
        elif change_pct < -2.0:
            recommendation = 'SELL'  
            confidence = min(0.9, 0.7 + (abs(change_pct) * 0.05))
        else:
            recommendation = 'HOLD'
            confidence = 0.65 + (abs(change_pct) * 0.1)
        
        confidence = round(confidence, 2)

        # セクター特性に基づく戦略選択
        sector_strategies = {
            '自動車': '中期', '医薬品': '中期', '金融': 'スイング',
            'テクノロジー': 'デイトレ', 'ゲーム': 'スイング', 
            '半導体': 'デイトレ', 'エネルギー': '中期'
        }
        
        strategy = sector_strategies.get(symbol_data['sector'], 'スイング')
        
        # 戦略別の予測日数（固定値、非ランダム）
        strategy_days = {
            'デイトレ': 1,
            'スイング': 5 if confidence > 0.8 else 8,
            '中期': 30 if confidence > 0.8 else 45
        }
        
        predicted_sell_day = strategy_days[strategy]
        
        # 利確・損切り目標（セクター基準）
        sector_targets = {
            '自動車': (8, 4), 'テクノロジー': (15, 8), '医薬品': (10, 5),
            '金融': (12, 6), 'ゲーム': (20, 10), '半導体': (18, 9)
        }
        target_profit_pct, stop_loss_pct = sector_targets.get(symbol_data['sector'], (10, 5))
        
        # 売却理由
        sell_reasons = []
        if confidence > 0.85:
            sell_reasons.append(f'利確目標{target_profit_pct}%達成時')
        if confidence < 0.7:
            sell_reasons.append(f'損切り-{stop_loss_pct}%到達時')
        sell_reasons.append(f'{predicted_sell_day}日後のテクニカル転換点')
        
        # 売買タイミング予測（実データベース）
        buy_timing, sell_timing = self._calculate_trading_timing(
            price, previous_close, change_pct, symbol_data['sector'], strategy, confidence
        )

        return {
            'action': self._get_action_advice(recommendation, confidence),
            'timing': self._get_timing_advice(recommendation),
            'amount_suggestion': self._get_amount_suggestion(price, category),
            'symbol': symbol,
            'name': symbol_data['name'],
            'sector': symbol_data['sector'],
            'category': category,
            'recommendation': recommendation,
            'recommendation_friendly': recommendation,
            'confidence': confidence,
            'confidence_friendly': self._get_confidence_friendly(confidence),
            'star_rating': self._get_star_rating(confidence),
            'price': price,
            'change': change,
            'risk_level': self._get_risk_level(category),
            'risk_friendly': self._get_risk_level(category),
            'stability': symbol_data.get('stability', '中安定'),
            'who_suitable': self._get_who_suitable(symbol_data),
            'reason': f"{symbol_data['sector']}セクター・実際の価格変動{change_pct:+.1f}%に基づく分析",
            'friendly_reason': f"実際の株価データ(¥{price:,.0f})に基づき{symbol_data['sector']}セクターで{self._get_confidence_friendly(confidence)}と判定",
            'timestamp': time.time(),
            'real_data': True,  # 実データベースに変更
            'model_used': 'real_price_analysis',
            'buy_timing': buy_timing,
            'sell_timing': sell_timing,
            'sell_prediction': {
                'strategy': strategy,
                'predicted_days': predicted_sell_day,
                'sell_reasons': sell_reasons,
                'target_profit': f"{target_profit_pct}%",
                'stop_loss': f"-{stop_loss_pct}%",
                'description': f"{predicted_sell_day}日間保有（{strategy}戦略）"
            }
        }

    def _calculate_trading_timing(self, current_price: float, previous_close: float, 
                                change_pct: float, sector: str, strategy: str, confidence: float) -> tuple:
        """実際のデータに基づく売買タイミング予測（時間詳細版）"""
        
        # 現在の日時を基準
        from datetime import datetime, timedelta
        now = datetime.now()
        
        # 購入タイミング予測（時間詳細・決定論的）
        if change_pct > 1.5:  # 大幅上昇
            buy_days = 2
            buy_hour = 10 if confidence > 0.8 else 14  # 高信頼度なら朝、そうでなければ午後
            buy_timing_desc = "押し目待ち"
        elif change_pct < -1.5:  # 大幅下落
            buy_days = 0
            buy_hour = 10 if abs(change_pct) > 3.0 else 13  # 大幅下落なら午前、中程度なら午後
            buy_timing_desc = "今すぐ購入"
        elif abs(change_pct) < 0.5:  # 小動き
            buy_days = 6
            buy_hour = 9  # 様子見後の寄り付き
            buy_timing_desc = "様子見後購入"
        else:  # 通常の動き
            buy_days = 1
            buy_hour = 9  # 翌日寄り付き
            buy_timing_desc = "寄り付き購入"
        
        # セクター特性を考慮した時間調整
        sector_adjustments = {
            'テクノロジー': {'volatility': 'high', 'timing_factor': 0.8, 'preferred_hours': [9, 10, 14, 15]},
            '半導体': {'volatility': 'high', 'timing_factor': 0.7, 'preferred_hours': [9, 10, 13, 14]},
            'ゲーム': {'volatility': 'high', 'timing_factor': 0.9, 'preferred_hours': [10, 11, 14, 15]},
            '自動車': {'volatility': 'low', 'timing_factor': 1.2, 'preferred_hours': [9, 10, 11]},
            '医薬品': {'volatility': 'low', 'timing_factor': 1.3, 'preferred_hours': [9, 10, 11, 13]},
            '金融': {'volatility': 'medium', 'timing_factor': 1.1, 'preferred_hours': [9, 10, 14]}
        }
        
        sector_info = sector_adjustments.get(sector, {
            'volatility': 'medium', 
            'timing_factor': 1.0, 
            'preferred_hours': [9, 10, 11, 14]
        })
        
        # セクター特性に応じた時間調整（決定論的）
        if sector in sector_adjustments:
            # セクターボラティリティに基づく時間選択
            if sector_info['volatility'] == 'high':
                buy_hour = 9 if confidence > 0.8 else 14  # 高ボラ銘柄は寄り付きまたは後場開始
            elif sector_info['volatility'] == 'low':
                buy_hour = 10  # 低ボラ銘柄は相場が落ち着いた午前中
            else:
                buy_hour = 11  # 中ボラ銘柄は前場後半
        
        # 売却タイミング予測（時間詳細・決定論的）
        strategy_days = {'デイトレ': 1, 'スイング': 5, '中期': 30}
        base_days = strategy_days.get(strategy, 5)
        
        # 戦略別の売却時間帯（決定論的）
        if strategy == 'デイトレ':
            sell_hour = 14 if confidence > 0.75 else 15  # 高信頼度なら早めに利確
            sell_timing_desc = "当日利確"
        elif strategy == 'スイング':
            sell_hour = 10 if sector_info['volatility'] == 'high' else 14  # ボラに応じて調整
            sell_timing_desc = "スイング利確"
        else:  # 中期
            sell_hour = 9 if confidence > 0.8 else 11  # 高信頼度なら寄り付きで利確
            sell_timing_desc = "中期利確"
        
        # 信頼度とセクターボラティリティで日数調整
        if confidence > 0.8 and sector_info['volatility'] == 'high':
            actual_sell_days = base_days
        elif confidence < 0.7:
            actual_sell_days = max(1, base_days // 2)
        else:
            actual_sell_days = base_days
        
        # 具体的な購入日時
        buy_datetime = now + timedelta(days=buy_days, hours=buy_hour-now.hour, minutes=-now.minute, seconds=-now.second)
        
        # 平日に調整（土日を避ける）
        while buy_datetime.weekday() >= 5:  # 5=土曜日, 6=日曜日
            buy_datetime += timedelta(days=1)
        
        # 具体的な売却日時
        sell_datetime = buy_datetime + timedelta(days=actual_sell_days)
        sell_datetime = sell_datetime.replace(hour=sell_hour, minute=0, second=0)
        
        # 平日に調整
        while sell_datetime.weekday() >= 5:
            sell_datetime += timedelta(days=1)
        
        # 目標価格計算
        target_buy_price = current_price * (0.98 if change_pct > 0 else 1.02)
        target_sell_price = current_price * (1.05 if confidence > 0.8 else 1.03)
        
        # フォーマットされたタイミング文字列生成
        if buy_days == 0:
            buy_timing = f"今日 {buy_datetime.strftime('%H:%M')} ({buy_timing_desc})\n¥{target_buy_price:,.0f}付近"
        else:
            buy_timing = f"{buy_datetime.strftime('%m/%d %H:%M')} ({buy_timing_desc})\n¥{target_buy_price:,.0f}付近"
        
        sell_timing = f"{sell_datetime.strftime('%m/%d %H:%M')} ({sell_timing_desc})\n¥{target_sell_price:,.0f}目標"
        
        return buy_timing, sell_timing

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