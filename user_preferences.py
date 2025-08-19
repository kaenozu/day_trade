#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Preferences - ユーザー設定・個人化システム
Issue #949対応: 個人設定 + 学習機能 + アダプティブUI
"""

import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque

# セキュリティ関連
import secrets
import base64


class PreferenceCategory(Enum):
    """設定カテゴリ"""
    DISPLAY = "DISPLAY"           # 表示設定
    TRADING = "TRADING"           # 取引設定  
    AI_ANALYSIS = "AI_ANALYSIS"   # AI分析設定
    NOTIFICATIONS = "NOTIFICATIONS" # 通知設定
    SECURITY = "SECURITY"         # セキュリティ設定
    ACCESSIBILITY = "ACCESSIBILITY" # アクセシビリティ
    PERFORMANCE = "PERFORMANCE"   # パフォーマンス設定


class ThemeMode(Enum):
    """テーマモード"""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"  # システム設定に従う


class LanguageCode(Enum):
    """言語コード"""
    JA = "ja"  # 日本語
    EN = "en"  # 英語


@dataclass
class DisplayPreferences:
    """表示設定"""
    theme_mode: ThemeMode = ThemeMode.LIGHT
    language: LanguageCode = LanguageCode.JA
    currency_format: str = "JPY"
    timezone: str = "Asia/Tokyo"
    chart_style: str = "candlestick"  # candlestick, line, bar
    default_time_range: str = "1d"    # 1h, 1d, 1w, 1m
    show_grid_lines: bool = True
    animation_enabled: bool = True
    compact_mode: bool = False
    font_size: str = "medium"  # small, medium, large
    color_blind_friendly: bool = False


@dataclass 
class TradingPreferences:
    """取引設定"""
    default_symbols: List[str] = None
    risk_tolerance: str = "medium"  # low, medium, high
    investment_style: str = "balanced"  # conservative, balanced, aggressive
    price_alerts_enabled: bool = True
    auto_analysis_enabled: bool = True
    preferred_sectors: List[str] = None
    blacklist_symbols: List[str] = None
    min_confidence_threshold: float = 0.7
    max_risk_per_trade: float = 0.05  # 5%


@dataclass
class AIAnalysisPreferences:
    """AI分析設定"""
    analysis_frequency: str = "realtime"  # realtime, hourly, daily
    preferred_indicators: List[str] = None
    enable_sentiment_analysis: bool = True
    enable_technical_analysis: bool = True
    enable_fundamental_analysis: bool = True
    ai_confidence_display: bool = True
    explanation_level: str = "detailed"  # basic, detailed, expert
    enable_ml_learning: bool = True


@dataclass
class NotificationPreferences:
    """通知設定"""
    email_notifications: bool = True
    push_notifications: bool = True
    sound_notifications: bool = True
    price_change_threshold: float = 0.05  # 5%変動で通知
    ai_alert_threshold: float = 0.8      # 80%信頼度で通知
    trading_hours_only: bool = True
    weekend_notifications: bool = False
    notification_frequency: str = "important"  # all, important, critical


@dataclass
class UserBehavior:
    """ユーザー行動データ"""
    most_viewed_symbols: Dict[str, int]
    preferred_analysis_types: Dict[str, int]
    interaction_patterns: Dict[str, float]
    session_durations: List[float]
    feature_usage_count: Dict[str, int]
    error_encounters: Dict[str, int]
    last_login: datetime
    login_frequency: float  # 1日あたりのログイン回数


@dataclass
class UserPreferences:
    """ユーザー設定統合"""
    user_id: str
    display: DisplayPreferences
    trading: TradingPreferences
    ai_analysis: AIAnalysisPreferences
    notifications: NotificationPreferences
    behavior: UserBehavior
    created_at: datetime
    updated_at: datetime
    version: str = "1.0"


class AdaptiveUIEngine:
    """アダプティブUI エンジン"""
    
    def __init__(self):
        self.learning_weights = {
            'view_frequency': 0.3,
            'interaction_time': 0.25,
            'success_rate': 0.2,
            'user_feedback': 0.15,
            'error_rate': 0.1
        }
    
    def analyze_user_patterns(self, behavior: UserBehavior) -> Dict[str, Any]:
        """ユーザーパターン分析"""
        patterns = {}
        
        # 最も使用される機能
        if behavior.feature_usage_count:
            top_features = sorted(
                behavior.feature_usage_count.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            patterns['top_features'] = top_features
        
        # 閲覧パターン
        if behavior.most_viewed_symbols:
            patterns['favorite_symbols'] = list(behavior.most_viewed_symbols.keys())[:10]
        
        # セッション時間パターン
        if behavior.session_durations:
            avg_session = sum(behavior.session_durations) / len(behavior.session_durations)
            patterns['avg_session_minutes'] = avg_session / 60
            patterns['session_type'] = 'long' if avg_session > 1800 else 'short'  # 30分
        
        # エラー率分析
        if behavior.error_encounters:
            total_errors = sum(behavior.error_encounters.values())
            total_interactions = sum(behavior.feature_usage_count.values()) if behavior.feature_usage_count else 1
            patterns['error_rate'] = total_errors / total_interactions
        
        return patterns
    
    def generate_ui_recommendations(self, preferences: UserPreferences) -> List[Dict[str, Any]]:
        """UI最適化推奨"""
        recommendations = []
        patterns = self.analyze_user_patterns(preferences.behavior)
        
        # セッション時間に基づく推奨
        if patterns.get('session_type') == 'short':
            recommendations.append({
                'type': 'layout',
                'suggestion': 'compact_dashboard',
                'reason': '短時間セッションに最適化',
                'confidence': 0.8
            })
        
        # よく使う機能のクイックアクセス
        if patterns.get('top_features'):
            recommendations.append({
                'type': 'quick_access',
                'suggestion': 'add_toolbar_shortcuts',
                'features': patterns['top_features'][:3],
                'reason': 'よく使用する機能への迅速なアクセス',
                'confidence': 0.9
            })
        
        # エラー率に基づく推奨
        if patterns.get('error_rate', 0) > 0.1:  # 10%以上
            recommendations.append({
                'type': 'assistance',
                'suggestion': 'enable_guided_mode',
                'reason': 'エラー率が高いため、ガイダンス機能を推奨',
                'confidence': 0.7
            })
        
        # お気に入り銘柄の自動表示
        if patterns.get('favorite_symbols'):
            recommendations.append({
                'type': 'content',
                'suggestion': 'auto_watchlist',
                'symbols': patterns['favorite_symbols'][:5],
                'reason': 'よく閲覧される銘柄の自動表示',
                'confidence': 0.85
            })
        
        return recommendations


class UserPreferenceManager:
    """ユーザー設定管理システム"""
    
    def __init__(self):
        self.preferences_dir = 'data/user_preferences'
        self.backup_dir = 'data/user_preferences/backups'
        self.adaptive_engine = AdaptiveUIEngine()
        
        os.makedirs(self.preferences_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # デフォルト設定
        self.default_preferences = self._create_default_preferences()
        
        # 学習データ
        self.interaction_log = deque(maxlen=1000)
        
    def _create_default_preferences(self) -> UserPreferences:
        """デフォルト設定作成"""
        return UserPreferences(
            user_id="default",
            display=DisplayPreferences(),
            trading=TradingPreferences(
                default_symbols=['7203', '8306', '9984', '6758', '4689'],
                preferred_sectors=['Technology', 'Finance', 'Automotive'],
                blacklist_symbols=[]
            ),
            ai_analysis=AIAnalysisPreferences(
                preferred_indicators=['RSI', 'MACD', 'Moving Average', 'Bollinger Bands']
            ),
            notifications=NotificationPreferences(),
            behavior=UserBehavior(
                most_viewed_symbols={},
                preferred_analysis_types={},
                interaction_patterns={},
                session_durations=[],
                feature_usage_count={},
                error_encounters={},
                last_login=datetime.now(),
                login_frequency=1.0
            ),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def get_user_preferences(self, user_id: str) -> UserPreferences:
        """ユーザー設定取得"""
        prefs_file = os.path.join(self.preferences_dir, f"{user_id}.json")
        
        if not os.path.exists(prefs_file):
            # デフォルト設定をコピーして新規作成
            preferences = self._create_default_preferences()
            preferences.user_id = user_id
            self.save_user_preferences(preferences)
            return preferences
        
        try:
            with open(prefs_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON から UserPreferences へ復元
            preferences = UserPreferences(
                user_id=data['user_id'],
                display=DisplayPreferences(**data['display']),
                trading=TradingPreferences(**data['trading']),
                ai_analysis=AIAnalysisPreferences(**data['ai_analysis']),
                notifications=NotificationPreferences(**data['notifications']),
                behavior=UserBehavior(
                    most_viewed_symbols=data['behavior']['most_viewed_symbols'],
                    preferred_analysis_types=data['behavior']['preferred_analysis_types'],
                    interaction_patterns=data['behavior']['interaction_patterns'],
                    session_durations=data['behavior']['session_durations'],
                    feature_usage_count=data['behavior']['feature_usage_count'],
                    error_encounters=data['behavior']['error_encounters'],
                    last_login=datetime.fromisoformat(data['behavior']['last_login']),
                    login_frequency=data['behavior']['login_frequency']
                ),
                created_at=datetime.fromisoformat(data['created_at']),
                updated_at=datetime.fromisoformat(data['updated_at']),
                version=data.get('version', '1.0')
            )
            
            return preferences
            
        except Exception as e:
            logging.error(f"Failed to load preferences for {user_id}: {e}")
            return self._create_default_preferences()
    
    def save_user_preferences(self, preferences: UserPreferences):
        """ユーザー設定保存"""
        preferences.updated_at = datetime.now()
        
        # バックアップ作成
        self._create_backup(preferences)
        
        # メイン設定ファイル保存
        prefs_file = os.path.join(self.preferences_dir, f"{preferences.user_id}.json")
        
        try:
            with open(prefs_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(preferences), f, default=str, ensure_ascii=False, indent=2)
            
            logging.info(f"Saved preferences for user {preferences.user_id}")
            
        except Exception as e:
            logging.error(f"Failed to save preferences for {preferences.user_id}: {e}")
    
    def _create_backup(self, preferences: UserPreferences):
        """設定バックアップ作成"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(
            self.backup_dir, 
            f"{preferences.user_id}_{timestamp}.json"
        )
        
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(preferences), f, default=str, ensure_ascii=False, indent=2)
            
            # 古いバックアップファイルをクリーンアップ（7日より古い）
            self._cleanup_old_backups(preferences.user_id)
            
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
    
    def _cleanup_old_backups(self, user_id: str):
        """古いバックアップクリーンアップ"""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        try:
            for filename in os.listdir(self.backup_dir):
                if not filename.startswith(user_id):
                    continue
                
                file_path = os.path.join(self.backup_dir, filename)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_time < cutoff_time:
                    os.remove(file_path)
                    
        except Exception as e:
            logging.error(f"Backup cleanup failed: {e}")
    
    def update_preference(self, user_id: str, category: PreferenceCategory, 
                         key: str, value: Any):
        """個別設定更新"""
        preferences = self.get_user_preferences(user_id)
        
        category_obj = {
            PreferenceCategory.DISPLAY: preferences.display,
            PreferenceCategory.TRADING: preferences.trading,
            PreferenceCategory.AI_ANALYSIS: preferences.ai_analysis,
            PreferenceCategory.NOTIFICATIONS: preferences.notifications,
        }.get(category)
        
        if category_obj and hasattr(category_obj, key):
            setattr(category_obj, key, value)
            self.save_user_preferences(preferences)
            logging.info(f"Updated {category.value}.{key} for user {user_id}")
        else:
            logging.warning(f"Invalid preference key: {category.value}.{key}")
    
    def record_user_interaction(self, user_id: str, feature: str, 
                              duration: float = None, success: bool = True):
        """ユーザー操作記録"""
        preferences = self.get_user_preferences(user_id)
        
        # 機能使用回数
        preferences.behavior.feature_usage_count[feature] = \
            preferences.behavior.feature_usage_count.get(feature, 0) + 1
        
        # 操作時間
        if duration:
            if feature not in preferences.behavior.interaction_patterns:
                preferences.behavior.interaction_patterns[feature] = []
            
            # 最近の操作時間のみ保持（最大10回）
            if len(preferences.behavior.interaction_patterns[feature]) >= 10:
                preferences.behavior.interaction_patterns[feature].pop(0)
            
            preferences.behavior.interaction_patterns[feature].append(duration)
        
        # エラー記録
        if not success:
            preferences.behavior.error_encounters[feature] = \
                preferences.behavior.error_encounters.get(feature, 0) + 1
        
        # ログ記録
        self.interaction_log.append({
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'feature': feature,
            'duration': duration,
            'success': success
        })
        
        self.save_user_preferences(preferences)
    
    def record_symbol_view(self, user_id: str, symbol: str):
        """銘柄閲覧記録"""
        preferences = self.get_user_preferences(user_id)
        
        preferences.behavior.most_viewed_symbols[symbol] = \
            preferences.behavior.most_viewed_symbols.get(symbol, 0) + 1
        
        self.save_user_preferences(preferences)
    
    def get_personalized_recommendations(self, user_id: str) -> Dict[str, Any]:
        """個人化推奨生成"""
        preferences = self.get_user_preferences(user_id)
        ui_recommendations = self.adaptive_engine.generate_ui_recommendations(preferences)
        
        # 銘柄推奨
        favorite_symbols = list(preferences.behavior.most_viewed_symbols.keys())[:5]
        
        # カスタマイズ推奨
        customization_suggestions = []
        
        # ダークモード推奨（夜間使用が多い場合）
        if self._is_night_user(preferences):
            customization_suggestions.append({
                'type': 'theme',
                'suggestion': 'dark_mode',
                'reason': '夜間使用が多いためダークモードを推奨'
            })
        
        return {
            'ui_recommendations': ui_recommendations,
            'favorite_symbols': favorite_symbols,
            'customization_suggestions': customization_suggestions,
            'last_updated': datetime.now().isoformat()
        }
    
    def _is_night_user(self, preferences: UserPreferences) -> bool:
        """夜間ユーザーかどうか判定"""
        # ログイン時間パターンから判定（簡略版）
        current_hour = datetime.now().hour
        return 20 <= current_hour or current_hour <= 6
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """ユーザーデータエクスポート"""
        preferences = self.get_user_preferences(user_id)
        
        return {
            'user_id': user_id,
            'preferences': asdict(preferences),
            'recommendations': self.get_personalized_recommendations(user_id),
            'interaction_log': list(self.interaction_log),
            'export_timestamp': datetime.now().isoformat()
        }
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """使用統計取得"""
        stats = {
            'total_users': len(os.listdir(self.preferences_dir)),
            'interaction_records': len(self.interaction_log),
            'most_used_features': {},
            'common_preferences': {}
        }
        
        # 最も使用される機能
        feature_counts = defaultdict(int)
        for record in self.interaction_log:
            feature_counts[record['feature']] += 1
        
        stats['most_used_features'] = dict(sorted(
            feature_counts.items(), key=lambda x: x[1], reverse=True
        )[:10])
        
        return stats


# グローバルインスタンス
user_preference_manager = UserPreferenceManager()


def get_user_preferences(user_id: str = "default") -> UserPreferences:
    """ユーザー設定取得"""
    return user_preference_manager.get_user_preferences(user_id)


def update_user_preference(user_id: str, category: PreferenceCategory, 
                          key: str, value: Any):
    """ユーザー設定更新"""
    user_preference_manager.update_preference(user_id, category, key, value)


def record_interaction(user_id: str, feature: str, duration: float = None, 
                      success: bool = True):
    """ユーザー操作記録"""
    user_preference_manager.record_user_interaction(user_id, feature, duration, success)


def get_personalized_recommendations(user_id: str = "default") -> Dict[str, Any]:
    """個人化推奨取得"""
    return user_preference_manager.get_personalized_recommendations(user_id)


if __name__ == "__main__":
    print("=== User Preferences System Test ===")
    
    # デフォルト設定取得
    prefs = get_user_preferences("test_user")
    print(f"Default theme: {prefs.display.theme_mode.value}")
    print(f"Default symbols: {prefs.trading.default_symbols}")
    
    # 設定更新テスト
    update_user_preference("test_user", PreferenceCategory.DISPLAY, "theme_mode", ThemeMode.DARK)
    print("Updated theme to dark mode")
    
    # 操作記録テスト
    record_interaction("test_user", "stock_analysis", duration=45.5, success=True)
    record_interaction("test_user", "ai_recommendations", duration=12.3, success=True)
    
    # 銘柄閲覧記録
    user_preference_manager.record_symbol_view("test_user", "7203")
    user_preference_manager.record_symbol_view("test_user", "7203")  # 重複閲覧
    user_preference_manager.record_symbol_view("test_user", "8306")
    
    # 個人化推奨
    recommendations = get_personalized_recommendations("test_user")
    print(f"UI recommendations: {len(recommendations['ui_recommendations'])}")
    print(f"Favorite symbols: {recommendations['favorite_symbols']}")
    
    # 使用統計
    stats = user_preference_manager.get_usage_statistics()
    print(f"Total users: {stats['total_users']}")
    print(f"Most used features: {list(stats['most_used_features'].keys())[:3]}")
    
    print("User preferences test completed!")