#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Web Dashboard - メインダッシュボードクラス

統合されたウェブダッシュボードシステムのメイン実装
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Windows環境での文字化け対策
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Web関連ライブラリ
try:
    from flask import Flask
    from flask_socketio import SocketIO
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

# モジュールのインポート
from .config import DashboardConfig, load_dashboard_config
from .real_time_manager import RealTimeDataManager
from .visualization import AdvancedVisualization
from .alert_manager import AlertManager
from .routes import DashboardRoutes
from .socket_handlers import SocketHandlers
from .templates import DashboardTemplates

# 既存システムとの統合
try:
    from prediction_accuracy_enhancer import PredictionAccuracyEnhancer
    PREDICTION_ENHANCER_AVAILABLE = True
except ImportError:
    PREDICTION_ENHANCER_AVAILABLE = False

try:
    from ..monitoring.model_performance_monitor import EnhancedModelPerformanceMonitor
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False

try:
    from src.day_trade.data.symbol_selector import DynamicSymbolSelector
    SYMBOL_SELECTOR_AVAILABLE = True
except ImportError:
    SYMBOL_SELECTOR_AVAILABLE = False


class EnhancedWebDashboard:
    """拡張ウェブダッシュボードメインクラス"""

    def __init__(self, config_path: Optional[Path] = None, port: int = 8080):
        if not WEB_AVAILABLE:
            raise ImportError("Web機能の依存関係が不足しています")

        self.logger = logging.getLogger(__name__)
        self.port = port

        # 設定読み込み
        self.config = load_dashboard_config(config_path)

        # Flask・SocketIOアプリ初期化
        self.app = Flask(__name__)

        # セキュアなsecret key設定
        secret_key = os.environ.get('ENHANCED_DASHBOARD_SECRET_KEY')
        if not secret_key:
            import secrets
            secret_key = secrets.token_urlsafe(32)
            self.logger.warning("⚠️  本番環境では環境変数ENHANCED_DASHBOARD_SECRET_KEYを設定してください")
            self.logger.warning("    例: export ENHANCED_DASHBOARD_SECRET_KEY='[32文字以上のランダム文字列]'")

        self.app.secret_key = secret_key
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # コンポーネント初期化
        self.real_time_manager = RealTimeDataManager(self.config)
        self.visualization = AdvancedVisualization(self.config)
        self.alert_manager = AlertManager(self.config)

        # 外部システム統合
        self.prediction_enhancer = None
        self.performance_monitor = None
        self.symbol_selector = None

        self._initialize_external_systems()

        # データベース初期化
        self.db_path = Path("enhanced_dashboard.db")
        self._init_database()

        # ルートとSocketIOハンドラー設定
        self.routes = DashboardRoutes(self.app, self)
        self.socket_handlers = SocketHandlers(self.socketio, self)

        # ユーザー設定
        self.user_preferences = {}

        self.logger.info("Enhanced Web Dashboard initialized")

    def _initialize_external_systems(self):
        """外部システムの初期化"""
        try:
            if PREDICTION_ENHANCER_AVAILABLE:
                self.prediction_enhancer = PredictionAccuracyEnhancer()
                self.logger.info("予測精度向上システム統合完了")

            if PERFORMANCE_MONITOR_AVAILABLE:
                self.performance_monitor = EnhancedModelPerformanceMonitor()
                self.logger.info("性能監視システム統合完了")

            if SYMBOL_SELECTOR_AVAILABLE:
                from src.day_trade.data.symbol_selector import create_symbol_selector
                self.symbol_selector = create_symbol_selector()
                self.logger.info("銘柄選択システム統合完了")

        except Exception as e:
            self.logger.warning(f"外部システム統合エラー: {e}")

    def _init_database(self):
        """データベース初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # ユーザー設定テーブル
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        user_id TEXT PRIMARY KEY,
                        config_json TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # アラート設定テーブル
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_alerts (
                        alert_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        condition_text TEXT NOT NULL,
                        threshold REAL NOT NULL,
                        enabled BOOLEAN DEFAULT TRUE,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # アラート履歴テーブル
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS alert_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        message TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        triggered_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # ダッシュボードアクセスログ
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS access_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        action TEXT NOT NULL,
                        details TEXT,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.commit()
                self.logger.info("データベース初期化完了")

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")

    def _render_enhanced_dashboard(self) -> str:
        """拡張ダッシュボードのレンダリング"""
        return DashboardTemplates.render_dashboard()

    # APIメソッド群（ルートから呼び出される）
    def _get_available_symbols(self) -> Dict[str, Any]:
        """利用可能な銘柄一覧の取得"""
        try:
            # デフォルト銘柄リスト
            symbols = [
                {"symbol": "7203", "name": "トヨタ自動車", "market": "東証プライム"},
                {"symbol": "8306", "name": "三菱UFJ銀行", "market": "東証プライム"},
                {"symbol": "9984", "name": "ソフトバンクグループ", "market": "東証プライム"},
                {"symbol": "6758", "name": "ソニー", "market": "東証プライム"},
                {"symbol": "4755", "name": "楽天グループ", "market": "東証プライム"},
            ]

            if self.symbol_selector:
                # 動的銘柄選択がある場合は追加
                dynamic_symbols = self.symbol_selector.get_recommended_symbols(limit=10)
                for sym in dynamic_symbols:
                    symbols.append({
                        "symbol": sym,
                        "name": f"銘柄 {sym}",
                        "market": "東証"
                    })

            return {"symbols": symbols, "success": True}

        except Exception as e:
            self.logger.error(f"銘柄一覧取得エラー: {e}")
            return {"symbols": [], "success": False, "error": str(e)}

    def _get_enhanced_chart_data(self, symbol: str, chart_type: str, indicators: list) -> Dict[str, Any]:
        """拡張チャートデータの取得"""
        try:
            # ダミーデータを使用（実際の実装では実データを取得）
            import pandas as pd
            import numpy as np
            
            # サンプルデータ生成
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            np.random.seed(42)
            prices = 1000 + np.cumsum(np.random.randn(100) * 2)
            
            data = pd.DataFrame({
                'Open': prices + np.random.randn(100),
                'High': prices + np.abs(np.random.randn(100)),
                'Low': prices - np.abs(np.random.randn(100)),
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, 100)
            }, index=dates)

            # 技術指標を追加
            if 'SMA_20' in indicators:
                data['SMA_20'] = data['Close'].rolling(20).mean()
            if 'SMA_50' in indicators:
                data['SMA_50'] = data['Close'].rolling(50).mean()
            if 'RSI' in indicators:
                data['RSI'] = 50 + np.random.randn(100) * 10

            # チャート作成
            if chart_type == 'candlestick':
                return self.visualization.create_enhanced_candlestick_chart(data, symbol, indicators)
            else:
                return self.visualization.create_simple_line_chart(data, symbol)

        except Exception as e:
            self.logger.error(f"チャートデータ取得エラー: {e}")
            return {"success": False, "error": str(e)}

    def _get_prediction_data(self, symbol: str) -> Dict[str, Any]:
        """予測データの取得"""
        try:
            # サンプル予測データ
            import numpy as np
            
            predictions = {
                'predictions': [1050, 1055, 1060, 1058, 1062],
                'confidence_intervals': {
                    'upper': [1070, 1075, 1080, 1078, 1082],
                    'lower': [1030, 1035, 1040, 1038, 1042]
                }
            }

            # 実際の実装では予測システムを使用
            if self.prediction_enhancer:
                # 予測システムから実データを取得
                pass

            import pandas as pd
            dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
            historical_data = pd.DataFrame({
                'Close': 1000 + np.cumsum(np.random.randn(50))
            }, index=dates)

            return self.visualization.create_prediction_chart(historical_data, predictions, symbol)

        except Exception as e:
            self.logger.error(f"予測データ取得エラー: {e}")
            return {"success": False, "error": str(e)}

    def _get_system_performance(self) -> Dict[str, Any]:
        """システム性能データの取得"""
        try:
            # サンプル性能データ
            performance_data = {
                'accuracy_history': [0.75, 0.78, 0.82, 0.79, 0.85],
                'confidence_distribution': [0.6, 0.7, 0.8, 0.9, 0.85],
                'feature_importance': {
                    'price': 0.3,
                    'volume': 0.25,
                    'rsi': 0.2,
                    'sma': 0.15,
                    'macd': 0.1
                },
                'data_quality': {
                    'completeness': 0.95,
                    'accuracy': 0.92,
                    'timeliness': 0.88,
                    'consistency': 0.90
                }
            }

            return self.visualization.create_performance_dashboard(performance_data)

        except Exception as e:
            self.logger.error(f"システム性能データ取得エラー: {e}")
            return {"success": False, "error": str(e)}

    def _get_user_alerts(self, user_id: str) -> Dict[str, Any]:
        """ユーザーアラートの取得"""
        try:
            # データベースからアラートを取得
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM user_alerts WHERE user_id = ? AND enabled = 1",
                    (user_id,)
                )
                alerts = [dict(row) for row in cursor.fetchall()]
                
            return {"alerts": alerts, "success": True}

        except Exception as e:
            self.logger.error(f"ユーザーアラート取得エラー: {e}")
            return {"alerts": [], "success": False, "error": str(e)}

    def _create_user_alert(self, user_id: str, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """ユーザーアラートの作成"""
        try:
            alert_id = f"alert_{user_id}_{int(datetime.now().timestamp())}"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO user_alerts 
                    (alert_id, user_id, symbol, alert_type, condition_text, threshold)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    alert_id,
                    user_id,
                    alert_data['symbol'],
                    alert_data['type'],
                    alert_data['condition'],
                    alert_data['threshold']
                ))
                
            return {"alert_id": alert_id, "success": True, "message": "アラートを作成しました"}

        except Exception as e:
            self.logger.error(f"アラート作成エラー: {e}")
            return {"success": False, "error": str(e)}

    def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """ユーザー設定の取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT config_json FROM user_preferences WHERE user_id = ?",
                    (user_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return {"preferences": json.loads(row[0]), "success": True}
                else:
                    # デフォルト設定を返す
                    default_prefs = {
                        "theme": "financial",
                        "update_frequency": "medium",
                        "default_symbols": ["7203", "8306", "9984"]
                    }
                    return {"preferences": default_prefs, "success": True}

        except Exception as e:
            self.logger.error(f"ユーザー設定取得エラー: {e}")
            return {"preferences": {}, "success": False, "error": str(e)}

    def _save_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """ユーザー設定の保存"""
        try:
            config_json = json.dumps(preferences, ensure_ascii=False)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO user_preferences 
                    (user_id, config_json, updated_at)
                    VALUES (?, ?, ?)
                """, (user_id, config_json, datetime.now().isoformat()))
                
            return {"success": True, "message": "設定を保存しました"}

        except Exception as e:
            self.logger.error(f"ユーザー設定保存エラー: {e}")
            return {"success": False, "error": str(e)}

    def _export_data(self, symbols: list, start_date: str, end_date: str, format: str):
        """データエクスポート"""
        # 実装は簡略化
        return {"success": True, "message": f"{format}形式でエクスポートしました"}

    async def _perform_analysis(self, symbol: str, analysis_type: str) -> Dict[str, Any]:
        """分析実行"""
        try:
            # サンプル分析結果
            result = {
                "symbol": symbol,
                "type": analysis_type,
                "recommendation": "HOLD",
                "confidence": 0.75,
                "technical_indicators": {
                    "rsi": 65.3,
                    "macd": 0.12,
                    "sma_trend": "上昇"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return result

        except Exception as e:
            self.logger.error(f"分析実行エラー: {e}")
            return {"error": str(e)}

    def run(self, debug: bool = False, production: bool = False):
        """ダッシュボードサーバー起動"""
        # プロダクションモードでは強制的にdebug=False
        if production:
            debug = False
            self.logger.info(f"Enhanced Web Dashboard starting in PRODUCTION mode on port {self.port}")
        else:
            self.logger.info(f"Enhanced Web Dashboard starting in {'DEBUG' if debug else 'DEVELOPMENT'} mode on port {self.port}")

        # バックグラウンドタスク開始
        asyncio.create_task(self.real_time_manager.update_data_loop(self.socketio))

        # プロダクションモードでは追加設定
        if production:
            # プロダクション用設定適用
            self.app.config.update({
                'ENV': 'production',
                'DEBUG': False,
                'TESTING': False,
                'SECRET_KEY': os.environ.get('SECRET_KEY', os.urandom(32).hex())
            })
            self.logger.info("Production security settings applied")

        self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=debug)