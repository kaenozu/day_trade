#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Web Dashboard - 拡張ウェブダッシュボード

Issue #871対応：リアルタイム・分析・予測・モニタリング・カスタマイズ機能拡張
包括的なウェブダッシュボードシステムの実装

主要機能：
1. リアルタイムデータ更新
2. 高度な分析ビジュアライゼーション
3. ML予測の可視化
4. パフォーマンス監視
5. カスタマイズ可能なダッシュボード
6. アラート・通知システム
7. データエクスポート機能
8. ユーザー設定管理
"""

import asyncio
import json
import logging
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import pandas as pd
import numpy as np

# Windows環境での文字化け対策
import sys
import os
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
    from flask import Flask, render_template, jsonify, request, send_from_directory, session
    from flask_socketio import SocketIO, emit, join_room, leave_room
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.utils
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

# 既存システムとの統合
try:
    from prediction_accuracy_enhancer import PredictionAccuracyEnhancer
    PREDICTION_ENHANCER_AVAILABLE = True
except ImportError:
    PREDICTION_ENHANCER_AVAILABLE = False

try:
    from real_data_provider_v2 import MultiSourceDataProvider
    DATA_PROVIDER_AVAILABLE = True
except ImportError:
    DATA_PROVIDER_AVAILABLE = False

try:
    from model_performance_monitor import EnhancedModelPerformanceMonitor
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False

try:
    from src.day_trade.data.symbol_selector import DynamicSymbolSelector
    SYMBOL_SELECTOR_AVAILABLE = True
except ImportError:
    SYMBOL_SELECTOR_AVAILABLE = False


class DashboardTheme(Enum):
    """ダッシュボードテーマ"""
    LIGHT = "light"
    DARK = "dark"
    FINANCIAL = "financial"
    CUSTOM = "custom"


class ChartType(Enum):
    """チャートタイプ"""
    CANDLESTICK = "candlestick"
    LINE = "line"
    AREA = "area"
    OHLC = "ohlc"
    VOLUME = "volume"
    INDICATORS = "indicators"


class UpdateFrequency(Enum):
    """更新頻度"""
    REAL_TIME = "real_time"  # 1秒
    HIGH = "high"           # 5秒
    MEDIUM = "medium"       # 30秒
    LOW = "low"             # 5分
    MANUAL = "manual"       # 手動


@dataclass
class DashboardConfig:
    """ダッシュボード設定"""
    # 基本設定
    theme: DashboardTheme = DashboardTheme.FINANCIAL
    update_frequency: UpdateFrequency = UpdateFrequency.MEDIUM
    auto_refresh: bool = True

    # 表示設定
    default_symbols: List[str] = field(default_factory=lambda: ["7203", "8306", "9984"])
    charts_per_row: int = 2
    show_volume: bool = True
    show_indicators: bool = True

    # データ設定
    data_retention_days: int = 30
    cache_duration_minutes: int = 5

    # 通知設定
    alerts_enabled: bool = True
    email_notifications: bool = False
    sound_alerts: bool = True

    # 高度な設定
    ml_predictions_enabled: bool = True
    performance_monitoring_enabled: bool = True
    custom_indicators: List[str] = field(default_factory=list)


@dataclass
class AlertConfig:
    """アラート設定"""
    alert_id: str
    symbol: str
    alert_type: str  # price_threshold, volume_spike, prediction_change, etc.
    condition: str   # >, <, ==, change_percent, etc.
    threshold: float
    enabled: bool = True
    last_triggered: Optional[datetime] = None


@dataclass
class UserPreferences:
    """ユーザー設定"""
    user_id: str
    dashboard_config: DashboardConfig
    custom_layouts: Dict[str, Any] = field(default_factory=dict)
    watchlist: List[str] = field(default_factory=list)
    alerts: List[AlertConfig] = field(default_factory=list)
    saved_analyses: List[str] = field(default_factory=list)


class RealTimeDataManager:
    """リアルタイムデータ管理"""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_subscriptions = set()
        self.data_cache = {}
        self.last_update = {}

        # データプロバイダー初期化
        if DATA_PROVIDER_AVAILABLE:
            self.data_provider = MultiSourceDataProvider()
        else:
            self.data_provider = None

    async def subscribe_symbol(self, symbol: str, socketio: SocketIO):
        """銘柄データの購読開始"""
        self.active_subscriptions.add(symbol)
        self.logger.info(f"リアルタイム購読開始: {symbol}")

        # 初回データ送信
        await self._send_initial_data(symbol, socketio)

    async def unsubscribe_symbol(self, symbol: str):
        """銘柄データの購読停止"""
        self.active_subscriptions.discard(symbol)
        self.logger.info(f"リアルタイム購読停止: {symbol}")

    async def _send_initial_data(self, symbol: str, socketio: SocketIO):
        """初回データの送信"""
        try:
            if self.data_provider:
                data = await self.data_provider.get_stock_data(symbol, "1d")
                if data is not None and not data.empty:
                    latest_data = {
                        'symbol': symbol,
                        'price': float(data['Close'].iloc[-1]),
                        'change': float(data['Close'].iloc[-1] - data['Close'].iloc[-2]) if len(data) > 1 else 0.0,
                        'volume': int(data['Volume'].iloc[-1]),
                        'timestamp': datetime.now().isoformat()
                    }
                    socketio.emit('price_update', latest_data)

        except Exception as e:
            self.logger.error(f"初回データ送信エラー {symbol}: {e}")

    async def update_data_loop(self, socketio: SocketIO):
        """データ更新ループ"""
        while True:
            try:
                for symbol in self.active_subscriptions.copy():
                    await self._update_symbol_data(symbol, socketio)

                # 更新頻度に応じた待機
                interval = self._get_update_interval()
                await asyncio.sleep(interval)

            except Exception as e:
                self.logger.error(f"データ更新ループエラー: {e}")
                await asyncio.sleep(5)

    async def _update_symbol_data(self, symbol: str, socketio: SocketIO):
        """個別銘柄データの更新"""
        try:
            # レート制限チェック
            last_update = self.last_update.get(symbol, datetime.min)
            min_interval = timedelta(seconds=self._get_update_interval())

            if datetime.now() - last_update < min_interval:
                return

            if self.data_provider:
                data = await self.data_provider.get_stock_data(symbol, "1d")
                if data is not None and not data.empty:
                    latest_data = {
                        'symbol': symbol,
                        'price': float(data['Close'].iloc[-1]),
                        'change': float(data['Close'].iloc[-1] - data['Close'].iloc[-2]) if len(data) > 1 else 0.0,
                        'change_percent': float(((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100)) if len(data) > 1 else 0.0,
                        'volume': int(data['Volume'].iloc[-1]),
                        'high': float(data['High'].iloc[-1]),
                        'low': float(data['Low'].iloc[-1]),
                        'timestamp': datetime.now().isoformat()
                    }

                    # キャッシュ更新
                    self.data_cache[symbol] = latest_data
                    self.last_update[symbol] = datetime.now()

                    # クライアントに送信
                    socketio.emit('price_update', latest_data)

        except Exception as e:
            self.logger.error(f"銘柄データ更新エラー {symbol}: {e}")

    def _get_update_interval(self) -> float:
        """更新間隔の取得"""
        intervals = {
            UpdateFrequency.REAL_TIME: 1.0,
            UpdateFrequency.HIGH: 5.0,
            UpdateFrequency.MEDIUM: 30.0,
            UpdateFrequency.LOW: 300.0,
            UpdateFrequency.MANUAL: 3600.0
        }
        return intervals.get(self.config.update_frequency, 30.0)


class AdvancedVisualization:
    """高度なビジュアライゼーション"""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_enhanced_candlestick_chart(self, data: pd.DataFrame, symbol: str,
                                        indicators: List[str] = None) -> Dict[str, Any]:
        """拡張ローソク足チャート作成"""
        try:
            # サブプロットの作成
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{symbol} Price Chart', 'Volume', 'Indicators'),
                row_width=[0.6, 0.2, 0.2]
            )

            # ローソク足チャート
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=symbol,
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff0000'
                ),
                row=1, col=1
            )

            # 移動平均線の追加
            if 'SMA_20' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )

            if 'SMA_50' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )

            # ボリンジャーバンド
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['BB_Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill=None
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['BB_Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)'
                    ),
                    row=1, col=1
                )

            # 出来高チャート
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='rgba(0,150,255,0.6)'
                ),
                row=2, col=1
            )

            # 技術指標（RSI）
            if 'RSI' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ),
                    row=3, col=1
                )

                # RSIの70/30ライン
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            # レイアウト設定
            fig.update_layout(
                title=f'{symbol} - Enhanced Technical Analysis',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark' if self.config.theme == DashboardTheme.DARK else 'plotly_white',
                showlegend=True,
                height=800
            )

            fig.update_xaxes(rangeslider_visible=False)

            return {
                'chart': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
                'success': True
            }

        except Exception as e:
            self.logger.error(f"チャート作成エラー: {e}")
            return {'success': False, 'error': str(e)}

    def create_prediction_chart(self, historical_data: pd.DataFrame,
                              predictions: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """予測チャート作成"""
        try:
            fig = go.Figure()

            # 過去データ
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Close'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                )
            )

            # 予測データ
            if 'predictions' in predictions and predictions['predictions']:
                pred_dates = pd.date_range(
                    start=historical_data.index[-1],
                    periods=len(predictions['predictions']),
                    freq='D'
                )

                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=predictions['predictions'],
                        mode='lines+markers',
                        name='Prediction',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=6)
                    )
                )

                # 信頼区間
                if 'confidence_intervals' in predictions:
                    upper_bound = predictions['confidence_intervals']['upper']
                    lower_bound = predictions['confidence_intervals']['lower']

                    fig.add_trace(
                        go.Scatter(
                            x=pred_dates,
                            y=upper_bound,
                            mode='lines',
                            name='Upper Bound',
                            line=dict(color='rgba(255,0,0,0.3)', width=1),
                            fill=None
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=pred_dates,
                            y=lower_bound,
                            mode='lines',
                            name='Lower Bound',
                            line=dict(color='rgba(255,0,0,0.3)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(255,0,0,0.1)'
                        )
                    )

            fig.update_layout(
                title=f'{symbol} - Price Prediction',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark' if self.config.theme == DashboardTheme.DARK else 'plotly_white',
                showlegend=True,
                height=500
            )

            return {
                'chart': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
                'success': True
            }

        except Exception as e:
            self.logger.error(f"予測チャート作成エラー: {e}")
            return {'success': False, 'error': str(e)}

    def create_performance_dashboard(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """パフォーマンスダッシュボード作成"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Model Accuracy Trends',
                    'Prediction Confidence',
                    'Feature Importance',
                    'Data Quality Metrics'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )

            # モデル精度トレンド
            if 'accuracy_history' in performance_data:
                accuracy_data = performance_data['accuracy_history']
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(accuracy_data))),
                        y=accuracy_data,
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='green', width=3)
                    ),
                    row=1, col=1
                )

            # 予測信頼度
            if 'confidence_distribution' in performance_data:
                conf_data = performance_data['confidence_distribution']
                fig.add_trace(
                    go.Histogram(
                        x=conf_data,
                        name='Confidence',
                        marker_color='blue',
                        opacity=0.7
                    ),
                    row=1, col=2
                )

            # 特徴量重要度
            if 'feature_importance' in performance_data:
                features = performance_data['feature_importance']
                fig.add_trace(
                    go.Bar(
                        x=list(features.values()),
                        y=list(features.keys()),
                        orientation='h',
                        name='Importance',
                        marker_color='orange'
                    ),
                    row=2, col=1
                )

            # データ品質メトリクス
            if 'data_quality' in performance_data:
                quality_metrics = performance_data['data_quality']
                fig.add_trace(
                    go.Bar(
                        x=list(quality_metrics.keys()),
                        y=list(quality_metrics.values()),
                        name='Quality Score',
                        marker_color='purple'
                    ),
                    row=2, col=2
                )

            fig.update_layout(
                title='System Performance Dashboard',
                template='plotly_dark' if self.config.theme == DashboardTheme.DARK else 'plotly_white',
                showlegend=True,
                height=700
            )

            return {
                'chart': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
                'success': True
            }

        except Exception as e:
            self.logger.error(f"パフォーマンスダッシュボード作成エラー: {e}")
            return {'success': False, 'error': str(e)}


class AlertManager:
    """アラート管理システム"""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_alerts = {}
        self.alert_history = []

    async def check_alerts(self, symbol: str, current_data: Dict[str, Any],
                          user_alerts: List[AlertConfig]) -> List[Dict[str, Any]]:
        """アラートチェック"""
        triggered_alerts = []

        try:
            for alert in user_alerts:
                if not alert.enabled or alert.symbol != symbol:
                    continue

                # クールダウンチェック
                if alert.last_triggered:
                    cooldown = timedelta(minutes=5)  # 5分間のクールダウン
                    if datetime.now() - alert.last_triggered < cooldown:
                        continue

                # アラート条件チェック
                if await self._check_alert_condition(alert, current_data):
                    alert_notification = {
                        'alert_id': alert.alert_id,
                        'symbol': symbol,
                        'type': alert.alert_type,
                        'message': self._generate_alert_message(alert, current_data),
                        'timestamp': datetime.now().isoformat(),
                        'severity': self._get_alert_severity(alert, current_data)
                    }

                    triggered_alerts.append(alert_notification)
                    alert.last_triggered = datetime.now()

                    # 履歴に追加
                    self.alert_history.append(alert_notification)

        except Exception as e:
            self.logger.error(f"アラートチェックエラー: {e}")

        return triggered_alerts

    async def _check_alert_condition(self, alert: AlertConfig, data: Dict[str, Any]) -> bool:
        """アラート条件の確認"""
        try:
            if alert.alert_type == 'price_threshold':
                current_price = data.get('price', 0)
                if alert.condition == '>':
                    return current_price > alert.threshold
                elif alert.condition == '<':
                    return current_price < alert.threshold

            elif alert.alert_type == 'price_change_percent':
                change_percent = data.get('change_percent', 0)
                if alert.condition == '>':
                    return abs(change_percent) > alert.threshold

            elif alert.alert_type == 'volume_spike':
                volume = data.get('volume', 0)
                # 通常の出来高と比較（簡略化）
                return volume > alert.threshold

            return False

        except Exception as e:
            self.logger.error(f"アラート条件チェックエラー: {e}")
            return False

    def _generate_alert_message(self, alert: AlertConfig, data: Dict[str, Any]) -> str:
        """アラートメッセージ生成"""
        try:
            if alert.alert_type == 'price_threshold':
                return f"{alert.symbol} 価格が{alert.threshold}円を{alert.condition}ました (現在: {data.get('price', 0):.2f}円)"
            elif alert.alert_type == 'price_change_percent':
                return f"{alert.symbol} 価格変動が{alert.threshold}%を超えました (変動: {data.get('change_percent', 0):.2f}%)"
            elif alert.alert_type == 'volume_spike':
                return f"{alert.symbol} 出来高急増を検知しました (出来高: {data.get('volume', 0):,})"
            else:
                return f"{alert.symbol} アラートが発生しました"

        except Exception as e:
            self.logger.error(f"アラートメッセージ生成エラー: {e}")
            return f"{alert.symbol} アラート"

    def _get_alert_severity(self, alert: AlertConfig, data: Dict[str, Any]) -> str:
        """アラート重要度の判定"""
        try:
            if alert.alert_type == 'price_change_percent':
                change = abs(data.get('change_percent', 0))
                if change > 10:
                    return 'critical'
                elif change > 5:
                    return 'warning'
                else:
                    return 'info'
            else:
                return 'info'

        except Exception as e:
            self.logger.error(f"アラート重要度判定エラー: {e}")
            return 'info'


class EnhancedWebDashboard:
    """拡張ウェブダッシュボードメインクラス"""

    def __init__(self, config_path: Optional[Path] = None, port: int = 8080):
        if not WEB_AVAILABLE:
            raise ImportError("Web機能の依存関係が不足しています")

        self.logger = logging.getLogger(__name__)
        self.port = port

        # 設定読み込み
        self.config = self._load_configuration(config_path)

        # Flask・SocketIOアプリ初期化
        self.app = Flask(__name__)
        
        # セキュアなsecret key設定
        secret_key = os.environ.get('ENHANCED_DASHBOARD_SECRET_KEY')
        if not secret_key:
            import secrets
            secret_key = secrets.token_urlsafe(32)
            self.logger.warning(f"⚠️  本番環境では環境変数ENHANCED_DASHBOARD_SECRET_KEYを設定してください")
            self.logger.warning(f"    例: export ENHANCED_DASHBOARD_SECRET_KEY='[32文字以上のランダム文字列]'")
        
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

        # ルート設定
        self._setup_routes()
        self._setup_socketio_events()

        # ユーザー設定
        self.user_preferences = {}

        self.logger.info("Enhanced Web Dashboard initialized")

    def _load_configuration(self, config_path: Optional[Path]) -> DashboardConfig:
        """設定の読み込み"""
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                return DashboardConfig(**config_data)
            except Exception as e:
                self.logger.warning(f"設定読み込みエラー: {e}. デフォルト設定を使用")

        return DashboardConfig()

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

    def _setup_routes(self):
        """APIルートの設定"""

        @self.app.route('/')
        def dashboard():
            """メインダッシュボード"""
            return self._render_enhanced_dashboard()

        @self.app.route('/api/symbols')
        def api_symbols():
            """銘柄一覧API"""
            return jsonify(self._get_available_symbols())

        @self.app.route('/api/chart/<symbol>')
        def api_chart(symbol):
            """高度なチャートAPI"""
            chart_type = request.args.get('type', 'candlestick')
            indicators = request.args.getlist('indicators')
            return jsonify(self._get_enhanced_chart_data(symbol, chart_type, indicators))

        @self.app.route('/api/prediction/<symbol>')
        def api_prediction(symbol):
            """予測データAPI"""
            return jsonify(self._get_prediction_data(symbol))

        @self.app.route('/api/performance')
        def api_performance():
            """システム性能API"""
            return jsonify(self._get_system_performance())

        @self.app.route('/api/alerts/<user_id>')
        def api_alerts(user_id):
            """ユーザーアラートAPI"""
            return jsonify(self._get_user_alerts(user_id))

        @self.app.route('/api/alerts/<user_id>', methods=['POST'])
        def api_create_alert(user_id):
            """アラート作成API"""
            alert_data = request.json
            return jsonify(self._create_user_alert(user_id, alert_data))

        @self.app.route('/api/preferences/<user_id>')
        def api_preferences(user_id):
            """ユーザー設定API"""
            return jsonify(self._get_user_preferences(user_id))

        @self.app.route('/api/preferences/<user_id>', methods=['POST'])
        def api_save_preferences(user_id):
            """ユーザー設定保存API"""
            preferences = request.json
            return jsonify(self._save_user_preferences(user_id, preferences))

        @self.app.route('/api/export/<format>')
        def api_export(format):
            """データエクスポートAPI"""
            symbols = request.args.getlist('symbols')
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            return self._export_data(symbols, start_date, end_date, format)

    def _setup_socketio_events(self):
        """SocketIOイベントの設定"""

        @self.socketio.on('connect')
        def handle_connect():
            """クライアント接続"""
            self.logger.info(f"クライアント接続: {request.sid}")
            emit('connected', {'status': 'success'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """クライアント切断"""
            self.logger.info(f"クライアント切断: {request.sid}")

        @self.socketio.on('subscribe')
        async def handle_subscribe(data):
            """リアルタイムデータ購読"""
            symbol = data.get('symbol')
            if symbol:
                await self.real_time_manager.subscribe_symbol(symbol, self.socketio)
                join_room(f'symbol_{symbol}')
                emit('subscribed', {'symbol': symbol, 'status': 'success'})

        @self.socketio.on('unsubscribe')
        async def handle_unsubscribe(data):
            """リアルタイムデータ購読停止"""
            symbol = data.get('symbol')
            if symbol:
                await self.real_time_manager.unsubscribe_symbol(symbol)
                leave_room(f'symbol_{symbol}')
                emit('unsubscribed', {'symbol': symbol, 'status': 'success'})

        @self.socketio.on('request_analysis')
        async def handle_request_analysis(data):
            """分析リクエスト"""
            symbol = data.get('symbol')
            analysis_type = data.get('type', 'technical')

            if symbol:
                analysis_result = await self._perform_analysis(symbol, analysis_type)
                emit('analysis_result', {
                    'symbol': symbol,
                    'type': analysis_type,
                    'result': analysis_result
                })

    def _render_enhanced_dashboard(self) -> str:
        """拡張ダッシュボードのレンダリング"""
        # HTMLテンプレートは別ファイルとして作成
        return """
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enhanced Trading Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .dashboard-card { margin-bottom: 20px; }
                .price-up { color: #28a745; }
                .price-down { color: #dc3545; }
                .alert-critical { border-left: 4px solid #dc3545; }
                .alert-warning { border-left: 4px solid #ffc107; }
                .alert-info { border-left: 4px solid #17a2b8; }
            </style>
        </head>
        <body>
            <div class="container-fluid">
                <h1 class="mt-3 mb-4">Enhanced Trading Dashboard</h1>

                <!-- リアルタイム価格表示 -->
                <div class="row">
                    <div class="col-12">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5>Real-time Prices</h5>
                            </div>
                            <div class="card-body" id="realtime-prices">
                                <!-- リアルタイム価格がここに表示される -->
                            </div>
                        </div>
                    </div>
                </div>

                <!-- チャート表示エリア -->
                <div class="row">
                    <div class="col-lg-8">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5>Price Chart</h5>
                                <select id="symbol-select" class="form-select" style="width: 200px; display: inline-block;">
                                    <option value="7203">7203 - トヨタ</option>
                                    <option value="8306">8306 - 三菱UFJ</option>
                                    <option value="9984">9984 - ソフトバンク</option>
                                </select>
                            </div>
                            <div class="card-body">
                                <div id="main-chart"></div>
                            </div>
                        </div>
                    </div>

                    <div class="col-lg-4">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5>Alerts</h5>
                            </div>
                            <div class="card-body" id="alerts-panel">
                                <!-- アラートがここに表示される -->
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 予測とパフォーマンス -->
                <div class="row">
                    <div class="col-lg-6">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5>ML Predictions</h5>
                            </div>
                            <div class="card-body">
                                <div id="prediction-chart"></div>
                            </div>
                        </div>
                    </div>

                    <div class="col-lg-6">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5>System Performance</h5>
                            </div>
                            <div class="card-body">
                                <div id="performance-chart"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                // Socket.IO接続
                const socket = io();

                // リアルタイム価格更新
                socket.on('price_update', function(data) {
                    updateRealtimePrice(data);
                });

                // アラート表示
                socket.on('alert_triggered', function(alert) {
                    showAlert(alert);
                });

                function updateRealtimePrice(data) {
                    const priceElement = document.getElementById('price-' + data.symbol);
                    if (priceElement) {
                        priceElement.innerHTML = `
                            <span class="${data.change >= 0 ? 'price-up' : 'price-down'}">
                                ¥${data.price.toFixed(2)}
                                (${data.change >= 0 ? '+' : ''}${data.change.toFixed(2)})
                            </span>
                        `;
                    }
                }

                function showAlert(alert) {
                    const alertsPanel = document.getElementById('alerts-panel');
                    const alertDiv = document.createElement('div');
                    alertDiv.className = `alert alert-${alert.severity} alert-dismissible`;
                    alertDiv.innerHTML = `
                        <strong>${alert.symbol}</strong> ${alert.message}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    `;
                    alertsPanel.appendChild(alertDiv);
                }

                // 初期化
                document.addEventListener('DOMContentLoaded', function() {
                    // デフォルト銘柄を購読
                    socket.emit('subscribe', {symbol: '7203'});
                    socket.emit('subscribe', {symbol: '8306'});
                    socket.emit('subscribe', {symbol: '9984'});

                    // チャート読み込み
                    loadChart('7203');
                });

                function loadChart(symbol) {
                    fetch(`/api/chart/${symbol}?type=candlestick&indicators=RSI,MACD`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                Plotly.newPlot('main-chart', data.chart);
                            }
                        });
                }
            </script>
        </body>
        </html>
        """

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


# ファクトリー関数
def create_enhanced_web_dashboard(config_path: Optional[str] = None,
                                port: int = 8080) -> EnhancedWebDashboard:
    """
    EnhancedWebDashboardインスタンスの作成

    Args:
        config_path: 設定ファイルパス
        port: ポート番号

    Returns:
        EnhancedWebDashboardインスタンス
    """
    config_path_obj = Path(config_path) if config_path else None
    return EnhancedWebDashboard(config_path=config_path_obj, port=port)


if __name__ == "__main__":
    # 基本動作確認
    if WEB_AVAILABLE:
        dashboard = EnhancedWebDashboard()
        dashboard.run(debug=True)
    else:
        print("Web機能の依存関係が不足しています")