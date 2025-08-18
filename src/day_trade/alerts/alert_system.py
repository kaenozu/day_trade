#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alert System - リアルタイムアラート・通知システム

デイトレード機会の即座通知・リスク警告・パフォーマンス監視
実戦投入向けリアルタイム監視システム
"""

import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from collections import deque, defaultdict
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

try:
    from enhanced_symbol_manager import EnhancedSymbolManager
    ENHANCED_SYMBOLS_AVAILABLE = True
except ImportError:
    ENHANCED_SYMBOLS_AVAILABLE = False

try:
    from theme_stock_analyzer import ThemeStockAnalyzer
    THEME_STOCK_AVAILABLE = True
except ImportError:
    THEME_STOCK_AVAILABLE = False

try:
    from performance_tracker import PerformanceTracker
    PERFORMANCE_TRACKER_AVAILABLE = True
except ImportError:
    PERFORMANCE_TRACKER_AVAILABLE = False

class AlertType(Enum):
    """アラートタイプ"""
    BUY_SIGNAL = "買いシグナル"
    SELL_SIGNAL = "売りシグナル"
    RISK_WARNING = "リスク警告"
    PROFIT_TARGET = "利益目標達成"
    STOP_LOSS = "損切りライン"
    VOLUME_SPIKE = "出来高急増"
    PRICE_BREAKOUT = "価格ブレイクアウト"
    NEWS_ALERT = "ニュース関連"
    PERFORMANCE_UPDATE = "パフォーマンス更新"
    SYSTEM_STATUS = "システム状況"

class AlertPriority(Enum):
    """アラート優先度"""
    CRITICAL = "緊急"     # 即座対応が必要
    HIGH = "高"          # 30分以内対応
    MEDIUM = "中"        # 1時間以内対応
    LOW = "低"           # 日中対応
    INFO = "情報"        # 参考情報

class NotificationMethod(Enum):
    """通知方法"""
    CONSOLE = "コンソール表示"
    FILE = "ファイル出力"
    EMAIL = "メール通知"
    DESKTOP = "デスクトップ通知"
    SOUND = "音声アラート"

@dataclass
class Alert:
    """アラート情報"""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    symbol: Optional[str] = None
    title: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # タイミング
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    # 状態管理
    is_active: bool = True
    is_acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None

    # 通知
    notification_methods: List[NotificationMethod] = field(default_factory=list)
    sent_notifications: List[NotificationMethod] = field(default_factory=list)

    # アクション
    suggested_actions: List[str] = field(default_factory=list)
    risk_level: str = "中"
    confidence: float = 75.0

@dataclass
class AlertRule:
    """アラートルール"""
    rule_id: str
    rule_name: str
    alert_type: AlertType
    priority: AlertPriority

    # 条件
    symbol_filter: Optional[List[str]] = None    # 対象銘柄
    condition_func: Optional[Callable] = None    # 発火条件関数
    threshold_values: Dict[str, float] = field(default_factory=dict)  # 閾値

    # タイミング制御
    cooldown_minutes: int = 15      # 同じルールの発火間隔
    last_triggered: Optional[datetime] = None

    # 通知設定
    notification_methods: List[NotificationMethod] = field(default_factory=list)

    is_active: bool = True

class RealTimeAlertSystem:
    """
    リアルタイムアラート・通知システム
    デイトレード機会の即座通知・リスク管理
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データベース初期化
        self.data_dir = Path("alert_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "alerts.db"
        self._init_database()

        # アラート管理
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_rules: Dict[str, AlertRule] = {}

        # 通知キュー
        self.notification_queue: deque = deque(maxlen=100)
        self.notification_thread: Optional[threading.Thread] = None
        self.is_running = False

        # 外部システム統合
        if ENHANCED_SYMBOLS_AVAILABLE:
            self.symbol_manager = EnhancedSymbolManager()

        if THEME_STOCK_AVAILABLE:
            self.theme_analyzer = ThemeStockAnalyzer()

        if PERFORMANCE_TRACKER_AVAILABLE:
            self.performance_tracker = PerformanceTracker()

        # デフォルトルール初期化
        self._initialize_default_rules()

        # 通知設定
        self.notification_settings = {
            "console_enabled": True,
            "file_enabled": True,
            "email_enabled": False,
            "email_smtp_server": "smtp.gmail.com",
            "email_port": 587,
            "email_user": "",
            "email_password": "",
            "email_recipients": [],
            "sound_enabled": False
        }

        self.logger.info("Real-time alert system initialized")

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # アラートテーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    symbol TEXT,
                    title TEXT,
                    message TEXT,
                    details TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    is_active BOOLEAN,
                    is_acknowledged BOOLEAN,
                    acknowledged_at TEXT,
                    risk_level TEXT,
                    confidence REAL,
                    suggested_actions TEXT
                )
            """)

            # アラートルールテーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_rules (
                    rule_id TEXT PRIMARY KEY,
                    rule_name TEXT NOT NULL,
                    alert_type TEXT,
                    priority TEXT,
                    symbol_filter TEXT,
                    threshold_values TEXT,
                    cooldown_minutes INTEGER,
                    last_triggered TEXT,
                    notification_methods TEXT,
                    is_active BOOLEAN,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # インデックス作成
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alerts(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type)")

    def _initialize_default_rules(self):
        """デフォルトアラートルール初期化"""

        # 買いシグナルルール
        buy_signal_rule = AlertRule(
            rule_id="BUY_SIGNAL_HIGH_CONFIDENCE",
            rule_name="高信頼度買いシグナル",
            alert_type=AlertType.BUY_SIGNAL,
            priority=AlertPriority.HIGH,
            threshold_values={"confidence": 90.0, "score": 85.0},
            cooldown_minutes=30,
            notification_methods=[NotificationMethod.CONSOLE, NotificationMethod.FILE]
        )

        # リスク警告ルール
        risk_warning_rule = AlertRule(
            rule_id="RISK_WARNING_HIGH_VOLATILITY",
            rule_name="高ボラティリティリスク警告",
            alert_type=AlertType.RISK_WARNING,
            priority=AlertPriority.CRITICAL,
            threshold_values={"volatility": 15.0, "max_drawdown": 10.0},
            cooldown_minutes=15,
            notification_methods=[NotificationMethod.CONSOLE, NotificationMethod.FILE]
        )

        # パフォーマンス更新ルール
        performance_rule = AlertRule(
            rule_id="PERFORMANCE_UPDATE_DAILY",
            rule_name="日次パフォーマンス更新",
            alert_type=AlertType.PERFORMANCE_UPDATE,
            priority=AlertPriority.INFO,
            threshold_values={"return_change": 5.0},
            cooldown_minutes=480,  # 8時間
            notification_methods=[NotificationMethod.CONSOLE, NotificationMethod.FILE]
        )

        # 出来高急増ルール
        volume_spike_rule = AlertRule(
            rule_id="VOLUME_SPIKE_ALERT",
            rule_name="出来高急増アラート",
            alert_type=AlertType.VOLUME_SPIKE,
            priority=AlertPriority.MEDIUM,
            threshold_values={"volume_ratio": 2.0},
            cooldown_minutes=60,
            notification_methods=[NotificationMethod.CONSOLE, NotificationMethod.FILE]
        )

        self.alert_rules.update({
            buy_signal_rule.rule_id: buy_signal_rule,
            risk_warning_rule.rule_id: risk_warning_rule,
            performance_rule.rule_id: performance_rule,
            volume_spike_rule.rule_id: volume_spike_rule
        })

    async def start_monitoring(self):
        """監視開始"""
        if self.is_running:
            self.logger.warning("Alert system already running")
            return

        self.is_running = True

        # 通知処理スレッド開始
        self.notification_thread = threading.Thread(target=self._notification_worker, daemon=True)
        self.notification_thread.start()

        self.logger.info("Real-time alert monitoring started")

        # 初期システム状況アラート
        await self.create_alert(Alert(
            alert_id=f"SYSTEM_START_{int(time.time())}",
            alert_type=AlertType.SYSTEM_STATUS,
            priority=AlertPriority.INFO,
            title="アラートシステム開始",
            message="リアルタイムアラート・通知システムが開始されました",
            details={"active_rules": len(self.alert_rules)},
            notification_methods=[NotificationMethod.CONSOLE, NotificationMethod.FILE]
        ))

    async def stop_monitoring(self):
        """監視停止"""
        self.is_running = False

        if self.notification_thread and self.notification_thread.is_alive():
            self.notification_thread.join(timeout=5)

        self.logger.info("Real-time alert monitoring stopped")

    async def create_alert(self, alert: Alert) -> bool:
        """
        アラート作成・発行

        Args:
            alert: アラート情報

        Returns:
            作成成功フラグ
        """
        try:
            # データベース保存
            await self._save_alert_to_db(alert)

            # アクティブアラートに追加
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)

            # 通知キューに追加
            self.notification_queue.append(alert)

            self.logger.info(f"Alert created: {alert.title} ({alert.priority.value})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create alert: {e}")
            return False

    async def _save_alert_to_db(self, alert: Alert):
        """アラートをデータベース保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alerts (
                    alert_id, alert_type, priority, symbol, title, message,
                    details, created_at, expires_at, is_active, is_acknowledged,
                    acknowledged_at, risk_level, confidence, suggested_actions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.alert_type.value,
                alert.priority.value,
                alert.symbol,
                alert.title,
                alert.message,
                json.dumps(alert.details, ensure_ascii=False),
                alert.created_at.isoformat(),
                alert.expires_at.isoformat() if alert.expires_at else None,
                alert.is_active,
                alert.is_acknowledged,
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                alert.risk_level,
                alert.confidence,
                json.dumps(alert.suggested_actions, ensure_ascii=False)
            ))

    def _notification_worker(self):
        """通知処理ワーカースレッド"""
        while self.is_running:
            try:
                if self.notification_queue:
                    alert = self.notification_queue.popleft()
                    self._process_notifications(alert)
                else:
                    time.sleep(1)  # 1秒待機

            except Exception as e:
                self.logger.error(f"Notification worker error: {e}")
                time.sleep(5)

    def _process_notifications(self, alert: Alert):
        """通知処理実行"""
        for method in alert.notification_methods:
            try:
                if method == NotificationMethod.CONSOLE and self.notification_settings["console_enabled"]:
                    self._send_console_notification(alert)
                elif method == NotificationMethod.FILE and self.notification_settings["file_enabled"]:
                    self._send_file_notification(alert)
                elif method == NotificationMethod.EMAIL and self.notification_settings["email_enabled"]:
                    self._send_email_notification(alert)

                # 送信完了記録
                if method not in alert.sent_notifications:
                    alert.sent_notifications.append(method)

            except Exception as e:
                self.logger.error(f"Failed to send {method.value} notification: {e}")

    def _send_console_notification(self, alert: Alert):
        """コンソール通知送信"""
        priority_icon = {
            AlertPriority.CRITICAL: "🚨",
            AlertPriority.HIGH: "⚠️",
            AlertPriority.MEDIUM: "📊",
            AlertPriority.LOW: "ℹ️",
            AlertPriority.INFO: "💡"
        }.get(alert.priority, "📢")

        print(f"\n{priority_icon} 【{alert.priority.value}】{alert.title}")
        print(f"    {alert.message}")
        if alert.symbol:
            print(f"    対象銘柄: {alert.symbol}")
        if alert.suggested_actions:
            print(f"    推奨アクション: {', '.join(alert.suggested_actions)}")
        print(f"    時刻: {alert.created_at.strftime('%H:%M:%S')}")
        print()

    def _send_file_notification(self, alert: Alert):
        """ファイル通知送信"""
        log_file = self.data_dir / "alert_log.txt"

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}] ")
            f.write(f"{alert.priority.value} - {alert.title}\n")
            f.write(f"  メッセージ: {alert.message}\n")
            if alert.symbol:
                f.write(f"  対象銘柄: {alert.symbol}\n")
            if alert.suggested_actions:
                f.write(f"  推奨アクション: {', '.join(alert.suggested_actions)}\n")
            f.write(f"  信頼度: {alert.confidence:.1f}%\n")
            f.write("\n")

    def _send_email_notification(self, alert: Alert):
        """メール通知送信"""
        if not self.notification_settings["email_user"] or not self.notification_settings["email_recipients"]:
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.notification_settings["email_user"]
            msg['To'] = ", ".join(self.notification_settings["email_recipients"])
            msg['Subject'] = f"[デイトレードアラート] {alert.title}"

            body = f"""
デイトレードアラート通知

優先度: {alert.priority.value}
アラート種別: {alert.alert_type.value}
対象銘柄: {alert.symbol or 'N/A'}
信頼度: {alert.confidence:.1f}%

メッセージ:
{alert.message}

推奨アクション:
{chr(10).join(f'• {action}' for action in alert.suggested_actions)}

発生時刻: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}

※このメールは自動送信されています。
"""

            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            server = smtplib.SMTP(self.notification_settings["email_smtp_server"],
                                self.notification_settings["email_port"])
            server.starttls()
            server.login(self.notification_settings["email_user"],
                       self.notification_settings["email_password"])
            text = msg.as_string()
            server.sendmail(self.notification_settings["email_user"],
                          self.notification_settings["email_recipients"], text)
            server.quit()

        except Exception as e:
            self.logger.error(f"Email notification failed: {e}")

    async def check_buy_signals(self, analysis_results: List[Dict[str, Any]]) -> List[Alert]:
        """買いシグナルチェック"""
        alerts = []

        for result in analysis_results:
            try:
                symbol = result.get('symbol')
                score = result.get('score', 0)
                confidence = result.get('confidence', 0)
                action = result.get('action', '')

                # 高信頼度買いシグナルチェック
                rule = self.alert_rules.get("BUY_SIGNAL_HIGH_CONFIDENCE")
                if (rule and rule.is_active and
                    confidence >= rule.threshold_values.get("confidence", 90) and
                    score >= rule.threshold_values.get("score", 85) and
                    action in ['買い', '強い買い']):

                    # クールダウンチェック
                    if (not rule.last_triggered or
                        datetime.now() - rule.last_triggered > timedelta(minutes=rule.cooldown_minutes)):

                        alert = Alert(
                            alert_id=f"BUY_SIGNAL_{symbol}_{int(time.time())}",
                            alert_type=AlertType.BUY_SIGNAL,
                            priority=AlertPriority.HIGH,
                            symbol=symbol,
                            title=f"高信頼度買いシグナル: {symbol}",
                            message=f"{symbol} で高信頼度の買いシグナルが発生しました。信頼度{confidence:.1f}%、スコア{score:.1f}点",
                            details={
                                "score": score,
                                "confidence": confidence,
                                "action": action,
                                "rule_id": rule.rule_id
                            },
                            suggested_actions=["エントリー検討", "リスク管理設定", "利確目標設定"],
                            risk_level="中",
                            confidence=confidence,
                            notification_methods=rule.notification_methods
                        )

                        alerts.append(alert)
                        rule.last_triggered = datetime.now()

            except Exception as e:
                self.logger.error(f"Buy signal check error for {result}: {e}")

        return alerts

    async def check_risk_warnings(self, portfolio_data: Dict[str, Any]) -> List[Alert]:
        """リスク警告チェック"""
        alerts = []

        try:
            if not portfolio_data:
                return alerts

            max_drawdown = portfolio_data.get('max_drawdown', 0)
            volatility = portfolio_data.get('volatility', 0)
            win_rate = portfolio_data.get('win_rate', 100)

            # 高ボラティリティリスク警告
            rule = self.alert_rules.get("RISK_WARNING_HIGH_VOLATILITY")
            if (rule and rule.is_active and
                (max_drawdown >= rule.threshold_values.get("max_drawdown", 10) or
                 volatility >= rule.threshold_values.get("volatility", 15))):

                # クールダウンチェック
                if (not rule.last_triggered or
                    datetime.now() - rule.last_triggered > timedelta(minutes=rule.cooldown_minutes)):

                    alert = Alert(
                        alert_id=f"RISK_WARNING_{int(time.time())}",
                        alert_type=AlertType.RISK_WARNING,
                        priority=AlertPriority.CRITICAL,
                        title="リスク警告: 高ボラティリティ",
                        message=f"ポートフォリオで高リスク状況を検知。最大DD{max_drawdown:.1f}%、ボラティリティ{volatility:.1f}%",
                        details={
                            "max_drawdown": max_drawdown,
                            "volatility": volatility,
                            "win_rate": win_rate,
                            "rule_id": rule.rule_id
                        },
                        suggested_actions=["ポジションサイズ縮小", "損切りライン見直し", "分散投資強化"],
                        risk_level="高",
                        confidence=95.0,
                        notification_methods=rule.notification_methods
                    )

                    alerts.append(alert)
                    rule.last_triggered = datetime.now()

        except Exception as e:
            self.logger.error(f"Risk warning check error: {e}")

        return alerts

    async def check_performance_updates(self, current_performance: Dict[str, Any]) -> List[Alert]:
        """パフォーマンス更新チェック"""
        alerts = []

        try:
            if not current_performance:
                return alerts

            total_return = current_performance.get('total_return', 0)
            daily_return = current_performance.get('daily_return', 0)

            # 大きなリターン変化をチェック
            rule = self.alert_rules.get("PERFORMANCE_UPDATE_DAILY")
            if (rule and rule.is_active and
                abs(daily_return) >= rule.threshold_values.get("return_change", 5)):

                # クールダウンチェック
                if (not rule.last_triggered or
                    datetime.now() - rule.last_triggered > timedelta(minutes=rule.cooldown_minutes)):

                    alert = Alert(
                        alert_id=f"PERFORMANCE_UPDATE_{int(time.time())}",
                        alert_type=AlertType.PERFORMANCE_UPDATE,
                        priority=AlertPriority.INFO,
                        title="パフォーマンス大幅変動",
                        message=f"日次リターン{daily_return:.1f}%の大幅変動。総リターン{total_return:.1f}%",
                        details={
                            "total_return": total_return,
                            "daily_return": daily_return,
                            "rule_id": rule.rule_id
                        },
                        suggested_actions=["ポジション確認", "リスク評価", "戦略見直し検討"],
                        risk_level="低" if daily_return > 0 else "中",
                        confidence=85.0,
                        notification_methods=rule.notification_methods
                    )

                    alerts.append(alert)
                    rule.last_triggered = datetime.now()

        except Exception as e:
            self.logger.error(f"Performance update check error: {e}")

        return alerts

    async def get_active_alerts(self) -> List[Alert]:
        """アクティブアラート取得"""
        current_time = datetime.now()
        active_alerts = []

        for alert in self.active_alerts.values():
            # 有効期限チェック
            if alert.expires_at and current_time > alert.expires_at:
                alert.is_active = False
                continue

            if alert.is_active:
                active_alerts.append(alert)

        # 優先度順でソート
        priority_order = {
            AlertPriority.CRITICAL: 0,
            AlertPriority.HIGH: 1,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 3,
            AlertPriority.INFO: 4
        }

        active_alerts.sort(key=lambda x: (priority_order.get(x.priority, 5), x.created_at), reverse=True)
        return active_alerts

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """アラート確認済み設定"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.is_acknowledged = True
                alert.acknowledged_at = datetime.now()

                # データベース更新
                await self._save_alert_to_db(alert)

                self.logger.info(f"Alert acknowledged: {alert_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to acknowledge alert {alert_id}: {e}")

        return False

    def get_alert_statistics(self) -> Dict[str, Any]:
        """アラート統計取得"""
        try:
            total_alerts = len(self.alert_history)
            active_count = len([a for a in self.active_alerts.values() if a.is_active])

            # タイプ別集計
            type_counts = defaultdict(int)
            priority_counts = defaultdict(int)

            for alert in self.alert_history:
                type_counts[alert.alert_type.value] += 1
                priority_counts[alert.priority.value] += 1

            return {
                "total_alerts": total_alerts,
                "active_alerts": active_count,
                "acknowledged_alerts": len([a for a in self.alert_history if a.is_acknowledged]),
                "alert_types": dict(type_counts),
                "priority_distribution": dict(priority_counts),
                "active_rules": len([r for r in self.alert_rules.values() if r.is_active]),
                "notification_queue_size": len(self.notification_queue),
                "system_running": self.is_running
            }

        except Exception as e:
            self.logger.error(f"Failed to get alert statistics: {e}")
            return {}

# テスト関数
async def test_alert_system():
    """リアルタイムアラート・通知システムのテスト"""
    print("=== リアルタイムアラート・通知システム テスト ===")

    alert_system = RealTimeAlertSystem()

    print(f"アラートルール数: {len(alert_system.alert_rules)}")
    print(f"通知設定: コンソール={alert_system.notification_settings['console_enabled']}")

    # 監視開始
    print(f"\n[ 監視開始 ]")
    await alert_system.start_monitoring()

    # テストアラート作成
    print(f"\n[ テストアラート作成 ]")

    # 買いシグナルアラート
    buy_alert = Alert(
        alert_id=f"TEST_BUY_{int(time.time())}",
        alert_type=AlertType.BUY_SIGNAL,
        priority=AlertPriority.HIGH,
        symbol="7203",
        title="テスト買いシグナル",
        message="トヨタ自動車で高信頼度の買いシグナルが発生（テスト）",
        details={"score": 92.5, "confidence": 95.0},
        suggested_actions=["エントリー検討", "リスク管理設定"],
        risk_level="中",
        confidence=95.0,
        notification_methods=[NotificationMethod.CONSOLE, NotificationMethod.FILE]
    )

    await alert_system.create_alert(buy_alert)

    # リスク警告アラート
    risk_alert = Alert(
        alert_id=f"TEST_RISK_{int(time.time())}",
        alert_type=AlertType.RISK_WARNING,
        priority=AlertPriority.CRITICAL,
        title="テストリスク警告",
        message="高ボラティリティ状況を検知（テスト）",
        details={"max_drawdown": 12.5, "volatility": 18.2},
        suggested_actions=["ポジション縮小", "損切り設定"],
        risk_level="高",
        confidence=98.0,
        notification_methods=[NotificationMethod.CONSOLE, NotificationMethod.FILE]
    )

    await alert_system.create_alert(risk_alert)

    # 少し待機（通知処理）
    await asyncio.sleep(2)

    # アクティブアラート確認
    print(f"\n[ アクティブアラート確認 ]")
    active_alerts = await alert_system.get_active_alerts()
    print(f"アクティブアラート数: {len(active_alerts)}")

    for alert in active_alerts:
        print(f"  {alert.priority.value}: {alert.title} ({alert.symbol})")

    # 買いシグナルチェックテスト
    print(f"\n[ 買いシグナルチェックテスト ]")
    test_analysis = [
        {"symbol": "6861", "score": 88.5, "confidence": 92.0, "action": "強い買い"},
        {"symbol": "4751", "score": 95.2, "confidence": 96.5, "action": "買い"}
    ]

    buy_signals = await alert_system.check_buy_signals(test_analysis)
    print(f"検出された買いシグナル: {len(buy_signals)}件")

    # リスクチェックテスト
    print(f"\n[ リスクチェックテスト ]")
    test_portfolio = {
        "max_drawdown": 11.2,
        "volatility": 16.8,
        "win_rate": 45.0
    }

    risk_warnings = await alert_system.check_risk_warnings(test_portfolio)
    print(f"検出されたリスク警告: {len(risk_warnings)}件")

    # アラート統計
    print(f"\n[ アラート統計 ]")
    stats = alert_system.get_alert_statistics()
    print(f"総アラート数: {stats['total_alerts']}")
    print(f"アクティブ数: {stats['active_alerts']}")
    print(f"アラートタイプ分布: {stats['alert_types']}")
    print(f"優先度分布: {stats['priority_distribution']}")

    # 監視停止
    print(f"\n[ 監視停止 ]")
    await alert_system.stop_monitoring()

    print(f"\n=== リアルタイムアラート・通知システム テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    asyncio.run(test_alert_system())