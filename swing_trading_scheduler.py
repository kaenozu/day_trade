#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trading Scheduler System - スイングトレード用スケジューラー機能
Issue #941 対応: 購入記録機能、売りタイミング監視システム、スケジュール管理UI
"""

import time
import json
import sqlite3
import threading
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import uuid

# カスタムモジュール
try:
    from performance_monitor import performance_monitor, track_performance
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False
    def track_performance(func):
        return func

try:
    from audit_logger import audit_logger
    HAS_AUDIT_LOGGER = True
except ImportError:
    HAS_AUDIT_LOGGER = False

try:
    from enhanced_ml_ensemble_system import enhanced_ml_ensemble
    HAS_ML_ENSEMBLE = True
except ImportError:
    HAS_ML_ENSEMBLE = False

try:
    from advanced_feature_engineering_system import advanced_feature_engineering
    HAS_FEATURE_ENGINEERING = True
except ImportError:
    HAS_FEATURE_ENGINEERING = False


class HoldingStatus(Enum):
    """保有状況"""
    MONITORING = "monitoring"      # 監視中
    SELL_CONSIDER = "sell_consider"  # 売却検討
    ATTENTION = "attention"        # 要注意
    PARTIAL_SOLD = "partial_sold"  # 部分売却済み
    SOLD = "sold"                  # 全売却済み


class PurchaseStrategy(Enum):
    """購入戦略"""
    GROWTH = "growth"              # 成長株投資
    VALUE = "value"                # バリュー投資  
    DIVIDEND = "dividend"          # 配当投資
    TECHNICAL = "technical"        # テクニカル分析
    MOMENTUM = "momentum"          # モメンタム投資
    MIXED = "mixed"                # 混合戦略


class SellSignalStrength(Enum):
    """売りシグナル強度"""
    NONE = 0      # なし
    WEAK = 1      # 弱い
    MODERATE = 2  # 中程度
    STRONG = 3    # 強い
    CRITICAL = 4  # 緊急


@dataclass
class PurchaseRecord:
    """購入記録"""
    id: str
    symbol: str
    symbol_name: str
    purchase_date: date
    purchase_price: float
    shares: int
    total_amount: float
    strategy: PurchaseStrategy
    purchase_reason: str
    expected_hold_period_days: int = 30
    target_profit_percent: float = 20.0
    stop_loss_percent: float = -10.0
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class MonitoringSchedule:
    """監視スケジュール"""
    id: str
    purchase_id: str
    symbol: str
    current_price: Optional[float] = None
    current_change_percent: Optional[float] = None
    
    # 売りシグナル分析
    sell_signal_strength: SellSignalStrength = SellSignalStrength.NONE
    sell_signal_reasons: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    
    # 予想売却日
    next_evaluation_date: date = field(default_factory=lambda: date.today())
    expected_sell_date: Optional[date] = None
    expected_sell_reason: str = ""
    
    # ステータス
    status: HoldingStatus = HoldingStatus.MONITORING
    alert_level: int = 0  # 0-5の警告レベル
    
    # 分析結果
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    sentiment_score: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class PartialSell:
    """部分売却記録"""
    id: str
    purchase_id: str
    sell_date: date
    sell_price: float
    shares_sold: int
    sell_amount: float
    profit_loss: float
    sell_reason: str
    remaining_shares: int
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """アラート"""
    id: str
    purchase_id: str
    symbol: str
    alert_type: str
    message: str
    priority: int  # 1-5（高-低）
    is_read: bool = False
    
    created_at: datetime = field(default_factory=datetime.now)


class SwingTradingScheduler:
    """スイングトレード用スケジューラーシステム"""
    
    def __init__(self, db_path: str = "data/swing_trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # データベース初期化
        self._initialize_database()
        
        # モニタリングスレッド
        self._monitoring_thread = None
        self._stop_monitoring = False
        
        # ロック
        self._lock = threading.Lock()
        
        # キャッシュ
        self.purchase_cache = {}
        self.monitoring_cache = {}
        
        print("Swing Trading Scheduler initialized")
    
    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # 購入記録テーブル
            conn.execute('''
                CREATE TABLE IF NOT EXISTS purchase_records (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    symbol_name TEXT NOT NULL,
                    purchase_date DATE NOT NULL,
                    purchase_price REAL NOT NULL,
                    shares INTEGER NOT NULL,
                    total_amount REAL NOT NULL,
                    strategy TEXT NOT NULL,
                    purchase_reason TEXT NOT NULL,
                    expected_hold_period_days INTEGER DEFAULT 30,
                    target_profit_percent REAL DEFAULT 20.0,
                    stop_loss_percent REAL DEFAULT -10.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 監視スケジュールテーブル
            conn.execute('''
                CREATE TABLE IF NOT EXISTS monitoring_schedule (
                    id TEXT PRIMARY KEY,
                    purchase_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    current_price REAL,
                    current_change_percent REAL,
                    sell_signal_strength INTEGER DEFAULT 0,
                    sell_signal_reasons TEXT,
                    confidence_score REAL DEFAULT 0.0,
                    next_evaluation_date DATE,
                    expected_sell_date DATE,
                    expected_sell_reason TEXT,
                    status TEXT DEFAULT 'monitoring',
                    alert_level INTEGER DEFAULT 0,
                    technical_score REAL DEFAULT 0.0,
                    fundamental_score REAL DEFAULT 0.0,
                    sentiment_score REAL DEFAULT 0.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (purchase_id) REFERENCES purchase_records (id)
                )
            ''')
            
            # 部分売却記録テーブル  
            conn.execute('''
                CREATE TABLE IF NOT EXISTS partial_sells (
                    id TEXT PRIMARY KEY,
                    purchase_id TEXT NOT NULL,
                    sell_date DATE NOT NULL,
                    sell_price REAL NOT NULL,
                    shares_sold INTEGER NOT NULL,
                    sell_amount REAL NOT NULL,
                    profit_loss REAL NOT NULL,
                    sell_reason TEXT NOT NULL,
                    remaining_shares INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (purchase_id) REFERENCES purchase_records (id)
                )
            ''')
            
            # アラートテーブル
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    purchase_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    is_read BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (purchase_id) REFERENCES purchase_records (id)
                )
            ''')
            
            # インデックス作成
            conn.execute('CREATE INDEX IF NOT EXISTS idx_purchase_symbol ON purchase_records(symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_purchase_date ON purchase_records(purchase_date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_monitoring_symbol ON monitoring_schedule(symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_monitoring_status ON monitoring_schedule(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts(priority)')
            
            conn.commit()
    
    @track_performance
    def record_purchase(self,
                       symbol: str,
                       symbol_name: str,
                       purchase_price: float,
                       shares: int,
                       strategy: PurchaseStrategy,
                       purchase_reason: str,
                       target_profit_percent: float = 20.0,
                       stop_loss_percent: float = -10.0,
                       expected_hold_days: int = 30) -> str:
        """購入記録を登録"""
        try:
            purchase_id = str(uuid.uuid4())
            total_amount = purchase_price * shares
            
            purchase_record = PurchaseRecord(
                id=purchase_id,
                symbol=symbol,
                symbol_name=symbol_name,
                purchase_date=date.today(),
                purchase_price=purchase_price,
                shares=shares,
                total_amount=total_amount,
                strategy=strategy,
                purchase_reason=purchase_reason,
                expected_hold_period_days=expected_hold_days,
                target_profit_percent=target_profit_percent,
                stop_loss_percent=stop_loss_percent
            )
            
            # データベースに保存
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO purchase_records 
                    (id, symbol, symbol_name, purchase_date, purchase_price, shares, 
                     total_amount, strategy, purchase_reason, expected_hold_period_days,
                     target_profit_percent, stop_loss_percent, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    purchase_record.id,
                    purchase_record.symbol,
                    purchase_record.symbol_name,
                    purchase_record.purchase_date,
                    purchase_record.purchase_price,
                    purchase_record.shares,
                    purchase_record.total_amount,
                    purchase_record.strategy.value,
                    purchase_record.purchase_reason,
                    purchase_record.expected_hold_period_days,
                    purchase_record.target_profit_percent,
                    purchase_record.stop_loss_percent,
                    purchase_record.created_at,
                    purchase_record.updated_at
                ))
                conn.commit()
            
            # 監視スケジュール作成
            self._create_monitoring_schedule(purchase_record)
            
            # アラート作成
            self.create_alert(
                purchase_id=purchase_id,
                symbol=symbol,
                alert_type="purchase_recorded",
                message=f"{symbol_name}の購入を記録しました。{shares}株 @ ¥{purchase_price:,.0f}",
                priority=3
            )
            
            # ログ記録
            if HAS_AUDIT_LOGGER:
                audit_logger.log_business_event(
                    "swing_trading_purchase_recorded",
                    {
                        "purchase_id": purchase_id,
                        "symbol": symbol,
                        "shares": shares,
                        "total_amount": total_amount,
                        "strategy": strategy.value
                    }
                )
            
            # キャッシュ更新
            self.purchase_cache[purchase_id] = purchase_record
            
            return purchase_id
            
        except Exception as e:
            print(f"購入記録エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"symbol": symbol, "context": "purchase_recording"})
            raise
    
    def _create_monitoring_schedule(self, purchase_record: PurchaseRecord):
        """監視スケジュールを作成"""
        schedule_id = str(uuid.uuid4())
        
        monitoring_schedule = MonitoringSchedule(
            id=schedule_id,
            purchase_id=purchase_record.id,
            symbol=purchase_record.symbol,
            next_evaluation_date=date.today() + timedelta(days=1),
            expected_sell_date=date.today() + timedelta(days=purchase_record.expected_hold_period_days)
        )
        
        # データベースに保存
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO monitoring_schedule 
                (id, purchase_id, symbol, next_evaluation_date, expected_sell_date,
                 status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                monitoring_schedule.id,
                monitoring_schedule.purchase_id,
                monitoring_schedule.symbol,
                monitoring_schedule.next_evaluation_date,
                monitoring_schedule.expected_sell_date,
                monitoring_schedule.status.value,
                monitoring_schedule.created_at,
                monitoring_schedule.updated_at
            ))
            conn.commit()
        
        # キャッシュ更新
        self.monitoring_cache[schedule_id] = monitoring_schedule
    
    @track_performance
    def evaluate_sell_timing(self, purchase_id: str) -> MonitoringSchedule:
        """売りタイミングの評価"""
        try:
            # 購入記録取得
            purchase_record = self.get_purchase_record(purchase_id)
            if not purchase_record:
                raise ValueError(f"Purchase record not found: {purchase_id}")
            
            # 現在価格取得（実際の実装では株価APIを使用）
            current_price = self._get_current_price(purchase_record.symbol)
            
            if current_price is None:
                print(f"価格取得失敗: {purchase_record.symbol}")
                return None
            
            # 変動率計算
            change_percent = ((current_price - purchase_record.purchase_price) / purchase_record.purchase_price) * 100
            
            # 売りシグナル分析
            sell_analysis = self._analyze_sell_signals(
                purchase_record, current_price, change_percent
            )
            
            # 監視スケジュール更新
            monitoring_schedule = self._update_monitoring_schedule(
                purchase_id, current_price, change_percent, sell_analysis
            )
            
            # アラート生成
            self._generate_alerts_if_needed(purchase_record, monitoring_schedule)
            
            return monitoring_schedule
            
        except Exception as e:
            print(f"売りタイミング評価エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"purchase_id": purchase_id, "context": "sell_timing_evaluation"})
            return None
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """現在価格を取得（ダミー実装）"""
        # 実際の実装では、yfinance、株価APIなどを使用
        # ここではダミー価格を返す
        base_price = 1000 + (hash(symbol) % 2000)  # 1000-3000の範囲
        time_factor = int(time.time() // 3600) % 100  # 時間による変動
        volatility = (hash(symbol + str(time_factor)) % 200 - 100) / 100  # -100% to +100%
        
        current_price = base_price * (1 + volatility * 0.1)  # 最大±10%の変動
        return max(current_price, 100)  # 最低100円
    
    def _analyze_sell_signals(self,
                             purchase_record: PurchaseRecord,
                             current_price: float,
                             change_percent: float) -> Dict[str, Any]:
        """売りシグナルを分析"""
        sell_signals = []
        signal_strength = SellSignalStrength.NONE
        confidence_score = 0.0
        
        # 1. 利益確定シグナル
        if change_percent >= purchase_record.target_profit_percent:
            sell_signals.append(f"目標利益達成: {change_percent:.1f}% >= {purchase_record.target_profit_percent:.1f}%")
            signal_strength = SellSignalStrength.STRONG
            confidence_score += 0.3
        
        # 2. 損切りシグナル
        if change_percent <= purchase_record.stop_loss_percent:
            sell_signals.append(f"ストップロス: {change_percent:.1f}% <= {purchase_record.stop_loss_percent:.1f}%")
            signal_strength = SellSignalStrength.CRITICAL
            confidence_score += 0.4
        
        # 3. 保有期間シグナル
        days_held = (date.today() - purchase_record.purchase_date).days
        if days_held >= purchase_record.expected_hold_period_days:
            sell_signals.append(f"予定保有期間経過: {days_held}日 >= {purchase_record.expected_hold_period_days}日")
            if signal_strength.value < SellSignalStrength.MODERATE.value:
                signal_strength = SellSignalStrength.MODERATE
            confidence_score += 0.2
        
        # 4. テクニカル分析シグナル（簡易版）
        technical_score = self._simple_technical_analysis(purchase_record.symbol, current_price)
        
        if technical_score < -0.5:
            sell_signals.append(f"テクニカル売りシグナル: スコア {technical_score:.2f}")
            if signal_strength.value < SellSignalStrength.MODERATE.value:
                signal_strength = SellSignalStrength.MODERATE
            confidence_score += abs(technical_score) * 0.2
        
        # 5. ML予測シグナル（利用可能な場合）
        if HAS_ML_ENSEMBLE:
            try:
                # 簡易特徴量作成
                import pandas as pd
                features = pd.DataFrame({
                    'current_price': [current_price],
                    'change_percent': [change_percent],
                    'days_held': [days_held],
                    'purchase_price': [purchase_record.purchase_price]
                })
                
                # 予測実行（実際の実装では適切な特徴量を使用）
                prediction_result = enhanced_ml_ensemble.predict(features, purchase_record.symbol)
                
                if prediction_result.ensemble_prediction < 0:  # 売りシグナル
                    sell_signals.append(f"AI予測: 売りシグナル (信頼度: {prediction_result.confidence_score:.2f})")
                    if signal_strength.value < SellSignalStrength.WEAK.value:
                        signal_strength = SellSignalStrength.WEAK
                    confidence_score += prediction_result.confidence_score * 0.1
                    
            except Exception as e:
                print(f"ML予測エラー: {e}")
        
        # 6. センチメント分析シグナル（利用可能な場合）
        if HAS_FEATURE_ENGINEERING:
            try:
                sentiment_result = advanced_feature_engineering.analyze_market_sentiment(purchase_record.symbol)
                
                if sentiment_result.sentiment_score < -0.3:  # ネガティブセンチメント
                    sell_signals.append(f"市場センチメント悪化: {sentiment_result.sentiment_score:.2f}")
                    if signal_strength.value < SellSignalStrength.WEAK.value:
                        signal_strength = SellSignalStrength.WEAK
                    confidence_score += sentiment_result.confidence * 0.1
                    
            except Exception as e:
                print(f"センチメント分析エラー: {e}")
        
        # 信頼度スコア正規化
        confidence_score = min(confidence_score, 1.0)
        
        return {
            'signal_strength': signal_strength,
            'signals': sell_signals,
            'confidence_score': confidence_score,
            'technical_score': technical_score,
            'fundamental_score': 0.0,  # 実装予定
            'sentiment_score': 0.0     # 実装予定
        }
    
    def _simple_technical_analysis(self, symbol: str, current_price: float) -> float:
        """簡易テクニカル分析"""
        # 実際の実装では移動平均、RSI、MACDなどを計算
        # ここでは簡易的なスコアを返す
        
        # シンボルと価格から疑似的なテクニカルスコアを生成
        hash_value = hash(f"{symbol}_{int(current_price)}_{int(time.time() // 86400)}")  # 日次で変更
        normalized_hash = (hash_value % 2000 - 1000) / 1000  # -1.0 to 1.0
        
        return normalized_hash
    
    def _update_monitoring_schedule(self,
                                  purchase_id: str,
                                  current_price: float,
                                  change_percent: float,
                                  sell_analysis: Dict[str, Any]) -> MonitoringSchedule:
        """監視スケジュールを更新"""
        
        # 現在の監視スケジュール取得
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute('''
                SELECT * FROM monitoring_schedule WHERE purchase_id = ?
            ''', (purchase_id,)).fetchone()
            
            if not row:
                return None
        
        # ステータス決定
        signal_strength = sell_analysis['signal_strength']
        status = HoldingStatus.MONITORING
        alert_level = 0
        
        if signal_strength == SellSignalStrength.CRITICAL:
            status = HoldingStatus.ATTENTION
            alert_level = 5
        elif signal_strength == SellSignalStrength.STRONG:
            status = HoldingStatus.SELL_CONSIDER
            alert_level = 4
        elif signal_strength == SellSignalStrength.MODERATE:
            status = HoldingStatus.SELL_CONSIDER
            alert_level = 3
        elif signal_strength == SellSignalStrength.WEAK:
            alert_level = 2
        
        # 次回評価日設定
        next_evaluation_date = date.today() + timedelta(days=1)
        if signal_strength.value >= SellSignalStrength.MODERATE.value:
            next_evaluation_date = date.today()  # 即座に再評価
        
        # データベース更新
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE monitoring_schedule SET
                    current_price = ?,
                    current_change_percent = ?,
                    sell_signal_strength = ?,
                    sell_signal_reasons = ?,
                    confidence_score = ?,
                    next_evaluation_date = ?,
                    status = ?,
                    alert_level = ?,
                    technical_score = ?,
                    fundamental_score = ?,
                    sentiment_score = ?,
                    updated_at = ?
                WHERE purchase_id = ?
            ''', (
                current_price,
                change_percent,
                signal_strength.value,
                json.dumps(sell_analysis['signals'], ensure_ascii=False),
                sell_analysis['confidence_score'],
                next_evaluation_date,
                status.value,
                alert_level,
                sell_analysis['technical_score'],
                sell_analysis['fundamental_score'],
                sell_analysis['sentiment_score'],
                datetime.now(),
                purchase_id
            ))
            conn.commit()
        
        # 更新されたスケジュール作成
        monitoring_schedule = MonitoringSchedule(
            id=row['id'],
            purchase_id=purchase_id,
            symbol=row['symbol'],
            current_price=current_price,
            current_change_percent=change_percent,
            sell_signal_strength=signal_strength,
            sell_signal_reasons=sell_analysis['signals'],
            confidence_score=sell_analysis['confidence_score'],
            next_evaluation_date=next_evaluation_date,
            expected_sell_date=date.fromisoformat(row['expected_sell_date']) if row['expected_sell_date'] else None,
            status=status,
            alert_level=alert_level,
            technical_score=sell_analysis['technical_score'],
            fundamental_score=sell_analysis['fundamental_score'],
            sentiment_score=sell_analysis['sentiment_score'],
            updated_at=datetime.now()
        )
        
        return monitoring_schedule
    
    def _generate_alerts_if_needed(self, purchase_record: PurchaseRecord, monitoring_schedule: MonitoringSchedule):
        """必要に応じてアラートを生成"""
        
        if monitoring_schedule.alert_level >= 4:  # 高優先度アラート
            change_percent = monitoring_schedule.current_change_percent or 0
            
            if monitoring_schedule.sell_signal_strength == SellSignalStrength.CRITICAL:
                message = f"🚨 緊急: {purchase_record.symbol_name} ストップロス発動 ({change_percent:+.1f}%)"
                priority = 1
            elif monitoring_schedule.sell_signal_strength == SellSignalStrength.STRONG:
                message = f"📈 売却検討: {purchase_record.symbol_name} 目標利益達成 ({change_percent:+.1f}%)"
                priority = 2
            else:
                message = f"⚠️ 注意: {purchase_record.symbol_name} 売りシグナル検出 ({change_percent:+.1f}%)"
                priority = 3
            
            self.create_alert(
                purchase_id=purchase_record.id,
                symbol=purchase_record.symbol,
                alert_type="sell_signal",
                message=message,
                priority=priority
            )
    
    def create_alert(self,
                    purchase_id: str,
                    symbol: str,
                    alert_type: str,
                    message: str,
                    priority: int) -> str:
        """アラートを作成"""
        alert_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO alerts (id, purchase_id, symbol, alert_type, message, priority, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (alert_id, purchase_id, symbol, alert_type, message, priority, datetime.now()))
            conn.commit()
        
        return alert_id
    
    def record_partial_sell(self,
                           purchase_id: str,
                           sell_price: float,
                           shares_sold: int,
                           sell_reason: str) -> str:
        """部分売却を記録"""
        try:
            purchase_record = self.get_purchase_record(purchase_id)
            if not purchase_record:
                raise ValueError(f"Purchase record not found: {purchase_id}")
            
            # 売却可能株数チェック
            current_shares = self._get_current_shares(purchase_id)
            if shares_sold > current_shares:
                raise ValueError(f"売却株数が保有株数を超過: {shares_sold} > {current_shares}")
            
            sell_id = str(uuid.uuid4())
            sell_amount = sell_price * shares_sold
            profit_loss = (sell_price - purchase_record.purchase_price) * shares_sold
            remaining_shares = current_shares - shares_sold
            
            partial_sell = PartialSell(
                id=sell_id,
                purchase_id=purchase_id,
                sell_date=date.today(),
                sell_price=sell_price,
                shares_sold=shares_sold,
                sell_amount=sell_amount,
                profit_loss=profit_loss,
                sell_reason=sell_reason,
                remaining_shares=remaining_shares
            )
            
            # データベースに保存
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO partial_sells 
                    (id, purchase_id, sell_date, sell_price, shares_sold, sell_amount,
                     profit_loss, sell_reason, remaining_shares, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    partial_sell.id,
                    partial_sell.purchase_id,
                    partial_sell.sell_date,
                    partial_sell.sell_price,
                    partial_sell.shares_sold,
                    partial_sell.sell_amount,
                    partial_sell.profit_loss,
                    partial_sell.sell_reason,
                    partial_sell.remaining_shares,
                    partial_sell.created_at
                ))
                conn.commit()
            
            # 監視ステータス更新
            if remaining_shares == 0:
                # 全売却の場合
                self._update_monitoring_status(purchase_id, HoldingStatus.SOLD)
            else:
                # 部分売却の場合
                self._update_monitoring_status(purchase_id, HoldingStatus.PARTIAL_SOLD)
            
            # アラート作成
            profit_loss_text = f"利益 ¥{profit_loss:,.0f}" if profit_loss >= 0 else f"損失 ¥{abs(profit_loss):,.0f}"
            self.create_alert(
                purchase_id=purchase_id,
                symbol=purchase_record.symbol,
                alert_type="partial_sell",
                message=f"部分売却完了: {shares_sold}株 @ ¥{sell_price:,.0f} ({profit_loss_text})",
                priority=3
            )
            
            # ログ記録
            if HAS_AUDIT_LOGGER:
                audit_logger.log_business_event(
                    "swing_trading_partial_sell",
                    {
                        "sell_id": sell_id,
                        "purchase_id": purchase_id,
                        "symbol": purchase_record.symbol,
                        "shares_sold": shares_sold,
                        "profit_loss": profit_loss
                    }
                )
            
            return sell_id
            
        except Exception as e:
            print(f"部分売却記録エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"purchase_id": purchase_id, "context": "partial_sell"})
            raise
    
    def _get_current_shares(self, purchase_id: str) -> int:
        """現在の保有株数を取得"""
        purchase_record = self.get_purchase_record(purchase_id)
        if not purchase_record:
            return 0
        
        # 部分売却の合計を取得
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute('''
                SELECT COALESCE(SUM(shares_sold), 0) as total_sold 
                FROM partial_sells 
                WHERE purchase_id = ?
            ''', (purchase_id,)).fetchone()
            
            total_sold = result[0] if result else 0
        
        return purchase_record.shares - total_sold
    
    def _update_monitoring_status(self, purchase_id: str, status: HoldingStatus):
        """監視ステータスを更新"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE monitoring_schedule SET status = ?, updated_at = ?
                WHERE purchase_id = ?
            ''', (status.value, datetime.now(), purchase_id))
            conn.commit()
    
    def get_purchase_record(self, purchase_id: str) -> Optional[PurchaseRecord]:
        """購入記録を取得"""
        # キャッシュチェック
        if purchase_id in self.purchase_cache:
            return self.purchase_cache[purchase_id]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute('''
                SELECT * FROM purchase_records WHERE id = ?
            ''', (purchase_id,)).fetchone()
            
            if row:
                purchase_record = PurchaseRecord(
                    id=row['id'],
                    symbol=row['symbol'],
                    symbol_name=row['symbol_name'],
                    purchase_date=date.fromisoformat(row['purchase_date']),
                    purchase_price=row['purchase_price'],
                    shares=row['shares'],
                    total_amount=row['total_amount'],
                    strategy=PurchaseStrategy(row['strategy']),
                    purchase_reason=row['purchase_reason'],
                    expected_hold_period_days=row['expected_hold_period_days'],
                    target_profit_percent=row['target_profit_percent'],
                    stop_loss_percent=row['stop_loss_percent'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
                
                # キャッシュに保存
                self.purchase_cache[purchase_id] = purchase_record
                return purchase_record
        
        return None
    
    def get_monitoring_list(self, status_filter: Optional[HoldingStatus] = None) -> List[Dict[str, Any]]:
        """監視対象銘柄一覧を取得"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = '''
                SELECT 
                    p.id as purchase_id,
                    p.symbol,
                    p.symbol_name,
                    p.purchase_date,
                    p.purchase_price,
                    p.shares,
                    p.strategy,
                    m.current_price,
                    m.current_change_percent,
                    m.status,
                    m.sell_signal_strength,
                    m.confidence_score,
                    m.alert_level,
                    m.expected_sell_date,
                    m.updated_at as last_evaluation
                FROM purchase_records p
                LEFT JOIN monitoring_schedule m ON p.id = m.purchase_id
                WHERE 1=1
            '''
            
            params = []
            if status_filter:
                query += " AND m.status = ?"
                params.append(status_filter.value)
            
            query += " ORDER BY m.alert_level DESC, m.updated_at DESC"
            
            rows = conn.execute(query, params).fetchall()
            
            monitoring_list = []
            for row in rows:
                current_shares = self._get_current_shares(row['purchase_id'])
                current_value = (row['current_price'] or 0) * current_shares
                unrealized_pl = current_value - (row['purchase_price'] * current_shares)
                
                monitoring_list.append({
                    'purchase_id': row['purchase_id'],
                    'symbol': row['symbol'],
                    'symbol_name': row['symbol_name'],
                    'purchase_date': row['purchase_date'],
                    'purchase_price': row['purchase_price'],
                    'original_shares': row['shares'],
                    'current_shares': current_shares,
                    'strategy': row['strategy'],
                    'current_price': row['current_price'],
                    'change_percent': row['current_change_percent'],
                    'current_value': current_value,
                    'unrealized_profit_loss': unrealized_pl,
                    'status': row['status'],
                    'sell_signal_strength': row['sell_signal_strength'],
                    'confidence_score': row['confidence_score'],
                    'alert_level': row['alert_level'],
                    'expected_sell_date': row['expected_sell_date'],
                    'last_evaluation': row['last_evaluation']
                })
        
        return monitoring_list
    
    def get_alerts(self, limit: int = 50, unread_only: bool = False) -> List[Dict[str, Any]]:
        """アラート一覧を取得"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = '''
                SELECT 
                    a.*,
                    p.symbol_name
                FROM alerts a
                LEFT JOIN purchase_records p ON a.purchase_id = p.id
                WHERE 1=1
            '''
            
            params = []
            if unread_only:
                query += " AND a.is_read = FALSE"
            
            query += " ORDER BY a.priority ASC, a.created_at DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            
            return [dict(row) for row in rows]
    
    def mark_alert_as_read(self, alert_id: str) -> bool:
        """アラートを既読にする"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                UPDATE alerts SET is_read = TRUE WHERE id = ?
            ''', (alert_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def start_monitoring(self, interval_minutes: int = 60):
        """定期監視を開始"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_minutes,),
            daemon=True
        )
        self._monitoring_thread.start()
        
        print(f"Monitoring started with {interval_minutes} minute intervals")
    
    def stop_monitoring(self):
        """定期監視を停止"""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        
        print("Monitoring stopped")
    
    def _monitoring_loop(self, interval_minutes: int):
        """監視ループ"""
        while not self._stop_monitoring:
            try:
                # 監視対象取得
                monitoring_list = self.get_monitoring_list()
                
                for item in monitoring_list:
                    if item['status'] not in ['sold']:
                        # 売りタイミング評価
                        self.evaluate_sell_timing(item['purchase_id'])
                
                print(f"Monitoring completed for {len(monitoring_list)} positions")
                
                # インターバル待機
                for _ in range(interval_minutes * 60):
                    if self._stop_monitoring:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Monitoring error: {e}")
                if HAS_AUDIT_LOGGER:
                    audit_logger.log_error_with_context(e, {"context": "monitoring_loop"})
                
                # エラー時は短時間待機
                time.sleep(60)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """ポートフォリオサマリーを取得"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # 基本統計
            stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_positions,
                    SUM(total_amount) as total_invested,
                    COUNT(CASE WHEN m.status = 'monitoring' THEN 1 END) as monitoring_count,
                    COUNT(CASE WHEN m.status = 'sell_consider' THEN 1 END) as sell_consider_count,
                    COUNT(CASE WHEN m.status = 'attention' THEN 1 END) as attention_count,
                    COUNT(CASE WHEN m.status = 'sold' THEN 1 END) as sold_count
                FROM purchase_records p
                LEFT JOIN monitoring_schedule m ON p.id = m.purchase_id
            ''').fetchone()
            
            # アラート統計
            alert_stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_alerts,
                    COUNT(CASE WHEN is_read = FALSE THEN 1 END) as unread_alerts,
                    COUNT(CASE WHEN priority <= 2 THEN 1 END) as high_priority_alerts
                FROM alerts
                WHERE created_at > datetime('now', '-7 days')
            ''').fetchone()
            
            # 部分売却統計
            sell_stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_sells,
                    SUM(profit_loss) as total_realized_pl,
                    AVG(profit_loss) as avg_profit_per_sell
                FROM partial_sells
            ''').fetchone()
        
        # 現在評価額計算
        monitoring_list = self.get_monitoring_list()
        current_value = sum(
            (item['current_price'] or 0) * item['current_shares'] 
            for item in monitoring_list 
            if item['status'] not in ['sold']
        )
        
        unrealized_pl = sum(
            item['unrealized_profit_loss'] 
            for item in monitoring_list 
            if item['status'] not in ['sold']
        )
        
        return {
            'portfolio_summary': {
                'total_positions': stats['total_positions'] or 0,
                'active_positions': (stats['total_positions'] or 0) - (stats['sold_count'] or 0),
                'total_invested': stats['total_invested'] or 0,
                'current_value': current_value,
                'unrealized_profit_loss': unrealized_pl,
                'realized_profit_loss': sell_stats['total_realized_pl'] or 0,
                'total_profit_loss': unrealized_pl + (sell_stats['total_realized_pl'] or 0)
            },
            'position_status': {
                'monitoring': stats['monitoring_count'] or 0,
                'sell_consider': stats['sell_consider_count'] or 0,
                'attention': stats['attention_count'] or 0,
                'sold': stats['sold_count'] or 0
            },
            'alerts': {
                'total_alerts': alert_stats['total_alerts'] or 0,
                'unread_alerts': alert_stats['unread_alerts'] or 0,
                'high_priority_alerts': alert_stats['high_priority_alerts'] or 0
            },
            'trading_activity': {
                'total_partial_sells': sell_stats['total_sells'] or 0,
                'average_profit_per_sell': sell_stats['avg_profit_per_sell'] or 0
            }
        }


# グローバルインスタンス
swing_trading_scheduler = SwingTradingScheduler()


if __name__ == "__main__":
    # テスト実行
    print("Swing Trading Scheduler テスト開始")
    
    # スケジューラーインスタンス作成
    scheduler = SwingTradingScheduler()
    
    print("\n1. 購入記録テスト")
    
    # テスト購入記録
    purchase_id1 = scheduler.record_purchase(
        symbol="7203",
        symbol_name="トヨタ自動車",
        purchase_price=2500,
        shares=100,
        strategy=PurchaseStrategy.VALUE,
        purchase_reason="PER低位、配当利回り良好、業績安定",
        target_profit_percent=25.0,
        stop_loss_percent=-8.0,
        expected_hold_days=45
    )
    print(f"購入記録1: {purchase_id1}")
    
    purchase_id2 = scheduler.record_purchase(
        symbol="9984",
        symbol_name="ソフトバンクグループ",
        purchase_price=8500,
        shares=50,
        strategy=PurchaseStrategy.GROWTH,
        purchase_reason="AI・テクノロジー投資のアップサイド期待",
        target_profit_percent=30.0,
        stop_loss_percent=-12.0,
        expected_hold_days=60
    )
    print(f"購入記録2: {purchase_id2}")
    
    print("\n2. 売りタイミング評価テスト")
    
    # 売りタイミング評価
    monitoring1 = scheduler.evaluate_sell_timing(purchase_id1)
    if monitoring1:
        print(f"評価結果1: {monitoring1.symbol} - {monitoring1.status.value}")
        print(f"  シグナル強度: {monitoring1.sell_signal_strength.value}")
        print(f"  信頼度: {monitoring1.confidence_score:.2f}")
        print(f"  アラートレベル: {monitoring1.alert_level}")
    
    monitoring2 = scheduler.evaluate_sell_timing(purchase_id2)
    if monitoring2:
        print(f"評価結果2: {monitoring2.symbol} - {monitoring2.status.value}")
    
    print("\n3. 監視対象一覧テスト")
    
    monitoring_list = scheduler.get_monitoring_list()
    print(f"監視対象銘柄数: {len(monitoring_list)}")
    
    for item in monitoring_list:
        print(f"  {item['symbol']} {item['symbol_name']}: {item['status']} "
              f"({item['change_percent']:+.1f}%, Alert: {item['alert_level']})")
    
    print("\n4. 部分売却テスト")
    
    # 部分売却実行
    try:
        sell_id = scheduler.record_partial_sell(
            purchase_id=purchase_id1,
            sell_price=2750,
            shares_sold=30,
            sell_reason="利益確定（一部）"
        )
        print(f"部分売却記録: {sell_id}")
    except Exception as e:
        print(f"部分売却エラー: {e}")
    
    print("\n5. アラート一覧テスト")
    
    alerts = scheduler.get_alerts(limit=10)
    print(f"アラート数: {len(alerts)}")
    
    for alert in alerts[:5]:  # 最初の5件を表示
        priority_text = ["🔴", "🟠", "🟡", "🔵", "⚪"][min(alert['priority']-1, 4)]
        read_status = "既読" if alert['is_read'] else "未読"
        print(f"  {priority_text} [{read_status}] {alert['message']}")
    
    print("\n6. ポートフォリオサマリーテスト")
    
    summary = scheduler.get_portfolio_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    
    print("\n7. 定期監視テスト（5秒間）")
    
    # 短時間の監視テスト
    scheduler.start_monitoring(interval_minutes=0.1)  # 6秒間隔
    time.sleep(5)
    scheduler.stop_monitoring()
    
    print("テスト完了 ✅")