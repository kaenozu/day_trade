#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Model - ポートフォリオとポジション管理
スイングトレード用のポジション管理システム
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

@dataclass
class Position:
    """個別ポジション"""
    id: str  # ユニークID
    symbol: str  # 銘柄コード
    name: str  # 銘柄名
    entry_date: str  # エントリー日
    entry_price: float  # エントリー価格
    quantity: int  # 数量
    position_type: str  # "LONG" or "SHORT"
    strategy_type: str  # スイング戦略タイプ
    target_price: Optional[float] = None  # 目標価格
    stop_loss_price: Optional[float] = None  # 損切り価格
    exit_date: Optional[str] = None  # エグジット日
    exit_price: Optional[float] = None  # エグジット価格
    status: str = "OPEN"  # "OPEN", "CLOSED", "PARTIAL"
    notes: str = ""  # メモ
    current_price: Optional[float] = None  # 現在価格（リアルタイム更新）
    
    @property
    def unrealized_pnl(self) -> float:
        """未実現損益"""
        if self.status == "CLOSED" or not self.current_price:
            return 0.0
        if self.position_type == "LONG":
            return (self.current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """未実現損益率"""
        if self.status == "CLOSED" or not self.current_price:
            return 0.0
        if self.position_type == "LONG":
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            return ((self.entry_price - self.current_price) / self.entry_price) * 100
    
    @property
    def realized_pnl(self) -> float:
        """実現損益"""
        if self.status != "CLOSED" or not self.exit_price:
            return 0.0
        if self.position_type == "LONG":
            return (self.exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - self.exit_price) * self.quantity
    
    @property
    def holding_days(self) -> int:
        """保有日数"""
        entry = datetime.strptime(self.entry_date, "%Y-%m-%d")
        if self.status == "CLOSED" and self.exit_date:
            exit_dt = datetime.strptime(self.exit_date, "%Y-%m-%d")
            return (exit_dt - entry).days
        else:
            return (datetime.now() - entry).days

@dataclass 
class PortfolioSummary:
    """ポートフォリオサマリー"""
    total_positions: int
    open_positions: int
    closed_positions: int
    total_investment: float
    current_value: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    average_holding_days: float
    best_position: Optional[str] = None
    worst_position: Optional[str] = None

class PortfolioManager:
    """ポートフォリオ管理クラス"""
    
    def __init__(self, db_path: str = "data/portfolio.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # データベースディレクトリの作成
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # データベース初期化
        self._init_db()
    
    def _init_db(self):
        """データベース初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # ポジションテーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS positions (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        name TEXT NOT NULL,
                        entry_date TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        quantity INTEGER NOT NULL,
                        position_type TEXT NOT NULL,
                        strategy_type TEXT NOT NULL,
                        target_price REAL,
                        stop_loss_price REAL,
                        exit_date TEXT,
                        exit_price REAL,
                        status TEXT DEFAULT 'OPEN',
                        notes TEXT DEFAULT '',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 価格履歴テーブル（パフォーマンス分析用）
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS price_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        price REAL NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # インデックス作成
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_history_symbol ON price_history(symbol)')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")
            raise
    
    def add_position(self, symbol: str, name: str, entry_price: float, 
                    quantity: int, position_type: str = "LONG", 
                    strategy_type: str = "スイングトレード", 
                    target_price: Optional[float] = None,
                    stop_loss_price: Optional[float] = None,
                    notes: str = "") -> str:
        """新規ポジション追加"""
        try:
            position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            entry_date = datetime.now().strftime("%Y-%m-%d")
            
            position = Position(
                id=position_id,
                symbol=symbol,
                name=name,
                entry_date=entry_date,
                entry_price=entry_price,
                quantity=quantity,
                position_type=position_type,
                strategy_type=strategy_type,
                target_price=target_price,
                stop_loss_price=stop_loss_price,
                notes=notes
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO positions 
                    (id, symbol, name, entry_date, entry_price, quantity, position_type, 
                     strategy_type, target_price, stop_loss_price, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position.id, position.symbol, position.name, position.entry_date,
                    position.entry_price, position.quantity, position.position_type,
                    position.strategy_type, position.target_price, position.stop_loss_price,
                    position.notes
                ))
                conn.commit()
            
            self.logger.info(f"新規ポジション追加: {symbol} ({position_id})")
            return position_id
            
        except Exception as e:
            self.logger.error(f"ポジション追加エラー: {e}")
            raise
    
    def close_position(self, position_id: str, exit_price: float, 
                      notes: str = "") -> bool:
        """ポジションクローズ"""
        try:
            exit_date = datetime.now().strftime("%Y-%m-%d")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE positions 
                    SET exit_date = ?, exit_price = ?, status = 'CLOSED', 
                        notes = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND status = 'OPEN'
                ''', (exit_date, exit_price, notes, position_id))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    self.logger.info(f"ポジションクローズ: {position_id}")
                    return True
                else:
                    self.logger.warning(f"ポジションが見つからないかすでにクローズ済み: {position_id}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"ポジションクローズエラー: {e}")
            raise
    
    def get_positions(self, status: Optional[str] = None) -> List[Position]:
        """ポジション一覧取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if status:
                    cursor.execute('''
                        SELECT * FROM positions WHERE status = ? 
                        ORDER BY entry_date DESC
                    ''', (status,))
                else:
                    cursor.execute('SELECT * FROM positions ORDER BY entry_date DESC')
                
                rows = cursor.fetchall()
                
                # カラム名取得
                columns = [description[0] for description in cursor.description]
                
                positions = []
                for row in rows:
                    position_data = dict(zip(columns, row))
                    
                    # 不要なカラムを除去
                    position_data.pop('created_at', None)
                    position_data.pop('updated_at', None)
                    
                    position = Position(**position_data)
                    positions.append(position)
                
                return positions
                
        except Exception as e:
            self.logger.error(f"ポジション取得エラー: {e}")
            return []
    
    def update_current_prices(self, price_data: Dict[str, float]):
        """現在価格の更新"""
        try:
            positions = self.get_positions(status="OPEN")
            
            for position in positions:
                if position.symbol in price_data:
                    position.current_price = price_data[position.symbol]
                    
                    # 価格履歴に記録
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO price_history (symbol, price)
                            VALUES (?, ?)
                        ''', (position.symbol, price_data[position.symbol]))
                        conn.commit()
                        
        except Exception as e:
            self.logger.error(f"価格更新エラー: {e}")
    
    def get_portfolio_summary(self) -> PortfolioSummary:
        """ポートフォリオサマリー取得"""
        try:
            all_positions = self.get_positions()
            open_positions = [p for p in all_positions if p.status == "OPEN"]
            closed_positions = [p for p in all_positions if p.status == "CLOSED"]
            
            # 基本統計
            total_positions = len(all_positions)
            open_count = len(open_positions)
            closed_count = len(closed_positions)
            
            # 投資額計算
            total_investment = sum(p.entry_price * p.quantity for p in open_positions)
            
            # 現在価値計算（オープンポジションのみ）
            current_value = 0.0
            total_unrealized_pnl = 0.0
            for position in open_positions:
                if position.current_price:
                    if position.position_type == "LONG":
                        current_value += position.current_price * position.quantity
                    else:  # SHORT
                        current_value += position.entry_price * position.quantity * 2 - position.current_price * position.quantity
                    total_unrealized_pnl += position.unrealized_pnl
                else:
                    current_value += position.entry_price * position.quantity
            
            # 実現損益計算
            total_realized_pnl = sum(p.realized_pnl for p in closed_positions)
            
            # 総損益
            total_pnl = total_unrealized_pnl + total_realized_pnl
            total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0.0
            
            # 勝率計算
            profitable_trades = len([p for p in closed_positions if p.realized_pnl > 0])
            win_rate = (profitable_trades / closed_count * 100) if closed_count > 0 else 0.0
            
            # 平均保有日数
            avg_holding_days = (sum(p.holding_days for p in closed_positions) / closed_count) if closed_count > 0 else 0.0
            
            # ベスト・ワーストポジション
            best_position = None
            worst_position = None
            if closed_positions:
                best_pos = max(closed_positions, key=lambda p: p.realized_pnl)
                worst_pos = min(closed_positions, key=lambda p: p.realized_pnl)
                best_position = f"{best_pos.symbol} (+{best_pos.realized_pnl:,.0f}円)"
                worst_position = f"{worst_pos.symbol} ({worst_pos.realized_pnl:,.0f}円)"
            
            return PortfolioSummary(
                total_positions=total_positions,
                open_positions=open_count,
                closed_positions=closed_count,
                total_investment=total_investment,
                current_value=current_value,
                total_unrealized_pnl=total_unrealized_pnl,
                total_realized_pnl=total_realized_pnl,
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl_pct,
                win_rate=win_rate,
                average_holding_days=avg_holding_days,
                best_position=best_position,
                worst_position=worst_position
            )
            
        except Exception as e:
            self.logger.error(f"ポートフォリオサマリー取得エラー: {e}")
            # エラー時のデフォルト値
            return PortfolioSummary(
                total_positions=0, open_positions=0, closed_positions=0,
                total_investment=0.0, current_value=0.0,
                total_unrealized_pnl=0.0, total_realized_pnl=0.0,
                total_pnl=0.0, total_pnl_pct=0.0,
                win_rate=0.0, average_holding_days=0.0
            )
    
    def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """特定ポジション取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM positions WHERE id = ?', (position_id,))
                row = cursor.fetchone()
                
                if row:
                    columns = [description[0] for description in cursor.description]
                    position_data = dict(zip(columns, row))
                    
                    # 不要なカラムを除去
                    position_data.pop('created_at', None)
                    position_data.pop('updated_at', None)
                    
                    return Position(**position_data)
                
                return None
                
        except Exception as e:
            self.logger.error(f"ポジション取得エラー: {e}")
            return None
    
    def get_performance_by_strategy(self) -> Dict[str, Dict]:
        """戦略別パフォーマンス分析"""
        try:
            positions = self.get_positions(status="CLOSED")
            strategy_performance = {}
            
            for position in positions:
                strategy = position.strategy_type
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {
                        'total_trades': 0,
                        'profitable_trades': 0,
                        'total_pnl': 0.0,
                        'avg_holding_days': 0.0,
                        'win_rate': 0.0
                    }
                
                perf = strategy_performance[strategy]
                perf['total_trades'] += 1
                perf['total_pnl'] += position.realized_pnl
                perf['avg_holding_days'] += position.holding_days
                
                if position.realized_pnl > 0:
                    perf['profitable_trades'] += 1
            
            # 平均値とパーセンテージの計算
            for strategy in strategy_performance:
                perf = strategy_performance[strategy]
                if perf['total_trades'] > 0:
                    perf['avg_holding_days'] /= perf['total_trades']
                    perf['win_rate'] = (perf['profitable_trades'] / perf['total_trades']) * 100
            
            return strategy_performance
            
        except Exception as e:
            self.logger.error(f"戦略別パフォーマンス分析エラー: {e}")
            return {}