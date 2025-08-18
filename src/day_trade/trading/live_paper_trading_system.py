#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Paper Trading System - ライブペーパートレーディングシステム

Issue #798実装：ライブ環境での実地テスト
リアルマネーを使わずに実際の市場でテスト
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path

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

class OrderType(Enum):
    """注文タイプ"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """注文状態"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"

@dataclass
class PaperOrder:
    """ペーパー注文"""
    id: str
    symbol: str
    order_type: OrderType
    quantity: int
    target_price: float
    current_price: float
    created_at: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    profit_loss: float = 0.0

@dataclass
class PaperPosition:
    """ペーパーポジション"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

@dataclass
class TradingSignal:
    """取引シグナル"""
    symbol: str
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float
    predicted_price: Optional[float]
    risk_score: float
    timestamp: datetime
    reasons: List[str] = field(default_factory=list)

class LivePaperTradingSystem:
    """ライブペーパートレーディングシステム"""

    def __init__(self, initial_capital: float = 1000000.0):
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_cash = initial_capital

        # データベース設定
        self.db_path = Path("trading_data/paper_trading.db")
        self.db_path.parent.mkdir(exist_ok=True)

        # ポートフォリオ管理
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: Dict[str, PaperOrder] = {}
        self.order_history: List[PaperOrder] = []

        # 取引設定
        self.max_position_size = 0.1  # 最大10%のポジション
        self.stop_loss_threshold = -0.05  # -5%でストップロス
        self.take_profit_threshold = 0.1  # +10%で利確

        # 取引対象銘柄
        self.target_symbols = [
            "7203",  # トヨタ自動車
            "8306",  # 三菱UFJ
            "4751",  # サイバーエージェント
            "6861",  # キーエンス
            "9984",  # ソフトバンクグループ
        ]

        # パフォーマンス追跡
        self.daily_pnl_history = []
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0

        self._init_database()
        self.logger.info(f"Paper trading system initialized with {initial_capital:,}円")

    def _init_database(self):
        """データベース初期化"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 注文履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS paper_orders (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        order_type TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        target_price REAL NOT NULL,
                        current_price REAL NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        filled_at TEXT,
                        filled_price REAL,
                        profit_loss REAL DEFAULT 0.0
                    )
                ''')

                # ポジション履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS paper_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        entry_time TEXT NOT NULL,
                        exit_time TEXT,
                        realized_pnl REAL DEFAULT 0.0,
                        hold_duration INTEGER DEFAULT 0
                    )
                ''')

                # パフォーマンス履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS paper_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        total_capital REAL NOT NULL,
                        available_cash REAL NOT NULL,
                        portfolio_value REAL NOT NULL,
                        daily_pnl REAL NOT NULL,
                        total_return REAL NOT NULL,
                        trade_count INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        UNIQUE(date)
                    )
                ''')

                conn.commit()

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")

    async def generate_trading_signals(self) -> List[TradingSignal]:
        """取引シグナル生成"""

        signals = []

        for symbol in self.target_symbols:
            try:
                # 最適化予測システムで予測
                from optimized_prediction_system import optimized_prediction_system
                prediction = await optimized_prediction_system.predict_with_optimized_models(symbol)

                # 現在価格取得
                from real_data_provider_v2 import real_data_provider
                current_data = await real_data_provider.get_latest_stock_price(symbol)
                current_price = current_data.get('current_price', 0) if current_data else 0

                if current_price <= 0:
                    continue

                # リスク管理システムでリスク評価
                from advanced_risk_management_system import advanced_risk_management_system
                risk_assessment = await advanced_risk_management_system.calculate_position_risk(
                    symbol, current_price, self.current_capital * self.max_position_size
                )

                # シグナル生成
                signal_type = "BUY" if prediction.prediction == 1 else "SELL"
                risk_score = risk_assessment.get('risk_score', 50) if risk_assessment else 50

                # ポジション管理を考慮したシグナル調整
                current_position = self.positions.get(symbol)
                reasons = []

                if signal_type == "BUY":
                    if current_position and current_position.quantity > 0:
                        signal_type = "HOLD"
                        reasons.append("既存ロングポジション保持中")
                    elif risk_score > 70:
                        signal_type = "HOLD"
                        reasons.append(f"高リスク (リスクスコア: {risk_score})")
                    elif prediction.confidence < 0.6:
                        signal_type = "HOLD"
                        reasons.append(f"信頼度不足 (信頼度: {prediction.confidence:.3f})")
                    else:
                        reasons.append(f"上昇予測 (信頼度: {prediction.confidence:.3f})")

                elif signal_type == "SELL":
                    if not current_position or current_position.quantity <= 0:
                        signal_type = "HOLD"
                        reasons.append("ポジションなし")
                    elif prediction.confidence < 0.6:
                        signal_type = "HOLD"
                        reasons.append(f"信頼度不足 (信頼度: {prediction.confidence:.3f})")
                    else:
                        reasons.append(f"下降予測 (信頼度: {prediction.confidence:.3f})")

                signal = TradingSignal(
                    symbol=symbol,
                    signal=signal_type,
                    confidence=prediction.confidence,
                    predicted_price=None,  # 予測価格は実装可能
                    risk_score=risk_score,
                    timestamp=datetime.now(),
                    reasons=reasons
                )

                signals.append(signal)

            except Exception as e:
                self.logger.warning(f"シグナル生成エラー {symbol}: {e}")
                continue

        return signals

    async def execute_paper_order(self, symbol: str, order_type: OrderType,
                                quantity: int, target_price: float) -> Optional[PaperOrder]:
        """ペーパー注文実行"""

        try:
            # 現在価格取得
            from real_data_provider_v2 import real_data_provider
            current_data = await real_data_provider.get_latest_stock_price(symbol)
            current_price = current_data.get('current_price', target_price) if current_data else target_price

            # 注文ID生成
            order_id = f"{symbol}_{order_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 資金チェック
            if order_type == OrderType.BUY:
                required_capital = quantity * current_price
                if required_capital > self.available_cash:
                    self.logger.warning(f"資金不足: 必要{required_capital:,}円 > 利用可能{self.available_cash:,}円")
                    return None

            # ペーパー注文作成
            order = PaperOrder(
                id=order_id,
                symbol=symbol,
                order_type=order_type,
                quantity=quantity,
                target_price=target_price,
                current_price=current_price,
                created_at=datetime.now(),
                status=OrderStatus.PENDING
            )

            # 即座に約定（実際の取引では注文執行時間がある）
            filled_price = current_price * (1 + np.random.uniform(-0.001, 0.001))  # 1%以内のスリッページシミュレート

            order.filled_at = datetime.now()
            order.filled_price = filled_price
            order.status = OrderStatus.FILLED

            # ポジション更新
            if order_type == OrderType.BUY:
                self._update_position_buy(symbol, quantity, filled_price)
                self.available_cash -= quantity * filled_price
            else:
                self._update_position_sell(symbol, quantity, filled_price)
                self.available_cash += quantity * filled_price

            # 注文履歴に追加
            self.orders[order_id] = order
            self.order_history.append(order)

            # データベースに保存
            await self._save_order_to_db(order)

            self.trade_count += 1
            self.logger.info(f"注文実行: {symbol} {order_type.value} {quantity}株 @{filled_price:.0f}円")

            return order

        except Exception as e:
            self.logger.error(f"注文実行エラー: {e}")
            return None

    def _update_position_buy(self, symbol: str, quantity: int, price: float):
        """買いポジション更新"""

        if symbol in self.positions:
            # 既存ポジションに追加
            position = self.positions[symbol]
            total_quantity = position.quantity + quantity
            total_cost = position.quantity * position.entry_price + quantity * price
            new_entry_price = total_cost / total_quantity

            position.quantity = total_quantity
            position.entry_price = new_entry_price
        else:
            # 新規ポジション
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                entry_time=datetime.now()
            )

    def _update_position_sell(self, symbol: str, quantity: int, price: float):
        """売りポジション更新"""

        if symbol in self.positions:
            position = self.positions[symbol]

            if position.quantity >= quantity:
                # 一部または全部売却
                realized_pnl = quantity * (price - position.entry_price)
                position.realized_pnl += realized_pnl
                position.quantity -= quantity

                # PnL記録
                if realized_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                # ポジションがゼロになったら削除
                if position.quantity <= 0:
                    del self.positions[symbol]
            else:
                self.logger.warning(f"売却数量が保有数量を超過: {symbol} 保有{position.quantity} < 売却{quantity}")

    async def _save_order_to_db(self, order: PaperOrder):
        """注文をデータベースに保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO paper_orders
                    (id, symbol, order_type, quantity, target_price, current_price,
                     status, created_at, filled_at, filled_price, profit_loss)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    order.id,
                    order.symbol,
                    order.order_type.value,
                    order.quantity,
                    order.target_price,
                    order.current_price,
                    order.status.value,
                    order.created_at.isoformat(),
                    order.filled_at.isoformat() if order.filled_at else None,
                    order.filled_price,
                    order.profit_loss
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"注文保存エラー: {e}")

    async def update_portfolio_prices(self):
        """ポートフォリオ価格更新"""

        for symbol, position in self.positions.items():
            try:
                # 最新価格取得
                from real_data_provider_v2 import real_data_provider
                current_data = await real_data_provider.get_latest_stock_price(symbol)
                current_price = current_data.get('current_price', position.current_price) if current_data else position.current_price

                # ポジション更新
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = position.quantity * (current_price - position.entry_price)

                # ストップロス・テイクプロフィットチェック
                pnl_ratio = position.unrealized_pnl / (position.quantity * position.entry_price)

                if pnl_ratio <= self.stop_loss_threshold:
                    # ストップロス実行
                    await self.execute_paper_order(symbol, OrderType.SELL, position.quantity, current_price)
                    self.logger.info(f"ストップロス実行: {symbol} 損失{pnl_ratio:.1%}")

                elif pnl_ratio >= self.take_profit_threshold:
                    # テイクプロフィット実行
                    await self.execute_paper_order(symbol, OrderType.SELL, position.quantity, current_price)
                    self.logger.info(f"利確実行: {symbol} 利益{pnl_ratio:.1%}")

            except Exception as e:
                self.logger.warning(f"価格更新エラー {symbol}: {e}")

    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """ポートフォリオメトリクス計算"""

        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())

        total_capital = self.available_cash + total_market_value
        total_return = (total_capital - self.initial_capital) / self.initial_capital

        win_rate = self.winning_trades / max(1, self.winning_trades + self.losing_trades)

        return {
            "total_capital": total_capital,
            "available_cash": self.available_cash,
            "portfolio_value": total_market_value,
            "unrealized_pnl": total_unrealized_pnl,
            "realized_pnl": total_realized_pnl,
            "total_return": total_return,
            "win_rate": win_rate,
            "trade_count": self.trade_count
        }

    async def run_paper_trading_session(self, duration_hours: int = 8):
        """ペーパートレーディングセッション実行"""

        session_start = datetime.now()
        session_end = session_start + timedelta(hours=duration_hours)

        print(f"=== 🚀 ライブペーパートレーディング開始 ===")
        print(f"開始資本: {self.initial_capital:,.0f}円")
        print(f"セッション時間: {duration_hours}時間")
        print(f"対象銘柄: {', '.join(self.target_symbols)}")

        session_count = 0

        while datetime.now() < session_end:
            session_count += 1
            cycle_start = datetime.now()

            print(f"\n--- サイクル #{session_count} ({cycle_start.strftime('%H:%M:%S')}) ---")

            try:
                # 1. ポートフォリオ価格更新
                await self.update_portfolio_prices()

                # 2. トレーディングシグナル生成
                signals = await self.generate_trading_signals()

                # 3. シグナルに基づく取引実行
                for signal in signals:
                    if signal.signal == "BUY" and signal.confidence > 0.6:
                        # 買いシグナル
                        position_size = int(self.current_capital * self.max_position_size / signal.predicted_price) if signal.predicted_price else 100
                        await self.execute_paper_order(signal.symbol, OrderType.BUY, position_size, 0)

                    elif signal.signal == "SELL" and signal.confidence > 0.6:
                        # 売りシグナル
                        if signal.symbol in self.positions:
                            position = self.positions[signal.symbol]
                            await self.execute_paper_order(signal.symbol, OrderType.SELL, position.quantity, 0)

                # 4. パフォーマンス表示
                metrics = self.calculate_portfolio_metrics()

                print(f"  総資本: {metrics['total_capital']:,.0f}円 (リターン: {metrics['total_return']:.2%})")
                print(f"  現金: {metrics['available_cash']:,.0f}円 | ポジション: {metrics['portfolio_value']:,.0f}円")
                print(f"  未実現PnL: {metrics['unrealized_pnl']:,.0f}円 | 実現PnL: {metrics['realized_pnl']:,.0f}円")
                print(f"  取引回数: {metrics['trade_count']} | 勝率: {metrics['win_rate']:.1%}")

                # シグナル表示
                active_signals = [s for s in signals if s.signal != "HOLD"]
                if active_signals:
                    print(f"  アクティブシグナル: {len(active_signals)}件")
                    for signal in active_signals[:3]:  # 上位3件表示
                        print(f"    {signal.symbol}: {signal.signal} (信頼度: {signal.confidence:.3f})")

                # 5. 待機（実際の取引間隔をシミュレート）
                await asyncio.sleep(300)  # 5分間隔

            except Exception as e:
                self.logger.error(f"トレーディングサイクルエラー: {e}")
                await asyncio.sleep(60)  # エラー時は1分待機
                continue

        # セッション終了
        final_metrics = self.calculate_portfolio_metrics()

        print(f"\n=== 📊 セッション終了レポート ===")
        print(f"初期資本: {self.initial_capital:,.0f}円")
        print(f"最終資本: {final_metrics['total_capital']:,.0f}円")
        print(f"総リターン: {final_metrics['total_return']:.2%}")
        print(f"実現損益: {final_metrics['realized_pnl']:,.0f}円")
        print(f"未実現損益: {final_metrics['unrealized_pnl']:,.0f}円")
        print(f"取引回数: {final_metrics['trade_count']}")
        print(f"勝率: {final_metrics['win_rate']:.1%}")

        return final_metrics

# グローバルインスタンス
live_paper_trading_system = LivePaperTradingSystem()

# テスト実行
async def run_paper_trading_test():
    """ペーパートレーディングテスト実行"""

    # 短時間テスト（30分）
    final_metrics = await live_paper_trading_system.run_paper_trading_session(duration_hours=0.5)

    return final_metrics

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(run_paper_trading_test())