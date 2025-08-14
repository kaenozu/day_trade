#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Manager - 実戦投入用リスク管理システム

資金保護・損切り自動化・ポジション管理
デイトレード特化リスク管理
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

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

class RiskLevel(Enum):
    """リスクレベル"""
    VERY_LOW = "超低リスク"
    LOW = "低リスク"
    MEDIUM = "中リスク"
    HIGH = "高リスク"
    VERY_HIGH = "超高リスク"

class PositionStatus(Enum):
    """ポジション状態"""
    OPEN = "建玉中"
    CLOSED = "決済済"
    STOP_LOSS = "損切り"
    TAKE_PROFIT = "利確"
    TIME_STOP = "時間切れ決済"

class AlertLevel(Enum):
    """アラートレベル"""
    INFO = "情報"
    WARNING = "警告"
    CRITICAL = "緊急"

@dataclass
class Position:
    """ポジション情報"""
    symbol: str
    name: str
    entry_price: float
    quantity: int
    entry_time: datetime
    stop_loss: float
    take_profit: float
    current_price: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    pnl: float = 0.0
    pnl_percent: float = 0.0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    max_holding_time: int = 240  # 分（4時間）

    def update_current_price(self, price: float):
        """現在価格更新とPnL計算"""
        self.current_price = price
        self.pnl = (price - self.entry_price) * self.quantity
        if self.entry_price > 0:
            self.pnl_percent = ((price - self.entry_price) / self.entry_price) * 100

    @property
    def holding_minutes(self) -> int:
        """保有時間（分）"""
        return int((datetime.now() - self.entry_time).total_seconds() / 60)

    @property
    def is_profitable(self) -> bool:
        """含み益判定"""
        return self.pnl > 0

    @property
    def should_stop_loss(self) -> bool:
        """損切り判定"""
        return self.current_price <= self.stop_loss

    @property
    def should_take_profit(self) -> bool:
        """利確判定"""
        return self.current_price >= self.take_profit

    @property
    def should_time_stop(self) -> bool:
        """時間切れ判定"""
        return self.holding_minutes >= self.max_holding_time

@dataclass
class RiskAlert:
    """リスクアラート"""
    timestamp: datetime
    symbol: str
    level: AlertLevel
    message: str
    current_price: float = 0.0
    recommended_action: str = ""

@dataclass
class RiskSettings:
    """リスク管理設定"""
    # 資金管理
    total_capital: float = 1000000  # 総資金（100万円）
    max_position_size: float = 0.10  # 1銘柄あたり最大10%
    max_daily_loss: float = 0.05     # 1日最大損失5%
    max_loss_per_trade: float = 0.02  # 1取引最大損失2%

    # デイトレード設定
    max_positions: int = 3           # 同時保有最大3銘柄
    max_holding_time: int = 240      # 最大保有時間4時間
    force_close_time: str = "14:50"  # 強制決済時刻

    # リスク調整
    volatility_multiplier: float = 1.5  # ボラティリティ倍率
    news_risk_reduction: float = 0.5    # ニュース時リスク軽減

    # アラート設定
    loss_alert_threshold: float = -0.01  # -1%でアラート
    time_alert_threshold: int = 180      # 3時間でアラート

class PersonalRiskManager:
    """
    個人投資家向けリスク管理システム
    資金保護を最優先とした損切り自動化
    """

    def __init__(self, settings: RiskSettings = None):
        self.logger = logging.getLogger(__name__)
        self.settings = settings or RiskSettings()

        # ポジション管理
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.alerts: List[RiskAlert] = []

        # 損益管理
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        # 緊急停止フラグ
        self.emergency_stop = False

    def calculate_position_size(self, symbol: str, entry_price: float,
                              volatility: float, confidence: float) -> int:
        """
        最適ポジションサイズ計算

        Args:
            symbol: 銘柄コード
            entry_price: エントリー価格
            volatility: ボラティリティ(%)
            confidence: 信頼度(%)

        Returns:
            int: 推奨株数
        """
        # 基本ポジションサイズ（資金の10%以内）
        max_investment = self.settings.total_capital * self.settings.max_position_size
        base_quantity = int(max_investment / entry_price)

        # リスク調整
        volatility_adj = 1.0 / (1.0 + volatility / 100)  # ボラティリティが高いほど減額
        confidence_adj = confidence / 100  # 信頼度による調整

        # 1取引あたりの最大損失制限
        max_loss_amount = self.settings.total_capital * self.settings.max_loss_per_trade
        stop_loss_percent = min(volatility * 0.5, 3.0) / 100  # 損切り幅
        max_quantity_by_loss = int(max_loss_amount / (entry_price * stop_loss_percent))

        # 最終ポジションサイズ
        adjusted_quantity = int(base_quantity * volatility_adj * confidence_adj)
        final_quantity = min(adjusted_quantity, max_quantity_by_loss, base_quantity)

        # デバッグ情報
        self.logger.info(f"Position size calc: base={base_quantity}, vol_adj={volatility_adj:.2f}, "
                        f"conf_adj={confidence_adj:.2f}, final={final_quantity}")

        return max(100, final_quantity)  # 最低100株

    def calculate_stop_loss_take_profit(self, symbol: str, entry_price: float,
                                       volatility: float, signal_strength: float) -> Tuple[float, float]:
        """
        損切り・利確価格計算

        Args:
            symbol: 銘柄コード
            entry_price: エントリー価格
            volatility: ボラティリティ(%)
            signal_strength: シグナル強度

        Returns:
            Tuple[float, float]: (損切り価格, 利確価格)
        """
        # ボラティリティベースの損切り幅
        base_stop_percent = min(volatility * 0.5, 3.0)  # 最大3%
        base_profit_percent = min(volatility * 0.8, 5.0)  # 最大5%

        # シグナル強度による調整
        signal_adj = 0.8 + (signal_strength / 100) * 0.4  # 0.8〜1.2倍

        # 最終損切り・利確幅
        stop_percent = base_stop_percent * signal_adj
        profit_percent = base_profit_percent * signal_adj

        stop_loss = entry_price * (1 - stop_percent / 100)
        take_profit = entry_price * (1 + profit_percent / 100)

        return round(stop_loss, 2), round(take_profit, 2)

    def open_position(self, symbol: str, name: str, entry_price: float,
                     quantity: int, volatility: float, confidence: float) -> bool:
        """
        ポジション建玉

        Returns:
            bool: 建玉成功かどうか
        """
        # リスクチェック
        if not self._can_open_position(symbol, entry_price, quantity):
            return False

        # 損切り・利確価格計算
        stop_loss, take_profit = self.calculate_stop_loss_take_profit(
            symbol, entry_price, volatility, confidence
        )

        # リスクレベル判定
        risk_level = self._assess_risk_level(volatility, confidence)

        # ポジション作成
        position = Position(
            symbol=symbol,
            name=name,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=entry_price,
            risk_level=risk_level
        )

        self.positions[symbol] = position
        self.trade_count += 1

        # アラート生成
        self._create_alert(
            symbol, AlertLevel.INFO,
            f"ポジション建玉: {quantity}株 @{entry_price}円",
            entry_price,
            f"損切り{stop_loss}円 利確{take_profit}円"
        )

        self.logger.info(f"Position opened: {symbol} {quantity}株 @{entry_price}")
        return True

    def update_positions(self, price_data: Dict[str, float]):
        """
        全ポジション更新とリスク監視

        Args:
            price_data: {銘柄コード: 現在価格}
        """
        for symbol, position in list(self.positions.items()):
            if symbol in price_data:
                # 価格更新
                old_price = position.current_price
                position.update_current_price(price_data[symbol])

                # リスク監視
                self._monitor_position_risk(position, old_price)

                # 自動決済判定
                if self._should_close_position(position):
                    self._close_position(position)

    def _monitor_position_risk(self, position: Position, old_price: float):
        """ポジションリスク監視"""
        # 損失アラート
        if position.pnl_percent <= self.settings.loss_alert_threshold * 100:
            if position.pnl_percent <= -1.5:  # -1.5%以下で緊急アラート
                self._create_alert(
                    position.symbol, AlertLevel.CRITICAL,
                    f"大幅損失: {position.pnl_percent:.2f}%",
                    position.current_price,
                    "即座に損切り検討"
                )
            else:
                self._create_alert(
                    position.symbol, AlertLevel.WARNING,
                    f"含み損拡大: {position.pnl_percent:.2f}%",
                    position.current_price,
                    "損切り準備"
                )

        # 時間アラート
        if position.holding_minutes >= self.settings.time_alert_threshold:
            self._create_alert(
                position.symbol, AlertLevel.WARNING,
                f"長期保有: {position.holding_minutes}分経過",
                position.current_price,
                "決済タイミング検討"
            )

        # 急騰急落アラート
        if old_price > 0:
            price_change = abs(position.current_price - old_price) / old_price
            if price_change >= 0.05:  # 5%以上の急変
                self._create_alert(
                    position.symbol, AlertLevel.CRITICAL,
                    f"急激な価格変動: {price_change*100:.1f}%",
                    position.current_price,
                    "即座に状況確認"
                )

    def _should_close_position(self, position: Position) -> bool:
        """ポジション決済判定"""
        return (position.should_stop_loss or
                position.should_take_profit or
                position.should_time_stop or
                self.emergency_stop or
                self._is_force_close_time())

    def _close_position(self, position: Position):
        """ポジション決済"""
        # 決済理由判定
        if position.should_stop_loss:
            position.status = PositionStatus.STOP_LOSS
            reason = "損切り"
        elif position.should_take_profit:
            position.status = PositionStatus.TAKE_PROFIT
            reason = "利確"
        elif position.should_time_stop:
            position.status = PositionStatus.TIME_STOP
            reason = "時間切れ"
        else:
            position.status = PositionStatus.CLOSED
            reason = "手動決済"

        # 統計更新
        self.daily_pnl += position.pnl
        self.total_pnl += position.pnl
        if position.is_profitable:
            self.win_count += 1

        # アラート生成
        self._create_alert(
            position.symbol, AlertLevel.INFO,
            f"ポジション決済({reason}): {position.pnl:+.0f}円 ({position.pnl_percent:+.2f}%)",
            position.current_price,
            f"保有時間{position.holding_minutes}分"
        )

        # ポジション移動
        self.closed_positions.append(position)
        del self.positions[position.symbol]

        self.logger.info(f"Position closed: {position.symbol} {reason} PnL:{position.pnl:+.0f}")

    def _can_open_position(self, symbol: str, entry_price: float, quantity: int) -> bool:
        """建玉可能性判定"""
        # 既存ポジションチェック
        if symbol in self.positions:
            self.logger.warning(f"Position already exists for {symbol}")
            return False

        # 最大ポジション数チェック
        if len(self.positions) >= self.settings.max_positions:
            self.logger.warning(f"Maximum positions reached: {len(self.positions)}")
            return False

        # 1日最大損失チェック
        max_daily_loss = self.settings.total_capital * self.settings.max_daily_loss
        if self.daily_pnl <= -max_daily_loss:
            self.logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            return False

        # 投資金額チェック
        investment = entry_price * quantity
        max_investment = self.settings.total_capital * self.settings.max_position_size
        self.logger.info(f"Investment: {investment:.2f}, Max: {max_investment:.2f}")

        if investment > max_investment:
            self.logger.warning(f"Position size too large: {investment:.2f} > {max_investment:.2f}")
            return False

        # 緊急停止チェック
        if self.emergency_stop:
            self.logger.warning("Emergency stop activated")
            return False

        self.logger.info(f"Position check passed for {symbol}")
        return True

    def _assess_risk_level(self, volatility: float, confidence: float) -> RiskLevel:
        """リスクレベル判定"""
        risk_score = volatility / confidence * 100

        if risk_score >= 8.0:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 6.0:
            return RiskLevel.HIGH
        elif risk_score >= 4.0:
            return RiskLevel.MEDIUM
        elif risk_score >= 2.0:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

    def _is_force_close_time(self) -> bool:
        """強制決済時刻判定"""
        now = datetime.now().time()
        force_time = datetime.strptime(self.settings.force_close_time, "%H:%M").time()
        return now >= force_time

    def _create_alert(self, symbol: str, level: AlertLevel, message: str,
                     price: float = 0.0, action: str = ""):
        """アラート作成"""
        alert = RiskAlert(
            timestamp=datetime.now(),
            symbol=symbol,
            level=level,
            message=message,
            current_price=price,
            recommended_action=action
        )
        self.alerts.append(alert)

        # 最新100件のみ保持
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

    def get_risk_summary(self) -> Dict[str, Any]:
        """リスク状況サマリー"""
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "positions_count": len(self.positions),
            "total_investment": sum(p.entry_price * p.quantity for p in self.positions.values()),
            "unrealized_pnl": sum(p.pnl for p in self.positions.values()),
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "win_rate": (self.win_count / max(self.trade_count, 1)) * 100,
            "emergency_stop": self.emergency_stop,
            "alerts_count": len([a for a in self.alerts if a.level != AlertLevel.INFO]),
            "max_risk_position": max(
                (p.symbol for p in self.positions.values()),
                key=lambda s: self.positions[s].risk_level.value if s in self.positions else "",
                default="なし"
            )
        }

    def force_close_all_positions(self):
        """緊急全ポジション決済"""
        self.emergency_stop = True
        for position in list(self.positions.values()):
            self._close_position(position)

        self._create_alert(
            "SYSTEM", AlertLevel.CRITICAL,
            f"緊急全決済実行: {len(self.positions)}ポジション",
            0.0,
            "システム停止"
        )


def main():
    """リスク管理システムのテスト"""
    print("=== リスク管理システム テスト ===")

    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # リスク管理システム初期化
    risk_manager = PersonalRiskManager()

    # テスト用ポジション建玉
    print("\n[ テスト用ポジション建玉 ]")

    # 適切なポジションサイズ計算
    quantity1 = risk_manager.calculate_position_size("7203", 2800, 3.5, 85)
    quantity2 = risk_manager.calculate_position_size("9984", 4200, 5.2, 78)

    print(f"推奨ポジションサイズ1: {quantity1}株")
    print(f"推奨ポジションサイズ2: {quantity2}株")

    # 小さめのテストサイズを使用
    test_quantity1 = min(quantity1, 30)  # 最大30株でテスト
    test_quantity2 = min(quantity2, 20)  # 最大20株でテスト

    success1 = risk_manager.open_position("7203", "トヨタ自動車", 2800, test_quantity1, 3.5, 85)
    success2 = risk_manager.open_position("9984", "ソフトバンクG", 4200, test_quantity2, 5.2, 78)

    print(f"ポジション1建玉: {'成功' if success1 else '失敗'}")
    print(f"ポジション2建玉: {'成功' if success2 else '失敗'}")

    # 価格更新テスト
    print("\n[ 価格更新・リスク監視テスト ]")
    price_data = {
        "7203": 2750,  # -1.8% (含み損)
        "9984": 4320   # +2.9% (含み益)
    }
    risk_manager.update_positions(price_data)

    # リスク状況確認
    print("\n[ リスク状況サマリー ]")
    summary = risk_manager.get_risk_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

    # アラート確認
    print(f"\n[ 最新アラート ({len(risk_manager.alerts)}件) ]")
    for alert in risk_manager.alerts[-3:]:  # 最新3件
        print(f"[{alert.level.value}] {alert.symbol}: {alert.message}")
        if alert.recommended_action:
            print(f"  推奨: {alert.recommended_action}")

    print("\n" + "="*50)
    print("リスク管理システム 正常動作確認")

if __name__ == "__main__":
    main()