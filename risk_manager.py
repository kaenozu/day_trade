#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Manager - Risk management system for practical use

Fund protection, automatic loss cutting, position management
Day trade specialized risk management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml

from src.day_trade.database_manager import DatabaseManager
from src.day_trade.data_models import Position, PositionStatus, RiskLevel, AlertLevel, RiskAlert
from src.day_trade.data_provider import AbstractDataProvider, ManualDataProvider
from src.day_trade.utils.encoding_fix import apply_windows_encoding_fix

# Apply Windows encoding fix
apply_windows_encoding_fix()

@dataclass
class RiskSettings:
    """Risk management settings"""
    # Fund management
    total_capital: float
    max_position_size: float
    max_daily_loss: float
    max_loss_per_trade: float

    # Day trade settings
    max_positions: int
    max_holding_time: int
    force_close_time: str

    # Risk adjustment
    volatility_multiplier: float
    news_risk_reduction: float

    # Alert settings
    loss_alert_threshold: float
    time_alert_threshold: int

class PersonalRiskManager:
    """
    Risk management system for individual investors
    Automated loss cutting with top priority on fund protection
    """

    def __init__(self, data_provider: AbstractDataProvider, settings_profile: str = None, settings_file: str = 'config/risk_settings.yaml', db_path: str = 'data/daytrade.db'):
        self.logger = logging.getLogger(__name__)
        self.data_provider = data_provider
        self.settings = self._load_settings(settings_profile, settings_file)

        # Database management
        self.db_manager = DatabaseManager(db_path)
        self.db_manager.connect()

        # Position management
        self.positions: Dict[str, Position] = {p.symbol: p for p in self.db_manager.load_open_positions()}
        self.closed_positions: List[Position] = []
        self.alerts: List[RiskAlert] = []

        # PnL management
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        # Emergency stop flag
        self.emergency_stop = False

    def _load_settings(self, profile: str = None, file_path: str = 'config/risk_settings.yaml') -> RiskSettings:
        """Load settings file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if profile is None:
                profile = config.get('default_profile', 'Moderate')

            settings_data = config['profiles'][profile]
            self.logger.info(f"Loaded risk settings profile '{profile}'")
            return RiskSettings(**settings_data)
        except (FileNotFoundError, KeyError) as e:
            self.logger.error(f"Failed to load risk settings: {e}")
            # Fallback to default settings
            return RiskSettings(
                total_capital=1000000, max_position_size=0.10, max_daily_loss=0.05,
                max_loss_per_trade=0.02, max_positions=3, max_holding_time=240,
                force_close_time="14:50", volatility_multiplier=1.5,
                news_risk_reduction=0.5, loss_alert_threshold=-0.01,
                time_alert_threshold=180
            )

    def calculate_position_size(self, symbol: str, entry_price: float,
                              volatility: float, confidence: float) -> int:
        """
        Calculate optimal position size

        Args:
            symbol: Symbol code
            entry_price: Entry price
            volatility: Volatility (%)
            confidence: Confidence (%)

        Returns:
            int: Recommended number of shares
        """
        # Base position size (within 10% of funds)
        max_investment = self.settings.total_capital * self.settings.max_position_size
        base_quantity = int(max_investment / entry_price)

        # Risk adjustment
        volatility_adj = 1.0 / (1.0 + volatility / 100)  # Reduce amount as volatility increases
        confidence_adj = confidence / 100  # Adjustment by confidence

        # Max loss limit per trade
        max_loss_amount = self.settings.total_capital * self.settings.max_loss_per_trade
        stop_loss_percent = min(volatility * 0.5, 3.0) / 100  # Stop loss range
        max_quantity_by_loss = int(max_loss_amount / (entry_price * stop_loss_percent))

        # Final position size
        adjusted_quantity = int(base_quantity * volatility_adj * confidence_adj)
        final_quantity = min(adjusted_quantity, max_quantity_by_loss, base_quantity)

        # Debug info
        self.logger.debug(f"Position size calc for {symbol}: base={base_quantity}, vol_adj={volatility_adj:.2f}, "
                         f"conf_adj={confidence_adj:.2f}, final={final_quantity}")

        return max(100, final_quantity)  # Minimum 100 shares

    def calculate_stop_loss_take_profit(self, symbol: str, entry_price: float,
                                       volatility: float, signal_strength: float,
                                       risk_tolerance: float = 1.0) -> Tuple[float, float]:
        """
        Calculate stop-loss and take-profit prices

        Args:
            symbol: Symbol code
            entry_price: Entry price
            volatility: Volatility (%)
            signal_strength: Signal strength
            risk_tolerance: Risk tolerance (0.5: low, 1.0: medium, 1.5: high)

        Returns:
            Tuple[float, float]: (Stop-loss price, Take-profit price)
        """
        # Volatility-based stop-loss range
        base_stop_percent = min(volatility * 0.5, 3.0)  # Max 3%
        base_profit_percent = min(volatility * 0.8, 5.0)  # Max 5%

        # Adjustment by signal strength
        signal_adj = 0.8 + (signal_strength / 100) * 0.4  # 0.8-1.2x

        # Adjustment by risk tolerance
        risk_tolerance = max(0.5, min(risk_tolerance, 1.5))

        # Final stop-loss and take-profit range
        stop_percent = base_stop_percent * signal_adj * risk_tolerance
        profit_percent = base_profit_percent * signal_adj / risk_tolerance

        stop_loss = entry_price * (1 - stop_percent / 100)
        take_profit = entry_price * (1 + profit_percent / 100)

        self.logger.debug(f"SL/TP calc for {symbol}: SL={stop_loss:.2f}, TP={take_profit:.2f}")
        return round(stop_loss, 2), round(take_profit, 2)

    def open_position(self, symbol: str, name: str, entry_price: float,
                     quantity: int, volatility: float, confidence: float) -> bool:
        """
        Open a position

        Returns:
            bool: Whether the position was successfully opened
        """
        # Risk check
        if not self._can_open_position(symbol, entry_price, quantity):
            return False

        # Calculate stop-loss and take-profit
        stop_loss, take_profit = self.calculate_stop_loss_take_profit(
            symbol, entry_price, volatility, confidence
        )

        # Assess risk level
        risk_level = self._assess_risk_level(symbol, volatility, confidence)

        # Create position
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
        self.db_manager.save_position(position)

        # Generate alert
        self._create_alert(
            symbol, AlertLevel.INFO,
            f"Position opened: {quantity} shares @{entry_price}",
            entry_price,
            f"Stop-loss {stop_loss} Take-profit {take_profit}"
        )

        self.logger.info(f"Position opened: {symbol} {quantity} shares @{entry_price}")
        return True

    def update_positions(self):
        """
        Update all positions and monitor risk
        """
        price_data = self.data_provider.get_latest_prices()
        self.logger.debug(f"Updating positions with price data: {price_data}")
        for symbol, position in list(self.positions.items()):
            if symbol in price_data:
                # Update price
                old_price = position.current_price
                position.update_current_price(price_data[symbol])
                self.db_manager.save_position(position) # Save price update to DB

                # Monitor risk
                self._monitor_position_risk(position, old_price)

                # Check for automatic closing
                if self._should_close_position(position):
                    self._close_position(position)

    def _monitor_position_risk(self, position: Position, old_price: float):
        """Monitor position risk"""
        # Loss alert
        if position.pnl_percent <= self.settings.loss_alert_threshold * 100:
            if position.pnl_percent <= -1.5:  # Emergency alert below -1.5%
                self._create_alert(
                    position.symbol, AlertLevel.CRITICAL,
                    f"Significant loss: {position.pnl_percent:.2f}%",
                    position.current_price,
                    "Consider immediate stop-loss"
                )
            else:
                self._create_alert(
                    position.symbol, AlertLevel.WARNING,
                    f"Unrealized loss increasing: {position.pnl_percent:.2f}%",
                    position.current_price,
                    "Prepare for stop-loss"
                )

        # Time alert
        if position.holding_minutes >= self.settings.time_alert_threshold:
            self._create_alert(
                position.symbol, AlertLevel.WARNING,
                f"Long holding time: {position.holding_minutes} minutes elapsed",
                position.current_price,
                "Consider closing position"
            )

        # Sudden price change alert
        if old_price > 0:
            price_change = abs(position.current_price - old_price) / old_price
            if price_change >= 0.05:  # Sudden change of 5% or more
                self._create_alert(
                    position.symbol, AlertLevel.CRITICAL,
                    f"Sudden price change: {price_change*100:.1f}%",
                    position.current_price,
                    "Check situation immediately"
                )

    def _should_close_position(self, position: Position) -> bool:
        """Check if position should be closed"""
        return (
                position.should_stop_loss or
                position.should_take_profit or
                position.should_time_stop or
                self.emergency_stop or
                self._is_force_close_time()
        )

    def _close_position(self, position: Position):
        """Close a position"""
        # Determine closing reason
        if position.should_stop_loss:
            position.status = PositionStatus.STOP_LOSS
            reason = "Stop-loss"
        elif position.should_take_profit:
            position.status = PositionStatus.TAKE_PROFIT
            reason = "Take-profit"
        elif position.should_time_stop:
            position.status = PositionStatus.TIME_STOP
            reason = "Time-stop"
        else:
            position.status = PositionStatus.CLOSED
            reason = "Manual close"

        # Update statistics
        self.daily_pnl += position.pnl
        self.total_pnl += position.pnl
        if position.is_profitable:
            self.win_count += 1

        # Generate alert
        self._create_alert(
            position.symbol, AlertLevel.INFO,
            f"Position closed ({reason}): {position.pnl:+.0f} ({position.pnl_percent:+.2f}%)",
            position.current_price,
            f"Holding time {position.holding_minutes} minutes"
        )

        # Move position
        self.closed_positions.append(position)
        self.db_manager.save_position(position)
        del self.positions[position.symbol]

        self.logger.info(f"Position closed: {position.symbol} {reason} PnL:{position.pnl:+.0f}")

    def _can_open_position(self, symbol: str, entry_price: float, quantity: int) -> bool:
        """Check if a position can be opened"""
        # Check for existing position
        if symbol in self.positions:
            self.logger.warning(f"Position already exists for {symbol}")
            return False

        # Check max positions
        if len(self.positions) >= self.settings.max_positions:
            self.logger.warning(f"Maximum positions reached: {len(self.positions)}")
            return False

        # Check max daily loss
        max_daily_loss = self.settings.total_capital * self.settings.max_daily_loss
        if self.daily_pnl <= -max_daily_loss:
            self.logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            self.emergency_stop = True # Emergency stop if loss limit is reached
            return False

        # Check investment amount
        investment = entry_price * quantity
        max_investment = self.settings.total_capital * self.settings.max_position_size
        self.logger.debug(f"Investment check for {symbol}: {investment:.2f}, Max: {max_investment:.2f}")

        if investment > max_investment:
            self.logger.warning(f"Position size too large for {symbol}: {investment:.2f} > {max_investment:.2f}")
            return False

        # Check emergency stop
        if self.emergency_stop:
            self.logger.warning("Emergency stop activated, cannot open new positions.")
            return False

        self.logger.debug(f"Position check passed for {symbol}")
        return True

    def _assess_risk_level(self, symbol: str, volatility: float, confidence: float) -> RiskLevel:
        """Assess risk level"""
        # New risk score calculation
        # Combines volatility (around 0-10) and lack of confidence (0-100)
        risk_score = (volatility * 0.6) + ((100 - confidence) * 0.4)
        self.logger.debug(f"Risk assessment for {symbol}: volatility={volatility}, confidence={confidence}, score={risk_score:.2f}")

        if risk_score >= 30:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 20:
            return RiskLevel.HIGH
        elif risk_score >= 10:
            return RiskLevel.MEDIUM
        elif risk_score >= 5:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

    def _is_force_close_time(self) -> bool:
        """Check for forced closing time"""
        now = datetime.now().time()
        force_time = datetime.strptime(self.settings.force_close_time, "%H:%M").time()
        return now >= force_time

    def _create_alert(self, symbol: str, level: AlertLevel, message: str,
                     price: float = 0.0, action: str = ""):
        """Create an alert"""
        alert = RiskAlert(
            timestamp=datetime.now(),
            symbol=symbol,
            level=level,
            message=message,
            current_price=price,
            recommended_action=action
        )
        self.alerts.append(alert)

        log_level_map = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        log_level = log_level_map.get(level, logging.INFO)
        self.logger.log(log_level, f"ALERT for {symbol}: {message}")

        # Keep only the latest 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary"""
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
                default="None"
            )
        }

    def force_close_all_positions(self):
        """Force close all positions"""
        self.logger.critical("FORCE CLOSING ALL POSITIONS!")
        self.emergency_stop = True
        for position in list(self.positions.values()):
            self._close_position(position)

        self._create_alert(
            "SYSTEM", AlertLevel.CRITICAL,
            f"Emergency closing of {len(self.positions)} positions executed",
            0.0,
            "System stop"
        )

    def close(self):
        """Clean up resources"""
        self.db_manager.close()