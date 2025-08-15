#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for Risk Manager
"""

import logging
import pytest
from risk_manager import PersonalRiskManager
from src.day_trade.data_provider import ManualDataProvider
from src.day_trade.data_models import RiskLevel

@pytest.fixture
def risk_manager():
    """Fixture for PersonalRiskManager"""
    price_data = {
        "7203": 2750,  # -1.8% (unrealized loss)
        "9984": 4320   # +2.9% (unrealized gain)
    }
    data_provider = ManualDataProvider(price_data)
    rm = PersonalRiskManager(data_provider)
    rm.db_manager.delete_all_positions()
    rm.positions = {}
    yield rm
    rm.close()

def test_open_position(risk_manager):
    """Test opening a position"""
    success = risk_manager.open_position("7203", "Toyota Motor", 2800, 30, 3.5, 85)
    assert success
    assert "7203" in risk_manager.positions
    assert risk_manager.positions["7203"].quantity == 30

def test_update_positions(risk_manager):
    """Test updating positions"""
    risk_manager.open_position("7203", "Toyota Motor", 2800, 30, 3.5, 85)
    risk_manager.open_position("9984", "Softbank Group", 4200, 20, 5.2, 78)

    risk_manager.update_positions()

    assert risk_manager.positions["7203"].current_price == 2750
    assert risk_manager.positions["9984"].current_price == 4320
    assert risk_manager.positions["7203"].pnl == -1500
    assert risk_manager.positions["9984"].pnl == 2400

def test_risk_assessment(risk_manager):
    """Test risk assessment logic"""
    risk_level = risk_manager._assess_risk_level("7203", 3.5, 85)
    assert risk_level == RiskLevel.LOW

    risk_level = risk_manager._assess_risk_level("9984", 5.2, 78)
    assert risk_level == RiskLevel.MEDIUM

def test_stop_loss_take_profit(risk_manager):
    """Test stop-loss and take-profit logic"""
    stop_loss, take_profit = risk_manager.calculate_stop_loss_take_profit("7203", 2800, 3.5, 85)
    assert stop_loss < 2800
    assert take_profit > 2800
