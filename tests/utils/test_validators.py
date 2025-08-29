"""
Validators module tests
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional


class MockDataValidator:
    """Mock data validator for testing"""
    
    def __init__(self):
        self.validation_rules = {}
        self.error_history = []
        self.validation_cache = {}
    
    def validate_stock_symbol(self, symbol: str) -> bool:
        """Validate stock symbol format"""
        if not isinstance(symbol, str):
            return False
        # Japanese stock code format: 4 digits + optional suffix
        return len(symbol) >= 4 and symbol[:4].isdigit()
    
    def validate_price_data(self, price: float) -> bool:
        """Validate price data"""
        if not isinstance(price, (int, float)):
            return False
        return price > 0 and price < 1000000  # Reasonable price range
    
    def validate_volume_data(self, volume: int) -> bool:
        """Validate volume data"""
        if not isinstance(volume, int):
            return False
        return volume >= 0
    
    def validate_datetime(self, dt: datetime) -> bool:
        """Validate datetime object"""
        if not isinstance(dt, datetime):
            return False
        # Check if datetime is within reasonable range
        min_date = datetime(2000, 1, 1)
        max_date = datetime.now() + timedelta(days=365)
        return min_date <= dt <= max_date


class MockPortfolioValidator:
    """Mock portfolio validator for testing"""
    
    def __init__(self):
        self.validation_errors = []
        self.warning_threshold = 0.1
    
    def validate_position(self, symbol: str, quantity: int, price: float) -> Dict[str, Any]:
        """Validate portfolio position"""
        errors = []
        warnings = []
        
        if not symbol or len(symbol) < 4:
            errors.append("Invalid symbol format")
        
        if quantity <= 0:
            errors.append("Quantity must be positive")
        
        if price <= 0:
            errors.append("Price must be positive")
        
        # Calculate position value
        position_value = quantity * price
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'position_value': position_value
        }
    
    def validate_portfolio_balance(self, positions: List[Dict], cash: float) -> Dict[str, Any]:
        """Validate overall portfolio balance"""
        total_value = sum(pos['quantity'] * pos['price'] for pos in positions)
        total_portfolio = total_value + cash
        
        warnings = []
        if cash / total_portfolio < 0.05:  # Less than 5% cash
            warnings.append("Low cash ratio")
        
        return {
            'valid': True,
            'total_value': total_value,
            'cash': cash,
            'total_portfolio': total_portfolio,
            'warnings': warnings
        }


class MockOrderValidator:
    """Mock order validator for testing"""
    
    def __init__(self):
        self.market_hours = {
            'open': 9,
            'close': 15,
            'lunch_start': 11.5,
            'lunch_end': 12.5
        }
    
    def validate_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trading order"""
        errors = []
        warnings = []
        
        required_fields = ['symbol', 'quantity', 'price', 'order_type']
        for field in required_fields:
            if field not in order:
                errors.append(f"Missing required field: {field}")
        
        if 'quantity' in order and order['quantity'] <= 0:
            errors.append("Quantity must be positive")
        
        if 'price' in order and order['price'] <= 0:
            errors.append("Price must be positive")
        
        # Validate order type
        valid_types = ['market', 'limit', 'stop', 'stop_limit']
        if 'order_type' in order and order['order_type'] not in valid_types:
            errors.append("Invalid order type")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def validate_market_hours(self, order_time: datetime) -> bool:
        """Check if order is placed during market hours"""
        hour = order_time.hour + order_time.minute / 60.0
        
        # Check if within trading hours (9:00-11:30, 12:30-15:00)
        morning_session = 9 <= hour <= 11.5
        afternoon_session = 12.5 <= hour <= 15
        
        return morning_session or afternoon_session


class MockRiskValidator:
    """Mock risk validator for testing"""
    
    def __init__(self):
        self.max_position_size = 0.1  # 10% of portfolio
        self.max_daily_loss = 0.05    # 5% daily loss limit
        self.sector_limits = {}
    
    def validate_position_size(self, position_value: float, portfolio_value: float) -> Dict[str, Any]:
        """Validate position size against portfolio"""
        position_ratio = position_value / portfolio_value if portfolio_value > 0 else 0
        
        errors = []
        warnings = []
        
        if position_ratio > self.max_position_size:
            errors.append(f"Position size exceeds limit: {position_ratio:.1%} > {self.max_position_size:.1%}")
        
        if position_ratio > self.max_position_size * 0.8:
            warnings.append("Position size approaching limit")
        
        return {
            'valid': len(errors) == 0,
            'position_ratio': position_ratio,
            'errors': errors,
            'warnings': warnings
        }
    
    def validate_daily_loss(self, current_pnl: float, portfolio_value: float) -> Dict[str, Any]:
        """Validate daily P&L against risk limits"""
        loss_ratio = abs(current_pnl) / portfolio_value if portfolio_value > 0 and current_pnl < 0 else 0
        
        errors = []
        warnings = []
        
        if loss_ratio > self.max_daily_loss:
            errors.append(f"Daily loss exceeds limit: {loss_ratio:.1%} > {self.max_daily_loss:.1%}")
        
        if loss_ratio > self.max_daily_loss * 0.8:
            warnings.append("Daily loss approaching limit")
        
        return {
            'valid': len(errors) == 0,
            'loss_ratio': loss_ratio,
            'errors': errors,
            'warnings': warnings
        }


class TestDataValidator:
    """Test data validator functionality"""
    
    def test_validate_stock_symbol(self):
        validator = MockDataValidator()
        
        # Valid symbols
        assert validator.validate_stock_symbol("7203") == True    # Toyota
        assert validator.validate_stock_symbol("9984") == True    # SoftBank
        assert validator.validate_stock_symbol("6758T") == True   # Sony with suffix
        
        # Invalid symbols
        assert validator.validate_stock_symbol("ABC") == False    # Too short
        assert validator.validate_stock_symbol("12345") == False  # Too long for basic
        assert validator.validate_stock_symbol(1234) == False     # Not string
        assert validator.validate_stock_symbol("") == False       # Empty
        assert validator.validate_stock_symbol(None) == False     # None
    
    def test_validate_price_data(self):
        validator = MockDataValidator()
        
        # Valid prices
        assert validator.validate_price_data(1000.0) == True
        assert validator.validate_price_data(100) == True
        assert validator.validate_price_data(0.01) == True
        
        # Invalid prices
        assert validator.validate_price_data(-100) == False       # Negative
        assert validator.validate_price_data(0) == False          # Zero
        assert validator.validate_price_data(2000000) == False    # Too high
        assert validator.validate_price_data("100") == False      # String
        assert validator.validate_price_data(None) == False       # None
    
    def test_validate_volume_data(self):
        validator = MockDataValidator()
        
        # Valid volumes
        assert validator.validate_volume_data(1000) == True
        assert validator.validate_volume_data(0) == True          # Zero volume allowed
        
        # Invalid volumes
        assert validator.validate_volume_data(-100) == False      # Negative
        assert validator.validate_volume_data(100.5) == False     # Float
        assert validator.validate_volume_data("1000") == False    # String
        assert validator.validate_volume_data(None) == False      # None
    
    def test_validate_datetime(self):
        validator = MockDataValidator()
        
        # Valid datetimes
        valid_dt = datetime(2023, 1, 1, 12, 0, 0)
        assert validator.validate_datetime(valid_dt) == True
        
        current_dt = datetime.now()
        assert validator.validate_datetime(current_dt) == True
        
        # Invalid datetimes
        old_dt = datetime(1990, 1, 1)
        assert validator.validate_datetime(old_dt) == False
        
        future_dt = datetime.now() + timedelta(days=400)
        assert validator.validate_datetime(future_dt) == False
        
        assert validator.validate_datetime("2023-01-01") == False  # String
        assert validator.validate_datetime(None) == False          # None


class TestPortfolioValidator:
    """Test portfolio validator functionality"""
    
    def test_validate_position(self):
        validator = MockPortfolioValidator()
        
        # Valid position
        result = validator.validate_position("7203", 100, 2500.0)
        assert result['valid'] == True
        assert result['position_value'] == 250000.0
        assert len(result['errors']) == 0
        
        # Invalid symbol
        result = validator.validate_position("", 100, 2500.0)
        assert result['valid'] == False
        assert "Invalid symbol format" in result['errors']
        
        # Invalid quantity
        result = validator.validate_position("7203", -100, 2500.0)
        assert result['valid'] == False
        assert "Quantity must be positive" in result['errors']
        
        # Invalid price
        result = validator.validate_position("7203", 100, -2500.0)
        assert result['valid'] == False
        assert "Price must be positive" in result['errors']
    
    def test_validate_portfolio_balance(self):
        validator = MockPortfolioValidator()
        
        positions = [
            {'quantity': 100, 'price': 2500.0},
            {'quantity': 200, 'price': 1000.0}
        ]
        cash = 50000.0
        
        result = validator.validate_portfolio_balance(positions, cash)
        assert result['valid'] == True
        assert result['total_value'] == 450000.0  # 250k + 200k
        assert result['cash'] == 50000.0
        assert result['total_portfolio'] == 500000.0
        
        # Low cash warning
        low_cash_result = validator.validate_portfolio_balance(positions, 10000.0)
        assert "Low cash ratio" in low_cash_result['warnings']


class TestOrderValidator:
    """Test order validator functionality"""
    
    def test_validate_order(self):
        validator = MockOrderValidator()
        
        # Valid order
        valid_order = {
            'symbol': '7203',
            'quantity': 100,
            'price': 2500.0,
            'order_type': 'limit'
        }
        result = validator.validate_order(valid_order)
        assert result['valid'] == True
        assert len(result['errors']) == 0
        
        # Missing fields
        incomplete_order = {'symbol': '7203'}
        result = validator.validate_order(incomplete_order)
        assert result['valid'] == False
        assert len(result['errors']) > 0
        
        # Invalid quantity
        invalid_order = {
            'symbol': '7203',
            'quantity': -100,
            'price': 2500.0,
            'order_type': 'limit'
        }
        result = validator.validate_order(invalid_order)
        assert result['valid'] == False
        assert "Quantity must be positive" in result['errors']
    
    def test_validate_market_hours(self):
        validator = MockOrderValidator()
        
        # Valid market hours (morning session)
        morning_time = datetime(2023, 1, 1, 10, 30, 0)
        assert validator.validate_market_hours(morning_time) == True
        
        # Valid market hours (afternoon session)
        afternoon_time = datetime(2023, 1, 1, 14, 30, 0)
        assert validator.validate_market_hours(afternoon_time) == True
        
        # Invalid market hours (lunch break)
        lunch_time = datetime(2023, 1, 1, 12, 0, 0)
        assert validator.validate_market_hours(lunch_time) == False
        
        # Invalid market hours (after close)
        after_close = datetime(2023, 1, 1, 16, 0, 0)
        assert validator.validate_market_hours(after_close) == False


class TestRiskValidator:
    """Test risk validator functionality"""
    
    def test_validate_position_size(self):
        validator = MockRiskValidator()
        
        # Valid position size
        result = validator.validate_position_size(50000.0, 1000000.0)  # 5%
        assert result['valid'] == True
        assert result['position_ratio'] == 0.05
        assert len(result['errors']) == 0
        
        # Position size exceeds limit
        result = validator.validate_position_size(150000.0, 1000000.0)  # 15%
        assert result['valid'] == False
        assert "Position size exceeds limit" in result['errors'][0]
        
        # Position size approaching limit (warning)
        result = validator.validate_position_size(85000.0, 1000000.0)  # 8.5%
        assert result['valid'] == True
        assert "Position size approaching limit" in result['warnings']
    
    def test_validate_daily_loss(self):
        validator = MockRiskValidator()
        
        # Acceptable daily loss
        result = validator.validate_daily_loss(-30000.0, 1000000.0)  # -3%
        assert result['valid'] == True
        assert result['loss_ratio'] == 0.03
        assert len(result['errors']) == 0
        
        # Daily loss exceeds limit
        result = validator.validate_daily_loss(-70000.0, 1000000.0)  # -7%
        assert result['valid'] == False
        assert "Daily loss exceeds limit" in result['errors'][0]
        
        # Daily loss approaching limit (warning)
        result = validator.validate_daily_loss(-45000.0, 1000000.0)  # -4.5%
        assert result['valid'] == True
        assert "Daily loss approaching limit" in result['warnings']
        
        # Positive P&L (no risk)
        result = validator.validate_daily_loss(30000.0, 1000000.0)   # +3%
        assert result['valid'] == True
        assert result['loss_ratio'] == 0.0


class TestValidatorIntegration:
    """Test validator integration scenarios"""
    
    def test_comprehensive_trade_validation(self):
        """Test complete trade validation workflow"""
        data_validator = MockDataValidator()
        order_validator = MockOrderValidator()
        risk_validator = MockRiskValidator()
        
        # Sample trade data
        trade_data = {
            'symbol': '7203',
            'quantity': 100,
            'price': 2500.0,
            'order_type': 'limit',
            'timestamp': datetime(2023, 1, 1, 10, 30, 0)
        }
        
        portfolio_value = 1000000.0
        position_value = trade_data['quantity'] * trade_data['price']
        
        # Run all validations
        symbol_valid = data_validator.validate_stock_symbol(trade_data['symbol'])
        price_valid = data_validator.validate_price_data(trade_data['price'])
        order_result = order_validator.validate_order(trade_data)
        market_hours_valid = order_validator.validate_market_hours(trade_data['timestamp'])
        risk_result = risk_validator.validate_position_size(position_value, portfolio_value)
        
        # All validations should pass
        assert symbol_valid == True
        assert price_valid == True
        assert order_result['valid'] == True
        assert market_hours_valid == True
        assert risk_result['valid'] == True
    
    def test_validation_error_aggregation(self):
        """Test aggregation of validation errors"""
        validators = {
            'data': MockDataValidator(),
            'order': MockOrderValidator(),
            'risk': MockRiskValidator()
        }
        
        # Invalid trade data
        invalid_trade = {
            'symbol': '',  # Invalid symbol
            'quantity': -100,  # Invalid quantity
            'price': -2500.0,  # Invalid price
            'order_type': 'invalid'  # Invalid order type
        }
        
        all_errors = []
        all_warnings = []
        
        # Collect all validation errors
        if not validators['data'].validate_stock_symbol(invalid_trade['symbol']):
            all_errors.append("Invalid stock symbol")
        
        if not validators['data'].validate_price_data(invalid_trade['price']):
            all_errors.append("Invalid price data")
        
        order_result = validators['order'].validate_order(invalid_trade)
        all_errors.extend(order_result['errors'])
        all_warnings.extend(order_result['warnings'])
        
        # Should have multiple validation errors
        assert len(all_errors) > 0
        assert "Invalid stock symbol" in all_errors
        assert "Invalid price data" in all_errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])