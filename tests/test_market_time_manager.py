
import unittest
from datetime import datetime, time, timedelta
from market_time_manager import MarketTimeManager, MarketSession, MarketStatus

class TestMarketTimeManager(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.manager = MarketTimeManager()

    async def test_is_market_day(self):
        # Test a weekday (e.g., 2025-08-18 is a Monday)
        self.assertTrue(self.manager.is_market_day(datetime(2025, 8, 18)))
        # Test a Saturday
        self.assertFalse(self.manager.is_market_day(datetime(2025, 8, 16)))
        # Test a Sunday
        self.assertFalse(self.manager.is_market_day(datetime(2025, 8, 17)))
        # Test a holiday (e.g., 2025-09-15 is Respect for the Aged Day)
        self.assertFalse(self.manager.is_market_day(datetime(2025, 9, 15)))
        # Test a special holiday (hardcoded in manager for now)
        self.assertFalse(self.manager.is_market_day(datetime(2025, 1, 1)))
        # Test a special trading day (hardcoded in manager for now)
        self.assertTrue(self.manager.is_market_day(datetime(2025, 1, 4))) # Assuming 2025-01-04 is a special trading day

    async def test_get_current_session(self):
        # Market Closed (Sunday)
        self.assertEqual(self.manager.get_current_session(datetime(2025, 8, 17, 10, 0)), MarketSession.MARKET_CLOSED)
        # Pre-market
        self.assertEqual(self.manager.get_current_session(datetime(2025, 8, 18, 8, 30)), MarketSession.PRE_MARKET)
        # Morning Session
        self.assertEqual(self.manager.get_current_session(datetime(2025, 8, 18, 9, 30)), MarketSession.MORNING_SESSION)
        # Lunch Break
        self.assertEqual(self.manager.get_current_session(datetime(2025, 8, 18, 12, 0)), MarketSession.LUNCH_BREAK)
        # Afternoon Session
        self.assertEqual(self.manager.get_current_session(datetime(2025, 8, 18, 13, 0)), MarketSession.AFTERNOON_SESSION)
        # After Market
        self.assertEqual(self.manager.get_current_session(datetime(2025, 8, 18, 16, 0)), MarketSession.AFTER_MARKET)

    async def test_get_market_status(self):
        self.assertEqual(self.manager.get_market_status(datetime(2025, 8, 17, 10, 0)), MarketStatus.CLOSED)
        self.assertEqual(self.manager.get_market_status(datetime(2025, 8, 18, 8, 30)), MarketStatus.PRE_OPEN)
        self.assertEqual(self.manager.get_market_status(datetime(2025, 8, 18, 9, 30)), MarketStatus.OPEN)
        self.assertEqual(self.manager.get_market_status(datetime(2025, 8, 18, 12, 0)), MarketStatus.POST_CLOSE)
        self.assertEqual(self.manager.get_market_status(datetime(2025, 8, 18, 13, 0)), MarketStatus.OPEN)
        self.assertEqual(self.manager.get_market_status(datetime(2025, 8, 18, 16, 0)), MarketStatus.POST_CLOSE)

    async def test_is_market_open(self):
        self.assertFalse(self.manager.is_market_open(datetime(2025, 8, 18, 8, 30)))
        self.assertTrue(self.manager.is_market_open(datetime(2025, 8, 18, 9, 30)))
        self.assertFalse(self.manager.is_market_open(datetime(2025, 8, 18, 12, 0)))
        self.assertTrue(self.manager.is_market_open(datetime(2025, 8, 18, 13, 0)))

    async def test_get_next_market_open(self):
        # Pre-market -> Today's morning open
        self.assertEqual(self.manager.get_next_market_open(datetime(2025, 8, 18, 8, 30)), datetime(2025, 8, 18, 9, 0))
        # Morning session -> Today's afternoon open
        self.assertEqual(self.manager.get_next_market_open(datetime(2025, 8, 18, 10, 0)), datetime(2025, 8, 18, 12, 30))
        # Lunch break -> Today's afternoon open
        self.assertEqual(self.manager.get_next_market_open(datetime(2025, 8, 18, 12, 0)), datetime(2025, 8, 18, 12, 30))
        # After market -> Next market day's morning open (e.g., Monday 16:00 -> Tuesday 9:00)
        self.assertEqual(self.manager.get_next_market_open(datetime(2025, 8, 18, 16, 0)), datetime(2025, 8, 19, 9, 0))
        # Market closed (Sunday) -> Next market day's morning open (e.g., Sunday 10:00 -> Monday 9:00)
        self.assertEqual(self.manager.get_next_market_open(datetime(2025, 8, 17, 10, 0)), datetime(2025, 8, 18, 9, 0))
        # Test across a holiday (e.g., Friday before a Monday holiday)
        self.assertEqual(self.manager.get_next_market_open(datetime(2025, 9, 12, 16, 0)), datetime(2025, 9, 16, 9, 0)) # Assuming 2025-09-15 is a holiday

    async def test_get_next_market_close(self):
        # Morning session -> Today's morning close
        self.assertEqual(self.manager.get_next_market_close(datetime(2025, 8, 18, 10, 0)), datetime(2025, 8, 18, 11, 30))
        # Lunch break -> Today's afternoon close
        self.assertEqual(self.manager.get_next_market_close(datetime(2025, 8, 18, 12, 0)), datetime(2025, 8, 18, 15, 0))
        # Afternoon session -> Today's afternoon close
        self.assertEqual(self.manager.get_next_market_close(datetime(2025, 8, 18, 14, 0)), datetime(2025, 8, 18, 15, 0))
        # After market -> Next market day's afternoon close
        self.assertEqual(self.manager.get_next_market_close(datetime(2025, 8, 18, 16, 0)), datetime(2025, 8, 19, 15, 0))
        # Market closed (Sunday) -> Next market day's afternoon close
        self.assertEqual(self.manager.get_next_market_close(datetime(2025, 8, 17, 10, 0)), datetime(2025, 8, 18, 15, 0))

    async def test_get_time_until_next_event(self):
        # Pre-market
        event, seconds = self.manager.get_time_until_next_event(datetime(2025, 8, 18, 8, 59, 50))
        self.assertEqual(event, "前場開始")
        self.assertEqual(seconds, 10)
        # Morning session
        event, seconds = self.manager.get_time_until_next_event(datetime(2025, 8, 18, 11, 29, 50))
        self.assertEqual(event, "前場終了")
        self.assertEqual(seconds, 10)
        # Lunch break
        event, seconds = self.manager.get_time_until_next_event(datetime(2025, 8, 18, 12, 29, 50))
        self.assertEqual(event, "後場開始")
        self.assertEqual(seconds, 10)
        # Afternoon session
        event, seconds = self.manager.get_time_until_next_event(datetime(2025, 8, 18, 14, 59, 50))
        self.assertEqual(event, "大引け")
        self.assertEqual(seconds, 10)
        # After market
        event, seconds = self.manager.get_time_until_next_event(datetime(2025, 8, 18, 15, 0, 0))
        self.assertEqual(event, "次回開場")
        self.assertEqual(seconds, (datetime(2025, 8, 19, 9, 0) - datetime(2025, 8, 18, 15, 0, 0)).total_seconds())
        # Market closed
        event, seconds = self.manager.get_time_until_next_event(datetime(2025, 8, 17, 10, 0, 0))
        self.assertEqual(event, "次回開場")
        self.assertEqual(seconds, (datetime(2025, 8, 18, 9, 0) - datetime(2025, 8, 17, 10, 0, 0)).total_seconds())

    async def test_get_market_summary(self):
        summary = self.manager.get_market_summary(datetime(2025, 8, 18, 9, 30))
        self.assertEqual(summary["session"], "前場")
        self.assertEqual(summary["status"], "開場中")
        self.assertTrue(summary["is_open"])
        self.assertEqual(summary["next_event"], "前場終了")

    async def test_get_session_advice(self):
        advice = self.manager.get_session_advice(datetime(2025, 8, 18, 9, 30))
        self.assertIn("前場取引中", advice)
        advice = self.manager.get_session_advice(datetime(2025, 8, 17, 10, 0))
        self.assertIn("休場日", advice)

if __name__ == '__main__':
    unittest.main()
