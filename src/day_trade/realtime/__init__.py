"""
リアルタイム取引システムモジュール

Phase 3: リアルタイム取引システムの実装
WebSocket、イベント駆動アーキテクチャ、自動取引エンジンを提供
"""

from .realtime_feed import RealtimeDataFeed, WebSocketClient

__all__ = ["RealtimeDataFeed", "WebSocketClient"]

__version__ = "1.0.0"
