#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for Improved Day Trading Engine
改善版デイトレードエンジンのテストケース

Issue #849対応: 包括的テストフレームワーク
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# テスト対象のインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))

from day_trading_engine import (
    PersonalDayTradingEngine,
    DayTradingSignal,
    TradingSession,
    DayTradingRecommendation,
    create_day_trading_engine
)

class TestPersonalDayTradingEngine:
    """PersonalDayTradingEngineのテストクラス"""

    @pytest.fixture
    def temp_config_file(self):
        """テスト用設定ファイル"""
        config_data = {
            "symbol_mapping": {
                "7203": "トヨタ自動車",
                "8306": "三菱UFJ",
                "6758": "ソニーG"
            },
            "data_fallback": {
                "enable_mock_data": True,
                "mock_data_notification": False
            },
            "market_timing": {
                "session_multipliers": {
                    "MORNING_SESSION": 1.5,
                    "AFTERNOON_SESSION": 1.2
                }
            },
            "logging": {
                "level": "DEBUG"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
            yield f.name

        Path(f.name).unlink()

    @pytest.fixture
    def engine(self, temp_config_file):
        """テスト用エンジンインスタンス"""
        return PersonalDayTradingEngine(config_path=temp_config_file)

    def test_initialization(self, engine):
        """初期化テスト"""
        assert engine.config is not None
        assert engine.daytrading_symbols is not None
        assert hasattr(engine, 'data_mode')
        assert hasattr(engine, 'time_mode')

    def test_config_loading(self, temp_config_file):
        """設定ファイル読み込みテスト"""
        engine = PersonalDayTradingEngine(config_path=temp_config_file)

        assert "symbol_mapping" in engine.config
        assert engine.config["symbol_mapping"]["7203"] == "トヨタ自動車"
        assert engine.config["data_fallback"]["enable_mock_data"] == True

    def test_symbol_name_from_config(self, engine):
        """設定ファイルからの銘柄名取得テスト"""
        assert engine._get_symbol_name_from_config("7203") == "トヨタ自動車"
        assert engine._get_symbol_name_from_config("9999") == "銘柄9999"

    def test_session_multiplier(self, engine):
        """時間帯別係数テスト"""
        morning_multiplier = engine._get_session_multiplier(TradingSession.MORNING_SESSION)
        assert morning_multiplier == 1.5  # 設定ファイルの値

        afternoon_multiplier = engine._get_session_multiplier(TradingSession.AFTERNOON_SESSION)
        assert afternoon_multiplier == 1.2

    def test_data_fallback_handling(self, engine):
        """データフォールバック処理テスト"""
        # モックデータが有効の場合
        mock_data = engine._handle_data_fallback("7203", "20250101")

        assert isinstance(mock_data, dict)
        assert "Open" in mock_data
        assert "High" in mock_data
        assert "Low" in mock_data
        assert "Close" in mock_data
        assert "Volume" in mock_data

        # データの妥当性チェック
        assert mock_data["High"] >= max(mock_data["Open"], mock_data["Close"])
        assert mock_data["Low"] <= min(mock_data["Open"], mock_data["Close"])

    @pytest.mark.asyncio
    async def test_daytrading_recommendations(self, engine):
        """デイトレード推奨取得テスト"""
        recommendations = await engine.get_today_daytrading_recommendations(limit=3)

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3

        if recommendations:
            rec = recommendations[0]
            assert isinstance(rec, DayTradingRecommendation)
            assert hasattr(rec, 'symbol')
            assert hasattr(rec, 'signal')
            assert isinstance(rec.signal, DayTradingSignal)
            assert isinstance(rec.confidence, float)
            assert 0 <= rec.confidence <= 100

    @pytest.mark.asyncio
    async def test_tomorrow_premarket_forecast(self, engine):
        """翌日前場予想テスト"""
        recommendations = await engine.get_tomorrow_premarket_forecast(limit=2)

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 2

        if recommendations:
            rec = recommendations[0]
            assert isinstance(rec, DayTradingRecommendation)
            assert "翌日" in rec.entry_timing or "寄り" in rec.entry_timing

    def test_trading_session_detection(self, engine):
        """取引時間帯判定テスト"""
        current_session = engine._get_current_trading_session()
        assert isinstance(current_session, TradingSession)

    def test_session_advice(self, engine):
        """時間帯アドバイステスト"""
        advice = engine.get_session_advice()
        assert isinstance(advice, str)
        assert len(advice) > 0

    def test_improved_mock_data_generation(self, engine):
        """改善されたモックデータ生成テスト"""
        mock_data = engine._generate_improved_mock_data("7203", "20250101")

        # データ品質チェック
        assert mock_data["Open"] > 0
        assert mock_data["High"] >= max(mock_data["Open"], mock_data["Close"])
        assert mock_data["Low"] <= min(mock_data["Open"], mock_data["Close"])
        assert mock_data["Volume"] > 0

        # 再現性チェック（同じシードで同じ結果）
        mock_data2 = engine._generate_improved_mock_data("7203", "20250101")
        assert mock_data["Open"] == mock_data2["Open"]

class TestFactoryFunction:
    """ファクトリー関数のテスト"""

    def test_create_day_trading_engine(self):
        """エンジン作成関数テスト"""
        engine = create_day_trading_engine()
        assert isinstance(engine, PersonalDayTradingEngine)

    def test_create_with_custom_config(self, temp_config_file):
        """カスタム設定でのエンジン作成テスト"""
        engine = create_day_trading_engine(config_path=temp_config_file)
        assert isinstance(engine, PersonalDayTradingEngine)
        assert "symbol_mapping" in engine.config

class TestDataFallbackConfiguration:
    """データフォールバック設定のテスト"""

    @pytest.fixture
    def mock_disabled_config(self):
        """モックデータ無効設定"""
        config_data = {
            "data_fallback": {
                "enable_mock_data": False,
                "mock_data_notification": True
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            yield f.name

        Path(f.name).unlink()

    def test_mock_data_disabled(self, mock_disabled_config):
        """モックデータ無効時のテスト"""
        engine = PersonalDayTradingEngine(config_path=mock_disabled_config)

        with pytest.raises(ValueError, match="モックデータの使用が無効化"):
            engine._handle_data_fallback("7203", "20250101")

class TestConfigurationManagement:
    """設定管理のテスト"""

    def test_default_config_creation(self):
        """デフォルト設定作成テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            engine = PersonalDayTradingEngine(config_path=str(config_path))

            # デフォルト設定が作成されることを確認
            assert config_path.exists()

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            assert "symbol_mapping" in config
            assert "data_fallback" in config
            assert "market_timing" in config

    def test_config_error_handling(self):
        """設定エラー処理テスト"""
        # 存在しないディレクトリのパス
        invalid_path = "/invalid/path/config.json"

        # エラーが発生してもデフォルト設定で動作することを確認
        engine = PersonalDayTradingEngine(config_path=invalid_path)
        assert engine.config is not None
        assert "symbol_mapping" in engine.config

class TestIntegration:
    """統合テスト"""

    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self, temp_config_file):
        """全体ワークフローのシミュレーションテスト"""
        engine = PersonalDayTradingEngine(config_path=temp_config_file)

        # 設定確認
        assert engine.config["data_fallback"]["enable_mock_data"] == True

        # 銘柄取得
        assert len(engine.daytrading_symbols) > 0

        # 推奨取得
        recommendations = await engine.get_today_daytrading_recommendations(limit=2)
        assert len(recommendations) <= 2

        # 時間帯アドバイス
        advice = engine.get_session_advice()
        assert isinstance(advice, str)

if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])