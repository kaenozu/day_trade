"""
Issue #132 改善点のテスト
アラートマネージャーの最適化、設定管理の改善、永続化機能のテスト
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

from src.day_trade.core.alerts import AlertManager, AlertCondition, AlertTrigger, AlertPriority
from src.day_trade.core.alert_strategies import (
    AlertStrategyFactory, PriceAboveStrategy, VolumeSpikeStrategy
)
from src.day_trade.core.config import ConfigManager
from src.day_trade.core.persistent_alerts import PersistentAlertsManager
from src.day_trade.core.security_config import SecurityConfig, SecurityError
from src.day_trade.models.enums import AlertType


class TestAlertManagerOptimizations:
    """AlertManager最適化のテスト"""

    @pytest.fixture
    def mock_stock_fetcher(self):
        """モックStockFetcher"""
        mock_fetcher = Mock()

        # バルクデータ取得のモック
        mock_fetcher.get_bulk_current_prices = Mock(return_value={
            'AAPL': {
                'current_price': 150.0,
                'volume': 1000000,
                'change_percent': 2.5
            },
            'MSFT': {
                'current_price': 300.0,
                'volume': 800000,
                'change_percent': -1.2
            }
        })

        # 個別データ取得のモック
        mock_fetcher.get_current_price = Mock(return_value={
            'current_price': 150.0,
            'volume': 1000000,
            'change_percent': 2.5
        })

        return mock_fetcher

    @pytest.fixture
    def alert_manager(self, mock_stock_fetcher):
        """テスト用AlertManager"""
        manager = AlertManager(stock_fetcher=mock_stock_fetcher)
        return manager

    def test_bulk_data_fetch(self, alert_manager):
        """バルクデータ取得のテスト"""
        symbols = ['AAPL', 'MSFT']
        bulk_data = alert_manager._fetch_bulk_market_data(symbols)

        assert 'AAPL' in bulk_data
        assert 'MSFT' in bulk_data
        assert bulk_data['AAPL']['current_data']['current_price'] == 150.0
        assert bulk_data['MSFT']['current_data']['current_price'] == 300.0

    def test_optimized_alert_check(self, alert_manager):
        """最適化されたアラートチェックのテスト"""
        # テスト用のアラート条件を追加
        condition = AlertCondition(
            alert_id="test_alert_1",
            symbol="AAPL",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("140.0"),
            priority=AlertPriority.HIGH
        )
        alert_manager.add_alert(condition)

        # アラートチェック実行
        with patch.object(alert_manager, '_handle_alert_trigger') as mock_handle:
            alert_manager.check_all_alerts()

            # トリガーが呼ばれたかチェック
            mock_handle.assert_called_once()

    def test_strategy_factory(self):
        """アラート戦略ファクトリーのテスト"""
        factory = AlertStrategyFactory()

        # 各戦略が正しく取得できるかテスト
        price_strategy = factory.get_strategy(AlertType.PRICE_ABOVE)
        assert isinstance(price_strategy, PriceAboveStrategy)

        volume_strategy = factory.get_strategy(AlertType.VOLUME_SPIKE)
        assert isinstance(volume_strategy, VolumeSpikeStrategy)

    def test_price_above_strategy(self):
        """価格上昇戦略のテスト"""
        strategy = PriceAboveStrategy()

        # トリガー条件
        is_triggered, message, current_value = strategy.evaluate(
            condition_value=Decimal("140.0"),
            current_price=Decimal("150.0"),
            volume=1000000,
            change_percent=2.5
        )

        assert is_triggered == True
        assert "価格が 140.0 を上回りました" in message
        assert current_value == Decimal("150.0")

        # 非トリガー条件
        is_triggered, message, current_value = strategy.evaluate(
            condition_value=Decimal("160.0"),
            current_price=Decimal("150.0"),
            volume=1000000,
            change_percent=2.5
        )

        assert is_triggered == False


class TestConfigManagerImprovements:
    """ConfigManager改善のテスト"""

    def test_save_config_error_handling(self):
        """設定保存のエラーハンドリングテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            manager = ConfigManager(config_dir=Path(temp_dir))

            # 正常な保存
            manager.save_config()
            assert config_path.exists()

    def test_export_import_config(self):
        """設定エクスポート・インポートのテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            export_path = Path(temp_dir) / "exported_config.json"

            manager = ConfigManager(config_dir=config_dir)

            # 設定を変更
            manager.set("api.timeout", 30)

            # エクスポート
            manager.export_config(export_path)
            assert export_path.exists()

            # 新しいマネージャーでインポート
            manager2 = ConfigManager(config_dir=config_dir / "imported")
            manager2.import_config(export_path)

            # 設定が正しくインポートされたかチェック
            assert manager2.get("api.timeout") == 30

    def test_import_nonexistent_file(self):
        """存在しないファイルのインポートテスト"""
        manager = ConfigManager()
        nonexistent_path = Path("/nonexistent/file.json")

        with pytest.raises(FileNotFoundError):
            manager.import_config(nonexistent_path)

    def test_import_invalid_json(self):
        """無効なJSONファイルのインポートテスト"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            f.flush()

            manager = ConfigManager()

            with pytest.raises(json.JSONDecodeError):
                manager.import_config(Path(f.name))


class TestPersistentAlertsManager:
    """永続化アラートマネージャーのテスト"""

    @pytest.fixture
    def mock_db_manager(self):
        """モックデータベースマネージャー"""
        mock_db = Mock()
        mock_session = Mock()
        mock_db.session_scope.return_value.__enter__ = Mock(return_value=mock_session)
        mock_db.session_scope.return_value.__exit__ = Mock(return_value=None)
        return mock_db

    @pytest.fixture
    def persistent_manager(self, mock_db_manager):
        """テスト用永続化マネージャー"""
        return PersistentAlertsManager(db_manager_instance=mock_db_manager)

    def test_save_alert_condition(self, persistent_manager):
        """アラート条件保存のテスト"""
        condition = AlertCondition(
            alert_id="test_persistent_alert",
            symbol="AAPL",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("150.0"),
            priority=AlertPriority.HIGH
        )

        # モックセッションの設定
        mock_session = persistent_manager.db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = persistent_manager.save_alert_condition(condition)
        assert result == True

    def test_load_alert_conditions(self, persistent_manager):
        """アラート条件読み込みのテスト"""
        # モックデータの設定
        mock_session = persistent_manager.db_manager.session_scope.return_value.__enter__.return_value
        mock_persistent_condition = Mock()
        mock_persistent_condition.expires_at = None
        mock_persistent_condition.to_alert_condition.return_value = AlertCondition(
            alert_id="test_alert",
            symbol="AAPL",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("150.0")
        )

        mock_session.query.return_value.filter.return_value.all.return_value = [mock_persistent_condition]

        conditions = persistent_manager.load_alert_conditions()
        assert len(conditions) == 1
        assert conditions[0].alert_id == "test_alert"

    def test_cleanup_expired_conditions(self, persistent_manager):
        """期限切れ条件クリーンアップのテスト"""
        mock_session = persistent_manager.db_manager.session_scope.return_value.__enter__.return_value
        mock_session.query.return_value.filter.return_value.update.return_value = 2

        cleaned_count = persistent_manager.cleanup_expired_conditions()
        assert cleaned_count == 2


class TestSecurityConfig:
    """セキュリティ設定のテスト"""

    def test_validate_safe_function(self):
        """安全な関数の検証テスト"""
        security = SecurityConfig()

        def safe_function(x):
            return x * 2

        # 手動で承認
        import inspect
        source = inspect.getsource(safe_function)
        func_hash = security._calculate_function_hash(source)
        security.approve_function("safe_function", func_hash)

        # 検証
        assert security.validate_custom_function(safe_function) == True

    def test_validate_unsafe_function(self):
        """危険な関数の検証テスト"""
        security = SecurityConfig()

        # 危険な関数（evalを使用）
        exec_code = '''def unsafe_function(x):
    return eval(x)'''

        # 文字列から関数を作成
        namespace = {}
        exec(exec_code, namespace)
        unsafe_function = namespace['unsafe_function']

        # 検証（失敗するはず）
        assert security.validate_custom_function(unsafe_function) == False

    def test_ast_analysis(self):
        """AST解析のテスト"""
        security = SecurityConfig()

        # 安全なコード
        safe_code = '''def safe_func(x):
    import math
    return math.sqrt(x)'''

        assert security._analyze_function_ast(safe_code) == True

        # 危険なコード
        unsafe_code = '''def unsafe_func(x):
    import os
    return os.system(x)'''

        assert security._analyze_function_ast(unsafe_code) == False

    def test_create_safe_globals(self):
        """安全なグローバル環境の作成テスト"""
        security = SecurityConfig()
        safe_globals = security.create_safe_globals()

        # 許可された要素が含まれているかチェック
        assert 'math' in safe_globals
        assert 'datetime' in safe_globals
        assert 'Decimal' in safe_globals

        # 危険な要素が含まれていないかチェック
        builtins = safe_globals.get('__builtins__', {})
        assert 'eval' not in builtins
        assert 'exec' not in builtins
        assert 'open' not in builtins

    def test_execute_safe_function(self):
        """安全な関数実行のテスト"""
        security = SecurityConfig()

        def safe_math_function(x, y):
            import math
            return math.sqrt(x * x + y * y)

        # 手動で承認
        import inspect
        source = inspect.getsource(safe_math_function)
        func_hash = security._calculate_function_hash(source)
        security.approve_function("safe_math_function", func_hash)

        # 実行
        result = security.execute_safe_function(safe_math_function, 3, 4)
        assert result == 5.0

    def test_security_report(self):
        """セキュリティレポートのテスト"""
        security = SecurityConfig()
        report = security.get_security_report()

        assert 'custom_functions_enabled' in report
        assert 'approved_functions_count' in report
        assert 'allowed_modules' in report
        assert isinstance(report['allowed_modules'], list)


if __name__ == "__main__":
    pytest.main([__file__])
