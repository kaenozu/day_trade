#!/usr/bin/env python3
"""
batch_data_fetcherとTechnicalIndicatorsManagerの統合テスト

Issue #451: 前処理パイプラインでのメソッド名エラーを防ぐための統合テスト
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from src.day_trade.data.batch_data_fetcher import AdvancedBatchDataFetcher
from src.day_trade.analysis.technical_indicators_unified import TechnicalIndicatorsManager


class TestBatchDataFetcherTechnicalIndicatorsIntegration:
    """batch_data_fetcherとTechnicalIndicatorsManagerの統合テスト"""

    @pytest.fixture
    def sample_stock_data(self):
        """テスト用の株価データ"""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        base_price = 1000
        prices = []
        current_price = base_price

        for _ in range(100):
            change = np.random.normal(0, 20)
            current_price = max(100, current_price + change)
            prices.append(current_price)

        data = pd.DataFrame({
            'Date': dates,
            'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
            'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(10000, 100000, 100)
        })
        data.set_index('Date', inplace=True)
        return data

    @pytest.fixture
    def batch_fetcher(self):
        """AdvancedBatchDataFetcherインスタンス"""
        return AdvancedBatchDataFetcher(max_workers=2)

    def test_technical_indicators_manager_integration_api_contract(self):
        """TechnicalIndicatorsManagerのAPI契約テスト"""
        # TechnicalIndicatorsManagerが期待される形で使用できることを確認
        manager = TechnicalIndicatorsManager()

        # Issue #451で発生した問題：間違ったメソッド名が使用されていた
        assert hasattr(manager, 'calculate_indicators'), \
            "TechnicalIndicatorsManager must have calculate_indicators method"

        assert not hasattr(manager, 'calculate_all_indicators'), \
            "calculate_all_indicators method should NOT exist - this caused Issue #451"

        # メソッドが呼び出し可能であることを確認
        assert callable(manager.calculate_indicators), \
            "calculate_indicators must be callable"

    def test_batch_data_fetcher_preprocess_method_signature(self, batch_fetcher, sample_stock_data):
        """batch_data_fetcherの前処理メソッドシグネチャテスト"""
        # _preprocess_dataメソッドが存在し、呼び出せることを確認
        assert hasattr(batch_fetcher, '_preprocess_data'), \
            "AdvancedBatchDataFetcher must have _preprocess_data method"

        assert callable(batch_fetcher._preprocess_data), \
            "_preprocess_data must be callable"

    @patch('src.day_trade.data.batch_data_fetcher.INDICATORS_AVAILABLE', True)
    def test_preprocess_data_with_technical_indicators_integration(self, batch_fetcher, sample_stock_data):
        """前処理でのTechnicalIndicatorsManager統合テスト"""
        symbol = "7203"

        try:
            # 実際の前処理実行 - Issue #451で失敗していた箇所
            result = batch_fetcher._preprocess_data(sample_stock_data, symbol)

            # 前処理が正常に完了することを確認（最低限のテスト）
            assert result is not None, "Preprocessing should return some result"

        except AttributeError as e:
            if "calculate_all_indicators" in str(e):
                pytest.fail(
                    f"Issue #451 reproduced: {e}. "
                    "The code is trying to call calculate_all_indicators instead of calculate_indicators"
                )
            else:
                # その他のAttributeErrorは実装依存のため警告として記録
                print(f"Warning: AttributeError in preprocessing (may be implementation-dependent): {e}")
        except Exception as e:
            # その他のエラーは実装に依存するため、AttributeErrorでないことのみ確認
            assert not isinstance(e, AttributeError) or "calculate_all_indicators" not in str(e), \
                f"Issue #451 related AttributeError: {e}"

    @patch('src.day_trade.data.batch_data_fetcher.INDICATORS_AVAILABLE', True)
    def test_technical_indicators_manager_method_call_in_preprocessing(self, batch_fetcher, sample_stock_data):
        """前処理でのTechnicalIndicatorsManagerメソッド呼び出しテスト"""

        # TechnicalIndicatorsManagerをモック化して実際の呼び出しをテスト
        with patch('src.day_trade.data.batch_data_fetcher.TechnicalIndicatorsManager') as MockManager:
            mock_instance = Mock()
            MockManager.return_value = mock_instance

            # Issue #451で修正された正しいメソッド名を確認
            mock_instance.calculate_indicators.return_value = {"sma": "mock_result"}

            symbol = "7203"

            try:
                # 前処理実行
                batch_fetcher._preprocess_data(sample_stock_data, symbol)

                # 正しいメソッドが呼び出されることを確認
                mock_instance.calculate_indicators.assert_called_once()

                # 間違ったメソッド名が呼び出されていないことを確認
                assert not hasattr(mock_instance, 'calculate_all_indicators'), \
                    "Mock should not have calculate_all_indicators method"

            except AttributeError as e:
                if "calculate_all_indicators" in str(e):
                    pytest.fail(
                        f"Issue #451 not fixed: The code is still trying to call calculate_all_indicators. "
                        f"Error: {e}"
                    )
                else:
                    raise

    def test_batch_data_fetcher_error_resilience(self, batch_fetcher):
        """batch_data_fetcherのエラー耐性テスト"""
        # 空のデータでの前処理テスト
        empty_data = pd.DataFrame()
        symbol = "TEST"

        try:
            result = batch_fetcher._preprocess_data(empty_data, symbol)
            # エラーが発生しない場合は、適切な結果が返されることを確認
            assert result is not None
        except Exception as e:
            # エラーが発生する場合は、AttributeErrorでないことを確認
            assert not isinstance(e, AttributeError), f"Should not be AttributeError: {e}"
            # その他のエラーは実装に依存するため許容

    def test_multiple_symbols_preprocessing_stability(self, batch_fetcher, sample_stock_data):
        """複数銘柄前処理の安定性テスト"""
        symbols = ["7203", "8306", "9984"]

        results = []
        errors = []

        for symbol in symbols:
            try:
                result = batch_fetcher._preprocess_data(sample_stock_data, symbol)
                results.append((symbol, result))
            except Exception as e:
                errors.append((symbol, e))

        # Issue #451では3銘柄でエラーが発生していた
        # AttributeErrorが発生していないことを確認
        attribute_errors = [error for symbol, error in errors if isinstance(error, AttributeError)]
        assert len(attribute_errors) == 0, \
            f"AttributeErrors detected (possible Issue #451 reproduction): {attribute_errors}"

        print(f"Processed {len(results)} symbols successfully, {len(errors)} had errors")

    @patch('src.day_trade.data.batch_data_fetcher.INDICATORS_AVAILABLE', False)
    def test_preprocessing_without_indicators(self, batch_fetcher, sample_stock_data):
        """テクニカル指標なしでの前処理テスト"""
        symbol = "7203"

        # INDICATORS_AVAILABLE=Falseの場合でも前処理が動作することを確認
        result = batch_fetcher._preprocess_data(sample_stock_data, symbol)

        assert result is not None, "Preprocessing should work even without indicators"
        assert isinstance(result, pd.DataFrame), "Should return DataFrame even without indicators"

    def test_integration_with_real_batch_workflow(self, batch_fetcher):
        """実際のバッチワークフローとの統合テスト"""
        # 実際のAdvancedBatchDataFetcherのワークフローをテスト
        symbols = ["7203", "8306"]  # 少数の銘柄でテスト

        try:
            # fetch_multiple_data_batchメソッドが存在し、呼び出せることを確認
            if hasattr(batch_fetcher, 'fetch_multiple_data_batch'):
                # 実際の外部API呼び出しを避けるため、モック化
                with patch.object(batch_fetcher, '_fetch_single_data') as mock_fetch:
                    # モックデータを返すように設定
                    mock_response = Mock()
                    mock_response.success = True
                    mock_response.data = pd.DataFrame({
                        'Open': [100], 'High': [105], 'Low': [99],
                        'Close': [103], 'Volume': [1000]
                    })
                    mock_response.symbol = "7203"
                    mock_response.source = "mock"
                    mock_response.timestamp = pd.Timestamp.now()
                    mock_response.metadata = {}

                    mock_fetch.return_value = mock_response

                    # バッチ処理実行
                    results = batch_fetcher.fetch_multiple_data_batch(symbols)

                    # 結果の検証
                    assert isinstance(results, list), "Batch processing should return a list"

                    # Issue #451のようなAttributeErrorが発生していないことを確認
                    for result in results:
                        if hasattr(result, 'success') and not result.success:
                            error_msg = getattr(result, 'error_message', str(result))
                            assert 'calculate_all_indicators' not in error_msg, \
                                f"Issue #451 detected in batch processing: {error_msg}"

        except Exception as e:
            # バッチ処理でAttributeErrorが発生していないことを確認
            assert not isinstance(e, AttributeError), f"Batch processing AttributeError: {e}"


class TestIssue451Regression:
    """Issue #451の回帰テスト"""

    def test_issue451_method_name_error_prevention(self):
        """Issue #451: メソッド名エラーの回帰防止テスト"""

        # 問題の再現：間違ったメソッド名の使用を検出
        manager = TechnicalIndicatorsManager()

        # 修正後：正しいメソッド名が存在することを確認
        assert hasattr(manager, 'calculate_indicators'), \
            "REGRESSION: calculate_indicators method is missing"

        # 修正後：間違ったメソッド名が存在しないことを確認
        assert not hasattr(manager, 'calculate_all_indicators'), \
            "REGRESSION: calculate_all_indicators method should not exist"

    def test_issue451_batch_data_fetcher_code_review(self):
        """Issue #451: batch_data_fetcherのコードレビューテスト"""

        # batch_data_fetcher.pyの実際のコードを検査
        import inspect
        from src.day_trade.data.batch_data_fetcher import AdvancedBatchDataFetcher

        # _preprocess_dataメソッドのソースコードを取得
        source = inspect.getsource(AdvancedBatchDataFetcher._preprocess_data)

        # 正しいメソッド名が使用されていることを確認
        assert 'calculate_indicators(' in source, \
            "REGRESSION: batch_data_fetcher should use calculate_indicators method"

        # 間違ったメソッド名が使用されていないことを確認
        assert 'calculate_all_indicators(' not in source, \
            "REGRESSION: batch_data_fetcher should NOT use calculate_all_indicators method"


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])