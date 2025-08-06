#!/usr/bin/env python3
"""
銘柄一括登録スクリプトのテスト

bulk_stock_registration.pyの各コンポーネントをテストする。
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bulk_stock_registration import (
    JPXDataDownloader,
    StockDataParser,
    BulkStockRegistrar
)


class TestJPXDataDownloader(unittest.TestCase):
    """JPXDataDownloaderのテスト"""

    def setUp(self):
        self.downloader = JPXDataDownloader(timeout=5, retries=1)

    def test_init(self):
        """初期化テスト"""
        assert self.downloader.timeout == 5
        assert isinstance(self.downloader.session, requests.Session)
        assert 'User-Agent' in self.downloader.session.headers

    @patch('scripts.bulk_stock_registration.requests.Session.get')
    def test_download_success(self, mock_get):
        """ダウンロード成功テスト"""
        # モックレスポンス
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content.return_value = [b'test,data\n1,test\n']
        mock_get.return_value = mock_response

        # テスト実行
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test.csv'
            result = self.downloader._download_from_url('http://test.com/data.csv', output_path)

            assert result is True
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    @patch('scripts.bulk_stock_registration.requests.Session.get')
    def test_download_http_error(self, mock_get):
        """HTTPエラーテスト"""
        mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test.csv'
            result = self.downloader._download_from_url('http://test.com/data.csv', output_path)

            assert result is False

    @patch('scripts.bulk_stock_registration.requests.Session.get')
    def test_download_timeout(self, mock_get):
        """タイムアウトテスト"""
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test.csv'
            result = self.downloader._download_from_url('http://test.com/data.csv', output_path)

            assert result is False


class TestStockDataParser(unittest.TestCase):
    """StockDataParserのテスト"""

    def setUp(self):
        self.parser = StockDataParser()

    def test_init(self):
        """初期化テスト"""
        assert self.parser.parsed_stocks == []
        assert self.parser.error_count == 0

    def test_detect_field_mapping_success(self):
        """フィールドマッピング検出成功テスト"""
        headers = ['証券コード', '銘柄名', '市場区分', 'セクター', '業種名']
        mapping = self.parser._detect_field_mapping(headers)

        assert mapping is not None
        assert mapping['code'] == '証券コード'
        assert mapping['name'] == '銘柄名'
        assert mapping['market'] == '市場区分'
        assert mapping['sector'] == 'セクター'
        assert mapping['industry'] == '業種名'

    def test_detect_field_mapping_english(self):
        """英語ヘッダーのマッピング検出テスト"""
        headers = ['Code', 'Name', 'Market', 'Sector', 'Industry']
        mapping = self.parser._detect_field_mapping(headers)

        assert mapping is not None
        assert mapping['code'] == 'Code'
        assert mapping['name'] == 'Name'

    def test_detect_field_mapping_failure(self):
        """フィールドマッピング検出失敗テスト"""
        headers = ['unknown1', 'unknown2', 'unknown3']
        mapping = self.parser._detect_field_mapping(headers)

        assert mapping is None

    def test_extract_stock_data_success(self):
        """株式データ抽出成功テスト"""
        row = {
            '証券コード': '7203',
            '銘柄名': 'トヨタ自動車',
            '市場区分': '東証プライム',
            'セクター': '輸送用機器',
            '業種名': '自動車'
        }

        field_mapping = {
            'code': '証券コード',
            'name': '銘柄名',
            'market': '市場区分',
            'sector': 'セクター',
            'industry': '業種名'
        }

        result = self.parser._extract_stock_data(row, field_mapping)

        assert result is not None
        assert result['code'] == '7203'
        assert result['name'] == 'トヨタ自動車'
        assert result['market'] == '東証プライム'
        assert result['sector'] == '輸送用機器'
        assert result['industry'] == '自動車'

    def test_extract_stock_data_invalid_code(self):
        """無効なコードのテスト"""
        row = {
            '証券コード': 'INVALID',  # 4桁数字でない
            '銘柄名': 'テスト会社'
        }

        field_mapping = {'code': '証券コード', 'name': '銘柄名'}
        result = self.parser._extract_stock_data(row, field_mapping)

        assert result is None

    def test_extract_stock_data_empty_name(self):
        """空の名前のテスト"""
        row = {
            '証券コード': '7203',
            '銘柄名': ''  # 空の名前
        }

        field_mapping = {'code': '証券コード', 'name': '銘柄名'}
        result = self.parser._extract_stock_data(row, field_mapping)

        assert result is None

    def test_parse_csv_success(self):
        """CSV解析成功テスト"""
        csv_content = """証券コード,銘柄名,市場区分,セクター,業種名
7203,トヨタ自動車,東証プライム,輸送用機器,自動車
9984,ソフトバンクグループ,東証プライム,情報・通信,通信業
6758,ソニーグループ,東証プライム,電気機器,電気機器"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            stocks, error_count = self.parser.parse_csv(csv_path)

            assert len(stocks) == 3
            assert error_count == 0

            # 最初の銘柄をチェック
            assert stocks[0]['code'] == '7203'
            assert stocks[0]['name'] == 'トヨタ自動車'
            assert stocks[0]['market'] == '東証プライム'

        finally:
            csv_path.unlink()

    def test_parse_csv_with_errors(self):
        """エラーを含むCSV解析テスト"""
        csv_content = """証券コード,銘柄名,市場区分
7203,トヨタ自動車,東証プライム
INVALID,無効なコード,東証プライム
9984,ソフトバンクグループ,東証プライム"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            stocks, error_count = self.parser.parse_csv(csv_path)

            assert len(stocks) == 2  # 有効なもののみ
            assert error_count == 1   # 1つのエラー

        finally:
            csv_path.unlink()


class TestBulkStockRegistrar(unittest.TestCase):
    """BulkStockRegistrarのテスト"""

    def setUp(self):
        self.registrar = BulkStockRegistrar(batch_size=10)

    @patch('scripts.bulk_stock_registration.create_stock_master_manager')
    @patch('scripts.bulk_stock_registration.get_default_database_manager')
    def test_init(self, mock_db_manager, mock_stock_master):
        """初期化テスト"""
        registrar = BulkStockRegistrar(batch_size=20)

        assert registrar.batch_size == 20
        assert 'total_processed' in registrar.stats
        assert registrar.stats['newly_added'] == 0

    @patch('scripts.bulk_stock_registration.BulkStockRegistrar._get_existing_codes')
    @patch('scripts.bulk_stock_registration.BulkStockRegistrar._process_batch')
    def test_register_stocks_success(self, mock_process_batch, mock_existing_codes):
        """銘柄登録成功テスト"""
        mock_existing_codes.return_value = {'7203'}
        mock_process_batch.return_value = None

        stock_data = [
            {
                'code': '7203',
                'name': 'トヨタ自動車',
                'market': '東証プライム',
                'sector': '輸送用機器',
                'industry': '自動車'
            },
            {
                'code': '9984',
                'name': 'ソフトバンクグループ',
                'market': '東証プライム',
                'sector': '情報・通信',
                'industry': '通信業'
            }
        ]

        stats = self.registrar.register_stocks(stock_data)

        assert stats['total_processed'] == 2
        assert mock_process_batch.call_count == 1  # バッチサイズが10なので1回

    def test_get_existing_codes_success(self):
        """既存コード取得成功テスト"""
        # モックセッション
        mock_session = Mock()
        mock_session.query.return_value.all.return_value = [('7203',), ('9984',)]

        # コンテキストマネージャーのモック設定
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_session)
        context_manager.__exit__ = Mock(return_value=False)

        self.registrar.db_manager = Mock()
        self.registrar.db_manager.session_scope.return_value = context_manager

        existing_codes = self.registrar._get_existing_codes()

        assert existing_codes == {'7203', '9984'}

    def test_get_existing_codes_error(self):
        """既存コード取得エラーテスト"""
        self.registrar.db_manager = Mock()
        self.registrar.db_manager.session_scope.side_effect = Exception("DB Error")

        existing_codes = self.registrar._get_existing_codes()

        assert existing_codes == set()  # エラー時は空のset


class TestIntegration(unittest.TestCase):
    """統合テスト"""

    def test_full_workflow_dry_run(self):
        """完全ワークフローのドライランテスト"""
        # テスト用CSVファイル作成
        csv_content = """証券コード,銘柄名,市場区分,セクター,業種名
7203,トヨタ自動車,東証プライム,輸送用機器,自動車
9984,ソフトバンクグループ,東証プライム,情報・通信,通信業"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            # 1. CSV解析
            parser = StockDataParser()
            stocks, error_count = parser.parse_csv(csv_path)

            assert len(stocks) == 2
            assert error_count == 0

            # 2. データ検証
            for stock in stocks:
                assert 'code' in stock
                assert 'name' in stock
                assert len(stock['code']) == 4
                assert stock['name'].strip() != ''

            print(f"統合テスト成功: {len(stocks)}件の銘柄を解析")

        finally:
            csv_path.unlink()


if __name__ == '__main__':
    # テスト実行
    unittest.main(verbosity=2)
