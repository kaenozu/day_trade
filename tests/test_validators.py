"""
validators.pyのテスト
"""

from src.day_trade.utils.validators import (
    normalize_stock_codes,
    suggest_stock_code_correction,
    validate_interval,
    validate_period,
    validate_stock_code,
)


class TestValidateStockCode:
    """validate_stock_code のテスト"""

    def test_valid_japanese_stock_codes(self):
        """有効な日本株コード"""
        valid_codes = ["7203", "9984", "4755", "1234"]
        for code in valid_codes:
            assert validate_stock_code(code) is True

    def test_valid_japanese_stock_codes_with_market(self):
        """市場コード付きの有効な日本株コード"""
        valid_codes = ["7203.T", "9984.t", "4755.T", "1234.T"]
        for code in valid_codes:
            assert validate_stock_code(code) is True

    def test_valid_alphanumeric_codes(self):
        """アルファベット含む有効なコード"""
        valid_codes = ["AAPL", "GOOGL", "MSFT", "TSLA", "A1B2C3"]
        for code in valid_codes:
            assert validate_stock_code(code) is True

    def test_valid_alphanumeric_codes_with_market(self):
        """市場コード付きのアルファベット含むコード"""
        valid_codes = ["AAPL.T", "GOOGL.T", "ABC123.T"]
        for code in valid_codes:
            assert validate_stock_code(code) is True

    def test_invalid_stock_codes(self):
        """無効な証券コード"""
        invalid_codes = [
            "",              # 空文字
            "ABC123456789A", # 長すぎる（10文字超）
            "7203.X",        # 無効な市場コード
            "7203.TT",       # 無効な市場コード
            "7203 ",         # スペース含む
            " 7203",         # 先頭スペース
            "7203.T.T",      # 複数ドット
        ]
        for code in invalid_codes:
            assert validate_stock_code(code) is False

    def test_none_input(self):
        """None入力のテスト"""
        # 現在の実装ではNoneはFalseを返す
        assert validate_stock_code(None) is False

    def test_empty_string(self):
        """空文字入力のテスト"""
        assert validate_stock_code("") is False

    def test_case_insensitive(self):
        """大文字小文字混合のテスト"""
        assert validate_stock_code("aapl") is True
        assert validate_stock_code("AaPl") is True
        assert validate_stock_code("7203.t") is True


class TestValidatePeriod:
    """validate_period のテスト"""

    def test_valid_periods(self):
        """有効な期間"""
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        for period in valid_periods:
            assert validate_period(period) is True

    def test_invalid_periods(self):
        """無効な期間"""
        invalid_periods = [
            "",           # 空文字
            "1D",         # 大文字
            "1day",       # 異なる形式
            "1month",     # 異なる形式
            "7d",         # サポートされていない日数
            "15y",        # サポートされていない年数
            "invalid",    # 完全無効
            "1w",         # 週は期間では無効
            None,         # None
        ]
        for period in invalid_periods:
            if period is not None:
                assert validate_period(period) is False

    def test_none_input(self):
        """None入力のテスト"""
        # 現在の実装ではTypeErrorではなくFalseを返す可能性
        assert validate_period(None) is False


class TestValidateInterval:
    """validate_interval のテスト"""

    def test_valid_intervals(self):
        """有効な間隔"""
        valid_intervals = [
            "1m", "2m", "5m", "15m", "30m", "60m", "90m",
            "1h", "1d", "5d", "1wk", "1mo", "3mo"
        ]
        for interval in valid_intervals:
            assert validate_interval(interval) is True

    def test_invalid_intervals(self):
        """無効な間隔"""
        invalid_intervals = [
            "",           # 空文字
            "1M",         # 大文字
            "3m",         # サポートされていない分数
            "10m",        # サポートされていない分数
            "2h",         # サポートされていない時間
            "1day",       # 異なる形式
            "6mo",        # サポートされていない月数
            "invalid",    # 完全無効
            "1y",         # 年は間隔では無効
            None,         # None
        ]
        for interval in invalid_intervals:
            if interval is not None:
                assert validate_interval(interval) is False

    def test_none_input(self):
        """None入力のテスト"""
        # 現在の実装ではTypeErrorではなくFalseを返す可能性
        assert validate_interval(None) is False


class TestNormalizeStockCodes:
    """normalize_stock_codes のテスト"""

    def test_normalize_japanese_codes(self):
        """日本株コードの正規化"""
        input_codes = ["7203", "9984", "4755"]
        expected = ["7203.T", "9984.T", "4755.T"]
        result = normalize_stock_codes(input_codes)
        assert result == expected

    def test_normalize_mixed_codes(self):
        """混合コードの正規化"""
        input_codes = ["7203", "AAPL", "9984.T", "googl"]
        expected = ["7203.T", "AAPL", "9984.T", "GOOGL"]
        result = normalize_stock_codes(input_codes)
        assert result == expected

    def test_normalize_with_invalid_codes(self):
        """無効なコードを含む正規化"""
        input_codes = ["7203", "invalid", "AAPL", ""]
        expected = ["7203.T", "INVALID", "AAPL"]  # 実装では"invalid"も有効として処理される
        result = normalize_stock_codes(input_codes)
        assert result == expected

    def test_normalize_empty_list(self):
        """空リストの正規化"""
        result = normalize_stock_codes([])
        assert result == []

    def test_normalize_all_invalid(self):
        """すべて無効なコードの正規化"""
        input_codes = ["", "ABC123456789AB", "7203 "]  # 実際に無効なコード
        result = normalize_stock_codes(input_codes)
        assert result == []

    def test_normalize_case_handling(self):
        """大文字小文字処理"""
        input_codes = ["aapl", "7203", "GoOgL"]
        expected = ["AAPL", "7203.T", "GOOGL"]
        result = normalize_stock_codes(input_codes)
        assert result == expected

    def test_normalize_with_existing_market_code(self):
        """既に市場コードがあるケースの正規化"""
        input_codes = ["7203.T", "9984.t", "AAPL.T"]
        expected = ["7203.T", "9984.T", "AAPL.T"]
        result = normalize_stock_codes(input_codes)
        assert result == expected


class TestSuggestStockCodeCorrection:
    """suggest_stock_code_correction のテスト"""

    def test_suggest_add_market_code(self):
        """市場コード追加の提案"""
        assert suggest_stock_code_correction("7203") == "7203.T"
        assert suggest_stock_code_correction("9984") == "9984.T"
        assert suggest_stock_code_correction("0001") == "0001.T"

    def test_suggest_uppercase_conversion(self):
        """大文字化の提案"""
        assert suggest_stock_code_correction("aapl") == "AAPL"
        assert suggest_stock_code_correction("googl") == "GOOGL"
        assert suggest_stock_code_correction("MsF") == "MSF"

    def test_suggest_no_correction_needed(self):
        """修正不要なケース"""
        assert suggest_stock_code_correction("AAPL") is None
        assert suggest_stock_code_correction("7203.T") is None
        assert suggest_stock_code_correction("GOOGL") is None

    def test_suggest_empty_input(self):
        """空入力"""
        assert suggest_stock_code_correction("") is None

    def test_suggest_none_input(self):
        """None入力"""
        assert suggest_stock_code_correction(None) is None

    def test_suggest_invalid_codes(self):
        """無効なコードに対する提案"""
        # 無効なコードでも部分的に修正可能な場合はNoneを返す（現在の実装）
        assert suggest_stock_code_correction("12345") is None  # 5桁は修正不可
        assert suggest_stock_code_correction("!@#$") is None   # 特殊文字は修正不可

    def test_suggest_priority_handling(self):
        """複数の修正候補がある場合の優先処理"""
        # 4桁の数字で小文字が含まれている場合、市場コード追加が優先される
        assert suggest_stock_code_correction("7203") == "7203.T"  # 数字優先

        # アルファベットの小文字は大文字化
        assert suggest_stock_code_correction("aapl") == "AAPL"
