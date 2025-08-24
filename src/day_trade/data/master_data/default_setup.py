#!/usr/bin/env python3
"""
マスターデータ管理 デフォルト設定
デフォルトデータ要素、ポリシー、スチュワードの設定
"""

import logging
from typing import Dict

from .types import (
    DataClassification,
    DataDomain,
    DataElement,
)

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class DefaultSetup:
    """デフォルト設定クラス"""

    def __init__(self):
        """初期化"""
        self.data_elements = self._setup_default_data_elements()

    def _setup_default_data_elements(self) -> Dict[str, DataElement]:
        """デフォルトデータ要素設定"""
        elements = {}

        # 株式関連データ要素
        stock_symbol = DataElement(
            element_id="stock_symbol",
            name="株式コード",
            description="証券取引所における株式の識別コード",
            data_type="string",
            domain=DataDomain.SECURITY,
            classification=DataClassification.PUBLIC,
            business_rules=["4桁の数字", "日本株式市場"],
            validation_rules=["^[0-9]{4}$"],
            source_systems=["trading_system", "market_data"],
        )
        elements[stock_symbol.element_id] = stock_symbol

        company_name = DataElement(
            element_id="company_name",
            name="企業名",
            description="上場企業の正式名称",
            data_type="string",
            domain=DataDomain.SECURITY,
            classification=DataClassification.PUBLIC,
            business_rules=["正式な企業名", "日本語表記"],
            validation_rules=["最大長255文字"],
            source_systems=["corporate_data", "regulatory_filing"],
        )
        elements[company_name.element_id] = company_name

        stock_price = DataElement(
            element_id="stock_price",
            name="株価",
            description="株式の現在価格または終値",
            data_type="decimal",
            domain=DataDomain.MARKET,
            classification=DataClassification.PUBLIC,
            business_rules=["正の数値", "円建て価格"],
            validation_rules=[">0", "小数点以下2桁まで"],
            source_systems=["market_data", "trading_system"],
        )
        elements[stock_price.element_id] = stock_price

        market_cap = DataElement(
            element_id="market_cap",
            name="時価総額",
            description="企業の時価総額",
            data_type="decimal",
            domain=DataDomain.MARKET,
            classification=DataClassification.PUBLIC,
            business_rules=["正の数値", "円建て"],
            validation_rules=[">0"],
            source_systems=["market_data", "calculation_engine"],
        )
        elements[market_cap.element_id] = market_cap

        # 企業関連データ要素
        industry = DataElement(
            element_id="industry",
            name="業界",
            description="企業が属する業界分類",
            data_type="string",
            domain=DataDomain.REFERENCE,
            classification=DataClassification.PUBLIC,
            business_rules=["標準業界分類", "日本語表記"],
            validation_rules=["最大長100文字"],
            source_systems=["corporate_data", "classification_system"],
        )
        elements[industry.element_id] = industry

        sector = DataElement(
            element_id="sector",
            name="セクター",
            description="企業が属するセクター分類",
            data_type="string",
            domain=DataDomain.REFERENCE,
            classification=DataClassification.PUBLIC,
            business_rules=["標準セクター分類", "日本語表記"],
            validation_rules=["最大長100文字"],
            source_systems=["corporate_data", "classification_system"],
        )
        elements[sector.element_id] = sector

        employees = DataElement(
            element_id="employees",
            name="従業員数",
            description="企業の従業員数",
            data_type="integer",
            domain=DataDomain.REFERENCE,
            classification=DataClassification.PUBLIC,
            business_rules=["正の整数", "単体ベース"],
            validation_rules=[">=0"],
            source_systems=["corporate_data", "hr_system"],
        )
        elements[employees.element_id] = employees

        # 通貨関連データ要素
        currency_code = DataElement(
            element_id="currency_code",
            name="通貨コード",
            description="ISO 4217準拠の通貨コード",
            data_type="string",
            domain=DataDomain.FINANCIAL,
            classification=DataClassification.PUBLIC,
            business_rules=["ISO 4217標準", "3文字大文字"],
            validation_rules=["^[A-Z]{3}$"],
            source_systems=["currency_master", "central_bank"],
        )
        elements[currency_code.element_id] = currency_code

        exchange_rate = DataElement(
            element_id="exchange_rate",
            name="為替レート",
            description="基準通貨に対する為替レート",
            data_type="decimal",
            domain=DataDomain.MARKET,
            classification=DataClassification.PUBLIC,
            business_rules=["正の数値", "対円レート"],
            validation_rules=[">0", "小数点以下4桁まで"],
            source_systems=["forex_data", "central_bank"],
        )
        elements[exchange_rate.element_id] = exchange_rate

        # 取引所関連データ要素
        exchange_code = DataElement(
            element_id="exchange_code",
            name="取引所コード",
            description="証券取引所の識別コード",
            data_type="string",
            domain=DataDomain.REFERENCE,
            classification=DataClassification.PUBLIC,
            business_rules=["MIC標準準拠", "4文字大文字"],
            validation_rules=["^[A-Z]{4}$"],
            source_systems=["exchange_master", "regulatory_data"],
        )
        elements[exchange_code.element_id] = exchange_code

        exchange_name = DataElement(
            element_id="exchange_name",
            name="取引所名",
            description="証券取引所の正式名称",
            data_type="string",
            domain=DataDomain.REFERENCE,
            classification=DataClassification.PUBLIC,
            business_rules=["正式名称", "日本語または英語"],
            validation_rules=["最大長200文字"],
            source_systems=["exchange_master", "regulatory_data"],
        )
        elements[exchange_name.element_id] = exchange_name

        # 時価データ要素
        trading_volume = DataElement(
            element_id="trading_volume",
            name="出来高",
            description="株式の取引出来高",
            data_type="integer",
            domain=DataDomain.MARKET,
            classification=DataClassification.PUBLIC,
            business_rules=["非負整数", "株数単位"],
            validation_rules=[">=0"],
            source_systems=["market_data", "trading_system"],
        )
        elements[trading_volume.element_id] = trading_volume

        open_price = DataElement(
            element_id="open_price",
            name="始値",
            description="株式の始値",
            data_type="decimal",
            domain=DataDomain.MARKET,
            classification=DataClassification.PUBLIC,
            business_rules=["正の数値", "円建て価格"],
            validation_rules=[">0", "小数点以下2桁まで"],
            source_systems=["market_data", "trading_system"],
        )
        elements[open_price.element_id] = open_price

        high_price = DataElement(
            element_id="high_price",
            name="高値",
            description="株式の高値",
            data_type="decimal",
            domain=DataDomain.MARKET,
            classification=DataClassification.PUBLIC,
            business_rules=["正の数値", "円建て価格"],
            validation_rules=[">0", "小数点以下2桁まで"],
            source_systems=["market_data", "trading_system"],
        )
        elements[high_price.element_id] = high_price

        low_price = DataElement(
            element_id="low_price",
            name="安値",
            description="株式の安値",
            data_type="decimal",
            domain=DataDomain.MARKET,
            classification=DataClassification.PUBLIC,
            business_rules=["正の数値", "円建て価格"],
            validation_rules=[">0", "小数点以下2桁まで"],
            source_systems=["market_data", "trading_system"],
        )
        elements[low_price.element_id] = low_price

        # 財務データ要素
        revenue = DataElement(
            element_id="revenue",
            name="売上高",
            description="企業の年間売上高",
            data_type="decimal",
            domain=DataDomain.FINANCIAL,
            classification=DataClassification.PUBLIC,
            business_rules=["正の数値", "円建て", "年間ベース"],
            validation_rules=[">=0"],
            source_systems=["financial_data", "corporate_filing"],
        )
        elements[revenue.element_id] = revenue

        net_income = DataElement(
            element_id="net_income",
            name="純利益",
            description="企業の年間純利益",
            data_type="decimal",
            domain=DataDomain.FINANCIAL,
            classification=DataClassification.PUBLIC,
            business_rules=["数値", "円建て", "年間ベース"],
            validation_rules=["数値"],
            source_systems=["financial_data", "corporate_filing"],
        )
        elements[net_income.element_id] = net_income

        total_assets = DataElement(
            element_id="total_assets",
            name="総資産",
            description="企業の総資産額",
            data_type="decimal",
            domain=DataDomain.FINANCIAL,
            classification=DataClassification.PUBLIC,
            business_rules=["正の数値", "円建て"],
            validation_rules=[">0"],
            source_systems=["financial_data", "corporate_filing"],
        )
        elements[total_assets.element_id] = total_assets

        # 国・地域関連データ要素
        country_code = DataElement(
            element_id="country_code",
            name="国コード",
            description="ISO 3166-1準拠の国コード",
            data_type="string",
            domain=DataDomain.REFERENCE,
            classification=DataClassification.PUBLIC,
            business_rules=["ISO 3166-1 alpha-2標準", "2文字大文字"],
            validation_rules=["^[A-Z]{2}$"],
            source_systems=["reference_data", "regulatory_data"],
        )
        elements[country_code.element_id] = country_code

        country_name = DataElement(
            element_id="country_name",
            name="国名",
            description="国の正式名称",
            data_type="string",
            domain=DataDomain.REFERENCE,
            classification=DataClassification.PUBLIC,
            business_rules=["正式国名", "日本語表記"],
            validation_rules=["最大長100文字"],
            source_systems=["reference_data", "regulatory_data"],
        )
        elements[country_name.element_id] = country_name

        logger.info(f"デフォルトデータ要素設定完了: {len(elements)}件")
        return elements

    def get_data_elements(self) -> Dict[str, DataElement]:
        """データ要素取得"""
        return self.data_elements

    def get_required_fields_for_entity_type(self, entity_type: str) -> list:
        """エンティティタイプの必須フィールド取得"""
        required_fields_map = {
            "stock": ["symbol", "company_name"],
            "company": ["name", "industry"],
            "currency": ["code", "name"],
            "exchange": ["code", "name", "country"],
            "financial": ["revenue", "net_income"],
        }
        return required_fields_map.get(entity_type, [])

    def get_numeric_fields_for_entity_type(self, entity_type: str) -> list:
        """エンティティタイプの数値フィールド取得"""
        numeric_fields_map = {
            "stock": [
                "last_price", "open_price", "high_price", "low_price", 
                "trading_volume", "market_cap"
            ],
            "company": ["employees", "revenue", "net_income", "total_assets"],
            "currency": ["exchange_rate"],
            "exchange": [],
            "financial": ["revenue", "net_income", "total_assets"],
        }
        return numeric_fields_map.get(entity_type, [])

    def get_validation_rules_for_field(self, field_name: str) -> list:
        """フィールドの検証ルール取得"""
        for element in self.data_elements.values():
            element_field_name = element.name.lower().replace(" ", "_")
            if element_field_name == field_name or element.element_id == field_name:
                return element.validation_rules
        return []

    def validate_data_element_consistency(self) -> Dict[str, list]:
        """データ要素整合性検証"""
        issues = {
            "missing_validation_rules": [],
            "invalid_data_types": [],
            "missing_business_rules": [],
        }

        valid_data_types = ["string", "integer", "decimal", "boolean", "date", "datetime"]

        for element_id, element in self.data_elements.items():
            # バリデーションルールチェック
            if not element.validation_rules:
                issues["missing_validation_rules"].append(element_id)

            # データタイプチェック
            if element.data_type not in valid_data_types:
                issues["invalid_data_types"].append(f"{element_id}: {element.data_type}")

            # ビジネスルールチェック
            if not element.business_rules:
                issues["missing_business_rules"].append(element_id)

        return issues