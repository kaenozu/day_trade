#!/usr/bin/env python3
"""
マスターデータ統合ルール
データ統合の抽象基底クラスと具体実装
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class DataIntegrationRule(ABC):
    """抽象データ統合ルール"""

    @abstractmethod
    async def integrate(
        self, source_data: Dict[str, Any], existing_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """データ統合実行"""
        pass

    @abstractmethod
    def get_rule_info(self) -> Dict[str, Any]:
        """ルール情報取得"""
        pass


class StockDataIntegrationRule(DataIntegrationRule):
    """株式データ統合ルール"""

    async def integrate(
        self, source_data: Dict[str, Any], existing_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """株式データ統合実行"""
        try:
            integrated = source_data.copy()

            if existing_data:
                # 既存データとの統合ロジック
                # 最新の価格情報を優先
                if "last_price" in source_data and "last_updated" in source_data:
                    existing_updated = existing_data.get("last_updated")
                    source_updated = source_data.get("last_updated")

                    if existing_updated and source_updated:
                        if pd.to_datetime(existing_updated) > pd.to_datetime(
                            source_updated
                        ):
                            integrated["last_price"] = existing_data["last_price"]
                            integrated["last_updated"] = existing_data["last_updated"]

                # 企業情報は手動変更を優先
                manual_fields = ["company_name", "industry", "sector"]
                for field in manual_fields:
                    if field in existing_data and existing_data.get(
                        f"{field}_manual", False
                    ):
                        integrated[field] = existing_data[field]
                        integrated[f"{field}_manual"] = True

                # バージョン管理
                integrated["version"] = existing_data.get("version", 1) + 1
                integrated["previous_version"] = existing_data.get("version", 1)

            # 品質チェック
            integrated = await self._apply_quality_rules(integrated)

            return integrated

        except Exception as e:
            logger.error(f"株式データ統合エラー: {e}")
            return source_data

    async def _apply_quality_rules(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """品質ルール適用"""
        # 必須フィールドチェック
        required_fields = ["symbol", "company_name"]
        for required_field in required_fields:
            if required_field not in data or not data[required_field]:
                data[f"{required_field}_quality_issue"] = "missing_required_field"

        # データ型検証
        if "last_price" in data:
            try:
                data["last_price"] = float(data["last_price"])
            except (ValueError, TypeError):
                data["last_price_quality_issue"] = "invalid_price_format"

        return data

    def get_rule_info(self) -> Dict[str, Any]:
        """ルール情報取得"""
        return {
            "rule_type": "stock_data_integration",
            "version": "1.0",
            "priority_fields": ["last_price", "last_updated"],
            "manual_override_fields": ["company_name", "industry", "sector"],
        }


class CompanyDataIntegrationRule(DataIntegrationRule):
    """企業データ統合ルール"""

    async def integrate(
        self, source_data: Dict[str, Any], existing_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """企業データ統合実行"""
        try:
            integrated = source_data.copy()

            if existing_data:
                # 手動更新された情報を優先
                manual_priority_fields = ["name", "industry", "headquarters"]
                for field in manual_priority_fields:
                    if field in existing_data and existing_data.get(
                        f"{field}_manual_override", False
                    ):
                        integrated[field] = existing_data[field]
                        integrated[f"{field}_manual_override"] = True

                # 従業員数などの数値データは最新を優先
                numeric_fields = ["employees", "revenue", "market_cap"]
                for field in numeric_fields:
                    if field in source_data:
                        try:
                            integrated[field] = float(source_data[field])
                        except (ValueError, TypeError):
                            if field in existing_data:
                                integrated[field] = existing_data[field]

                integrated["version"] = existing_data.get("version", 1) + 1

            # 品質チェック適用
            integrated = await self._apply_quality_rules(integrated)

            return integrated

        except Exception as e:
            logger.error(f"企業データ統合エラー: {e}")
            return source_data

    async def _apply_quality_rules(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """品質ルール適用"""
        # 必須フィールドチェック
        required_fields = ["name", "industry"]
        for required_field in required_fields:
            if required_field not in data or not data[required_field]:
                data[f"{required_field}_quality_issue"] = "missing_required_field"

        # 数値フィールドの検証
        numeric_fields = ["employees", "revenue", "market_cap"]
        for field in numeric_fields:
            if field in data and data[field] is not None:
                try:
                    value = float(data[field])
                    if value < 0:
                        data[f"{field}_quality_issue"] = "negative_value"
                    data[field] = value
                except (ValueError, TypeError):
                    data[f"{field}_quality_issue"] = "invalid_numeric_format"

        return data

    def get_rule_info(self) -> Dict[str, Any]:
        """ルール情報取得"""
        return {
            "rule_type": "company_data_integration",
            "version": "1.0",
            "manual_priority_fields": ["name", "industry", "headquarters"],
            "latest_priority_fields": ["employees", "revenue", "market_cap"],
        }


class CurrencyDataIntegrationRule(DataIntegrationRule):
    """通貨データ統合ルール"""

    async def integrate(
        self, source_data: Dict[str, Any], existing_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """通貨データ統合実行"""
        try:
            integrated = source_data.copy()

            if existing_data:
                # 為替レートは常に最新を優先
                rate_fields = ["exchange_rate", "buy_rate", "sell_rate"]
                for field in rate_fields:
                    if field in source_data:
                        # タイムスタンプをチェックして最新を採用
                        source_timestamp = source_data.get("rate_timestamp")
                        existing_timestamp = existing_data.get("rate_timestamp")

                        if source_timestamp and existing_timestamp:
                            if pd.to_datetime(existing_timestamp) > pd.to_datetime(
                                source_timestamp
                            ):
                                integrated[field] = existing_data[field]
                                integrated["rate_timestamp"] = existing_data[
                                    "rate_timestamp"
                                ]

                integrated["version"] = existing_data.get("version", 1) + 1

            # 品質チェック適用
            integrated = await self._apply_quality_rules(integrated)

            return integrated

        except Exception as e:
            logger.error(f"通貨データ統合エラー: {e}")
            return source_data

    async def _apply_quality_rules(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """品質ルール適用"""
        # 必須フィールドチェック
        required_fields = ["code", "name"]
        for required_field in required_fields:
            if required_field not in data or not data[required_field]:
                data[f"{required_field}_quality_issue"] = "missing_required_field"

        # 通貨コードフォーマット検証
        if "code" in data:
            code = data["code"]
            if not isinstance(code, str) or len(code) != 3 or not code.isupper():
                data["code_quality_issue"] = "invalid_currency_code_format"

        # 為替レート検証
        rate_fields = ["exchange_rate", "buy_rate", "sell_rate"]
        for field in rate_fields:
            if field in data and data[field] is not None:
                try:
                    rate = float(data[field])
                    if rate <= 0:
                        data[f"{field}_quality_issue"] = "invalid_rate_value"
                    data[field] = rate
                except (ValueError, TypeError):
                    data[f"{field}_quality_issue"] = "invalid_rate_format"

        return data

    def get_rule_info(self) -> Dict[str, Any]:
        """ルール情報取得"""
        return {
            "rule_type": "currency_data_integration",
            "version": "1.0",
            "timestamp_priority_fields": ["exchange_rate", "buy_rate", "sell_rate"],
            "required_format": {"code": "3-letter uppercase ISO code"},
        }


def get_default_integration_rules() -> Dict[str, DataIntegrationRule]:
    """デフォルト統合ルール取得"""
    return {
        "stock": StockDataIntegrationRule(),
        "company": CompanyDataIntegrationRule(),
        "currency": CurrencyDataIntegrationRule(),
    }