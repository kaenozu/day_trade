#!/usr/bin/env python3
"""
データ検証プロセッサー
Issue #376: バッチ処理の強化

データ検証ステージの処理ロジック
"""

import time
from typing import Any, Dict

from .core_types import StageResult, WorkflowStage
from .stage_processor import StageProcessor


class DataValidateProcessor(StageProcessor):
    """データ検証プロセッサー"""

    def __init__(self):
        super().__init__(WorkflowStage.DATA_VALIDATE)

    async def process(self, data: Any, parameters: Dict[str, Any]) -> StageResult:
        """データ検証処理"""
        start_time = time.time()

        try:
            validation_rules = parameters.get("validation_rules", {})
            strict_mode = parameters.get("strict_mode", False)

            validation_errors = []
            valid_records = []
            invalid_records = []

            if not data:
                return StageResult(
                    stage=self.stage,
                    success=False,
                    error_message="No data provided for validation",
                )

            # データが辞書形式の場合（データタイプ別）
            if isinstance(data, dict):
                validated_data = {}

                for data_type, type_data in data.items():
                    if not isinstance(type_data, list):
                        continue

                    valid_type_records = []

                    for record in type_data:
                        validation_result = self._validate_record(
                            record, validation_rules, data_type
                        )

                        if validation_result["valid"]:
                            valid_type_records.append(record)
                            valid_records.append(record)
                        else:
                            validation_errors.extend(validation_result["errors"])
                            invalid_records.append(record)

                            if strict_mode:
                                return StageResult(
                                    stage=self.stage,
                                    success=False,
                                    error_message=f"Validation failed in strict mode: {validation_result['errors']}",
                                )

                    validated_data[data_type] = valid_type_records

                final_data = validated_data

            # データがリスト形式の場合（フラット）
            elif isinstance(data, list):
                validated_records = []

                for record in data:
                    validation_result = self._validate_record(record, validation_rules)

                    if validation_result["valid"]:
                        validated_records.append(record)
                        valid_records.append(record)
                    else:
                        validation_errors.extend(validation_result["errors"])
                        invalid_records.append(record)

                        if strict_mode:
                            return StageResult(
                                stage=self.stage,
                                success=False,
                                error_message=f"Validation failed: {validation_result['errors']}",
                            )

                final_data = validated_records

            else:
                return StageResult(
                    stage=self.stage,
                    success=False,
                    error_message="Unsupported data format for validation",
                )

            processing_time_ms = (time.time() - start_time) * 1000

            success = len(validation_errors) == 0 or not strict_mode

            return StageResult(
                stage=self.stage,
                success=success,
                data=final_data,
                processing_time_ms=processing_time_ms,
                statistics={
                    "total_records": len(valid_records) + len(invalid_records),
                    "valid_records": len(valid_records),
                    "invalid_records": len(invalid_records),
                    "validation_errors": validation_errors,
                    "validation_success_rate": len(valid_records)
                    / max(len(valid_records) + len(invalid_records), 1),
                },
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000

            return StageResult(
                stage=self.stage,
                success=False,
                processing_time_ms=processing_time_ms,
                error_message=str(e),
            )

    def _validate_record(
        self, record: Dict[str, Any], rules: Dict[str, Any], data_type: str = None
    ) -> Dict[str, Any]:
        """個別レコード検証"""
        errors = []

        try:
            # 必須フィールド検証
            required_fields = rules.get("required_fields", [])
            for field in required_fields:
                if field not in record or record[field] is None:
                    errors.append(f"Required field missing: {field}")

            # データタイプ別検証
            if data_type == "current_price":
                # 価格データ検証
                if "price" in record:
                    price = record["price"]
                    if not isinstance(price, (int, float)) or price < 0:
                        errors.append(f"Invalid price: {price}")

                if "volume" in record:
                    volume = record["volume"]
                    if not isinstance(volume, int) or volume < 0:
                        errors.append(f"Invalid volume: {volume}")

            # カスタム検証ルール
            custom_validations = rules.get("custom_validations", [])
            for validation in custom_validations:
                field = validation.get("field")
                condition = validation.get("condition")

                if field in record:
                    value = record[field]
                    if condition == "positive" and value <= 0:
                        errors.append(f"Field {field} must be positive")
                    elif condition == "not_empty" and (value is None or value == ""):
                        errors.append(f"Field {field} must not be empty")

            return {"valid": len(errors) == 0, "errors": errors}

        except Exception as e:
            return {"valid": False, "errors": [f"Validation error: {str(e)}"]}