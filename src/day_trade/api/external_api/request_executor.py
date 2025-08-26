#!/usr/bin/env python3
"""
外部APIクライアント - リクエスト実行エンジン
"""

import asyncio
import time
from datetime import datetime
from typing import Optional

import aiohttp
from aiohttp import ClientError, ClientTimeout

from ...utils.logging_config import get_context_logger
from .enums import RequestMethod
from .models import APIRequest, APIResponse

logger = get_context_logger(__name__)


class RequestExecutor:
    """リクエスト実行エンジン"""

    def __init__(self, session: aiohttp.ClientSession, 
                 url_builder, auth_manager, data_normalizer, error_handler):
        self.session = session
        self.url_builder = url_builder
        self.auth_manager = auth_manager
        self.data_normalizer = data_normalizer
        self.error_handler = error_handler

    async def execute_with_retry(self, request: APIRequest) -> Optional[APIResponse]:
        """リトライ付きリクエスト実行"""
        last_error = None

        for attempt in range(request.endpoint.max_retries + 1):
            try:
                response = await self._execute_single_request(request)
                if response.success:
                    return response

                # エラーレスポンスの場合、リトライするかどうか判定
                if not self.error_handler.should_retry(response, attempt):
                    return response

                last_error = response.error_message

            except Exception as e:
                # セキュリティ強化: エラーメッセージをサニタイズして保存
                error_type = self.error_handler.categorize_error(e)
                last_error = self.error_handler.sanitize_error_message(str(e), error_type)
                logger.warning(
                    f"APIリクエストエラー (試行 {attempt + 1}): {last_error}"
                )

                if attempt >= request.endpoint.max_retries:
                    break

            # リトライ前の待機
            if attempt < request.endpoint.max_retries:
                delay = self.error_handler.calculate_retry_delay(
                    attempt,
                    {
                        "retry_delay_seconds": request.endpoint.retry_delay_seconds,
                        "exponential_backoff": True,
                        "max_backoff_seconds": 60.0,
                    }
                )
                await asyncio.sleep(delay)

        # 全てのリトライが失敗
        logger.error(f"APIリクエスト最終失敗: {last_error}")
        return APIResponse(
            request=request,
            status_code=0,
            response_data=None,
            headers={},
            response_time_ms=0,
            timestamp=datetime.now(),
            success=False,
            error_message=last_error,
        )

    async def _execute_single_request(self, request: APIRequest) -> APIResponse:
        """単一APIリクエスト実行"""
        start_time = time.time()

        # URL構築
        url = self.url_builder.build_url(request)

        # ヘッダー構築
        headers = self.auth_manager.add_auth_to_headers(dict(request.headers), request.endpoint)

        try:
            # HTTPリクエスト実行
            async with self.session.request(
                method=request.endpoint.method.value,
                url=url,
                params=(
                    request.params
                    if request.endpoint.method == RequestMethod.GET
                    else None
                ),
                json=(
                    request.data
                    if request.endpoint.method != RequestMethod.GET
                    else None
                ),
                headers=headers,
                timeout=ClientTimeout(total=request.endpoint.timeout_seconds),
            ) as response:
                response_time = (time.time() - start_time) * 1000
                response_headers = dict(response.headers)

                # レスポンスデータ取得
                if request.endpoint.response_format == "json":
                    response_data = await response.json()
                elif request.endpoint.response_format == "csv":
                    text_data = await response.text()
                    response_data = self.data_normalizer.parse_csv_response(text_data)
                else:
                    response_data = await response.text()

                # レスポンス作成
                api_response = APIResponse(
                    request=request,
                    status_code=response.status,
                    response_data=response_data,
                    headers=response_headers,
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    success=response.status == 200,
                    error_message=(
                        None if response.status == 200 else f"HTTP {response.status}"
                    ),
                )

                # データ正規化
                if api_response.success:
                    api_response.normalized_data = await self.data_normalizer.normalize_response_data(
                        api_response
                    )

                return api_response

        except asyncio.TimeoutError:
            return APIResponse(
                request=request,
                status_code=0,
                response_data=None,
                headers={},
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                success=False,
                error_message="Request timeout",
            )

        except ClientError as e:
            # セキュリティ強化: エラーメッセージの機密情報をサニタイズ
            safe_error_message = self.error_handler.sanitize_error_message(str(e), "ClientError")

            return APIResponse(
                request=request,
                status_code=0,
                response_data=None,
                headers={},
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                success=False,
                error_message=safe_error_message,
            )