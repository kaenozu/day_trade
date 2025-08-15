
import asyncio
import logging
import threading # Remove if no longer used
from concurrent.futures import ThreadPoolExecutor # Remove if no longer used

logger = logging.getLogger(__name__)

class AsyncUtils:
    """非同期処理ユーティリティ"""

    @staticmethod
    async def run_in_thread(func, *args, **kwargs):
        """別スレッドで同期関数を実行"""
        # asyncio.to_threadはPython 3.9+で利用可能
        # それ以前のバージョンではThreadPoolExecutorを直接使う必要があります
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except AttributeError:
            logger.warning("asyncio.to_thread not available. Falling back to ThreadPoolExecutor.")
            # Fallback for Python versions < 3.9
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, func, *args, **kwargs)

    @staticmethod
    def safe_create_task(coro):
        """安全にタスクを作成"""
        try:
            # asyncio.create_taskは実行中のループがある場合にそれを使用する
            return asyncio.create_task(coro)
        except RuntimeError as e:
            logger.error(f"Failed to create task: {e}")
            raise # Re-raise the exception after logging

# 使用例:
# await AsyncUtils.run_in_thread(some_sync_function, arg1, arg2)
# AsyncUtils.safe_create_task(some_async_function())
