
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class AsyncUtils:
    """非同期処理ユーティリティ"""

    @staticmethod
    def run_in_thread(coro):
        """別スレッドで非同期関数を実行"""
        def run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        with ThreadPoolExecutor() as executor:
            future = executor.submit(run)
            return future.result()

    @staticmethod
    def safe_create_task(coro):
        """安全にタスクを作成"""
        try:
            loop = asyncio.get_running_loop()
            return loop.create_task(coro)
        except RuntimeError:
            # イベントループが存在しない場合
            return asyncio.create_task(coro)

# 使用例:
# AsyncUtils.run_in_thread(some_async_function())
# AsyncUtils.safe_create_task(some_async_function())
