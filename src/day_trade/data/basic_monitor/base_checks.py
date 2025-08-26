#!/usr/bin/env python3
"""
Basic Monitor Base Checks
基本監視システムの抽象チェッククラス

監視チェック処理の基底クラス
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from .models import MonitorAlert


class MonitorCheck(ABC):
    """抽象監視チェッククラス"""

    @abstractmethod
    async def execute_check(
        self, data_source: str, data: Any, context: Dict[str, Any]
    ) -> Tuple[bool, Optional[MonitorAlert]]:
        """チェック実行"""
        pass

    @abstractmethod
    def get_check_info(self) -> Dict[str, Any]:
        """チェック情報取得"""
        pass