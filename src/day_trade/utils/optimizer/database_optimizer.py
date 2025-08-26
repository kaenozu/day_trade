#!/usr/bin/env python3
"""
データベース操作最適化

データベースの一括挿入・更新処理を効率的に実行するための機能を提供します。
"""

import time
from typing import Any, Dict, List


class DatabaseOptimizer:
    """データベース操作の最適化クラス"""

    @staticmethod
    def optimize_bulk_insert(
        data: List[Dict], session, model_class
    ) -> Dict[str, Any]:
        """一括挿入の最適化"""
        start_time = time.perf_counter()

        try:
            # SQLAlchemyの一括挿入を使用
            session.bulk_insert_mappings(model_class, data)
            session.commit()

            execution_time = time.perf_counter() - start_time
            return {
                "success": True,
                "inserted_count": len(data),
                "execution_time": execution_time,
                "records_per_second": len(data) / execution_time,
            }
        except Exception as e:
            session.rollback()
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.perf_counter() - start_time,
            }

    @staticmethod
    def optimize_bulk_update(
        data: List[Dict], session, model_class
    ) -> Dict[str, Any]:
        """一括更新の最適化"""
        start_time = time.perf_counter()

        try:
            # SQLAlchemyの一括更新を使用
            session.bulk_update_mappings(model_class, data)
            session.commit()

            execution_time = time.perf_counter() - start_time
            return {
                "success": True,
                "updated_count": len(data),
                "execution_time": execution_time,
                "records_per_second": len(data) / execution_time,
            }
        except Exception as e:
            session.rollback()
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.perf_counter() - start_time,
            }
