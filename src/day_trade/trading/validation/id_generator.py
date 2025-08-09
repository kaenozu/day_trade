"""
ID生成

取引ID・注文ID・レポートIDの一意生成機能
"""

import hashlib
import time
import uuid
from datetime import datetime
from typing import Dict, Optional, Set
import threading

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class IDGenerator:
    """
    ID生成クラス

    各種一意識別子の生成・管理機能を提供
    """

    def __init__(self):
        """初期化"""
        self._lock = threading.Lock()
        self._sequence_counters: Dict[str, int] = {}
        self._used_ids: Set[str] = set()
        self._startup_time = int(time.time())

        logger.info("ID生成機初期化完了")

    def generate_trade_id(self, symbol: Optional[str] = None) -> str:
        """
        取引ID生成

        Args:
            symbol: 銘柄コード（オプション）

        Returns:
            一意の取引ID
        """
        try:
            with self._lock:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                microseconds = datetime.now().strftime("%f")[:3]  # ミリ秒

                # シーケンス番号
                sequence = self._get_next_sequence("trade")

                # ランダム部分
                random_part = str(uuid.uuid4())[:8]

                # 基本ID構築
                base_id = f"TRD_{timestamp}{microseconds}_{sequence:04d}_{random_part}"

                # 銘柄コードがある場合は追加
                if symbol:
                    base_id = f"{base_id}_{symbol}"

                # 重複チェックと調整
                final_id = self._ensure_unique_id(base_id, "TRD")

                self._used_ids.add(final_id)
                logger.debug(f"取引ID生成: {final_id}")

                return final_id

        except Exception as e:
            logger.error(f"取引ID生成エラー: {e}")
            # フォールバック：タイムスタンプベース
            return f"TRD_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"

    def generate_order_id(self, order_type: str = "MKT") -> str:
        """
        注文ID生成

        Args:
            order_type: 注文タイプ（MKT, LMT, STP等）

        Returns:
            一意の注文ID
        """
        try:
            with self._lock:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                sequence = self._get_next_sequence("order")
                random_part = str(uuid.uuid4())[:6]

                base_id = f"ORD_{order_type}_{timestamp}_{sequence:04d}_{random_part}"
                final_id = self._ensure_unique_id(base_id, "ORD")

                self._used_ids.add(final_id)
                logger.debug(f"注文ID生成: {final_id}")

                return final_id

        except Exception as e:
            logger.error(f"注文ID生成エラー: {e}")
            return f"ORD_{order_type}_{int(time.time() * 1000)}_{str(uuid.uuid4())[:6]}"

    def generate_position_id(self, symbol: str) -> str:
        """
        ポジションID生成

        Args:
            symbol: 銘柄コード

        Returns:
            一意のポジションID
        """
        try:
            with self._lock:
                date_str = datetime.now().strftime("%Y%m%d")
                sequence = self._get_next_sequence(f"position_{symbol}")

                # 銘柄コードハッシュ（短縮）
                symbol_hash = hashlib.md5(symbol.encode()).hexdigest()[:6]

                base_id = f"POS_{symbol}_{date_str}_{sequence:03d}_{symbol_hash}"
                final_id = self._ensure_unique_id(base_id, "POS")

                self._used_ids.add(final_id)
                logger.debug(f"ポジションID生成: {final_id}")

                return final_id

        except Exception as e:
            logger.error(f"ポジションID生成エラー: {e}")
            return f"POS_{symbol}_{int(time.time())}_{str(uuid.uuid4())[:6]}"

    def generate_report_id(self, report_type: str = "RPT") -> str:
        """
        レポートID生成

        Args:
            report_type: レポートタイプ（RPT, TAX, PNL等）

        Returns:
            一意のレポートID
        """
        try:
            with self._lock:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                sequence = self._get_next_sequence("report")

                base_id = f"{report_type}_{timestamp}_{sequence:04d}"
                final_id = self._ensure_unique_id(base_id, report_type)

                self._used_ids.add(final_id)
                logger.debug(f"レポートID生成: {final_id}")

                return final_id

        except Exception as e:
            logger.error(f"レポートID生成エラー: {e}")
            return f"{report_type}_{int(time.time())}_{str(uuid.uuid4())[:8]}"

    def generate_batch_id(self, batch_type: str = "BATCH") -> str:
        """
        バッチ処理ID生成

        Args:
            batch_type: バッチタイプ

        Returns:
            一意のバッチID
        """
        try:
            with self._lock:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                sequence = self._get_next_sequence("batch")
                process_id = str(uuid.uuid4())[:8]

                base_id = f"{batch_type}_{timestamp}_{sequence:04d}_{process_id}"
                final_id = self._ensure_unique_id(base_id, batch_type)

                self._used_ids.add(final_id)
                logger.debug(f"バッチID生成: {final_id}")

                return final_id

        except Exception as e:
            logger.error(f"バッチID生成エラー: {e}")
            return f"{batch_type}_{int(time.time())}_{str(uuid.uuid4())[:8]}"

    def generate_uuid(self) -> str:
        """
        汎用UUID生成

        Returns:
            UUID文字列
        """
        return str(uuid.uuid4())

    def generate_short_uuid(self, length: int = 8) -> str:
        """
        短縮UUID生成

        Args:
            length: 文字列長

        Returns:
            短縮UUID文字列
        """
        return str(uuid.uuid4()).replace("-", "")[:length].upper()

    def generate_numeric_id(self, prefix: str = "", digits: int = 10) -> str:
        """
        数値ID生成

        Args:
            prefix: プレフィックス
            digits: 桁数

        Returns:
            数値ID文字列
        """
        try:
            # タイムスタンプベース数値
            timestamp_ms = int(time.time() * 1000)

            # 指定桁数に調整
            numeric_part = str(timestamp_ms)[-digits:]
            if len(numeric_part) < digits:
                numeric_part = numeric_part.zfill(digits)

            final_id = f"{prefix}{numeric_part}" if prefix else numeric_part

            logger.debug(f"数値ID生成: {final_id}")
            return final_id

        except Exception as e:
            logger.error(f"数値ID生成エラー: {e}")
            return f"{prefix}{'9' * digits}"

    def generate_hash_id(self, data: str, algorithm: str = "md5", length: int = 16) -> str:
        """
        ハッシュID生成

        Args:
            data: ハッシュ対象データ
            algorithm: ハッシュアルゴリズム（md5, sha1, sha256）
            length: 出力長

        Returns:
            ハッシュID文字列
        """
        try:
            # タイムスタンプを追加してユニーク性を確保
            unique_data = f"{data}_{time.time()}_{uuid.uuid4()}"

            if algorithm.lower() == "md5":
                hash_obj = hashlib.md5(unique_data.encode())
            elif algorithm.lower() == "sha1":
                hash_obj = hashlib.sha1(unique_data.encode())
            elif algorithm.lower() == "sha256":
                hash_obj = hashlib.sha256(unique_data.encode())
            else:
                hash_obj = hashlib.md5(unique_data.encode())

            hash_id = hash_obj.hexdigest()[:length].upper()

            logger.debug(f"ハッシュID生成: {hash_id}")
            return hash_id

        except Exception as e:
            logger.error(f"ハッシュID生成エラー: {e}")
            return str(uuid.uuid4())[:length].upper()

    def _get_next_sequence(self, sequence_type: str) -> int:
        """次のシーケンス番号取得"""
        if sequence_type not in self._sequence_counters:
            self._sequence_counters[sequence_type] = 0

        self._sequence_counters[sequence_type] += 1

        # オーバーフロー対策
        if self._sequence_counters[sequence_type] > 9999:
            self._sequence_counters[sequence_type] = 1

        return self._sequence_counters[sequence_type]

    def _ensure_unique_id(self, base_id: str, prefix: str) -> str:
        """ID重複回避"""
        if base_id not in self._used_ids:
            return base_id

        # 重複がある場合、追加サフィックスで調整
        counter = 1
        while True:
            candidate_id = f"{base_id}_{counter:03d}"
            if candidate_id not in self._used_ids:
                logger.debug(f"ID重複回避: {base_id} -> {candidate_id}")
                return candidate_id
            counter += 1

            # 無限ループ対策
            if counter > 999:
                unique_suffix = str(uuid.uuid4())[:8]
                fallback_id = f"{prefix}_{int(time.time())}_{unique_suffix}"
                logger.warning(f"ID重複回避失敗、フォールバック使用: {fallback_id}")
                return fallback_id

    def validate_id_format(self, id_string: str, expected_prefix: str = None) -> Dict[str, bool]:
        """
        ID形式検証

        Args:
            id_string: 検証対象ID
            expected_prefix: 期待されるプレフィックス

        Returns:
            検証結果辞書
        """
        validation_result = {
            "is_valid": True,
            "has_expected_prefix": True,
            "is_unique_format": True,
            "length_appropriate": True,
        }

        try:
            if not id_string or not isinstance(id_string, str):
                validation_result["is_valid"] = False
                return validation_result

            # プレフィックスチェック
            if expected_prefix:
                if not id_string.startswith(expected_prefix):
                    validation_result["has_expected_prefix"] = False
                    validation_result["is_valid"] = False

            # 長さチェック
            if len(id_string) < 10 or len(id_string) > 100:
                validation_result["length_appropriate"] = False
                validation_result["is_valid"] = False

            # 一意性フォーマットチェック（基本的な構造確認）
            if expected_prefix:
                parts = id_string.split("_")
                if len(parts) < 3:  # 最低限の構造
                    validation_result["is_unique_format"] = False
                    validation_result["is_valid"] = False

            return validation_result

        except Exception as e:
            logger.error(f"ID形式検証エラー: {e}")
            validation_result["is_valid"] = False
            return validation_result

    def is_id_used(self, id_string: str) -> bool:
        """
        ID使用済みチェック

        Args:
            id_string: チェック対象ID

        Returns:
            使用済みの場合True
        """
        return id_string in self._used_ids

    def register_used_id(self, id_string: str) -> None:
        """
        使用済みID登録

        Args:
            id_string: 登録するID
        """
        with self._lock:
            self._used_ids.add(id_string)
            logger.debug(f"使用済みID登録: {id_string}")

    def get_id_statistics(self) -> Dict[str, any]:
        """
        ID生成統計取得

        Returns:
            統計情報辞書
        """
        try:
            with self._lock:
                stats = {
                    "total_generated_ids": len(self._used_ids),
                    "sequence_counters": dict(self._sequence_counters),
                    "startup_time": datetime.fromtimestamp(self._startup_time).isoformat(),
                    "uptime_seconds": int(time.time() - self._startup_time),
                    "memory_usage_estimate": len(self._used_ids) * 50,  # バイト概算
                }

                # シーケンスタイプ別統計
                sequence_stats = {}
                for seq_type, count in self._sequence_counters.items():
                    sequence_stats[seq_type] = {
                        "current_sequence": count,
                        "estimated_total": count,  # 簡易推定
                    }

                stats["sequence_statistics"] = sequence_stats

                return stats

        except Exception as e:
            logger.error(f"ID統計取得エラー: {e}")
            return {"error": str(e)}

    def cleanup_old_ids(self, max_ids: int = 100000) -> int:
        """
        古いID情報クリーンアップ

        Args:
            max_ids: 保持する最大ID数

        Returns:
            削除されたID数
        """
        try:
            with self._lock:
                if len(self._used_ids) <= max_ids:
                    return 0

                # 古いIDをクリア（実際の実装では使用頻度や時刻ベースで）
                ids_to_remove = len(self._used_ids) - max_ids

                # セット操作でランダムに削除（簡易実装）
                ids_list = list(self._used_ids)
                for _ in range(ids_to_remove):
                    if ids_list:
                        self._used_ids.discard(ids_list.pop(0))

                logger.info(f"ID履歴クリーンアップ: {ids_to_remove}件削除")
                return ids_to_remove

        except Exception as e:
            logger.error(f"IDクリーンアップエラー: {e}")
            return 0

    def reset_sequences(self) -> None:
        """シーケンスカウンターリセット"""
        try:
            with self._lock:
                sequence_count = len(self._sequence_counters)
                self._sequence_counters.clear()
                logger.info(f"シーケンスカウンターリセット: {sequence_count}種類")
        except Exception as e:
            logger.error(f"シーケンスリセットエラー: {e}")

    def export_id_history(self) -> Dict[str, any]:
        """
        ID履歴エクスポート

        Returns:
            ID履歴データ
        """
        try:
            with self._lock:
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_ids": len(self._used_ids),
                    "sequence_counters": dict(self._sequence_counters),
                    "used_ids": sorted(list(self._used_ids)),
                    "statistics": self.get_id_statistics(),
                }

                logger.info(f"ID履歴エクスポート: {len(self._used_ids)}件")
                return export_data

        except Exception as e:
            logger.error(f"ID履歴エクスポートエラー: {e}")
            return {"error": str(e)}
