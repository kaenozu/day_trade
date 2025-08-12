#!/usr/bin/env python3
"""
Cache Serializers
キャッシュシリアライザー

オブジェクトのシリアライゼーション・デシリアライゼーション
"""

import gzip
import json
import pickle
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from ..exceptions.risk_exceptions import CacheError
from ..interfaces.cache_interfaces import ICacheSerializer


class PickleSerializer(ICacheSerializer):
    """Pickleシリアライザー"""

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol

    def serialize(self, obj: Any) -> bytes:
        """オブジェクトをバイト列にシリアライズ"""
        try:
            return pickle.dumps(obj, protocol=self.protocol)
        except Exception as e:
            raise CacheError(
                f"Failed to pickle serialize object: {type(obj)}",
                operation="serialize",
                cause=e,
            )

    def deserialize(self, data: bytes) -> Any:
        """バイト列からオブジェクトにデシリアライズ"""
        try:
            return pickle.loads(data)
        except Exception as e:
            raise CacheError(
                "Failed to pickle deserialize data", operation="deserialize", cause=e
            )

    def get_content_type(self) -> str:
        """コンテンツタイプ取得"""
        return "application/pickle"


class JsonSerializer(ICacheSerializer):
    """JSONシリアライザー"""

    def __init__(self, ensure_ascii: bool = False, indent: Optional[int] = None):
        self.ensure_ascii = ensure_ascii
        self.indent = indent

    def serialize(self, obj: Any) -> bytes:
        """オブジェクトをJSONバイト列にシリアライズ"""
        try:
            json_str = json.dumps(
                obj,
                ensure_ascii=self.ensure_ascii,
                indent=self.indent,
                default=self._json_serializer,
                separators=(",", ":") if self.indent is None else None,
            )
            return json_str.encode("utf-8")
        except Exception as e:
            raise CacheError(
                f"Failed to JSON serialize object: {type(obj)}",
                operation="serialize",
                cause=e,
            )

    def deserialize(self, data: bytes) -> Any:
        """JSONバイト列からオブジェクトにデシリアライズ"""
        try:
            json_str = data.decode("utf-8")
            return json.loads(json_str, object_hook=self._json_deserializer)
        except Exception as e:
            raise CacheError(
                "Failed to JSON deserialize data", operation="deserialize", cause=e
            )

    def get_content_type(self) -> str:
        """コンテンツタイプ取得"""
        return "application/json"

    def _json_serializer(self, obj: Any) -> Any:
        """JSON シリアライゼーション用カスタムハンドラー"""
        if isinstance(obj, datetime):
            return {"__datetime__": True, "value": obj.isoformat()}
        elif isinstance(obj, Decimal):
            return {"__decimal__": True, "value": str(obj)}
        elif isinstance(obj, set):
            return {"__set__": True, "value": list(obj)}
        elif isinstance(obj, bytes):
            return {"__bytes__": True, "value": obj.hex()}
        elif hasattr(obj, "__dict__"):
            # データクラスや通常のオブジェクト
            return {
                "__object__": True,
                "class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                "data": obj.__dict__,
            }
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _json_deserializer(self, dct: Dict[str, Any]) -> Any:
        """JSON デシリアライゼーション用カスタムハンドラー"""
        if "__datetime__" in dct:
            return datetime.fromisoformat(dct["value"])
        elif "__decimal__" in dct:
            return Decimal(dct["value"])
        elif "__set__" in dct:
            return set(dct["value"])
        elif "__bytes__" in dct:
            return bytes.fromhex(dct["value"])
        elif "__object__" in dct:
            # オブジェクト復元（簡易版）
            class_path = dct["class"]
            if "." in class_path:
                module_name, class_name = class_path.rsplit(".", 1)
                try:
                    import importlib

                    module = importlib.import_module(module_name)
                    cls = getattr(module, class_name)
                    obj = cls.__new__(cls)
                    obj.__dict__.update(dct["data"])
                    return obj
                except (ImportError, AttributeError):
                    # クラスを復元できない場合は辞書として返す
                    return dct["data"]
            return dct["data"]
        return dct


class MsgPackSerializer(ICacheSerializer):
    """MessagePackシリアライザー"""

    def __init__(self):
        try:
            import msgpack

            self.msgpack = msgpack
        except ImportError:
            raise CacheError(
                "msgpack is not available. Please install: pip install msgpack",
                operation="init",
            )

    def serialize(self, obj: Any) -> bytes:
        """オブジェクトをMessagePackバイト列にシリアライズ"""
        try:
            return self.msgpack.packb(
                obj, default=self._msgpack_serializer, use_bin_type=True
            )
        except Exception as e:
            raise CacheError(
                f"Failed to msgpack serialize object: {type(obj)}",
                operation="serialize",
                cause=e,
            )

    def deserialize(self, data: bytes) -> Any:
        """MessagePackバイト列からオブジェクトにデシリアライズ"""
        try:
            return self.msgpack.unpackb(
                data, object_hook=self._msgpack_deserializer, raw=False
            )
        except Exception as e:
            raise CacheError(
                "Failed to msgpack deserialize data", operation="deserialize", cause=e
            )

    def get_content_type(self) -> str:
        """コンテンツタイプ取得"""
        return "application/msgpack"

    def _msgpack_serializer(self, obj: Any) -> Any:
        """MessagePack シリアライゼーション用カスタムハンドラー"""
        if isinstance(obj, datetime):
            return {"__datetime__": obj.isoformat()}
        elif isinstance(obj, Decimal):
            return {"__decimal__": str(obj)}
        elif isinstance(obj, set):
            return {"__set__": list(obj)}
        else:
            return obj

    def _msgpack_deserializer(self, code: int, obj: Any) -> Any:
        """MessagePack デシリアライゼーション用カスタムハンドラー"""
        if isinstance(obj, dict):
            if "__datetime__" in obj:
                return datetime.fromisoformat(obj["__datetime__"])
            elif "__decimal__" in obj:
                return Decimal(obj["__decimal__"])
            elif "__set__" in obj:
                return set(obj["__set__"])
        return obj


class CompressionSerializer(ICacheSerializer):
    """圧縮シリアライザーラッパー"""

    def __init__(self, base_serializer: ICacheSerializer, compression: str = "gzip"):
        self.base_serializer = base_serializer
        self.compression = compression.lower()

        if self.compression == "gzip":
            self._compress = gzip.compress
            self._decompress = gzip.decompress
        elif self.compression == "lz4":
            try:
                import lz4.frame

                self._compress = lz4.frame.compress
                self._decompress = lz4.frame.decompress
            except ImportError:
                raise CacheError(
                    "lz4 is not available. Please install: pip install lz4",
                    operation="init",
                )
        else:
            raise CacheError(
                f"Unsupported compression type: {compression}", operation="init"
            )

    def serialize(self, obj: Any) -> bytes:
        """オブジェクトを圧縮バイト列にシリアライズ"""
        try:
            serialized_data = self.base_serializer.serialize(obj)
            compressed_data = self._compress(serialized_data)

            # 圧縮ヘッダー付加（圧縮タイプと元のサイズ）
            header = f"{self.compression}:{len(serialized_data)}:".encode()
            return header + compressed_data

        except Exception as e:
            raise CacheError(
                f"Failed to compress serialize object: {type(obj)}",
                operation="serialize",
                cause=e,
            )

    def deserialize(self, data: bytes) -> Any:
        """圧縮バイト列からオブジェクトにデシリアライズ"""
        try:
            # ヘッダー解析
            header_end = data.find(b":", data.find(b":") + 1) + 1
            header = data[:header_end].decode("utf-8")
            compressed_data = data[header_end:]

            compression_type, original_size = header.rstrip(":").split(":")

            if compression_type != self.compression:
                raise ValueError(
                    f"Compression type mismatch: expected {self.compression}, got {compression_type}"
                )

            # 解凍
            decompressed_data = self._decompress(compressed_data)

            # 元のサイズ検証
            if len(decompressed_data) != int(original_size):
                raise ValueError("Decompressed data size mismatch")

            return self.base_serializer.deserialize(decompressed_data)

        except Exception as e:
            raise CacheError(
                "Failed to decompress deserialize data",
                operation="deserialize",
                cause=e,
            )

    def get_content_type(self) -> str:
        """コンテンツタイプ取得"""
        base_content_type = self.base_serializer.get_content_type()
        return f"{base_content_type}+{self.compression}"


class EncryptionSerializer(ICacheSerializer):
    """暗号化シリアライザーラッパー"""

    def __init__(self, base_serializer: ICacheSerializer, encryption_key: str):
        self.base_serializer = base_serializer
        self.encryption_key = encryption_key.encode("utf-8")

        try:
            import base64
            import os

            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

            # キー派生
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"risk_cache_salt",  # 実際の環境では動的salt使用を推奨
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.encryption_key))
            self.fernet = Fernet(key)

        except ImportError:
            raise CacheError(
                "cryptography is not available. Please install: pip install cryptography",
                operation="init",
            )

    def serialize(self, obj: Any) -> bytes:
        """オブジェクトを暗号化バイト列にシリアライズ"""
        try:
            serialized_data = self.base_serializer.serialize(obj)
            encrypted_data = self.fernet.encrypt(serialized_data)
            return encrypted_data
        except Exception as e:
            raise CacheError(
                f"Failed to encrypt serialize object: {type(obj)}",
                operation="serialize",
                cause=e,
            )

    def deserialize(self, data: bytes) -> Any:
        """暗号化バイト列からオブジェクトにデシリアライズ"""
        try:
            decrypted_data = self.fernet.decrypt(data)
            return self.base_serializer.deserialize(decrypted_data)
        except Exception as e:
            raise CacheError(
                "Failed to decrypt deserialize data", operation="deserialize", cause=e
            )

    def get_content_type(self) -> str:
        """コンテンツタイプ取得"""
        base_content_type = self.base_serializer.get_content_type()
        return f"{base_content_type}+encrypted"


class TypedSerializer(ICacheSerializer):
    """型情報付きシリアライザー"""

    def __init__(self, base_serializer: ICacheSerializer):
        self.base_serializer = base_serializer

    def serialize(self, obj: Any) -> bytes:
        """型情報付きでオブジェクトをシリアライズ"""
        try:
            # 型情報とデータをセットで保存
            typed_data = {
                "type": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                "data": obj,
            }
            return self.base_serializer.serialize(typed_data)
        except Exception as e:
            raise CacheError(
                f"Failed to typed serialize object: {type(obj)}",
                operation="serialize",
                cause=e,
            )

    def deserialize(self, data: bytes) -> Any:
        """型情報を使用してオブジェクトをデシリアライズ"""
        try:
            typed_data = self.base_serializer.deserialize(data)

            if (
                isinstance(typed_data, dict)
                and "type" in typed_data
                and "data" in typed_data
            ):
                # 型検証（オプション）
                expected_type = typed_data["type"]
                actual_data = typed_data["data"]

                # 型情報をログ記録（デバッグ用）
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Deserializing object of type: {expected_type}")

                return actual_data
            else:
                # 型情報がない場合はそのまま返す
                return typed_data

        except Exception as e:
            raise CacheError(
                "Failed to typed deserialize data", operation="deserialize", cause=e
            )

    def get_content_type(self) -> str:
        """コンテンツタイプ取得"""
        base_content_type = self.base_serializer.get_content_type()
        return f"{base_content_type}+typed"


# シリアライザーファクトリー


def create_serializer(
    serializer_type: str,
    compression: Optional[str] = None,
    encryption_key: Optional[str] = None,
    typed: bool = False,
    **kwargs,
) -> ICacheSerializer:
    """シリアライザー作成"""

    # ベースシリアライザー作成
    if serializer_type.lower() == "pickle":
        base_serializer = PickleSerializer(**kwargs)
    elif serializer_type.lower() == "json":
        base_serializer = JsonSerializer(**kwargs)
    elif serializer_type.lower() == "msgpack":
        base_serializer = MsgPackSerializer()
    else:
        raise CacheError(
            f"Unsupported serializer type: {serializer_type}",
            operation="create_serializer",
        )

    # ラッパー適用
    serializer = base_serializer

    if typed:
        serializer = TypedSerializer(serializer)

    if compression:
        serializer = CompressionSerializer(serializer, compression)

    if encryption_key:
        serializer = EncryptionSerializer(serializer, encryption_key)

    return serializer
