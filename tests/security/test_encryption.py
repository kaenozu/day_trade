"""
Encryption module tests
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import base64
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Dict, Any, Optional, Union, Tuple, List
import hashlib
import hmac
import json


class MockSymmetricEncryption:
    """Mock symmetric encryption for testing"""
    
    def __init__(self, key: bytes = None):
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self.encrypted_data_cache = {}
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data using symmetric encryption"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        try:
            encrypted = self.cipher.encrypt(data)
            # Cache for testing purposes
            self.encrypted_data_cache[encrypted] = data
            return encrypted
        except Exception as e:
            raise RuntimeError(f"Encryption failed: {e}")
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using symmetric encryption"""
        try:
            return self.cipher.decrypt(encrypted_data)
        except Exception as e:
            raise RuntimeError(f"Decryption failed: {e}")
    
    def encrypt_string(self, text: str) -> str:
        """Encrypt string and return base64 encoded result"""
        encrypted_bytes = self.encrypt(text)
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """Decrypt base64 encoded string"""
        encrypted_bytes = base64.b64decode(encrypted_text.encode('utf-8'))
        decrypted_bytes = self.decrypt(encrypted_bytes)
        return decrypted_bytes.decode('utf-8')
    
    def encrypt_dict(self, data_dict: Dict[str, Any]) -> str:
        """Encrypt dictionary as JSON"""
        json_string = json.dumps(data_dict, ensure_ascii=False)
        return self.encrypt_string(json_string)
    
    def decrypt_dict(self, encrypted_json: str) -> Dict[str, Any]:
        """Decrypt and parse JSON dictionary"""
        json_string = self.decrypt_string(encrypted_json)
        return json.loads(json_string)
    
    def get_key(self) -> bytes:
        """Get encryption key"""
        return self.key
    
    def rotate_key(self) -> bytes:
        """Generate new encryption key"""
        old_key = self.key
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        return old_key


class MockAsymmetricEncryption:
    """Mock asymmetric (RSA) encryption for testing"""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        self.public_key = self.private_key.public_key()
        self.max_message_length = (key_size // 8) - 42  # OAEP padding overhead
    
    def encrypt_with_public_key(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data with public key"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if len(data) > self.max_message_length:
            raise ValueError(f"Message too long for RSA encryption: {len(data)} > {self.max_message_length}")
        
        try:
            encrypted = self.public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return encrypted
        except Exception as e:
            raise RuntimeError(f"RSA encryption failed: {e}")
    
    def decrypt_with_private_key(self, encrypted_data: bytes) -> bytes:
        """Decrypt data with private key"""
        try:
            decrypted = self.private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted
        except Exception as e:
            raise RuntimeError(f"RSA decryption failed: {e}")
    
    def sign_data(self, data: Union[str, bytes]) -> bytes:
        """Sign data with private key"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        try:
            signature = self.private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature
        except Exception as e:
            raise RuntimeError(f"Signing failed: {e}")
    
    def verify_signature(self, data: Union[str, bytes], signature: bytes) -> bool:
        """Verify signature with public key"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        try:
            self.public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def get_public_key_pem(self) -> str:
        """Get public key in PEM format"""
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode('utf-8')
    
    def get_private_key_pem(self, password: bytes = None) -> str:
        """Get private key in PEM format"""
        encryption_algorithm = (
            serialization.BestAvailableEncryption(password)
            if password else serialization.NoEncryption()
        )
        
        pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption_algorithm
        )
        return pem.decode('utf-8')


class MockHashingService:
    """Mock hashing service for testing"""
    
    def __init__(self):
        self.salt_length = 32
        self.hash_iterations = 100000
        self.supported_algorithms = ['sha256', 'sha512', 'blake2b']
    
    def hash_data(self, data: Union[str, bytes], algorithm: str = 'sha256') -> str:
        """Hash data with specified algorithm"""
        if algorithm not in self.supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == 'sha256':
            hash_obj = hashlib.sha256(data)
        elif algorithm == 'sha512':
            hash_obj = hashlib.sha512(data)
        elif algorithm == 'blake2b':
            hash_obj = hashlib.blake2b(data)
        
        return hash_obj.hexdigest()
    
    def hash_password(self, password: str, salt: bytes = None) -> Tuple[str, bytes]:
        """Hash password with salt using PBKDF2"""
        if salt is None:
            salt = os.urandom(self.salt_length)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.hash_iterations
        )
        
        password_bytes = password.encode('utf-8')
        key = kdf.derive(password_bytes)
        hash_hex = key.hex()
        
        return hash_hex, salt
    
    def verify_password(self, password: str, hash_hex: str, salt: bytes) -> bool:
        """Verify password against hash"""
        try:
            computed_hash, _ = self.hash_password(password, salt)
            return hmac.compare_digest(hash_hex, computed_hash)
        except Exception:
            return False
    
    def hmac_sign(self, data: Union[str, bytes], key: Union[str, bytes]) -> str:
        """Create HMAC signature"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        if isinstance(key, str):
            key = key.encode('utf-8')
        
        signature = hmac.new(key, data, hashlib.sha256)
        return signature.hexdigest()
    
    def hmac_verify(self, data: Union[str, bytes], signature: str, key: Union[str, bytes]) -> bool:
        """Verify HMAC signature"""
        try:
            expected_signature = self.hmac_sign(data, key)
            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False
    
    def generate_checksum(self, file_path: str = None, data: bytes = None) -> str:
        """Generate file or data checksum"""
        if file_path and data:
            raise ValueError("Provide either file_path or data, not both")
        
        if file_path:
            # Mock file reading
            data = f"mock_file_content_{file_path}".encode('utf-8')
        elif data is None:
            raise ValueError("Must provide either file_path or data")
        
        return hashlib.sha256(data).hexdigest()


class MockKeyManager:
    """Mock key management service for testing"""
    
    def __init__(self):
        self.keys = {}
        self.key_metadata = {}
        self.key_versions = {}
        self.master_key = Fernet.generate_key()
    
    def generate_key(self, key_id: str, key_type: str = 'symmetric') -> str:
        """Generate and store encryption key"""
        if key_type == 'symmetric':
            key = Fernet.generate_key()
        elif key_type == 'asymmetric':
            # For asymmetric, store key pair
            rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            key = {
                'private': rsa_key,
                'public': rsa_key.public_key()
            }
        else:
            raise ValueError(f"Unsupported key type: {key_type}")
        
        # Encrypt key with master key for storage
        if key_type == 'symmetric':
            encrypted_key = Fernet(self.master_key).encrypt(key)
        else:
            # For asymmetric, serialize and encrypt private key
            private_pem = rsa_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            encrypted_key = Fernet(self.master_key).encrypt(private_pem)
        
        self.keys[key_id] = encrypted_key
        self.key_metadata[key_id] = {
            'type': key_type,
            'created_at': secrets.token_hex(8),  # Mock timestamp
            'version': 1,
            'active': True
        }
        self.key_versions[key_id] = [1]
        
        return key_id
    
    def get_key(self, key_id: str, version: int = None) -> Optional[Union[bytes, Dict[str, Any]]]:
        """Retrieve decrypted key"""
        if key_id not in self.keys:
            return None
        
        if version and version not in self.key_versions.get(key_id, []):
            return None
        
        metadata = self.key_metadata[key_id]
        encrypted_key = self.keys[key_id]
        
        try:
            if metadata['type'] == 'symmetric':
                # Decrypt symmetric key
                decrypted_key = Fernet(self.master_key).decrypt(encrypted_key)
                return decrypted_key
            else:
                # Decrypt and deserialize asymmetric key
                decrypted_pem = Fernet(self.master_key).decrypt(encrypted_key)
                private_key = serialization.load_pem_private_key(decrypted_pem, password=None)
                return {
                    'private': private_key,
                    'public': private_key.public_key()
                }
        except Exception:
            return None
    
    def rotate_key(self, key_id: str) -> bool:
        """Rotate (generate new version of) key"""
        if key_id not in self.keys:
            return False
        
        metadata = self.key_metadata[key_id]
        key_type = metadata['type']
        
        # Generate new key version
        new_version = max(self.key_versions[key_id]) + 1
        
        if key_type == 'symmetric':
            new_key = Fernet.generate_key()
            encrypted_key = Fernet(self.master_key).encrypt(new_key)
        else:
            rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            private_pem = rsa_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            encrypted_key = Fernet(self.master_key).encrypt(private_pem)
        
        # Store new version
        self.keys[f"{key_id}_v{new_version}"] = encrypted_key
        self.key_versions[key_id].append(new_version)
        metadata['version'] = new_version
        
        return True
    
    def deactivate_key(self, key_id: str) -> bool:
        """Deactivate key (mark as inactive)"""
        if key_id in self.key_metadata:
            self.key_metadata[key_id]['active'] = False
            return True
        return False
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all keys with metadata"""
        key_list = []
        for key_id, metadata in self.key_metadata.items():
            key_info = {
                'key_id': key_id,
                **metadata
            }
            key_list.append(key_info)
        return key_list
    
    def backup_keys(self) -> Dict[str, Any]:
        """Create backup of all keys"""
        return {
            'keys': self.keys.copy(),
            'metadata': self.key_metadata.copy(),
            'versions': self.key_versions.copy(),
            'master_key': base64.b64encode(self.master_key).decode('utf-8')
        }
    
    def restore_keys(self, backup_data: Dict[str, Any]) -> bool:
        """Restore keys from backup"""
        try:
            self.keys = backup_data['keys'].copy()
            self.key_metadata = backup_data['metadata'].copy()
            self.key_versions = backup_data['versions'].copy()
            self.master_key = base64.b64decode(backup_data['master_key'])
            return True
        except Exception:
            return False


class MockSecureStorage:
    """Mock secure storage for testing"""
    
    def __init__(self, encryption_key: bytes = None):
        self.encryption = MockSymmetricEncryption(encryption_key)
        self.storage = {}
        self.access_log = []
    
    def store_data(self, key: str, data: Union[str, bytes, Dict[str, Any]]) -> bool:
        """Store encrypted data"""
        try:
            if isinstance(data, dict):
                encrypted_data = self.encryption.encrypt_dict(data)
            elif isinstance(data, str):
                encrypted_data = self.encryption.encrypt_string(data)
            else:
                encrypted_data = base64.b64encode(self.encryption.encrypt(data)).decode('utf-8')
            
            self.storage[key] = {
                'data': encrypted_data,
                'type': type(data).__name__,
                'timestamp': secrets.token_hex(8)  # Mock timestamp
            }
            
            self._log_access(key, 'store')
            return True
        except Exception:
            return False
    
    def retrieve_data(self, key: str) -> Optional[Union[str, bytes, Dict[str, Any]]]:
        """Retrieve and decrypt data"""
        if key not in self.storage:
            return None
        
        try:
            stored_item = self.storage[key]
            encrypted_data = stored_item['data']
            data_type = stored_item['type']
            
            if data_type == 'dict':
                decrypted_data = self.encryption.decrypt_dict(encrypted_data)
            elif data_type == 'str':
                decrypted_data = self.encryption.decrypt_string(encrypted_data)
            else:
                encrypted_bytes = base64.b64decode(encrypted_data)
                decrypted_data = self.encryption.decrypt(encrypted_bytes)
            
            self._log_access(key, 'retrieve')
            return decrypted_data
        except Exception:
            return None
    
    def delete_data(self, key: str) -> bool:
        """Delete data"""
        if key in self.storage:
            del self.storage[key]
            self._log_access(key, 'delete')
            return True
        return False
    
    def list_keys(self) -> List[str]:
        """List all stored keys"""
        return list(self.storage.keys())
    
    def get_access_log(self, key: str = None) -> List[Dict[str, Any]]:
        """Get access log"""
        if key:
            return [log for log in self.access_log if log['key'] == key]
        return self.access_log.copy()
    
    def _log_access(self, key: str, action: str) -> None:
        """Log data access"""
        self.access_log.append({
            'key': key,
            'action': action,
            'timestamp': secrets.token_hex(8)  # Mock timestamp
        })
    
    def secure_wipe(self, key: str) -> bool:
        """Securely wipe data (overwrite with random data)"""
        if key in self.storage:
            # Overwrite with random data before deletion
            random_data = secrets.token_bytes(1024)
            self.storage[key]['data'] = base64.b64encode(random_data).decode('utf-8')
            del self.storage[key]
            self._log_access(key, 'secure_wipe')
            return True
        return False


class TestSymmetricEncryption:
    """Test symmetric encryption functionality"""
    
    def test_basic_encryption_decryption(self):
        enc = MockSymmetricEncryption()
        
        # Test with string
        original_text = "Hello, World! This is a test message."
        encrypted = enc.encrypt(original_text)
        decrypted = enc.decrypt(encrypted)
        
        assert decrypted.decode('utf-8') == original_text
        assert encrypted != original_text.encode('utf-8')
    
    def test_string_encryption(self):
        enc = MockSymmetricEncryption()
        
        original = "Sensitive trading data: AAPL $150.25"
        encrypted = enc.encrypt_string(original)
        decrypted = enc.decrypt_string(encrypted)
        
        assert decrypted == original
        assert encrypted != original
        assert len(encrypted) > len(original)  # Base64 encoded
    
    def test_dictionary_encryption(self):
        enc = MockSymmetricEncryption()
        
        original_dict = {
            "symbol": "AAPL",
            "price": 150.25,
            "quantity": 100,
            "metadata": {
                "timestamp": "2023-01-01T12:00:00Z",
                "source": "API"
            }
        }
        
        encrypted = enc.encrypt_dict(original_dict)
        decrypted = enc.decrypt_dict(encrypted)
        
        assert decrypted == original_dict
        assert isinstance(encrypted, str)
    
    def test_key_rotation(self):
        enc = MockSymmetricEncryption()
        
        original_text = "Test message"
        
        # Encrypt with first key
        encrypted_v1 = enc.encrypt_string(original_text)
        
        # Rotate key
        old_key = enc.rotate_key()
        
        # Old encrypted data should still decrypt with old key
        old_enc = MockSymmetricEncryption(old_key)
        assert old_enc.decrypt_string(encrypted_v1) == original_text
        
        # New encryption with new key
        encrypted_v2 = enc.encrypt_string(original_text)
        assert enc.decrypt_string(encrypted_v2) == original_text
        
        # Old and new encrypted data should be different
        assert encrypted_v1 != encrypted_v2
    
    def test_encryption_error_handling(self):
        enc = MockSymmetricEncryption()
        
        # Test decryption with invalid data
        with pytest.raises(RuntimeError):
            enc.decrypt(b"invalid_encrypted_data")
        
        with pytest.raises(Exception):  # Should fail on invalid base64
            enc.decrypt_string("invalid_base64!")


class TestAsymmetricEncryption:
    """Test asymmetric (RSA) encryption functionality"""
    
    def test_rsa_encryption_decryption(self):
        rsa_enc = MockAsymmetricEncryption()
        
        message = "Secret trading algorithm parameters"
        
        # Encrypt with public key
        encrypted = rsa_enc.encrypt_with_public_key(message)
        
        # Decrypt with private key
        decrypted = rsa_enc.decrypt_with_private_key(encrypted)
        
        assert decrypted.decode('utf-8') == message
        assert encrypted != message.encode('utf-8')
    
    def test_message_size_limits(self):
        rsa_enc = MockAsymmetricEncryption(key_size=2048)
        
        # Message too large for RSA encryption
        large_message = "x" * (rsa_enc.max_message_length + 1)
        
        with pytest.raises(ValueError):
            rsa_enc.encrypt_with_public_key(large_message)
    
    def test_digital_signatures(self):
        rsa_enc = MockAsymmetricEncryption()
        
        message = "Important trading order: Buy 100 AAPL at $150"
        
        # Sign message
        signature = rsa_enc.sign_data(message)
        
        # Verify signature
        assert rsa_enc.verify_signature(message, signature) == True
        
        # Tampered message should fail verification
        tampered_message = "Important trading order: Buy 1000 AAPL at $150"
        assert rsa_enc.verify_signature(tampered_message, signature) == False
    
    def test_key_serialization(self):
        rsa_enc = MockAsymmetricEncryption()
        
        # Get PEM formatted keys
        public_pem = rsa_enc.get_public_key_pem()
        private_pem = rsa_enc.get_private_key_pem()
        
        assert public_pem.startswith("-----BEGIN PUBLIC KEY-----")
        assert private_pem.startswith("-----BEGIN PRIVATE KEY-----")
        
        # Test password-protected private key
        password = b"test_password"
        protected_private_pem = rsa_enc.get_private_key_pem(password)
        assert "BEGIN ENCRYPTED PRIVATE KEY" in protected_private_pem


class TestHashingService:
    """Test hashing service functionality"""
    
    def test_data_hashing(self):
        hasher = MockHashingService()
        
        data = "Trading data to hash"
        
        # Test different algorithms
        sha256_hash = hasher.hash_data(data, 'sha256')
        sha512_hash = hasher.hash_data(data, 'sha512')
        blake2b_hash = hasher.hash_data(data, 'blake2b')
        
        assert len(sha256_hash) == 64  # SHA256 produces 32 bytes = 64 hex chars
        assert len(sha512_hash) == 128  # SHA512 produces 64 bytes = 128 hex chars
        assert len(blake2b_hash) == 128  # Blake2b produces 64 bytes = 128 hex chars
        
        # Same data should produce same hash
        assert hasher.hash_data(data, 'sha256') == sha256_hash
        
        # Different data should produce different hash
        assert hasher.hash_data("Different data", 'sha256') != sha256_hash
    
    def test_password_hashing(self):
        hasher = MockHashingService()
        
        password = "SecurePassword123!"
        
        # Hash password
        hash_hex, salt = hasher.hash_password(password)
        
        assert len(hash_hex) == 64  # 32 bytes = 64 hex chars
        assert len(salt) == hasher.salt_length
        
        # Verify password
        assert hasher.verify_password(password, hash_hex, salt) == True
        assert hasher.verify_password("WrongPassword", hash_hex, salt) == False
        
        # Same password with different salt should produce different hash
        hash_hex2, salt2 = hasher.hash_password(password)
        assert hash_hex != hash_hex2
        assert salt != salt2
    
    def test_hmac_operations(self):
        hasher = MockHashingService()
        
        data = "Important trading message"
        key = "secret_hmac_key"
        
        # Create HMAC signature
        signature = hasher.hmac_sign(data, key)
        
        assert len(signature) == 64  # SHA256 HMAC = 32 bytes = 64 hex chars
        
        # Verify signature
        assert hasher.hmac_verify(data, signature, key) == True
        assert hasher.hmac_verify("Tampered message", signature, key) == False
        assert hasher.hmac_verify(data, signature, "wrong_key") == False
    
    def test_checksum_generation(self):
        hasher = MockHashingService()
        
        # Test with data
        test_data = b"File content for checksum"
        checksum = hasher.generate_checksum(data=test_data)
        
        assert len(checksum) == 64  # SHA256 = 64 hex chars
        
        # Test with file path (mock)
        file_checksum = hasher.generate_checksum(file_path="test_file.txt")
        assert len(file_checksum) == 64
        
        # Error cases
        with pytest.raises(ValueError):
            hasher.generate_checksum()  # No data or file path
        
        with pytest.raises(ValueError):
            hasher.generate_checksum(file_path="test.txt", data=b"data")  # Both provided


class TestKeyManager:
    """Test key management functionality"""
    
    def test_key_generation(self):
        km = MockKeyManager()
        
        # Generate symmetric key
        key_id = km.generate_key("test_symmetric", "symmetric")
        assert key_id == "test_symmetric"
        assert key_id in km.keys
        
        # Generate asymmetric key
        key_id2 = km.generate_key("test_asymmetric", "asymmetric")
        assert key_id2 == "test_asymmetric"
        
        # Check metadata
        metadata = km.key_metadata[key_id]
        assert metadata['type'] == 'symmetric'
        assert metadata['active'] == True
        assert metadata['version'] == 1
    
    def test_key_retrieval(self):
        km = MockKeyManager()
        
        # Generate and retrieve symmetric key
        key_id = km.generate_key("test_key", "symmetric")
        retrieved_key = km.get_key(key_id)
        
        assert retrieved_key is not None
        assert isinstance(retrieved_key, bytes)
        
        # Generate and retrieve asymmetric key
        asym_key_id = km.generate_key("test_asym", "asymmetric")
        retrieved_asym_key = km.get_key(asym_key_id)
        
        assert retrieved_asym_key is not None
        assert isinstance(retrieved_asym_key, dict)
        assert 'private' in retrieved_asym_key
        assert 'public' in retrieved_asym_key
    
    def test_key_rotation(self):
        km = MockKeyManager()
        
        key_id = km.generate_key("rotatable_key", "symmetric")
        original_key = km.get_key(key_id)
        
        # Rotate key
        assert km.rotate_key(key_id) == True
        
        # Metadata should show new version
        metadata = km.key_metadata[key_id]
        assert metadata['version'] == 2
        assert 2 in km.key_versions[key_id]
        
        # Should be able to get specific version
        # (In real implementation, would specify version parameter)
    
    def test_key_lifecycle(self):
        km = MockKeyManager()
        
        key_id = km.generate_key("lifecycle_key", "symmetric")
        
        # Key should be active initially
        metadata = km.key_metadata[key_id]
        assert metadata['active'] == True
        
        # Deactivate key
        assert km.deactivate_key(key_id) == True
        assert km.key_metadata[key_id]['active'] == False
        
        # Non-existent key
        assert km.deactivate_key("non_existent") == False
    
    def test_key_listing(self):
        km = MockKeyManager()
        
        # Generate multiple keys
        km.generate_key("key1", "symmetric")
        km.generate_key("key2", "asymmetric")
        km.generate_key("key3", "symmetric")
        
        key_list = km.list_keys()
        
        assert len(key_list) == 3
        key_ids = [k['key_id'] for k in key_list]
        assert "key1" in key_ids
        assert "key2" in key_ids
        assert "key3" in key_ids
    
    def test_key_backup_restore(self):
        km = MockKeyManager()
        
        # Generate some keys
        km.generate_key("backup_key1", "symmetric")
        km.generate_key("backup_key2", "asymmetric")
        
        # Create backup
        backup = km.backup_keys()
        
        assert 'keys' in backup
        assert 'metadata' in backup
        assert 'versions' in backup
        assert 'master_key' in backup
        
        # Create new key manager and restore
        km2 = MockKeyManager()
        assert km2.restore_keys(backup) == True
        
        # Should have same keys
        assert len(km2.list_keys()) == 2
        assert "backup_key1" in km2.keys
        assert "backup_key2" in km2.keys


class TestSecureStorage:
    """Test secure storage functionality"""
    
    def test_basic_storage_operations(self):
        storage = MockSecureStorage()
        
        # Store different types of data
        assert storage.store_data("string_key", "Secret string data") == True
        assert storage.store_data("dict_key", {"secret": "value", "amount": 1000}) == True
        assert storage.store_data("bytes_key", b"Binary secret data") == True
        
        # Retrieve data
        string_data = storage.retrieve_data("string_key")
        dict_data = storage.retrieve_data("dict_key")
        bytes_data = storage.retrieve_data("bytes_key")
        
        assert string_data == "Secret string data"
        assert dict_data == {"secret": "value", "amount": 1000}
        assert bytes_data == b"Binary secret data"
    
    def test_data_not_found(self):
        storage = MockSecureStorage()
        
        # Non-existent key
        assert storage.retrieve_data("non_existent") is None
    
    def test_data_deletion(self):
        storage = MockSecureStorage()
        
        storage.store_data("delete_me", "Data to delete")
        
        # Verify data exists
        assert storage.retrieve_data("delete_me") == "Data to delete"
        
        # Delete data
        assert storage.delete_data("delete_me") == True
        
        # Data should be gone
        assert storage.retrieve_data("delete_me") is None
        
        # Deleting non-existent data should return False
        assert storage.delete_data("non_existent") == False
    
    def test_key_listing(self):
        storage = MockSecureStorage()
        
        storage.store_data("key1", "data1")
        storage.store_data("key2", "data2")
        storage.store_data("key3", "data3")
        
        keys = storage.list_keys()
        
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys
    
    def test_access_logging(self):
        storage = MockSecureStorage()
        
        storage.store_data("logged_key", "logged_data")
        storage.retrieve_data("logged_key")
        storage.delete_data("logged_key")
        
        # Check access log
        full_log = storage.get_access_log()
        assert len(full_log) == 3
        
        key_log = storage.get_access_log("logged_key")
        assert len(key_log) == 3
        
        actions = [log['action'] for log in key_log]
        assert "store" in actions
        assert "retrieve" in actions
        assert "delete" in actions
    
    def test_secure_wipe(self):
        storage = MockSecureStorage()
        
        storage.store_data("wipe_key", "sensitive_data")
        
        # Secure wipe
        assert storage.secure_wipe("wipe_key") == True
        
        # Data should be gone
        assert storage.retrieve_data("wipe_key") is None
        
        # Should be logged
        log = storage.get_access_log("wipe_key")
        actions = [l['action'] for l in log]
        assert "secure_wipe" in actions


class TestEncryptionIntegration:
    """Test encryption system integration scenarios"""
    
    def test_hybrid_encryption(self):
        """Test hybrid encryption (RSA + AES)"""
        # Use RSA for key exchange, AES for data encryption
        rsa_enc = MockAsymmetricEncryption()
        
        # Generate AES key
        aes_key = Fernet.generate_key()
        
        # Encrypt AES key with RSA
        encrypted_aes_key = rsa_enc.encrypt_with_public_key(aes_key)
        
        # Use AES to encrypt large data
        aes_enc = MockSymmetricEncryption(aes_key)
        large_data = "Large trading data that exceeds RSA limits: " + "x" * 1000
        encrypted_data = aes_enc.encrypt_string(large_data)
        
        # Decrypt process
        # 1. Decrypt AES key with RSA
        decrypted_aes_key = rsa_enc.decrypt_with_private_key(encrypted_aes_key)
        
        # 2. Use decrypted AES key to decrypt data
        aes_dec = MockSymmetricEncryption(decrypted_aes_key)
        decrypted_data = aes_dec.decrypt_string(encrypted_data)
        
        assert decrypted_data == large_data
    
    def test_encrypted_storage_with_key_manager(self):
        """Test integration of secure storage with key manager"""
        km = MockKeyManager()
        
        # Generate encryption key
        key_id = km.generate_key("storage_key", "symmetric")
        encryption_key = km.get_key(key_id)
        
        # Create secure storage with managed key
        storage = MockSecureStorage(encryption_key)
        
        # Store sensitive trading data
        trading_data = {
            "strategy": "momentum_trading",
            "positions": [
                {"symbol": "AAPL", "quantity": 100, "price": 150.25},
                {"symbol": "GOOGL", "quantity": 50, "price": 2500.00}
            ],
            "total_value": 140012.5
        }
        
        storage.store_data("trading_positions", trading_data)
        
        # Retrieve and verify
        retrieved_data = storage.retrieve_data("trading_positions")
        assert retrieved_data == trading_data
        
        # Key rotation test
        old_key = km.rotate_key(key_id)
        
        # Old data should still be accessible with old key
        old_storage = MockSecureStorage(old_key)
        # In real implementation, would need key versioning in storage
    
    def test_end_to_end_data_protection(self):
        """Test complete data protection workflow"""
        # Setup components
        hasher = MockHashingService()
        rsa_enc = MockAsymmetricEncryption()
        km = MockKeyManager()
        
        # Original sensitive data
        sensitive_data = {
            "user_id": "trader_001",
            "api_key": "sk-1234567890abcdef",
            "account_balance": 100000.00,
            "trading_history": [
                {"date": "2023-01-01", "symbol": "AAPL", "profit": 250.00},
                {"date": "2023-01-02", "symbol": "GOOGL", "profit": -150.00}
            ]
        }
        
        # 1. Create hash of original data for integrity check
        original_json = json.dumps(sensitive_data, sort_keys=True)
        data_hash = hasher.hash_data(original_json)
        
        # 2. Encrypt data with symmetric key
        sym_key_id = km.generate_key("data_encryption_key", "symmetric")
        sym_key = km.get_key(sym_key_id)
        sym_enc = MockSymmetricEncryption(sym_key)
        encrypted_data = sym_enc.encrypt_dict(sensitive_data)
        
        # 3. Sign encrypted data for authenticity
        signature = rsa_enc.sign_data(encrypted_data)
        
        # 4. Create secure package
        secure_package = {
            "encrypted_data": encrypted_data,
            "signature": base64.b64encode(signature).decode('utf-8'),
            "data_hash": data_hash,
            "key_id": sym_key_id
        }
        
        # Verification process
        # 1. Verify signature
        package_signature = base64.b64decode(secure_package["signature"])
        assert rsa_enc.verify_signature(secure_package["encrypted_data"], package_signature) == True
        
        # 2. Decrypt data
        decryption_key = km.get_key(secure_package["key_id"])
        dec_enc = MockSymmetricEncryption(decryption_key)
        decrypted_data = dec_enc.decrypt_dict(secure_package["encrypted_data"])
        
        # 3. Verify data integrity
        decrypted_json = json.dumps(decrypted_data, sort_keys=True)
        verified_hash = hasher.hash_data(decrypted_json)
        assert verified_hash == secure_package["data_hash"]
        
        # 4. Verify content
        assert decrypted_data == sensitive_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])