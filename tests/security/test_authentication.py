"""
Authentication module tests
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import secrets


class MockPasswordManager:
    """Mock password manager for testing"""
    
    def __init__(self):
        self.salt_rounds = 12
        self.password_history = {}
        self.failed_attempts = {}
        self.lockout_duration = 900  # 15 minutes
    
    def hash_password(self, password: str, salt: str = None) -> str:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Mock bcrypt-style hashing
        password_bytes = password.encode('utf-8')
        salt_bytes = salt.encode('utf-8')
        
        # Simple hash for testing (not secure in production)
        combined = password_bytes + salt_bytes
        hash_result = hashlib.pbkdf2_hmac('sha256', combined, salt_bytes, 100000)
        
        return f"$pbkdf2${salt}${hash_result.hex()}"
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            parts = hashed_password.split('$')
            if len(parts) != 4 or parts[0] != '' or parts[1] != 'pbkdf2':
                return False
            
            salt = parts[2]
            stored_hash = parts[3]
            
            # Hash the provided password with the same salt
            test_hash = self.hash_password(password, salt)
            test_hash_part = test_hash.split('$')[3]
            
            return hmac.compare_digest(stored_hash, test_hash_part)
        except Exception:
            return False
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        result = {
            'valid': True,
            'score': 0,
            'issues': []
        }
        
        # Length check
        if len(password) < 8:
            result['issues'].append("Password must be at least 8 characters")
            result['valid'] = False
        else:
            result['score'] += 20
        
        # Character variety checks
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        if not has_lower:
            result['issues'].append("Password must contain lowercase letters")
            result['valid'] = False
        else:
            result['score'] += 20
        
        if not has_upper:
            result['issues'].append("Password must contain uppercase letters")
            result['valid'] = False
        else:
            result['score'] += 20
        
        if not has_digit:
            result['issues'].append("Password must contain digits")
            result['valid'] = False
        else:
            result['score'] += 20
        
        if not has_special:
            result['issues'].append("Password must contain special characters")
            result['valid'] = False
        else:
            result['score'] += 20
        
        return result
    
    def check_password_history(self, user_id: str, password: str) -> bool:
        """Check if password was used recently"""
        if user_id not in self.password_history:
            return True  # No history, password is allowed
        
        for old_hash in self.password_history[user_id]:
            if self.verify_password(password, old_hash):
                return False  # Password was used before
        
        return True
    
    def add_to_password_history(self, user_id: str, password_hash: str) -> None:
        """Add password hash to user's history"""
        if user_id not in self.password_history:
            self.password_history[user_id] = []
        
        self.password_history[user_id].append(password_hash)
        
        # Keep only last 5 passwords
        if len(self.password_history[user_id]) > 5:
            self.password_history[user_id] = self.password_history[user_id][-5:]
    
    def record_failed_attempt(self, user_id: str) -> None:
        """Record failed login attempt"""
        current_time = time.time()
        
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        # Remove old attempts (older than lockout duration)
        cutoff_time = current_time - self.lockout_duration
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id]
            if attempt > cutoff_time
        ]
        
        self.failed_attempts[user_id].append(current_time)
    
    def is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if user_id not in self.failed_attempts:
            return False
        
        current_time = time.time()
        cutoff_time = current_time - self.lockout_duration
        
        # Count recent failed attempts
        recent_attempts = [
            attempt for attempt in self.failed_attempts[user_id]
            if attempt > cutoff_time
        ]
        
        return len(recent_attempts) >= 5  # Lock after 5 failed attempts
    
    def clear_failed_attempts(self, user_id: str) -> None:
        """Clear failed attempts for user (after successful login)"""
        if user_id in self.failed_attempts:
            self.failed_attempts[user_id] = []


class MockTokenManager:
    """Mock JWT token manager for testing"""
    
    def __init__(self):
        self.secret_key = "test_secret_key_do_not_use_in_production"
        self.algorithm = "HS256"
        self.access_token_expiry = 3600  # 1 hour
        self.refresh_token_expiry = 86400  # 24 hours
        self.revoked_tokens = set()
    
    def generate_access_token(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate mock JWT access token"""
        now = datetime.utcnow()
        payload = {
            'user_id': user_id,
            'permissions': permissions or [],
            'token_type': 'access',
            'iat': int(now.timestamp()),
            'exp': int((now + timedelta(seconds=self.access_token_expiry)).timestamp()),
            'jti': secrets.token_hex(16)  # Token ID for revocation
        }
        
        # Mock JWT encoding (base64 encode the payload as JSON)
        import base64
        import json
        payload_json = json.dumps(payload, default=str)
        encoded_payload = base64.b64encode(payload_json.encode()).decode()
        
        # Simple mock JWT format: header.payload.signature
        header = base64.b64encode(json.dumps({'typ': 'JWT', 'alg': 'HS256'}).encode()).decode()
        signature = hashlib.sha256(f"{header}.{encoded_payload}.{self.secret_key}".encode()).hexdigest()
        
        return f"{header}.{encoded_payload}.{signature}"
    
    def generate_refresh_token(self, user_id: str) -> str:
        """Generate mock JWT refresh token"""
        now = datetime.utcnow()
        payload = {
            'user_id': user_id,
            'token_type': 'refresh',
            'iat': int(now.timestamp()),
            'exp': int((now + timedelta(seconds=self.refresh_token_expiry)).timestamp()),
            'jti': secrets.token_hex(16)
        }
        
        # Mock JWT encoding
        import base64
        import json
        payload_json = json.dumps(payload, default=str)
        encoded_payload = base64.b64encode(payload_json.encode()).decode()
        
        header = base64.b64encode(json.dumps({'typ': 'JWT', 'alg': 'HS256'}).encode()).decode()
        signature = hashlib.sha256(f"{header}.{encoded_payload}.{self.secret_key}".encode()).hexdigest()
        
        return f"{header}.{encoded_payload}.{signature}"
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify mock JWT token"""
        try:
            # Parse mock JWT token
            parts = token.split('.')
            if len(parts) != 3:
                return {'valid': False, 'error': 'Invalid token format'}
            
            header, encoded_payload, signature = parts
            
            # Verify signature
            expected_signature = hashlib.sha256(f"{header}.{encoded_payload}.{self.secret_key}".encode()).hexdigest()
            if signature != expected_signature:
                return {'valid': False, 'error': 'Invalid signature'}
            
            # Decode payload
            import base64
            import json
            try:
                payload_json = base64.b64decode(encoded_payload).decode()
                payload = json.loads(payload_json)
            except:
                return {'valid': False, 'error': 'Invalid payload'}
            
            # Check if token is revoked
            if payload.get('jti') in self.revoked_tokens:
                return {'valid': False, 'error': 'Token revoked'}
            
            # Check expiration
            exp = payload.get('exp')
            if exp and exp < int(datetime.utcnow().timestamp()):
                return {'valid': False, 'error': 'Token expired'}
            
            return {
                'valid': True,
                'payload': payload,
                'user_id': payload.get('user_id'),
                'permissions': payload.get('permissions', []),
                'token_type': payload.get('token_type')
            }
        
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Generate new access token from refresh token"""
        verification = self.verify_token(refresh_token)
        
        if not verification['valid']:
            return {'success': False, 'error': verification['error']}
        
        if verification['payload'].get('token_type') != 'refresh':
            return {'success': False, 'error': 'Invalid token type'}
        
        user_id = verification['user_id']
        # In real implementation, would fetch user permissions from database
        permissions = ['read', 'write']  # Mock permissions
        
        new_access_token = self.generate_access_token(user_id, permissions)
        
        return {
            'success': True,
            'access_token': new_access_token,
            'user_id': user_id
        }
    
    def revoke_token(self, token: str) -> bool:
        """Revoke token (add to blacklist)"""
        try:
            # Parse mock JWT token to get jti
            parts = token.split('.')
            if len(parts) != 3:
                return False
            
            import base64
            import json
            encoded_payload = parts[1]
            payload_json = base64.b64decode(encoded_payload).decode()
            payload = json.loads(payload_json)
            
            jti = payload.get('jti')
            if jti:
                self.revoked_tokens.add(jti)
                return True
            
            return False
        except:
            return False
    
    def cleanup_expired_tokens(self) -> int:
        """Clean up expired revoked tokens"""
        # In real implementation, would remove expired JTIs from revoked set
        # For mock, just return count
        return len(self.revoked_tokens)


class MockTwoFactorAuth:
    """Mock two-factor authentication for testing"""
    
    def __init__(self):
        self.totp_secrets = {}
        self.backup_codes = {}
        self.pending_verifications = {}
    
    def generate_totp_secret(self, user_id: str) -> str:
        """Generate TOTP secret for user"""
        secret = secrets.token_bytes(20).hex().upper()
        self.totp_secrets[user_id] = secret
        return secret
    
    def generate_qr_code_url(self, user_id: str, issuer: str = "TradingApp") -> str:
        """Generate QR code URL for TOTP setup"""
        secret = self.totp_secrets.get(user_id)
        if not secret:
            secret = self.generate_totp_secret(user_id)
        
        return f"otpauth://totp/{issuer}:{user_id}?secret={secret}&issuer={issuer}"
    
    def verify_totp_code(self, user_id: str, code: str) -> bool:
        """Verify TOTP code"""
        if user_id not in self.totp_secrets:
            return False
        
        # Mock verification - accept codes ending with user_id last digit
        # In real implementation, would use proper TOTP algorithm
        try:
            code_int = int(code)
            user_last_digit = int(user_id[-1]) if user_id[-1].isdigit() else 0
            return len(code) == 6 and (code_int % 10) == user_last_digit
        except (ValueError, IndexError):
            return False
    
    def generate_backup_codes(self, user_id: str, count: int = 10) -> List[str]:
        """Generate backup codes for user"""
        codes = []
        for _ in range(count):
            code = '-'.join([secrets.token_hex(4) for _ in range(2)])
            codes.append(code.upper())
        
        self.backup_codes[user_id] = codes.copy()
        return codes
    
    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify backup code (single use)"""
        if user_id not in self.backup_codes:
            return False
        
        code_upper = code.upper()
        if code_upper in self.backup_codes[user_id]:
            self.backup_codes[user_id].remove(code_upper)
            return True
        
        return False
    
    def get_remaining_backup_codes(self, user_id: str) -> int:
        """Get count of remaining backup codes"""
        return len(self.backup_codes.get(user_id, []))
    
    def is_2fa_enabled(self, user_id: str) -> bool:
        """Check if 2FA is enabled for user"""
        return user_id in self.totp_secrets
    
    def disable_2fa(self, user_id: str) -> bool:
        """Disable 2FA for user"""
        removed = False
        if user_id in self.totp_secrets:
            del self.totp_secrets[user_id]
            removed = True
        if user_id in self.backup_codes:
            del self.backup_codes[user_id]
            removed = True
        return removed


class MockSessionManager:
    """Mock session manager for testing"""
    
    def __init__(self):
        self.sessions = {}
        self.session_timeout = 3600  # 1 hour
        self.max_sessions_per_user = 3
    
    def create_session(self, user_id: str, ip_address: str = None) -> str:
        """Create new session"""
        # Clean up expired sessions first
        self._cleanup_expired_sessions()
        
        # Check session limit per user
        user_sessions = [s for s in self.sessions.values() if s['user_id'] == user_id]
        if len(user_sessions) >= self.max_sessions_per_user:
            # Remove oldest session
            oldest_session = min(user_sessions, key=lambda s: s['created_at'])
            del self.sessions[oldest_session['session_id']]
        
        # Create new session
        session_id = secrets.token_urlsafe(32)
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'ip_address': ip_address,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'data': {}
        }
        
        self.sessions[session_id] = session_data
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session expired
        if time.time() - session['last_accessed'] > self.session_timeout:
            del self.sessions[session_id]
            return None
        
        # Update last accessed time
        session['last_accessed'] = time.time()
        
        return session.copy()
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data"""
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id]['data'].update(data)
        self.sessions[session_id]['last_accessed'] = time.time()
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def delete_user_sessions(self, user_id: str) -> int:
        """Delete all sessions for user"""
        user_sessions = [s_id for s_id, s in self.sessions.items() if s['user_id'] == user_id]
        
        for session_id in user_sessions:
            del self.sessions[session_id]
        
        return len(user_sessions)
    
    def get_active_sessions(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get active sessions"""
        self._cleanup_expired_sessions()
        
        sessions = []
        for session in self.sessions.values():
            if user_id is None or session['user_id'] == user_id:
                session_copy = session.copy()
                # Remove sensitive data
                session_copy.pop('data', None)
                sessions.append(session_copy)
        
        return sessions
    
    def _cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session['last_accessed'] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        return len(expired_sessions)


class TestPasswordManager:
    """Test password manager functionality"""
    
    def test_password_hashing(self):
        pm = MockPasswordManager()
        
        password = "TestPassword123!"
        hashed = pm.hash_password(password)
        
        assert hashed.startswith("$pbkdf2$")
        assert pm.verify_password(password, hashed) == True
        assert pm.verify_password("WrongPassword", hashed) == False
    
    def test_password_strength_validation(self):
        pm = MockPasswordManager()
        
        # Strong password
        result = pm.validate_password_strength("StrongPass123!")
        assert result['valid'] == True
        assert result['score'] == 100
        
        # Weak password
        result = pm.validate_password_strength("weak")
        assert result['valid'] == False
        assert len(result['issues']) > 0
        
        # Missing uppercase
        result = pm.validate_password_strength("lowercase123!")
        assert result['valid'] == False
        assert "uppercase" in str(result['issues'])
    
    def test_password_history(self):
        pm = MockPasswordManager()
        
        user_id = "test_user"
        old_password = "OldPassword123!"
        new_password = "NewPassword456!"
        
        # Add password to history
        old_hash = pm.hash_password(old_password)
        pm.add_to_password_history(user_id, old_hash)
        
        # Check history
        assert pm.check_password_history(user_id, old_password) == False  # Should be blocked
        assert pm.check_password_history(user_id, new_password) == True   # Should be allowed
    
    def test_account_lockout(self):
        pm = MockPasswordManager()
        
        user_id = "test_user"
        
        # Initially not locked
        assert pm.is_account_locked(user_id) == False
        
        # Record failed attempts
        for _ in range(5):
            pm.record_failed_attempt(user_id)
        
        # Should be locked now
        assert pm.is_account_locked(user_id) == True
        
        # Clear attempts
        pm.clear_failed_attempts(user_id)
        assert pm.is_account_locked(user_id) == False


class TestTokenManager:
    """Test JWT token manager functionality"""
    
    def test_token_generation(self):
        tm = MockTokenManager()
        
        user_id = "test_user"
        permissions = ["read", "write"]
        
        # Generate tokens
        access_token = tm.generate_access_token(user_id, permissions)
        refresh_token = tm.generate_refresh_token(user_id)
        
        assert isinstance(access_token, str)
        assert isinstance(refresh_token, str)
        assert access_token != refresh_token
    
    def test_token_verification(self):
        tm = MockTokenManager()
        
        user_id = "test_user"
        permissions = ["read", "write"]
        
        # Generate and verify access token
        token = tm.generate_access_token(user_id, permissions)
        result = tm.verify_token(token)
        
        assert result['valid'] == True
        assert result['user_id'] == user_id
        assert result['permissions'] == permissions
        assert result['token_type'] == 'access'
    
    def test_token_expiration(self):
        tm = MockTokenManager()
        tm.access_token_expiry = -1  # Force immediate expiration
        
        user_id = "test_user"
        token = tm.generate_access_token(user_id)
        
        # Token should be expired
        result = tm.verify_token(token)
        assert result['valid'] == False
        assert 'expired' in result['error'].lower()
    
    def test_token_refresh(self):
        tm = MockTokenManager()
        
        user_id = "test_user"
        refresh_token = tm.generate_refresh_token(user_id)
        
        # Refresh access token
        result = tm.refresh_access_token(refresh_token)
        
        assert result['success'] == True
        assert 'access_token' in result
        assert result['user_id'] == user_id
    
    def test_token_revocation(self):
        tm = MockTokenManager()
        
        user_id = "test_user"
        token = tm.generate_access_token(user_id)
        
        # Verify token works
        result = tm.verify_token(token)
        assert result['valid'] == True
        
        # Revoke token
        assert tm.revoke_token(token) == True
        
        # Token should no longer work
        result = tm.verify_token(token)
        assert result['valid'] == False
        assert 'revoked' in result['error'].lower()


class TestTwoFactorAuth:
    """Test two-factor authentication functionality"""
    
    def test_totp_setup(self):
        tfa = MockTwoFactorAuth()
        
        user_id = "test_user"
        
        # Generate secret
        secret = tfa.generate_totp_secret(user_id)
        assert len(secret) > 0
        
        # Generate QR code URL
        qr_url = tfa.generate_qr_code_url(user_id)
        assert qr_url.startswith("otpauth://totp/")
        assert user_id in qr_url
        assert secret in qr_url
    
    def test_totp_verification(self):
        tfa = MockTwoFactorAuth()
        
        user_id = "12345"  # Ends with 5
        tfa.generate_totp_secret(user_id)
        
        # Valid code (mock: must end with user_id last digit)
        assert tfa.verify_totp_code(user_id, "123455") == True
        
        # Invalid code
        assert tfa.verify_totp_code(user_id, "123456") == False
        assert tfa.verify_totp_code(user_id, "invalid") == False
    
    def test_backup_codes(self):
        tfa = MockTwoFactorAuth()
        
        user_id = "test_user"
        
        # Generate backup codes
        codes = tfa.generate_backup_codes(user_id, 5)
        assert len(codes) == 5
        assert all(len(code) > 0 for code in codes)
        
        # Test code usage
        test_code = codes[0]
        assert tfa.verify_backup_code(user_id, test_code) == True
        
        # Code should be consumed (single use)
        assert tfa.verify_backup_code(user_id, test_code) == False
        
        # Check remaining codes
        assert tfa.get_remaining_backup_codes(user_id) == 4
    
    def test_2fa_management(self):
        tfa = MockTwoFactorAuth()
        
        user_id = "test_user"
        
        # Initially disabled
        assert tfa.is_2fa_enabled(user_id) == False
        
        # Enable 2FA
        tfa.generate_totp_secret(user_id)
        assert tfa.is_2fa_enabled(user_id) == True
        
        # Disable 2FA
        assert tfa.disable_2fa(user_id) == True
        assert tfa.is_2fa_enabled(user_id) == False


class TestSessionManager:
    """Test session manager functionality"""
    
    def test_session_creation(self):
        sm = MockSessionManager()
        
        user_id = "test_user"
        ip_address = "192.168.1.100"
        
        # Create session
        session_id = sm.create_session(user_id, ip_address)
        assert len(session_id) > 0
        
        # Get session
        session = sm.get_session(session_id)
        assert session['user_id'] == user_id
        assert session['ip_address'] == ip_address
    
    def test_session_timeout(self):
        sm = MockSessionManager()
        sm.session_timeout = -1  # Force immediate timeout
        
        user_id = "test_user"
        session_id = sm.create_session(user_id)
        
        # Session should be expired
        session = sm.get_session(session_id)
        assert session is None
    
    def test_session_limit(self):
        sm = MockSessionManager()
        sm.max_sessions_per_user = 2
        
        user_id = "test_user"
        
        # Create sessions up to limit
        session1 = sm.create_session(user_id)
        session2 = sm.create_session(user_id)
        
        # Both should exist
        assert sm.get_session(session1) is not None
        assert sm.get_session(session2) is not None
        
        # Create third session (should remove oldest)
        session3 = sm.create_session(user_id)
        
        # First session should be gone
        assert sm.get_session(session1) is None
        assert sm.get_session(session2) is not None
        assert sm.get_session(session3) is not None
    
    def test_session_management(self):
        sm = MockSessionManager()
        
        user_id = "test_user"
        session_id = sm.create_session(user_id)
        
        # Update session data
        test_data = {"key": "value"}
        assert sm.update_session(session_id, test_data) == True
        
        session = sm.get_session(session_id)
        assert session['data']['key'] == "value"
        
        # Delete session
        assert sm.delete_session(session_id) == True
        assert sm.get_session(session_id) is None
    
    def test_user_session_management(self):
        sm = MockSessionManager()
        
        user_id = "test_user"
        
        # Create multiple sessions for user
        session1 = sm.create_session(user_id)
        session2 = sm.create_session(user_id)
        
        # Get active sessions
        active_sessions = sm.get_active_sessions(user_id)
        assert len(active_sessions) == 2
        
        # Delete all user sessions
        deleted_count = sm.delete_user_sessions(user_id)
        assert deleted_count == 2
        
        # No active sessions remaining
        active_sessions = sm.get_active_sessions(user_id)
        assert len(active_sessions) == 0


class TestAuthenticationIntegration:
    """Test authentication integration scenarios"""
    
    def test_complete_authentication_flow(self):
        """Test complete authentication workflow"""
        pm = MockPasswordManager()
        tm = MockTokenManager()
        sm = MockSessionManager()
        
        user_id = "test_user"
        password = "SecurePassword123!"
        
        # 1. Register user (hash password)
        password_hash = pm.hash_password(password)
        pm.add_to_password_history(user_id, password_hash)
        
        # 2. Login attempt
        if pm.verify_password(password, password_hash):
            # 3. Create session
            session_id = sm.create_session(user_id, "192.168.1.100")
            
            # 4. Generate tokens
            access_token = tm.generate_access_token(user_id, ["read", "write"])
            refresh_token = tm.generate_refresh_token(user_id)
            
            # All operations should succeed
            assert len(session_id) > 0
            assert len(access_token) > 0
            assert len(refresh_token) > 0
            
            # 5. Verify token
            token_result = tm.verify_token(access_token)
            assert token_result['valid'] == True
            assert token_result['user_id'] == user_id
    
    def test_authentication_with_2fa(self):
        """Test authentication with two-factor authentication"""
        pm = MockPasswordManager()
        tfa = MockTwoFactorAuth()
        tm = MockTokenManager()
        
        user_id = "12345"
        password = "SecurePassword123!"
        
        # Setup user
        password_hash = pm.hash_password(password)
        tfa.generate_totp_secret(user_id)
        
        # Step 1: Password authentication
        if pm.verify_password(password, password_hash):
            # Step 2: 2FA verification
            if tfa.verify_totp_code(user_id, "123455"):  # Valid code for user_id ending in 5
                # Step 3: Generate tokens after successful 2FA
                access_token = tm.generate_access_token(user_id, ["read", "write"])
                
                assert len(access_token) > 0
                
                token_result = tm.verify_token(access_token)
                assert token_result['valid'] == True
    
    def test_failed_authentication_handling(self):
        """Test failed authentication attempt handling"""
        pm = MockPasswordManager()
        tm = MockTokenManager()
        sm = MockSessionManager()
        
        user_id = "test_user"
        correct_password = "CorrectPassword123!"
        wrong_password = "WrongPassword"
        
        # Setup user
        password_hash = pm.hash_password(correct_password)
        
        # Multiple failed attempts
        for _ in range(5):
            if not pm.verify_password(wrong_password, password_hash):
                pm.record_failed_attempt(user_id)
        
        # Account should be locked
        assert pm.is_account_locked(user_id) == True
        
        # Even correct password should be rejected when locked
        if pm.is_account_locked(user_id):
            # Don't create session or tokens
            assert sm.get_active_sessions(user_id) == []
    
    def test_session_security(self):
        """Test session security features"""
        sm = MockSessionManager()
        tm = MockTokenManager()
        
        user_id = "test_user"
        
        # Create session from specific IP
        session_id = sm.create_session(user_id, "192.168.1.100")
        session = sm.get_session(session_id)
        
        # Simulate IP change detection (security concern)
        current_ip = "10.0.0.1"
        if session['ip_address'] != current_ip:
            # Security response: invalidate session and tokens
            sm.delete_session(session_id)
            
            # In real implementation, would also revoke associated tokens
            # tm.revoke_user_tokens(user_id)
        
        # Session should be gone
        assert sm.get_session(session_id) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])