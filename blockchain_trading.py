#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blockchain Trading System - ブロックチェーン統合取引記録システム
Issue #939対応: 分散台帳 + スマートコントラクト + 暗号化取引記録
"""

import hashlib
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import secrets
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import logging

# 統合モジュール
try:
    from quantum_ai_engine import quantum_ai_engine, QuantumPrediction
    HAS_QUANTUM_AI = True
except ImportError:
    HAS_QUANTUM_AI = False

try:
    from performance_monitor import performance_monitor
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False

try:
    from data_persistence import data_persistence
    HAS_DATA_PERSISTENCE = True
except ImportError:
    HAS_DATA_PERSISTENCE = False


@dataclass
class Transaction:
    """取引記録"""
    tx_id: str
    from_address: str
    to_address: str
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: float
    price: float
    fee: float
    timestamp: datetime
    signature: str
    ai_confidence: float
    quantum_signature: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Block:
    """ブロック構造"""
    index: int
    transactions: List[Transaction]
    previous_hash: str
    merkle_root: str
    timestamp: datetime
    nonce: int
    hash: str
    validator: str
    difficulty: int = 4


@dataclass
class SmartContract:
    """スマートコントラクト"""
    contract_id: str
    contract_code: str
    owner: str
    created_at: datetime
    state: Dict[str, Any]
    is_active: bool = True


class CryptographicSecurity:
    """暗号化セキュリティ"""
    
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        # ノード固有の鍵ペア
        self.node_keys = {}
        self.symmetric_keys = {}
    
    def generate_key_pair(self, node_id: str) -> Tuple[bytes, bytes]:
        """鍵ペア生成"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        # PEM形式でシリアライズ
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        self.node_keys[node_id] = {'private': private_key, 'public': public_key}
        
        return private_pem, public_pem
    
    def sign_transaction(self, transaction_data: str, node_id: str = None) -> str:
        """取引署名"""
        key = self.node_keys.get(node_id, {}).get('private', self.private_key)
        
        signature = key.sign(
            transaction_data.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode('utf-8')
    
    def verify_signature(self, transaction_data: str, signature: str, node_id: str = None) -> bool:
        """署名検証"""
        try:
            key = self.node_keys.get(node_id, {}).get('public', self.public_key)
            signature_bytes = base64.b64decode(signature.encode('utf-8'))
            
            key.verify(
                signature_bytes,
                transaction_data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def encrypt_data(self, data: str, node_id: str = None) -> str:
        """データ暗号化"""
        # 対称鍵生成
        key = secrets.token_bytes(32)  # 256-bit key
        iv = secrets.token_bytes(16)   # 128-bit IV
        
        # AES暗号化
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # パディング
        data_bytes = data.encode('utf-8')
        padding_length = 16 - (len(data_bytes) % 16)
        padded_data = data_bytes + bytes([padding_length]) * padding_length
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # 公開鍵で対称鍵を暗号化
        pub_key = self.node_keys.get(node_id, {}).get('public', self.public_key)
        encrypted_key = pub_key.encrypt(
            key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # 結果をbase64エンコード
        result = {
            'encrypted_key': base64.b64encode(encrypted_key).decode('utf-8'),
            'iv': base64.b64encode(iv).decode('utf-8'),
            'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8')
        }
        
        return base64.b64encode(json.dumps(result).encode('utf-8')).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str, node_id: str = None) -> str:
        """データ復号化"""
        try:
            # base64デコード
            data_dict = json.loads(base64.b64decode(encrypted_data.encode('utf-8')).decode('utf-8'))
            
            encrypted_key = base64.b64decode(data_dict['encrypted_key'].encode('utf-8'))
            iv = base64.b64decode(data_dict['iv'].encode('utf-8'))
            ciphertext = base64.b64decode(data_dict['encrypted_data'].encode('utf-8'))
            
            # 秘密鍵で対称鍵を復号化
            priv_key = self.node_keys.get(node_id, {}).get('private', self.private_key)
            symmetric_key = priv_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # AES復号化
            cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            
            decrypted_padded = decryptor.update(ciphertext) + decryptor.finalize()
            
            # パディング除去
            padding_length = decrypted_padded[-1]
            decrypted_data = decrypted_padded[:-padding_length]
            
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logging.error(f"Decryption error: {e}")
            return ""


class MerkleTree:
    """マークル木実装"""
    
    @staticmethod
    def calculate_merkle_root(transactions: List[Transaction]) -> str:
        """マークル根計算"""
        if not transactions:
            return hashlib.sha256(b'').hexdigest()
        
        # 取引ハッシュ計算
        tx_hashes = []
        for tx in transactions:
            tx_data = f"{tx.tx_id}{tx.symbol}{tx.action}{tx.quantity}{tx.price}{tx.timestamp.isoformat()}"
            tx_hash = hashlib.sha256(tx_data.encode()).hexdigest()
            tx_hashes.append(tx_hash)
        
        # マークル木構築
        while len(tx_hashes) > 1:
            next_level = []
            
            # ペアでハッシュ化
            for i in range(0, len(tx_hashes), 2):
                left = tx_hashes[i]
                right = tx_hashes[i + 1] if i + 1 < len(tx_hashes) else left
                
                combined = left + right
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(parent_hash)
            
            tx_hashes = next_level
        
        return tx_hashes[0]
    
    @staticmethod
    def verify_transaction_in_merkle_tree(transaction: Transaction, merkle_root: str, 
                                        merkle_path: List[str]) -> bool:
        """取引のマークル木検証"""
        # 簡略化された検証実装
        tx_data = f"{transaction.tx_id}{transaction.symbol}{transaction.action}"
        current_hash = hashlib.sha256(tx_data.encode()).hexdigest()
        
        for path_hash in merkle_path:
            combined = current_hash + path_hash
            current_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return current_hash == merkle_root


class SmartContractEngine:
    """スマートコントラクト実行エンジン"""
    
    def __init__(self):
        self.contracts = {}
        self.contract_storage = defaultdict(dict)
        self.execution_gas_limit = 1000000
        
        # 組み込み関数
        self.builtin_functions = {
            'get_balance': self._get_balance,
            'transfer': self._transfer,
            'get_time': lambda: int(time.time()),
            'get_block_height': self._get_block_height,
            'hash256': lambda x: hashlib.sha256(str(x).encode()).hexdigest(),
            'verify_signature': self._verify_signature
        }
    
    def deploy_contract(self, contract_code: str, owner: str) -> str:
        """コントラクトデプロイ"""
        contract_id = hashlib.sha256(
            f"{contract_code}{owner}{time.time()}".encode()
        ).hexdigest()[:16]
        
        contract = SmartContract(
            contract_id=contract_id,
            contract_code=contract_code,
            owner=owner,
            created_at=datetime.now(),
            state={}
        )
        
        self.contracts[contract_id] = contract
        
        # 初期化実行
        try:
            self._execute_contract(contract_id, 'init', {})
        except Exception as e:
            logging.error(f"Contract initialization failed: {e}")
        
        return contract_id
    
    def execute_contract(self, contract_id: str, function_name: str, 
                        parameters: Dict[str, Any]) -> Dict[str, Any]:
        """コントラクト実行"""
        if contract_id not in self.contracts:
            return {'error': 'Contract not found'}
        
        contract = self.contracts[contract_id]
        if not contract.is_active:
            return {'error': 'Contract is not active'}
        
        try:
            result = self._execute_contract(contract_id, function_name, parameters)
            return {'success': True, 'result': result}
        except Exception as e:
            return {'error': str(e)}
    
    def _execute_contract(self, contract_id: str, function_name: str, parameters: Dict[str, Any]) -> Any:
        """コントラクト実行（簡略実装）"""
        contract = self.contracts[contract_id]
        
        # セキュリティコンテキスト
        context = {
            'contract_id': contract_id,
            'storage': self.contract_storage[contract_id],
            'parameters': parameters,
            'builtin': self.builtin_functions,
            'gas_used': 0
        }
        
        # シンプルなスクリプト実行エンジン（実際の実装ではより安全な実行環境が必要）
        if function_name == 'init':
            return self._execute_init_contract(contract, context)
        elif function_name == 'execute_trade':
            return self._execute_trade_contract(contract, context)
        elif function_name == 'validate_ai_prediction':
            return self._validate_ai_prediction_contract(contract, context)
        else:
            return {'error': f'Function {function_name} not found'}
    
    def _execute_init_contract(self, contract: SmartContract, context: Dict[str, Any]) -> Dict[str, Any]:
        """初期化コントラクト"""
        context['storage']['initialized'] = True
        context['storage']['owner'] = contract.owner
        context['storage']['balance'] = 1000000.0  # 初期残高
        
        return {'initialized': True}
    
    def _execute_trade_contract(self, contract: SmartContract, context: Dict[str, Any]) -> Dict[str, Any]:
        """取引実行コントラクト"""
        params = context['parameters']
        
        symbol = params.get('symbol', '')
        action = params.get('action', '')
        quantity = float(params.get('quantity', 0))
        price = float(params.get('price', 0))
        ai_confidence = float(params.get('ai_confidence', 0))
        
        # 取引条件チェック
        if ai_confidence < 0.7:
            return {'error': 'AI confidence too low', 'required': 0.7, 'actual': ai_confidence}
        
        if quantity <= 0 or price <= 0:
            return {'error': 'Invalid quantity or price'}
        
        # 残高チェック
        balance = context['storage'].get('balance', 0)
        required_amount = quantity * price
        
        if action == 'BUY' and balance < required_amount:
            return {'error': 'Insufficient balance', 'required': required_amount, 'available': balance}
        
        # 取引実行
        if action == 'BUY':
            context['storage']['balance'] -= required_amount
            context['storage'][f'holdings_{symbol}'] = context['storage'].get(f'holdings_{symbol}', 0) + quantity
        elif action == 'SELL':
            holdings = context['storage'].get(f'holdings_{symbol}', 0)
            if holdings < quantity:
                return {'error': 'Insufficient holdings', 'required': quantity, 'available': holdings}
            
            context['storage']['balance'] += required_amount
            context['storage'][f'holdings_{symbol}'] = holdings - quantity
        
        # 取引記録
        trade_id = hashlib.sha256(f"{symbol}{action}{quantity}{price}{time.time()}".encode()).hexdigest()[:16]
        context['storage']['last_trade'] = {
            'trade_id': trade_id,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'trade_executed': True,
            'trade_id': trade_id,
            'new_balance': context['storage']['balance'],
            'holdings': context['storage'].get(f'holdings_{symbol}', 0)
        }
    
    def _validate_ai_prediction_contract(self, contract: SmartContract, context: Dict[str, Any]) -> Dict[str, Any]:
        """AI予測検証コントラクト"""
        params = context['parameters']
        
        prediction_data = params.get('prediction', {})
        actual_result = params.get('actual_result', {})
        
        # 予測精度計算
        predicted_confidence = prediction_data.get('confidence', 0)
        actual_accuracy = actual_result.get('accuracy', 0)
        
        accuracy_score = 1.0 - abs(predicted_confidence - actual_accuracy)
        
        # 報酬計算
        reward = accuracy_score * 100  # スコアに基づく報酬
        
        context['storage']['total_predictions'] = context['storage'].get('total_predictions', 0) + 1
        context['storage']['cumulative_accuracy'] = context['storage'].get('cumulative_accuracy', 0) + accuracy_score
        context['storage']['average_accuracy'] = context['storage']['cumulative_accuracy'] / context['storage']['total_predictions']
        
        return {
            'validation_completed': True,
            'accuracy_score': accuracy_score,
            'reward': reward,
            'average_accuracy': context['storage']['average_accuracy']
        }
    
    def _get_balance(self, address: str) -> float:
        """残高取得"""
        # 簡略実装
        return 1000000.0
    
    def _transfer(self, from_addr: str, to_addr: str, amount: float) -> bool:
        """送金"""
        # 簡略実装
        return True
    
    def _get_block_height(self) -> int:
        """ブロック高取得"""
        return 100  # 簡略実装
    
    def _verify_signature(self, message: str, signature: str, public_key: str) -> bool:
        """署名検証"""
        # 簡略実装
        return len(signature) > 50  # 最低限の検証


class Blockchain:
    """ブロックチェーン実装"""
    
    def __init__(self, node_id: str = "node_1"):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.node_id = node_id
        self.difficulty = 4
        
        self.crypto = CryptographicSecurity()
        self.smart_contracts = SmartContractEngine()
        
        # ネットワーク
        self.peers: List[str] = []
        self.validators = ['validator_1', 'validator_2', 'validator_3']
        
        # 統計
        self.stats = {
            'total_transactions': 0,
            'total_blocks': 0,
            'total_contracts': 0,
            'network_hash_rate': 0
        }
        
        # ジェネシスブロック作成
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """ジェネシスブロック作成"""
        genesis_transactions = []
        
        genesis_block = Block(
            index=0,
            transactions=genesis_transactions,
            previous_hash="0",
            merkle_root=MerkleTree.calculate_merkle_root(genesis_transactions),
            timestamp=datetime.now(),
            nonce=0,
            hash="",
            validator=self.node_id,
            difficulty=self.difficulty
        )
        
        # ジェネシスブロックのハッシュ計算
        genesis_block.hash = self._calculate_block_hash(genesis_block)
        self.chain.append(genesis_block)
        
        logging.info(f"Genesis block created: {genesis_block.hash}")
    
    def create_transaction(self, from_address: str, to_address: str, symbol: str,
                          action: str, quantity: float, price: float, 
                          ai_confidence: float = 0.0, metadata: Dict[str, Any] = None) -> str:
        """取引作成"""
        tx_id = hashlib.sha256(
            f"{from_address}{to_address}{symbol}{action}{quantity}{price}{time.time()}".encode()
        ).hexdigest()
        
        # 手数料計算
        fee = quantity * price * 0.001  # 0.1%の手数料
        
        # 取引データ作成
        tx_data = f"{tx_id}{from_address}{to_address}{symbol}{action}{quantity}{price}"
        signature = self.crypto.sign_transaction(tx_data)
        
        # 量子署名（利用可能な場合）
        quantum_signature = None
        if HAS_QUANTUM_AI:
            quantum_signature = hashlib.sha256(f"quantum_{tx_data}".encode()).hexdigest()
        
        transaction = Transaction(
            tx_id=tx_id,
            from_address=from_address,
            to_address=to_address,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            fee=fee,
            timestamp=datetime.now(),
            signature=signature,
            ai_confidence=ai_confidence,
            quantum_signature=quantum_signature,
            metadata=metadata or {}
        )
        
        # 取引検証
        if self._validate_transaction(transaction):
            self.pending_transactions.append(transaction)
            self.stats['total_transactions'] += 1
            
            # スマートコントラクト実行（該当する場合）
            if metadata and 'contract_id' in metadata:
                self._execute_transaction_contract(transaction, metadata['contract_id'])
            
            logging.info(f"Transaction created: {tx_id}")
            return tx_id
        else:
            raise ValueError("Transaction validation failed")
    
    def _validate_transaction(self, transaction: Transaction) -> bool:
        """取引検証"""
        # 基本検証
        if transaction.quantity <= 0 or transaction.price <= 0:
            return False
        
        if transaction.action not in ['BUY', 'SELL', 'HOLD']:
            return False
        
        # 署名検証
        tx_data = f"{transaction.tx_id}{transaction.from_address}{transaction.to_address}{transaction.symbol}{transaction.action}{transaction.quantity}{transaction.price}"
        
        if not self.crypto.verify_signature(tx_data, transaction.signature):
            return False
        
        # 重複取引チェック
        for block in self.chain:
            for tx in block.transactions:
                if tx.tx_id == transaction.tx_id:
                    return False
        
        for tx in self.pending_transactions:
            if tx.tx_id == transaction.tx_id:
                return False
        
        return True
    
    def _execute_transaction_contract(self, transaction: Transaction, contract_id: str):
        """取引関連スマートコントラクト実行"""
        contract_params = {
            'symbol': transaction.symbol,
            'action': transaction.action,
            'quantity': transaction.quantity,
            'price': transaction.price,
            'ai_confidence': transaction.ai_confidence
        }
        
        result = self.smart_contracts.execute_contract(contract_id, 'execute_trade', contract_params)
        
        # 実行結果をメタデータに追加
        transaction.metadata['contract_result'] = result
    
    def mine_block(self) -> Optional[Block]:
        """ブロックマイニング"""
        if not self.pending_transactions:
            return None
        
        # 新しいブロック作成
        new_index = len(self.chain)
        previous_hash = self.chain[-1].hash if self.chain else "0"
        
        # トランザクション選択（手数料順）
        selected_transactions = sorted(
            self.pending_transactions,
            key=lambda tx: tx.fee,
            reverse=True
        )[:10]  # 最大10取引
        
        merkle_root = MerkleTree.calculate_merkle_root(selected_transactions)
        
        new_block = Block(
            index=new_index,
            transactions=selected_transactions,
            previous_hash=previous_hash,
            merkle_root=merkle_root,
            timestamp=datetime.now(),
            nonce=0,
            hash="",
            validator=self.node_id,
            difficulty=self.difficulty
        )
        
        # プルーフオブワーク
        new_block = self._proof_of_work(new_block)
        
        # ブロックチェーンに追加
        self.chain.append(new_block)
        
        # 処理済み取引を削除
        for tx in selected_transactions:
            if tx in self.pending_transactions:
                self.pending_transactions.remove(tx)
        
        self.stats['total_blocks'] += 1
        
        # データ永続化
        if HAS_DATA_PERSISTENCE:
            self._persist_block_data(new_block)
        
        logging.info(f"Block mined: {new_block.index} - {new_block.hash}")
        return new_block
    
    def _proof_of_work(self, block: Block) -> Block:
        """プルーフオブワーク"""
        target = "0" * block.difficulty
        
        while not block.hash.startswith(target):
            block.nonce += 1
            block.hash = self._calculate_block_hash(block)
        
        return block
    
    def _calculate_block_hash(self, block: Block) -> str:
        """ブロックハッシュ計算"""
        block_string = json.dumps({
            'index': block.index,
            'previous_hash': block.previous_hash,
            'merkle_root': block.merkle_root,
            'timestamp': block.timestamp.isoformat(),
            'nonce': block.nonce,
            'validator': block.validator
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def _persist_block_data(self, block: Block):
        """ブロックデータ永続化"""
        try:
            block_data = {
                'block_index': block.index,
                'block_hash': block.hash,
                'transaction_count': len(block.transactions),
                'validator': block.validator,
                'timestamp': block.timestamp.isoformat(),
                'merkle_root': block.merkle_root
            }
            
            data_persistence.save_analysis_result(
                symbol='BLOCKCHAIN',
                analysis_type='block_mining',
                duration_ms=0,
                result_data=block_data,
                confidence_score=1.0,
                session_id=f'blockchain_{self.node_id}'
            )
        except Exception as e:
            logging.error(f"Block persistence error: {e}")
    
    def validate_chain(self) -> bool:
        """チェーン検証"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # ブロックハッシュ検証
            if current_block.hash != self._calculate_block_hash(current_block):
                return False
            
            # 前ブロックハッシュ検証
            if current_block.previous_hash != previous_block.hash:
                return False
            
            # マークル根検証
            if current_block.merkle_root != MerkleTree.calculate_merkle_root(current_block.transactions):
                return False
        
        return True
    
    def get_balance(self, address: str, symbol: str = None) -> Dict[str, float]:
        """残高取得"""
        balances = defaultdict(float)
        
        for block in self.chain:
            for tx in block.transactions:
                if tx.from_address == address:
                    balances[tx.symbol] -= tx.quantity
                if tx.to_address == address:
                    balances[tx.symbol] += tx.quantity
        
        if symbol:
            return {symbol: balances.get(symbol, 0.0)}
        
        return dict(balances)
    
    def get_transaction_history(self, address: str, limit: int = 100) -> List[Transaction]:
        """取引履歴取得"""
        transactions = []
        
        for block in reversed(self.chain):
            for tx in block.transactions:
                if tx.from_address == address or tx.to_address == address:
                    transactions.append(tx)
                    
                if len(transactions) >= limit:
                    return transactions
        
        return transactions
    
    def deploy_smart_contract(self, contract_code: str, owner: str) -> str:
        """スマートコントラクトデプロイ"""
        contract_id = self.smart_contracts.deploy_contract(contract_code, owner)
        self.stats['total_contracts'] += 1
        
        # デプロイメント取引作成
        self.create_transaction(
            from_address=owner,
            to_address=f"contract_{contract_id}",
            symbol="CONTRACT",
            action="DEPLOY",
            quantity=1.0,
            price=0.0,
            metadata={'contract_id': contract_id, 'contract_code': contract_code[:100]}
        )
        
        return contract_id
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """ブロックチェーン統計"""
        chain_size = len(self.chain)
        total_tx = sum(len(block.transactions) for block in self.chain)
        
        latest_block = self.chain[-1] if self.chain else None
        
        return {
            'node_id': self.node_id,
            'chain_length': chain_size,
            'total_transactions': total_tx,
            'pending_transactions': len(self.pending_transactions),
            'difficulty': self.difficulty,
            'latest_block_hash': latest_block.hash if latest_block else None,
            'latest_block_time': latest_block.timestamp.isoformat() if latest_block else None,
            'validators': self.validators,
            'peers_count': len(self.peers),
            'smart_contracts_deployed': self.stats['total_contracts'],
            'is_valid_chain': self.validate_chain()
        }


class TradingBlockchainIntegration:
    """取引ブロックチェーン統合システム"""
    
    def __init__(self, node_id: str = "trading_node_1"):
        self.blockchain = Blockchain(node_id)
        self.node_id = node_id
        
        # AI統合
        self.ai_predictions_cache = {}
        
        # 取引戦略コントラクトをデプロイ
        self.strategy_contract_id = self._deploy_trading_strategy_contract()
        
        # 統計
        self.integration_stats = {
            'ai_predictions_recorded': 0,
            'automated_trades_executed': 0,
            'quantum_signatures_verified': 0,
            'total_trading_volume': 0.0
        }
    
    def _deploy_trading_strategy_contract(self) -> str:
        """取引戦略スマートコントラクトデプロイ"""
        contract_code = """
        contract TradingStrategy {
            function execute_trade(symbol, action, quantity, price, ai_confidence) {
                if (ai_confidence < 0.7) {
                    return error("AI confidence too low");
                }
                
                // 取引ロジック
                return execute_market_trade(symbol, action, quantity, price);
            }
            
            function validate_prediction(prediction, actual) {
                accuracy = calculate_accuracy(prediction, actual);
                reward = accuracy * 100;
                return reward;
            }
        }
        """
        
        return self.blockchain.deploy_smart_contract(contract_code, self.node_id)
    
    def record_ai_prediction(self, symbol: str, prediction: Any) -> str:
        """AI予測記録"""
        try:
            if HAS_QUANTUM_AI and hasattr(prediction, 'hybrid_confidence'):
                # 量子AI予測
                prediction_data = {
                    'symbol': symbol,
                    'quantum_confidence': prediction.quantum_confidence,
                    'classical_confidence': prediction.classical_confidence,
                    'hybrid_confidence': prediction.hybrid_confidence,
                    'quantum_advantage': prediction.quantum_advantage,
                    'entanglement_strength': prediction.entanglement_strength,
                    'algorithm_used': prediction.algorithm_used,
                    'prediction_probabilities': prediction.prediction_probabilities
                }
                ai_confidence = prediction.hybrid_confidence
            else:
                # 従来AI予測
                prediction_data = {
                    'symbol': symbol,
                    'confidence': getattr(prediction, 'confidence', 0.5),
                    'signal_type': getattr(prediction, 'signal_type', 'HOLD'),
                    'strength': getattr(prediction, 'strength', 0.0)
                }
                ai_confidence = getattr(prediction, 'confidence', 0.5)
            
            # ブロックチェーンに記録
            tx_id = self.blockchain.create_transaction(
                from_address=self.node_id,
                to_address='AI_PREDICTION_POOL',
                symbol=symbol,
                action='PREDICT',
                quantity=1.0,
                price=0.0,
                ai_confidence=ai_confidence,
                metadata={
                    'prediction_data': prediction_data,
                    'contract_id': self.strategy_contract_id,
                    'prediction_type': 'quantum' if HAS_QUANTUM_AI else 'classical'
                }
            )
            
            self.ai_predictions_cache[symbol] = prediction_data
            self.integration_stats['ai_predictions_recorded'] += 1
            
            return tx_id
            
        except Exception as e:
            logging.error(f"AI prediction recording error: {e}")
            return ""
    
    def execute_automated_trade(self, symbol: str, action: str, quantity: float, price: float) -> str:
        """自動取引実行"""
        try:
            # AI予測に基づく信頼度取得
            ai_confidence = 0.5
            
            if symbol in self.ai_predictions_cache:
                pred_data = self.ai_predictions_cache[symbol]
                ai_confidence = pred_data.get('hybrid_confidence', pred_data.get('confidence', 0.5))
            elif HAS_QUANTUM_AI:
                # リアルタイムAI分析
                market_data = [price + i * 5 for i in range(20)]  # 模擬データ
                prediction = quantum_ai_engine.quantum_market_analysis(symbol, market_data)
                ai_confidence = prediction.hybrid_confidence
                
                # 予測をキャッシュ
                self.ai_predictions_cache[symbol] = {
                    'hybrid_confidence': prediction.hybrid_confidence,
                    'quantum_advantage': prediction.quantum_advantage
                }
            
            # 取引実行
            tx_id = self.blockchain.create_transaction(
                from_address=self.node_id,
                to_address='MARKET',
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                ai_confidence=ai_confidence,
                metadata={
                    'automated_trade': True,
                    'contract_id': self.strategy_contract_id,
                    'trade_type': 'ai_driven'
                }
            )
            
            self.integration_stats['automated_trades_executed'] += 1
            self.integration_stats['total_trading_volume'] += quantity * price
            
            # パフォーマンス記録
            if HAS_PERFORMANCE_MONITOR:
                performance_monitor.track_analysis_time(
                    symbol, 0.1, 'blockchain_trade'
                )
            
            return tx_id
            
        except Exception as e:
            logging.error(f"Automated trade execution error: {e}")
            return ""
    
    def validate_prediction_accuracy(self, symbol: str, actual_result: Dict[str, float]) -> Dict[str, Any]:
        """予測精度検証"""
        if symbol not in self.ai_predictions_cache:
            return {'error': 'No prediction found for symbol'}
        
        prediction_data = self.ai_predictions_cache[symbol]
        
        # 精度計算
        predicted_confidence = prediction_data.get('hybrid_confidence', prediction_data.get('confidence', 0.5))
        actual_accuracy = actual_result.get('accuracy', 0.5)
        
        # スマートコントラクト実行
        validation_result = self.blockchain.smart_contracts.execute_contract(
            self.strategy_contract_id,
            'validate_ai_prediction',
            {
                'prediction': prediction_data,
                'actual_result': actual_result
            }
        )
        
        return validation_result
    
    def get_trading_analytics(self) -> Dict[str, Any]:
        """取引分析データ取得"""
        # ブロックチェーンから取引データ抽出
        all_trades = []
        ai_trades = []
        
        for block in self.blockchain.chain:
            for tx in block.transactions:
                if tx.action in ['BUY', 'SELL']:
                    all_trades.append(tx)
                    
                    if tx.metadata.get('automated_trade'):
                        ai_trades.append(tx)
        
        # 統計計算
        total_volume = sum(tx.quantity * tx.price for tx in all_trades)
        ai_volume = sum(tx.quantity * tx.price for tx in ai_trades)
        
        avg_ai_confidence = (
            sum(tx.ai_confidence for tx in ai_trades) / len(ai_trades)
            if ai_trades else 0.0
        )
        
        return {
            'blockchain_stats': self.blockchain.get_blockchain_stats(),
            'integration_stats': self.integration_stats,
            'trading_summary': {
                'total_trades': len(all_trades),
                'ai_driven_trades': len(ai_trades),
                'total_volume': total_volume,
                'ai_volume': ai_volume,
                'ai_volume_percentage': (ai_volume / total_volume * 100) if total_volume > 0 else 0,
                'average_ai_confidence': avg_ai_confidence
            },
            'recent_predictions': list(self.ai_predictions_cache.values())[-10:],
            'quantum_integration': HAS_QUANTUM_AI,
            'data_persistence': HAS_DATA_PERSISTENCE
        }
    
    def mine_next_block(self) -> Optional[Block]:
        """次ブロックマイニング"""
        return self.blockchain.mine_block()
    
    def get_portfolio_on_chain(self, address: str) -> Dict[str, Any]:
        """オンチェーンポートフォリオ取得"""
        balances = self.blockchain.get_balance(address)
        transaction_history = self.blockchain.get_transaction_history(address, 50)
        
        # ポートフォリオ価値計算
        total_value = 0.0
        for symbol, quantity in balances.items():
            # 最新価格取得（簡略化）
            latest_price = 1500.0  # 実際の実装では市場データから取得
            total_value += quantity * latest_price
        
        return {
            'address': address,
            'balances': balances,
            'total_value': total_value,
            'transaction_count': len(transaction_history),
            'recent_transactions': [
                {
                    'tx_id': tx.tx_id,
                    'symbol': tx.symbol,
                    'action': tx.action,
                    'quantity': tx.quantity,
                    'price': tx.price,
                    'timestamp': tx.timestamp.isoformat(),
                    'ai_confidence': tx.ai_confidence
                }
                for tx in transaction_history[:10]
            ]
        }


# グローバルインスタンス
trading_blockchain = TradingBlockchainIntegration()


if __name__ == "__main__":
    print("=== Blockchain Trading System Test ===")
    
    # AI予測記録テスト
    if HAS_QUANTUM_AI:
        print("\nQuantum AI Prediction Recording:")
        market_data = [1500 + i * 5 for i in range(20)]
        prediction = quantum_ai_engine.quantum_market_analysis("7203", market_data)
        
        tx_id = trading_blockchain.record_ai_prediction("7203", prediction)
        print(f"Prediction recorded: {tx_id}")
        
        print(f"Quantum Confidence: {prediction.quantum_confidence:.3f}")
        print(f"Hybrid Confidence: {prediction.hybrid_confidence:.3f}")
    
    # 自動取引テスト
    print("\nAutomated Trade Execution:")
    trade_tx = trading_blockchain.execute_automated_trade("7203", "BUY", 100.0, 1550.0)
    print(f"Trade executed: {trade_tx}")
    
    # ブロックマイニング
    print("\nMining Block:")
    new_block = trading_blockchain.mine_next_block()
    if new_block:
        print(f"Block mined: {new_block.index} - {new_block.hash}")
        print(f"Transactions in block: {len(new_block.transactions)}")
    
    # ポートフォリオ確認
    print("\nOn-chain Portfolio:")
    portfolio = trading_blockchain.get_portfolio_on_chain(trading_blockchain.node_id)
    print(f"Total Value: {portfolio['total_value']:.2f}")
    print(f"Balances: {portfolio['balances']}")
    
    # 取引分析
    print("\n=== Trading Analytics ===")
    analytics = trading_blockchain.get_trading_analytics()
    
    print(f"Total Trades: {analytics['trading_summary']['total_trades']}")
    print(f"AI-Driven Trades: {analytics['trading_summary']['ai_driven_trades']}")
    print(f"Total Volume: {analytics['trading_summary']['total_volume']:.2f}")
    print(f"Average AI Confidence: {analytics['trading_summary']['average_ai_confidence']:.3f}")
    
    # ブロックチェーン統計
    print(f"\nBlockchain Length: {analytics['blockchain_stats']['chain_length']}")
    print(f"Chain Valid: {analytics['blockchain_stats']['is_valid_chain']}")
    print(f"Smart Contracts: {analytics['blockchain_stats']['smart_contracts_deployed']}")
    print(f"Quantum Integration: {analytics['quantum_integration']}")