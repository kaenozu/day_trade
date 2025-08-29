#!/usr/bin/env python3
"""
Blockchain Engine Implementation
ブロックチェーンエンジン実装
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
import uuid
import logging
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from ..functional.monads import Either, TradingResult

logger = logging.getLogger(__name__)

class TransactionType(Enum):
    """取引タイプ"""
    TRADE = "trade"
    SETTLEMENT = "settlement"
    SMART_CONTRACT = "smart_contract"
    TOKEN_TRANSFER = "token_transfer"
    GOVERNANCE = "governance"

class TransactionStatus(Enum):
    """取引状態"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REJECTED = "rejected"

class ConsensusType(Enum):
    """合意アルゴリズムタイプ"""
    PROOF_OF_WORK = "proof_of_work"
    PROOF_OF_STAKE = "proof_of_stake"
    PRACTICAL_BFT = "practical_bft"
    RAFT = "raft"

@dataclass
class Transaction:
    """ブロックチェーン取引"""
    transaction_id: str
    transaction_type: TransactionType
    from_address: str
    to_address: str
    amount: float
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    nonce: int = 0
    gas_limit: int = 21000
    gas_price: float = 0.001
    signature: Optional[str] = None
    status: TransactionStatus = TransactionStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書変換"""
        return {
            'transaction_id': self.transaction_id,
            'type': self.transaction_type.value,
            'from': self.from_address,
            'to': self.to_address,
            'amount': self.amount,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'nonce': self.nonce,
            'gas_limit': self.gas_limit,
            'gas_price': self.gas_price,
            'signature': self.signature,
            'status': self.status.value
        }
    
    def calculate_hash(self) -> str:
        """取引ハッシュ計算"""
        transaction_string = json.dumps({
            'transaction_id': self.transaction_id,
            'type': self.transaction_type.value,
            'from': self.from_address,
            'to': self.to_address,
            'amount': self.amount,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'nonce': self.nonce
        }, sort_keys=True)
        
        return hashlib.sha256(transaction_string.encode()).hexdigest()
    
    def calculate_fee(self) -> float:
        """取引手数料計算"""
        return self.gas_limit * self.gas_price

@dataclass  
class Block:
    """ブロック"""
    block_number: int
    previous_hash: str
    transactions: List[Transaction] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    nonce: int = 0
    difficulty: int = 4
    miner_address: str = ""
    
    def calculate_merkle_root(self) -> str:
        """マークルルート計算"""
        if not self.transactions:
            return "0" * 64
        
        # 取引ハッシュ取得
        tx_hashes = [tx.calculate_hash() for tx in self.transactions]
        
        # マークルツリー構築
        while len(tx_hashes) > 1:
            next_level = []
            
            for i in range(0, len(tx_hashes), 2):
                if i + 1 < len(tx_hashes):
                    combined = tx_hashes[i] + tx_hashes[i + 1]
                else:
                    combined = tx_hashes[i] + tx_hashes[i]  # 奇数の場合は自分自身と結合
                
                hash_obj = hashlib.sha256(combined.encode())
                next_level.append(hash_obj.hexdigest())
            
            tx_hashes = next_level
        
        return tx_hashes[0]
    
    def calculate_hash(self) -> str:
        """ブロックハッシュ計算"""
        merkle_root = self.calculate_merkle_root()
        
        block_string = json.dumps({
            'block_number': self.block_number,
            'previous_hash': self.previous_hash,
            'merkle_root': merkle_root,
            'timestamp': self.timestamp.isoformat(),
            'nonce': self.nonce,
            'difficulty': self.difficulty,
            'miner_address': self.miner_address
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def is_valid(self, previous_block: Optional['Block'] = None) -> bool:
        """ブロック妥当性検証"""
        # ハッシュ検証
        calculated_hash = self.calculate_hash()
        if not calculated_hash.startswith('0' * self.difficulty):
            return False
        
        # 前ブロックハッシュ検証
        if previous_block and self.previous_hash != previous_block.calculate_hash():
            return False
        
        # 取引妥当性検証
        for tx in self.transactions:
            if not self._validate_transaction(tx):
                return False
        
        return True
    
    def _validate_transaction(self, transaction: Transaction) -> bool:
        """取引妥当性検証"""
        # 基本検証
        if transaction.amount < 0:
            return False
        
        if not transaction.from_address or not transaction.to_address:
            return False
        
        # 署名検証（簡略化）
        if not transaction.signature:
            return False
        
        return True

@dataclass
class SmartContract:
    """スマートコントラクト"""
    contract_id: str
    contract_address: str
    code: str
    abi: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    creator: str = ""
    creation_block: int = 0
    gas_limit: int = 1000000
    
    def execute_function(self, function_name: str, parameters: Dict[str, Any],
                        caller: str) -> TradingResult[Any]:
        """関数実行"""
        try:
            if function_name not in self.abi:
                return TradingResult.failure('FUNCTION_NOT_FOUND', f'Function {function_name} not found')
            
            # 簡略化された実行エンジン
            if function_name == 'transfer':
                return self._execute_transfer(parameters, caller)
            elif function_name == 'approve':
                return self._execute_approve(parameters, caller)
            elif function_name == 'trade':
                return self._execute_trade(parameters, caller)
            else:
                return TradingResult.failure('UNSUPPORTED_FUNCTION', f'Function {function_name} not supported')
                
        except Exception as e:
            return TradingResult.failure('EXECUTION_ERROR', str(e))
    
    def _execute_transfer(self, parameters: Dict[str, Any], caller: str) -> TradingResult[Any]:
        """転送実行"""
        to_address = parameters.get('to')
        amount = parameters.get('amount', 0)
        
        # 残高確認
        caller_balance = self.state.get('balances', {}).get(caller, 0)
        if caller_balance < amount:
            return TradingResult.failure('INSUFFICIENT_BALANCE', 'Insufficient balance')
        
        # 残高更新
        if 'balances' not in self.state:
            self.state['balances'] = {}
        
        self.state['balances'][caller] = caller_balance - amount
        self.state['balances'][to_address] = self.state['balances'].get(to_address, 0) + amount
        
        return TradingResult.success({'success': True, 'amount': amount})
    
    def _execute_approve(self, parameters: Dict[str, Any], caller: str) -> TradingResult[Any]:
        """承認実行"""
        spender = parameters.get('spender')
        amount = parameters.get('amount', 0)
        
        if 'allowances' not in self.state:
            self.state['allowances'] = {}
        
        if caller not in self.state['allowances']:
            self.state['allowances'][caller] = {}
        
        self.state['allowances'][caller][spender] = amount
        
        return TradingResult.success({'success': True, 'amount': amount})
    
    def _execute_trade(self, parameters: Dict[str, Any], caller: str) -> TradingResult[Any]:
        """取引実行"""
        symbol = parameters.get('symbol')
        action = parameters.get('action')  # 'buy' or 'sell'
        quantity = parameters.get('quantity', 0)
        price = parameters.get('price', 0)
        
        # 取引記録
        trade_id = str(uuid.uuid4())
        trade_record = {
            'trade_id': trade_id,
            'trader': caller,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if 'trades' not in self.state:
            self.state['trades'] = []
        
        self.state['trades'].append(trade_record)
        
        return TradingResult.success(trade_record)


class ConsensusEngine(ABC):
    """合意エンジン基底クラス"""
    
    @abstractmethod
    async def validate_block(self, block: Block, previous_block: Block) -> TradingResult[bool]:
        """ブロック検証"""
        pass
    
    @abstractmethod
    async def mine_block(self, transactions: List[Transaction], 
                        previous_block: Block, miner_address: str) -> TradingResult[Block]:
        """ブロックマイニング"""
        pass


class ProofOfWorkConsensus(ConsensusEngine):
    """Proof of Work合意"""
    
    def __init__(self, difficulty: int = 4):
        self.difficulty = difficulty
    
    async def validate_block(self, block: Block, previous_block: Block) -> TradingResult[bool]:
        """ブロック検証"""
        try:
            # 基本妥当性検証
            if not block.is_valid(previous_block):
                return TradingResult.success(False)
            
            # PoW検証
            block_hash = block.calculate_hash()
            if not block_hash.startswith('0' * self.difficulty):
                return TradingResult.success(False)
            
            return TradingResult.success(True)
            
        except Exception as e:
            return TradingResult.failure('VALIDATION_ERROR', str(e))
    
    async def mine_block(self, transactions: List[Transaction],
                        previous_block: Block, miner_address: str) -> TradingResult[Block]:
        """ブロックマイニング"""
        try:
            block = Block(
                block_number=previous_block.block_number + 1,
                previous_hash=previous_block.calculate_hash(),
                transactions=transactions,
                difficulty=self.difficulty,
                miner_address=miner_address
            )
            
            # PoW実行
            start_time = time.time()
            nonce = 0
            target = '0' * self.difficulty
            
            while True:
                block.nonce = nonce
                block_hash = block.calculate_hash()
                
                if block_hash.startswith(target):
                    mining_time = time.time() - start_time
                    logger.info(f"Block mined in {mining_time:.2f}s with nonce {nonce}")
                    break
                
                nonce += 1
                
                # 長時間のマイニング防止
                if nonce > 1000000:
                    return TradingResult.failure('MINING_TIMEOUT', 'Mining took too long')
            
            return TradingResult.success(block)
            
        except Exception as e:
            return TradingResult.failure('MINING_ERROR', str(e))


class ProofOfStakeConsensus(ConsensusEngine):
    """Proof of Stake合意"""
    
    def __init__(self, validators: Dict[str, float]):
        self.validators = validators  # address -> stake amount
        self.total_stake = sum(validators.values())
    
    async def validate_block(self, block: Block, previous_block: Block) -> TradingResult[bool]:
        """ブロック検証"""
        try:
            # 基本妥当性検証
            if not block.is_valid(previous_block):
                return TradingResult.success(False)
            
            # バリデーター検証
            if block.miner_address not in self.validators:
                return TradingResult.success(False)
            
            return TradingResult.success(True)
            
        except Exception as e:
            return TradingResult.failure('VALIDATION_ERROR', str(e))
    
    async def mine_block(self, transactions: List[Transaction],
                        previous_block: Block, miner_address: str) -> TradingResult[Block]:
        """ブロック生成（PoS）"""
        try:
            # バリデーター確認
            if miner_address not in self.validators:
                return TradingResult.failure('INVALID_VALIDATOR', 'Not a valid validator')
            
            # ステーク比例選択（簡略化）
            import random
            validator_stake = self.validators[miner_address]
            selection_probability = validator_stake / self.total_stake
            
            if random.random() > selection_probability:
                return TradingResult.failure('NOT_SELECTED', 'Validator not selected for this block')
            
            block = Block(
                block_number=previous_block.block_number + 1,
                previous_hash=previous_block.calculate_hash(),
                transactions=transactions,
                difficulty=1,  # PoSでは難易度不要
                miner_address=miner_address
            )
            
            return TradingResult.success(block)
            
        except Exception as e:
            return TradingResult.failure('BLOCK_CREATION_ERROR', str(e))


class BlockchainEngine:
    """ブロックチェーンエンジン"""
    
    def __init__(self, consensus: ConsensusEngine, genesis_block: Optional[Block] = None):
        self.consensus = consensus
        self.chain: List[Block] = [genesis_block or self._create_genesis_block()]
        self.pending_transactions: List[Transaction] = []
        self.smart_contracts: Dict[str, SmartContract] = {}
        self.accounts: Dict[str, Dict[str, Any]] = {}  # address -> account info
        self.mining_reward = 10.0
        
    def _create_genesis_block(self) -> Block:
        """ジェネシスブロック作成"""
        return Block(
            block_number=0,
            previous_hash="0" * 64,
            transactions=[],
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            difficulty=4
        )
    
    async def submit_transaction(self, transaction: Transaction) -> TradingResult[str]:
        """取引投入"""
        try:
            # 取引検証
            validation_result = await self._validate_transaction(transaction)
            if validation_result.is_left():
                return validation_result
            
            # 取引プールに追加
            self.pending_transactions.append(transaction)
            
            logger.info(f"Transaction {transaction.transaction_id} submitted")
            
            return TradingResult.success(transaction.transaction_id)
            
        except Exception as e:
            return TradingResult.failure('TRANSACTION_SUBMISSION_ERROR', str(e))
    
    async def mine_next_block(self, miner_address: str, 
                            max_transactions: int = 10) -> TradingResult[Block]:
        """次ブロックマイニング"""
        try:
            if not self.pending_transactions:
                return TradingResult.failure('NO_TRANSACTIONS', 'No pending transactions')
            
            # 取引選択（手数料順）
            selected_transactions = sorted(
                self.pending_transactions[:max_transactions],
                key=lambda tx: tx.calculate_fee(),
                reverse=True
            )
            
            # マイニング報酬取引追加
            reward_transaction = Transaction(
                transaction_id=str(uuid.uuid4()),
                transaction_type=TransactionType.SETTLEMENT,
                from_address="system",
                to_address=miner_address,
                amount=self.mining_reward
            )
            selected_transactions.append(reward_transaction)
            
            # 前ブロック取得
            previous_block = self.chain[-1]
            
            # ブロックマイニング
            mining_result = await self.consensus.mine_block(
                selected_transactions, previous_block, miner_address
            )
            
            if mining_result.is_left():
                return mining_result
            
            new_block = mining_result.get_right()
            
            # ブロックチェーンに追加
            self.chain.append(new_block)
            
            # 処理済み取引を保留中から削除
            for tx in selected_transactions[:-1]:  # 報酬取引を除く
                if tx in self.pending_transactions:
                    self.pending_transactions.remove(tx)
                    tx.status = TransactionStatus.CONFIRMED
            
            # アカウント残高更新
            await self._update_account_balances(selected_transactions)
            
            logger.info(f"Block {new_block.block_number} mined with {len(selected_transactions)} transactions")
            
            return TradingResult.success(new_block)
            
        except Exception as e:
            return TradingResult.failure('MINING_ERROR', str(e))
    
    async def deploy_smart_contract(self, contract: SmartContract,
                                  deployer_address: str) -> TradingResult[str]:
        """スマートコントラクトデプロイ"""
        try:
            # デプロイ取引作成
            deploy_transaction = Transaction(
                transaction_id=str(uuid.uuid4()),
                transaction_type=TransactionType.SMART_CONTRACT,
                from_address=deployer_address,
                to_address="",
                amount=0.0,
                data={
                    'action': 'deploy',
                    'contract_id': contract.contract_id,
                    'code': contract.code
                }
            )
            
            # 取引投入
            submission_result = await self.submit_transaction(deploy_transaction)
            if submission_result.is_left():
                return submission_result
            
            # コントラクト登録
            contract.creator = deployer_address
            contract.creation_block = len(self.chain)
            self.smart_contracts[contract.contract_id] = contract
            
            logger.info(f"Smart contract {contract.contract_id} deployed")
            
            return TradingResult.success(contract.contract_address)
            
        except Exception as e:
            return TradingResult.failure('CONTRACT_DEPLOYMENT_ERROR', str(e))
    
    async def call_smart_contract(self, contract_id: str, function_name: str,
                                parameters: Dict[str, Any], caller: str) -> TradingResult[Any]:
        """スマートコントラクト呼び出し"""
        try:
            if contract_id not in self.smart_contracts:
                return TradingResult.failure('CONTRACT_NOT_FOUND', f'Contract {contract_id} not found')
            
            contract = self.smart_contracts[contract_id]
            
            # 関数実行
            execution_result = contract.execute_function(function_name, parameters, caller)
            
            if execution_result.is_right():
                # 呼び出し記録取引作成
                call_transaction = Transaction(
                    transaction_id=str(uuid.uuid4()),
                    transaction_type=TransactionType.SMART_CONTRACT,
                    from_address=caller,
                    to_address=contract.contract_address,
                    amount=0.0,
                    data={
                        'action': 'call',
                        'function': function_name,
                        'parameters': parameters,
                        'result': execution_result.get_right()
                    }
                )
                
                await self.submit_transaction(call_transaction)
            
            return execution_result
            
        except Exception as e:
            return TradingResult.failure('CONTRACT_CALL_ERROR', str(e))
    
    async def get_account_balance(self, address: str) -> float:
        """アカウント残高取得"""
        return self.accounts.get(address, {}).get('balance', 0.0)
    
    async def get_transaction_by_id(self, transaction_id: str) -> Optional[Transaction]:
        """取引ID検索"""
        # 確認済み取引検索
        for block in reversed(self.chain):
            for tx in block.transactions:
                if tx.transaction_id == transaction_id:
                    return tx
        
        # 保留中取引検索
        for tx in self.pending_transactions:
            if tx.transaction_id == transaction_id:
                return tx
        
        return None
    
    async def get_block_by_number(self, block_number: int) -> Optional[Block]:
        """ブロック番号検索"""
        if 0 <= block_number < len(self.chain):
            return self.chain[block_number]
        return None
    
    def get_blockchain_info(self) -> Dict[str, Any]:
        """ブロックチェーン情報取得"""
        return {
            'chain_length': len(self.chain),
            'pending_transactions': len(self.pending_transactions),
            'smart_contracts': len(self.smart_contracts),
            'accounts': len(self.accounts),
            'latest_block': {
                'number': self.chain[-1].block_number,
                'hash': self.chain[-1].calculate_hash(),
                'transactions': len(self.chain[-1].transactions),
                'timestamp': self.chain[-1].timestamp.isoformat()
            } if self.chain else None
        }
    
    async def _validate_transaction(self, transaction: Transaction) -> TradingResult[None]:
        """取引検証"""
        try:
            # 基本検証
            if transaction.amount < 0:
                return TradingResult.failure('INVALID_AMOUNT', 'Amount cannot be negative')
            
            if not transaction.from_address or not transaction.to_address:
                return TradingResult.failure('INVALID_ADDRESS', 'Invalid from/to address')
            
            # 残高検証
            if transaction.from_address != "system":  # システム取引以外
                sender_balance = await self.get_account_balance(transaction.from_address)
                required_amount = transaction.amount + transaction.calculate_fee()
                
                if sender_balance < required_amount:
                    return TradingResult.failure('INSUFFICIENT_FUNDS', 'Insufficient balance')
            
            return TradingResult.success(None)
            
        except Exception as e:
            return TradingResult.failure('VALIDATION_ERROR', str(e))
    
    async def _update_account_balances(self, transactions: List[Transaction]) -> None:
        """アカウント残高更新"""
        for tx in transactions:
            # 送信者残高減少
            if tx.from_address != "system":
                if tx.from_address not in self.accounts:
                    self.accounts[tx.from_address] = {'balance': 0.0}
                
                self.accounts[tx.from_address]['balance'] -= (tx.amount + tx.calculate_fee())
            
            # 受信者残高増加
            if tx.to_address not in self.accounts:
                self.accounts[tx.to_address] = {'balance': 0.0}
            
            self.accounts[tx.to_address]['balance'] += tx.amount
    
    async def validate_chain(self) -> TradingResult[bool]:
        """チェーン妥当性検証"""
        try:
            for i in range(1, len(self.chain)):
                current_block = self.chain[i]
                previous_block = self.chain[i - 1]
                
                validation_result = await self.consensus.validate_block(current_block, previous_block)
                
                if validation_result.is_left() or not validation_result.get_right():
                    return TradingResult.success(False)
            
            return TradingResult.success(True)
            
        except Exception as e:
            return TradingResult.failure('CHAIN_VALIDATION_ERROR', str(e))