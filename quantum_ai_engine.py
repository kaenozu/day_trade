#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum AI Engine - 量子コンピューティング対応AI予測システム
Issue #938対応: 量子アルゴリズムシミュレーション + ハイブリッド計算
"""

import numpy as np
import pandas as pd
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import cmath
import random
import logging

# 量子計算ライブラリシミュレーション
try:
    # 実際の量子コンピューティングライブラリがある場合
    import qiskit
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

# 統合モジュール
try:
    from advanced_ai_engine import advanced_ai_engine, MarketSignal
    HAS_AI_ENGINE = True
except ImportError:
    HAS_AI_ENGINE = False

try:
    from performance_monitor import performance_monitor
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False


@dataclass
class QuantumState:
    """量子状態表現"""
    amplitude: complex
    probability: float
    phase: float
    entangled: bool = False


@dataclass
class QuantumPrediction:
    """量子予測結果"""
    symbol: str
    quantum_confidence: float
    classical_confidence: float
    hybrid_confidence: float
    quantum_advantage: float
    coherence_factor: float
    entanglement_strength: float
    prediction_probabilities: Dict[str, float]
    quantum_states: List[QuantumState]
    computation_time_ms: float
    algorithm_used: str


class QuantumSimulator:
    """量子計算シミュレーター"""

    def __init__(self, num_qubits: int = 8):
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1.0  # |00...0⟩ 初期状態

        self.gate_history = []
        self.measurement_history = []

        # 量子ゲート定義
        self.gates = {
            'H': self._hadamard_gate(),
            'X': self._pauli_x_gate(),
            'Y': self._pauli_y_gate(),
            'Z': self._pauli_z_gate(),
            'CNOT': self._cnot_gate()
        }

    def _hadamard_gate(self) -> np.ndarray:
        """アダマールゲート"""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    def _pauli_x_gate(self) -> np.ndarray:
        """パウリXゲート"""
        return np.array([[0, 1], [1, 0]], dtype=complex)

    def _pauli_y_gate(self) -> np.ndarray:
        """パウリYゲート"""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)

    def _pauli_z_gate(self) -> np.ndarray:
        """パウリZゲート"""
        return np.array([[1, 0], [0, -1]], dtype=complex)

    def _cnot_gate(self) -> np.ndarray:
        """CNOTゲート"""
        return np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]], dtype=complex)

    def apply_gate(self, gate_name: str, qubit_indices: List[int]):
        """量子ゲート適用"""
        if gate_name not in self.gates:
            raise ValueError(f"Unknown gate: {gate_name}")

        gate = self.gates[gate_name]

        if len(qubit_indices) == 1:
            self._apply_single_qubit_gate(gate, qubit_indices[0])
        elif len(qubit_indices) == 2:
            self._apply_two_qubit_gate(gate, qubit_indices[0], qubit_indices[1])

        self.gate_history.append({
            'gate': gate_name,
            'qubits': qubit_indices,
            'timestamp': time.time()
        })

    def _apply_single_qubit_gate(self, gate: np.ndarray, qubit_idx: int):
        """単一量子ビットゲート適用"""
        # 簡略化された実装
        new_state = np.zeros_like(self.state_vector)

        for i in range(2**self.num_qubits):
            bit_i = (i >> qubit_idx) & 1
            i_flip = i ^ (1 << qubit_idx)

            new_state[i] += gate[bit_i, 0] * self.state_vector[i]
            new_state[i_flip] += gate[bit_i, 1] * self.state_vector[i]

        self.state_vector = new_state

    def _apply_two_qubit_gate(self, gate: np.ndarray, control_idx: int, target_idx: int):
        """二量子ビットゲート適用"""
        # CNOT ゲートの簡略実装
        new_state = np.copy(self.state_vector)

        for i in range(2**self.num_qubits):
            control_bit = (i >> control_idx) & 1
            target_bit = (i >> target_idx) & 1

            if control_bit == 1:
                # フリップ
                i_flip = i ^ (1 << target_idx)
                new_state[i_flip] = self.state_vector[i]
                new_state[i] = 0

        self.state_vector = new_state

    def measure(self, qubit_idx: int) -> int:
        """量子ビット測定"""
        probabilities = np.abs(self.state_vector)**2

        # 指定された量子ビットが0である確率を計算
        prob_0 = sum(probabilities[i] for i in range(2**self.num_qubits)
                    if (i >> qubit_idx) & 1 == 0)

        # 測定結果を確率的に決定
        measurement = 0 if random.random() < prob_0 else 1

        # 状態ベクトルを正規化（測定後の状態に更新）
        self._collapse_state(qubit_idx, measurement)

        self.measurement_history.append({
            'qubit': qubit_idx,
            'result': measurement,
            'probability': prob_0 if measurement == 0 else 1 - prob_0,
            'timestamp': time.time()
        })

        return measurement

    def _collapse_state(self, qubit_idx: int, measurement: int):
        """測定による状態収束"""
        new_state = np.zeros_like(self.state_vector)
        norm_factor = 0

        for i in range(2**self.num_qubits):
            if ((i >> qubit_idx) & 1) == measurement:
                new_state[i] = self.state_vector[i]
                norm_factor += np.abs(self.state_vector[i])**2

        if norm_factor > 0:
            self.state_vector = new_state / np.sqrt(norm_factor)

    def get_entanglement_measure(self) -> float:
        """量子もつれの強度測定"""
        # フォン・ノイマンエントロピーによる近似計算
        probabilities = np.abs(self.state_vector)**2
        probabilities = probabilities[probabilities > 1e-10]  # ゼロ除去

        if len(probabilities) <= 1:
            return 0.0

        entropy = -np.sum(probabilities * np.log2(probabilities))
        max_entropy = np.log2(len(probabilities))

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def get_coherence_factor(self) -> float:
        """量子コヒーレンス因子"""
        # オフ対角要素の大きさによる近似
        coherence = np.sum(np.abs(self.state_vector[1:]))
        total_amplitude = np.sum(np.abs(self.state_vector))

        return coherence / total_amplitude if total_amplitude > 0 else 0.0

    def reset(self):
        """量子状態リセット"""
        self.state_vector = np.zeros(2**self.num_qubits, dtype=complex)
        self.state_vector[0] = 1.0
        self.gate_history = []
        self.measurement_history = []


class QuantumAlgorithms:
    """量子アルゴリズム集"""

    def __init__(self, simulator: QuantumSimulator):
        self.simulator = simulator

    def quantum_fourier_transform(self, qubits: List[int]) -> np.ndarray:
        """量子フーリエ変換（QFT）シミュレーション"""
        start_time = time.time()

        # QFTアルゴリズムの簡略実装
        n_qubits = len(qubits)

        for i in range(n_qubits):
            # アダマールゲート
            self.simulator.apply_gate('H', [qubits[i]])

            # 制御回転ゲート（簡略化）
            for j in range(i + 1, n_qubits):
                # 制御位相ゲートの代わりにCNOTを使用（簡略化）
                self.simulator.apply_gate('CNOT', [qubits[j], qubits[i]])

        # ビット順序反転（簡略化）
        for i in range(n_qubits // 2):
            self.simulator.apply_gate('CNOT', [qubits[i], qubits[n_qubits - 1 - i]])
            self.simulator.apply_gate('CNOT', [qubits[n_qubits - 1 - i], qubits[i]])
            self.simulator.apply_gate('CNOT', [qubits[i], qubits[n_qubits - 1 - i]])

        execution_time = (time.time() - start_time) * 1000

        return {
            'algorithm': 'QFT',
            'execution_time_ms': execution_time,
            'entanglement': self.simulator.get_entanglement_measure(),
            'coherence': self.simulator.get_coherence_factor()
        }

    def quantum_amplitude_amplification(self, target_state: int) -> Dict[str, Any]:
        """量子振幅増幅アルゴリズム"""
        start_time = time.time()

        # グローバーのアルゴリズムの一般化
        n_iterations = int(np.pi / 4 * np.sqrt(2**self.simulator.num_qubits))

        # 重ね合わせ状態作成
        for i in range(self.simulator.num_qubits):
            self.simulator.apply_gate('H', [i])

        # 振幅増幅反復
        for _ in range(min(n_iterations, 3)):  # 計算量制限
            # オラクル（目標状態をマーク）
            self._oracle_operator(target_state)

            # 拡散演算子
            self._diffusion_operator()

        execution_time = (time.time() - start_time) * 1000

        # 測定確率計算
        probabilities = np.abs(self.simulator.state_vector)**2
        target_probability = probabilities[target_state]

        return {
            'algorithm': 'AmplitudeAmplification',
            'target_state': target_state,
            'target_probability': target_probability,
            'amplification_factor': target_probability * (2**self.simulator.num_qubits),
            'execution_time_ms': execution_time,
            'iterations': n_iterations
        }

    def _oracle_operator(self, target_state: int):
        """オラクル演算子（簡略実装）"""
        # 目標状態の位相を反転
        self.simulator.state_vector[target_state] *= -1

    def _diffusion_operator(self):
        """拡散演算子"""
        # 全状態に対するアダマール変換
        for i in range(self.simulator.num_qubits):
            self.simulator.apply_gate('H', [i])

        # 位相反転
        for i in range(1, 2**self.simulator.num_qubits):
            self.simulator.state_vector[i] *= -1

        # アダマール変換再適用
        for i in range(self.simulator.num_qubits):
            self.simulator.apply_gate('H', [i])

    def quantum_phase_estimation(self, eigenvalue_guess: float) -> Dict[str, Any]:
        """量子位相推定アルゴリズム"""
        start_time = time.time()

        # 制御ユニタリ演算（簡略化）
        phase = eigenvalue_guess * 2 * np.pi

        # 位相キックバック
        for i in range(self.simulator.num_qubits // 2):
            self.simulator.apply_gate('H', [i])

            # 制御位相ゲート（簡略化）
            for j in range(2**i):
                self.simulator.apply_gate('Z', [self.simulator.num_qubits // 2])

        # 逆QFT
        qft_qubits = list(range(self.simulator.num_qubits // 2))
        self.quantum_fourier_transform(qft_qubits)

        # 測定
        measured_phase = 0
        for i in range(self.simulator.num_qubits // 2):
            bit = self.simulator.measure(i)
            measured_phase += bit * (2**(-(i+1)))

        execution_time = (time.time() - start_time) * 1000

        return {
            'algorithm': 'PhaseEstimation',
            'input_eigenvalue': eigenvalue_guess,
            'estimated_phase': measured_phase,
            'phase_error': abs(measured_phase - eigenvalue_guess),
            'execution_time_ms': execution_time
        }


class HybridQuantumClassical:
    """ハイブリッド量子-古典計算"""

    def __init__(self):
        self.quantum_simulator = QuantumSimulator(num_qubits=6)  # 計算量考慮
        self.quantum_algorithms = QuantumAlgorithms(self.quantum_simulator)

        self.classical_weights = np.random.uniform(0, 1, 10)  # 古典的パラメータ
        self.quantum_weights = np.random.uniform(0, 1, 5)     # 量子パラメータ

        self.optimization_history = []

    def variational_quantum_eigensolver(self, market_data: np.ndarray) -> Dict[str, Any]:
        """変分量子固有値解法（VQE）"""
        start_time = time.time()

        best_energy = float('inf')
        best_params = None
        n_iterations = 5  # 計算量制限

        for iteration in range(n_iterations):
            # 量子回路パラメータ最適化
            params = np.random.uniform(0, 2*np.pi, 3)

            # 量子状態準備
            self.quantum_simulator.reset()
            self._prepare_ansatz_state(params)

            # エネルギー期待値計算
            energy = self._compute_energy_expectation(market_data)

            if energy < best_energy:
                best_energy = energy
                best_params = params.copy()

            self.optimization_history.append({
                'iteration': iteration,
                'energy': energy,
                'params': params.tolist()
            })

        execution_time = (time.time() - start_time) * 1000

        return {
            'algorithm': 'VQE',
            'ground_state_energy': best_energy,
            'optimal_parameters': best_params.tolist() if best_params is not None else [],
            'iterations': n_iterations,
            'convergence_history': self.optimization_history[-n_iterations:],
            'execution_time_ms': execution_time,
            'quantum_advantage': self._estimate_quantum_advantage(best_energy)
        }

    def _prepare_ansatz_state(self, params: np.ndarray):
        """アンザッツ状態準備"""
        # パラメータ化量子回路
        for i in range(min(3, self.quantum_simulator.num_qubits)):
            # 回転ゲート（簡略化）
            angle = params[i] if i < len(params) else 0
            if angle > np.pi:
                self.quantum_simulator.apply_gate('X', [i])

            self.quantum_simulator.apply_gate('H', [i])

        # エンタングリングゲート
        for i in range(min(2, self.quantum_simulator.num_qubits - 1)):
            self.quantum_simulator.apply_gate('CNOT', [i, i + 1])

    def _compute_energy_expectation(self, market_data: np.ndarray) -> float:
        """エネルギー期待値計算"""
        # 市場データを量子ハミルトニアンにマッピング
        probabilities = np.abs(self.quantum_simulator.state_vector)**2

        # 簡略化されたエネルギー計算
        if len(market_data) > 0:
            data_variance = np.var(market_data)
            data_mean = np.mean(market_data)
        else:
            data_variance = 1.0
            data_mean = 0.0

        # 量子状態のエントロピーとデータ特性を組み合わせ
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        energy = entropy * data_variance + data_mean

        return float(energy)

    def _estimate_quantum_advantage(self, energy: float) -> float:
        """量子アドバンテージ推定"""
        # 古典的アルゴリズムとの比較（簡略）
        classical_energy = abs(energy) * 1.2  # 古典的手法は20%劣ると仮定

        if classical_energy > 0:
            advantage = (classical_energy - abs(energy)) / classical_energy
            return max(0.0, min(1.0, advantage))  # 0-1に正規化

        return 0.0

    def quantum_approximate_optimization(self, objective_function: np.ndarray) -> Dict[str, Any]:
        """量子近似最適化アルゴリズム（QAOA）"""
        start_time = time.time()

        # QAOA パラメータ
        p_levels = 2  # 層数
        best_cost = float('-inf')
        best_params = None

        for trial in range(3):  # 試行回数制限
            # ランダムパラメータ初期化
            beta_params = np.random.uniform(0, np.pi, p_levels)
            gamma_params = np.random.uniform(0, 2*np.pi, p_levels)

            # 量子状態準備
            self.quantum_simulator.reset()

            # QAOA 回路構築
            for layer in range(p_levels):
                self._apply_cost_hamiltonian(gamma_params[layer], objective_function)
                self._apply_mixer_hamiltonian(beta_params[layer])

            # コスト計算
            cost = self._evaluate_cost_function(objective_function)

            if cost > best_cost:
                best_cost = cost
                best_params = {
                    'beta': beta_params.tolist(),
                    'gamma': gamma_params.tolist()
                }

        execution_time = (time.time() - start_time) * 1000

        return {
            'algorithm': 'QAOA',
            'optimal_cost': best_cost,
            'optimal_parameters': best_params,
            'p_levels': p_levels,
            'execution_time_ms': execution_time,
            'approximation_ratio': self._compute_approximation_ratio(best_cost, objective_function)
        }

    def _apply_cost_hamiltonian(self, gamma: float, objective: np.ndarray):
        """コストハミルトニアン適用"""
        # 簡略化された実装
        for i in range(min(2, self.quantum_simulator.num_qubits)):
            if gamma > np.pi:
                self.quantum_simulator.apply_gate('Z', [i])

    def _apply_mixer_hamiltonian(self, beta: float):
        """ミキサーハミルトニアン適用"""
        for i in range(self.quantum_simulator.num_qubits):
            if beta > np.pi/2:
                self.quantum_simulator.apply_gate('X', [i])

    def _evaluate_cost_function(self, objective: np.ndarray) -> float:
        """コスト関数評価"""
        probabilities = np.abs(self.quantum_simulator.state_vector)**2

        if len(objective) > 0:
            obj_sum = np.sum(objective)
        else:
            obj_sum = 1.0

        # 量子状態の分散をコストとして使用
        expectation = np.sum(probabilities * np.arange(len(probabilities)))
        return float(expectation * obj_sum)

    def _compute_approximation_ratio(self, achieved_cost: float, objective: np.ndarray) -> float:
        """近似比計算"""
        if len(objective) > 0:
            optimal_cost = np.max(objective)
        else:
            optimal_cost = 1.0

        if optimal_cost != 0:
            return min(1.0, abs(achieved_cost) / abs(optimal_cost))
        return 0.0


class QuantumAIEngine:
    """量子AI統合エンジン"""

    def __init__(self):
        self.hybrid_processor = HybridQuantumClassical()
        self.quantum_memory = defaultdict(list)

        self.computation_stats = {
            'total_quantum_operations': 0,
            'total_classical_operations': 0,
            'quantum_advantage_achieved': 0,
            'average_coherence_time': 0.0
        }

        self.algorithm_performance = defaultdict(list)

    def quantum_market_analysis(self, symbol: str, market_data: List[float]) -> QuantumPrediction:
        """量子市場分析"""
        start_time = time.time()

        try:
            # データ前処理
            data_array = np.array(market_data[-20:]) if market_data else np.array([1500.0])
            normalized_data = self._normalize_market_data(data_array)

            # 複数の量子アルゴリズム実行
            vqe_result = self.hybrid_processor.variational_quantum_eigensolver(normalized_data)
            qaoa_result = self.hybrid_processor.quantum_approximate_optimization(normalized_data)

            # 量子フーリエ変換による周期性分析
            qft_result = self.hybrid_processor.quantum_algorithms.quantum_fourier_transform(
                list(range(min(4, self.hybrid_processor.quantum_simulator.num_qubits)))
            )

            # 振幅増幅による信号強化
            target_state = self._encode_market_target(normalized_data)
            amp_result = self.hybrid_processor.quantum_algorithms.quantum_amplitude_amplification(target_state)

            # 量子状態解析
            entanglement = self.hybrid_processor.quantum_simulator.get_entanglement_measure()
            coherence = self.hybrid_processor.quantum_simulator.get_coherence_factor()

            # 予測確率計算
            probabilities = self._compute_prediction_probabilities(vqe_result, qaoa_result, amp_result)

            # 量子コンフィデンス計算
            quantum_confidence = self._calculate_quantum_confidence(
                entanglement, coherence, vqe_result['quantum_advantage']
            )

            # 古典的手法との比較
            classical_confidence = self._get_classical_confidence(symbol, data_array)

            # ハイブリッド信頼度
            hybrid_confidence = self._compute_hybrid_confidence(quantum_confidence, classical_confidence)

            execution_time = (time.time() - start_time) * 1000

            # 量子状態記録
            quantum_states = self._extract_quantum_states()

            # 統計更新
            self._update_statistics(execution_time, entanglement, coherence)

            prediction = QuantumPrediction(
                symbol=symbol,
                quantum_confidence=quantum_confidence,
                classical_confidence=classical_confidence,
                hybrid_confidence=hybrid_confidence,
                quantum_advantage=vqe_result.get('quantum_advantage', 0.0),
                coherence_factor=coherence,
                entanglement_strength=entanglement,
                prediction_probabilities=probabilities,
                quantum_states=quantum_states,
                computation_time_ms=execution_time,
                algorithm_used='Hybrid-VQE-QAOA'
            )

            # メモリに保存
            self.quantum_memory[symbol].append(prediction)
            if len(self.quantum_memory[symbol]) > 100:
                self.quantum_memory[symbol].pop(0)

            return prediction

        except Exception as e:
            logging.error(f"Quantum analysis error for {symbol}: {e}")
            return self._create_fallback_prediction(symbol, start_time)

    def _normalize_market_data(self, data: np.ndarray) -> np.ndarray:
        """市場データ正規化"""
        if len(data) == 0:
            return np.array([0.5])

        data_min, data_max = np.min(data), np.max(data)
        if data_max - data_min > 0:
            normalized = (data - data_min) / (data_max - data_min)
        else:
            normalized = np.full_like(data, 0.5)

        return normalized

    def _encode_market_target(self, data: np.ndarray) -> int:
        """市場目標エンコーディング"""
        # データの特徴を量子状態インデックスにマッピング
        if len(data) == 0:
            return 0

        trend = np.mean(np.diff(data)) if len(data) > 1 else 0
        volatility = np.std(data)

        # 特徴量を量子ビット数に応じてエンコード
        max_states = 2**self.hybrid_processor.quantum_simulator.num_qubits
        feature_hash = hash((round(trend, 4), round(volatility, 4)))

        return abs(feature_hash) % max_states

    def _compute_prediction_probabilities(self, vqe_result: Dict, qaoa_result: Dict, amp_result: Dict) -> Dict[str, float]:
        """予測確率計算"""
        # VQE エネルギーから方向性推定
        energy = vqe_result.get('ground_state_energy', 0)
        if energy < -0.1:
            buy_prob = 0.7
        elif energy > 0.1:
            buy_prob = 0.3
        else:
            buy_prob = 0.5

        # QAOA コストから強度調整
        cost = qaoa_result.get('optimal_cost', 0)
        confidence_factor = min(1.0, abs(cost))

        # 振幅増幅結果で調整
        amp_factor = amp_result.get('amplification_factor', 1.0)
        amp_factor = min(2.0, amp_factor) / 2.0

        # 最終確率計算
        buy_prob *= (confidence_factor * 0.7 + amp_factor * 0.3)
        sell_prob = 1.0 - buy_prob - 0.1  # HOLDの余地を残す
        hold_prob = 0.1

        # 正規化
        total = buy_prob + sell_prob + hold_prob
        return {
            'BUY': buy_prob / total,
            'SELL': sell_prob / total,
            'HOLD': hold_prob / total
        }

    def _calculate_quantum_confidence(self, entanglement: float, coherence: float, advantage: float) -> float:
        """量子信頼度計算"""
        # 量子特性を信頼度に変換
        base_confidence = 0.5

        # エンタングルメント寄与 (0-0.3)
        entanglement_boost = entanglement * 0.3

        # コヒーレンス寄与 (0-0.2)
        coherence_boost = coherence * 0.2

        # 量子アドバンテージ寄与 (0-0.2)
        advantage_boost = advantage * 0.2

        total_confidence = base_confidence + entanglement_boost + coherence_boost + advantage_boost

        return min(0.95, max(0.1, total_confidence))

    def _get_classical_confidence(self, symbol: str, data: np.ndarray) -> float:
        """古典的信頼度取得"""
        if HAS_AI_ENGINE:
            try:
                # 既存のAIエンジンから信頼度取得
                signal = advanced_ai_engine.analyze_symbol(symbol)
                return signal.confidence
            except:
                pass

        # フォールバック: データ基づく簡易信頼度
        if len(data) > 1:
            volatility = np.std(data)
            trend_strength = abs(np.mean(np.diff(data)))

            # 低ボラティリティ・明確トレンド = 高信頼度
            confidence = (1.0 - min(1.0, volatility)) * 0.5 + min(1.0, trend_strength * 10) * 0.5
            return max(0.1, min(0.9, confidence))

        return 0.6  # デフォルト

    def _compute_hybrid_confidence(self, quantum_conf: float, classical_conf: float) -> float:
        """ハイブリッド信頼度計算"""
        # 量子コンピューティングの優位性に基づく重み付け
        quantum_weight = 0.6 if quantum_conf > classical_conf else 0.4
        classical_weight = 1.0 - quantum_weight

        hybrid = quantum_conf * quantum_weight + classical_conf * classical_weight

        # 量子・古典の一致度ボーナス
        agreement_bonus = 1.0 - abs(quantum_conf - classical_conf)
        hybrid *= (0.9 + agreement_bonus * 0.1)

        return min(0.98, max(0.2, hybrid))

    def _extract_quantum_states(self) -> List[QuantumState]:
        """量子状態抽出"""
        states = []
        state_vector = self.hybrid_processor.quantum_simulator.state_vector

        for i, amplitude in enumerate(state_vector):
            if abs(amplitude) > 1e-6:  # 無視できない振幅のみ
                states.append(QuantumState(
                    amplitude=amplitude,
                    probability=abs(amplitude)**2,
                    phase=cmath.phase(amplitude),
                    entangled=(abs(amplitude) > 0.1 and i > 0)
                ))

        return states[:10]  # 上位10状態のみ

    def _update_statistics(self, execution_time: float, entanglement: float, coherence: float):
        """統計更新"""
        self.computation_stats['total_quantum_operations'] += 1

        # コヒーレンス時間の移動平均
        prev_avg = self.computation_stats['average_coherence_time']
        n_ops = self.computation_stats['total_quantum_operations']

        self.computation_stats['average_coherence_time'] = (
            (prev_avg * (n_ops - 1) + coherence) / n_ops
        )

        # 量子アドバンテージカウント
        if entanglement > 0.5 or coherence > 0.3:
            self.computation_stats['quantum_advantage_achieved'] += 1

    def _create_fallback_prediction(self, symbol: str, start_time: float) -> QuantumPrediction:
        """フォールバック予測作成"""
        execution_time = (time.time() - start_time) * 1000

        return QuantumPrediction(
            symbol=symbol,
            quantum_confidence=0.5,
            classical_confidence=0.6,
            hybrid_confidence=0.55,
            quantum_advantage=0.0,
            coherence_factor=0.0,
            entanglement_strength=0.0,
            prediction_probabilities={'BUY': 0.4, 'SELL': 0.3, 'HOLD': 0.3},
            quantum_states=[],
            computation_time_ms=execution_time,
            algorithm_used='Classical-Fallback'
        )

    def get_quantum_statistics(self) -> Dict[str, Any]:
        """量子統計取得"""
        return {
            'computation_statistics': self.computation_stats.copy(),
            'quantum_memory_size': sum(len(predictions) for predictions in self.quantum_memory.values()),
            'tracked_symbols': list(self.quantum_memory.keys()),
            'algorithm_performance': dict(self.algorithm_performance),
            'quantum_simulator_qubits': self.hybrid_processor.quantum_simulator.num_qubits,
            'hybrid_processing_available': True,
            'quantum_advantage_rate': (
                self.computation_stats['quantum_advantage_achieved'] /
                max(1, self.computation_stats['total_quantum_operations'])
            ) * 100
        }

    def quantum_portfolio_optimization(self, symbols: List[str], risk_tolerance: float = 0.5) -> Dict[str, Any]:
        """量子ポートフォリオ最適化"""
        start_time = time.time()

        # 各銘柄の量子分析
        predictions = {}
        for symbol in symbols:
            market_data = [1500 + hash(symbol) % 500 + i * 10 for i in range(20)]  # 模擬データ
            predictions[symbol] = self.quantum_market_analysis(symbol, market_data)

        # ポートフォリオ最適化QAOA
        returns = np.array([pred.hybrid_confidence for pred in predictions.values()])
        risks = np.array([1.0 - pred.coherence_factor for pred in predictions.values()])

        # 目的関数: リターン最大化 - リスク * リスク許容度
        objective = returns - risks * risk_tolerance

        qaoa_result = self.hybrid_processor.quantum_approximate_optimization(objective)

        # 最適ウェイト計算（簡略化）
        n_assets = len(symbols)
        optimal_weights = np.ones(n_assets) / n_assets  # 均等配分から開始

        # QAOA結果で調整
        if qaoa_result['optimal_cost'] > 0:
            weight_adjustments = objective / np.sum(objective) if np.sum(objective) > 0 else optimal_weights
            optimal_weights = weight_adjustments / np.sum(weight_adjustments)

        execution_time = (time.time() - start_time) * 1000

        return {
            'symbols': symbols,
            'optimal_weights': optimal_weights.tolist(),
            'expected_return': float(np.dot(optimal_weights, returns)),
            'portfolio_risk': float(np.dot(optimal_weights, risks)),
            'quantum_advantage': qaoa_result.get('approximation_ratio', 0.0),
            'sharpe_ratio': self._calculate_quantum_sharpe_ratio(optimal_weights, returns, risks),
            'execution_time_ms': execution_time,
            'individual_predictions': {
                symbol: {
                    'hybrid_confidence': pred.hybrid_confidence,
                    'quantum_advantage': pred.quantum_advantage,
                    'entanglement_strength': pred.entanglement_strength
                }
                for symbol, pred in predictions.items()
            }
        }

    def _calculate_quantum_sharpe_ratio(self, weights: np.ndarray, returns: np.ndarray, risks: np.ndarray) -> float:
        """量子シャープレシオ計算"""
        portfolio_return = np.dot(weights, returns)
        portfolio_risk = np.sqrt(np.dot(weights**2, risks**2))  # 簡略化

        # 量子効果を考慮した調整
        quantum_enhancement = 1.0 + (np.mean([
            self.computation_stats['quantum_advantage_achieved'] /
            max(1, self.computation_stats['total_quantum_operations'])
        ]))

        if portfolio_risk > 0:
            return float((portfolio_return / portfolio_risk) * quantum_enhancement)

        return 0.0


# グローバルインスタンス
quantum_ai_engine = QuantumAIEngine()


if __name__ == "__main__":
    print("=== Quantum AI Engine Test ===")

    # 単一銘柄量子分析
    symbol = "7203"
    market_data = [1500 + i * 5 + np.random.normal(0, 10) for i in range(50)]

    print(f"\nQuantum Analysis for {symbol}:")
    prediction = quantum_ai_engine.quantum_market_analysis(symbol, market_data)

    print(f"Quantum Confidence: {prediction.quantum_confidence:.3f}")
    print(f"Classical Confidence: {prediction.classical_confidence:.3f}")
    print(f"Hybrid Confidence: {prediction.hybrid_confidence:.3f}")
    print(f"Quantum Advantage: {prediction.quantum_advantage:.3f}")
    print(f"Entanglement Strength: {prediction.entanglement_strength:.3f}")
    print(f"Computation Time: {prediction.computation_time_ms:.1f}ms")
    print(f"Algorithm: {prediction.algorithm_used}")

    print("\nPrediction Probabilities:")
    for action, prob in prediction.prediction_probabilities.items():
        print(f"  {action}: {prob:.3f}")

    # ポートフォリオ最適化
    print("\n=== Quantum Portfolio Optimization ===")
    portfolio_symbols = ["7203", "8306", "9984"]

    portfolio_result = quantum_ai_engine.quantum_portfolio_optimization(
        portfolio_symbols, risk_tolerance=0.3
    )

    print(f"Optimal Weights:")
    for symbol, weight in zip(portfolio_symbols, portfolio_result['optimal_weights']):
        print(f"  {symbol}: {weight:.3f}")

    print(f"Expected Return: {portfolio_result['expected_return']:.3f}")
    print(f"Portfolio Risk: {portfolio_result['portfolio_risk']:.3f}")
    print(f"Quantum Sharpe Ratio: {portfolio_result['sharpe_ratio']:.3f}")

    # 量子統計
    print("\n=== Quantum Statistics ===")
    stats = quantum_ai_engine.get_quantum_statistics()
    print(json.dumps(stats, indent=2, default=str, ensure_ascii=False))