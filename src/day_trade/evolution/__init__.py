#!/usr/bin/env python3
"""
Self-Evolution System
自己進化システム

This module implements self-evolving trading systems that continuously
improve through genetic algorithms and autonomous learning.
"""

from .self_evolution_engine import (
    SelfEvolutionEngine,
    EvolutionEngine,
    GeneticOperator,
    FitnessEvaluator,
    ArchitectureEvolution,
    GeneticCode,
    Chromosome,
    Genome,
    EvolutionaryTrait,
    FitnessEvaluation,
    EvolutionaryEvent,
    EvolutionStage,
    MutationType,
    SelectionPressure,
    FitnessMetric,
    start_system_evolution,
    get_evolution_status,
    stop_system_evolution
)

__all__ = [
    # Core Evolution Engine
    'SelfEvolutionEngine',
    'EvolutionEngine',
    'GeneticOperator', 
    'FitnessEvaluator',
    'ArchitectureEvolution',
    
    # Genetic Data Structures
    'GeneticCode',
    'Chromosome',
    'Genome',
    'EvolutionaryTrait',
    'FitnessEvaluation',
    'EvolutionaryEvent',
    
    # Enums
    'EvolutionStage',
    'MutationType',
    'SelectionPressure', 
    'FitnessMetric',
    
    # Global Functions
    'start_system_evolution',
    'get_evolution_status',
    'stop_system_evolution'
]