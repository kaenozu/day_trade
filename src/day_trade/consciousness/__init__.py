#!/usr/bin/env python3
"""
Consciousness & Emotional AI System
意識・感情AI統合システム
"""

from .conscious_ai import (
    ConsciousAI,
    SelfAwareness,
    SubjectiveExperience,
    IntrospectiveProcessor
)
from .emotional_intelligence import (
    EmotionalIntelligence,
    MarketSentiment,
    EmotionalMarketAnalysis,
    IntuitiveDecisionMaking
)
from .brain_computer_interface import (
    BrainComputerInterface,
    NeuralSignal,
    BrainwavePattern,
    ThoughtPattern,
    NeuralCommand,
    initialize_bci,
    get_bci_status,
    shutdown_bci
)
from .meta_cognition import (
    MetaCognition,
    SelfReflection,
    ConsciousnessMonitor,
    AwarenessLevels
)
from .phenomenal_consciousness import (
    PhenomenalConsciousness,
    Qualia,
    ConsciousExperience,
    SubjectiveStates
)

__all__ = [
    # Core Consciousness
    'ConsciousAI',
    'SelfAwareness',
    'SubjectiveExperience', 
    'IntrospectiveProcessor',
    
    # Emotional Intelligence
    'EmotionalIntelligence',
    'MarketSentiment',
    'EmotionalMarketAnalysis',
    'IntuitiveDecisionMaking',
    
    # Brain-Computer Interface
    'BrainComputerInterface',
    'NeuralSignal',
    'BrainwavePattern',
    'ThoughtPattern',
    'NeuralCommand',
    'initialize_bci',
    'get_bci_status',
    'shutdown_bci',
    
    # Meta-Cognition
    'MetaCognition',
    'SelfReflection',
    'ConsciousnessMonitor',
    'AwarenessLevels',
    
    # Phenomenal Consciousness
    'PhenomenalConsciousness',
    'Qualia',
    'ConsciousExperience',
    'SubjectiveStates'
]