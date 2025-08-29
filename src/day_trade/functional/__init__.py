#!/usr/bin/env python3
"""
Functional Programming System
関数型プログラミング統合システム
"""

from .monads import Maybe, Either, IO, Reader, Writer, State
from .immutable import ImmutableList, ImmutableDict, ImmutableSet, FrozenRecord
from .functions import curry, compose, pipe, partial_application
from .streams import LazyStream, InfiniteStream, FilteredStream
from .combinators import Parser, Applicative, Alternative
from .category_theory import Functor, Monad, Foldable, Traversable

__all__ = [
    # Monads
    'Maybe', 'Either', 'IO', 'Reader', 'Writer', 'State',
    
    # Immutable Data Structures
    'ImmutableList', 'ImmutableDict', 'ImmutableSet', 'FrozenRecord',
    
    # Higher-Order Functions
    'curry', 'compose', 'pipe', 'partial_application',
    
    # Lazy Streams
    'LazyStream', 'InfiniteStream', 'FilteredStream',
    
    # Parser Combinators
    'Parser', 'Applicative', 'Alternative',
    
    # Category Theory
    'Functor', 'Monad', 'Foldable', 'Traversable'
]