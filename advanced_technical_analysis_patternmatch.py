#!/usr/bin/env python3
"""
advanced_technical_analysis.py - PatternMatch

リファクタリングにより分割されたモジュール
"""

class PatternMatch:
    """パターンマッチ結果"""
    pattern_name: str
    match_score: float
    start_index: int
    end_index: int
    pattern_type: str  # "continuation", "reversal"
    reliability: float
    target_price: Optional[float] = None
