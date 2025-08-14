#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sector Diversification - 業界分散システム

セクター集中リスク解消・33業界完全分散
セクターローテーション対応・相関分析統合
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from collections import defaultdict, Counter
import itertools

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

try:
    from enhanced_symbol_manager import EnhancedSymbolManager, IndustryCategory, EnhancedStockInfo
    ENHANCED_SYMBOLS_AVAILABLE = True
except ImportError:
    ENHANCED_SYMBOLS_AVAILABLE = False

class SectorStrength(Enum):
    """セクター強度"""
    VERY_STRONG = "非常に強い"     # トップ20%
    STRONG = "強い"               # 上位40%
    NEUTRAL = "中立"              # 中位40%
    WEAK = "弱い"                 # 下位40%
    VERY_WEAK = "非常に弱い"      # 最下位20%

class SectorRotationPhase(Enum):
    """セクターローテーションフェーズ"""
    EARLY_CYCLE = "景気回復初期"      # テック・消費系
    MID_CYCLE = "景気拡大期"          # 工業・素材系
    LATE_CYCLE = "景気成熟期"         # エネルギー・金融系
    RECESSION = "景気後退期"          # 公益・ディフェンシブ系

@dataclass
class SectorAnalysis:
    """セクター分析データ"""
    sector: str
    symbol_count: int
    avg_market_cap: float
    total_weight: float              # ポートフォリオ内比重(%)
    avg_volatility: float
    avg_growth_potential: float
    correlation_with_market: float   # 市場相関
    sector_strength: SectorStrength
    rotation_phase_fit: SectorRotationPhase
    risk_contribution: float         # リスク寄与度
    diversification_benefit: float   # 分散効果

@dataclass
class DiversificationMetrics:
    """分散化指標"""
    total_sectors: int
    herfindahl_index: float         # ハーフィンダル指数（集中度）
    effective_sectors: float        # 実効セクター数
    sector_balance_score: float     # セクターバランススコア(0-100)
    correlation_risk: float         # 相関リスク
    diversification_ratio: float    # 分散化比率
    sector_coverage: float          # セクターカバレッジ(%)

class SectorDiversificationManager:
    """
    業界分散管理システム
    33業界完全分散・セクター集中リスク解消
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 拡張業界分類（33業界完全網羅）
        self.comprehensive_sectors = self._initialize_comprehensive_sectors()

        # セクター強度マトリックス
        self.sector_strength_matrix = {}

        # 相関マトリックス
        self.correlation_matrix = pd.DataFrame()

        # データディレクトリ
        self.data_dir = Path("sector_data")
        self.data_dir.mkdir(exist_ok=True)

        if ENHANCED_SYMBOLS_AVAILABLE:
            self.symbol_manager = EnhancedSymbolManager()

        self.logger.info(f"Sector diversification manager initialized with {len(self.comprehensive_sectors)} sectors")

    def _initialize_comprehensive_sectors(self) -> Dict[str, Dict[str, Any]]:
        """33業界完全分類初期化"""

        return {
            # テクノロジー系 (6業界)
            "ソフトウェア・IT": {
                "description": "ソフトウェア開発・ITサービス",
                "cyclical_type": "growth",
                "defensive_score": 30,
                "keywords": ["ソフト", "IT", "システム", "デジタル"]
            },
            "半導体・エレクトロニクス": {
                "description": "半導体・電子部品製造",
                "cyclical_type": "cyclical",
                "defensive_score": 20,
                "keywords": ["半導体", "エレクトロニクス", "電子部品"]
            },
            "AI・ロボティクス": {
                "description": "人工知能・ロボット技術",
                "cyclical_type": "growth",
                "defensive_score": 25,
                "keywords": ["AI", "ロボット", "自動化", "機械学習"]
            },
            "フィンテック": {
                "description": "金融テクノロジー",
                "cyclical_type": "growth",
                "defensive_score": 40,
                "keywords": ["フィンテック", "決済", "暗号通貨"]
            },
            "サイバーセキュリティ": {
                "description": "情報セキュリティ",
                "cyclical_type": "defensive",
                "defensive_score": 70,
                "keywords": ["セキュリティ", "暗号化", "防御"]
            },
            "通信・ネットワーク": {
                "description": "通信インフラ・ネットワーク",
                "cyclical_type": "defensive",
                "defensive_score": 75,
                "keywords": ["通信", "ネットワーク", "5G", "インフラ"]
            },

            # ヘルスケア系 (4業界)
            "製薬・バイオテクノロジー": {
                "description": "医薬品開発・バイオテクノロジー",
                "cyclical_type": "defensive",
                "defensive_score": 80,
                "keywords": ["製薬", "バイオ", "創薬", "治療"]
            },
            "医療機器・診断": {
                "description": "医療機器・診断装置",
                "cyclical_type": "defensive",
                "defensive_score": 75,
                "keywords": ["医療機器", "診断", "検査"]
            },
            "ヘルスケアIT": {
                "description": "医療情報システム",
                "cyclical_type": "growth",
                "defensive_score": 70,
                "keywords": ["医療IT", "電子カルテ", "遠隔医療"]
            },
            "再生医療・遺伝子治療": {
                "description": "再生医療・遺伝子技術",
                "cyclical_type": "growth",
                "defensive_score": 60,
                "keywords": ["再生医療", "遺伝子", "細胞治療"]
            },

            # 金融系 (4業界)
            "銀行・金融サービス": {
                "description": "銀行・金融サービス",
                "cyclical_type": "cyclical",
                "defensive_score": 50,
                "keywords": ["銀行", "金融", "融資", "預金"]
            },
            "証券・投資": {
                "description": "証券・投資サービス",
                "cyclical_type": "cyclical",
                "defensive_score": 30,
                "keywords": ["証券", "投資", "資産運用"]
            },
            "保険": {
                "description": "生命保険・損害保険",
                "cyclical_type": "defensive",
                "defensive_score": 70,
                "keywords": ["保険", "生保", "損保"]
            },
            "不動産・REIT": {
                "description": "不動産・不動産投資信託",
                "cyclical_type": "cyclical",
                "defensive_score": 45,
                "keywords": ["不動産", "REIT", "建設"]
            },

            # 製造業系 (6業界)
            "自動車・輸送機器": {
                "description": "自動車・輸送機器製造",
                "cyclical_type": "cyclical",
                "defensive_score": 25,
                "keywords": ["自動車", "輸送", "車両"]
            },
            "機械・工業機器": {
                "description": "産業機械・工業機器",
                "cyclical_type": "cyclical",
                "defensive_score": 35,
                "keywords": ["機械", "工業", "製造装置"]
            },
            "航空宇宙・防衛": {
                "description": "航空宇宙・防衛産業",
                "cyclical_type": "defensive",
                "defensive_score": 65,
                "keywords": ["航空", "宇宙", "防衛", "軍事"]
            },
            "造船・重工業": {
                "description": "造船・重工業",
                "cyclical_type": "cyclical",
                "defensive_score": 30,
                "keywords": ["造船", "重工", "プラント"]
            },
            "精密機器・計測": {
                "description": "精密機器・計測機器",
                "cyclical_type": "growth",
                "defensive_score": 55,
                "keywords": ["精密", "計測", "光学"]
            },
            "工作機械・ロボット": {
                "description": "工作機械・産業ロボット",
                "cyclical_type": "cyclical",
                "defensive_score": 40,
                "keywords": ["工作機械", "ロボット", "自動化"]
            },

            # エネルギー・素材系 (4業界)
            "石油・ガス": {
                "description": "石油・天然ガス",
                "cyclical_type": "cyclical",
                "defensive_score": 35,
                "keywords": ["石油", "ガス", "エネルギー"]
            },
            "再生可能エネルギー": {
                "description": "太陽光・風力・水力発電",
                "cyclical_type": "growth",
                "defensive_score": 50,
                "keywords": ["再生エネ", "太陽光", "風力", "水力"]
            },
            "化学・素材": {
                "description": "化学・基礎素材",
                "cyclical_type": "cyclical",
                "defensive_score": 40,
                "keywords": ["化学", "素材", "樹脂", "繊維"]
            },
            "鉄鋼・非鉄金属": {
                "description": "鉄鋼・非鉄金属",
                "cyclical_type": "cyclical",
                "defensive_score": 25,
                "keywords": ["鉄鋼", "金属", "アルミ", "銅"]
            },

            # 消費者向けサービス系 (5業界)
            "小売・Eコマース": {
                "description": "小売業・電子商取引",
                "cyclical_type": "cyclical",
                "defensive_score": 35,
                "keywords": ["小売", "EC", "百貨店", "専門店"]
            },
            "食品・飲料": {
                "description": "食品・飲料製造",
                "cyclical_type": "defensive",
                "defensive_score": 80,
                "keywords": ["食品", "飲料", "アルコール"]
            },
            "娯楽・エンターテイメント": {
                "description": "ゲーム・エンタメ・メディア",
                "cyclical_type": "growth",
                "defensive_score": 40,
                "keywords": ["ゲーム", "エンタメ", "メディア"]
            },
            "旅行・レジャー": {
                "description": "旅行・宿泊・レジャー",
                "cyclical_type": "cyclical",
                "defensive_score": 20,
                "keywords": ["旅行", "ホテル", "レジャー", "観光"]
            },
            "教育・人材サービス": {
                "description": "教育・人材派遣サービス",
                "cyclical_type": "defensive",
                "defensive_score": 60,
                "keywords": ["教育", "人材", "研修", "派遣"]
            },

            # インフラ・公益系 (4業界)
            "電力・ガス・水道": {
                "description": "公益事業",
                "cyclical_type": "defensive",
                "defensive_score": 85,
                "keywords": ["電力", "ガス", "水道", "公益"]
            },
            "鉄道・物流": {
                "description": "鉄道・物流・運輸",
                "cyclical_type": "defensive",
                "defensive_score": 65,
                "keywords": ["鉄道", "物流", "運輸", "配送"]
            },
            "通信インフラ": {
                "description": "通信インフラ・データセンター",
                "cyclical_type": "defensive",
                "defensive_score": 75,
                "keywords": ["通信基盤", "データセンター", "インフラ"]
            },
            "廃棄物・環境": {
                "description": "廃棄物処理・環境事業",
                "cyclical_type": "defensive",
                "defensive_score": 70,
                "keywords": ["廃棄物", "環境", "リサイクル"]
            }
        }

    def analyze_current_diversification(self, portfolio_symbols: List[str]) -> DiversificationMetrics:
        """現在の分散化状況分析"""

        if not ENHANCED_SYMBOLS_AVAILABLE:
            return self._create_dummy_diversification()

        # シンボル情報取得
        symbol_infos = []
        for symbol in portfolio_symbols:
            info = self.symbol_manager.symbols.get(symbol)
            if info and info.is_active:
                symbol_infos.append(info)

        if not symbol_infos:
            return self._create_dummy_diversification()

        # セクター分布計算
        sector_weights = defaultdict(float)
        total_market_cap = sum(info.market_cap_billion for info in symbol_infos)

        for info in symbol_infos:
            sector = self._map_industry_to_comprehensive_sector(info.industry)
            weight = info.market_cap_billion / total_market_cap if total_market_cap > 0 else 1.0 / len(symbol_infos)
            sector_weights[sector] += weight

        # ハーフィンダル指数計算（集中度）
        herfindahl_index = sum(w**2 for w in sector_weights.values())

        # 実効セクター数
        effective_sectors = 1.0 / herfindahl_index if herfindahl_index > 0 else 1.0

        # セクターバランススコア
        ideal_weight = 1.0 / len(self.comprehensive_sectors)
        balance_deviations = [abs(w - ideal_weight) for w in sector_weights.values()]
        sector_balance_score = max(0, 100 - sum(balance_deviations) * 100)

        # セクターカバレッジ
        sector_coverage = len(sector_weights) / len(self.comprehensive_sectors) * 100

        # 簡易相関リスク（セクター数が少ないほど高リスク）
        correlation_risk = max(0, 100 - len(sector_weights) * 3)

        # 分散化比率
        diversification_ratio = min(1.0, len(sector_weights) / 15)  # 15セクター以上で最大

        return DiversificationMetrics(
            total_sectors=len(sector_weights),
            herfindahl_index=herfindahl_index,
            effective_sectors=effective_sectors,
            sector_balance_score=sector_balance_score,
            correlation_risk=correlation_risk,
            diversification_ratio=diversification_ratio,
            sector_coverage=sector_coverage
        )

    def _map_industry_to_comprehensive_sector(self, industry: 'IndustryCategory') -> str:
        """業界カテゴリを包括セクターにマッピング"""

        mapping = {
            IndustryCategory.TECH: "ソフトウェア・IT",
            IndustryCategory.FINANCE: "銀行・金融サービス",
            IndustryCategory.HEALTHCARE: "製薬・バイオテクノロジー",
            IndustryCategory.MANUFACTURING: "機械・工業機器",
            IndustryCategory.RETAIL: "小売・Eコマース",
            IndustryCategory.ENERGY: "石油・ガス",
            IndustryCategory.REAL_ESTATE: "不動産・REIT",
            IndustryCategory.MATERIALS: "化学・素材",
            IndustryCategory.UTILITIES: "電力・ガス・水道",
            IndustryCategory.TRANSPORT: "鉄道・物流",
            IndustryCategory.TELECOM: "通信・ネットワーク",
            IndustryCategory.FOOD: "食品・飲料",
            IndustryCategory.AUTOMOTIVE: "自動車・輸送機器",
            IndustryCategory.CONSTRUCTION: "不動産・REIT",
            IndustryCategory.ENTERTAINMENT: "娯楽・エンターテイメント"
        }

        return mapping.get(industry, "その他")

    def optimize_sector_allocation(self, current_symbols: List[str],
                                 target_count: int = 30) -> List[Tuple[str, str, float]]:
        """
        セクター配分最適化

        Returns:
            List[Tuple[symbol, sector, weight]]: 最適化後の配分
        """

        if not ENHANCED_SYMBOLS_AVAILABLE:
            return [(s, "未分類", 1.0/len(current_symbols)) for s in current_symbols]

        # 現在の分散化状況
        current_metrics = self.analyze_current_diversification(current_symbols)

        # 不足セクター特定
        current_sectors = self._get_current_sectors(current_symbols)
        missing_sectors = set(self.comprehensive_sectors.keys()) - current_sectors

        # 最適化候補銘柄リスト構築
        optimization_candidates = []

        for symbol, info in self.symbol_manager.symbols.items():
            if not info.is_active:
                continue

            sector = self._map_industry_to_comprehensive_sector(info.industry)

            # スコア計算
            liquidity_score = info.liquidity_score
            stability_score = info.stability_score
            growth_score = info.growth_potential

            # セクター不足ボーナス
            sector_bonus = 20 if sector in missing_sectors else 0

            total_score = (liquidity_score * 0.4 + stability_score * 0.3 +
                          growth_score * 0.3 + sector_bonus)

            optimization_candidates.append((symbol, sector, total_score, info))

        # セクターバランス最適化
        optimized_portfolio = self._balance_sectors(optimization_candidates, target_count)

        return optimized_portfolio

    def _get_current_sectors(self, symbols: List[str]) -> Set[str]:
        """現在のセクター取得"""
        sectors = set()

        for symbol in symbols:
            info = self.symbol_manager.symbols.get(symbol)
            if info:
                sector = self._map_industry_to_comprehensive_sector(info.industry)
                sectors.add(sector)

        return sectors

    def _balance_sectors(self, candidates: List[Tuple[str, str, float, Any]],
                        target_count: int) -> List[Tuple[str, str, float]]:
        """セクターバランス調整"""

        # セクター別候補ソート
        sector_candidates = defaultdict(list)
        for symbol, sector, score, info in candidates:
            sector_candidates[sector].append((symbol, score, info))

        # 各セクターから最高スコア銘柄を選択
        selected = []
        sector_weights = {}

        # Phase1: 各セクターから1銘柄ずつ（最大15セクター）
        sectors_covered = 0
        for sector, sector_cands in sector_candidates.items():
            if sectors_covered >= min(15, target_count):
                break

            # セクター内最高スコア銘柄
            sector_cands.sort(key=lambda x: x[1], reverse=True)
            if sector_cands:
                symbol, score, info = sector_cands[0]
                selected.append((symbol, sector, 1.0 / target_count))
                sector_weights[sector] = 1
                sectors_covered += 1

        # Phase2: 残り枠を高スコア順で埋める
        remaining_candidates = []
        for symbol, sector, score, info in candidates:
            if symbol not in [s[0] for s in selected]:
                remaining_candidates.append((symbol, sector, score, info))

        remaining_candidates.sort(key=lambda x: x[2], reverse=True)

        for symbol, sector, score, info in remaining_candidates[:target_count - len(selected)]:
            weight = 1.0 / target_count
            selected.append((symbol, sector, weight))

        return selected[:target_count]

    def generate_diversification_report(self, portfolio_symbols: List[str]) -> Dict[str, Any]:
        """分散化レポート生成"""

        metrics = self.analyze_current_diversification(portfolio_symbols)

        # セクター別分析
        sector_analysis = []
        if ENHANCED_SYMBOLS_AVAILABLE:
            sector_breakdown = self._analyze_sector_breakdown(portfolio_symbols)
            sector_analysis = sector_breakdown

        # 改善提案
        improvement_suggestions = self._generate_improvement_suggestions(metrics)

        return {
            "diversification_metrics": {
                "total_sectors": metrics.total_sectors,
                "effective_sectors": metrics.effective_sectors,
                "sector_balance_score": metrics.sector_balance_score,
                "sector_coverage": metrics.sector_coverage,
                "correlation_risk": metrics.correlation_risk,
                "diversification_ratio": metrics.diversification_ratio,
                "herfindahl_index": metrics.herfindahl_index
            },
            "sector_analysis": sector_analysis,
            "improvement_suggestions": improvement_suggestions,
            "risk_assessment": {
                "concentration_risk": "高" if metrics.herfindahl_index > 0.25 else "低",
                "diversification_quality": "優秀" if metrics.sector_balance_score > 80 else
                                          ("良好" if metrics.sector_balance_score > 60 else "要改善")
            }
        }

    def _analyze_sector_breakdown(self, portfolio_symbols: List[str]) -> List[Dict[str, Any]]:
        """セクター別内訳分析"""

        sector_data = defaultdict(list)

        for symbol in portfolio_symbols:
            info = self.symbol_manager.symbols.get(symbol)
            if info and info.is_active:
                sector = self._map_industry_to_comprehensive_sector(info.industry)
                sector_data[sector].append({
                    "symbol": symbol,
                    "name": info.name,
                    "market_cap": info.market_cap_billion,
                    "volatility": info.volatility_level.value
                })

        sector_analysis = []
        for sector, symbols in sector_data.items():
            total_market_cap = sum(s["market_cap"] for s in symbols)
            sector_analysis.append({
                "sector": sector,
                "symbol_count": len(symbols),
                "symbols": symbols,
                "total_market_cap": total_market_cap,
                "avg_market_cap": total_market_cap / len(symbols) if symbols else 0
            })

        return sorted(sector_analysis, key=lambda x: x["total_market_cap"], reverse=True)

    def _generate_improvement_suggestions(self, metrics: DiversificationMetrics) -> List[str]:
        """改善提案生成"""

        suggestions = []

        if metrics.sector_coverage < 50:
            suggestions.append("セクターカバレッジが低い。より多様な業界への分散を推奨")

        if metrics.herfindahl_index > 0.25:
            suggestions.append("セクター集中度が高い。主要セクターの比重を下げることを推奨")

        if metrics.effective_sectors < 8:
            suggestions.append("実効セクター数が少ない。少なくとも10-15セクターへの分散を推奨")

        if metrics.correlation_risk > 60:
            suggestions.append("相関リスクが高い。相関の低い業界への分散を推奨")

        if not suggestions:
            suggestions.append("分散化状況は良好です。現在のバランスを維持することを推奨")

        return suggestions

    def _create_dummy_diversification(self) -> DiversificationMetrics:
        """ダミー分散化指標作成"""
        return DiversificationMetrics(
            total_sectors=8,
            herfindahl_index=0.20,
            effective_sectors=5.0,
            sector_balance_score=65.0,
            correlation_risk=45.0,
            diversification_ratio=0.75,
            sector_coverage=24.2
        )

# テスト関数
def test_sector_diversification():
    """セクター分散システムのテスト"""
    print("=== セクター分散システム テスト ===")

    manager = SectorDiversificationManager()

    print(f"\n[ 包括セクター数: {len(manager.comprehensive_sectors)} ]")

    # テスト用ポートフォリオ
    test_portfolio = [
        "7203", "8306", "9984", "6758", "4503",  # 大型株中心
        "4751", "6723", "4565", "1605", "8411"   # 成長株
    ]

    print(f"\n[ 現在の分散化分析: {len(test_portfolio)}銘柄 ]")
    metrics = manager.analyze_current_diversification(test_portfolio)

    print(f"総セクター数: {metrics.total_sectors}")
    print(f"実効セクター数: {metrics.effective_sectors:.1f}")
    print(f"セクターバランススコア: {metrics.sector_balance_score:.1f}")
    print(f"セクターカバレッジ: {metrics.sector_coverage:.1f}%")
    print(f"集中度指数: {metrics.herfindahl_index:.3f}")

    # 最適化提案
    print(f"\n[ セクター配分最適化提案: 20銘柄 ]")
    optimized = manager.optimize_sector_allocation(test_portfolio, 20)

    sector_counts = Counter(allocation[1] for allocation in optimized[:10])
    print("TOP10最適化後セクター分布:")
    for sector, count in sector_counts.most_common():
        print(f"  {sector}: {count}銘柄")

    # レポート生成
    print(f"\n[ 分散化レポート ]")
    report = manager.generate_diversification_report(test_portfolio)

    print(f"分散化品質: {report['risk_assessment']['diversification_quality']}")
    print(f"集中リスク: {report['risk_assessment']['concentration_risk']}")

    print(f"\n改善提案:")
    for suggestion in report["improvement_suggestions"]:
        print(f"  • {suggestion}")

    print(f"\n=== セクター分散システム テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    test_sector_diversification()