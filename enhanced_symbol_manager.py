#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Symbol Manager - 本格運用向け100銘柄体制管理システム

実利益獲得に向けた銘柄の体系的管理
Tier構造による段階的銘柄管理・リスク分散・機会拡大
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

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

class SymbolTier(Enum):
    """銘柄ティア分類"""
    TIER1_CORE = "コア大型株"      # 15-20銘柄
    TIER2_GROWTH = "成長中型株"    # 30銘柄
    TIER3_OPPORTUNITY = "機会小型株" # 25銘柄
    TIER4_SPECIAL = "特殊銘柄"     # 25銘柄

class IndustryCategory(Enum):
    """業界カテゴリ"""
    TECH = "テクノロジー"
    FINANCE = "金融"
    HEALTHCARE = "医療・バイオ"
    MANUFACTURING = "製造業"
    RETAIL = "小売・消費"
    ENERGY = "エネルギー"
    REAL_ESTATE = "不動産"
    MATERIALS = "素材・化学"
    UTILITIES = "公益事業"
    TRANSPORT = "運輸・物流"
    TELECOM = "通信"
    FOOD = "食品・飲料"
    AUTOMOTIVE = "自動車"
    CONSTRUCTION = "建設"
    ENTERTAINMENT = "エンタメ"

class VolatilityLevel(Enum):
    """ボラティリティレベル"""
    VERY_LOW = "超低ボラ"      # <1%
    LOW = "低ボラ"            # 1-2%
    MEDIUM = "中ボラ"         # 2-4%
    HIGH = "高ボラ"           # 4-7%
    VERY_HIGH = "超高ボラ"    # >7%

@dataclass
class EnhancedStockInfo:
    """拡張銘柄情報"""
    symbol: str
    name: str
    tier: SymbolTier
    industry: IndustryCategory
    volatility_level: VolatilityLevel
    market_cap_billion: float  # 時価総額(億円)
    avg_volume_million: float  # 平均出来高(百万円/日)
    liquidity_score: float     # 流動性スコア(0-100)
    stability_score: float     # 安定性スコア(0-100)
    growth_potential: float    # 成長性スコア(0-100)
    dividend_yield: float      # 配当利回り(%)
    is_active: bool = True     # アクティブフラグ
    last_updated: datetime = field(default_factory=datetime.now)
    notes: str = ""           # 備考

    @property
    def yahoo_symbol(self) -> str:
        """Yahoo Finance用シンボル"""
        return f"{self.symbol}.T"

    @property
    def risk_score(self) -> float:
        """総合リスクスコア(0-100, 低いほど安全)"""
        volatility_risk = {
            VolatilityLevel.VERY_LOW: 10,
            VolatilityLevel.LOW: 25,
            VolatilityLevel.MEDIUM: 50,
            VolatilityLevel.HIGH: 75,
            VolatilityLevel.VERY_HIGH: 90
        }

        vol_risk = volatility_risk.get(self.volatility_level, 50)
        liquidity_risk = max(0, 100 - self.liquidity_score)
        stability_risk = max(0, 100 - self.stability_score)

        return (vol_risk * 0.4 + liquidity_risk * 0.3 + stability_risk * 0.3)

class EnhancedSymbolManager:
    """
    100銘柄本格運用向け銘柄管理システム
    Tier構造・業界分散・リスク管理統合
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データファイル管理
        self.data_dir = Path("enhanced_symbols_data")
        self.data_dir.mkdir(exist_ok=True)
        self.symbols_file = self.data_dir / "symbols_database.json"

        # 銘柄データベース
        self.symbols: Dict[str, EnhancedStockInfo] = {}

        # 初期化
        self._initialize_symbol_database()
        self._load_symbols_from_file()

    def _initialize_symbol_database(self):
        """銘柄データベース初期化"""

        # Tier1: コア大型株 (15銘柄)
        tier1_symbols = [
            EnhancedStockInfo("7203", "トヨタ自動車", SymbolTier.TIER1_CORE, IndustryCategory.AUTOMOTIVE,
                            VolatilityLevel.LOW, 37000, 800, 95, 90, 70, 2.4),
            EnhancedStockInfo("8306", "三菱UFJ", SymbolTier.TIER1_CORE, IndustryCategory.FINANCE,
                            VolatilityLevel.MEDIUM, 11000, 600, 90, 85, 60, 4.2),
            EnhancedStockInfo("9984", "ソフトバンクG", SymbolTier.TIER1_CORE, IndustryCategory.TECH,
                            VolatilityLevel.HIGH, 8000, 500, 85, 70, 80, 6.1),
            EnhancedStockInfo("6758", "ソニーG", SymbolTier.TIER1_CORE, IndustryCategory.ENTERTAINMENT,
                            VolatilityLevel.MEDIUM, 15000, 400, 88, 85, 85, 0.8),
            EnhancedStockInfo("7974", "任天堂", SymbolTier.TIER1_CORE, IndustryCategory.ENTERTAINMENT,
                            VolatilityLevel.HIGH, 8500, 350, 82, 88, 90, 2.5),
            EnhancedStockInfo("8316", "三井住友FG", SymbolTier.TIER1_CORE, IndustryCategory.FINANCE,
                            VolatilityLevel.MEDIUM, 6500, 400, 87, 87, 65, 4.8),
            EnhancedStockInfo("6861", "キーエンス", SymbolTier.TIER1_CORE, IndustryCategory.MANUFACTURING,
                            VolatilityLevel.MEDIUM, 12000, 300, 85, 92, 88, 1.2),
            EnhancedStockInfo("4503", "アステラス製薬", SymbolTier.TIER1_CORE, IndustryCategory.HEALTHCARE,
                            VolatilityLevel.LOW, 3200, 250, 80, 85, 75, 3.8),
            EnhancedStockInfo("9437", "NTTドコモ", SymbolTier.TIER1_CORE, IndustryCategory.TELECOM,
                            VolatilityLevel.LOW, 6800, 200, 85, 90, 60, 4.5),
            EnhancedStockInfo("8058", "三菱商事", SymbolTier.TIER1_CORE, IndustryCategory.MATERIALS,
                            VolatilityLevel.MEDIUM, 7500, 350, 88, 80, 70, 5.2),
            EnhancedStockInfo("6501", "日立製作所", SymbolTier.TIER1_CORE, IndustryCategory.MANUFACTURING,
                            VolatilityLevel.MEDIUM, 6200, 280, 85, 82, 78, 2.8),
            EnhancedStockInfo("8035", "東京エレクトロン", SymbolTier.TIER1_CORE, IndustryCategory.TECH,
                            VolatilityLevel.HIGH, 8800, 400, 82, 75, 85, 1.8),
            EnhancedStockInfo("6954", "ファナック", SymbolTier.TIER1_CORE, IndustryCategory.MANUFACTURING,
                            VolatilityLevel.MEDIUM, 4500, 200, 80, 85, 80, 2.1),
            EnhancedStockInfo("9983", "ファーストリテイリング", SymbolTier.TIER1_CORE, IndustryCategory.RETAIL,
                            VolatilityLevel.HIGH, 10000, 350, 85, 78, 85, 1.4),
            EnhancedStockInfo("2914", "日本たばこ", SymbolTier.TIER1_CORE, IndustryCategory.FOOD,
                            VolatilityLevel.LOW, 4800, 150, 85, 90, 50, 6.8)
        ]

        # Tier2: 成長中型株 (主要15銘柄、全30銘柄予定)
        tier2_symbols = [
            EnhancedStockInfo("4751", "サイバーエージェント", SymbolTier.TIER2_GROWTH, IndustryCategory.TECH,
                            VolatilityLevel.VERY_HIGH, 2800, 200, 75, 70, 95, 0.5),
            EnhancedStockInfo("4385", "メルカリ", SymbolTier.TIER2_GROWTH, IndustryCategory.TECH,
                            VolatilityLevel.VERY_HIGH, 800, 150, 70, 65, 90, 0.0),
            EnhancedStockInfo("6723", "ルネサス", SymbolTier.TIER2_GROWTH, IndustryCategory.TECH,
                            VolatilityLevel.HIGH, 3200, 300, 78, 72, 88, 1.2),
            EnhancedStockInfo("4478", "フリー", SymbolTier.TIER2_GROWTH, IndustryCategory.TECH,
                            VolatilityLevel.VERY_HIGH, 1200, 100, 65, 60, 92, 0.0),
            EnhancedStockInfo("6098", "リクルートHD", SymbolTier.TIER2_GROWTH, IndustryCategory.TECH,
                            VolatilityLevel.MEDIUM, 4500, 250, 82, 80, 85, 1.8),
            EnhancedStockInfo("4689", "LINEヤフー", SymbolTier.TIER2_GROWTH, IndustryCategory.TECH,
                            VolatilityLevel.HIGH, 1800, 150, 75, 70, 80, 0.0),
            EnhancedStockInfo("7735", "SCREENホールディングス", SymbolTier.TIER2_GROWTH, IndustryCategory.TECH,
                            VolatilityLevel.HIGH, 2200, 180, 75, 75, 85, 2.2),
            EnhancedStockInfo("6857", "アドバンテスト", SymbolTier.TIER2_GROWTH, IndustryCategory.TECH,
                            VolatilityLevel.HIGH, 1800, 120, 72, 78, 82, 1.5),
            EnhancedStockInfo("4565", "そーせいG", SymbolTier.TIER2_GROWTH, IndustryCategory.HEALTHCARE,
                            VolatilityLevel.VERY_HIGH, 1500, 80, 60, 50, 95, 0.0),
            EnhancedStockInfo("4587", "ペプチドリーム", SymbolTier.TIER2_GROWTH, IndustryCategory.HEALTHCARE,
                            VolatilityLevel.VERY_HIGH, 800, 60, 55, 55, 90, 0.0),
            EnhancedStockInfo("8411", "みずほFG", SymbolTier.TIER2_GROWTH, IndustryCategory.FINANCE,
                            VolatilityLevel.MEDIUM, 2800, 400, 85, 70, 65, 4.5),
            EnhancedStockInfo("1605", "INPEX", SymbolTier.TIER2_GROWTH, IndustryCategory.ENERGY,
                            VolatilityLevel.HIGH, 2200, 200, 78, 65, 70, 3.8),
            EnhancedStockInfo("8604", "野村HD", SymbolTier.TIER2_GROWTH, IndustryCategory.FINANCE,
                            VolatilityLevel.HIGH, 1800, 150, 75, 68, 75, 3.2),
            EnhancedStockInfo("3659", "ネクソン", SymbolTier.TIER2_GROWTH, IndustryCategory.ENTERTAINMENT,
                            VolatilityLevel.HIGH, 1600, 120, 70, 72, 85, 2.8),
            EnhancedStockInfo("2432", "DeNA", SymbolTier.TIER2_GROWTH, IndustryCategory.TECH,
                            VolatilityLevel.HIGH, 800, 80, 68, 70, 80, 1.5)
        ]

        # Tier3: 高ボラティリティ機会銘柄 (実証済み高成績15銘柄追加)
        tier3_high_vol_symbols = [
            EnhancedStockInfo("2370", "メディネット", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.HEALTHCARE,
                            VolatilityLevel.HIGH, 250, 50, 60, 55, 98, 0.0, notes="デイトレスコア97.8"),
            EnhancedStockInfo("4597", "ソレイジア・ファーマ", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.HEALTHCARE,
                            VolatilityLevel.HIGH, 180, 35, 70, 60, 96, 0.0, notes="デイトレスコア96.2"),
            EnhancedStockInfo("4563", "アンジェス", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.HEALTHCARE,
                            VolatilityLevel.HIGH, 400, 80, 70, 65, 94, 0.0, notes="デイトレスコア94.4"),
            EnhancedStockInfo("3664", "モブキャスト", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.TECH,
                            VolatilityLevel.VERY_HIGH, 120, 25, 65, 60, 94, 0.0, notes="デイトレスコア93.8"),
            EnhancedStockInfo("2158", "フロンティア", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.TECH,
                            VolatilityLevel.HIGH, 300, 45, 60, 65, 92, 0.0, notes="デイトレスコア91.8"),
            EnhancedStockInfo("6920", "レーザーテック", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.TECH,
                            VolatilityLevel.HIGH, 800, 120, 75, 80, 89, 1.5, notes="デイトレスコア88.9"),
            EnhancedStockInfo("3778", "さくらインターネット", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.TECH,
                            VolatilityLevel.HIGH, 600, 90, 65, 70, 83, 0.0, notes="デイトレスコア83.4"),
            EnhancedStockInfo("4592", "サンバイオ", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.HEALTHCARE,
                            VolatilityLevel.VERY_HIGH, 200, 40, 50, 55, 95, 0.0, notes="高ボラバイオ"),
            EnhancedStockInfo("4596", "窪田製薬HD", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.HEALTHCARE,
                            VolatilityLevel.HIGH, 150, 30, 55, 60, 91, 0.0, notes="デイトレスコア90.7"),
            EnhancedStockInfo("4579", "ラクオリア創薬", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.HEALTHCARE,
                            VolatilityLevel.VERY_HIGH, 100, 20, 50, 55, 88, 0.0, notes="創薬高ボラ"),
            EnhancedStockInfo("3687", "フィックスターズ", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.TECH,
                            VolatilityLevel.HIGH, 250, 40, 60, 75, 85, 0.0, notes="AI・HPC"),
            EnhancedStockInfo("4716", "オルトプラス", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.TECH,
                            VolatilityLevel.VERY_HIGH, 80, 15, 45, 50, 90, 0.0, notes="ゲーム高ボラ"),
            EnhancedStockInfo("3765", "ガンホー・オンライン・エンターテイメント", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.ENTERTAINMENT,
                            VolatilityLevel.HIGH, 400, 70, 65, 70, 85, 1.2, notes="ゲーム大手"),
            EnhancedStockInfo("4493", "サイバーセキュリティクラウド", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.TECH,
                            VolatilityLevel.HIGH, 180, 35, 55, 75, 88, 0.0, notes="セキュリティ"),
            EnhancedStockInfo("5020", "ENEOSホールディングス", SymbolTier.TIER3_OPPORTUNITY, IndustryCategory.ENERGY,
                            VolatilityLevel.MEDIUM, 3500, 600, 80, 75, 65, 5.8, notes="エネルギー大手")
        ]

        # Tier4: 業界分散強化銘柄 (新セクター15銘柄追加)
        tier4_diversification_symbols = [
            # 建設・不動産
            EnhancedStockInfo("1925", "大和ハウス工業", SymbolTier.TIER4_SPECIAL, IndustryCategory.CONSTRUCTION,
                            VolatilityLevel.MEDIUM, 2500, 200, 82, 80, 70, 3.8, notes="住宅・建設大手"),
            EnhancedStockInfo("8802", "三菱地所", SymbolTier.TIER4_SPECIAL, IndustryCategory.REAL_ESTATE,
                            VolatilityLevel.MEDIUM, 2800, 180, 85, 82, 65, 2.9, notes="不動産開発"),

            # 運輸・物流
            EnhancedStockInfo("9020", "JR東日本", SymbolTier.TIER4_SPECIAL, IndustryCategory.TRANSPORT,
                            VolatilityLevel.LOW, 3200, 250, 88, 85, 60, 2.1, notes="鉄道インフラ"),
            EnhancedStockInfo("9301", "三菱倉庫", SymbolTier.TIER4_SPECIAL, IndustryCategory.TRANSPORT,
                            VolatilityLevel.MEDIUM, 800, 120, 70, 75, 65, 2.8, notes="物流・倉庫"),

            # 公益事業
            EnhancedStockInfo("9501", "東京電力", SymbolTier.TIER4_SPECIAL, IndustryCategory.UTILITIES,
                            VolatilityLevel.MEDIUM, 1500, 300, 75, 70, 55, 0.0, notes="電力大手"),
            EnhancedStockInfo("9531", "東京ガス", SymbolTier.TIER4_SPECIAL, IndustryCategory.UTILITIES,
                            VolatilityLevel.LOW, 1200, 180, 80, 85, 60, 2.5, notes="都市ガス"),

            # 商社
            EnhancedStockInfo("8001", "伊藤忠商事", SymbolTier.TIER4_SPECIAL, IndustryCategory.MATERIALS,
                            VolatilityLevel.MEDIUM, 7200, 400, 90, 85, 75, 4.5, notes="総合商社"),
            EnhancedStockInfo("8031", "三井物産", SymbolTier.TIER4_SPECIAL, IndustryCategory.MATERIALS,
                            VolatilityLevel.MEDIUM, 5500, 350, 88, 82, 73, 5.2, notes="資源・エネルギー商社"),

            # 繊維・化学
            EnhancedStockInfo("4061", "デンカ", SymbolTier.TIER4_SPECIAL, IndustryCategory.MATERIALS,
                            VolatilityLevel.HIGH, 800, 100, 65, 70, 75, 1.8, notes="特殊化学"),
            EnhancedStockInfo("4188", "三菱ケミカル", SymbolTier.TIER4_SPECIAL, IndustryCategory.MATERIALS,
                            VolatilityLevel.MEDIUM, 1800, 200, 75, 75, 70, 3.2, notes="化学総合"),

            # 鉄鋼・金属
            EnhancedStockInfo("5401", "日本製鉄", SymbolTier.TIER4_SPECIAL, IndustryCategory.MATERIALS,
                            VolatilityLevel.HIGH, 3500, 400, 78, 70, 65, 2.8, notes="鉄鋼最大手"),
            EnhancedStockInfo("5406", "神戸製鋼所", SymbolTier.TIER4_SPECIAL, IndustryCategory.MATERIALS,
                            VolatilityLevel.HIGH, 600, 150, 60, 65, 70, 0.0, notes="特殊鋼・機械"),

            # 航空・海運
            EnhancedStockInfo("9201", "日本航空", SymbolTier.TIER4_SPECIAL, IndustryCategory.TRANSPORT,
                            VolatilityLevel.HIGH, 800, 200, 70, 65, 75, 0.0, notes="航空大手"),
            EnhancedStockInfo("9104", "商船三井", SymbolTier.TIER4_SPECIAL, IndustryCategory.TRANSPORT,
                            VolatilityLevel.HIGH, 1200, 300, 75, 70, 80, 2.5, notes="海運大手"),

            # 紙・パルプ
            EnhancedStockInfo("3861", "王子ホールディングス", SymbolTier.TIER4_SPECIAL, IndustryCategory.MATERIALS,
                            VolatilityLevel.MEDIUM, 800, 120, 70, 75, 60, 2.1, notes="製紙・パルプ")
        ]

        # 全銘柄を辞書に格納（60銘柄体制完成）
        all_symbols = tier1_symbols + tier2_symbols + tier3_high_vol_symbols + tier4_diversification_symbols

        for symbol_info in all_symbols:
            self.symbols[symbol_info.symbol] = symbol_info

        self.logger.info(f"Initialized symbol database with {len(self.symbols)} symbols")

    def get_symbols_by_tier(self, tier: SymbolTier) -> List[EnhancedStockInfo]:
        """ティア別銘柄取得"""
        return [symbol for symbol in self.symbols.values() if symbol.tier == tier and symbol.is_active]

    def get_symbols_by_industry(self, industry: IndustryCategory) -> List[EnhancedStockInfo]:
        """業界別銘柄取得"""
        return [symbol for symbol in self.symbols.values() if symbol.industry == industry and symbol.is_active]

    def get_symbols_by_volatility(self, volatility: VolatilityLevel) -> List[EnhancedStockInfo]:
        """ボラティリティ別銘柄取得"""
        return [symbol for symbol in self.symbols.values() if symbol.volatility_level == volatility and symbol.is_active]

    def get_top_symbols_by_criteria(self, criteria: str = "liquidity", limit: int = 10) -> List[EnhancedStockInfo]:
        """条件別TOP銘柄取得"""
        active_symbols = [s for s in self.symbols.values() if s.is_active]

        if criteria == "liquidity":
            return sorted(active_symbols, key=lambda x: x.liquidity_score, reverse=True)[:limit]
        elif criteria == "stability":
            return sorted(active_symbols, key=lambda x: x.stability_score, reverse=True)[:limit]
        elif criteria == "growth":
            return sorted(active_symbols, key=lambda x: x.growth_potential, reverse=True)[:limit]
        elif criteria == "low_risk":
            return sorted(active_symbols, key=lambda x: x.risk_score)[:limit]
        elif criteria == "high_volatility":
            return sorted(active_symbols, key=lambda x: x.volatility_level.name, reverse=True)[:limit]
        else:
            return active_symbols[:limit]

    def get_diversified_portfolio(self, count: int = 20) -> List[EnhancedStockInfo]:
        """業界分散ポートフォリオ生成"""
        portfolio = []
        industries_covered = set()

        # 各業界から1銘柄ずつ選択
        for symbol in self.symbols.values():
            if not symbol.is_active:
                continue

            if symbol.industry not in industries_covered and len(portfolio) < count:
                portfolio.append(symbol)
                industries_covered.add(symbol.industry)

        # 残り枠を流動性順で埋める
        remaining_symbols = [s for s in self.symbols.values()
                           if s.is_active and s not in portfolio]
        remaining_symbols.sort(key=lambda x: x.liquidity_score, reverse=True)

        portfolio.extend(remaining_symbols[:count - len(portfolio)])

        return portfolio[:count]

    def get_high_volatility_opportunities(self, count: int = 15) -> List[EnhancedStockInfo]:
        """高ボラティリティ機会銘柄取得"""
        high_vol_symbols = [
            s for s in self.symbols.values()
            if s.is_active and s.volatility_level in [VolatilityLevel.HIGH, VolatilityLevel.VERY_HIGH]
        ]

        # 成長性とボラティリティの複合スコア順
        high_vol_symbols.sort(key=lambda x: x.growth_potential +
                             (20 if x.volatility_level == VolatilityLevel.VERY_HIGH else 10),
                             reverse=True)

        return high_vol_symbols[:count]

    def get_daytrading_optimized_portfolio(self, count: int = 20) -> List[EnhancedStockInfo]:
        """デイトレード最適化ポートフォリオ"""

        # ティア3の高成績銘柄優先
        tier3_symbols = self.get_symbols_by_tier(SymbolTier.TIER3_OPPORTUNITY)

        # デイトレード適性スコア計算
        def daytrading_score(symbol: EnhancedStockInfo) -> float:
            vol_score = {
                VolatilityLevel.VERY_LOW: 10,
                VolatilityLevel.LOW: 30,
                VolatilityLevel.MEDIUM: 70,
                VolatilityLevel.HIGH: 90,
                VolatilityLevel.VERY_HIGH: 100
            }.get(symbol.volatility_level, 50)

            liquidity_bonus = min(20, symbol.liquidity_score / 5)
            growth_bonus = symbol.growth_potential / 5

            return vol_score + liquidity_bonus + growth_bonus

        # 全銘柄をデイトレードスコア順でソート
        all_active = [s for s in self.symbols.values() if s.is_active]
        all_active.sort(key=daytrading_score, reverse=True)

        return all_active[:count]

    def get_summary_statistics(self) -> Dict[str, any]:
        """銘柄統計サマリー"""
        active_symbols = [s for s in self.symbols.values() if s.is_active]

        tier_counts = {}
        for tier in SymbolTier:
            tier_counts[tier.value] = len([s for s in active_symbols if s.tier == tier])

        industry_counts = {}
        for industry in IndustryCategory:
            industry_counts[industry.value] = len([s for s in active_symbols if s.industry == industry])

        volatility_counts = {}
        for vol in VolatilityLevel:
            volatility_counts[vol.value] = len([s for s in active_symbols if s.volatility_level == vol])

        return {
            "total_symbols": len(active_symbols),
            "tier_distribution": tier_counts,
            "industry_distribution": industry_counts,
            "volatility_distribution": volatility_counts,
            "avg_market_cap": sum(s.market_cap_billion for s in active_symbols) / len(active_symbols),
            "avg_liquidity_score": sum(s.liquidity_score for s in active_symbols) / len(active_symbols),
            "avg_stability_score": sum(s.stability_score for s in active_symbols) / len(active_symbols)
        }

    def _save_symbols_to_file(self):
        """銘柄データをファイルに保存"""
        try:
            symbols_data = {}
            for symbol, info in self.symbols.items():
                symbols_data[symbol] = {
                    "symbol": info.symbol,
                    "name": info.name,
                    "tier": info.tier.value,
                    "industry": info.industry.value,
                    "volatility_level": info.volatility_level.value,
                    "market_cap_billion": info.market_cap_billion,
                    "avg_volume_million": info.avg_volume_million,
                    "liquidity_score": info.liquidity_score,
                    "stability_score": info.stability_score,
                    "growth_potential": info.growth_potential,
                    "dividend_yield": info.dividend_yield,
                    "is_active": info.is_active,
                    "last_updated": info.last_updated.isoformat(),
                    "notes": info.notes
                }

            with open(self.symbols_file, 'w', encoding='utf-8') as f:
                json.dump(symbols_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save symbols data: {e}")

    def _load_symbols_from_file(self):
        """ファイルから銘柄データを読み込み"""
        if not self.symbols_file.exists():
            return

        try:
            with open(self.symbols_file, 'r', encoding='utf-8') as f:
                symbols_data = json.load(f)

            # 既存データと比較して更新があれば反映
            for symbol, data in symbols_data.items():
                if symbol in self.symbols:
                    # アクティブフラグなどの管理情報のみ更新
                    self.symbols[symbol].is_active = data.get("is_active", True)
                    self.symbols[symbol].notes = data.get("notes", "")

        except Exception as e:
            self.logger.error(f"Failed to load symbols data: {e}")

# テスト関数
def test_enhanced_symbol_manager():
    """拡張銘柄管理システムのテスト"""
    print("=== 拡張銘柄管理システム テスト ===")

    manager = EnhancedSymbolManager()

    # 統計情報
    print("\n[ 銘柄統計サマリー ]")
    stats = manager.get_summary_statistics()
    print(f"総銘柄数: {stats['total_symbols']}")
    print(f"平均時価総額: {stats['avg_market_cap']:.0f}億円")
    print(f"平均流動性スコア: {stats['avg_liquidity_score']:.1f}")

    # Tier別銘柄数
    print("\n[ Tier別銘柄分布 ]")
    for tier, count in stats['tier_distribution'].items():
        if count > 0:
            print(f"{tier}: {count}銘柄")

    # 業界分散確認
    print("\n[ 業界分散状況 ]")
    for industry, count in stats['industry_distribution'].items():
        if count > 0:
            print(f"{industry}: {count}銘柄")

    # TOP10銘柄（流動性順）
    print("\n[ TOP10銘柄 (流動性順) ]")
    top10 = manager.get_top_symbols_by_criteria("liquidity", 10)
    for i, symbol in enumerate(top10, 1):
        print(f"{i:2d}. {symbol.symbol} ({symbol.name}) - 流動性:{symbol.liquidity_score}")

    # 分散ポートフォリオ
    print("\n[ 業界分散ポートフォリオ (20銘柄) ]")
    portfolio = manager.get_diversified_portfolio(20)
    for symbol in portfolio[:10]:  # 最初の10銘柄のみ表示
        print(f"{symbol.symbol} ({symbol.name}) - {symbol.industry.value}")

    print(f"\n=== テスト完了: {len(manager.symbols)}銘柄システム稼働中 ===")

if __name__ == "__main__":
    test_enhanced_symbol_manager()