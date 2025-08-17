#!/usr/bin/env python3
"""
銘柄範囲拡張システム
Issue #881: 自動更新の更新時間を考える - 銘柄範囲拡張

動的に監視銘柄を拡張し、市場機会を最大化
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


class ExpansionStrategy(Enum):
    """拡張戦略"""
    SECTOR_ROTATION = "sector_rotation"      # セクターローテーション
    VOLATILITY_BASED = "volatility_based"    # ボラティリティベース
    CORRELATION_BASED = "correlation_based"  # 相関ベース
    MOMENTUM_BASED = "momentum_based"        # モメンタムベース
    NEWS_DRIVEN = "news_driven"              # ニュース主導


class SymbolStatus(Enum):
    """銘柄状態"""
    ACTIVE = "active"                # アクティブ監視
    CANDIDATE = "candidate"          # 候補
    WATCHLIST = "watchlist"          # ウォッチリスト
    SUSPENDED = "suspended"          # 一時停止
    EXCLUDED = "excluded"            # 除外


@dataclass
class SymbolCandidate:
    """銘柄候補"""
    code: str
    name: str
    sector: str
    market_cap: Optional[float] = None
    avg_volume: Optional[float] = None
    volatility_score: float = 0.0
    momentum_score: float = 0.0
    correlation_score: float = 0.0
    opportunity_score: float = 0.0
    risk_score: float = 0.0
    priority_score: float = 0.0
    last_analysis: Optional[datetime] = None
    status: SymbolStatus = SymbolStatus.CANDIDATE


@dataclass
class ExpansionMetrics:
    """拡張メトリクス"""
    symbols_added: int = 0
    symbols_removed: int = 0
    opportunities_found: int = 0
    false_positives: int = 0
    accuracy_rate: float = 0.0
    coverage_improvement: float = 0.0


class SymbolExpansionSystem:
    """銘柄範囲拡張システム"""
    
    def __init__(self, config_path: str = "config/settings.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.current_symbols: Set[str] = set()
        self.candidates: Dict[str, SymbolCandidate] = {}
        self.metrics = ExpansionMetrics()
        self.expansion_active = True
        
        # 拡張可能銘柄プール（日本市場主要銘柄）
        self.symbol_pool = self._initialize_symbol_pool()
        
        # 拡張設定
        self.max_symbols = 200  # 最大監視銘柄数
        self.min_opportunity_score = 0.6  # 最小機会スコア
        self.expansion_batch_size = 5  # バッチサイズ
        
        self.setup_logging()
    
    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/symbol_expansion.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> dict:
        """設定ファイル読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"設定ファイルが見つかりません: {self.config_path}")
            return {}
    
    def _initialize_symbol_pool(self) -> Dict[str, dict]:
        """銘柄プール初期化"""
        # 日本市場主要銘柄データ（TOPIX500ベース）
        pool = {
            # 追加Technology銘柄
            "4768": {"name": "大塚商会", "sector": "Technology", "market_cap": "large"},
            "4816": {"name": "東映アニメーション", "sector": "Entertainment", "market_cap": "medium"},
            "9449": {"name": "GMOインターネット", "sector": "Technology", "market_cap": "medium"},
            "3782": {"name": "DDS", "sector": "Technology", "market_cap": "small"},
            "3681": {"name": "ブイキューブ", "sector": "Technology", "market_cap": "small"},
            
            # 追加Financial銘柄
            "8601": {"name": "大和証券", "sector": "Financial", "market_cap": "large"},
            "8572": {"name": "アコム", "sector": "Financial", "market_cap": "medium"},
            "8698": {"name": "マネックス", "sector": "Financial", "market_cap": "small"},
            
            # 追加Healthcare銘柄
            "4568": {"name": "第一三共", "sector": "Healthcare", "market_cap": "large"},
            "4507": {"name": "塩野義製薬", "sector": "Healthcare", "market_cap": "large"},
            "4519": {"name": "中外製薬", "sector": "Healthcare", "market_cap": "large"},
            "4578": {"name": "大塚HD", "sector": "Healthcare", "market_cap": "large"},
            
            # 追加Consumer銘柄
            "2269": {"name": "明治HD", "sector": "Consumer", "market_cap": "large"},
            "2282": {"name": "日本ハム", "sector": "Consumer", "market_cap": "large"},
            "8167": {"name": "リコーリース", "sector": "Consumer", "market_cap": "medium"},
            
            # 追加Industrial銘柄
            "6762": {"name": "TDK", "sector": "Industrial", "market_cap": "large"},
            "6770": {"name": "アルプスアルパイン", "sector": "Industrial", "market_cap": "medium"},
            "6902": {"name": "デンソー", "sector": "Industrial", "market_cap": "large"},
            
            # 新興市場銘柄
            "4477": {"name": "BASE", "sector": "DayTrading", "market_cap": "medium"},
            "4488": {"name": "AI inside", "sector": "DayTrading", "market_cap": "small"},
            "4490": {"name": "ビザスク", "sector": "DayTrading", "market_cap": "small"},
            "4425": {"name": "Kudan", "sector": "DayTrading", "market_cap": "small"},
            "4431": {"name": "スマレジ", "sector": "DayTrading", "market_cap": "small"},
            
            # バイオテック拡張
            "4889": {"name": "カルナバイオサイエンス", "sector": "BioTech", "market_cap": "small"},
            "4880": {"name": "セルソース", "sector": "BioTech", "market_cap": "small"},
            "4998": {"name": "フマキラー", "sector": "BioTech", "market_cap": "medium"},
            
            # エネルギー・素材
            "5108": {"name": "ブリヂストン", "sector": "Materials", "market_cap": "large"},
            "5214": {"name": "日本電気硝子", "sector": "Materials", "market_cap": "medium"},
            "5333": {"name": "日本ガイシ", "sector": "Materials", "market_cap": "medium"},
            
            # ESG関連
            "1605": {"name": "INPEX", "sector": "Energy", "market_cap": "large"},
            "9531": {"name": "東京ガス", "sector": "Utilities", "market_cap": "large"},
            "9532": {"name": "大阪ガス", "sector": "Utilities", "market_cap": "large"},
        }
        
        self.logger.info(f"銘柄プール初期化完了: {len(pool)}銘柄")
        return pool
    
    def initialize_current_symbols(self) -> None:
        """現在の監視銘柄初期化"""
        symbols = self.config.get('watchlist', {}).get('symbols', [])
        self.current_symbols = {symbol['code'] for symbol in symbols}
        self.logger.info(f"現在の監視銘柄: {len(self.current_symbols)}銘柄")
    
    def analyze_expansion_opportunities(self) -> List[SymbolCandidate]:
        """拡張機会分析"""
        self.logger.info("拡張機会分析開始")
        opportunities = []
        
        progress_bar = tqdm(
            total=len(self.symbol_pool),
            desc="機会分析",
            unit="銘柄"
        )
        
        for code, data in self.symbol_pool.items():
            if code in self.current_symbols:
                progress_bar.update(1)
                continue
                
            candidate = self._analyze_symbol_candidate(code, data)
            if candidate.opportunity_score >= self.min_opportunity_score:
                opportunities.append(candidate)
                self.candidates[code] = candidate
                
            progress_bar.update(1)
        
        progress_bar.close()
        
        # 機会スコア順でソート
        opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)
        
        self.logger.info(f"拡張機会発見: {len(opportunities)}銘柄")
        return opportunities
    
    def _analyze_symbol_candidate(self, code: str, data: dict) -> SymbolCandidate:
        """銘柄候補分析"""
        candidate = SymbolCandidate(
            code=code,
            name=data['name'],
            sector=data['sector']
        )
        
        # 各種スコア計算
        candidate.volatility_score = self._calculate_volatility_score(code, data)
        candidate.momentum_score = self._calculate_momentum_score(code, data)
        candidate.correlation_score = self._calculate_correlation_score(code, data)
        candidate.risk_score = self._calculate_risk_score(code, data)
        
        # 総合機会スコア計算
        candidate.opportunity_score = self._calculate_opportunity_score(candidate)
        candidate.priority_score = self._calculate_priority_score(candidate)
        candidate.last_analysis = datetime.now()
        
        return candidate
    
    def _calculate_volatility_score(self, code: str, data: dict) -> float:
        """ボラティリティスコア計算"""
        # 仮想的なボラティリティ分析
        sector = data['sector']
        market_cap = data.get('market_cap', 'medium')
        
        base_score = 0.5
        
        # セクター調整
        if sector in ['DayTrading', 'BioTech']:
            base_score += 0.3
        elif sector in ['Technology', 'Gaming']:
            base_score += 0.2
        elif sector in ['Financial', 'Industrial']:
            base_score += 0.1
        
        # 時価総額調整
        if market_cap == 'small':
            base_score += 0.2
        elif market_cap == 'large':
            base_score -= 0.1
        
        # ランダム要素（実際はリアルデータ分析）
        base_score += np.random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_momentum_score(self, code: str, data: dict) -> float:
        """モメンタムスコア計算"""
        # 仮想的なモメンタム分析
        sector = data['sector']
        
        base_score = 0.5
        
        # 成長セクター
        if sector in ['Technology', 'DayTrading', 'BioTech']:
            base_score += 0.25
        
        # ランダム要素
        base_score += np.random.uniform(-0.2, 0.2)
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_correlation_score(self, code: str, data: dict) -> float:
        """相関スコア計算"""
        # 既存銘柄との相関分析
        sector = data['sector']
        
        # 既存銘柄セクター分布チェック
        existing_sectors = set()
        for symbol_data in self.config.get('watchlist', {}).get('symbols', []):
            existing_sectors.add(symbol_data.get('sector', ''))
        
        # 新規セクターまたは不足セクターは高スコア
        if sector not in existing_sectors:
            return 0.8
        
        # セクター内での分散効果
        sector_count = sum(1 for s in self.config.get('watchlist', {}).get('symbols', [])
                          if s.get('sector') == sector)
        
        if sector_count < 5:
            return 0.6
        elif sector_count < 10:
            return 0.4
        else:
            return 0.2
    
    def _calculate_risk_score(self, code: str, data: dict) -> float:
        """リスクスコア計算"""
        sector = data['sector']
        market_cap = data.get('market_cap', 'medium')
        
        base_risk = 0.3
        
        # 高リスクセクター
        if sector in ['BioTech', 'DayTrading']:
            base_risk += 0.3
        elif sector in ['Energy', 'Materials']:
            base_risk += 0.2
        
        # 時価総額リスク
        if market_cap == 'small':
            base_risk += 0.2
        elif market_cap == 'large':
            base_risk -= 0.1
        
        return max(0.0, min(1.0, base_risk))
    
    def _calculate_opportunity_score(self, candidate: SymbolCandidate) -> float:
        """機会スコア計算"""
        # 重み付け総合スコア
        weights = {
            'volatility': 0.25,
            'momentum': 0.25,
            'correlation': 0.3,
            'risk_penalty': 0.2
        }
        
        score = (
            candidate.volatility_score * weights['volatility'] +
            candidate.momentum_score * weights['momentum'] +
            candidate.correlation_score * weights['correlation'] -
            candidate.risk_score * weights['risk_penalty']
        )
        
        return max(0.0, min(1.0, score))
    
    def _calculate_priority_score(self, candidate: SymbolCandidate) -> float:
        """優先度スコア計算"""
        priority = candidate.opportunity_score
        
        # セクター重要度調整
        if candidate.sector in ['DayTrading', 'BioTech']:
            priority += 0.2
        elif candidate.sector in ['Technology', 'Gaming']:
            priority += 0.1
        
        return max(0.0, min(1.0, priority))
    
    def select_expansion_candidates(self, limit: int = None) -> List[SymbolCandidate]:
        """拡張候補選択"""
        if limit is None:
            limit = self.expansion_batch_size
        
        opportunities = self.analyze_expansion_opportunities()
        
        # 現在の銘柄数チェック
        current_count = len(self.current_symbols)
        max_additions = min(limit, self.max_symbols - current_count)
        
        if max_additions <= 0:
            self.logger.warning("最大銘柄数に達しているため拡張できません")
            return []
        
        # トップ候補選択
        selected = opportunities[:max_additions]
        
        self.logger.info(f"拡張候補選択完了: {len(selected)}銘柄")
        for candidate in selected:
            self.logger.info(
                f"  {candidate.code} ({candidate.name}): "
                f"機会スコア {candidate.opportunity_score:.3f}"
            )
        
        return selected
    
    def add_symbols_to_watchlist(self, candidates: List[SymbolCandidate]) -> bool:
        """ウォッチリストに銘柄追加"""
        if not candidates:
            return False
        
        try:
            # 設定ファイル更新
            symbols_data = self.config.get('watchlist', {}).get('symbols', [])
            
            for candidate in candidates:
                symbol_entry = {
                    "code": candidate.code,
                    "name": candidate.name,
                    "group": candidate.sector,
                    "priority": self._determine_priority(candidate),
                    "sector": candidate.sector
                }
                symbols_data.append(symbol_entry)
                self.current_symbols.add(candidate.code)
                candidate.status = SymbolStatus.ACTIVE
            
            # 設定保存
            self.config['watchlist']['symbols'] = symbols_data
            self._save_config()
            
            self.metrics.symbols_added += len(candidates)
            self.logger.info(f"ウォッチリストに追加完了: {len(candidates)}銘柄")
            
            return True
            
        except Exception as e:
            self.logger.error(f"ウォッチリスト追加エラー: {e}")
            return False
    
    def _determine_priority(self, candidate: SymbolCandidate) -> str:
        """優先度決定"""
        if candidate.priority_score >= 0.8:
            return "high"
        elif candidate.priority_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _save_config(self) -> None:
        """設定ファイル保存"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"設定保存エラー: {e}")
    
    def generate_expansion_report(self) -> dict:
        """拡張レポート生成"""
        return {
            "timestamp": datetime.now().isoformat(),
            "current_symbols_count": len(self.current_symbols),
            "candidates_analyzed": len(self.candidates),
            "expansion_metrics": {
                "symbols_added": self.metrics.symbols_added,
                "symbols_removed": self.metrics.symbols_removed,
                "opportunities_found": self.metrics.opportunities_found,
                "accuracy_rate": self.metrics.accuracy_rate
            },
            "top_candidates": [
                {
                    "code": candidate.code,
                    "name": candidate.name,
                    "sector": candidate.sector,
                    "opportunity_score": round(candidate.opportunity_score, 3),
                    "priority_score": round(candidate.priority_score, 3)
                }
                for candidate in sorted(
                    self.candidates.values(),
                    key=lambda x: x.opportunity_score,
                    reverse=True
                )[:10]
            ]
        }
    
    async def run_expansion_cycle(self) -> None:
        """拡張サイクル実行"""
        self.logger.info("銘柄拡張システム開始")
        
        # 初期化
        self.initialize_current_symbols()
        
        while self.expansion_active:
            try:
                # 拡張候補分析・選択
                candidates = self.select_expansion_candidates()
                
                if candidates:
                    success = self.add_symbols_to_watchlist(candidates)
                    if success:
                        self.metrics.opportunities_found += len(candidates)
                
                # レポート生成（30分毎）
                await asyncio.sleep(1800)  # 30分待機
                
            except KeyboardInterrupt:
                self.logger.info("拡張システム停止要求受信")
                self.expansion_active = False
            except Exception as e:
                self.logger.error(f"拡張サイクルエラー: {e}")
                await asyncio.sleep(300)  # エラー時は5分待機
        
        self.logger.info("銘柄拡張システム終了")


def main():
    """メイン実行"""
    system = SymbolExpansionSystem()
    
    try:
        # 一回だけ実行
        system.initialize_current_symbols()
        candidates = system.select_expansion_candidates(10)
        
        if candidates:
            print(f"\n=== 拡張候補 {len(candidates)}銘柄 ===")
            for candidate in candidates:
                print(f"{candidate.code} ({candidate.name}): "
                      f"機会スコア {candidate.opportunity_score:.3f}")
            
            # レポート生成
            report = system.generate_expansion_report()
            print(f"\n=== 拡張レポート ===")
            print(f"現在監視銘柄数: {report['current_symbols_count']}")
            print(f"候補分析数: {report['candidates_analyzed']}")
            
        else:
            print("拡張候補が見つかりませんでした")
            
    except KeyboardInterrupt:
        print("\n銘柄拡張システム停止")
    except Exception as e:
        print(f"システムエラー: {e}")


if __name__ == "__main__":
    main()