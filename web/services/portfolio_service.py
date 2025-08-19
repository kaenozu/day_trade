#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Management Service - ポートフォリオ管理サービス
個人投資家向けの包括的なポートフォリオ追跡・分析機能
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

@dataclass
class Position:
    """保有ポジション"""
    symbol: str
    name: str
    quantity: int
    average_price: float
    current_price: float
    purchase_date: str
    sector: str
    category: str
    last_updated: str
    
    @property
    def total_value(self) -> float:
        """現在の総価値"""
        return self.quantity * self.current_price
    
    @property
    def total_cost(self) -> float:
        """総取得コスト"""
        return self.quantity * self.average_price
    
    @property
    def unrealized_pnl(self) -> float:
        """未実現損益"""
        return self.total_value - self.total_cost
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """未実現損益率（%）"""
        if self.total_cost > 0:
            return (self.unrealized_pnl / self.total_cost) * 100
        return 0.0
    
    @property
    def holding_days(self) -> int:
        """保有日数"""
        try:
            purchase_dt = datetime.fromisoformat(self.purchase_date)
            return (datetime.now() - purchase_dt).days
        except:
            return 0

@dataclass
class Transaction:
    """取引記録"""
    id: str
    symbol: str
    name: str
    action: str  # BUY, SELL
    quantity: int
    price: float
    total_amount: float
    commission: float
    date: str
    notes: str = ""
    
    @property
    def net_amount(self) -> float:
        """手数料込み実際取引金額"""
        if self.action == "BUY":
            return self.total_amount + self.commission
        else:
            return self.total_amount - self.commission

@dataclass
class PortfolioSummary:
    """ポートフォリオサマリー"""
    total_value: float
    total_cost: float
    total_pnl: float
    total_pnl_pct: float
    cash_balance: float
    total_assets: float
    positions_count: int
    sectors_exposure: Dict[str, float]
    top_holdings: List[Dict[str, Any]]
    daily_change: float
    daily_change_pct: float
    last_updated: str

class PortfolioService:
    """ポートフォリオ管理サービス"""
    
    def __init__(self, data_dir: str = "data/portfolio"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.positions_file = self.data_dir / "positions.json"
        self.transactions_file = self.data_dir / "transactions.json"
        self.settings_file = self.data_dir / "settings.json"
        
        self.logger = logging.getLogger(__name__)
        
        # 初期化
        self._initialize_files()
        
        # 設定読み込み
        self.settings = self._load_settings()
    
    def _initialize_files(self):
        """データファイルの初期化"""
        if not self.positions_file.exists():
            self._save_positions([])
        
        if not self.transactions_file.exists():
            self._save_transactions([])
        
        if not self.settings_file.exists():
            default_settings = {
                "initial_cash": 1000000,  # 初期資金100万円
                "current_cash": 1000000,
                "commission_rate": 0.01,  # 手数料1%
                "currency": "JPY",
                "risk_tolerance": "medium",
                "investment_style": "balanced"
            }
            self._save_settings(default_settings)
    
    def _load_positions(self) -> List[Position]:
        """ポジション読み込み"""
        try:
            with open(self.positions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [Position(**pos) for pos in data]
        except Exception as e:
            self.logger.error(f"ポジション読み込みエラー: {e}")
            return []
    
    def _save_positions(self, positions: List[Position]):
        """ポジション保存"""
        try:
            data = [asdict(pos) for pos in positions]
            with open(self.positions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"ポジション保存エラー: {e}")
    
    def _load_transactions(self) -> List[Transaction]:
        """取引履歴読み込み"""
        try:
            with open(self.transactions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [Transaction(**txn) for txn in data]
        except Exception as e:
            self.logger.error(f"取引履歴読み込みエラー: {e}")
            return []
    
    def _save_transactions(self, transactions: List[Transaction]):
        """取引履歴保存"""
        try:
            data = [asdict(txn) for txn in transactions]
            with open(self.transactions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"取引履歴保存エラー: {e}")
    
    def _load_settings(self) -> Dict[str, Any]:
        """設定読み込み"""
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"設定読み込みエラー: {e}")
            return {}
    
    def _save_settings(self, settings: Dict[str, Any]):
        """設定保存"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"設定保存エラー: {e}")
    
    def add_position(self, symbol: str, name: str, quantity: int, price: float, 
                    sector: str = "Unknown", category: str = "Stock") -> bool:
        """新規ポジション追加"""
        try:
            positions = self._load_positions()
            
            # 既存ポジションチェック
            existing_pos = next((p for p in positions if p.symbol == symbol), None)
            
            if existing_pos:
                # 平均取得価格の再計算
                total_quantity = existing_pos.quantity + quantity
                total_cost = (existing_pos.quantity * existing_pos.average_price) + (quantity * price)
                new_average_price = total_cost / total_quantity
                
                existing_pos.quantity = total_quantity
                existing_pos.average_price = new_average_price
                existing_pos.current_price = price
                existing_pos.last_updated = datetime.now().isoformat()
            else:
                # 新規ポジション作成
                new_position = Position(
                    symbol=symbol,
                    name=name,
                    quantity=quantity,
                    average_price=price,
                    current_price=price,
                    purchase_date=datetime.now().isoformat(),
                    sector=sector,
                    category=category,
                    last_updated=datetime.now().isoformat()
                )
                positions.append(new_position)
            
            # 取引記録追加
            self._add_transaction("BUY", symbol, name, quantity, price)
            
            # 現金残高更新
            total_cost = quantity * price
            commission = total_cost * self.settings.get("commission_rate", 0.01)
            self._update_cash_balance(-(total_cost + commission))
            
            self._save_positions(positions)
            return True
            
        except Exception as e:
            self.logger.error(f"ポジション追加エラー: {e}")
            return False
    
    def sell_position(self, symbol: str, quantity: int, price: float) -> bool:
        """ポジション売却"""
        try:
            positions = self._load_positions()
            position = next((p for p in positions if p.symbol == symbol), None)
            
            if not position:
                self.logger.error(f"ポジションが見つかりません: {symbol}")
                return False
            
            if position.quantity < quantity:
                self.logger.error(f"売却数量が保有数量を超えています: {symbol}")
                return False
            
            # ポジション更新
            position.quantity -= quantity
            position.current_price = price
            position.last_updated = datetime.now().isoformat()
            
            # 数量が0になった場合はポジション削除
            if position.quantity == 0:
                positions.remove(position)
            
            # 取引記録追加
            self._add_transaction("SELL", symbol, position.name, quantity, price)
            
            # 現金残高更新
            total_amount = quantity * price
            commission = total_amount * self.settings.get("commission_rate", 0.01)
            self._update_cash_balance(total_amount - commission)
            
            self._save_positions(positions)
            return True
            
        except Exception as e:
            self.logger.error(f"ポジション売却エラー: {e}")
            return False
    
    def update_prices(self, price_data: Dict[str, float]) -> bool:
        """価格一括更新"""
        try:
            positions = self._load_positions()
            updated = False
            
            for position in positions:
                if position.symbol in price_data:
                    position.current_price = price_data[position.symbol]
                    position.last_updated = datetime.now().isoformat()
                    updated = True
            
            if updated:
                self._save_positions(positions)
            
            return updated
            
        except Exception as e:
            self.logger.error(f"価格更新エラー: {e}")
            return False
    
    def _add_transaction(self, action: str, symbol: str, name: str, quantity: int, price: float):
        """取引記録追加"""
        try:
            transactions = self._load_transactions()
            
            total_amount = quantity * price
            commission = total_amount * self.settings.get("commission_rate", 0.01)
            
            transaction = Transaction(
                id=f"{symbol}_{action}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                name=name,
                action=action,
                quantity=quantity,
                price=price,
                total_amount=total_amount,
                commission=commission,
                date=datetime.now().isoformat()
            )
            
            transactions.append(transaction)
            self._save_transactions(transactions)
            
        except Exception as e:
            self.logger.error(f"取引記録追加エラー: {e}")
    
    def _update_cash_balance(self, amount: float):
        """現金残高更新"""
        self.settings["current_cash"] = self.settings.get("current_cash", 0) + amount
        self._save_settings(self.settings)
    
    def get_portfolio_summary(self) -> PortfolioSummary:
        """ポートフォリオサマリー取得"""
        try:
            positions = self._load_positions()
            
            total_value = sum(pos.total_value for pos in positions)
            total_cost = sum(pos.total_cost for pos in positions)
            total_pnl = total_value - total_cost
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
            
            cash_balance = self.settings.get("current_cash", 0)
            total_assets = total_value + cash_balance
            
            # セクター別構成
            sectors_exposure = {}
            for pos in positions:
                sector = pos.sector
                if sector not in sectors_exposure:
                    sectors_exposure[sector] = 0
                sectors_exposure[sector] += pos.total_value
            
            # 上位保有銘柄
            top_holdings = sorted(
                [{"symbol": pos.symbol, "name": pos.name, "value": pos.total_value, "weight": pos.total_value/total_value*100 if total_value > 0 else 0}
                 for pos in positions],
                key=lambda x: x["value"],
                reverse=True
            )[:5]
            
            # 日次変動（簡易計算）
            daily_change = sum(pos.quantity * (pos.current_price - pos.average_price) for pos in positions) * 0.01  # 仮の値
            daily_change_pct = (daily_change / total_value * 100) if total_value > 0 else 0
            
            return PortfolioSummary(
                total_value=total_value,
                total_cost=total_cost,
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl_pct,
                cash_balance=cash_balance,
                total_assets=total_assets,
                positions_count=len(positions),
                sectors_exposure=sectors_exposure,
                top_holdings=top_holdings,
                daily_change=daily_change,
                daily_change_pct=daily_change_pct,
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"ポートフォリオサマリー取得エラー: {e}")
            return PortfolioSummary(
                total_value=0, total_cost=0, total_pnl=0, total_pnl_pct=0,
                cash_balance=0, total_assets=0, positions_count=0,
                sectors_exposure={}, top_holdings=[], daily_change=0,
                daily_change_pct=0, last_updated=datetime.now().isoformat()
            )
    
    def get_positions(self) -> List[Position]:
        """全ポジション取得"""
        return self._load_positions()
    
    def get_transactions(self, limit: int = 50) -> List[Transaction]:
        """取引履歴取得"""
        transactions = self._load_transactions()
        return sorted(transactions, key=lambda t: t.date, reverse=True)[:limit]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンス指標計算"""
        try:
            positions = self._load_positions()
            transactions = self._load_transactions()
            
            if not positions and not transactions:
                return {"error": "データが不足しています"}
            
            # 基本指標
            total_value = sum(pos.total_value for pos in positions)
            total_cost = sum(pos.total_cost for pos in positions)
            
            # 勝率計算（売却済み取引のみ）
            sell_transactions = [t for t in transactions if t.action == "SELL"]
            profitable_sells = len([t for t in sell_transactions if t.price > t.total_amount/t.quantity])
            win_rate = (profitable_sells / len(sell_transactions) * 100) if sell_transactions else 0
            
            # 保有期間分析
            if positions:
                avg_holding_days = sum(pos.holding_days for pos in positions) / len(positions)
                max_holding_days = max(pos.holding_days for pos in positions)
            else:
                avg_holding_days = 0
                max_holding_days = 0
            
            # セクター分散度（簡易Herfindahl指数）
            sector_weights = {}
            for pos in positions:
                sector = pos.sector
                if sector not in sector_weights:
                    sector_weights[sector] = 0
                sector_weights[sector] += pos.total_value
            
            if total_value > 0:
                sector_hhi = sum((weight/total_value)**2 for weight in sector_weights.values())
                diversification_score = 1 - sector_hhi
            else:
                diversification_score = 0
            
            return {
                "total_return_pct": ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0,
                "win_rate": win_rate,
                "avg_holding_days": avg_holding_days,
                "max_holding_days": max_holding_days,
                "diversification_score": diversification_score,
                "sectors_count": len(sector_weights),
                "largest_position_pct": (max(pos.total_value for pos in positions) / total_value * 100) if positions and total_value > 0 else 0,
                "cash_ratio": (self.settings.get("current_cash", 0) / (total_value + self.settings.get("current_cash", 0)) * 100) if total_value + self.settings.get("current_cash", 0) > 0 else 100
            }
            
        except Exception as e:
            self.logger.error(f"パフォーマンス指標計算エラー: {e}")
            return {"error": str(e)}
    
    def get_risk_analysis(self) -> Dict[str, Any]:
        """リスク分析"""
        try:
            positions = self._load_positions()
            
            if not positions:
                return {"warning": "ポジションがありません"}
            
            total_value = sum(pos.total_value for pos in positions)
            
            # 集中リスク分析
            position_weights = [(pos.symbol, pos.total_value/total_value*100) for pos in positions if total_value > 0]
            position_weights.sort(key=lambda x: x[1], reverse=True)
            
            # 上位3銘柄の集中度
            top3_concentration = sum(weight for _, weight in position_weights[:3])
            
            # セクター集中度
            sector_exposure = {}
            for pos in positions:
                sector = pos.sector
                if sector not in sector_exposure:
                    sector_exposure[sector] = 0
                sector_exposure[sector] += pos.total_value
            
            max_sector_exposure = (max(sector_exposure.values()) / total_value * 100) if total_value > 0 else 0
            
            # リスクレベル判定
            risk_level = "低"
            risk_factors = []
            
            if top3_concentration > 60:
                risk_level = "高"
                risk_factors.append("上位3銘柄の集中度が高い")
            elif top3_concentration > 40:
                risk_level = "中"
                risk_factors.append("銘柄集中度がやや高い")
            
            if max_sector_exposure > 50:
                risk_level = "高"
                risk_factors.append("特定セクターへの集中度が高い")
            elif max_sector_exposure > 30:
                if risk_level == "低":
                    risk_level = "中"
                risk_factors.append("セクター集中度がやや高い")
            
            # 現金比率チェック
            cash_ratio = self.settings.get("current_cash", 0) / (total_value + self.settings.get("current_cash", 0)) * 100
            if cash_ratio < 5:
                risk_factors.append("現金比率が低い")
            
            return {
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "top3_concentration": top3_concentration,
                "max_sector_exposure": max_sector_exposure,
                "cash_ratio": cash_ratio,
                "positions_count": len(positions),
                "largest_position": position_weights[0] if position_weights else None,
                "recommendations": self._get_risk_recommendations(risk_level, risk_factors)
            }
            
        except Exception as e:
            self.logger.error(f"リスク分析エラー: {e}")
            return {"error": str(e)}
    
    def _get_risk_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """リスク改善提案"""
        recommendations = []
        
        if "上位3銘柄の集中度が高い" in risk_factors:
            recommendations.append("ポートフォリオの分散を検討してください")
        
        if "特定セクターへの集中度が高い" in risk_factors:
            recommendations.append("異なるセクターへの投資を検討してください")
        
        if "現金比率が低い" in risk_factors:
            recommendations.append("一部利益確定を検討してください")
        
        if risk_level == "高":
            recommendations.append("リスク軽減のための売却を検討してください")
        
        if not recommendations:
            recommendations.append("現在のポートフォリオは適切に分散されています")
        
        return recommendations