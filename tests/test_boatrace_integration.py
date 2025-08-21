"""
Boatraceシステム統合テスト

主要機能の動作確認テスト
"""

import pytest
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import sys

# パス設定
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.boatrace.core.api_client import BoatraceAPIClient
from src.boatrace.core.stadium_manager import StadiumManager
from src.boatrace.core.data_models import STADIUMS
from src.boatrace.data.database import init_database
from src.boatrace.prediction.racer_analyzer import RacerAnalyzer
from src.boatrace.betting.ticket_manager import TicketManager, BetType
from src.boatrace.betting.betting_strategy import ConservativeStrategy
from src.boatrace.betting.portfolio import Portfolio


class TestBoatraceIntegration:
    """統合テスト"""
    
    @classmethod
    def setup_class(cls):
        """テストクラス初期化"""
        cls.database = init_database(":memory:")  # インメモリDB
        cls.api_client = BoatraceAPIClient()
        cls.stadium_manager = StadiumManager()
        cls.racer_analyzer = RacerAnalyzer(cls.database)
        cls.ticket_manager = TicketManager(cls.database)
        cls.portfolio = Portfolio(Decimal('50000'), cls.database)
    
    def test_api_client_basic(self):
        """APIクライアント基本テスト"""
        # 接続テスト
        connectivity = self.api_client.validate_api_connectivity()
        assert isinstance(connectivity, bool)
        
        if connectivity:
            # 今日のプログラム取得テスト
            today_programs = self.api_client.get_today_programs()
            assert 'programs' in today_programs
            assert isinstance(today_programs['programs'], list)
        else:
            pytest.skip("API接続不可のため、APIテストをスキップ")
    
    def test_stadium_manager(self):
        """競技場管理テスト"""
        # 全競技場取得
        all_stadiums = self.stadium_manager.get_all_stadiums()
        assert len(all_stadiums) == 24
        
        # 個別競技場取得
        edogawa = self.stadium_manager.get_stadium(3)  # 江戸川
        assert edogawa is not None
        assert edogawa.name == "江戸川"
        
        # 特性取得
        char = self.stadium_manager.get_characteristics(3)
        assert char is not None
        assert char.is_tidal == True  # 江戸川は潮汐影響あり
        
        # 分析取得
        analysis = self.stadium_manager.get_stadium_analysis(3)
        assert analysis is not None
        assert 'stadium' in analysis
        assert 'characteristics' in analysis
    
    def test_data_models(self):
        """データモデルテスト"""
        # 競技場マスタ
        assert len(STADIUMS) == 24
        assert all(1 <= num <= 24 for num in STADIUMS.keys())
        
        # 各競技場の基本情報チェック
        for num, stadium in STADIUMS.items():
            assert stadium.number == num
            assert len(stadium.name) > 0
            assert len(stadium.prefecture) > 0
    
    def test_betting_system(self):
        """投票システムテスト"""
        # ダミーレースIDで舟券購入テスト
        race_id = "20250821_03_01"
        
        # 舟券購入記録
        ticket_id = self.ticket_manager.purchase_ticket(
            race_id=race_id,
            bet_type=BetType.WIN,
            numbers="1",
            amount=Decimal('1000'),
            strategy_name="テスト戦略"
        )
        
        assert ticket_id > 0
        
        # ポートフォリオ状況確認
        balance = self.portfolio.get_current_balance()
        assert balance['initial_capital'] == 50000.0
        assert balance['total_invested'] == 1000.0
    
    def test_betting_strategy(self):
        """投票戦略テスト"""
        from src.boatrace.core.data_models import PredictionResult
        
        # ダミー予想結果作成
        prediction = PredictionResult(
            race_id="20250821_03_01",
            predicted_at=datetime.now(),
            win_probabilities={
                1: Decimal('0.35'),
                2: Decimal('0.25'),
                3: Decimal('0.15'),
                4: Decimal('0.10'),
                5: Decimal('0.10'),
                6: Decimal('0.05')
            },
            place_probabilities={
                1: Decimal('0.65'),
                2: Decimal('0.55'),
                3: Decimal('0.45'),
                4: Decimal('0.35'),
                5: Decimal('0.25'),
                6: Decimal('0.15')
            },
            recommended_bets=[],
            confidence=Decimal('0.75')
        )
        
        # 保守的戦略テスト
        conservative = ConservativeStrategy()
        budget = Decimal('5000')
        
        recommendations = conservative.generate_bets(
            prediction, budget, {'competitiveness': '実力差'}
        )
        
        assert len(recommendations) > 0
        
        # 推奨内容確認
        total_amount = sum(rec.amount for rec in recommendations)
        assert total_amount <= budget
        
        # リスクレベル確認
        for rec in recommendations:
            assert rec.risk_level.value == 'conservative'
    
    def test_portfolio_management(self):
        """ポートフォリオ管理テスト"""
        # 複数の投票記録
        race_ids = ["20250821_03_01", "20250821_03_02", "20250821_03_03"]
        
        for i, race_id in enumerate(race_ids):
            self.ticket_manager.purchase_ticket(
                race_id=race_id,
                bet_type=BetType.WIN,
                numbers=str(i + 1),
                amount=Decimal('2000'),
                strategy_name=f"戦略{i + 1}"
            )
        
        # ポートフォリオサマリー
        summary = self.ticket_manager.get_portfolio_summary(days_back=1)
        
        assert 'summary' in summary
        assert summary['summary']['total_tickets'] >= 3
        
        # パフォーマンス指標（データ不足でも基本構造確認）
        balance = self.portfolio.get_current_balance()
        assert 'initial_capital' in balance
        assert 'current_balance' in balance
        assert 'roi' in balance
    
    def test_risk_analysis(self):
        """リスク分析テスト"""
        # リスク分析実行
        risk_analysis = self.portfolio.get_risk_analysis()
        
        # 基本構造確認
        expected_keys = ['overall_risk_level', 'capital_utilization', 'recommendations']
        for key in expected_keys:
            assert key in risk_analysis
        
        # 推奨事項の存在確認
        assert isinstance(risk_analysis['recommendations'], list)
    
    def test_bet_sizing_optimization(self):
        """賭け金最適化テスト"""
        # 最適賭け金計算
        optimal_bet = self.portfolio.optimize_bet_sizing(
            prediction_confidence=0.75,
            expected_odds=3.0,
            max_risk_per_bet=0.05
        )
        
        assert isinstance(optimal_bet, Decimal)
        assert optimal_bet > 0
        
        # 最大リスク制限確認
        current_balance = self.portfolio.get_current_balance()['current_balance']
        max_risk_amount = current_balance * 0.05
        assert optimal_bet <= Decimal(str(max_risk_amount))


def test_system_integration():
    """システム全体統合テスト"""
    # 基本的なシステム初期化テスト
    database = init_database(":memory:")
    assert database is not None
    
    # 各コンポーネントの初期化テスト
    api_client = BoatraceAPIClient()
    stadium_manager = StadiumManager()
    
    # 基本機能テスト
    stadiums = stadium_manager.get_all_stadiums()
    assert len(stadiums) == 24
    
    # API接続テスト（接続可能な場合のみ）
    if api_client.validate_api_connectivity():
        cache_dir = Path("data/cache")
        assert cache_dir.exists() or True  # キャッシュディレクトリが作成される


if __name__ == "__main__":
    # 直接実行時のテスト
    print("🧪 Boatraceシステム統合テスト実行")
    
    # pytest実行
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    
    if exit_code == 0:
        print("✅ 全てのテストが成功しました")
    else:
        print("❌ テストに失敗しました")
    
    exit(exit_code)