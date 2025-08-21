"""
BoatraceOpenAPIクライアント

競艇公式データを取得するためのAPIクライアント実装
"""

import json
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class APIEndpoints:
    """API エンドポイント定義"""
    BASE_URL = "https://boatraceopenapi.github.io"
    PROGRAMS = f"{BASE_URL}/programs/v2"
    PREVIEWS = f"{BASE_URL}/previews/v2" 
    RESULTS = f"{BASE_URL}/results/v2"
    
    @classmethod
    def programs_url(cls, date_str: str) -> str:
        """出走表APIのURL"""
        if date_str.lower() == "today":
            return f"{cls.PROGRAMS}/today.json"
        return f"{cls.PROGRAMS}/{date_str}.json"
    
    @classmethod
    def previews_url(cls, date_str: str) -> str:
        """直前情報APIのURL"""
        return f"{cls.PREVIEWS}/{date_str}.json"
    
    @classmethod
    def results_url(cls, date_str: str) -> str:
        """結果APIのURL"""
        return f"{cls.RESULTS}/{date_str}.json"


class BoatraceAPIClient:
    """BoatraceOpenAPIクライアント"""
    
    def __init__(self, 
                 timeout: int = 30,
                 max_retries: int = 3,
                 backoff_factor: float = 0.3,
                 cache_dir: Optional[Path] = None):
        """
        初期化
        
        Args:
            timeout: タイムアウト秒数
            max_retries: 最大リトライ回数
            backoff_factor: リトライ間隔の係数
            cache_dir: キャッシュディレクトリ
        """
        self.timeout = timeout
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # HTTP セッション設定
        self.session = requests.Session()
        
        # リトライ戦略設定
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=backoff_factor
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # ヘッダー設定
        self.session.headers.update({
            'User-Agent': 'BoatraceSystem/1.0.0',
            'Accept': 'application/json',
            'Accept-Charset': 'utf-8'
        })
    
    def _get_cache_path(self, endpoint: str, date_str: str) -> Path:
        """キャッシュファイルパスを取得"""
        return self.cache_dir / f"{endpoint}_{date_str}.json"
    
    def _save_cache(self, endpoint: str, date_str: str, data: Dict[str, Any]) -> None:
        """キャッシュに保存"""
        try:
            cache_path = self._get_cache_path(endpoint, date_str)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"キャッシュ保存: {cache_path}")
        except Exception as e:
            logger.warning(f"キャッシュ保存失敗: {e}")
    
    def _load_cache(self, endpoint: str, date_str: str) -> Optional[Dict[str, Any]]:
        """キャッシュから読み込み"""
        try:
            cache_path = self._get_cache_path(endpoint, date_str)
            if cache_path.exists():
                # 今日のデータはキャッシュしない（リアルタイム性重視）
                if date_str.lower() == "today":
                    cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
                    if cache_age > timedelta(minutes=10):  # 10分でキャッシュ無効
                        return None
                
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.debug(f"キャッシュ読み込み: {cache_path}")
                return data
        except Exception as e:
            logger.warning(f"キャッシュ読み込み失敗: {e}")
        return None
    
    def _fetch_data(self, url: str) -> Dict[str, Any]:
        """APIからデータを取得"""
        try:
            logger.info(f"API呼び出し: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"API呼び出し成功: {len(data.get('programs', []))} programs")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API呼び出し失敗: {url} - {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失敗: {url} - {e}")
            raise
    
    def get_programs(self, 
                    race_date: Optional[date] = None,
                    use_cache: bool = True) -> Dict[str, Any]:
        """
        出走表データを取得
        
        Args:
            race_date: レース日付（Noneの場合は今日）
            use_cache: キャッシュを使用するか
            
        Returns:
            出走表データ
        """
        date_str = "today" if race_date is None else race_date.strftime("%Y%m%d")
        
        # キャッシュチェック
        if use_cache:
            cached_data = self._load_cache("programs", date_str)
            if cached_data:
                return cached_data
        
        # API呼び出し
        url = APIEndpoints.programs_url(date_str)
        data = self._fetch_data(url)
        
        # キャッシュ保存
        if use_cache:
            self._save_cache("programs", date_str, data)
        
        return data
    
    def get_previews(self, 
                    race_date: date,
                    use_cache: bool = True) -> Dict[str, Any]:
        """
        直前情報データを取得
        
        Args:
            race_date: レース日付
            use_cache: キャッシュを使用するか
            
        Returns:
            直前情報データ
        """
        date_str = race_date.strftime("%Y%m%d")
        
        # キャッシュチェック
        if use_cache:
            cached_data = self._load_cache("previews", date_str)
            if cached_data:
                return cached_data
        
        # API呼び出し
        url = APIEndpoints.previews_url(date_str)
        data = self._fetch_data(url)
        
        # キャッシュ保存
        if use_cache:
            self._save_cache("previews", date_str, data)
        
        return data
    
    def get_results(self, 
                   race_date: date,
                   use_cache: bool = True) -> Dict[str, Any]:
        """
        結果データを取得
        
        Args:
            race_date: レース日付
            use_cache: キャッシュを使用するか
            
        Returns:
            結果データ
        """
        date_str = race_date.strftime("%Y%m%d")
        
        # キャッシュチェック
        if use_cache:
            cached_data = self._load_cache("results", date_str)
            if cached_data:
                return cached_data
        
        # API呼び出し
        url = APIEndpoints.results_url(date_str)
        data = self._fetch_data(url)
        
        # キャッシュ保存
        if use_cache:
            self._save_cache("results", date_str, data)
        
        return data
    
    def get_today_programs(self) -> Dict[str, Any]:
        """今日の出走表データを取得"""
        return self.get_programs(race_date=None)
    
    def get_date_range_programs(self, 
                               start_date: date, 
                               end_date: date,
                               use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        日付範囲の出走表データを取得
        
        Args:
            start_date: 開始日
            end_date: 終了日
            use_cache: キャッシュを使用するか
            
        Returns:
            出走表データのリスト
        """
        results = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                data = self.get_programs(current_date, use_cache)
                if data.get('programs'):
                    results.append(data)
            except Exception as e:
                logger.warning(f"データ取得スキップ {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        return results
    
    def get_stadium_programs(self, 
                           race_date: Optional[date] = None,
                           stadium_number: int = None) -> List[Dict[str, Any]]:
        """
        特定競技場の出走表データを取得
        
        Args:
            race_date: レース日付
            stadium_number: 競技場番号（1-24）
            
        Returns:
            競技場のレースデータリスト
        """
        data = self.get_programs(race_date)
        programs = data.get('programs', [])
        
        if stadium_number is not None:
            programs = [p for p in programs if p.get('race_stadium_number') == stadium_number]
        
        return programs
    
    def validate_api_connectivity(self) -> bool:
        """API接続性をテスト"""
        try:
            self.get_today_programs()
            logger.info("API接続テスト成功")
            return True
        except Exception as e:
            logger.error(f"API接続テスト失敗: {e}")
            return False
    
    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("キャッシュクリア完了")
        except Exception as e:
            logger.error(f"キャッシュクリア失敗: {e}")
    
    def __del__(self):
        """デストラクタ"""
        if hasattr(self, 'session'):
            self.session.close()