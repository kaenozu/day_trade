"""
ELK Stack統合システム
Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

Elasticsearch, Logstash, Kibanaの統合による
エンタープライズレベルのログ管理システム。
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import elasticsearch
    from elasticsearch import Elasticsearch, helpers

    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from .log_aggregation_system import (
        LogAlert,
        LogEntry,
        LogLevel,
        LogPattern,
        LogSource,
    )

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


class ElasticSearchConnectionStatus(Enum):
    """Elasticsearch接続状態"""

    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ERROR = "error"
    INDEXING = "indexing"


class KibanaVisualizationType(Enum):
    """Kibana可視化タイプ"""

    LINE_CHART = "line"
    BAR_CHART = "bar"
    PIE_CHART = "pie"
    DATA_TABLE = "table"
    HEATMAP = "heatmap"
    METRIC = "metric"


@dataclass
class ElasticSearchConfig:
    """Elasticsearch設定"""

    hosts: List[str] = field(default_factory=lambda: ["localhost:9200"])
    username: Optional[str] = None
    password: Optional[str] = None
    use_ssl: bool = False
    verify_certs: bool = False
    ca_certs: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_on_timeout: bool = True


@dataclass
class LogstashConfig:
    """Logstash設定"""

    host: str = "localhost"
    port: int = 5044
    protocol: str = "tcp"  # tcp, udp, http
    ssl_enabled: bool = False
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None


@dataclass
class KibanaConfig:
    """Kibana設定"""

    host: str = "localhost"
    port: int = 5601
    username: Optional[str] = None
    password: Optional[str] = None
    use_ssl: bool = False


@dataclass
class ELKIndexTemplate:
    """Elasticsearchインデックステンプレート"""

    name: str
    pattern: str
    settings: Dict[str, Any]
    mappings: Dict[str, Any]
    version: int = 1


class ElasticsearchManager:
    """Elasticsearch管理システム"""

    def __init__(self, config: ElasticSearchConfig):
        self.config = config
        self.client: Optional[Elasticsearch] = None
        self.connection_status = ElasticSearchConnectionStatus.DISCONNECTED
        self.logger = logging.getLogger(__name__)

    async def connect(self) -> bool:
        """Elasticsearch接続"""
        try:
            if not ELASTICSEARCH_AVAILABLE:
                self.logger.error("Elasticsearchライブラリが利用できません")
                return False

            # 接続設定構築
            connection_params = {
                "hosts": self.config.hosts,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
                "retry_on_timeout": self.config.retry_on_timeout,
            }

            if self.config.username and self.config.password:
                connection_params["http_auth"] = (
                    self.config.username,
                    self.config.password,
                )

            if self.config.use_ssl:
                connection_params["use_ssl"] = True
                connection_params["verify_certs"] = self.config.verify_certs
                if self.config.ca_certs:
                    connection_params["ca_certs"] = self.config.ca_certs

            # Elasticsearchクライアント作成
            self.client = Elasticsearch(**connection_params)

            # 接続テスト
            if self.client.ping():
                self.connection_status = ElasticSearchConnectionStatus.CONNECTED
                cluster_health = self.client.cluster.health()
                self.logger.info(f"Elasticsearch接続成功: {cluster_health['cluster_name']}")
                return True
            else:
                self.connection_status = ElasticSearchConnectionStatus.ERROR
                self.logger.error("Elasticsearch接続失敗: pingテスト失敗")
                return False

        except Exception as e:
            self.connection_status = ElasticSearchConnectionStatus.ERROR
            self.logger.error(f"Elasticsearch接続エラー: {e}")
            return False

    async def create_index_template(self, template: ELKIndexTemplate) -> bool:
        """インデックステンプレート作成"""
        try:
            if not self.client or self.connection_status != ElasticSearchConnectionStatus.CONNECTED:
                self.logger.error("Elasticsearchに接続されていません")
                return False

            template_body = {
                "index_patterns": [template.pattern],
                "template": {
                    "settings": template.settings,
                    "mappings": template.mappings,
                },
                "version": template.version,
            }

            response = self.client.indices.put_index_template(
                name=template.name, body=template_body
            )

            if response.get("acknowledged"):
                self.logger.info(f"インデックステンプレート作成成功: {template.name}")
                return True
            else:
                self.logger.error(f"インデックステンプレート作成失敗: {template.name}")
                return False

        except Exception as e:
            self.logger.error(f"インデックステンプレート作成エラー: {e}")
            return False

    async def index_log_entry(self, log_entry: LogEntry, index_name: str) -> bool:
        """ログエントリインデックス化"""
        try:
            if not self.client or self.connection_status != ElasticSearchConnectionStatus.CONNECTED:
                return False

            # Elasticsearch用ドキュメント作成
            doc = {
                "@timestamp": log_entry.timestamp.isoformat(),
                "log_id": log_entry.id,
                "level": log_entry.level.value,
                "source": log_entry.source.value,
                "component": log_entry.component,
                "message": log_entry.message,
                "structured_data": log_entry.structured_data,
                "tags": log_entry.tags,
                "trace_id": log_entry.trace_id,
                "user_id": log_entry.user_id,
                "session_id": log_entry.session_id,
                "parsed_at": log_entry.parsed_at.isoformat(),
            }

            # インデックス実行
            response = self.client.index(index=index_name, id=log_entry.id, body=doc)

            return response.get("result") in ["created", "updated"]

        except Exception as e:
            self.logger.error(f"ログインデックス化エラー: {e}")
            return False

    async def bulk_index_logs(self, log_entries: List[LogEntry], index_name: str) -> int:
        """ログエントリ一括インデックス化"""
        try:
            if not self.client or self.connection_status != ElasticSearchConnectionStatus.CONNECTED:
                return 0

            # バルクインデックス用データ準備
            actions = []
            for log_entry in log_entries:
                doc = {
                    "@timestamp": log_entry.timestamp.isoformat(),
                    "log_id": log_entry.id,
                    "level": log_entry.level.value,
                    "source": log_entry.source.value,
                    "component": log_entry.component,
                    "message": log_entry.message,
                    "structured_data": log_entry.structured_data,
                    "tags": log_entry.tags,
                    "trace_id": log_entry.trace_id,
                    "user_id": log_entry.user_id,
                    "session_id": log_entry.session_id,
                    "parsed_at": log_entry.parsed_at.isoformat(),
                }

                action = {"_index": index_name, "_id": log_entry.id, "_source": doc}
                actions.append(action)

            # バルクインデックス実行
            success_count, failed_items = helpers.bulk(
                self.client, actions, chunk_size=1000, request_timeout=60
            )

            if failed_items:
                self.logger.warning(f"バルクインデックス: {len(failed_items)}件失敗")

            self.logger.info(f"バルクインデックス完了: {success_count}件成功")
            return success_count

        except Exception as e:
            self.logger.error(f"バルクインデックスエラー: {e}")
            return 0

    async def search_logs(
        self, index_pattern: str, query: Dict[str, Any], size: int = 100, from_: int = 0
    ) -> Dict[str, Any]:
        """ログ検索"""
        try:
            if not self.client or self.connection_status != ElasticSearchConnectionStatus.CONNECTED:
                return {"hits": {"total": {"value": 0}, "hits": []}}

            response = self.client.search(index=index_pattern, body=query, size=size, from_=from_)

            return response

        except Exception as e:
            self.logger.error(f"ログ検索エラー: {e}")
            return {"hits": {"total": {"value": 0}, "hits": []}}

    async def get_cluster_health(self) -> Dict[str, Any]:
        """クラスター健全性取得"""
        try:
            if not self.client or self.connection_status != ElasticSearchConnectionStatus.CONNECTED:
                return {}

            health = self.client.cluster.health()
            stats = self.client.cluster.stats()

            return {
                "cluster_name": health.get("cluster_name"),
                "status": health.get("status"),
                "number_of_nodes": health.get("number_of_nodes"),
                "number_of_data_nodes": health.get("number_of_data_nodes"),
                "active_primary_shards": health.get("active_primary_shards"),
                "active_shards": health.get("active_shards"),
                "relocating_shards": health.get("relocating_shards"),
                "initializing_shards": health.get("initializing_shards"),
                "unassigned_shards": health.get("unassigned_shards"),
                "indices_count": stats.get("indices", {}).get("count", 0),
                "total_store_size": stats.get("indices", {})
                .get("store", {})
                .get("size_in_bytes", 0),
            }

        except Exception as e:
            self.logger.error(f"クラスター健全性取得エラー: {e}")
            return {}


class LogstashIntegration:
    """Logstash統合システム"""

    def __init__(self, config: LogstashConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def send_log_to_logstash(self, log_entry: LogEntry) -> bool:
        """LogstashにログエントリSend"""
        try:
            # Logstash用JSONフォーマット
            logstash_doc = {
                "@timestamp": log_entry.timestamp.isoformat(),
                "log_id": log_entry.id,
                "level": log_entry.level.value,
                "source": log_entry.source.value,
                "component": log_entry.component,
                "message": log_entry.message,
                "structured_data": log_entry.structured_data,
                "tags": log_entry.tags,
                "trace_id": log_entry.trace_id,
                "user_id": log_entry.user_id,
                "session_id": log_entry.session_id,
                "host": "day_trade_system",
                "environment": "production",
            }

            if self.config.protocol == "http":
                # HTTP経由でLogstashに送信
                url = f"http{'s' if self.config.ssl_enabled else ''}://{self.config.host}:{self.config.port}"

                headers = {"Content-Type": "application/json"}

                if REQUESTS_AVAILABLE:
                    response = requests.post(
                        url,
                        json=logstash_doc,
                        headers=headers,
                        timeout=10,
                        verify=self.config.ssl_enabled,
                    )
                    return response.status_code == 200
                else:
                    self.logger.warning("requestsライブラリが利用できません")
                    return False

            elif self.config.protocol in ["tcp", "udp"]:
                # TCP/UDP経由でLogstashに送信（简单実装）
                import socket

                sock_type = (
                    socket.SOCK_STREAM if self.config.protocol == "tcp" else socket.SOCK_DGRAM
                )

                with socket.socket(socket.AF_INET, sock_type) as sock:
                    sock.settimeout(10)

                    if self.config.protocol == "tcp":
                        sock.connect((self.config.host, self.config.port))

                    message = json.dumps(logstash_doc) + "\n"

                    if self.config.protocol == "tcp":
                        sock.sendall(message.encode("utf-8"))
                    else:
                        sock.sendto(
                            message.encode("utf-8"),
                            (self.config.host, self.config.port),
                        )

                    return True

            else:
                self.logger.error(f"サポートされていないプロトコル: {self.config.protocol}")
                return False

        except Exception as e:
            self.logger.error(f"Logstash送信エラー: {e}")
            return False

    def generate_logstash_config(self, elasticsearch_config: ElasticSearchConfig) -> str:
        """Logstash設定ファイル生成"""
        config_template = f"""
input {{
  beats {{
    port => {self.config.port}
  }}

  tcp {{
    port => {self.config.port + 1}
    codec => json_lines
  }}

  http {{
    port => {self.config.port + 2}
    codec => json
  }}
}}

filter {{
  # タイムスタンプパース
  date {{
    match => [ "@timestamp", "ISO8601" ]
  }}

  # ログレベル正規化
  mutate {{
    uppercase => [ "level" ]
  }}

  # 構造化データ展開
  if [structured_data] {{
    ruby {{
      code => "
        structured = event.get('structured_data')
        if structured.is_a?(Hash)
          structured.each {{ |k, v| event.set(k, v) }}
        end
      "
    }}
  }}

  # GeoIPエンリッチメント（IPアドレスがある場合）
  if [client_ip] {{
    geoip {{
      source => "client_ip"
    }}
  }}

  # 日付ベースインデックス名生成
  mutate {{
    add_field => {{ "[@metadata][index_name]" => "day-trade-logs-%{{+YYYY.MM.dd}}" }}
  }}
}}

output {{
  elasticsearch {{
    hosts => {elasticsearch_config.hosts}
    {"user => '" + elasticsearch_config.username + "'" if elasticsearch_config.username else ""}
    {"password => '" + elasticsearch_config.password + "'" if elasticsearch_config.password else ""}
    {"ssl => true" if elasticsearch_config.use_ssl else ""}
    {"ssl_certificate_verification => false" if not elasticsearch_config.verify_certs else ""}
    index => "%{{[@metadata][index_name]}}"
    template_name => "day-trade-logs"
    template_pattern => "day-trade-logs-*"
    template_overwrite => true
    template => "/etc/logstash/templates/day-trade-template.json"
  }}

  # デバッグ用stdout出力
  stdout {{
    codec => rubydebug {{
      metadata => false
    }}
  }}
}}
"""
        return config_template


class KibanaIntegration:
    """Kibana統合システム"""

    def __init__(self, config: KibanaConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def create_index_pattern(self, pattern_name: str, time_field: str = "@timestamp") -> bool:
        """Kibanaインデックスパターン作成"""
        try:
            if not REQUESTS_AVAILABLE:
                self.logger.error("requestsライブラリが利用できません")
                return False

            base_url = (
                f"http{'s' if self.config.use_ssl else ''}://{self.config.host}:{self.config.port}"
            )
            api_url = f"{base_url}/api/saved_objects/index-pattern/{pattern_name}"

            # リクエストヘッダー
            headers = {"Content-Type": "application/json", "kbn-xsrf": "true"}

            # 認証設定
            auth = None
            if self.config.username and self.config.password:
                auth = (self.config.username, self.config.password)

            # インデックスパターン作成ペイロード
            payload = {"attributes": {"title": pattern_name, "timeFieldName": time_field}}

            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                auth=auth,
                timeout=30,
                verify=self.config.use_ssl,
            )

            if response.status_code in [200, 201]:
                self.logger.info(f"Kibanaインデックスパターン作成成功: {pattern_name}")
                return True
            else:
                self.logger.error(
                    f"Kibanaインデックスパターン作成失敗: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Kibanaインデックスパターン作成エラー: {e}")
            return False

    async def create_dashboard(
        self, dashboard_name: str, visualizations: List[Dict[str, Any]]
    ) -> bool:
        """Kibanaダッシュボード作成"""
        try:
            if not REQUESTS_AVAILABLE:
                return False

            base_url = (
                f"http{'s' if self.config.use_ssl else ''}://{self.config.host}:{self.config.port}"
            )

            # 認証ヘッダー
            headers = {"Content-Type": "application/json", "kbn-xsrf": "true"}

            auth = None
            if self.config.username and self.config.password:
                auth = (self.config.username, self.config.password)

            # ダッシュボード作成
            dashboard_payload = {
                "attributes": {
                    "title": dashboard_name,
                    "type": "dashboard",
                    "description": f"Automated dashboard for {dashboard_name}",
                    "panelsJSON": json.dumps(visualizations),
                    "version": 1,
                }
            }

            api_url = (
                f"{base_url}/api/saved_objects/dashboard/{dashboard_name.lower().replace(' ', '-')}"
            )

            response = requests.post(
                api_url, json=dashboard_payload, headers=headers, auth=auth, timeout=30
            )

            if response.status_code in [200, 201]:
                self.logger.info(f"Kibanaダッシュボード作成成功: {dashboard_name}")
                return True
            else:
                self.logger.error(f"Kibanaダッシュボード作成失敗: {response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"Kibanaダッシュボード作成エラー: {e}")
            return False

    def generate_log_level_visualization(self) -> Dict[str, Any]:
        """ログレベル別可視化設定生成"""
        return {
            "version": "7.10.0",
            "gridData": {"x": 0, "y": 0, "w": 24, "h": 15},
            "panelIndex": "log-levels-pie",
            "type": "visualization",
            "id": "log-levels-distribution",
            "embeddableConfig": {
                "title": "Log Levels Distribution",
                "vis": {
                    "type": "pie",
                    "params": {
                        "addTooltip": True,
                        "addLegend": True,
                        "legendPosition": "right",
                    },
                },
            },
        }

    def generate_component_visualization(self) -> Dict[str, Any]:
        """コンポーネント別可視化設定生成"""
        return {
            "version": "7.10.0",
            "gridData": {"x": 24, "y": 0, "w": 24, "h": 15},
            "panelIndex": "components-bar",
            "type": "visualization",
            "id": "components-activity",
            "embeddableConfig": {
                "title": "Components Activity",
                "vis": {
                    "type": "histogram",
                    "params": {
                        "addTooltip": True,
                        "addLegend": False,
                        "mode": "stacked",
                    },
                },
            },
        }


class ELKStackIntegration:
    """ELK Stack統合管理システム"""

    def __init__(
        self,
        elasticsearch_config: ElasticSearchConfig,
        logstash_config: LogstashConfig,
        kibana_config: KibanaConfig,
    ):
        self.elasticsearch = ElasticsearchManager(elasticsearch_config)
        self.logstash = LogstashIntegration(logstash_config)
        self.kibana = KibanaIntegration(kibana_config)
        self.logger = logging.getLogger(__name__)

    async def initialize_elk_stack(self) -> bool:
        """ELK Stack初期化"""
        try:
            self.logger.info("ELK Stack初期化開始")

            # Elasticsearch接続
            if not await self.elasticsearch.connect():
                self.logger.error("Elasticsearch接続失敗")
                return False

            # インデックステンプレート作成
            log_template = ELKIndexTemplate(
                name="day-trade-logs",
                pattern="day-trade-logs-*",
                settings={
                    "number_of_shards": 2,
                    "number_of_replicas": 1,
                    "index.refresh_interval": "5s",
                    "index.codec": "best_compression",
                },
                mappings={
                    "properties": {
                        "@timestamp": {"type": "date"},
                        "log_id": {"type": "keyword"},
                        "level": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "component": {"type": "keyword"},
                        "message": {"type": "text", "analyzer": "standard"},
                        "structured_data": {"type": "object"},
                        "tags": {"type": "keyword"},
                        "trace_id": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                        "session_id": {"type": "keyword"},
                        "parsed_at": {"type": "date"},
                        "host": {"type": "keyword"},
                        "environment": {"type": "keyword"},
                    }
                },
            )

            if not await self.elasticsearch.create_index_template(log_template):
                self.logger.warning("インデックステンプレート作成失敗")

            # Kibanaインデックスパターン作成
            await asyncio.sleep(2)  # Elasticsearchの準備待ち

            if not await self.kibana.create_index_pattern("day-trade-logs-*"):
                self.logger.warning("Kibanaインデックスパターン作成失敗")

            # Kibanaダッシュボード作成
            dashboard_visualizations = [
                self.kibana.generate_log_level_visualization(),
                self.kibana.generate_component_visualization(),
            ]

            if not await self.kibana.create_dashboard(
                "Day Trade System Logs", dashboard_visualizations
            ):
                self.logger.warning("Kibanaダッシュボード作成失敗")

            self.logger.info("ELK Stack初期化完了")
            return True

        except Exception as e:
            self.logger.error(f"ELK Stack初期化エラー: {e}")
            return False

    async def process_log_entry(self, log_entry: LogEntry) -> bool:
        """ログエントリ処理（ELK Stackに送信）"""
        try:
            # 現在の日付ベースのインデックス名
            index_name = f"day-trade-logs-{datetime.utcnow().strftime('%Y.%m.%d')}"

            # 並列処理でElasticsearchとLogstashに送信
            tasks = []

            # Elasticsearchに直接インデックス
            if self.elasticsearch.connection_status == ElasticSearchConnectionStatus.CONNECTED:
                tasks.append(self.elasticsearch.index_log_entry(log_entry, index_name))

            # Logstashに送信（必要に応じて）
            tasks.append(self.logstash.send_log_to_logstash(log_entry))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success_count = sum(1 for result in results if isinstance(result, bool) and result)

                return success_count > 0
            else:
                return False

        except Exception as e:
            self.logger.error(f"ログエントリ処理エラー: {e}")
            return False

    async def bulk_process_logs(self, log_entries: List[LogEntry]) -> int:
        """ログエントリ一括処理"""
        try:
            if not log_entries:
                return 0

            # 日付ベースのインデックス名
            index_name = f"day-trade-logs-{datetime.utcnow().strftime('%Y.%m.%d')}"

            # Elasticsearchに一括インデックス
            success_count = await self.elasticsearch.bulk_index_logs(log_entries, index_name)

            # 統計ログ出力
            if success_count > 0:
                self.logger.info(f"ELK Stack一括処理完了: {success_count}/{len(log_entries)}件")

            return success_count

        except Exception as e:
            self.logger.error(f"一括ログ処理エラー: {e}")
            return 0

    async def search_logs_with_elk(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        log_levels: Optional[List[str]] = None,
        components: Optional[List[str]] = None,
        size: int = 100,
    ) -> Dict[str, Any]:
        """ELK Stack経由でのログ検索"""
        try:
            # Elasticsearchクエリ構築
            es_query = {
                "query": {"bool": {"must": [], "filter": []}},
                "sort": [{"@timestamp": {"order": "desc"}}],
            }

            # テキスト検索
            if query:
                es_query["query"]["bool"]["must"].append(
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["message", "component", "structured_data.*"],
                            "type": "best_fields",
                        }
                    }
                )

            # 時間範囲フィルター
            if start_time or end_time:
                time_filter = {"range": {"@timestamp": {}}}
                if start_time:
                    time_filter["range"]["@timestamp"]["gte"] = start_time.isoformat()
                if end_time:
                    time_filter["range"]["@timestamp"]["lte"] = end_time.isoformat()
                es_query["query"]["bool"]["filter"].append(time_filter)

            # ログレベルフィルター
            if log_levels:
                es_query["query"]["bool"]["filter"].append({"terms": {"level": log_levels}})

            # コンポーネントフィルター
            if components:
                es_query["query"]["bool"]["filter"].append({"terms": {"component": components}})

            # 検索実行
            index_pattern = "day-trade-logs-*"
            response = await self.elasticsearch.search_logs(index_pattern, es_query, size=size)

            return response

        except Exception as e:
            self.logger.error(f"ELK検索エラー: {e}")
            return {"hits": {"total": {"value": 0}, "hits": []}}

    async def get_elk_stack_health(self) -> Dict[str, Any]:
        """ELK Stack健全性チェック"""
        try:
            health_status = {
                "elasticsearch": {"status": "unknown", "details": {}},
                "logstash": {"status": "unknown", "details": {}},
                "kibana": {"status": "unknown", "details": {}},
                "overall_status": "unknown",
            }

            # Elasticsearch健全性
            try:
                es_health = await self.elasticsearch.get_cluster_health()
                if es_health:
                    health_status["elasticsearch"]["status"] = es_health.get("status", "unknown")
                    health_status["elasticsearch"]["details"] = es_health
                else:
                    health_status["elasticsearch"]["status"] = "disconnected"
            except Exception as e:
                health_status["elasticsearch"]["status"] = "error"
                health_status["elasticsearch"]["error"] = str(e)

            # Logstash健全性（簡易チェック）
            try:
                # TCP接続テスト
                import socket

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(5)
                    result = sock.connect_ex((self.logstash.config.host, self.logstash.config.port))
                    if result == 0:
                        health_status["logstash"]["status"] = "connected"
                    else:
                        health_status["logstash"]["status"] = "disconnected"
            except Exception as e:
                health_status["logstash"]["status"] = "error"
                health_status["logstash"]["error"] = str(e)

            # Kibana健全性
            try:
                if REQUESTS_AVAILABLE:
                    base_url = f"http{'s' if self.kibana.config.use_ssl else ''}://{self.kibana.config.host}:{self.kibana.config.port}"
                    response = requests.get(f"{base_url}/api/status", timeout=10)
                    if response.status_code == 200:
                        health_status["kibana"]["status"] = "connected"
                        health_status["kibana"]["details"] = response.json()
                    else:
                        health_status["kibana"]["status"] = "error"
                else:
                    health_status["kibana"]["status"] = "no_requests_library"
            except Exception as e:
                health_status["kibana"]["status"] = "error"
                health_status["kibana"]["error"] = str(e)

            # 総合ステータス判定
            statuses = [
                health_status["elasticsearch"]["status"],
                health_status["logstash"]["status"],
                health_status["kibana"]["status"],
            ]

            if all(status == "connected" for status in statuses):
                health_status["overall_status"] = "healthy"
            elif any(status in ["connected", "green", "yellow"] for status in statuses):
                health_status["overall_status"] = "partial"
            else:
                health_status["overall_status"] = "unhealthy"

            return health_status

        except Exception as e:
            self.logger.error(f"ELK健全性チェックエラー: {e}")
            return {"overall_status": "error", "error": str(e)}


# Factory function
def create_elk_stack_integration(
    elasticsearch_hosts: List[str] = None,
    elasticsearch_user: str = None,
    elasticsearch_password: str = None,
    logstash_host: str = "localhost",
    logstash_port: int = 5044,
    kibana_host: str = "localhost",
    kibana_port: int = 5601,
) -> ELKStackIntegration:
    """ELK Stack統合システム作成"""

    if elasticsearch_hosts is None:
        elasticsearch_hosts = ["localhost:9200"]

    es_config = ElasticSearchConfig(
        hosts=elasticsearch_hosts,
        username=elasticsearch_user,
        password=elasticsearch_password,
    )

    logstash_config = LogstashConfig(host=logstash_host, port=logstash_port)

    kibana_config = KibanaConfig(host=kibana_host, port=kibana_port)

    return ELKStackIntegration(es_config, logstash_config, kibana_config)


if __name__ == "__main__":
    # テスト実行
    async def test_elk_integration():
        print("=== ELK Stack統合システムテスト ===")

        try:
            # ELK Stack統合システム初期化
            elk_system = create_elk_stack_integration()

            print("\n1. ELK Stack統合システム初期化完了")

            # ELK Stack健全性チェック
            print("\n2. ELK Stack健全性チェック...")
            health_status = await elk_system.get_elk_stack_health()

            print(f"   総合ステータス: {health_status.get('overall_status')}")
            print(f"   Elasticsearch: {health_status.get('elasticsearch', {}).get('status')}")
            print(f"   Logstash: {health_status.get('logstash', {}).get('status')}")
            print(f"   Kibana: {health_status.get('kibana', {}).get('status')}")

            # モックログエントリでのテスト（Elasticsearchが利用可能な場合のみ）
            if MONITORING_AVAILABLE:
                print("\n3. ログエントリ処理テスト...")

                # テストログエントリ作成
                test_log = LogEntry(
                    id=f"test_log_{int(time.time())}",
                    timestamp=datetime.utcnow(),
                    level=LogLevel.INFO,
                    source=LogSource.APPLICATION,
                    component="elk_test",
                    message="ELK統合テストメッセージ",
                    structured_data={"test": True, "integration": "elk"},
                    tags=["test", "elk", "integration"],
                )

                # ログ処理
                success = await elk_system.process_log_entry(test_log)
                print(f"   ログエントリ処理: {'成功' if success else '失敗'}")

            # Logstash設定生成テスト
            print("\n4. Logstash設定生成テスト...")
            es_config = ElasticSearchConfig()
            logstash_config = elk_system.logstash.generate_logstash_config(es_config)

            config_lines = len(logstash_config.split("\n"))
            print(f"   生成された設定: {config_lines}行")
            print("   入力プラグイン: beats, tcp, http")
            print("   出力プラグイン: elasticsearch, stdout")

            # ELK検索テスト（モック）
            print("\n5. ELK検索テスト...")
            search_result = await elk_system.search_logs_with_elk(
                query="test",
                start_time=datetime.utcnow() - timedelta(hours=1),
                end_time=datetime.utcnow(),
                size=10,
            )

            total_hits = search_result.get("hits", {}).get("total", {}).get("value", 0)
            print(f"   検索結果: {total_hits}件")

            print("\n✅ ELK Stack統合システムテスト完了")

        except Exception as e:
            print(f"❌ テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_elk_integration())
