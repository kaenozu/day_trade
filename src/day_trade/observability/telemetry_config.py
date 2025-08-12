"""
APM・オブザーバビリティ統合基盤 - Issue #442
OpenTelemetry 設定と分散トレーシング初期化

このモジュールは以下を提供します:
- OpenTelemetry 自動計装設定
- カスタムトレーサー・メーター初期化
- 構造化ログ設定
- HFT対応の低オーバーヘッド監視
"""

import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Optional

import structlog

# OpenTelemetry imports
from opentelemetry import metrics, trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.propagators.composite import CompositeHTTPPropagator
from opentelemetry.propagators.jaeger import JaegerPropagator
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv.resource import ResourceAttributes


class TelemetryConfig:
    """
    APM・オブザーバビリティ統合基盤設定クラス

    Features:
    - 分散トレーシング (Jaeger/OpenTelemetry)
    - メトリクス収集 (Prometheus)
    - 構造化ログ (JSON)
    - HFT対応低レイテンシ監視
    """

    def __init__(
        self,
        service_name: str = "day-trade-app",
        service_version: str = "1.0.0",
        environment: str = "production",
        otlp_endpoint: Optional[str] = None,
        jaeger_endpoint: Optional[str] = None,
        enable_console_export: bool = False,
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.otlp_endpoint = otlp_endpoint or os.getenv(
            "OTLP_ENDPOINT", "http://otel-collector:4317"
        )
        self.jaeger_endpoint = jaeger_endpoint or os.getenv(
            "JAEGER_ENDPOINT", "http://jaeger-all-in-one:14268/api/traces"
        )
        self.enable_console_export = (
            enable_console_export
            or os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true"
        )

        # HFT最適化設定
        self.hft_mode = os.getenv("HFT_MODE", "false").lower() == "true"
        self.batch_timeout = 100 if self.hft_mode else 1000  # ms
        self.max_queue_size = 512 if self.hft_mode else 2048
        self.max_export_batch_size = 256 if self.hft_mode else 512

        # 初期化完了フラグ
        self._initialized = False
        self._tracer: Optional[trace.Tracer] = None
        self._meter: Optional[metrics.Meter] = None

    def initialize_telemetry(self) -> None:
        """テレメトリシステムの初期化"""
        if self._initialized:
            return

        try:
            # リソース設定
            resource = self._create_resource()

            # トレーシング初期化
            self._initialize_tracing(resource)

            # メトリクス初期化
            self._initialize_metrics(resource)

            # 構造化ログ初期化
            self._initialize_structured_logging()

            # 自動計装設定
            self._setup_auto_instrumentation()

            # プロパゲーター設定
            self._setup_propagators()

            self._initialized = True

            # 初期化完了ログ
            logger = structlog.get_logger()
            logger.info(
                "Telemetry initialization completed",
                service_name=self.service_name,
                environment=self.environment,
                hft_mode=self.hft_mode,
                otlp_endpoint=self.otlp_endpoint,
            )

        except Exception as e:
            logging.error(f"Failed to initialize telemetry: {e}")
            raise

    def _create_resource(self) -> Resource:
        """OpenTelemetryリソース作成"""
        return Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: self.service_name,
                ResourceAttributes.SERVICE_VERSION: self.service_version,
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.environment,
                ResourceAttributes.SERVICE_NAMESPACE: "day-trade",
                ResourceAttributes.SERVICE_INSTANCE_ID: os.getenv(
                    "HOSTNAME", "unknown"
                ),
                "service.language": "python",
                "service.runtime": f"python-{sys.version_info.major}.{sys.version_info.minor}",
            }
        )

    def _initialize_tracing(self, resource: Resource) -> None:
        """分散トレーシング初期化"""
        # TracerProvider設定
        tracer_provider = TracerProvider(resource=resource)

        # エクスポーター設定
        exporters = []

        # OTLP エクスポーター
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.otlp_endpoint,
                insecure=True,  # 本番環境では適切なTLS設定を行う
            )
            exporters.append(otlp_exporter)
        except Exception as e:
            logging.warning(f"OTLP span exporter initialization failed: {e}")

        # Jaeger エクスポーター（フォールバック）
        try:
            jaeger_exporter = JaegerExporter(
                agent_host_name="jaeger-all-in-one",
                agent_port=6831,
            )
            exporters.append(jaeger_exporter)
        except Exception as e:
            logging.warning(f"Jaeger span exporter initialization failed: {e}")

        # コンソールエクスポーター（デバッグ用）
        if self.enable_console_export:
            exporters.append(ConsoleSpanExporter())

        # BatchSpanProcessor設定（HFT最適化）
        for exporter in exporters:
            processor = BatchSpanProcessor(
                exporter,
                max_queue_size=self.max_queue_size,
                schedule_delay_millis=self.batch_timeout,
                export_timeout_millis=self.batch_timeout * 2,
                max_export_batch_size=self.max_export_batch_size,
            )
            tracer_provider.add_span_processor(processor)

        # グローバルTracerProvider設定
        trace.set_tracer_provider(tracer_provider)

        # トレーサー取得
        self._tracer = trace.get_tracer(
            __name__,
            self.service_version,
        )

    def _initialize_metrics(self, resource: Resource) -> None:
        """メトリクス収集初期化"""
        # MetricReader設定
        readers = []

        # OTLP メトリクスエクスポーター
        try:
            otlp_metric_exporter = OTLPMetricExporter(
                endpoint=self.otlp_endpoint, insecure=True
            )
            otlp_reader = PeriodicExportingMetricReader(
                otlp_metric_exporter,
                export_interval_millis=(
                    5000 if self.hft_mode else 10000
                ),  # HFT用高頻度エクスポート
            )
            readers.append(otlp_reader)
        except Exception as e:
            logging.warning(f"OTLP metric reader initialization failed: {e}")

        # コンソールメトリクスエクスポーター（デバッグ用）
        if self.enable_console_export:
            console_reader = PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=30000,
            )
            readers.append(console_reader)

        # MeterProvider設定
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=readers,
        )

        # グローバルMeterProvider設定
        metrics.set_meter_provider(meter_provider)

        # メーター取得
        self._meter = metrics.get_meter(
            __name__,
            self.service_version,
        )

    def _initialize_structured_logging(self) -> None:
        """構造化ログ初期化"""

        def add_trace_context(logger, method_name, event_dict):
            """トレースコンテキストをログに追加"""
            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                ctx = span.get_span_context()
                event_dict["trace_id"] = format(ctx.trace_id, "032x")
                event_dict["span_id"] = format(ctx.span_id, "016x")
                event_dict["trace_flags"] = format(ctx.trace_flags, "02x")
            return event_dict

        def add_service_context(logger, method_name, event_dict):
            """サービスコンテキストをログに追加"""
            event_dict["service.name"] = self.service_name
            event_dict["service.version"] = self.service_version
            event_dict["environment"] = self.environment
            return event_dict

        # structlog設定
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                add_service_context,
                add_trace_context,
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # 標準ログレベル設定
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, log_level),
        )

    def _setup_auto_instrumentation(self) -> None:
        """自動計装設定"""
        try:
            # FastAPI自動計装（存在する場合）
            try:
                FastAPIInstrumentor().instrument()
            except Exception:
                pass  # FastAPIが利用されていない場合はスキップ

            # SQLAlchemy自動計装
            SQLAlchemyInstrumentor().instrument()

            # Redis自動計装
            RedisInstrumentor().instrument()

            # Requests自動計装
            RequestsInstrumentor().instrument()

            # PostgreSQL自動計装
            try:
                Psycopg2Instrumentor().instrument()
            except Exception:
                pass  # psycopg2が利用されていない場合はスキップ

        except Exception as e:
            logging.warning(f"Auto instrumentation setup failed: {e}")

    def _setup_propagators(self) -> None:
        """プロパゲーター設定"""
        # 複数のプロパゲーター形式をサポート
        set_global_textmap(
            CompositeHTTPPropagator(
                [
                    B3MultiFormat(),
                    JaegerPropagator(),
                ]
            )
        )

    @property
    def tracer(self) -> trace.Tracer:
        """トレーサー取得"""
        if not self._initialized:
            raise RuntimeError(
                "Telemetry not initialized. Call initialize_telemetry() first."
            )
        return self._tracer

    @property
    def meter(self) -> metrics.Meter:
        """メーター取得"""
        if not self._initialized:
            raise RuntimeError(
                "Telemetry not initialized. Call initialize_telemetry() first."
            )
        return self._meter

    @contextmanager
    def trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """カスタムスパン作成"""
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

    def record_exception(self, exception: Exception, span: Optional[trace.Span] = None):
        """例外記録"""
        current_span = span or trace.get_current_span()
        if current_span:
            current_span.record_exception(exception)
            current_span.set_status(
                trace.Status(trace.StatusCode.ERROR, str(exception))
            )


# グローバルテレメトリ設定インスタンス
telemetry_config = TelemetryConfig()


def initialize_observability(
    service_name: str = "day-trade-app", environment: str = None
) -> TelemetryConfig:
    """
    オブザーバビリティ初期化

    Args:
        service_name: サービス名
        environment: 実行環境

    Returns:
        TelemetryConfig: 設定済みテレメトリ設定
    """
    global telemetry_config

    if environment is None:
        environment = os.getenv("ENVIRONMENT", "production")

    telemetry_config = TelemetryConfig(
        service_name=service_name, environment=environment
    )

    telemetry_config.initialize_telemetry()

    return telemetry_config


def get_tracer() -> trace.Tracer:
    """グローバルトレーサー取得"""
    return telemetry_config.tracer


def get_meter() -> metrics.Meter:
    """グローバルメーター取得"""
    return telemetry_config.meter


def get_logger():
    """構造化ログ取得"""
    return structlog.get_logger()
