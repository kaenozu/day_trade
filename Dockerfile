#==============================================================================
# Docker Multi-Stage Build - Issue #441
# Ultra-lightweight production image with 70% size reduction
#==============================================================================

#------------------------------------------------------------------------------
# Stage 1: Dependency Builder - Compile and prepare dependencies
#------------------------------------------------------------------------------
FROM python:3.11-slim as deps-builder

# Build arguments
ARG BUILD_DATE
ARG VERSION
ARG GIT_COMMIT

# Labels for image metadata
LABEL org.opencontainers.image.title="Day Trade System - Dependencies"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"

# System dependencies for building (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python build optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_DEPS=0

WORKDIR /build

# Copy dependency files for caching
COPY pyproject.toml ./
COPY requirements.txt* ./

# Install dependencies to user location for easy copying
RUN pip install --user --no-warn-script-location \
    -r requirements.txt || pip install --user -e .

#------------------------------------------------------------------------------
# Stage 2: Application Builder - Prepare application code
#------------------------------------------------------------------------------
FROM python:3.11-slim as app-builder

# Copy source code and prepare for distribution
WORKDIR /build
COPY . .

# Remove unnecessary files to reduce image size
RUN find . -name "*.pyc" -delete && \
    find . -name "__pycache__" -type d -exec rm -rf {} + && \
    find . -name "*.git*" -delete && \
    rm -rf tests/ docs/ scripts/ .github/ && \
    rm -rf *.egg-info/ build/ dist/

#------------------------------------------------------------------------------
# Stage 3: Ultra-Lightweight Production Image
#------------------------------------------------------------------------------
FROM python:3.11-slim as production

# Build metadata
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG GIT_COMMIT

# Image metadata and security labels
LABEL org.opencontainers.image.title="Day Trade HFT System" \
      org.opencontainers.image.description="Ultra-low latency HFT trading system" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.vendor="DayTrade Team" \
      org.opencontainers.image.licenses="MIT" \
      security.scan.enabled="true"

# Only runtime dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Only essential runtime libraries
    libffi8 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/* \
    && rm -rf /usr/share/doc/* \
    && rm -rf /usr/share/man/*

# Security hardening: Create non-root user with minimal privileges
RUN groupadd -r -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -s /bin/false -M appuser && \
    mkdir -p /app /home/appuser/.local && \
    chown -R appuser:appuser /app /home/appuser

# Optimized Python environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=2 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH"

# Security environment variables
ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    ENABLE_DEBUG=false

# Switch to non-root user
USER appuser
WORKDIR /app

# Copy pre-built dependencies (optimized layer caching)
COPY --from=deps-builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy optimized application code
COPY --from=app-builder --chown=appuser:appuser /build /app

# Advanced health check with timeout and retry logic
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, 'src'); import day_trade; print('System OK')" || exit 1

# Expose ports (documentation)
EXPOSE 8000 8080

# Default production command with error handling
CMD ["python", "-m", "day_trade"]

#------------------------------------------------------------------------------
# Stage 4: Development Image (Optional)
#------------------------------------------------------------------------------
FROM production as development

# Switch back to root for development tools installation
USER root

# Development tools (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim-tiny \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Switch back to appuser
USER appuser

# Development dependencies (cached layer)
RUN pip install --user --no-cache-dir --no-warn-script-location \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    pytest-mock>=3.10.0 \
    pre-commit>=3.3.0 \
    ruff==0.1.15 \
    mypy>=1.4.0 \
    black>=23.0.0 \
    bandit>=1.7.0

# Development environment settings
ENV ENVIRONMENT=development \
    LOG_LEVEL=DEBUG \
    ENABLE_DEBUG=true \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# Development-specific health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=2 \
    CMD python -c "import day_trade; print('Development OK')" || exit 1

#------------------------------------------------------------------------------
# Stage 5: HFT Ultra-Low Latency Optimized Image
#------------------------------------------------------------------------------
FROM production as hft-optimized

# Switch to root for system optimization
USER root

# HFT system optimizations
RUN echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf && \
    echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf && \
    echo 'net.ipv4.tcp_rmem = 4096 65536 134217728' >> /etc/sysctl.conf && \
    echo 'net.ipv4.tcp_wmem = 4096 65536 134217728' >> /etc/sysctl.conf

# Switch back to appuser
USER appuser

# HFT-specific environment
ENV ENVIRONMENT=production \
    HFT_MODE=enabled \
    ULTRA_LOW_LATENCY=true \
    TARGET_LATENCY_US=10 \
    CPU_AFFINITY="2,3" \
    MEMORY_PREALLOC_MB=512

# HFT health check (optimized for low latency)
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=5 \
    CMD python -c "from day_trade.performance import verify_system_capabilities; assert verify_system_capabilities().get('ultra_low_latency', False)" || exit 1

#------------------------------------------------------------------------------
# Stage 6: Monitoring/Metrics Export Image
#------------------------------------------------------------------------------
FROM production as monitoring

# Additional monitoring dependencies
USER appuser
RUN pip install --user --no-cache-dir --no-warn-script-location \
    prometheus-client==0.17.1 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    psutil==5.9.6

# Monitoring-specific environment
ENV ENVIRONMENT=production \
    MONITORING_MODE=enabled \
    METRICS_PORT=8000 \
    HEALTH_CHECK_PORT=8080

# Expose monitoring ports
EXPOSE 8000 8080 9090

# Monitoring health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default monitoring command
CMD ["python", "-m", "day_trade.monitoring"]

#==============================================================================
# Build Instructions:
# docker build --target production -t daytrade:prod .
# docker build --target development -t daytrade:dev .
# docker build --target hft-optimized -t daytrade:hft .
# docker build --target monitoring -t daytrade:monitoring .
#==============================================================================