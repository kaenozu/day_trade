# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a sophisticated day trading support system built in Python, featuring ensemble trading strategies, automated analysis, and comprehensive market data processing capabilities. It's designed as both a CLI application and an interactive system for stock market analysis and trading decision support.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies (recommended)
pip install -e .[dev]

# Alternative installation
pip install -r requirements.txt -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

### Development Workflow
```bash
# Start development
make dev                    # Setup complete development environment

# Code quality checks
make lint                  # Run ruff linter with auto-fix
make format               # Format code with ruff
make pre-commit          # Run all pre-commit hooks

# Testing
make test                # Run tests with coverage
make test-fast           # Quick tests without coverage
make test-unit           # Unit tests only
make test-integration    # Integration tests only

# Coverage analysis
make coverage            # Comprehensive coverage report
make coverage-html       # Generate HTML coverage report
make coverage-goals      # Check coverage goals (target: 35%+)

# Build and CI
make build               # Build package
make ci                  # Run full CI pipeline locally
make ci-fast            # Run fast CI checks
```

### Specific Testing Commands
```bash
# Run specific test files
pytest tests/test_ensemble.py -v
pytest tests/integration/test_end_to_end_workflow.py

# Run with coverage targeting src/day_trade
pytest tests/ --cov=src/day_trade --cov-report=html --cov-report=term

# Coverage with specific thresholds
pytest --cov=src/day_trade --cov-fail-under=35
```

### Database Operations
```bash
# Initialize database
python -m day_trade.models.database --init

# Run Alembic migrations
alembic upgrade head

# Reset database (development)
python -m day_trade.models.database --reset
```

### Application Usage
```bash
# Interactive CLI mode
python -m day_trade.cli.main
# or simply: daytrade

# Specific analysis commands
daytrade analyze 7203                                    # Analyze specific stock
daytrade watchlist add 7203 6758 4755                   # Manage watchlist
daytrade backtest --start-date 2024-01-01 --end-date 2024-12-31
daytrade screen --strategy momentum --min-volume 1000000
```

## Architecture and Code Structure

### High-Level Architecture
This is a layered architecture with clean separation between:
- **Presentation Layer**: CLI interfaces (main.py, interactive.py)
- **Business Logic**: Core analysis engines (analysis/, automation/)
- **Data Layer**: Models, database operations, external API integration
- **Infrastructure**: Utilities, caching, logging, configuration

### Key Modules and Their Purpose

#### Core Analysis Engine (`src/day_trade/analysis/`)
- **ensemble.py**: Multi-strategy ensemble trading system - the heart of the application
- **enhanced_ensemble.py**: Advanced ensemble features with weighted voting
- **indicators.py**: Technical analysis indicators (RSI, MACD, Bollinger Bands, etc.)
- **patterns.py**: Chart pattern recognition algorithms
- **signals.py**: Buy/sell signal generation and evaluation
- **backtest.py**: Strategy backtesting framework
- **screener.py**: Stock screening and filtering
- **ml_models.py**: Machine learning models for prediction

#### Automation Framework (`src/day_trade/automation/`)
- **orchestrator.py**: Main automation orchestrator
- **optimized_orchestrator.py**: Performance-optimized automation engine
- **auto_optimizer.py**: Automatic parameter optimization

#### Data Management (`src/day_trade/data/`)
- **stock_fetcher.py**: Primary data fetching from yfinance and other APIs
- **enhanced_stock_fetcher.py**: Enhanced data fetching with resilience features
- **stock_master.py**: Master data management for stock symbols and metadata

#### Database Layer (`src/day_trade/models/`)
- **database.py**: Core database operations with SQLAlchemy
- **optimized_database.py**: Performance-optimized database operations
- **enums.py**: Database enumerations and type definitions
- **stock.py**: Stock data models and relationships

#### CLI and User Interface (`src/day_trade/cli/`)
- **main.py**: Main CLI entry point with Click framework
- **interactive.py**: Interactive mode with rich/prompt_toolkit
- **enhanced_interactive.py**: Advanced interactive features

#### Infrastructure (`src/day_trade/utils/`)
- **logging_config.py**: Structured logging configuration using structlog
- **cache_utils.py**: Caching layer for performance optimization
- **api_resilience.py**: API retry and failure handling mechanisms
- **transaction_manager.py**: Database transaction management
- **formatters.py**: Output formatting utilities
- **exceptions.py**: Custom exception hierarchy

### Data Flow and Integration Points
1. **Data Ingestion**: yfinance API → stock_fetcher.py → database models
2. **Analysis Pipeline**: Raw data → technical indicators → ensemble strategies → signals
3. **Decision Making**: Ensemble signals → risk management → trade recommendations
4. **User Interface**: CLI commands → business logic → formatted output

### Configuration Management
- **pyproject.toml**: Main project configuration with tool settings
- **config/settings.json**: Application runtime configuration
- **config/indicators_config.json**: Technical indicator parameters
- **Makefile**: Development automation and task management
- **alembic.ini**: Database migration configuration

### Testing Strategy
- **Unit Tests**: Individual component testing in tests/
- **Integration Tests**: End-to-end workflow testing in tests/integration/
- **Coverage Target**: Minimum 35% (current), aiming for 80%+
- **Test Configuration**: Comprehensive pytest setup with coverage reporting

### Important Development Notes
- Uses **structured logging** with structlog - check logging_config.py for setup
- **Database transactions** are managed through transaction_manager.py
- **API resilience** is built-in through api_resilience.py with retry mechanisms
- **Caching layer** for performance - see cache_utils.py
- **Type checking** with mypy, **linting** with ruff
- **Pre-commit hooks** for code quality enforcement

### Performance Considerations
- Heavy use of pandas/numpy for data processing
- Database query optimization through optimized_database.py
- Caching strategies for API calls and computed results
- Parallel processing capabilities in automation modules

### Key Design Patterns
- **Factory Pattern**: For creating different types of analyzers and strategies
- **Strategy Pattern**: Ensemble system supports pluggable trading strategies
- **Observer Pattern**: Alert and notification systems
- **Repository Pattern**: Database access abstraction
- **Builder Pattern**: Complex configuration and setup processes

This codebase emphasizes reliability, performance, and extensibility for financial data analysis and automated trading support.