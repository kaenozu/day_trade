# Day Trade Development Automation
.PHONY: help install dev test lint format clean docker-build docker-up docker-down ci clean-db

# デフォルトターゲット
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# 環境セットアップ
install: ## Install dependencies
	@echo "🔧 Installing dependencies..."
	pip install --upgrade pip
	pip install -e .
	pip install -r requirements-dev.txt || echo "requirements-dev.txt not found, skipping"
	pre-commit install

dev: ## Setup development environment
	@echo "🚀 Setting up development environment..."
	$(MAKE) install
	pre-commit install --hook-type commit-msg
	@echo "✅ Development environment ready!"

# コード品質
lint: ## Run linters
	@echo "🔍 Running linters..."
	ruff check . --fix
	ruff format .

format: ## Format code
	@echo "✨ Formatting code..."
	ruff format .

pre-commit: ## Run pre-commit hooks
	@echo "🪝 Running pre-commit hooks..."
	pre-commit run --all-files

# データベースクリーンアップ
clean-db: ## Clean up database files
	@echo "🧹 Cleaning up database files..."
	-del day_trade.db day_trade.db-shm day_trade.db-wal 2>NUL || true

# テスト
test: clean-db ## Run tests
	@echo "🧪 Running tests..."
	pytest tests/ -v --cov=src/day_trade --cov-report=html --cov-report=term

test-fast: clean-db ## Run tests (fast, no coverage)
	@echo "⚡ Running fast tests..."
	pytest tests/ -x --tb=short

test-unit: clean-db ## Run unit tests only
	@echo "🧪 Running unit tests..."
	pytest tests/ -v --ignore=tests/integration/

test-integration: clean-db ## Run integration tests only
	@echo "🔗 Running integration tests..."
	pytest tests/integration/ -v

# カバレッジ
coverage: ## Generate comprehensive coverage report
	@echo "📊 Generating coverage report..."
	python scripts/coverage_report.py

coverage-goals: ## Check coverage goals and generate progress report
	@echo "🎯 Checking coverage goals..."
	python scripts/coverage_goals.py

coverage-html: ## Generate HTML coverage report
	@echo "🌐 Generating HTML coverage report..."
	pytest tests/ --cov=src/day_trade --cov-report=html --cov-report=term-missing
	@echo "📄 HTML report available at: htmlcov/index.html"

coverage-xml: ## Generate XML coverage report for CI
	@echo "📄 Generating XML coverage report..."
	pytest tests/ --cov=src/day_trade --cov-report=xml --cov-report=term-missing

# Docker
docker-build: ## Build Docker images
	@echo "🐳 Building Docker images..."
	docker-compose build


docker-up: ## Start Docker services
	@echo "🚀 Starting Docker services..."
	docker-compose up -d


docker-down: ## Stop Docker services
	@echo "🛑 Stopping Docker services..."
	docker-compose down


docker-test: ## Run tests in Docker
	@echo "🧪 Running tests in Docker..."
	docker-compose run --rm test


docker-quality: ## Run quality checks in Docker
	@echo "🔍 Running quality checks in Docker..."
	docker-compose run --rm quality

# CI/CD
ci: ## Run full CI pipeline locally
	@echo "🔄 Running full CI pipeline..."
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) build
	@echo "✅ CI pipeline completed!"

ci-fast: ## Run fast CI checks
	@echo "⚡ Running fast CI checks..."
	$(MAKE) pre-commit
	$(MAKE) test-fast
	@echo "✅ Fast CI completed!"

# ビルド
build: ## Build package
	@echo "📦 Building package..."
	python -m build
	python -m twine check dist/*

# クリーンアップ
clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-docker: ## Clean Docker resources
	@echo "🧹 Cleaning Docker resources..."
	docker-compose down -v
	docker system prune -f

# デバッグ
debug: ## Show environment info
	@echo "🔍 Environment Information:"
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "Current directory: $(shell pwd)"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Git status: $(shell git status --porcelain 2>/dev/null | wc -l || echo 'N/A') files changed"

# パフォーマンス
profile: ## Profile application performance
	@echo "📊 Profiling application..."
	python -m cProfile -o profile_output.prof -m day_trade --help
	@echo "Profile saved to profile_output.prof"

benchmark: ## Run performance benchmarks
	@echo "⚡ Running benchmarks..."
	python -m pytest tests/ --benchmark-only || echo "No benchmarks found"

# 依存関係管理
deps-update: ## Update dependencies
	@echo "📦 Updating dependencies..."
	pip install --upgrade pip
	python scripts/dependency_manager.py update --dry-run
	pre-commit autoupdate


deps-check: ## Check for security vulnerabilities
	@echo "🔒 Checking dependencies for vulnerabilities..."
	python scripts/dependency_manager.py check
	safety check || echo "Safety not installed"
	pip-audit || echo "pip-audit not installed"


deps-report: ## Generate dependency report
	@echo "📊 Generating dependency report..."
	python scripts/dependency_manager.py report


deps-sync: ## Sync requirements files
	@echo "🔄 Syncing requirements files..."
	python scripts/dependency_manager.py sync


deps-unused: ## Check for unused dependencies
	@echo "🔍 Checking for unused dependencies..."
	deptry . --extend-exclude "tests" --ignore-notebooks || echo "deptry not installed"


deps-validate: ## Validate dependency configuration
	@echo "✅ Validating dependency configuration..."
	pip check
	pip-check-reqs . || echo "pip-check-reqs not installed"


deps-audit: ## Comprehensive dependency audit
	@echo "🔒 Running comprehensive dependency audit..."
	safety check || echo "safety not installed"
	pip-audit || echo "pip-audit not installed"
	bandit -r src/ -f json -o reports/dependencies/security_audit.json || echo "bandit not installed"

# リリース
release-patch: ## Create patch release
	@echo "🏷️ Creating patch release..."
	bump2version patch
	git push --tags

release-minor: ## Create minor release
	@echo "🏷️ Creating minor release..."
	bump2version minor
	git push --tags

release-major: ## Create major release
	@echo "🏷️ Creating major release..."
	bump2version major
	git push --tags

# 開発サーバー
serve: ## Start development server
	@echo "🚀 Starting development server..."
	python -m day_trade --interactive

# ドキュメント
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	sphinx-build -b html docs/ docs/_build/html || echo "Sphinx not configured"

# ヘルパー
watch: ## Watch for file changes and run tests
	@echo "👀 Watching for changes..."
	watchexec --exts py "make test-fast" || echo "watchexec not installed"
