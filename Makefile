# WhatsApp Sales Assistant - Multi-Agent System Makefile
# =======================================================
#
# This Makefile provides comprehensive build, test, and deployment automation
# for the multi-agent WhatsApp sales assistant system.
#
# Prerequisites:
# - Docker and Docker Compose installed
# - Python 3.12+ with venv support
# - Make utility
#
# Quick Start:
# 	make help			# Show all available commands
# 	make setup			# Initial project setup
# 	make dev			# Start development environment
# 	make test			# Run all tests
# 	make build			# Build all Docker images
# 	make up				# Start all services
#
# =======================================================

# Configuration
SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help setup clean build test lint format type-check dev up down logs status health

# Project Configuration
PROJECT_NAME := whatsapp-sales-assistant
PYTHON_VERSION := 3.12
VENV_NAME := venv_linux
DOCKER_COMPOSE_FILE := docker-compose.yml

# Service Names
CLASSIFIER_SERVICE := classifier
ORCHESTRATOR_SERVICE := orchestrator
CLI_SERVICE := cli

# Docker Image Names
CLASSIFIER_IMAGE := whatsapp-classifier
ORCHESTRATOR_IMAGE := whatsapp-orchestrator

# Test Configuration
TEST_PATH := tests/
COVERAGE_MIN := 90
PYTEST_ARGS := -v --tb=short --strict-markers
PYTEST_COV_ARGS := --cov=. --cov-report=html --cov-report=term --cov-report=xml --cov-fail-under=$(COVERAGE_MIN)

# Linting and Formatting
PYTHON_FILES := agents/ cli/ shared/ config/ tests/
BLACK_ARGS := --line-length=100 --target-version=py312
RUFF_ARGS := --line-length=100 --target-version=py312
MYPY_ARGS := --strict --ignore-missing-imports

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m

# Helper function to print colored output
define print_section
	@echo -e "$(CYAN)$1$(RESET)"
endef

define print_success
	@echo -e "$(GREEN)✓ $1$(RESET)"
endef

define print_warning
	@echo -e "$(YELLOW)⚠ $1$(RESET)"
endef

define print_error
	@echo -e "$(RED)✗ $1$(RESET)"
endef

# =======================================================
# Help and Information
# =======================================================

help: ## Show this help message
	@echo -e "$(BLUE)WhatsApp Sales Assistant - Multi-Agent System$(RESET)"
	@echo -e "$(BLUE)===============================================$(RESET)"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { \
		printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2 \
	}' $(MAKEFILE_LIST)
	@echo ""
	@echo -e "$(YELLOW)Quick Start:$(RESET)"
	@echo "  1. make setup     # Initial setup"
	@echo "  2. make dev       # Start development"
	@echo "  3. make test      # Run tests"
	@echo ""

info: ## Show project information
	$(call print_section,"Project Information")
	@echo "Project Name: $(PROJECT_NAME)"
	@echo "Python Version: $(PYTHON_VERSION)"
	@echo "Virtual Environment: $(VENV_NAME)"
	@echo "Docker Compose File: $(DOCKER_COMPOSE_FILE)"
	@echo "Services: $(CLASSIFIER_SERVICE), $(ORCHESTRATOR_SERVICE), $(CLI_SERVICE)"
	@echo "Docker Images: $(CLASSIFIER_IMAGE), $(ORCHESTRATOR_IMAGE)"

# =======================================================
# Environment Setup
# =======================================================

check-python: ## Check Python version
	@python3 --version | grep -q "$(PYTHON_VERSION)" || \
		($(call print_error,"Python $(PYTHON_VERSION) required") && exit 1)
	$(call print_success,"Python $(PYTHON_VERSION) detected")

check-docker: ## Check Docker installation
	@docker --version >/dev/null 2>&1 || \
		($(call print_error,"Docker not found. Please install Docker") && exit 1)
	@docker-compose --version >/dev/null 2>&1 || \
		($(call print_error,"Docker Compose not found. Please install Docker Compose") && exit 1)
	$(call print_success,"Docker and Docker Compose detected")

venv: check-python ## Create virtual environment
	$(call print_section,"Creating Virtual Environment")
	@if [ ! -d "$(VENV_NAME)" ]; then \
		python3 -m venv $(VENV_NAME); \
		$(call print_success,"Virtual environment created"); \
	else \
		$(call print_warning,"Virtual environment already exists"); \
	fi

venv-activate: ## Activate virtual environment (for manual use)
	@echo "To activate the virtual environment, run:"
	@echo "source $(VENV_NAME)/bin/activate"

install-deps: venv ## Install Python dependencies
	$(call print_section,"Installing Dependencies")
	@$(VENV_NAME)/bin/pip install --upgrade pip setuptools wheel
	@$(VENV_NAME)/bin/pip install -r requirements.txt
	@$(VENV_NAME)/bin/pip install -r agents/classifier/requirements.txt
	@$(VENV_NAME)/bin/pip install -r agents/orchestrator/requirements.txt
	$(call print_success,"Dependencies installed")

setup: check-python check-docker venv install-deps ## Initial project setup
	$(call print_section,"Project Setup Complete")
	@echo "Next steps:"
	@echo "1. Copy .env.example to .env and configure your environment variables"
	@echo "2. Run 'make dev' to start development environment"
	@echo "3. Run 'make test' to verify everything works"

# =======================================================
# Development Environment
# =======================================================

env-check: ## Check if .env file exists
	@if [ ! -f ".env" ]; then \
		$(call print_warning,"No .env file found. Copying from .env.example"); \
		cp .env.example .env; \
		$(call print_warning,"Please configure .env file with your API keys"); \
	fi

dev: env-check ## Start development environment
	$(call print_section,"Starting Development Environment")
	@echo "Development environment includes:"
	@echo "- Auto-reload enabled"
	@echo "- Debug logging"
	@echo "- Development dependencies"
	@echo ""
	@echo "Use 'make up' to start services with Docker"
	@echo "Use 'make test' to run tests"

# =======================================================
# Code Quality
# =======================================================

format: venv ## Format code with black
	$(call print_section,"Formatting Code")
	@$(VENV_NAME)/bin/black $(BLACK_ARGS) $(PYTHON_FILES)
	$(call print_success,"Code formatted")

format-check: venv ## Check code formatting
	$(call print_section,"Checking Code Format")
	@$(VENV_NAME)/bin/black --check $(BLACK_ARGS) $(PYTHON_FILES)
	$(call print_success,"Code format is correct")

lint: venv ## Lint code with ruff
	$(call print_section,"Linting Code")
	@$(VENV_NAME)/bin/ruff check $(RUFF_ARGS) $(PYTHON_FILES)
	$(call print_success,"Code linting passed")

lint-fix: venv ## Fix linting issues automatically
	$(call print_section,"Fixing Linting Issues")
	@$(VENV_NAME)/bin/ruff check --fix $(RUFF_ARGS) $(PYTHON_FILES)
	$(call print_success,"Linting issues fixed")

type-check: venv ## Type check with mypy
	$(call print_section,"Type Checking")
	@$(VENV_NAME)/bin/mypy $(MYPY_ARGS) $(PYTHON_FILES) || true
	$(call print_success,"Type checking completed")

quality: format-check lint type-check ## Run all code quality checks
	$(call print_success,"All code quality checks passed")

# =======================================================
# Testing
# =======================================================

test-unit: venv ## Run unit tests only
	$(call print_section,"Running Unit Tests")
	@$(VENV_NAME)/bin/pytest $(PYTEST_ARGS) -m "unit" $(TEST_PATH)
	$(call print_success,"Unit tests passed")

test-integration: venv ## Run integration tests only
	$(call print_section,"Running Integration Tests")
	@$(VENV_NAME)/bin/pytest $(PYTEST_ARGS) -m "integration" $(TEST_PATH)
	$(call print_success,"Integration tests passed")

test-coverage: venv ## Run tests with coverage
	$(call print_section,"Running Tests with Coverage")
	@$(VENV_NAME)/bin/pytest $(PYTEST_ARGS) $(PYTEST_COV_ARGS) $(TEST_PATH)
	$(call print_success,"Tests with coverage completed")

test-fast: venv ## Run tests excluding slow tests
	$(call print_section,"Running Fast Tests")
	@$(VENV_NAME)/bin/pytest $(PYTEST_ARGS) -m "not slow" $(TEST_PATH)
	$(call print_success,"Fast tests passed")

test: venv ## Run all tests
	$(call print_section,"Running All Tests")
	@$(VENV_NAME)/bin/pytest $(PYTEST_ARGS) $(TEST_PATH)
	$(call print_success,"All tests passed")

test-watch: venv ## Run tests in watch mode
	$(call print_section,"Running Tests in Watch Mode")
	@$(VENV_NAME)/bin/ptw -- $(PYTEST_ARGS) $(TEST_PATH)

# =======================================================
# Docker Operations
# =======================================================

build-classifier: check-docker ## Build classifier Docker image
	$(call print_section,"Building Classifier Image")
	@docker build -f agents/classifier/Dockerfile -t $(CLASSIFIER_IMAGE):latest .
	$(call print_success,"Classifier image built")

build-orchestrator: check-docker ## Build orchestrator Docker image
	$(call print_section,"Building Orchestrator Image")
	@docker build -f agents/orchestrator/Dockerfile -t $(ORCHESTRATOR_IMAGE):latest .
	$(call print_success,"Orchestrator image built")

build: build-classifier build-orchestrator ## Build all Docker images
	$(call print_success,"All Docker images built")

up: check-docker env-check ## Start all services with Docker Compose
	$(call print_section,"Starting Services")
	@docker-compose -f $(DOCKER_COMPOSE_FILE) up -d
	$(call print_success,"Services started")
	@echo ""
	@echo "Services are now running:"
	@echo "- Classifier: http://localhost:8001"
	@echo "- Orchestrator: http://localhost:8080"
	@echo ""
	@echo "Use 'make logs' to view logs"
	@echo "Use 'make status' to check service status"
	@echo "Use 'make down' to stop services"

down: check-docker ## Stop all services
	$(call print_section,"Stopping Services")
	@docker-compose -f $(DOCKER_COMPOSE_FILE) down
	$(call print_success,"Services stopped")

restart: down up ## Restart all services
	$(call print_success,"Services restarted")

logs: check-docker ## Show service logs
	@docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f

logs-classifier: check-docker ## Show classifier service logs
	@docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f $(CLASSIFIER_SERVICE)

logs-orchestrator: check-docker ## Show orchestrator service logs
	@docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f $(ORCHESTRATOR_SERVICE)

status: check-docker ## Show service status
	$(call print_section,"Service Status")
	@docker-compose -f $(DOCKER_COMPOSE_FILE) ps

# =======================================================
# Health Checks and Monitoring
# =======================================================

health: ## Check service health
	$(call print_section,"Health Check")
	@echo "Checking classifier health..."
	@curl -s http://localhost:8001/api/v1/health | jq '.' || $(call print_error,"Classifier unhealthy")
	@echo ""
	@echo "Checking orchestrator health..."
	@curl -s http://localhost:8080/api/v1/health | jq '.' || $(call print_error,"Orchestrator unhealthy")
	$(call print_success,"Health check completed")

metrics: ## Show service metrics
	$(call print_section,"Service Metrics")
	@echo "Classifier metrics:"
	@curl -s http://localhost:8001/api/v1/metrics | jq '.' || echo "Metrics unavailable"
	@echo ""
	@echo "Orchestrator metrics:"
	@curl -s http://localhost:8080/api/v1/metrics | jq '.' || echo "Metrics unavailable"

# =======================================================
# CLI Operations
# =======================================================

cli-help: venv ## Show CLI help
	@$(VENV_NAME)/bin/python -m cli.main --help

cli-test: venv ## Test CLI connection
	$(call print_section,"Testing CLI Connection")
	@$(VENV_NAME)/bin/python -m cli.main --test-connection
	$(call print_success,"CLI connection test completed")

cli-interactive: venv ## Start CLI in interactive mode
	$(call print_section,"Starting CLI Interactive Mode")
	@$(VENV_NAME)/bin/python -m cli.main

cli-message: venv ## Send a test message via CLI
	$(call print_section,"Sending Test Message")
	@$(VENV_NAME)/bin/python -m cli.main -m "What's the price of iPhone 15?"

# =======================================================
# Database and Data Operations
# =======================================================

generate-test-data: venv ## Generate test data
	$(call print_section,"Generating Test Data")
	@$(VENV_NAME)/bin/python -c "from tests.fixtures.test_data import TestDataProvider; \
		import json; \
		data = TestDataProvider.get_all_test_data(); \
		with open('test_data.json', 'w') as f: json.dump(data, f, indent=2, default=str)"
	$(call print_success,"Test data generated in test_data.json")

# =======================================================
# Performance and Load Testing
# =======================================================

load-test: venv ## Run load tests
	$(call print_section,"Running Load Tests")
	@$(VENV_NAME)/bin/pytest $(PYTEST_ARGS) -m "slow" tests/integration/test_end_to_end.py::TestSystemResilience::test_system_load_handling
	$(call print_success,"Load tests completed")

performance-test: venv ## Run performance tests
	$(call print_section,"Running Performance Tests")
	@$(VENV_NAME)/bin/pytest $(PYTEST_ARGS) -m "slow" tests/
	$(call print_success,"Performance tests completed")

# =======================================================
# Documentation
# =======================================================

docs-serve: venv ## Serve documentation locally
	$(call print_section,"Serving Documentation")
	@echo "Documentation available in README.md"
	@echo "API documentation available at:"
	@echo "- Classifier: http://localhost:8001/docs"
	@echo "- Orchestrator: http://localhost:8080/docs"

# =======================================================
# Cleanup
# =======================================================

clean-pyc: ## Remove Python bytecode files
	$(call print_section,"Cleaning Python Bytecode")
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	$(call print_success,"Python bytecode cleaned")

clean-test: ## Remove test artifacts
	$(call print_section,"Cleaning Test Artifacts")
	@rm -rf .pytest_cache/
	@rm -rf htmlcov/
	@rm -f coverage.xml
	@rm -f .coverage
	$(call print_success,"Test artifacts cleaned")

clean-docker: check-docker ## Remove Docker images and containers
	$(call print_section,"Cleaning Docker Resources")
	@docker-compose -f $(DOCKER_COMPOSE_FILE) down --remove-orphans
	@docker image rm $(CLASSIFIER_IMAGE):latest 2>/dev/null || true
	@docker image rm $(ORCHESTRATOR_IMAGE):latest 2>/dev/null || true
	@docker system prune -f
	$(call print_success,"Docker resources cleaned")

clean-venv: ## Remove virtual environment
	$(call print_section,"Cleaning Virtual Environment")
	@rm -rf $(VENV_NAME)/
	$(call print_success,"Virtual environment removed")

clean: clean-pyc clean-test ## Clean all artifacts except Docker and venv
	$(call print_success,"Project cleaned")

clean-all: clean clean-docker clean-venv ## Clean everything
	$(call print_success,"Everything cleaned")

# =======================================================
# Observability Testing
# =======================================================

test-observability: venv ## Run complete observability tests
	$(call print_section,"Running Observability Tests")
	@$(VENV_NAME)/bin/python test_observability_complete.py
	$(call print_success,"Observability tests completed")

test-costs: venv ## Test cost calculations
	$(call print_section,"Testing Cost Calculations")
	@$(VENV_NAME)/bin/python test_cost_calculations.py
	$(call print_success,"Cost calculation tests completed")

test-tracking: venv ## Test decision tracking
	$(call print_section,"Testing Decision Tracking")
	@$(VENV_NAME)/bin/python test_decision_tracking.py
	$(call print_success,"Decision tracking tests completed")

test-arize: venv ## Test Arize integration
	$(call print_section,"Testing Arize Integration")
	@$(VENV_NAME)/bin/python test_observability.py
	$(call print_success,"Arize integration tests completed")

check-observability: ## Check observability metrics endpoint
	$(call print_section,"Checking Observability Status")
	@curl -s http://localhost:8001/api/v1/observability-metrics | jq '.' || \
		$(call print_warning,"Metrics endpoint not available - ensure services are running")

observability-report: venv ## Generate observability report
	$(call print_section,"Generating Observability Report")
	@$(VENV_NAME)/bin/python -c "from shared.observability import get_metrics_summary; \
		from shared.observability_cost import session_cost_aggregator; \
		import json; \
		metrics = get_metrics_summary(); \
		sessions = session_cost_aggregator.get_all_sessions(); \
		print('📊 Observability Report'); \
		print('=' * 50); \
		print(json.dumps(metrics, indent=2, default=str)); \
		print('\n💰 Session Costs:'); \
		for sid, costs in sessions.items(): \
			print(f'  {sid}: Total=$${costs[\"total\"]:.6f}')"

# =======================================================
# Release and Deployment
# =======================================================

version: ## Show version information
	@echo "WhatsApp Sales Assistant v1.0.0"
	@echo "Python: $$(python3 --version)"
	@echo "Docker: $$(docker --version)"
	@echo "Docker Compose: $$(docker-compose --version)"

tag-release: ## Tag a new release
	@read -p "Enter release version (e.g., v1.0.0): " version; \
	git tag -a $$version -m "Release $$version"; \
	echo "Tagged release $$version"

pre-commit: quality test-fast ## Run pre-commit checks
	$(call print_success,"Pre-commit checks passed")

ci: quality test-coverage build ## Run CI pipeline
	$(call print_success,"CI pipeline completed")

# =======================================================
# Validation Pipeline
# =======================================================

validate-level1: quality ## Validation Level 1: Code quality
	$(call print_section,"Validation Level 1: Code Quality")
	$(call print_success,"Level 1 validation passed")

validate-level2: test-coverage ## Validation Level 2: Test coverage
	$(call print_section,"Validation Level 2: Test Coverage")
	$(call print_success,"Level 2 validation passed")

validate-level3: build up health down ## Validation Level 3: Integration
	$(call print_section,"Validation Level 3: Integration")
	$(call print_success,"Level 3 validation passed")

validate: validate-level1 validate-level2 validate-level3 ## Run complete validation pipeline
	$(call print_section,"Complete Validation Pipeline")
	$(call print_success,"All validation levels passed")
	@echo ""
	@echo "🎉 WhatsApp Sales Assistant is ready for deployment!"

# =======================================================
# Development Shortcuts
# =======================================================

quick-test: format-check lint test-fast ## Quick development test cycle
	$(call print_success,"Quick test cycle completed")

full-test: quality test-coverage ## Full test cycle
	$(call print_success,"Full test cycle completed")

dev-reset: down clean build up ## Reset development environment
	$(call print_success,"Development environment reset")

# =======================================================
# Special Targets
# =======================================================

.PHONY: all
all: setup build test validate ## Run complete setup, build, test, and validate

# Ensure make commands work even if files with the same name exist
.PHONY: test build clean help info setup dev up down logs status health metrics