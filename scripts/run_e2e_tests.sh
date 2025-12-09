#!/bin/bash
# Run E2E Tests for Crypto Scalper Bot

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================"
echo "  Crypto Scalper Bot - E2E Test Suite  "
echo "========================================"
echo ""

# Check if required services are available (optional for integration)
check_services() {
    echo "Checking optional services..."

    # Check Redis
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping &> /dev/null; then
            echo "  ✓ Redis is available"
            export REDIS_AVAILABLE=true
        else
            echo "  ⚠ Redis is not running (tests will use mocks)"
            export REDIS_AVAILABLE=false
        fi
    else
        echo "  ⚠ Redis CLI not found (tests will use mocks)"
        export REDIS_AVAILABLE=false
    fi
    echo ""
}

# Run tests
run_tests() {
    echo "Running E2E Tests..."
    echo ""

    # Set test environment
    export ENVIRONMENT=testnet
    export ENABLE_PAPER_TRADING=true
    export LOG_LEVEL=WARNING

    # Run pytest with coverage
    if [ "$1" == "--coverage" ]; then
        python -m pytest tests/e2e/ \
            -v \
            --tb=short \
            --cov=src \
            --cov-report=term-missing \
            --cov-report=html:coverage_e2e \
            "${@:2}"
    elif [ "$1" == "--fast" ]; then
        # Run only fast tests (exclude slow marked tests)
        python -m pytest tests/e2e/ \
            -v \
            --tb=short \
            -m "not slow" \
            "${@:2}"
    elif [ "$1" == "--specific" ]; then
        # Run specific test file or test
        python -m pytest "$2" \
            -v \
            --tb=long \
            "${@:3}"
    else
        # Run all tests
        python -m pytest tests/e2e/ \
            -v \
            --tb=short \
            "$@"
    fi
}

# Main
main() {
    check_services

    case "$1" in
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --coverage     Run tests with coverage report"
            echo "  --fast         Run only fast tests (exclude slow)"
            echo "  --specific     Run specific test file/function"
            echo "  --help, -h     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Run all E2E tests"
            echo "  $0 --coverage               # Run with coverage"
            echo "  $0 --fast                   # Run fast tests only"
            echo "  $0 --specific tests/e2e/test_trading_flow.py"
            echo "  $0 --specific tests/e2e/test_trading_flow.py::TestFullTradingCycle"
            ;;
        *)
            run_tests "$@"
            ;;
    esac
}

main "$@"
