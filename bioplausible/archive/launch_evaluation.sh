#!/bin/bash
#
# Quick Launch Script for Novel Hybrid Algorithm Evaluation
#
# Usage: ./launch_evaluation.sh [HOURS]
#

set -e

# Configuration
HOURS=${1:-1.0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

echo "=========================================="
echo "Novel Hybrid Algorithm Evaluation"
echo "=========================================="
echo "Duration: ${HOURS} hours"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Pre-flight check
echo "Running pre-flight checks..."
python tests/test_algorithms.py -q 2>&1 | tail -5
if [ $? -ne 0 ]; then
    echo "ERROR: Unit tests failed!"
    echo "Run: python tests/test_algorithms.py -v"
    exit 1
fi
echo "✓ All algorithms verified"
echo ""

# Launch evaluation
echo "Launching evaluation..."
echo ""
python run_evaluation.py --hours "$HOURS"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Check results/algorithm_comparison/ for outputs"
echo "=========================================="
