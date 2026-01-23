#!/bin/bash
# Leaderboard Server Launch Script

echo "🎯 Starting Bioplausible Leaderboard"
echo "===================================="
echo ""
echo "Default database: examples/shallow_benchmark.db"
echo "Default port: 5000"
echo ""
echo "Options:"
echo "  --db PATH     : Specify database path"
echo "  --port N      : Specify port number"
echo ""
echo "📊 Server will start at http://localhost:5000"
echo "   Open this URL in your browser to view the leaderboard"
echo ""

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask not found. Installing..."
    pip install flask flask-cors
fi

# Start server
cd bioplausible_ui
python3 leaderboard_server.py "$@"
