"""
Leaderboard Server

Lightweight Flask server for serving leaderboard UI and API.
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bioplausible_ui.leaderboard_data import (
    load_trials,
    compute_pareto_frontier,
    format_for_frontend,
)

app = Flask(__name__, static_folder=".")
CORS(app)  # Enable CORS for development

# Configuration
DB_PATH = "examples/shallow_benchmark.db"  # Default path


@app.route("/")
def index():
    """Serve main leaderboard HTML."""
    return send_from_directory(".", "leaderboard_app.html")


@app.route("/api/trials")
def get_trials():
    """
    API endpoint for trial data.
    
    Returns:
        JSON with all trials, Pareto frontier, and statistics
    """
    try:
        trials = load_trials(DB_PATH)
        
        if not trials:
            return jsonify({
                'trials': [],
                'pareto_ids': [],
                'statistics': {},
                'best_per_model': {},
                'error': None,
            })
        
        pareto_ids = compute_pareto_frontier(trials)
        data = format_for_frontend(trials, pareto_ids)
        data['error'] = None
        
        return jsonify(data)
    
    except Exception as e:
        return jsonify({
            'trials': [],
            'pareto_ids': [],
            'statistics': {},
            'best_per_model': {},
            'error': str(e),
        })


@app.route("/api/config")
def get_config():
    """Get server configuration."""
    return jsonify({
        'db_path': DB_PATH,
        'refresh_interval': 5000,  # ms
    })


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Leaderboard Server")
    parser.add_argument(
        "--db",
        default="examples/shallow_benchmark.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Server port"
    )
    args = parser.parse_args()
    
    DB_PATH = args.db
    
    print(f"🚀 Leaderboard server starting...")
    print(f"   Database: {DB_PATH}")
    print(f"   URL: http://localhost:{args.port}")
    print(f"\n📊 Open browser to view leaderboard!\n")
    
    app.run(host="0.0.0.0", port=args.port, debug=True)
