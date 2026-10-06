"""
Production Web Server for Transit Harassment Simulation & Telemetry Suite.
Compatible with Render, Heroku, Railway, Docker, and local development.
Binds to 0.0.0.0:$PORT as specified by hosting platforms.
"""

import os
import sys
import json
import http.server
import socketserver
from pathlib import Path

# Render assigns PORT dynamically via environment variable (default 10000)
PORT = int(os.environ.get("PORT", 10000))
BASE_DIR = Path(__file__).resolve().parent


class TransitHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def end_headers(self):
        # Enable CORS and control caching for live continuous telemetry
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        if self.path.endswith(".html") or self.path.endswith(".json") or self.path == "/":
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        else:
            self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()

    def do_GET(self):
        # Health check endpoint for Render health monitoring
        if self.path in ("/healthz", "/health", "/api/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "status": "healthy",
                "service": "transit-harassment-telemetry",
                "version": "2.4.0",
                "colombo_timezone": "Asia/Colombo"
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return

        # Default root route serves index.html
        if self.path == "" or self.path == "/":
            self.path = "/index.html"

        return super().do_GET()


def main():
    handler = TransitHTTPRequestHandler
    # Allow socket address reuse to prevent bind errors during restarts
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("0.0.0.0", PORT), handler) as httpd:
        print("=" * 65)
        print("  SRI LANKA TRANSIT HARASSMENT TELEMETRY WEB SERVER")
        print("=" * 65)
        print(f" • Host URL       : http://0.0.0.0:{PORT}")
        print(f" • Local Access   : http://localhost:{PORT}")
        print(f" • Health Check   : http://0.0.0.0:{PORT}/healthz")
        print(f" • Static Root    : {BASE_DIR}")
        print(f" • Real-time Sync : Asia/Colombo (Sri Lanka Standard Time)")
        print("=" * 65)
        print("Server is ready and accepting requests...\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server gracefully...")
            httpd.shutdown()


if __name__ == "__main__":
    main()
