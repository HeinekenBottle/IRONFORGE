from __future__ import annotations
from typing import Any, Dict


class DashboardGenerator:
    def create_html_dashboard(self, visualization_data: Dict[str, Any]) -> str:
        # Minimal HTML generator
        return "<html><body><h1>Minidash</h1></body></html>"

    def export_png_dashboard(self, html_content: str) -> str:
        return "runs/latest/minidash.png"
