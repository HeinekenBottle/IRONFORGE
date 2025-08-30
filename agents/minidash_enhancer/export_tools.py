from __future__ import annotations


class ExportTools:
    def create_html_dashboard(self, visualization_data):
        return "<html></html>"

    def export_png_dashboard(self, html_content: str) -> str:
        return "runs/latest/minidash.png"
