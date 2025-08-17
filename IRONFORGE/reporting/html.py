from __future__ import annotations

from base64 import b64encode
from io import BytesIO


def _img_to_data_uri(img) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + b64encode(buf.getvalue()).decode("ascii")


def build_report_html(title: str, images: list[tuple[str, object]]) -> str:
    """
    images: List of (caption, PIL.Image)
    """
    blocks = []
    for caption, im in images:
        blocks.append(
            f"<figure><img src='{_img_to_data_uri(im)}' /><figcaption>{caption}</figcaption></figure>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,Helvetica,Arial;margin:24px}}figure{{margin:0 0 24px}}figcaption{{color:#555}}</style>
</head><body><h1>{title}</h1>{''.join(blocks)}</body></html>"""
