"""Increase a PDF's byte size without changing its visible pages.

The script adds a deterministic image resource outside the page boundary.
The original vector page, searchable text, page count and page geometry are
preserved. This is useful for upload systems that enforce a minimum PDF size.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import shutil
import tempfile
from pathlib import Path

import fitz
from PIL import Image


def page_digest(document: fitz.Document, dpi: int = 180) -> str:
    page = document[0]
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return hashlib.sha256(pix.samples).hexdigest()


def padding_jpeg(minimum_bytes: int) -> bytes:
    edge = max(1100, math.ceil(math.sqrt(minimum_bytes * 1.1)))
    samples = hashlib.shake_256(b"ywzs-pdf-size-resource").digest(edge * edge * 3)
    image = Image.frombytes("RGB", (edge, edge), samples)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, optimize=False)
    return buffer.getvalue()


def ensure_min_size(source: Path, output: Path, minimum_bytes: int) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if source.stat().st_size >= minimum_bytes:
        if source != output:
            shutil.copy2(source, output)
        return {
            "changed": False,
            "bytes": output.stat().st_size,
            "pages": len(fitz.open(output)),
        }

    with fitz.open(source) as original:
        before_pages = len(original)
        before_rect = original[0].rect
        before_text = original[0].get_text("text")
        before_digest = page_digest(original)
        resource = padding_jpeg(minimum_bytes)

        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False, dir=output.parent
        ) as handle:
            temporary = Path(handle.name)

        try:
            page = original[0]
            padding_rect = fitz.Rect(
                page.rect.x1 + 20,
                page.rect.y1 + 20,
                page.rect.x1 + 120,
                page.rect.y1 + 120,
            )
            page.insert_image(padding_rect, stream=resource, overlay=False)
            original.save(temporary, garbage=4, clean=True, deflate=False)

            if temporary.stat().st_size < minimum_bytes:
                raise RuntimeError(
                    f"Generated PDF is still below the requested minimum: "
                    f"{temporary.stat().st_size} < {minimum_bytes}"
                )

            with fitz.open(temporary) as checked:
                if len(checked) != before_pages:
                    raise RuntimeError("Page count changed")
                if checked[0].rect != before_rect:
                    raise RuntimeError("Page geometry changed")
                if checked[0].get_text("text") != before_text:
                    raise RuntimeError("Searchable page text changed")
                if page_digest(checked) != before_digest:
                    raise RuntimeError("Visible page rendering changed")

            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)

    with fitz.open(output) as final_doc:
        return {
            "changed": True,
            "bytes": output.stat().st_size,
            "pages": len(final_doc),
            "embedded_files": final_doc.embfile_count(),
            "visible_page_sha256": page_digest(final_doc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-bytes", type=int, default=1_100_000)
    args = parser.parse_args()
    print(ensure_min_size(args.source, args.output, args.minimum_bytes))


if __name__ == "__main__":
    main()
