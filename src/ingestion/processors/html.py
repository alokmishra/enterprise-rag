"""
Enterprise RAG System - HTML Document Processor
"""

import re
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from bs4 import BeautifulSoup

from src.core.types import ContentType
from src.ingestion.processors.base import DocumentProcessor, ProcessedDocument


class HTMLProcessor(DocumentProcessor):
    """Processor for HTML documents."""

    supported_extensions = [".html", ".htm", ".xhtml"]
    supported_mimetypes = ["text/html", "application/xhtml+xml"]

    # Tags to remove entirely (including content)
    REMOVE_TAGS = ["script", "style", "noscript", "iframe", "svg", "canvas"]

    # Tags that represent structural breaks
    BLOCK_TAGS = [
        "p", "div", "section", "article", "header", "footer",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "table", "tr", "br", "hr",
        "blockquote", "pre", "code",
    ]

    async def process(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProcessedDocument:
        """Process an HTML file and extract text content."""
        html_content = self._read_text(source)

        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove unwanted tags
        for tag in self.REMOVE_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

        # Extract title
        title = None
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # If no title tag, try h1
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)

        # Extract metadata from meta tags
        html_metadata = {}
        for meta in soup.find_all("meta"):
            name = meta.get("name", meta.get("property", ""))
            content = meta.get("content", "")
            if name and content:
                if name in ("description", "author", "keywords"):
                    html_metadata[f"html_{name}"] = content
                elif name.startswith("og:"):
                    html_metadata[f"html_{name}"] = content

        # Extract main content
        # Try to find main content area
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find(id="content") or
            soup.find(class_="content") or
            soup.find("body") or
            soup
        )

        # Convert to text with structure preservation
        text = self._extract_text(main_content)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        # Merge metadata
        combined_metadata = {**html_metadata, **(metadata or {})}

        return ProcessedDocument(
            content=text,
            content_type=ContentType.TEXT,
            metadata=combined_metadata,
            title=title,
        )

    def _extract_text(self, element) -> str:
        """Extract text from HTML element with structure preservation."""
        parts = []

        for child in element.children:
            if child.name is None:
                # Text node
                text = str(child).strip()
                if text:
                    parts.append(text)
            elif child.name in self.REMOVE_TAGS:
                continue
            elif child.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(child.name[1])
                text = child.get_text(strip=True)
                if text:
                    parts.append(f"\n{'#' * level} {text}\n")
            elif child.name == "li":
                text = child.get_text(strip=True)
                if text:
                    parts.append(f"  - {text}")
            elif child.name == "br":
                parts.append("\n")
            elif child.name == "hr":
                parts.append("\n---\n")
            elif child.name in ("pre", "code"):
                text = child.get_text()
                if text:
                    parts.append(f"\n```\n{text}\n```\n")
            elif child.name == "blockquote":
                text = child.get_text(strip=True)
                if text:
                    quoted = "\n".join(f"> {line}" for line in text.split("\n"))
                    parts.append(f"\n{quoted}\n")
            elif child.name in self.BLOCK_TAGS:
                text = self._extract_text(child)
                if text:
                    parts.append(f"\n{text}\n")
            elif child.name == "a":
                text = child.get_text(strip=True)
                href = child.get("href", "")
                if text and href and not href.startswith("#"):
                    parts.append(f"[{text}]({href})")
                elif text:
                    parts.append(text)
            else:
                text = self._extract_text(child)
                if text:
                    parts.append(text)

        return " ".join(parts)
