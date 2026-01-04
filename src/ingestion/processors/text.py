"""
Enterprise RAG System - Text Document Processors
"""

from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from src.core.types import ContentType
from src.ingestion.processors.base import DocumentProcessor, ProcessedDocument


class PlainTextProcessor(DocumentProcessor):
    """Processor for plain text files."""

    supported_extensions = [".txt", ".text", ".log"]
    supported_mimetypes = ["text/plain"]

    async def process(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProcessedDocument:
        """Process a plain text file."""
        content = self._read_text(source)

        # Try to extract title from first line
        lines = content.strip().split("\n")
        title = lines[0][:100] if lines else None

        return ProcessedDocument(
            content=content,
            content_type=ContentType.TEXT,
            metadata=metadata or {},
            title=title,
        )


class MarkdownProcessor(DocumentProcessor):
    """Processor for Markdown files."""

    supported_extensions = [".md", ".markdown", ".mdown"]
    supported_mimetypes = ["text/markdown", "text/x-markdown"]

    async def process(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProcessedDocument:
        """Process a Markdown file."""
        content = self._read_text(source)

        # Extract title from first heading
        title = None
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                break

        return ProcessedDocument(
            content=content,
            content_type=ContentType.TEXT,
            metadata={**(metadata or {}), "format": "markdown"},
            title=title,
        )


class CodeProcessor(DocumentProcessor):
    """Processor for source code files."""

    supported_extensions = [
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".cpp", ".c", ".h", ".hpp",
        ".go", ".rs", ".rb", ".php", ".swift",
        ".kt", ".scala", ".sql", ".sh", ".bash",
        ".yaml", ".yml", ".json", ".xml", ".html",
        ".css", ".scss", ".less",
    ]
    supported_mimetypes = [
        "text/x-python",
        "application/javascript",
        "application/typescript",
        "text/x-java-source",
    ]

    # Map extensions to language names
    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".sql": "sql",
        ".sh": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
    }

    async def process(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProcessedDocument:
        """Process a source code file."""
        content = self._read_text(source)

        # Detect language from extension
        language = None
        if isinstance(source, (str, Path)):
            ext = Path(source).suffix.lower()
            language = self.LANGUAGE_MAP.get(ext)

        # Try to extract module/class name as title
        title = self._extract_title(content, language)

        return ProcessedDocument(
            content=content,
            content_type=ContentType.CODE,
            metadata={
                **(metadata or {}),
                "language": language,
            },
            title=title,
            language=language,
        )

    def _extract_title(self, content: str, language: Optional[str]) -> Optional[str]:
        """Extract a title from code content."""
        lines = content.split("\n")

        if language == "python":
            # Look for module docstring or class name
            for line in lines[:20]:
                if line.strip().startswith("class "):
                    return line.strip().split("(")[0].replace("class ", "")
                if line.strip().startswith("def "):
                    return line.strip().split("(")[0].replace("def ", "")

        elif language in ("javascript", "typescript"):
            for line in lines[:20]:
                if "class " in line:
                    parts = line.split("class ")
                    if len(parts) > 1:
                        return parts[1].split()[0].strip("{")
                if "function " in line:
                    parts = line.split("function ")
                    if len(parts) > 1:
                        return parts[1].split("(")[0]

        return None
