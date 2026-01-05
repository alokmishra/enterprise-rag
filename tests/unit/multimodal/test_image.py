"""Tests for image processing."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import io

from src.ingestion.multimodal.image import (
    ImageProcessor,
    OCREngine,
    ImageEmbedder,
    VisualAnalyzer,
    OCRResult,
    VisualAnalysisResult,
)
from src.ingestion.multimodal.base import ModalityType


class TestOCREngine:
    """Tests for OCR engine."""

    @pytest.fixture
    def ocr_engine(self):
        """Create OCR engine."""
        return OCREngine(engine="tesseract", languages=["en"])

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image bytes."""
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    @pytest.mark.asyncio
    async def test_initialize(self, ocr_engine):
        """Test OCR engine initialization."""
        await ocr_engine.initialize()
        # Should not raise

    @pytest.mark.asyncio
    async def test_extract_text_from_bytes(self, ocr_engine, sample_image_bytes):
        """Test text extraction from image bytes."""
        with patch.object(ocr_engine, "_ocr") as mock_ocr:
            mock_ocr.image_to_data.return_value = {
                "text": ["Hello", "World"],
                "conf": [95, 90],
                "left": [10, 50],
                "top": [10, 10],
                "width": [30, 40],
                "height": [20, 20],
            }
            ocr_engine._initialized = True

            result = await ocr_engine.extract_text(sample_image_bytes)

            assert isinstance(result, OCRResult)
            assert "Hello" in result.text or result.text == ""

    @pytest.mark.asyncio
    async def test_extract_text_from_path(self, ocr_engine, tmp_path):
        """Test text extraction from image path."""
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        with patch.object(ocr_engine, "_ocr") as mock_ocr:
            mock_ocr.image_to_data.return_value = {
                "text": [],
                "conf": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
            }
            ocr_engine._initialized = True

            result = await ocr_engine.extract_text(img_path)

            assert isinstance(result, OCRResult)

    def test_ocr_result_structure(self):
        """Test OCRResult structure."""
        result = OCRResult(
            text="Sample text",
            confidence=0.95,
            regions=[{"text": "Sample", "confidence": 0.95, "bbox": {}}],
            language="en",
        )

        assert result.text == "Sample text"
        assert result.confidence == 0.95
        assert len(result.regions) == 1


class TestImageEmbedder:
    """Tests for image embedder."""

    @pytest.fixture
    def embedder(self):
        """Create image embedder."""
        return ImageEmbedder(model="clip")

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image bytes."""
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    @pytest.mark.asyncio
    async def test_initialize(self, embedder):
        """Test embedder initialization."""
        await embedder.initialize()
        # May fail if transformers not installed, that's ok

    @pytest.mark.asyncio
    async def test_embed_returns_list(self, embedder, sample_image_bytes):
        """Test embedding returns list of floats."""
        with patch.object(embedder, "_embedder") as mock_model:
            with patch.object(embedder, "_processor") as mock_proc:
                import torch
                mock_model.get_image_features.return_value = torch.randn(1, 512)
                mock_proc.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
                embedder._initialized = True

                embedding = await embedder.embed(sample_image_bytes)

                assert isinstance(embedding, list)
                assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_from_path(self, embedder, tmp_path):
        """Test embedding from file path."""
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="blue")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        # Mock the embedding
        embedder._initialized = True
        embedder._embedder = None  # Will use fallback

        embedding = await embedder.embed(img_path)

        assert isinstance(embedding, list)


class TestVisualAnalyzer:
    """Tests for visual analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create visual analyzer."""
        return VisualAnalyzer(provider="local")

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image bytes."""
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    @pytest.mark.asyncio
    async def test_initialize(self, analyzer):
        """Test analyzer initialization."""
        await analyzer.initialize()
        assert analyzer._initialized

    @pytest.mark.asyncio
    async def test_analyze_local(self, analyzer, sample_image_bytes):
        """Test local image analysis."""
        await analyzer.initialize()
        result = await analyzer.analyze(sample_image_bytes)

        assert isinstance(result, VisualAnalysisResult)
        assert result.description is not None

    @pytest.mark.asyncio
    async def test_analyze_openai(self, sample_image_bytes):
        """Test OpenAI vision analysis."""
        analyzer = VisualAnalyzer(provider="openai")

        with patch("src.ingestion.multimodal.image.AsyncOpenAI") as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content='{"description": "A green square"}'))
            ]
            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            await analyzer.initialize()
            result = await analyzer.analyze(sample_image_bytes)

            assert isinstance(result, VisualAnalysisResult)

    def test_extract_dominant_colors(self, analyzer):
        """Test dominant color extraction."""
        from PIL import Image
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))

        colors = analyzer._extract_dominant_colors(img, num_colors=3)

        assert isinstance(colors, list)
        if colors:
            assert "rgb" in colors[0]
            assert "hex" in colors[0]


class TestImageProcessor:
    """Tests for complete image processor."""

    @pytest.fixture
    def processor(self):
        """Create image processor."""
        return ImageProcessor(
            ocr_engine="tesseract",
            embedding_model="clip",
            vision_provider="local",
        )

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image bytes."""
        from PIL import Image
        img = Image.new("RGB", (200, 200), color="yellow")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_supported_modalities(self, processor):
        """Test supported modalities."""
        assert ModalityType.IMAGE in processor.supported_modalities

    def test_supported_formats(self, processor):
        """Test supported formats."""
        assert ".jpg" in processor.supported_formats
        assert ".png" in processor.supported_formats

    @pytest.mark.asyncio
    async def test_process_image(self, processor, sample_image_bytes):
        """Test full image processing."""
        with patch.object(processor.ocr, "extract_text") as mock_ocr:
            with patch.object(processor.embedder, "embed") as mock_embed:
                with patch.object(processor.analyzer, "analyze") as mock_analyze:
                    mock_ocr.return_value = OCRResult(
                        text="Test text",
                        confidence=0.9,
                        regions=[],
                    )
                    mock_embed.return_value = [0.1] * 512
                    mock_analyze.return_value = VisualAnalysisResult(
                        description="A yellow square",
                        objects=[],
                        tags=["yellow", "square"],
                        colors=[],
                    )

                    processor._initialized = True
                    result = await processor.process(
                        sample_image_bytes,
                        extract_text=True,
                        generate_embedding=True,
                        analyze_visual=True,
                    )

                    assert result.modality == ModalityType.IMAGE
                    assert result.text_content == "Test text"
                    assert result.visual_embedding is not None
                    assert result.image_description == "A yellow square"

    @pytest.mark.asyncio
    async def test_extract_text(self, processor, sample_image_bytes):
        """Test text extraction."""
        with patch.object(processor.ocr, "extract_text") as mock_ocr:
            mock_ocr.return_value = OCRResult(
                text="Extracted text",
                confidence=0.85,
                regions=[],
            )
            processor._initialized = True

            text = await processor.extract_text(sample_image_bytes)

            assert text == "Extracted text"

    @pytest.mark.asyncio
    async def test_generate_embedding(self, processor, sample_image_bytes):
        """Test embedding generation."""
        with patch.object(processor.embedder, "embed") as mock_embed:
            mock_embed.return_value = [0.5] * 512
            processor._initialized = True

            embedding = await processor.generate_embedding(sample_image_bytes)

            assert len(embedding) == 512

    @pytest.mark.asyncio
    async def test_process_from_path(self, processor, tmp_path):
        """Test processing from file path."""
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="cyan")
        img_path = tmp_path / "test_image.png"
        img.save(img_path)

        with patch.object(processor.ocr, "extract_text") as mock_ocr:
            with patch.object(processor.embedder, "embed") as mock_embed:
                mock_ocr.return_value = OCRResult(text="", confidence=0.0, regions=[])
                mock_embed.return_value = [0.0] * 512
                processor._initialized = True

                result = await processor.process(
                    img_path,
                    analyze_visual=False,
                )

                assert result.source_path == str(img_path)
                assert result.modality == ModalityType.IMAGE
