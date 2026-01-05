"""Image processing for multi-modal RAG."""

import asyncio
import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
from datetime import datetime
import hashlib

from src.ingestion.multimodal.base import (
    MultiModalProcessor,
    MultiModalContent,
    ModalityType,
)
from src.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    confidence: float
    regions: list[dict]  # Bounding boxes and text per region
    language: Optional[str] = None


@dataclass
class VisualAnalysisResult:
    """Result from visual analysis."""
    description: str
    objects: list[dict]  # Detected objects with labels and confidence
    tags: list[str]
    colors: list[dict]  # Dominant colors
    scene_type: Optional[str] = None


class OCREngine:
    """OCR engine for text extraction from images."""

    def __init__(
        self,
        engine: str = "tesseract",  # tesseract, easyocr, paddleocr, cloud
        languages: list[str] = None,
        config: Optional[dict] = None,
    ):
        self.engine = engine
        self.languages = languages or ["en"]
        self.config = config or {}
        self._initialized = False
        self._ocr = None

    async def initialize(self) -> None:
        """Initialize OCR engine."""
        if self._initialized:
            return

        if self.engine == "tesseract":
            try:
                import pytesseract
                self._ocr = pytesseract
                self._initialized = True
            except ImportError:
                logger.warning("pytesseract not installed, OCR will be limited")

        elif self.engine == "easyocr":
            try:
                import easyocr
                self._ocr = easyocr.Reader(self.languages, gpu=False)
                self._initialized = True
            except ImportError:
                logger.warning("easyocr not installed")

        elif self.engine == "cloud":
            # Use cloud OCR (Google Vision, AWS Textract, Azure)
            self._initialized = True

    async def extract_text(
        self,
        image: Union[bytes, str, Path],
        **kwargs,
    ) -> OCRResult:
        """Extract text from image using OCR."""
        if not self._initialized:
            await self.initialize()

        # Load image
        from PIL import Image

        if isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, (str, Path)):
            img = Image.open(image)
        else:
            img = image

        regions = []
        full_text = ""
        confidence = 0.0

        if self.engine == "tesseract" and self._ocr:
            # Use pytesseract
            data = self._ocr.image_to_data(
                img,
                output_type=self._ocr.Output.DICT,
                lang="+".join(self.languages),
            )

            texts = []
            for i, text in enumerate(data["text"]):
                if text.strip():
                    texts.append(text)
                    conf = data["conf"][i]
                    if conf > 0:
                        regions.append({
                            "text": text,
                            "confidence": conf / 100,
                            "bbox": {
                                "x": data["left"][i],
                                "y": data["top"][i],
                                "width": data["width"][i],
                                "height": data["height"][i],
                            },
                        })

            full_text = " ".join(texts)
            if regions:
                confidence = sum(r["confidence"] for r in regions) / len(regions)

        elif self.engine == "easyocr" and self._ocr:
            # Use EasyOCR
            img_array = await asyncio.to_thread(lambda: list(img.getdata()))
            results = await asyncio.to_thread(
                self._ocr.readtext,
                img,
            )

            texts = []
            for bbox, text, conf in results:
                texts.append(text)
                regions.append({
                    "text": text,
                    "confidence": conf,
                    "bbox": {
                        "points": bbox,
                    },
                })

            full_text = " ".join(texts)
            if regions:
                confidence = sum(r["confidence"] for r in regions) / len(regions)

        return OCRResult(
            text=full_text,
            confidence=confidence,
            regions=regions,
            language=self.languages[0] if self.languages else None,
        )


class ImageEmbedder:
    """Generate embeddings for images."""

    def __init__(
        self,
        model: str = "clip",  # clip, blip, openai
        model_name: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self.model = model
        self.model_name = model_name
        self.config = config or {}
        self._initialized = False
        self._embedder = None
        self._processor = None

    async def initialize(self) -> None:
        """Initialize embedding model."""
        if self._initialized:
            return

        if self.model == "clip":
            try:
                import torch
                from transformers import CLIPProcessor, CLIPModel

                model_name = self.model_name or "openai/clip-vit-base-patch32"
                self._embedder = CLIPModel.from_pretrained(model_name)
                self._processor = CLIPProcessor.from_pretrained(model_name)
                self._initialized = True
            except ImportError:
                logger.warning("transformers/torch not installed for CLIP")

        elif self.model == "openai":
            # Use OpenAI's vision API for embeddings
            self._initialized = True

    async def embed(
        self,
        image: Union[bytes, str, Path],
        **kwargs,
    ) -> list[float]:
        """Generate embedding for image."""
        if not self._initialized:
            await self.initialize()

        from PIL import Image

        if isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, (str, Path)):
            img = Image.open(image)
        else:
            img = image

        if self.model == "clip" and self._embedder and self._processor:
            import torch

            inputs = self._processor(images=img, return_tensors="pt")

            with torch.no_grad():
                image_features = self._embedder.get_image_features(**inputs)

            # Normalize and convert to list
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)
            return embedding[0].tolist()

        elif self.model == "openai":
            # For OpenAI, we'd use their vision API
            # Return placeholder for now
            return [0.0] * 512

        # Fallback: return zero embedding
        return [0.0] * 512


class VisualAnalyzer:
    """Analyze images for objects, scenes, and descriptions."""

    def __init__(
        self,
        provider: str = "local",  # local, openai, anthropic, google
        model: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self.provider = provider
        self.model = model
        self.config = config or {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize visual analyzer."""
        self._initialized = True

    async def analyze(
        self,
        image: Union[bytes, str, Path],
        **kwargs,
    ) -> VisualAnalysisResult:
        """Analyze image and extract visual information."""
        if not self._initialized:
            await self.initialize()

        from PIL import Image

        if isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
            img_bytes = image
        elif isinstance(image, (str, Path)):
            img = Image.open(image)
            with open(image, "rb") as f:
                img_bytes = f.read()
        else:
            img = image
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()

        if self.provider == "openai":
            return await self._analyze_openai(img_bytes, **kwargs)
        elif self.provider == "anthropic":
            return await self._analyze_anthropic(img_bytes, **kwargs)
        else:
            return await self._analyze_local(img, **kwargs)

    async def _analyze_local(
        self,
        image,
        **kwargs,
    ) -> VisualAnalysisResult:
        """Local image analysis using pre-trained models."""
        # Basic analysis using PIL
        colors = self._extract_dominant_colors(image)

        return VisualAnalysisResult(
            description="Image analysis (local mode)",
            objects=[],
            tags=[],
            colors=colors,
            scene_type=None,
        )

    async def _analyze_openai(
        self,
        image_bytes: bytes,
        **kwargs,
    ) -> VisualAnalysisResult:
        """Analyze image using OpenAI Vision API."""
        try:
            from openai import AsyncOpenAI

            settings = get_settings()
            client = AsyncOpenAI(api_key=settings.openai_api_key)

            # Encode image to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            response = await client.chat.completions.create(
                model=self.model or "gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image. Provide: 1) A detailed description, 2) List of objects detected, 3) Relevant tags, 4) Scene type. Format as JSON.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
            )

            # Parse response
            content = response.choices[0].message.content

            # Try to parse as JSON
            import json
            try:
                data = json.loads(content)
                return VisualAnalysisResult(
                    description=data.get("description", content),
                    objects=data.get("objects", []),
                    tags=data.get("tags", []),
                    colors=[],
                    scene_type=data.get("scene_type"),
                )
            except json.JSONDecodeError:
                return VisualAnalysisResult(
                    description=content,
                    objects=[],
                    tags=[],
                    colors=[],
                )

        except Exception as e:
            logger.error(f"OpenAI vision analysis failed: {e}")
            return VisualAnalysisResult(
                description=f"Analysis failed: {str(e)}",
                objects=[],
                tags=[],
                colors=[],
            )

    async def _analyze_anthropic(
        self,
        image_bytes: bytes,
        **kwargs,
    ) -> VisualAnalysisResult:
        """Analyze image using Anthropic Claude Vision."""
        try:
            from anthropic import AsyncAnthropic

            settings = get_settings()
            client = AsyncAnthropic(api_key=settings.anthropic_api_key)

            # Encode image to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            response = await client.messages.create(
                model=self.model or "claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Analyze this image. Provide: 1) A detailed description, 2) List of objects detected, 3) Relevant tags, 4) Scene type. Format as JSON.",
                            },
                        ],
                    }
                ],
            )

            content = response.content[0].text

            import json
            try:
                data = json.loads(content)
                return VisualAnalysisResult(
                    description=data.get("description", content),
                    objects=data.get("objects", []),
                    tags=data.get("tags", []),
                    colors=[],
                    scene_type=data.get("scene_type"),
                )
            except json.JSONDecodeError:
                return VisualAnalysisResult(
                    description=content,
                    objects=[],
                    tags=[],
                    colors=[],
                )

        except Exception as e:
            logger.error(f"Anthropic vision analysis failed: {e}")
            return VisualAnalysisResult(
                description=f"Analysis failed: {str(e)}",
                objects=[],
                tags=[],
                colors=[],
            )

    def _extract_dominant_colors(self, image, num_colors: int = 5) -> list[dict]:
        """Extract dominant colors from image."""
        try:
            # Resize for faster processing
            img = image.copy()
            img.thumbnail((100, 100))

            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Get colors
            from collections import Counter
            pixels = list(img.getdata())
            color_counts = Counter(pixels)

            # Get top colors
            top_colors = color_counts.most_common(num_colors)

            return [
                {
                    "rgb": color,
                    "hex": "#{:02x}{:02x}{:02x}".format(*color),
                    "count": count,
                }
                for color, count in top_colors
            ]
        except Exception:
            return []


class ImageProcessor(MultiModalProcessor):
    """Complete image processing pipeline."""

    def __init__(
        self,
        ocr_engine: str = "tesseract",
        embedding_model: str = "clip",
        vision_provider: str = "local",
        config: Optional[dict] = None,
    ):
        super().__init__(config)
        self.ocr = OCREngine(engine=ocr_engine)
        self.embedder = ImageEmbedder(model=embedding_model)
        self.analyzer = VisualAnalyzer(provider=vision_provider)

    @property
    def supported_modalities(self) -> list[ModalityType]:
        return [ModalityType.IMAGE]

    @property
    def supported_formats(self) -> list[str]:
        return [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"]

    async def initialize(self) -> None:
        """Initialize all components."""
        await asyncio.gather(
            self.ocr.initialize(),
            self.embedder.initialize(),
            self.analyzer.initialize(),
        )
        self._initialized = True

    async def process(
        self,
        content: Union[bytes, str, Path],
        modality: Optional[ModalityType] = None,
        extract_text: bool = True,
        generate_embedding: bool = True,
        analyze_visual: bool = True,
        **kwargs,
    ) -> MultiModalContent:
        """Process image and extract all information."""
        start_time = datetime.now()

        if not self._initialized:
            await self.initialize()

        from PIL import Image

        # Load image
        if isinstance(content, bytes):
            img_bytes = content
            img = Image.open(io.BytesIO(content))
            source_path = None
        elif isinstance(content, (str, Path)):
            source_path = str(content)
            with open(content, "rb") as f:
                img_bytes = f.read()
            img = Image.open(io.BytesIO(img_bytes))
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

        # Create result object
        result = MultiModalContent(
            id=hashlib.sha256(img_bytes).hexdigest()[:16],
            modality=ModalityType.IMAGE,
            source_path=source_path,
            width=img.width,
            height=img.height,
            file_size_bytes=len(img_bytes),
            mime_type=f"image/{img.format.lower() if img.format else 'unknown'}",
        )
        result.compute_hash(img_bytes)

        # Run processing tasks
        tasks = []

        if extract_text:
            tasks.append(("ocr", self.ocr.extract_text(img)))

        if generate_embedding:
            tasks.append(("embedding", self.embedder.embed(img)))

        if analyze_visual:
            tasks.append(("analysis", self.analyzer.analyze(img_bytes)))

        # Execute tasks
        results = {}
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Task {name} failed: {e}")
                results[name] = None

        # Populate result
        if results.get("ocr"):
            ocr_result = results["ocr"]
            result.text_content = ocr_result.text
            result.detected_text_regions = ocr_result.regions

        if results.get("embedding"):
            result.visual_embedding = results["embedding"]

        if results.get("analysis"):
            analysis = results["analysis"]
            result.image_description = analysis.description
            result.detected_objects = analysis.objects
            result.metadata["tags"] = analysis.tags
            result.metadata["colors"] = analysis.colors
            result.metadata["scene_type"] = analysis.scene_type

        # Calculate processing time
        result.processed_at = datetime.now()
        result.processing_time_ms = (
            result.processed_at - start_time
        ).total_seconds() * 1000

        return result

    async def extract_text(
        self,
        content: Union[bytes, str, Path],
        **kwargs,
    ) -> str:
        """Extract text from image using OCR."""
        if not self._initialized:
            await self.initialize()

        result = await self.ocr.extract_text(content, **kwargs)
        return result.text

    async def generate_embedding(
        self,
        content: Union[bytes, str, Path],
        **kwargs,
    ) -> list[float]:
        """Generate embedding for image."""
        if not self._initialized:
            await self.initialize()

        return await self.embedder.embed(content, **kwargs)
