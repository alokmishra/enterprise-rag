"""Audio processing for multi-modal RAG."""
from __future__ import annotations

import asyncio
import io
import logging
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
from datetime import datetime

from src.ingestion.multimodal.base import (
    MultiModalProcessor,
    MultiModalContent,
    ModalityType,
)
from src.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio."""
    text: str
    start_time: float  # seconds
    end_time: float  # seconds
    confidence: float
    speaker: Optional[str] = None
    language: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Result from audio transcription."""
    text: str
    segments: list[TranscriptionSegment]
    language: Optional[str] = None
    duration_seconds: float = 0.0
    confidence: float = 0.0
    word_count: int = 0


@dataclass
class SpeakerSegment:
    """A segment attributed to a specific speaker."""
    speaker_id: str
    start_time: float
    end_time: float
    text: Optional[str] = None


@dataclass
class AudioAnalysisResult:
    """Result from audio analysis."""
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    speakers: list[str] = field(default_factory=list)
    speaker_segments: list[SpeakerSegment] = field(default_factory=list)
    language: Optional[str] = None
    noise_level: Optional[float] = None  # 0-1 scale


class Transcriber:
    """Transcribe audio to text."""

    def __init__(
        self,
        provider: str = "whisper",  # whisper, openai, google, azure, deepgram
        model: Optional[str] = None,
        language: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self.provider = provider
        self.model = model
        self.language = language
        self.config = config or {}
        self._initialized = False
        self._transcriber = None

    async def initialize(self) -> None:
        """Initialize transcription engine."""
        if self._initialized:
            return

        if self.provider == "whisper":
            try:
                import whisper
                self._transcriber = whisper.load_model(self.model or "base")
                self._initialized = True
            except ImportError:
                logger.warning("whisper not installed")

        elif self.provider == "openai":
            # Use OpenAI Whisper API
            self._initialized = True

        elif self.provider == "deepgram":
            # Use Deepgram API
            self._initialized = True

        else:
            self._initialized = True

    async def transcribe(
        self,
        audio: Union[bytes, str, Path],
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio to text."""
        if not self._initialized:
            await self.initialize()

        if self.provider == "whisper":
            return await self._transcribe_whisper(audio, **kwargs)
        elif self.provider == "openai":
            return await self._transcribe_openai(audio, **kwargs)
        elif self.provider == "deepgram":
            return await self._transcribe_deepgram(audio, **kwargs)
        else:
            return TranscriptionResult(
                text="",
                segments=[],
                duration_seconds=0.0,
            )

    async def _transcribe_whisper(
        self,
        audio: Union[bytes, str, Path],
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe using local Whisper model."""
        if not self._transcriber:
            return TranscriptionResult(text="", segments=[], duration_seconds=0.0)

        import tempfile
        import os

        # Save to temp file if bytes
        if isinstance(audio, bytes):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                audio_path = f.name
        else:
            audio_path = str(audio)

        try:
            # Run transcription in thread pool
            result = await asyncio.to_thread(
                self._transcriber.transcribe,
                audio_path,
                language=self.language,
            )

            segments = []
            for seg in result.get("segments", []):
                segments.append(TranscriptionSegment(
                    text=seg["text"],
                    start_time=seg["start"],
                    end_time=seg["end"],
                    confidence=seg.get("confidence", 0.0) if "confidence" in seg else 0.9,
                ))

            full_text = result.get("text", "")

            return TranscriptionResult(
                text=full_text,
                segments=segments,
                language=result.get("language"),
                duration_seconds=segments[-1].end_time if segments else 0.0,
                confidence=sum(s.confidence for s in segments) / len(segments) if segments else 0.0,
                word_count=len(full_text.split()),
            )

        finally:
            if isinstance(audio, bytes) and os.path.exists(audio_path):
                os.unlink(audio_path)

    async def _transcribe_openai(
        self,
        audio: Union[bytes, str, Path],
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper API."""
        try:
            from openai import AsyncOpenAI

            settings = get_settings()
            client = AsyncOpenAI(api_key=settings.openai_api_key)

            # Prepare audio file
            if isinstance(audio, bytes):
                audio_file = io.BytesIO(audio)
                audio_file.name = "audio.wav"
            else:
                audio_file = open(audio, "rb")

            try:
                response = await client.audio.transcriptions.create(
                    model=self.model or "whisper-1",
                    file=audio_file,
                    language=self.language,
                    response_format="verbose_json",
                )

                segments = []
                for seg in response.segments or []:
                    segments.append(TranscriptionSegment(
                        text=seg.text,
                        start_time=seg.start,
                        end_time=seg.end,
                        confidence=seg.confidence if hasattr(seg, "confidence") else 0.9,
                    ))

                return TranscriptionResult(
                    text=response.text,
                    segments=segments,
                    language=response.language,
                    duration_seconds=response.duration if hasattr(response, "duration") else 0.0,
                    confidence=0.9,
                    word_count=len(response.text.split()),
                )

            finally:
                if not isinstance(audio, bytes):
                    audio_file.close()

        except Exception as e:
            logger.error(f"OpenAI transcription failed: {e}")
            return TranscriptionResult(text="", segments=[], duration_seconds=0.0)

    async def _transcribe_deepgram(
        self,
        audio: Union[bytes, str, Path],
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe using Deepgram API."""
        try:
            from deepgram import Deepgram

            settings = get_settings()
            client = Deepgram(settings.deepgram_api_key)

            # Prepare audio
            if isinstance(audio, (str, Path)):
                with open(audio, "rb") as f:
                    audio_bytes = f.read()
            else:
                audio_bytes = audio

            source = {"buffer": audio_bytes, "mimetype": "audio/wav"}

            response = await client.transcription.prerecorded(
                source,
                {
                    "punctuate": True,
                    "utterances": True,
                    "language": self.language or "en",
                    "model": self.model or "general",
                },
            )

            # Parse response
            results = response.get("results", {})
            channels = results.get("channels", [{}])
            alternatives = channels[0].get("alternatives", [{}]) if channels else [{}]
            transcript = alternatives[0].get("transcript", "") if alternatives else ""

            segments = []
            for word in alternatives[0].get("words", []) if alternatives else []:
                segments.append(TranscriptionSegment(
                    text=word.get("word", ""),
                    start_time=word.get("start", 0.0),
                    end_time=word.get("end", 0.0),
                    confidence=word.get("confidence", 0.0),
                ))

            return TranscriptionResult(
                text=transcript,
                segments=segments,
                language=self.language,
                duration_seconds=results.get("duration", 0.0),
                confidence=alternatives[0].get("confidence", 0.0) if alternatives else 0.0,
                word_count=len(transcript.split()),
            )

        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}")
            return TranscriptionResult(text="", segments=[], duration_seconds=0.0)


class AudioEmbedder:
    """Generate embeddings for audio."""

    def __init__(
        self,
        model: str = "wav2vec2",  # wav2vec2, clap, openai
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
        """Initialize audio embedding model."""
        if self._initialized:
            return

        if self.model == "wav2vec2":
            try:
                import torch
                from transformers import Wav2Vec2Processor, Wav2Vec2Model

                model_name = self.model_name or "facebook/wav2vec2-base-960h"
                self._processor = Wav2Vec2Processor.from_pretrained(model_name)
                self._embedder = Wav2Vec2Model.from_pretrained(model_name)
                self._initialized = True
            except ImportError:
                logger.warning("transformers/torch not installed for wav2vec2")

        elif self.model == "clap":
            try:
                import torch
                from transformers import ClapProcessor, ClapModel

                model_name = self.model_name or "laion/clap-htsat-unfused"
                self._processor = ClapProcessor.from_pretrained(model_name)
                self._embedder = ClapModel.from_pretrained(model_name)
                self._initialized = True
            except ImportError:
                logger.warning("transformers/torch not installed for CLAP")

        else:
            self._initialized = True

    async def embed(
        self,
        audio: Union[bytes, str, Path],
        sample_rate: int = 16000,
        **kwargs,
    ) -> list[float]:
        """Generate embedding for audio."""
        if not self._initialized:
            await self.initialize()

        # Load audio
        audio_array, sr = await self._load_audio(audio, sample_rate)

        if self.model == "wav2vec2" and self._embedder and self._processor:
            import torch

            inputs = self._processor(
                audio_array,
                sampling_rate=sr,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self._embedder(**inputs)

            # Mean pool over time dimension
            embedding = outputs.last_hidden_state.mean(dim=1)
            return embedding[0].tolist()

        elif self.model == "clap" and self._embedder and self._processor:
            import torch

            inputs = self._processor(
                audios=audio_array,
                sampling_rate=sr,
                return_tensors="pt",
            )

            with torch.no_grad():
                audio_features = self._embedder.get_audio_features(**inputs)

            embedding = audio_features / audio_features.norm(dim=-1, keepdim=True)
            return embedding[0].tolist()

        # Fallback
        return [0.0] * 768

    async def _load_audio(
        self,
        audio: Union[bytes, str, Path],
        target_sr: int = 16000,
    ) -> tuple[Any, int]:
        """Load audio and resample if needed."""
        try:
            import librosa
            import soundfile as sf
            import numpy as np

            if isinstance(audio, bytes):
                audio_array, sr = sf.read(io.BytesIO(audio))
            else:
                audio_array, sr = librosa.load(str(audio), sr=None)

            # Resample if needed
            if sr != target_sr:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
                sr = target_sr

            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)

            return audio_array, sr

        except ImportError:
            logger.warning("librosa/soundfile not installed")
            return [], target_sr


class SpeakerDiarizer:
    """Identify and segment speakers in audio."""

    def __init__(
        self,
        provider: str = "pyannote",  # pyannote, azure, aws
        config: Optional[dict] = None,
    ):
        self.provider = provider
        self.config = config or {}
        self._initialized = False
        self._diarizer = None

    async def initialize(self) -> None:
        """Initialize speaker diarization."""
        if self._initialized:
            return

        if self.provider == "pyannote":
            try:
                from pyannote.audio import Pipeline

                self._diarizer = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=self.config.get("hf_token"),
                )
                self._initialized = True
            except ImportError:
                logger.warning("pyannote not installed")

        else:
            self._initialized = True

    async def diarize(
        self,
        audio: Union[bytes, str, Path],
        **kwargs,
    ) -> list[SpeakerSegment]:
        """Identify speakers in audio."""
        if not self._initialized:
            await self.initialize()

        if self.provider == "pyannote" and self._diarizer:
            return await self._diarize_pyannote(audio, **kwargs)

        return []

    async def _diarize_pyannote(
        self,
        audio: Union[bytes, str, Path],
        **kwargs,
    ) -> list[SpeakerSegment]:
        """Diarize using pyannote."""
        import tempfile
        import os

        if isinstance(audio, bytes):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                audio_path = f.name
        else:
            audio_path = str(audio)

        try:
            diarization = await asyncio.to_thread(
                self._diarizer,
                audio_path,
            )

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(SpeakerSegment(
                    speaker_id=speaker,
                    start_time=turn.start,
                    end_time=turn.end,
                ))

            return segments

        finally:
            if isinstance(audio, bytes) and os.path.exists(audio_path):
                os.unlink(audio_path)


class AudioProcessor(MultiModalProcessor):
    """Complete audio processing pipeline."""

    def __init__(
        self,
        transcription_provider: str = "whisper",
        embedding_model: str = "wav2vec2",
        enable_diarization: bool = False,
        config: Optional[dict] = None,
    ):
        super().__init__(config)
        self.transcriber = Transcriber(provider=transcription_provider)
        self.embedder = AudioEmbedder(model=embedding_model)
        self.enable_diarization = enable_diarization
        self.diarizer = SpeakerDiarizer() if enable_diarization else None

    @property
    def supported_modalities(self) -> list[ModalityType]:
        return [ModalityType.AUDIO]

    @property
    def supported_formats(self) -> list[str]:
        return [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"]

    async def initialize(self) -> None:
        """Initialize all components."""
        tasks = [
            self.transcriber.initialize(),
            self.embedder.initialize(),
        ]
        if self.diarizer:
            tasks.append(self.diarizer.initialize())

        await asyncio.gather(*tasks)
        self._initialized = True

    async def process(
        self,
        content: Union[bytes, str, Path],
        modality: Optional[ModalityType] = None,
        transcribe: bool = True,
        generate_embedding: bool = True,
        diarize: bool = False,
        **kwargs,
    ) -> MultiModalContent:
        """Process audio and extract all information."""
        start_time = datetime.now()

        if not self._initialized:
            await self.initialize()

        # Load audio bytes
        if isinstance(content, bytes):
            audio_bytes = content
            source_path = None
        elif isinstance(content, (str, Path)):
            source_path = str(content)
            with open(content, "rb") as f:
                audio_bytes = f.read()
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

        # Get audio metadata
        audio_info = await self._get_audio_info(audio_bytes)

        # Create result object
        result = MultiModalContent(
            id=hashlib.sha256(audio_bytes).hexdigest()[:16],
            modality=ModalityType.AUDIO,
            source_path=source_path,
            file_size_bytes=len(audio_bytes),
            duration_seconds=audio_info.get("duration", 0.0),
            sample_rate=audio_info.get("sample_rate", 0),
            channels=audio_info.get("channels", 0),
            mime_type=audio_info.get("mime_type", "audio/unknown"),
        )
        result.compute_hash(audio_bytes)

        # Run processing tasks
        tasks = []

        if transcribe:
            tasks.append(("transcription", self.transcriber.transcribe(audio_bytes)))

        if generate_embedding:
            tasks.append(("embedding", self.embedder.embed(audio_bytes)))

        if diarize and self.diarizer:
            tasks.append(("diarization", self.diarizer.diarize(audio_bytes)))

        # Execute tasks
        results = {}
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Task {name} failed: {e}")
                results[name] = None

        # Populate result
        if results.get("transcription"):
            trans = results["transcription"]
            result.text_content = trans.text
            result.transcription = trans.text
            result.metadata["transcription_segments"] = [
                {
                    "text": s.text,
                    "start": s.start_time,
                    "end": s.end_time,
                    "confidence": s.confidence,
                }
                for s in trans.segments
            ]
            result.metadata["language"] = trans.language
            result.metadata["word_count"] = trans.word_count

        if results.get("embedding"):
            result.audio_embedding = results["embedding"]

        if results.get("diarization"):
            result.speaker_segments = [
                {
                    "speaker": s.speaker_id,
                    "start": s.start_time,
                    "end": s.end_time,
                }
                for s in results["diarization"]
            ]

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
        """Extract text from audio via transcription."""
        if not self._initialized:
            await self.initialize()

        result = await self.transcriber.transcribe(content, **kwargs)
        return result.text

    async def generate_embedding(
        self,
        content: Union[bytes, str, Path],
        **kwargs,
    ) -> list[float]:
        """Generate embedding for audio."""
        if not self._initialized:
            await self.initialize()

        return await self.embedder.embed(content, **kwargs)

    async def _get_audio_info(self, audio_bytes: bytes) -> dict:
        """Get audio file information."""
        try:
            import soundfile as sf

            info = sf.info(io.BytesIO(audio_bytes))
            return {
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "format": info.format,
                "mime_type": f"audio/{info.format.lower()}",
            }
        except Exception:
            return {
                "duration": 0.0,
                "sample_rate": 0,
                "channels": 0,
                "format": "unknown",
                "mime_type": "audio/unknown",
            }
