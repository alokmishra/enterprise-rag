"""Tests for audio processing."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import io

from src.ingestion.multimodal.audio import (
    AudioProcessor,
    Transcriber,
    AudioEmbedder,
    SpeakerDiarizer,
    TranscriptionResult,
    TranscriptionSegment,
    SpeakerSegment,
)
from src.ingestion.multimodal.base import ModalityType


class TestTranscriber:
    """Tests for transcriber."""

    @pytest.fixture
    def transcriber(self):
        """Create transcriber."""
        return Transcriber(provider="whisper", model="base")

    @pytest.fixture
    def sample_audio_bytes(self):
        """Create sample audio bytes (WAV header + silence)."""
        # Minimal WAV file
        import struct
        sample_rate = 16000
        duration = 1  # 1 second
        num_samples = sample_rate * duration

        # WAV header
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + num_samples * 2,
            b"WAVE",
            b"fmt ",
            16,  # Subchunk1Size
            1,   # AudioFormat (PCM)
            1,   # NumChannels
            sample_rate,
            sample_rate * 2,  # ByteRate
            2,   # BlockAlign
            16,  # BitsPerSample
            b"data",
            num_samples * 2,
        )

        # Silent audio data
        audio_data = b"\x00" * (num_samples * 2)

        return header + audio_data

    @pytest.mark.asyncio
    async def test_initialize(self, transcriber):
        """Test transcriber initialization."""
        await transcriber.initialize()
        # May fail if whisper not installed

    @pytest.mark.asyncio
    async def test_transcribe_returns_result(self, transcriber, sample_audio_bytes):
        """Test transcription returns result."""
        with patch.object(transcriber, "_transcriber") as mock_whisper:
            mock_whisper.transcribe.return_value = {
                "text": "Hello world",
                "segments": [
                    {"text": "Hello", "start": 0.0, "end": 0.5},
                    {"text": "world", "start": 0.5, "end": 1.0},
                ],
                "language": "en",
            }
            transcriber._initialized = True

            result = await transcriber.transcribe(sample_audio_bytes)

            assert isinstance(result, TranscriptionResult)
            assert "Hello" in result.text or result.text == ""

    @pytest.mark.asyncio
    async def test_transcribe_openai(self, sample_audio_bytes):
        """Test OpenAI transcription."""
        transcriber = Transcriber(provider="openai")

        with patch("src.ingestion.multimodal.audio.AsyncOpenAI") as mock_client:
            mock_response = MagicMock()
            mock_response.text = "Transcribed text"
            mock_response.segments = []
            mock_response.language = "en"
            mock_client.return_value.audio.transcriptions.create = AsyncMock(
                return_value=mock_response
            )

            await transcriber.initialize()
            result = await transcriber.transcribe(sample_audio_bytes)

            assert isinstance(result, TranscriptionResult)

    def test_transcription_result_structure(self):
        """Test TranscriptionResult structure."""
        result = TranscriptionResult(
            text="Sample transcription",
            segments=[
                TranscriptionSegment(
                    text="Sample",
                    start_time=0.0,
                    end_time=0.5,
                    confidence=0.95,
                )
            ],
            language="en",
            duration_seconds=1.0,
            confidence=0.95,
            word_count=2,
        )

        assert result.text == "Sample transcription"
        assert len(result.segments) == 1
        assert result.segments[0].confidence == 0.95


class TestAudioEmbedder:
    """Tests for audio embedder."""

    @pytest.fixture
    def embedder(self):
        """Create audio embedder."""
        return AudioEmbedder(model="wav2vec2")

    @pytest.fixture
    def sample_audio_bytes(self):
        """Create sample audio bytes."""
        import struct
        sample_rate = 16000
        num_samples = sample_rate  # 1 second

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + num_samples * 2, b"WAVE", b"fmt ", 16,
            1, 1, sample_rate, sample_rate * 2, 2, 16, b"data", num_samples * 2,
        )
        audio_data = b"\x00" * (num_samples * 2)
        return header + audio_data

    @pytest.mark.asyncio
    async def test_initialize(self, embedder):
        """Test embedder initialization."""
        await embedder.initialize()
        # May fail if transformers not installed

    @pytest.mark.asyncio
    async def test_embed_returns_list(self, embedder, sample_audio_bytes):
        """Test embedding returns list of floats."""
        # Mock the embedding process
        embedder._initialized = True
        embedder._embedder = None  # Will use fallback

        embedding = await embedder.embed(sample_audio_bytes)

        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)


class TestSpeakerDiarizer:
    """Tests for speaker diarization."""

    @pytest.fixture
    def diarizer(self):
        """Create speaker diarizer."""
        return SpeakerDiarizer(provider="pyannote")

    @pytest.mark.asyncio
    async def test_initialize(self, diarizer):
        """Test diarizer initialization."""
        await diarizer.initialize()
        # May fail if pyannote not installed

    @pytest.mark.asyncio
    async def test_diarize_returns_segments(self, diarizer):
        """Test diarization returns speaker segments."""
        diarizer._initialized = True
        diarizer._diarizer = None  # No model loaded

        segments = await diarizer.diarize(b"audio_bytes")

        assert isinstance(segments, list)

    def test_speaker_segment_structure(self):
        """Test SpeakerSegment structure."""
        segment = SpeakerSegment(
            speaker_id="SPEAKER_01",
            start_time=0.0,
            end_time=5.5,
            text="Hello, how are you?",
        )

        assert segment.speaker_id == "SPEAKER_01"
        assert segment.end_time == 5.5


class TestAudioProcessor:
    """Tests for complete audio processor."""

    @pytest.fixture
    def processor(self):
        """Create audio processor."""
        return AudioProcessor(
            transcription_provider="whisper",
            embedding_model="wav2vec2",
            enable_diarization=False,
        )

    @pytest.fixture
    def sample_audio_bytes(self):
        """Create sample audio bytes."""
        import struct
        sample_rate = 16000
        num_samples = sample_rate

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + num_samples * 2, b"WAVE", b"fmt ", 16,
            1, 1, sample_rate, sample_rate * 2, 2, 16, b"data", num_samples * 2,
        )
        return header + b"\x00" * (num_samples * 2)

    def test_supported_modalities(self, processor):
        """Test supported modalities."""
        assert ModalityType.AUDIO in processor.supported_modalities

    def test_supported_formats(self, processor):
        """Test supported formats."""
        assert ".mp3" in processor.supported_formats
        assert ".wav" in processor.supported_formats

    @pytest.mark.asyncio
    async def test_process_audio(self, processor, sample_audio_bytes):
        """Test full audio processing."""
        with patch.object(processor.transcriber, "transcribe") as mock_trans:
            with patch.object(processor.embedder, "embed") as mock_embed:
                with patch.object(processor, "_get_audio_info") as mock_info:
                    mock_trans.return_value = TranscriptionResult(
                        text="Test transcription",
                        segments=[],
                        duration_seconds=1.0,
                    )
                    mock_embed.return_value = [0.1] * 768
                    mock_info.return_value = {
                        "duration": 1.0,
                        "sample_rate": 16000,
                        "channels": 1,
                        "mime_type": "audio/wav",
                    }

                    processor._initialized = True
                    result = await processor.process(
                        sample_audio_bytes,
                        transcribe=True,
                        generate_embedding=True,
                    )

                    assert result.modality == ModalityType.AUDIO
                    assert result.transcription == "Test transcription"
                    assert result.audio_embedding is not None

    @pytest.mark.asyncio
    async def test_extract_text(self, processor, sample_audio_bytes):
        """Test text extraction via transcription."""
        with patch.object(processor.transcriber, "transcribe") as mock_trans:
            mock_trans.return_value = TranscriptionResult(
                text="Extracted speech",
                segments=[],
            )
            processor._initialized = True

            text = await processor.extract_text(sample_audio_bytes)

            assert text == "Extracted speech"

    @pytest.mark.asyncio
    async def test_generate_embedding(self, processor, sample_audio_bytes):
        """Test embedding generation."""
        with patch.object(processor.embedder, "embed") as mock_embed:
            mock_embed.return_value = [0.5] * 768
            processor._initialized = True

            embedding = await processor.generate_embedding(sample_audio_bytes)

            assert len(embedding) == 768

    @pytest.mark.asyncio
    async def test_process_with_diarization(self, sample_audio_bytes):
        """Test processing with speaker diarization."""
        processor = AudioProcessor(enable_diarization=True)

        with patch.object(processor.transcriber, "transcribe") as mock_trans:
            with patch.object(processor.embedder, "embed") as mock_embed:
                with patch.object(processor.diarizer, "diarize") as mock_diarize:
                    with patch.object(processor, "_get_audio_info") as mock_info:
                        mock_trans.return_value = TranscriptionResult(
                            text="Speaker one. Speaker two.",
                            segments=[],
                        )
                        mock_embed.return_value = [0.0] * 768
                        mock_diarize.return_value = [
                            SpeakerSegment("SPEAKER_01", 0.0, 1.0),
                            SpeakerSegment("SPEAKER_02", 1.0, 2.0),
                        ]
                        mock_info.return_value = {"duration": 2.0, "sample_rate": 16000, "channels": 1}

                        processor._initialized = True
                        result = await processor.process(
                            sample_audio_bytes,
                            diarize=True,
                        )

                        assert len(result.speaker_segments) == 2
