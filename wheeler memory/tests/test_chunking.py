"""Unit tests for domain chunking."""
import pytest
from wheeler_memory.chunking import DEFAULT_CHUNK, select_chunk, select_recall_chunks


class TestSelectChunk:
    """Tests for select_chunk function."""

    def test_select_chunk_code_keywords(self):
        """'fix the python bug' → 'code'."""
        chunk = select_chunk("fix the python bug")
        assert chunk == "code"

    def test_select_chunk_science_keywords(self):
        """'quantum entanglement' → 'science'."""
        chunk = select_chunk("quantum entanglement")
        assert chunk == "science"

    def test_select_chunk_hardware_keywords(self):
        """'3d printer filament' → 'hardware'."""
        chunk = select_chunk("3d printer filament")
        assert chunk == "hardware"

    def test_select_chunk_daily_tasks_keywords(self):
        """'grocery list for tomorrow' → 'daily_tasks'."""
        chunk = select_chunk("grocery list for tomorrow")
        assert chunk == "daily_tasks"

    def test_select_chunk_meta_keywords(self):
        """'wheeler attractor' → 'meta'."""
        chunk = select_chunk("wheeler attractor")
        assert chunk == "meta"

    def test_select_chunk_general_default(self):
        """'random unrelated text xyz' → DEFAULT_CHUNK ('general')."""
        chunk = select_chunk("random unrelated text xyz")
        assert chunk == DEFAULT_CHUNK


class TestSelectRecallChunks:
    """Tests for select_recall_chunks function."""

    def test_select_recall_chunks_always_includes_general(self):
        """any text → result includes 'general'."""
        result = select_recall_chunks("random text")
        assert DEFAULT_CHUNK in result

    def test_select_recall_chunks_code_text(self):
        """code text → result includes 'code' and 'general'."""
        result = select_recall_chunks("fix the python bug")
        assert "code" in result
        assert DEFAULT_CHUNK in result

    def test_select_recall_chunks_multiple_domains(self):
        """text with multiple domain keywords → result includes both."""
        result = select_recall_chunks("python bug in 3d printer")
        assert "code" in result
        assert "hardware" in result
        assert DEFAULT_CHUNK in result
