import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import Mock, patch

# Patch where GenerativeModel is used, not where it's from
@patch('gemini_agents_toolkit.core.llm.GenerativeModel')
def test_streaming_response(mock_generative_model):
    # Create mock model instance
    mock_model_instance = mock_generative_model.return_value

    # Simulate streaming output: 2 chunks + completion marker logic
    mock_chunks = [Mock(text="First chunk"), Mock(text="Second chunk")]
    mock_model_instance.generate_content.return_value = mock_chunks

    # Now import and test
    from gemini_agents_toolkit.core.llm import LLM
    llm = LLM()
    chunks = list(llm.generate("test prompt", stream=True))

    assert len(chunks) == 3  # 2 chunks + 1 completion marker
    assert chunks[0]["content"] == "First chunk"
    assert chunks[1]["content"] == "Second chunk"
    assert chunks[2]["is_complete"] is True
