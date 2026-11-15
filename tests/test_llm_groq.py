import os, pytest
from agents.llm import LLMResponse, ToolCallRequest

@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="no GROQ_API_KEY")
def test_groq_smoke():
    from agents.llm_groq import GroqLLMClient
    from agents.llm import LLMMessage
    r = GroqLLMClient().generate([LLMMessage(role="user", content="Reply with the single word: ok")])
    assert isinstance(r, LLMResponse)
    assert r.content