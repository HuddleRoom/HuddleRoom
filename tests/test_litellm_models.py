import pytest

from huddleroom.services.litellm_models import build_litellm_model_name


@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("openrouter", "nvidia/nemotron-3", "openrouter/nvidia/nemotron-3"),
        ("openrouter", "openrouter/nvidia/nemotron-3", "openrouter/nvidia/nemotron-3"),
        ("openai", "groq/llama-3.1-70b", "groq/llama-3.1-70b"),
        (None, "gpt-6.1-sol", "gpt-6.1-sol"),
        ("openai", "gpt-6.1-sol", "openai/gpt-6.1-sol"),
    ],
)
def test_build_litellm_model_name(provider, model, expected) -> None:
    assert build_litellm_model_name(provider, model) == expected
