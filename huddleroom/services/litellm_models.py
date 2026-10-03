from __future__ import annotations


def build_litellm_model_name(provider: str | None, model: str) -> str:
    if not provider or model.startswith(f"{provider}/"):
        return model
    if provider == "openrouter":
        return f"{provider}/{model}"
    if "/" in model:
        return model
    return f"{provider}/{model}"
