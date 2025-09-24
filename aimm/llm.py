"""LLM helper utilities."""
from __future__ import annotations

import json
from typing import Any, List

from .secrets import get_api_key

try:  # pragma: no cover
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:  # pragma: no cover
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore


class MissingProviderError(RuntimeError):
    """Raised when the requested LLM provider is unavailable."""


def get_llm_client(provider: str = "openai") -> Any:
    api_key = get_api_key(provider)
    if not api_key:
        raise ValueError(f"{provider.title()} API key is not configured.")
    if provider == "openai":
        if OpenAI is None:
            raise MissingProviderError("openai package is required for OpenAI provider")
        return OpenAI(api_key=api_key)
    if provider == "anthropic":
        if anthropic is None:
            raise MissingProviderError("anthropic package is required for Anthropic provider")
        return anthropic.Anthropic(api_key=api_key)
    raise NotImplementedError(f"Unsupported LLM provider: {provider}")


def llm_plan_keyframes(shot_description: str, num_keyframes: int = 3, context: str = "") -> List[str]:
    try:
        client = get_llm_client()
        system_prompt = (
            "You are an award-winning storyboard artist assisting a director."
            " Create vivid visual prompts for an image generation model."
            f" Provide exactly {num_keyframes} prompts in JSON format with the key 'prompts'."
        )
        user_prompt = json.dumps(
            {
                "context": context,
                "shot_description": shot_description,
                "instructions": "Focus on cinematic detail, camera placement, and mood.",
            },
            indent=2,
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        parsed = json.loads(response.choices[0].message.content)
        prompts = parsed.get("prompts", [])
        if not prompts:
            raise ValueError("LLM returned empty prompt list")
        return prompts[:num_keyframes]
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"⚠️ LLM keyframe planning failed: {exc}. Falling back to heuristic prompts.")
        return [
            f"{shot_description} – establishing shot with cinematic lighting",
            f"{shot_description} – mid-action, dynamic angle",
            f"{shot_description} – dramatic close-up finale",
        ][:num_keyframes]


def llm_refine_captions(raw_text: str, context: str = "") -> str:
    try:
        client = get_llm_client()
        system_prompt = (
            "You are a professional subtitle editor. Improve punctuation, grammar, and readability"
            " while preserving intent. Return only the cleaned text."
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps({"context": context, "raw": raw_text})},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"⚠️ Caption refinement skipped: {exc}")
        return raw_text
