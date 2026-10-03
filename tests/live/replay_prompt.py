#!/usr/bin/env python3
"""Minimal diagnostic script for testing LLM providers directly."""

import asyncio
import json
import sys
from pathlib import Path

import litellm


async def main():
    if len(sys.argv) < 2:
        print("Usage: replay_prompt.py <path-to-json>", file=sys.stderr)
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as f:
        params = json.load(f)

    model = params.pop("model")

    response = await litellm.acompletion(model=model, **params)

    content = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason

    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    raw_content_len = len(content)

    preview = content[:300].replace("\n", " ")
    print(f"Preview: {preview}", file=sys.stderr)

    result = {
        "content": content,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "raw_content_len": raw_content_len,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
