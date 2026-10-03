from __future__ import annotations

import logging
from huddleroom.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    async def generate_embedding(self, text: str) -> list[float] | None:
        try:
            import litellm
            response = await litellm.aembedding(
                model=settings.embedding_model,
                input=[text],
            )
            return response.data[0]["embedding"]
        except Exception as e:
            logger.warning("Embedding generation failed: %s", e)
            return None
