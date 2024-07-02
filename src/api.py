import asyncio
import time
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

import numpy as np
from cachetools import TTLCache, cached
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer  # type: ignore

T = TypeVar("T")
P = ParamSpec("P")

cache = TTLCache(maxsize=1000, ttl=3600)  # type: ignore
MODEL_NAME = "all-mpnet-base-v2"


class Content(BaseModel):
    content: str


@cached(cache)  # type: ignore
def load_model():  # type: ignore
    return SentenceTransformer(MODEL_NAME)  # type: ignore


model = load_model()


def asyncify(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


@asyncify
def make_embedding(text: str) -> np.ndarray[np.float32, Any]:
    return model.encode(  # type: ignore
        text,
        batch_size=256,
        show_progress_bar=False,
        output_value="sentence_embedding",
        precision="float32",
        convert_to_numpy=True,
        convert_to_tensor=False,
        device="cpu",
        normalize_embeddings=False,
    )


def create_app():
    app = FastAPI(
        title="QEmbeddings",
        description="API for making embeddings of text using Sentence Transformers",
        version="0.1.0",
    )

    @app.post("/api/embeddings", response_class=ORJSONResponse)
    async def _(request: Content):
        start = time.perf_counter()
        vectors = await make_embedding(request.content)
        return ORJSONResponse(
            content={
                "dim": len(vectors),
                "model": MODEL_NAME,
                "process_time": time.perf_counter() - start,
                "content": vectors.tolist(),
            },
            status_code=200,
        )
    return app
