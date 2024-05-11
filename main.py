import asyncio
import time
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

import numpy as np
from cachetools import TTLCache, cached
from docs import app as docs_app
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer  # type: ignore

T = TypeVar("T")
P = ParamSpec("P")

cache = TTLCache(maxsize=1000, ttl=3600)
MODEL_NAME = "all-MiniLM-L6-v2"


class Content(BaseModel):
    content: str | list[str]


@cached(cache)
def load_model():
    return SentenceTransformer(MODEL_NAME)


model = load_model()


def asyncify(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


@asyncify
def make_embedding(text: str | list[str]) -> list[np.ndarray[np.float32, Any]]:
    if isinstance(text, str):
        text = [text]
    return model.encode(  # type: ignore
        text,
        batch_size=256,
        show_progress_bar=True,
        output_value="sentence_embedding",
        precision="float32",
        convert_to_numpy=True,
        convert_to_tensor=False,
        device="cpu",
        normalize_embeddings=False,
    )


app = FastAPI()


@app.post("/embeddings", response_class=ORJSONResponse)
async def embed(request: Content):
    start = time.perf_counter()
    embeddings = await make_embedding(request.content)
    return ORJSONResponse(
        content={
            "total": len(embeddings),
            "process_time": time.perf_counter() - start,
            "content": embeddings,
        },
        status_code=200,
    )

app.include_router(docs_app)
