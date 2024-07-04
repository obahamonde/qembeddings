import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, ParamSpec, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
from cachetools import TTLCache, cached
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from PIL import Image
from pydantic import BaseModel
from requests import get
from transformers import AutoTokenizer  # type: ignore
from transformers import (  # type: ignore
    AutoImageProcessor,
    AutoModel,
    PreTrainedModel,
    PreTrainedTokenizer,
)

T = TypeVar("T")
P = ParamSpec("P")


def asyncify(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


@cached(TTLCache[str, PreTrainedModel](maxsize=1, ttl=3600 * 60))
def load_model(model_name: str) -> PreTrainedModel:  # type: ignore
    return AutoModel.from_pretrained(model_name, trust_remote_code=True)  # type: ignore


@cached(TTLCache[str, PreTrainedTokenizer](maxsize=1, ttl=3600 * 60))
def load_tokenizer(tokenizer_name: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_name)  # type: ignore


class Content(BaseModel):
    content: str


@dataclass
class ImageTextEmbedder:
    processor: AutoImageProcessor = field(init=False)
    vision_model: PreTrainedModel = field(init=False)
    tokenizer: PreTrainedTokenizer = field(init=False)
    text_model: PreTrainedModel = field(init=False)
    vision_model_name: Literal["nomic-ai/nomic-embed-vision-v1.5"] = field(
        default="nomic-ai/nomic-embed-vision-v1.5"
    )
    text_model_name: Literal["nomic-ai/nomic-embed-text-v1.5"] = field(
        default="nomic-ai/nomic-embed-text-v1.5"
    )

    def __post_init__(self):
        self.processor = AutoImageProcessor.from_pretrained(self.vision_model_name)
        self.vision_model = load_model(self.vision_model_name)
        self.tokenizer = load_tokenizer(self.text_model_name)
        self.text_model = load_model(self.text_model_name)
        self.vision_model.eval()
        self.text_model.eval()

    def compute_image_embedding(self, image: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(get(image, stream=True).raw).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.vision_model(**inputs)

        if hasattr(outputs, "last_hidden_state"):
            embedding = outputs.last_hidden_state.mean(dim=1)
        else:
            embedding = outputs.pooler_output

        return F.normalize(embedding, p=2, dim=1)

    def compute_text_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=8192
        )

        with torch.no_grad():
            outputs = self.text_model(**inputs)

        # Use the [CLS] token embedding or mean pooling, depending on the model's output
        if hasattr(outputs, "last_hidden_state"):
            embedding = outputs.last_hidden_state.mean(dim=1)
        else:
            embedding = outputs.pooler_output

        return F.normalize(embedding, p=2, dim=1)

    def compute_similarity(
        self, embedding1: torch.Tensor, embedding2: torch.Tensor
    ) -> float:
        return F.cosine_similarity(embedding1, embedding2, dim=1).item()

    def compute_image_text_similarity(
        self, image: Union[str, Image.Image], text: str
    ) -> float:
        image_embedding = self.compute_image_embedding(image)
        text_embedding = self.compute_text_embedding(text)
        return self.compute_similarity(image_embedding, text_embedding)


model = ImageTextEmbedder()


@asyncify
def make_embedding(text: str) -> list[np.ndarray[np.float32, Any]]:
    return model.compute_text_embedding(text).tolist()


@asyncify
def make_image_embedding(image: str) -> list[np.ndarray[np.float32, Any]]:
    return model.compute_image_embedding(image).tolist()


def create_app():
    app = FastAPI(
        title="QEmbeddings",
        description="API for making embeddings of text using Sentence Transformers",
        version="0.1.0",
    )

    @app.post("/api/embeddings", response_class=ORJSONResponse)
    async def _(request: Content):
        start = time.perf_counter()
        embeddings = await make_embedding(request.content)
        return ORJSONResponse(
            content={
                "total": len(embeddings),
                "dim": len(embeddings),
                "model": model.text_model_name,
                "process_time": time.perf_counter() - start,
                "content": embeddings,
            },
            status_code=200,
        )

    @app.post("/api/image-embeddings", response_class=ORJSONResponse)
    async def _(request: Content):
        start = time.perf_counter()
        embeddings = await make_image_embedding(request.content)
        return ORJSONResponse(
            content={
                "total": len(embeddings),
                "dim": len(embeddings),
                "model": model.vision_model_name,
                "process_time": time.perf_counter() - start,
                "content": embeddings,
            },
            status_code=200,
        )

    return app
