import dspy
import numpy as np
from transformers import AutoModel

from router_poc import settings as S


class ModelSingleton:
    _instance = None
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = AutoModel.from_pretrained(
                "jinaai/jina-embeddings-v3",
                trust_remote_code=True,
            )
            cls._model.to("cuda")
            cls._model.eval()
        return cls._model


def _embed(text: str) -> np.ndarray:
    model = ModelSingleton.get_model()
    return model.encode(text, task="classification")


embed = dspy.Embedder(_embed)