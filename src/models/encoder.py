from __future__ import annotations

import inspect
from contextlib import contextmanager

import numpy as np
from transformers import AutoModel
from FlagEmbedding import BGEM3FlagModel


@contextmanager
def _fix_dtype_kwarg():
    """FlagEmbedding >=1.3 passes dtype= to AutoModel.from_pretrained, but
    transformers expects torch_dtype=. Patch only during model construction."""
    orig = AutoModel.from_pretrained.__func__

    @classmethod  # type: ignore[misc]
    def _patched(cls, *args, **kwargs):
        if "dtype" in kwargs:
            kwargs.setdefault("torch_dtype", kwargs.pop("dtype"))
        return orig(cls, *args, **kwargs)

    AutoModel.from_pretrained = _patched
    try:
        yield
    finally:
        AutoModel.from_pretrained = classmethod(orig)


class BGEM3Encoder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        self.device = device
        # FlagEmbedding only enables fp16 paths meaningfully on CUDA.
        kwargs: dict = {"use_fp16": (device == "cuda")}

        # The constructor kwarg that selects the compute device changed across
        # FlagEmbedding versions ("device" → "devices"). Pass whichever this
        # installed version accepts so the model actually lands on `device`
        # instead of silently auto-detecting (and falling back to CPU).
        params = inspect.signature(BGEM3FlagModel.__init__).parameters
        if "devices" in params:
            kwargs["devices"] = device
        elif "device" in params:
            kwargs["device"] = device

        with _fix_dtype_kwarg():
            self.model = BGEM3FlagModel(model_name, **kwargs)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 256,
        max_length: int = 512,
    ) -> dict:
        """Encode texts and return dense vectors and sparse weights.

        Returns:
            {
                "dense":  np.ndarray of shape (N, 1024), L2-normalized,
                "sparse": list of dict {token_id: weight} per passage,
            }
        """
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "dense": out["dense_vecs"],       # np.ndarray (N, 1024), already L2-normalized
            "sparse": out["lexical_weights"],  # list of {token_id: weight}
        }
