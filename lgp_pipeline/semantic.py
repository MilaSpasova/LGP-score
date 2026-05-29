from __future__ import annotations

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
except ModuleNotFoundError as e: 
    raise ModuleNotFoundError(
        "Missing dependency 'sentence-transformers'. Install project requirements first:\n"
        "  python -m pip install -r requirements.txt"
    ) from e


_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        # Load lazily because SBERT is only needed for the meaning-preservation
        # experiment and downloads model weights on first use.
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def compute_sbert_cosine_similarity(text_a: str, text_b: str) -> float:
    model = _get_model()
    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
    return float(cos_sim(embeddings[0], embeddings[1]).item())
