import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab", quiet=True)

MIN_CHARS = 40   # discard passages shorter than this (fragments, URL-only comments)
MAX_CHARS = 600  # discard passages longer than this (wall-of-text, off-topic)


def chunk_text(text: str) -> list[dict]:
    """Segment text into single-sentence passages.

    Returns a list of dicts with key: text.
    Passages outside [MIN_CHARS, MAX_CHARS] are discarded.
    """
    sentences = sent_tokenize(text.strip())
    chunks = []
    for s in sentences:
        s = s.strip()
        if MIN_CHARS <= len(s) <= MAX_CHARS:
            chunks.append({"text": s})
    return chunks
