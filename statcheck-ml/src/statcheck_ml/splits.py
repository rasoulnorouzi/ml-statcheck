"""Make deterministic document-level train/dev splits."""
import random
from typing import List, Dict


def make_splits(documents: List[str], dev_share: float = 0.15, seed: int = 0) -> Dict:
    """Split documents into train and dev sets.

    Shuffles the unique documents with a given seed and assigns the first
    round(len * dev_share) to dev, the rest to train.

    Args:
        documents: List of document paths (may contain duplicates).
        dev_share: Fraction of documents for dev set (default 0.15).
        seed: Random seed for reproducibility (default 0).

    Returns:
        Dict with keys:
        - "seed": the seed used
        - "dev_share": the dev_share used
        - "documents": mapping of document to "train" or "dev"
    """
    # Get unique documents and sort for stability
    unique_docs = sorted(set(documents))

    # Shuffle with seed
    rng = random.Random(seed)
    shuffled = unique_docs.copy()
    rng.shuffle(shuffled)

    # Calculate split point
    n_dev = round(len(shuffled) * dev_share)
    dev_docs = set(shuffled[:n_dev])

    # Create mapping
    doc_mapping = {doc: "dev" if doc in dev_docs else "train" for doc in unique_docs}

    return {
        "seed": seed,
        "dev_share": dev_share,
        "documents": doc_mapping,
    }
