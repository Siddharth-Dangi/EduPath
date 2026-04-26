"""
knowledge_base.py
-----------------
Lightweight semantic search for EduPulse using sentence-transformers
and numpy cosine similarity — no external vector DB required.
Works on all Python versions and cloud environments.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Curated study resources
# ---------------------------------------------------------------------------

STUDY_RESOURCES: list[dict] = [
    {
        "content": (
            "Gradient Boosting builds an ensemble of decision trees sequentially, "
            "where each tree corrects the residual errors of its predecessor. "
            "It is effective for both classification and regression tasks."
        ),
        "source": "Advanced ML Guide",
        "topic": "Machine Learning",
    },
    {
        "content": (
            "Python list comprehensions provide a concise way to create lists. "
            "Syntax: [expression for item in iterable if condition]. "
            "They are generally faster than equivalent for-loop constructions."
        ),
        "source": "Python Mastery",
        "topic": "Python Programming",
    },
    {
        "content": (
            "Calculus: The chain rule states d/dx[f(g(x))] = f'(g(x)) * g'(x). "
            "It is fundamental for computing derivatives of composite functions "
            "and is the backbone of backpropagation in neural networks."
        ),
        "source": "Calculus Essentials",
        "topic": "Mathematics",
    },
    {
        "content": (
            "Active recall is one of the most evidence-backed study techniques. "
            "Instead of re-reading notes, regularly test yourself on the material "
            "using flashcards, practice problems, or the Feynman technique."
        ),
        "source": "Learning Science",
        "topic": "Study Skills",
    },
    {
        "content": (
            "K-Means clustering iteratively assigns data points to the nearest "
            "centroid and recomputes centroids until convergence. Choosing k with "
            "the elbow method or silhouette score is recommended."
        ),
        "source": "Unsupervised Learning 101",
        "topic": "Machine Learning",
    },
    {
        "content": (
            "SQL JOINs: INNER JOIN returns rows with matching keys in both tables. "
            "LEFT JOIN keeps all rows from the left table. RIGHT JOIN keeps all from "
            "the right table. FULL OUTER JOIN returns all rows from both tables."
        ),
        "source": "SQL Fundamentals",
        "topic": "Databases",
    },
    {
        "content": (
            "Time-blocking is a productivity method where you schedule dedicated "
            "blocks of time for specific tasks. Pair it with the 2-minute rule: "
            "if a task takes less than 2 minutes, do it immediately."
        ),
        "source": "Productivity Playbook",
        "topic": "Study Skills",
    },
    {
        "content": (
            "Logistic regression models the probability of a binary outcome using "
            "the sigmoid function σ(z) = 1/(1+e^-z). Coefficients are optimised "
            "via maximum likelihood estimation, not least squares."
        ),
        "source": "Statistical Learning",
        "topic": "Machine Learning",
    },
]


# ---------------------------------------------------------------------------
# KnowledgeHub
# ---------------------------------------------------------------------------


class KnowledgeHub:
    """
    Semantic search over study resources using sentence-transformers
    and cosine similarity — no vector database required.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model: SentenceTransformer | None = None
        self._embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Load the embedding model and pre-compute resource embeddings."""
        print("[KnowledgeHub] Loading embedding model…")
        self._model = SentenceTransformer(self.MODEL_NAME)
        texts = [r["content"] for r in STUDY_RESOURCES]
        self._embeddings = self._model.encode(texts, normalize_embeddings=True)
        print(f"[KnowledgeHub] Indexed {len(STUDY_RESOURCES)} study resources.")

    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Dot product of unit-normed vectors = cosine similarity."""
        return b @ a  # b: (n, dim), a: (dim,) → (n,)

    def find_resources(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top_k most semantically similar resources for a query."""
        if self._model is None or self._embeddings is None:
            self.initialise()

        query_vec = self._model.encode(query, normalize_embeddings=True)
        scores = self._cosine_similarity(query_vec, self._embeddings)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "content": STUDY_RESOURCES[i]["content"],
                "source": STUDY_RESOURCES[i]["source"],
                "score": float(scores[i]),
            }
            for i in top_indices
        ]
