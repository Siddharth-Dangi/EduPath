"""
knowledge_base.py
-----------------
Manages the RAG knowledge base for EduPulse using ChromaDB and
HuggingFace sentence embeddings. Stores topic-specific learning
resources that the AI tutor can retrieve on demand.
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


STUDY_RESOURCES: list[Document] = [
    Document(
        page_content=(
            "Gradient Boosting builds an ensemble of decision trees sequentially, "
            "where each tree corrects the residual errors of its predecessor. "
            "It is effective for both classification and regression tasks."
        ),
        metadata={"topic": "Machine Learning", "source": "Advanced ML Guide"},
    ),
    Document(
        page_content=(
            "Python list comprehensions provide a concise way to create lists. "
            "Syntax: [expression for item in iterable if condition]. "
            "They are generally faster than equivalent for-loop constructions."
        ),
        metadata={"topic": "Python Programming", "source": "Python Mastery"},
    ),
    Document(
        page_content=(
            "Calculus: The chain rule states d/dx[f(g(x))] = f'(g(x)) * g'(x). "
            "It is fundamental for computing derivatives of composite functions "
            "and is the backbone of backpropagation in neural networks."
        ),
        metadata={"topic": "Mathematics", "source": "Calculus Essentials"},
    ),
    Document(
        page_content=(
            "Active recall is one of the most evidence-backed study techniques. "
            "Instead of re-reading notes, regularly test yourself on the material "
            "using flashcards, practice problems, or the Feynman technique."
        ),
        metadata={"topic": "Study Skills", "source": "Learning Science"},
    ),
    Document(
        page_content=(
            "K-Means clustering iteratively assigns data points to the nearest "
            "centroid and recomputes centroids until convergence. Choosing k with "
            "the elbow method or silhouette score is recommended."
        ),
        metadata={"topic": "Machine Learning", "source": "Unsupervised Learning 101"},
    ),
    Document(
        page_content=(
            "SQL JOINs: INNER JOIN returns rows with matching keys in both tables. "
            "LEFT JOIN keeps all rows from the left table. RIGHT JOIN keeps all from "
            "the right table. FULL OUTER JOIN returns all rows from both tables."
        ),
        metadata={"topic": "Databases", "source": "SQL Fundamentals"},
    ),
    Document(
        page_content=(
            "Time-blocking is a productivity method where you schedule dedicated "
            "blocks of time for specific tasks. Pair it with the 2-minute rule: "
            "if a task takes less than 2 minutes, do it immediately."
        ),
        metadata={"topic": "Study Skills", "source": "Productivity Playbook"},
    ),
    Document(
        page_content=(
            "Logistic regression models the probability of a binary outcome using "
            "the sigmoid function σ(z) = 1/(1+e^-z). Coefficients are optimised "
            "via maximum likelihood estimation, not least squares."
        ),
        metadata={"topic": "Machine Learning", "source": "Statistical Learning"},
    ),
]


class KnowledgeHub:
    """Wraps ChromaDB to provide semantic search over curated study resources.

    Uses an in-memory store so it works on ephemeral cloud deployments
    (e.g. Streamlit Community Cloud) without requiring a persistent filesystem.
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self._store: Chroma | None = None

    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Build the knowledge base in memory from curated study resources."""
        print("[KnowledgeHub] Initialising in-memory knowledge base…")
        self._store = Chroma.from_documents(
            documents=STUDY_RESOURCES,
            embedding=self.embeddings,
        )

    # ------------------------------------------------------------------

    def find_resources(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top_k most semantically similar resources for a query."""
        if self._store is None:
            self.initialise()
        hits = self._store.similarity_search(query, k=top_k)
        return [
            {"content": doc.page_content, "source": doc.metadata.get("source", "—")}
            for doc in hits
        ]

    def as_retriever(self, top_k: int = 3):
        """Expose a LangChain-compatible retriever interface."""
        if self._store is None:
            self.initialise()
        return self._store.as_retriever(search_kwargs={"k": top_k})
