"""
Central configuration for the medAI RAG pipeline.

All tunable parameters live here so they can be imported from a single place.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DOCUMENTS_DIR: Path = PROJECT_ROOT / "raw_documents"
DATA_DIR: Path = PROJECT_ROOT / "data"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

CLEANED_OUTPUT_PATH: Path = DATA_DIR / "cleaned_output.json"
CHUNKS_OUTPUT_PATH: Path = DATA_DIR / "chunks.json"
CHUNKS_BACKUP_PATH: Path = DATA_DIR / "chunks_backup_v1.json"
VECTOR_DB_PATH: str = str(DATA_DIR / "vector_db")
RETRIEVAL_LOG_PATH: str = str(LOGS_DIR / "retrieval.log")

# ──────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 512           # max tokens per chunk
CHUNK_OVERLAP: int = 50         # overlap tokens between consecutive chunks
MIN_CHUNK_CHARS: int = 100      # discard chunks shorter than this

CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

# ──────────────────────────────────────────────────────────────────────
# Embeddings & Vector Store (ChromaDB)
# ──────────────────────────────────────────────────────────────────────
# Upgraded from all-MiniLM-L6-v2 based on tuning_experiment.py A/B test:
# paraphrase-MiniLM-L6-v2 achieved 100.0% P@3 vs 94.9% for all-MiniLM-L6-v2
EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE: int = 64
COLLECTION_NAME: str = "uspstf_depression_guidelines"

# ──────────────────────────────────────────────────────────────────────
# Retrieval & Reranking (Day 2 Optimization)
# ──────────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = 15       # candidate count per retriever (semantic & BM25)
TOP_K_FINAL: int = 3            # final reranked chunks returned to generator
RRF_K: int = 60                 # Reciprocal Rank Fusion constant
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Calibrated confidence threshold for out-of-scope rejection
# Calibrated as midpoint between max OOS top-1 confidence (69.3%) and min in-scope top-1 confidence (92.6%)
CONFIDENCE_THRESHOLD: float = 0.81

# ──────────────────────────────────────────────────────────────────────
# Section Prior Boosts (Updated: wider spread for clinical authority)
# ──────────────────────────────────────────────────────────────────────
# Wider prior spread: Recommendation sections get 1.30x boost,
# while References are heavily demoted to 0.50x to prevent
# bibliographic entries from dominating retrieval results.
SECTION_PRIORS: dict[str, float] = {
    "Recommendation": 1.30,
    "Clinical Considerations": 1.20,
    "Practice Considerations": 1.15,
    "General": 1.10,
    "Table": 1.00,
    "Recommendations of Others": 0.70,
    "References": 0.50,
}

# ──────────────────────────────────────────────────────────────────────
# Perinatal Query Boost
# ──────────────────────────────────────────────────────────────────────
PERINATAL_QUERY_KEYWORDS: list[str] = [
    "pregnant", "postpartum", "perinatal", "epds", "edinburgh",
]
PERINATAL_CHUNK_KEYWORDS: list[str] = [
    "EPDS", "Edinburgh", "postpartum", "perinatal",
]
PERINATAL_BOOST: float = 1.25

# ──────────────────────────────────────────────────────────────────────
# Older Adults Query Boost
# ──────────────────────────────────────────────────────────────────────
OLDER_ADULTS_QUERY_KEYWORDS: list[str] = [
    "over 65", "older adults", "geriatric", "elderly", "seniors", "gds",
]
OLDER_ADULTS_CHUNK_KEYWORDS: list[str] = [
    "GDS", "Geriatric Depression Scale", "older adults", "65 years", "geriatric", "elderly",
]
OLDER_ADULTS_BOOST: float = 1.20

# ──────────────────────────────────────────────────────────────────────
# Screening‑tool keywords & patterns (for chunk tagging & scope checks)
# ──────────────────────────────────────────────────────────────────────
SCREENING_TOOL_KEYWORDS: list[str] = [
    "PHQ-2", "PHQ-9", "PHQ-10", "EPDS", "Edinburgh",
    "BDI", "Beck Depression Inventory",
    "CES-D", "GDS", "Geriatric Depression Scale",
    "HAM-D", "HDRS", "Hamilton",
    "MADRS", "K6", "Kessler", "Zung",
    "Columbia", "C-SSRS", "ASQ", "GAD-2", "GAD-7",
    "SRQ",
]
