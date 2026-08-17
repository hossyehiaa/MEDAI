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
# Calibrated midpoint of min_in_scope_top1 and max_oos_top1; recompute after any retrieval change
CONFIDENCE_THRESHOLD: float = 0.76

# Section Prior Boosts (Updated: Day 3.5 Clinical Prior Hierarchy)
# ──────────────────────────────────────────────────────────────────────
# Recommendation sections get 1.30x boost, while low-value sections
# (Bibliography, Metadata, References) and Tables are demoted to ensure
# substantive clinical prose dominates retrieval.
SECTION_PRIORS: dict[str, float] = {
    "Recommendation": 1.30,
    "Clinical Considerations": 1.20,
    "Practice Considerations": 1.15,
    "General": 1.10,
    "Table": 0.85,
    "Recommendations of Others": 0.70,
    "References": 0.40,
    "Bibliography": 0.30,
    "Metadata": 0.30,
}

# ──────────────────────────────────────────────────────────────────────
# Source Display Names (Human-readable citations)
# ──────────────────────────────────────────────────────────────────────
SOURCE_DISPLAY_NAMES: dict[str, str] = {
    "Bookshelf_NBK592805": "AHRQ Evidence Review (USPSTF Bookshelf)",
    "depression-suicide-risk-adults-clinician-summ (2)": "USPSTF Clinician Summary (JAMA 2023)",
    "depression-suicide-risk-adults-clinician-summ": "USPSTF Clinician Summary (JAMA 2023)",
    "depression-suicide-risk-adults-final-evidence-summary (2)": "USPSTF Final Evidence Summary (2023)",
    "depression-suicide-risk-adults-final-evidence-summary": "USPSTF Final Evidence Summary (2023)",
}


def get_source_display_name(doc_name: str) -> str:
    """Return clean human-readable source display name."""
    clean_key = str(doc_name).replace(".pdf", "").strip()
    if clean_key in SOURCE_DISPLAY_NAMES:
        return SOURCE_DISPLAY_NAMES[clean_key]
    for k, v in SOURCE_DISPLAY_NAMES.items():
        if k in clean_key or clean_key in k:
            return v
    return clean_key


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
OLDER_ADULTS_BOOST: float = 1.10

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

# ──────────────────────────────────────────────────────────────────────
# LLM Generation (Day 3 — Groq)
# ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = "groq"
LLM_MODEL: str = "allam-2-7b"
LLM_FALLBACK_MODEL: str = "groq/compound"
LLM_TIMEOUT_SEC: int = 30
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
TEMPERATURE: float = 0.1
MAX_CONTEXT_CHUNKS: int = 3
GENERATION_LOG_PATH: str = str(LOGS_DIR / "generation.log")

