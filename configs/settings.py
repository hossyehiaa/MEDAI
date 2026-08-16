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

CLEANED_OUTPUT_PATH: Path = DATA_DIR / "cleaned_output.json"
CHUNKS_OUTPUT_PATH: Path = DATA_DIR / "chunks.json"
VECTOR_DB_PATH: str = str(DATA_DIR / "vector_db")

# ──────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 512           # max tokens per chunk
CHUNK_OVERLAP: int = 50         # overlap tokens between consecutive chunks
MIN_CHUNK_CHARS: int = 100      # discard chunks shorter than this

CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

# ──────────────────────────────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE: int = 64

# ──────────────────────────────────────────────────────────────────────
# Vector Store (ChromaDB)
# ──────────────────────────────────────────────────────────────────────
COLLECTION_NAME: str = "uspstf_depression_guidelines"

# ──────────────────────────────────────────────────────────────────────
# Screening‑tool keywords (for table tagging)
# ──────────────────────────────────────────────────────────────────────
SCREENING_TOOL_KEYWORDS: list[str] = [
    "PHQ-2", "PHQ-9", "PHQ-A", "EPDS", "Edinburgh",
    "BDI", "Beck Depression Inventory",
    "CES-D", "GDS", "Geriatric Depression Scale",
    "HAM-D", "HDRS", "Hamilton",
    "MADRS", "K6", "Kessler", "Zung",
    "Columbia", "C-SSRS", "ASQ", "GAD-2", "GAD-7",
    "SRQ",
]
