🏥 medAI — Clinical Decision Support RAG System
A Retrieval-Augmented Generation (RAG) system that delivers evidence-based clinical recommendations grounded strictly in official USPSTF guidelines. Built for the AI Clinical Decision Support Lite Hackathon.
🎯 Clinical Scope
Topic: Depression Screening in Adults
Source: U.S. Preventive Services Task Force (USPSTF), June 2023
Grade B Recommendation: Screening for depression in adults, including pregnant and postpartum persons
🏗️ Architecture (4-Layer Design)
Layer 1: Document Ingestion ✅ (Day 1 Complete)
PDF parsing with PyMuPDF + pdfplumber
Section-aware chunking (512 tokens, 50 overlap)
Boilerplate cleaning (running headers, page numbers, URLs)
Embeddings via all-MiniLM-L6-v2 (384-dim)
Persistent ChromaDB vector store
1,800 indexed chunks (1,362 text + 438 tables)
Layer 2: Retrieval 🚧 (In Progress)
Semantic search with ChromaDB
Planned: Hybrid search (BM25 + vectors)
Planned: Cross-encoder reranking
Layer 3: Generation 🚧 (Planned)
Strict grounding prompts
Structured citations (Doc | Section | Page)
Refusal mechanisms
Layer 4: Safety 🚧 (Planned)
Hallucination detection
Confidence thresholds
Out-of-scope query handling
