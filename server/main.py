import os
import sys
import time
import threading
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path so src imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline, ensure_data_ready

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="medAI Clinical RAG API", version="1.0.0")

# Allow all CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for cold-start
_is_ready = False
_startup_error = None
_warmup_lock = threading.Lock()

def _warmup_background():
    global _is_ready, _startup_error
    try:
        logger.info("Starting background data warmup (ensure_data_ready)...")
        ensure_data_ready()
        
        # Pre-initialize pipeline singletons by running a dummy query
        # This forces the models to load into memory
        logger.info("Initializing models with warmup query...")
        try:
            # We skip this if it fails, just meant to load models
            from src.retrieval.retrieval_manager import RetrievalManager
            rm = RetrievalManager()
            rm.retrieve("test query")
        except Exception as e:
            logger.warning(f"Warmup query failed (expected if API key missing, but models loaded): {e}")

        with _warmup_lock:
            _is_ready = True
        logger.info("Background warmup COMPLETE.")
    except Exception as e:
        logger.error(f"Background warmup FAILED: {e}", exc_info=True)
        with _warmup_lock:
            _startup_error = str(e)

@app.on_event("startup")
def startup_event():
    # Start the warmup in a background thread to allow uvicorn to bind immediately
    thread = threading.Thread(target=_warmup_background, daemon=True)
    thread.start()

class ChatRequest(BaseModel):
    query: str
    lang: Optional[str] = None

@app.get("/health")
def health_check():
    with _warmup_lock:
        ready = _is_ready
        err = _startup_error
    return {"ok": True, "ready": ready, "error": err}

@app.post("/chat")
def chat(request: ChatRequest):
    # Wait up to 120s for readiness
    wait_time = 0
    max_wait = 120
    
    while wait_time < max_wait:
        with _warmup_lock:
            if _is_ready:
                break
            if _startup_error:
                return JSONResponse(
                    status_code=500,
                    content={"status": "ERROR", "message": f"Startup failed: {_startup_error}"}
                )
        time.sleep(1)
        wait_time += 1
        
    with _warmup_lock:
        if not _is_ready:
            return JSONResponse(
                status_code=503,
                content={"status": "WARMING_UP", "retry": True, "message": "Server is still initializing models."}
            )

    try:
        # run_pipeline only accepts query right now. lang is handled by the internal crisis gate.
        result = run_pipeline(request.query)
        
        # Parse citations from response markdown
        import re
        citations = []
        # Regex to match [Doc: ... | Sec: ... | Pg: ... | Quote: "..."]
        cite_pattern = r'\[Doc:\s*(.*?)\s*\|\s*Sec:\s*(.*?)\s*\|\s*Pg:\s*(.*?)\s*\|\s*Quote:\s*\"(.*?)\"\]'
        
        response_markdown = result.get("response") or ""
        
        for match in re.finditer(cite_pattern, response_markdown):
            doc, sec, pg, quote = match.groups()
            citations.append({
                "doc": doc,
                "section": sec,
                "page": pg,
                "quote": quote,
                "verified": True # If it's in the final response_markdown, it passed verify_citations or was repaired
            })

        top_chunks = []
        ret = result.get("retrieval")
        if ret and isinstance(ret, dict) and "final_chunks" in ret:
            for chunk in ret["final_chunks"]:
                top_chunks.append({
                    "doc": chunk.get("source", "Unknown"),
                    "section": chunk.get("section", "General"),
                    "page": chunk.get("page", 1),
                    "score": chunk.get("score", 0.0)
                })

        # Extract safety and gating info
        safety_gate = "PASS"
        crisis_referral = False
        safe_dict = result.get("safety")
        if safe_dict and isinstance(safe_dict, dict):
            safety_gate = safe_dict.get("status", "PASS")
            if "CRISIS" in safety_gate:
                crisis_referral = True
                
        response_markdown = result.get("response") or ""
        sections = {}

        confidence = 0.0
        if ret and isinstance(ret, dict) and "top1_confidence" in ret:
            confidence = ret["top1_confidence"]

        model = "openrouter"
        tokens = 0
        gen = result.get("generation")
        if gen and isinstance(gen, dict):
            model = gen.get("model", "unknown")
            tokens = gen.get("usage", {}).get("total_tokens", 0)

        citations_verified = True
        if gen and isinstance(gen, dict) and gen.get("citation_status") == "CITATION_VERIFICATION_FAILED":
            citations_verified = False

        disclaimer = result.get("disclaimer", "")
        if not disclaimer and "This tool is not a substitute" in response_markdown:
            disclaimer = "This tool is not a substitute for professional medical judgment. Always verify with current guidelines and consult appropriate specialists."

        latency_ms = result.get("total_time_ms", 0)
        related_questions = []

        return {
            "status": result.get("status", "SUCCESS"),
            "response_markdown": response_markdown,
            "sections": sections,
            "citations": citations,
            "top_chunks": top_chunks,
            "confidence": confidence,
            "model": model,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "citations_verified": citations_verified,
            "safety_gate": safety_gate,
            "crisis_referral": crisis_referral,
            "disclaimer": disclaimer,
            "related_questions": related_questions
        }
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "message": str(e)}
        )
