import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai_service.config import settings
from ai_service.api.routes_qa import router as qa_router

logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("coverageiq.main")

app = FastAPI(
    title="CoverageIQ AI Service",
    description="Evidence-Grounded Insurance Policy Intelligence System (Hybrid Search, Grounded LLM & Evaluator Agent)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount internal AI routes
app.include_router(qa_router, prefix="/ai", tags=["Policy Q&A & Verification"])

@app.get("/ai/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "database_mode": "mock_memory" if settings.USE_MOCK_STORAGE else "postgresql_pgvector",
        "reranker_model": settings.RERANKER_MODEL_NAME,
        "retrieval": {
            "rrf_k": settings.RRF_K,
            "weights": {
                "vector": settings.WEIGHT_VECTOR,
                "bm25": settings.WEIGHT_BM25
            }
        },
        "evaluator": {
            "max_retries": settings.MAX_RETRIES,
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_service.main:app", host=settings.HOST, port=settings.PORT, reload=True)
