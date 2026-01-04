"""
Enterprise RAG System - Admin API Routes
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/stats")
async def get_stats():
    """Get system statistics."""
    return {
        "documents": {"total": 0, "pending": 0, "processed": 0},
        "chunks": {"total": 0},
        "queries": {"total": 0, "today": 0},
    }


@router.post("/reindex")
async def trigger_reindex():
    """Trigger a full reindex."""
    return {"status": "queued", "message": "Reindex job queued"}
