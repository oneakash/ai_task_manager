import logging
import chromadb
from app.config import CHROMA_PATH, CHROMA_COLLECTION

logger = logging.getLogger("TaskFlowAgent")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)


def sync_task_to_chroma(task) -> None:
    """Upserts task embedding & sanitized metadata into ChromaDB."""
    doc_text = f"{task.title}. {task.description or ''}".strip()
    metadata = {
        "title": str(task.title or ""),
        "status": str(task.status or "To do"),
        "priority": str(task.priority or "Medium"),
        "due_date": str(task.due_date or ""),
    }
    collection.upsert(ids=[task.id], documents=[doc_text], metadatas=[metadata])
    logger.info(
        f"[CHROMA_UPSERT] id={task.id} | title='{task.title}' | metadata={metadata}"
    )


def remove_task_from_chroma(task_id: str) -> None:
    try:
        collection.delete(ids=[task_id])
        logger.info(f"[CHROMA_DELETE] Removed vector embedding for id={task_id}")
    except Exception as e:
        logger.error(f"[CHROMA_DELETE_ERROR] Failed removing id={task_id}: {e}")