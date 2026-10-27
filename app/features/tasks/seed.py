import logging
from app.core.database import SessionLocal, engine, Base
from app.core.chroma import sync_task_to_chroma
from app.features.tasks.models import TaskModel

logger = logging.getLogger("TaskFlowAgent")

INITIAL_SEED_TASKS = [
    {"id": "task-1", "title": "Finish AI presentation",
     "description": "Prepare slides and live demo for the AI agent presentation",
     "due_date": "Tomorrow", "status": "To do", "priority": "High"},
    {"id": "task-2", "title": "Review pull request #82",
     "description": "Code review for backend authentication PR #82",
     "due_date": "Today", "status": "In progress", "priority": "Medium"},
    {"id": "task-3", "title": "Send weekly report",
     "description": "Compile engineering metrics and email the weekly status report",
     "due_date": "Friday", "status": "To do", "priority": "High"},
    {"id": "task-4", "title": "Update API documentation",
     "description": "Document new FastAPI endpoints and ChromaDB retrieval schemas",
     "due_date": "Next week", "status": "To do", "priority": "Low"},
    {"id": "task-5", "title": "Set up database",
     "description": "Configure SQLite persistence and ChromaDB vector collection",
     "due_date": "Completed", "status": "Done", "priority": "Medium"},
    {"id": "task-6", "title": "Write project README",
     "description": "Write setup guide, architecture overview, and prompt debugging log",
     "due_date": "Completed", "status": "Done", "priority": "Low"},
]


def seed_database() -> None:
    """Creates tables if needed, seeds empty DB, and re-syncs Chroma on restart."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        count = db.query(TaskModel).count()
        if count == 0:
            logger.info("[STARTUP_SEED] Empty SQLite database detected. Seeding 6 initial reference tasks...")
            for item in INITIAL_SEED_TASKS:
                task = TaskModel(**item)
                db.add(task)
                db.commit()
                sync_task_to_chroma(task)
            logger.info("[STARTUP_SEED] Completed seeding 6 reference tasks.")
        else:
            logger.info(f"[STARTUP_SYNC] Found {count} existing tasks. Verifying ChromaDB sync...")
            for task in db.query(TaskModel).all():
                sync_task_to_chroma(task)
    finally:
        db.close()