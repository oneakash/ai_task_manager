from sqlalchemy import Column, String, Text
from app.core.database import Base


class TaskModel(Base):
    __tablename__ = "tasks"
    id = Column(String, primary_key=True, index=True)  # e.g. "task-1"
    title = Column(String, nullable=False)
    description = Column(Text, default="")
    status = Column(String, default="To do")     # "To do", "In progress", "Done"
    priority = Column(String, default="Medium")  # "Low", "Medium", "High"
    due_date = Column(String, default="")