from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class QueryRequest(BaseModel):
    query: str
    workspace: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    final_answer: str
    agent_path: list
    workspace_type: str
    sql_query: Optional[str]
    execution_time: float
    timestamp: datetime