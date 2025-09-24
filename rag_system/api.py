from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from typing import Optional
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from .graph import RAGSystem
from .schemas import QueryRequest, QueryResponse

# Global RAG system instance
rag_system: Optional[RAGSystem] = None

@asynccontextmanager
async def lifespan():
    """Initialize the RAG system on startup"""
    global rag_system
    try:
        rag_system = RAGSystem()
        await rag_system.initialize()
        print("RAG System initialized successfully")
        
        yield
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        raise e
    

app = FastAPI(
    title="RAG System API", 
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if rag_system and rag_system.db_manager.pool:
            # Test database connection
            await rag_system.db_manager.execute_query("SELECT 1")
            return {"status": "healthy", "database": "connected"}
        else:
            return {"status": "unhealthy", "database": "disconnected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query through the RAG system"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        result = await rag_system.process_query(request.query)
        
        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time
        
        return QueryResponse(
            query=result["query"],
            final_answer=result["final_answer"],
            agent_path=result["agent_path"],
            workspace_type=result["workspace_type"],
            sql_query=result["sql_query"],
            execution_time=execution_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/workspaces")
async def get_workspaces():
    """Get available workspaces"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    workspaces = await rag_system.db_manager.get_workspaces()
    return {"workspaces": workspaces}

@app.get("/tables/{workspace}")
async def get_workspace_tables(workspace: str):
    """Get tables available in a specific workspace"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        tables = await rag_system.db_manager.get_workspace_tables(workspace)
    except KeyError:
        raise HTTPException(status_code=404, detail="Workspace not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workspace tables: {str(e)}")

    table_schemas = {}
    for table in tables:
        schema = await rag_system.db_manager.get_table_schema(table)
        table_schemas[table] = schema

    return {"workspace": workspace, "tables": table_schemas}

@app.get("/sample-data/{table}")
async def get_sample_data(table: str, limit: int = 5):
    """Get sample data from a table"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Validate table name to prevent SQL injection using registered tables
        registered = await rag_system.db_manager.get_all_registered_tables()
        valid_tables = {entry["table"] for entry in registered}
        if table not in valid_tables:
            raise HTTPException(status_code=400, detail="Invalid table name")

        query = f"SELECT * FROM {table} LIMIT $1"
        results = await rag_system.db_manager.execute_query(query, [limit])

        return {"table": table, "sample_data": results, "count": len(results)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sample data: {str(e)}")

@app.post("/execute-sql")
async def execute_custom_sql(request: dict):
    """Execute a custom SQL query (be careful with this in production!)"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        sql_query = request.get("query", "")
        if not sql_query:
            raise HTTPException(status_code=400, detail="SQL query is required")
        
        # Basic safety check - only allow SELECT statements
        if not sql_query.strip().upper().startswith("SELECT"):
            raise HTTPException(status_code=400, detail="Only SELECT statements are allowed")
        
        results = await rag_system.db_manager.execute_query(sql_query)
        
        return {
            "query": sql_query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL execution failed: {str(e)}")

@app.get("/agent-info")
async def get_agent_info():
    """Get information about available agents"""
    return {
        "agents": [
            {
                "name": "Intent Agent",
                "description": "Classifies user intent and routes to appropriate workspace",
                "capabilities": ["intent_classification", "workspace_routing"]
            },
            {
                "name": "SQL Agent", 
                "description": "Generates and executes SQL queries based on user requests",
                "capabilities": ["sql_generation", "query_execution", "schema_analysis"]
            },
            {
                "name": "Table Agent",
                "description": "Processes and formats table data for user consumption",
                "capabilities": ["data_formatting", "result_processing", "insights_generation"]
            },
            {
                "name": "Column Pragma Agent",
                "description": "Provides column-level insights and data quality analysis", 
                "capabilities": ["column_analysis", "data_quality", "statistics"]
            },
            {
                "name": "Query Evaluation Agent",
                "description": "Evaluates query adequacy and suggests improvements",
                "capabilities": ["query_evaluation", "improvement_suggestions", "confidence_scoring"]
            }
        ],
        "workflow": [
            "intent_classification",
            "sql_generation", 
            "sql_execution",
            "table_processing",
            "column_analysis",
            "query_evaluation",
            "final_response"
        ]
    }