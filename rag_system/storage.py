from typing import Optional, List, Dict
import json

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncpg

class DatabaseManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        
    async def initialize(self):
        """Initialize database connection pool and create tables"""
        self.pool = await asyncpg.create_pool(self.database_url)
        await self.create_tables()
        
    async def create_tables(self):
        """Create necessary tables for the RAG system"""
        async with self.pool.acquire() as conn:
            # Metadata tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    description TEXT,
                    config JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Sample business tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS rides (
                    id SERIAL PRIMARY KEY,
                    ride_date DATE,
                    driver_name VARCHAR(100),
                    passenger_count INTEGER,
                    distance_km DECIMAL(8,2),
                    fare_amount DECIMAL(10,2),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS platform_eng (
                    id SERIAL PRIMARY KEY,
                    service_name VARCHAR(100),
                    deployment_date DATE,
                    status VARCHAR(50),
                    cpu_usage DECIMAL(5,2),
                    memory_usage DECIMAL(5,2),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(100),
                    metric_value DECIMAL(12,2),
                    metric_date DATE,
                    category VARCHAR(50),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cogs (
                    id SERIAL PRIMARY KEY,
                    product_name VARCHAR(100),
                    cost_per_unit DECIMAL(10,2),
                    quantity INTEGER,
                    total_cost DECIMAL(12,2),
                    cost_date DATE,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
    async def execute_query(self, query: str, params: Optional[List] = None) -> List[Dict]:
        """Execute SQL query and return results"""
        async with self.pool.acquire() as conn:
            if params:
                rows = await conn.fetch(query, *params)
            else:
                rows = await conn.fetch(query)
            return [dict(row) for row in rows]
    
    async def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get table schema information"""
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = $1
            ORDER BY ordinal_position
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, table_name)
            return {row['column_name']: row['data_type'] for row in rows}

    # Workspace management helpers
    async def register_workspace(self, name: str, description: str, tables: Optional[List[str]] = None, config: Optional[Dict] = None):
        """Register a workspace in the metadata `workspaces` table.

        `tables` will be stored inside `config` under the `tables` key.
        This allows dynamic workspace definitions without sprinkling literals across the codebase.
        """
        cfg = config.copy() if config else {}
        if tables is not None:
            cfg.setdefault("tables", tables)

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO workspaces (name, description, config)
                VALUES ($1, $2, $3)
                ON CONFLICT (name) DO UPDATE SET description = EXCLUDED.description, config = EXCLUDED.config
                """,
                name, description, json.dumps(cfg)
            )

    async def get_workspaces(self) -> List[Dict]:
        """Return list of registered workspaces with parsed config."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT name, description, config FROM workspaces ORDER BY id")
            result = []
            for r in rows:
                cfg = r["config"] if r["config"] else {}
                # If stored as string, try to parse
                if isinstance(cfg, str):
                    try:
                        cfg = json.loads(cfg)
                    except:
                        cfg = {}
                result.append({"name": r["name"], "description": r["description"], "config": cfg})
            return result

    async def get_workspace_tables(self, workspace_name: str) -> List[str]:
        """Return the tables configured for a workspace (raises KeyError if workspace not found)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT config FROM workspaces WHERE name = $1", workspace_name)
            if not row:
                raise KeyError(f"Workspace not found: {workspace_name}")
            cfg = row["config"] or {}
            if isinstance(cfg, str):
                try:
                    cfg = json.loads(cfg)
                except:
                    cfg = {}
            return cfg.get("tables", [])

    async def get_all_registered_tables(self) -> List[Dict[str, str]]:
        """Return a list of dicts with table -> workspace mapping from registered workspaces.

        Example: [{"table": "rides", "workspace": "system"}, ...]
        """
        workspaces = await self.get_workspaces()
        result: List[Dict[str, str]] = []
        for ws in workspaces:
            tables = ws.get("config", {}).get("tables", [])
            for t in tables:
                result.append({"table": t, "workspace": ws["name"]})
        return result

    async def ensure_default_workspaces(self):
        """Seed default workspaces if none are registered."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT count(*) as c FROM workspaces")
            if row and row["c"] > 0:
                return

        # Register default workspaces
        await self.register_workspace(
            "system", "System workspace for rides, platform engineering, and metrics", ["rides", "platform_eng", "metrics"]
        )
        await self.register_workspace(
            "custom", "Custom workspace for COGS and business analysis", ["cogs", "metrics"]
        )

class VectorDatabaseManager:
    """Manages vector database operations for semantic search and retrieval"""
    
    def __init__(self, vector_db_url: str, openai_api_key: str):
        self.vector_db_url = vector_db_url
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.pool = None
        
    async def initialize(self):
        """Initialize vector database connection and create tables"""
        self.pool = await asyncpg.create_pool(self.vector_db_url)
        await self.create_vector_tables()
        
    async def create_vector_tables(self):
        """Create vector extension and tables"""
        async with self.pool.acquire() as conn:
            # Enable vector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create embeddings table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    embedding VECTOR(1536),
                    workspace VARCHAR(50),
                    document_type VARCHAR(50),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create index for similarity search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS document_embeddings_embedding_idx 
                ON document_embeddings USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100)
            """)
            
            # Create metadata indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_embeddings_workspace 
                ON document_embeddings(workspace)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_embeddings_type 
                ON document_embeddings(document_type)
            """)
    
    async def embed_table_schemas(self, db_manager):
        """Embed table schemas and sample data for semantic search.

        This implementation reads the registered workspaces and their tables from
        `db_manager.get_all_registered_tables()` so the system no longer depends on
        hard-coded table lists.
        """
        # Get registered tables and their workspace mapping
        try:
            tables_info = await db_manager.get_all_registered_tables()
        except Exception as e:
            print(f"Could not load registered tables: {e}")
            tables_info = []

        # Build mapping table -> workspace (if multiple workspaces reference same table, mark as 'both')
        table_ws: Dict[str, str] = {}
        for entry in tables_info:
            t = entry.get("table")
            ws = entry.get("workspace")
            if not t:
                continue
            if t not in table_ws:
                table_ws[t] = ws
            else:
                if table_ws[t] != ws:
                    table_ws[t] = "both"

        tables = list(table_ws.keys())

        for table in tables:
            try:
                # Get schema information
                schema = await db_manager.get_table_schema(table)

                # Get sample data
                sample_data = await db_manager.execute_query(f"SELECT * FROM {table} LIMIT 3")

                # Create document content
                schema_text = f"Table: {table}\n"
                schema_text += "Columns:\n"
                for col, dtype in schema.items():
                    schema_text += f"- {col}: {dtype}\n"

                if sample_data:
                    schema_text += "\nSample Data:\n"
                    for i, row in enumerate(sample_data[:2], 1):
                        schema_text += f"Row {i}: {dict(row)}\n"

                # Determine workspace from mapping
                workspace = table_ws.get(table, "custom")

                # Create embedding
                embedding = await self.embeddings.aembed_query(schema_text)

                # Store in vector database
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO document_embeddings 
                        (document_id, content, metadata, embedding, workspace, document_type)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT DO NOTHING
                        """,
                        f"schema_{table}", schema_text,
                        {"table_name": table, "columns": list(schema.keys())},
                        embedding, workspace, "table_schema"
                    )

                print(f"Embedded schema for table: {table} (workspace={workspace})")

            except Exception as e:
                print(f"Error embedding schema for {table}: {e}")
    
    async def embed_queries(self, queries: List[Dict]):
        """Embed sample queries and their SQL translations"""
        for sample in queries:
            try:
                content = f"Query: {sample['query']}\nSQL: {sample['sql']}\nDescription: {sample['description']}"
                embedding = await self.embeddings.aembed_query(content)
                
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO document_embeddings 
                        (document_id, content, metadata, embedding, workspace, document_type)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT DO NOTHING
                    """, f"sample_query_{hash(sample['query'])}", content,
                    {"original_query": sample['query'], "sql_template": sample['sql']},
                    embedding, sample['workspace'], "sample_query")
                
                print(f"Embedded sample query: {sample['query']}")
                
            except Exception as e:
                print(f"Error embedding sample query: {e}")
    
    async def semantic_search(self, query: str, workspace: str = None, limit: int = 5) -> List[Dict]:
        """Perform semantic search on embedded documents"""
        try:
            # Create query embedding
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Build search query
            sql_query = """
                SELECT document_id, content, metadata, workspace, document_type,
                       1 - (embedding <=> $1) as similarity
                FROM document_embeddings
                WHERE 1 = 1
            """
            params = [query_embedding]
            
            if workspace and workspace != "both":
                sql_query += " AND (workspace = $2 OR workspace = 'both')"
                params.append(workspace)
            
            sql_query += " ORDER BY embedding <=> $1 LIMIT $" + str(len(params) + 1)
            params.append(limit)
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(sql_query, *params)
                
            return [
                {
                    "document_id": row["document_id"],
                    "content": row["content"],
                    "metadata": row["metadata"],
                    "workspace": row["workspace"],
                    "document_type": row["document_type"],
                    "similarity": float(row["similarity"])
                }
                for row in rows
            ]
            
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    async def get_relevant_schemas(self, query: str, workspace: str = None) -> List[Dict]:
        """Get relevant table schemas for a query"""
        results = await self.semantic_search(
            query, workspace, limit=3
        )
        
        return [
            result for result in results 
            if result["document_type"] == "table_schema"
        ]
    
    async def get_similar_queries(self, query: str, workspace: str = None) -> List[Dict]:
        """Get similar sample queries for reference"""
        results = await self.semantic_search(
            query, workspace, limit=3
        )
        
        return [
            result for result in results 
            if result["document_type"] == "sample_query"
        ]