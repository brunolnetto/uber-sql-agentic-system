import os
import asyncio
from typing import List, Dict, Any
import asyncpg
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

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
        """Embed table schemas and sample data for semantic search"""
        # Get table schemas from main database
        tables = ["rides", "platform_eng", "metrics", "cogs"]
        
        for table in tables:
            try:
                # Get schema information
                schema = await db_manager.get_table_schema(table)
                
                # Get sample data
                sample_data = await db_manager.execute_query(
                    f"SELECT * FROM {table} LIMIT 3"
                )
                
                # Create document content
                schema_text = f"Table: {table}\n"
                schema_text += "Columns:\n"
                for col, dtype in schema.items():
                    schema_text += f"- {col}: {dtype}\n"
                
                if sample_data:
                    schema_text += "\nSample Data:\n"
                    for i, row in enumerate(sample_data[:2], 1):
                        schema_text += f"Row {i}: {dict(row)}\n"
                
                # Determine workspace
                workspace = "system" if table in ["rides", "platform_eng"] else "custom"
                if table == "metrics":
                    workspace = "both"
                
                # Create embedding
                embedding = await self.embeddings.aembed_query(schema_text)
                
                # Store in vector database
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO document_embeddings 
                        (document_id, content, metadata, embedding, workspace, document_type)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT DO NOTHING
                    """, f"schema_{table}", schema_text, 
                    {"table_name": table, "columns": list(schema.keys())},
                    embedding, workspace, "table_schema")
                    
                print(f"Embedded schema for table: {table}")
                
            except Exception as e:
                print(f"Error embedding schema for {table}: {e}")
    
    async def embed_sample_queries(self):
        """Embed sample queries and their SQL translations"""
        sample_queries = [
            {
                "query": "Show me all rides data",
                "sql": "SELECT * FROM rides ORDER BY ride_date DESC",
                "workspace": "system",
                "description": "Retrieve all ride records with details"
            },
            {
                "query": "What's the average cost per unit for our products?",
                "sql": "SELECT AVG(cost_per_unit) as avg_cost FROM cogs",
                "workspace": "custom", 
                "description": "Calculate average cost per unit across all products"
            },
            {
                "query": "How many platform services are currently active?",
                "sql": "SELECT COUNT(*) FROM platform_eng WHERE status = 'active'",
                "workspace": "system",
                "description": "Count active platform services"
            },
            {
                "query": "What's the total revenue from rides this month?",
                "sql": "SELECT SUM(fare_amount) as total_revenue FROM rides WHERE DATE_TRUNC('month', ride_date) = DATE_TRUNC('month', CURRENT_DATE)",
                "workspace": "system",
                "description": "Sum ride revenues for current month"
            },
            {
                "query": "Show me the highest cost products",
                "sql": "SELECT product_name, cost_per_unit FROM cogs ORDER BY cost_per_unit DESC LIMIT 10",
                "workspace": "custom",
                "description": "List products with highest cost per unit"
            },
            {
                "query": "Which services have high CPU usage?", 
                "sql": "SELECT service_name, cpu_usage FROM platform_eng WHERE cpu_usage > 50 ORDER BY cpu_usage DESC",
                "workspace": "system",
                "description": "Find services with CPU usage above 50%"
            }
        ]
        
        for sample in sample_queries:
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

# Enhanced RAG Agent with Vector Search
class EnhancedSQLAgent:
    """Enhanced SQL Agent with semantic search capabilities"""
    
    def __init__(self, llm, db_manager, vector_manager):
        self.llm = llm
        self.db_manager = db_manager
        self.vector_manager = vector_manager
        
    async def generate_sql_with_context(self, user_query: str, workspace_type: str) -> str:
        """Generate SQL with semantic context from vector search"""
        # Get relevant schemas
        relevant_schemas = await self.vector_manager.get_relevant_schemas(
            user_query, workspace_type
        )
        
        # Get similar queries for reference
        similar_queries = await self.vector_manager.get_similar_queries(
            user_query, workspace_type
        )
        
        # Build context
        context = "Relevant table information:\n"
        for schema in relevant_schemas:
            context += f"{schema['content']}\n\n"
        
        if similar_queries:
            context += "Similar query examples:\n"
            for similar in similar_queries:
                context += f"{similar['content']}\n\n"
        
        prompt = f"""
        Generate a SQL query to answer the following question using the provided context:
        
        Question: {user_query}
        Workspace: {workspace_type}
        
        Context:
        {context}
        
        Generate only the SQL query, no explanations.
        """
        
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content.strip()

# Setup script
async def setup_vector_database():
    """Setup and populate vector database"""
    from main import DatabaseManager
    
    # Initialize managers
    vector_db_url = os.getenv("VECTOR_DATABASE_URL", "postgresql://postgres:password@localhost:5433/vector_db")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return
    
    vector_manager = VectorDatabaseManager(vector_db_url, openai_api_key)
    await vector_manager.initialize()
    
    # Initialize main database manager for schema extraction
    main_db_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/rag_system")
    db_manager = DatabaseManager(main_db_url)
    await db_manager.initialize()
    
    # Embed schemas and sample queries
    await vector_manager.embed_table_schemas(db_manager)
    await vector_manager.embed_sample_queries()
    
    print("Vector database setup complete!")

if __name__ == "__main__":
    asyncio.run(setup_vector_database())