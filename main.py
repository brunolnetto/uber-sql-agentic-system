import os
import asyncio
from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass
from datetime import datetime
import json

import asyncpg
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub
import pandas as pd

# Configuration
DATABASE_URL = "postgresql://postgres:password@localhost:5432/rag_system"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# State definition for LangGraph
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    user_query: str
    workspace_type: str
    sql_query: Optional[str]
    sql_results: Optional[List[Dict]]
    retrieved_docs: Optional[List[Document]]
    final_answer: Optional[str]
    agent_path: List[str]

@dataclass
class WorkspaceConfig:
    name: str
    description: str
    tables: List[str]
    custom_instructions: str

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

class IntentAgent:
    """Agent to classify user intent and route to appropriate workspace"""
    
    def __init__(self, llm):
        self.llm = llm
        
    async def classify_intent(self, user_query: str) -> str:
        """Classify user intent and return appropriate workspace"""
        prompt = f"""
        Analyze the following user query and classify it into one of these workspaces:
        - system: For queries about rides, platform engineering, or system metrics
        - custom: For queries about COGS (Cost of Goods Sold), financial metrics, or business analysis
        
        Query: {user_query}
        
        Return only the workspace name (system or custom).
        """
        
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        workspace = response.content.strip().lower()
        return "system" if workspace == "system" else "custom"

class SQLAgent:
    """Agent to generate and execute SQL queries"""
    
    def __init__(self, llm, db_manager: DatabaseManager):
        self.llm = llm
        self.db_manager = db_manager
        
    async def generate_sql(self, user_query: str, workspace_type: str) -> str:
        """Generate SQL query based on user query and workspace"""
        # Get relevant table schemas
        if workspace_type == "system":
            tables = ["rides", "platform_eng", "metrics"]
        else:
            tables = ["cogs", "metrics"]
            
        schema_info = {}
        for table in tables:
            schema_info[table] = await self.db_manager.get_table_schema(table)
            
        prompt = f"""
        Generate a SQL query to answer the following question based on the available tables:
        
        Question: {user_query}
        
        Available tables and schemas:
        {json.dumps(schema_info, indent=2)}
        
        Return only the SQL query, no explanations.
        """
        
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content.strip()
        
    async def execute_sql(self, sql_query: str) -> List[Dict]:
        """Execute SQL query and return results"""
        try:
            return await self.db_manager.execute_query(sql_query)
        except Exception as e:
            print(f"SQL execution error: {e}")
            return []

class TableAgent:
    """Agent to process and format table data"""
    
    def __init__(self, llm):
        self.llm = llm
        
    async def process_results(self, sql_results: List[Dict], user_query: str) -> str:
        """Process SQL results and format them appropriately"""
        if not sql_results:
            return "No data found for your query."
            
        prompt = f"""
        Format the following SQL results to answer the user's question:
        
        Question: {user_query}
        
        Results:
        {json.dumps(sql_results, indent=2, default=str)}
        
        Provide a clear, concise answer with relevant insights.
        """
        
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content

class ColumnPragmaAgent:
    """Agent to provide column-level insights and recommendations"""
    
    def __init__(self, llm):
        self.llm = llm
        
    async def analyze_columns(self, sql_results: List[Dict], user_query: str) -> str:
        """Analyze column patterns and provide insights"""
        if not sql_results:
            return "No data available for column analysis."
            
        # Extract column statistics
        df = pd.DataFrame(sql_results)
        stats = df.describe(include='all').to_dict()
        
        prompt = f"""
        Provide column-level insights for the following data:
        
        Original Question: {user_query}
        
        Column Statistics:
        {json.dumps(stats, indent=2, default=str)}
        
        Provide insights about data quality, patterns, and recommendations.
        """
        
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content

class QueryEvaluationAgent:
    """Agent to evaluate and refine queries"""
    
    def __init__(self, llm):
        self.llm = llm
        
    async def evaluate_query(self, original_query: str, sql_query: str, results: List[Dict]) -> Dict[str, Any]:
        """Evaluate if the query and results adequately answer the user's question"""
        prompt = f"""
        Evaluate if the SQL query and results adequately answer the user's question:
        
        User Question: {original_query}
        SQL Query: {sql_query}
        Number of Results: {len(results)}
        
        Respond with a JSON object containing:
        {{
            "adequate": true/false,
            "confidence": 0-100,
            "suggestions": "suggestions for improvement if inadequate"
        }}
        """
        
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        try:
            return json.loads(response.content)
        except:
            return {"adequate": True, "confidence": 70, "suggestions": ""}

class RAGSystem:
    """Main RAG system orchestrator using LangGraph"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.db_manager = DatabaseManager(DATABASE_URL)
        self.intent_agent = IntentAgent(self.llm)
        self.sql_agent = SQLAgent(self.llm, self.db_manager)
        self.table_agent = TableAgent(self.llm)
        self.column_agent = ColumnPragmaAgent(self.llm)
        self.eval_agent = QueryEvaluationAgent(self.llm)
        
        # Initialize LangGraph
        self.graph = self._create_graph()
        
    async def initialize(self):
        """Initialize the system"""
        await self.db_manager.initialize()
        await self._populate_sample_data()
        
    async def _populate_sample_data(self):
        """Populate sample data for testing"""
        # Sample rides data
        rides_data = [
            ("2024-01-15", "John Doe", 2, 15.5, 25.00),
            ("2024-01-16", "Jane Smith", 1, 8.2, 12.50),
            ("2024-01-17", "Mike Johnson", 3, 22.1, 35.75),
        ]
        
        for ride in rides_data:
            await self.db_manager.execute_query(
                "INSERT INTO rides (ride_date, driver_name, passenger_count, distance_km, fare_amount) VALUES ($1, $2, $3, $4, $5)",
                list(ride)
            )
            
        # Sample platform engineering data
        platform_data = [
            ("auth-service", "2024-01-10", "active", 45.5, 78.2),
            ("user-service", "2024-01-12", "active", 32.1, 65.4),
            ("payment-service", "2024-01-14", "maintenance", 12.3, 45.7),
        ]
        
        for platform in platform_data:
            await self.db_manager.execute_query(
                "INSERT INTO platform_eng (service_name, deployment_date, status, cpu_usage, memory_usage) VALUES ($1, $2, $3, $4, $5)",
                list(platform)
            )
            
        # Sample COGS data
        cogs_data = [
            ("Product A", 5.50, 100, 550.00, "2024-01-15"),
            ("Product B", 12.25, 75, 918.75, "2024-01-16"),
            ("Product C", 8.90, 200, 1780.00, "2024-01-17"),
        ]
        
        for cogs in cogs_data:
            await self.db_manager.execute_query(
                "INSERT INTO cogs (product_name, cost_per_unit, quantity, total_cost, cost_date) VALUES ($1, $2, $3, $4, $5)",
                list(cogs)
            )
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("intent_classification", self._intent_node)
        graph.add_node("sql_generation", self._sql_node)
        graph.add_node("sql_execution", self._execution_node)
        graph.add_node("table_processing", self._table_node)
        graph.add_node("column_analysis", self._column_node)
        graph.add_node("query_evaluation", self._evaluation_node)
        graph.add_node("final_response", self._response_node)
        
        # Define edges
        graph.add_edge("intent_classification", "sql_generation")
        graph.add_edge("sql_generation", "sql_execution")
        graph.add_edge("sql_execution", "table_processing")
        graph.add_edge("table_processing", "column_analysis")
        graph.add_edge("column_analysis", "query_evaluation")
        graph.add_edge("query_evaluation", "final_response")
        graph.add_edge("final_response", END)
        
        # Set entry point
        graph.set_entry_point("intent_classification")
        
        return graph.compile()
    
    async def _intent_node(self, state: AgentState) -> AgentState:
        """Intent classification node"""
        workspace = await self.intent_agent.classify_intent(state["user_query"])
        state["workspace_type"] = workspace
        state["agent_path"].append("intent_classification")
        return state
    
    async def _sql_node(self, state: AgentState) -> AgentState:
        """SQL generation node"""
        sql_query = await self.sql_agent.generate_sql(
            state["user_query"], 
            state["workspace_type"]
        )
        state["sql_query"] = sql_query
        state["agent_path"].append("sql_generation")
        return state
    
    async def _execution_node(self, state: AgentState) -> AgentState:
        """SQL execution node"""
        results = await self.sql_agent.execute_sql(state["sql_query"])
        state["sql_results"] = results
        state["agent_path"].append("sql_execution")
        return state
    
    async def _table_node(self, state: AgentState) -> AgentState:
        """Table processing node"""
        processed_results = await self.table_agent.process_results(
            state["sql_results"], 
            state["user_query"]
        )
        state["messages"].append({
            "agent": "table_agent",
            "content": processed_results,
            "timestamp": datetime.now().isoformat()
        })
        state["agent_path"].append("table_processing")
        return state
    
    async def _column_node(self, state: AgentState) -> AgentState:
        """Column analysis node"""
        column_insights = await self.column_agent.analyze_columns(
            state["sql_results"], 
            state["user_query"]
        )
        state["messages"].append({
            "agent": "column_agent",
            "content": column_insights,
            "timestamp": datetime.now().isoformat()
        })
        state["agent_path"].append("column_analysis")
        return state
    
    async def _evaluation_node(self, state: AgentState) -> AgentState:
        """Query evaluation node"""
        evaluation = await self.eval_agent.evaluate_query(
            state["user_query"], 
            state["sql_query"], 
            state["sql_results"]
        )
        state["messages"].append({
            "agent": "evaluation_agent",
            "content": evaluation,
            "timestamp": datetime.now().isoformat()
        })
        state["agent_path"].append("query_evaluation")
        return state
    
    async def _response_node(self, state: AgentState) -> AgentState:
        """Final response compilation node"""
        # Compile final answer from all agent outputs
        table_response = next((msg["content"] for msg in state["messages"] if msg["agent"] == "table_agent"), "")
        column_insights = next((msg["content"] for msg in state["messages"] if msg["agent"] == "column_agent"), "")
        
        final_answer = f"""
        **Query Results:**
        {table_response}
        
        **Additional Insights:**
        {column_insights}
        
        **Query Details:**
        - Workspace: {state["workspace_type"]}
        - SQL Query: {state["sql_query"]}
        - Records Found: {len(state["sql_results"]) if state["sql_results"] else 0}
        """
        
        state["final_answer"] = final_answer
        state["agent_path"].append("final_response")
        return state
    
    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query through the RAG system"""
        initial_state = AgentState(
            messages=[],
            user_query=user_query,
            workspace_type="",
            sql_query=None,
            sql_results=None,
            retrieved_docs=None,
            final_answer=None,
            agent_path=[]
        )
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        return {
            "query": user_query,
            "final_answer": result["final_answer"],
            "agent_path": result["agent_path"],
            "workspace_type": result["workspace_type"],
            "sql_query": result["sql_query"],
            "sql_results": result["sql_results"],
            "messages": result["messages"]
        }

# Example usage
async def main():
    """Main function to demonstrate the RAG system"""
    rag_system = RAGSystem()
    await rag_system.initialize()
    
    # Example queries
    queries = [
        "Show me all rides data",
        "What's the average cost per unit for our products?",
        "How many platform services are currently active?",
        "What's the total revenue from rides this month?"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        result = await rag_system.process_query(query)
        print(f"Answer: {result['final_answer']}")
        print(f"Agent Path: {' -> '.join(result['agent_path'])}")

if __name__ == "__main__":
    asyncio.run(main())