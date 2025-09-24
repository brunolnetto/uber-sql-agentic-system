from typing import List, Dict, Any, Optional, TypedDict
from pathlib import Path
import os
import datetime
import json

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .agents import (
    IntentAgent, 
    SQLAgent, 
    TableAgent, 
    ColumnPragmaAgent, 
    QueryEvaluationAgent,
)
from .storage import DatabaseManager, VectorDatabaseManager

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
VECTOR_DATABASE_URL = os.getenv("VECTOR_DATABASE_URL")
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


class RAGSystem:
    """Main RAG system orchestrator using LangGraph"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.db_manager = DatabaseManager(DATABASE_URL)
        self.vector_db_manager = VectorDatabaseManager(VECTOR_DATABASE_URL, OPENAI_API_KEY)
        
        self.intent_agent = IntentAgent(self.llm)
        self.sql_agent = SQLAgent(self.llm, self.db_manager, self.vector_db_manager)
        self.table_agent = TableAgent(self.llm)
        self.column_agent = ColumnPragmaAgent(self.llm)
        self.eval_agent = QueryEvaluationAgent(self.llm)
        
        # Initialize LangGraph
        self.graph = self._create_graph()
        
    async def initialize(self):
        """Initialize the system"""
        await self.db_manager.initialize()
        await self._populate_sample_data()
        await self._populate_sample_queries()
        
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
                """
                INSERT INTO rides (ride_date, driver_name, passenger_count, distance_km, fare_amount) 
                VALUES ($1, $2, $3, $4, $5)
                """,
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
                """
                INSERT INTO platform_eng (service_name, deployment_date, status, cpu_usage, memory_usage)
                VALUES ($1, $2, $3, $4, $5)
                """,
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
                """
                INSERT INTO cogs (product_name, cost_per_unit, quantity, total_cost, cost_date) 
                VALUES ($1, $2, $3, $4, $5)
                """,
                list(cogs)
            )
    
    def _populate_sample_queries(self):
        base = Path(__file__).parent  # use Path.cwd() if running interactively
        path = base / 'samples' / 'sample_queries.json'

        with path.open('r', encoding='utf-8') as f:
            sample_queries = json.load(f)

        self.vector_db_manager.embed_queries(sample_queries)
    
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