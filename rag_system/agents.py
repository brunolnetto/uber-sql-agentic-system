from typing import List, Dict, Any
import json

import pandas as pd

class IntentAgent:
    """Agent to classify user intent and route to appropriate workspace

    The agent now accepts a `db_manager` so it can validate predicted workspace
    labels against the registered workspaces in the system. This allows workspace
    names to be configured in the database instead of hard-coded.
    """

    def __init__(self, llm, db_manager=None):
        self.llm = llm
        self.db_manager = db_manager

    async def classify_intent(self, user_query: str) -> str:
        """Classify user intent and return an existing workspace name.

        The LLM is asked to propose a workspace label (free text). If that label
        matches a registered workspace name it is returned; otherwise, if there
        is a close match the first matching workspace is returned; otherwise
        the method falls back to the first registered workspace or 'custom'.
        """
        prompt = f"""
        Analyze the following user query and suggest the best workspace name to
        handle it. Return only a single workspace name (short, one word) that may
        correspond to a workspace registered in the system.

        Query: {user_query}
        """

        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        proposed = response.content.strip().lower()

        # If we have a db_manager, validate against registered workspaces
        if self.db_manager:
            try:
                workspaces = await self.db_manager.get_workspaces()
                names = [w["name"] for w in workspaces]
                if proposed in names:
                    return proposed
                # simple fallback: return first workspace containing the token
                for name in names:
                    if proposed in name:
                        return name
                # final fallback: return first registered workspace
                if names:
                    return names[0]
            except Exception:
                pass

        # No db_manager or validation failed -> conservative fallback
        return proposed if proposed in ("system", "custom") else "custom"

class SQLAgent:
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