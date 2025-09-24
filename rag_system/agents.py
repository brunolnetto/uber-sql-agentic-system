from typing import List, Dict, Any
import json

import pandas

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