#!/usr/bin/env python3
"""
Test script to validate the RAG system functionality
"""

import asyncio
import json
import aiohttp
from typing import List, Dict, Any

class RAGSystemTester:
    """Test suite for the RAG System"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_check(self) -> bool:
        """Test if the API is healthy"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Health check passed")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Database: {data.get('database')}")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_workspaces(self) -> bool:
        """Test workspace listing"""
        try:
            async with self.session.get(f"{self.base_url}/workspaces") as response:
                if response.status == 200:
                    data = await response.json()
                    workspaces = data.get('workspaces', [])
                    print(f"✅ Found {len(workspaces)} workspaces:")
                    for workspace in workspaces:
                        print(f"   - {workspace['name']}: {workspace['description']}")
                    return True
                else:
                    print(f"❌ Workspaces test failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Workspaces test error: {e}")
            return False
    
    async def test_table_schemas(self, workspace: str) -> bool:
        """Test table schema retrieval"""
        try:
            async with self.session.get(f"{self.base_url}/tables/{workspace}") as response:
                if response.status == 200:
                    data = await response.json()
                    tables = data.get('tables', {})
                    print(f"✅ {workspace} workspace has {len(tables)} tables:")
                    for table_name, schema in tables.items():
                        columns = list(schema.keys())
                        print(f"   - {table_name}: {len(columns)} columns")
                    return True
                else:
                    print(f"❌ Table schemas test failed for {workspace}: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Table schemas test error for {workspace}: {e}")
            return False
    
    async def test_sample_data(self, table: str) -> bool:
        """Test sample data retrieval"""
        try:
            async with self.session.get(f"{self.base_url}/sample-data/{table}?limit=3") as response:
                if response.status == 200:
                    data = await response.json()
                    count = data.get('count', 0)
                    print(f"✅ Sample data for {table}: {count} records")
                    return True
                else:
                    print(f"❌ Sample data test failed for {table}: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Sample data test error for {table}: {e}")
            return False
    
    async def test_query_processing(self, query: str) -> Dict[str, Any]:
        """Test query processing through the RAG system"""
        try:
            payload = {"query": query}
            async with self.session.post(
                f"{self.base_url}/query", 
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Query processed successfully: '{query}'")
                    print(f"   Workspace: {data.get('workspace_type')}")
                    print(f"   Execution Time: {data.get('execution_time', 0):.3f}s")
                    print(f"   Agent Path: {' -> '.join(data.get('agent_path', []))}")
                    print(f"   SQL Query: {data.get('sql_query', 'N/A')}")
                    return data
                else:
                    error_data = await response.text()
                    print(f"❌ Query processing failed: {response.status}")
                    print(f"   Error: {error_data}")
                    return {}
        except Exception as e:
            print(f"❌ Query processing error: {e}")
            return {}
    
    async def test_custom_sql(self, sql_query: str) -> bool:
        """Test custom SQL execution"""
        try:
            payload = {"query": sql_query}
            async with self.session.post(
                f"{self.base_url}/execute-sql",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    count = data.get('count', 0)
                    print(f"✅ Custom SQL executed: {count} results")
                    return True
                else:
                    error_data = await response.text()
                    print(f"❌ Custom SQL failed: {response.status}")
                    print(f"   Error: {error_data}")
                    return False
        except Exception as e:
            print(f"❌ Custom SQL error: {e}")
            return False
    
    async def run_comprehensive_test(self):
        """Run a comprehensive test suite"""
        print("🧪 Starting RAG System Comprehensive Test Suite")
        print("=" * 60)
        
        # Test 1: Health Check
        print("\n1. Testing Health Check...")
        health_ok = await self.test_health_check()
        if not health_ok:
            print("❌ System is not healthy. Stopping tests.")
            return
        
        # Test 2: Workspaces
        print("\n2. Testing Workspaces...")
        await self.test_workspaces()
        
        # Test 3: Table Schemas
        print("\n3. Testing Table Schemas...")
        for workspace in ["system", "custom"]:
            await self.test_table_schemas(workspace)
        
        # Test 4: Sample Data
        print("\n4. Testing Sample Data...")
        for table in ["rides", "platform_eng", "metrics", "cogs"]:
            await self.test_sample_data(table)
        
        # Test 5: Query Processing
        print("\n5. Testing Query Processing...")
        test_queries = [
            "Show me all rides data",
            "What's the average cost per unit for our products?", 
            "How many platform services are currently active?",
            "What's the total revenue from rides?",
            "Which services have high CPU usage?",
            "Show me the most expensive products",
            "What are today's metrics?",
            "List all drivers and their total fares"
        ]
        
        successful_queries = 0
        for query in test_queries:
            print(f"\n   Testing: '{query}'")
            result = await self.test_query_processing(query)
            if result:
                successful_queries += 1
                # Print a sample of the answer
                answer = result.get('final_answer', '')
                if answer:
                    print(f"   Sample Answer: {answer[:200]}...")
        
        print(f"\n   Query Success Rate: {successful_queries}/{len(test_queries)} ({100*successful_queries/len(test_queries):.1f}%)")
        
        # Test 6: Custom SQL
        print("\n6. Testing Custom SQL...")
        custom_queries = [
            "SELECT COUNT(*) as total_rides FROM rides",
            "SELECT service_name, status FROM platform_eng LIMIT 5",
            "SELECT AVG(cost_per_unit) as avg_cost FROM cogs"
        ]
        
        for sql in custom_queries:
            print(f"   Testing SQL: {sql}")
            await self.test_custom_sql(sql)
        
        print("\n" + "=" * 60)
        print("🎉 Test Suite Complete!")

async def main():
    """Main test function"""
    print("RAG System Test Suite")
    print("Make sure the system is running on http://localhost:8000")
    print()
    
    async with RAGSystemTester() as tester:
        await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())