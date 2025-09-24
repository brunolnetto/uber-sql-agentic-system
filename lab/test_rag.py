import asyncio

from rag_system import RAGSystem

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