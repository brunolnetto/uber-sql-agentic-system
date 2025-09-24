# RAG System with LangGraph

A sophisticated Retrieval-Augmented Generation (RAG) system built with Python, PostgreSQL, and LangGraph that processes natural language queries and converts them to SQL queries with intelligent agent orchestration.

## 🏗️ Architecture

The system implements a multi-agent architecture:

### Workspaces
- **System Workspace**: Handles queries about rides, platform engineering, and system metrics
- **Custom Workspace**: Manages COGS (Cost of Goods Sold) and business analysis queries

### Agent Pipeline
1. **Intent Agent**: Classifies user intent and routes to appropriate workspace
2. **SQL Agent**: Generates SQL queries based on user requests and table schemas
3. **Table Agent**: Processes and formats query results
4. **Column Pragma Agent**: Provides column-level insights and data quality analysis
5. **Query Evaluation Agent**: Evaluates query adequacy and suggests improvements

### Components
- **LangGraph Orchestration**: Manages agent workflow and state transitions
- **PostgreSQL Database**: Stores business data (rides, platform metrics, COGS)
- **PGVector Database**: Handles semantic search and embeddings
- **FastAPI**: Provides REST API interface
- **Redis**: Caching layer for improved performance

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### 1. Clone and Setup
```bash
git clone <repository-url>
cd rag-system
```

### 2. Environment Configuration
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the System
```bash
chmod +x run.sh
./run.sh
```

This script will:
- Start Docker containers (PostgreSQL, PGVector, Redis)
- Create Python virtual environment
- Install dependencies
- Initialize databases with sample data
- Setup vector embeddings
- Start the FastAPI server

### 4. Access the API
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 📊 Sample Data

The system comes pre-loaded with sample data:

### Rides Table
```sql
rides (id, ride_date, driver_name, passenger_count, distance_km, fare_amount, pickup_location, dropoff_location, status)
```

### Platform Engineering Table
```sql
platform_eng (id, service_name, deployment_date, status, cpu_usage, memory_usage, version, environment)
```

### Metrics Table
```sql
metrics (id, metric_name, metric_value, metric_date, category, subcategory, unit)
```

### COGS Table
```sql
cogs (id, product_name, cost_per_unit, quantity, total_cost, cost_date, supplier, category)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_queries.py
```

This will test:
- API health and connectivity
- Workspace functionality
- Table schema retrieval
- Sample data access
- Query processing through all agents
- Custom SQL execution

## 🔍 Example Queries

Try these sample queries:

### System Workspace Queries
- "Show me all rides data"
- "How many platform services are currently active?"
- "Which services have high CPU usage?"
- "What's the total revenue from rides this month?"

### Custom Workspace Queries
- "What's the average cost per unit for our products?"
- "Show me the highest cost products"
- "What's our total COGS for January?"
- "Which suppliers provide the most expensive materials?"

### Cross-Workspace Queries
- "What are today's key business metrics?"
- "Show me performance indicators across all systems"

## 📡 API Endpoints

### Core Endpoints
- `POST /query` - Process natural language queries
- `GET /health` - System health check
- `GET /workspaces` - List available workspaces
- `GET /tables/{workspace}` - Get table schemas for workspace
- `GET /sample-data/{table}` - Get sample data from table
- `POST /execute-sql` - Execute custom SQL (SELECT only)

### Response Format
```json
{
  "query": "Show me all rides data",
  "final_answer": "Here are the ride records...",
  "agent_path": ["intent_classification", "sql_generation", "sql_execution", "table_processing", "column_analysis", "query_evaluation", "final_response"],
  "workspace_type": "system",
  "sql_query": "SELECT * FROM rides ORDER BY ride_date DESC",
  "execution_time": 1.234,
  "timestamp": "2024-01-22T10:30:00"
}
```

## 🛠️ Development

### Project Structure
```
rag-system/
├── main.py              # Core RAG system implementation
├── api.py              # FastAPI application
├── vector_setup.py     # Vector database setup
├── test_queries.py     # Test suite
├── requirements.txt    # Python dependencies
├── docker-compose.yml  # Docker configuration
├── Dockerfile         # Container definition
├── init.sql          # Database initialization
├── run.sh            # Startup script
└── .env.example      # Environment template
```

### Key Classes
- `RAGSystem`: Main orchestrator using LangGraph
- `DatabaseManager`: PostgreSQL operations
- `VectorDatabaseManager`: Semantic search and embeddings
- `IntentAgent`: Query classification
- `SQLAgent`: SQL generation and execution
- `TableAgent`: Result processing
- `ColumnPragmaAgent`: Column analysis
- `QueryEvaluationAgent`: Quality assessment

### Adding New Tables
1. Add table schema to `init.sql`
2. Update workspace configuration in `main.py`
3. Add sample data and embeddings
4. Update agent logic if needed

### Customizing Agents
Each agent can be extended by:
- Modifying the prompt templates
- Adding new capabilities
- Integrating external tools
- Enhancing error handling

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_system
VECTOR_DATABASE_URL=postgresql://postgres:password@localhost:5433/vector_db
REDIS_URL=redis://localhost:6379
```

### Database Configuration
The system uses two PostgreSQL instances:
- **Main DB** (port 5432): Business data storage
- **Vector DB** (port 5433): Embeddings and semantic search

### LangGraph Configuration
The agent workflow is defined as a state graph with:
- State transitions between agents
- Error handling and retry logic
- Parallel processing where applicable
- Comprehensive logging

## 📈 Performance

### Optimization Features
- Connection pooling for database operations
- Vector indexing for fast semantic search
- Redis caching for frequent queries
- Async operations throughout the pipeline
- Query result pagination

### Monitoring
- Query execution time tracking
- Agent performance metrics
- Database query logging
- Error rate monitoring

## 🔒 Security

### Current Measures
- SQL injection prevention through parameterized queries
- Input validation and sanitization
- Limited SQL execution (SELECT only for custom queries)
- Environment variable protection

### Production Considerations
- Add authentication and authorization
- Implement rate limiting
- Use secrets management
- Add request/response encryption
- Database access controls

## 🚧 Roadmap

### Planned Features
- [ ] Multi-tenant support
- [ ] Real-time query streaming
- [ ] Advanced visualization integration
- [ ] ML model integration for better intent classification
- [ ] Query caching and optimization
- [ ] Audit logging and compliance features
- [ ] Integration with BI tools
- [ ] Advanced security features

### Known Limitations
- Currently supports only SELECT operations for safety
- Limited to predefined table schemas
- OpenAI API dependency for embeddings
- Single-language support (English)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Troubleshooting

**Database Connection Issues:**
- Ensure Docker containers are running: `docker-compose ps`
- Check database logs: `docker-compose logs postgres`
- Verify environment variables in `.env`

**API Errors:**
- Check API health: `curl http://localhost:8000/health`
- Review API logs for detailed error messages
- Ensure OpenAI API key is valid and has sufficient credits

**Vector Search Issues:**
- Run vector setup script: `python vector_setup.py`
- Check PGVector container: `docker-compose logs pgvector`
- Verify embeddings are created: Check `document_embeddings` table

### Getting Help
- Check the API documentation at `/docs`
- Review the test suite for usage examples
- Check Docker container logs for system issues
- Verify all environment variables are properly set

## 🙏 Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) for LLM integration
- [FastAPI](https://fastapi.tiangolo.com/) for the web API
- [PostgreSQL](https://www.postgresql.org/) for data storage
- [PGVector](https://github.com/pgvector/pgvector) for vector operations