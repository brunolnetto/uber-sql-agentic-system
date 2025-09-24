## Copilot / AI Agent Quick Instructions

Short, actionable summary to get productive in this repo (RAG system with FastAPI + LangGraph).

1) Big picture: `main.py::RAGSystem` wires LLM-based agents (Intent, SQL, Table, ColumnPragma, QueryEvaluation) into a LangGraph workflow. `api.py` exposes endpoints that call `RAGSystem`.

2) Key files: `main.py` (agents, DBManager), `api.py` (endpoints), `vector_setup.py` (PGVector schema + embeddings), `run.sh`, `docker-compose.yaml`, `test_api.py`.

3) Project-specific conventions you must follow when editing code:
  - Workspace names are literal: `system` and `custom` (used across `main.py` and `api.py`).
  - Use asyncpg placeholders `$1`, `$2`, ... for parameterized SQL.
  - Agent methods (e.g., `SQLAgent.generate_sql`) must return content-only strings (no explanation text).
  - Vector embeddings use OpenAI dims (1536) and are stored in `document_embeddings.embedding VECTOR(1536)`.
  - `/execute-sql` endpoint enforces `SELECT`-only SQL — preserve this unless intentionally changing the security model.

4) Common edits and where to change them:
  - Add table: `DatabaseManager.create_tables` in `main.py`, `init.sql`, sample rows in `RAGSystem._populate_sample_data`, and embed schema via `vector_setup.py`.
  - Improve SQL generation: update `SQLAgent.generate_sql` or swap in `EnhancedSQLAgent.generate_sql_with_context` (vector search lives in `vector_setup.py` / `VectorDatabaseManager`).

5) Tests & debugging:
  - Integration tests: run `python3 test_api.py` against `http://localhost:8000`.
  - If startup stalls, check container health: `docker-compose ps` and `docker-compose logs postgres`.
  - DB readiness checks used in `run.sh`: `docker-compose exec postgres pg_isready -U postgres` and same for `pgvector`.

6) Examples to reuse:
  - SQL param style: `SELECT * FROM rides LIMIT $1` (used in `api.py`).
  - Sample query embedding: `"Show me all rides data" -> SELECT * FROM rides ORDER BY ride_date DESC` (in `vector_setup.py`).

If you want a longer merged version, or inclusion of prompt templates and unit test mocks for `ChatOpenAI` / `OpenAIEmbeddings`, tell me which sections to expand.
