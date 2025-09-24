# Copilot / AI Agent Instructions for this repository

This file contains concise, actionable guidance so an AI coding agent can be productive immediately in this repo.

## Big picture
- This is a Retrieval-Augmented-Generation (RAG) system implemented in Python using LangGraph + LangChain-style components and a FastAPI front end (`api.py`).
- Two Postgres instances are used:
  - Main DB (business data): `postgres:5432` (`rag_system`) — see `docker-compose.yaml` and `main.py:DatabaseManager`.
  - Vector DB (PGVector): `pgvector:5433` (`vector_db`) — see `vector_setup.py:VectorDatabaseManager`.
- Core orchestrator: `main.py::RAGSystem` (creates LangGraph nodes and composes agents).
- Agents to know and edit: `IntentAgent`, `SQLAgent`, `TableAgent`, `ColumnPragmaAgent`, `QueryEvaluationAgent` (all in `main.py`).

## Key files to reference
- `main.py` — central orchestrator, DB manager, agent implementations, sample data population.
- `api.py` — FastAPI app and endpoints (`/query`, `/health`, `/workspaces`, `/tables/{workspace}`, `/sample-data/{table}`, `/execute-sql`).
- `vector_setup.py` — creates `document_embeddings` table and embeds schemas & sample queries.
- `run.sh` — convenience start script (creates venv, starts Docker containers, initializes DBs, runs vector setup, and launches API).
- `docker-compose.yaml` & `Dockerfile` — container orchestration and image build instructions.
- `requirements.txt` — pinned dependencies and versions used in the code.
- `init.sql` — DB initialization SQL (mounted into Postgres container).
- `test_api.py` — integration-style test script that calls the running API endpoints.

## Developer workflows & commands (explicit)
- Local quick start (preferred):
  - Ensure `.env` exists with `OPENAI_API_KEY`.
  - Run the provided script:
    ```bash
    chmod +x run.sh
    ./run.sh
    ```
  - The script does: `docker-compose up -d postgres pgvector redis`, creates venv, installs deps, initializes main DB, runs `vector_setup.py`, then starts `uvicorn api:app`.
- Docker-only development (no run.sh):
  - `docker-compose up -d` then `docker-compose logs -f rag_app` (or `docker-compose exec rag_app sh`).
- Run tests (integration against running service):
  - With app running: `python3 test_api.py` (calls `/health`, `/query`, `/sample-data`, etc.).

## Project-specific conventions & patterns
- Workspace names are explicit strings: `system` and `custom`. Code relies on these in `main.py` (e.g., `SQLAgent.generate_sql` and `api.py:/workspaces`).
- SQL parameters use asyncpg-style placeholders: `$1`, `$2`, ... — follow this when generating queries or edits to `DatabaseManager.execute_query` usage.
- The app expects agents to return content-only answers in many places (for example `SQLAgent.generate_sql` must return the SQL string only — prompts in code enforce that).
- Vector embeddings use OpenAI embeddings with dimension 1536 and are stored in `document_embeddings.embedding VECTOR(1536)` (see `vector_setup.py`).
- Safety: `api.py` enforces only `SELECT` statements for the `/execute-sql` endpoint — any code changes should preserve this constraint unless intentionally relaxing it.

## Integration points & external dependencies
- OpenAI: `OPENAI_API_KEY` env var. Embeddings and LLM calls are async: methods like `llm.ainvoke(...)` and `embeddings.aembed_query(...)` are used across the codebase.
- Databases: two Postgres instances. Connection strings are provided via env vars: `DATABASE_URL` and `VECTOR_DATABASE_URL` (see `docker-compose.yaml` and `run.sh`).
- Redis: included in `docker-compose.yaml` for caching (not heavily used in current code but present).

## Typical code edits an AI might perform
- Add a new sample table: update `main.py::DatabaseManager.create_tables`, `init.sql`, and add sample rows in `RAGSystem._populate_sample_data`. Also embed schema via `vector_setup.py`.
- Improve SQL generation: edit `SQLAgent.generate_sql` or create `EnhancedSQLAgent.generate_sql_with_context` and wire it into `RAGSystem._sql_node`.
- Make `/execute-sql` stricter or allow parameterized queries: update `api.py` and keep asyncpg `$n` placeholders.

## Debugging tips (project-specific)
- If startup hangs: the API startup calls `await rag_system.initialize()` in `api.py` which initializes DBs — ensure DB containers are healthy first (`docker-compose ps`, `docker-compose logs postgres`).
- Check DB readiness the same way `run.sh` does: `docker-compose exec postgres pg_isready -U postgres` and `docker-compose exec pgvector pg_isready -U postgres`.
- To inspect embeddings and similarity results: query `document_embeddings` in the vector DB (`psql -h localhost -p 5433 -U postgres -d vector_db`).

## Examples useful for AI-generated code or tests
- Example SQL placeholder style (use `$1`):
  - `SELECT * FROM rides LIMIT $1` — used in `api.py:/sample-data`.
- Example quick curl after startup:
  - `curl -X GET http://localhost:8000/health`
  - `curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{"query":"Show me all rides data"}' | jq .` 

## Quick pointers on unit / integration testing
- `test_api.py` is an integration tester that expects the service running on `http://localhost:8000`. Use it to validate end-to-end behavior.
- For unit testing of agents, mock `ChatOpenAI` and `OpenAIEmbeddings` methods (`ainvoke`, `aembed_query`) and `DatabaseManager` calls (`get_table_schema`, `execute_query`).

---
If any section is unclear or you want more examples (prompt templates, how to mock the async LLM calls, or a recommended unit test scaffold), tell me which area to expand and I will iterate.
