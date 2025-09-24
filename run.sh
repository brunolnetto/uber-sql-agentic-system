#!/bin/bash

# RAG System Startup Script
# This script helps you get the system up and running

set -e

echo "🚀 Starting RAG System Setup..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your actual API keys before continuing!"
    echo "   Required: OPENAI_API_KEY"
    read -p "Press enter when you've updated the .env file..."
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY is not set in .env file"
    exit 1
fi

echo "🐳 Starting Docker containers..."
docker-compose up -d postgres pgvector redis

echo "⏳ Waiting for databases to be ready..."
sleep 30

# Check if databases are ready
echo "🔍 Checking database connections..."
until docker-compose exec postgres pg_isready -U postgres; do
    echo "Waiting for main database..."
    sleep 5
done

until docker-compose exec pgvector pg_isready -U postgres; do
    echo "Waiting for vector database..."
    sleep 5
done

echo "✅ Databases are ready!"

# Install Python dependencies if needed
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "📦 Installing Python dependencies..."
source venv/bin/activate
pip install -r requirements.txt

# Initialize the main database
echo "🗄️ Initializing main database..."
python3 -c "
import asyncio
from main import DatabaseManager
async def init_db():
    db = DatabaseManager('postgresql://postgres:password@localhost:5432/rag_system')
    await db.initialize()
    print('Main database initialized!')
asyncio.run(init_db())
"

# Setup vector database
echo "🔍 Setting up vector database..."
python3 vector_setup.py

# Start the FastAPI application
echo "🚀 Starting RAG System API..."
echo "API will be available at: http://localhost:8000"
echo "API docs will be available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the service"

python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload