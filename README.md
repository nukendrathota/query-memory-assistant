# Query Memory Assistant

A lightweight Retrieval-Augmented Generation (RAG) assistant that stores user queries and responses in a PostgreSQL database using vector embeddings. If a similar question has been asked before, the assistant retrieves the previous response — avoiding redundant API calls to OpenAI.

## What It Does

This Streamlit-based chatbot:
- Embeds every incoming user query.
- Searches for semantically similar past queries in a PostgreSQL + pgvector database.
- If a close match is found, returns the previously generated response.
- If no match is found, queries OpenAI's API to generate a fresh answer and stores the new query-response pair.

## Tech Stack

- **Python 3.11+**
- [Streamlit](https://streamlit.io/)
- [OpenAI API](https://platform.openai.com/)
- [PostgreSQL + pgvector](https://github.com/pgvector/pgvector)
- [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings)

## Getting Started

### 1. Clone the repository

git clone https://github.com/nukendrathota/query-memory-assistant.git
cd query-memory-assistant

### 2. Install dependencies

https://platform.openai.com/docs/api-reference/embeddings

### 3. Setup Environment Files

OPENAI_API_KEY=your_openai_api_key_here
DB_URL=postgresql://user:password@host:port/dbname

### 4. Run the App

streamlit run app.py
